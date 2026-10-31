"""Experimental whole-solve CUDA graph for fixed, fresh inner SAC workloads.

This is a benchmark API, not a production execution mode. PyTorch 2.3 cannot
capture AMBI's private generators or per-phase reseeding. The graph and its
uncaptured control therefore use one isolated CUDA RNG stream, with fresh
draws on every action. They do not reproduce native AMBI's seed sequence and
must not be used to resume a native training lineage.
"""

from contextlib import contextmanager
import time

import torch


class _GraphRNG:
    """Use the default CUDA generator only inside the runner's isolated fork."""

    @contextmanager
    def fork(self, name):
        yield None

    @contextmanager
    def action_fork(self):
        yield

    def generator(self, name):
        return None

    def training_state_dict(self):
        raise RuntimeError("Experimental graph RNG cannot serialize a native AMBI checkpoint.")


def validate_graph_workload(engine):
    """Reject dynamic or persistent protocols instead of silently changing them."""
    cfg = engine.cfg
    expected = {
        "inner_operator": "sac", "episodic": False,
        "inner_actor_adaptation": "clone", "inner_critic_adaptation": "clone",
        "inner_actor_initialization": "prior", "inner_critic_initialization": "prior",
        "inner_explorer_mode": "none", "inner_update_timing": "round",
        "inner_critic_bernoulli_p": 0.0, "inner_outer_replay_fraction": 0.0,
        "inner_actor_writeback_coef": 0.0, "inner_critic_writeback_coef": 0.0,
        "inner_rebase_persistent": False, "sac_actor_loss_scale_mode": "none",
        "critic_value_mode": "single", "inner_horizon_conditioning": "none",
    }
    expected.update({f"inner_{name}_scope": "action" for name in (
        "actor", "critic", "temperature", "replay", "actor_optimizer",
        "critic_optimizer", "temperature_optimizer",
    )})
    for key, value in expected.items():
        actual = getattr(cfg, key, value)
        if actual != value:
            raise ValueError(f"Whole inner graph requires {key}={value!r}; got {actual!r}.")
    if int(cfg.inner_rounds) <= 0 or engine._retrace_enabled:
        raise ValueError("Whole inner graph requires positive rounds and ordinary SAC replay.")
    if cfg.compile and not cfg.compile_strict:
        raise ValueError("Compiled whole inner graph requires compile_strict=True.")
    if engine._active_trace is not None:
        raise ValueError("Whole inner graph does not support observational traces.")


def disable_phase_timers(engine):
    """Apply equally to all benchmark arms; external synchronized timers remain."""
    engine._timer_start = lambda: None
    engine._timer_stop = lambda key, start: None


class WholeInnerGraph:
    """Own a fixed engine and replay all its solve operations in one CUDA graph.

    ``capture=False`` is the otherwise identical uncaptured control. Inputs,
    alpha validation and RNG save/restore are included in the public call's
    latency. Tensor initialization, rollout, replay, losses, backwards, Adam,
    target updates, final action and ordinary scalar metrics are captured.
    Diagnostic probes, Python counters and host materialization are outside.
    The returned tensors are overwritten by the next invocation.
    """

    def __init__(self, engine, root, *, capture=True, seed=8173, warmup=3):
        validate_graph_workload(engine)
        if engine.device.type != "cuda" or root.device.type != "cuda":
            raise ValueError("WholeInnerGraph requires a CUDA engine and CUDA root.")
        if root.ndim != 2 or root.shape[0] != 1:
            raise ValueError("WholeInnerGraph requires one [1, latent_dim] root.")
        self.engine, self.device = engine, engine.device
        # Keep the native entry point when a benchmark installs this runner
        # as engine.act after capture.
        self._engine_act = engine.act
        self.root = root.detach().clone()
        self.capture_enabled = bool(capture)
        self.graph = None
        self.calls = 0
        self.seed = int(seed)
        self._initial_alpha = engine._initial_inner_alpha
        self.alpha_input = self._initial_alpha().detach().clone()
        engine._initial_inner_alpha = lambda: self.alpha_input
        engine.rng = _GraphRNG()
        disable_phase_timers(engine)
        original_prepare = engine._prepare_workspace

        def prepare(*, t0):
            original_prepare(t0=t0)
            for name in ("actor_optim", "critic_optim", "temperature_optim"):
                optimizer = getattr(engine.state, name)
                if optimizer is not None:
                    for group in optimizer.param_groups:
                        group["capturable"] = True
                    for state in optimizer.state.values():
                        if "step" in state and state["step"].device != self.device:
                            state["step"] = state["step"].to(self.device)

        engine._prepare_workspace = prepare
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        self.device_index = index
        with torch.random.fork_rng(devices=[index]):
            torch.cuda.manual_seed(self.seed)
            self.rng_state = torch.cuda.get_rng_state(self.device)
        started = time.perf_counter()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream), self._rng_fork():
            from torch._dynamo.utils import counters
            for _ in range(max(3, int(warmup))):
                before = int(counters["stats"]["unique_graphs"])
                self.output = self._solve()
            after = int(counters["stats"]["unique_graphs"])
            if after != before:
                raise RuntimeError("Inner regions did not stabilize before capture; increase warmup.")
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self.warmup_seconds = time.perf_counter() - started
        self.capture_seconds = 0.0
        if self.capture_enabled:
            started = time.perf_counter()
            self.graph = torch.cuda.CUDAGraph()
            with self._rng_fork():
                with torch.cuda.graph(self.graph, stream=stream):
                    self.output = self._solve()
            torch.cuda.synchronize(self.device)
            self.capture_seconds = time.perf_counter() - started
        self.reset_rng()

    @contextmanager
    def _rng_fork(self):
        with torch.random.fork_rng(devices=[self.device_index]):
            torch.cuda.set_rng_state(self.rng_state, self.device)
            try:
                yield
            finally:
                self.rng_state = torch.cuda.get_rng_state(self.device)

    def reset_rng(self):
        """Reset only this experimental RNG stream, for paired validation."""
        with torch.random.fork_rng(devices=[self.device_index]):
            torch.cuda.manual_seed(self.seed)
            self.rng_state = torch.cuda.get_rng_state(self.device)

    def _solve(self):
        return self._engine_act(self.root, collect_diagnostics=False, eval_mode=False)

    def __call__(self, root):
        if root.shape != self.root.shape or root.dtype != self.root.dtype or root.device != self.root.device:
            raise ValueError("Whole inner graph root shape, dtype and device must remain fixed.")
        # Validate outside capture, preserving the native finite/nonnegative
        # contract and refreshing inherited alpha after any outer update.
        self.alpha_input.copy_(self._initial_alpha())
        self.root.copy_(root)
        with self._rng_fork():
            if self.graph is None:
                self.output = self._solve()
            else:
                self.graph.replay()
        self.calls += 1
        return self.output
