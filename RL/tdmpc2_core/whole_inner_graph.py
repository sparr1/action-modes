"""Experimental whole-solve CUDA graph for fixed, fresh inner SAC workloads.

PyTorch 2.3 cannot
capture AMBI's private generators or per-phase reseeding. The graph and its
uncaptured control therefore use one isolated CUDA RNG stream, with fresh
draws on every action. They do not reproduce native AMBI's seed sequence and
must not be used to resume a native training lineage. The opt-in training
adapter below checkpoints its own stream in a distinct graph lineage.
"""

from contextlib import contextmanager
from copy import deepcopy
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
        "inner_rebase_persistent": False,
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
    scale_mode = getattr(cfg, "sac_actor_loss_scale_mode", "none")
    scale_scope = getattr(cfg, "inner_actor_loss_scale_scope", "action")
    scale_update = getattr(cfg, "inner_actor_loss_scale_update", "per_action")
    if scale_mode not in ("none", "tdmpc2_percentile_range"):
        raise ValueError("Whole inner graph supports no Q scale or TD-MPC2 percentile-range Q scale.")
    if (scale_scope, scale_update) not in {
        ("action", "per_action"), ("action", "per_update"), ("run", "per_update"),
    } or (scale_mode == "none" and scale_scope != "action"):
        raise ValueError("Whole inner graph requires a validated actor loss scale scope/update pair.")
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
        self.actor_loss_scale_input = None
        if engine._sac_actor_loss_scale_enabled:
            self.actor_loss_scale_input = engine._critic_owner.actor_loss_scale.detach().clone()
            # The live engine selects the inherited/carried initial value before
            # every call. Capture must not close over the worker's warmup carry.
            engine._initial_actor_loss_scale = lambda outer_scale: self.actor_loss_scale_input.detach().clone()
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

    def __call__(self, root, *, initial_actor_loss_scale=None):
        if root.shape != self.root.shape or root.dtype != self.root.dtype or root.device != self.root.device:
            raise ValueError("Whole inner graph root shape, dtype and device must remain fixed.")
        # Preflight every input before changing captured buffers or RNG state.
        if self.actor_loss_scale_input is not None:
            if initial_actor_loss_scale is None:
                initial_actor_loss_scale = self.engine._critic_owner.actor_loss_scale
            target = self.actor_loss_scale_input
            if (initial_actor_loss_scale.shape != target.shape
                    or initial_actor_loss_scale.dtype != target.dtype
                    or initial_actor_loss_scale.device != target.device):
                raise ValueError("Whole inner graph Q scale shape, dtype and device must remain fixed.")
            if not bool((torch.isfinite(initial_actor_loss_scale) & (initial_actor_loss_scale >= 1)).all()):
                raise ValueError("Whole inner graph Q scale must be finite and at least one.")
        elif initial_actor_loss_scale is not None:
            raise ValueError("An unnormalized whole inner graph does not accept a Q scale.")
        # Preserve the native finite/nonnegative alpha contract and refresh
        # inherited alpha after any outer update.
        alpha = self._initial_alpha()
        self.alpha_input.copy_(alpha)
        if self.actor_loss_scale_input is not None:
            self.actor_loss_scale_input.copy_(initial_actor_loss_scale.detach())
        self.root.copy_(root)
        with self._rng_fork():
            if self.graph is None:
                self.output = self._solve()
            else:
                self.graph.replay()
        self.calls += 1
        return self.output


_TRAINING_PROTOCOL = "isolated_cuda_graph_with_native_diagnostics_v1"


def validate_training_graph(engine):
    """Keep the first training mode limited to the validated fresh K1 recipe."""
    validate_graph_workload(engine)
    for key, expected in {
        "obs": "state", "inner_solve_interval": 1,
        "inner_execution_action": "policy_sample",
        "inner_execution_policy_source": "primary",
        "outer_behavior_policy_kl_schedule": "none",
        "aux_return_mode": "off", "outer_critic_target": "reward_only",
        "inner_sac_critic_target": "reward_only",
        "compile": True, "compile_strict": True,
    }.items():
        if getattr(engine.cfg, key, expected) != expected:
            raise ValueError(f"Training CUDA graph requires {key}={expected!r}.")
    if engine.device.type != "cuda":
        raise ValueError("Training CUDA graph requires CUDA.")


def training_graph_state(engine):
    """Serialize only persistent graph randomness; its workspace is fresh."""
    if engine._whole_graph_runner is None and engine._whole_graph_resume is not None:
        return deepcopy(engine._whole_graph_resume)
    runner = engine._whole_graph_runner
    return {
        "schema": "ambi-training-whole-inner-graph-v1",
        "protocol": _TRAINING_PROTOCOL,
        "seed": int(engine.cfg.seed) + 8173,
        "calls": 0 if runner is None else runner.calls,
        "rng_state": None if runner is None else runner.rng_state.detach().cpu().clone(),
    }


def preflight_training_graph_state(engine, saved):
    """Reject native/graph continuation and malformed RNG before live mutation."""
    enabled = bool(getattr(engine.cfg, "inner_cuda_graph", False))
    if not enabled:
        if saved is not None:
            raise ValueError("A CUDA graph lineage cannot resume in native mode.")
        return None
    validate_training_graph(engine)
    expected = {"schema", "protocol", "seed", "calls", "rng_state"}
    if not isinstance(saved, dict) or saved.keys() != expected:
        raise ValueError("CUDA graph resume requires its complete graph RNG state.")
    if (saved["schema"] != "ambi-training-whole-inner-graph-v1"
            or saved["protocol"] != _TRAINING_PROTOCOL
            or type(saved["seed"]) is not int or saved["seed"] != int(engine.cfg.seed) + 8173
            or type(saved["calls"]) is not int or saved["calls"] < 0):
        raise ValueError("Incompatible CUDA graph RNG protocol or counters.")
    rng = saved["rng_state"]
    if rng is None:
        if saved["calls"] != 0:
            raise ValueError("A used CUDA graph stream must include its RNG state.")
    else:
        if not torch.is_tensor(rng) or rng.dtype != torch.uint8 or rng.ndim != 1:
            raise ValueError("CUDA graph RNG state must be a one-dimensional byte tensor.")
        # A private generator validates layout without changing global state.
        torch.Generator(device=engine.device).set_state(rng.detach().cpu())
    return deepcopy(saved)


def training_graph_act(engine, root):
    """Replay in a private workspace, retaining native diagnostics and reset logic.

    The native engine handles the configured diagnostic decisions and keeps its
    normal private streams. This workspace never aliases its pooled allocations,
    so an episode boundary/checkpoint cannot invalidate captured addresses.
    """
    if engine._whole_graph_runner is None:
        from .inner_improvement import InnerImprovementEngine
        worker = InnerImprovementEngine(engine.agent)
        worker._whole_graph_worker = True
        runner = WholeInnerGraph(worker, root, seed=int(engine.cfg.seed) + 8173, warmup=4)
        saved = engine._whole_graph_resume
        if saved is not None:
            if saved["rng_state"] is not None:
                runner.rng_state = saved["rng_state"].detach().cpu().clone()
            runner.calls = saved["calls"]
        engine._whole_graph_runner = runner
        engine._whole_graph_resume = None
    engine._pending_timers = {}
    started = engine._timer_start()
    if engine._sac_actor_loss_scale_enabled:
        initial_scale = engine._initial_actor_loss_scale(engine._critic_owner.actor_loss_scale)
        action, values, lengths = engine._whole_graph_runner(
            root, initial_actor_loss_scale=initial_scale,
        )
        if engine._carry_actor_loss_scale:
            # Preserve native [1] shape and own storage independently of the
            # captured outputs. A following native diagnostic uses this value,
            # and its successful final scale becomes the next graph input.
            engine._carried_actor_loss_scale = values["inner_actor_loss_scale_final"].detach().reshape_as(
                initial_scale
            ).clone()
    else:
        action, values, lengths = engine._whole_graph_runner(root)
    engine._timer_stop("inner_action_seconds", started)
    engine.action_index += 1
    # The live engine records the prior version used by this logical action,
    # even though its disposable tensor workspace belongs to the graph worker.
    engine.state.outer_version = int(engine.agent.outer_version)
    # Do not mutate the static result dictionary owned by the capture.
    values = dict(values, inner_cuda_graph_used=1.0)
    return action, values, lengths
