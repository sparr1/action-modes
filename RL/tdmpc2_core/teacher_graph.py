"""Whole-solve CUDA capture for independent, fresh inner-SAC teachers.

This reuses the whole-inner-graph execution design validated by the AMBI CUDA
graph benchmarks. It has an explicit RNG protocol: all random operations in a
root solve share one isolated CUDA stream of draws, seeded by the caller. This
is distinct from native AMBI's phase generators; it is never a silent fallback.
"""

from contextlib import contextmanager
import threading
import time

import torch

from .common.compile_regions import _preserve_host_rng_state


TEACHER_GRAPH_PROTOCOL = "isolated_cuda_teacher_per_root_seed_v1"
_RNG_LOCK = threading.RLock()
_ALPHA_FLOOR = 1e-8


def validate_teacher_graph_config(cfg, *, allow_gsde_collection=False):
    """Check the supported static protocol without allocating CUDA state."""
    gsde_collection = (
        allow_gsde_collection and getattr(cfg, "actor_distribution", "gaussian") == "gsde"
    )
    if getattr(cfg, "actor_distribution", "gaussian") == "gsde" and not gsde_collection:
        raise ValueError("gSDE graph sampling requires the live collection noise controller.")
    required = {
        "obs": "state",
        "inner_operator": "sac",
        "episodic": False,
        "inner_actor_adaptation": "clone",
        "inner_critic_adaptation": "clone",
        "inner_actor_initialization": "prior",
        "inner_critic_initialization": "prior",
        "inner_actor_source": "sac",
        "inner_critic_source": "sac",
        "inner_explorer_mode": "none",
        "inner_execution_policy_source": "primary",
        "inner_update_timing": "round",
        "inner_steps_per_update": None,
        "inner_sac_return_estimator": "one_step",
        "inner_critic_bernoulli_p": 0.0,
        "inner_outer_replay_fraction": 0.0,
        "inner_actor_writeback_coef": 0.0,
        "inner_critic_writeback_coef": 0.0,
        "critic_value_mode": "single",
        "aux_return_mode": "off",
        "inner_horizon_conditioning": "none",
    }
    if gsde_collection:
        # A collection graph unrolls a fixed number of non-episodic steps.
        # Interleaved updates are static too; the live controller supplies one
        # matrix per trajectory and round across every captured optimizer step.
        required.pop("inner_update_timing")
        required.pop("inner_steps_per_update")
        if getattr(cfg, "inner_update_timing", "round") not in {"round", "step"}:
            raise ValueError("gSDE collection graphs require round or step update timing.")
    for component in (
        "actor", "critic", "temperature", "replay", "actor_optimizer",
        "critic_optimizer", "temperature_optimizer",
    ):
        required[f"inner_{component}_scope"] = "action"
    for key, expected in required.items():
        actual = getattr(cfg, key, expected)
        if actual != expected:
            raise ValueError(
                f"CUDA teacher graph requires {key}={expected!r}; got {actual!r}."
            )
    if int(getattr(cfg, "inner_rounds", 0)) <= 0 or int(
        getattr(cfg, "inner_actor_updates_per_action", 0)
    ) <= 0:
        raise ValueError("CUDA teacher graph requires positive rounds and actor updates.")
    if getattr(cfg, "compile", False) and not getattr(cfg, "compile_strict", False):
        raise ValueError("Compiled CUDA teacher graph requires compile_strict=True.")
    if getattr(cfg, "sac_actor_loss_scale_mode", "none") not in {
        "none", "tdmpc2_percentile_range",
    }:
        raise ValueError("CUDA teacher graph does not support this Q-scale mode.")


class _GraphRNG:
    """Route explicit and implicit draws to the isolated CUDA default RNG."""

    _action_fork_depth = 0

    @contextmanager
    def fork(self, name):
        yield None

    @contextmanager
    def action_fork(self):
        yield

    def generator(self, name):
        return None

    def training_state_dict(self):
        raise RuntimeError(
            "A CUDA teacher worker is disposable; checkpoint the reanalysis seed generator."
        )


class TeacherGraph:
    """Own one private worker and enqueue fresh teacher solves on ``stream``.

    The worker must not be used or reset elsewhere after construction. Each
    runner owns its allocations and CUDA graph pool. Outer priors are shared
    read-only and must remain fixed until the caller joins every worker stream.

    ``__call__`` waits for its input's current producer stream, enqueues copies
    and a solve, then returns immediately. Its tensor outputs are overwritten
    on the next call. Clone outputs on this runner's stream and join that stream
    before consuming them elsewhere. Setup synchronizes; replay never does.

    ``capture=False`` is the otherwise matching uncaptured graph-protocol
    control, intended for parity validation rather than a fallback backend.
    """

    protocol = TEACHER_GRAPH_PROTOCOL

    def __init__(self, engine, root, *, capture=True, warmup=3):
        validate_teacher_graph_config(engine.cfg)
        if not isinstance(capture, bool):
            raise TypeError("capture must be bool.")
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 3:
            raise ValueError("Teacher graph warmup must be an integer of at least three.")
        if (not torch.is_tensor(root) or root.ndim != 2 or root.shape[0] != 1
                or root.shape[1] != int(engine.cfg.latent_dim)):
            raise ValueError("TeacherGraph requires one [1, latent_dim] root.")
        if engine.device.type != "cuda" or root.device.type != "cuda":
            raise ValueError("TeacherGraph requires a CUDA engine and CUDA root.")
        engine_index = engine.device.index
        if engine_index is None:
            engine_index = torch.cuda.current_device()
        if root.device.index != engine_index:
            raise ValueError("TeacherGraph engine and root must use the same CUDA device.")
        if getattr(engine, "_active_trace", None) is not None:
            raise ValueError("TeacherGraph cannot capture an active observational trace.")
        if engine.model.training:
            raise ValueError("TeacherGraph requires the shared world model in eval mode.")
        self.engine = engine
        self.device = root.device
        self.device_index = engine_index
        self.capture_enabled = capture
        self.graph = None
        self.calls = 0
        self.stream = torch.cuda.Stream(device=self.device)
        try:
            self._initialize(root, capture=capture, warmup=warmup)
        except BaseException:
            # A coordinator cannot join a constructor that never returned.
            # Finish this slot's outstanding reads before it restores shared
            # model modes or releases inputs; retain the original error.
            try:
                self.stream.synchronize()
            except Exception:
                pass
            raise

    def _initialize(self, root, *, capture, warmup):
        engine = self.engine
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        self._initial_alpha = engine._initial_inner_alpha
        # Validate inherited/fixed alpha once before capture. Replay refreshes
        # its value with device operations, avoiding the native .item() check.
        initial_alpha = self._initial_alpha().detach()
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            self.root = root.detach().clone()
            self.alpha_input = initial_alpha.clone()
        engine._initial_inner_alpha = lambda: self.alpha_input
        engine.rng = _GraphRNG()
        original_prepare = engine._prepare_workspace

        def prepare(*, t0):
            original_prepare(t0=t0)
            for name in ("actor_optim", "critic_optim", "temperature_optim"):
                optimizer = getattr(engine.state, name)
                if optimizer is None:
                    continue
                for group in optimizer.param_groups:
                    group["capturable"] = True
                for state in optimizer.state.values():
                    if "step" in state and state["step"].device != self.device:
                        state["step"] = state["step"].to(self.device)

        engine._prepare_workspace = prepare
        from torch._dynamo.utils import counters

        started = time.perf_counter()
        with torch.cuda.stream(self.stream), self._seeded_rng(0), _preserve_host_rng_state():
            for _ in range(warmup):
                before = int(counters["stats"]["unique_graphs"])
                self.output = self._solve()
            after = int(counters["stats"]["unique_graphs"])
            if after != before:
                raise RuntimeError(
                    "Teacher compiled regions did not stabilize before capture; increase warmup."
                )
            if any(region.failed for region in engine._compile_regions.values()):
                raise RuntimeError("Teacher graph cannot use compiler fallback.")
        self.stream.synchronize()
        self.warmup_seconds = time.perf_counter() - started
        self.capture_seconds = 0.0
        if capture:
            started = time.perf_counter()
            self.graph = torch.cuda.CUDAGraph()
            with self._seeded_rng(0), _preserve_host_rng_state():
                with torch.cuda.graph(self.graph, stream=self.stream):
                    self.output = self._solve()
            self.stream.synchronize()
            self.capture_seconds = time.perf_counter() - started

    @contextmanager
    def _seeded_rng(self, seed):
        # PyTorch 2.3 CUDAGraph stores per-graph device seed/offset tensors.
        # Its generator contract permits overlapping streams when enqueueing
        # from one host thread. The lock also serializes our host RNG swaps;
        # graph execution remains asynchronous on each worker's own stream.
        with _RNG_LOCK, torch.cuda.device(self.device), torch.random.fork_rng(
            devices=[self.device_index]
        ):
            torch.cuda.manual_seed(seed)
            yield

    def _solve(self):
        return self.engine.solve_teacher(self.root, t0=True)

    def _current_initial_alpha(self):
        cfg = self.engine.cfg
        if cfg.inner_temperature_initialization == "inherit_outer":
            alpha = self.engine._actor_owner.alpha.detach().reshape(-1)[0]
        else:
            alpha = self.alpha_input.new_full((), float(cfg.inner_temperature))
        # The normal outer learner already validates and floors temperatures.
        # Keep invalid-value failures explicit without synchronizing each root.
        torch._assert_async(
            torch.isfinite(alpha) & (alpha >= 0),
            "Teacher initial entropy coefficient must be finite and nonnegative.",
        )
        return alpha.clamp_min(_ALPHA_FLOOR)

    def __call__(self, root, *, seed):
        if (not torch.is_tensor(root) or root.shape != self.root.shape
                or root.dtype != self.root.dtype or root.device != self.device):
            raise ValueError("Teacher graph root shape, dtype, and device must remain fixed.")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("Teacher graph seed must be an integer in [0, 2**63).")
        if self.engine.model.training:
            raise ValueError("Teacher graph requires the shared world model in eval mode.")
        producer = torch.cuda.current_stream(self.device)
        if producer != self.stream:
            self.stream.wait_stream(producer)
            # Standalone callers may pass a temporary latent tensor. Keep its
            # storage alive until this asynchronous cross-stream copy finishes.
            root.record_stream(self.stream)
        # Root/Q-scale/alpha and current prior copies are ordered on this
        # stream. Q scaling is read directly from the stable outer estimator
        # inside _act_rl on each graph replay, with private local copies.
        with torch.cuda.stream(self.stream), self._seeded_rng(seed):
            self.root.copy_(root.detach())
            self.alpha_input.copy_(self._current_initial_alpha())
            if self.engine._sac_actor_loss_scale_enabled:
                scale = self.engine._critic_owner.actor_loss_scale
                torch._assert_async(
                    (torch.isfinite(scale) & (scale >= 1)).all(),
                    "Teacher Q scale must be finite and at least one.",
                )
            if self.graph is None:
                self.output = self._solve()
            else:
                self.graph.replay()
        self.calls += 1
        return self.output
