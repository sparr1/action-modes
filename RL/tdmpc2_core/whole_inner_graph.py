"""Resumable, isolated CUDA graphs for fresh K1 SAC collection.

This retains the production collection protocol: one continuous private CUDA
RNG stream for graph actions, and the ordinary phase streams for diagnostic
actions. It deliberately differs from per-root-seeded reanalysis graphs.
"""

from contextlib import contextmanager
from copy import deepcopy
import time

import torch

from .common.compile_regions import _preserve_host_rng_state
from .teacher_graph import validate_teacher_graph_config


_TRAINING_PROTOCOL = "isolated_cuda_graph_with_native_diagnostics_v1"
_GSDE_TRAINING_PROTOCOL = "isolated_cuda_graph_with_live_gsde_inputs_v1"


def training_graph_protocol(cfg):
    """Name the graph RNG contract without altering historical identities."""
    return (_GSDE_TRAINING_PROTOCOL
            if getattr(cfg, "actor_distribution", "gaussian") == "gsde"
            else _TRAINING_PROTOCOL)


def _validate_fresh_config(cfg, *, require_compile):
    gsde = getattr(cfg, "actor_distribution", "gaussian") == "gsde"
    validate_teacher_graph_config(cfg, allow_gsde_collection=gsde)
    expected = {
        "inner_solve_interval": 1,
        "inner_actor_loss_scale_scope": "action",
        "inner_execution_action": "policy_sample",
        "outer_behavior_policy_kl_schedule": "none",
        "outer_critic_target": "reward_only",
        "inner_sac_critic_target": "reward_only",
        "inner_root_entropy_samples": 0,
    }
    if gsde:
        # Both targets are fixed-shape tensor equations. The legacy Gaussian
        # graph recipe retains its original reward-only contract.
        expected.pop("outer_critic_target")
        expected.pop("inner_sac_critic_target")
        for name in ("outer_critic_target", "inner_sac_critic_target"):
            if getattr(cfg, name) not in {"reward_only", "entropy_augmented"}:
                raise ValueError(f"gSDE collection graph has unsupported {name}.")
    if require_compile:
        expected.update(compile=True, compile_strict=True)
    for key, value in expected.items():
        if getattr(cfg, key, value) != value:
            raise ValueError(f"Collection CUDA graph requires {key}={value!r}.")
    if getattr(cfg, "inner_actor_loss_scale_update", "per_action") not in {"per_action", "per_update"}:
        raise ValueError("Collection CUDA graph requires per_action or per_update Q-scale updates.")


def validate_collection_graph_config(cfg):
    """Validate the supported fresh K1 recipe without allocating a device."""
    _validate_fresh_config(cfg, require_compile=True)


def validate_training_graph(engine):
    validate_collection_graph_config(engine.cfg)
    if engine.device.type != "cuda":
        raise ValueError("Collection CUDA graph requires CUDA.")


class _GraphRNG:
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
        raise RuntimeError("Checkpoint collection graph RNG through its live engine, not its disposable worker.")


class _GsdeGraphInputs:
    """Stable graph buffers; the live controller owns all matrix lifetimes.

    Round scopes need one buffer per round. Solve and real-step scopes share
    one buffer across every round, avoiding J redundant matrix copies. The
    cursor is only used while tracing/capturing a solve, never for replay.
    """

    def __init__(self, engine, root):
        cfg, actor = engine.cfg, engine._actor_base
        self.rounds = int(cfg.inner_rounds)
        self.per_round = cfg.inner_gsde_noise_scope == "round"
        self.collection_enabled = (
            cfg.inner_behavior_action == "policy_sample"
            and float(cfg.inner_behavior_std_scale) > 0.0
        )
        self.execution_enabled = (
            cfg.inner_execution_action == "policy_sample"
            and float(cfg.inner_execution_std_scale) > 0.0
        )
        shape = (int(cfg.inner_rollouts_per_round), actor.sde_feature_dim, actor.sde_action_dim)
        self.collection = tuple(
            root.new_zeros(shape)
            for _ in range((self.rounds if self.per_round else 1) if self.collection_enabled else 0)
        )
        self.execution = (
            root.new_zeros((int(root.shape[0]), actor.sde_feature_dim, actor.sde_action_dim))
            if self.execution_enabled else None
        )
        self._cursor = 0

    def install(self, engine):
        # Replaying a CUDA graph skips Python. Bind explicit buffers here so
        # neither a captured t0=True nor captured interval branches decide when
        # to resample behavior noise.
        engine._begin_gsde_action = self.begin
        engine._gsde_noise_for_round = self.next_round
        engine._gsde_noise_for_execution = self.for_execution

    def begin(self, *, t0, eval_mode):
        del t0, eval_mode
        self._cursor = 0

    def next_round(self, root, count, generator):
        del root, count, generator
        if not self.collection_enabled:
            return None
        if self._cursor >= self.rounds:
            raise RuntimeError("gSDE graph requested more rollout rounds than captured.")
        index = self._cursor if self.per_round else 0
        self._cursor += 1
        return self.collection[index]

    def for_execution(self, root, policy, generator):
        del root, policy, generator
        return self.execution

    def copy_(self, collection, execution):
        if not isinstance(collection, (tuple, list)) or len(collection) != len(self.collection):
            raise ValueError("gSDE graph requires its complete collection matrix inputs.")
        pairs = list(zip(self.collection, collection))
        if self.execution is None:
            if execution is not None:
                raise ValueError("This gSDE graph does not sample execution noise.")
        else:
            pairs.append((self.execution, execution))
        # Validate every tensor before copying any, without reading device data.
        for destination, source in pairs:
            if (not torch.is_tensor(source) or source.shape != destination.shape
                    or source.dtype != destination.dtype or source.device != destination.device):
                raise ValueError("gSDE graph matrix shape, dtype and device must remain fixed.")
        for destination, source in pairs:
            destination.copy_(source)


def _gsde_inputs_for_action(engine, root):
    """Draw on the live behavior streams, independently of captured gradients."""
    collection = []
    generator = engine.rng.generator("collection")
    for round_index in range(int(engine.cfg.inner_rounds)):
        # Explicit generators need no default-RNG fork. Collection/execution
        # contain no implicit draws here; captured optimizer randomness stays
        # on the graph's private stream.
        matrix = engine._gsde_noise_for_round(
            root, int(engine.cfg.inner_rollouts_per_round), generator,
        )
        if matrix is not None and (round_index == 0 or engine.cfg.inner_gsde_noise_scope == "round"):
            collection.append(matrix)
    execution = (
        engine._gsde_noise_for_execution(root, engine._actor_base, engine.rng.generator("execution"))
        if float(engine.cfg.inner_execution_std_scale) > 0.0 else None
    )
    return collection, execution


class WholeInnerGraph:
    """Capture a private fresh solve, including sampled execution and teachers.

    ``capture=False`` uses exactly the same RNG protocol and provides the
    uncaptured arithmetic reference. Outputs own captured storage and must be
    copied before another call. The live adapter below owns those policy copies.
    """

    def __init__(self, engine, root, *, capture=True, seed=8173, warmup=4,
                 capture_teacher_policy=False, return_behavior_policy=False):
        _validate_fresh_config(engine.cfg, require_compile=False)
        if engine.device.type != "cuda" or root.device.type != "cuda":
            raise ValueError("WholeInnerGraph requires a CUDA engine and root.")
        if (root.ndim != 2 or root.shape != (1, int(engine.cfg.latent_dim))
                or root.dtype != torch.float32):
            raise ValueError("WholeInnerGraph requires one float32 [1, latent_dim] root.")
        if not all(isinstance(flag, bool) for flag in
                   (capture, capture_teacher_policy, return_behavior_policy)):
            raise TypeError("Collection graph capture controls must be bool.")
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 3:
            raise ValueError("Collection graph warmup must be an integer of at least three.")
        self.engine, self.device = engine, engine.device
        engine._whole_graph_worker = True
        self.device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        if root.device.index != self.device_index:
            raise ValueError("Collection graph engine and root must use the same CUDA device.")
        if engine._active_trace is not None:
            raise ValueError("Collection graph cannot capture an active trace.")
        self._engine_act = engine.act
        self.capture_enabled = capture
        self.capture_teacher_policy = capture_teacher_policy
        self.return_behavior_policy = return_behavior_policy
        self.calls, self.seed = 0, int(seed)
        self.graph = None
        self.root = root.detach().clone()
        self.gsde_inputs = _GsdeGraphInputs(engine, root) if engine._gsde_active else None
        if self.gsde_inputs is not None:
            self.gsde_inputs.install(engine)
        self._initial_alpha = engine._initial_inner_alpha
        self.alpha_input = self._initial_alpha().detach().clone()
        engine._initial_inner_alpha = lambda: self.alpha_input
        engine.rng = _GraphRNG()
        engine._timer_start = lambda: None
        engine._timer_stop = lambda key, start: None
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
        self.reset_rng()
        self.stream = torch.cuda.Stream(device=self.device)
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        started = time.perf_counter()
        try:
            with torch.cuda.stream(self.stream), self._rng_fork(), _preserve_host_rng_state():
                from torch._dynamo.utils import counters
                for _ in range(warmup):
                    before = int(counters["stats"]["unique_graphs"])
                    self.output = self._solve()
                if int(counters["stats"]["unique_graphs"]) != before:
                    raise RuntimeError("Collection graph compilation did not stabilize before capture.")
                if any(region.failed for region in engine._compile_regions.values()):
                    raise RuntimeError("Collection graph cannot use compiler fallback.")
            self.stream.synchronize()
            self.warmup_seconds = time.perf_counter() - started
            self.capture_seconds = 0.0
            if capture:
                started = time.perf_counter()
                self.graph = torch.cuda.CUDAGraph()
                with self._rng_fork(), _preserve_host_rng_state():
                    with torch.cuda.graph(self.graph, stream=self.stream):
                        self.output = self._solve()
                self.stream.synchronize()
                self.capture_seconds = time.perf_counter() - started
            self.reset_rng()
        except BaseException:
            self.stream.synchronize()
            raise

    @contextmanager
    def _rng_fork(self):
        with torch.cuda.device(self.device), torch.random.fork_rng(devices=[self.device_index]):
            torch.cuda.set_rng_state(self.rng_state, self.device)
            try:
                yield
            finally:
                self.rng_state = torch.cuda.get_rng_state(self.device)

    def reset_rng(self):
        with torch.cuda.device(self.device), torch.random.fork_rng(devices=[self.device_index]):
            torch.cuda.manual_seed(self.seed)
            self.rng_state = torch.cuda.get_rng_state(self.device)

    def _solve(self):
        result = self._engine_act(
            self.root, t0=True, collect_diagnostics=False, eval_mode=False,
            capture_teacher_policy=self.capture_teacher_policy,
            return_behavior_policy=self.return_behavior_policy,
        )
        if self.capture_teacher_policy:
            return (*result, self.engine.last_teacher_policy)
        return result

    def __call__(self, root, *, gsde_collection=None, gsde_execution=None):
        if root.shape != self.root.shape or root.dtype != self.root.dtype or root.device != self.root.device:
            raise ValueError("Collection graph root shape, dtype and device must remain fixed.")
        # Canonical fresh-SAC clones this live outer scalar inside every solve.
        # Its storage is stable and the clone is a captured device operation;
        # inner per-update normalization never writes into the outer estimator.
        if self.engine._sac_actor_loss_scale_enabled:
            scale = self.engine._critic_owner.actor_loss_scale
            valid_scale = (torch.isfinite(scale) & (scale >= 1)).all()
            if self.gsde_inputs is not None:
                torch._assert_async(valid_scale, "Collection graph Q scale must be finite and at least one.")
            elif not bool(valid_scale):
                raise ValueError("Collection graph Q scale must be finite and at least one.")
        if self.gsde_inputs is not None:
            self.gsde_inputs.copy_(gsde_collection, gsde_execution)
        elif gsde_collection is not None or gsde_execution is not None:
            raise ValueError("Gaussian collection graphs do not accept gSDE matrix inputs.")
        alpha = self._initial_alpha()
        self.alpha_input.copy_(alpha)
        self.root.copy_(root)
        with self._rng_fork(), _preserve_host_rng_state():
            if self.graph is None:
                self.output = self._solve()
            else:
                self.graph.replay()
        self.calls += 1
        return self.output


def training_graph_state(engine):
    if engine._whole_graph_runner is None and engine._whole_graph_resume is not None:
        return deepcopy(engine._whole_graph_resume)
    runner = engine._whole_graph_runner
    return {
        "schema": "ambi-training-whole-inner-graph-v1", "protocol": training_graph_protocol(engine.cfg),
        "seed": int(engine.cfg.seed) + 8173,
        "calls": 0 if runner is None else runner.calls,
        "rng_state": None if runner is None else runner.rng_state.detach().cpu().clone(),
    }


def preflight_training_graph_state(engine, saved):
    if not getattr(engine.cfg, "inner_cuda_graph", False):
        if saved is not None:
            raise ValueError("A collection CUDA graph lineage cannot resume in native mode.")
        return None
    validate_collection_graph_config(engine.cfg)
    expected = {"schema", "protocol", "seed", "calls", "rng_state"}
    if not isinstance(saved, dict) or saved.keys() != expected:
        raise ValueError("Collection CUDA graph resume requires its complete RNG state.")
    if (saved["schema"] != "ambi-training-whole-inner-graph-v1"
            or saved["protocol"] != training_graph_protocol(engine.cfg)
            or type(saved["seed"]) is not int or saved["seed"] != int(engine.cfg.seed) + 8173
            or type(saved["calls"]) is not int or saved["calls"] < 0):
        raise ValueError("Incompatible collection CUDA graph RNG protocol or counters.")
    rng = saved["rng_state"]
    if rng is None:
        if saved["calls"] != 0:
            raise ValueError("A used collection CUDA graph must include its RNG state.")
    else:
        if not torch.is_tensor(rng) or rng.dtype != torch.uint8 or rng.ndim != 1:
            raise ValueError("Collection CUDA graph RNG must be a one-dimensional byte tensor.")
        torch.Generator(device=engine.device).set_state(rng.detach().cpu())
    return deepcopy(saved)


def training_graph_act(engine, root, *, t0=False, capture_teacher_policy=False,
                       return_behavior_policy=False):
    """Collect with the production private stream and owned policy metadata."""
    capture_behavior = bool(getattr(engine.cfg, "store_behavior_policy", False)) or return_behavior_policy
    runner = engine._whole_graph_runner
    if (runner is None or runner.capture_teacher_policy != capture_teacher_policy
            or getattr(runner, "return_behavior_policy", False) != capture_behavior):
        from .inner_improvement import InnerImprovementEngine
        saved = training_graph_state(engine) if runner is not None else engine._whole_graph_resume
        worker = InnerImprovementEngine(engine.agent)
        worker._whole_graph_worker = True
        with _preserve_host_rng_state():
            runner = WholeInnerGraph(worker, root, seed=int(engine.cfg.seed) + 8173,
                                     capture_teacher_policy=capture_teacher_policy,
                                     return_behavior_policy=capture_behavior)
        if saved is not None:
            if saved["rng_state"] is not None:
                runner.rng_state = saved["rng_state"].detach().cpu().clone()
            runner.calls = saved["calls"]
        engine._whole_graph_runner = runner
        engine._whole_graph_resume = None
    engine._pending_timers = {}
    started = engine._timer_start()
    if engine._gsde_active:
        engine._begin_gsde_action(t0=t0, eval_mode=False)
        collection, execution = _gsde_inputs_for_action(engine, root)
        result = runner(root, gsde_collection=collection, gsde_execution=execution)
        if execution is not None:
            engine._gsde_execution_uses += 1
        engine._finish_gsde_action()
    else:
        result = runner(root)
    action, values, lengths = result[:3]
    if capture_teacher_policy:
        # Captured outputs are overwritten at the next action. Replay staging
        # and live callers must instead retain their own unscaled statistics.
        teacher_index = 4 if capture_behavior else 3
        engine.last_teacher_policy = {
            key: value.detach().clone() for key, value in result[teacher_index].items()
        }
    engine._timer_stop("inner_action_seconds", started)
    engine.action_index += 1
    engine.state.outer_version = int(engine.agent.outer_version)
    output = action, dict(values, inner_cuda_graph_used=1.0), lengths
    if return_behavior_policy:
        # Behavior statistics describe the executed adapted actor (including
        # execution std scaling), independently of unscaled teacher targets.
        behavior = {key: value.detach().clone() for key, value in result[3].items()}
        return (*output, behavior)
    return output
