#!/usr/bin/env python3
"""Exact-shape, environment-free AMBI-XQC compute benchmark.

Each iteration performs the compute owned by one steady-state training
decision: one canonical action-local XQC solve followed by one recurrent TOLD
and persistent-XQC update over a fixed synthetic H=3, B=256 batch. DMControl,
replay sampling, logging, evaluation, and checkpoint I/O are deliberately
excluded so eager/compiled differences stay attributable to learner compute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL.AMBIXQC import AMBIXQC
from RL.tdmpc2_core.ambixqc_agent import AMBIXQCAgent


DEFAULT_CONFIG = (
    ROOT / "configs/dmcontrol/algs/ambixqc_humanoid_walk_state.json"
)
PRODUCTION_OBS_DIM = 67
PRODUCTION_ACTION_DIM = 21
PRODUCTION_LATENT_DIM = 512
PRODUCTION_OUTER_BATCH = 256
PRODUCTION_TRAIN_HORIZON = 3
CANONICAL_INNER = {
    "rounds": 2,
    "rollouts_per_round": 32,
    "horizon": 3,
    "updates_per_round": 4,
    "batch_size": 64,
    "replay_capacity": 192,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measured", type=int, default=50)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--compile-strict", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--require-compiled", action="store_true")
    parser.add_argument(
        "--optimizer-backend",
        choices=("auto", "single_tensor", "foreach", "fused"),
        default="auto",
    )
    parser.add_argument("--max-projection-residual", type=float, default=1e-6)
    parser.add_argument("--min-cycles-per-second", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    return parser


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


class _ShapeOnlyHumanoidEnv:
    """Only the spaces and horizon needed to resolve the real AMBI-XQC cfg."""

    observation_type = "state"

    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(PRODUCTION_OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(PRODUCTION_ACTION_DIM,),
            dtype=np.float32,
        )
        self.spec = SimpleNamespace(max_episode_steps=500)

    def get_wrapper_attr(self, name):
        if name == "observation_type":
            return self.observation_type
        if name == "_max_episode_steps":
            return self.spec.max_episode_steps
        raise AttributeError(name)


class _SyntheticOuterBuffer:
    """A fixed device-resident H3/B256 batch with the real tensor contract."""

    def __init__(self, cfg, *, generator: torch.Generator):
        horizon = int(cfg.train_unroll_horizon)
        batch = int(cfg.batch_size)
        device = torch.device(cfg.device)
        obs_shape = (horizon + 1, batch, PRODUCTION_OBS_DIM)
        action_shape = (horizon, batch, PRODUCTION_ACTION_DIM)
        reward_shape = (horizon, batch, 1)
        self.obs = torch.randn(
            obs_shape, device=device, dtype=torch.float32, generator=generator
        )
        self.action = torch.randn(
            action_shape, device=device, dtype=torch.float32, generator=generator
        ).tanh()
        self.reward = torch.randn(
            reward_shape, device=device, dtype=torch.float32, generator=generator
        )
        self.terminated = torch.zeros(
            reward_shape, device=device, dtype=torch.float32
        )

    def sample(self):
        return self.obs, self.action, self.reward, self.terminated, None


class _CycleTimer:
    def __init__(self, device: torch.device):
        self.device = device

    def start(self):
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def stop(self, start):
        if self.device.type == "cuda":
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            return start, end
        return time.perf_counter() - start

    def seconds(self, token) -> float:
        if self.device.type == "cuda":
            start, end = token
            return start.elapsed_time(end) / 1000.0
        return float(token)


def _build_cfg(config: dict, args: argparse.Namespace, device: torch.device):
    params = dict(config["alg_params"])
    params["compile"] = bool(args.compile)
    params["compile_strict"] = bool(args.compile_strict)
    params["xqc_optimizer_backend"] = args.optimizer_backend
    seed = int(config["seed"] if args.seed is None else args.seed)

    algorithm = object.__new__(AMBIXQC)
    algorithm.env = _ShapeOnlyHumanoidEnv()
    algorithm.run_params = {
        "seed": seed,
        "device": str(device),
        "env": config["env"],
        "total_steps": int(config["total_steps"]),
    }
    algorithm.experiment_params = {}
    algorithm.custom_params = params
    cfg = algorithm._build_cfg(params)
    return cfg


def _validate_exact_shape(cfg) -> None:
    actual = {
        "obs_dim": int(cfg.obs_shape["state"][0]),
        "action_dim": int(cfg.action_dim),
        "latent_dim": int(cfg.latent_dim),
        "outer_batch": int(cfg.batch_size),
        "train_horizon": int(cfg.train_unroll_horizon),
        "actor_arch": tuple(cfg.xqc_actor_net_arch),
        "critic_arch": tuple(cfg.xqc_critic_net_arch),
        "atoms": int(cfg.xqc_num_atoms),
        "rounds": int(cfg.inner_rounds),
        "rollouts_per_round": int(cfg.inner_rollouts_per_round),
        "inner_horizon": int(cfg.inner_rollout_horizon),
        "updates_per_round": int(cfg.inner_updates_per_round),
        "inner_batch": int(cfg.inner_batch_size),
        "replay_capacity": int(cfg.inner_replay_capacity),
    }
    expected = {
        "obs_dim": PRODUCTION_OBS_DIM,
        "action_dim": PRODUCTION_ACTION_DIM,
        "latent_dim": PRODUCTION_LATENT_DIM,
        "outer_batch": PRODUCTION_OUTER_BATCH,
        "train_horizon": PRODUCTION_TRAIN_HORIZON,
        "actor_arch": (256, 256, 256, 256),
        "critic_arch": (512, 512, 512, 512),
        "atoms": 101,
        "rounds": CANONICAL_INNER["rounds"],
        "rollouts_per_round": CANONICAL_INNER["rollouts_per_round"],
        "inner_horizon": CANONICAL_INNER["horizon"],
        "updates_per_round": CANONICAL_INNER["updates_per_round"],
        "inner_batch": CANONICAL_INNER["batch_size"],
        "replay_capacity": CANONICAL_INNER["replay_capacity"],
    }
    if actual != expected:
        raise ValueError(
            "AMBI-XQC compute benchmark requires the exact production shape: "
            f"expected {expected}, got {actual}"
        )


def _controller_snapshot(controller):
    return {
        key: value.detach().clone()
        for key, value in controller.state_dict().items()
        if torch.is_tensor(value)
    }


def _snapshot_equal(controller, snapshot) -> bool:
    current = controller.state_dict()
    return current.keys() == snapshot.keys() and all(
        torch.equal(current[key], value) for key, value in snapshot.items()
    )


def _snapshot_changed(controller, snapshot) -> bool:
    current = controller.state_dict()
    return any(not torch.equal(current[key], value) for key, value in snapshot.items())


def _global_rng_state(device: torch.device):
    state = {"cpu": torch.random.get_rng_state().clone()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device).clone()
    return state


def _rng_equal(left, right) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[key], right[key]) for key in left
    )


def _floating_state(agent: AMBIXQCAgent):
    for tensor in (*agent.parameters(), *agent.buffers()):
        if tensor.is_floating_point():
            yield tensor
    optimizers = [
        agent.world_optimizer,
        agent.xqc_workspace.actor_optimizer,
        agent.xqc_workspace.critic_optimizer,
        agent.xqc_workspace.temperature_optimizer,
    ]
    inner = agent.inner_engine._workspace_pool
    if inner is not None:
        optimizers.extend(
            (
                inner.actor_optimizer,
                inner.critic_optimizer,
                inner.temperature_optimizer,
            )
        )
    for optimizer in optimizers:
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value) and value.is_floating_point():
                    yield value


def _all_finite(agent: AMBIXQCAgent) -> bool:
    checks = [torch.isfinite(tensor).all() for tensor in _floating_state(agent)]
    return bool(torch.stack(checks).all().cpu()) if checks else True


def _projection_residual(agent: AMBIXQCAgent) -> float:
    controllers = [agent.xqc_controller]
    if agent.inner_engine._workspace_pool is not None:
        controllers.append(agent.inner_engine._workspace_pool.controller)
    residuals = []
    for controller in controllers:
        residuals.extend(
            (torch.linalg.vector_norm(weight, dim=1) - 1.0).abs().max()
            for weight in (
                *controller._actor_linear_weights,
                *controller._critic_linear_weights,
            )
        )
    return float(torch.stack(residuals).max().cpu())


def _optimizer_backend(optimizer) -> str:
    if optimizer.defaults.get("fused"):
        return "fused"
    if optimizer.defaults.get("foreach"):
        return "foreach"
    return "single_tensor"


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _source_sha() -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive(path: Path, result: dict) -> None:
    path = path.expanduser().resolve()
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {path.parent}")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _run_cycle(agent, buffer, observation, timer):
    start = timer.start()
    action = agent.act(observation, collect_diagnostics=False)
    outer_metrics = agent.update(buffer)
    return timer.stop(start), action, outer_metrics


def run_benchmark(args: argparse.Namespace) -> dict:
    if args.warmup < 2:
        raise ValueError(
            "warmup must be at least two to isolate cold timing and lifecycle checks"
        )
    if args.measured <= 0:
        raise ValueError("measured must be positive")
    if args.compile_strict and not args.compile:
        raise ValueError("compile-strict requires compile")
    if args.require_compiled and not args.compile:
        raise ValueError("require-compiled requires compile")
    if args.max_projection_residual < 0 or args.min_cycles_per_second < 0:
        raise ValueError("benchmark thresholds must be non-negative")

    config_path = args.config.expanduser().resolve()
    config = _load_json(config_path)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("AMBI-XQC compute benchmark supports CPU or CUDA")

    seed = int(config["seed"] if args.seed is None else args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    cfg = _build_cfg(config, args, device)
    _validate_exact_shape(cfg)
    agent = AMBIXQCAgent(cfg)
    synthetic_generator = torch.Generator(device=device)
    synthetic_generator.manual_seed(seed + 1_000_003)
    buffer = _SyntheticOuterBuffer(cfg, generator=synthetic_generator)
    observation = torch.randn(
        PRODUCTION_OBS_DIM,
        dtype=torch.float32,
        device=device,
        generator=synthetic_generator,
    )
    timer = _CycleTimer(device)

    warmup_seconds = []
    outer_prior_preserved = False
    outer_update_changed_prior = False
    global_rng_preserved = True
    workspace_ids = []
    reset_counters_fresh = True
    cold_wall_seconds = None
    for index in range(args.warmup):
        prior_before_action = (
            _controller_snapshot(agent.xqc_controller) if index == 1 else None
        )
        rng_before = _global_rng_state(device)
        if index == 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            cold_wall_start = time.perf_counter()
            token, action, _ = _run_cycle(agent, buffer, observation, timer)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            cold_wall_seconds = time.perf_counter() - cold_wall_start
        elif index == 1:
            start = timer.start()
            action = agent.act(observation, collect_diagnostics=False)
            outer_prior_preserved = _snapshot_equal(
                agent.xqc_controller, prior_before_action
            )
            prior_after_action = _controller_snapshot(agent.xqc_controller)
            agent.update(buffer)
            outer_update_changed_prior = _snapshot_changed(
                agent.xqc_controller, prior_after_action
            )
            token = timer.stop(start)
        else:
            token, action, _ = _run_cycle(agent, buffer, observation, timer)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        warmup_seconds.append(timer.seconds(token))
        global_rng_preserved &= _rng_equal(rng_before, _global_rng_state(device))
        workspace = agent.inner_engine._workspace_pool
        workspace_ids.append(id(workspace))
        reset_counters_fresh &= bool(
            workspace is not None
            and workspace.update_step == 8
            and agent.last_inner_metrics["inner_update_slots"] == 8.0
        )
        if not bool(torch.isfinite(action).all()):
            raise RuntimeError("warmup produced a non-finite action")

    workspace_reused = len(set(workspace_ids)) == 1
    if not outer_prior_preserved or not outer_update_changed_prior:
        raise RuntimeError("outer-prior lifecycle checks failed during warmup")
    if not workspace_reused or not reset_counters_fresh:
        raise RuntimeError("action-local workspace reset/reuse checks failed")
    if not global_rng_preserved:
        raise RuntimeError("AMBI-XQC compute advanced global Torch RNG state")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    measured_tokens = []
    inner_rollout_seconds = []
    last_outer_metrics = None
    for _ in range(args.measured):
        token, action, last_outer_metrics = _run_cycle(
            agent, buffer, observation, timer
        )
        measured_tokens.append(token)
        rollout_seconds = float(
            agent.last_inner_metrics["inner_rollout_seconds"]
        )
        if not math.isfinite(rollout_seconds) or rollout_seconds <= 0.0:
            raise RuntimeError(
                "measured iteration produced an invalid inner rollout timing"
            )
        inner_rollout_seconds.append(rollout_seconds)
        if not bool(torch.isfinite(action).all()):
            raise RuntimeError("measured iteration produced a non-finite action")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    iteration_seconds = [timer.seconds(token) for token in measured_tokens]
    elapsed = sum(iteration_seconds)

    inner = agent.inner_engine._workspace_pool
    if inner is None:
        raise RuntimeError("benchmark ended without a reusable inner workspace")
    outer_compile = dict(agent.xqc_controller.compile_status)
    inner_compile = dict(inner.controller.compile_status)
    inner_rollout_compile = dict(agent.inner_engine.rollout_compile_status)
    compile_fallback = bool(
        outer_compile["fallback"]
        or inner_compile["fallback"]
        or inner_rollout_compile["fallback"]
    )
    compiled_regions = bool(
        outer_compile["critic_compiled"]
        and outer_compile["actor_compiled"]
        and inner_compile["critic_compiled"]
        and inner_compile["actor_compiled"]
        and inner_rollout_compile["compiled"]
    )
    if compile_fallback:
        raise RuntimeError("AMBI-XQC benchmark observed a compile fallback")
    if args.require_compiled and not compiled_regions:
        raise RuntimeError("all five AMBI-XQC compiled regions were required")

    expected_cycles = args.warmup + args.measured
    counters = {
        "inner_action_index": int(agent.inner_engine.action_index),
        "outer_num_updates": int(agent.num_updates),
        "outer_version": int(agent.outer_version),
        "outer_xqc_update_step": int(agent.xqc_workspace.update_step),
        "last_inner_update_slots": int(
            agent.last_inner_metrics["inner_update_slots"]
        ),
        "last_inner_actor_steps": int(
            agent.last_inner_metrics["inner_actor_optimizer_steps"]
        ),
        "last_inner_temperature_steps": int(
            agent.last_inner_metrics["inner_temperature_optimizer_steps"]
        ),
        "last_inner_model_steps": int(
            agent.last_inner_metrics["inner_model_steps"]
        ),
        "last_inner_replay_draws": int(
            agent.last_inner_metrics["inner_replay_draws"]
        ),
    }
    expected_counters = {
        "inner_action_index": expected_cycles,
        "outer_num_updates": expected_cycles,
        "outer_version": expected_cycles,
        "outer_xqc_update_step": expected_cycles,
        "last_inner_update_slots": 8,
        "last_inner_actor_steps": 3,
        "last_inner_temperature_steps": 3,
        "last_inner_model_steps": 192,
        "last_inner_replay_draws": 512,
    }
    if counters != expected_counters:
        raise RuntimeError(
            f"AMBI-XQC benchmark counters diverged: {counters} != {expected_counters}"
        )

    rollout_lengths = list(agent.last_inner_rollout_lengths)
    exact_inner_workload = {
        "rounds": int(agent.last_inner_metrics["inner_rounds"]),
        "requested_rollouts": int(
            agent.last_inner_metrics["inner_requested_rollouts"]
        ),
        "realized_rollouts": int(agent.last_inner_metrics["inner_rollouts"]),
        "rollout_horizon_min": int(
            agent.last_inner_metrics["inner_rollout_len_min"]
        ),
        "rollout_horizon_max": int(
            agent.last_inner_metrics["inner_rollout_len_max"]
        ),
        "model_step_budget": int(
            agent.last_inner_metrics["inner_model_steps_budget"]
        ),
        "realized_model_steps": int(
            agent.last_inner_metrics["inner_realized_model_steps"]
        ),
        "replay_size": int(agent.last_inner_metrics["inner_buffer_size"]),
        "replay_capacity": int(
            agent.last_inner_metrics["inner_buffer_capacity"]
        ),
        "replay_draws": int(agent.last_inner_metrics["inner_replay_draws"]),
        "policy_evaluations": int(
            agent.last_inner_metrics["inner_policy_evaluations"]
        ),
        "q_evaluations": int(
            agent.last_inner_metrics["inner_q_evaluations"]
        ),
        "update_slots": int(agent.last_inner_metrics["inner_update_slots"]),
        "requested_update_slots": int(
            agent.last_inner_metrics["inner_requested_update_slots"]
        ),
    }
    expected_inner_workload = {
        "rounds": 2,
        "requested_rollouts": 64,
        "realized_rollouts": 64,
        "rollout_horizon_min": 3,
        "rollout_horizon_max": 3,
        "model_step_budget": 192,
        "realized_model_steps": 192,
        "replay_size": 192,
        "replay_capacity": 192,
        "replay_draws": 512,
        "policy_evaluations": 1217,
        "q_evaluations": 2560,
        "update_slots": 8,
        "requested_update_slots": 8,
    }
    if exact_inner_workload != expected_inner_workload:
        raise RuntimeError(
            "AMBI-XQC benchmark inner workload diverged: "
            f"{exact_inner_workload} != {expected_inner_workload}"
        )
    if rollout_lengths != [PRODUCTION_TRAIN_HORIZON] * 64:
        raise RuntimeError(
            "AMBI-XQC benchmark did not realize exactly 64 full H=3 rollouts"
        )
    replay_pool = agent.inner_engine._replay_pool
    exact_replay_state = (
        None
        if replay_pool is None
        else {
            "size": int(replay_pool.size),
            "capacity": int(replay_pool.capacity),
            "pos": int(replay_pool.pos),
            "full": bool(replay_pool.full),
            "next_sample_id": int(replay_pool.next_sample_id),
        }
    )
    expected_replay_state = {
        "size": 192,
        "capacity": 192,
        "pos": 0,
        "full": True,
        "next_sample_id": 192,
    }
    if exact_replay_state != expected_replay_state:
        raise RuntimeError(
            "AMBI-XQC benchmark replay state diverged: "
            f"{exact_replay_state} != {expected_replay_state}"
        )

    finite = _all_finite(agent) and all(
        math.isfinite(float(value.detach().cpu() if torch.is_tensor(value) else value))
        for value in last_outer_metrics.values()
    )
    if not finite:
        raise RuntimeError("AMBI-XQC benchmark produced non-finite learned state")
    projection_residual = _projection_residual(agent)
    if projection_residual > args.max_projection_residual:
        raise RuntimeError(
            "AMBI-XQC projection residual exceeded the requested maximum: "
            f"{projection_residual} > {args.max_projection_residual}"
        )
    throughput = args.measured / elapsed
    if throughput < args.min_cycles_per_second:
        raise RuntimeError(
            "AMBI-XQC throughput was below the requested minimum: "
            f"{throughput} < {args.min_cycles_per_second}"
        )

    peak_allocated = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(device))
        if device.type == "cuda"
        else 0
    )
    result = {
        "schema": "ambixqc-exact-compute-benchmark-v2",
        "source_sha": _source_sha(),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _file_sha256(config_path),
        "source_config": config,
        "runtime_overrides": {
            "device": str(device),
            "seed": seed,
            "compile": bool(args.compile),
            "compile_strict": bool(args.compile_strict),
            "optimizer_backend": args.optimizer_backend,
        },
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "seed": seed,
        "compile_requested": bool(args.compile),
        "compile_strict": bool(args.compile_strict),
        "outer_compile_status": outer_compile,
        "inner_compile_status": inner_compile,
        "inner_rollout_compile_status": inner_rollout_compile,
        "all_five_regions_compiled": compiled_regions,
        "compile_fallback": compile_fallback,
        "optimizer_backend_requested": args.optimizer_backend,
        "outer_optimizer_backend": _optimizer_backend(
            agent.xqc_workspace.critic_optimizer
        ),
        "inner_optimizer_backend": _optimizer_backend(inner.critic_optimizer),
        "workload": {
            "observation_dim": PRODUCTION_OBS_DIM,
            "action_dim": PRODUCTION_ACTION_DIM,
            "latent_dim": PRODUCTION_LATENT_DIM,
            "outer_batch_size": PRODUCTION_OUTER_BATCH,
            "train_unroll_horizon": PRODUCTION_TRAIN_HORIZON,
            "inner": CANONICAL_INNER,
        },
        "warmup_cycles": args.warmup,
        "cold_cycle_seconds": cold_wall_seconds,
        "cold_wall_seconds": cold_wall_seconds,
        "cold_cuda_event_seconds": warmup_seconds[0],
        "warmup_steady_p50_seconds": _percentile(warmup_seconds[1:], 0.50),
        "warmup_steady_p95_seconds": _percentile(warmup_seconds[1:], 0.95),
        "measured_cycles": args.measured,
        "iteration_seconds": iteration_seconds,
        "iteration_p50_seconds": _percentile(iteration_seconds, 0.50),
        "iteration_p95_seconds": _percentile(iteration_seconds, 0.95),
        "inner_rollout_seconds": inner_rollout_seconds,
        "inner_rollout_p50_seconds": _percentile(
            inner_rollout_seconds, 0.50
        ),
        "inner_rollout_p95_seconds": _percentile(
            inner_rollout_seconds, 0.95
        ),
        "elapsed_seconds": elapsed,
        "cycles_per_second": throughput,
        "actions_per_second": throughput,
        "outer_updates_per_second": throughput,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "outer_prior_preserved_by_inner_action": outer_prior_preserved,
        "outer_prior_changed_by_outer_update": outer_update_changed_prior,
        "workspace_reused": workspace_reused,
        "workspace_reset_counters_fresh": reset_counters_fresh,
        "global_torch_rng_preserved": global_rng_preserved,
        "counters": counters,
        "exact_inner_workload": exact_inner_workload,
        "exact_replay_state": exact_replay_state,
        "rollout_lengths": rollout_lengths,
        "projection_residual": projection_residual,
        "all_finite": finite,
    }
    if args.output is not None:
        _write_exclusive(args.output, result)
    return result


def main() -> None:
    result = run_benchmark(_parser().parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
