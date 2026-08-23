#!/usr/bin/env python3
"""Deterministic, environment-free XQC learner throughput gate.

This benchmark intentionally includes replay sampling and host-to-device batch
transfer, because both occur once per Action Modes interaction. It excludes
DMControl and W&B so a regression points directly at the learner hot path.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RL.sac_core import ReplayBuffer
from RL.xqc_core import XQCAgent, XQCConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measured", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--utd", type=int, default=2)
    parser.add_argument("--replay-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--debug-checks", action="store_true")
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--compile-strict", action="store_true")
    parser.add_argument("--require-compiled", action="store_true")
    parser.add_argument(
        "--optimizer-backend",
        choices=("auto", "single_tensor", "foreach", "fused"),
        default="auto",
    )
    parser.add_argument("--min-updates-per-second", type=float, default=0.0)
    parser.add_argument("--max-projection-residual", type=float, default=1e-6)
    return parser


def _make_replay(size: int, obs_dim: int, action_dim: int, seed: int) -> ReplayBuffer:
    rng = np.random.default_rng(seed)
    replay = ReplayBuffer(obs_dim, action_dim, size)
    replay.obs[:] = rng.standard_normal(replay.obs.shape, dtype=np.float32)
    replay.next_obs[:] = rng.standard_normal(replay.next_obs.shape, dtype=np.float32)
    replay.actions[:] = np.tanh(
        rng.standard_normal(replay.actions.shape, dtype=np.float32)
    )
    replay.rewards[:] = rng.standard_normal(replay.rewards.shape, dtype=np.float32)
    replay.terminated[:] = (
        rng.random(replay.terminated.shape) < 0.02
    ).astype(np.float32)
    replay.pos = 0
    replay.full = True
    return replay


def _floating_state(agent: XQCAgent):
    for module in (agent.actor, agent.critic, agent.critic_target):
        for tensor in (*module.parameters(), *module.buffers()):
            if tensor.is_floating_point():
                yield tensor
    yield agent.log_temperature
    for optimizer in (
        agent.actor_optimizer,
        agent.critic_optimizer,
        agent.temperature_optimizer,
    ):
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value) and value.is_floating_point():
                    yield value


def _all_finite(agent: XQCAgent) -> bool:
    checks_by_device = {}
    for tensor in _floating_state(agent):
        checks_by_device.setdefault(tensor.device, []).append(torch.isfinite(tensor).all())
    return all(
        bool(torch.stack(checks).all().cpu())
        for checks in checks_by_device.values()
    )


def _projection_residual(agent: XQCAgent) -> float:
    residuals = [
        (torch.linalg.vector_norm(weight, dim=1) - 1.0).abs().max()
        for weight in (
            *agent._actor_linear_weights,
            *agent._critic_linear_weights,
        )
    ]
    return float(torch.stack(residuals).max().cpu())


def run_benchmark(args: argparse.Namespace) -> dict:
    if args.warmup < 0 or args.measured <= 0:
        raise ValueError("warmup must be non-negative and measured must be positive")
    if args.batch_size <= 0 or args.utd <= 0 or args.replay_size <= 0:
        raise ValueError("batch-size, utd, and replay-size must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obs_dim, action_dim = (2, 1) if args.tiny else (67, 21)
    actor_arch = (8,) if args.tiny else (256, 256, 256, 256)
    critic_arch = (8,) if args.tiny else (512, 512, 512, 512)
    num_atoms = 5 if args.tiny else 101
    config = XQCConfig(
        actor_net_arch=actor_arch,
        critic_net_arch=critic_arch,
        num_atoms=num_atoms,
        vmin=-2.0 if args.tiny else -5.0,
        vmax=2.0 if args.tiny else 5.0,
        num_interactions=max(1, args.warmup + args.measured),
        updates_per_step=args.utd,
        gradient_steps=args.utd,
        batch_size=args.batch_size,
        reward_normalization=False,
        debug_checks=args.debug_checks,
        compile=args.compile,
        compile_strict=args.compile_strict,
        optimizer_backend=args.optimizer_backend,
        seed=args.seed,
        device=str(device),
        verbose=0,
    )
    agent = XQCAgent(obs_dim, action_dim, config)
    replay = _make_replay(args.replay_size, obs_dim, action_dim, args.seed + 1)

    for _ in range(args.warmup):
        agent.update(replay, args.utd, args.batch_size)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    last_metrics = None
    for _ in range(args.measured):
        last_metrics = agent.update(replay, args.utd, args.batch_size)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    measured_updates = args.measured * args.utd
    projection_residual = _projection_residual(agent)
    finite = _all_finite(agent) and all(
        math.isfinite(value) for value in last_metrics.values()
    )
    result = {
        "torch_version": torch.__version__,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "optimizer_backend": (
            "fused"
            if agent.critic_optimizer.defaults.get("fused")
            else "foreach"
            if agent.critic_optimizer.defaults.get("foreach")
            else "single_tensor"
        ),
        "debug_checks": bool(args.debug_checks),
        "compile_requested": bool(args.compile),
        "compile_strict": bool(args.compile_strict),
        "critic_region_compiled": agent._critic_loss_region._compiled is not None,
        "actor_region_compiled": agent._actor_loss_region._compiled is not None,
        "compile_fallback": bool(
            agent._critic_loss_region.failed or agent._actor_loss_region.failed
        ),
        "warmup_decisions": args.warmup,
        "measured_decisions": args.measured,
        "utd": args.utd,
        "measured_updates": measured_updates,
        "elapsed_seconds": elapsed,
        "decisions_per_second": args.measured / elapsed,
        "updates_per_second": measured_updates / elapsed,
        "projection_residual": projection_residual,
        "all_finite": finite,
        "final_update_step": agent.update_step,
    }
    if agent.update_step != (args.warmup + args.measured) * args.utd:
        raise RuntimeError("XQC benchmark observed an incorrect update count")
    if not finite:
        raise RuntimeError("XQC benchmark produced non-finite learned state")
    if projection_residual > args.max_projection_residual:
        raise RuntimeError(
            "XQC projection residual exceeded the requested maximum: "
            f"{projection_residual} > {args.max_projection_residual}"
        )
    if result["updates_per_second"] < args.min_updates_per_second:
        raise RuntimeError(
            "XQC throughput was below the requested minimum: "
            f"{result['updates_per_second']} < {args.min_updates_per_second}"
        )
    if args.require_compiled and not (
        result["critic_region_compiled"] and result["actor_region_compiled"]
    ):
        raise RuntimeError("XQC compiled learner regions were required but unavailable")
    return result


def main() -> None:
    result = run_benchmark(_parser().parse_args())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
