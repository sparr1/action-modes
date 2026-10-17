"""Bounded separate/joint critic benchmark using resident synthetic batches.

Run from the repository root, for example::

    environments/dmcontrol/.venv/bin/python -m tests.benchmark_aux_return_joint_vmap \
        --device cpu --steps 20 --warmup 3 --repeats 3

No environment rollouts, replay sampling, training jobs, or online logging. The
forward measurement excludes actor/optimizer work; the complete update includes
both actors (when selected), target EMA, and normal diagnostics. With
``--operator mppi``, also measure actual ``agent.act`` calls and a complete
MPPI action plus one outer update. CUDA timings
synchronize before and after each sample; warmup and setup are excluded.
"""

import argparse
from copy import deepcopy
import gc
import json
import math
from numbers import Real
from pathlib import Path
import platform
import statistics
import time

import gymnasium as gym
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent


def _build_cfg(**params):
    """Resolve configuration without importing test or evaluation modules."""
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {
        "seed": 3, "device": "cpu", "env": "test-env", "total_steps": 12,
    }
    algorithm.custom_params = params
    try:
        return algorithm._build_cfg({"device": "cpu", **params})
    finally:
        algorithm.env.close()


def _tiny_params(**overrides):
    """Retain the benchmark's original small-config defaults explicitly."""
    params = {
        "device": "cpu", "model_size": None,
        "enc_dim": 16, "mlp_dim": 16, "latent_dim": 8,
        "num_enc_layers": 2, "simnorm_dim": 4,
        "num_bins": 5, "vmin": -5, "vmax": 5,
        "batch_size": 2, "train_unroll_horizon": 2,
        "outer_planning_horizon": 2, "buffer_size": 32,
        "seed_steps": 4, "pretrain_steps": 1, "utd": 1,
        "compile": False, "episodic": False, "discount": 0.99,
        "wandb": False, "dropout": 0.0,
        "q_representation": "scalar", "num_q": 2,
        "q_num_bins": 5, "q_vmin": -5, "q_vmax": 5,
        "inner_operator": "sac", "inner_rounds": 3,
        "inner_rollouts_per_round": 2, "inner_rollout_horizon": 2,
        "inner_updates_per_round": "auto", "inner_batch_size": 4,
        "inner_replay_capacity": 12,
        "inner_replay_sampling": "with_replacement",
        "inner_actor_adaptation": "clone", "inner_critic_adaptation": "clone",
        "inner_temperature_mode": "auto",
        "inner_temperature_initialization": "inherit_outer",
        "inner_target_entropy": "inherit_outer",
        "inner_critic_target_tau": 0.005,
        "inner_critic_target_update_interval": 1,
        "inner_diagnostic_rollouts": 0,
        "inner_mppi_num_elites": 2, "inner_mppi_num_pi_trajs": 0,
    }
    params.update(overrides)
    return params


def make_agent(args, joint, *, compile_enabled=None):
    compile_enabled = args.compile if compile_enabled is None else compile_enabled
    params = _tiny_params(
        aux_return_mode=args.aux_mode, aux_return_joint_vmap=joint,
        critic_ensemble_backend="vmap", inner_operator=args.operator, inner_rounds=1,
        inner_updates_per_round=1,
        q_representation=args.representation,
        num_q=args.heads if args.representation == "distributional" else 2,
        enc_dim=args.encoder_width or args.width, mlp_dim=args.width, latent_dim=args.latent_dim,
        num_bins=args.bins, q_num_bins=args.bins,
        batch_size=args.batch_size, train_unroll_horizon=args.horizon,
        dropout=args.dropout, ent_coef=.1, aux_return_ent_coef=.1,
        compile=compile_enabled, compile_strict=compile_enabled,
    )
    if args.operator == "mppi":
        for key in ("inner_rounds", "inner_rollouts_per_round", "inner_updates_per_round"):
            params.pop(key)
        params.update(
            inner_rollout_horizon=args.mppi_horizon,
            inner_mppi_iterations=args.mppi_iterations,
            inner_mppi_num_elites=args.mppi_elites,
            inner_mppi_num_pi_trajs=args.mppi_pi_trajs,
            inner_model_step_budget=(
                args.mppi_iterations * args.mppi_samples * args.mppi_horizon
                + args.mppi_pi_trajs * (args.mppi_horizon - 1)
            ),
            inner_temperature_mode="inherit_outer",
            inner_horizon_critic_source="aux_return",
            inner_horizon_actor_source="sac" if args.aux_mode == "sac" else "return_actor",
        )
    cfg = _build_cfg(**params)
    # Resolve configuration using a tiny Gym environment.
    # Replace its shape metadata before constructing any model or optimizer.
    cfg.obs_shape = {"state": (args.obs_dim,)}
    cfg.obs_shapes = [cfg.obs_shape]
    cfg.action_dim = args.action_dim
    cfg.action_dims = [args.action_dim]
    cfg.device = args.device
    return AMBITDMPC2Agent(cfg)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(operation, *, device, steps, warmup, compile_enabled=False):
    for _ in range(warmup):
        operation()
    _synchronize(device)
    baseline_memory = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    from torch._dynamo.utils import counters

    graphs_before = int(counters["stats"]["unique_graphs"])
    started = time.perf_counter()
    for _ in range(steps):
        operation()
    _synchronize(device)
    elapsed = time.perf_counter() - started
    graphs_after = int(counters["stats"]["unique_graphs"])
    if compile_enabled:
        assert graphs_after == graphs_before, (
            "Compilation occurred during the timed window; increase --warmup. "
            f"Unique graphs changed from {graphs_before} to {graphs_after}."
        )
    return {
        "seconds": elapsed, "milliseconds_per_step": 1000. * elapsed / steps,
        "unique_graphs_before_timing": graphs_before,
        "unique_graphs_after_timing": graphs_after,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_incremental_bytes": torch.cuda.max_memory_allocated(device) - baseline_memory if device.type == "cuda" else None,
    }


def _assert_finite(value):
    """Validate outside timed regions so checks do not add GPU synchronizations."""
    if torch.is_tensor(value):
        assert torch.isfinite(value).all(), "Nonfinite benchmark tensor."
    elif isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_finite(item)
    elif isinstance(value, Real):
        assert math.isfinite(value), "Nonfinite benchmark metric."


def benchmark(args):
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable on this host.")
    torch.manual_seed(718)
    initial = make_agent(args, False)
    resolved_configuration = {
        key: getattr(initial.cfg, key, None)
        for key in (
            "enc_dim", "mlp_dim", "latent_dim", "num_enc_layers", "simnorm_dim",
            "num_bins", "vmin", "vmax", "q_num_bins", "q_vmin", "q_vmax",
            "num_q", "dropout", "batch_size", "train_unroll_horizon",
            "critic_coef", "aux_return_critic_coef", "actor_lr", "critic_lr",
            "aux_return_critic_lr", "aux_return_mode", "aux_return_detach_representation",
            "ent_coef", "target_entropy", "outer_actor_entropy_mode",
            "critic_entropy_mode", "outer_q_target_reduction", "outer_q_actor_reduction",
            "inner_operator", "inner_temperature_mode", "inner_horizon_critic_source",
            "inner_horizon_actor_source", "mppi_terminal_q_reduction",
        )
    }
    resolved_configuration["aux_return_actor_cfg"] = deepcopy(initial.cfg.aux_return_actor_cfg)
    with torch.no_grad():
        for name in ("_Qs", "_target_Qs", "_aux_return_Qs", "_target_aux_return_Qs"):
            for head in getattr(initial.model, name):
                head[-1].weight.normal_(std=.1)
    saved = deepcopy(initial.checkpoint_state())
    generator = torch.Generator().manual_seed(811)
    h, b = args.horizon, args.batch_size
    data = (
        torch.randn(h + 1, b, args.obs_dim, generator=generator).to(device),
        torch.randn(h, b, args.action_dim, generator=generator).tanh().to(device),
        torch.randn(h, b, 1, generator=generator).to(device),
        torch.zeros(h, b, 1, device=device),
    )
    latent = torch.randn(h, b, args.latent_dim, generator=generator).to(device)
    auxiliary_latent = latent
    root_observation = data[0][0, 0].cpu()
    del initial
    rows = []
    workloads = ["critic_forward", "complete_update"]
    if args.operator == "mppi":
        workloads.extend(("mppi_action", "mppi_action_and_update"))
    for repeat in range(args.repeats):
        # Alternate order to reduce thermal/warmup bias in the paired comparison.
        for joint in ((False, True) if repeat % 2 == 0 else (True, False)):
            for workload in workloads:
                gc.collect()
                if args.compile and workload != "critic_forward":
                    # Each case owns new modules. Discard guards for previous
                    # instances before warmup; keep the compiler's disk cache.
                    torch._dynamo.reset()
                agent = make_agent(args, joint, compile_enabled=args.compile and workload != "critic_forward")
                agent.load(deepcopy(saved))
                torch.manual_seed(991 + repeat)
                last = {}

                def update():
                    info = agent._update(*data)
                    assert info["aux_return_joint_vmap_online"] == float(joint)
                    assert info["aux_return_joint_vmap_target"] == float(joint)
                    assert not info["compile_fallback"]
                    last["update"] = info

                def act():
                    last["action"] = agent.act(root_observation, collect_diagnostics=False)
                    last["planning"] = agent.last_inner_metrics

                def cycle():
                    act()
                    update()

                if workload == "critic_forward":
                    agent.model.train()

                    @torch.no_grad()
                    def forward():
                        if joint:
                            last["predictions"] = agent.model.joint_q_predictions(latent, data[1], auxiliary_latent, data[1])
                        else:
                            last["predictions"] = (agent.model.q_predictions(latent, data[1]),
                                                   agent.model.aux_return_q_predictions(auxiliary_latent, data[1]))

                    # This isolated primitive benchmark is eager even when the
                    # complete-update compilation option is enabled.
                    operation = forward
                elif workload == "complete_update":
                    operation = update
                elif workload == "mppi_action":
                    operation = act
                else:
                    operation = cycle
                measured = _measure(
                    operation, device=device, steps=args.steps, warmup=args.warmup,
                    compile_enabled=args.compile and workload != "critic_forward",
                )
                _assert_finite(last)
                assert torch.stack([torch.isfinite(p).all() for p in agent.model.parameters()]).all()
                for name, value in last.get("planning", {}).items():
                    if "compile" in name and "fallback" in name:
                        assert not value, f"Planning compilation fallback: {name}"
                row = dict(repeat=repeat, mode="joint" if joint else "separate", workload=workload,
                           joint_online=last.get("update", {}).get("aux_return_joint_vmap_online"),
                           joint_target=last.get("update", {}).get("aux_return_joint_vmap_target"),
                           **measured)
                rows.append(row)
                print(json.dumps(row), flush=True)
                if workload in {"complete_update", "mppi_action_and_update"}:
                    assert agent.num_updates == args.steps + args.warmup
                    assert not agent._outer_update_region.failed
                    assert not agent._joint_target_critic_region.failed
                del operation, agent, last
                gc.collect()
    summary = {}
    for workload in workloads:
        means = {}
        for mode in ("separate", "joint"):
            samples = [row["milliseconds_per_step"] for row in rows if row["workload"] == workload and row["mode"] == mode]
            means[mode] = statistics.mean(samples)
            summary[f"{workload}_{mode}"] = {
                "mean_ms": means[mode], "min_ms": min(samples), "max_ms": max(samples),
                "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.,
            }
        summary[f"{workload}_speedup"] = means["separate"] / means["joint"]
    return dict(
        platform=platform.platform(), torch_version=torch.__version__, device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        cpu_threads=torch.get_num_threads(), configuration={key: value for key, value in vars(args).items() if key != "output"},
        resolved_configuration=resolved_configuration,
        rows=rows, summary=summary,
        limitations="Synthetic resident replay batches and configurable networks; no environment stepping, replay sampling or publication. MPPI action timings include actual agent.act, a CPU observation input, and normal action/metric materialization. The combined cycle has one action and one outer update (UTD=1). Forward-only timings are eager and omit backward and optimizers; other workloads use the requested compilation mode. CUDA allocated-memory totals include the persistent reference checkpoint and input batches; incremental peaks subtract post-warmup allocation. CPU results do not establish GPU speedup. Numerical parity is covered by the integration tests; random draws differ between execution modes.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--encoder-width", type=int, help="Encoder width; defaults to --width.")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--obs-dim", type=int, default=3)
    parser.add_argument("--action-dim", type=int, default=1)
    parser.add_argument("--bins", type=int, default=5, help="Reward and distributional critic bin count.")
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--representation", choices=("scalar", "distributional"), default="distributional")
    parser.add_argument("--aux-mode", choices=("sac", "return_actor"), default="sac")
    parser.add_argument("--operator", choices=("sac", "mppi"), default="sac")
    parser.add_argument("--mppi-horizon", type=int, default=3)
    parser.add_argument("--mppi-samples", type=int, default=128)
    parser.add_argument("--mppi-iterations", type=int, default=2)
    parser.add_argument("--mppi-elites", type=int, default=8)
    parser.add_argument("--mppi-pi-trajs", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=.01)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--compile", action="store_true", help="Strictly compile model/update regions; primitive timing stays eager.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.width, args.encoder_width or args.width, args.latent_dim,
           args.obs_dim, args.action_dim, args.horizon, args.steps, args.repeats, args.threads,
           args.mppi_horizon, args.mppi_samples, args.mppi_iterations, args.mppi_elites) < 1:
        parser.error("Shape dimensions, steps, repeats, and threads must be positive.")
    if args.heads < 2 or args.latent_dim % 4 or args.warmup < 0 or not 0 <= args.dropout < 1:
        parser.error("Require heads >= 2, latent dimension divisible by 4, warmup >= 0, and 0 <= dropout < 1.")
    if args.encoder_width is not None and args.encoder_width < 1:
        parser.error("--encoder-width must be positive when specified.")
    if args.bins < 2 or not 0 <= args.mppi_pi_trajs <= args.mppi_samples or args.mppi_elites > args.mppi_samples:
        parser.error("Require bins >= 2, 0 <= MPPI policy trajectories <= samples, and elites <= samples.")
    result = benchmark(args)
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
