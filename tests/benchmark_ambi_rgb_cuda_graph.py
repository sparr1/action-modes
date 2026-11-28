"""Time one short, real Humanoid Walk AMBI training arm on an allocated CUDA GPU."""

import argparse
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
from unittest.mock import patch


CONFIG = "ambi_humanoid_walk_mrsq_q5_seed55"
SEED_DECISIONS = 500
PRETRAIN_UPDATES = 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs", choices=("state", "rgb"), required=True)
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--decisions", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 2 or args.decisions < 1:
        parser.error("Use at least 2 warmup decisions and 1 measured decision.")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if not args.output.parent.is_dir():
        parser.error(f"Output directory does not exist: {args.output.parent}")
    return args


def recipe(root, args):
    base = root / "configs" / "dmcontrol"
    run = json.loads((base / "algs" / f"{CONFIG}.json").read_text())
    manifest = json.loads((base / "experiments" / f"{CONFIG}.json").read_text())
    first_step = SEED_DECISIONS + 2 + args.warmup
    last_step = first_step + args.decisions - 1
    overrides = {
        "obs": args.obs,
        "inner_cuda_graph": args.graph,
        "compile_strict": args.strict,
        "seed_steps": SEED_DECISIONS,
        "pretrain_steps": PRETRAIN_UPDATES,
        "wandb": False,
        "eval_freq": None,
        "critic_geometry_diagnostics": False,
        "inner_diagnostics_every": last_step + 1,
    }
    run["alg_params"].update(overrides)
    run.update(total_steps=last_step, checkpoint_every=None, save_strat="none")
    manifest["overrides_alg"]["total_steps"] = last_step
    manifest["env_params"]["obs"] = args.obs
    manifest.update(checkpoint_every=None, save_strat="none", save_trials="none", logs="none")
    return run, manifest, first_step, last_step, overrides


def benchmark(root, args):
    os.environ["WANDB_MODE"] = "disabled"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    sys.path.insert(0, str(root))

    import gymnasium as gym
    import torch

    import domains  # noqa: F401  Registers the project's DMControl adapter.
    from RL.AMBITDMPC2 import AMBITDMPC2
    from RL.tdmpc2_core.common import buffer as replay_module

    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires an allocated CUDA GPU.")
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")

    run, manifest, first_step, last_step, overrides = recipe(root, args)
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    try:
        learner = AMBITDMPC2(CONFIG, env, run["alg_params"], run, manifest)
        cfg, agent = learner.cfg, learner.agent
        assert (cfg.num_q, cfg.inner_rollout_horizon, cfg.inner_rounds, cfg.inner_solve_interval) == (5, 3, 8, 1)
        assert (cfg.batch_size, cfg.train_unroll_horizon, cfg.utd) == (256, 3, 1)
        assert (cfg.inner_rollouts_per_round, cfg.inner_batch_size, cfg.inner_replay_capacity) == (128, 256, 3072)
        assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action) == (128, 32)
        assert cfg.compile and cfg.obs == args.obs and cfg.inner_cuda_graph == args.graph
        assert cfg.critic_ensemble_backend == "vmap" and cfg.inner_execution_action == "policy_sample"

        action_times, environment_times, update_times, decision_times = [], [], [], []
        graph_flags, inner_fallbacks, outer_fallbacks = [], [], []
        steps = []
        started_at = None
        previous = None
        started_updates = None
        graphs_before = None
        runner_calls_before = None
        act, step, update, log_step = learner._act_agent, env.step, agent.update, learner._log_wandb_step

        def compile_count():
            return int(torch._dynamo.utils.counters["stats"]["unique_graphs"])

        def runner_calls():
            runner = getattr(agent.inner_engine, "_whole_graph_runner", None)
            return runner.calls if runner is not None else 0

        def observed_action(*values, **kwargs):
            nonlocal started_at, previous, started_updates, graphs_before, runner_calls_before
            decision = learner._global_step + 1
            if decision == SEED_DECISIONS + 2:
                print("Starting AMBI action warmup", flush=True)
            if decision == first_step:
                torch.cuda.synchronize()
                graphs_before = compile_count()
                runner_calls_before = runner_calls()
                started_updates = learner._num_updates
                print(f"Timing decisions {first_step} through {last_step}", flush=True)
                started_at = previous = time.perf_counter()
            start = time.perf_counter()
            action = act(*values, **kwargs)
            if first_step <= decision <= last_step:
                action_times.append(time.perf_counter() - start)
                assert action.device.type == "cpu" and bool(torch.isfinite(action).all())
                metrics = agent.last_inner_metrics
                graph_flags.append(int(metrics.get("inner_cuda_graph_used", 0)))
                inner_fallbacks.append(int(metrics["inner_compile_fallback"]))
                assert metrics["inner_critic_optimizer_steps"] == 128
                assert metrics["inner_actor_optimizer_steps"] == 32
            return action

        def observed_step(action):
            decision = learner._global_step + 1
            start = time.perf_counter()
            result = step(action)
            if first_step <= decision <= last_step:
                environment_times.append(time.perf_counter() - start)
                assert math.isfinite(float(result[1]))
            return result

        def observed_update(*values, **kwargs):
            start = time.perf_counter()
            metrics = update(*values, **kwargs)
            if first_step <= learner._global_step <= last_step:
                update_times.append(time.perf_counter() - start)
                outer_fallbacks.append(int(metrics["compile_fallback"]))
            return metrics

        def observed_log_step(*values, **kwargs):
            nonlocal previous
            result = log_step(*values, **kwargs)
            decision = learner._global_step
            if first_step <= decision <= last_step:
                if decision == last_step:
                    torch.cuda.synchronize()
                now = time.perf_counter()
                decision_times.append(now - previous)
                previous = now
                steps.append(decision)
            return result

        learner._act_agent = observed_action
        env.step = observed_step
        agent.update = observed_update
        learner._log_wandb_step = observed_log_step

        # A small probe would otherwise select GPU replay. Exercise the ordinary
        # CPU storage and pinned transfer path used by large RGB experiments.
        with patch.object(replay_module, "cuda_mem_get_info", return_value=(0, 0)):
            learner.learn(last_step)

        assert steps == list(range(first_step, last_step + 1))
        assert all(len(samples) == args.decisions for samples in (action_times, environment_times, update_times, graph_flags))
        assert learner._num_updates - started_updates == args.decisions
        assert sum(graph_flags) == runner_calls() - runner_calls_before == args.decisions * int(args.graph)
        assert learner.buffer._storage_device.type == "cpu"
        graph_runner = getattr(agent.inner_engine, "_whole_graph_runner", None)
        graph_captured = graph_runner is not None and graph_runner.graph is not None
        assert graph_captured == args.graph
        assert all(bool(torch.isfinite(parameter).all()) for parameter in agent.model.parameters())
        assert all(bool(torch.isfinite(torch.as_tensor(metric)).all()) for metric in learner._last_train_metrics.values())

        graphs_after = compile_count()
        wall = previous - started_at
        return {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "config": CONFIG,
            "overrides": overrides,
            "task": manifest["env_params"]["task"],
            "seed": int(cfg.seed),
            "gpu": torch.cuda.get_device_name(),
            "host": os.uname().nodename,
            "job": os.environ.get("SLURM_JOB_ID"),
            "torch": torch.__version__,
            "torch_threads": threads,
            "outer_batch_size": int(cfg.batch_size),
            "outer_train_unroll": int(cfg.train_unroll_horizon),
            "inner_batch_size": int(cfg.inner_batch_size),
            "inner_rounds": int(cfg.inner_rounds),
            "inner_rollout_horizon": int(cfg.inner_rollout_horizon),
            "inner_rollouts_per_round": int(cfg.inner_rollouts_per_round),
            "inner_critic_updates_per_action": int(cfg.inner_critic_updates_per_action),
            "inner_actor_updates_per_action": int(cfg.inner_actor_updates_per_action),
            "warmup_active_decisions": args.warmup,
            "measured_decisions": args.decisions,
            "outer_updates_in_window": learner._num_updates - started_updates,
            "graph_captured": graph_captured,
            "graph_solves_in_window": sum(graph_flags),
            "graph_calls_total": runner_calls(),
            "graph_setup_seconds": {
                "warmup": graph_runner.warmup_seconds if graph_captured else None,
                "capture": graph_runner.capture_seconds if graph_captured else None,
            },
            "compile_unique_graphs_before": graphs_before,
            "compile_unique_graphs_after": graphs_after,
            "compiled_during_window": graphs_before != graphs_after,
            "inner_compile_fallback_steps": sum(inner_fallbacks),
            "outer_compile_fallback_steps": sum(outer_fallbacks),
            "replay_device": str(learner.buffer._storage_device),
            "replay_capacity": learner.buffer.capacity,
            "replay_rows": learner.buffer.size,
            "rss_peak_linux_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "seconds_per_decision": wall / args.decisions,
            "median_seconds_per_decision": statistics.median(decision_times),
            "decisions_per_second": args.decisions / wall,
            "projected_training_hours_per_million_decisions": wall / args.decisions * 1_000_000 / 3600,
            "host_timing_seconds_per_decision": {
                "action": statistics.mean(action_times),
                "environment_step": statistics.mean(environment_times),
                "outer_update": statistics.mean(update_times),
            },
            "timing_scope": "Warmed-up early replay, including environment, CPU replay and one outer update per decision. CUDA synchronized at window boundaries; host phase timings assign asynchronous work to the phase in which the host waits.",
        }
    finally:
        if learner is not None:
            learner.close()
        env.close()


def main():
    args = parse_args()
    report = benchmark(Path.cwd(), args)
    data = json.dumps(report, indent=2, allow_nan=False)
    with args.output.open("x") as stream:
        stream.write(data + "\n")
    print(data, flush=True)


if __name__ == "__main__":
    main()
