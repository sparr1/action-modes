"""Gate and launch the two H3 MPPI training controls on a scheduled GPU.

The gate times complete training decisions at production network/planner shapes,
using isolated 1/2-core subprocesses. It shortens replay collection and pretraining
only; the production launcher always delegates the unmodified recipe to main.py.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_transfer_training import binding, digest, read, recipe, resolve_manifest, runtime, write_new
from slurm.benchmark_ambi_training_cpu import TrainingWindow, physical_core_cpus

MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_mppi_training.json"
CPU_COUNTS = (1, 2)
SEED_STEPS = 500


def campaign_names(manifest_path=MANIFEST):
    names = read(resolve_manifest(manifest_path))["configs"]
    assert len(names) == 2 and len(set(names)) == 2, "Expected exactly two distinct MPPI controls"
    return names


def smoke_recipe(index, warmup, decisions, manifest_path=MANIFEST):
    name, manifest, run = recipe(index, manifest_path)
    manifest, run = deepcopy(manifest), deepcopy(run)
    overrides = {"wandb": False, "seed_steps": SEED_STEPS, "pretrain_steps": 2, "buffer_size": 4096}
    run["alg_params"].update(overrides)
    run["total_steps"] = SEED_STEPS + 1 + warmup + decisions
    # Checkpoint I/O is exercised after the timer's final complete decision.
    run.update(checkpoint_every=run["total_steps"], save_strat="all")
    manifest.update(checkpoint_every=run["total_steps"], save_strat="all", save_trials="none", logs="none")
    return name, manifest, run, overrides


def validate_recipe(cfg, agent, index):
    assert cfg.compile and cfg.compile_strict and cfg.critic_ensemble_backend == "vmap"
    assert cfg.utd == 1 and cfg.inner_operator == "mppi"
    assert (cfg.inner_rollout_horizon, cfg.inner_mppi_iterations, cfg.inner_mppi_num_samples) == (3, 8, 512)
    assert (cfg.inner_mppi_num_elites, cfg.inner_mppi_num_pi_trajs) == (64, 24)
    assert (cfg.inner_mppi_temperature, cfg.inner_mppi_min_std, cfg.inner_mppi_max_std) == (.5, .05, 2.)
    assert cfg.inner_mppi_warm_start_scope == "episode" and cfg.inner_mppi_eval_action == "weighted_elite"
    assert cfg.inner_actor_source == cfg.inner_horizon_actor_source == "sac"
    assert cfg.mppi_terminal_q_reduction == "mean_pair"
    assert cfg.inner_critic_updates_per_action == cfg.inner_actor_updates_per_action == 0
    if index == 0:
        assert cfg.aux_return_mode == "sac" and agent.aux_return is not None and not agent.aux_return.has_actor
        assert cfg.inner_critic_source == cfg.inner_horizon_critic_source == "aux_return"
        assert agent.inner_engine._horizon_critic is agent.model._aux_return_Qs
    else:
        assert cfg.aux_return_mode == "off" and agent.aux_return is None
        assert cfg.inner_critic_source == cfg.inner_horizon_critic_source == "sac"
        assert agent.inner_engine._horizon_critic is agent.model._Qs
        assert not hasattr(agent.model, "_aux_return_Qs")
        assert not any("aux_return" in key for key in agent.model.state_dict())


def worker(args):
    cpus = [int(cpu) for cpu in args.worker_cpus.split(",")]
    assert len(cpus) in CPU_COUNTS and len(cpus) == len(set(cpus))
    assert set(cpus) <= os.sched_getaffinity(0), "Requested CPU is outside this allocation"
    assert len(physical_core_cpus(cpus)) == len(cpus), "Use distinct physical cores"
    os.sched_setaffinity(0, cpus)
    import torch
    import gymnasium as gym
    import domains  # noqa: F401
    from RL.AMBITDMPC2 import AMBITDMPC2

    torch.set_num_threads(len(cpus))
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available(), "Gate requires a Slurm GPU allocation"
    torch.cuda.reset_peak_memory_stats()
    name, manifest, run, overrides = smoke_recipe(args.index, args.warmup, args.decisions, args.manifest)
    production = recipe(args.index, args.manifest)[2]
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    started = time.perf_counter()
    try:
        learner = AMBITDMPC2(name + "-gpu-gate", env, run["alg_params"], run, manifest)
        cfg, agent = learner.cfg, learner.agent
        validate_recipe(cfg, agent, args.index)
        timer = TrainingWindow(SEED_STEPS + args.warmup + 2, args.decisions, torch.cuda.synchronize)
        act, update, log_step = learner._act_agent, agent.update, learner._log_wandb_step
        active = 0
        compile_counts, q_calls, update_seconds, startup = {}, [], [], {}

        def observed_act(*positional, **kwargs):
            nonlocal active
            step = learner._global_step + 1
            if step == timer.first_step:
                compile_counts["before"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
                torch.cuda.synchronize()
                startup["seconds_through_compilation_and_warmup"] = time.perf_counter() - started
            timer.before_action(step, learner._num_updates)
            # Audit the actual first planner call, then restore Q before warmup
            # ends so instrumentation does not perturb measured throughput.
            original_q = agent.model.Q
            def checked_q(*q_args, **q_kwargs):
                selected = q_kwargs.get("qs", agent.model._Qs)
                assert selected is agent.inner_engine._horizon_critic
                assert not q_kwargs.get("target", False) and q_kwargs["reduction"] == "mean_pair"
                q_calls.append(True)
                return original_q(*q_args, **q_kwargs)
            if active == 0:
                agent.model.Q = checked_q
            action_started = time.perf_counter()
            try:
                action = act(*positional, **kwargs)
            finally:
                if active == 0:
                    agent.model.Q = original_q
            timer.after_action(step, time.perf_counter() - action_started)
            metrics = agent.last_inner_metrics
            assert action.device.type == "cpu" and torch.isfinite(action).all()
            for key in ("inner_critic_optimizer_steps", "inner_actor_optimizer_steps", "inner_temperature_optimizer_steps"):
                assert metrics[key] == 0
            assert metrics["inner_mppi_iterations"] == 8 and metrics["inner_model_steps"] == 12336
            assert metrics["inner_compile_fallback"] == 0
            assert all(math.isfinite(float(value)) for value in metrics.values())
            active += 1
            return action

        def observed_update(*positional, **kwargs):
            update_started = time.perf_counter()
            metrics = update(*positional, **kwargs)
            assert metrics["compile_fallback"] == 0
            if timer.first_step <= learner._global_step <= timer.last_step:
                update_seconds.append(time.perf_counter() - update_started)
            if args.index == 1:
                assert not any(key.startswith("aux_return_") for key in metrics)
            return metrics

        def observed_log_step(reward, *positional, **kwargs):
            assert math.isfinite(float(reward))
            result = log_step(reward, *positional, **kwargs)
            timer.after_decision(learner._global_step, learner._num_updates)
            return result

        learner._act_agent, agent.update, learner._log_wandb_step = observed_act, observed_update, observed_log_step
        with tempfile.TemporaryDirectory(prefix=name + "-checkpoint-") as temporary:
            models = Path(temporary) / "models"
            learner.set_checkpointing(run["total_steps"], str(models), "smoke", save_strat="all")
            learner.learn(run["total_steps"])
            learner.flush_checkpoints()
            compile_counts["after"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
            assert compile_counts["before"] == compile_counts["after"], "Compilation inside measured window; increase warmup"
            assert timer.result is not None and active == args.warmup + args.decisions and q_calls
            assert learner._global_step == run["total_steps"] and learner._pretrained
            assert all(torch.isfinite(parameter).all() for parameter in agent.model.parameters())
            assert all(torch.isfinite(torch.as_tensor(value)).all() for value in learner._last_train_metrics.values())
            checkpoint = models / f"smoke_{run['total_steps']}"
            assert checkpoint.is_file() and Path(str(checkpoint) + ".metadata.json").is_file()
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert ("aux_return_state" in saved) == (args.index == 0)
            agent.load(checkpoint)
            for key, value in agent.model.state_dict().items():
                torch.testing.assert_close(value.cpu(), saved["model"][key], rtol=0, atol=0)
            checkpoint_bytes = checkpoint.stat().st_size
        assert torch.get_num_threads() == len(cpus) and os.sched_getaffinity(0) == set(cpus)
        measured = timer.result
        measured["outer_update_seconds"] = statistics.mean(update_seconds)
        measured["projected_production_compute_seconds"] = (
            (production["total_steps"] - production["alg_params"]["seed_steps"]) * measured["seconds_per_decision"]
            + production["alg_params"]["pretrain_steps"] * measured["outer_update_seconds"]
        )
        storage = learner.buffer._buffer._storage._storage
        replay_bytes_per_row = sum(value.numel() * value.element_size() for value in storage.values()) / len(storage)
        report = {"passed": True, "config": name, "index": args.index,
                  "physical_cores": len(cpus), "logical_cpu_ids": cpus,
                  "torch_threads": torch.get_num_threads(), "runtime": runtime(),
                  "binding": binding(args.manifest), "measured": measured,
                  "warmup_active_decisions": args.warmup, "compile_graph_counts": compile_counts,
                  "compile_fallback": False, "finite_actions_and_model": True,
                  "online_terminal_critic_verified": True, "auxiliary_critic_present": args.index == 0,
                  "inner_optimizer_steps": 0, "checkpoint_roundtrip": True,
                  "checkpoint_sidecars": True, "checkpoint_bytes": checkpoint_bytes,
                  "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "startup": startup,
                  "replay_storage_device": str(learner.buffer._storage_device),
                  "replay_bytes_per_row": replay_bytes_per_row,
                  "production_replay_capacity": min(production["total_steps"], production["alg_params"]["buffer_size"]),
                  "estimated_production_replay_bytes": replay_bytes_per_row * min(production["total_steps"], production["alg_params"]["buffer_size"]),
                  "total_seconds": time.perf_counter() - started,
                  "production_overrides": {**overrides, "total_steps": run["total_steps"], "checkpoint_every": run["total_steps"]},
                  "created_utc": datetime.now(timezone.utc).isoformat()}
        write_new(args.output, report)
        print("MPPI_GPU_CASE_PASSED " + json.dumps({"config": name, "cores": len(cpus),
              "measured": {k: v for k, v in measured.items() if not isinstance(v, list)}}), flush=True)
    finally:
        if learner is not None:
            learner.close()
        env.close()


def smoke(args):
    assert os.environ.get("SLURM_JOB_ID"), "Run inside a scheduled GPU job"
    assert int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) >= max(CPU_COUNTS)
    names = campaign_names(args.manifest)
    cores = physical_core_cpus(os.sched_getaffinity(0))
    assert len(cores) >= max(CPU_COUNTS), "Request --hint=nomultithread with two physical cores"
    output = args.output.resolve()
    assert output != ROOT and ROOT not in output.parents, "Keep gate outputs outside source"
    output.mkdir(parents=True, exist_ok=False)
    order = [(index, count) for index in range(len(names)) for count in CPU_COUNTS]
    random.Random(55).shuffle(order)
    write_new(output / "protocol.json", {"binding": binding(args.manifest), "case_order": order,
        "physical_core_representatives": cores, "warmup_active_decisions": args.warmup,
        "measured_decisions": args.decisions, "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "scope": "Short early-replay throughput; excludes startup, compilation, W&B, diagnostics cadence and checkpoint I/O from measured window. Full-size replay and long-run performance are not measured."})
    with tempfile.TemporaryDirectory(prefix="ambi-mppi-gate-") as temporary:
        cache = Path(temporary)
        for index, count in order:
            environment = dict(os.environ)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                environment[key] = str(count)
            environment.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE", WANDB_MODE="disabled",
                               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", MUJOCO_GL="egl",
                               TORCHINDUCTOR_COMPILE_THREADS="1", TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
                               TRITON_CACHE_DIR=str(cache / "triton"))
            stem = f"cell-{index}-cores-{count}"
            command = [sys.executable, str(Path(__file__).resolve()), "smoke", "--worker-cpus", ",".join(map(str, cores[:count])),
                       "--index", str(index), "--manifest", str(resolve_manifest(args.manifest)),
                       "--warmup", str(args.warmup), "--decisions", str(args.decisions), "--output", str(output / (stem + ".json"))]
            with (output / (stem + ".log")).open("x") as log:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            print("MPPI_GPU_GATE_PROGRESS " + stem, flush=True)
    cases = [read(output / f"cell-{index}-cores-{count}.json") for index in range(len(names)) for count in CPU_COUNTS]
    assert all(row["passed"] for row in cases)
    assert len({json.dumps(row["runtime"], sort_keys=True) for row in cases}) == 1
    recommendations = {}
    for index, name in enumerate(names):
        rows = [row for row in cases if row["index"] == index]
        fastest = max(row["measured"]["decisions_per_second"] for row in rows)
        acceptable = [row for row in rows if row["measured"]["decisions_per_second"] >= .95 * fastest]
        recommendations[name] = min(row["physical_cores"] for row in acceptable)
    receipt = {"passed": True, "binding": binding(args.manifest), "runtime": cases[0]["runtime"],
               "cases": cases, "recommended_cpus_within_5pct_of_best": recommendations,
               "estimated_checkpoint_bytes": 20 * sum(next(row["checkpoint_bytes"] for row in cases if row["index"] == index) for index in range(len(names))),
               "created_utc": datetime.now(timezone.utc).isoformat()}
    write_new(output / "receipt.json", receipt)
    print("MPPI_GPU_GATE_PASSED " + str(output / "receipt.json"), flush=True)


def validate_receipt(receipt, campaign_binding, current_runtime, index, cpus):
    assert receipt["passed"] and receipt["binding"] == campaign_binding, "GPU gate does not match source/configuration"
    expected = {(index, count) for index in range(2) for count in CPU_COUNTS}
    assert {(row["index"], row["physical_cores"]) for row in receipt["cases"]} == expected
    assert len(receipt["cases"]) == len(expected) and all(row["passed"] for row in receipt["cases"])
    for key in ("python", "torch", "cuda", "executable", "gpu"):
        assert current_runtime[key] == receipt["runtime"][key], f"GPU gate used another {key}"
    assert (index, cpus) in expected, "Production CPU allocation was not measured"


def launch(args):
    manifest_path = resolve_manifest(args.manifest)
    campaign_names(manifest_path)
    campaign_binding, current_runtime = binding(manifest_path), runtime()
    receipt = read(args.gate)
    validate_receipt(receipt, campaign_binding, current_runtime, args.index, int(os.environ.get("SLURM_CPUS_PER_TASK", 0)))
    name, manifest, run = recipe(args.index, manifest_path)
    if not os.environ.get("WANDB_API_KEY"):
        import netrc
        auth = netrc.netrc().authenticators("api.wandb.ai")
        assert auth and auth[2], "W&B credential unavailable"
    args.output.mkdir(parents=True, exist_ok=False)
    launch_data = {"binding": campaign_binding, "config": name, "run_params": run,
                   "gate_receipt": str(args.gate), "gate_sha256": digest(args.gate),
                   "runtime": current_runtime, "created_utc": datetime.now(timezone.utc).isoformat(),
                   "slurm": {key: os.environ.get(key) for key in
                             ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK")},
                   "wandb_run_id": os.environ["WANDB_RUN_ID"]}
    write_new(args.output / "launch.json", launch_data)
    command = [sys.executable, "main.py", "--run", str(manifest_path), "--alg-dir", "configs/dmcontrol/algs",
               "--log-dir", str(args.output), "--alg-index", str(args.index), "--trial-index", "0", "--num-runs", "1"]
    print("MPPI_TRAINING_LAUNCH " + json.dumps(launch_data), flush=True)
    os.execv(sys.executable, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--index", type=int, choices=range(2))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--decisions", type=int, default=128)
    parser.add_argument("--worker-cpus", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode == "smoke":
        assert args.warmup >= 10 and args.decisions >= 100
        assert SEED_STEPS + 1 + args.warmup + args.decisions < 1000
        if args.worker_cpus:
            assert args.index is not None
            worker(args)
        else:
            smoke(args)
    else:
        assert args.gate is not None and args.index is not None and not args.worker_cpus
        launch(args)


if __name__ == "__main__":
    main()
