"""Validate and launch the single H3/J8 soft-only SAC training control."""

import argparse
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
from slurm.ambi_mppi_training import smoke_recipe as _short_recipe
from slurm.benchmark_ambi_training_cpu import TrainingWindow, physical_core_cpus

MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_soft_training.json"
CPU_COUNTS = (1, 2, 4)
SEED_STEPS = 500


def campaign_name(manifest_path=MANIFEST):
    names = read(resolve_manifest(manifest_path))["configs"]
    assert names == ["ambi_full_h3_j8_soft"], "Expected the one-cell H3/J8 soft-SAC control"
    return names[0]


def smoke_recipe(warmup, decisions, manifest_path=MANIFEST):
    return _short_recipe(0, warmup, decisions, manifest_path)


def validate_recipe(cfg, agent):
    assert cfg.compile and cfg.compile_strict and cfg.critic_ensemble_backend == "vmap"
    assert cfg.utd == 1 and cfg.inner_operator == "sac"
    assert (cfg.inner_rounds, cfg.inner_rollout_horizon, cfg.inner_critic_bernoulli_p) == (8, 3, 0.)
    assert (cfg.inner_rollouts_per_round, cfg.inner_batch_size, cfg.inner_replay_capacity) == (128, 256, 3072)
    assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action) == (128, 32)
    assert cfg.inner_execution_action == "policy_sample" and cfg.inner_finite_horizon
    assert cfg.inner_sac_critic_target == cfg.outer_critic_target == "entropy_augmented"
    assert cfg.inner_actor_source == cfg.inner_horizon_actor_source == "sac"
    assert cfg.inner_critic_source == cfg.inner_horizon_critic_source == "sac"
    assert cfg.inner_actor_initialization == cfg.inner_critic_initialization == "prior"
    assert cfg.aux_return_mode == "off" and agent.aux_return is None
    assert agent.inner_engine._horizon_critic is agent.model._Qs
    assert agent.inner_engine._critic_base is agent.model._Qs
    assert not hasattr(agent.model, "_aux_return_Qs")
    assert not any("aux_return" in key for key in agent.model.state_dict())


def validate_inner_metrics(metrics):
    """Fresh initialization omits Bernoulli-only diagnostics when p is zero."""
    assert metrics["inner_critic_optimizer_steps"] == 128
    assert metrics["inner_actor_optimizer_steps"] == 32
    assert metrics["inner_compile_fallback"] == 0
    assert metrics.get("inner_critic_bernoulli_applied", 0.) == 0
    assert all(math.isfinite(float(value)) for value in metrics.values())


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
    name, manifest, run, overrides = smoke_recipe(args.warmup, args.decisions, args.manifest)
    production = recipe(0, args.manifest)[2]
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    started = time.perf_counter()
    try:
        learner = AMBITDMPC2(name + "-gpu-gate", env, run["alg_params"], run, manifest)
        cfg, agent = learner.cfg, learner.agent
        validate_recipe(cfg, agent)
        timer = TrainingWindow(SEED_STEPS + args.warmup + 2, args.decisions, torch.cuda.synchronize)
        act, update, log_step = learner._act_agent, agent.update, learner._log_wandb_step
        active = 0
        compile_counts, update_seconds, startup = {}, [], {}

        def observed_act(*positional, **kwargs):
            nonlocal active
            step = learner._global_step + 1
            if step == timer.first_step:
                compile_counts["before"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
                torch.cuda.synchronize()
                startup["seconds_through_compilation_and_warmup"] = time.perf_counter() - started
            timer.before_action(step, learner._num_updates)
            action_started = time.perf_counter()
            action = act(*positional, **kwargs)
            timer.after_action(step, time.perf_counter() - action_started)
            metrics = agent.last_inner_metrics
            assert action.device.type == "cpu" and torch.isfinite(action).all()
            validate_inner_metrics(metrics)
            active += 1
            return action

        def observed_update(*positional, **kwargs):
            update_started = time.perf_counter()
            metrics = update(*positional, **kwargs)
            assert metrics["compile_fallback"] == 0
            assert not any(key.startswith("aux_return_") for key in metrics)
            if timer.first_step <= learner._global_step <= timer.last_step:
                update_seconds.append(time.perf_counter() - update_started)
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
            assert timer.result is not None and active == args.warmup + args.decisions
            assert learner._global_step == run["total_steps"] and learner._pretrained
            assert all(torch.isfinite(parameter).all() for parameter in agent.model.parameters())
            assert all(torch.isfinite(torch.as_tensor(value)).all() for value in learner._last_train_metrics.values())
            checkpoint = models / f"smoke_{run['total_steps']}"
            assert checkpoint.is_file() and Path(str(checkpoint) + ".metadata.json").is_file()
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert "aux_return_state" not in saved
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
        bytes_per_row = sum(value.numel() * value.element_size() for value in storage.values()) / len(storage)
        production_capacity = min(production["total_steps"], production["alg_params"]["buffer_size"])
        report = {"passed": True, "config": name, "physical_cores": len(cpus), "logical_cpu_ids": cpus,
                  "torch_threads": torch.get_num_threads(), "runtime": runtime(), "binding": binding(args.manifest),
                  "measured": measured, "warmup_active_decisions": args.warmup, "compile_graph_counts": compile_counts,
                  "compile_fallback": False, "finite_actions_and_model": True, "auxiliary_critic_present": False,
                  "critic_updates_per_action": 128, "actor_updates_per_action": 32, "bernoulli_transfer": False,
                  "checkpoint_roundtrip": True, "checkpoint_sidecars": True, "checkpoint_bytes": checkpoint_bytes,
                  "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(), "startup": startup,
                  "replay_storage_device": str(learner.buffer._storage_device), "replay_bytes_per_row": bytes_per_row,
                  "production_replay_capacity": production_capacity,
                  "estimated_production_replay_bytes": bytes_per_row * production_capacity,
                  "total_seconds": time.perf_counter() - started,
                  "production_overrides": {**overrides, "total_steps": run["total_steps"], "checkpoint_every": run["total_steps"]},
                  "created_utc": datetime.now(timezone.utc).isoformat()}
        write_new(args.output, report)
        print("SOFT_SAC_GPU_CASE_PASSED " + json.dumps({"config": name, "cores": len(cpus),
              "measured": {k: v for k, v in measured.items() if not isinstance(v, list)}}), flush=True)
    finally:
        if learner is not None:
            learner.close()
        env.close()


def smoke(args):
    assert os.environ.get("SLURM_JOB_ID"), "Run inside a scheduled GPU job"
    assert int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) >= max(CPU_COUNTS)
    name = campaign_name(args.manifest)
    cores = physical_core_cpus(os.sched_getaffinity(0))
    assert len(cores) >= max(CPU_COUNTS), "Request --hint=nomultithread with four physical cores"
    output = args.output.resolve()
    assert output != ROOT and ROOT not in output.parents, "Keep gate outputs outside source"
    output.mkdir(parents=True, exist_ok=False)
    order = list(CPU_COUNTS)
    random.Random(55).shuffle(order)
    write_new(output / "protocol.json", {"binding": binding(args.manifest), "case_order": order,
        "physical_core_representatives": cores, "warmup_active_decisions": args.warmup,
        "measured_decisions": args.decisions, "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "scope": "Short early-replay throughput; excludes startup, compilation, W&B, diagnostics cadence and checkpoint I/O from measured window. Full-size replay and long-run performance are not measured."})
    with tempfile.TemporaryDirectory(prefix="ambi-soft-gate-") as temporary:
        cache = Path(temporary)
        for count in order:
            environment = dict(os.environ)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                environment[key] = str(count)
            environment.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE", WANDB_MODE="disabled",
                               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", MUJOCO_GL="egl",
                               TORCHINDUCTOR_COMPILE_THREADS="1", TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
                               TRITON_CACHE_DIR=str(cache / "triton"))
            stem = f"cores-{count}"
            command = [sys.executable, str(Path(__file__).resolve()), "smoke", "--worker-cpus", ",".join(map(str, cores[:count])),
                       "--manifest", str(resolve_manifest(args.manifest)), "--warmup", str(args.warmup),
                       "--decisions", str(args.decisions), "--output", str(output / (stem + ".json"))]
            with (output / (stem + ".log")).open("x") as log:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            print("SOFT_SAC_GPU_GATE_PROGRESS " + stem, flush=True)
    cases = [read(output / f"cores-{count}.json") for count in CPU_COUNTS]
    assert all(row["passed"] for row in cases)
    assert len({json.dumps(row["runtime"], sort_keys=True) for row in cases}) == 1
    fastest = max(row["measured"]["decisions_per_second"] for row in cases)
    acceptable = [row for row in cases if row["measured"]["decisions_per_second"] >= .95 * fastest]
    receipt = {"passed": True, "binding": binding(args.manifest), "runtime": cases[0]["runtime"], "cases": cases,
               "recommended_cpus_within_5pct_of_best": {name: min(row["physical_cores"] for row in acceptable)},
               "estimated_checkpoint_bytes": 20 * max(row["checkpoint_bytes"] for row in cases),
               "created_utc": datetime.now(timezone.utc).isoformat()}
    write_new(output / "receipt.json", receipt)
    print("SOFT_SAC_GPU_GATE_PASSED " + str(output / "receipt.json"), flush=True)


def validate_receipt(receipt, campaign_binding, current_runtime, cpus):
    assert receipt["passed"] and receipt["binding"] == campaign_binding, "GPU gate does not match source/configuration"
    assert {row["physical_cores"] for row in receipt["cases"]} == set(CPU_COUNTS)
    assert len(receipt["cases"]) == len(CPU_COUNTS) and all(row["passed"] for row in receipt["cases"])
    for key in ("python", "torch", "cuda", "executable", "gpu"):
        assert current_runtime[key] == receipt["runtime"][key], f"GPU gate used another {key}"
    assert cpus in CPU_COUNTS, "Production CPU allocation was not measured"


def launch(args):
    manifest_path = resolve_manifest(args.manifest)
    campaign_name(manifest_path)
    campaign_binding, current_runtime = binding(manifest_path), runtime()
    receipt = read(args.gate)
    validate_receipt(receipt, campaign_binding, current_runtime, int(os.environ.get("SLURM_CPUS_PER_TASK", 0)))
    name, manifest, run = recipe(0, manifest_path)
    if not os.environ.get("WANDB_API_KEY"):
        import netrc
        auth = netrc.netrc().authenticators("api.wandb.ai")
        assert auth and auth[2], "W&B credential unavailable"
    args.output.mkdir(parents=True, exist_ok=False)
    launch_data = {"binding": campaign_binding, "config": name, "run_params": run,
                   "gate_receipt": str(args.gate), "gate_sha256": digest(args.gate), "runtime": current_runtime,
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "slurm": {key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK")},
                   "wandb_run_id": os.environ["WANDB_RUN_ID"]}
    write_new(args.output / "launch.json", launch_data)
    command = [sys.executable, "main.py", "--run", str(manifest_path), "--alg-dir", "configs/dmcontrol/algs",
               "--log-dir", str(args.output), "--alg-index", "0", "--trial-index", "0", "--num-runs", "1"]
    print("SOFT_SAC_TRAINING_LAUNCH " + json.dumps(launch_data), flush=True)
    os.execv(sys.executable, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--decisions", type=int, default=128)
    parser.add_argument("--worker-cpus", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode == "smoke":
        assert args.warmup >= 10 and args.decisions >= 100
        assert SEED_STEPS + 1 + args.warmup + args.decisions < 1000
        (worker if args.worker_cpus else smoke)(args)
    else:
        assert args.gate is not None and not args.worker_cpus
        launch(args)


if __name__ == "__main__":
    main()
