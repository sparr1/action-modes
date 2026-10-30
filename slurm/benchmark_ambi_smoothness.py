"""Smoke and time one production smoothness recipe on a scheduled CUDA GPU.

Each CPU count runs in a fresh process. Complete decisions include held actions,
environment work and outer updates. Startup, compilation, W&B and checkpoint I/O
are excluded; the short replay and reduced pretraining are not long-run timing.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_transfer_training import digest, read, runtime, write_new
from slurm.benchmark_ambi_training_cpu import TrainingWindow, physical_core_cpus

SEED_STEPS = 500
TERMS = ("mpr", "action_smoothness")
PREFIXES = ("outer_pave", "aux_return_pave")


def source_path(path):
    path = Path(path)
    path = (ROOT / path if not path.is_absolute() else path).resolve(strict=True)
    path.relative_to(ROOT)
    return path


def recipe(manifest_path, alg_dir, index, warmup, decisions):
    manifest = read(source_path(manifest_path))
    assert 0 <= index < len(manifest["configs"]), "Invalid --alg-index"
    name = manifest["configs"][index]
    config_path = source_path(source_path(alg_dir) / (name + ".json"))
    run = {"name": name, **read(config_path), **manifest.get("overrides_alg", {})}
    run, manifest = deepcopy(run), deepcopy(manifest)
    overrides = dict(wandb=False, seed_steps=SEED_STEPS, pretrain_steps=2, buffer_size=4096)
    run["alg_params"].update(overrides)
    run.update(total_steps=SEED_STEPS + 1 + warmup + decisions,
               checkpoint_every=None, save_strat="none")
    manifest.update(checkpoint_every=None, save_strat="none", save_trials="none", logs="none")
    return name, manifest, run, overrides


def binding(args):
    manifest = source_path(args.manifest)
    names = read(manifest)["configs"]
    config = source_path(source_path(args.alg_dir) / (names[args.alg_index] + ".json"))
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "manifest": str(manifest.relative_to(ROOT)), "manifest_sha256": digest(manifest),
            "config": str(config.relative_to(ROOT)), "config_sha256": digest(config),
            "alg_index": args.alg_index, "lock_sha256": digest(ROOT / "environments/dmcontrol/uv.lock")}


def active_terms(params):
    terms = [(prefix, term) for prefix in PREFIXES for term in TERMS
             if float(params.get(f"{prefix}_{term}_coef", 0.0)) > 0.0]
    assert terms, "This benchmark requires an active MPR or action-smoothness term"
    assert "outer_pave_batch_size" in params and params["outer_pave_batch_size"] in (None, 768), \
        "Keep the production full (768-sample) PAVE batch"
    return terms


def validate_penalties(rows, selected):
    """Validate materialized outer metrics, including every selected critic."""
    assert rows, "No measured outer updates"
    summary = {}
    for prefix, term in selected:
        key = f"{prefix}_{term}_loss"
        values = [float(row[key]) for row in rows]
        assert all(float(row[f"{prefix}_sample_count"]) == 768 for row in rows)
        assert all(math.isfinite(value) and value >= 0 for value in values), key
        assert max(values) > 0, f"Active regularizer stayed zero: {key}"
        weighted = [float(row[f"{prefix}_{term}_weighted_loss"]) for row in rows]
        assert all(math.isfinite(value) and value >= 0 for value in weighted), key
        assert max(weighted) > 0, f"Active weighted regularizer stayed zero: {key}"
        summary[key] = {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}
    return summary


def finite_metrics(metrics, torch):
    # Some optional diagnostics are absent/None, especially on held decisions.
    for key, value in (metrics or {}).items():
        if value is not None:
            assert bool(torch.isfinite(torch.as_tensor(value)).all()), f"Non-finite metric: {key}"


def configure_cpu_replay_without_checkpoints(learner, output):
    """Keep production CPU placement even if learn rebuilds the replay buffer."""
    learner.set_checkpointing(25_000, str(output.parent / "checkpointing-disabled"),
                              "smoothness-benchmark", save_strat="none", checkpoint_replay=True)


def worker(args):
    assert os.environ.get("SLURM_JOB_ID"), "Run inside a scheduled GPU job"
    cpus = [int(cpu) for cpu in args.worker_cpus.split(",")]
    assert cpus and len(cpus) == len(set(cpus))
    assert set(cpus) <= os.sched_getaffinity(0), "CPU outside the allocation"
    assert len(physical_core_cpus(cpus)) == len(cpus), "CPU affinity includes SMT siblings"
    assert args.output.resolve() != ROOT and ROOT not in args.output.resolve().parents
    os.sched_setaffinity(0, cpus)  # Before Torch, NumPy or MuJoCo starts threads.
    import torch
    import gymnasium as gym
    import domains  # noqa: F401
    from RL.AMBITDMPC2 import AMBITDMPC2

    torch.set_num_threads(len(cpus))
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available(), "CUDA GPU required"
    name, manifest, run, overrides = recipe(args.manifest, args.alg_dir, args.alg_index,
                                            args.warmup, args.decisions)
    selected = active_terms(run["alg_params"])
    env = gym.make(run["env"], **manifest.get("env_params", {}))
    learner = None
    started = time.perf_counter()
    try:
        learner = AMBITDMPC2(name + "-smoothness-smoke", env, run["alg_params"], run, manifest)
        # Match exact-resume production placement without enabling checkpoint I/O.
        configure_cpu_replay_without_checkpoints(learner, args.output)
        cfg, agent = learner.cfg, learner.agent
        assert (cfg.inner_rounds, cfg.inner_rollout_horizon, cfg.inner_solve_interval) == (10, 3, 3)
        assert cfg.compile and cfg.compile_strict and cfg.critic_ensemble_backend == "vmap"
        assert cfg.utd == 1 and cfg.outer_pave_batch_size in (None, 768)
        assert cfg.batch_size * cfg.train_unroll_horizon == 768
        assert not cfg.wandb and learner._wandb_run is None
        timer = TrainingWindow(SEED_STEPS + args.warmup + 2, args.decisions, torch.cuda.synchronize)
        act, update, log_step = learner._act_agent, agent.update, learner._log_wandb_step
        compile_counts, startup = {}, {}
        inner_rows, outer_rows, measured_outer = [], [], []
        torch.cuda.reset_peak_memory_stats()

        def observed_act(*positional, **kwargs):
            step = learner._global_step + 1
            if step == timer.first_step:
                torch.cuda.synchronize()
                compile_counts["before"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
                startup["seconds_through_compilation_and_warmup"] = time.perf_counter() - started
                startup["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
                startup["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
                torch.cuda.reset_peak_memory_stats()
            timer.before_action(step, learner._num_updates)
            action_started = time.perf_counter()
            action = act(*positional, **kwargs)
            timer.after_action(step, time.perf_counter() - action_started)
            assert action.device.type == "cpu" and bool(torch.isfinite(action).all())
            inner_rows.append((step, dict(agent.last_inner_metrics or {})))
            return action

        def observed_update(*positional, **kwargs):
            metrics = update(*positional, **kwargs)
            # These returned diagnostics are detached. Defer materialization and
            # assertions until after timing, avoiding extra GPU synchronizations.
            outer_rows.append(dict(metrics))
            if timer.first_step <= learner._global_step <= timer.last_step:
                measured_outer.append(dict(metrics))
            return metrics

        def observed_log_step(reward, *positional, **kwargs):
            assert math.isfinite(float(reward))
            result = log_step(reward, *positional, **kwargs)
            timer.after_decision(learner._global_step, learner._num_updates)
            return result

        learner._act_agent, agent.update, learner._log_wandb_step = observed_act, observed_update, observed_log_step
        learner.learn(run["total_steps"])
        compile_counts["after"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
        assert compile_counts["before"] == compile_counts["after"], "Compilation in measured window; increase --warmup"
        assert timer.result and len(inner_rows) == args.warmup + args.decisions
        assert len(measured_outer) == args.decisions
        assert learner.buffer.resumable_storage and learner.buffer._storage_device.type == "cpu"
        for _, metrics in inner_rows:
            finite_metrics(metrics, torch)
            assert float(metrics.get("inner_compile_fallback", 0) or 0) == 0
        for metrics in outer_rows:
            finite_metrics(metrics, torch)
            assert float(metrics["compile_fallback"]) == 0
        assert all(bool(torch.isfinite(parameter).all()) for parameter in agent.model.parameters())
        measured_inner = [row for step, row in inner_rows if timer.first_step <= step <= timer.last_step]
        solves = sum(int(float(row.get("inner_solve_performed", 0) or 0)) for row in measured_inner)
        assert solves == args.decisions // cfg.inner_solve_interval
        penalty_metrics = [{key: float(value) for key, value in row.items() if "pave" in key}
                           for row in measured_outer]
        penalties = validate_penalties(penalty_metrics, selected)
        assert torch.get_num_threads() == len(cpus) and os.sched_getaffinity(0) == set(cpus)
        measured = timer.result
        measured.update(solve_decisions=solves, held_decisions=args.decisions - solves,
                        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                        projected_1m_decision_hours=measured["seconds_per_decision"] * 1e6 / 3600)
        report = {"passed": True, "binding": binding(args), "runtime": runtime(), "config": name,
                  "physical_cores": len(cpus), "logical_cpu_ids": cpus,
                  "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                  "startup": startup, "warmup_decisions": args.warmup, "measured": measured,
                  "compile_graph_counts": compile_counts, "compile_fallback": False,
                  "finite_actions_model_metrics": True, "penalty_metrics": penalties,
                  "pave_sample_count": 768, "production_overrides": overrides,
                  "replay_storage_device": "cpu",
                  "total_steps": run["total_steps"], "replay_transitions": learner.buffer.num_transitions,
                  "checkpointing": False, "wandb": False, "created_utc": datetime.now(timezone.utc).isoformat()}
        write_new(args.output, report)
        print("SMOOTHNESS_BENCH_CASE " + json.dumps({"cores": len(cpus), "measured": {
            key: value for key, value in measured.items() if not isinstance(value, list)}}), flush=True)
    finally:
        if learner is not None:
            learner.close()
        env.close()


def driver(args):
    assert os.environ.get("SLURM_JOB_ID"), "Run inside a scheduled GPU job"
    assert int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) >= max(args.cores)
    allowed = sorted(os.sched_getaffinity(0))
    cores = physical_core_cpus(allowed)
    assert len(cores) >= max(args.cores), "Request distinct physical cores (--hint=nomultithread)"
    output = args.output.resolve()
    assert output != ROOT and ROOT not in output.parents, "Outputs must be outside the checkout"
    campaign_binding = binding(args)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "protocol.json", {"binding": campaign_binding, "case_order": args.cores,
              "physical_core_representatives": cores, "warmup_decisions": args.warmup,
              "measured_decisions": args.decisions, "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "host": os.uname().nodename, "scope": __doc__})
    with tempfile.TemporaryDirectory(prefix="ambi-smoothness-benchmark-") as temporary:
        # Compiled artifacts are keyed by Torch/Inductor, including applicable
        # compile options. Sharing disk caches does not share process RNG or
        # model state; each case still constructs a fresh seeded learner.
        cache = Path(temporary)
        for count in args.cores:
            environment = dict(os.environ)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                environment[key] = str(count)
            environment.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE", WANDB_MODE="disabled",
                               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", MUJOCO_GL="egl",
                               TORCHINDUCTOR_COMPILE_THREADS="1", TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
                               TRITON_CACHE_DIR=str(cache / "triton"))
            command = [sys.executable, str(Path(__file__).resolve()), "--manifest", str(source_path(args.manifest)),
                       "--alg-dir", str(source_path(args.alg_dir)), "--alg-index", str(args.alg_index),
                       "--worker-cpus", ",".join(map(str, cores[:count])), "--warmup", str(args.warmup),
                       "--decisions", str(args.decisions), "--output", str(output / f"cores-{count}.json")]
            with (output / f"cores-{count}.log").open("x") as log:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    cases = [read(output / f"cores-{count}.json") for count in args.cores]
    assert all(row["passed"] and row["binding"] == campaign_binding for row in cases)
    fastest = min(row["measured"]["seconds_per_decision"] for row in cases)
    recommended = min(row["physical_cores"] for row in cases if row["measured"]["seconds_per_decision"] <= fastest / .95)
    write_new(output / "results.json", {"passed": True, "cases": cases,
              "recommended_cpus_within_5pct_of_best": recommended, "case_order": args.cores})
    print("SMOOTHNESS_BENCH_PASSED " + str(output / "results.json"), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--alg-dir", type=Path, required=True)
    parser.add_argument("--alg-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cores", default="1,2", help="Comma-separated physical CPU counts")
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--decisions", type=int, default=30)
    parser.add_argument("--worker-cpus", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        args.cores = [int(value) for value in args.cores.split(",")]
    except ValueError:
        parser.error("--cores must be comma-separated positive integers")
    if not args.cores or min(args.cores) < 1 or len(set(args.cores)) != len(args.cores):
        parser.error("--cores must contain distinct positive integers")
    if args.warmup < 12 or args.decisions < 30 or args.warmup % 3 or args.decisions % 3:
        parser.error("Use at least 12 warmup and 30 measured decisions, both multiples of K=3")
    if SEED_STEPS + 1 + args.warmup + args.decisions >= 1000:
        parser.error("Keep this short benchmark inside one post-seed episode")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    (worker if arguments.worker_cpus else driver)(arguments)
