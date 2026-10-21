"""Compare CPU and CUDA replay on one scheduled GPU and one physical CPU.

Each occupancy uses CPU, CUDA, CUDA, CPU fresh subprocesses. The replay fixture
duplicates one real, seeded Humanoid episode; it measures capacity-dependent
execution cost, not mature-policy quality. Integrated learning runs first, so
the later isolated sample timings cannot perturb its minibatch RNG stream.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_reward_training import ENTROPY_MANIFESTS, validate_campaign, validate_learner
from slurm.ambi_transfer_training import binding, resolve_manifest, runtime, write_new
from slurm.benchmark_ambi_training_cpu import TrainingWindow, physical_core_cpus

CAPACITY = 1_000_000
SEED_STEPS = 500
ORDER = ("cpu", "cuda", "cuda", "cpu")
MANIFEST = ENTROPY_MANIFESTS[-10.5]


def replay_rows(value):
    rows = tuple(int(item) for item in value.split(","))
    if not rows or len(rows) != len(set(rows)) or min(rows) < 501 or max(rows) > CAPACITY - 501:
        raise ValueError("Replay occupancies must be distinct counts between 501 and 999499 rows")
    return rows


def benchmark_recipe(manifest_path=MANIFEST):
    name, manifest, run = validate_campaign(manifest_path)
    if resolve_manifest(manifest_path) not in ENTROPY_MANIFESTS.values():
        raise ValueError("Replay benchmark requires a one-million-decision no-inner recipe")
    manifest, run = deepcopy(manifest), deepcopy(run)
    overrides = {"wandb": False, "seed_steps": SEED_STEPS, "pretrain_steps": 2}
    run["alg_params"].update(overrides)
    run.update(checkpoint_every=None, save_strat="none")
    manifest.update(checkpoint_every=None, save_strat="none", save_trials="none", logs="none")
    assert run["total_steps"] == run["alg_params"]["buffer_size"] == CAPACITY
    return name, manifest, run, overrides


def populate_fixture(buffer, episode, target_rows, *, chunk_episodes=64):
    """Fill complete episode copies after the native first insertion, once only."""
    import torch

    rows = len(episode)
    if buffer.num_eps != 1 or buffer.size != rows or rows < 2:
        raise ValueError("Fixture population must immediately follow the first complete episode")
    copies = math.ceil(target_rows / rows)
    if target_rows < rows or copies * rows > buffer.capacity or chunk_episodes < 1:
        raise ValueError("Fixture must contain complete episodes and fit in replay")
    # Initial action/reward/termination rows intentionally contain native NaNs.
    for key in ("obs", "action", "reward", "terminated"):
        value = episode[key] if key == "obs" else episode[key][1:]
        if not torch.isfinite(value).all():
            raise ValueError(f"Invalid real-episode fixture field: {key}")
    digest = hashlib.sha256()
    for key in ("obs", "action", "reward", "terminated"):
        digest.update(episode[key].detach().cpu().contiguous().numpy().tobytes())
    remaining = copies - 1
    while remaining:
        count = min(remaining, chunk_episodes)
        # load assigns distinct episode IDs; no sampled or synthetic transitions.
        buffer.load(torch.stack([episode] * count))
        remaining -= count
    assert buffer.size == copies * rows and buffer.num_eps == copies
    return {"requested_rows": target_rows, "resident_rows": buffer.size,
            "resident_transitions": buffer.num_transitions, "episode_rows": rows,
            "episode_copies": copies, "source_episode_sha256": digest.hexdigest(),
            "scope": "Duplicated real warmup episode: performance fixture, not mature learning data"}


def sample_windows(buffer, synchronize, *, windows, samples):
    """Synchronized complete sample+transfer windows, after integrated training."""
    import torch

    for _ in range(8):
        batch = buffer.sample()
    for tensor in batch:
        if tensor is not None:
            assert tensor.device.type == "cuda" and torch.isfinite(tensor).all()
    seconds = []
    for _ in range(windows):
        synchronize()
        started = time.perf_counter()
        for _ in range(samples):
            buffer.sample()
        synchronize()
        seconds.append(time.perf_counter() - started)
    return {"windows": windows, "samples_per_window": samples, "wall_seconds": seconds,
            "seconds_per_sample": sum(seconds) / (windows * samples),
            "window_seconds_per_sample": [value / samples for value in seconds],
            "scope": "Isolated replay.sample including sampling, gather, pinning, transfer and contiguous outputs"}


def compare_cases(cases):
    """Report both counterbalanced pairs; avoid treating micro windows as replicas."""
    summaries = []
    for occupancy in sorted({row["fixture"]["requested_rows"] for row in cases}):
        selected = sorted((row for row in cases if row["fixture"]["requested_rows"] == occupancy),
                          key=lambda row: row["case_index"])
        assert [row["placement"] for row in selected] == list(ORDER)
        assert len({row["fixture"]["source_episode_sha256"] for row in selected}) == 1
        assert len({row["fixture"]["resident_rows"] for row in selected}) == 1
        pairings = ((selected[0], selected[1]), (selected[3], selected[2]))
        speedups = [cpu["measured"]["seconds_per_decision"] / gpu["measured"]["seconds_per_decision"]
                    for cpu, gpu in pairings]
        sampling = [cpu["replay_sample"]["seconds_per_sample"] / gpu["replay_sample"]["seconds_per_sample"]
                    for cpu, gpu in pairings]
        summaries.append({"requested_rows": occupancy, "gpu_training_speedups": speedups,
                          "gpu_training_speedup_geometric_mean": math.sqrt(math.prod(speedups)),
                          "gpu_sample_speedups": sampling,
                          "gpu_sample_speedup_geometric_mean": math.sqrt(math.prod(sampling)),
                          "scope": "Two counterbalanced pairs; speedup >1 favors GPU; no learning-quality inference"})
    return summaries


def worker(args):
    cpu = args.worker_cpu
    assert cpu in os.sched_getaffinity(0)
    os.sched_setaffinity(0, {cpu})
    import torch
    import gymnasium as gym
    import domains  # noqa: F401
    from RL.AMBITDMPC2 import AMBITDMPC2

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available(), "Replay benchmark requires CUDA"
    torch.cuda.reset_peak_memory_stats()
    name, manifest, run, overrides = benchmark_recipe(args.manifest)
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    started = time.perf_counter()
    created_utc = datetime.now(timezone.utc).isoformat()
    try:
        learner = AMBITDMPC2(name + "-replay-benchmark", env, run["alg_params"], run, manifest)
        cfg, agent, buffer = learner.cfg, learner.agent, learner.buffer
        validate_learner(cfg, agent, target_entropy=run["alg_params"]["target_entropy"], solve_interval=1)
        assert cfg.inner_operator == "none" and cfg.steps == buffer.capacity == CAPACITY
        if args.placement == "cpu":
            buffer.enable_resumable_storage()
        assert not learner._eval_freq and learner._checkpointing is None
        stop_step = SEED_STEPS + 1 + args.warmup + args.decisions
        timer = TrainingWindow(SEED_STEPS + args.warmup + 2, args.decisions, torch.cuda.synchronize)
        fixture, compile_counts = {}, {}
        active = 0
        add, act, update, log_step = buffer.add, learner._act_agent, agent.update, learner._log_wandb_step
        run_episode = learner._run_training_episode

        def first_add(episode):
            result = add(episode)
            assert len(episode) == SEED_STEPS + 1
            fixture.update(populate_fixture(buffer, episode, args.worker_rows))
            assert buffer._storage_device.type == args.placement, "Automatic GPU placement did not select CUDA"
            assert buffer._pin_memory == (args.placement == "cpu")
            buffer.add = add
            return result

        def observed_act(*positional, **kwargs):
            nonlocal active
            step = learner._global_step + 1
            if step == timer.first_step:
                compile_counts["before"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
            timer.before_action(step, learner._num_updates)
            action_started = time.perf_counter()
            action = act(*positional, **kwargs)
            timer.after_action(step, time.perf_counter() - action_started)
            assert action.device.type == "cpu" and torch.isfinite(action).all()
            assert agent.last_inner_metrics["inner_compile_fallback"] == 0
            assert agent.last_inner_metrics["inner_model_steps"] == 0
            active += 1
            return action

        def observed_update(*positional, **kwargs):
            metrics = update(*positional, **kwargs)
            assert metrics["compile_fallback"] == 0
            return metrics

        def observed_log(*positional, **kwargs):
            result = log_step(*positional, **kwargs)
            timer.after_decision(learner._global_step, learner._num_updates)
            return result

        def bounded_episode(obs, total_timesteps, **kwargs):
            # Preserve cfg.steps and the one-million-row allocation. learn(small)
            # would replace Buffer and silently shrink its capacity to that budget.
            return run_episode(obs, min(total_timesteps, stop_step), **kwargs)

        buffer.add = first_add
        learner._act_agent, agent.update, learner._log_wandb_step = observed_act, observed_update, observed_log
        learner._run_training_episode = bounded_episode
        learner.learn(CAPACITY)
        torch.cuda.synchronize()
        learning_seconds = time.perf_counter() - started
        compile_counts["after"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
        assert compile_counts["before"] == compile_counts["after"], "Compilation inside measured window; increase warmup"
        assert timer.result and active == args.warmup + args.decisions
        assert learner._global_step == stop_step and learner._num_updates == 2 + active
        assert cfg.steps == buffer.capacity == CAPACITY and buffer.size == fixture["resident_rows"]
        assert all(torch.isfinite(parameter).all() for parameter in agent.model.parameters())
        assert all(torch.isfinite(torch.as_tensor(value)).all() for value in learner._last_train_metrics.values())
        # Deliberately after the entire integrated window: no RNG save/restore or
        # sampler-cache approximation is needed to preserve its sampled batches.
        micro = sample_windows(buffer, torch.cuda.synchronize, windows=args.sample_windows, samples=args.samples)
        assert os.sched_getaffinity(0) == {cpu} and torch.get_num_threads() == 1
        report = {"passed": True, "case_index": args.case_index, "placement": args.placement,
                  "created_utc": created_utc, "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "host": os.uname().nodename, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "runtime": runtime(), "binding": binding(args.manifest), "config": name,
                  "production_overrides": overrides, "resolved_alg_params": dict(run["alg_params"]),
                  "physical_cores": 1, "logical_cpu_ids": [cpu], "fixture": fixture,
                  "replay_capacity": buffer.capacity, "replay_storage_device": str(buffer._storage_device),
                  "replay_pinned_staging": buffer._pin_memory, "cfg_steps": cfg.steps,
                  "learned_decisions": learner._global_step, "warmup_active_decisions": args.warmup,
                  "measured": timer.result, "replay_sample": micro, "compile_graph_counts": compile_counts,
                  "compile_fallback": False, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "startup_collection_prefill_compile_warmup_seconds": learning_seconds - timer.result["wall_seconds"],
                  "learn_wall_seconds": learning_seconds, "wandb": False, "checkpointing": False,
                  "timing_scope": "Synchronized native actions, environment and UTD1 learning; excludes compilation, fixture fill, W&B, periodic checkpoints and real mature-data learning"}
        write_new(args.output, report)
        print("REPLAY_BENCH_CASE " + json.dumps({"placement": args.placement, "rows": buffer.size,
              "decisions_per_second": timer.result["decisions_per_second"],
              "sample_seconds": micro["seconds_per_sample"]}), flush=True)
    finally:
        if learner is not None:
            learner.close()
        env.close()


def driver(args):
    assert os.environ.get("SLURM_JOB_ID"), "Run inside a scheduled GPU job"
    cores = physical_core_cpus(os.sched_getaffinity(0))
    assert cores and int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) >= 1
    output = args.output.resolve()
    assert output != ROOT and ROOT not in output.parents, "Keep benchmark results outside source"
    output.mkdir(parents=True, exist_ok=False)
    occupancies = replay_rows(args.replay_rows)
    cases = [(rows, placement) for rows in occupancies for placement in ORDER]
    write_new(output / "protocol.json", {
        "binding": binding(args.manifest), "cpu": cores[0], "case_order": cases,
        "warmup": args.warmup, "decisions": args.decisions, "sample_windows": args.sample_windows,
        "samples_per_window": args.samples, "capacity": CAPACITY,
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Two counterbalanced pairs per occupancy, one shared GPU and physical CPU; real episode duplication is a performance fixture, not a quality experiment. CPU uses production resumable placement; CUDA uses legacy automatic placement, asserted CUDA. Compilation cache is shared; every learner and RNG starts fresh. Sampling microbenchmarks follow integrated learning."})
    reports = []
    with tempfile.TemporaryDirectory(prefix="ambi-replay-benchmark-") as temporary:
        cache = Path(temporary)
        for index, (rows, placement) in enumerate(cases):
            environment = dict(os.environ)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                environment[key] = "1"
            environment.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE", WANDB_MODE="disabled",
                               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", MUJOCO_GL="egl",
                               TORCHINDUCTOR_COMPILE_THREADS="1", TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
                               TRITON_CACHE_DIR=str(cache / "triton"))
            prefix = f"case-{index}-{rows}-{placement}"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker-cpu", str(cores[0]),
                       "--worker-rows", str(rows), "--placement", placement, "--case-index", str(index),
                       "--manifest", str(args.manifest), "--warmup", str(args.warmup),
                       "--decisions", str(args.decisions), "--sample-windows", str(args.sample_windows),
                       "--samples", str(args.samples), "--output", str(output / (prefix + ".json"))]
            with (output / (prefix + ".log")).open("x") as log:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            reports.append(json.loads((output / (prefix + ".json")).read_text()))
            row = reports[-1]
            print("REPLAY_BENCH_PROGRESS " + json.dumps({"case": index, "placement": placement,
                  "rows": rows, "decisions_per_second": row["measured"]["decisions_per_second"],
                  "sample_seconds": row["replay_sample"]["seconds_per_sample"]}), flush=True)
    assert len({json.dumps(row["runtime"], sort_keys=True) for row in reports}) == 1
    write_new(output / "results.json", {"passed": True, "cases": reports, "comparisons": compare_cases(reports)})
    print("REPLAY_BENCH_PASSED " + str(output / "results.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--replay-rows", default="10000,750000")
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--decisions", type=int, default=200)
    parser.add_argument("--sample-windows", type=int, default=5)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--worker-cpu", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-rows", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--placement", choices=("cpu", "cuda"), help=argparse.SUPPRESS)
    parser.add_argument("--case-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.manifest = resolve_manifest(args.manifest)
    benchmark_recipe(args.manifest)
    replay_rows(args.replay_rows)
    assert args.warmup >= 24 and args.decisions >= 100
    assert SEED_STEPS + 1 + args.warmup + args.decisions < 1000, "Keep measured occupancy constant inside one partial episode"
    assert args.sample_windows >= 2 and args.samples >= 8
    if args.worker_cpu is not None:
        assert args.worker_rows is not None and args.placement is not None and args.case_index is not None
        replay_rows(str(args.worker_rows))
    (worker if args.worker_cpu is not None else driver)(args)


if __name__ == "__main__":
    main()
