"""Measure production-shape J8 H3 training at selected physical CPU counts.

Run inside a Slurm GPU allocation with at least the largest CPU count. Each case is a
fresh subprocess with a nested CPU affinity mask, equal thread counts and the
same seed. No W&B run, checkpoint, or production configuration is modified.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_transfer_training import binding, recipe, resolve_manifest, runtime, write_new

MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_transfer_training_j8.json"
COUNTS = (1, 2, 4, 8)
SEED_STEPS = 500


def physical_core_cpus(allowed, sysfs=Path("/sys/devices/system/cpu")):
    """Pick one allowed logical CPU per physical core, in stable topology order."""
    cores = {}
    for cpu in sorted(allowed):
        topology = sysfs / f"cpu{cpu}" / "topology"
        key = (int((topology / "physical_package_id").read_text()),
               int((topology / "core_id").read_text()))
        cores.setdefault(key, cpu)
    return [cores[key] for key in sorted(cores)]


def cpu_counts(value):
    counts = tuple(int(item) for item in value.split(","))
    if not counts or min(counts) < 1 or len(set(counts)) != len(counts):
        raise ValueError("CPU counts must be distinct positive integers")
    return counts


def benchmark_recipe(warmup, decisions, *, manifest_path=MANIFEST, cell_index=5):
    name, manifest, run = recipe(cell_index, manifest_path)
    run, manifest = deepcopy(run), deepcopy(manifest)
    overrides = {"wandb": False, "seed_steps": SEED_STEPS, "pretrain_steps": 2,
                 "buffer_size": 4096}
    run["alg_params"].update(overrides)
    # The first post-seed decision is still random, then pretraining executes.
    run["total_steps"] = SEED_STEPS + 1 + warmup + decisions
    run.update(checkpoint_every=None, save_strat="none")
    manifest.update(checkpoint_every=None, save_strat="none", save_trials="none", logs="none")
    return name, manifest, run, overrides


def action_work(metrics, *, decision, interval, probability, rounds=8):
    """Validate realized work; held decisions must not perform an inner solve."""
    assert type(rounds) is int and rounds > 0
    solve = decision % interval == 0
    if interval > 1:
        assert metrics["inner_solve_performed"] == float(solve)
        assert metrics["inner_policy_held"] == float(not solve)
        assert metrics["inner_action_age"] == decision % interval
        assert metrics["inner_episode_decision_index"] == decision
    expected = {"inner_critic_optimizer_steps": 16 * rounds, "inner_actor_optimizer_steps": 4 * rounds,
                "inner_temperature_optimizer_steps": 4 * rounds, "inner_model_steps": 128 * 3 * rounds}
    for key, count in expected.items():
        assert metrics[key] == (count if solve else 0), (decision, key, metrics[key])
    assert metrics["inner_compile_fallback"] == 0
    assert metrics.get("inner_critic_bernoulli_applied", 0.) == float(solve and probability > 0 and decision > 0)
    if not solve:
        for key in ("inner_updates", "inner_steps", "inner_rollouts", "inner_replay_draws",
                    "inner_realized_model_steps", "inner_total_model_steps", "inner_q_evaluations"):
            assert metrics[key] == 0, (decision, key, metrics[key])
        assert metrics["inner_policy_evaluations"] == 1
    assert all(math.isfinite(float(value)) for value in metrics.values())
    return solve


def phase_summary(rows):
    """Keep complete-action timings distinct from asynchronous update enqueue times."""
    result = {"solve_decisions": sum(row["solve"] for row in rows),
              "held_decisions": sum(not row["solve"] for row in rows)}
    for label, solve in (("solve", True), ("held", False)):
        seconds = [row["action_seconds"] for row in rows if row["solve"] == solve]
        result[f"{label}_action_seconds_mean"] = statistics.mean(seconds) if seconds else None
        result[f"{label}_action_seconds_median"] = statistics.median(seconds) if seconds else None
    result["outer_update_enqueue_seconds_mean"] = statistics.mean(
        row["outer_update_enqueue_seconds"] for row in rows
    ) if rows else None
    return result


class TrainingWindow:
    """Time complete decisions, including their outer update, after warmup."""

    def __init__(self, first_step, decisions, synchronize):
        self.first_step = first_step
        self.last_step = first_step + decisions - 1
        self.decisions = decisions
        self.synchronize = synchronize
        self.inner_seconds = []
        self.decision_seconds = []
        self.completed_steps = []
        self.result = None

    def before_action(self, step, updates):
        if step == self.first_step:
            self.synchronize()
            self.started_wall = time.perf_counter()
            self.previous_wall = self.started_wall
            self.started_cpu = time.process_time()
            self.started_updates = updates

    def after_action(self, step, seconds):
        if self.first_step <= step <= self.last_step:
            self.inner_seconds.append(seconds)

    def after_decision(self, step, updates):
        if not self.first_step <= step <= self.last_step:
            return
        self.completed_steps.append(step)
        if step == self.last_step:
            self.synchronize()
        now = time.perf_counter()
        self.decision_seconds.append(now - self.previous_wall)
        self.previous_wall = now
        if step == self.last_step:
            wall = now - self.started_wall
            cpu = time.process_time() - self.started_cpu
            assert self.completed_steps == list(range(self.first_step, self.last_step + 1))
            assert len(self.inner_seconds) == self.decisions
            assert updates - self.started_updates == self.decisions
            self.result = {
                "decisions": self.decisions, "outer_updates": updates - self.started_updates,
                "wall_seconds": wall, "process_cpu_seconds": cpu,
                "average_cpu_cores_used": cpu / wall,
                "decisions_per_second": self.decisions / wall,
                "seconds_per_decision": wall / self.decisions,
                "inner_seconds_per_action": statistics.mean(self.inner_seconds),
                "inner_seconds_median": statistics.median(self.inner_seconds),
                "inner_seconds_stdev": statistics.pstdev(self.inner_seconds),
                "inner_seconds": self.inner_seconds,
                "end_to_end_seconds": self.decision_seconds,
                "per_decision_rates": [1 / seconds for seconds in self.decision_seconds],
                "projected_500k_decision_days": wall / self.decisions * 500000 / 86400,
            }


def worker(args):
    cpus = [int(cpu) for cpu in args.worker_cpus.split(",")]
    counts = cpu_counts(args.cpu_counts)
    assert len(cpus) in counts and len(cpus) == len(set(cpus))
    assert set(cpus) <= os.sched_getaffinity(0), "Requested CPU is outside the allocation"
    assert len(physical_core_cpus(cpus)) == len(cpus), "Affinity includes SMT siblings"
    # Do this before importing Torch, NumPy or MuJoCo so their threads inherit it.
    os.sched_setaffinity(0, cpus)
    assert os.sched_getaffinity(0) == set(cpus)
    import torch
    import gymnasium as gym
    import domains  # noqa: F401
    from RL.AMBITDMPC2 import AMBITDMPC2

    torch.set_num_threads(len(cpus))
    torch.set_num_interop_threads(1)
    assert torch.get_num_threads() == len(cpus) and torch.cuda.is_available()
    name, manifest, run, overrides = benchmark_recipe(
        args.warmup, args.decisions, manifest_path=args.manifest, cell_index=args.cell_index,
    )
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    try:
        learner = AMBITDMPC2(name + "-cpu-benchmark", env, run["alg_params"], run, manifest)
        cfg, agent = learner.cfg, learner.agent
        interval, probability = cfg.inner_solve_interval, cfg.inner_critic_bernoulli_p
        assert (cfg.inner_rounds, cfg.inner_rollout_horizon) == (8, 3)
        assert probability == run["alg_params"]["inner_critic_bernoulli_p"]
        if interval > 1:
            # The durable full-training launcher fixes replay on CPU. Match its
            # placement and sampling path before the first episode allocation.
            learner.buffer.enable_resumable_storage()
            assert args.warmup % interval == args.decisions % interval == 0
        assert (cfg.inner_rollouts_per_round, cfg.inner_batch_size, cfg.inner_replay_capacity) == (128, 256, 3072)
        assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action) == (128, 32)
        assert cfg.compile and cfg.compile_strict and cfg.critic_ensemble_backend == "vmap"
        assert cfg.inner_execution_action == "policy_sample" and cfg.utd == 1
        assert cfg.aux_return_mode == "sac" and not agent.aux_return.has_actor
        timer = TrainingWindow(SEED_STEPS + args.warmup + 2, args.decisions, torch.cuda.synchronize)
        act, update, log_step = learner._act_agent, agent.update, learner._log_wandb_step
        compile_counts = {}
        active = 0
        action_rows = []
        learning_started = time.perf_counter()

        def observed_act(*positional, **kwargs):
            nonlocal active
            step = learner._global_step + 1
            if step == timer.first_step:
                compile_counts["before"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
            timer.before_action(step, learner._num_updates)
            graphs_before = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
            started = time.perf_counter()
            action = act(*positional, **kwargs)
            seconds = time.perf_counter() - started
            timer.after_action(step, seconds)
            metrics = agent.last_inner_metrics
            assert action.device.type == "cpu", "Action timing expects the native CPU materialization"
            assert torch.isfinite(action).all()
            solve = action_work(metrics, decision=active, interval=interval, probability=probability)
            if not solve:
                assert graphs_before == int(torch._dynamo.utils.counters["stats"]["unique_graphs"]), "Held action captured a graph"
            action_rows.append({"decision": active, "step": step, "solve": solve,
                                "action_seconds": seconds, "outer_update_enqueue_seconds": 0.,
                                "outer_updates": 0})
            active += 1
            return action

        def observed_update(*positional, **kwargs):
            started = time.perf_counter()
            metrics = update(*positional, **kwargs)
            seconds = time.perf_counter() - started
            assert metrics["compile_fallback"] == 0
            if action_rows and action_rows[-1]["step"] == learner._global_step:
                action_rows[-1]["outer_update_enqueue_seconds"] += seconds
                action_rows[-1]["outer_updates"] += 1
            return metrics

        def observed_log_step(reward, *positional, **kwargs):
            assert math.isfinite(float(reward))
            result = log_step(reward, *positional, **kwargs)
            timer.after_decision(learner._global_step, learner._num_updates)
            return result

        learner._act_agent, agent.update, learner._log_wandb_step = observed_act, observed_update, observed_log_step
        learner.learn(run["total_steps"])
        torch.cuda.synchronize()
        learning_seconds = time.perf_counter() - learning_started
        compile_counts["after"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
        assert compile_counts["before"] == compile_counts["after"], "Compilation occurred in measured window; increase --warmup"
        assert timer.result is not None and active == args.warmup + args.decisions
        assert all(torch.isfinite(parameter).all() for parameter in agent.model.parameters())
        assert all(torch.isfinite(torch.as_tensor(value)).all() for value in learner._last_train_metrics.values())
        assert torch.get_num_threads() == len(cpus)
        assert os.sched_getaffinity(0) == set(cpus)
        measured = timer.result
        measured_rows = action_rows[args.warmup:]
        assert all(row["outer_updates"] == 1 for row in action_rows)
        measured.update(phase_summary(measured_rows))
        if interval > 1:
            assert measured["solve_decisions"] == args.decisions // interval
            assert measured["held_decisions"] == args.decisions - args.decisions // interval
        measured["action_rows"] = measured_rows
        storage_device = str(learner.buffer._storage_device)
        if interval > 1:
            assert storage_device == "cpu" and learner.buffer.resumable_storage
        report = {"passed": True, "physical_cores": len(cpus), "logical_cpu_ids": cpus,
                  "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                  "runtime": runtime(), "binding": binding(args.manifest), "config": name,
                  "cell_index": args.cell_index, "inner_solve_interval": interval,
                  "bernoulli_probability": probability, "replay_storage_device": storage_device,
                  "warmup_active_decisions": args.warmup, "measured": measured,
                  "compile_graph_counts": compile_counts, "compile_fallback": False,
                  "finite_actions_and_model": True, "critic_updates_per_action": 128,
                  "actor_updates_per_action": 32, "production_overrides": overrides,
                  "optimizer_dose_scope": "per solve; zero on held decisions",
                  "timing_scope": "Synchronized whole-window training throughput. Native CPU action latency can include pending preceding outer GPU work; outer update timings are host enqueue latency. Excludes periodic logging/checkpoint I/O and mature replay costs.",
                  "cold_first_solve_seconds": action_rows[0]["action_seconds"],
                  "cold_collection_pretraining_and_warmup_seconds": learning_seconds - measured["wall_seconds"],
                  "warmup": phase_summary(action_rows[:args.warmup]),
                  "learn_wall_seconds": learning_seconds,
                  "total_steps": run["total_steps"], "replay_transitions": learner.buffer.num_transitions,
                  "checkpointing": False, "wandb": False,
                  "created_utc": datetime.now(timezone.utc).isoformat()}
        write_new(args.output, report)
        print("CPU_BENCH_CASE " + json.dumps({k: v for k, v in report.items() if k not in ("binding", "measured")}
              | {"measured": {k: v for k, v in measured.items() if not isinstance(v, list)}}), flush=True)
    finally:
        if learner is not None:
            learner.close()
        env.close()


def driver(args):
    assert os.environ.get("SLURM_JOB_ID"), "Run inside a scheduled GPU job"
    counts = cpu_counts(args.cpu_counts)
    assert int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) >= max(counts)
    allowed = sorted(os.sched_getaffinity(0))
    cores = physical_core_cpus(allowed)
    assert len(cores) >= max(counts), "Need distinct physical cores; request --hint=nomultithread"
    output = args.output.resolve()
    assert output != ROOT and ROOT not in output.parents, "Keep benchmark outputs outside source"
    output.mkdir(parents=True, exist_ok=False)
    order = list(counts)
    random.Random(args.order_seed).shuffle(order)
    write_new(output / "protocol.json", {
        "binding": binding(args.manifest), "cell_index": args.cell_index,
        "allowed_logical_cpus": allowed,
        "physical_core_representatives": cores, "case_order": order, "order_seed": args.order_seed,
        "warmup_active_decisions": args.warmup, "measured_decisions_per_case": args.decisions,
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "host": os.uname().nodename,
        "scope": "Short early-replay training; excludes startup/compilation, W&B, periodic diagnostics and checkpoint I/O. One case per CPU count; no long-run or statistical guarantee.",
    })
    with tempfile.TemporaryDirectory(prefix="ambi-cpu-benchmark-") as temporary:
        cache = Path(temporary)
        for count in order:
            environment = dict(os.environ)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                environment[key] = str(count)
            environment.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE", WANDB_MODE="disabled",
                               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", MUJOCO_GL="egl",
                               TORCHINDUCTOR_COMPILE_THREADS="1", TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
                               TRITON_CACHE_DIR=str(cache / "triton"))
            command = [sys.executable, str(Path(__file__).resolve()), "--worker-cpus", ",".join(map(str, cores[:count])),
                       "--warmup", str(args.warmup), "--decisions", str(args.decisions),
                       "--manifest", str(args.manifest), "--cell-index", str(args.cell_index),
                       "--cpu-counts", args.cpu_counts,
                       "--output", str(output / f"cores-{count}.json")]
            with (output / f"cores-{count}.log").open("x") as log:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            row = json.loads((output / f"cores-{count}.json").read_text())
            print("CPU_BENCH_PROGRESS " + json.dumps({"cores": count, "measured": {
                k: v for k, v in row["measured"].items() if not isinstance(v, list)}}), flush=True)
    cases = [json.loads((output / f"cores-{count}.json").read_text()) for count in counts]
    assert len({json.dumps(row["runtime"], sort_keys=True) for row in cases}) == 1
    fastest = max(row["measured"]["decisions_per_second"] for row in cases)
    for row in cases:
        row["relative_throughput"] = row["measured"]["decisions_per_second"] / fastest
    recommended = min(row["physical_cores"] for row in cases if row["relative_throughput"] >= .95)
    write_new(output / "results.json", {"passed": True, "cases": cases, "case_order": order,
                                      "smallest_cores_within_five_percent": recommended,
                                      "recommendation_scope": "one short sample per count, no statistical guarantee"})
    print("CPU_BENCH_PASSED " + str(output / "results.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--decisions", type=int, default=128)
    parser.add_argument("--order-seed", type=int, default=55)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--cell-index", type=int, default=5)
    parser.add_argument("--cpu-counts", default=",".join(map(str, COUNTS)))
    parser.add_argument("--worker-cpus", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.manifest = resolve_manifest(args.manifest)
    assert args.cell_index >= 0
    cpu_counts(args.cpu_counts)
    assert args.warmup >= 10 and args.decisions >= 100
    assert SEED_STEPS + 1 + args.warmup + args.decisions < 1000, "Keep this short benchmark inside one post-seed episode"
    (worker if args.worker_cpus else driver)(args)


if __name__ == "__main__":
    main()
