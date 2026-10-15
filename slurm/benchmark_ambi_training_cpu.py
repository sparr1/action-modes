"""Measure J8 H3 Bernoulli training on 1/2/4/8 physical CPU cores, one GPU.

Run once inside an eight-physical-core Slurm GPU allocation. Each case is a
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
from slurm.ambi_transfer_training import binding, recipe, runtime, write_new

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


def benchmark_recipe(warmup, decisions):
    name, manifest, run = recipe(5, MANIFEST)
    run, manifest = deepcopy(run), deepcopy(manifest)
    overrides = {"wandb": False, "seed_steps": SEED_STEPS, "pretrain_steps": 2,
                 "buffer_size": 4096}
    run["alg_params"].update(overrides)
    # The first post-seed decision is still random, then pretraining executes.
    run["total_steps"] = SEED_STEPS + 1 + warmup + decisions
    run.update(checkpoint_every=None, save_strat="none")
    manifest.update(checkpoint_every=None, save_strat="none", save_trials="none", logs="none")
    return name, manifest, run, overrides


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
    assert len(cpus) in COUNTS and len(cpus) == len(set(cpus))
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
    name, manifest, run, overrides = benchmark_recipe(args.warmup, args.decisions)
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    try:
        learner = AMBITDMPC2(name + "-cpu-benchmark", env, run["alg_params"], run, manifest)
        cfg, agent = learner.cfg, learner.agent
        assert (cfg.inner_rounds, cfg.inner_rollout_horizon, cfg.inner_critic_bernoulli_p) == (8, 3, .5)
        assert (cfg.inner_rollouts_per_round, cfg.inner_batch_size, cfg.inner_replay_capacity) == (128, 256, 3072)
        assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action) == (128, 32)
        assert cfg.compile and cfg.compile_strict and cfg.critic_ensemble_backend == "vmap"
        assert cfg.inner_execution_action == "policy_sample" and cfg.utd == 1
        assert cfg.aux_return_mode == "sac" and not agent.aux_return.has_actor
        timer = TrainingWindow(SEED_STEPS + args.warmup + 2, args.decisions, torch.cuda.synchronize)
        act, update, log_step = learner._act_agent, agent.update, learner._log_wandb_step
        compile_counts = {}
        active = 0

        def observed_act(*positional, **kwargs):
            nonlocal active
            step = learner._global_step + 1
            if step == timer.first_step:
                compile_counts["before"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
            timer.before_action(step, learner._num_updates)
            started = time.perf_counter()
            action = act(*positional, **kwargs)
            timer.after_action(step, time.perf_counter() - started)
            metrics = agent.last_inner_metrics
            assert action.device.type == "cpu", "Action timing expects the native CPU materialization"
            assert torch.isfinite(action).all()
            assert metrics["inner_critic_optimizer_steps"] == 128
            assert metrics["inner_actor_optimizer_steps"] == 32
            assert metrics["inner_compile_fallback"] == 0
            assert metrics["inner_critic_bernoulli_applied"] == float(active > 0)
            assert all(math.isfinite(float(value)) for value in metrics.values())
            active += 1
            return action

        def observed_update(*positional, **kwargs):
            metrics = update(*positional, **kwargs)
            assert metrics["compile_fallback"] == 0
            return metrics

        def observed_log_step(reward, *positional, **kwargs):
            assert math.isfinite(float(reward))
            result = log_step(reward, *positional, **kwargs)
            timer.after_decision(learner._global_step, learner._num_updates)
            return result

        learner._act_agent, agent.update, learner._log_wandb_step = observed_act, observed_update, observed_log_step
        learner.learn(run["total_steps"])
        compile_counts["after"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
        assert compile_counts["before"] == compile_counts["after"], "Compilation occurred in measured window; increase --warmup"
        assert timer.result is not None and active == args.warmup + args.decisions
        assert all(torch.isfinite(parameter).all() for parameter in agent.model.parameters())
        assert all(torch.isfinite(torch.as_tensor(value)).all() for value in learner._last_train_metrics.values())
        assert torch.get_num_threads() == len(cpus)
        assert os.sched_getaffinity(0) == set(cpus)
        measured = timer.result
        report = {"passed": True, "physical_cores": len(cpus), "logical_cpu_ids": cpus,
                  "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                  "runtime": runtime(), "binding": binding(MANIFEST), "config": name,
                  "warmup_active_decisions": args.warmup, "measured": measured,
                  "compile_graph_counts": compile_counts, "compile_fallback": False,
                  "finite_actions_and_model": True, "critic_updates_per_action": 128,
                  "actor_updates_per_action": 32, "production_overrides": overrides,
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
    assert int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) >= 8
    allowed = sorted(os.sched_getaffinity(0))
    cores = physical_core_cpus(allowed)
    assert len(cores) >= 8, "Need eight distinct physical cores; request --hint=nomultithread"
    output = args.output.resolve()
    assert output != ROOT and ROOT not in output.parents, "Keep benchmark outputs outside source"
    output.mkdir(parents=True, exist_ok=False)
    order = list(COUNTS)
    random.Random(args.order_seed).shuffle(order)
    write_new(output / "protocol.json", {
        "binding": binding(MANIFEST), "allowed_logical_cpus": allowed,
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
                       "--output", str(output / f"cores-{count}.json")]
            with (output / f"cores-{count}.log").open("x") as log:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            row = json.loads((output / f"cores-{count}.json").read_text())
            print("CPU_BENCH_PROGRESS " + json.dumps({"cores": count, "measured": {
                k: v for k, v in row["measured"].items() if not isinstance(v, list)}}), flush=True)
    cases = [json.loads((output / f"cores-{count}.json").read_text()) for count in COUNTS]
    assert len({json.dumps(row["runtime"], sort_keys=True) for row in cases}) == 1
    fastest = max(row["measured"]["decisions_per_second"] for row in cases)
    for row in cases:
        row["relative_throughput"] = row["measured"]["decisions_per_second"] / fastest
    write_new(output / "results.json", {"passed": True, "cases": cases, "case_order": order})
    print("CPU_BENCH_PASSED " + str(output / "results.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--decisions", type=int, default=128)
    parser.add_argument("--order-seed", type=int, default=55)
    parser.add_argument("--worker-cpus", help=argparse.SUPPRESS)
    args = parser.parse_args()
    assert args.warmup >= 10 and args.decisions >= 100
    assert SEED_STEPS + 1 + args.warmup + args.decisions < 1000, "Keep this short benchmark inside one post-seed episode"
    (worker if args.worker_cpus else driver)(args)


if __name__ == "__main__":
    main()
