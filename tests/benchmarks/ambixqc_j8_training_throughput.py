"""Short real-environment AMBI-XQC J8 timing, including replay and outer learning.

Matches H3/J8/N128/B256/C16/A4 through XQC's inner policy delay of four.
Keeps the baseline XQC architecture, soft critics, reward normalization and
optimizer semantics. This is an early-replay throughput test, not a learning
experiment: random collection/pretraining are shortened, W&B and saves disabled.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SOURCE_CONFIG = ROOT / "configs/dmcontrol/algs/ambixqc_humanoid_walk_backbone_replay_500k_baseline_utd1.json"
SEED_STEPS = 500
J8 = {
    "inner_operator": "xqc", "inner_rounds": 8,
    "inner_rollout_horizon": 3, "inner_rollouts_per_round": 128,
    "inner_batch_size": 256, "inner_replay_capacity": 3072,
    "inner_updates_per_round": 16, "inner_policy_delay": 4,
    "inner_update_timing": "round",
}


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def recipe(mode, compiled, warmup, measured):
    if mode not in {"prior", "j8"} or warmup < 2 or measured < 3:
        raise ValueError("Invalid mode or timing window")
    if SEED_STEPS + 1 + warmup + measured >= 1000:
        raise ValueError("Keep timing inside one post-seed episode")
    run = json.loads(SOURCE_CONFIG.read_text())
    run = deepcopy(run)
    params = run["alg_params"]
    params.update(J8)
    params.update(
        inner_operator="none" if mode == "prior" else "xqc",
        wandb=False, compile=compiled, compile_strict=compiled,
        seed_steps=SEED_STEPS, pretrain_steps=2,
    )
    run.update(total_steps=SEED_STEPS + 1 + warmup + measured,
               checkpoint_every=None, save_strat="none")
    manifest = {"env_params": {"task": "humanoid-walk", "obs": "state", "render_mode": None},
                "logs": "none", "save_trials": "none", "save_strat": "none",
                "checkpoint_every": None, "log_info": False}
    return run, manifest


class TrainingWindow:
    """Time whole decisions, starting before action and ending after learning."""

    def __init__(self, first, count, synchronize, wall=time.perf_counter, cpu=time.process_time):
        self.first, self.last = first, first + count - 1
        self.count, self.synchronize, self.wall, self.cpu = count, synchronize, wall, cpu
        self.rows = []
        self.result = None

    def before_action(self, step, updates):
        if step == self.first:
            self.synchronize()
            self.start = self.previous = self.wall()
            self.cpu_start, self.updates_start = self.cpu(), updates

    def after_decision(self, step, updates, action_seconds):
        if not self.first <= step <= self.last:
            return
        if step == self.last:
            self.synchronize()
        now = self.wall()
        self.rows.append({"step": step, "seconds": now - self.previous,
                          "action_seconds": action_seconds})
        self.previous = now
        if step == self.last:
            if [r["step"] for r in self.rows] != list(range(self.first, self.last + 1)):
                raise RuntimeError("Incomplete or duplicate measured decisions")
            if updates - self.updates_start != self.count:
                raise RuntimeError("Measured decisions must include exactly one outer update each")
            elapsed = now - self.start
            block_rates = []
            for start in range(0, self.count, max(1, self.count // 3)):
                block = self.rows[start:start + max(1, self.count // 3)]
                block_rates.append(len(block) / sum(r["seconds"] for r in block))
            self.result = {
                "decisions": self.count, "outer_updates": updates - self.updates_start,
                "wall_seconds": elapsed, "decisions_per_second": self.count / elapsed,
                "seconds_per_decision": elapsed / self.count,
                "average_cpu_cores_used": (self.cpu() - self.cpu_start) / elapsed,
                "action_seconds_mean": statistics.mean(r["action_seconds"] for r in self.rows),
                "block_decisions_per_second": block_rates,
                "projected_500k_hours": elapsed / self.count * 500000 / 3600,
                "rows": self.rows,
            }


def physical_cpus(allowed):
    cores = {}
    for cpu in sorted(allowed):
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = ((base / "physical_package_id").read_text().strip(),
               (base / "core_id").read_text().strip())
        cores.setdefault(key, cpu)
    return list(cores.values())


def worker(args):
    cpus = [int(value) for value in args.worker_cpus.split(",")]
    if not set(cpus) <= os.sched_getaffinity(0) or len(physical_cpus(cpus)) != len(cpus):
        raise RuntimeError("CPU selection is outside allocation or repeats physical cores")
    os.sched_setaffinity(0, cpus)
    import torch
    import gymnasium as gym
    import domains  # noqa: F401
    from RL.AMBIXQC import AMBIXQC

    torch.set_num_threads(len(cpus))
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Requires one scheduled CUDA GPU")
    run, manifest = recipe(args.mode, args.compile, args.warmup, args.measured)
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    try:
        learner = AMBIXQC("xqc-j8-throughput", env, run["alg_params"], run, manifest)
        cfg, agent = learner.cfg, learner.agent
        expected = (0, 0, 0, 0) if args.mode == "prior" else (128, 32, 32, 3072)
        resolved = (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action,
                    cfg.inner_temperature_updates_per_action, cfg.inner_model_step_budget)
        if resolved != expected or cfg.utd != 1 or cfg.xqc_utd != 1 or cfg.aux_return_mode != "off":
            raise RuntimeError(f"Unexpected resolved workload: {resolved}")
        timer = TrainingWindow(SEED_STEPS + args.warmup + 2, args.measured, torch.cuda.synchronize)
        act, log_step = learner._act_agent, learner._log_wandb_step
        graph_counts, actions, last_action_seconds = {}, 0, 0.0
        last_counts = None

        def observed_act(*positional, **kwargs):
            nonlocal actions, last_action_seconds, last_counts
            step = learner._global_step + 1
            if step == timer.first:
                graph_counts["before"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
                torch.cuda.reset_peak_memory_stats()
            timer.before_action(step, learner._num_updates)
            start = time.perf_counter()
            action = act(*positional, **kwargs)
            last_action_seconds = time.perf_counter() - start
            if action.device.type != "cpu" or not bool(torch.isfinite(action).all()):
                raise RuntimeError("Action must be finite and materialized on CPU")
            metrics = agent.last_inner_metrics
            last_counts = tuple(int(metrics[k]) for k in (
                "inner_critic_optimizer_steps", "inner_actor_optimizer_steps",
                "inner_temperature_optimizer_steps", "inner_model_steps"))
            if last_counts != expected:
                raise RuntimeError(f"Executed workload differs: {last_counts} != {expected}")
            actions += 1
            return action

        def observed_log(reward, *positional, **kwargs):
            if not math.isfinite(float(reward)):
                raise RuntimeError("Nonfinite environment reward")
            result = log_step(reward, *positional, **kwargs)
            timer.after_decision(learner._global_step, learner._num_updates, last_action_seconds)
            return result

        learner._act_agent, learner._log_wandb_step = observed_act, observed_log
        torch.cuda.synchronize()
        started = time.perf_counter()
        learner.learn(run["total_steps"])
        torch.cuda.synchronize()
        total_seconds = time.perf_counter() - started
        graph_counts["after"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
        if timer.result is None or actions != args.warmup + args.measured:
            raise RuntimeError("Missing timing window or incorrect action count")
        if graph_counts["before"] != graph_counts["after"]:
            raise RuntimeError("Compilation occurred during measurement; increase warmup")
        statuses = {"outer": dict(agent.xqc_controller.compile_status)}
        if args.mode == "j8":
            statuses["inner"] = dict(agent.inner_engine._workspace_pool.controller.compile_status)
            statuses["rollout"] = dict(agent.inner_engine.rollout_compile_status)
        if any(s.get("fallback", False) for s in statuses.values()):
            raise RuntimeError("Compile fallback")
        if args.compile:
            for key in ("outer", "inner"):
                if key in statuses and not all(statuses[key].get(k) for k in ("critic_compiled", "actor_compiled")):
                    raise RuntimeError(f"Missing compiled region: {key}")
            if args.mode == "j8" and not statuses["rollout"].get("compiled"):
                raise RuntimeError("Inner rollout did not compile")
        if not all(bool(torch.isfinite(p).all()) for p in agent.parameters()):
            raise RuntimeError("Nonfinite learned parameters")
        result = {
            "passed": True, "mode": args.mode, "compiled": args.compile,
            "physical_cores": len(cpus), "cpu_ids": cpus,
            "host": os.uname().nodename, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "device": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_config_sha256": hashlib.sha256(SOURCE_CONFIG.read_bytes()).hexdigest(),
            "run_config": run, "resolved_config": vars(cfg),
            "warmup_active_decisions": args.warmup, "total_training_seconds": total_seconds,
            "measured": timer.result, "compile_graph_counts": graph_counts, "compile_status": statuses,
            "last_inner_counts": last_counts,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "replay_transitions": learner.buffer.num_transitions,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Early-replay real training; excludes startup/compilation, W&B, periodic diagnostics and checkpoint I/O. Shortened 500-step random collection and 2-update pretraining; XQC schedule follows shortened run budget.",
        }
        write_new(args.output, result)
        print("XQC_TIMING_PASSED " + json.dumps({"mode": args.mode, "compile": args.compile,
              "cpus": len(cpus), "measured": {k: v for k, v in timer.result.items() if k != "rows"}}), flush=True)
    finally:
        if learner is not None:
            learner.flush_checkpoints()
            learner.agent.inner_engine.clear_all()
        env.close()


def driver(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Submit the driver through Slurm")
    cpus = physical_cpus(os.sched_getaffinity(0))
    if len(cpus) < 2:
        raise RuntimeError("Two physical cores required for the CPU comparison")
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Write benchmark results outside source")
    output.mkdir(parents=True, exist_ok=False)
    # Eager first also warms shared libraries; compiler startup is excluded.
    cases = [("prior", False, 1), ("j8", False, 1), ("j8", False, 2),
             ("prior", True, 1), ("j8", True, 1), ("j8", True, 2)]
    cache = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / ("xqc-j8-" + os.environ["SLURM_JOB_ID"])
    write_new(output / "protocol.json", {"cases": cases, "warmup": args.warmup,
               "measured": args.measured, "j8": J8, "physical_cpu_ids": cpus})
    results = []
    for mode, compiled, count in cases:
        label = f"{mode}-{'compiled' if compiled else 'eager'}-cpu{count}"
        environment = dict(os.environ)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[key] = str(count)
        environment.update(WANDB_MODE="disabled", WANDB_DISABLED="true", MUJOCO_GL="egl",
            PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", TORCHINDUCTOR_COMPILE_THREADS="1",
            TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), TRITON_CACHE_DIR=str(cache / "triton"))
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-cpus", ",".join(map(str, cpus[:count])),
                   "--mode", mode, "--compile" if compiled else "--no-compile",
                   "--warmup", str(args.warmup), "--measured", str(args.measured), "--output", str(output / (label + ".json"))]
        print("XQC_TIMING_START " + label, flush=True)
        with (output / (label + ".log")).open("x") as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads((output / (label + ".json")).read_text())
        results.append(result)
        print("XQC_TIMING_RESULT " + json.dumps({"case": label, "decisions_per_second": result["measured"]["decisions_per_second"]}), flush=True)
    write_new(output / "results.json", {"passed": True, "cases": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--measured", type=int, default=96)
    parser.add_argument("--worker-cpus", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("prior", "j8"), default="j8")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    recipe(args.mode, args.compile, args.warmup, args.measured)
    (worker if args.worker_cpus else driver)(args)


if __name__ == "__main__":
    main()
