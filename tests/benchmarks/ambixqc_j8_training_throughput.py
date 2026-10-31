"""Short real-environment AMBI-XQC J8 timing, including replay and outer learning.

Matches H3/J8/N128/B256/C16/A4 through XQC's inner policy delay of four.
Keeps the baseline XQC architecture, soft critics, reward normalization and
optimizer semantics. This is an early-replay throughput test, not a learning
experiment: random collection/pretraining are shortened, W&B and saves disabled.

--solve-interval 3 enables a benchmark-only held-policy adapter: snapshot the
encoder and adapted actor at each solve, then evaluate that policy on each new
observation for three real decisions. Outer learning still runs every decision.
This matches native AMBI's K semantics; it does not add a production XQC config
option, checkpoint format, or resume support. --single-case runs only the
requested mode/compile/cadence on one physical CPU for parallel Slurm trials.
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
WORK_KEYS = ("inner_critic_optimizer_steps", "inner_actor_optimizer_steps",
             "inner_temperature_optimizer_steps", "inner_model_steps")


class HeldPolicyBenchmark:
    """Timing-only adapter, installed on an agent with no saves/evaluation loop.

    Actor parameters and running BN buffers are copied, as is the encoder.
    Held decisions consume only execution RNG, and never touch the pooled
    learner, replay, optimizers, or their RNG streams. Reset/mode change starts
    a new block; failed actions invalidate the current block.
    """

    def __init__(self, agent, interval):
        if interval < 2 or agent.cfg.inner_operator != "xqc" or agent.cfg.obs != "state":
            raise ValueError("Held timing requires state XQC and K >= 2")
        self.agent, self.interval = agent, interval
        self.original_act, self.original_reset = agent.act, agent.reset
        self.encoder = self.actor = None
        self.clear()
        agent.act, agent.reset = self.act, self.reset

    def clear(self):
        self.index, self.mode, self.ready = 0, None, False

    def reset(self):
        self.clear()
        return self.original_reset()

    @staticmethod
    def snapshot(source, destination):
        if destination is None:
            destination = deepcopy(source)
        else:
            destination.load_state_dict(source.state_dict())
        return destination.requires_grad_(False).eval()

    def act(self, obs, t0=False, eval_mode=False, task=None, *, collect_diagnostics=True):
        import torch

        if task is not None:
            raise ValueError("Held timing supports single-task training only")
        if t0 or self.mode != bool(eval_mode):
            self.clear()
        self.mode = bool(eval_mode)
        age = self.index % self.interval
        try:
            if age == 0:
                with torch.no_grad():
                    self.encoder = self.snapshot(self.agent.model._encoder["state"], self.encoder)
                action = self.original_act(obs, t0=t0, eval_mode=eval_mode,
                                           collect_diagnostics=collect_diagnostics)
                with torch.no_grad():
                    source = self.agent.inner_engine._workspace_pool.controller.actor
                    self.actor = self.snapshot(source, self.actor)
                self.ready = True
            else:
                if not self.ready:
                    raise RuntimeError("Held action requires a completed solve")
                with torch.no_grad():
                    z = self.encoder(obs.to(self.agent.device, non_blocking=True).unsqueeze(0))
                    # Match XQC's explicit execution-noise draw, including eval.
                    noise = torch.randn((*z.shape[:-1], int(self.agent.cfg.action_dim)),
                                        dtype=z.dtype, device=z.device,
                                        generator=self.agent.inner_engine.rng.generator("execution"))
                    action, _ = self.actor.sample(z, deterministic=bool(eval_mode),
                                                  bn_mode="running", noise=noise)
                    action = action[0].detach().cpu()
                self.agent.last_inner_metrics = dict.fromkeys(WORK_KEYS, 0.0)
                self.agent.last_inner_metrics.update(inner_active=0.0, inner_diagnostics_sampled=0.0)
                self.agent.last_inner_rollout_lengths = []
            self.agent.last_inner_metrics.update(
                inner_solve_performed=float(age == 0), inner_policy_held=float(age != 0),
                inner_solve_interval=float(self.interval), inner_action_age=float(age),
                inner_episode_decision_index=float(self.index),
            )
            self.index += 1
            return action
        except BaseException:
            self.clear()
            raise


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def recipe(mode, compiled, warmup, measured, solve_interval=1):
    if mode not in {"prior", "j8"} or warmup < 2 or measured < 3:
        raise ValueError("Invalid mode or timing window")
    if SEED_STEPS + 1 + warmup + measured >= 1000:
        raise ValueError("Keep timing inside one post-seed episode")
    if solve_interval not in {1, 3} or (solve_interval > 1 and mode != "j8"):
        raise ValueError("Supported solve intervals are K1 or J8/K3")
    if warmup % solve_interval or measured % solve_interval:
        raise ValueError("Warmup and measured decisions must contain complete solve blocks")
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
    run, manifest = recipe(args.mode, args.compile, args.warmup, args.measured, args.solve_interval)
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    try:
        learner = AMBIXQC("xqc-j8-throughput", env, run["alg_params"], run, manifest)
        cfg, agent = learner.cfg, learner.agent
        if args.solve_interval > 1:
            HeldPolicyBenchmark(agent, args.solve_interval)
        expected = (0, 0, 0, 0) if args.mode == "prior" else (128, 32, 32, 3072)
        resolved = (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action,
                    cfg.inner_temperature_updates_per_action, cfg.inner_model_step_budget)
        if resolved != expected or cfg.utd != 1 or cfg.xqc_utd != 1 or cfg.aux_return_mode != "off":
            raise RuntimeError(f"Unexpected resolved workload: {resolved}")
        timer = TrainingWindow(SEED_STEPS + args.warmup + 2, args.measured, torch.cuda.synchronize)
        act, log_step = learner._act_agent, learner._log_wandb_step
        graph_counts, actions, last_action_seconds = {}, 0, 0.0
        last_counts = None
        work_rows = []

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
            last_counts = tuple(int(metrics[k]) for k in WORK_KEYS)
            age = actions % args.solve_interval
            solve = args.mode == "j8" and age == 0
            expected_this_action = expected if solve else (0, 0, 0, 0)
            if last_counts != expected_this_action:
                raise RuntimeError(f"Executed workload differs: {last_counts} != {expected_this_action}")
            if args.solve_interval > 1 and (metrics["inner_action_age"] != age
                                           or metrics["inner_solve_performed"] != float(solve)):
                raise RuntimeError("Incorrect held-policy cadence")
            if timer.first <= step <= timer.last:
                work_rows.append({"step": step, "solve": solve, "age": age,
                                  "counts": last_counts, "action_seconds": last_action_seconds})
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
        solves = sum(row["solve"] for row in work_rows)
        expected_solves = args.measured // args.solve_interval if args.mode == "j8" else 0
        if len(work_rows) != args.measured or solves != expected_solves:
            raise RuntimeError("Incomplete measured solve/held blocks")
        timer.result.update(
            inner_solves=solves, held_decisions=args.measured - solves if args.mode == "j8" else 0,
            inner_work_totals=dict(zip(WORK_KEYS, (sum(row["counts"][i] for row in work_rows)
                                                 for i in range(len(WORK_KEYS))))),
        )
        for kind, is_solve in (("solve", True), ("held", False)):
            times = [row["action_seconds"] for row in work_rows if row["solve"] == is_solve]
            if times:
                timer.result[kind + "_action_seconds_mean"] = statistics.mean(times)
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
            "inner_solve_interval": args.solve_interval,
            "held_policy_implementation": "benchmark_adapter_solve_snapshot" if args.solve_interval > 1 else None,
            "physical_cores": len(cpus), "cpu_ids": cpus,
            "host": os.uname().nodename, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "device": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_config_sha256": hashlib.sha256(SOURCE_CONFIG.read_bytes()).hexdigest(),
            "run_config": run, "resolved_config": vars(cfg),
            "warmup_active_decisions": args.warmup, "total_training_seconds": total_seconds,
            "measured": timer.result, "compile_graph_counts": graph_counts, "compile_status": statuses,
            "last_inner_counts": last_counts,
            "measured_work": work_rows,
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
    if len(cpus) < (1 if args.single_case else 2):
        raise RuntimeError("Two physical cores required for the CPU comparison")
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Write benchmark results outside source")
    output.mkdir(parents=True, exist_ok=False)
    # Eager first also warms shared libraries; compiler startup is excluded.
    cases = [("prior", False, 1), ("j8", False, 1), ("j8", False, 2),
             ("prior", True, 1), ("j8", True, 1), ("j8", True, 2)]
    if args.single_case:
        cases = [(args.mode, args.compile, 1)]
    elif args.solve_interval != 1:
        raise ValueError("K3 requires --single-case")
    cache = Path(os.environ.get("SLURM_TMPDIR", "/tmp")) / ("xqc-j8-" + os.environ["SLURM_JOB_ID"])
    write_new(output / "protocol.json", {"cases": cases, "warmup": args.warmup,
               "measured": args.measured, "j8": J8, "physical_cpu_ids": cpus,
               "inner_solve_interval": args.solve_interval})
    results = []
    for mode, compiled, count in cases:
        label = f"{mode}-{'compiled' if compiled else 'eager'}-cpu{count}"
        if args.solve_interval > 1:
            label += f"-k{args.solve_interval}"
        environment = dict(os.environ)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[key] = str(count)
        environment.update(WANDB_MODE="disabled", WANDB_DISABLED="true", MUJOCO_GL="egl",
            PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", TORCHINDUCTOR_COMPILE_THREADS="1",
            TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), TRITON_CACHE_DIR=str(cache / "triton"))
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-cpus", ",".join(map(str, cpus[:count])),
                   "--mode", mode, "--compile" if compiled else "--no-compile",
                   "--solve-interval", str(args.solve_interval),
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
    parser.add_argument("--solve-interval", type=int, choices=(1, 3), default=1)
    parser.add_argument("--single-case", action="store_true")
    args = parser.parse_args()
    recipe(args.mode, args.compile, args.warmup, args.measured, args.solve_interval)
    (worker if args.worker_cpus else driver)(args)


if __name__ == "__main__":
    main()
