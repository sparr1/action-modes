"""GPU validation, CPU sizing, and gated launch of the two prior entropy runs."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_transfer_training import binding, recipe, runtime, write_new
from slurm.benchmark_ambi_training_cpu import TrainingWindow, physical_core_cpus

MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_prior_entropy.json"


def worker(args):
    cpus = [int(cpu) for cpu in args.worker_cpus.split(",")]
    assert set(cpus) <= os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpus)
    import gymnasium as gym
    import torch
    import domains  # noqa: F401
    from RL.AMBITDMPC2 import AMBITDMPC2

    torch.set_num_threads(len(cpus))
    torch.set_num_interop_threads(1)
    name, manifest, run = recipe(args.index, MANIFEST)
    run, manifest = deepcopy(run), deepcopy(manifest)
    overrides = {"wandb": False, "seed_steps": 500, "pretrain_steps": 2}
    run["alg_params"].update(overrides)
    warmup, measured = 128, 384
    total = 501 + warmup + measured
    run.update(total_steps=total, checkpoint_every=None, save_strat="none")
    manifest.update(logs="none", save_trials="none", checkpoint_every=None, save_strat="none")
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    started = time.perf_counter()
    try:
        learner = AMBITDMPC2(name + "-disabled-benchmark", env, run["alg_params"], run, manifest)
        agent, cfg = learner.agent, learner.cfg
        assert cfg.action_dim == 21 and cfg.inner_operator == "none" and not cfg.mpc
        assert cfg.compile and cfg.compile_strict and cfg.critic_ensemble_backend == "vmap"
        assert cfg.outer_critic_target == "reward_only" and cfg.aux_return_mode == "off"
        assert cfg.sac_actor_loss_scale_mode == "tdmpc2_percentile_range"
        timer = TrainingWindow(502 + warmup, measured, torch.cuda.synchronize)
        act, log = learner._act_agent, learner._log_wandb_step
        graph_counts = {}

        def observed_act(*positional, **kwargs):
            step = learner._global_step + 1
            if step == timer.first_step:
                graph_counts["before"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
            timer.before_action(step, learner._num_updates)
            t = time.perf_counter()
            action = act(*positional, **kwargs)
            timer.after_action(step, time.perf_counter() - t)
            assert torch.isfinite(action).all()
            assert agent.last_inner_metrics.get("inner_actor_optimizer_steps", 0) == 0
            assert agent.last_inner_metrics.get("inner_critic_optimizer_steps", 0) == 0
            return action

        def observed_log(*positional, **kwargs):
            result = log(*positional, **kwargs)
            timer.after_decision(learner._global_step, learner._num_updates)
            return result

        learner._act_agent, learner._log_wandb_step = observed_act, observed_log
        learner.learn(total)
        assert timer.result and learner._global_step == total
        graph_counts["after"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
        assert graph_counts["before"] == graph_counts["after"], graph_counts
        metrics = learner._last_train_metrics
        assert metrics["compile_fallback"] == 0
        assert all(math.isfinite(float(v)) for v in metrics.values())
        for key in ("actor_pre_tanh_entropy", "actor_post_tanh_entropy",
                    "actor_mean_action_abs_ge_0p99_fraction",
                    "actor_entropy_coefficient_joint_units", "actor_q_range_instant"):
            assert key in metrics, key
        assert all(torch.isfinite(p).all() for p in agent.model.parameters())
        assert learner.buffer._storage_device.type == "cuda"
        with tempfile.TemporaryDirectory(prefix="prior-entropy-checkpoint-") as temp:
            checkpoint = Path(learner.save(temp, "checkpoint"))
            learner.flush_checkpoints()
            assert checkpoint.is_file() and Path(str(checkpoint) + ".metadata.json").is_file()
            size = checkpoint.stat().st_size
            scale = agent.actor_loss_scale.detach().clone()
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            agent.load(checkpoint)
            for key, value in agent.model.state_dict().items():
                torch.testing.assert_close(value.cpu(), saved["model"][key], rtol=0, atol=0)
            torch.testing.assert_close(agent.actor_loss_scale, scale, rtol=0, atol=0)
        report = {"passed": True, "index": args.index, "name": name,
                  "physical_cpus": len(cpus), "runtime": runtime(),
                  "binding": binding(MANIFEST), "measured": timer.result,
                  "checkpoint_bytes": size, "checkpoint_roundtrip": True,
                  "checkpoint_scale_roundtrip": True, "checkpoint_sidecar": True,
                  "compile_counts": graph_counts, "compile_fallback": False,
                  "gpu_replay": True, "metrics": {k: float(v) for k, v in metrics.items()},
                  "diagnostic_episodes": learner._training_diagnostics.episode_metrics(),
                  "seconds": time.perf_counter() - started,
                  "benchmark_overrides": {**overrides, "total_steps": total}, "wandb": False}
        write_new(args.output, report)
    finally:
        if learner is not None:
            learner.close()
        env.close()


def gate(args):
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("WANDB_MODE") == "disabled"
    cores = physical_core_cpus(os.sched_getaffinity(0))
    assert len(cores) >= 2
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for index in (0, 1):
        for count in (1, 2):
            output = args.output / f"arm-{index}-cpu-{count}.json"
            env = dict(os.environ)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[key] = str(count)
            command = [sys.executable, str(Path(__file__).resolve()), "worker", "--index", str(index),
                       "--worker-cpus", ",".join(map(str, cores[:count])), "--output", str(output)]
            with output.with_suffix(".log").open("x") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            row = json.loads(output.read_text())
            cases.append(row)
            print("GATE_CASE", index, count, row["measured"]["seconds_per_decision"], flush=True)
    plans = []
    for index in (0, 1):
        rows = [r for r in cases if r["index"] == index]
        fastest = min(rows, key=lambda r: r["measured"]["seconds_per_decision"])
        # Prefer one core if two buy less than 5% throughput.
        one = next(r for r in rows if r["physical_cpus"] == 1)
        selected = one if one["measured"]["seconds_per_decision"] <= 1.05 * fastest["measured"]["seconds_per_decision"] else fastest
        seconds = selected["measured"]["seconds_per_decision"]
        plans.append({"index": index, "cpus": selected["physical_cpus"],
                      "seconds_per_decision": seconds,
                      "wall_hours": math.ceil((seconds * 1000000 * 1.25 + 900) / 3600),
                      "checkpoint_bytes": selected["checkpoint_bytes"]})
    bank_bytes = 40 * sum(p["checkpoint_bytes"] for p in plans)
    free = shutil.disk_usage(args.output).free
    assert free >= bank_bytes * 1.3 + 2 * 2**30, (free, bank_bytes)
    write_new(args.output / "receipt.json", {
        "passed": True, "binding": binding(MANIFEST), "runtime": cases[0]["runtime"],
        "plans": plans, "checkpoint_bank_bytes": bank_bytes, "free_bytes": free,
        "storage_margin": "30 percent plus 2 GiB", "cases": [r["name"] for r in cases],
        "scope": "Real Humanoid early-replay training, 1/2 cores, production model/batch/diagnostics; compilation excluded from steady-state timing; W&B disabled."})


def train(args):
    receipt = json.loads(args.gate.read_text())
    assert receipt["passed"] and receipt["binding"] == binding(MANIFEST)
    current = runtime()
    for key in ("python", "torch", "cuda", "gpu", "executable"):
        assert current[key] == receipt["runtime"][key], key
    plan = receipt["plans"][args.index]
    assert int(os.environ["SLURM_CPUS_PER_TASK"]) == plan["cpus"]
    assert os.environ.get("WANDB_RUN_ID")
    args.output.mkdir(parents=True, exist_ok=False)
    free = shutil.disk_usage(args.output).free
    assert free >= receipt["checkpoint_bank_bytes"] * 1.3 + 2 * 2**30
    write_new(args.output / "launch.json", {
        "binding": binding(MANIFEST), "plan": plan, "runtime": current,
        "output": str(args.output), "gate": str(args.gate), "free_bytes": free,
        "job_id": os.environ["SLURM_JOB_ID"], "wandb_run_id": os.environ["WANDB_RUN_ID"]})
    command = [sys.executable, "main.py", "--run", str(MANIFEST), "--alg-dir", "configs/dmcontrol/algs",
               "--log-dir", str(args.output), "--alg-index", str(args.index), "--trial-index", "0", "--num-runs", "1"]
    os.execv(sys.executable, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("gate", "worker", "train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--index", type=int, choices=(0, 1))
    parser.add_argument("--worker-cpus")
    args = parser.parse_args()
    {"gate": gate, "worker": worker, "train": train}[args.mode](args)


if __name__ == "__main__":
    main()
