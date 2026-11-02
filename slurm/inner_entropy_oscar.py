"""Per-arm GPU validation, measured resource sizing, and six-cell Oscar launch."""
import argparse
from copy import deepcopy
import json
import math
import os
import runpy
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

MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_inner_entropy.json"


def accelerated_runtime():
    import torch
    torch.set_float32_matmul_precision("high")
    return {**runtime(), "float32_matmul_precision": torch.get_float32_matmul_precision()}


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
    execution = accelerated_runtime()
    name, manifest, run = recipe(args.index, MANIFEST)
    run, manifest = deepcopy(run), deepcopy(manifest)
    overrides = {"wandb": False, "seed_steps": 500, "pretrain_steps": 2}
    run["alg_params"].update(overrides)
    warmup, measured = 8, 24
    total = 501 + warmup + measured
    run.update(total_steps=total, checkpoint_every=None, save_strat="none")
    manifest.update(logs="none", save_trials="none", checkpoint_every=None, save_strat="none")
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    started = time.perf_counter()
    try:
        learner = AMBITDMPC2(name + "-disabled-benchmark", env, run["alg_params"], run, manifest)
        agent, cfg = learner.agent, learner.cfg
        assert cfg.action_dim == 21 and cfg.inner_operator == "sac" and not cfg.mpc
        assert (cfg.inner_rollout_horizon, cfg.inner_rounds, cfg.inner_solve_interval) == (3, 8, 1)
        assert cfg.inner_finite_horizon
        assert cfg.outer_actor_entropy_mode == cfg.inner_actor_entropy_mode
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
            before = agent.actor_loss_scale.detach().clone()
            carried = agent.inner_engine._carried_actor_loss_scale
            initial = before if carried is None else carried.detach().clone()
            action = act(*positional, **kwargs)
            timer.after_action(step, time.perf_counter() - t)
            assert torch.isfinite(action).all()
            inner = agent.last_inner_metrics
            assert inner["inner_actor_optimizer_steps"] == 32
            assert inner["inner_critic_optimizer_steps"] == 128
            assert inner["inner_temperature_optimizer_steps"] == 0
            assert inner["inner_compile_fallback"] == 0
            assert all(math.isfinite(float(v)) for v in inner.values())
            expected = initial if cfg.inner_actor_loss_scale_scope == "run" else before
            assert math.isclose(inner["inner_actor_loss_scale_initial"], expected.item(), rel_tol=1e-6)
            if cfg.inner_actor_loss_scale_update == "per_action":
                assert inner["inner_actor_loss_scale_final"] == inner["inner_actor_loss_scale_initial"]
            torch.testing.assert_close(agent.actor_loss_scale, before, rtol=0, atol=0)
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
            engine = agent.inner_engine
            engine.prepare_training_resume_boundary()
            inner_state = deepcopy(engine.training_state_dict())
            engine.load_training_state_dict(inner_state)
            if cfg.inner_actor_loss_scale_scope == "run":
                torch.testing.assert_close(engine._carried_actor_loss_scale,
                                           inner_state["actor_loss_scale_carry"], rtol=0, atol=0)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            agent.load(checkpoint)
            for key, value in agent.model.state_dict().items():
                torch.testing.assert_close(value.cpu(), saved["model"][key], rtol=0, atol=0)
            torch.testing.assert_close(agent.actor_loss_scale, scale, rtol=0, atol=0)
        report = {"passed": True, "index": args.index, "name": name,
                  "physical_cpus": len(cpus), "runtime": execution,
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


def checkpoint_space(root, bank_bytes):
    # Shared-filesystem free space alone does not represent this user's quota.
    quota = subprocess.check_output(["checkquota"], text=True)
    rows = [line.split() for line in quota.splitlines()]
    row = next(r for r in rows if len(r) > 5 and r[0] == "rgao48" and r[1] == "/oscar/scratch")
    quota_free = int((float(row[4]) - float(row[2])) * 2**30)
    disk_free = shutil.disk_usage(root).free
    required = bank_bytes * 1.3 + 2 * 2**30
    assert min(quota_free, disk_free) >= required, (quota_free, disk_free, required)
    return {"quota_free_bytes": quota_free, "filesystem_free_bytes": disk_free,
            "checkpoint_bank_bytes": bank_bytes, "required_with_margin_bytes": required}


def select_plan(rows):
    fastest = min(rows, key=lambda r: r["measured"]["seconds_per_decision"])
    one = next(r for r in rows if r["physical_cpus"] == 1)
    selected = one if one["measured"]["seconds_per_decision"] <= 1.05 * fastest["measured"]["seconds_per_decision"] else fastest
    seconds = selected["measured"]["seconds_per_decision"]
    return {"index": selected["index"], "cpus": selected["physical_cpus"],
            "seconds_per_decision": seconds,
            "wall_hours": math.ceil(seconds * 1000000 * 1.25 / 3600 + .5),
            "checkpoint_bytes": selected["checkpoint_bytes"]}


def slurm_fields(job):
    return dict(item.split("=", 1) for item in subprocess.check_output(
        ["scontrol", "show", "job", str(job), "-o"], text=True).split() if "=" in item)


def resize_pending(job, plan):
    fields = slurm_fields(job)
    assert fields["JobState"] == "PENDING" and fields["UserId"].startswith("rgao48("), fields
    assert fields["WorkDir"] == str(ROOT), fields["WorkDir"]
    subprocess.run(["scontrol", "hold", str(job)], check=True)
    count = str(plan["cpus"])
    subprocess.run(["scontrol", "update", f"JobId={job}",
                    f"TimeLimit={plan['wall_hours']}:00:00", f"NumCPUs={count}",
                    f"CPUsPerTask={count}", f"TresPerTask=cpu:{count}",
                    f"MinCPUsNode={count}"], check=True)
    fields = slurm_fields(job)
    assert fields["JobState"] == "PENDING"
    for name in ("NumCPUs", "CPUs/Task", "MinCPUsNode"):
        assert fields[name] == count, (name, fields)
    assert fields["TresPerTask"] in {f"cpu:{count}", f"cpu={count}"}, fields
    limit = fields["TimeLimit"]
    days, clock = limit.split("-") if "-" in limit else ("0", limit)
    hh, mm, ss = map(int, clock.split(":"))
    assert int(days) * 24 + hh == plan["wall_hours"] and mm == ss == 0, fields
    subprocess.run(["scontrol", "release", str(job)], check=True)
    return fields


def gate(args):
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("WANDB_MODE") == "disabled"
    cores = physical_core_cpus(os.sched_getaffinity(0))
    assert len(cores) >= 2
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for count in (1, 2):
        output = args.output / f"cpu-{count}.json"
        env = dict(os.environ)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = str(count)
        command = [sys.executable, str(Path(__file__).resolve()), "worker", "--index", str(args.index),
                   "--worker-cpus", ",".join(map(str, cores[:count])), "--output", str(output)]
        with output.with_suffix(".log").open("x") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows.append(json.loads(output.read_text()))
    plan = select_plan(rows)
    bank_bytes = 6 * 40 * max(r["checkpoint_bytes"] for r in rows)
    space = checkpoint_space(args.output, bank_bytes)
    receipt = {"passed": True, "binding": binding(MANIFEST), "runtime": rows[0]["runtime"],
               "plan": plan, "space": space, "cases": [str(args.output/f"cpu-{n}.json") for n in (1,2)],
               "scope": "Real H3/J8/K1 Humanoid training, production model/batch/diagnostics; 1/2 CPUs; no W&B; portable checkpoint and carried-scale resume checks."}
    write_new(args.output / "receipt.json", receipt)
    campaign = Path(os.environ["AMBI_ENTROPY_CAMPAIGN"])
    submission = json.loads((campaign / "submission.json").read_text())
    assert submission["commit"] == receipt["binding"]["commit"]
    arm = submission["runs"][args.index]
    assert str(arm["gate_job"]) == os.environ["SLURM_JOB_ID"]
    applied = resize_pending(arm["job"], plan)
    write_new(args.output / "allocation.json", {"production_job": arm["job"], "plan": plan, "verified": applied})
    print("GATE_PASSED_AND_PRODUCTION_RELEASED", arm["job"], flush=True)


def train(args):
    import torch
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    torch.set_num_interop_threads(1)
    receipt = json.loads(args.gate.read_text())
    assert receipt["passed"] and receipt["binding"] == binding(MANIFEST)
    current = accelerated_runtime()
    for key in ("python", "torch", "cuda", "gpu", "executable", "float32_matmul_precision"):
        assert current[key] == receipt["runtime"][key], key
    plan = receipt["plan"]
    assert plan["index"] == args.index
    assert int(os.environ["SLURM_CPUS_PER_TASK"]) == plan["cpus"]
    assert os.environ.get("WANDB_RUN_ID")
    args.output.mkdir(parents=True, exist_ok=False)
    space = checkpoint_space(args.output, receipt["space"]["checkpoint_bank_bytes"])
    write_new(args.output / "launch.json", {
        "binding": binding(MANIFEST), "plan": plan, "runtime": current,
        "output": str(args.output), "gate": str(args.gate), "space": space,
        "job_id": os.environ["SLURM_JOB_ID"], "wandb_run_id": os.environ["WANDB_RUN_ID"]})
    command = [sys.executable, "main.py", "--run", str(MANIFEST), "--alg-dir", "configs/dmcontrol/algs",
               "--log-dir", str(args.output), "--alg-index", str(args.index), "--trial-index", "0", "--num-runs", "1"]
    # Keep the validated matmul setting in the same process as the trainer.
    sys.argv = command[1:]
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("gate", "worker", "train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--index", type=int, choices=range(6))
    parser.add_argument("--worker-cpus")
    args = parser.parse_args()
    {"gate": gate, "worker": worker, "train": train}[args.mode](args)


if __name__ == "__main__":
    main()
