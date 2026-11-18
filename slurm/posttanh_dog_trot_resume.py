"""Restart Dog Trot seed56 with the native full-state resume contract."""

import argparse
from copy import deepcopy
import json
import math
import tempfile
import time
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_posttanh_outer_adapt_dog_trot_seed56_resume.json"
DURABLE_ROOT = Path("/oscar/home/rgao48/ambi-durable")


def validate_recipe():
    from slurm.ambi_transfer_training import recipe
    from slurm.posttanh_dog_seed_replicates import validate_recipe as original_recipe

    _, original_manifest, original = original_recipe("trot", 0)
    name, manifest, run = recipe(0, MANIFEST)
    comparable = deepcopy(run)
    comparable["name"] = original["name"]
    for key in ("wandb_run_name", "wandb_tags", "wandb_group"):
        comparable["alg_params"][key] = original["alg_params"][key]
    assert comparable == original, "Restart changed scientific parameters"
    assert manifest["configs"] == [name]
    for key in ("overrides_alg", "env_params", "trials", "logs", "save_trials",
                "checkpoint_every", "save_strat", "log_info", "log_type"):
        assert manifest[key] == original_manifest[key], key
    return name, manifest, run


def train(args):
    import torch
    from slurm.ambi_transfer_training import binding, write_new
    from slurm.inner_entropy_oscar import accelerated_runtime
    from slurm.posttanh_dog_seed_replicates import validate_reference

    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("WANDB_MODE") == "online"
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == os.environ["AMBI_EXPECTED_COMMIT"]
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True).strip()
    name, _, run = validate_recipe()
    reference_path, reference = validate_reference("trot")
    torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
    torch.set_num_interop_threads(1)
    runtime = accelerated_runtime()
    assert runtime == reference["runtime"], (runtime, reference["runtime"])
    lineage = Path(os.environ["AMBI_LINEAGE_DIR"]).resolve()
    assert lineage.is_relative_to(DURABLE_ROOT.resolve())
    assert args.output.is_absolute() and not args.output.is_relative_to(ROOT)
    assert args.snapshot_root.is_absolute() and not args.snapshot_root.is_relative_to(ROOT)
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / "launch.json", {
        "binding": binding(MANIFEST), "config": name, "seed": run["seed"],
        "reference_gate": str(reference_path), "runtime": runtime,
        "resume_mode": args.resume_mode, "lineage_dir": str(lineage),
        "resume_checkpoint_minutes": args.checkpoint_minutes,
        "drain_after_seconds": args.drain_seconds,
        "snapshot_root": str(args.snapshot_root),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "segment_id": os.environ["AMBI_SEGMENT_ID"],
    })
    sys.argv = ["main.py", "--run", str(MANIFEST), "--alg-dir", "configs/dmcontrol/algs",
                "--num-runs", "1", "--log-dir", str(args.output),
                "--lineage-dir", str(lineage), "--resume-mode", args.resume_mode,
                "--resume-wandb-mode", "online", "--resume-model-snapshot-root", str(args.snapshot_root),
                "--resume-checkpoint-minutes", str(args.checkpoint_minutes),
                "--drain-after-seconds", str(args.drain_seconds)]
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")


def benchmark_worker(args):
    import resource
    from slurm.ambi_transfer_training import binding, recipe, write_new
    from slurm.benchmark_ambi_training_cpu import TrainingWindow
    from slurm.inner_entropy_oscar import accelerated_runtime, check_geometry
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
    graph_enabled = not args.native_reference
    overrides = {"wandb": False, "wandb_mode": "disabled", "seed_steps": 500,
                 "pretrain_steps": 2, "inner_cuda_graph": graph_enabled}
    run["alg_params"].update(overrides)
    warmup, measured = 8, 32 if args.native_reference else 128
    total = 501 + warmup + measured
    run.update(total_steps=total, checkpoint_every=None, save_strat="none")
    manifest.update(logs="none", save_trials="none", checkpoint_every=None, save_strat="none")
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    started = time.perf_counter()
    try:
        learner = AMBITDMPC2(name + "-disabled-benchmark", env, run["alg_params"], run, manifest)
        from RL.tdmpc2_core.common.buffer import Buffer
        learner.buffer = Buffer(learner.cfg, resumable=True)
        learner.agent.set_outer_replay_buffer(learner.buffer)
        agent, cfg = learner.agent, learner.cfg
        expected_shape = (223, 38) if manifest["env_params"]["task"].startswith("dog-") else (67, 21)
        assert (env.observation_space.shape[0], env.action_space.shape[0]) == expected_shape
        assert cfg.action_dim == expected_shape[1] and cfg.inner_operator == "sac" and not cfg.mpc
        assert env.unwrapped.action_repeat == 2 and env.spec.max_episode_steps == 500
        assert (cfg.inner_rollout_horizon, cfg.inner_rounds, cfg.inner_solve_interval) == (3, 8, 1)
        assert cfg.inner_finite_horizon
        assert cfg.outer_actor_entropy_mode == cfg.inner_actor_entropy_mode
        assert cfg.compile and cfg.compile_strict and cfg.critic_ensemble_backend == "vmap"
        assert cfg.inner_cuda_graph is graph_enabled
        assert cfg.outer_critic_target == "reward_only" and cfg.aux_return_mode == "off"
        assert cfg.sac_actor_loss_scale_mode == "tdmpc2_percentile_range"
        timer = TrainingWindow(502 + warmup, measured, torch.cuda.synchronize)
        act, log = learner._act_agent, learner._log_wandb_step
        graph_counts = {}
        replay_counts = {}

        def observed_act(*positional, **kwargs):
            step = learner._global_step + 1
            if step == timer.first_step:
                graph_counts["before"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
                runner = agent.inner_engine._whole_graph_runner
                replay_counts["before"] = 0 if runner is None else runner.calls
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
            # Native-only execution need not emit the graph telemetry key;
            # graph-enabled measurements still require an explicit one.
            assert inner.get("inner_cuda_graph_used", 0.0) == float(graph_enabled)
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
        runner = agent.inner_engine._whole_graph_runner
        replay_counts["after"] = 0 if runner is None else runner.calls
        replay_counts["timed"] = replay_counts["after"] - replay_counts["before"]
        assert replay_counts["timed"] == (measured if graph_enabled else 0), replay_counts
        metrics = learner._last_train_metrics
        assert metrics["compile_fallback"] == 0
        assert all(math.isfinite(float(v)) for v in metrics.values())
        for key in ("actor_pre_tanh_entropy", "actor_post_tanh_entropy",
                    "actor_mean_action_abs_ge_0p99_fraction",
                    "actor_entropy_coefficient_joint_units", "actor_q_range_instant"):
            assert key in metrics, key
        assert all(torch.isfinite(p).all() for p in agent.model.parameters())
        assert learner.buffer._storage_device.type == "cpu"
        # Expensive sparse observations and recapture checks are deliberately
        # outside the steady-state timing window.
        geometry = check_geometry(agent, learner.buffer)
        if graph_enabled:
            from tests.test_inner_entropy_graph import assert_graph_resume_continuation
            assert_graph_resume_continuation(agent)
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
                  "inner_cuda_graph": graph_enabled, "graph_replays": replay_counts,
                  "graph_native_graph_continuation": graph_enabled,
                  "critic_geometry": geometry,
                  "gpu_replay": False, "cpu_replay": True, "metrics": {k: float(v) for k, v in metrics.items()},
                  "diagnostic_episodes": learner._training_diagnostics.episode_metrics(),
                  "seconds": time.perf_counter() - started,
                  "peak_host_memory_bytes": int((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                      + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)),
                  "benchmark_overrides": {**overrides, "total_steps": total}, "wandb": False}
        write_new(args.output, report)
    finally:
        if learner is not None:
            learner.close()
        env.close()


def benchmark(args):
    from slurm.ambi_transfer_training import write_new
    from slurm.benchmark_ambi_training_cpu import physical_core_cpus

    assert os.environ.get("WANDB_MODE") == "disabled"
    cpus = physical_core_cpus(os.sched_getaffinity(0))
    assert len(cpus) >= 2
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for count in (1, 2):
        path = args.output / f"cpu{count}.json"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "benchmark-worker",
                        "--output", str(path), "--worker-cpus", ",".join(map(str, cpus[:count]))],
                       cwd=ROOT, check=True)
        reports.append(json.loads(path.read_text()))
    for report in reports:
        check_speed(report)
    fastest = min(row["measured"]["seconds_per_decision"] for row in reports)
    selected = min((row for row in reports if row["measured"]["seconds_per_decision"] <= fastest * 1.05),
                   key=lambda row: row["physical_cpus"])
    write_new(args.output / "plan.json", {"cpus": selected["physical_cpus"],
              "seconds_per_decision": selected["measured"]["seconds_per_decision"],
              "binding": selected["binding"], "runtime": selected["runtime"],
              "cpu_replay": True, "wandb": False,
              "max_seconds_per_decision": MAX_SECONDS_PER_DECISION,
              "peak_host_memory_bytes": max(row["peak_host_memory_bytes"] for row in reports),
              "wall_hours": math.ceil(max(50., selected["measured"]["seconds_per_decision"] * 1_000_000 / 3600) * 1.20 + 0.5)})



# Completed matching runs took about 50 h (~0.18 s/decision). The timed-out
# node delivered ~0.31 s/decision. Refuse that scale of slowdown before training.
MAX_SECONDS_PER_DECISION = 0.225


def check_speed(report):
    assert report["passed"] and report["cpu_replay"] and not report["gpu_replay"]
    assert report["wandb"] is False and report["compile_fallback"] is False
    elapsed = report["measured"]["seconds_per_decision"]
    assert 0 < elapsed <= MAX_SECONDS_PER_DECISION, (
        f"Slow allocation: {elapsed:.4f} s/decision exceeds {MAX_SECONDS_PER_DECISION:.3f}; "
        "refusing to start production training on this allocation")


def node_check(args):
    from slurm.ambi_transfer_training import binding
    from slurm.benchmark_ambi_training_cpu import physical_core_cpus
    assert os.environ.get("WANDB_MODE") == "disabled"
    cpus = physical_core_cpus(os.sched_getaffinity(0))
    count = int(os.environ["SLURM_CPUS_PER_TASK"])
    assert len(cpus) >= count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "benchmark-worker",
                    "--output", str(args.output), "--worker-cpus", ",".join(map(str, cpus[:count]))],
                   cwd=ROOT, check=True)
    report = json.loads(args.output.read_text())
    assert report["binding"] == binding(MANIFEST) and report["physical_cpus"] == count
    check_speed(report)
    print("NODE_THROUGHPUT_PASSED", report["measured"], flush=True)


def verify_allocation(args):
    from slurm.ambi_transfer_training import binding
    plan = json.loads((args.output / "plan.json").read_text())
    assert plan["binding"] == binding(MANIFEST)
    assert plan["cpu_replay"] and plan["wandb"] is False
    assert int(os.environ["SLURM_CPUS_PER_TASK"]) == plan["cpus"]
    value = subprocess.check_output(["squeue", "--noheader", "--jobs", os.environ["SLURM_JOB_ID"],
                                     "--format", "%l"], text=True).strip()
    days, clock = value.split("-") if "-" in value else ("0", value)
    parts = list(map(int, clock.split(":")))
    if len(parts) == 2:
        parts.insert(0, 0)
    assert len(parts) == 3
    seconds = int(days) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    assert seconds == int(os.environ["AMBI_TRAIN_WALLTIME_SECONDS"])


def record_handoff(args):
    from slurm.ambi_transfer_training import write_new
    lineage = Path(os.environ["AMBI_LINEAGE_DIR"])
    generation = (lineage / "LATEST").read_text().strip()
    metadata = json.loads((lineage / "generations" / generation / "manifest.json").read_text())["metadata"]
    # The first allocation must exercise the trained state, not only genesis.
    assert metadata["global_step"] > 2500 and metadata["num_updates"] > 0, metadata
    assert metadata["replay_rows"] > 0
    write_new(args.output / "bootstrap-handoff.json", {"generation": generation, **metadata})


def publish(args):
    from slurm.ambi_transfer_training import write_new
    lineage = Path(os.environ["AMBI_LINEAGE_DIR"])
    metadata = json.loads((lineage / "LINEAGE.json").read_text())["metadata"]
    path = args.output / "publication.json"
    if not path.exists():
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        name, _, run = validate_recipe()
        write_new(path, {
            "commit": commit, "output": str(args.output),
            "workspace_name": "Dog Trot — seed56 resumable restart",
            "protocol": (
                "**Dog Trot: fresh seed56 restart with full-state resume.**\n\n"
                "Same 1,000,000-decision H3/J8/K1, N128/C16/A4/B256, fixed post-tanh alpha0.0021 "
                "recipe as the interrupted seed56 run. Sampled execution, reward-only critics, "
                "strict compilation, whole-inner CUDA graphs, vmap and TF32-high are preserved. "
                "Critic geometry remains enabled every1,000 outer updates. Native resume uses CPU replay; "
                "this fresh random trajectory need not match the old GPU-replay run.\n\n"
                "Full trainer, replay, optimizer, environment and RNG state is committed every15minutes "
                "at an episode boundary; latest and predecessor are retained in Oscar home storage. "
                "All40 portable model checkpoints are retained every25k decisions in scratch. "
                "The first real training segment is checkpointed and resumed in the same allocation. "
                "Time-limit handoffs are verified before automatic requeue, preserving one W&B run. "
                "Curves update only after their state is durably checkpointed.\n\n"
                f"Source `{commit}`. Lineage `{lineage}`."
            ),
            "runs": [{"id": metadata["initial_wandb_run_id"], "seed": 56,
                      "name": run["alg_params"]["wandb_run_name"], "label": "seed56 resumable restart",
                      "color": "#D55E00", "job": int(os.environ["SLURM_JOB_ID"])}],
            "controls": [{"id": "409d693629b04b6393ee277443e02a86", "label": "seed55 reference", "color": "#777777"}, {"id": "ab6f56f4db5648c0b14646cca39c9856", "label": "seed57 complete", "color": "#0072B2"}, {"id": "005d672e237748b68ca77ba721e36876", "label": "seed56 timed out (original)", "color": "#CC79A7"}],
        })
    subprocess.run([os.environ["AMBI_WORKSPACE_PYTHON"], "slurm/inner_entropy_workspace.py", str(path)],
                   cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "benchmark-worker", "allocation", "node-check", "record-handoff", "publish", "train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-cpus")
    parser.add_argument("--resume-mode", choices=("new", "required"))
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--drain-seconds", type=int)
    parser.add_argument("--checkpoint-minutes", type=int, default=15)
    args = parser.parse_args()
    validate_recipe()
    if args.mode == "benchmark-worker":
        assert os.environ.get("WANDB_MODE") == "disabled"
        args.index, args.native_reference = 0, False
        benchmark_worker(args)
    elif args.mode == "train":
        if args.resume_mode is None or args.snapshot_root is None or not args.drain_seconds or args.drain_seconds <= 0:
            parser.error("train requires resume mode, snapshot root and positive drain seconds")
        if args.checkpoint_minutes <= 0:
            parser.error("checkpoint minutes must be positive")
        train(args)
    else:
        {"benchmark": benchmark, "allocation": verify_allocation,
         "node-check": node_check, "record-handoff": record_handoff, "publish": publish}[args.mode](args)


if __name__ == "__main__":
    main()
