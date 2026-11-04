"""Gate and execute one fresh post-tanh AMBI distillation run on Oscar.

The gate uses the production algorithm and real environment, shortens only its
startup/workload, and disables W&B. Training requires an exact-source/config
receipt whose measured projection fits the original completion deadline.
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
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_transfer_training import runtime, write_new
from slurm.benchmark_ambi_training_cpu import TrainingWindow, physical_core_cpus


GPU_PARITY_TEST = "tests/test_ambi_distillation_collection_graph.py"
GPU_PARITY_CASES = {
    "test_collection_graph_matches_uncaptured_protocol_teachers_and_optimizer_state[small]",
    "test_collection_graph_matches_uncaptured_protocol_teachers_and_optimizer_state[production_budget]",
    "test_collection_graph_matches_uncaptured_protocol_teachers_and_optimizer_state[production_budget_tf32]",
    "test_teacher_capture_does_not_change_collection_action_or_cuda_rng",
    "test_collection_graph_checkpoint_replays_next_teachers_actions_and_native_diagnostics_exactly[small]",
    "test_collection_graph_checkpoint_replays_next_teachers_actions_and_native_diagnostics_exactly[production_budget]",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def deadline(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "The deadline must include its UTC offset.")
    return parsed.astimezone(timezone.utc)


def read_recipe(path):
    path = path.resolve(strict=True)
    path.relative_to(ROOT)
    run = json.loads(path.read_text())
    require(run["env"] == "DMControl-v0" and run["alg"] == "AMBITDMPC2/AMBITDMPC2",
            "This launcher is for the matched AMBI DMControl run.")
    require(type(run["total_steps"]) is int and run["total_steps"] > 500, "Invalid decision budget.")
    env_params = run.get("env_params", {"task": "humanoid-walk", "obs": "state", "render_mode": None})
    require(env_params == {"task": "humanoid-walk", "obs": "state", "render_mode": None},
            "Keep the original Humanoid Walk state-observation environment.")
    manifest = {"env_params": env_params, "trials": 1, "logs": "none", "save_trials": "none",
                "checkpoint_every": run["checkpoint_every"], "save_strat": run["save_strat"],
                "log_info": False, "log_type": "summary"}
    return path.stem, manifest, run


def source_binding(config):
    config = config.resolve(strict=True)
    _, manifest, _ = read_recipe(config)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "config": str(config.relative_to(ROOT)), "config_sha256": digest(config),
        "lock_sha256": digest(ROOT / "environments/dmcontrol/uv.lock"),
        "environment": manifest["env_params"],
    }


def inspect_parity_xml(path):
    cases = list(ET.parse(path).iter("testcase"))
    require(cases and all(not any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
                          for case in cases), "Collection graph parity must pass without skips or errors.")
    names = {case.attrib["name"] for case in cases}
    require(GPU_PARITY_CASES <= names, "Collection graph parity omitted a required CUDA/production-budget case.")
    return len(cases)


def gpu_parity(output, bound, current):
    """Run correctness in a separate process before any timing/compile state."""
    log, xml = output / "gpu-parity.log", output / "gpu-parity.xml"
    started = time.perf_counter()
    with log.open("x") as stream:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", GPU_PARITY_TEST, "--junitxml", str(xml), "-rA"],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=600, check=False,
        )
    require(result.returncode == 0, "Collection graph parity failed; inspect gpu-parity.log.")
    count = inspect_parity_xml(xml)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    proof = {"passed": True, "binding": bound, "runtime": current,
             "test": GPU_PARITY_TEST, "test_sha256": digest(ROOT / GPU_PARITY_TEST),
             "junit_file": xml.name, "junit_sha256": digest(xml),
             "log_file": log.name, "log_sha256": digest(log),
             "passed_count": count, "required_gpu_cases": sorted(GPU_PARITY_CASES),
             "seconds": time.perf_counter() - started}
    write_new(output / "gpu-parity.json", proof)
    return proof


def validate_parity(proof, directory, bound, current):
    require(proof.get("passed") is True and proof["binding"] == bound and proof["runtime"] == current,
            "GPU parity source/config/runtime does not match this launch.")
    require(proof["test"] == GPU_PARITY_TEST and proof["required_gpu_cases"] == sorted(GPU_PARITY_CASES),
            "GPU parity protocol changed.")
    require(hashlib.sha256((ROOT / GPU_PARITY_TEST).read_bytes()).hexdigest() == proof["test_sha256"],
            "GPU parity test source changed.")
    for field, expected_name in (("junit", "gpu-parity.xml"), ("log", "gpu-parity.log")):
        require(proof[field + "_file"] == expected_name, "Unexpected GPU parity artifact path.")
        artifact = directory / expected_name
        require(artifact.is_file() and hashlib.sha256(artifact.read_bytes()).hexdigest() == proof[field + "_sha256"],
                "GPU parity artifact is missing or changed: " + expected_name)
    require(inspect_parity_xml(directory / "gpu-parity.xml") == proof["passed_count"],
            "GPU parity test accounting changed.")


def gate_recipe(run, manifest, *, decisions=None, warmup=None):
    interval = int(run["alg_params"]["reanalyze_interval"])
    require(interval > 0, "The gate requires active lazy reanalysis.")
    measured = max(128, 2 * interval) if decisions is None else decisions
    # Seed collection ends at 501. Keep native diagnostics at their original
    # step-1000 cadence, but compile that path before the timing window starts.
    minimum_warmup = max(500, interval)
    warmup = minimum_warmup if warmup is None else warmup
    require(measured >= 2 * interval and warmup >= minimum_warmup,
            "Warm up the first refresh/native diagnostic and measure at least two refresh intervals.")
    shortened, experiment = deepcopy(run), deepcopy(manifest)
    overrides = {"seed_steps": 500, "pretrain_steps": 2, "wandb": False, "wandb_mode": "disabled"}
    shortened["alg_params"].update(overrides)
    shortened.update(total_steps=501 + warmup + measured, checkpoint_every=None, save_strat="none")
    experiment.update(checkpoint_every=None, save_strat="none")
    return shortened, experiment, warmup, measured, {
        "alg_params": overrides, "total_steps": shortened["total_steps"],
        "checkpoint_every": None, "save_strat": "none",
        "extra_observers": "One synchronous portable checkpoint and critic geometry check inside timing.",
        "replay_capacity": "Allocate the original production capacity; only resident data are from the short gate.",
    }


def execution():
    require(os.environ.get("SLURM_JOB_ID"), "Run inside a scheduled Oscar GPU allocation.")
    require(not os.environ.get("SLURM_ARRAY_TASK_ID"), "This is one run, not an array.")
    require(int(os.environ.get("SLURM_RESTART_COUNT", "0")) == 0, "Fresh execution cannot requeue/resume.")
    count = int(os.environ["SLURM_CPUS_PER_TASK"])
    cores = physical_core_cpus(os.sched_getaffinity(0))
    require(count > 0 and len(cores) >= count, "Insufficient allocated physical CPUs.")
    os.sched_setaffinity(0, cores[:count])
    import torch
    require(torch.cuda.is_available(), "A CUDA GPU is required.")
    torch.set_num_threads(count)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    require(not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True).strip(),
        "Execute a clean, pinned checkout.")
    current = runtime()
    require("L40S" in current["gpu"] or "H100" in current["gpu"], "Gate on an approved L40S or H100 GPU.")
    return {**current, "float32_matmul_precision": "high", "physical_cpus": count}


def validate_recipe(cfg):
    require(cfg.outer_actor_objective == "distillation", "Select the distillation objective.")
    require(cfg.inner_cuda_graph and cfg.compile and cfg.compile_strict, "Require strict compiled collection graphs.")
    require(cfg.reanalyze_backend == "batched_cuda_graph", "Require the validated batched reanalysis graph.")
    require((cfg.inner_rounds, cfg.inner_rollout_horizon, cfg.inner_solve_interval) == (8, 3, 1),
            "Keep the original J8/H3/K1 collection budget.")
    require((cfg.inner_rollouts_per_round, cfg.inner_batch_size,
             cfg.inner_critic_updates_per_round, cfg.inner_actor_updates_per_round) == (128, 256, 16, 4),
            "Keep the original N128/B256/C16/A4 SAC schedule.")
    require(cfg.utd == 1 and cfg.action_dim == 21 and cfg.inner_finite_horizon, "Unexpected matched recipe.")
    require(cfg.inner_diagnostics_every == 1000, "Keep the original native diagnostic cadence.")
    require(cfg.outer_actor_entropy_mode == cfg.inner_actor_entropy_mode == "squashed", "Require post-tanh entropy.")
    require(cfg.inner_actor_loss_scale_update == "per_update" and cfg.inner_actor_loss_scale_scope == "action",
            "Require fresh outer-initialized, adapting inner Q normalization.")
    require(cfg.sac_actor_loss_scale_mode == "tdmpc2_percentile_range", "Keep Q normalization enabled.")
    require(cfg.outer_critic_target == cfg.inner_sac_critic_target == "reward_only", "Keep reward-only Q targets.")
    require(cfg.critic_geometry_diagnostics and cfg.critic_geometry_every == 1000
            and cfg.critic_geometry_batch_size == 64, "Retain production critic geometry diagnostics.")
    require(cfg.critic_geometry_obs_noise_std == .01 and cfg.critic_geometry_action_noise_std == .02,
            "Keep the configured geometry perturbations.")


def check_geometry(agent, buffer):
    import torch
    from RL.tdmpc2_core.critic_geometry import critic_geometry_diagnostics
    from RL.tdmpc2_core.reanalyze import _preserve_global_rng
    before = {key: value.detach().clone() for key, value in agent.model.state_dict().items()}
    parameters = tuple(agent.model.parameters())
    gradients = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    modes = [module.training for module in agent.model.modules()]
    with _preserve_global_rng():
        obs, action = buffer.sample()[:2]
        torch.cuda.synchronize()
        started = time.perf_counter()
        values = critic_geometry_diagnostics(agent.model, obs, action, agent.cfg, update_number=1000)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    require(float(values["critic_geometry_sample_count"]) == 64, "Geometry observer omitted replay rows.")
    for key, value in agent.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            require(parameter.grad is None, "Geometry observer created parameter gradients.")
        else:
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    require([module.training for module in agent.model.modules()] == modes, "Geometry changed model modes.")
    report = {}
    for key, value in values.items():
        number = float(value)
        require(not math.isinf(number), "Geometry returned infinity.")
        # Undefined direction/relative-gradient statistics carry explicit validity.
        if math.isnan(number):
            require(key.endswith(("_cosine_similarity", "_relative_gradient_change")),
                    "Unexpected NaN geometry statistic: " + key)
            if key.endswith("_cosine_similarity"):
                validity = key.removesuffix("_cosine_similarity") + "_cosine_valid_fraction"
            else:
                critic = key.split("_mpr_")[0].split("_action_smoothness_")[0]
                validity = critic + "_gradient_scale_valid"
            require(float(values[validity]) == 0, "Undefined geometry statistic has nonzero validity: " + key)
            report[key] = None
        else:
            report[key] = number
    return {"passed": True, "seconds": seconds, "metrics": report, "state_unchanged": True}


def project_duration(run, measured, *, update_seconds, checkpoint_seconds, geometry_seconds,
                     extra_observer_seconds, startup_seconds, seed_seconds, margin=1.1):
    params = run["alg_params"]
    total, seeds, pretrain = int(run["total_steps"]), int(params["seed_steps"]), int(params["pretrain_steps"])
    active = max(0, total - seeds - 1)
    base = max(0., measured["wall_seconds"] - extra_observer_seconds) / measured["decisions"]
    checkpoints = math.ceil(total / int(run["checkpoint_every"]))
    geometry_events = (active + pretrain) // int(params.get("critic_geometry_every", 1000))
    raw = (active * base + pretrain * update_seconds + seeds * seed_seconds / 500
           + checkpoints * checkpoint_seconds + geometry_events * geometry_seconds + startup_seconds)
    return {"total_decisions": total, "active_decisions": active, "pretraining_updates": pretrain,
            "checkpoint_count": checkpoints, "geometry_events": geometry_events,
            "seconds_per_decision_without_extra_gate_observers": base,
            "outer_update_seconds_including_reanalysis": update_seconds,
            "startup_upper_bound_seconds": startup_seconds,
            "projected_seconds_before_margin": raw, "margin_multiplier": margin,
            "projected_seconds": raw * margin, "projected_hours": raw * margin / 3600}


def gate(args, bound, current):
    require(os.environ.get("WANDB_MODE") == "disabled", "Gate W&B must be disabled.")
    import torch
    import gymnasium as gym
    import domains  # noqa: F401
    from RL.AMBITDMPC2 import AMBITDMPC2
    from RL.tdmpc2_core.common.buffer import Buffer

    name, manifest, production = read_recipe(args.config)
    run, experiment, warmup, decisions, overrides = gate_recipe(
        production, manifest, decisions=args.decisions, warmup=args.warmup,
    )
    write_new(args.output / "protocol.json", {"binding": bound, "runtime": current, "overrides": overrides,
              "warmup_active_decisions": warmup, "measured_decisions": decisions,
              "deadline_utc": args.deadline_utc.isoformat(), "wandb": False})
    parity = gpu_parity(args.output, bound, current)
    started = time.perf_counter()
    env = gym.make(run["env"], **experiment["env_params"])
    learner = None
    try:
        learner = AMBITDMPC2(name + "-gate", env, run["alg_params"], run, experiment)
        agent, cfg = learner.agent, learner.cfg
        validate_recipe(cfg)
        replay_cfg = deepcopy(cfg)
        replay_cfg.steps = production["total_steps"]
        learner.buffer = Buffer(replay_cfg)
        timer = TrainingWindow(502 + warmup, decisions, torch.cuda.synchronize)
        act, update, maybe_checkpoint = learner._act_agent, agent.update, learner._maybe_checkpoint
        stage_teacher = learner._stage_teacher_policy
        stats = {"graph_actions": 0, "native_diagnostic_actions": 0, "valid_collected_teachers": 0,
                 "timed_refreshes": 0, "timed_skips": 0, "coverage": [], "update_seconds": []}
        observed, checkpoints = {}, {}
        midpoint = timer.first_step + decisions // 2

        def budget_check():
            require(time.perf_counter() - started <= args.max_gate_seconds,
                    "Gate exceeded its bounded runtime; do not launch production from this receipt.")

        def observed_act(*positional, **keywords):
            budget_check()
            step = learner._global_step + 1
            if step == timer.first_step:
                observed["compile_before"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
                observed["actor_before"] = agent.distillation.actor_updates
                observed["refresh_before"] = agent.distillation.reanalyzer.refreshes
                observed["graph_before"] = agent.inner_engine._whole_graph_runner.calls
                observed["startup_seconds"] = time.perf_counter() - started
            timer.before_action(step, learner._num_updates)
            action_started = time.perf_counter()
            result = act(*positional, **keywords)
            timer.after_action(step, time.perf_counter() - action_started)
            require(torch.isfinite(result).all().item(), "Collection returned a nonfinite action.")
            inner = agent.last_inner_metrics
            require(inner["inner_actor_optimizer_steps"] == 32 and inner["inner_critic_optimizer_steps"] == 128,
                    "Collection adaptation budget changed.")
            require(inner["inner_compile_fallback"] == 0, "Collection compilation fell back.")
            diagnostic = step % int(cfg.inner_diagnostics_every) == 0
            require(inner.get("inner_cuda_graph_used") == float(not diagnostic),
                    "Collection did not use the required graph/native diagnostic path.")
            require(agent.last_teacher_policy is not None, "Collection failed to capture the final teacher.")
            if timer.first_step <= step <= timer.last_step:
                stats["native_diagnostic_actions" if diagnostic else "graph_actions"] += 1
            return result

        def observed_teacher(row, teacher):
            result = stage_teacher(row, teacher)
            if teacher is not None:
                require(bool(learner._episode_staging["teacher_valid"][row]), "Teacher was not staged on its successor row.")
                stats["valid_collected_teachers"] += 1
            return result

        def observed_update(*positional, **keywords):
            budget_check()
            before = agent.distillation.actor_updates
            update_started = time.perf_counter()
            metrics = update(*positional, **keywords)
            elapsed = time.perf_counter() - update_started
            require(float(metrics["compile_fallback"]) == 0, "Outer compilation fell back.")
            skipped = int(float(metrics["distillation_actor_skipped"]))
            require(agent.distillation.actor_updates - before == 1 - skipped, "Actor optimizer count disagrees with masking.")
            if timer.first_step <= learner._global_step <= timer.last_step:
                stats["update_seconds"].append(elapsed)
                stats["coverage"].append(float(metrics["distillation_valid_fraction"]))
                stats["timed_refreshes"] += int(float(metrics["distillation_reanalyze_ran"]))
                stats["timed_skips"] += skipped
                if float(metrics["distillation_reanalyze_ran"]):
                    require(float(metrics["distillation_reanalyze_cuda_graph"]) == 1
                            and float(metrics["distillation_reanalyze_batched"]) == 1,
                            "Refresh did not execute the required batched CUDA graph.")
            return metrics

        def observed_checkpoint():
            result = maybe_checkpoint()
            step = learner._global_step
            if step == 500:
                observed["seed_seconds"] = time.perf_counter() - started
            if step == midpoint:
                extra_started = time.perf_counter()
                checkpoint_started = time.perf_counter()
                path = Path(learner.save(str(args.output / "checkpoints"), "gate"))
                learner.flush_checkpoints()
                checkpoints.update(path=str(path), seconds=time.perf_counter() - checkpoint_started,
                                   bytes=path.stat().st_size)
                require(Path(str(path) + ".metadata.json").is_file(), "Checkpoint sidecar is missing.")
                observed["geometry"] = check_geometry(agent, learner.buffer)
                observed["extra_observer_seconds"] = time.perf_counter() - extra_started
            # Logging runs before periodic checkpointing; finish timing here so
            # real checkpoint creation and flush are part of the measured window.
            timer.after_decision(step, learner._num_updates)
            return result

        learner._act_agent, agent.update = observed_act, observed_update
        learner._maybe_checkpoint, learner._stage_teacher_policy = observed_checkpoint, observed_teacher
        learner.learn(run["total_steps"])
        learner.flush_checkpoints()
        budget_check()
        require(timer.result is not None and learner._global_step == run["total_steps"], "Incomplete gate window.")
        observed["compile_after"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
        require(observed["compile_before"] == observed["compile_after"], "Compilation occurred during timing.")
        require(stats["timed_refreshes"] >= 2 and stats["timed_skips"] == 0, "Refresh/teacher coverage gate failed.")
        require(min(stats["coverage"]) > 0 and max(stats["coverage"]) <= 1, "Invalid teacher coverage.")
        require(agent.distillation.actor_updates - observed["actor_before"] == decisions, "Missing timed actor updates.")
        require(agent.distillation.reanalyzer.refreshes - observed["refresh_before"] == stats["timed_refreshes"],
                "Reanalysis refresh accounting changed.")
        require(agent.inner_engine._whole_graph_runner.calls - observed["graph_before"] == stats["graph_actions"],
                "Collection graph replay count disagrees with observed actions.")
        require(stats["valid_collected_teachers"] == warmup + decisions, "Missing collected teacher labels.")
        for optimizer in (agent.pi_optim, agent.ent_coef_optim):
            if optimizer is not None:
                require(bool(optimizer.state) and all(int(state["step"].item()) == agent.distillation.actor_updates
                            for state in optimizer.state.values()), "Actor/temperature Adam counts disagree.")
        require(all(torch.isfinite(parameter).all().item() for parameter in agent.model.parameters()), "Nonfinite model.")
        require(learner.buffer._storage_device.type == "cuda", "Replay fell back to CPU placement.")
        saved = torch.load(checkpoints["path"], map_location="cpu", weights_only=False)
        from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
        restored = AMBITDMPC2Agent(deepcopy(cfg))
        restored.load(checkpoints["path"])
        for key, value in restored.model.state_dict().items():
            torch.testing.assert_close(value.cpu(), saved["model"][key], rtol=0, atol=0)
        require(restored.distillation.actor_updates == saved["distillation_state"]["actor_updates"],
                "Checkpoint lost the actual actor-update count.")
        torch.testing.assert_close(restored.distillation.scale.cpu(), saved["distillation_state"]["scale"], rtol=0, atol=0)
        measured = dict(timer.result)
        measured.pop("projected_500k_decision_days", None)
        projection = project_duration(production, measured,
            update_seconds=statistics.mean(stats["update_seconds"]), checkpoint_seconds=checkpoints["seconds"],
            geometry_seconds=observed["geometry"]["seconds"], extra_observer_seconds=observed["extra_observer_seconds"],
            startup_seconds=observed["startup_seconds"], seed_seconds=observed["seed_seconds"])
        available = min(args.max_wall_hours * 3600, (args.deadline_utc - datetime.now(timezone.utc)).total_seconds())
        report = {"schema": "ambi-posttanh-distillation-gate-v1", "passed": projection["projected_seconds"] <= available,
                  "binding": bound, "runtime": current, "deadline_utc": args.deadline_utc.isoformat(),
                  "gpu_parity": parity,
                  "max_wall_hours": args.max_wall_hours, "available_seconds": available,
                  "measured": measured, "projection": projection, "checkpoint": checkpoints,
                  "checkpoint_roundtrip": True, "observed": observed, "validation": stats,
                  "actor_updates": agent.distillation.actor_updates, "total_outer_updates": agent.num_updates,
                  "teacher_coverage_mean": statistics.mean(stats["coverage"]),
                  "replay_capacity": learner.buffer.capacity, "replay_resident_rows": learner.buffer.size,
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "timing_includes_native_diagnostic": stats["native_diagnostic_actions"] > 0,
                  "overrides": overrides, "wandb": False, "job_id": os.environ["SLURM_JOB_ID"],
                  "scope": "Real environment, collection, refresh, outer learning and checkpoint/geometry I/O; early resident replay and no W&B. Native diagnostics compile during warmup at their original cadence. Production projection includes original pretraining and checkpoint/geometry cadence; the 10% margin also allows for W&B, rare native diagnostic actions outside the measured window, and longer-run throughput variation."}
        write_new(args.output / "receipt.json", report)
        require(report["passed"], "Measured projection exceeds the completion deadline; reduce the versioned workload and regate.")
        print("DISTILLATION_GATE_PASSED " + json.dumps(projection), flush=True)
    finally:
        if learner is not None:
            learner.close()
        env.close()


def train(args, bound, current):
    require(args.gate is not None, "Training requires --gate.")
    receipt = json.loads(args.gate.read_text())
    require(receipt.get("schema") == "ambi-posttanh-distillation-gate-v1" and receipt.get("passed") is True,
            "A passing distillation gate is required.")
    require(receipt["binding"] == bound and receipt["runtime"] == current, "Gate source/config/runtime changed.")
    validate_parity(receipt["gpu_parity"], args.gate.resolve().parent, bound, current)
    require(deadline(receipt["deadline_utc"]) == args.deadline_utc, "Keep the original completion deadline.")
    require(receipt["max_wall_hours"] == args.max_wall_hours, "Gate wall-budget policy changed.")
    projected = float(receipt["projection"]["projected_seconds"])
    available = min(args.max_wall_hours * 3600, (args.deadline_utc - datetime.now(timezone.utc)).total_seconds())
    require(projected <= available, "The projected run no longer fits before the original deadline.")
    if os.environ.get("SLURM_JOB_END_TIME"):
        require(projected <= float(os.environ["SLURM_JOB_END_TIME"]) - time.time(), "Slurm walltime is too short.")
    require(os.environ.get("WANDB_MODE") == "online" and os.environ.get("WANDB_RUN_ID")
            and os.environ.get("WANDB_RESUME") == "never", "Use a new explicit W&B identity for training.")
    import gymnasium as gym
    import domains  # noqa: F401
    from RL.AMBITDMPC2 import AMBITDMPC2
    name, manifest, run = read_recipe(args.config)
    require(run["alg_params"].get("wandb") is True, "Production W&B configuration must be enabled.")
    write_new(args.output / "launch.json", {"binding": bound, "runtime": current,
              "gate": str(args.gate.resolve()), "gate_sha256": hashlib.sha256(args.gate.read_bytes()).hexdigest(),
              "projection": receipt["projection"], "deadline_utc": args.deadline_utc.isoformat(),
              "job_id": os.environ["SLURM_JOB_ID"], "wandb_run_id": os.environ["WANDB_RUN_ID"]})
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    started = time.perf_counter()
    try:
        learner = AMBITDMPC2(name, env, run["alg_params"], run, manifest)
        validate_recipe(learner.cfg)
        learner.set_checkpointing(run["checkpoint_every"], str(args.output / "models"), name,
                                  save_strat=run["save_strat"])
        learner.learn(run["total_steps"])
        learner.flush_checkpoints()
        require(learner._global_step == run["total_steps"], "Training ended before its configured workload.")
        write_new(args.output / "completed.json", {"binding": bound, "decisions": learner._global_step,
                  "outer_updates": learner._num_updates, "actor_updates": learner.agent.distillation.actor_updates,
                  "refreshes": learner.agent.distillation.reanalyzer.refreshes,
                  "seconds": time.perf_counter() - started, "completed_utc": datetime.now(timezone.utc).isoformat()})
    finally:
        if learner is not None:
            learner.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("gate", "train"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--deadline-utc", type=deadline, required=True)
    parser.add_argument("--max-wall-hours", type=float, default=46.)
    parser.add_argument("--max-gate-seconds", type=float, default=600.)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--decisions", type=int)
    args = parser.parse_args()
    require(0 < args.max_wall_hours <= 48 and args.max_gate_seconds > 0, "Invalid wall-clock bounds.")
    args.config = (ROOT / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    args.output = args.output.resolve()
    require(args.output != ROOT and ROOT not in args.output.parents, "Keep new output outside the source checkout.")
    require(not args.output.exists(), "Output must be a fresh path; never overwrite or resume this run.")
    current = execution()
    bound = source_binding(args.config)
    require(bound["commit"] == os.environ.get("AMBI_EXPECTED_COMMIT"), "Source commit is not the explicitly pinned revision.")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        (gate if args.mode == "gate" else train)(args, bound, current)
    except BaseException as error:
        write_new(args.output / "failure.json", {"type": type(error).__name__, "message": str(error),
                  "binding": bound, "utc": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
