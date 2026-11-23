"""W&B-disabled Assembly full-ensemble training, evaluation, and throughput gate.

Run on one scheduled GPU/CPU from a clean, pinned checkout. The production
seed/pretraining budget and scientific settings are retained. Only the smoke
budget, replay capacity, evaluation count/cadence, and artifact output differ.
The timed window includes complete training decisions and outer updates;
evaluation, compilation, seed collection, and checkpoint I/O are excluded.
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
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.benchmark_ambi_training_cpu import TrainingWindow, physical_core_cpus
from slurm.metaworld_campaign import PROTOCOL, read_json, validate_algorithm

ALGORITHM = ROOT / "configs/metaworld/algs/ambi_state_mrsq_q5.json"
MANIFEST = ROOT / "configs/metaworld/experiments/assembly_mrsq_q5.json"
ENSEMBLE_SIZES = {"mrsq_q5": 5, "mrsq_q10": 10}
WARMUP = MEASURED = 100
REDUCTIONS = {
    "outer_q_actor_reduction": "mean_all",
    "inner_q_actor_reduction": "mean_all",
    "outer_q_target_reduction": "min_all",
    "inner_q_target_reduction": "min_all",
    "mppi_terminal_q_reduction": "min_all",
}


def recipe_paths(variant):
    num_q = ENSEMBLE_SIZES[variant]
    if num_q == 5:
        return ALGORITHM, MANIFEST
    return (ALGORITHM.with_name(f"ambi_state_mrsq_q{num_q}.json"),
            MANIFEST.with_name(f"assembly_mrsq_q{num_q}.json"))


def source_binding(variant="mrsq_q5"):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    assert not git("status", "--porcelain", "--untracked-files=all"), "Dirty source checkout"
    commit = git("rev-parse", "HEAD")
    expected = os.environ.get("AMBI_EXPECTED_COMMIT")
    if variant == "mrsq_q10":
        assert expected, "Pin AMBI_EXPECTED_COMMIT for the Q10 gate"
    assert expected is None or commit == expected, "Unexpected source commit"
    paths = (*recipe_paths(variant), Path(__file__).resolve())
    return {"commit": commit, "tree": git("rev-parse", "HEAD^{tree}"), "clean": True,
            "files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in paths}}


def recipe(output, variant="mrsq_q5"):
    """Derive an explicitly bounded smoke from the versioned production recipe."""
    algorithm_path, manifest_path = recipe_paths(variant)
    run, manifest = deepcopy(read_json(algorithm_path)), deepcopy(read_json(manifest_path))
    validate_algorithm(run)
    assert manifest["configs"] == [algorithm_path.stem]
    assert not manifest.get("overrides_alg"), "Unexpected shallow manifest override"
    assert manifest["env_params"] == {
        "task": "assembly", "obs": "state", "render_mode": None, "protocol": PROTOCOL,
    }
    params = run["alg_params"]
    assert run["seed"] == 55 and run["total_steps"] == 1_000_000
    assert params["seed_steps"] == params["pretrain_steps"] == 1000
    assert params["num_q"] == params["q_pair_size"] == ENSEMBLE_SIZES[variant]
    assert all(params[key] == value for key, value in REDUCTIONS.items())
    assert params["critic_geometry_diagnostics"] is True
    assert params["wandb"] is False and params["wandb_mode"] == "disabled"
    overrides = {"buffer_size": 4096, "eval_episodes": 1, "eval_freq": 1000,
                 "eval_csv_path": str(output / "evaluation.csv")}
    params.update(overrides)
    # The first decision after the seed budget is still random and performs
    # the full pretraining burst. Active decisions begin at step 1002.
    run["total_steps"] = params["seed_steps"] + 1 + WARMUP + MEASURED
    run.update(checkpoint_every=None, save_strat="none")
    manifest.update(checkpoint_every=None, save_strat="none", save_trials="none", logs="none")
    return run, manifest, {"alg_params": overrides, "total_steps": run["total_steps"],
                           "checkpoint_every": None, "save_strat": "none", "logs": "none"}


def finite_metrics(metrics, torch):
    for key, value in metrics.items():
        if value is not None:
            assert bool(torch.isfinite(torch.as_tensor(value)).all()), f"Non-finite metric: {key}"


def validate_graph_usage(rows, diagnostic_every, first_measured, last_measured):
    """Match the engine's intentional native evaluation/diagnostic paths."""
    assert diagnostic_every > 0
    counts = {"training_graph": 0, "training_native_diagnostic": 0,
              "evaluation_native": 0, "measured_training_graph": 0}
    for row in rows:
        diagnostic = row["step"] % diagnostic_every == 0
        scope = ("evaluation_native" if row["evaluation"] else
                 "training_native_diagnostic" if diagnostic else "training_graph")
        used = float(row["metrics"]["inner_cuda_graph_used"])
        expected = float(scope == "training_graph")
        assert used == expected, f"Unexpected graph usage for {scope} at step {row['step']}: {used}"
        counts[scope] += 1
        if not row["evaluation"] and first_measured <= row["step"] <= last_measured:
            assert used == 1, "Timed training must use the whole-inner CUDA graph"
            counts["measured_training_graph"] += 1
    return counts


def validate_trained_controller(learner, num_q):
    extra_checks = {}
    if num_q == 10:
        from slurm.inner_entropy_oscar import check_geometry
        from tests.test_inner_entropy_graph import assert_graph_resume_continuation

        # Native evaluation restores the training state and drops the captured
        # runner. Test continuation from the actual sampled training graph first.
        assert_graph_resume_continuation(learner.agent)
        extra_checks["graph_native_graph_continuation"] = True
        extra_checks["critic_geometry"] = check_geometry(learner.agent, learner.buffer)
    # Exercise the trained model's real online success/return path outside timing.
    learner._evaluate_policy(learner._global_step)
    return extra_checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New directory outside source")
    parser.add_argument("--variant", choices=ENSEMBLE_SIZES, default="mrsq_q5")
    args = parser.parse_args(argv)
    num_q = ENSEMBLE_SIZES[args.variant]
    assert os.environ.get("SLURM_JOB_ID"), "Use a scheduled GPU allocation"
    assert int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) == 1, "Gate requires one CPU"
    assert os.environ.get("WANDB_MODE") == "disabled", "Smoke must never create a W&B run"
    output = args.output.resolve()
    assert output != ROOT and ROOT not in output.parents, "Keep smoke output outside source"
    binding = source_binding(args.variant)
    cpus = physical_core_cpus(os.sched_getaffinity(0))
    assert cpus, "No allocated physical CPU"
    cpu = cpus[0]
    os.sched_setaffinity(0, {cpu})
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE", TORCHINDUCTOR_COMPILE_THREADS="1")
    output.mkdir(parents=True, exist_ok=False)
    run, manifest, overrides = recipe(output, args.variant)

    import torch
    import gymnasium as gym
    import domains  # noqa: F401
    from RL.AMBITDMPC2 import AMBITDMPC2

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    assert torch.cuda.is_available(), "CUDA GPU required"
    gpu = torch.cuda.get_device_name()
    if num_q == 10:
        assert gpu == "NVIDIA L40S", f"Q10 Assembly production sizing requires L40S, received {gpu}"
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    started = time.perf_counter()
    try:
        assert env.observation_space.shape == (39,) and env.action_space.shape == (4,)
        assert env.unwrapped.action_repeat == 2 and env.spec.max_episode_steps == 100
        learner = AMBITDMPC2(f"assembly-mrsq-q{num_q}-smoke", env, run["alg_params"], run, manifest)
        # Match full-state-resume production replay placement, while the
        # tracker has no save strategy and therefore writes no checkpoints.
        learner.set_checkpointing(25_000, str(output / "checkpointing-disabled"),
                                  f"assembly-q{num_q}-smoke", save_strat="none", checkpoint_replay=True)
        cfg, agent = learner.cfg, learner.agent
        assert cfg.compile and cfg.compile_strict and cfg.inner_cuda_graph
        assert cfg.critic_ensemble_backend == "vmap" and cfg.utd == 1
        assert cfg.num_q == cfg.q_pair_size == num_q
        assert all(getattr(cfg, key) == value for key, value in REDUCTIONS.items())
        assert cfg.seed_steps == cfg.pretrain_steps == 1000
        assert (cfg.inner_rounds, cfg.inner_rollout_horizon, cfg.inner_solve_interval) == (8, 3, 1)
        assert (cfg.inner_rollouts_per_round, cfg.inner_batch_size, cfg.inner_replay_capacity) == (128, 256, 3072)
        assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action) == (128, 32)
        assert cfg.inner_total_optimizer_steps_per_action == 160
        assert cfg.critic_geometry_diagnostics and cfg.critic_geometry_every == 1000
        assert not cfg.wandb and cfg.wandb_mode == "disabled" and learner._wandb_run is None
        assert learner._eval_freq == 1000 and learner._eval_episodes == 1
        timer = TrainingWindow(cfg.seed_steps + WARMUP + 2, MEASURED, torch.cuda.synchronize)
        act, update, log_step = learner._act_agent, agent.update, learner._log_wandb_step
        evaluate, record_evaluation = learner._evaluate_policy, learner._record_evaluation
        inner_rows, outer_rows, evaluations = [], [], []
        compile_counts, startup = {}, {}
        evaluation = None
        torch.cuda.reset_peak_memory_stats()

        def observed_act(*positional, **kwargs):
            is_eval = bool(kwargs.get("eval_mode", False))
            step = learner._global_step + 1
            if not is_eval:
                if step == timer.first_step:
                    torch.cuda.synchronize()
                    compile_counts["before"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
                    startup["seconds_through_compilation_and_warmup"] = time.perf_counter() - started
                timer.before_action(step, learner._num_updates)
            action_started = time.perf_counter()
            action = act(*positional, **kwargs)
            if not is_eval:
                timer.after_action(step, time.perf_counter() - action_started)
            assert action.device.type == "cpu" and bool(torch.isfinite(action).all())
            inner_rows.append({"step": step, "evaluation": is_eval, "metrics": dict(agent.last_inner_metrics)})
            if is_eval:
                assert evaluation is not None
                evaluation["decisions"] += 1
            return action

        def observed_update(*positional, **kwargs):
            metrics = update(*positional, **kwargs)
            # Defer detached metric checks until timing ends, avoiding an
            # extra GPU synchronization in every outer update.
            outer_rows.append({"step": learner._global_step, "metrics": dict(metrics)})
            return metrics

        def observed_log_step(reward, *positional, **kwargs):
            assert math.isfinite(float(reward))
            result = log_step(reward, *positional, **kwargs)
            timer.after_decision(learner._global_step, learner._num_updates)
            return result

        def observed_record(step, mean_return, **kwargs):
            assert evaluation is not None and int(step) == evaluation["step"]
            evaluation.update(mean_return=float(mean_return), success_rate=float(kwargs["success_rate"]))
            assert math.isfinite(evaluation["mean_return"])
            assert evaluation["success_rate"] in (0.0, 1.0)
            return record_evaluation(step, mean_return, **kwargs)

        def observed_evaluate(step, **kwargs):
            nonlocal evaluation
            assert evaluation is None
            evaluation = {"step": int(step), "episodes": 1, "decisions": 0}
            torch.cuda.synchronize()
            began = time.perf_counter()
            try:
                result = evaluate(step, **kwargs)
                torch.cuda.synchronize()
                evaluation["seconds"] = time.perf_counter() - began
                assert evaluation["decisions"] == 100
                assert "success_rate" in evaluation
                evaluations.append(evaluation)
                return result
            finally:
                evaluation = None

        learner._act_agent, agent.update, learner._log_wandb_step = observed_act, observed_update, observed_log_step
        learner._evaluate_policy, learner._record_evaluation = observed_evaluate, observed_record
        learner.learn(run["total_steps"])
        compile_counts["after"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
        # Preserve completed timing even if a later gate fails. This file is
        # explicitly unvalidated and cannot authorize production; only the
        # final results.json below carries passed=true.
        with (output / "measurements.json").open("x") as stream:
            json.dump({"passed": False, "validation": "pending", "source": binding,
                       "measured": timer.result, "startup": startup,
                       "compile_graph_counts": compile_counts,
                       "evaluations_before_final_gate": evaluations,
                       "num_q": num_q,
                       "created_utc": datetime.now(timezone.utc).isoformat()},
                      stream, indent=2, allow_nan=False)
            stream.write("\n")
        assert compile_counts["before"] == compile_counts["after"], "Compilation occurred in timed window"
        assert timer.result is not None
        assert len([r for r in inner_rows if not r["evaluation"]]) == WARMUP + MEASURED
        assert len([r for r in outer_rows if timer.first_step <= r["step"] <= timer.last_step]) == MEASURED
        assert learner._global_step == 1201 and learner._num_updates == 1200
        assert learner.buffer.resumable_storage and learner.buffer._storage_device.type == "cpu"
        extra_checks = validate_trained_controller(learner, num_q)
        assert [r["step"] for r in evaluations] == [0, 1000, 1201]
        graph_usage = validate_graph_usage(inner_rows, cfg.inner_diagnostics_every,
                                          timer.first_step, timer.last_step)
        assert graph_usage == {"training_graph": 200, "training_native_diagnostic": 0,
                               "evaluation_native": 300, "measured_training_graph": 100}
        for row in inner_rows:
            metrics = row["metrics"]
            finite_metrics(metrics, torch)
            assert float(metrics["inner_critic_optimizer_steps"]) == 128
            assert float(metrics["inner_actor_optimizer_steps"]) == 32
            assert float(metrics["inner_temperature_optimizer_steps"]) == 0
            assert float(metrics["inner_compile_fallback"]) == 0
        for row in outer_rows:
            finite_metrics(row["metrics"], torch)
            assert float(row["metrics"]["compile_fallback"]) == 0
        assert all(bool(torch.isfinite(p).all()) for p in agent.model.parameters())
        finite_metrics(learner._last_train_metrics, torch)
        assert learner._wandb_run is None and not list(output.rglob("*.pt"))
        assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
        assert os.sched_getaffinity(0) == {cpu}
        measured = dict(timer.result)
        measured["projected_1m_training_hours"] = measured["seconds_per_decision"] * 1e6 / 3600
        report = {
            "passed": True, "source": binding, "runtime": {"python": sys.version.split()[0],
                "executable": sys.executable, "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": gpu, "float32_matmul_precision": torch.get_float32_matmul_precision()},
            "slurm_job_id": os.environ["SLURM_JOB_ID"], "host": os.uname().nodename,
            "physical_cores": 1, "logical_cpu_ids": [cpu], "warmup_active_decisions": WARMUP,
            "measured": measured, "startup": startup, "evaluations": evaluations,
            "wall_seconds": time.perf_counter() - started, "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(), "compile_graph_counts": compile_counts,
            "compile_fallback": False, "inner_cuda_graph_used": True, "finite_actions_model_metrics": True,
            "graph_usage_counts": graph_usage,
            "critic_updates_per_action": 128, "actor_updates_per_action": 32,
            f"q{num_q}_reductions": REDUCTIONS, "num_q": num_q, "q_pair_size": num_q,
            "critic_geometry_diagnostics": True, "production_overrides": overrides,
            "total_steps": run["total_steps"], "seed_steps": 1000, "pretrain_steps": 1000,
            "replay_storage_device": "cpu", "replay_transitions": learner.buffer.num_transitions,
            "checkpointing": False, "wandb": False, "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Early-replay steady training only; add production evaluation, diagnostics, checkpoint I/O and startup overhead when sizing a full run.",
            **extra_checks,
        }
        with (output / "results.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"ASSEMBLY_Q{num_q}_SMOKE " + json.dumps({"passed": True, "output": str(output),
              "seconds_per_decision": measured["seconds_per_decision"], "evaluations": evaluations}), flush=True)
    finally:
        if learner is not None:
            learner.close()
        env.close()


if __name__ == "__main__":
    main()
