"""CUDA validation, resource measurements and launch for the eight alpha-carry cells."""

import argparse
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_transfer_training import (
    binding as source_binding, digest, read, recipe as source_recipe,
    resolve_manifest, runtime, write_new,
)
from slurm.benchmark_ambi_training_cpu import TrainingWindow, physical_core_cpus

MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_alpha_carry_training.json"
COUNTS = (1, 2, 4)
SEED_STEPS = 500


def binding():
    return source_binding(MANIFEST)


def recipe(index):
    if not 0 <= index < 8:
        raise ValueError("Expected one of the eight campaign cells")
    return source_recipe(index, MANIFEST)


def assert_recipe(cfg, agent, run):
    """Check resolved settings, rather than trusting only serialized JSON."""
    assert cfg.compile and cfg.compile_strict and cfg.critic_ensemble_backend == "vmap"
    assert (cfg.inner_rollout_horizon, cfg.inner_rounds) == (3, 8)
    assert (cfg.inner_rollouts_per_round, cfg.inner_batch_size, cfg.inner_replay_capacity) == (128, 256, 3072)
    assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action) == (128, 32)
    assert cfg.inner_execution_action == cfg.inner_behavior_action == "policy_sample"
    assert cfg.inner_temperature_mode == "auto" and cfg.inner_temperature_lr == 3e-4
    assert cfg.inner_temperature_scope == cfg.inner_temperature_optimizer_scope
    assert cfg.inner_temperature_scope in {"episode", "run"}
    for component in ("actor", "critic", "replay", "actor_optimizer", "critic_optimizer"):
        assert getattr(cfg, f"inner_{component}_scope") == "action"
    target = run["alg_params"]["target_entropy"]
    assert target in (-10.5, -21.0)
    assert agent.target_entropy == agent.inner_engine._resolved_inner_target_entropy() == target
    assert cfg.aux_return_mode == "sac" and not agent.aux_return.has_actor
    assert cfg.aux_return_detach_representation == run["alg_params"]["aux_return_detach_representation"]
    assert cfg.inner_critic_source == cfg.inner_horizon_critic_source == "aux_return"
    assert cfg.inner_critic_bernoulli_p == 0 and cfg.inner_outer_policy_kl_coef == 0


def check_temperature_transition(previous, current, scope):
    """Both alpha and Adam carry, except at the selected episode boundary."""
    reset = previous is None or (current["t0"] and scope == "episode")
    expected_alpha = current["outer_alpha"] if reset else previous["alpha_final"]
    assert math.isclose(current["alpha_initial"], expected_alpha, rel_tol=2e-6, abs_tol=1e-7)
    expected_step = 0 if reset else previous["optimizer_step_final"]
    assert current["optimizer_step_initial"] == expected_step
    assert current["optimizer_step_final"] == expected_step + 32
    if not reset:
        assert current["alpha_identity"] == previous["alpha_identity"]
        assert current["optimizer_identity"] == previous["optimizer_identity"]
    assert current["episode_reset"] == float(current["t0"] and scope == "episode")


def checkpoint_bank_estimate(bytes_per_row, *, capacity=1_000_000,
                             total_steps=1_000_000, checkpoint_every=25_000,
                             episode_length=500, checkpoint_bytes=0, cells=8):
    """Count resident rows at each retained snapshot, including initial rows.

    State replay stores one extra initial-observation row per episode, and its
    FIFO capacity counts rows. This estimates the payload rather than assuming
    all forty early snapshots already contain a full million-row replay.
    """
    assert bytes_per_row > 0 and min(capacity, total_steps, checkpoint_every, episode_length, cells) > 0
    assert total_steps % checkpoint_every == checkpoint_every % episode_length == 0
    snapshots = []
    for step in range(checkpoint_every, total_steps + 1, checkpoint_every):
        rows = min(capacity, step + step // episode_length)
        snapshots.append({"step": step, "resident_rows": rows,
                          "replay_payload_bytes": math.ceil(rows * bytes_per_row)})
    replay_bytes = sum(row["replay_payload_bytes"] for row in snapshots)
    model_bytes = len(snapshots) * checkpoint_bytes
    retained = cells * (replay_bytes + model_bytes)
    # Latest + predecessor plus one in-flight native generation coexist. Their
    # full optimizer/model states exceed portable models; estimate separately.
    live_replay_generations = cells * 3 * math.ceil(capacity * bytes_per_row)
    return {"snapshots_per_cell": len(snapshots), "cells": cells,
            "per_checkpoint": snapshots, "per_cell_retained_replay_bytes": replay_bytes,
            "per_cell_retained_model_bytes": model_bytes,
            "all_cells_retained_replay_bytes": cells * replay_bytes,
            "all_cells_retained_models_and_replay_bytes": retained,
            "all_cells_three_full_native_replay_generations_bytes": live_replay_generations,
            "payload_plus_native_replay_with_20pct_margin_bytes": math.ceil(1.2 * (retained + live_replay_generations)),
            "excluded": "Native model/optimizer generations, small shard metadata, W&B logs/media and unrelated campaigns; budget these separately."}


def assert_state_equal(left, right):
    """Exact recursive equality for tensors, RNG state, and replay metadata."""
    import numpy as np
    import torch
    if torch.is_tensor(left):
        assert torch.is_tensor(right)
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping) and set(left) == set(right)
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def verify_checkpoint_replay(learner, checkpoint, step):
    """Read back every replay row while the originating learner is still paused."""
    from RL.tdmpc2_core.common.buffer import Buffer
    from utils.checkpoint_replay import load_checkpoint_replay

    restored = Buffer(deepcopy(learner.cfg), resumable=True)
    loaded = load_checkpoint_replay(checkpoint, restored)
    assert loaded["step"] == step
    assert_state_equal(learner.buffer.training_state_metadata(), restored.training_state_metadata())
    source_shards = list(learner.buffer.iter_training_state_shards(max_rows=32768))
    restored_shards = list(restored.iter_training_state_shards(max_rows=32768))
    assert_state_equal(source_shards, restored_shards)
    pending = loaded["pending_episode"]
    if learner._last_terminated or learner._last_truncated:
        assert pending is None
    else:
        assert pending["length"] == learner._episode_len
        assert pending["episode_index"] == learner._episode_idx
        assert pending["return"] == learner._episode_return
        expected = learner._episode_staging[:learner._episode_len + 1].cpu()
        assert_state_equal(expected.to_dict(), pending["rows"].to_dict())
    replay_dir = Path(str(checkpoint) + ".replay")
    return {"step": step, "roundtrip": True, "resident_rows": restored.size,
            "pending_episode_length": 0 if pending is None else pending["length"],
            "replay_artifact_bytes": sum(path.stat().st_size for path in replay_dir.rglob("*") if path.is_file())}


def memory_report(learner, production_capacity):
    """Measure allocated tensors and bound replay growth from its actual schema."""
    import torch
    storage = learner.buffer._buffer._storage._storage
    tensors = list(storage.values(True, True))
    allocated = sum(value.numel() * value.element_size() for value in tensors)
    per_row = allocated / learner.buffer.capacity
    # ru_maxrss is KiB on Linux, bytes on macOS (production is Linux).
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    full_replay = math.ceil(per_row * production_capacity)
    return {
        "peak_process_rss_bytes": rss,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "smoke_replay_allocated_bytes": allocated,
        "replay_bytes_per_row": per_row,
        "production_replay_capacity": production_capacity,
        "production_full_replay_bytes": full_replay,
        "production_replay_device": str(learner.buffer._storage_device),
        "conservative_host_rss_plus_full_replay_bytes": rss + full_replay,
        "retained_25k_replay_bank": checkpoint_bank_estimate(per_row, capacity=production_capacity),
        "note": "Early-training measurement; host bound adds the entire full replay even when storage is on CUDA. Excludes concurrent checkpoint staging and W&B subprocesses.",
    }


def run_smoke_case(index, output):
    import gymnasium as gym
    import torch
    import domains  # noqa: F401
    from RL.AMBITDMPC2 import AMBITDMPC2
    from utils.resume_runtime import capture_global_rng_state

    assert torch.cuda.is_available(), "Run the gate inside a Slurm GPU allocation"
    name, manifest, production = recipe(index)
    overrides = {"wandb": False, "seed_steps": SEED_STEPS, "pretrain_steps": 2,
                 "buffer_size": 4096, "inner_diagnostics_every": 1}
    run = deepcopy(production)
    run["alg_params"].update(overrides)
    run["total_steps"] = 516
    env = gym.make(run["env"], **manifest["env_params"])

    class ShortSmokeEpisodes(gym.Wrapper):
        """Keep the 500-decision seed episode, then test actual resets quickly."""
        total_decisions = 0

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.total_decisions += 1
            if self.total_decisions > SEED_STEPS and (self.total_decisions - SEED_STEPS) % 8 == 0:
                truncated = True
            return obs, reward, terminated, truncated, info

    env = ShortSmokeEpisodes(env)
    learner = None
    started = time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats()
        learner = AMBITDMPC2(name + "-cuda-gate", env, run["alg_params"], run, manifest)
        learner.buffer.enable_resumable_storage()
        cfg, agent = learner.cfg, learner.agent
        assert_recipe(cfg, agent, production)
        actions = []
        replay_checks = []
        act = learner._act_agent

        def observed_act(*args, **kwargs):
            outer_alpha = float(agent.alpha.detach().cpu())
            torch.cuda.synchronize()
            begin = time.perf_counter()
            action = act(*args, **kwargs)
            torch.cuda.synchronize()
            metrics, state = agent.last_inner_metrics, agent.inner_engine.state
            assert metrics["inner_compile_fallback"] == 0
            assert metrics["inner_critic_optimizer_steps"] == 128
            assert metrics["inner_actor_optimizer_steps"] == 32
            assert metrics["inner_temperature_optimizer_steps"] == 32
            assert torch.isfinite(action).all()
            row = {
                "t0": bool(kwargs["t0"]), "outer_alpha": outer_alpha,
                "alpha_initial": float(metrics["inner_alpha_initial"]),
                "alpha_final": float(metrics["inner_alpha_final"]),
                "optimizer_step_initial": float(metrics["inner_temperature_optimizer_step_initial"]),
                "optimizer_step_final": float(metrics["inner_temperature_optimizer_step_final"]),
                "episode_reset": float(metrics["inner_temperature_episode_reset"]),
                "alpha_identity": id(state.log_alpha), "optimizer_identity": id(state.temperature_optim),
                "outer_updates": agent.num_updates, "seconds": time.perf_counter() - begin,
            }
            check_temperature_transition(actions[-1] if actions else None, row, cfg.inner_temperature_scope)
            actions.append(row)
            return action

        learner._act_agent = observed_act
        with tempfile.TemporaryDirectory(prefix=name + "-") as temporary:
            models = Path(temporary) / "models"
            learner.set_checkpointing(258, str(models), "smoke", save_strat="all", checkpoint_replay=True)
            maybe_checkpoint = learner._maybe_checkpoint

            def observed_checkpoint():
                step = learner._global_step
                if step % 258:
                    return maybe_checkpoint()
                before = capture_global_rng_state()
                inner_rng_before = agent.inner_engine.rng.training_state_dict()
                result = maybe_checkpoint()
                learner.flush_checkpoints()
                assert_state_equal(before, capture_global_rng_state())
                replay_checks.append(verify_checkpoint_replay(learner, models / f"smoke_{step}", step))
                assert_state_equal(before, capture_global_rng_state())
                assert_state_equal(inner_rng_before, agent.inner_engine.rng.training_state_dict())
                return result

            learner._maybe_checkpoint = observed_checkpoint
            learner.learn(run["total_steps"])
            learner.flush_checkpoints()
            assert learner._global_step == 516 and agent.num_updates >= 16
            assert len(actions) == 15 and any(row["t0"] for row in actions)
            assert len({row["outer_updates"] for row in actions}) > 1
            assert learner._last_train_metrics["compile_fallback"] == 0
            assert [check["step"] for check in replay_checks] == [258, 516]
            assert all(torch.isfinite(parameter).all() for parameter in agent.model.parameters())
            for step in (258, 516):
                checkpoint = models / f"smoke_{step}"
                assert checkpoint.is_file() and Path(str(checkpoint) + ".metadata.json").is_file()
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            measured_memory = memory_report(learner, production["alg_params"]["buffer_size"])
            agent.load(checkpoint)
            for key, value in agent.model.state_dict().items():
                torch.testing.assert_close(value.cpu(), saved["model"][key], rtol=0, atol=0)
            case = {
                "passed": True, "index": index, "name": name, "binding": binding(), "runtime": runtime(),
                "real_decisions": 516, "inner_actions": len(actions), "outer_updates": agent.num_updates,
                "temperature_scope": cfg.inner_temperature_scope, "temperature_optimizer_scope": cfg.inner_temperature_optimizer_scope,
                "target_entropy": agent.target_entropy, "aux_return_detach_representation": cfg.aux_return_detach_representation,
                "alpha_and_adam_lifecycle": True, "checkpoint_roundtrip": True, "checkpoint_sidecars": True,
                "checkpoint_replay_roundtrip": True, "checkpoint_replay_rng_unchanged": True,
                "checkpoint_replay_checks": replay_checks,
                "checkpoint_bytes": checkpoint.stat().st_size, "compile_fallback": False,
                "warm_control_seconds": statistics.median(row["seconds"] for row in actions[-5:]),
                "memory": measured_memory, "seconds": time.perf_counter() - started,
                "production_overrides": {**overrides, "total_steps": 516, "checkpoint_every": 258,
                                         "smoke_episode_length_after_seed": 8},
                "actions": actions, "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            write_new(output, case)
            print("ALPHA_CARRY_GPU_CASE " + json.dumps({k: v for k, v in case.items() if k not in {"actions", "binding"}}), flush=True)
    finally:
        if learner is not None:
            learner.close()
        env.close()


def seal(output):
    """Merge independently scheduled cells; fail closed on missing/mismatched work."""
    expected = binding()
    cases = [read(output / f"case-{index}.json") for index in range(8)]
    selected_runtime = cases[0]["runtime"]
    for index, case in enumerate(cases):
        assert case["index"] == index and case["name"] == recipe(index)[0]
        assert case["passed"] and case["binding"] == expected
        assert case["alpha_and_adam_lifecycle"] and case["checkpoint_sidecars"] and case["checkpoint_roundtrip"]
        assert case["checkpoint_replay_roundtrip"] and case["checkpoint_replay_rng_unchanged"]
        for key in ("python", "torch", "cuda", "executable", "gpu"):
            assert case["runtime"][key] == selected_runtime[key]
    receipt = {"passed": True, "binding": expected, "runtime": selected_runtime, "cases": cases,
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "estimated_all_25k_checkpoint_bytes": 40 * sum(row["checkpoint_bytes"] for row in cases)}
    # Equal architecture/replay schemas across all factorial cells are required;
    # use the measured largest payload to keep this campaign estimate conservative.
    receipt["retained_checkpoint_bank_estimate"] = checkpoint_bank_estimate(
        max(row["memory"]["replay_bytes_per_row"] for row in cases),
        checkpoint_bytes=max(row["checkpoint_bytes"] for row in cases),
    )
    write_new(output / "receipt.json", receipt)
    print("ALPHA_CARRY_GPU_GATE_PASSED " + str(output / "receipt.json"), flush=True)


def configure_benchmark_replay(learner, output):
    """Keep CPU replay through learn's rebuild without writing checkpoints."""
    learner.set_checkpointing(25_000, str(output.parent / "checkpointing-disabled"),
                              "benchmark", save_strat="none", checkpoint_replay=True)


def benchmark_worker(args):
    cpus = [int(cpu) for cpu in args.worker_cpus.split(",")]
    assert len(cpus) in COUNTS and len(cpus) == len(set(cpus))
    assert set(cpus) <= os.sched_getaffinity(0)
    assert len(physical_core_cpus(cpus)) == len(cpus)
    os.sched_setaffinity(0, cpus)
    import gymnasium as gym
    import torch
    import domains  # noqa: F401
    from RL.AMBITDMPC2 import AMBITDMPC2

    torch.set_num_threads(len(cpus))
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    name, manifest, production = recipe(7)
    run = deepcopy(production)
    overrides = {"wandb": False, "seed_steps": SEED_STEPS, "pretrain_steps": 2, "buffer_size": 4096}
    run["alg_params"].update(overrides)
    run["total_steps"] = SEED_STEPS + 1 + args.warmup + args.decisions
    run.update(checkpoint_every=None, save_strat="none")
    manifest = {**manifest, "checkpoint_every": None, "save_strat": "none", "save_trials": "none", "logs": "none"}
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    try:
        torch.cuda.reset_peak_memory_stats()
        learner = AMBITDMPC2(name + "-cpu-benchmark", env, run["alg_params"], run, manifest)
        # Legacy learn() reconstructs replay. Keep the production CPU placement
        # through that reset while disabling every actual checkpoint write.
        configure_benchmark_replay(learner, args.output)
        agent = learner.agent
        assert_recipe(learner.cfg, agent, production)
        timer = TrainingWindow(SEED_STEPS + args.warmup + 2, args.decisions, torch.cuda.synchronize)
        act, update, log_step = learner._act_agent, agent.update, learner._log_wandb_step
        compile_counts = {}

        def observed_act(*positional, **kwargs):
            step = learner._global_step + 1
            if step == timer.first_step:
                compile_counts["before"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
            timer.before_action(step, learner._num_updates)
            started = time.perf_counter()
            action = act(*positional, **kwargs)
            timer.after_action(step, time.perf_counter() - started)
            assert action.device.type == "cpu" and torch.isfinite(action).all()
            assert agent.last_inner_metrics["inner_compile_fallback"] == 0
            return action

        def observed_update(*positional, **kwargs):
            metrics = update(*positional, **kwargs)
            assert metrics["compile_fallback"] == 0
            return metrics

        def observed_log_step(*positional, **kwargs):
            result = log_step(*positional, **kwargs)
            timer.after_decision(learner._global_step, learner._num_updates)
            return result

        learner._act_agent, agent.update, learner._log_wandb_step = observed_act, observed_update, observed_log_step
        learner.learn(run["total_steps"])
        assert learner.buffer.resumable_storage and str(learner.buffer._storage_device) == "cpu"
        compile_counts["after"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
        assert compile_counts["before"] == compile_counts["after"], "Compilation occurred in timing window"
        assert timer.result is not None
        measured = timer.result
        measured["projected_1m_decision_days"] = measured["seconds_per_decision"] * 1_000_000 / 86400
        report = {"passed": True, "physical_cores": len(cpus), "logical_cpu_ids": cpus,
                  "runtime": runtime(), "binding": binding(), "config": name,
                  "warmup_active_decisions": args.warmup, "measured": measured,
                  "compile_graph_counts": compile_counts, "memory": memory_report(learner, production["alg_params"]["buffer_size"]),
                  "production_overrides": overrides, "checkpointing": False, "wandb": False,
                  "created_utc": datetime.now(timezone.utc).isoformat()}
        write_new(args.output, report)
        print("ALPHA_CARRY_CPU_CASE " + json.dumps({"cores": len(cpus), "seconds_per_decision": measured["seconds_per_decision"],
                                                   "projected_1m_decision_days": measured["projected_1m_decision_days"]}), flush=True)
    finally:
        if learner is not None:
            learner.close()
        env.close()


def benchmark(args):
    assert os.environ.get("SLURM_JOB_ID"), "Benchmark requires a scheduled GPU allocation"
    cores = physical_core_cpus(os.sched_getaffinity(0))
    assert len(cores) >= 4 and int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) >= 4
    args.output.mkdir(parents=True, exist_ok=False)
    for count in COUNTS:
        environment = dict(os.environ)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[key] = str(count)
        environment.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE", WANDB_MODE="disabled")
        command = [sys.executable, str(Path(__file__).resolve()), "benchmark-worker", "--worker-cpus",
                   ",".join(map(str, cores[:count])), "--output", str(args.output / f"cores-{count}.json"),
                   "--warmup", str(args.warmup), "--decisions", str(args.decisions)]
        with (args.output / f"cores-{count}.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        print("ALPHA_CARRY_CPU_PROGRESS " + str(count), flush=True)
    cases = [read(args.output / f"cores-{count}.json") for count in COUNTS]
    fastest = min(row["measured"]["seconds_per_decision"] for row in cases)
    selected = min(row["physical_cores"] for row in cases if row["measured"]["seconds_per_decision"] <= fastest * 1.05)
    write_new(args.output / "results.json", {"passed": True, "binding": binding(), "cases": cases,
              "recommended_cpus_within_5_percent_of_fastest": selected,
              "scope": "Short early-replay training; excludes compilation, W&B, checkpoint I/O and scheduled diagnostics. Not a production walltime guarantee."})


def launch(index, output, gate):
    expected = binding()
    receipt = read(gate)
    assert receipt["passed"] and receipt["binding"] == expected, "GPU gate does not match this source/configuration"
    cases = receipt["cases"]
    assert len(cases) == 8 and {row["index"] for row in cases} == set(range(8))
    for row in cases:
        assert row["passed"] and row["name"] == recipe(row["index"])[0]
        assert row["binding"] == expected and row["alpha_and_adam_lifecycle"]
        assert row["checkpoint_replay_roundtrip"] and row["checkpoint_replay_rng_unchanged"]
    current_runtime = runtime()
    for key in ("python", "torch", "cuda", "executable", "gpu"):
        assert current_runtime[key] == receipt["runtime"][key], f"GPU gate used another {key}"
    assert "L40S" in current_runtime["gpu"].upper(), "This campaign requests L40S GPUs"
    name, manifest, run = recipe(index)
    assert run["total_steps"] == 1_000_000 and run["checkpoint_every"] == 25_000
    assert run["checkpoint_replay"] is True
    if not os.environ.get("WANDB_API_KEY"):
        import netrc
        auth = netrc.netrc().authenticators("api.wandb.ai")
        assert auth and auth[2], "W&B credential unavailable"
    lineage = Path(os.environ["AMBI_LINEAGE_DIR"])
    assert lineage.is_absolute()
    resume_mode = os.environ.get("AMBI_RESUME_MODE", "new")
    assert resume_mode in {"new", "required"}
    drain_seconds = int(os.environ["AMBI_DRAIN_AFTER_SECONDS"])
    assert 0 < drain_seconds <= 341700
    output.mkdir(parents=True, exist_ok=False)
    single_manifest = {**deepcopy(manifest), "configs": [name], "trials": 1}
    single_path = output / "experiment.json"
    write_new(single_path, single_manifest)
    launch_data = {"binding": expected, "config": name, "run_params": run,
                   "gate_receipt": str(gate), "gate_sha256": digest(gate), "runtime": current_runtime,
                   "slurm": {key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE")},
                   "lineage_dir": str(lineage), "resume_mode": resume_mode,
                   "resume_checkpoint_minutes": 15, "drain_after_seconds": drain_seconds,
                   "single_cell_manifest": str(single_path), "single_cell_manifest_sha256": digest(single_path),
                   "wandb_run_id_source": str(lineage / "LINEAGE.json") + ":metadata.initial_wandb_run_id",
                   "created_utc": datetime.now(timezone.utc).isoformat()}
    write_new(output / "launch.json", launch_data)
    command = [sys.executable, "main.py", "--run", str(single_path), "--alg-dir", "configs/dmcontrol/algs",
               "--log-dir", str(output), "--num-runs", "1", "--lineage-dir", str(lineage),
               "--resume-mode", resume_mode, "--resume-wandb-mode", "online",
               "--resume-checkpoint-minutes", "15", "--drain-after-seconds", str(drain_seconds)]
    print("ALPHA_CARRY_TRAINING_LAUNCH " + json.dumps(launch_data), flush=True)
    os.execv(sys.executable, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "seal", "benchmark", "benchmark-worker", "train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--index", type=int, choices=range(8))
    parser.add_argument("--worker-cpus")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--decisions", type=int, default=128)
    args = parser.parse_args()
    args.output = args.output.resolve()
    assert args.output != ROOT and ROOT not in args.output.parents, "Store outputs outside source"
    if args.mode == "smoke":
        args.output.mkdir(parents=True, exist_ok=True)
        for index in range(8) if args.index is None else (args.index,):
            run_smoke_case(index, args.output / f"case-{index}.json")
            # Discard specialization guards between cells; Inductor's disk cache
            # remains shared. Independent --index jobs are also supported.
            import torch
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
        if args.index is None:
            seal(args.output)
    elif args.mode == "seal":
        seal(args.output)
    elif args.mode in {"benchmark", "benchmark-worker"}:
        assert args.warmup >= 10 and args.decisions >= 100
        assert SEED_STEPS + 1 + args.warmup + args.decisions < 1000
        (benchmark_worker if args.mode == "benchmark-worker" else benchmark)(args)
    else:
        assert args.index is not None and args.gate is not None
        launch(args.index, args.output, args.gate)


if __name__ == "__main__":
    main()
