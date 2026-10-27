"""Single-cell K3 alpha-carry CUDA gate, CPU sizing and native-resume launch."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm import ambi_alpha_carry_training as shared
from slurm.ambi_transfer_training import binding as source_binding, digest, read, recipe as source_recipe, runtime, write_new
from slurm.benchmark_ambi_training_cpu import TrainingWindow, physical_core_cpus

MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_alpha_carry_k3_training.json"
SUPPORTED_MANIFESTS = (
    str(MANIFEST.relative_to(ROOT)),
    "configs/dmcontrol/experiments/ambi_alpha_carry_k3_j10_training.json",
)
COUNTS = (1, 2)
SEED_STEPS = 500
HELD_ZERO_WORK = ("inner_rollouts", "inner_steps", "inner_updates", "inner_model_steps",
                  "inner_replay_draws", "inner_critic_optimizer_steps", "inner_actor_optimizer_steps",
                  "inner_temperature_optimizer_steps", "inner_q_evaluations")


def selected_manifest():
    selected = os.environ.get("AMBI_ALPHA_CARRY_K3_MANIFEST", SUPPORTED_MANIFESTS[0])
    assert selected in SUPPORTED_MANIFESTS, "Select a supported K3 campaign manifest"
    return ROOT / selected


def binding():
    return source_binding(selected_manifest())


def recipe():
    return source_recipe(0, selected_manifest())


def assert_recipe(cfg, agent, run):
    expected_rounds = run["alg_params"]["inner_rounds"]
    assert expected_rounds in (8, 10)
    shared.assert_recipe(cfg, agent, run, expected_rounds=expected_rounds)
    assert cfg.inner_solve_interval == 3 and cfg.inner_hold_encoder == "solve_snapshot"
    assert cfg.inner_temperature_scope == cfg.inner_temperature_optimizer_scope == "run"
    assert cfg.aux_return_detach_representation and agent.target_entropy == -10.5


def assert_cadence(metrics, decision, *, expected_rounds=8):
    solve = decision % 3 == 0
    assert metrics["inner_solve_performed"] == float(solve)
    assert metrics["inner_policy_held"] == float(not solve)
    assert metrics["inner_action_age"] == decision % 3
    assert all(math.isfinite(float(value)) for value in metrics.values())
    if solve:
        assert metrics["inner_compile_fallback"] == 0
        assert metrics["inner_critic_optimizer_steps"] == 16 * expected_rounds
        assert metrics["inner_actor_optimizer_steps"] == metrics["inner_temperature_optimizer_steps"] == 4 * expected_rounds
    else:
        assert all(metrics[key] == 0 for key in HELD_ZERO_WORK)
        assert metrics["inner_policy_evaluations"] == 1
    return solve


def measured_memory(learner, production, checkpoint_bytes=0):
    result = shared.memory_report(learner, production["alg_params"]["buffer_size"])
    result["retained_25k_replay_bank"] = shared.checkpoint_bank_estimate(
        result["replay_bytes_per_row"], cells=1, checkpoint_bytes=checkpoint_bytes)
    return result


def smoke(output):
    import gymnasium as gym
    import torch
    import domains  # noqa: F401
    from RL.AMBITDMPC2 import AMBITDMPC2
    from utils.resume_runtime import capture_global_rng_state

    assert torch.cuda.is_available(), "Run inside a scheduled GPU allocation"
    name, manifest, production = recipe()
    run = deepcopy(production)
    overrides = dict(wandb=False, seed_steps=SEED_STEPS, pretrain_steps=2,
                     buffer_size=4096, inner_diagnostics_every=1)
    run["alg_params"].update(overrides)
    run["total_steps"] = 516

    class ShortEpisodes(gym.Wrapper):
        total_decisions = 0

        def step(self, action):
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.total_decisions += 1
            if self.total_decisions > SEED_STEPS and (self.total_decisions - SEED_STEPS) % 8 == 0:
                truncated = True
            return observation, reward, terminated, truncated, info

    env = ShortEpisodes(gym.make(run["env"], **manifest["env_params"]))
    learner = None
    actions, solves, replay_checks = [], [], []
    started = time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats()
        learner = AMBITDMPC2(name + "-cuda-gate", env, run["alg_params"], run, manifest)
        learner.buffer.enable_resumable_storage()
        agent = learner.agent
        assert_recipe(learner.cfg, agent, production)
        act = learner._act_agent
        decision = 0

        def observed_act(*args, **kwargs):
            nonlocal decision
            if kwargs["t0"]:
                decision = 0
            engine = agent.inner_engine
            outer_alpha = float(agent.alpha.detach().cpu())
            held = decision % 3 != 0
            if held:
                before_alpha = engine.state.log_alpha.detach().clone()
                before_optimizer = deepcopy(engine.state.temperature_optim.state_dict())
                before_graphs = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
            torch.cuda.synchronize()
            begin = time.perf_counter()
            action = act(*args, **kwargs)
            torch.cuda.synchronize()
            metrics, state = agent.last_inner_metrics, engine.state
            solve = assert_cadence(metrics, decision, expected_rounds=learner.cfg.inner_rounds)
            assert torch.isfinite(action).all()
            row = {"decision": decision, "solve": solve, "t0": bool(kwargs["t0"]),
                   "outer_updates": agent.num_updates, "seconds": time.perf_counter() - begin}
            if solve:
                row.update(outer_alpha=outer_alpha, alpha_initial=float(metrics["inner_alpha_initial"]),
                           alpha_final=float(metrics["inner_alpha_final"]),
                           optimizer_step_initial=float(metrics["inner_temperature_optimizer_step_initial"]),
                           optimizer_step_final=float(metrics["inner_temperature_optimizer_step_final"]),
                           episode_reset=float(metrics["inner_temperature_episode_reset"]),
                           alpha_identity=id(state.log_alpha), optimizer_identity=id(state.temperature_optim))
                shared.check_temperature_transition(solves[-1] if solves else None, row, "run",
                                                    temperature_steps=4 * learner.cfg.inner_rounds)
                solves.append(row)
            else:
                shared.assert_state_equal(before_alpha, state.log_alpha)
                shared.assert_state_equal(before_optimizer, state.temperature_optim.state_dict())
                assert before_graphs == int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
            actions.append(row)
            decision += 1
            return action

        learner._act_agent = observed_act
        with tempfile.TemporaryDirectory(prefix="ambi-k3-smoke-") as temporary:
            models = Path(temporary) / "models"
            learner.set_checkpointing(258, str(models), "smoke", save_strat="all", checkpoint_replay=True)
            maybe_checkpoint = learner._maybe_checkpoint

            def observed_checkpoint():
                step = learner._global_step
                if step % 258:
                    return maybe_checkpoint()
                before = capture_global_rng_state()
                inner_rng = agent.inner_engine.rng.training_state_dict()
                result = maybe_checkpoint()
                learner.flush_checkpoints()
                shared.assert_state_equal(before, capture_global_rng_state())
                replay_checks.append(shared.verify_checkpoint_replay(learner, models / f"smoke_{step}", step))
                shared.assert_state_equal(before, capture_global_rng_state())
                shared.assert_state_equal(inner_rng, agent.inner_engine.rng.training_state_dict())
                return result

            learner._maybe_checkpoint = observed_checkpoint
            learner.learn(516)
            learner.flush_checkpoints()
            assert learner._global_step == 516 and agent.num_updates >= 16
            assert len(actions) == 15 and len(solves) == 6
            assert len({row["outer_updates"] for row in actions}) == len(actions)
            assert any(row["t0"] for row in solves[1:]), "Must test run-alpha across an episode boundary"
            assert learner._last_train_metrics["compile_fallback"] == 0
            assert [check["step"] for check in replay_checks] == [258, 516]
            assert all(torch.isfinite(parameter).all() for parameter in agent.model.parameters())
            checkpoint = models / "smoke_516"
            assert checkpoint.is_file() and Path(str(checkpoint) + ".metadata.json").is_file()
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            memory = measured_memory(learner, production, checkpoint.stat().st_size)
            agent.load(checkpoint)
            shared.assert_state_equal({k: v.cpu() for k, v in agent.model.state_dict().items()}, saved["model"])
            receipt = {"passed": True, "binding": binding(), "runtime": runtime(), "config": name,
                       "inner_rounds": learner.cfg.inner_rounds,
                       "real_decisions": 516, "inner_actions": len(actions), "inner_solves": len(solves),
                       "held_cadence": True, "held_alpha_adam_unchanged": True, "alpha_and_adam_lifecycle": True,
                       "checkpoint_roundtrip": True, "checkpoint_replay_roundtrip": True,
                       "checkpoint_replay_rng_unchanged": True, "checkpoint_replay_checks": replay_checks,
                       "compile_fallback": False, "memory": memory, "seconds": time.perf_counter() - started,
                       "production_overrides": {**overrides, "total_steps": 516, "checkpoint_every": 258,
                                                "smoke_episode_length_after_seed": 8},
                       "actions": actions, "created_utc": datetime.now(timezone.utc).isoformat()}
            output.mkdir(parents=True, exist_ok=False)
            write_new(output / "receipt.json", receipt)
            print("K3_GPU_GATE_PASSED " + str(output / "receipt.json"), flush=True)
    finally:
        if learner is not None:
            learner.close()
        env.close()


def benchmark_worker(args):
    cpus = [int(cpu) for cpu in args.worker_cpus.split(",")]
    assert len(cpus) in COUNTS and len(cpus) == len(set(cpus))
    assert set(cpus) <= os.sched_getaffinity(0) and len(physical_core_cpus(cpus)) == len(cpus)
    os.sched_setaffinity(0, cpus)
    import gymnasium as gym
    import torch
    import domains  # noqa: F401
    from RL.AMBITDMPC2 import AMBITDMPC2

    torch.set_num_threads(len(cpus))
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    name, manifest, production = recipe()
    run = deepcopy(production)
    overrides = dict(wandb=False, seed_steps=SEED_STEPS, pretrain_steps=2, buffer_size=4096)
    run["alg_params"].update(overrides)
    run["total_steps"] = SEED_STEPS + 1 + args.warmup + args.decisions
    run.update(checkpoint_every=None, save_strat="none")
    manifest = {**manifest, "checkpoint_every": None, "save_strat": "none", "save_trials": "none", "logs": "none"}
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    try:
        torch.cuda.reset_peak_memory_stats()
        learner = AMBITDMPC2(name + "-cpu-benchmark", env, run["alg_params"], run, manifest)
        shared.configure_benchmark_replay(learner, args.output)
        agent = learner.agent
        assert_recipe(learner.cfg, agent, production)
        timer = TrainingWindow(SEED_STEPS + args.warmup + 2, args.decisions, torch.cuda.synchronize)
        act, update, log_step = learner._act_agent, agent.update, learner._log_wandb_step
        compile_counts = {}
        active, measured_solves = 0, 0

        def observed_act(*positional, **kwargs):
            nonlocal active, measured_solves
            step = learner._global_step + 1
            if step == timer.first_step:
                compile_counts["before"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
            timer.before_action(step, learner._num_updates)
            started = time.perf_counter()
            action = act(*positional, **kwargs)
            timer.after_action(step, time.perf_counter() - started)
            assert action.device.type == "cpu" and torch.isfinite(action).all()
            solve = assert_cadence(agent.last_inner_metrics, active, expected_rounds=learner.cfg.inner_rounds)
            if timer.first_step <= step <= timer.last_step:
                measured_solves += solve
            active += 1
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
        assert timer.result and active == args.warmup + args.decisions
        assert measured_solves == args.decisions // 3
        measured = timer.result
        measured["projected_1m_decision_days"] = measured["seconds_per_decision"] * 1_000_000 / 86400
        measured["inner_solves"] = measured_solves
        report = {"passed": True, "physical_cores": len(cpus), "logical_cpu_ids": cpus,
                  "inner_rounds": learner.cfg.inner_rounds,
                  "runtime": runtime(), "binding": binding(), "config": name,
                  "warmup_active_decisions": args.warmup, "measured": measured,
                  "compile_graph_counts": compile_counts, "memory": measured_memory(learner, production),
                  "production_overrides": overrides, "checkpointing": False, "wandb": False,
                  "created_utc": datetime.now(timezone.utc).isoformat()}
        write_new(args.output, report)
        print("K3_CPU_CASE " + json.dumps({"cores": len(cpus), "seconds_per_decision": measured["seconds_per_decision"]}), flush=True)
    finally:
        if learner is not None:
            learner.close()
        env.close()


def benchmark(args):
    assert os.environ.get("SLURM_JOB_ID"), "Benchmark requires a scheduled GPU allocation"
    cores = physical_core_cpus(os.sched_getaffinity(0))
    assert len(cores) >= 2 and int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) >= 2
    args.output.mkdir(parents=True, exist_ok=False)
    for count in COUNTS:
        environment = dict(os.environ)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[key] = str(count)
        environment.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE", WANDB_MODE="disabled",
                           AMBI_ALPHA_CARRY_K3_MANIFEST=str(selected_manifest().relative_to(ROOT)))
        command = [sys.executable, str(Path(__file__).resolve()), "benchmark-worker", "--worker-cpus",
                   ",".join(map(str, cores[:count])), "--output", str(args.output / f"cores-{count}.json"),
                   "--warmup", str(args.warmup), "--decisions", str(args.decisions)]
        with (args.output / f"cores-{count}.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        print("K3_CPU_PROGRESS " + str(count), flush=True)
    cases = [read(args.output / f"cores-{count}.json") for count in COUNTS]
    assert cases[0]["runtime"] == cases[1]["runtime"]
    fastest = min(row["measured"]["seconds_per_decision"] for row in cases)
    selected = min(row["physical_cores"] for row in cases if row["measured"]["seconds_per_decision"] <= fastest * 1.05)
    write_new(args.output / "results.json", {"passed": True, "binding": binding(), "cases": cases,
              "recommended_cpus_within_5_percent_of_fastest": selected,
              "scope": "Short early-replay K3 training; excludes compilation,W&B,checkpoint I/O and scheduled diagnostics. Add measured startup/I/O and a modest margin when requesting walltime."})


def validate_gate(receipt, expected):
    assert receipt["passed"] and receipt["binding"] == expected, "GPU gate source/configuration mismatch"
    assert receipt["config"] == recipe()[0]
    assert receipt.get("inner_rounds", 8) == recipe()[2]["alg_params"]["inner_rounds"]
    for key in ("held_cadence", "held_alpha_adam_unchanged", "alpha_and_adam_lifecycle",
                "checkpoint_roundtrip", "checkpoint_replay_roundtrip", "checkpoint_replay_rng_unchanged"):
        assert receipt[key], f"GPU gate missing {key}"
    assert receipt["compile_fallback"] is False


def launch(output, gate, benchmark_path):
    expected = binding()
    receipt = read(gate)
    validate_gate(receipt, expected)
    benchmark_report = read(benchmark_path)
    assert benchmark_report["passed"] and benchmark_report["binding"] == expected
    cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    assert cpus == benchmark_report["recommended_cpus_within_5_percent_of_fastest"]
    current_runtime = runtime()
    assert "L40S" in current_runtime["gpu"].upper()
    for row in [receipt, *benchmark_report["cases"]]:
        assert row["passed"] and row["binding"] == expected
        assert row["runtime"] == current_runtime, "GPU gate/benchmark runtime mismatch"
    assert {row["physical_cores"] for row in benchmark_report["cases"]} == set(COUNTS)
    name, manifest, run = recipe()
    lineage = Path(os.environ["AMBI_LINEAGE_DIR"])
    assert lineage.is_absolute()
    resume_mode = os.environ.get("AMBI_RESUME_MODE", "new")
    assert resume_mode in {"new", "required"}
    drain_seconds = int(os.environ["AMBI_DRAIN_AFTER_SECONDS"])
    walltime_seconds = int(os.environ["AMBI_TRAIN_WALLTIME_SECONDS"])
    assert 0 < drain_seconds <= walltime_seconds - 1800, "Reserve at least30minutes for final checkpoint publication"
    assert run["total_steps"] == 1_000_000 and run["checkpoint_every"] == 25_000 and run["checkpoint_replay"]
    output.mkdir(parents=True, exist_ok=False)
    single_path = output / "experiment.json"
    write_new(single_path, manifest)
    launch_data = {"binding": expected, "config": name, "run_params": run,
                   "gate_receipt": str(gate), "gate_sha256": digest(gate), "runtime": current_runtime,
                   "benchmark_receipt": str(benchmark_path), "benchmark_sha256": digest(benchmark_path),
                   "slurm": {key: os.environ.get(key) for key in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE")},
                   "lineage_dir": str(lineage), "resume_mode": resume_mode, "resume_checkpoint_minutes": 15,
                   "drain_after_seconds": drain_seconds, "requested_walltime_seconds": walltime_seconds,
                   "single_cell_manifest": str(single_path), "single_cell_manifest_sha256": digest(single_path),
                   "wandb_run_id_source": str(lineage / "LINEAGE.json") + ":metadata.initial_wandb_run_id",
                   "created_utc": datetime.now(timezone.utc).isoformat()}
    write_new(output / "launch.json", launch_data)
    command = [sys.executable, "main.py", "--run", str(single_path), "--alg-dir", "configs/dmcontrol/algs",
               "--log-dir", str(output), "--num-runs", "1", "--lineage-dir", str(lineage),
               "--resume-mode", resume_mode, "--resume-wandb-mode", "online",
               "--resume-checkpoint-minutes", "15", "--drain-after-seconds", str(drain_seconds)]
    print("K3_TRAINING_LAUNCH " + json.dumps(launch_data), flush=True)
    os.execv(sys.executable, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "benchmark", "benchmark-worker", "train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--worker-cpus")
    parser.add_argument("--warmup", type=int, default=24)
    parser.add_argument("--decisions", type=int, default=96)
    args = parser.parse_args()
    args.output = args.output.resolve()
    assert args.output != ROOT and ROOT not in args.output.parents, "Keep outputs outside source"
    if args.mode in ("benchmark", "benchmark-worker"):
        assert args.warmup >= 12 and args.decisions >= 60
        assert args.warmup % 3 == args.decisions % 3 == 0
        assert SEED_STEPS + 1 + args.warmup + args.decisions < 1000
        (benchmark_worker if args.mode == "benchmark-worker" else benchmark)(args)
    elif args.mode == "smoke":
        smoke(args.output)
    else:
        assert args.gate and args.benchmark
        launch(args.output, args.gate, args.benchmark)


if __name__ == "__main__":
    main()
