"""Gate one reward-only primary-Q training recipe and launch durable resume.

Each selected manifest has one cell. The GPU gate compares one and two physical
CPU cores (also four for the no-inner baseline), using real short training, production shapes and update dose, strict
compilation, and the CPU replay placement used by exact resume. Timing excludes
periodic checkpoint I/O, W&B, mature replay and the production pretraining dose.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_transfer_training import binding, digest, read, recipe, resolve_manifest, runtime, write_new
from slurm.ambi_mppi_training import smoke_recipe as _smoke_recipe
from slurm.benchmark_ambi_training_cpu import TrainingWindow, action_work, phase_summary, physical_core_cpus

CPU_COUNTS = (1, 2)
SEED_STEPS = 500
MANIFESTS = {
    "sac": ROOT / "configs/dmcontrol/experiments/ambi_return_only_sac_training.json",
    "mppi": ROOT / "configs/dmcontrol/experiments/ambi_return_only_mppi_training.json",
    "none": ROOT / "configs/dmcontrol/experiments/ambi_return_only_no_inner_training.json",
}


def cpu_counts(operator):
    assert operator in MANIFESTS
    return (1, 2, 4) if operator == "none" else CPU_COUNTS


def smoke_defaults(operator):
    return (24, 400) if operator == "none" else (12, 120)


def validate_campaign(manifest_path):
    path = resolve_manifest(manifest_path)
    manifest = read(path)
    assert len(manifest["configs"]) == 1 and manifest["trials"] == 1
    name, _, run = recipe(0, path)
    assert (run["seed"], run["total_steps"], run["checkpoint_every"], run["save_strat"]) == (55, 500000, 25000, "all")
    assert manifest["env_params"] == {"task": "humanoid-walk", "obs": "state", "render_mode": None}
    # These unchanged defaults are omitted by the older MPPI/SAC recipes.
    # Validate the explicit resolved meaning without modifying the manifest.
    p = {"inner_actor_entropy_mode": "squashed", "inner_entropy_enabled": True,
         "inner_critic_bernoulli_p": 0., "mppi_terminal_q_reduction": "mean_pair",
         **run["alg_params"]}
    expected = {
        "model_size": 5, "batch_size": 256, "train_unroll_horizon": 3,
        "critic_value_mode": "single", "aux_return_mode": "off",
        "outer_critic_target": "reward_only", "inner_sac_critic_target": "reward_only",
        "inner_actor_source": "sac", "inner_critic_source": "sac",
        "inner_horizon_actor_source": "sac", "inner_horizon_critic_source": "sac",
        "ent_coef": "auto_1.0", "target_entropy": -10.5,
        "outer_actor_entropy_mode": "squashed", "inner_actor_entropy_mode": "squashed",
        "inner_entropy_enabled": True, "sac_actor_loss_scale_mode": "none",
        "compile": True, "compile_strict": True, "critic_ensemble_backend": "vmap",
        "wandb": True, "wandb_project": "ambi", "wandb_mode": "online",
        "seed_steps": 2500, "pretrain_steps": 2500, "utd": 1,
        "inner_rollout_horizon": 3, "num_q": 5, "q_pair_size": 2,
        "inner_critic_bernoulli_p": 0., "mppi_terminal_q_reduction": "mean_pair",
    }
    assert all(p.get(key) == value for key, value in expected.items()), "Reward-only production recipe changed"
    assert p["inner_operator"] in MANIFESTS
    interval = p.get("inner_solve_interval", 1)
    if p["inner_operator"] == "sac":
        assert interval in (1, 3)
        expected = {"inner_rounds": 8, "inner_rollouts_per_round": 128,
                    "inner_batch_size": 256, "inner_replay_capacity": 3072,
                    "inner_critic_updates_per_round": 16, "inner_actor_updates_per_round": 4,
                    "inner_execution_action": "policy_sample", "inner_temperature_mode": "auto",
                    "inner_temperature_initialization": "inherit_outer", "inner_target_entropy": "inherit_outer"}
        assert all(p.get(key) == value for key, value in expected.items())
        for component in ("actor", "critic", "temperature", "replay", "actor_optimizer", "critic_optimizer", "temperature_optimizer"):
            assert p[f"inner_{component}_scope"] == "action"
    elif p["inner_operator"] == "mppi":
        assert interval == 1, "MPPI does not support held-policy cadence"
        expected = {"inner_mppi_iterations": 8,
                    "inner_mppi_num_elites": 64, "inner_mppi_num_pi_trajs": 24,
                    "inner_mppi_temperature": .5, "inner_mppi_min_std": .05, "inner_mppi_max_std": 2.,
                    "inner_mppi_warm_start_scope": "episode", "inner_mppi_eval_action": "weighted_elite"}
        assert all(p.get(key) == value for key, value in expected.items())
    else:
        assert interval == 1 and p["inner_model_step_budget"] == 0
        assert p["inner_execution_action"] == "policy_sample"
        assert p.get("inner_execution_std_scale", 1.) == 1.
    return name, manifest, run


def smoke_recipe(warmup, decisions, manifest_path):
    validate_campaign(manifest_path)
    return _smoke_recipe(0, warmup, decisions, manifest_path)


def validate_learner(cfg, agent):
    assert cfg.compile and cfg.compile_strict and cfg.critic_ensemble_backend == "vmap"
    assert (cfg.model_size, cfg.enc_dim, cfg.mlp_dim, cfg.latent_dim, cfg.num_q) == (5, 256, 512, 512, 5)
    assert (cfg.action_dim, cfg.batch_size, cfg.train_unroll_horizon) == (21, 256, 3)
    assert cfg.aux_return_mode == "off" and agent.aux_return is None
    assert cfg.critic_value_mode == "single" and not agent.model.value_spec.is_split
    assert cfg.outer_critic_target == cfg.inner_sac_critic_target == "reward_only"
    assert not hasattr(agent.model, "_aux_return_Qs") and not hasattr(agent.model, "_target_aux_return_Qs")
    assert not any("aux_return" in key for key in agent.model.state_dict())
    assert agent.inner_engine._critic_base is agent.inner_engine._horizon_critic is agent.model._Qs
    assert agent.ent_coef_optim is not None and agent.log_ent_coef is not None
    assert agent.target_entropy == -10.5 and cfg.sac_actor_loss_scale_mode == "none"
    assert cfg.inner_entropy_enabled and cfg.utd == 1
    assert cfg.mppi_terminal_q_reduction == "mean_pair" and cfg.inner_critic_bernoulli_p == 0.
    assert cfg.outer_actor_entropy_mode == cfg.inner_actor_entropy_mode == "squashed"
    if cfg.inner_operator == "mppi":
        assert (cfg.inner_rollout_horizon, cfg.inner_mppi_iterations, cfg.inner_mppi_num_samples) == (3, 8, 512)
        assert (cfg.inner_mppi_num_elites, cfg.inner_mppi_num_pi_trajs) == (64, 24)
        assert cfg.inner_critic_updates_per_action == cfg.inner_actor_updates_per_action == 0
    elif cfg.inner_operator == "sac":
        assert (cfg.inner_rollout_horizon, cfg.inner_rounds, cfg.inner_rollouts_per_round) == (3, 8, 128)
        assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action, cfg.inner_temperature_updates_per_action) == (128, 32, 32)
    else:
        assert cfg.inner_operator == "none" and cfg.inner_solve_interval == 1
        assert cfg.inner_execution_action == "policy_sample" and cfg.inner_execution_std_scale == 1.
        assert agent.inner_engine._actor_base is agent.model._pi
        for key in ("inner_model_step_budget", "inner_rounds", "inner_rollouts_per_round",
                    "inner_critic_updates_per_action", "inner_actor_updates_per_action",
                    "inner_temperature_updates_per_action", "inner_mppi_iterations"):
            assert getattr(cfg, key) == 0, key


def validate_action(metrics, *, operator, decision, interval):
    if operator == "sac":
        return action_work(metrics, decision=decision, interval=interval, probability=0.)
    assert operator in {"none", "mppi"} and interval == 1
    for key in ("inner_critic_optimizer_steps", "inner_actor_optimizer_steps", "inner_temperature_optimizer_steps"):
        assert metrics[key] == 0
    if operator == "none":
        for key in ("inner_active", "inner_model_steps_budget", "inner_steps", "inner_model_steps",
                    "inner_total_model_steps", "inner_realized_model_steps", "inner_nominal_model_steps",
                    "inner_rounds", "inner_iterations", "inner_mppi_iterations", "inner_rollouts",
                    "inner_updates", "inner_update_slots", "inner_target_updates", "inner_q_evaluations",
                    "inner_replay_draws", "inner_buffer_size"):
            assert metrics[key] == 0, (decision, key, metrics[key])
        assert metrics["inner_policy_evaluations"] == 1
    else:
        assert metrics["inner_mppi_iterations"] == 8 and metrics["inner_model_steps"] == 12336
    assert metrics["inner_compile_fallback"] == 0
    assert all(math.isfinite(float(value)) for value in metrics.values())
    return operator == "mppi"


def training_phases(rows, operator):
    """A direct prior action is neither a solve nor a held inner policy."""
    summary = phase_summary(rows)
    if operator == "none":
        assert not any(row["solve"] for row in rows)
        summary.update(solve_decisions=0, held_decisions=0, prior_decisions=len(rows))
        for statistic in ("mean", "median"):
            summary[f"prior_action_seconds_{statistic}"] = summary[f"held_action_seconds_{statistic}"]
            summary[f"held_action_seconds_{statistic}"] = None
    return summary


def _optimizer_steps(optimizer):
    steps = [int(state["step"].item()) for state in optimizer.state.values() if "step" in state]
    assert steps and len(set(steps)) == 1, "Optimizer parameters did not receive equal update counts"
    return steps[0]


def worker(args):
    cpus = [int(cpu) for cpu in args.worker_cpus.split(",")]
    _, _, production = validate_campaign(args.manifest)
    assert len(cpus) in cpu_counts(production["alg_params"]["inner_operator"]) and len(cpus) == len(set(cpus))
    assert set(cpus) <= os.sched_getaffinity(0) and len(physical_core_cpus(cpus)) == len(cpus)
    os.sched_setaffinity(0, cpus)
    import torch
    import gymnasium as gym
    import domains  # noqa: F401
    from RL.AMBITDMPC2 import AMBITDMPC2

    torch.set_num_threads(len(cpus))
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available(), "Gate requires an allocated CUDA GPU"
    torch.cuda.reset_peak_memory_stats()
    name, manifest, run, overrides = smoke_recipe(args.warmup, args.decisions, args.manifest)
    env = gym.make(run["env"], **manifest["env_params"])
    learner = None
    started = time.perf_counter()
    try:
        learner = AMBITDMPC2(name + "-gpu-gate", env, run["alg_params"], run, manifest)
        cfg, agent = learner.cfg, learner.agent
        validate_learner(cfg, agent)
        learner.buffer.enable_resumable_storage()
        operator, interval = cfg.inner_operator, cfg.inner_solve_interval
        assert args.warmup % interval == args.decisions % interval == 0
        initial_actor = {key: value.detach().clone() for key, value in agent.model._pi.state_dict().items()}
        alpha_before = float(agent.alpha.item())
        timer = TrainingWindow(SEED_STEPS + args.warmup + 2, args.decisions, torch.cuda.synchronize)
        act, update, log_step = learner._act_agent, agent.update, learner._log_wandb_step
        rows, compile_counts, startup, terminal_calls, prior_actions = [], {}, {}, [], []

        def observed_act(*positional, **kwargs):
            decision, step = len(rows), learner._global_step + 1
            if step == timer.first_step:
                torch.cuda.synchronize()
                compile_counts["before"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
                startup["seconds_through_compilation_and_warmup"] = time.perf_counter() - started
            timer.before_action(step, learner._num_updates)
            graphs_before = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
            original_q, original_pi = agent.model.Q, agent.model.pi
            original_policy_action = agent.inner_engine._policy_action

            def checked_q(*q_args, **q_kwargs):
                assert q_kwargs.get("qs", agent.model._Qs) is agent.model._Qs
                assert not q_kwargs.get("target", False) and q_kwargs["reduction"] == "mean_pair"
                terminal_calls.append(True)
                return original_q(*q_args, **q_kwargs)

            def no_entropy_statistics(*unused_args, **unused_kwargs):
                raise AssertionError("MPPI requested policy log probability/entropy")

            def checked_prior(z, policy, **policy_kwargs):
                assert policy is agent.model._pi
                assert policy_kwargs["mode"] == "policy_sample" and not policy_kwargs["inner_bounds"]
                assert policy_kwargs["generator"] is not None and policy_kwargs["std_scale"] == 1.
                sampled, info = original_policy_action(z, policy, **policy_kwargs)
                mean = agent.model.pi_action(z, policy=policy, deterministic=True,
                                            **agent.inner_engine._actor_options)
                assert not torch.equal(sampled, mean), "No-inner action used the deterministic prior mean"
                prior_actions.append(sampled[0].detach().cpu().clone())
                return sampled, info

            audit_mppi = operator == "mppi" and decision == 0
            audit_prior = operator == "none" and decision == 0
            if audit_mppi:
                agent.model.Q, agent.model.pi = checked_q, no_entropy_statistics
            if audit_prior:
                agent.inner_engine._policy_action = checked_prior
            action_started = time.perf_counter()
            try:
                action = act(*positional, **kwargs)
            finally:
                if audit_mppi:
                    agent.model.Q, agent.model.pi = original_q, original_pi
                if audit_prior:
                    agent.inner_engine._policy_action = original_policy_action
            seconds = time.perf_counter() - action_started
            timer.after_action(step, seconds)
            assert action.device.type == "cpu" and torch.isfinite(action).all()
            if audit_prior:
                assert len(prior_actions) == 1
                torch.testing.assert_close(action, prior_actions[0], rtol=0, atol=0)
            solve = validate_action(agent.last_inner_metrics, operator=operator, decision=decision, interval=interval)
            if not solve:
                assert graphs_before == int(torch._dynamo.utils.counters["stats"]["unique_graphs"]), "Non-solving action captured a graph"
            rows.append({"decision": decision, "step": step, "solve": solve, "action_seconds": seconds,
                         "outer_update_enqueue_seconds": 0., "outer_updates": 0})
            return action

        def observed_update(*positional, **kwargs):
            update_started = time.perf_counter()
            metrics = update(*positional, **kwargs)
            seconds = time.perf_counter() - update_started
            assert metrics["compile_fallback"] == 0
            assert not any(key.startswith("aux_return_") for key in metrics)
            if rows and rows[-1]["step"] == learner._global_step:
                rows[-1]["outer_update_enqueue_seconds"] += seconds
                rows[-1]["outer_updates"] += 1
            return metrics

        def observed_log(reward, *positional, **kwargs):
            assert math.isfinite(float(reward))
            result = log_step(reward, *positional, **kwargs)
            timer.after_decision(learner._global_step, learner._num_updates)
            return result

        learner._act_agent, agent.update, learner._log_wandb_step = observed_act, observed_update, observed_log
        with tempfile.TemporaryDirectory(prefix=name + "-checkpoint-") as temporary:
            models = Path(temporary) / "models"
            learner.set_checkpointing(run["total_steps"], str(models), "smoke", save_strat="all")
            learner.learn(run["total_steps"])
            learner.flush_checkpoints()
            torch.cuda.synchronize()
            compile_counts["after"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
            assert compile_counts["before"] == compile_counts["after"], "Compilation inside measured window; increase warmup"
            assert timer.result and len(rows) == args.warmup + args.decisions
            assert all(row["outer_updates"] == 1 for row in rows)
            assert operator != "mppi" or terminal_calls
            assert operator != "none" or prior_actions
            assert all(torch.isfinite(parameter).all() for parameter in agent.model.parameters())
            assert all(torch.isfinite(torch.as_tensor(value)).all() for value in learner._last_train_metrics.values())
            actor_steps, alpha_steps = _optimizer_steps(agent.pi_optim), _optimizer_steps(agent.ent_coef_optim)
            assert actor_steps == alpha_steps == agent.num_updates == learner._num_updates == 2 + len(rows)
            assert any(not torch.equal(value, initial_actor[key]) for key, value in agent.model._pi.state_dict().items())
            alpha_after = float(agent.alpha.item())
            assert math.isfinite(alpha_after) and alpha_after > 0
            checkpoint = models / f"smoke_{run['total_steps']}"
            assert checkpoint.is_file() and Path(str(checkpoint) + ".metadata.json").is_file()
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert "aux_return_state" not in saved
            agent.load(checkpoint)
            for key, value in agent.model.state_dict().items():
                torch.testing.assert_close(value.cpu(), saved["model"][key], rtol=0, atol=0)
            checkpoint_bytes = checkpoint.stat().st_size
            # Validate the agent boundary payload; full replay/lineage resume
            # remains covered by the canonical resume suite and launcher.
            agent.prepare_training_resume_boundary()
            state = deepcopy(agent.training_state_dict())
            agent.load_training_state_dict(state)
            assert agent.training_state_dict()["boundary_prepared"] is True
            assert _optimizer_steps(agent.pi_optim) == actor_steps
            assert _optimizer_steps(agent.ent_coef_optim) == alpha_steps
        assert learner.buffer.resumable_storage and str(learner.buffer._storage_device) == "cpu"
        assert torch.get_num_threads() == len(cpus) and os.sched_getaffinity(0) == set(cpus)
        measured = timer.result
        measured_rows = rows[args.warmup:]
        measured.update(training_phases(measured_rows, operator))
        measured["action_rows"] = measured_rows
        expected_solves = 0 if operator == "none" else args.decisions // interval
        assert measured["solve_decisions"] == expected_solves
        assert measured["held_decisions"] == (0 if operator == "none" else args.decisions - expected_solves)
        storage = learner.buffer._buffer._storage._storage
        replay_bytes_per_row = sum(value.numel() * value.element_size() for value in storage.values()) / len(storage)
        report = {"passed": True, "binding": binding(args.manifest), "runtime": runtime(), "config": name,
                  "operator": operator, "inner_solve_interval": interval, "physical_cores": len(cpus),
                  "logical_cpu_ids": cpus, "torch_threads": torch.get_num_threads(),
                  "measured": measured, "warmup_active_decisions": args.warmup,
                  "compile_graph_counts": compile_counts, "compile_fallback": False,
                  "auxiliary_critic_present": False, "reward_only_targets": True,
                  "primary_critic_sources": True, "outer_actor_entropy_auto": True,
                  "outer_target_entropy": agent.target_entropy,
                  "outer_actor_optimizer_steps": actor_steps, "outer_temperature_optimizer_steps": alpha_steps,
                  "alpha_before": alpha_before, "alpha_after": alpha_after,
                  "mppi_primary_online_mean_pair_no_entropy_verified": bool(terminal_calls),
                  "sampled_prior_execution_verified": bool(prior_actions),
                  "zero_inner_work_verified": operator == "none",
                  "checkpoint_roundtrip": True, "checkpoint_sidecars": True,
                  "boundary_training_state_roundtrip": True, "full_replay_resume_roundtrip": False,
                  "checkpoint_bytes": checkpoint_bytes, "replay_storage_device": "cpu",
                  "replay_bytes_per_row": replay_bytes_per_row,
                  "estimated_production_replay_bytes": replay_bytes_per_row * min(production["total_steps"], production["alg_params"]["buffer_size"]),
                  "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                  ("cold_first_prior_action_seconds" if operator == "none" else "cold_first_solve_seconds"): rows[0]["action_seconds"], "startup": startup,
                  "total_seconds": time.perf_counter() - started,
                  "production_overrides": {**overrides, "total_steps": run["total_steps"], "checkpoint_every": run["total_steps"]},
                  "timing_scope": "Synchronized complete training window. Native action latency may include pending preceding outer GPU work; outer timings are host enqueue latency. Excludes W&B, periodic checkpoint I/O and mature replay.",
                  "created_utc": datetime.now(timezone.utc).isoformat()}
        write_new(args.output, report)
        print("REWARD_TRAINING_CASE_PASSED " + json.dumps({"config": name, "cores": len(cpus),
              "measured": {k: v for k, v in measured.items() if not isinstance(v, list)}}), flush=True)
    finally:
        if learner is not None:
            learner.close()
        env.close()


def validate_receipt(receipt, campaign_binding, current_runtime, cpus, *, name, operator, interval):
    assert receipt["passed"] is True and receipt["binding"] == campaign_binding, "Gate source/configuration mismatch"
    counts = cpu_counts(operator)
    assert cpus in counts, "Production CPU allocation was not measured"
    assert receipt["runtime"] == current_runtime, "Gate used another runtime or GPU"
    cases = receipt["cases"]
    assert len(cases) == len(counts) and {row["physical_cores"] for row in cases} == set(counts)
    for row in cases:
        assert row["passed"] is True and row["binding"] == campaign_binding and row["runtime"] == current_runtime
        assert (row["config"], row["operator"], row["inner_solve_interval"]) == (name, operator, interval)
        assert row["compile_fallback"] is False and row["replay_storage_device"] == "cpu"
        assert row["compile_graph_counts"]["before"] == row["compile_graph_counts"]["after"]
        assert row["auxiliary_critic_present"] is False
        for key in ("reward_only_targets", "primary_critic_sources", "outer_actor_entropy_auto",
                    "checkpoint_roundtrip", "checkpoint_sidecars", "boundary_training_state_roundtrip"):
            assert row[key] is True, key
        assert row["outer_target_entropy"] == -10.5
        measured = row["measured"]
        assert measured["decisions"] >= 100 and measured["decisions"] == measured["outer_updates"]
        if operator == "none":
            assert measured["decisions"] >= 400 and row["warmup_active_decisions"] >= 24
            assert measured["solve_decisions"] == measured["held_decisions"] == 0
            assert measured["prior_decisions"] == measured["decisions"]
            assert row["sampled_prior_execution_verified"] is row["zero_inner_work_verified"] is True
        else:
            assert measured["solve_decisions"] == measured["decisions"] // interval
            assert measured["held_decisions"] == measured["decisions"] - measured["solve_decisions"]
        assert row["outer_actor_optimizer_steps"] == row["outer_temperature_optimizer_steps"] == 2 + row["warmup_active_decisions"] + measured["decisions"]
        assert math.isfinite(measured["seconds_per_decision"]) and measured["seconds_per_decision"] > 0
        if operator == "mppi":
            assert row["mppi_primary_online_mean_pair_no_entropy_verified"] is True


def smoke(args):
    assert os.environ.get("SLURM_JOB_ID")
    name, _, run = validate_campaign(args.manifest)
    operator = run["alg_params"]["inner_operator"]
    counts = cpu_counts(operator)
    assert int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) == max(counts)
    interval = run["alg_params"].get("inner_solve_interval", 1)
    cores = physical_core_cpus(os.sched_getaffinity(0))
    assert len(cores) >= max(counts), f"Request {max(counts)} physical CPUs with --hint=nomultithread"
    output = args.output.resolve()
    assert output != ROOT and ROOT not in output.parents
    output.mkdir(parents=True, exist_ok=False)
    order = list(counts)
    random.Random(55).shuffle(order)
    write_new(output / "protocol.json", {"binding": binding(args.manifest), "config": name,
              "case_order": order, "physical_core_representatives": cores,
              "warmup_active_decisions": args.warmup, "measured_decisions": args.decisions,
              "slurm_job_id": os.environ["SLURM_JOB_ID"]})
    with tempfile.TemporaryDirectory(prefix="ambi-reward-gate-") as temporary:
        cache = Path(temporary)
        for count in order:
            environment = dict(os.environ)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                environment[key] = str(count)
            environment.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE", WANDB_MODE="disabled",
                               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", MUJOCO_GL="egl",
                               TORCHINDUCTOR_COMPILE_THREADS="1", TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
                               TRITON_CACHE_DIR=str(cache / "triton"))
            command = [sys.executable, str(Path(__file__).resolve()), "smoke", "--worker-cpus", ",".join(map(str, cores[:count])),
                       "--manifest", str(resolve_manifest(args.manifest)), "--warmup", str(args.warmup),
                       "--decisions", str(args.decisions), "--output", str(output / f"cores-{count}.json")]
            with (output / f"cores-{count}.log").open("x") as log:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f"REWARD_TRAINING_GATE_PROGRESS cores={count}", flush=True)
    paths = [output / f"cores-{count}.json" for count in counts]
    cases = [read(path) for path in paths]
    receipt = {"passed": True, "binding": binding(args.manifest), "runtime": cases[0]["runtime"], "cases": cases,
               "artifacts": {str(path): digest(path) for path in paths}, "created_utc": datetime.now(timezone.utc).isoformat()}
    validate_receipt(receipt, receipt["binding"], receipt["runtime"], 1, name=name, operator=operator, interval=interval)
    fastest = min(row["measured"]["seconds_per_decision"] for row in cases)
    receipt["recommended_cpus_within_5pct_of_best"] = min(row["physical_cores"] for row in cases
        if row["measured"]["seconds_per_decision"] <= fastest / .95)
    write_new(output / "receipt.json", receipt)
    print("REWARD_TRAINING_GATE_PASSED " + str(output / "receipt.json"), flush=True)


def launch(args):
    name, _, run = validate_campaign(args.manifest)
    current_binding, current_runtime = binding(args.manifest), runtime()
    receipt = read(args.gate)
    validate_receipt(receipt, current_binding, current_runtime, int(os.environ["SLURM_CPUS_PER_TASK"]),
                     name=name, operator=run["alg_params"]["inner_operator"], interval=run["alg_params"].get("inner_solve_interval", 1))
    for path, checksum in receipt["artifacts"].items():
        assert digest(path) == checksum, "Gate artifact changed"
    assert os.environ["AMBI_DURABLE_ROOT"] == "/oscar/home/rgao48/ambi-durable"
    lineage = Path(os.environ["AMBI_LINEAGE_DIR"])
    assert lineage.is_absolute() and lineage != Path(os.environ["AMBI_DURABLE_ROOT"])
    lineage.resolve().relative_to(Path(os.environ["AMBI_DURABLE_ROOT"]).resolve())
    assert int(os.environ["AMBI_DRAIN_AFTER_SECONDS"]) > 0
    segment = os.environ["SLURM_JOB_ID"] + "." + os.environ.get("SLURM_RESTART_COUNT", "0")
    args.output.mkdir(parents=True, exist_ok=True)
    write_new(args.output / f"launch-{segment}.json", {
        "binding": current_binding, "config": name, "run_params": run, "runtime": current_runtime,
        "gate_receipt": str(args.gate), "gate_sha256": digest(args.gate), "lineage_dir": str(lineage),
        "segment": segment, "resume_mode": "required" if os.environ.get("SLURM_RESTART_COUNT", "0") != "0" else os.environ.get("AMBI_RESUME_MODE", "new"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    os.environ.update(AMBI_RUN_CONFIG=str(resolve_manifest(args.manifest)), AMBI_ALG_DIR=str(ROOT / "configs/dmcontrol/algs"),
                      AMBI_PYTHON=sys.executable, SLURM_SUBMIT_DIR=str(ROOT))
    os.execv("/bin/bash", ["/bin/bash", str(ROOT / "run_ambi_oscar.sh")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "train"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--decisions", type=int)
    parser.add_argument("--worker-cpus", help=argparse.SUPPRESS)
    args = parser.parse_args()
    assert binding(args.manifest)["commit"] == os.environ["AMBI_EXPECTED_COMMIT"]
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True).strip()
    if args.mode == "smoke":
        operator = validate_campaign(args.manifest)[2]["alg_params"]["inner_operator"]
        warmup, decisions = smoke_defaults(operator)
        args.warmup = warmup if args.warmup is None else args.warmup
        args.decisions = decisions if args.decisions is None else args.decisions
        assert args.warmup >= 12 and args.decisions >= 100
        assert SEED_STEPS + 1 + args.warmup + args.decisions < 1000
        (worker if args.worker_cpus else smoke)(args)
    else:
        assert args.gate is not None and not args.worker_cpus
        launch(args)


if __name__ == "__main__":
    main()
