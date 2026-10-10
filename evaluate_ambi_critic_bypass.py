"""Full-episode 800k evaluation of three selectors on a fixed H1/J6 SAC solve.

Each independently owned task is one arm and environment seed. Selection and
held-out shadow diagnostics have named streams separate from SAC learning.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from evaluate_ambi_action_audit import atomic_json, policy_bank, resolved_base
from evaluate_ambi_checkpoint import (
    _close_resources, _file_sha256, _initialize_frozen_model, _make_env,
    _outer_state_digest, _seed_spaces,
)
from evaluate_ambi_transfer_diagnostics import resolved_setting
from utils.ambi_benchmark import solver_seed
from utils.matched_action_audit import mppi_candidate, score_action_bank
from utils.transfer_diagnostics import Reference, validate_controller

PROTOCOL = "closed-loop-critic-bypass-v1"
ARMS = ("actor_mean", "learned_q", "model_score", "prior", "mppi_h3")


def load_config(path, *, smoke=False):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate configuration key: {key}")
            result[key] = value
        return result
    config = json.loads(Path(path).read_text(), object_pairs_hook=unique)
    if (config["protocol"] != PROTOCOL or config["arms"] != list(ARMS)
            or config["seeds"] != list(range(101, 121)) or config["controller_seed"] != 55
            or config["max_steps"] != 500 or config["horizon"] != 1 or config["rounds"] != 6
            or config["mc_rollouts"] != 32 or config["validation_rollouts"] != 32):
        raise ValueError("Configuration differs from the authorized 800k H1/J6 protocol.")
    if (config["checkpoint"]["step"] != 800000
            or any(len(config["checkpoint"][key]) != 64 for key in ("sha256", "metadata_sha256"))):
        raise ValueError("Expected pinned 800k checkpoint and metadata hashes.")
    decisions = config["diagnostic_decisions"]
    if (not decisions or len(set(decisions)) != len(decisions)
            or any(type(value) is not int or not 0 <= value < 500 for value in decisions)):
        raise ValueError("Invalid diagnostic decisions.")
    if type(config["max_expanded_batch"]) is not int or config["max_expanded_batch"] < 32:
        raise ValueError("Invalid model scoring batch bound.")
    config["smoke"] = bool(smoke)
    if smoke:
        config.update(seeds=[101], max_steps=4, diagnostic_decisions=[0, 3])
    return config


def task_list(config):
    # Interleave arms so the more expensive direct-model jobs do not all start
    # after the other methods. Stable identities do not depend on task indices.
    scheduling_order = ("model_score", "learned_q", "actor_mean", "mppi_h3", "prior")
    return [dict(task_id=f"{arm}-seed-{seed}", arm=arm, env_seed=seed,
                 controller_seed=config["controller_seed"])
            for seed in config["seeds"] for arm in scheduling_order]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def campaign_identity(config, checkpoint, config_path):
    source = Path(__file__).resolve().parent
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD^{tree}"], text=True).strip()
    result = {**deepcopy(config), "config": deepcopy(config), "source_commit": commit,
              "source_tree": tree, "config_sha256": _file_sha256(config_path),
              "checkpoint_path": str(checkpoint), "task_list": task_list(config)}
    result["campaign_id"] = fingerprint(result)
    return result


def synchronized_time(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def timing_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid controller timing samples.")
    return float(values.mean()), float(np.percentile(values, 95))


def run(args):
    from utils.critic_bypass import run_bypass_decision
    config = load_config(args.config, smoke=args.smoke)
    checkpoint = args.checkpoint.resolve()
    if _file_sha256(checkpoint) != config["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint hash differs from authorized 800k checkpoint.")
    if _file_sha256(Path(str(checkpoint) + ".metadata.json")) != config["checkpoint"]["metadata_sha256"]:
        raise ValueError("Checkpoint metadata hash mismatch.")
    identity = campaign_identity(config, checkpoint, args.config)
    campaign = args.campaign_root.resolve()
    base = resolved_setting(resolved_base(config, checkpoint), 1, 6)
    if args.prepare:
        campaign.mkdir(parents=True, exist_ok=False)
        atomic_json(campaign / "campaign.json", identity)
        atomic_json(campaign / "resolved.json", base)
        print(json.dumps(dict(campaign_id=identity["campaign_id"], tasks=len(identity["task_list"]))))
        return
    if json.loads((campaign / "campaign.json").read_text()) != identity:
        raise ValueError("Worker configuration/source/checkpoint differs from prepared campaign.")
    if args.task_index is None or not 0 <= args.task_index < len(identity["task_list"]):
        raise ValueError("Valid task-index required.")
    task = identity["task_list"][args.task_index]
    task_dir = campaign / "tasks" / task["task_id"]
    task_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        device=str(device), hostname=platform.node(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
        execution="eager", timing="synchronized wall time; shadow scoring excluded")
    manifest = dict(schema_version=1, protocol=PROTOCOL, campaign_id=identity["campaign_id"],
                    **task, runtime=runtime, status="running", source_commit=identity["source_commit"])
    atomic_json(task_dir / "manifest.started.json", manifest)
    env = wrapped = None
    started = time.monotonic()
    try:
        env = _make_env(base)
        wrapped, _ = _initialize_frozen_model(base, env, checkpoint, config["controller_seed"], device=args.device)
        validate_controller(wrapped)
        before = _outer_state_digest(wrapped)
        reference = Reference(wrapped)
        _seed_spaces(env, task["env_seed"])
        observation, _ = env.reset(seed=task["env_seed"])
        episode_seed = solver_seed(config["controller_seed"], "episode", task["env_seed"])
        episode_return, previous_mean = 0.0, None
        control_times, selection_times, diagnostic_times, diagnostics, actions = [], [], [], [], []
        smoke_checks = {}
        terminated = truncated = False
        for decision in range(config["max_steps"]):
            if decision % 10 == 0:
                atomic_json(task_dir / "progress.json", dict(**task, status="running", decision=decision,
                    episode_return=episode_return, elapsed_seconds=time.monotonic()-started))
            if task["arm"] in ARMS[:3]:
                root_seed = solver_seed(episode_seed, "root", decision)
                output = run_bypass_decision(wrapped, observation, arm=task["arm"],
                    solve_seed=solver_seed(root_seed, "solve"), selection_seed=solver_seed(root_seed, "selection"),
                    validation_seed=solver_seed(root_seed, "validation"),
                    shadow=decision in config["diagnostic_decisions"], mc_rollouts=config["mc_rollouts"],
                    max_expanded_batch=config["max_expanded_batch"])
                if config["smoke"] and decision == 0:
                    without_shadow = run_bypass_decision(wrapped, observation, arm=task["arm"],
                        solve_seed=solver_seed(root_seed, "solve"), selection_seed=solver_seed(root_seed, "selection"),
                        validation_seed=solver_seed(root_seed, "validation"), shadow=False,
                        mc_rollouts=config["mc_rollouts"], max_expanded_batch=config["max_expanded_batch"])
                    if not np.array_equal(output["action"], without_shadow["action"]):
                        raise RuntimeError("Enabling shadow diagnostics changed the executed action.")
                    smoke_checks["shadow_action_invariance"] = True
                    choices = output["diagnostics"]["choices"]
                    labels = list(choices)
                    audited = score_action_bank(reference, observation,
                        actions=[choices[label]["normalized_action"] for label in labels], labels=labels,
                        critics={"final": reference.engine._action_pool.critic},
                        seed=solver_seed(root_seed, "validation"), mc_rollouts=32, horizons=(1,))
                    for row in audited["actions"]:
                        if not np.allclose(row["model"]["h1"]["draws"],
                                choices[row["label"]]["validation_model"]["draws"], atol=5e-5, rtol=2e-6):
                            raise RuntimeError("Optimized model scoring differs from the existing audit scorer.")
                    smoke_checks["model_score_audit_parity"] = True
                action = output["action"]
                timing = output["timing"]
                control_times.append(float(timing["control_seconds"]))
                selection_times.append(float(timing["selection_seconds"]))
                diagnostic_times.append(float(timing["diagnostics_seconds"]))
                if output["diagnostics"] is not None:
                    diagnostics.append(dict(decision=decision, **output["diagnostics"]))
                if output["bank_count"] != 770 or output["replay_count"] != 768:
                    raise RuntimeError("SAC candidate bank has missing or extra actions.")
            else:
                start = synchronized_time(device)
                if task["arm"] == "prior":
                    normalized_action = wrapped.agent.act_outer_policy(
                        torch.as_tensor(observation, dtype=torch.float32), deterministic=True).numpy()
                else:
                    planned = mppi_candidate(reference, observation, horizon=3,
                        seed=solver_seed(episode_seed, "source-mppi", decision),
                        planner_options=config["mppi"], previous_mean=previous_mean)
                    previous_mean = torch.as_tensor(planned["proposal_mean"], device=device)
                    normalized_action = np.asarray(planned["normalized_action"], dtype=np.float32)
                action = wrapped._unscale_action(normalized_action)
                control_times.append(synchronized_time(device)-start)
                selection_times.append(0.0)
                diagnostic_times.append(0.0)
                if config["smoke"] and task["arm"] == "prior":
                    mean, _, _ = policy_bank(reference, reference.engine._actor_base, observation,
                        seed=0, count=1, bounds=reference.engine._actor_options)
                    if not np.array_equal(normalized_action, mean[0].cpu().numpy()):
                        raise RuntimeError("Prior action differs between standard execution and audit paths.")
                    smoke_checks["prior_execution_path_parity"] = True
            action = np.asarray(action, dtype=np.float32)
            if not np.isfinite(action).all() or not env.action_space.contains(action):
                raise RuntimeError("Controller returned an invalid environment action.")
            actions.append(action.tolist())
            observation, reward, terminated, truncated, _ = env.step(action)
            if not np.isfinite(float(reward)):
                raise RuntimeError("Environment produced a nonfinite reward.")
            episode_return += float(reward)
            if terminated or truncated:
                break
        steps = len(control_times)
        if _outer_state_digest(wrapped) != before:
            raise RuntimeError("Evaluation modified frozen outer state.")
        control_mean, control_p95 = timing_summary(control_times)
        select_mean, select_p95 = timing_summary(selection_times)
        result = dict(**{key: value for key, value in manifest.items() if key != "status"}, complete=True,
            episode_return=episode_return, episode_length=steps, terminated=bool(terminated),
            truncated=bool(truncated), evaluator_capped=bool(not (terminated or truncated)),
            controller_time_mean_s=control_mean, controller_time_p95_s=control_p95,
            selection_time_mean_s=select_mean, selection_time_p95_s=select_p95,
            controller_times_s=control_times, selection_times_s=selection_times,
            diagnostic_times_s=diagnostic_times, diagnostics=diagnostics,
            elapsed_seconds=time.monotonic()-started,
            checks=dict(outer_state_unchanged=True, full_episode=bool(terminated or truncated or steps == config["max_steps"]),
                finite_returns=True, action_bounds=True, complete_candidate_banks=True,
                independent_selection_validation=True, **smoke_checks),
            action_sha256=fingerprint(actions), solver_seed=episode_seed)
        atomic_json(task_dir / "actions.json", actions)
        atomic_json(task_dir / "result.json", result)
        atomic_json(task_dir / "manifest.json", {**manifest, "status": "complete", "result_sha256": _file_sha256(task_dir/"result.json")})
        atomic_json(task_dir / "progress.json", dict(**task, status="complete", decision=steps,
                    episode_return=episode_return, elapsed_seconds=time.monotonic()-started))
        print(json.dumps(dict(**task, episode_return=episode_return, episode_length=steps,
                              control_seconds=control_mean*steps, status="complete")))
    except BaseException as error:
        atomic_json(task_dir / "progress.json", dict(**task, status="failed", error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        _close_resources(wrapped, env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/research/ambi_800k_critic_bypass.json"))
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
