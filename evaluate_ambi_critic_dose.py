"""800k full-episode critic-dose experiment with fresh H1/J6 SAC at each root.

Only C changes (32 or 64 updates per round) and execution selects the actor
mean or learned-Q bank maximum. Earlier C16 results are references, never new
tasks. Common named random streams do not imply identical replay banks across
critic doses: extra updates change later actors and imagined action sampling.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time

import numpy as np
import torch

from evaluate_ambi_action_audit import atomic_json, resolved_base
from evaluate_ambi_checkpoint import (
    _close_resources, _file_sha256, _initialize_frozen_model, _make_env,
    _outer_state_digest, _seed_spaces,
)
from evaluate_ambi_critic_bypass import fingerprint, timing_summary
from evaluate_ambi_transfer_diagnostics import resolved_setting
from utils.ambi_benchmark import solver_seed
from utils.critic_bypass import run_bypass_decision, validate_bypass_controller
from utils.matched_action_audit import score_action_bank
from utils.transfer_diagnostics import Reference

PROTOCOL = "closed-loop-critic-dose-v1"
ARM_SETTINGS = {
    "actor_mean_c32": ("actor_mean", 32), "learned_q_c32": ("learned_q", 32),
    "actor_mean_c64": ("actor_mean", 64), "learned_q_c64": ("learned_q", 64),
}
ARMS = tuple(ARM_SETTINGS)
DIAGNOSTIC_DECISIONS = [0, 25, 100, 250, 400, 499]


def load_config(path, *, smoke=False):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate configuration key: {key}")
            result[key] = value
        return result
    config = json.loads(Path(path).read_text(), object_pairs_hook=unique)
    expected = dict(protocol=PROTOCOL, arms=list(ARMS), seeds=list(range(101, 121)),
        controller_seed=55, max_steps=500, horizon=1, rounds=6, rollouts_per_round=128,
        actor_updates_per_round=4, batch_size=256, critic_updates_per_round=[32, 64],
        mc_rollouts=32, validation_rollouts=32, diagnostic_decisions=DIAGNOSTIC_DECISIONS)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Configuration differs from the authorized 800k critic-dose protocol.")
    if (config["checkpoint"].get("step") != 800000 or any(not re.fullmatch(r"[0-9a-f]{64}",
            str(config["checkpoint"].get(key, ""))) for key in ("sha256", "metadata_sha256"))):
        raise ValueError("Expected pinned 800k checkpoint and metadata hashes.")
    for key in ("reference_campaign_id", "reference_snapshot_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(config.get(key, ""))):
            raise ValueError(f"Verified C16 reference identity required: {key}")
    if type(config.get("max_expanded_batch")) is not int or config["max_expanded_batch"] < 32:
        raise ValueError("Invalid model scoring batch bound.")
    config["smoke"] = bool(smoke)
    if smoke:
        config.update(seeds=[101], max_steps=4, diagnostic_decisions=[0, 3])
    return config


def task_list(config):
    # Begin the more expensive cells early and interleave doses/selectors.
    order = ("learned_q_c64", "actor_mean_c64", "learned_q_c32", "actor_mean_c32")
    return [dict(task_id=f"{arm}-seed-{seed}", arm=arm, selector=ARM_SETTINGS[arm][0],
                 critic_updates_per_round=ARM_SETTINGS[arm][1], env_seed=seed,
                 controller_seed=config["controller_seed"])
            for seed in config["seeds"] for arm in order]


def resolved_dose(base, dose):
    if isinstance(dose, bool) or dose not in (32, 64):
        raise ValueError("New critic-dose tasks must use C32 or C64.")
    result = resolved_setting(base, 1, 6)
    result["algorithm_config"]["alg_params"]["inner_critic_updates_per_round"] = dose
    return result


def campaign_identity(config, checkpoint, config_path):
    source = Path(__file__).resolve().parent
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD^{tree}"], text=True).strip()
    result = {**deepcopy(config), "config": deepcopy(config), "source_commit": commit,
              "source_tree": tree, "config_sha256": _file_sha256(config_path),
              "checkpoint_path": str(checkpoint), "task_list": task_list(config)}
    result["campaign_id"] = fingerprint(result)
    return result


def root_seeds(controller_seed, env_seed, decision):
    # Deliberately identical to the completed C16 campaign's derivation.
    episode_seed = solver_seed(controller_seed, "episode", env_seed)
    root = solver_seed(episode_seed, "root", decision)
    return {f"{stream}_seed": solver_seed(root, stream)
            for stream in ("solve", "selection", "validation")}


def smoke_validate(reference, wrapped, observation, task, output, options):
    """Check the actual GPU path, dose accounting and independent audit scorer."""
    repeated = run_bypass_decision(wrapped, observation, arm=task["selector"],
        critic_updates_per_round=task["critic_updates_per_round"], shadow=False, **options)
    if not np.array_equal(output["action"], repeated["action"]):
        raise RuntimeError("Enabling shadow diagnostics changed the executed action.")
    if output["work"] != repeated["work"]:
        raise RuntimeError("Shadow diagnostics changed realized learner work.")
    choices = output["diagnostics"]["choices"]
    labels = list(choices)
    audited = score_action_bank(reference, observation,
        actions=[choices[label]["normalized_action"] for label in labels], labels=labels,
        critics={"final": reference.engine._action_pool.critic},
        seed=options["validation_seed"], mc_rollouts=32, horizons=(1,))
    for row in audited["actions"]:
        if not np.allclose(row["model"]["h1"]["draws"],
                choices[row["label"]]["validation_model"]["draws"], atol=5e-5, rtol=2e-6):
            raise RuntimeError("Optimized model scoring differs from the existing audit scorer.")
        if not np.isclose(row["q"]["final"], choices[row["label"]]["learned_q"],
                          atol=5e-5, rtol=2e-6):
            raise RuntimeError("Learned-Q scoring differs from the existing audit scorer.")
    return dict(shadow_action_invariance=True, model_score_audit_parity=True,
                learned_q_audit_parity=True)


def run(args):
    config = load_config(args.config, smoke=args.smoke)
    checkpoint = args.checkpoint.resolve()
    if _file_sha256(checkpoint) != config["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint hash differs from authorized 800k checkpoint.")
    if _file_sha256(Path(str(checkpoint) + ".metadata.json")) != config["checkpoint"]["metadata_sha256"]:
        raise ValueError("Checkpoint metadata hash mismatch.")
    identity = campaign_identity(config, checkpoint, args.config)
    campaign = args.campaign_root.resolve()
    base = resolved_base(config, checkpoint)
    if args.prepare:
        campaign.mkdir(parents=True, exist_ok=False)
        atomic_json(campaign / "campaign.json", identity)
        atomic_json(campaign / "resolved.json", {f"c{dose}": resolved_dose(base, dose) for dose in (32, 64)})
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
        resolved = resolved_dose(base, task["critic_updates_per_round"])
        env = _make_env(resolved)
        wrapped, _ = _initialize_frozen_model(resolved, env, checkpoint, config["controller_seed"], device=args.device)
        validate_bypass_controller(wrapped, critic_updates_per_round=task["critic_updates_per_round"])
        before = _outer_state_digest(wrapped)
        reference = Reference(wrapped)
        _seed_spaces(env, task["env_seed"])
        observation, _ = env.reset(seed=task["env_seed"])
        episode_seed = solver_seed(config["controller_seed"], "episode", task["env_seed"])
        episode_return = 0.0
        control_times, selection_times, diagnostic_times, diagnostics, actions = [], [], [], [], []
        expected_work = dict(actor_updates=24, critic_updates=6 * task["critic_updates_per_round"], model_steps=768)
        work_totals = dict.fromkeys(expected_work, 0)
        smoke_checks = {}
        terminated = truncated = False
        for decision in range(config["max_steps"]):
            if decision % 10 == 0:
                atomic_json(task_dir / "progress.json", dict(**task, status="running", decision=decision,
                    episode_return=episode_return, elapsed_seconds=time.monotonic()-started))
            options = dict(**root_seeds(config["controller_seed"], task["env_seed"], decision),
                mc_rollouts=config["mc_rollouts"], max_expanded_batch=config["max_expanded_batch"])
            output = run_bypass_decision(wrapped, observation, arm=task["selector"],
                critic_updates_per_round=task["critic_updates_per_round"],
                shadow=decision in config["diagnostic_decisions"], **options)
            if output["work"] != expected_work:
                raise RuntimeError("The learner did not execute the requested critic dose.")
            for key in work_totals:
                work_totals[key] += output["work"][key]
            if config["smoke"] and decision == 0:
                smoke_checks.update(smoke_validate(reference, wrapped, observation, task, output, options))
            timing = output["timing"]
            control_times.append(float(timing["control_seconds"]))
            selection_times.append(float(timing["selection_seconds"]))
            diagnostic_times.append(float(timing["diagnostics_seconds"]))
            if output["diagnostics"] is not None:
                diagnostics.append(dict(decision=decision, **output["diagnostics"]))
            if output["bank_count"] != 770 or output["replay_count"] != 768:
                raise RuntimeError("SAC candidate bank has missing or extra actions.")
            action = np.asarray(output["action"], dtype=np.float32)
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
        if [row["decision"] for row in diagnostics] != [d for d in config["diagnostic_decisions"] if d < steps]:
            raise RuntimeError("Missing same-state diagnostics.")
        control_mean, control_p95 = timing_summary(control_times)
        select_mean, select_p95 = timing_summary(selection_times)
        result = dict(**{key: value for key, value in manifest.items() if key != "status"}, complete=True,
            episode_return=episode_return, episode_length=steps, terminated=bool(terminated),
            truncated=bool(truncated), evaluator_capped=bool(not (terminated or truncated)),
            controller_time_mean_s=control_mean, controller_time_p95_s=control_p95,
            selection_time_mean_s=select_mean, selection_time_p95_s=select_p95,
            controller_times_s=control_times, selection_times_s=selection_times,
            diagnostic_times_s=diagnostic_times, diagnostics=diagnostics,
            expected_work_per_decision=expected_work, realized_work_totals=work_totals,
            elapsed_seconds=time.monotonic()-started,
            checks=dict(outer_state_unchanged=True, full_episode=bool(terminated or truncated or steps == config["max_steps"]),
                finite_returns=True, action_bounds=True, complete_candidate_banks=True,
                critic_update_dose_verified=True, diagnostic_coverage=True,
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
    parser.add_argument("--config", type=Path, default=Path("configs/research/ambi_800k_critic_dose.json"))
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
