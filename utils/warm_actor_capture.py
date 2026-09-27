"""Immutable warm-history roots and matched actor interventions.

This runner never trains the frozen backbone. Source episodes use the original
actor-transfer-v2 RNG and solve schedule. Cold solves run in another controller
with a copy of the warm solve's pre-action private RNG. Captures are trusted
local artifacts: actor modules use Python/PyTorch serialization.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from RL.tdmpc2_core.inner_trace import FrozenActorSnapshot, InnerActionTrace
from utils.ambi_benchmark import atomic_json, solver_seed
from utils.ambi_real_calibration import (
    SimulatorSnapshot, capture_simulator_snapshot, restore_simulator_snapshot,
)


DEFAULT_DECISIONS = (25, 75, 150, 250, 350, 450)
DEFAULT_ROUNDS = (0, 1, 2, 4, 6, 8, 10)
PROTOCOL = "warm-actor-branch-calibration-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bytes(path, payload):
    """Exclusive creation: an incomplete/retried run never overwrites evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def save_actor_snapshot(snapshot, directory, family):
    filename = f"{family}-round-{snapshot.round_index}.pt"
    write_bytes(Path(directory) / filename, snapshot.payload)
    return {
        "family": family, "round": snapshot.round_index,
        "actor_updates": snapshot.actor_updates,
        "critic_updates": snapshot.critic_updates,
        "temperature_updates": snapshot.temperature_updates,
        "inner": snapshot.inner, "bounds": snapshot.policy_bounds,
        "path": filename, "sha256": snapshot.sha256,
    }


def load_actor_snapshot(root_path, actor):
    root_path = Path(root_path).resolve()
    path = (root_path.parent / actor["path"]).resolve()
    if not path.is_relative_to(root_path.parent):
        raise ValueError("Actor path must stay within the captured root directory.")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != actor["sha256"]:
        raise ValueError("Actor snapshot checksum mismatch.")
    return FrozenActorSnapshot(
        round_index=int(actor["round"]), actor_updates=int(actor["actor_updates"]),
        critic_updates=int(actor["critic_updates"]),
        temperature_updates=int(actor["temperature_updates"]),
        inner=bool(actor["inner"]), bounds=tuple(actor["bounds"].items()), payload=payload,
    )


def prior_snapshot(model):
    engine = model.agent.inner_engine
    policy = copy.deepcopy(engine._actor_base).cpu().eval().requires_grad_(False)
    output = io.BytesIO()
    torch.save(policy, output)
    bounds = {name: getattr(model.cfg, name) for name in (
        "log_std_mapping", "log_std_min", "log_std_max")}
    bounds.update(engine._actor_options)
    return FrozenActorSnapshot(0, 0, 0, 0, False, tuple(bounds.items()), output.getvalue())


def resolve_source(matrix_path, checkpoint, selector, metadata_path=None):
    from utils.ambi_research import load_preset_matrix, resolve_preset
    from utils.checkpoint_context import load_checkpoint_context

    matrix = load_preset_matrix(matrix_path)
    context = (load_checkpoint_context(checkpoint, metadata_path=metadata_path)
               if matrix["base_alg_config"] == "checkpoint" else None)
    return matrix, resolve_preset(matrix_path, selector, matrix=matrix,
                                  checkpoint_context=context)


def validate_source(cfg):
    expected = {
        "inner_operator": "sac", "inner_actor_scope": "episode",
        "inner_critic_scope": "action", "inner_temperature_scope": "action",
        "inner_replay_scope": "action", "inner_actor_optimizer_scope": "action",
        "inner_critic_optimizer_scope": "action", "inner_temperature_optimizer_scope": "action",
        "inner_actor_adaptation": "clone", "inner_critic_adaptation": "clone",
        "inner_first_action_rounds": None, "inner_eval_execution_action": "mean",
        "inner_actor_writeback_coef": 0., "inner_critic_writeback_coef": 0.,
    }
    mismatch = {key: (getattr(cfg, key, None), value) for key, value in expected.items()
                if getattr(cfg, key, None) != value}
    if mismatch:
        raise ValueError(f"Source must be actor-only, uniform-J transfer: {mismatch}")
    if getattr(cfg, "obs", "state") != "state":
        raise ValueError("Warm branch capture currently supports state observations only.")


def make_trace(model, controller_seed, seed, decision, *, capture=False, rollouts=32):
    rounds = [r for r in DEFAULT_ROUNDS if r <= int(model.cfg.inner_rounds)]
    return InnerActionTrace(
        probes=True, probe_mode="outer_tail", transfer_probes=True,
        probe_rollouts=rollouts, probe_horizon=int(model.cfg.inner_rollout_horizon),
        probe_seed=solver_seed(controller_seed, "togo_probe", seed, decision),
        capture_actors=capture, actor_rounds=rounds,
    )


def captured_cold_solve(model, observation, rng_state, *, controller_seed, seed,
                        decision, rollouts=32):
    """Fresh actor/critic/Adam solve with the warm pre-solve random streams."""
    engine = model.agent.inner_engine
    engine.reset_for_evaluation(0, reuse_action_pool=True)
    engine.rng.load_training_state_dict(rng_state)
    trace = make_trace(model, controller_seed, seed, decision, capture=True, rollouts=rollouts)
    action, _ = model.predict(observation, deterministic=True, episode_start=True, trace=trace)
    return action, trace


def completed_capture(output, *, checkpoint, matrix_path, resolved, selector, seed,
                      controller_seed, max_steps, decisions, device):
    """Reuse only a fully verified immutable source episode; never partial state."""
    output = Path(output)
    path = output / "manifest.json"
    if not path.is_file():
        raise FileExistsError(f"Capture directory exists without a complete manifest: {output}")
    manifest = json.loads(path.read_text())
    expected = {
        "status": "complete", "protocol": PROTOCOL, "selector": selector,
        "seed": int(seed), "controller_seed": int(controller_seed),
        "episode_max_steps": int(max_steps), "decisions": list(decisions),
        "checkpoint_sha256": sha256_file(checkpoint),
        "matrix_sha256": sha256_file(matrix_path), "resolved_preset": resolved,
        "captured_roots": [f"decision-{decision}/root.json" for decision in sorted(decisions)],
    }
    mismatch = [key for key, value in expected.items() if manifest.get(key) != value]
    if device is not None and str(manifest.get("resolved_config", {}).get("device")) != str(device):
        mismatch.append("device")
    if mismatch:
        raise FileExistsError(f"Existing capture cannot be resumed; identity differs or incomplete: {mismatch}")
    for field in ("source_trace", "source_trajectory"):
        artifact = output / manifest[field]
        if not artifact.is_file() or sha256_file(artifact) != manifest.get(field + "_sha256"):
            raise ValueError(f"Completed capture {field} is missing or has changed.")
    for relative in manifest["captured_roots"]:
        root_path = output / relative
        root = json.loads(root_path.read_text())
        SimulatorSnapshot.from_dict(root["snapshot"])
        if root["checkpoint_sha256"] != expected["checkpoint_sha256"] or root["matrix_sha256"] != expected["matrix_sha256"]:
            raise ValueError("Completed root identity differs from its source manifest.")
        for actor in root["actors"]:
            load_actor_snapshot(root_path, actor)  # Verifies bytes without deserializing modules.
        if sha256_file(root_path.parent / root["learner_rng_path"]) != root["learner_rng_sha256"]:
            raise ValueError("Completed root learner RNG checksum mismatch.")
    return {**manifest, "reused_complete_capture": True}


def capture_episode(*, checkpoint, matrix_path, selector, seed, output,
                    decisions=DEFAULT_DECISIONS, controller_seed=None,
                    max_steps=500, device=None, metadata_path=None):
    """Run one original warm episode and persist immutable, branchable roots."""
    from evaluate_ambi_checkpoint import (
        _close_resources, _initialize_frozen_model, _jsonable, _make_env,
        _outer_state_digest, _seed_spaces,
    )

    decisions = tuple(int(value) for value in decisions)
    if len(set(decisions)) != len(decisions) or any(d < 0 or d >= max_steps for d in decisions):
        raise ValueError("Capture decisions must be unique and within the episode cap.")
    output = Path(output).resolve()
    matrix, resolved = resolve_source(matrix_path, checkpoint, selector, metadata_path)
    controller_seed = int(matrix["evaluation"].get("controller_seed", 55)
                          if controller_seed is None else controller_seed)
    if output.exists():
        return completed_capture(output, checkpoint=checkpoint, matrix_path=matrix_path,
            resolved=resolved, selector=selector, seed=seed, controller_seed=controller_seed,
            max_steps=max_steps, decisions=decisions, device=device)
    output.mkdir(parents=True, exist_ok=False)
    rollouts = int(matrix["evaluation"].get("togo_return_rollouts", 32))
    env, cold_env = _make_env(resolved), _make_env(resolved)
    source = cold = None
    started = time.perf_counter()
    try:
        source, run_config = _initialize_frozen_model(
            resolved, env, checkpoint, controller_seed, device=device)
        cold, _ = _initialize_frozen_model(
            resolved, cold_env, checkpoint, controller_seed, device=device)
        validate_source(source.cfg)
        frozen_before = _outer_state_digest(source)
        cold_before = _outer_state_digest(cold)
        # Match the original bundle evaluator's unscored compile/allocation solve.
        warm_obs = env.reset(seed=int(seed))[0]
        source.predict(warm_obs, deterministic=True, episode_start=True)
        cold.predict(warm_obs, deterministic=True, episode_start=True)
        warmup_seconds = time.perf_counter() - started
        engine = source.agent.inner_engine
        episode_seed = solver_seed(controller_seed, "episode", int(seed))
        engine.reset_for_evaluation(episode_seed, reuse_action_pool=True)
        _seed_spaces(env, int(seed))
        observation, _ = env.reset(seed=int(seed))
        prior = prior_snapshot(source)
        H, J = int(source.cfg.inner_rollout_horizon), int(source.cfg.inner_rounds)
        cell = selector.rsplit("/", 1)[-1]
        manifest = {
            "schema_version": 1, "protocol": PROTOCOL, "status": "running",
            "source_cell": cell, "selector": selector, "H": H, "J": J,
            "seed": int(seed), "controller_seed": controller_seed,
            "episode_solver_seed": episode_seed, "episode_max_steps": int(max_steps),
            "matrix": str(Path(matrix_path).resolve()), "matrix_sha256": sha256_file(matrix_path),
            "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": sha256_file(checkpoint),
            "resolved_preset": resolved, "resolved_config": _jsonable(vars(source.cfg)),
            "run_config": _jsonable(run_config), "decisions": list(decisions),
            "captured_roots": [], "source_trace": "source-trace.jsonl.gz",
            "source_trajectory": "source-trajectory.jsonl.gz",
            "warmup_including_initialization_seconds": warmup_seconds,
            "source_frozen_digest_before": frozen_before,
            "cold_pairing": "Exact copy of warm pre-action private learner RNG, fresh prior actor/critic/Adam/replay/alpha; no source-state mutation.",
        }
        try:
            manifest["source_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            manifest["source_commit"] = None
        atomic_json(output / "manifest.json", manifest)
        total = 0.
        terminated = truncated = False
        with gzip.open(output / "source-trace.jsonl.gz", "wt") as trace_file, \
                gzip.open(output / "source-trajectory.jsonl.gz", "wt") as trajectory_file:
            for decision in range(max_steps):
                selected = decision in decisions
                snapshot = capture_simulator_snapshot(env) if selected else None
                learner_rng = engine.rng.training_state_dict() if selected else None
                lifetime_before = int(engine.state.actor_lifetime_steps)
                trace = make_trace(source, controller_seed, seed, decision,
                                   capture=selected, rollouts=rollouts)
                action, _ = source.predict(observation, deterministic=True,
                                            episode_start=decision == 0, trace=trace)
                for event in trace.events:
                    trace_file.write(json.dumps({"decision_index": decision, **event}, allow_nan=False) + "\n")
                if selected:
                    root_dir = output / f"decision-{decision}"
                    root_dir.mkdir()
                    actors = [save_actor_snapshot(item, root_dir, "warm")
                              for item in trace.actor_snapshots]
                    cold_action, cold_trace = captured_cold_solve(
                        cold, observation.copy(), learner_rng, controller_seed=controller_seed,
                        seed=seed, decision=decision, rollouts=rollouts)
                    actors.extend(save_actor_snapshot(item, root_dir, "cold")
                                  for item in cold_trace.actor_snapshots)
                    actors.append(save_actor_snapshot(prior, root_dir, "prior"))
                    rng_bytes = io.BytesIO()
                    torch.save(learner_rng, rng_bytes)
                    write_bytes(root_dir / "learner-rng.pt", rng_bytes.getvalue())
                    if capture_simulator_snapshot(env).sha256 != snapshot.sha256:
                        raise RuntimeError("Capture/cold solve changed the source simulator.")
                    root = {
                        "schema_version": 1, "protocol": PROTOCOL,
                        "root_id": f"{cell}-seed-{seed}-decision-{decision}",
                        "source_cell": cell, "selector": selector, "H": H, "J": J,
                        "seed": int(seed), "decision_index": decision,
                        "episode_max_steps": max_steps, "episode_return_before": total,
                        "observation": np.asarray(observation).tolist(),
                        "snapshot": snapshot.to_dict(), "simulator_sha256": snapshot.sha256,
                        "actors": actors, "learner_rng_path": "learner-rng.pt",
                        "learner_rng_sha256": sha256_file(root_dir / "learner-rng.pt"),
                        "actor_lifetime_updates_before": lifetime_before,
                        "controller_seed": controller_seed, "episode_solver_seed": episode_seed,
                        "matrix": manifest["matrix"], "matrix_sha256": manifest["matrix_sha256"],
                        "checkpoint": manifest["checkpoint"],
                        "checkpoint_sha256": manifest["checkpoint_sha256"],
                        "manifest": "../manifest.json", "resolved_config": manifest["resolved_config"],
                        "diagnostic_events": {"warm": trace.events, "cold": cold_trace.events},
                        "cold_pairing": manifest["cold_pairing"],
                        "source_action": np.asarray(action).tolist(),
                        "cold_action": np.asarray(cold_action).tolist(),
                    }
                    atomic_json(root_dir / "root.json", root)
                    manifest["captured_roots"].append(f"decision-{decision}/root.json")
                    atomic_json(output / "manifest.json", manifest, overwrite=True)
                    print(json.dumps({"event": "root_captured", "root": str(root_dir / "root.json")}), flush=True)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                trajectory_file.write(json.dumps({
                    "decision_index": decision, "observation": np.asarray(observation).tolist(),
                    "action": np.asarray(action).tolist(), "reward": float(reward),
                    "terminated": bool(terminated), "truncated": bool(truncated),
                }, allow_nan=False) + "\n")
                total += float(reward)
                observation = next_observation
                if terminated or truncated:
                    break
        final_digest = _outer_state_digest(source)
        if final_digest != frozen_before or _outer_state_digest(cold) != cold_before:
            raise RuntimeError("Capture changed a frozen outer learner.")
        missing = sorted(set(decisions) - {int(Path(p).parent.name.split("-")[-1])
                                          for p in manifest["captured_roots"]})
        if missing:
            raise RuntimeError(f"Episode ended before requested roots: {missing}")
        manifest.update(status="complete", source_return=total, source_steps=decision + 1,
                        terminated=bool(terminated), truncated=bool(truncated),
                        source_frozen_digest_after=final_digest,
                        elapsed_seconds=time.perf_counter() - started)
        manifest["source_trajectory_sha256"] = sha256_file(output / "source-trajectory.jsonl.gz")
        manifest["source_trace_sha256"] = sha256_file(output / "source-trace.jsonl.gz")
        atomic_json(output / "manifest.json", manifest, overwrite=True)
        return manifest
    finally:
        _close_resources(source, cold, env, cold_env)


@torch.no_grad()
def snapshot_mean_action(model, observation, snapshot):
    policy = snapshot.make_policy(model.agent.device)
    obs = model._obs_to_tensor(observation).to(model.agent.device).unsqueeze(0)
    z = model.agent.model.encode(obs)
    action, _ = model.agent.model.pi(z, policy=policy, deterministic=True,
                                      **snapshot.policy_bounds)
    return model._unscale_action(action[0].cpu().numpy())


def install_carried_actor(model, snapshot, *, future_seed, decision_index,
                          lifetime_updates=0):
    """Install only actor weights in an isolated controller, preserving pools.

    The next prediction is a new solve with fresh critic/target/Adam/replay/alpha.
    Preparation occurs before setting future RNG, so all intervention branches
    begin their actual continuation from identical named random streams.
    """
    validate_source(model.cfg)
    engine = model.agent.inner_engine
    engine.reset_for_evaluation(int(future_seed), reuse_action_pool=True)
    with engine.rng.action_fork(), engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    policy = snapshot.make_policy(engine.device)
    engine.state.actor.load_state_dict(policy.state_dict(), strict=True)
    engine.state.actor.requires_grad_(True)
    engine.state.actor_lifetime_steps = int(lifetime_updates)
    engine._clear_expired(t0=False, include_action=True)
    engine.rng = engine._new_rng(int(future_seed))
    engine.action_index = int(decision_index) + 1
    engine.episode_index = 1
    model._predict_t0 = False


def run_replanning_branch(model, env, root, snapshot, *, future_seed):
    observation = restore_simulator_snapshot(
        env, SimulatorSnapshot.from_dict(root["snapshot"]), continuing=False)
    np.testing.assert_array_equal(observation, np.asarray(root["observation"], dtype=observation.dtype))
    install_carried_actor(
        model, snapshot, future_seed=future_seed, decision_index=root["decision_index"],
        lifetime_updates=(int(root.get("actor_lifetime_updates_before", 0)) + snapshot.actor_updates
                          if snapshot.inner else 0))
    remaining = int(root["episode_max_steps"]) - int(root["decision_index"])
    rewards, actions = [], []
    terminated = truncated = False
    for offset in range(remaining):
        action = (snapshot_mean_action(model, observation, snapshot) if offset == 0
                  else model.predict(observation, deterministic=True, episode_start=False)[0])
        observation, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        actions.append(np.asarray(action).tolist())
        if terminated or truncated:
            break
    if len(rewards) != remaining and not terminated:
        raise RuntimeError("Replanning branch truncated before the original episode cutoff.")
    return {
        "real_mc_return": float(sum(rewards)),
        "real_mc_discounted_return": float(sum(float(model.agent.discount) ** k * r
                                               for k, r in enumerate(rewards))),
        "return_semantics": "remaining_episode_undiscounted",
        "rewards": rewards, "actions": actions, "steps": len(rewards),
        "terminated": bool(terminated), "truncated": bool(truncated),
        "future_solver_seed": int(future_seed), "future_J": int(model.cfg.inner_rounds),
        "first_action": "captured_actor_mean", "future_solve_cadence": 1,
        "future_policy": "actor_only_warm_transfer", "gamma": float(model.agent.discount),
    }


def evaluate_replanning_root(*, root_path, output, checkpoint=None, matrix_path=None,
                             device=None, metadata_path=None):
    from evaluate_ambi_checkpoint import _close_resources, _initialize_frozen_model, _make_env, _outer_state_digest

    root_path, output = Path(root_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    root = json.loads(root_path.read_text())
    checkpoint = checkpoint or root["checkpoint"]
    matrix_path = matrix_path or root["matrix"]
    if sha256_file(checkpoint) != root["checkpoint_sha256"]:
        raise ValueError("Continuation checkpoint differs from captured checkpoint.")
    if sha256_file(matrix_path) != root["matrix_sha256"]:
        raise ValueError("Continuation matrix differs from captured matrix.")
    _, resolved = resolve_source(matrix_path, checkpoint, root["selector"], metadata_path)
    env = _make_env(resolved)
    model = None
    try:
        model, _ = _initialize_frozen_model(resolved, env, checkpoint,
                                           root["controller_seed"], device=device)
        validate_source(model.cfg)
        digest = _outer_state_digest(model)
        env.reset(seed=root["seed"])
        model.predict(np.asarray(root["observation"], dtype=np.float32),
                      deterministic=True, episode_start=True)
        candidates = [("prior", 0), ("warm", 0), ("warm", 4), ("warm", root["J"])]
        candidates = list(dict.fromkeys(candidates))
        future_seed = solver_seed(root["controller_seed"], "replan_continuation", root["root_id"])
        records = []
        for family, round_index in candidates:
            actor = next(item for item in root["actors"]
                         if item["family"] == family and item["round"] == round_index)
            snapshot = load_actor_snapshot(root_path, actor)
            result = run_replanning_branch(model, env, root, snapshot, future_seed=future_seed)
            records.append({"branch_kind": "replan", "actor_family": family,
                            "round": round_index, "action_mode": "mean", "replicate": 0,
                            "actor_sha256": snapshot.sha256, **result})
            print(json.dumps({"event": "replan_branch_complete", "root": root["root_id"],
                              "family": family, "round": round_index,
                              "return": result["real_mc_return"]}), flush=True)
        if _outer_state_digest(model) != digest:
            raise RuntimeError("Replanning changed the frozen outer learner.")
        shard = {"schema_version": 1, "protocol": PROTOCOL, "status": "complete",
                 "root_id": root["root_id"], "source_cell": root["source_cell"],
                 "H": root["H"], "J": root["J"], "seed": root["seed"],
                 "decision": root["decision_index"], "records": records,
                 "root_sha256": sha256_file(root_path),
                 "checkpoint_sha256": root["checkpoint_sha256"],
                 "frozen_digest": digest,
                 "future_rng_pairing": "Identical root-derived named learner streams after initialization, independent of stage; original simulator clocks retained."}
        atomic_json(output, shard)
        return shard
    finally:
        _close_resources(model, env)
