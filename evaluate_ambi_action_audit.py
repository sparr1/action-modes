"""Matched-state, frozen-800k action audit; not a full-episode policy evaluation.

One worker owns a source-history/episode-seed pair. Model action selection,
held-out scoring, and real calibration use separate named random streams.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from evaluate_ambi_checkpoint import (
    _close_resources, _file_sha256, _initialize_frozen_model, _make_env,
    _outer_state_digest, _seed_spaces, _validate_checkpoint_contract,
)
from evaluate_ambi_transfer_diagnostics import resolved_setting
from utils.ambi_benchmark import solver_seed
from utils.ambi_research import load_preset_matrix, resolve_preset
from utils.checkpoint_context import load_checkpoint_context
from utils.matched_action_audit import mppi_candidate, score_action_bank, score_real_candidates
from utils.transfer_diagnostic_real import capture_simulator_snapshot, preserve_simulator
from utils.transfer_diagnostics import Reference, evaluating, json_value, solve_fork, validate_controller, write_json


PROTOCOL = "matched-action-audit-v1"
HISTORIES = ("prior", "sac_j6", "mppi_h3")
MEANS = ("prior_mean", "sac_j4_mean", "sac_j6_mean", "sac_j8_mean", "mppi_h1_mean", "mppi_h3_mean")
PUBLICATION_CANDIDATES = (*MEANS, "broad_best_h1", "broad_best_h3", "replay_best_h1", "replay_best_h3")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(json_value(value), stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def load_config(path, *, smoke=False):
    def unique_pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate configuration key: {key}")
            result[key] = value
        return result
    config = json.loads(Path(path).read_text(), object_pairs_hook=unique_pairs)
    if config["protocol"] != PROTOCOL or config["J"] != [4, 6, 8] or config["histories"] != list(HISTORIES):
        raise ValueError("Unsupported action audit contract.")
    if config["publication_candidates"] != list(PUBLICATION_CANDIDATES):
        raise ValueError("Publication candidates differ from the action audit contract.")
    if config["checkpoint"]["step"] != 800000 or len(config["checkpoint"]["sha256"]) != 64:
        raise ValueError("This audit is pinned to the 800k checkpoint.")
    for name in ("seeds", "decisions"):
        values = config[name]
        if not values or len(set(values)) != len(values) or any(type(v) is not int or v < 0 for v in values):
            raise ValueError(f"Invalid {name}.")
    if max(config["decisions"]) >= config["source_max_steps"]:
        raise ValueError("Audit roots must precede source_max_steps.")
    for name in ("mc_rollouts", "real_rollouts", "real_tail_steps", "policy_samples", "uniform_actions", "local_actions_per_scale"):
        if type(config[name]) is not int or config[name] < 2:
            raise ValueError(f"Invalid {name}.")
    config["smoke"] = bool(smoke)
    if smoke:
        config.update(seeds=[101], decisions=[2], source_max_steps=3,
                      mc_rollouts=2, real_rollouts=2, real_tail_steps=4,
                      policy_samples=2, uniform_actions=4, local_actions_per_scale=2)
    return config


def task_pairs(config):
    return [(history, seed) for history in config["histories"] for seed in config["seeds"]]


def state_hash(state):
    """Hash tensor values rather than storage identities or pickle records."""
    digest = hashlib.sha256()
    def add(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, np.ndarray):
            digest.update(str((value.dtype.str, value.shape)).encode())
            digest.update(value.tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                add(str(key)); add(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                add(item)
        else:
            digest.update(repr(value).encode())
    add(state)
    return digest.hexdigest()


def select_candidates(scores):
    """Select once on selection draws; aliases preserve provenance."""
    rows = scores["actions"]
    indexed = {row["label"]: row for row in rows}
    selected = [(name, name) for name in MEANS]
    for family, prefix in (("broad", "broad/"), ("replay", "replay/")):
        eligible = [row for row in rows if row["label"].startswith(prefix)]
        if not eligible:
            raise ValueError(f"Empty candidate family: {family}")
        for horizon in (1, 3):
            winner = max(eligible, key=lambda row: row["model"][f"h{horizon}"]["mean"])
            selected.append((f"{family}_best_h{horizon}", winner["label"]))
    return dict(labels=[name for name, _ in selected],
                actions=[indexed[source]["action"] for _, source in selected],
                provenance={name: source for name, source in selected})


def replay_actions(reference, expected):
    engine = reference.engine
    replay = engine.state.replay if engine.state.replay is not None else engine._action_pool.replay
    if replay is None or replay.size != expected or replay.next_sample_id != expected:
        raise RuntimeError("SAC replay is incomplete or overflowed.")
    if not bool((replay.horizon_end[:expected] == 1).all()):
        raise RuntimeError("H1 replay contains a non-boundary transition.")
    roots = replay.z[:expected]
    if not torch.allclose(roots, roots[:1].expand_as(roots), atol=0., rtol=0.):
        raise RuntimeError("H1 replay contains states other than the common root.")
    return replay.action[:expected].detach().clone()


def policy_bank(reference, actor, observation, *, seed, count, bounds=None):
    z = reference.encode(np.asarray(observation)[None])
    options = reference.bounds if bounds is None else bounds
    with torch.no_grad(), evaluating(reference.model, actor):
        stats = reference.model.policy_stats(z, policy=actor, **options)
        samples = reference.model.pi(z.expand(count, -1), policy=actor,
            noise=reference.noise((count, reference.cfg.action_dim), seed), **options)[0]
    return stats["mean"].detach(), samples.detach(), stats


def audit_root(audits, env, observation, config, *, history, episode_seed, decision, task_dir):
    started = time.monotonic()
    reference = Reference(audits[4], rollouts=config["mc_rollouts"])
    seed = solver_seed(config["controller_seed"], "action-audit", episode_seed, decision)
    # Same solver stream across J, and across source histories at equal seed/time;
    # J only extends the number of learning rounds.
    solve_seed = solver_seed(seed, "solve")
    actions, labels, critics, learners, planner_rows = [], [], {}, {}, {}
    def append(name, values):
        values = torch.as_tensor(values, device=reference.device, dtype=torch.float32).reshape(-1, reference.cfg.action_dim)
        actions.append(values.detach())
        labels.extend([name] if len(values) == 1 else [f"{name}/{i:04d}" for i in range(len(values))])

    snapshot = capture_simulator_snapshot(env)
    write_json(task_dir / f"simulator-{decision}.json", snapshot.to_dict())
    prior_mean, prior_samples, _ = policy_bank(reference, reference.engine._actor_base, observation,
        seed=solver_seed(seed, "policy-samples"), count=config["policy_samples"], bounds=reference.engine._actor_options)
    append("prior_mean", prior_mean)
    append("policy/prior", prior_samples)
    for rounds, wrapped in audits.items():
        local = Reference(wrapped, rollouts=config["mc_rollouts"])
        rng = local.engine._new_rng(solve_seed).training_state_dict()
        action, trace, final = solve_fork(wrapped, observation, rng, capture_rounds=())
        actor = final.make_module("actor", reference.device)
        critic = final.make_module("critic", reference.device)
        critics[f"sac_j{rounds}_final"] = critic
        if "initial" not in critics:
            initial = next(s for s in trace.learner_snapshots if s.stage == "initial")
            critics["initial"] = initial.make_module("critic", reference.device)
        mean, samples, stats = policy_bank(local, actor, observation,
            seed=solver_seed(seed, "policy-samples"), count=config["policy_samples"], bounds=final.policy_bounds)
        executed = torch.as_tensor(wrapped._scale_action(action), device=local.device).reshape_as(mean)
        if not torch.allclose(mean, executed, atol=2e-6, rtol=2e-6):
            raise RuntimeError("Snapshot actor mean disagrees with executed action.")
        append(f"sac_j{rounds}_mean", mean)
        append(f"policy/sac_j{rounds}", samples)
        replay = replay_actions(local, rounds * int(local.cfg.inner_rollouts_per_round))
        encoded_root = local.encode(np.asarray(observation)[None])
        pooled = local.engine._action_pool.replay
        if not torch.allclose(pooled.z[:1], encoded_root, atol=1e-6, rtol=1e-6):
            raise RuntimeError("Replay root disagrees with the saved simulator observation.")
        append(f"replay/j{rounds}", replay)
        learners[f"j{rounds}"] = dict(alpha=final.alpha, actor_loss_scale=final.actor_loss_scale,
            actor_updates=final.actor_updates, critic_updates=final.critic_updates,
            temperature_updates=final.temperature_updates, replay_size=len(replay),
            policy_stats={name: value.detach().cpu().tolist() for name, value in stats.items() if torch.is_tensor(value)},
            bounds=final.policy_bounds)

    for horizon in (1, 3):
        candidate = mppi_candidate(reference, observation, horizon=horizon,
            seed=solver_seed(seed, "mppi", horizon), planner_options=config["mppi"])
        append(f"mppi_h{horizon}_mean", candidate["normalized_action"])
        planner_rows[f"h{horizon}"] = candidate
    generator = torch.Generator(device=reference.device).manual_seed(solver_seed(seed, "broad"))
    append("broad/uniform", 2 * torch.rand((config["uniform_actions"], reference.cfg.action_dim),
        generator=generator, device=reference.device) - 1)
    center = torch.atanh(prior_mean.clamp(-1 + 1e-6, 1 - 1e-6))
    for scale in config["local_scales"]:
        noise = torch.randn((config["local_actions_per_scale"], reference.cfg.action_dim),
                            generator=generator, device=reference.device)
        append(f"broad/local-{scale}", torch.tanh(center + float(scale) * noise))
    bank = torch.cat(actions)
    selection = score_action_bank(reference, observation, actions=bank, labels=labels,
        critics=critics, seed=solver_seed(seed, "selection"), mc_rollouts=config["mc_rollouts"])
    selected = select_candidates(selection)
    heldout = score_action_bank(reference, observation, actions=selected["actions"], labels=selected["labels"],
        critics=critics, seed=solver_seed(seed, "validation"), mc_rollouts=config["mc_rollouts"])
    # Persist expensive model work before beginning real counterfactuals.
    write_json(task_dir / f"model-{decision}.json", dict(selection_scores=selection,
        heldout_scores=heldout, selected=selected, learners=learners, planners=planner_rows))
    atomic_json(task_dir / "progress.json", dict(status="running", stage="real_calibration", decision=decision))
    real = score_real_candidates(reference, env, snapshot, actions=selected["actions"], labels=selected["labels"],
        seed=solver_seed(seed, "real"), real_rollouts=config["real_rollouts"], tail_steps=config["real_tail_steps"])
    if capture_simulator_snapshot(env).sha256 != snapshot.sha256:
        raise RuntimeError("Action audit changed source simulator state.")
    return dict(identity=dict(history=history, seed=episode_seed, decision=decision),
        observation=np.asarray(observation).tolist(), simulator_sha256=snapshot.sha256,
        selection_scores=selection, heldout_scores=heldout, real_scores=real,
        selected=selected, learners=learners, planners=planner_rows,
        elapsed_seconds=time.monotonic()-started,
        checks=dict(simulator_restored=True, h1_replay_root_only=True, h1_replay_complete=True,
                    independent_selection_validation=True, execution_mean_verified=True))


def resolved_base(config, checkpoint):
    matrix_path = Path(config["matrix"])
    matrix = load_preset_matrix(matrix_path)
    matrix["checkpoint_steps"] = [config["checkpoint"]["step"]]
    matrix["checkpoint_contract"] = {k: config["checkpoint"][k] for k in ("step", "sha256")}
    context = load_checkpoint_context(checkpoint)
    base = resolve_preset(matrix_path, "return_return/fresh", matrix=matrix, checkpoint_context=context)
    _validate_checkpoint_contract(matrix, checkpoint, context, [base])
    return base


def run(args):
    config = load_config(args.config, smoke=args.smoke)
    checkpoint = args.checkpoint.resolve()
    if _file_sha256(checkpoint) != config["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint hash does not match the 800k audit.")
    metadata = Path(str(checkpoint) + ".metadata.json")
    if _file_sha256(metadata) != config["checkpoint"]["metadata_sha256"]:
        raise ValueError("Checkpoint metadata hash does not match the audit.")
    source_dir = Path(__file__).resolve().parent
    commit = subprocess.check_output(["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True).strip()
    config.update(checkpoint={**config["checkpoint"], "path": str(checkpoint)}, source_commit=commit,
        config_sha256=_file_sha256(args.config))
    campaign = args.campaign_root.resolve()
    base = resolved_base(config, checkpoint)
    if args.prepare:
        campaign.mkdir(parents=True, exist_ok=False)
        write_json(campaign / "campaign.json", config)
        print(json.dumps(dict(campaign_root=str(campaign), tasks=len(task_pairs(config)), source_commit=commit)))
        return
    saved = json.loads((campaign / "campaign.json").read_text())
    if saved != config:
        raise ValueError("Campaign identity differs from worker configuration/source/checkpoint.")
    pairs = task_pairs(config)
    if args.task_index is None or not 0 <= args.task_index < len(pairs):
        raise ValueError("A valid task-index is required.")
    history, episode_seed = pairs[args.task_index]
    task_dir = campaign / "tasks" / f"{history}-seed-{episode_seed}"
    task_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    device = torch.device(args.device)
    torch.set_num_threads(1)
    runtime = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        device=str(device), hostname=platform.node(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        gpu_memory_bytes=torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None)
    manifest = dict(protocol=PROTOCOL, status="running", identity=dict(history=history, seed=episode_seed),
        config=config, runtime=runtime, task_index=args.task_index)
    write_json(task_dir / "manifest.started.json", manifest)
    env = source = None
    audits = {}
    try:
        source_resolved = resolved_setting(base, 1, 6)
        env = _make_env(source_resolved)
        source, _ = _initialize_frozen_model(source_resolved, env, checkpoint, config["controller_seed"], device=args.device)
        for rounds in config["J"]:
            audits[rounds], _ = _initialize_frozen_model(resolved_setting(base, 1, rounds), env, checkpoint,
                config["controller_seed"], device=args.device)
            validate_controller(audits[rounds])
        fingerprints = [_outer_state_digest(wrapped) for wrapped in (source, *audits.values())]
        ref = Reference(source)
        source_seed = solver_seed(config["controller_seed"], "episode", episode_seed)
        source.agent.inner_engine.reset_for_evaluation(source_seed, reuse_action_pool=True)
        _seed_spaces(env, episode_seed)
        observation, _ = env.reset(seed=episode_seed)
        total_reward, captured, previous_mean = 0., [], None
        for decision in range(config["source_max_steps"]):
            if decision in config["decisions"]:
                atomic_json(task_dir / "progress.json", dict(status="running", stage="model_audit", decision=decision))
                rng_before = state_hash(source.agent.inner_engine.rng.training_state_dict())
                with preserve_simulator(env):
                    root = audit_root(audits, env, observation, config, history=history,
                        episode_seed=episode_seed, decision=decision, task_dir=task_dir)
                rng_after = state_hash(source.agent.inner_engine.rng.training_state_dict())
                if rng_before != rng_after:
                    raise RuntimeError("Audit advanced source controller RNG.")
                root["checks"]["source_rng_unchanged"] = True
                if fingerprints != [_outer_state_digest(wrapped) for wrapped in (source, *audits.values())]:
                    raise RuntimeError("Audit changed a frozen outer learner.")
                root["checks"]["outer_state_unchanged"] = True
                atomic_json(task_dir / f"root-{decision}.json", root)
                captured.append(decision)
            if history == "prior":
                mean, _, _ = policy_bank(ref, ref.engine._actor_base, observation, seed=0, count=1,
                                         bounds=ref.engine._actor_options)
                action = source._unscale_action(mean[0].cpu().numpy())
            elif history == "sac_j6":
                action, _ = source.predict(observation, deterministic=True, episode_start=decision == 0)
            else:
                result = mppi_candidate(ref, observation, horizon=3,
                    seed=solver_seed(source_seed, "source-mppi", decision), planner_options=config["mppi"],
                    previous_mean=previous_mean)
                previous_mean = torch.as_tensor(result["proposal_mean"], device=ref.device)
                action = source._unscale_action(np.asarray(result["normalized_action"], dtype=np.float32))
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        if fingerprints != [_outer_state_digest(wrapped) for wrapped in (source, *audits.values())]:
            raise RuntimeError("Source trajectory changed a frozen outer learner.")
        manifest.update(status="complete", roots=captured, root_count=len(captured),
            unreached_decisions=sorted(set(config["decisions"])-set(captured)),
            source_prefix_return=total_reward, source_steps=decision+1,
            outer_state_unchanged=True, elapsed_seconds=time.monotonic()-started)
        atomic_json(task_dir / "manifest.json", manifest)
        atomic_json(task_dir / "progress.json", dict(status="complete", roots=captured))
        print(json.dumps({key: manifest[key] for key in ("identity", "status", "root_count", "elapsed_seconds")}))
    except BaseException as error:
        atomic_json(task_dir / "progress.json", dict(status="failed", error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        _close_resources(source, *audits.values(), env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/research/ambi_800k_action_audit.json"))
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
