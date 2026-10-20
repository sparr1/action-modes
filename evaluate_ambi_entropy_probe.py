"""Measure inner-SAC entropy dynamics at frozen, matched-protocol roots.

One worker owns one checkpoint and one environment seed. It saves the entire
mean-action source prefix before running any diagnostics. Nothing is published
to W&B and the checkpoint's outer model, optimizers and temperature stay frozen.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time
import json

import numpy as np
import torch

from evaluate_ambi_checkpoint import (
    _close_resources, _file_sha256, _initialize_frozen_model, _jsonable,
    _make_env, _outer_state_digest, _seed_spaces,
)
from utils.ambi_benchmark import (
    atomic_json, canonical_hash, capture_root, frozen_execution_action,
    protocol_for, read_json, solver_seed,
)
from utils.ambi_research import resolve_preset
from utils.checkpoint_context import load_checkpoint_context


ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE = ROOT / "configs/research/ambi_entropy_probe.json"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--environment-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh immutable bundle directory.")
    parser.add_argument("--decisions", type=int, nargs="+", help="Subset of profile roots, for a smoke check.")
    parser.add_argument("--root-bank", type=Path, help="Reuse a compatible saved bank from this exact cell/source.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true", help="Validate identities/config; do not load model or simulator.")
    return parser


def _science_identity():
    paths = [Path(__file__), ROOT / "utils/ambi_entropy_probe.py",
             ROOT / "utils/ambi_benchmark.py", ROOT / "utils/ambi_research.py",
             ROOT / "utils/checkpoint_context.py", ROOT / "evaluate_ambi_checkpoint.py",
             ROOT / "domains/dmcontrol.py", ROOT / "RL/AMBITDMPC2.py",
             *sorted((ROOT / "RL/tdmpc2_core").rglob("*.py"))]
    files = {str(path.relative_to(ROOT)): _file_sha256(path) for path in paths}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    return {"sha256": canonical_hash(files), "git_commit": commit, "files": files}


def _prepare(args):
    profile_path = args.profile.expanduser().resolve()
    profile = read_json(profile_path)
    if profile.get("schema_version") != 1:
        raise ValueError("Entropy probe profile schema_version must be 1.")
    if args.torch_threads < 1:
        raise ValueError("torch_threads must be positive.")
    if args.environment_seed not in profile["environment_seeds"]:
        raise ValueError("environment-seed must be one of the paired profile seeds.")
    decisions = sorted(profile["decisions"] if args.decisions is None else args.decisions)
    if (not decisions or len(set(decisions)) != len(decisions)
            or any(value not in profile["decisions"] for value in decisions)):
        raise ValueError("decisions must be a nonempty distinct subset of the profile roots.")
    try:
        selected = profile["checkpoints"][args.checkpoint_id]
    except KeyError as exc:
        raise ValueError(f"Unknown checkpoint-id: {args.checkpoint_id}") from exc
    checkpoint = args.checkpoint.expanduser().resolve()
    context = load_checkpoint_context(checkpoint, args.metadata)
    if _file_sha256(checkpoint) != selected["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA256 does not match the pinned profile.")
    if _file_sha256(context.source) != selected["metadata_sha256"]:
        raise ValueError("Checkpoint metadata SHA256 does not match the pinned profile.")
    if context.metadata["checkpoint"]["step"] != selected["step"]:
        raise ValueError("Checkpoint step does not match the pinned profile.")
    matrix_path = (profile_path.parent / profile["matrix"]).resolve()
    if _file_sha256(matrix_path) != profile["matrix_sha256"]:
        raise ValueError("Source preset matrix changed; review and repin the entropy profile.")
    resolved = resolve_preset(matrix_path, profile["preset"], checkpoint_context=context)
    if frozen_execution_action(resolved) != "mean":
        raise ValueError("Entropy root collection requires mean execution.")
    params = resolved["algorithm_config"]["alg_params"]
    if params.get("aux_return_mode") != "sac" or params.get("obs", "state") != "state":
        raise ValueError("This protocol requires a state-observation auxiliary-return checkpoint.")
    if resolved["evaluation"]["controller_seed"] != profile["controller_seed"]:
        raise ValueError("Source controller seed differs from the entropy profile.")
    return profile, selected, checkpoint, context, resolved, decisions


def collect_roots(env, model, *, environment_seed, controller_seed, decisions, identity):
    """Capture every requested pre-action root, then return; no probe runs here.

    Warmup and private episode RNG reset match the benchmark-bundle evaluator.
    Historical traces did not save root observations/actions, so these are newly
    collected protocol-matched trajectories, not historical trajectory replay.
    """
    started = time.perf_counter()
    warm_observation, _ = env.reset(seed=environment_seed)
    warm_started = time.perf_counter()
    model.predict(warm_observation, deterministic=True, episode_start=True)
    warmup_seconds = time.perf_counter() - warm_started
    episode_seed = solver_seed(controller_seed, "episode", environment_seed)
    model.agent.inner_engine.reset_for_evaluation(episode_seed, reuse_action_pool=True)
    _seed_spaces(env, environment_seed)
    observation, _ = env.reset(seed=environment_seed)
    roots, transitions, cumulative_return = [], [], 0.0
    for decision in range(max(decisions) + 1):
        if decision in decisions:
            roots.append(capture_root(observation, environment_seed, decision, cumulative_return))
        if decision == max(decisions):
            break
        action_started = time.perf_counter()
        action, _ = model.predict(observation, deterministic=True, episode_start=(decision == 0))
        seconds = time.perf_counter() - action_started
        action = np.asarray(action).copy()
        if action.ndim != 1 or not np.isfinite(action).all():
            raise ValueError("Source controller returned a nonfinite or non-vector action.")
        observation, reward, terminated, truncated, _ = env.step(action)
        cumulative_return += float(reward)
        transitions.append(dict(decision_index=decision, action=action.tolist(), reward=float(reward),
                                terminated=bool(terminated), truncated=bool(truncated),
                                controller_seconds=seconds))
        if terminated or truncated:
            raise RuntimeError(f"Source episode ended after decision {decision}; cannot collect all requested roots.")
    bank = dict(schema_version=1, kind="ambi_entropy_matched_prefix_roots", identity=identity,
                environment_seed=environment_seed, episode_controller_seed=episode_seed,
                root_source="newly_collected_frozen_H3_J8_mean_prefix",
                historical_trajectory_replay=False, roots=roots, transitions=transitions)
    bank["id"] = canonical_hash(bank)
    return bank, dict(collection_seconds=time.perf_counter() - started,
                      warmup_seconds=warmup_seconds, warmup_solves=1,
                      source_controller_solves=len(transitions),
                      source_simulator_decisions=len(transitions))


def load_root_bank(path, *, identity, environment_seed, decisions):
    bank = read_json(path)
    if bank.get("schema_version") != 1 or bank.get("kind") != "ambi_entropy_matched_prefix_roots":
        raise ValueError("Not an entropy matched-prefix root bank.")
    if bank.get("id") != canonical_hash({key: value for key, value in bank.items() if key != "id"}):
        raise ValueError("Root bank content hash mismatch.")
    if bank.get("identity") != identity or bank.get("environment_seed") != environment_seed:
        raise ValueError("Root bank identity does not match checkpoint, source, protocol and seed.")
    if [row["decision_index"] for row in bank["roots"]] != decisions:
        raise ValueError("Root bank coverage differs from requested decisions.")
    if len(bank["transitions"]) != max(decisions):
        raise ValueError("Root bank lacks the complete source prefix.")
    for row in bank["roots"]:
        expected = capture_root(np.asarray(row["observation"], dtype=np.float32),
                                environment_seed, row["decision_index"], row["return_before"])
        if row != expected:
            raise ValueError("Root bank observation metadata is inconsistent.")
    return bank


def run_probe(args):
    profile, selected, checkpoint, context, resolved, decisions = _prepare(args)
    plan = dict(checkpoint_id=args.checkpoint_id, source_run=selected["source_run"],
                checkpoint_step=selected["step"], environment_seed=args.environment_seed,
                decisions=decisions, roots=len(decisions), controller_seed=profile["controller_seed"],
                actor_steps=profile["actor_steps"], anchor_samples=profile["anchor_samples"],
                exact_engine=True, source_controller_solves=0 if args.root_bank else max(decisions),
                warmup_solves=0 if args.root_bank else 1,
                root_source="newly_collected_frozen_H3_J8_mean_prefix",
                historical_trajectory_replay=False,
                diagnostic_seed_scheme="solver_seed(controller_seed, entropy_probe, environment_seed, decision)",
                diagnostic_rng="independent_of_source_episode; paired_across_backbones",
                diagnostic_order="complete_and_save_source_prefix_before_any_root_diagnostics")
    if args.plan_only:
        return plan
    # Import after preflight; --plan-only does not initialize the diagnostic core.
    from utils.ambi_entropy_probe import probe_entropy_root
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh output directory; existing results are immutable.")
    torch.set_num_threads(args.torch_threads)
    science = _science_identity()
    protocol = protocol_for(resolved, profile["controller_seed"], resolved["evaluation"]["max_steps"])
    identity = dict(checkpoint_sha256=selected["checkpoint_sha256"],
                    metadata_sha256=selected["metadata_sha256"], source_sha256=science["sha256"],
                    profile_sha256=_file_sha256(args.profile.expanduser().resolve()),
                    resolved_sha256=canonical_hash(resolved), protocol=protocol)
    bank = load_root_bank(args.root_bank, identity=identity, environment_seed=args.environment_seed,
                          decisions=decisions) if args.root_bank else None
    model, env, error = None, None, None
    started, records = time.perf_counter(), []
    output.mkdir(parents=True)
    atomic_json(output / "plan.json", plan)
    atomic_json(output / "checkpoint.metadata.json", context.metadata)
    atomic_json(output / "source.json", science)
    atomic_json(output / "resolved.json", resolved)
    try:
        env = _make_env(resolved)
        model, run_config = _initialize_frozen_model(resolved, env, checkpoint,
                                                    profile["controller_seed"], device=args.device)
        digest_before = _outer_state_digest(model)
        atomic_json(output / "runtime-config.json", _jsonable(vars(model.cfg)))
        collection = dict(collection_seconds=0., warmup_seconds=0., warmup_solves=0,
                          source_controller_solves=0, source_simulator_decisions=0)
        if bank is None:
            bank, collection = collect_roots(env, model, environment_seed=args.environment_seed,
                controller_seed=profile["controller_seed"], decisions=decisions, identity=identity)
        digest_after_collection = _outer_state_digest(model)
        if digest_after_collection != digest_before:
            raise RuntimeError("Frozen outer state changed during root collection.")
        # This durable bank precedes every diagnostic, so failed diagnostics can
        # be retried with --root-bank without recomputing source controller work.
        atomic_json(output / "root-bank.json", bank)
        print(f"Saved {len(bank['roots'])} roots; source prefix complete.", flush=True)
        for index, root in enumerate(bank["roots"]):
            root_started = time.perf_counter()
            seed = solver_seed(profile["controller_seed"], "entropy_probe", args.environment_seed,
                               root["decision_index"])
            measured = probe_entropy_root(model, np.asarray(root["observation"], dtype=np.float32),
                seed=seed, actor_steps=profile["actor_steps"], anchor_samples=profile["anchor_samples"],
                exact_engine=True)
            if not measured.get("outer_state_unchanged") or not measured.get("global_rng_unchanged"):
                raise RuntimeError("Diagnostic core did not verify outer-state and RNG preservation.")
            if _outer_state_digest(model) != digest_before:
                raise RuntimeError("Frozen outer state changed during root diagnostics.")
            record = dict(root_id=root["root_id"], environment_seed=args.environment_seed,
                          decision_index=root["decision_index"], root_sha256=canonical_hash(root),
                          diagnostic_seed=seed, seconds=time.perf_counter() - root_started,
                          diagnostics=measured)
            atomic_json(output / "roots" / f"{index:04d}.json", record)
            records.append(dict(path=f"roots/{index:04d}.json", sha256=canonical_hash(record),
                                root_id=root["root_id"], diagnostic_seed=seed, seconds=record["seconds"]))
            print(f"Completed entropy root {index + 1}/{len(bank['roots'])}: {root['root_id']}", flush=True)
        if _science_identity()["sha256"] != science["sha256"]:
            raise RuntimeError("Source changed while the diagnostic was running.")
        result = dict(schema_version=1, kind="ambi_entropy_probe", complete=True, protocol=plan,
                      identity=identity, source_git_commit=science["git_commit"], root_bank_id=bank["id"],
                      resolved_run_config=run_config, outer_state_unchanged=True,
                      outer_state_sha256_before=digest_before,
                      outer_state_sha256_after_collection=digest_after_collection,
                      outer_state_sha256_after_diagnostics=_outer_state_digest(model),
                      global_rng_unchanged_for_each_probe=True, root_records=records,
                      timing=dict(total_seconds=time.perf_counter() - started, **collection),
                      uncertainty_unit="environment_seed; roots_within_seed_are_clustered")
        result["sha256"] = canonical_hash(result)
        atomic_json(output / "result.json", result)
        return result
    except BaseException as exc:
        error = exc
        atomic_json(output / "failure.json", dict(complete=False, error=str(exc), completed_roots=len(records)))
        raise
    finally:
        _close_resources(model, env, primary_error=error)


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run_probe(args)
    print(json.dumps(result if args.plan_only else dict(complete=result["complete"],
          output=str(args.output.resolve()), timing=result["timing"]), indent=2))


if __name__ == "__main__":
    main()
