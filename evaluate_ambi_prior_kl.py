"""Evaluate prior-KL mitigation on pinned, previously saved entropy-probe roots."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from evaluate_ambi_checkpoint import _close_resources, _file_sha256, _initialize_frozen_model, _jsonable, _make_env, _outer_state_digest
from evaluate_ambi_entropy_probe import _prepare, _science_identity as _entropy_source
from utils.ambi_benchmark import atomic_json, canonical_hash, capture_root, read_json, solver_seed

ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE = ROOT / "configs/research/ambi_prior_kl_probe.json"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--environment-seed", type=int, required=True)
    parser.add_argument("--root-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions", nargs="+", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def validate_saved_bank(path, *, profile, checkpoint_id, environment_seed, resolved):
    """Accept the original immutable identity, never rewrite it to current code."""
    bank, source = read_json(path), read_json(path.parent / "source.json")
    selected = profile["checkpoints"][checkpoint_id]
    expected_bank = profile["root_banks"][checkpoint_id][str(environment_seed)]["id"]
    if bank.get("id") != canonical_hash({k: v for k, v in bank.items() if k != "id"}) or bank.get("id") != expected_bank:
        raise ValueError("Saved root bank content hash does not match the pinned original bank.")
    if source.get("git_commit") != profile["source_git_commit"] or source.get("sha256") != canonical_hash(source.get("files", {})):
        raise ValueError("Saved root bank source is not the pinned original source.")
    identity = bank["identity"]
    expected = dict(checkpoint_sha256=selected["checkpoint_sha256"], metadata_sha256=selected["metadata_sha256"],
                    profile_sha256=profile["source_profile_sha256"], source_sha256=source["sha256"],
                    resolved_sha256=canonical_hash(resolved))
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved root bank checkpoint/profile/resolved/source identity mismatch.")
    if bank.get("environment_seed") != environment_seed or bank.get("kind") != "ambi_entropy_matched_prefix_roots":
        raise ValueError("Saved root bank seed or kind mismatch.")
    if [r["decision_index"] for r in bank["roots"]] != profile["decisions"] or len(bank["transitions"]) != max(profile["decisions"]):
        raise ValueError("Saved root bank has incomplete root or prefix coverage.")
    for row in bank["roots"]:
        expected = capture_root(np.asarray(row["observation"], dtype=np.float32), environment_seed,
                                row["decision_index"], row["return_before"])
        if row != expected or not np.isfinite(row["observation"]).all():
            raise ValueError("Saved root observation metadata is inconsistent.")
    return bank, source


def science_identity():
    source = _entropy_source()
    for relative in ("evaluate_ambi_prior_kl.py", "utils/ambi_prior_kl.py"):
        source["files"][relative] = _file_sha256(ROOT / relative)
    source["sha256"] = canonical_hash(source["files"])
    return source


def run(args):
    profile, selected, checkpoint, context, resolved, decisions = _prepare(args)
    bank_path = args.root_bank.expanduser().resolve()
    bank, original_source = validate_saved_bank(bank_path, profile=profile, checkpoint_id=args.checkpoint_id,
        environment_seed=args.environment_seed, resolved=resolved)
    plan = dict(checkpoint_id=args.checkpoint_id, checkpoint_step=selected["step"], source_run=selected["source_run"],
        environment_seed=args.environment_seed, decisions=decisions, roots=len(decisions),
        actor_steps=profile["actor_steps"], anchor_samples=profile["anchor_samples"],
        source_controller_solves=0, source_simulator_decisions=0,
        methods=["baseline_entropy", "prior_kl"], beta="inherited_outer_alpha_held_fixed",
        root_bank_path=str(bank_path), root_bank_id=bank["id"], root_bank_sha256=_file_sha256(bank_path),
        original_source_git_commit=original_source["git_commit"], uncertainty_unit="environment_seed")
    if args.plan_only:
        return plan
    from utils.ambi_prior_kl import probe_prior_kl_root
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.torch_threads)
    science = science_identity()
    atomic_json(output / "plan.json", plan)
    atomic_json(output / "source.json", science)
    atomic_json(output / "original-source.json", original_source)
    atomic_json(output / "root-bank.json", bank)
    atomic_json(output / "resolved.json", resolved)
    model, env, error, records = None, None, None, []
    start = time.perf_counter()
    try:
        env = _make_env(resolved)
        model, _ = _initialize_frozen_model(resolved, env, checkpoint, profile["controller_seed"], device=args.device)
        initial_digest = _outer_state_digest(model)
        atomic_json(output / "runtime-config.json", _jsonable(vars(model.cfg)))
        for index, root in enumerate(r for r in bank["roots"] if r["decision_index"] in decisions):
            seed = solver_seed(profile["controller_seed"], "entropy_probe", args.environment_seed, root["decision_index"])
            before = time.perf_counter()
            diagnostics = probe_prior_kl_root(model, root["observation"], seed=seed,
                actor_steps=profile["actor_steps"], anchor_samples=profile["anchor_samples"])
            if not diagnostics["outer_state_unchanged"] or not diagnostics["global_rng_unchanged"] or _outer_state_digest(model) != initial_digest:
                raise RuntimeError("Prior-KL diagnostic did not preserve frozen state and RNG.")
            record = dict(root_id=root["root_id"], environment_seed=args.environment_seed,
                decision_index=root["decision_index"], root_sha256=canonical_hash(root), diagnostic_seed=seed,
                seconds=time.perf_counter() - before, diagnostics=diagnostics)
            path = f"roots/{index:04d}.json"
            atomic_json(output / path, record)
            records.append(dict(path=path, root_id=root["root_id"], sha256=canonical_hash(record), seconds=record["seconds"]))
            print(f"Completed paired prior-KL root {index+1}/{len(decisions)}", flush=True)
        if science_identity()["sha256"] != science["sha256"]:
            raise RuntimeError("Source changed during diagnostic.")
        result = dict(schema_version=1, kind="ambi_prior_kl_probe", complete=True, protocol=plan,
            original_root_bank_identity=bank["identity"], original_root_bank_id=bank["id"],
            original_root_bank_sha256=plan["root_bank_sha256"], original_source_git_commit=original_source["git_commit"],
            new_source=science, root_records=records, outer_state_unchanged=True,
            outer_state_sha256=initial_digest, global_rng_unchanged_for_each_probe=True,
            target_entropy=float(model.agent.target_entropy), action_dim=int(model.cfg.action_dim),
            timing=dict(total_seconds=time.perf_counter()-start), uncertainty_unit="environment_seed; roots within seed are clustered")
        result["sha256"] = canonical_hash(result)
        atomic_json(output / "result.json", result)
        return result
    except BaseException as exc:
        error = exc
        atomic_json(output / "failure.json", dict(complete=False, error=str(exc), completed_roots=len(records)))
        raise
    finally:
        _close_resources(model, env, primary_error=error)


if __name__ == "__main__":
    import json
    arguments = build_parser().parse_args()
    result = run(arguments)
    print(json.dumps(result if arguments.plan_only else dict(complete=result["complete"], timing=result["timing"]), indent=2))
