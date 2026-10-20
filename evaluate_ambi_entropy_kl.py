"""Compare joint automatic entropy and prior-KL penalties on pinned saved roots."""
import argparse
from pathlib import Path
import time

import torch

from evaluate_ambi_checkpoint import (
    _close_resources, _file_sha256, _initialize_frozen_model,
    _jsonable, _make_env, _outer_state_digest,
)
from evaluate_ambi_entropy_probe import _prepare, _science_identity as _entropy_source
from evaluate_ambi_prior_kl import validate_saved_bank
from utils.ambi_entropy_kl import probe_entropy_kl_root, beta_multiplier_key, validate_beta_multipliers
from utils.ambi_benchmark import atomic_json, canonical_hash, solver_seed


ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE = ROOT / "configs/research/ambi_entropy_kl_probe.json"


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


def science_identity():
    source = _entropy_source()
    for relative in ("evaluate_ambi_entropy_kl.py", "utils/ambi_entropy_kl.py",
                     "evaluate_ambi_prior_kl.py", "utils/ambi_prior_kl.py",
                     "utils/ambi_gradient_probe.py"):
        source["files"][relative] = _file_sha256(ROOT / relative)
    source["sha256"] = canonical_hash(source["files"])
    return source


def run(args):
    profile, selected, checkpoint, context, resolved, decisions = _prepare(args)
    multipliers = validate_beta_multipliers(profile["beta_multipliers"])
    bank_path = args.root_bank.expanduser().resolve()
    bank, original_source = validate_saved_bank(bank_path, profile=profile,
        checkpoint_id=args.checkpoint_id, environment_seed=args.environment_seed, resolved=resolved)
    plan = dict(checkpoint_id=args.checkpoint_id, checkpoint_step=selected["step"],
        source_run=selected["source_run"], environment_seed=args.environment_seed,
        decisions=decisions, roots=len(decisions), actor_steps=profile["actor_steps"],
        anchor_samples=profile["anchor_samples"], source_controller_solves=0,
        source_simulator_decisions=0, full_solves_per_root=len(multipliers),
        methods=[beta_multiplier_key(value) for value in multipliers], beta_multipliers=list(multipliers),
        changed_parameter="inner_outer_policy_kl_coef", inner_temperature_lr=0.0003,
        beta_selection="multiplier_times_inherited_outer_alpha_held_fixed", hard_kl_budget_enforced=False,
        root_bank_path=str(bank_path),
        root_bank_id=bank["id"], root_bank_sha256=_file_sha256(bank_path),
        profile_sha256=_file_sha256(args.profile.expanduser().resolve()),
        original_source_git_commit=original_source["git_commit"], uncertainty_unit="environment_seed")
    if args.plan_only:
        return plan
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.torch_threads)
    science = science_identity()
    atomic_json(output / "plan.json", plan)
    atomic_json(output / "source.json", science)
    atomic_json(output / "profile.json", profile)
    atomic_json(output / "original-source.json", original_source)
    atomic_json(output / "root-bank.json", bank)
    atomic_json(output / "resolved.json", resolved)
    model, env, error, records = None, None, None, []
    start = time.perf_counter()
    try:
        env = _make_env(resolved)
        model, _ = _initialize_frozen_model(resolved, env, checkpoint,
                                            profile["controller_seed"], device=args.device)
        initial_digest = _outer_state_digest(model)
        if int(model.cfg.inner_actor_updates_per_action) != int(profile["actor_steps"]):
            raise ValueError("Profile actor_steps does not match the complete configured solve.")
        atomic_json(output / "runtime-config.json", _jsonable(vars(model.cfg)))
        for index, root in enumerate(r for r in bank["roots"] if r["decision_index"] in decisions):
            seed = solver_seed(profile["controller_seed"], "entropy_probe",
                               args.environment_seed, root["decision_index"])
            before = time.perf_counter()
            diagnostics = probe_entropy_kl_root(model, root["observation"], seed=seed,
                beta_multipliers=multipliers, anchor_samples=profile["anchor_samples"])
            if (not diagnostics["outer_state_unchanged"] or not diagnostics["global_rng_unchanged"]
                    or _outer_state_digest(model) != initial_digest):
                raise RuntimeError("Joint entropy-KL diagnostic did not preserve frozen state and RNG.")
            record = dict(root_id=root["root_id"], environment_seed=args.environment_seed,
                decision_index=root["decision_index"], root_sha256=canonical_hash(root),
                diagnostic_seed=seed, seconds=time.perf_counter() - before, diagnostics=diagnostics)
            path = f"roots/{index:04d}.json"
            atomic_json(output / path, record)
            records.append(dict(path=path, root_id=root["root_id"], sha256=canonical_hash(record),
                                seconds=record["seconds"]))
            print(f"Completed paired entropy-KL root {index + 1}/{len(decisions)}", flush=True)
        if science_identity()["sha256"] != science["sha256"]:
            raise RuntimeError("Source changed during diagnostic.")
        if _file_sha256(args.profile.expanduser().resolve()) != plan["profile_sha256"]:
            raise RuntimeError("Profile changed during diagnostic.")
        result = dict(schema_version=1, kind="ambi_entropy_kl_probe", complete=True, protocol=plan,
            original_root_bank_identity=bank["identity"], original_root_bank_id=bank["id"],
            original_root_bank_sha256=plan["root_bank_sha256"],
            original_source_git_commit=original_source["git_commit"], new_source=science,
            root_records=records, outer_state_unchanged=True, outer_state_sha256=initial_digest,
            global_rng_unchanged_for_each_probe=True, target_entropy=float(model.agent.target_entropy),
            action_dim=int(model.cfg.action_dim),
            timing=dict(total_seconds=time.perf_counter() - start, includes_observer_overhead=True),
            uncertainty_unit="environment_seed; roots within seed are clustered")
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
    print(json.dumps(result if arguments.plan_only else
                     dict(complete=result["complete"], timing=result["timing"]), indent=2))
