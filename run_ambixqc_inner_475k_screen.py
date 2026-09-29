"""Four frozen H1 inner-XQC conditions on one pinned shared-UTD2 checkpoint."""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import run_ambixqc_backbone_mppi_evaluation as banks
from run_ambixqc_mppi_evaluation import file_sha256

ROOT = Path(__file__).resolve().parent
MATRIX = ROOT / "configs/research/ambixqc_humanoid_shared_utd2_475k_h1_screen.json"
CHECKPOINT_SHA = "00d9347f00eb7719f55d4e3ef7abff8c24d7f2e214c923cdeba7b8e34afa1368"
CELL, STEP, INVENTORY_INDEX = "aux_shared_utd2", 475_000, 98
SELECTORS = tuple(f"controller/{route}_j{j}" for j in (1, 4)
                  for route in ("return_return", "soft_soft"))
MAP_SCHEMA = "ambixqc-inner-475k-run-map-v1"
REFERENCE_SCHEMA = "ambixqc-inner-475k-prior-reference-v1"
SEEDS = [101, 102, 103, 104, 105]
PROTOCOL = {
    "action_rule": "tanh_mean", "controller_seed": 12345,
    "env_wrapper": None, "env_wrappers": [],
    "environment": {"id": "DMControl-v0", "params": {
        "obs": "state", "render_mode": None, "task": "humanoid-walk"}},
    "max_steps": 500, "observation": "state", "seed_scheme": "sha256-v1",
}


def selector_for(index):
    if type(index) is not int or not 0 <= index < len(SELECTORS):
        raise ValueError("Condition index must be an integer from 0 through 3.")
    return SELECTORS[index]


def expected_settings(index):
    selector = selector_for(index)
    j = 1 if index < 2 else 4
    source = "aux_return" if "return_return" in selector else "xqc"
    matrix = banks.read_json(MATRIX)
    return {**matrix["shared_alg_params"],
            **matrix["comparisons"]["controller"]["variants"][selector.split("/")[1]]["alg_params"],
            "inner_temperature_mode": "auto", "inner_temperature_initialization": "inherit_outer",
            "inner_temperature_lr": 5e-5 / j,
            "inner_actor_scope": "action", "inner_critic_scope": "action",
            "inner_temperature_scope": "action", "inner_replay_scope": "action",
            "inner_actor_optimizer_scope": "action", "inner_critic_optimizer_scope": "action",
            "inner_temperature_optimizer_scope": "action", "inner_actor_adaptation": "clone",
            "inner_critic_adaptation": "clone", "inner_rebase_persistent": False,
            "aux_return_mode": "xqc", "aux_return_detach_representation": False,
            "xqc_utd": 2, "utd": 1,
            "inner_critic_source": source, "inner_horizon_critic_source": source}


def select_checkpoint(manifest, *, checkpoint_root=None):
    row = banks.select_checkpoint(manifest, INVENTORY_INDEX, checkpoint_root=checkpoint_root)
    if (row["cell"], row["step"], row["sha256"]) != (CELL, STEP, CHECKPOINT_SHA):
        raise ValueError("This screen requires the pinned shared UTD2 475k checkpoint.")
    return row


def select_reference(path, row, manifest):
    if path is None:
        raise ValueError("Production requires a pinned prior-reference index.")
    data = banks.read_json(path)
    if (data.get("schema") != REFERENCE_SCHEMA
            or data.get("checkpoint_manifest_sha256") != file_sha256(manifest)
            or data.get("checkpoint_sha256") != CHECKPOINT_SHA):
        raise ValueError("Prior-reference index checkpoint or inventory association differs.")
    bundle = Path(data["bundle_path"])
    if not bundle.is_absolute() or not (bundle / "manifest.json").is_file():
        raise ValueError("Prior reference must be an existing absolute bundle directory.")
    if file_sha256(bundle / "manifest.json") != data.get("manifest_sha256"):
        raise ValueError("Prior reference manifest hash differs.")
    saved = banks.read_json(bundle / "manifest.json")
    if (saved.get("checkpoint", {}).get("metadata_sha256") != row["metadata_sha256"]
            or saved.get("code", {}).get("dirty") is not False):
        raise ValueError("Prior reference sidecar or clean-source evidence differs.")
    from utils.ambi_benchmark import reference_returns
    values = reference_returns(bundle, CHECKPOINT_SHA, PROTOCOL)
    prior, = [run for run in saved["runs"] if run.get("selector") == "controller/prior"]
    result = prior["result"]
    if (list(values) != SEEDS or result.get("outer_state_unchanged") is not True
            or result.get("outer_updates_before") != result.get("outer_updates_after")
            or any(e["length"] != 500 or e.get("capped") for e in prior["episodes"])
            or result.get("resolved_config", {}).get("inner_operator") != "none"):
        raise ValueError("Prior reference must contain the five complete frozen actor-mean episodes.")
    # The earlier MPPI checkout has a different evaluator fingerprint. Reuse is
    # justified by checkpoint/sidecar, environment, action and episode-seed
    # contracts, not by claiming the two evaluators have identical source.
    return bundle


def resolve_run_map(path, index, mode):
    selector = selector_for(index)
    if mode == "smoke":
        if path is not None:
            raise ValueError("Smoke must not assign publication curves.")
        return {}
    if path is None:
        raise ValueError("Production requires four explicit New evaluation curves.")
    data = banks.read_json(path)
    mapping = data.get("runs", {})
    if (data.get("schema") != MAP_SCHEMA or set(mapping) != set(SELECTORS)
            or data.get("checkpoint_sha256") != CHECKPOINT_SHA):
        raise ValueError("Run map must assign exactly these four checkpoint conditions.")
    if any(not isinstance(v, str) or not Path(v).is_absolute() for v in mapping.values()):
        raise ValueError("Evaluation curve directories must be absolute.")
    if len({str(Path(v).resolve()) for v in mapping.values()}) != len(SELECTORS):
        raise ValueError("Each condition requires a distinct evaluation curve.")
    from utils.ambi_benchmark import resolve_eval_run_map
    return resolve_eval_run_map([selector], run_map={selector: mapping[selector]})


def validate_bundle(bundle_path, index, *, seeds, max_steps, reference_bundle=None,
                    source_sha=None):
    path = Path(bundle_path)
    saved = banks.read_json(path / "manifest.json")
    selector = selector_for(index)
    cfg_expected = expected_settings(index)
    j = cfg_expected["inner_rounds"]
    if (saved.get("status") != "complete" or saved.get("code", {}).get("dirty") is not False
            or (source_sha is not None and saved["code"].get("commit") != source_sha)
            or saved.get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA
            or saved.get("protocol") != {**PROTOCOL, "max_steps": max_steps}
            or len(saved.get("runs", [])) != 1):
        raise ValueError("Bundle source, checkpoint, protocol or completion differs.")
    run = saved["runs"][0]
    result = run["result"]
    if (run.get("selector") != selector or run.get("status") != "complete"
            or result.get("action_rule") != "tanh_mean"
            or result.get("outer_state_unchanged") is not True
            or result.get("outer_updates_before") != result.get("outer_updates_after")
            or result.get("nonfinite_model_metrics") or result.get("nonfinite_trace_metrics")):
        raise ValueError("Evaluation changed frozen outer state or did not finish with finite metrics.")
    cfg = result["resolved_config"]
    if any(cfg.get(key) != value for key, value in cfg_expected.items()):
        bad = [key for key, value in cfg_expected.items() if cfg.get(key) != value]
        raise ValueError(f"Resolved inner settings differ: {bad}")
    provenance = result.get("checkpoint_evaluation_provenance", {})
    evaluated = provenance.get("evaluated_semantic_signature", {})
    if evaluated.get("inner_actor_bn_mode") != "running":
        raise ValueError("Frozen-evaluation provenance does not record running actor BN.")
    if [e["seed"] for e in run["episodes"]] != list(seeds):
        raise ValueError("Episode seeds differ.")
    from utils.ambi_benchmark import reference_returns
    reference = reference_returns(reference_bundle, CHECKPOINT_SHA, saved["protocol"]) if reference_bundle else None
    for episode in run["episodes"]:
        if (episode["length"] != max_steps or not math.isfinite(episode["return"])
                or max_steps == 500 and episode.get("capped")):
            raise ValueError("Episode length, cap or return is invalid.")
        if reference is not None and not math.isclose(
                episode["paired_return_delta"], episode["return"] - reference[episode["seed"]],
                rel_tol=0, abs_tol=1e-9):
            raise ValueError("Paired gain differs from the exact reused prior episode.")
    seen, traces = set(), {}
    for relative in run["trace_files"]:
        trace = (path / relative).resolve()
        if path.resolve() not in trace.parents or str(relative) in traces:
            raise ValueError("Trace paths escape the bundle or are duplicated.")
        traces[str(relative)] = file_sha256(trace)
        with gzip.open(trace, "rt") as stream:
            for line in stream:
                event = json.loads(line)
                identity = (event["episode_id"], event["decision_index"])
                if identity in seen or event["phase"] != "decision" or event["nonfinite"]:
                    raise ValueError("Trace decisions are duplicated, nonfinite or the wrong phase.")
                seen.add(identity)
                if (event["critic_updates"], event["actor_updates"], event["temperature_updates"]) != (3*j, j, j):
                    raise ValueError("Actual per-decision optimizer counts differ.")
                metrics = event["metrics"]
                required = {"decision/inner_model_steps": 256*j,
                            "decision/inner_reward_scale_delta": 0,
                            "decision/inner_reward_normalizer_imagined_updates": 0,
                            "decision/inner_diagnostics_sampled": 1}
                if any(metrics.get(key) != value for key, value in required.items()):
                    raise ValueError("Trace work, frozen scale or sampling differs.")
                routing = ("decision/inner_critic_source_aux_return",
                           "decision/inner_horizon_critic_source_aux_return",
                           "decision/inner_critic_target_reward_only")
                # InnerXQCEngine intentionally omits these three diagnostics
                # for the unchanged xqc/xqc/entropy_augmented route. Config
                # and frozen-evaluation provenance were checked above; only
                # these known default metrics may be absent, never work/scale.
                expected_route = float(index % 2 == 0)
                if any(metrics.get(key, 0.0 if not expected_route else None) != expected_route
                       for key in routing):
                    raise ValueError("Trace critic routing differs.")
                if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in metrics.values()):
                    raise ValueError("Trace contains nonfinite metrics.")
    if seen != {(f"seed-{seed}", decision) for seed in seeds for decision in range(max_steps)}:
        raise ValueError("Trace decisions are missing, extra or misaligned.")
    return {"outer_state_unchanged": True, "episodes": len(seeds), "decisions": len(seen),
            "critic_updates_per_decision": 3*j, "actor_updates_per_decision": j,
            "temperature_updates_per_decision": j, "model_steps_per_decision": 256*j,
            "actor_bn_mode": "running", "prior_reused": reference is not None,
            "trace_sha256": traces, "bundle_manifest_sha256": file_sha256(path / "manifest.json")}


def prepare_specs(manifest, index, directory, *, checkpoint_root=None):
    selector = selector_for(index)
    row = select_checkpoint(manifest, checkpoint_root=checkpoint_root)
    from evaluate_ambi_checkpoint import evaluate_matrix
    return evaluate_matrix(MATRIX, row["path"], selectors=[selector], checkpoint_inventory=manifest,
                           source_run=row["source_run"], eval_series_spec_dir=directory)


def run(manifest, index, result_root, *, mode="production", device="cuda", eval_run_map=None,
        checkpoint_root=None, reference_index=None):
    if mode not in ("smoke", "production"):
        raise ValueError("Mode must be smoke or production.")
    selector = selector_for(index)
    row = select_checkpoint(manifest, checkpoint_root=checkpoint_root)
    assigned = resolve_run_map(eval_run_map, index, mode)
    reference = select_reference(reference_index, row, manifest) if mode == "production" else None
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.ambi_benchmark import atomic_json, code_identity, stage_completed_bundle
    code = code_identity()
    if code.get("dirty") is not False or not code.get("commit"):
        raise ValueError("Evaluation requires a clean committed checkout.")
    output = Path(result_root).resolve() / selector.split("/")[1]
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Results must be outside the source checkout.")
    output.mkdir(parents=True, exist_ok=False)
    seeds, steps = ([101, 102], 3) if mode == "smoke" else (SEEDS, 500)
    provenance = {"source_sha": code["commit"], "checkpoint": row, "index": index,
                  "selector": selector, "mode": mode, "seeds": seeds, "max_steps": steps,
                  "manifest_sha256": file_sha256(manifest), "matrix_sha256": file_sha256(MATRIX),
                  "controller_seed": 12345, "replay_used": False, "eval_run_map": assigned,
                  "reference_bundle": str(reference) if reference else None,
                  "reference_index_sha256": file_sha256(reference_index) if reference else None}
    atomic_json(output / "provenance.json", provenance)
    payload = evaluate_matrix(MATRIX, row["path"], selectors=[selector], seeds=seeds,
                              controller_seed=12345, max_steps=steps, device=device,
                              bundle_dir=output / "bundle", reference_bundle=reference,
                              checkpoint_inventory=manifest, source_run=row["source_run"],
                              eval_run_map=assigned or None, stage_results=False)
    atomic_json(output / "results.json", payload)
    if payload["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("Evaluated checkpoint differs from the pinned checkpoint.")
    validation = validate_bundle(output / "bundle", index, seeds=seeds, max_steps=steps,
                                 reference_bundle=reference, source_sha=code["commit"])
    atomic_json(output / "validation.json", {**provenance, **validation})
    if assigned:
        staged = stage_completed_bundle(output / "bundle", assigned, source_run=row["source_run"], inventory_path=manifest)
        if set(staged) != {selector} or staged[selector]["status"] != "queued":
            raise RuntimeError("Failed to stage the expected curve; preserve results for upload recovery.")
    (output / "PASS").write_text("PASS\n")
    print(json.dumps({"output": str(output), "selector": selector, **validation}))
    return output


def verify_smokes(smoke_root, manifest, source_sha):
    paths = sorted(Path(smoke_root).glob("job*-task*/*/validation.json"))
    if len(paths) != len(SELECTORS):
        raise ValueError("Require exactly four successful smoke evaluations.")
    seen = []
    for path in paths:
        saved = banks.read_json(path)
        index = saved["index"]
        selector = selector_for(index)
        expected = {"source_sha": source_sha, "manifest_sha256": file_sha256(manifest),
                    "matrix_sha256": file_sha256(MATRIX), "mode": "smoke", "selector": selector}
        if any(saved.get(k) != value for k, value in expected.items()):
            raise ValueError("Smoke source, inventory or settings differ.")
        job = path.parent.parent
        if (job / "FAILED").exists() or any(not p.is_file() or p.read_text().strip() != "PASS"
                for p in (job / "PASS", path.parent / "PASS")):
            raise ValueError("Smoke job is failed or incomplete.")
        actual = validate_bundle(path.parent / "bundle", index, seeds=[101, 102], max_steps=3, source_sha=source_sha)
        if any(saved.get(key) != value for key, value in actual.items()):
            raise ValueError("Smoke artifacts changed after validation.")
        if index == 0:
            import re
            if not re.search(r"\b[1-9][0-9]* passed\b", (job / "pytest.log").read_text()):
                raise ValueError("Actor-BN CUDA regression tests did not pass.")
        seen.append(index)
    if sorted(seen) != list(range(len(SELECTORS))):
        raise ValueError("Smoke conditions are duplicated or missing.")
    return {"validated_indices": sorted(seen), "source_sha": source_sha}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path)
    p.add_argument("--index", type=int)
    p.add_argument("--result-root", type=Path)
    p.add_argument("--mode", choices=("smoke", "production"), default="production")
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-run-map", type=Path)
    p.add_argument("--reference-index", type=Path)
    p.add_argument("--eval-series-spec-dir", type=Path)
    p.add_argument("--verify-smoke-root", type=Path)
    p.add_argument("--source-sha")
    args = p.parse_args(argv)
    if args.verify_smoke_root:
        print(json.dumps(verify_smokes(args.verify_smoke_root, args.manifest, args.source_sha)))
    elif args.index is None:
        p.error("--index is required")
    elif args.eval_series_spec_dir:
        print(json.dumps(prepare_specs(args.manifest, args.index, args.eval_series_spec_dir,
                                       checkpoint_root=args.checkpoint_root)))
    elif args.result_root is None:
        p.error("--result-root is required")
    else:
        run(args.manifest, args.index, args.result_root, mode=args.mode, device=args.device,
            eval_run_map=args.eval_run_map, checkpoint_root=args.checkpoint_root,
            reference_index=args.reference_index)


if __name__ == "__main__":
    main()
