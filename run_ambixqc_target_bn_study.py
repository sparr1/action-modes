"""Test H2/J2/G6 running target BatchNorm against the completed batch-statistics control.

--completed-sweep-root identifies the completed Stage 10 H2 update comparison. Every
parent result is verified; only the exact Stage 8 H2/J2/G6 control is reused.
Historical results retain source 10852a8; the new condition has its own pinned execution revision.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
import json
import math
from pathlib import Path
import sys

import run_ambixqc_h2_update_study as previous

horizon = previous.horizon

extension = horizon.extension
sweep = horizon.sweep
continuation = horizon.continuation
TOOLING_ROOT = horizon.TOOLING_ROOT
PARENT_SOURCE_SHA = horizon.SOURCE_SHA
SOURCE_SHA = "f2877284b6ac0210682e03f886b1e0548d5bb702"
SCHEMA = "ambixqc-target-bn-study-v1"
STAGE = "stage11"
ROUNDS = (2,)
UPDATES = (6,)
BASELINE_CONDITIONS = 35
BASELINE_EPISODES = 175
BASELINE_RUN_ID = "4b2d4437ffd84ca49093e303f794a11b"
PARENT_TOOLING_SHA = "51c49abb7266393ba39d9f82b40575d4327f607c"
STUDY_ID = "ambixqc-h2-target-bn-20261001"
WORKER_TIME = "02:00:00"
expected_counts = sweep.expected_counts
worker_root = sweep.worker_root
parent_arguments = extension.parent_arguments


def baseline_condition(study):
    return horizon.conditions(study)[0]


def conditions(study):
    cell = deepcopy(baseline_condition(study))
    cell.update(target_bn_mode="running", selector="controller/return_return_running_h2_j2_g6_target_running")
    cell["settings"]["inner_critic_target_bn_mode"] = "running"
    return [cell]


def split_conditions(study):
    return conditions(study), [baseline_condition(study)]


def matrix_for(study, cells):
    matrix = study.matrix_for(cells)
    matrix["description"] = "H2/J2/G6 target BatchNorm ablation; historical batch-statistics control retained."
    for cell in cells:
        matrix["comparisons"]["controller"]["variants"][cell["selector"].split("/")[1]]["description"] = (
            "Return/return; H2 J2 G6 delay3 N256 B256, all learning rates5e-5. "
            "Only the inner target critic now uses inherited running statistics; outer tail unchanged.")
    return matrix


def _baseline_registry(evidence, study):
    binding = previous._baseline_registry(evidence["previous"], study)
    if study.read(study.check_binding(binding)).get("run_id") != BASELINE_RUN_ID:
        raise ValueError("H2 update comparison requires the exact published Stage 8 J2/G6 run")
    return binding


def parent_evidence(args, study, publication):
    root = args.completed_sweep_root
    state_path, complete_path = root / "coordinator-state.json", root / "COMPLETE.json"
    state, complete = study.read(state_path), study.read(complete_path)
    if ({k: v for k, v in complete.items() if k != "workspace"} != state
            or state.get("schema") != previous.SCHEMA or state.get("source_sha") != PARENT_SOURCE_SHA
            or state.get("progress_run_id") != args.progress_run_id or state.get("stage10_complete") is not True
            or state.get("tooling", {}).get("commit") != PARENT_TOOLING_SHA
            or not str(state.get("job", "")).isdigit() or state.get("failures")
            or (root / "FAILED.json").exists()):
        raise ValueError("Require the complete, published 51c49ab Stage 10 H2 update comparison")
    plan_path = study.check_binding(state["plan"])
    plan = previous.load_plan(plan_path, study, PARENT_SOURCE_SHA, PARENT_TOOLING_SHA)
    if (plan_path != (root / "stage10-plan.json").resolve() or plan.get("result_root") != str(root)
            or plan["tooling"] != state["tooling"] or plan["execution"] != state["execution"]
            or plan["progress_run_id"] != args.progress_run_id
            or plan["completed_sweep_root"] != state["completed_sweep_root"]):
        raise ValueError("Stage 10 plan differs from its completed coordinator")
    prior_args = previous.parent_arguments(plan)
    inherited, baseline = previous.parent_evidence(prior_args, study, publication)
    if inherited != plan["parent"] or baseline != plan["reused"]:
        raise ValueError("Stage 10 inherited baseline or parent changed")
    for key in ("manifest", "reference_index", "smoke_root", "checkpoint_root"):
        setattr(args, key, getattr(prior_args, key))
    if (plan["inputs"] != {key: study.bind(getattr(args, key)) for key in ("manifest", "reference_index")}
            or plan["smoke_root"] != str(args.smoke_root)
            or plan["checkpoint_root"] != (str(args.checkpoint_root) if args.checkpoint_root else None)):
        raise ValueError("Stage 10 inherited checkpoint or reference inputs differ")
    row = study.screen.select_checkpoint(args.manifest, checkpoint_root=args.checkpoint_root)
    if str(study.screen.select_reference(args.reference_index, row, args.manifest)) != plan["reference_bundle"]:
        raise ValueError("Stage 10 paired reference differs")
    results_path = study.check_binding(state["stage10_results"])
    result = study.read(results_path)
    if (results_path != (root / "stage10-results.json").resolve()
            or result.get("schema") != previous.SCHEMA or result.get("stage") != "stage10"
            or result.get("source_sha") != PARENT_SOURCE_SHA or result.get("plan") != study.bind(plan_path)
            or result.get("reused") != baseline or len(result.get("entries", [])) != 2):
        raise ValueError("Stage 10 results are incomplete or incompatible")
    receipts, published = [], {}
    for index, cell in enumerate(plan["conditions"]):
        entry = previous.validate_worker(plan, index, state["job"], study)
        if result["entries"][index] != entry or entry["condition"] != cell:
            raise ValueError("Stage 10 result differs from its validated H2 worker")
        run_dir = Path(plan["runs"][cell["selector"]]["path"]).parent
        binding, receipt = sweep.publication_receipt(run_dir, cell["selector"], study,
                                                     publication, PARENT_SOURCE_SHA, entry)
        receipts.append(binding)
        published[cell["selector"]] = receipt
    if state.get("published") != published:
        raise ValueError("Stage 10 lacks both acknowledged publications")
    if [item["condition"] for item in baseline] != [baseline_condition(study)]:
        raise ValueError("Stage 10 must retain the exact H2/J2/G6 baseline")
    evidence = {"source_sha": PARENT_SOURCE_SHA, "previous": inherited, "stage10": {"state": study.bind(state_path),
        "complete": study.bind(complete_path), "plan": study.bind(plan_path),
        "results": study.bind(results_path), "publication_receipts": receipts}}
    evidence["baseline_registry"] = _baseline_registry(evidence, study)
    return evidence, baseline


def prepare_plan(args, study, publication, provenance):
    return extension.prepare_plan(args, study, publication, provenance, policy=sys.modules[__name__])


def load_plan(path, study, source_sha, tooling_sha):
    plan = extension.load_plan(path, study, source_sha, tooling_sha, policy=sys.modules[__name__])
    if (plan["parent"].get("source_sha") != PARENT_SOURCE_SHA
            or plan["parent"]["baseline_registry"] != _baseline_registry(plan["parent"], study)):
        raise ValueError("H2 update study changed its exact Stage 8 baseline registry")
    return plan


def validate_bundle(plan, index, root, study, smoke):
    if plan["conditions"][index] not in conditions(study):
        raise ValueError("Target-BN validation requires the one running-target condition")
    result = sweep.validate_bundle(plan, index, root, study, smoke)
    result = horizon.validate_h2_boundaries(plan, index, root, study, result)
    run = study.read(Path(root) / "bundle/manifest.json")["runs"][0]
    provenance = run["result"]["checkpoint_evaluation_provenance"]
    if (provenance["evaluated_semantic_signature"].get("inner_critic_target_bn_mode") != "running"
            or provenance["saved_semantic_signature"].get("inner_critic_target_bn_mode", "batch_no_update") != "batch_no_update"):
        raise ValueError("Target-BN saved/evaluated semantics differ")
    bundle = Path(root) / "bundle"
    for relative in study.read(bundle / "manifest.json")["runs"][0]["trace_files"]:
        with gzip.open(bundle / relative, "rt") as stream:
            if any(json.loads(line)["metrics"].get("decision/inner_critic_target_bn_running") != 1 for line in stream):
                raise ValueError("Target-BN trace does not record running targets")
    return {**result, "target_bn_mode": "running"}


def validate_default_bundle(plan, index, root, study, smoke):
    if not smoke or plan["conditions"] != [baseline_condition(study)]:
        raise ValueError("Compatibility canary must be the exact short historical baseline")
    result = sweep.validate_bundle(plan, index, root, study, smoke)
    return horizon.validate_h2_boundaries(plan, index, root, study, result)


def compare_default_canary(plan, root, study):
    """Compare matched roots; retain later closed-loop drift as diagnostics.

    CUDA categorical projection is not deterministic in the historical protocol.
    Same-source repeats can diverge after the first executed action. Comparing
    later losses then compares different observations, not implementation parity.
    """
    old_root = Path(plan["reused"][0]["result"]["output_path"]) / "smoke"
    new_root = Path(root) / "default-compat"
    default_plan = {**plan, "conditions": [baseline_condition(study)]}
    validate_default_bundle(default_plan, 0, new_root, study, True)
    validate_default_bundle({**default_plan, "source_sha": PARENT_SOURCE_SHA}, 0, old_root, study, True)
    manifests = [study.read(p / "bundle/manifest.json") for p in (old_root, new_root)]
    old_run, new_run = [m["runs"][0] for m in manifests]
    report = {"old_source_sha": PARENT_SOURCE_SHA, "new_source_sha": plan["source_sha"],
        "old_manifest": study.bind(old_root / "bundle/manifest.json"),
        "new_manifest": study.bind(new_root / "bundle/manifest.json"),
        "rtol": 1e-5, "atol": 1e-6, "max_absolute_difference": 0.0,
        "numeric_comparisons": 0, "decisions": 0, "matched_root_decisions": 0,
        "comparison_scope": "first_decision_per_seed_and_all_discrete_or_reward_statistics",
        "action_comparison": False, "closed_loop_differences": [],
        "old_traces": [], "new_traces": []}
    def compare(a, b, key, *, matched_root, decision=None):
        tokens = set(key.split("/")[-1].split("_"))
        exact = bool(tokens & {"count", "steps", "updates", "rows", "slots", "size", "capacity",
                              "rollouts", "evaluations", "fallback", "compiled", "flag"})
        exact = exact or key.startswith("decision/inner_reward_")
        if not matched_root and not exact:
            if a != b:
                report["closed_loop_differences"].append({"key": key, "decision": decision,
                    "old": a, "new": b, "absolute_difference": abs(a-b)})
            return
        if (a != b if exact else not math.isclose(a, b, rel_tol=report["rtol"], abs_tol=report["atol"])):
            raise ValueError(f"Default-mode compatibility canary differs at {key}: {a} vs {b}")
        report["max_absolute_difference"] = max(report["max_absolute_difference"], abs(a-b))
        report["numeric_comparisons"] += 1
    for old, new in zip(old_run["episodes"], new_run["episodes"]):
        if old["seed"] != new["seed"] or old["length"] != new["length"]:
            raise ValueError("Default-mode canary episode alignment differs")
        compare(old["return"], new["return"], "episode_return", matched_root=False,
                decision=old["seed"])
    if len(old_run["trace_files"]) != len(new_run["trace_files"]):
        raise ValueError("Default-mode canary trace count differs")
    for old_file, new_file in zip(old_run["trace_files"], new_run["trace_files"]):
        old_path, new_path = old_root / "bundle" / old_file, new_root / "bundle" / new_file
        report["old_traces"].append(study.bind(old_path)); report["new_traces"].append(study.bind(new_path))
        def rows(path):
            with gzip.open(path, "rt") as stream:
                return [json.loads(line) for line in stream]
        old_rows, new_rows = rows(old_path), rows(new_path)
        if len(old_rows) != len(new_rows):
            raise ValueError("Default-mode canary trace length differs")
        if not old_rows or old_rows[0]["decision_index"] != 0 or new_rows[0]["decision_index"] != 0:
            raise ValueError("Default-mode canary must include the matched initial root")
        for old, new in zip(old_rows, new_rows):
            for key in ("decision_index", "event_index", "phase", "round_index",
                        "critic_updates", "actor_updates", "temperature_updates"):
                if old[key] != new[key]:
                    raise ValueError("Default-mode canary counters differ: " + key)
            old_metrics = {k: v for k, v in old["metrics"].items() if "_seconds" not in k}
            new_metrics = {k: v for k, v in new["metrics"].items()
                           if "_seconds" not in k and k != "decision/inner_critic_target_bn_running"}
            if old_metrics.keys() != new_metrics.keys():
                raise ValueError("Default-mode canary scientific metric schema differs")
            if new["metrics"].get("decision/inner_critic_target_bn_running") != 0:
                raise ValueError("Default-mode canary used running target BN")
            matched_root = old["decision_index"] == 0
            for key in old_metrics:
                compare(old_metrics[key], new_metrics[key], key, matched_root=matched_root,
                        decision=[old.get("episode_id"), old["decision_index"]])
            report["matched_root_decisions"] += int(matched_root)
            report["decisions"] += 1
    return report


def before_smoke(plan, index, root, study):
    from types import SimpleNamespace
    if index != 0:
        raise ValueError("Target-BN comparison has exactly one new condition")
    baseline = baseline_condition(study)
    matrix_path = Path(root) / "default-compat.matrix.json"
    study.immutable_json(matrix_path, study.matrix_for([baseline]))
    default_plan = {**plan, "conditions": [baseline], "matrix": study.bind(matrix_path)}
    sweep.evaluate_cell(default_plan, 0, Path(root) / "default-compat", study, smoke=True,
                        policy=SimpleNamespace(validate_bundle=validate_default_bundle))
    study.immutable_json(Path(root) / "default-compatibility.json", compare_default_canary(plan, root, study))
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_cell(plan, index, root, study, *, smoke):
    return sweep.evaluate_cell(plan, index, root, study, smoke=smoke, policy=sys.modules[__name__])


def validate_worker(plan, index, job, study):
    root = worker_root(plan, index, job)
    if study.read(root / "default-compatibility.json") != compare_default_canary(plan, root, study):
        raise ValueError("Default-mode compatibility evidence changed")
    return sweep.validate_worker(plan, index, job, study, policy=sys.modules[__name__])


def label_curve(api, registry, cell, publication):
    label = "475k shared UTD2 · H2 J2 G6 · target BN running · all LR5e-5"
    api.flush()
    run = api.run(f"{publication.ENTITY}/{publication.PROJECT}/{registry['run_id']}")
    for key in ("update_sweep_id", "actor_lr_study_id", "horizon_study_id", "horizon_j_study_id", "h2_update_study_id"):
        run.config.pop(key, None)
    run.config.update(curve_label=label, target_bn_study_id=STUDY_ID)
    query = '''mutation LabelTargetBNRun($id:String!,$display_name:String!,$config:JSONString!){
        upsertBucket(input:{id:$id,displayName:$display_name,config:$config}){bucket{id displayName}}}'''
    variables = {"id": run.storage_id, "display_name": label, "config": run.json_config}
    service = getattr(api, "__dict__", {}).get("_service_api")
    if service is not None and hasattr(service, "execute_graphql"):
        response = service.execute_graphql(query, variables=variables)
    else:
        from wandb_gql import gql
        response = api.client.execute(gql(query), variable_values=variables)
    if response.get("upsertBucket", {}).get("bucket") != {"id": run.storage_id, "displayName": label}:
        raise RuntimeError("Target-BN display label was not acknowledged")


def worker(args, coordinator, study, publication, provenance):
    return sweep.worker(args, coordinator, study, publication, provenance, policy=sys.modules[__name__])


def submit(args, plan_path, state, coordinator, study):
    return sweep.submit(args, plan_path, state, coordinator, study, policy=sys.modules[__name__])


def publish_finished(args, plan, state, coordinator, study, publication, api, *, timeout=10800):
    return sweep.publish_finished(args, plan, state, coordinator, study, publication, api,
                                  timeout=timeout, policy=sys.modules[__name__])


def coordinate(args, coordinator, study, publication, provenance, *, api):
    return extension.coordinate(args, coordinator, study, publication, provenance,
                                api=api, policy=sys.modules[__name__])


def run(args):
    return extension.run(args, policy=sys.modules[__name__])


def main(argv=None):
    return extension.main(argv, policy=sys.modules[__name__])


if __name__ == "__main__":
    main()
