"""Compare H2/J2/G12 at delays 6 and 3, reusing the completed G6 baseline.

--completed-sweep-root identifies the completed Stage 9 H2/J sweep. Every
parent result is verified; only the exact Stage 8 H2/J2/G6 control is reused.
Scientific evaluation remains at the original clean 10852a8 source.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import run_ambixqc_horizon_j_sweep as rounds

horizon = rounds.horizon

extension = horizon.extension
sweep = horizon.sweep
continuation = horizon.continuation
TOOLING_ROOT = horizon.TOOLING_ROOT
SOURCE_SHA = horizon.SOURCE_SHA
SCHEMA = "ambixqc-h2-update-study-v1"
STAGE = "stage10"
ROUNDS = (2,)
UPDATES = (12,)
BASELINE_CONDITIONS = 33
BASELINE_EPISODES = 165
BASELINE_RUN_ID = "4b2d4437ffd84ca49093e303f794a11b"
PARENT_TOOLING_SHA = "543f5c9776b26a3bfb459bf22e157a833914ad61"
STUDY_ID = "ambixqc-h2-updates-20261001"
WORKER_TIME = "02:00:00"
expected_counts = sweep.expected_counts
worker_root = sweep.worker_root
parent_arguments = extension.parent_arguments


def baseline_condition(study):
    return horizon.conditions(study)[0]


def conditions(study):
    cells = []
    for delay in (6, 3):
        cell = deepcopy(baseline_condition(study))
        cell.update(updates_per_round=12, policy_delay=delay,
                    selector=f"controller/return_return_running_h2_j2_g12_delay{delay}")
        cell["settings"].update(inner_updates_per_round=12, inner_policy_delay=delay)
        cells.append(cell)
    return cells


def split_conditions(study):
    return conditions(study), [baseline_condition(study)]


def matrix_for(study, cells):
    matrix = study.matrix_for(cells)
    matrix["description"] = "H2/J2 update-dose comparison; exact completed G6/delay3 control reused."
    for cell in cells:
        matrix["comparisons"]["controller"]["variants"][cell["selector"].split("/")[1]]["description"] = (
            f"Return/return, running actor/critic BN; H2 J2 G12 delay{cell['policy_delay']} N256 B256; "
            "all learning rates 5e-5; replay capacity 1024. Only final-depth rows use the frozen outer tail.")
    return matrix


def _baseline_registry(evidence, study):
    binding = rounds._baseline_registry(evidence["previous"], study)
    if study.read(study.check_binding(binding)).get("run_id") != BASELINE_RUN_ID:
        raise ValueError("H2 update comparison requires the exact published Stage 8 J2/G6 run")
    return binding


def parent_evidence(args, study, publication):
    root = args.completed_sweep_root
    state_path, complete_path = root / "coordinator-state.json", root / "COMPLETE.json"
    state, complete = study.read(state_path), study.read(complete_path)
    if ({k: v for k, v in complete.items() if k != "workspace"} != state
            or state.get("schema") != rounds.SCHEMA or state.get("source_sha") != args.source_sha
            or state.get("progress_run_id") != args.progress_run_id or state.get("stage9_complete") is not True
            or state.get("tooling", {}).get("commit") != PARENT_TOOLING_SHA
            or not str(state.get("job", "")).isdigit() or state.get("failures")
            or (root / "FAILED.json").exists()):
        raise ValueError("Require the complete, published 543f5c9 Stage 9 H2/J sweep")
    plan_path = study.check_binding(state["plan"])
    plan = rounds.load_plan(plan_path, study, args.source_sha, PARENT_TOOLING_SHA)
    if (plan_path != (root / "stage9-plan.json").resolve() or plan.get("result_root") != str(root)
            or plan["tooling"] != state["tooling"] or plan["execution"] != state["execution"]
            or plan["progress_run_id"] != args.progress_run_id
            or plan["completed_sweep_root"] != state["completed_sweep_root"]):
        raise ValueError("Stage 9 plan differs from its completed coordinator")
    prior_args = rounds.parent_arguments(plan)
    previous, baseline = rounds.parent_evidence(prior_args, study, publication)
    if previous != plan["parent"] or baseline != plan["reused"]:
        raise ValueError("Stage 9 inherited baseline or parent changed")
    for key in ("manifest", "reference_index", "smoke_root", "checkpoint_root"):
        setattr(args, key, getattr(prior_args, key))
    if (plan["inputs"] != {key: study.bind(getattr(args, key)) for key in ("manifest", "reference_index")}
            or plan["smoke_root"] != str(args.smoke_root)
            or plan["checkpoint_root"] != (str(args.checkpoint_root) if args.checkpoint_root else None)):
        raise ValueError("Stage 9 inherited checkpoint or reference inputs differ")
    row = study.screen.select_checkpoint(args.manifest, checkpoint_root=args.checkpoint_root)
    if str(study.screen.select_reference(args.reference_index, row, args.manifest)) != plan["reference_bundle"]:
        raise ValueError("Stage 9 paired reference differs")
    results_path = study.check_binding(state["stage9_results"])
    result = study.read(results_path)
    if (results_path != (root / "stage9-results.json").resolve()
            or result.get("schema") != rounds.SCHEMA or result.get("stage") != "stage9"
            or result.get("source_sha") != args.source_sha or result.get("plan") != study.bind(plan_path)
            or result.get("reused") != baseline or len(result.get("entries", [])) != 4):
        raise ValueError("Stage 9 results are incomplete or incompatible")
    receipts, published = [], {}
    for index, cell in enumerate(plan["conditions"]):
        entry = rounds.validate_worker(plan, index, state["job"], study)
        if result["entries"][index] != entry or entry["condition"] != cell:
            raise ValueError("Stage 9 result differs from its validated H2 worker")
        run_dir = Path(plan["runs"][cell["selector"]]["path"]).parent
        binding, receipt = sweep.publication_receipt(run_dir, cell["selector"], study,
                                                     publication, args.source_sha, entry)
        receipts.append(binding)
        published[cell["selector"]] = receipt
    if state.get("published") != published:
        raise ValueError("Stage 9 lacks all four acknowledged publications")
    if [item["condition"] for item in baseline] != [baseline_condition(study)]:
        raise ValueError("Stage 9 must retain the exact H2/J2/G6 baseline")
    evidence = {"previous": previous, "stage9": {"state": study.bind(state_path),
        "complete": study.bind(complete_path), "plan": study.bind(plan_path),
        "results": study.bind(results_path), "publication_receipts": receipts}}
    evidence["baseline_registry"] = _baseline_registry(evidence, study)
    return evidence, baseline


def prepare_plan(args, study, publication, provenance):
    return extension.prepare_plan(args, study, publication, provenance, policy=sys.modules[__name__])


def load_plan(path, study, source_sha, tooling_sha):
    plan = extension.load_plan(path, study, source_sha, tooling_sha, policy=sys.modules[__name__])
    if plan["parent"]["baseline_registry"] != _baseline_registry(plan["parent"], study):
        raise ValueError("H2 update study changed its exact Stage 8 baseline registry")
    return plan


def validate_bundle(plan, index, root, study, smoke):
    if plan["conditions"][index] not in conditions(study):
        raise ValueError("H2 update validation requires one of the two new conditions")
    result = sweep.validate_bundle(plan, index, root, study, smoke)
    result = horizon.validate_h2_boundaries(plan, index, root, study, result)
    # Record and verify the changed actor cadence as well as optimizer totals.
    import gzip
    import json
    delay = plan["conditions"][index]["settings"]["inner_policy_delay"]
    bundle = Path(root) / "bundle"
    for relative in study.read(bundle / "manifest.json")["runs"][0]["trace_files"]:
        with gzip.open(bundle / relative, "rt") as stream:
            if any(json.loads(line)["metrics"].get("decision/inner_policy_delay") != delay for line in stream):
                raise ValueError("H2 update trace policy delay differs")
    return {**result, "policy_delay": delay}


def evaluate_cell(plan, index, root, study, *, smoke):
    return sweep.evaluate_cell(plan, index, root, study, smoke=smoke, policy=sys.modules[__name__])


def validate_worker(plan, index, job, study):
    return sweep.validate_worker(plan, index, job, study, policy=sys.modules[__name__])


def label_curve(api, registry, cell, publication):
    label = f"475k shared UTD2 · return/running · H2 J2 G12 delay{cell['policy_delay']} · all LR5e-5"
    api.flush()
    run = api.run(f"{publication.ENTITY}/{publication.PROJECT}/{registry['run_id']}")
    for key in ("update_sweep_id", "actor_lr_study_id", "horizon_study_id", "horizon_j_study_id"):
        run.config.pop(key, None)
    run.config.update(curve_label=label, h2_update_study_id=STUDY_ID)
    query = '''mutation LabelH2UpdateRun($id:String!,$display_name:String!,$config:JSONString!){
        upsertBucket(input:{id:$id,displayName:$display_name,config:$config}){bucket{id displayName}}}'''
    variables = {"id": run.storage_id, "display_name": label, "config": run.json_config}
    service = getattr(api, "__dict__", {}).get("_service_api")
    if service is not None and hasattr(service, "execute_graphql"):
        response = service.execute_graphql(query, variables=variables)
    else:
        from wandb_gql import gql
        response = api.client.execute(gql(query), variable_values=variables)
    if response.get("upsertBucket", {}).get("bucket") != {"id": run.storage_id, "displayName": label}:
        raise RuntimeError("H2 update display label was not acknowledged")


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
