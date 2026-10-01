"""Extend H2/G6 across J1/4/6/8, reusing the completed Stage 8 J2 result.

--completed-sweep-root identifies the completed Stage 8 horizon comparison,
not the older Stage 5 J/G sweep. Only orchestration changes; evaluation remains
at the original clean 10852a8 source and pinned 475k checkpoint.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import run_ambixqc_horizon_study as horizon

extension = horizon.extension
sweep = horizon.sweep
continuation = horizon.continuation
TOOLING_ROOT = horizon.TOOLING_ROOT
SOURCE_SHA = horizon.SOURCE_SHA
SCHEMA = "ambixqc-horizon-j-sweep-v1"
STAGE = "stage9"
ROUNDS = (1, 2, 4, 6, 8)
UPDATES = (6,)
BASELINE_CONDITIONS = 29
BASELINE_EPISODES = 145
BASELINE_RUN_ID = "4b2d4437ffd84ca49093e303f794a11b"
PARENT_TOOLING_SHA = "dd0cd9a8ac5a054ffdd61cd3fa358a48dd6de5ef"
STUDY_ID = "ambixqc-h2-j-20261001"
WORKER_TIME = "04:00:00"
expected_counts = sweep.expected_counts
worker_root = sweep.worker_root
parent_arguments = extension.parent_arguments


def baseline_condition(study):
    return horizon.conditions(study)[0]


def conditions(study):
    baseline = baseline_condition(study)
    cells = []
    for j in (8, 6, 4, 1):
        cell = deepcopy(baseline)
        cell.update(rounds=j, selector=f"controller/return_return_running_h2_j{j}_g6_actor_high")
        cell["settings"].update(inner_rounds=j, inner_replay_capacity=max(1024, 512*j))
        cells.append(cell)
    return cells


def split_conditions(study):
    return conditions(study), [baseline_condition(study)]


def matrix_for(study, cells):
    matrix = study.matrix_for(cells)
    matrix["description"] = "H2/G6 collection-round sweep; exact completed J2 result reused."
    for cell in cells:
        matrix["comparisons"]["controller"]["variants"][cell["selector"].split("/")[1]]["description"] = (
            f"Return/return, running actor/critic BN; H2 J{cell['rounds']} G6 N256 B256 delay3; "
            f"all learning rates 5e-5; replay capacity {cell['settings']['inner_replay_capacity']}. "
            "Only final-depth rows use the frozen outer tail.")
    return matrix


def _baseline_registry(evidence, study):
    plan = study.read(study.check_binding(evidence["stage8"]["plan"]))
    if plan["tooling"]["commit"] != PARENT_TOOLING_SHA:
        raise ValueError("H2/J2 baseline must come from the completed dd0cd9a horizon study")
    binding = plan["runs"][baseline_condition(study)["selector"]]
    if study.read(study.check_binding(binding)).get("run_id") != BASELINE_RUN_ID:
        raise ValueError("H2 J sweep requires the exact published Stage 8 J2 run")
    return binding


def parent_evidence(args, study, publication):
    root = args.completed_sweep_root
    state_path, complete_path = root / "coordinator-state.json", root / "COMPLETE.json"
    state, complete = study.read(state_path), study.read(complete_path)
    if ({k: v for k, v in complete.items() if k != "workspace"} != state
            or state.get("schema") != horizon.SCHEMA or state.get("source_sha") != args.source_sha
            or state.get("progress_run_id") != args.progress_run_id or state.get("stage8_complete") is not True
            or state.get("tooling", {}).get("commit") != PARENT_TOOLING_SHA
            or not str(state.get("job", "")).isdigit() or state.get("failures")
            or (root / "FAILED.json").exists()):
        raise ValueError("Require the complete, published dd0cd9a Stage 8 horizon comparison")
    plan_path = study.check_binding(state["plan"])
    plan = horizon.load_plan(plan_path, study, args.source_sha, PARENT_TOOLING_SHA)
    if (plan_path != (root / "stage8-plan.json").resolve() or plan.get("result_root") != str(root)
            or plan["tooling"] != state["tooling"] or plan["execution"] != state["execution"]
            or plan["progress_run_id"] != args.progress_run_id
            or plan["completed_sweep_root"] != state["completed_sweep_root"]):
        raise ValueError("Stage 8 plan differs from its completed coordinator")
    prior_args = horizon.parent_arguments(plan)
    previous, baseline = horizon.parent_evidence(prior_args, study, publication)
    if previous != plan["parent"] or baseline != plan["reused"]:
        raise ValueError("Stage 8 inherited baseline or parent changed")
    for key in ("manifest", "reference_index", "smoke_root", "checkpoint_root"):
        setattr(args, key, getattr(prior_args, key))
    if (plan["inputs"] != {key: study.bind(getattr(args, key)) for key in ("manifest", "reference_index")}
            or plan["smoke_root"] != str(args.smoke_root)
            or plan["checkpoint_root"] != (str(args.checkpoint_root) if args.checkpoint_root else None)):
        raise ValueError("Stage 8 inherited checkpoint or reference inputs differ")
    row = study.screen.select_checkpoint(args.manifest, checkpoint_root=args.checkpoint_root)
    if str(study.screen.select_reference(args.reference_index, row, args.manifest)) != plan["reference_bundle"]:
        raise ValueError("Stage 8 paired reference differs")
    results_path = study.check_binding(state["stage8_results"])
    result = study.read(results_path)
    if (results_path != (root / "stage8-results.json").resolve()
            or result.get("schema") != horizon.SCHEMA or result.get("stage") != "stage8"
            or result.get("source_sha") != args.source_sha or result.get("plan") != study.bind(plan_path)
            or result.get("reused") != baseline or len(result.get("entries", [])) != 1):
        raise ValueError("Stage 8 results are incomplete or incompatible")
    entry = horizon.validate_worker(plan, 0, state["job"], study)
    cell = baseline_condition(study)
    if result["entries"] != [entry] or entry["condition"] != cell:
        raise ValueError("Stage 8 result differs from its validated H2/J2 worker")
    run_dir = Path(plan["runs"][cell["selector"]]["path"]).parent
    receipt_binding, receipt = sweep.publication_receipt(run_dir, cell["selector"], study,
                                                       publication, args.source_sha, entry)
    if state.get("published") != {cell["selector"]: receipt}:
        raise ValueError("Stage 8 lacks its acknowledged H2/J2 publication")
    evidence = {"previous": previous, "stage8": {"state": study.bind(state_path),
        "complete": study.bind(complete_path), "plan": study.bind(plan_path),
        "results": study.bind(results_path), "publication_receipt": receipt_binding}}
    evidence["baseline_registry"] = _baseline_registry(evidence, study)
    return evidence, [{"condition": cell, "result": entry}]


def prepare_plan(args, study, publication, provenance):
    return extension.prepare_plan(args, study, publication, provenance, policy=sys.modules[__name__])


def load_plan(path, study, source_sha, tooling_sha):
    plan = extension.load_plan(path, study, source_sha, tooling_sha, policy=sys.modules[__name__])
    if plan["parent"]["baseline_registry"] != _baseline_registry(plan["parent"], study):
        raise ValueError("H2 J sweep changed its exact Stage 8 baseline registry")
    return plan


def validate_bundle(plan, index, root, study, smoke):
    if plan["conditions"][index] not in conditions(study):
        raise ValueError("H2 J validation requires one of the four new conditions")
    result = sweep.validate_bundle(plan, index, root, study, smoke)
    return horizon.validate_h2_boundaries(plan, index, root, study, result)


def evaluate_cell(plan, index, root, study, *, smoke):
    return sweep.evaluate_cell(plan, index, root, study, smoke=smoke, policy=sys.modules[__name__])


def validate_worker(plan, index, job, study):
    return sweep.validate_worker(plan, index, job, study, policy=sys.modules[__name__])


def label_curve(api, registry, cell, publication):
    label = f"475k shared UTD2 · return/running · H2 J{cell['rounds']} G6 · all LR5e-5"
    api.flush()
    run = api.run(f"{publication.ENTITY}/{publication.PROJECT}/{registry['run_id']}")
    for key in ("update_sweep_id", "actor_lr_study_id", "horizon_study_id"):
        run.config.pop(key, None)
    run.config.update(curve_label=label, horizon_j_study_id=STUDY_ID)
    query = '''mutation LabelHorizonJSweepRun($id:String!,$display_name:String!,$config:JSONString!){
        upsertBucket(input:{id:$id,displayName:$display_name,config:$config}){bucket{id displayName}}}'''
    variables = {"id": run.storage_id, "display_name": label, "config": run.json_config}
    service = getattr(api, "__dict__", {}).get("_service_api")
    if service is not None and hasattr(service, "execute_graphql"):
        response = service.execute_graphql(query, variables=variables)
    else:
        from wandb_gql import gql
        response = api.client.execute(gql(query), variable_values=variables)
    if response.get("upsertBucket", {}).get("bucket") != {"id": run.storage_id, "displayName": label}:
        raise RuntimeError("H2 J-sweep display label was not acknowledged")


def worker(args, coordinator, study, publication, provenance):
    return sweep.worker(args, coordinator, study, publication, provenance, policy=sys.modules[__name__])


def submit(args, plan_path, state, coordinator, study):
    return sweep.submit(args, plan_path, state, coordinator, study, policy=sys.modules[__name__])


def publish_finished(args, plan, state, coordinator, study, publication, api, *, timeout=18000):
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
