"""One frozen J2/G6 actor-LR comparison, reusing the completed 5e-5 baseline.

Only actor LR changes to 1e-4; critic and temperature retain 5e-5. Scientific
execution stays at 10852a8. The completed Stage 5 baseline is hash-bound and
never rerun or republished. This actor variant is excluded from J/G curves.
"""
from __future__ import annotations

import sys

import run_ambixqc_update_extension as extension

sweep = extension.sweep
continuation = extension.continuation
TOOLING_ROOT = extension.TOOLING_ROOT
SOURCE_SHA = extension.SOURCE_SHA
SCHEMA = "ambixqc-actor-lr-study-v1"
STAGE = "stage7"
ROUNDS = (2,)
UPDATES = (6,)
BASELINE_CONDITIONS = 27
BASELINE_EPISODES = 135
BASELINE_RUN_ID = "1a763d57173e4efdb3d228d123232e2f"
PARENT_TOOLING_SHA = "cffd5b289e8f54b62dae137611deb8f82f134408"
STUDY_ID = "ambixqc-actor-lr-20261001"
WORKER_TIME = "01:30:00"
expected_counts = sweep.expected_counts
validate_bundle = sweep.validate_bundle
worker_root = sweep.worker_root
evaluate_cell = sweep.evaluate_cell
validate_worker = sweep.validate_worker
parent_arguments = extension.parent_arguments


def baseline_condition(study):
    return next(cell for cell in sweep.conditions(study)
                if (cell["rounds"], cell["updates_per_round"]) == (2, 6))


def conditions(study):
    cell = baseline_condition(study)
    cell["selector"] = "controller/return_return_running_j2_g6_actor_1e4"
    cell["actor_lr"] = cell["settings"]["inner_actor_lr"] = 1e-4
    return [cell]


def split_conditions(study):
    return conditions(study), [baseline_condition(study)]


def matrix_for(study, cells):
    matrix = study.matrix_for(cells)
    matrix["description"] = "Single J2/G6 actor LR1e-4 comparison; exact completed actor LR5e-5 baseline reused."
    for cell in cells:
        matrix["comparisons"]["controller"]["variants"][cell["selector"].split("/")[1]]["description"] = (
            "Return/return, running actor/critic BN; H1 J2 G6 N256 B256 delay3; "
            "actor LR1e-4, critic and temperature LR5e-5; replay capacity1024.")
    return matrix


def _baseline_registry(evidence, study):
    plan = study.read(study.check_binding(evidence["stage5"]["plan"]))
    if plan["tooling"]["commit"] != PARENT_TOOLING_SHA:
        raise ValueError("Actor-rate baseline must come from the completed cffd5b2 sweep")
    binding = plan["runs"][baseline_condition(study)["selector"]]
    if study.read(study.check_binding(binding)).get("run_id") != BASELINE_RUN_ID:
        raise ValueError("Actor-rate comparison requires the exact published J2/G6 baseline run")
    return binding


def parent_evidence(args, study, publication):
    evidence, grid = extension.parent_evidence(args, study, publication)
    selected = [item for item in grid if item["condition"] == baseline_condition(study)]
    if len(selected) != 1:
        raise ValueError("Actor-rate comparison requires one exact completed J2/G6 baseline")
    return {"completed_sweep": evidence, "baseline_registry": _baseline_registry(evidence, study)}, selected


def prepare_plan(args, study, publication, provenance):
    return extension.prepare_plan(args, study, publication, provenance, policy=sys.modules[__name__])


def load_plan(path, study, source_sha, tooling_sha):
    plan = extension.load_plan(path, study, source_sha, tooling_sha, policy=sys.modules[__name__])
    if plan["parent"]["baseline_registry"] != _baseline_registry(plan["parent"]["completed_sweep"], study):
        raise ValueError("Actor-rate plan changed its baseline registry")
    return plan


def label_curve(api, registry, cell, publication):
    label = "475k shared UTD2 · return/running · J2 G6 · actor LR1e-4 · critic/temp LR5e-5"
    api.flush()
    run = api.run(f"{publication.ENTITY}/{publication.PROJECT}/{registry['run_id']}")
    run.config.pop("update_sweep_id", None)
    run.config.update(curve_label=label, actor_lr_study_id=STUDY_ID)
    # Run.update also rewrites summary. Only modify display/config metadata so
    # publication acknowledgements cannot be overwritten by cached summaries.
    query = '''mutation LabelActorRateRun($id:String!,$display_name:String!,$config:JSONString!){
        upsertBucket(input:{id:$id,displayName:$display_name,config:$config}){bucket{id displayName}}}'''
    variables = {"id": run.storage_id, "display_name": label, "config": run.json_config}
    service = getattr(api, "__dict__", {}).get("_service_api")
    if service is not None and hasattr(service, "execute_graphql"):
        response = service.execute_graphql(query, variables=variables)
    else:
        from wandb_gql import gql
        response = api.client.execute(gql(query), variable_values=variables)
    if response.get("upsertBucket", {}).get("bucket") != {"id": run.storage_id, "displayName": label}:
        raise RuntimeError("Actor-rate display label was not acknowledged")


def worker(args, coordinator, study, publication, provenance):
    return sweep.worker(args, coordinator, study, publication, provenance, policy=sys.modules[__name__])


def submit(args, plan_path, state, coordinator, study):
    return sweep.submit(args, plan_path, state, coordinator, study, policy=sys.modules[__name__])


def publish_finished(args, plan, state, coordinator, study, publication, api, *, timeout=7200):
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
