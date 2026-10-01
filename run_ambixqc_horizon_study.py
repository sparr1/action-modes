"""One frozen H2/J2/G6 comparison, reusing the completed H1 baseline.

Only rollout horizon changes; all learning rates remain 5e-5. Scientific
execution stays at 10852a8. Earlier-depth transitions use the inner Bellman
target and final-depth transitions use the frozen outer return critic.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import run_ambixqc_actor_lr_study as actor

extension = actor.extension
sweep = actor.sweep
continuation = actor.continuation
TOOLING_ROOT = actor.TOOLING_ROOT
SOURCE_SHA = actor.SOURCE_SHA
SCHEMA = "ambixqc-horizon-study-v1"
STAGE = "stage8"
ROUNDS = (2,)
UPDATES = (6,)
BASELINE_CONDITIONS = 28
BASELINE_EPISODES = 140
BASELINE_RUN_ID = actor.BASELINE_RUN_ID
PARENT_TOOLING_SHA = actor.PARENT_TOOLING_SHA
STUDY_ID = "ambixqc-horizon-20261001"
WORKER_TIME = "01:30:00"
expected_counts = sweep.expected_counts
worker_root = sweep.worker_root
baseline_condition = actor.baseline_condition
parent_arguments = actor.parent_arguments
parent_evidence = actor.parent_evidence


def conditions(study):
    cell = baseline_condition(study)
    cell["selector"] = "controller/return_return_running_h2_j2_g6_actor_high"
    cell["horizon"] = cell["settings"]["inner_rollout_horizon"] = 2
    return [cell]


def split_conditions(study):
    return conditions(study), [baseline_condition(study)]


def matrix_for(study, cells):
    matrix = study.matrix_for(cells)
    matrix["description"] = "Single H2/J2/G6 comparison; exact completed H1 baseline reused."
    for cell in cells:
        matrix["comparisons"]["controller"]["variants"][cell["selector"].split("/")[1]]["description"] = (
            "Return/return, running actor/critic BN; H2 J2 G6 N256 B256 delay3; "
            "all learning rates5e-5; replay capacity1024. Only final-depth rows use the frozen outer tail.")
    return matrix


def prepare_plan(args, study, publication, provenance):
    return extension.prepare_plan(args, study, publication, provenance, policy=sys.modules[__name__])


def load_plan(path, study, source_sha, tooling_sha):
    plan = extension.load_plan(path, study, source_sha, tooling_sha, policy=sys.modules[__name__])
    if plan["parent"]["baseline_registry"] != actor._baseline_registry(plan["parent"]["completed_sweep"], study):
        raise ValueError("Horizon plan changed its exact H1 baseline registry")
    return plan


def validate_bundle(plan, index, root, study, smoke):
    """Preserve every shared check and verify the two-depth boundary contract."""
    result = sweep.validate_bundle(plan, index, root, study, smoke)
    cell = plan["conditions"][index]
    if cell != conditions(study)[0]:
        raise ValueError("Horizon validation requires the exact H2 intervention")
    cfg = cell["settings"]
    counts = expected_counts(cell)
    rollouts = cfg["inner_rounds"] * cfg["inner_rollouts_per_round"]
    required = {
        "decision/inner_rollout_count": rollouts,
        "decision/inner_rollout_len_min": 2,
        "decision/inner_rollout_len_mean": 2,
        "decision/inner_rollout_len_max": 2,
        "decision/inner_rollout_len_std": 0,
        "decision/inner_buffer_capacity": cfg["inner_replay_capacity"],
        "decision/inner_termination_rate": 0,
        "decision/inner_terminal_bootstrap_outer": 1,
        "decision/inner_outer_terminal_boundary_rows": rollouts,
        # The frozen outer policy/Q are evaluated on the complete update batch;
        # only the sampled final-depth mask selects their targets afterwards.
        "decision/inner_outer_terminal_policy_evaluations": counts["replay_draws"],
        "decision/inner_outer_terminal_q_evaluations": counts["replay_draws"],
    }
    bundle = Path(root) / "bundle"
    run = study.read(bundle / "manifest.json")["runs"][0]
    for relative in run["trace_files"]:
        with gzip.open(bundle / relative, "rt") as stream:
            for line in stream:
                metrics = json.loads(line)["metrics"]
                if any(metrics.get(key) != value for key, value in required.items()):
                    raise ValueError("H2 rollout or outer-terminal boundary diagnostics differ")
                sampled = metrics.get("decision/inner_outer_terminal_bootstrap_rows")
                if (type(sampled) not in (int, float) or not 0 <= sampled <= counts["replay_draws"]
                        or int(sampled) != sampled):
                    raise ValueError("H2 sampled outer-terminal count must be an integer within replay draws")
    return {**result, "rollout_horizon": 2, "outer_terminal_boundary_rows_per_decision": rollouts}


def evaluate_cell(plan, index, root, study, *, smoke):
    return sweep.evaluate_cell(plan, index, root, study, smoke=smoke, policy=sys.modules[__name__])


def validate_worker(plan, index, job, study):
    return sweep.validate_worker(plan, index, job, study, policy=sys.modules[__name__])


def label_curve(api, registry, cell, publication):
    label = "475k shared UTD2 · return/running · H2 J2 G6 · all LR5e-5"
    api.flush()
    run = api.run(f"{publication.ENTITY}/{publication.PROJECT}/{registry['run_id']}")
    run.config.pop("update_sweep_id", None)
    run.config.pop("actor_lr_study_id", None)
    run.config.update(curve_label=label, horizon_study_id=STUDY_ID)
    # Update metadata only: Run.update can overwrite a newer result summary.
    query = '''mutation LabelHorizonRun($id:String!,$display_name:String!,$config:JSONString!){
        upsertBucket(input:{id:$id,displayName:$display_name,config:$config}){bucket{id displayName}}}'''
    variables = {"id": run.storage_id, "display_name": label, "config": run.json_config}
    service = getattr(api, "__dict__", {}).get("_service_api")
    if service is not None and hasattr(service, "execute_graphql"):
        response = service.execute_graphql(query, variables=variables)
    else:
        from wandb_gql import gql
        response = api.client.execute(gql(query), variable_values=variables)
    if response.get("upsertBucket", {}).get("bucket") != {"id": run.storage_id, "displayName": label}:
        raise RuntimeError("Horizon display label was not acknowledged")


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
