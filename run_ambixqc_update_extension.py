"""Extend the completed frozen J/G sweep with exactly eight J8/G12 cells.

All twelve original grid cells are verified and reused. Scientific execution
remains pinned to 10852a8; this module owns only the continuation plan and
publication, sharing the validated GPU lifecycle with the original sweep.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import run_ambixqc_update_sweep as sweep

critic = sweep.critic
continuation = sweep.continuation
TOOLING_ROOT = Path(__file__).resolve().parent
SOURCE_SHA = sweep.SOURCE_SHA
SCHEMA = "ambixqc-update-extension-v1"
STAGE = "stage6"
BASELINE_CONDITIONS = 19
BASELINE_EPISODES = 95
ROUNDS = (1, 2, 4, 6, 8)
UPDATES = (3, 6, 9, 12)
REUSED = {(j, g) for j in sweep.ROUNDS for g in sweep.UPDATES}
expected_counts = sweep.expected_counts
validate_bundle = sweep.validate_bundle
worker_root = sweep.worker_root
evaluate_cell = sweep.evaluate_cell
validate_worker = sweep.validate_worker
label_curve = sweep.label_curve


def conditions(study):
    old = {(cell["rounds"], cell["updates_per_round"]): cell for cell in sweep.conditions(study)}
    cells = []
    for j in ROUNDS:
        for g in UPDATES:
            if (j, g) in old:
                cell = old[j, g]
            else:
                cell = study.condition("running", "return_return", 1 if j == 1 else 4, 5e-5)
                cell.update(rounds=j, updates_per_round=g,
                            selector=f"controller/return_return_running_j{j}_g{g}_actor_high")
                cell["settings"].update(inner_rounds=j, inner_updates_per_round=g,
                                        inner_replay_capacity=max(1024, 256*j))
            cells.append(cell)
    return cells


def split_conditions(study):
    cells = conditions(study)
    new = [cell for cell in cells if (cell["rounds"], cell["updates_per_round"]) not in REUSED]
    new.sort(key=lambda c: (c["rounds"]*c["updates_per_round"], c["rounds"], c["updates_per_round"]), reverse=True)
    return new, [cell for cell in cells if (cell["rounds"], cell["updates_per_round"]) in REUSED]


def matrix_for(study, cells):
    matrix = sweep.matrix_for(study, cells)
    matrix["description"] = "Frozen J8/G12 extension; all twelve completed J1/2/4/6 by G3/6/9 cells reused."
    return matrix


def parent_arguments(plan):
    return SimpleNamespace(completed_sweep_root=Path(plan["completed_sweep_root"]),
                           source_sha=plan["source_sha"], progress_run_id=plan["progress_run_id"])


def parent_evidence(args, study, publication):
    root = args.completed_sweep_root
    state_path, complete_path = root / "coordinator-state.json", root / "COMPLETE.json"
    state, complete = study.read(state_path), study.read(complete_path)
    if ({k: v for k, v in complete.items() if k != "workspace"} != state
            or state.get("schema") != sweep.SCHEMA or state.get("source_sha") != args.source_sha
            or state.get("progress_run_id") != args.progress_run_id or state.get("stage5_complete") is not True
            or not str(state.get("job", "")).isdigit() or state.get("failures")
            or (root / "FAILED.json").exists() or state.get("rounds") != list(sweep.ROUNDS)):
        raise ValueError("Require the complete, published full Stage 5 sweep")
    plan_path = study.check_binding(state["plan"])
    plan = sweep.load_plan(plan_path, study, args.source_sha, state["tooling"]["commit"])
    if (plan_path != (root / "stage5-plan.json").resolve() or plan.get("result_root") != str(root)
            or plan.get("rounds") != list(sweep.ROUNDS) or len(plan["conditions"]) != 10
            or plan["tooling"] != state["tooling"] or plan["execution"] != state["execution"]
            or plan["progress_run_id"] != args.progress_run_id
            or plan["parent_root"] != state["parent_root"]
            or plan["previous_stage_root"] != state["previous_stage_root"]):
        raise ValueError("Completed sweep plan differs from its coordinator and full grid")
    previous_args = sweep.parent_arguments(plan)
    previous, baseline = sweep.parent_evidence(previous_args, study, publication)
    if previous != plan["parent"] or baseline != plan["reused"]:
        raise ValueError("Completed sweep inherited parent or reused results changed")
    for key in ("manifest", "reference_index", "smoke_root", "checkpoint_root"):
        setattr(args, key, getattr(previous_args, key))
    if (plan["inputs"] != {key: study.bind(getattr(args, key)) for key in ("manifest", "reference_index")}
            or plan["smoke_root"] != str(args.smoke_root)
            or plan["checkpoint_root"] != (str(args.checkpoint_root) if args.checkpoint_root else None)):
        raise ValueError("Completed sweep inherited checkpoint or reference inputs differ")
    row = study.screen.select_checkpoint(args.manifest, checkpoint_root=args.checkpoint_root)
    if str(study.screen.select_reference(args.reference_index, row, args.manifest)) != plan["reference_bundle"]:
        raise ValueError("Completed sweep paired reference differs")
    results_path = study.check_binding(state["stage5_results"])
    result = study.read(results_path)
    if (results_path != (root / "stage5-results.json").resolve()
            or result.get("schema") != sweep.SCHEMA or result.get("stage") != "stage5"
            or result.get("source_sha") != args.source_sha or result.get("plan") != study.bind(plan_path)
            or result.get("reused") != baseline or len(result.get("entries", [])) != 10):
        raise ValueError("Completed sweep results are incomplete or incompatible")
    receipts, published = [], {}
    entries = [item["result"] for item in baseline]
    for index, cell in enumerate(plan["conditions"]):
        entry = sweep.validate_worker(plan, index, state["job"], study)
        if result["entries"][index] != entry:
            raise ValueError("Completed sweep result differs from its validated GPU task")
        run_dir = Path(plan["runs"][cell["selector"]]["path"]).parent
        binding, receipt = sweep.publication_receipt(run_dir, cell["selector"], study, publication, args.source_sha, entry)
        receipts.append(binding)
        published[cell["selector"]] = receipt
        entries.append(entry)
    if state.get("published") != published:
        raise ValueError("Completed sweep lacks all ten acknowledged publications")
    reused = []
    for cell in split_conditions(study)[1]:
        matches = [entry for entry in entries if entry["condition"]["selector"] == cell["selector"]
                   and entry["condition"]["settings"] == cell["settings"]]
        if len(matches) != 1:
            raise ValueError("Every completed grid cell must have exactly one compatible result")
        reused.append({"condition": cell, "result": matches[0]})
    evidence = {"previous": previous, "stage5": {"state": study.bind(state_path),
        "complete": study.bind(complete_path), "plan": study.bind(plan_path),
        "results": study.bind(results_path), "publication_receipts": receipts}}
    return evidence, reused


def prepare_plan(args, study, publication, provenance, *, policy=None):
    policy = policy or sys.modules[__name__]
    stage = policy.STAGE
    evidence, reused = policy.parent_evidence(args, study, publication)
    study.verify_smokes(args.smoke_root, args.manifest, getattr(policy, "PARENT_SOURCE_SHA", args.source_sha))
    row = study.screen.select_checkpoint(args.manifest, checkpoint_root=args.checkpoint_root)
    reference = study.screen.select_reference(args.reference_index, row, args.manifest)
    cells, _ = policy.split_conditions(study)
    matrix_path = args.result_root / (stage + ".matrix.json")
    study.immutable_json(matrix_path, policy.matrix_for(study, cells))
    from evaluate_ambi_checkpoint import evaluate_matrix
    runs = {}
    for index, cell in enumerate(cells):
        with tempfile.TemporaryDirectory(dir=args.result_root) as temporary:
            prepared = evaluate_matrix(matrix_path, row["path"], selectors=[cell["selector"]],
                checkpoint_inventory=args.manifest, source_run=row["source_run"],
                eval_series_spec_dir=Path(temporary) / "specs")
            if set(prepared.get("specs", {})) != {cell["selector"]}:
                raise ValueError("Each extension cell must resolve exactly one scientific identity")
            spec = study.read(prepared["specs"][cell["selector"]])
        study.immutable_json(args.result_root / "specs" / f"{index}.json", spec)
        registry = publication.allocate_curve(args.result_root, spec, stage, cell["selector"])
        runs[cell["selector"]] = study.bind(Path(registry["run_dir"]) / "run.json")
    plan = {"schema": policy.SCHEMA, "stage": stage, "source_sha": args.source_sha, **provenance,
        "campaign": publication.CAMPAIGN, "checkpoint_sha256": study.CHECKPOINT_SHA, "checkpoint_step": 475000,
        "rounds": list(policy.ROUNDS), "updates": list(policy.UPDATES), "conditions": cells, "reused": reused, "parent": evidence,
        "completed_sweep_root": str(args.completed_sweep_root), "progress_run_id": args.progress_run_id,
        "result_root": str(args.result_root), "matrix": study.bind(matrix_path), "runs": runs,
        "inputs": {key: study.bind(getattr(args, key)) for key in ("manifest", "reference_index")},
        "checkpoint_root": str(args.checkpoint_root) if args.checkpoint_root else None,
        "reference_bundle": str(reference), "smoke_root": str(args.smoke_root),
        "environment_seeds": study.SEEDS, "controller_seed": 12345, "max_steps": 500,
        "smoke_seeds": [101, 102], "smoke_max_steps": 3,
        "baseline_conditions": policy.BASELINE_CONDITIONS, "baseline_episodes": policy.BASELINE_EPISODES}
    plan["plan_sha256"] = study.digest(plan)
    path = args.result_root / (stage + "-plan.json")
    study.immutable_json(path, plan)
    return path, plan


def load_plan(path, study, source_sha, tooling_sha, *, policy=None):
    policy = policy or sys.modules[__name__]
    plan = study.read(path)
    cells, reused = policy.split_conditions(study)
    signed = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if (plan.get("schema") != policy.SCHEMA or plan.get("stage") != policy.STAGE
            or plan.get("campaign") != "ambixqc-bn-study-20260929" or plan.get("plan_sha256") != study.digest(signed)
            or source_sha != getattr(policy, "SOURCE_SHA", SOURCE_SHA) or plan.get("source_sha") != source_sha
            or plan.get("tooling", {}).get("commit") != tooling_sha
            or plan.get("execution", {}).get("commit") != source_sha
            or plan.get("checkpoint_sha256") != study.CHECKPOINT_SHA or plan.get("checkpoint_step") != 475000
            or plan.get("rounds") != list(policy.ROUNDS) or plan.get("updates") != list(policy.UPDATES)
            or plan.get("conditions") != cells or [item.get("condition") for item in plan.get("reused", [])] != reused
            or plan.get("environment_seeds") != study.SEEDS or plan.get("controller_seed") != 12345
            or plan.get("max_steps") != 500 or plan.get("smoke_seeds") != [101, 102] or plan.get("smoke_max_steps") != 3
            or plan.get("baseline_conditions") != policy.BASELINE_CONDITIONS
            or plan.get("baseline_episodes") != policy.BASELINE_EPISODES):
        raise ValueError("Extension plan changed source, controlled settings, reuse or protocol")
    critic.bindings_unchanged(plan, study)
    if study.read(plan["matrix"]["path"]) != policy.matrix_for(study, cells):
        raise ValueError("Extension matrix differs from its plan")
    if (set(plan["runs"]) != {cell["selector"] for cell in cells}
            or len({binding["path"] for binding in plan["runs"].values()}) != len(cells)):
        raise ValueError("Only the planned new cells may have distinct allocated registries")
    for item in plan["reused"]:
        old = item["result"]["condition"]
        if old["settings"] != item["condition"]["settings"] or old["selector"] != item["condition"]["selector"]:
            raise ValueError("Reused extension result is not exactly compatible")
    return plan


def worker(args, coordinator, study, publication, provenance):
    return sweep.worker(args, coordinator, study, publication, provenance, policy=sys.modules[__name__])


def submit(args, plan_path, state, coordinator, study):
    return sweep.submit(args, plan_path, state, coordinator, study, policy=sys.modules[__name__])


def publish_finished(args, plan, state, coordinator, study, publication, api, *, timeout=25200):
    return sweep.publish_finished(args, plan, state, coordinator, study, publication, api,
                                  timeout=timeout, policy=sys.modules[__name__])


def coordinate(args, coordinator, study, publication, provenance, *, api, policy=None):
    policy = policy or sys.modules[__name__]
    stage = policy.STAGE
    workspace = continuation.verify_workspace(api, args.workspace_spec, publication)
    expected = {"schema": policy.SCHEMA, "source_sha": args.source_sha, **provenance,
        "completed_sweep_root": str(args.completed_sweep_root), "max_concurrent": args.max_concurrent,
        "workspace_spec": study.bind(args.workspace_spec), "progress_run_id": args.progress_run_id,
        "gpu_type": args.gpu_type, "worker_launcher": study.bind(args.worker_launcher)}
    state_path = args.result_root / "coordinator-state.json"
    state = study.read(state_path) if state_path.exists() else {**expected, "published": {}}
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError("Extension state belongs to different source, scope or publication inputs")
    coordinator.atomic_json(state_path, state, overwrite=True)
    plan_path, plan = policy.prepare_plan(args, study, publication, provenance)
    policy.load_plan(plan_path, study, args.source_sha, args.tooling_sha)
    state["plan"] = study.bind(plan_path)
    for cell in plan["conditions"]:
        policy.label_curve(api, study.read(plan["runs"][cell["selector"]]["path"]), cell, publication)
    total_conditions = plan["baseline_conditions"] + len(plan["conditions"])
    total_episodes = plan["baseline_episodes"] + 5 * len(plan["conditions"])
    publication.update_progress(api, args.progress_run_id, phase=stage, conditions_expected=total_conditions,
        episodes_expected=total_episodes, conditions_completed=plan["baseline_conditions"] + len(state["published"]),
        episodes_completed=plan["baseline_episodes"] + 5 * len(state["published"]))
    policy.submit(args, plan_path, state, coordinator, study)
    entries = policy.publish_finished(args, plan, state, coordinator, study, publication, api)
    result_path = args.result_root / (stage + "-results.json")
    study.immutable_json(result_path, {"schema": policy.SCHEMA, "stage": stage,
        "plan": study.bind(plan_path), "source_sha": args.source_sha, "reused": plan["reused"], "entries": entries})
    critic.bindings_unchanged(plan, study)
    continuation.require_checkout(TOOLING_ROOT, args.tooling_sha)
    coordinator.require_source(args.source_sha)
    state[stage + "_complete"] = True
    state[stage + "_results"] = study.bind(result_path)
    coordinator.atomic_json(state_path, state, overwrite=True)
    study.immutable_json(args.result_root / "COMPLETE.json", {**state, "workspace": workspace})
    publication.update_progress(api, args.progress_run_id, phase="complete", **{stage + "_complete": 1},
                                conditions_completed=total_conditions, episodes_completed=total_episodes)
    (args.result_root / "FAILED.json").unlink(missing_ok=True)
    return state


def run(args, *, policy=None):
    policy = policy or sys.modules[__name__]
    if not os.environ.get("SLURM_JOB_ID") or args.source_sha != getattr(policy, "SOURCE_SHA", SOURCE_SHA):
        raise ValueError("Use a scheduler allocation and the policy-pinned experiment source")
    args.execution_root = args.execution_root.resolve()
    provenance = {"tooling": continuation.require_checkout(TOOLING_ROOT, args.tooling_sha),
                  "execution": continuation.require_checkout(args.execution_root, args.source_sha)}
    continuation.require_runtime(args.execution_root)
    coordinator, study, publication = continuation.execution_modules(args.execution_root)
    if args.command == "worker":
        args.plan = args.plan.resolve(strict=True)
        return policy.worker(args, coordinator, study, publication, provenance)
    if type(args.max_concurrent) is not int or args.max_concurrent < 1:
        raise ValueError("max-concurrent must be positive")
    for key in ("completed_sweep_root", "result_root", "workspace_spec", "worker_launcher"):
        setattr(args, key, getattr(args, key).resolve())
    if (args.result_root.is_relative_to(args.completed_sweep_root)
            or args.completed_sweep_root.is_relative_to(args.result_root)
            or any(args.result_root.is_relative_to(root) for root in (TOOLING_ROOT, args.execution_root))
            or not args.worker_launcher.is_relative_to(TOOLING_ROOT)):
        raise ValueError("Require separate extension storage and the pinned tooling worker launcher")
    args.result_root.mkdir(parents=True, exist_ok=True)
    from utils.eval_series import _lock
    import wandb
    with _lock(args.result_root / ".update-extension.lock", blocking=False):
        api = None
        try:
            api = wandb.Api(timeout=60)
            return policy.coordinate(args, coordinator, study, publication, provenance, api=api)
        except Exception as error:
            coordinator.atomic_json(args.result_root / "FAILED.json", {"type": type(error).__name__,
                                    "message": str(error), **provenance}, overwrite=True)
            try:
                publication.update_progress(api, args.progress_run_id, phase=policy.STAGE + "_failed", failure_type=type(error).__name__)
            except Exception:
                pass
            raise


def main(argv=None, *, policy=None):
    policy = policy or sys.modules[__name__]
    parser = argparse.ArgumentParser(description=policy.__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("coordinate", "worker"):
        sub = commands.add_parser(command)
        sub.add_argument("--execution-root", type=Path, required=True)
        sub.add_argument("--source-sha", required=True)
        sub.add_argument("--tooling-sha", required=True)
        if command == "coordinate":
            for name in ("completed-sweep-root", "result-root", "workspace-spec", "worker-launcher"):
                sub.add_argument("--" + name, type=Path, required=True)
            sub.add_argument("--progress-run-id", required=True)
            sub.add_argument("--gpu-type", default="prefer_l40s",
                help="Slurm GPU type, or prefer_l40s for L40S preference with A5000 fallback")
            sub.add_argument("--max-concurrent", type=int, required=True)
        else:
            sub.add_argument("--plan", type=Path, required=True)
            sub.add_argument("--index", type=int, required=True)
    print(json.dumps(policy.run(parser.parse_args(argv)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
