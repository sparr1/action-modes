"""Launch only the authorized Stage 3 extension using the original experiment source.

This operator has its own tooling commit and output directory. It imports the
unchanged, separately pinned experiment checkout only after validating both
checkouts. The completed parent study and its publication history are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

TOOLING_ROOT = Path(__file__).resolve().parent
SCHEMA = "ambixqc-bn-stage3-continuation-v1"
VIEW_NAME = "nw-xqcbn290926-v"
VIEW_URL = "https://wandb.ai/rwgao_b-brown-university/ambi-inner-bench?nw=xqcbn290926"
LOCK_SHA = "f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6"


def require_checkout(root, sha):
    root = Path(root).resolve(strict=True)
    if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{40}", sha) is None:
        raise ValueError("Both execution and tooling require full source commits")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if git("rev-parse", "HEAD") != sha or git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Require the exact clean checkout: " + str(root))
    return {"root": str(root), "commit": sha, "tree": git("rev-parse", "HEAD^{tree}")}


def execution_modules(root):
    """Reject mixed imports instead of silently executing the tooling checkout."""
    root = Path(root).resolve(strict=True)
    for name, module in tuple(sys.modules.items()):
        if name in {"orchestrate_ambixqc_bn_study", "run_ambixqc_bn_study", "publish_ambixqc_bn_study", "utils"} or name.startswith("utils."):
            origin = getattr(module, "__file__", None)
            if origin and not Path(origin).resolve().is_relative_to(root):
                raise ValueError("Experiment dependency was already imported from another checkout: " + name)
    os.chdir(root)
    sys.path.insert(0, str(root))
    modules = [importlib.import_module(name) for name in (
        "orchestrate_ambixqc_bn_study", "run_ambixqc_bn_study", "publish_ambixqc_bn_study")]
    if any(not Path(module.__file__).resolve().is_relative_to(root) for module in modules):
        raise ValueError("Experiment modules must come from the execution checkout")
    return modules


def require_runtime(execution_root):
    if sys.version_info[:2] != (3, 10):
        raise ValueError("Use the locked Python 3.10 DMControl runtime")
    for lock in (Path(execution_root) / "environments/dmcontrol/uv.lock", Path(sys.prefix).parent / "uv.lock"):
        if not lock.is_file() or hashlib.sha256(lock.read_bytes()).hexdigest() != LOCK_SHA:
            raise ValueError("Execution/runtime dependency lock differs: " + str(lock))


def verify_workspace(api, path, publication):
    proposed = json.loads(Path(path).read_text())
    if proposed.get("campaign") != publication.CAMPAIGN or proposed.get("name") != VIEW_NAME:
        raise ValueError("Extension workspace must identify the working native study view")
    query = publication.VIEW_QUERY
    variables = {"entityName": publication.ENTITY, "name": publication.PROJECT}
    service = getattr(api, "__dict__", {}).get("_service_api")
    if service is not None and hasattr(service, "execute_graphql"):
        response = service.execute_graphql(query, variables=variables)
    else:
        from wandb_gql import gql
        response = api.client.execute(gql(query), variable_values=variables)
    nodes = [item["node"] for item in response["project"]["allViews"]["edges"]
             if item["node"]["name"] == VIEW_NAME]
    if len(nodes) != 1:
        raise ValueError("Working study workspace is missing or ambiguous")
    spec = nodes[0]["spec"]
    if (json.loads(spec) if isinstance(spec, str) else spec) != proposed.get("spec"):
        raise ValueError("Working study workspace differs from the approved extension panels")
    return {"url": VIEW_URL, "view_id": nodes[0]["id"], "verified": True}


def parent_evidence(args, study, publication):
    parent = args.parent_root
    state_path, complete_path = parent / "coordinator-state.json", parent / "COMPLETE.json"
    state, complete = study.read(state_path), study.read(complete_path)
    if ({key: value for key, value in complete.items() if key != "workspace"} != state
            or state.get("campaign") != publication.CAMPAIGN or state.get("source_sha") != args.source_sha
            or state.get("include_stage3") is not False or state.get("stage2_complete") is not True
            or state.get("stage3_complete") or set(state.get("jobs", {})) != {"stage2"}
            or state.get("progress_run_id") != args.progress_run_id):
        raise ValueError("Require the completed, matching Stage 2 parent without a prior Stage 3")
    result_path = study.check_binding(state["stage2_results"])
    if result_path.resolve() != (parent / "stage2-results.json").resolve():
        raise ValueError("Parent result index is outside its completed study")
    result = study.validate_result_index(result_path, source_sha=args.source_sha)
    if result["stage"] != "stage2" or len(result["entries"]) != 4:
        raise ValueError("Parent must contain all four Stage 2 conditions")
    plan = study.load_plan(study.check_binding(result["plan"]), source_sha=args.source_sha)
    mapping_path = parent / "stage2-run-map.json"
    mapping = study.read(mapping_path)
    receipts = []
    for cell in plan["conditions"]:
        study.run_map(mapping_path, plan, cell)
        run_dir = Path(mapping["runs"][cell["selector"]])
        registry, rid, entry = publication._validated_record(run_dir, cell["selector"], study.CHECKPOINT_SHA, args.source_sha)
        receipt_path = run_dir / "bn-study-publication-verified.json"
        receipt = study.read(receipt_path)
        expected = {"selector": cell["selector"], "checkpoint_sha256": study.CHECKPOINT_SHA,
                    "source_sha": args.source_sha, "run_id": registry["run_id"], "record_id": rid,
                    "record_sha256": entry["record_sha256"], "accepted": 1, "published": 1}
        if entry["status"] != "published" or receipt != expected:
            raise ValueError("Parent publication acknowledgement is incomplete or incompatible")
        receipts.append(study.bind(receipt_path))
    args.manifest = study.check_binding(state["inputs"]["manifest"])
    args.reference_index = study.check_binding(state["inputs"]["reference_index"])
    args.smoke_root = Path(state["roots"]["smoke_root"])
    args.checkpoint_root = Path(state["roots"]["checkpoint_root"]) if state["roots"]["checkpoint_root"] else None
    return {"state": study.bind(state_path), "complete": study.bind(complete_path),
            "results": study.bind(result_path), "run_map": study.bind(mapping_path),
            "publication_receipts": receipts}, result_path


def continue_stage3(args, coordinator, study, publication, provenance, *, api):
    evidence, stage2_path = parent_evidence(args, study, publication)
    study.verify_smokes(args.smoke_root, args.manifest, args.source_sha)
    row = study.screen.select_checkpoint(args.manifest, checkpoint_root=args.checkpoint_root)
    study.screen.select_reference(args.reference_index, row, args.manifest)
    workspace = verify_workspace(api, args.workspace_spec, publication)
    expected = {"schema": SCHEMA, "campaign": publication.CAMPAIGN,
                "source_sha": args.source_sha, "tooling": provenance["tooling"],
                "execution": provenance["execution"], "parent": evidence,
                "workspace_spec": study.bind(args.workspace_spec),
                "progress_run_id": args.progress_run_id, "gpu_type": args.gpu_type}
    state_path = args.result_root / "coordinator-state.json"
    state = study.read(state_path) if state_path.exists() else {**expected, "jobs": {}, "published": {}}
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError("Extension state belongs to different tooling, execution, parent or workspace")
    coordinator.atomic_json(state_path, state, overwrite=True)
    study.immutable_json(args.result_root / "continuation-provenance.json", expected)
    coordinator.atomic_json(args.result_root / "workspace-verified.json", workspace, overwrite=True)
    plan_path = args.result_root / "stage3-plan.json"
    study.prepare("stage3", plan_path, source_sha=args.source_sha, stage2_results=stage2_path)
    plan, mapping_path = coordinator.allocate_stage(args, plan_path)
    if plan["stage"] != "stage3" or len(plan["conditions"]) != 3 or len(plan["reused"]) != 1:
        raise ValueError("Extension must evaluate three new cells and reuse one Stage 2 result")
    publication.update_progress(api, args.progress_run_id, phase="stage3", conditions_expected=7,
                                episodes_expected=35, conditions_completed=4 + len(state["published"]),
                                episodes_completed=20 + 5 * len(state["published"]))
    job = coordinator.submit_stage(args, plan_path, mapping_path, state)
    coordinator.wait_jobs([f"{job}_{index}" for index in range(3)])
    result_path = args.result_root / "stage3-results.json"
    coordinator.collect_stage(args, plan_path, job, result_path)
    mapping = study.read(mapping_path)
    for cell in plan["conditions"]:
        selector = cell["selector"]
        receipt = publication.publish_curve(mapping["runs"][selector], selector, study.CHECKPOINT_SHA,
                                            source_sha=args.source_sha)
        state["published"][selector] = receipt
        coordinator.atomic_json(state_path, state, overwrite=True)
        publication.update_progress(api, args.progress_run_id, conditions_completed=4 + len(state["published"]),
                                    episodes_completed=20 + 5 * len(state["published"]))
    state["stage3_complete"] = True
    state["stage3_results"] = study.bind(result_path)
    coordinator.atomic_json(state_path, state, overwrite=True)
    # Keep the parent COMPLETE/state files byte-for-byte unchanged.
    for key in ("state", "complete", "results", "run_map"):
        study.check_binding(evidence[key])
    coordinator.require_source(args.source_sha)
    study.immutable_json(args.result_root / "COMPLETE.json", {**state, "workspace": workspace})
    publication.update_progress(api, args.progress_run_id, phase="complete", stage3_complete=1,
                                conditions_completed=7, episodes_completed=35)
    (args.result_root / "CONTINUATION_FAILED.json").unlink(missing_ok=True)
    return state


def run(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Run the continuation on an allocated CPU scheduler job")
    for key in ("execution_root", "parent_root", "result_root", "workspace_spec"):
        setattr(args, key, getattr(args, key).resolve())
    if (args.result_root == args.parent_root or args.parent_root.is_relative_to(args.result_root)
            or args.result_root.is_relative_to(TOOLING_ROOT)
            or args.result_root.is_relative_to(args.execution_root)):
        raise ValueError("Use a separate extension output directory outside both source checkouts")
    provenance = {"tooling": require_checkout(TOOLING_ROOT, args.tooling_sha),
                  "execution": require_checkout(args.execution_root, args.source_sha)}
    require_runtime(args.execution_root)
    coordinator, study, publication = execution_modules(args.execution_root)
    from utils.eval_series import _lock
    import wandb
    args.result_root.mkdir(parents=True, exist_ok=True)
    with _lock(args.result_root / ".continuation.lock", blocking=False):
        api = None
        try:
            api = wandb.Api(timeout=60)
            return continue_stage3(args, coordinator, study, publication, provenance, api=api)
        except Exception as error:
            coordinator.atomic_json(args.result_root / "CONTINUATION_FAILED.json",
                {"type": type(error).__name__, "message": str(error), **provenance}, overwrite=True)
            try:
                publication.update_progress(api, args.progress_run_id, phase="stage3_failed",
                                            failure_type=type(error).__name__)
            except Exception:
                pass  # The durable failure receipt remains authoritative offline.
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("execution-root", "parent-root", "result-root", "workspace-spec"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-sha", "tooling-sha", "progress-run-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--gpu-type", default="nvidia_rtx_a5000")
    result = run(parser.parse_args(argv))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
