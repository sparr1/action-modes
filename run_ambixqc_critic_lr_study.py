"""Two authorized critic-rate cells, executed with the unchanged 10852a8 science.

This continuation owns a separate tooling commit, immutable plan and registry.
It reuses the completed Stage 3 baseline and paired prior without rerunning or
republishing them. GPU workers smoke their exact cell before five full episodes;
the CPU owner publishes each completed cell independently.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import continue_ambixqc_bn_study as continuation

TOOLING_ROOT = Path(__file__).resolve().parent
SOURCE_SHA = "10852a8cb6501f53b800cf50a774e732fac1be90"
SCHEMA = "ambixqc-critic-lr-study-v1"
RATES = (1e-4, 2e-4)


def conditions(study):
    cells = []
    for rate, token in zip(RATES, ("1e4", "2e4")):
        cell = study.condition("running", "return_return", 4, 5e-5)
        cell["selector"] += "_critic_" + token
        cell["critic_lr"] = rate
        cell["settings"]["inner_critic_lr"] = rate
        cells.append(cell)
    return cells


def matrix_for(study, cells):
    matrix = study.matrix_for(cells)
    matrix["description"] = "Authorized J4 critic-rate extension; reuse completed critic LR5e-5 baseline."
    for cell in cells:
        matrix["comparisons"]["controller"]["variants"][cell["selector"].split("/")[1]]["description"] = (
            f"Return/return, running actor/critic BN, H1 J4 N256 B256 G3 delay3; "
            f"critic LR {cell['critic_lr']:g}; actor and temperature LR 5e-5.")
    return matrix


def bindings_unchanged(value, study):
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            study.check_binding(value)
        else:
            for child in value.values():
                bindings_unchanged(child, study)
    elif isinstance(value, list):
        for child in value:
            bindings_unchanged(child, study)


def parent_evidence(args, study, publication):
    root = args.parent_root
    state_path, complete_path = root / "coordinator-state.json", root / "COMPLETE.json"
    state, complete = study.read(state_path), study.read(complete_path)
    if ({k: v for k, v in complete.items() if k != "workspace"} != state
            or state.get("schema") != continuation.SCHEMA
            or state.get("campaign") != publication.CAMPAIGN
            or state.get("source_sha") != args.source_sha
            or state.get("progress_run_id") != args.progress_run_id
            or state.get("stage3_complete") is not True
            or set(state.get("jobs", {})) != {"stage3"}):
        raise ValueError("Require the completed matching Stage 3 continuation")
    bindings_unchanged(state["parent"], study)
    original = study.check_binding(state["parent"]["state"]).parent
    original_args = SimpleNamespace(parent_root=original, source_sha=args.source_sha,
                                    progress_run_id=args.progress_run_id)
    original_evidence, _ = continuation.parent_evidence(original_args, study, publication)
    if original_evidence != state["parent"]:
        raise ValueError("Stage 3 parent evidence changed")
    for key in ("manifest", "reference_index", "smoke_root", "checkpoint_root"):
        setattr(args, key, getattr(original_args, key))
    result_path = study.check_binding(state["stage3_results"])
    if result_path != (root / "stage3-results.json").resolve():
        raise ValueError("Stage 3 result index is outside its completed continuation")
    result = study.validate_result_index(result_path, source_sha=args.source_sha)
    plan = study.load_plan(study.check_binding(result["plan"]), source_sha=args.source_sha)
    if result["stage"] != "stage3" or len(result["entries"]) != 3 or len(plan["reused"]) != 1:
        raise ValueError("Require all three Stage 3 conditions and its reused Stage 2 result")
    baseline, = [entry for entry in result["entries"]
                 if entry["condition"] == study.condition("running", "return_return", 4, 5e-5)]
    mapping_path = root / "stage3-run-map.json"
    mapping = study.read(mapping_path)
    receipts = []
    expected_published = {}
    for entry in result["entries"]:
        cell = entry["condition"]
        study.run_map(mapping_path, plan, cell)
        run_dir = Path(mapping["runs"][cell["selector"]])
        registry, rid, record = publication._validated_record(run_dir, cell["selector"], study.CHECKPOINT_SHA, args.source_sha)
        receipt_path = run_dir / "bn-study-publication-verified.json"
        receipt = study.read(receipt_path)
        expected = {"selector": cell["selector"], "checkpoint_sha256": study.CHECKPOINT_SHA,
                    "source_sha": args.source_sha, "run_id": registry["run_id"], "record_id": rid,
                    "record_sha256": record["record_sha256"], "accepted": 1, "published": 1}
        raw = study.read(run_dir / "records" / (rid + ".json"))
        returns = {str(ep["seed"]): ep["return"] for ep in raw["episodes"]}
        if record["status"] != "published" or receipt != expected or returns != entry["episode_returns"]:
            raise ValueError("Stage 3 publication acknowledgement or episode evidence differs")
        expected_published[cell["selector"]] = receipt
        receipts.append(study.bind(receipt_path))
    if state.get("published") != expected_published:
        raise ValueError("Stage 3 completion does not acknowledge every published condition")
    return {"state": study.bind(state_path), "complete": study.bind(complete_path),
            "results": study.bind(result_path), "run_map": study.bind(mapping_path),
            "publication_receipts": receipts, "original": original_evidence}, baseline


def prepare_plan(args, study, publication, provenance):
    evidence, baseline = parent_evidence(args, study, publication)
    study.verify_smokes(args.smoke_root, args.manifest, args.source_sha)
    row = study.screen.select_checkpoint(args.manifest, checkpoint_root=args.checkpoint_root)
    reference = study.screen.select_reference(args.reference_index, row, args.manifest)
    cells = conditions(study)
    matrix_path = args.result_root / "stage4.matrix.json"
    study.immutable_json(matrix_path, matrix_for(study, cells))
    from evaluate_ambi_checkpoint import evaluate_matrix
    runs = {}
    for index, cell in enumerate(cells):
        with tempfile.TemporaryDirectory(dir=args.result_root) as temporary:
            prepared = evaluate_matrix(matrix_path, row["path"], selectors=[cell["selector"]],
                                       checkpoint_inventory=args.manifest, source_run=row["source_run"],
                                       eval_series_spec_dir=Path(temporary) / "specs")
            if set(prepared.get("specs", {})) != {cell["selector"]}:
                raise ValueError("Each critic rate must resolve exactly one scientific identity")
            spec = study.read(prepared["specs"][cell["selector"]])
        study.immutable_json(args.result_root / "specs" / f"{index}.json", spec)
        registry = publication.allocate_curve(args.result_root, spec, "stage4", cell["selector"])
        runs[cell["selector"]] = study.bind(Path(registry["run_dir"]) / "run.json")
    plan = {"schema": SCHEMA, "stage": "stage4", "source_sha": args.source_sha,
            "tooling": provenance["tooling"], "execution": provenance["execution"],
            "campaign": publication.CAMPAIGN, "checkpoint_sha256": study.CHECKPOINT_SHA,
            "checkpoint_step": 475000, "conditions": cells, "reused": baseline,
            "parent": evidence, "parent_root": str(args.parent_root),
            "progress_run_id": args.progress_run_id, "result_root": str(args.result_root),
            "matrix": study.bind(matrix_path), "runs": runs,
            "inputs": {key: study.bind(getattr(args, key)) for key in ("manifest", "reference_index")},
            "checkpoint_root": str(args.checkpoint_root) if args.checkpoint_root else None,
            "reference_bundle": str(reference), "smoke_root": str(args.smoke_root),
            "environment_seeds": study.SEEDS, "controller_seed": 12345, "max_steps": 500,
            "smoke_seeds": [101, 102], "smoke_max_steps": 3}
    plan["plan_sha256"] = study.digest(plan)
    path = args.result_root / "stage4-plan.json"
    study.immutable_json(path, plan)
    return path, plan


def load_plan(path, study, source_sha, tooling_sha):
    plan = study.read(path)
    signed = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if (plan.get("schema") != SCHEMA or plan.get("stage") != "stage4"
            or plan.get("campaign") != "ambixqc-bn-study-20260929"
            or plan.get("plan_sha256") != study.digest(signed)
            or source_sha != SOURCE_SHA or plan.get("source_sha") != source_sha
            or plan.get("tooling", {}).get("commit") != tooling_sha
            or plan.get("execution", {}).get("commit") != source_sha
            or plan.get("checkpoint_sha256") != study.CHECKPOINT_SHA
            or plan.get("checkpoint_step") != 475000 or plan.get("conditions") != conditions(study)
            or plan.get("environment_seeds") != study.SEEDS or plan.get("controller_seed") != 12345
            or plan.get("max_steps") != 500 or plan.get("smoke_seeds") != [101, 102]
            or plan.get("smoke_max_steps") != 3
            or plan.get("reused", {}).get("condition") != study.condition("running", "return_return", 4, 5e-5)):
        raise ValueError("Critic-rate plan changed source, controlled settings, baseline or protocol")
    bindings_unchanged(plan, study)
    if study.read(plan["matrix"]["path"]) != matrix_for(study, plan["conditions"]):
        raise ValueError("Critic-rate matrix differs from its plan")
    if (set(plan["runs"]) != {c["selector"] for c in conditions(study)}
            or len({v["path"] for v in plan["runs"].values()}) != 2):
        raise ValueError("Each critic rate requires its own registry")
    return plan


def worker_root(plan, index, job):
    if type(index) is not int or index not in (0, 1) or not re.fullmatch(r"\d+", str(job)):
        raise ValueError("Require one of the two submitted GPU array tasks")
    return Path(plan["result_root"]) / "stage4" / f"job{job}-task{index}"


def validate_bundle(plan, index, root, study, smoke):
    return study.screen.validate_bundle(
        root / "bundle", 0, seeds=plan["smoke_seeds"] if smoke else plan["environment_seeds"],
        max_steps=plan["smoke_max_steps"] if smoke else plan["max_steps"], source_sha=plan["source_sha"],
        reference_bundle=None if smoke else plan["reference_bundle"],
        expected_selector=plan["conditions"][index]["selector"],
        expected_config=study.expected_settings(plan["conditions"][index]))


def evaluate_cell(plan, index, root, study, *, smoke):
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.ambi_benchmark import stage_completed_bundle
    cell = plan["conditions"][index]
    root.mkdir(parents=True, exist_ok=False)
    manifest = study.check_binding(plan["inputs"]["manifest"])
    row = study.screen.select_checkpoint(manifest, checkpoint_root=plan["checkpoint_root"])
    assigned = None if smoke else {cell["selector"]: str(Path(plan["runs"][cell["selector"]]["path"]).parent)}
    payload = evaluate_matrix(
        plan["matrix"]["path"], row["path"], selectors=[cell["selector"]],
        seeds=plan["smoke_seeds"] if smoke else plan["environment_seeds"],
        controller_seed=12345, max_steps=plan["smoke_max_steps"] if smoke else 500, device="cuda",
        bundle_dir=root / "bundle", reference_bundle=None if smoke else plan["reference_bundle"],
        checkpoint_inventory=manifest, source_run=row["source_run"], eval_run_map=assigned, stage_results=False)
    if payload["checkpoint_sha256"] != study.CHECKPOINT_SHA:
        raise ValueError("Critic-rate worker evaluated a different checkpoint")
    study.immutable_json(root / "results.json", payload)
    validation = validate_bundle(plan, index, root, study, smoke)
    study.immutable_json(root / "validation.json", validation)
    if not smoke:
        staged = stage_completed_bundle(root / "bundle", assigned, source_run=row["source_run"], inventory_path=manifest)
        if set(staged) != {cell["selector"]} or staged[cell["selector"]]["status"] != "queued":
            raise RuntimeError("Complete critic-rate result could not be queued for publication")
    (root / "PASS").write_text("PASS\n")


def worker(args, coordinator, study, publication, provenance):
    plan = load_plan(args.plan, study, args.source_sha, args.tooling_sha)
    if provenance != {"tooling": plan["tooling"], "execution": plan["execution"]}:
        raise ValueError("Worker checkouts differ from the allocated plan")
    parent_args = SimpleNamespace(parent_root=Path(plan["parent_root"]), source_sha=args.source_sha,
                                  progress_run_id=plan["progress_run_id"])
    evidence, baseline = parent_evidence(parent_args, study, publication)
    if (evidence != plan["parent"] or baseline != plan["reused"]
            or plan["inputs"] != {key: study.bind(getattr(parent_args, key))
                                  for key in ("manifest", "reference_index")}
            or plan["smoke_root"] != str(parent_args.smoke_root)
            or plan["checkpoint_root"] != (str(parent_args.checkpoint_root) if parent_args.checkpoint_root else None)):
        raise ValueError("Worker parent or reused baseline differs from the plan")
    study.verify_smokes(parent_args.smoke_root, parent_args.manifest, args.source_sha)
    row = study.screen.select_checkpoint(parent_args.manifest, checkpoint_root=parent_args.checkpoint_root)
    reference = study.screen.select_reference(parent_args.reference_index, row, parent_args.manifest)
    if str(reference) != plan["reference_bundle"]:
        raise ValueError("Worker paired reference differs")
    job = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", ""))
    if os.environ.get("SLURM_ARRAY_TASK_ID") != str(args.index):
        raise ValueError("Worker index differs from its allocated task")
    root = worker_root(plan, args.index, job)
    root.mkdir(parents=True, exist_ok=False)
    try:
        import torch
        if (torch.__version__.split("+")[0] != "2.3.1" or not torch.cuda.is_available()
                or torch.cuda.device_count() != 1 or torch.ones(1, device="cuda").sum().item() != 1):
            raise ValueError("Worker requires the locked PyTorch 2.3.1 runtime and one CUDA device")
        study.immutable_json(root / "runtime.json", {"python": sys.version, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0), "cuda_device_count": 1})
        study.immutable_json(root / "provenance.json", {"plan": study.bind(args.plan), "index": args.index,
            "job": job, **provenance})
        # No production evaluation or staging can occur if its exact-cell smoke fails.
        evaluate_cell(plan, args.index, root / "smoke", study, smoke=True)
        evaluate_cell(plan, args.index, root / "full", study, smoke=False)
        continuation.require_checkout(TOOLING_ROOT, args.tooling_sha)
        coordinator.require_source(args.source_sha)
        (root / "PASS").write_text("PASS\n")
    except Exception as error:
        coordinator.atomic_json(root / "FAILED", {"type": type(error).__name__, "message": str(error)}, overwrite=True)
        raise
    return {"output": str(root), "index": args.index}


def validate_worker(plan, index, job, study):
    root = worker_root(plan, index, job)
    if (root / "FAILED").exists() or not (root / "PASS").is_file() or (root / "PASS").read_text().strip() != "PASS":
        raise ValueError("Submitted critic-rate GPU task did not finish successfully")
    runtime = study.read(root / "runtime.json")
    if not runtime.get("gpu") or not runtime.get("torch", "").startswith("2.3.1") or runtime.get("cuda_device_count") != 1:
        raise ValueError("Missing allocated CUDA runtime evidence")
    provenance = study.read(root / "provenance.json")
    if provenance != {"plan": study.bind(Path(plan["result_root"]) / "stage4-plan.json"), "index": index,
                      "job": str(job), "tooling": plan["tooling"], "execution": plan["execution"]}:
        raise ValueError("Worker provenance differs from the exact submitted plan")
    for phase in ("smoke", "full"):
        directory = root / phase
        if not (directory / "PASS").is_file() or (directory / "PASS").read_text().strip() != "PASS":
            raise ValueError("Missing smoke or full evaluation completion")
        if study.read(directory / "validation.json") != validate_bundle(plan, index, directory, study, phase == "smoke"):
            raise ValueError("Worker bundle changed after validation")
    episodes = study.read(root / "full/bundle/manifest.json")["runs"][0]["episodes"]
    return {"index": index, "condition": plan["conditions"][index], "output_path": str(root),
            "mean_return": statistics.mean(ep["return"] for ep in episodes),
            "episode_returns": {str(ep["seed"]): ep["return"] for ep in episodes},
            "validation": study.bind(root / "full/validation.json"),
            "bundle_manifest": study.bind(root / "full/bundle/manifest.json")}


def submit(args, plan_path, state, coordinator, study):
    coordinator.require_source(args.source_sha)
    inputs = {"plan": study.bind(plan_path), "worker_launcher": study.bind(args.worker_launcher)}
    if state.get("job"):
        if state.get("submission_intent", {}).get("inputs") != inputs or not str(state["job"]).isdigit():
            raise ValueError("Saved array does not match this exact critic-rate plan")
        return state["job"]
    if state.get("submission_intent"):
        raise RuntimeError("Uncertain prior sbatch attempt; inspect the named job before retrying")
    state["submission_intent"] = {"inputs": inputs, "unix_time": time.time(),
                                  "job_name": "axqc-critic-lr-" + state["plan"]["sha256"][:8]}
    state_path = args.result_root / "coordinator-state.json"
    coordinator.atomic_json(state_path, state, overwrite=True)
    env = dict(os.environ)
    env.update(AMBI_EXECUTION_ROOT=str(args.execution_root), AMBI_SOURCE_SHA=args.source_sha,
               AMBI_TOOLING_SHA=args.tooling_sha, AMBI_TOOLING_ROOT=str(TOOLING_ROOT),
               AMBI_CRITIC_LR_PLAN=str(plan_path), AMBI_PYTHON=sys.executable,
               AMBIXQC_PYTHON=sys.executable)
    (args.result_root / "slurm").mkdir(exist_ok=True)
    command = ["sbatch", "--parsable", "--job-name=" + state["submission_intent"]["job_name"],
               "--array=0-1%2", "--export=ALL", "--gres=gpu:" + args.gpu_type + ":1",
               "--cpus-per-task=6", "--mem=32G", "--time=02:00:00",
               "--output=" + str(args.result_root / "slurm/stage4-%A_%a.out"),
               "--error=" + str(args.result_root / "slurm/stage4-%A_%a.err"), str(args.worker_launcher)]
    job = subprocess.check_output(command, env=env, text=True, timeout=60).strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError("Unrecognized sbatch receipt; inspect the persisted intent before retrying")
    state["job"] = job
    coordinator.atomic_json(state_path, state, overwrite=True)
    print(json.dumps({"stage": "stage4", "submitted_job": job, "conditions": 2}), flush=True)
    return job


def label_curve(api, registry, cell, publication):
    # Stock science identities already include critic LR, but stock display names
    # do not. Apply metadata after each Publisher session, which resets labels.
    label = f"475k shared UTD2 · return/running · J4 · critic LR {cell['critic_lr']:g} · actor/temp 5e-5"
    # Api.run caches the pre-publication object. Run.update also writes its
    # summary, so even an innocuous label refresh can erase completed results.
    # Fetch current config and issue a metadata-only mutation: never write summary.
    api.flush()
    run = api.run(f"{publication.ENTITY}/{publication.PROJECT}/{registry['run_id']}")
    run.config["curve_label"] = label
    query = '''mutation LabelCriticRateRun($id:String!,$display_name:String!,$config:JSONString!){
        upsertBucket(input:{id:$id,displayName:$display_name,config:$config}){
            bucket{id displayName}}}'''
    variables = {"id": run.storage_id, "display_name": label, "config": run.json_config}
    service = getattr(api, "__dict__", {}).get("_service_api")
    if service is not None and hasattr(service, "execute_graphql"):
        response = service.execute_graphql(query, variables=variables)
    else:
        from wandb_gql import gql
        response = api.client.execute(gql(query), variable_values=variables)
    if response.get("upsertBucket", {}).get("bucket") != {"id": run.storage_id, "displayName": label}:
        raise RuntimeError("Critic-rate display label was not acknowledged")


def publish_finished(args, plan, state, coordinator, study, publication, api, *, timeout=12600):
    start = time.monotonic()
    pending = set(range(2))
    observed = {index: set() for index in pending}
    entries = {}
    failures = {}
    while pending:
        for index in sorted(pending):
            job = f"{state['job']}_{index}"
            if not coordinator.scheduler_done([job], observed[index]):
                continue
            pending.remove(index)
            try:
                coordinator.wait_jobs([job])
                entries[index] = validate_worker(plan, index, state["job"], study)
                cell = plan["conditions"][index]
                registry = study.read(plan["runs"][cell["selector"]]["path"])
                receipt = publication.publish_curve(registry["run_dir"], cell["selector"], study.CHECKPOINT_SHA,
                                                    source_sha=args.source_sha)
                label_curve(api, registry, cell, publication)
                state["published"][cell["selector"]] = receipt
                coordinator.atomic_json(args.result_root / "coordinator-state.json", state, overwrite=True)
                publication.update_progress(api, args.progress_run_id, conditions_completed=7 + len(state["published"]),
                                            episodes_completed=35 + 5 * len(state["published"]))
            except Exception as error:
                failures[str(index)] = {"type": type(error).__name__, "message": str(error)}
                state["failures"] = failures
                coordinator.atomic_json(args.result_root / "coordinator-state.json", state, overwrite=True)
        if pending:
            if time.monotonic() - start >= timeout:
                raise TimeoutError("Critic-rate array did not finish before the coordinator deadline")
            time.sleep(20)
    if failures:
        raise RuntimeError("Critic-rate completion/publication failed: " + json.dumps(failures))
    state.pop("failures", None)
    return [entries[index] for index in range(2)]


def coordinate(args, coordinator, study, publication, provenance, *, api):
    workspace = continuation.verify_workspace(api, args.workspace_spec, publication)
    # Bind scope before creating any runs; retry cannot silently change its parent,
    # tooling, launcher, or dashboard input and then allocate a duplicate attempt.
    expected = {"schema": SCHEMA, "source_sha": args.source_sha, **provenance,
                "parent_root": str(args.parent_root), "workspace_spec": study.bind(args.workspace_spec),
                "progress_run_id": args.progress_run_id, "gpu_type": args.gpu_type,
                "worker_launcher": study.bind(args.worker_launcher)}
    state_path = args.result_root / "coordinator-state.json"
    state = study.read(state_path) if state_path.exists() else {**expected, "published": {}}
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError("Critic-rate state belongs to different source, scope or publication inputs")
    coordinator.atomic_json(state_path, state, overwrite=True)
    plan_path, plan = prepare_plan(args, study, publication, provenance)
    load_plan(plan_path, study, args.source_sha, args.tooling_sha)
    state["plan"] = study.bind(plan_path)
    for cell in plan["conditions"]:
        label_curve(api, study.read(plan["runs"][cell["selector"]]["path"]), cell, publication)
    publication.update_progress(api, args.progress_run_id, phase="stage4", conditions_expected=9, episodes_expected=45,
                                conditions_completed=7 + len(state["published"]), episodes_completed=35 + 5 * len(state["published"]))
    submit(args, plan_path, state, coordinator, study)
    entries = publish_finished(args, plan, state, coordinator, study, publication, api)
    results = {"schema": SCHEMA, "stage": "stage4", "plan": study.bind(plan_path),
               "source_sha": args.source_sha, "reused": plan["reused"], "entries": entries}
    study.immutable_json(args.result_root / "stage4-results.json", results)
    bindings_unchanged(plan, study)
    continuation.require_checkout(TOOLING_ROOT, args.tooling_sha)
    coordinator.require_source(args.source_sha)
    state["stage4_complete"] = True
    state["stage4_results"] = study.bind(args.result_root / "stage4-results.json")
    coordinator.atomic_json(state_path, state, overwrite=True)
    study.immutable_json(args.result_root / "COMPLETE.json", {**state, "workspace": workspace})
    publication.update_progress(api, args.progress_run_id, phase="complete", stage4_complete=1,
                                conditions_completed=9, episodes_completed=45)
    (args.result_root / "FAILED.json").unlink(missing_ok=True)
    return state


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or args.source_sha != SOURCE_SHA:
        raise ValueError("Use a scheduler allocation and the original 10852a8 experiment source")
    args.execution_root = args.execution_root.resolve()
    provenance = {"tooling": continuation.require_checkout(TOOLING_ROOT, args.tooling_sha),
                  "execution": continuation.require_checkout(args.execution_root, args.source_sha)}
    continuation.require_runtime(args.execution_root)
    coordinator, study, publication = continuation.execution_modules(args.execution_root)
    if args.command == "worker":
        args.plan = args.plan.resolve(strict=True)
        return worker(args, coordinator, study, publication, provenance)
    for key in ("parent_root", "result_root", "workspace_spec", "worker_launcher"):
        setattr(args, key, getattr(args, key).resolve())
    if (args.result_root == args.parent_root or args.parent_root.is_relative_to(args.result_root)
            or args.result_root.is_relative_to(args.parent_root)
            or any(args.result_root.is_relative_to(root) for root in (TOOLING_ROOT, args.execution_root))
            or not args.worker_launcher.is_relative_to(TOOLING_ROOT)):
        raise ValueError("Require separate result storage and the pinned tooling checkout's worker launcher")
    args.result_root.mkdir(parents=True, exist_ok=True)
    from utils.eval_series import _lock
    import wandb
    with _lock(args.result_root / ".critic-lr.lock", blocking=False):
        api = None
        try:
            api = wandb.Api(timeout=60)
            return coordinate(args, coordinator, study, publication, provenance, api=api)
        except Exception as error:
            coordinator.atomic_json(args.result_root / "FAILED.json", {"type": type(error).__name__,
                                    "message": str(error), **provenance}, overwrite=True)
            try:
                publication.update_progress(api, args.progress_run_id, phase="stage4_failed", failure_type=type(error).__name__)
            except Exception:
                pass
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("coordinate", "worker"):
        sub = commands.add_parser(command)
        sub.add_argument("--execution-root", type=Path, required=True)
        sub.add_argument("--source-sha", required=True)
        sub.add_argument("--tooling-sha", required=True)
        if command == "coordinate":
            for name in ("parent-root", "result-root", "workspace-spec", "worker-launcher"):
                sub.add_argument("--" + name, type=Path, required=True)
            sub.add_argument("--progress-run-id", required=True)
            sub.add_argument("--gpu-type", default="nvidia_rtx_a5000")
        else:
            sub.add_argument("--plan", type=Path, required=True)
            sub.add_argument("--index", type=int, choices=(0, 1), required=True)
    print(json.dumps(run(parser.parse_args(argv)), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
