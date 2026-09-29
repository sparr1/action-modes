"""Continue the authorized BN study on an Oscar CPU allocation.

Wait for the probe and controller smoke checks, then launch four independent
normalization/route conditions. The optional --include-stage3 extension must be
explicitly authorized; it adds three budget/rate conditions and reuses the fourth.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import run_ambixqc_bn_study as study
from eval_series import scheduler_done
from publish_ambixqc_bn_study import (
    CAMPAIGN, ENTITY, PROJECT, allocate_curve, install_workspace,
    publish_curve, update_progress,
)
from utils.ambi_benchmark import atomic_json
from utils.eval_series import _lock


def require_source(sha):
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Require a full source commit")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain=v1", "--untracked-files=all"], text=True)
    if actual != sha or dirty.strip():
        raise ValueError("Coordinator requires the exact clean committed source")


def wait_jobs(jobs, timeout=14400):
    start, observed = time.monotonic(), set()
    while not scheduler_done(jobs, observed):
        if time.monotonic() - start >= timeout:
            raise TimeoutError("Study compute did not finish within the coordinator window")
        time.sleep(20)
    # A terminal failure is not completion; validators also require all artifacts.
    text = subprocess.check_output([
        "sacct", "-n", "-X", "--array", "-j", ",".join(jobs),
        "-o", "JobID%128,State%64,ExitCode", "-P"], text=True, timeout=30)
    rows = {fields[0].strip(): fields for line in text.splitlines()
            if len(fields := line.split("|")) >= 3 and re.fullmatch(r"\d+(?:_\d+)?", fields[0].strip())}
    failed = [r[:3] for r in rows.values()
              if r[1].strip().rstrip("+") != "COMPLETED" or r[2].strip() != "0:0"]
    missing = [job for job in jobs if not any(jid == job or jid.startswith(job + "_") for jid in rows)]
    if not rows or failed or missing:
        raise RuntimeError("Study prerequisites failed: " + json.dumps(failed))


def allocate_stage(args, plan_path):
    require_source(args.source_sha)
    plan = study.load_plan(plan_path, source_sha=args.source_sha)
    stage = plan["stage"]
    mapping = {"schema": study.MAP_SCHEMA, "plan_sha256": plan["plan_sha256"], "runs": {}}
    curves = []
    for index, cell in enumerate(plan["conditions"]):
        spec_dir = args.result_root / "specs" / stage / str(index)
        # Regenerate the full scientific identity, even on a retry. A same-name
        # stale/tampered template must never allocate an incompatible curve.
        with tempfile.TemporaryDirectory(dir=args.result_root) as temporary:
            prepared = study.prepare_spec(plan_path, index, args.manifest, Path(temporary) / "spec",
                                          checkpoint_root=args.checkpoint_root)
            if set(prepared.get("specs", {})) != {cell["selector"]}:
                raise ValueError("Each condition must produce exactly one identity template")
            spec = json.loads(Path(prepared["specs"][cell["selector"]]).read_text())
        if spec.get("selector") != cell["selector"]:
            raise ValueError("Wrong controller identity template")
        existing = list(spec_dir.glob("*.json"))
        spec_path = spec_dir / (cell["selector"].replace("/", "__") + ".json")
        if existing and existing != [spec_path]:
            raise ValueError("Unexpected identity templates in condition directory")
        study.immutable_json(spec_path, spec)
        registry = allocate_curve(args.result_root, spec, stage, cell["selector"])
        mapping["runs"][cell["selector"]] = registry["run_dir"]
        curves.append({"selector": cell["selector"], "run_id": registry["run_id"],
                       "run_dir": registry["run_dir"], "condition": cell})
    path = args.result_root / (stage + "-run-map.json")
    study.immutable_json(path, mapping)
    study.immutable_json(args.result_root / (stage + "-curves.json"), curves)
    return plan, path


def submit_stage(args, plan_path, mapping_path, state):
    require_source(args.source_sha)
    plan = study.load_plan(plan_path, source_sha=args.source_sha)
    stage = plan["stage"]
    if stage not in ("stage2", "stage3"):
        raise ValueError("Coordinator submits only production stages")
    for cell in plan["conditions"]:
        study.run_map(mapping_path, plan, cell)
    inputs = {"plan": study.bind(plan_path), "run_map": study.bind(mapping_path)}
    if stage in state["jobs"]:
        intent = state.get("submission_intents", {}).get(stage, {})
        if intent.get("inputs") != inputs or not re.fullmatch(r"\d+", str(state["jobs"][stage])):
            raise ValueError("Saved job does not match this exact stage plan and run map")
        return state["jobs"][stage]
    if stage in state.get("submission_intents", {}):
        raise RuntimeError("Uncertain prior sbatch attempt; inspect its named job before retrying")
    job_name = "axqc-bn-" + stage + "-" + plan["plan_sha256"][:8]
    state.setdefault("submission_intents", {})[stage] = {
        "job_name": job_name, "unix_time": time.time(), "inputs": inputs}
    atomic_json(args.result_root / "coordinator-state.json", state, overwrite=True)
    env = dict(os.environ)
    env.update(EXPECTED_ACTION_MODES_SHA=args.source_sha,
               AMBIXQC_ACTION_MODES_DIR=str(Path.cwd()),
               AMBIXQC_RESULTS_ROOT=str(args.result_root), AMBIXQC_PYTHON=sys.executable,
               AMBIXQC_CHECKPOINT_MANIFEST=str(args.manifest),
               AMBIXQC_REFERENCE_INDEX=str(args.reference_index),
               AMBIXQC_SMOKE_ROOT=str(args.smoke_root),
               AMBIXQC_BN_PLAN=str(plan_path), AMBIXQC_STUDY_STAGE=stage,
               EVAL_RUN_MAP=str(mapping_path))
    if args.checkpoint_root:
        env["AMBIXQC_CHECKPOINT_ROOT"] = str(args.checkpoint_root)
    count = len(plan["conditions"])
    (args.result_root / "slurm").mkdir(exist_ok=True)
    command = ["sbatch", "--parsable", "--job-name=" + job_name,
               "--array=0-" + str(count - 1) + "%" + str(count),
               "--gres=gpu:" + args.gpu_type + ":1", "--cpus-per-task=6", "--mem=32G",
               "--time=" + ("02:00:00" if stage == "stage3" else "01:00:00"),
               "--output=" + str(args.result_root / "slurm" / (stage + "-%A_%a.out")),
               "--error=" + str(args.result_root / "slurm" / (stage + "-%A_%a.err")),
               "slurm/run_ambixqc_bn_study_oscar.sbatch"]
    output = subprocess.check_output(command, env=env, text=True, timeout=60).strip()
    job = output.split(";")[0]
    if not job.isdigit():
        raise RuntimeError("Unrecognized sbatch receipt; inspect submission intent before retrying")
    state["jobs"][stage] = job
    atomic_json(args.result_root / "coordinator-state.json", state, overwrite=True)
    print(json.dumps({"stage": stage, "submitted_job": job, "conditions": count}), flush=True)
    return job


def collect_stage(args, plan_path, job, result_path):
    """Completion requires each submitted GPU task's final markers and bundles."""
    plan = study.load_plan(plan_path, source_sha=args.source_sha)
    stage_root = args.result_root / plan["stage"]
    expected = []
    for index, cell in enumerate(plan["conditions"]):
        job_root = stage_root / f"job{job}-task{index}"
        if ((job_root / "FAILED").exists() or not (job_root / "PASS").is_file()
                or (job_root / "PASS").read_text().strip() != "PASS"):
            raise ValueError("Submitted GPU task did not finish successfully")
        runtime = study.read(job_root / "runtime.json")
        if (not runtime.get("gpu") or not runtime.get("torch", "").startswith("2.3.1")
                or runtime.get("cuda_device_count") != 1):
            raise ValueError("Production result lacks allocated CUDA runtime evidence")
        output = job_root / cell["selector"].split("/")[1]
        expected.append(str(output.resolve()))
        study.validate_output(output, plan, index)
    result = study.collect(plan_path, stage_root, result_path)
    if [entry["output_path"] for entry in result["entries"]] != expected:
        raise ValueError("Collected results do not belong to the submitted GPU array")
    return study.validate_result_index(result_path, source_sha=args.source_sha)


def coordinator_state(args):
    expected = {"campaign": CAMPAIGN, "source_sha": args.source_sha,
                "initial_jobs": args.initial_jobs, "progress_run_id": args.progress_run_id,
                "gpu_type": args.gpu_type,
                "include_stage3": bool(getattr(args, "include_stage3", False)),
                "inputs": {key: study.bind(getattr(args, key))
                           for key in ("manifest", "reference_index", "workspace_spec")},
                "roots": {key: str(getattr(args, key).resolve()) if getattr(args, key) else None
                          for key in ("smoke_root", "probe_root", "checkpoint_root")}}
    path = args.result_root / "coordinator-state.json"
    state = study.read(path) if path.exists() else {**expected, "jobs": {}}
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError("Coordinator state belongs to different source, prerequisites or publication inputs")
    atomic_json(path, state, overwrite=True)
    return state


def run(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Run the coordinator on a scheduler CPU allocation")
    args.result_root = args.result_root.resolve()
    if args.result_root == study.ROOT or study.ROOT in args.result_root.parents:
        raise ValueError("Coordinator results must be outside the source checkout")
    args.result_root.mkdir(parents=True, exist_ok=True)
    with _lock(args.result_root / ".coordinator.lock", blocking=False):
        return _run_locked(args)


def _run_locked(args):
    require_source(args.source_sha)
    stages = ("stage2", "stage3") if getattr(args, "include_stage3", False) else ("stage2",)
    expected_conditions = 7 if len(stages) == 2 else 4
    state = coordinator_state(args)
    state_path = args.result_root / "coordinator-state.json"
    row = study.screen.select_checkpoint(args.manifest, checkpoint_root=args.checkpoint_root)
    study.screen.select_reference(args.reference_index, row, args.manifest)
    import wandb
    api = wandb.Api(timeout=60)
    workspace = install_workspace(api, args.workspace_spec)
    atomic_json(args.result_root / "workspace-verified.json", workspace, overwrite=True)
    update_progress(api, args.progress_run_id, phase="probe_and_smoke",
                    episodes_expected=expected_conditions * 5, conditions_expected=expected_conditions)
    wait_jobs(args.initial_jobs)
    study.verify_smokes(args.smoke_root, args.manifest, args.source_sha)
    selection = args.probe_root / "selection.json"
    from run_ambixqc_bn_probe import validate_selection
    probe = validate_selection(selection, source_sha=args.source_sha)
    # Preserve the full mechanism study separately from episode-performance curves.
    with wandb.init(id=args.progress_run_id, entity=ENTITY, project=PROJECT,
                    resume="must", dir=str(args.result_root), reinit=True) as summary:
        artifact = wandb.Artifact(CAMPAIGN + "-probe", type="normalization-diagnostic")
        for filename in ("results.json", "selection.json", "roots.json", "paired-inputs.json"):
            artifact.add_file(str(args.probe_root / filename), name=filename)
        summary.log_artifact(artifact).wait()
        for mode, score in probe["aggregate_scores"].items():
            summary.summary["probe/heldout_running_q_mse/" + mode] = score
    completed = 0
    for stage in stages:
        require_source(args.source_sha)
        plan_path = args.result_root / (stage + "-plan.json")
        if stage == "stage2":
            study.prepare(stage, plan_path, source_sha=args.source_sha, selection=selection)
        else:
            study.prepare(stage, plan_path, source_sha=args.source_sha,
                          stage2_results=args.result_root / "stage2-results.json")
        plan, mapping_path = allocate_stage(args, plan_path)
        update_progress(api, args.progress_run_id, phase=stage,
                        selected_critic_bn_mode=probe["selected_critic_bn_mode"], stage1_complete=1)
        job = submit_stage(args, plan_path, mapping_path, state)
        wait_jobs([f"{job}_{index}" for index in range(len(plan["conditions"]))])
        result_path = args.result_root / (stage + "-results.json")
        collect_stage(args, plan_path, job, result_path)
        mapping = json.loads(mapping_path.read_text())
        for cell in plan["conditions"]:
            publish_curve(mapping["runs"][cell["selector"]], cell["selector"], study.CHECKPOINT_SHA,
                          source_sha=args.source_sha)
            completed += 1
            update_progress(api, args.progress_run_id, conditions_completed=completed,
                            episodes_completed=completed * 5)
        update_progress(api, args.progress_run_id, **{stage + "_complete": 1})
        state[stage + "_complete"] = True
        state[stage + "_results"] = study.bind(result_path)
        atomic_json(state_path, state, overwrite=True)
    update_progress(api, args.progress_run_id, phase="complete", conditions_completed=expected_conditions,
                    episodes_completed=expected_conditions * 5)
    study.immutable_json(args.result_root / "COMPLETE.json", {**state, "workspace": workspace})
    (args.result_root / "COORDINATOR_FAILED.json").unlink(missing_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("result-root", "manifest", "reference-index", "smoke-root", "probe-root", "workspace-spec"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--checkpoint-root", type=Path)
    p.add_argument("--initial-jobs", nargs="+", required=True)
    p.add_argument("--progress-run-id", required=True)
    p.add_argument("--gpu-type", default="nvidia_rtx_a5000")
    p.add_argument("--include-stage3", action="store_true",
                   help="Explicitly include the separately authorized budget/rate extension")
    args = p.parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        atomic_json(args.result_root / "COORDINATOR_FAILED.json",
                    {"type": type(exc).__name__, "message": str(exc)}, overwrite=True)
        try:
            import wandb
            update_progress(wandb.Api(timeout=30), args.progress_run_id, phase="failed",
                            failure_type=type(exc).__name__)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
