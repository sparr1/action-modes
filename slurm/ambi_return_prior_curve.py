"""Three-seed frozen prior/SAC curves, with audited reuse of the 500k result.

GPU workers stay offline. The CPU publisher owns two explicit new curves and
accepts only complete, hash-bound checkpoint results. Existing five-seed source
bundles are immutable; seed-subset records retain their complete provenance.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm import ambi_return_prior_eval as original
from slurm import ambi_return_prior_publish as previous
from utils import eval_series as series
from utils import eval_series_data as data

ATTEMPT = "return-prior-curve-20261027"
SOURCE = "rwgao_b-brown-university/ambi/e83a27546bcd4ffb92650dade99e619f"
SEEDS = [101, 102, 103]
STEPS = list(range(200000, 500001, 50000))
ARMS = ("prior", "sac")
OWNER = previous.OWNER
ENTITY, PROJECT = previous.ENTITY, previous.PROJECT
MATRIX = original.MATRIX
VISIBLE = ("eval/return_mean", "eval/return_sample_std", "eval/paired_gain_mean",
           "eval/paired_gain_sample_std", "eval/paired_win_fraction",
           "runtime/control_seconds_per_decision", "eval/episodes")
require = original.require


def load_inventory(path):
    value = series._read(path)
    require(value.get("schema_version") == 1 and value.get("campaign_id") == ATTEMPT,
            "Unexpected campaign inventory")
    rows = value.get("entries", [])
    require([row.get("step") for row in rows] == STEPS, "Expected 200k through 500k every 50k")
    require(value.get("seeds", SEEDS) == SEEDS and value.get("expected_steps", STEPS) == STEPS,
            "Unexpected seed or checkpoint scope")
    for row in rows:
        require(row.get("source_run") == SOURCE and row.get("source_run_id") == SOURCE.rsplit("/", 1)[1]
                and row.get("J") == 8 and row.get("K") == 1, "Source or original H3 J8 K1 changed")
        for key in ("sha256", "metadata_sha256"):
            require(bool(re.fullmatch(r"[a-f0-9]{64}", row.get(key, ""))), "Missing pinned artifact hash")
        require(bool(row.get("reuse_output")) == (row["step"] == 500000),
                "500k must be reused, and only 500k is a reused result")
    return value


def verify_source(expected_commit):
    require(bool(re.fullmatch(r"[a-f0-9]{40}", expected_commit or "")), "Expected commit is required")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(actual == expected_commit, "Source commit mismatch")
    require(not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"],
                                        cwd=ROOT, text=True).strip(), "Source checkout is not clean")


def verify_entry(row):
    for field, hash_field in (("checkpoint", "sha256"), ("metadata", "metadata_sha256")):
        path = Path(row[field])
        require(path.is_absolute() and path.is_file(), "Missing absolute " + field)
        require(series._file_digest(path) == row[hash_field], field + " SHA256 mismatch")
    # Reuse the audited checkpoint contract without its old three-source and
    # min(500k,latest) campaign-selection policy.
    synthetic = [dict(source_run_id=run, source_run=f"rwgao_b-brown-university/ambi/{run}",
                      source_run_name="not-selected", J=j, K=k)
                 for run, j, k in original.SOURCES]
    synthetic[0] = {**row, "latest_at_selection": row["step"]}
    original.verify_entry({"schema_version": 1, "entries": synthetic}, 0)
    return row


def native_inventory(row):
    return {"source_run": SOURCE, "checkpoints": [{key: row[key]
            for key in ("step", "sha256", "metadata_sha256")}]}


def output_dir(root, index, mode="production"):
    return Path(root) / mode / f"cell-{index}"


def base_command(row, inventory_path):
    return [sys.executable, str(ROOT / "evaluate_ambi_checkpoint.py"), "--matrix", str(MATRIX),
            "--checkpoint", row["checkpoint"], "--metadata", row["metadata"],
            "--checkpoint-inventory", str(inventory_path), "--controller-seed", "55",
            "--seeds", *map(str, SEEDS), "--max-steps", "500", "--device", "cuda"]


def validate_spec(spec, arm):
    identity = spec["identity"]
    require(spec["selector"] == "comparison/" + arm and identity["backbone"] == SOURCE,
            "Specification source or selector mismatch")
    planner, protocol = identity["planner"], identity["protocol"]
    require(planner.get("type") == arm and planner.get("action_rule") == "tanh_mean",
            "Unexpected planner or action rule")
    require(protocol.get("environment_seeds") == SEEDS and protocol.get("max_steps") == 500
            and protocol.get("controller_seed") == 55 and protocol.get("action_rule") == "tanh_mean",
            "Expected three paired full episodes")
    if arm == "sac":
        settings = planner["settings"]
        require((settings.get("inner_rollout_horizon"), settings.get("inner_rounds"),
                 settings.get("inner_solve_interval", 1)) == (3, 8, 1), "Original H3 J8 K1 changed")
    return identity


def prepare(inventory, root, expected_commit):
    verify_source(expected_commit)
    root = Path(root).resolve()
    require(not root.is_relative_to(ROOT), "Outputs must be outside source")
    specs = {}
    for index, row in enumerate(inventory["entries"]):
        verify_entry(row)
        dest = output_dir(root, index, "prepare")
        receipt = dest / "receipt.json"
        if receipt.exists():
            require(series._read(receipt)["inventory_sha256"] == series._digest(inventory),
                    "Prepared inventory changed")
        else:
            dest.mkdir(parents=True, exist_ok=False)
            series._atomic_json(dest / "checkpoint-inventory.json", native_inventory(row))
            command = base_command(row, dest / "checkpoint-inventory.json") + [
                "--eval-series-spec-dir", str(dest / "specs"), "--output", str(dest / "prepared.json")]
            subprocess.run(command, cwd=ROOT, check=True)
            series._atomic_json(receipt, {"inventory_sha256": series._digest(inventory),
                                          "commit": expected_commit, "mode": "prepare"})
        for arm in ARMS:
            spec = series._read(dest / "specs" / f"comparison__{arm}.json")
            identity = validate_spec(spec, arm)
            if arm in specs:
                require(identity == specs[arm]["identity"], "Checkpoint-dependent curve identity")
            specs[arm] = spec
    return specs


def prepare_publication(inventory, root):
    root = Path(root)
    publication = root / "publication"
    publication.mkdir(parents=True, exist_ok=True)
    path = publication / "publication-map.json"
    with series._lock(publication / ".prepare.lock"):
        receipt = series._read(path) if path.exists() else {
            "campaign_id": ATTEMPT, "inventory_sha256": series._digest(inventory), "curves": {}}
        require(receipt["campaign_id"] == ATTEMPT and receipt["inventory_sha256"] == series._digest(inventory),
                "Prepared inventory changed")
        for arm in ARMS:
            specs = [series._read(output_dir(root, i, "prepare") / "specs" / f"comparison__{arm}.json")
                     for i in range(len(inventory["entries"]))]
            for spec in specs:
                validate_spec(spec, arm)
                require(spec["identity"] == specs[0]["identity"], "Checkpoint-dependent curve identity")
            existing = list((publication / "runs" / arm).glob("*/run.json"))
            if arm in receipt["curves"]:
                registry = series.validate_identity(receipt["curves"][arm]["run_dir"], specs[0]["identity"])
            elif existing:
                require(len(existing) == 1, "Ambiguous interrupted run allocation")
                registry = series.validate_identity(existing[0].parent, specs[0]["identity"])
            else:
                registry = series.create_run(publication / "runs" / arm, specs[0], ATTEMPT, PROJECT, ENTITY, OWNER)
            require(registry["attempt_label"] == ATTEMPT and registry["owner"] == OWNER, "Wrong run owner/attempt")
            receipt["curves"][arm] = {"run_id": registry["run_id"], "run_dir": registry["run_dir"],
                "arm": arm, "name": "Policy prior" if arm == "prior" else "AMBI · inner SAC H3 J8 K1",
                "color": "#64748B" if arm == "prior" else "#2563EB", "expected": len(STEPS)}
            series._atomic_json(path, receipt)
        series._atomic_json(publication / "eval-run-map.json", {
            "comparison/" + arm: row["run_dir"] for arm, row in receipt["curves"].items()})
    return receipt


def validate_results(payload, row, *, smoke=False, smoke_steps=3):
    seeds, steps = ([101], smoke_steps) if smoke else (SEEDS, 500)
    require(payload.get("frozen_outer_learning") is True and payload.get("deterministic_execution") is True,
            "Expected frozen mean-action evaluation")
    require(payload.get("checkpoint_sha256") == row["sha256"], "Result checkpoint mismatch")
    results = payload.get("results", [])
    require(len(results) == 2 and {r["selector"] for r in results} == set(original.SELECTORS),
            "Expected complete prior/SAC pair")
    for result in results:
        require(result.get("outer_state_unchanged") is True and
                result["outer_updates_before"] == result["outer_updates_after"], "Outer state changed")
        require(result.get("controller_seed") == 55 and result.get("environment_seeds") == seeds,
                "Evaluation seed mismatch")
        episodes = result.get("episodes", [])
        require([e["seed"] for e in episodes] == seeds and all(e["length"] == steps for e in episodes),
                "Incomplete episode coverage")
        require(all(math.isfinite(e["return"]) for e in episodes), "Nonfinite return")
        if not smoke:
            require(all(e.get("truncated") is True and e.get("terminated") is False and
                        e.get("truncated_by_evaluator") is False for e in episodes), "Shortened episode")
        require(not result.get("nonfinite_model_metrics") and not result.get("nonfinite_trace_metrics")
                and not result.get("togo_return_probe"), "Unexpected probe/nonfinite diagnostics")
        cfg = result["resolved_config"]
        if result["selector"] == "comparison/sac":
            require((cfg["inner_operator"], cfg["inner_rounds"], cfg["inner_solve_interval"],
                     cfg["inner_rollout_horizon"]) == ("sac", 8, 1, 3), "SAC recipe changed")
            require(cfg["inner_temperature_mode"] == "auto" and
                    cfg["inner_temperature_initialization"] == "inherit_outer", "Inner alpha changed")
        else:
            require(cfg["inner_operator"] == "none" and cfg["inner_rounds"] == 0, "Prior performed inner work")
            for key in ("inner_model_steps", "inner_actor_optimizer_steps", "inner_critic_optimizer_steps",
                        "inner_temperature_optimizer_steps"):
                metric = result.get("model_metrics", {}).get(key, {})
                require(metric.get("count") == len(seeds) * steps and
                        all(metric.get(f) == 0 for f in ("sum", "min", "max")), "Prior work counter mismatch")
    return {"episodes": 2 * len(seeds), "decisions": 2 * len(seeds) * steps}


def paired_record(inner, prior):
    require(inner["checkpoint"] == prior["checkpoint"] and
            inner["identity"]["backbone"] == prior["identity"]["backbone"], "Paired checkpoint/source mismatch")
    require(inner["identity"]["protocol"] == prior["identity"]["protocol"], "Paired protocol differs")
    require(prior["identity"]["planner"]["type"] == "prior", "Baseline must be prior")
    require(sorted(e["seed"] for e in inner["episodes"]) == SEEDS and
            sorted(e["seed"] for e in prior["episodes"]) == SEEDS, "Pairing requires three unique seeds")
    record = copy.deepcopy(inner)
    baseline = {e["seed"]: e["return"] for e in prior["episodes"]}
    for episode in record["episodes"]:
        episode["paired_return_delta"] = episode["return"] - baseline[episode["seed"]]
    record["metrics"] = data._metrics(record["episodes"])
    record["metrics"].update({"eval/base_return_mean": prior["metrics"]["eval/return_mean"],
        "eval/paired_win_fraction": sum(e["paired_return_delta"] > 0 for e in record["episodes"]) / len(SEEDS)})
    record["provenance"]["paired_baseline"] = {"record_id": prior["record_id"],
        "identity": prior["identity"], "checkpoint": prior["checkpoint"], "episodes": prior["episodes"]}
    record["record_id"] = series._digest({"result": inner["record_id"], "paired_baseline": prior["record_id"]})
    record["artifact_files"].update({"paired-prior/" + key: value for key, value in prior["artifact_files"].items()})
    return record


def subset_record(record, identity):
    """Derive an audited three-seed record without changing any source bytes."""
    compatible = copy.deepcopy(record["identity"])
    old_seeds = compatible["protocol"]["environment_seeds"]
    require(old_seeds == [101, 102, 103, 104, 105], "Reuse requires the original five-seed result")
    compatible["protocol"]["environment_seeds"] = SEEDS
    require(compatible == identity, "Reuse identity differs beyond the explicit seed subset")
    require(record["checkpoint"]["step"] == 500000, "Only 500k reuse is authorized")
    require(sorted(e["seed"] for e in record["episodes"]) == old_seeds, "Original seeds are incomplete or duplicated")
    episodes = [copy.deepcopy(e) for e in record["episodes"] if e["seed"] in SEEDS]
    require(all(e["length"] == 500 and e.get("truncated") and not e.get("terminated")
                and not e.get("truncated_by_evaluator") for e in episodes), "Reused episodes must be complete")
    provenance = copy.deepcopy(record["provenance"])
    provenance["seed_subset_reuse"] = {"source_record_id": record["record_id"],
        "source_identity": record["identity"], "source_result_path": record["source_result_path"],
        "source_seeds": old_seeds, "selected_seeds": SEEDS, "new_environment_decisions": 0}
    return data._record(identity, record["checkpoint"], episodes, record["artifact_files"], provenance,
                        record["source_result_path"], record["label"], record["selector"], record["controller"])


def raw_records(row, output):
    manifest = Path(output) / "bundle" / "manifest.json"
    return data.load_records(manifest, source_run=SOURCE,
                             inventory_path=Path(output) / "checkpoint-inventory.json")


def validate_completed(inventory, root, index):
    row = inventory["entries"][index]
    require(not row.get("reuse_output"), "500k is reuse-only; never reevaluate it")
    output = output_dir(root, index)
    result_path, manifest_path = output / "results.json", output / "bundle" / "manifest.json"
    if not result_path.is_file() or not manifest_path.is_file():
        return None
    manifest = series._read(manifest_path)
    if manifest.get("status") != "complete":
        return None
    require(all(r.get("status") == "complete" for r in manifest.get("runs", [])), "Incomplete bundle")
    counts = validate_results(series._read(result_path), row)
    require(series._read(output / "checkpoint-inventory.json") == native_inventory(row), "Output inventory differs")
    records = raw_records(row, output)
    require(len(records) == 2, "Expected exactly two records")
    for record in records:
        arm = record["selector"].split("/")[-1]
        require(arm in ARMS, "Unexpected result selector")
        expected = series._read(output_dir(root, index, "prepare") / "specs" / f"comparison__{arm}.json")
        require(record["identity"] == expected["identity"], "Completed scientific identity differs")
        require(record["checkpoint"] == {"step": row["step"], "sha256": row["sha256"]}, "Checkpoint changed")
    receipt = {"validated": True, "mode": "run", "source_run": SOURCE, "step": row["step"],
        "J": 8, "K": 1, "checkpoint_sha256": row["sha256"], "metadata_sha256": row["metadata_sha256"],
        "results_sha256": series._file_digest(result_path), "manifest_sha256": series._file_digest(manifest_path),
        "inventory_sha256": series._digest(inventory), **counts}
    receipt_path = output / "receipt.json"
    if receipt_path.exists():
        old = series._read(receipt_path)
        require(all(old.get(k) == v for k, v in receipt.items()), "Validated output changed after receipt")
    else:
        series._atomic_json(receipt_path, receipt)
    return receipt


def run_cell(inventory, root, index, expected_commit, *, smoke=False, smoke_steps=3):
    row = inventory["entries"][index]
    require(not row.get("reuse_output"), "500k is reuse-only; never reevaluate it")
    require(1 <= smoke_steps <= 3, "Smoke must use at most three decisions")
    verify_source(expected_commit)
    verify_entry(row)
    output = output_dir(Path(root).resolve(), index, "smoke" if smoke else "production")
    require(not output.is_relative_to(ROOT), "Outputs must be outside source")
    output.mkdir(parents=True, exist_ok=False)
    native = output / "checkpoint-inventory.json"
    series._atomic_json(native, native_inventory(row))
    command = base_command(row, native)
    if smoke:
        command += ["--seeds", "101", "--max-steps", str(smoke_steps)]
    command += ["--bundle-dir", str(output / "bundle"), "--output", str(output / "results.json")]
    series._atomic_json(output / "command.json", command)
    started = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    if smoke:
        result = {"mode": "smoke", "validated": True,
            **validate_results(series._read(output / "results.json"), row, smoke=True, smoke_steps=smoke_steps)}
        series._atomic_json(output / "receipt.json", result)
    else:
        result = validate_completed(inventory, root, index)
        require(result is not None, "Missing completed production result")
    return {**result, "elapsed_seconds": time.perf_counter() - started}


def completed_records(inventory, root, index):
    row = inventory["entries"][index]
    if row.get("reuse_output"):
        # This checks old receipt hashes, frozen/full episode contracts and
        # original five-seed normalization before any subset is derived.
        records = previous.completed_records(row, Path(row["reuse_output"]))
        require(records is not None, "Missing original 500k receipt")
        for arm in ARMS:
            spec = series._read(output_dir(root, index, "prepare") / "specs" / f"comparison__{arm}.json")
            records[arm] = subset_record(records[arm], spec["identity"])
    else:
        if validate_completed(inventory, root, index) is None:
            return None
        output = output_dir(root, index)
        records = {r["selector"].split("/")[-1]: r for r in raw_records(row, output)}
        for record in records.values():
            # Full traces remain durable on scratch with content hashes;
            # complete manifests/results/receipts remain upload artifacts.
            traces = {name: path for name, path in record["artifact_files"].items()
                      if name.endswith((".jsonl.gz", ".jsonl"))}
            record["provenance"]["remote_trace_files"] = [
                {"path": path, "sha256": series._file_digest(path)} for path in traces.values()]
            record["artifact_files"] = {n: p for n, p in record["artifact_files"].items() if n not in traces}
            record["artifact_files"].update({"results.json": str(output / "results.json"),
                                            "receipt.json": str(output / "receipt.json")})
    records["sac"] = paired_record(records["sac"], records["prior"])
    return records


def verify_workspace(receipt, path, wandb_module=None):
    require(Path(path).is_file(), "Install native workspace panels before publication")
    saved = series._read(path)
    ids = {row["run_id"] for row in receipt["curves"].values()}
    require(saved.get("campaign_id") == ATTEMPT and saved.get("api_verified") is True and
            set(saved.get("run_ids", [])) == ids, "Workspace receipt does not verify both curves")
    parsed = urlparse(saved["url"])
    require(parsed.hostname in ("wandb.ai", "forge.coreweave.com") and
            parsed.path.rstrip("/").endswith(f"/{ENTITY}/{PROJECT}"), "Wrong workspace project")
    view = parse_qs(parsed.query).get("nw", [None])[0]
    require(bool(view), "A dedicated named workspace is required")
    if wandb_module is None:
        import wandb as wandb_module
    try:
        from wandb_gql import gql
    except ImportError:
        from wandb.vendor.gql import gql
    query = gql('query View($entityName:String,$name:String,$viewName:String) { project(name:$name,entityName:$entityName) { allViews(viewType:"project-view",viewName:$viewName) { edges { node { id spec } } } } }')
    response = wandb_module.Api(timeout=30).client.execute(query, variable_values={
        "entityName": ENTITY, "name": PROJECT, "viewName": "nw-" + view + "-v"})
    edges = (response.get("project") or {}).get("allViews", {}).get("edges", [])
    require(len(edges) == 1, "Dedicated workspace is missing")
    wire = json.dumps(edges[0]["node"]["spec"], sort_keys=True)
    require(all(token in wire for token in [*ids, series.X_AXIS, "eval/return_mean",
        "eval/paired_gain_mean", "progress/checkpoints_complete", "progress/failed_cells"]),
        "Workspace lost required visible return, gain or progress panels")
    return saved


def progress_rows(receipt):
    rows = []
    for arm, curve in receipt["curves"].items():
        entries = list(series._read(Path(curve["run_dir"]) / "publication.json")["records"].values())
        rows.append({**curve, "arm": arm, "complete": len(entries),
                     "published": sum(e["status"] == "published" for e in entries)})
    return rows


def publish_once(inventory, root, *, wandb_module=None, workspace_verifier=verify_workspace):
    root = Path(root)
    publication = root / "publication"
    with series._lock(publication / ".publisher.lock", blocking=False):
        receipt = series._read(publication / "publication-map.json")
        require(receipt["inventory_sha256"] == series._digest(inventory), "Inventory changed after preparation")
        workspace = workspace_verifier(receipt, publication / "workspace.json", wandb_module)
        failures = {}
        accepted_steps = set(STEPS)
        for curve in receipt["curves"].values():
            accepted_steps &= {entry["checkpoint_step"] for entry in
                series._read(Path(curve["run_dir"]) / "publication.json")["records"].values()}
        # Reuse is available immediately, even while all six GPU cells queue.
        for index in [6, *range(6)]:
            # The immutable accepted records already pin all source hashes.
            # Avoid re-reading gigabytes of completed traces on every poll.
            if inventory["entries"][index]["step"] in accepted_steps:
                continue
            try:
                records = completed_records(inventory, root, index)
                if records:
                    for arm in ARMS:
                        series.stage_record(receipt["curves"][arm]["run_dir"], records[arm])
            except Exception as exc:
                failures[str(index)] = str(exc)
        for row in progress_rows(receipt):
            with series.Publisher(row["run_dir"], owner=OWNER, wandb_module=wandb_module) as publisher:
                publisher.run.name = row["name"]
                publisher.run.config.update({"campaign_id": ATTEMPT, "curve_role": row["arm"],
                    "source_run": SOURCE, "evaluation_seeds": SEEDS, "expected_checkpoint_steps": STEPS})
                for metric in VISIBLE:
                    publisher.run.define_metric(metric, step_metric=series.X_AXIS, step_sync=False, hidden=False)
                publisher.publish_pending()
                publisher.run.summary.update({"progress/checkpoints_complete": row["complete"],
                    "progress/checkpoints_expected": len(STEPS),
                    "progress/checkpoints_pending": len(STEPS) - row["complete"],
                    "progress/failed_cells": len(failures),
                    "progress/status": "failed validation" if failures else
                        ("complete" if row["complete"] == len(STEPS) else "pending results")})
        result = {"campaign_id": ATTEMPT, "runs": progress_rows(receipt), "failures": failures,
                  "workspace_url": workspace["url"]}
        series._atomic_json(publication / "status.json", result)
        require(not failures, "Evaluation validation failed: " + json.dumps(failures, sort_keys=True))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "prepare-publication", "run", "smoke", "validate", "publish", "watch"))
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--expected-commit")
    parser.add_argument("--smoke-steps", type=int, default=3)
    parser.add_argument("--compute-jobs")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    inventory = load_inventory(args.inventory)
    if args.mode == "prepare":
        result = prepare(inventory, args.root, args.expected_commit)
    elif args.mode == "prepare-publication":
        result = prepare_publication(inventory, args.root)
    elif args.mode in ("run", "smoke", "validate"):
        require(args.index is not None and args.index in range(7), "Expected a valid checkpoint index")
        result = (validate_completed(inventory, args.root, args.index) if args.mode == "validate" else
                  run_cell(inventory, args.root, args.index, args.expected_commit,
                           smoke=args.mode == "smoke", smoke_steps=args.smoke_steps))
    else:
        require(1 <= args.poll_seconds <= 60, "Poll interval must be 1–60 seconds")
        if args.mode == "watch":
            require(bool(re.fullmatch(r"\d+(,\d+)*", args.compute_jobs or "")), "Explicit compute job IDs required")
        try:
            while True:
                result = publish_once(inventory, args.root)
                if args.mode != "watch":
                    break
                active = subprocess.check_output(["squeue", "-h", "-j", args.compute_jobs], text=True).strip()
                if not active:
                    result = publish_once(inventory, args.root)
                    require(all(row["published"] == len(STEPS) for row in result["runs"]),
                            "Compute ended with incomplete publication")
                    break
                time.sleep(args.poll_seconds)
        except Exception as exc:
            series._atomic_json(args.root / "publication" / "failure.json", {"mode": args.mode, "error": str(exc)})
            raise
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
