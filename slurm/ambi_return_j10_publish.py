"""J10 refinement on two J8 checkpoints, with a separate J10-trained reference."""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm import ambi_return_prior_publish as prior
from slurm import ambi_return_cadence_publish as cadence
from slurm import ambi_return_h4_publish as shared
from utils import eval_series as series

ATTEMPT = "return-prior-j10-20261026"
SELECTOR = "rounds/j10"
METRICS = (
    ("Episode return: mean of five seeds", "eval/return_mean"),
    ("Improvement over the same checkpoint's prior", "eval/paired_gain_mean"),
    ("Eval J10 minus J8: same-checkpoint paired mean", "eval/rounds_delta_mean"),
    ("Eval J10 minus J8: paired sample SD", "eval/rounds_delta_sample_std"),
    ("Eval J10 exceeds J8: fraction of paired seeds", "eval/rounds_win_fraction"),
    ("Controller seconds per decision", "runtime/control_seconds_per_decision"),
)
LAYOUT = shared.Campaign(attempt=ATTEMPT, selector=SELECTOR, prefix="Frozen J10 comparison / ",
    title="500k checkpoints: J10 refinement and J10-trained reference", metrics=METRICS,
    description=("**All checkpoints are frozen at 500k decisions.** Train J/K identify how the checkpoint was learned; "
        "eval J/K describe the controller used here. The two J8-trained checkpoints are evaluated with J10, retaining "
        "H3 and their original K1 or K3. Their J8 evaluations and prior controls are reused unchanged. The separately "
        "J10-trained/K3 checkpoint receives a new prior/J10 pair at 500k. All arms use mean actions, environment "
        "seeds 101–105, controller seed 55, and complete 500-decision episodes. Prior gains use each checkpoint's own "
        "prior. J10-minus-J8 gains pair only the same J8-trained checkpoint and environment seed; this metric is "
        "absent for the independently trained J10 checkpoint. Cross-checkpoint differences do not isolate training "
        "J. Missing bars are pending or inapplicable, never zero returns. Sample SD describes episode variability, "
        "not training-seed uncertainty; J10 spends more inner optimization and model work than J8."))
install_workspace = partial(shared.install_workspace, campaign=LAYOUT)
verify_workspace = partial(shared.verify_workspace, campaign=LAYOUT)


def load_inventory(path):
    data = series._read(path)
    entries = data.get("entries", [])
    if data.get("schema_version") != 1 or data.get("campaign_id") != ATTEMPT or len(entries) != 3:
        raise ValueError("Expected the selected J10 comparison inventory")
    if len({entry["source_run"] for entry in entries}) != 3:
        raise ValueError("J10 comparison requires three distinct training runs")
    for index, entry in enumerate(entries):
        if (entry["J"], entry["K"], entry["step"]) != ((8, 8, 10)[index], (1, 3, 3)[index], 500000):
            raise ValueError("Expected the three selected 500k training J/K identities")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+/[\w.-]+", entry["source_run"]):
            raise ValueError("Invalid checkpoint source identity")
        if any(not re.fullmatch(r"[0-9a-f]{64}", entry[key]) for key in ("sha256", "metadata_sha256")):
            raise ValueError("Checkpoint and metadata require SHA256 pins")
    return data


def arms(index):
    return ("j10",) if index < 2 else prior.ARMS


def selector(index, arm):
    return SELECTOR if index < 2 else "comparison/" + arm


def validate_spec(entry, index, arm, spec):
    if spec.get("selector") != selector(index, arm):
        raise ValueError("Unexpected J10 comparison selector")
    wanted_arm = "sac" if index < 2 else arm
    prior.validate_spec({**entry, "J": 10}, wanted_arm, {**spec, "selector": "comparison/" + wanted_arm})
    if wanted_arm == "sac" and entry["K"] > 1 and spec["identity"]["planner"]["settings"].get("inner_hold_encoder") != "solve_snapshot":
        raise ValueError("K3 must hold the adapted policy's solve-time encoder")


def label(entry, arm):
    prefix = f"Train J{entry['J']} K{entry['K']} @ 500k | "
    return prefix + ("Prior only" if arm == "prior" else f"SAC eval J{8 if arm == 'j8' else 10} H3 K{entry['K']}")


def prepare(inventory, root, reference_root, prepared_root=None):
    root, reference_root = Path(root).resolve(), Path(reference_root).resolve()
    prepared_root = Path(prepared_root).resolve() if prepared_root else root / "prepare"
    publication = root / "publication"
    publication.mkdir(parents=True, exist_ok=True)
    original = series._read(reference_root / "publication/publication-map.json")
    if original.get("campaign_id") != prior.ATTEMPT:
        raise ValueError("Expected the original three-checkpoint reference campaign")
    path = publication / "publication-map.json"
    identity = {"campaign_id": ATTEMPT, "inventory_sha256": series._digest(inventory), "reference_root": str(reference_root)}
    with series._lock(publication / ".prepare.lock"):
        receipt = series._read(path) if path.exists() else {"schema_version": 1, **identity,
            "owner": prior.OWNER, "attempt_label": ATTEMPT, "curves": {}, "context_curves": {}, "references": {}, "cell_run_maps": {}}
        if any(receipt.get(key) != value for key, value in identity.items()):
            raise ValueError("Prepared J10 inventory or reference root changed")
        for index, entry in enumerate(inventory["entries"]):
            if index < 2:
                controls, pins = cadence.reference_records(entry, index, reference_root, receipt["references"].get(str(index)))
                receipt["references"][str(index)] = pins
                # Validate the actual controls per entry: the new third 500k
                # entry intentionally differs from the old 325k inventory.
                for arm in prior.ARMS:
                    curve = original["curves"][f"cell-{index}_{arm}"]
                    if curve["source_run"] != entry["source_run"] or curve["step"] != entry["step"]:
                        raise ValueError("Original control belongs to another source or checkpoint")
                    shared.validate_published_curve(curve, controls[arm])
                    key = f"cell-{index}_{'prior' if arm == 'prior' else 'j8'}"
                    row = {**curve, "name": label(entry, "prior" if arm == "prior" else "j8")}
                    if key in receipt["context_curves"] and receipt["context_curves"][key] != row:
                        raise ValueError("Previously selected J8 control changed")
                    receipt["context_curves"][key] = row
            run_map = {}
            for arm in arms(index):
                selected = selector(index, arm)
                spec = series._read(prepared_root / f"cell-{index}/specs" / (selected.replace("/", "__") + ".json"))
                validate_spec(entry, index, arm, spec)
                key = f"cell-{index}_{arm}"
                owned = list((publication / "runs" / key).glob("*/run.json"))
                if key in receipt["curves"]:
                    registry = series.validate_identity(receipt["curves"][key]["run_dir"], spec["identity"])
                elif owned:
                    if len(owned) != 1:
                        raise ValueError("Ambiguous interrupted J10 run allocation")
                    registry = series.validate_identity(owned[0].parent, spec["identity"])
                else:
                    registry = series.create_run(publication / "runs" / key, spec, ATTEMPT, prior.PROJECT, prior.ENTITY, prior.OWNER)
                if registry["attempt_label"] != ATTEMPT or registry["owner"] != prior.OWNER:
                    raise ValueError("J10 run owner or attempt changed")
                receipt["curves"][key] = {"run_id": registry["run_id"], "run_dir": registry["run_dir"], "index": index,
                    "arm": arm, "source_run": entry["source_run"], "source_J": entry["J"], "source_K": entry["K"],
                    "name": label(entry, arm), "color": ("#14B8A6", "#DB2777", "#94A3B8" if arm == "prior" else "#7C3AED")[index], "step": entry["step"]}
                run_map[selected] = registry["run_dir"]
                series._atomic_json(path, receipt)
            run_map_path = publication / f"cell-{index}/eval-run-map.json"
            series._atomic_json(run_map_path, run_map)
            receipt["cell_run_maps"][str(index)] = str(run_map_path)
        series._atomic_json(path, receipt)
    return receipt


def add_rounds_delta(record, j8):
    result = cadence.add_cadence_delta(record, j8)
    for suffix in ("delta_mean", "delta_sample_std", "win_fraction"):
        result["metrics"]["eval/rounds_" + suffix] = result["metrics"].pop("eval/cadence_" + suffix)
    for episode in result["episodes"]:
        episode["rounds_return_delta"] = episode.pop("cadence_return_delta")
    result["provenance"]["j8_reference"] = result["provenance"].pop("original_sac_reference")
    result["record_id"] = series._digest({"j10_j8_pair": result["record_id"]})
    return result


def completed_records(entry, index, output, reference_root, pins=None):
    output = Path(output)
    if not (output / "receipt.json").exists():
        return None
    receipt = series._read(output / "receipt.json")
    expected = {"validated": True, "mode": "run", "campaign_id": ATTEMPT, "source_run": entry["source_run"],
        "step": 500000, "J": 10, "source_J": entry["J"], "eval_J": 10, "source_K": entry["K"], "eval_K": entry["K"],
        "source_H": 3, "eval_H": 3, "checkpoint_sha256": entry["sha256"], "metadata_sha256": entry["metadata_sha256"]}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("J10 receipt disagrees with the authorized checkpoint/controller")
    if index == 2:
        records = prior.completed_records(entry, output)
        for arm, record in records.items():
            validate_spec(entry, index, arm, record)
        return records
    if receipt.get("selector") != SELECTOR or any(receipt.get("reference", {}).get(field) != pins[name] for name, field in (
            ("receipt.json", "receipt_sha256"), ("results.json", "results_sha256"), ("bundle/manifest.json", "manifest_sha256"))):
        raise ValueError("J10 receipt used a different original reference")
    if any(series._file_digest(output / name) != receipt.get(field) for name, field in (
            ("results.json", "results_sha256"), ("bundle/manifest.json", "manifest_sha256"))):
        raise ValueError("J10 result changed after validation")
    manifest = series._read(output / "bundle/manifest.json")
    if manifest.get("status") != "complete" or len(manifest.get("runs", [])) != 1 or manifest["runs"][0].get("status") != "complete":
        raise ValueError("Expected exactly one completed J10 arm")
    from utils.eval_series_data import load_records
    records = load_records(output / "bundle/manifest.json", source_run=entry["source_run"], inventory_path=output / "checkpoint-inventory.json")
    if len(records) != 1:
        raise ValueError("J10 bundle must normalize to one arm")
    record = records[0]
    validate_spec(entry, index, "j10", record)
    if record["checkpoint"] != {"step": 500000, "sha256": entry["sha256"]}:
        raise ValueError("J10 checkpoint changed")
    if sorted(ep["seed"] for ep in record["episodes"]) != prior.SEEDS or any(
            ep["length"] != 500 or not ep.get("truncated") or ep.get("terminated") or ep.get("truncated_by_evaluator")
            for ep in record["episodes"]):
        raise ValueError("J10 result lacks five complete environment-limited episodes")
    traces = set(manifest["runs"][0].get("trace_files", []))
    record["provenance"]["remote_trace_files"] = [{"path": path, "sha256": series._file_digest(path)}
        for name, path in record["artifact_files"].items() if name in traces]
    record["artifact_files"] = {name: path for name, path in record["artifact_files"].items() if name not in traces}
    record["artifact_files"].update({name: str(output / name) for name in ("results.json", "receipt.json")})
    controls, _ = cadence.reference_records(entry, index, reference_root, pins)
    return {"j10": add_rounds_delta(prior.paired_record(record, controls["prior"]), controls["sac"])}


def publish_once(inventory, root, reference_root, *, wandb_module=None, workspace_verifier=verify_workspace):
    root, reference_root = Path(root).resolve(), Path(reference_root).resolve()
    publication = root / "publication"
    with series._lock(publication / ".publisher.lock", blocking=False):
        receipt = series._read(publication / "publication-map.json")
        if (receipt["inventory_sha256"] != series._digest(inventory) or receipt["reference_root"] != str(reference_root)
                or receipt["campaign_id"] != ATTEMPT):
            raise ValueError("J10 inventory/reference identity changed")
        workspace = workspace_verifier(receipt, publication / "workspace.json", wandb_module)
        failures = {}
        for index, entry in enumerate(inventory["entries"]):
            try:
                records = completed_records(entry, index, root / f"production/cell-{index}", reference_root, receipt["references"].get(str(index)))
                if records is not None:
                    for arm, record in records.items():
                        series.stage_record(receipt["curves"][f"cell-{index}_{arm}"]["run_dir"], record)
            except Exception as exc:
                failures[str(index)] = str(exc)
        for row in prior.progress_rows(receipt):
            with series.Publisher(row["run_dir"], owner=prior.OWNER, wandb_module=wandb_module) as publisher:
                publisher.run.name = row["name"]
                publisher.run.config.update({"campaign_id": ATTEMPT, "source_J": row["source_J"], "source_K": row["source_K"],
                    "evaluation_J": 0 if row["arm"] == "prior" else 10, "evaluation_K": 1 if row["arm"] == "prior" else row["source_K"], "evaluation_H": 3})
                for _, metric in METRICS:
                    publisher.run.define_metric(metric, step_metric=series.X_AXIS, step_sync=False, hidden=False)
                publisher.publish_pending()
                publisher.run.summary.update({"progress/checkpoints_complete": row["complete"], "progress/checkpoints_expected": 1,
                    "progress/failed_cells": int(str(row["index"]) in failures), "progress/status": "failed validation" if str(row["index"]) in failures else
                    ("complete" if row["complete"] else "pending results")})
        result = {"campaign_id": ATTEMPT, "updated_at": prior.now(), "runs": prior.progress_rows(receipt),
            "failures": failures, "workspace_url": workspace["url"]}
        series._atomic_json(publication / "status.json", result)
        if failures:
            raise ValueError("J10 validation failed: " + json.dumps(failures))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "layout", "publish", "watch"))
    for name in ("inventory", "root", "reference-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--compute-jobs")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    try:
        inventory = load_inventory(args.inventory)
        if args.command == "prepare":
            result = prepare(inventory, args.root, args.reference_root, args.prepared_root)
        elif args.command == "layout":
            receipt = series._read(args.root / "publication/publication-map.json")
            if receipt["inventory_sha256"] != series._digest(inventory):
                raise ValueError("J10 inventory changed")
            result = install_workspace(receipt, args.root / "publication/workspace.json")
        else:
            if args.command == "watch" and (not args.compute_jobs or not re.fullmatch(r"\d+(,\d+)*", args.compute_jobs)):
                raise ValueError("watch requires explicit numeric compute job IDs")
            if not 1 <= args.poll_seconds <= 60:
                raise ValueError("Polling must be between one and 60 seconds")
            while True:
                result = publish_once(inventory, args.root, args.reference_root)
                if args.command != "watch":
                    break
                if not subprocess.check_output(["squeue", "-h", "-j", args.compute_jobs], text=True).strip():
                    result = publish_once(inventory, args.root, args.reference_root)
                    if any(row["published"] != 1 for row in result["runs"]):
                        raise ValueError("Compute ended before all four J10 comparison results were published")
                    break
                time.sleep(args.poll_seconds)
        print(json.dumps(result, allow_nan=False))
    except Exception as exc:
        series._atomic_json(args.root / "publication/failure.json", {"at": prior.now(), "command": args.command, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
