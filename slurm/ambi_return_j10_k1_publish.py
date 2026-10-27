"""Complete the J10/H3/K1 comparison, reusing eight published controls."""
from __future__ import annotations

import argparse
import copy
from functools import partial
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm import ambi_return_j10_publish as j10
from slurm import ambi_return_prior_publish as prior
from slurm import ambi_return_cadence_publish as cadence
from slurm import ambi_return_h4_publish as shared
from utils import eval_series as series

ATTEMPT = "return-prior-j10-k1-20261026"
SELECTOR = "rounds/j10_k1"
INDICES = (1, 2)
METRICS = (
    ("Episode return: mean of five seeds", "eval/return_mean"),
    ("Improvement over the same checkpoint's prior", "eval/paired_gain_mean"),
    ("J10 K1 minus K3: same-checkpoint paired mean", "eval/cadence_delta_mean"),
    ("J10 K1 minus K3: paired sample SD", "eval/cadence_delta_sample_std"),
    ("J10 K1 exceeds K3: fraction of paired seeds", "eval/cadence_win_fraction"),
    ("Controller seconds per decision", "runtime/control_seconds_per_decision"),
)
LAYOUT = shared.Campaign(attempt=ATTEMPT, selector=SELECTOR, prefix="Frozen J10 K1 comparison / ",
    title="500k checkpoints: J10 H3 at every real decision", metrics=METRICS,
    description=("**All checkpoints are frozen at 500k decisions.** Train J/K describe checkpoint training; eval J/K "
        "describe this controller. Two new J10/H3/K1 evaluations complete the K1 comparison for the J8/K3-trained "
        "and J10/K3-trained checkpoints. The J8/K1-trained checkpoint's J10/K1 evaluation already exists and is reused. "
        "All eight previous runs are reused unchanged. Mean actions, environment seeds 101–105, controller seed 55, "
        "and complete 500-decision episodes are shared. Prior gains subtract each checkpoint's own prior. K1-minus-K3 "
        "gains pair only the same checkpoint and seed at J10/H3; positive values favor solving every decision. "
        "These cadence gains appear only on the two new runs. Cross-checkpoint differences do not isolate training "
        "J or K. Missing bars are pending or inapplicable, never zero returns. Sample SD describes episode variability, "
        "not training-seed uncertainty. K1 spends more inner computation than K3."))
install_workspace = partial(shared.install_workspace, campaign=LAYOUT)
verify_workspace = partial(shared.verify_workspace, campaign=LAYOUT)
_CONTEXT_CACHE = {}


def load_inventory(path):
    data = series._read(path)
    if data.get("schema_version") != 1 or data.get("campaign_id") != ATTEMPT or len(data.get("entries", [])) != 3:
        raise ValueError("Expected the J10/K1 completion inventory")
    for index, entry in enumerate(data["entries"]):
        if (entry["J"], entry["K"], entry["step"]) != ((8, 8, 10)[index], (1, 3, 3)[index], 500000):
            raise ValueError("Expected three selected 500k training J/K identities")
        if not re.fullmatch(r"[\w.-]+/[\w.-]+/[\w.-]+", entry["source_run"]) or any(
                not re.fullmatch(r"[0-9a-f]{64}", entry[key]) for key in ("sha256", "metadata_sha256")):
            raise ValueError("Invalid source identity or checkpoint SHA256 pins")
    if len({entry["source_run"] for entry in data["entries"]}) != 3:
        raise ValueError("Expected three distinct training runs")
    return data


def validate_spec(entry, index, spec):
    if index not in INDICES or (entry["J"], entry["K"], entry["step"]) != ((8 if index == 1 else 10), 3, 500000):
        raise ValueError("Only the two selected K3-trained checkpoints need new K1 runs")
    if spec.get("selector") != SELECTOR:
        raise ValueError("Unexpected J10/K1 selector")
    prior.validate_spec({**entry, "J": 10, "K": 1}, "sac", {**spec, "selector": "comparison/sac"})


def context_records(inventory, reference_root, j10_root, pins=None):
    """Pin small files every pass; normalize and hash large trace bundles once."""
    reference_root, j10_root = Path(reference_root).resolve(), Path(j10_root).resolve()
    old_inventory = j10.load_inventory(j10_root / "inventory.json")
    if old_inventory["entries"] != inventory["entries"]:
        raise ValueError("J10/K1 must reuse the exact three J10 campaign checkpoints")
    map_path = j10_root / "publication/publication-map.json"
    old = series._read(map_path)
    if (old.get("campaign_id") != j10.ATTEMPT or old.get("inventory_sha256") != series._digest(old_inventory)
            or old.get("reference_root") != str(reference_root)):
        raise ValueError("The J10 reference publication map changed")
    curves = cadence.layout_curves(old)
    wanted = {"cell-0_prior", "cell-0_j8", "cell-1_prior", "cell-1_j8",
              "cell-0_j10", "cell-1_j10", "cell-2_prior", "cell-2_sac"}
    if set(curves) != wanted or len({row["run_id"] for row in curves.values()}) != 8:
        raise ValueError("Expected exactly eight distinct published J10 comparison controls")
    observed = {"publication_map": series._file_digest(map_path), "outputs": {}, "registries": {}}
    for index in range(3):
        observed["outputs"][f"j10-{index}"] = shared.file_pins(j10_root / f"production/cell-{index}")
        if index < 2:
            observed["outputs"][f"prior-{index}"] = shared.file_pins(reference_root / f"production/cell-{index}")
    for key, row in curves.items():
        observed["registries"][key] = {name: series._file_digest(Path(row["run_dir"]) / name)
            for name in ("run.json", "publication.json")}
    if pins is not None and observed != pins:
        raise ValueError("A frozen J10 comparison control changed")
    cache_key = (str(reference_root), str(j10_root), series._digest(inventory["entries"]), series._digest(observed))
    if cache_key not in _CONTEXT_CACHE:
        records = {}
        for index, entry in enumerate(inventory["entries"]):
            if index < 2:
                controls, _ = cadence.reference_records(entry, index, reference_root, old["references"][str(index)])
                records[f"cell-{index}_prior"], records[f"cell-{index}_j8"] = controls["prior"], controls["sac"]
            result = j10.completed_records(entry, index, j10_root / f"production/cell-{index}",
                reference_root, old["references"].get(str(index)))
            if result is None:
                raise ValueError("The J10 reference campaign must be complete")
            records.update({f"cell-{index}_{arm}": record for arm, record in result.items()})
        for key, record in records.items():
            shared.validate_published_curve(curves[key], record)
        _CONTEXT_CACHE[cache_key] = records
    return copy.deepcopy(_CONTEXT_CACHE[cache_key]), curves, observed


def prepare(inventory, root, reference_root, j10_root, prepared_root=None):
    root, reference_root, j10_root = (Path(path).resolve() for path in (root, reference_root, j10_root))
    prepared_root = Path(prepared_root).resolve() if prepared_root else root / "prepare"
    publication = root / "publication"
    publication.mkdir(parents=True, exist_ok=True)
    path = publication / "publication-map.json"
    identity = {"campaign_id": ATTEMPT, "inventory_sha256": series._digest(inventory),
        "reference_root": str(reference_root), "j10_root": str(j10_root)}
    with series._lock(publication / ".prepare.lock"):
        receipt = series._read(path) if path.exists() else {"schema_version": 1, **identity,
            "owner": prior.OWNER, "attempt_label": ATTEMPT, "curves": {}, "cell_run_maps": {}}
        if any(receipt.get(key) != value for key, value in identity.items()):
            raise ValueError("Prepared J10/K1 inventory or reference roots changed")
        _, curves, pins = context_records(inventory, reference_root, j10_root, receipt.get("context_pins"))
        receipt.update(context_curves=curves, context_pins=pins,
            references={"1": pins["outputs"]["prior-1"], "2": pins["outputs"]["j10-2"]})
        for index in INDICES:
            entry = inventory["entries"][index]
            spec = series._read(prepared_root / f"cell-{index}/specs/rounds__j10_k1.json")
            validate_spec(entry, index, spec)
            key = f"cell-{index}_k1"
            owned = list((publication / "runs" / key).glob("*/run.json"))
            if key in receipt["curves"]:
                registry = series.validate_identity(receipt["curves"][key]["run_dir"], spec["identity"])
            elif owned:
                if len(owned) != 1:
                    raise ValueError("Ambiguous interrupted J10/K1 run allocation")
                registry = series.validate_identity(owned[0].parent, spec["identity"])
            else:
                registry = series.create_run(publication / "runs" / key, spec, ATTEMPT, prior.PROJECT, prior.ENTITY, prior.OWNER)
            if registry["attempt_label"] != ATTEMPT or registry["owner"] != prior.OWNER:
                raise ValueError("J10/K1 run owner or attempt changed")
            receipt["curves"][key] = {"run_id": registry["run_id"], "run_dir": registry["run_dir"], "index": index,
                "source_run": entry["source_run"], "source_J": entry["J"], "source_K": 3, "eval_J": 10, "eval_K": 1,
                "name": f"Train J{entry['J']} K3 @ 500k | SAC eval J10 H3 K1",
                "color": ("#F59E0B" if index == 1 else "#22C55E"), "step": 500000}
            series._atomic_json(path, receipt)
            map_path = publication / f"cell-{index}/eval-run-map.json"
            series._atomic_json(map_path, {SELECTOR: registry["run_dir"]})
            receipt["cell_run_maps"][str(index)] = str(map_path)
        series._atomic_json(path, receipt)
    return receipt


def completed_record(entry, index, output, controls, pins):
    output = Path(output)
    if not (output / "receipt.json").exists():
        return None
    receipt = series._read(output / "receipt.json")
    expected = {"validated": True, "mode": "run", "campaign_id": ATTEMPT, "source_run": entry["source_run"],
        "step": 500000, "J": 10, "source_J": entry["J"], "eval_J": 10, "source_K": 3, "eval_K": 1,
        "source_H": 3, "eval_H": 3, "selector": SELECTOR, "episodes": 5, "decisions": 2500, "solves": 2500,
        "checkpoint_sha256": entry["sha256"], "metadata_sha256": entry["metadata_sha256"]}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("J10/K1 receipt disagrees with the authorized checkpoint/controller")
    if any(receipt.get("reference", {}).get(field) != pins[name] for name, field in (
            ("receipt.json", "receipt_sha256"), ("results.json", "results_sha256"), ("bundle/manifest.json", "manifest_sha256"))):
        raise ValueError("J10/K1 receipt used a different prior reference")
    if any(series._file_digest(output / name) != receipt.get(field) for name, field in (
            ("results.json", "results_sha256"), ("bundle/manifest.json", "manifest_sha256"))):
        raise ValueError("J10/K1 result changed after validation")
    manifest = series._read(output / "bundle/manifest.json")
    if manifest.get("status") != "complete" or len(manifest.get("runs", [])) != 1 or manifest["runs"][0].get("status") != "complete":
        raise ValueError("Expected exactly one completed J10/K1 arm")
    from utils.eval_series_data import load_records
    records = load_records(output / "bundle/manifest.json", source_run=entry["source_run"], inventory_path=output / "checkpoint-inventory.json")
    if len(records) != 1:
        raise ValueError("J10/K1 bundle must normalize to one arm")
    record = records[0]
    validate_spec(entry, index, record)
    if record["checkpoint"] != {"step": 500000, "sha256": entry["sha256"]}:
        raise ValueError("J10/K1 checkpoint changed")
    if sorted(ep["seed"] for ep in record["episodes"]) != prior.SEEDS or any(
            ep["length"] != 500 or not ep.get("truncated") or ep.get("terminated") or ep.get("truncated_by_evaluator")
            for ep in record["episodes"]):
        raise ValueError("J10/K1 result lacks five complete environment-limited episodes")
    traces = set(manifest["runs"][0].get("trace_files", []))
    record["provenance"]["remote_trace_files"] = [{"path": path, "sha256": series._file_digest(path)}
        for name, path in record["artifact_files"].items() if name in traces]
    record["artifact_files"] = {name: path for name, path in record["artifact_files"].items() if name not in traces}
    record["artifact_files"].update({name: str(output / name) for name in ("results.json", "receipt.json")})
    return cadence.add_cadence_delta(prior.paired_record(record, controls[f"cell-{index}_prior"]),
        controls[f"cell-{index}_{'j10' if index == 1 else 'sac'}"])


def publish_curve(row, failed, *, wandb_module=None, sleep=time.sleep):
    """Retry acknowledgment uncertainty only, always through journal reconciliation."""
    for attempt in range(3):
        try:
            with series.Publisher(row["run_dir"], owner=prior.OWNER, wandb_module=wandb_module,
                                  acknowledgement_timeout=60) as publisher:
                publisher.run.name = row["name"]
                publisher.run.config.update({"campaign_id": ATTEMPT, "source_J": row["source_J"], "source_K": 3,
                    "evaluation_J": 10, "evaluation_K": 1, "evaluation_H": 3})
                for _, metric in METRICS:
                    publisher.run.define_metric(metric, step_metric=series.X_AXIS, step_sync=False, hidden=False)
                publisher.publish_pending()
                publisher.run.summary.update({"progress/checkpoints_complete": row["complete"], "progress/checkpoints_expected": 1,
                    "progress/failed_cells": int(failed), "progress/status": "failed validation" if failed else
                    ("complete" if row["complete"] else "pending results")})
            return
        except series.PublicationUncertainError:
            if attempt == 2:
                raise
            sleep(15 * (attempt + 1))


def publish_once(inventory, root, reference_root, j10_root, *, wandb_module=None, workspace_verifier=verify_workspace):
    root, reference_root, j10_root = (Path(path).resolve() for path in (root, reference_root, j10_root))
    publication = root / "publication"
    with series._lock(publication / ".publisher.lock", blocking=False):
        receipt = series._read(publication / "publication-map.json")
        if (receipt["inventory_sha256"] != series._digest(inventory) or receipt["reference_root"] != str(reference_root)
                or receipt["j10_root"] != str(j10_root) or receipt["campaign_id"] != ATTEMPT):
            raise ValueError("J10/K1 inventory/reference identity changed")
        workspace = workspace_verifier(receipt, publication / "workspace.json", wandb_module)
        controls, _, _ = context_records(inventory, reference_root, j10_root, receipt["context_pins"])
        failures, publication_failures = {}, {}
        for index in INDICES:
            try:
                record = completed_record(inventory["entries"][index], index, root / f"production/cell-{index}",
                    controls, receipt["references"][str(index)])
                if record is not None:
                    series.stage_record(receipt["curves"][f"cell-{index}_k1"]["run_dir"], record)
            except Exception as exc:
                failures[str(index)] = str(exc)
        for row in prior.progress_rows(receipt):
            try:
                publish_curve(row, str(row["index"]) in failures, wandb_module=wandb_module)
            except series.PublicationUncertainError as exc:
                publication_failures[str(row["index"])] = str(exc)
        result = {"campaign_id": ATTEMPT, "updated_at": prior.now(), "runs": prior.progress_rows(receipt),
            "failures": failures, "publication_failures": publication_failures, "workspace_url": workspace["url"]}
        series._atomic_json(publication / "status.json", result)
        if failures:
            raise ValueError("J10/K1 validation failed: " + json.dumps(failures))
        if publication_failures:
            raise series.PublicationUncertainError("Retained J10/K1 journals need reconciliation: " + json.dumps(publication_failures))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "layout", "publish", "watch"))
    for name in ("inventory", "root", "reference-root", "j10-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--compute-jobs")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    try:
        inventory = load_inventory(args.inventory)
        if args.command == "prepare":
            result = prepare(inventory, args.root, args.reference_root, args.j10_root, args.prepared_root)
        elif args.command == "layout":
            receipt = series._read(args.root / "publication/publication-map.json")
            if receipt["inventory_sha256"] != series._digest(inventory):
                raise ValueError("J10/K1 inventory changed")
            result = install_workspace(receipt, args.root / "publication/workspace.json")
        else:
            if args.command == "watch" and (not args.compute_jobs or not re.fullmatch(r"\d+(,\d+)*", args.compute_jobs)):
                raise ValueError("watch requires explicit numeric compute job IDs")
            if not 1 <= args.poll_seconds <= 60:
                raise ValueError("Polling must be between one and 60 seconds")
            while True:
                result = publish_once(inventory, args.root, args.reference_root, args.j10_root)
                if args.command != "watch":
                    break
                if not subprocess.check_output(["squeue", "-h", "-j", args.compute_jobs], text=True).strip():
                    result = publish_once(inventory, args.root, args.reference_root, args.j10_root)
                    if any(row["published"] != 1 for row in result["runs"]):
                        raise ValueError("Compute ended before both J10/K1 results were published")
                    break
                time.sleep(args.poll_seconds)
        print(json.dumps(result, allow_nan=False))
    except Exception as exc:
        series._atomic_json(args.root / "publication/failure.json", {"at": prior.now(), "command": args.command, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
