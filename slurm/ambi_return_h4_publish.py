"""Publish two H4/K1 evaluations with immutable prior and H3/K1 controls."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm import ambi_return_cadence_publish as cadence
from slurm import ambi_return_prior_publish as prior
from utils import eval_series as series

ATTEMPT = "return-prior-h4-k1-20261026"
SELECTOR = "horizon/h4_k1"
PREFIX = "Frozen H4 K1 / "
METRICS = [
    ("Episode return: mean of five seeds", "eval/return_mean"),
    ("Paired improvement over the same checkpoint's prior", "eval/paired_gain_mean"),
    ("H4 minus H3 at K1: paired mean", "eval/horizon_delta_mean"),
    ("H4 minus H3 at K1: paired sample SD", "eval/horizon_delta_sample_std"),
    ("H4 exceeds H3: fraction of paired seeds", "eval/horizon_win_fraction"),
    ("Controller seconds per decision", "runtime/control_seconds_per_decision"),
]
_H3_CACHE = {}


def validate_spec(entry, index, spec):
    if index not in (0, 1) or (entry["J"], entry["K"], entry["step"]) != (8, (1, 3)[index], 500000):
        raise ValueError("H4 comparison requires the two selected J8 500k checkpoints")
    identity = spec["identity"]
    planner, protocol = identity["planner"], identity["protocol"]
    settings = planner.get("settings", {})
    if (spec.get("selector") != SELECTOR or identity["backbone"] != entry["source_run"]
            or planner.get("type") != "sac" or planner.get("action_rule") != "tanh_mean"
            or (settings.get("inner_rounds"), settings.get("inner_rollout_horizon"),
                settings.get("inner_solve_interval", 1)) != (8, 4, 1)):
        raise ValueError("Expected the selected checkpoint with SAC J8/H4/K1 mean execution")
    if (protocol.get("action_rule") != "tanh_mean" or protocol.get("environment_seeds") != prior.SEEDS
            or protocol.get("controller_seed") != 55 or protocol.get("max_steps") != 500):
        raise ValueError("H4 requires five paired complete mean-action episodes")


def file_pins(output):
    return {name: series._file_digest(Path(output) / name)
            for name in ("receipt.json", "results.json", "bundle/manifest.json")}


def reference_records(entry, index, reference_root, cadence_root, pins=None):
    original, prior_pins = cadence.reference_records(entry, index, reference_root, pins["prior"] if pins else None)
    h3 = original["sac"]
    h3_pins = prior_pins
    if index == 1:
        output = Path(cadence_root) / "production/cell-1"
        h3_pins = file_pins(output)
        if pins and h3_pins != pins["h3"]:
            raise ValueError("Pinned H3/K1 comparison changed")
        key = (str(output.resolve()), series._digest(entry), series._digest(h3_pins), series._digest(prior_pins))
        if key not in _H3_CACHE:
            _H3_CACHE[key] = cadence.completed_record(entry, 1, output, reference_root, prior_pins)
        h3 = copy.deepcopy(_H3_CACHE[key])
    observed = {"prior": prior_pins, "h3": h3_pins}
    if pins is not None and pins != observed:
        raise ValueError("Pinned H4 controls changed")
    if h3 is None:
        raise ValueError("H3/K1 control is incomplete")
    prior.validate_spec({**entry, "K": 1}, "sac", {**h3, "selector": "comparison/sac"})
    # This common pairing validator checks checkpoint, complete seed set and protocol.
    prior.paired_record(h3, original["prior"])
    return {"prior": original["prior"], "h3": h3}, observed


def validate_published_curve(curve, record):
    registry = series.validate_identity(curve["run_dir"], record["identity"])
    journal = series._read(Path(curve["run_dir"]) / "publication.json")["records"]
    accepted = journal.get(record["record_id"], {})
    hashes = {name: series._file_digest(path) for name, path in record["artifact_files"].items()}
    if (registry["run_id"] != curve["run_id"] or len(journal) != 1 or accepted.get("status") != "published"
            or accepted.get("record_sha256") != series._record_fingerprint(record, hashes)):
        raise ValueError("Reused curve does not contain the exact validated published control")


def prepare(inventory, root, reference_root, cadence_root, prepared_root=None):
    root, reference_root, cadence_root = map(lambda p: Path(p).resolve(), (root, reference_root, cadence_root))
    prepared_root = Path(prepared_root).resolve() if prepared_root else root / "prepare"
    publication = root / "publication"
    publication.mkdir(parents=True, exist_ok=True)
    original = series._read(reference_root / "publication/publication-map.json")
    swaps = series._read(cadence_root / "publication/publication-map.json")
    if any(item["inventory_sha256"] != series._digest(inventory) for item in (original, swaps)):
        raise ValueError("Reference campaign uses a different frozen inventory")
    path = publication / "publication-map.json"
    identity = {"campaign_id": ATTEMPT, "inventory_sha256": series._digest(inventory),
                "reference_root": str(reference_root), "cadence_root": str(cadence_root)}
    with series._lock(publication / ".prepare.lock"):
        receipt = series._read(path) if path.exists() else {"schema_version": 1, **identity,
            "owner": prior.OWNER, "attempt_label": ATTEMPT, "curves": {}, "context_curves": {},
            "references": {}, "cell_run_maps": {}}
        if any(receipt.get(key) != value for key, value in identity.items()):
            raise ValueError("Prepared H4 inventory or reference roots changed")
        for index, entry in enumerate(inventory["entries"][:2]):
            controls, pins = reference_records(entry, index, reference_root, cadence_root, receipt["references"].get(str(index)))
            receipt["references"][str(index)] = pins
            context = {"prior": original["curves"][f"cell-{index}_prior"],
                       "h3": original["curves"]["cell-0_sac"] if index == 0 else swaps["curves"]["cell-1_swap"]}
            for arm, curve in context.items():
                validate_published_curve(curve, controls[arm])
                name = f"Train K{entry['K']} @ 500k | " + ("Prior only" if arm == "prior" else "SAC H3 K1")
                key, row = f"cell-{index}_{arm}", {**curve, "name": name}
                if key in receipt["context_curves"] and receipt["context_curves"][key] != row:
                    raise ValueError("Previously selected H4 control identity changed")
                receipt["context_curves"][key] = row
            spec = series._read(prepared_root / f"cell-{index}/specs/horizon__h4_k1.json")
            validate_spec(entry, index, spec)
            key = f"cell-{index}_h4"
            owned = list((publication / "runs" / key).glob("*/run.json"))
            if key in receipt["curves"]:
                registry = series.validate_identity(receipt["curves"][key]["run_dir"], spec["identity"])
            elif owned:
                if len(owned) != 1:
                    raise ValueError("Ambiguous interrupted H4 run allocation")
                registry = series.validate_identity(owned[0].parent, spec["identity"])
            else:
                registry = series.create_run(publication / "runs" / key, spec, ATTEMPT, prior.PROJECT, prior.ENTITY, prior.OWNER)
            if registry["attempt_label"] != ATTEMPT or registry["owner"] != prior.OWNER:
                raise ValueError("H4 run owner or attempt changed")
            receipt["curves"][key] = {"run_id": registry["run_id"], "run_dir": registry["run_dir"], "index": index,
                "source_run": entry["source_run"], "source_K": entry["K"], "eval_K": 1, "eval_H": 4,
                "name": f"Train K{entry['K']} @ 500k | SAC H4 K1", "color": ("#14B8A6", "#F97316")[index], "step": entry["step"]}
            series._atomic_json(path, receipt)
            run_map = publication / f"cell-{index}/eval-run-map.json"
            series._atomic_json(run_map, {SELECTOR: registry["run_dir"]})
            receipt["cell_run_maps"][str(index)] = str(run_map)
        series._atomic_json(path, receipt)
    return receipt


def add_horizon_delta(record, h3):
    result = cadence.add_cadence_delta(record, h3)
    for suffix in ("delta_mean", "delta_sample_std", "win_fraction"):
        result["metrics"]["eval/horizon_" + suffix] = result["metrics"].pop("eval/cadence_" + suffix)
    for episode in result["episodes"]:
        episode["horizon_return_delta"] = episode.pop("cadence_return_delta")
    result["provenance"]["h3_k1_reference"] = result["provenance"].pop("original_sac_reference")
    result["record_id"] = series._digest({"h4_h3_pair": result["record_id"]})
    return result


def completed_record(entry, index, output, reference_root, cadence_root, pins):
    output = Path(output)
    if not (output / "receipt.json").exists():
        return None
    receipt = series._read(output / "receipt.json")
    expected = {"validated": True, "mode": "run", "campaign_id": ATTEMPT, "source_run": entry["source_run"],
        "step": entry["step"], "J": 8, "source_K": entry["K"], "eval_K": 1, "source_H": 3, "eval_H": 4,
        "selector": SELECTOR, "checkpoint_sha256": entry["sha256"], "metadata_sha256": entry["metadata_sha256"]}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("H4 receipt disagrees with the authorized checkpoint/controller")
    if any(receipt.get("reference", {}).get(field) != pins["prior"][name] for name, field in (
            ("receipt.json", "receipt_sha256"), ("results.json", "results_sha256"), ("bundle/manifest.json", "manifest_sha256"))):
        raise ValueError("H4 receipt used a different original prior reference")
    if any(series._file_digest(output / name) != receipt.get(field) for name, field in (
            ("results.json", "results_sha256"), ("bundle/manifest.json", "manifest_sha256"))):
        raise ValueError("H4 result changed after validation")
    manifest = series._read(output / "bundle/manifest.json")
    if manifest.get("status") != "complete" or len(manifest.get("runs", [])) != 1 or manifest["runs"][0].get("status") != "complete":
        raise ValueError("Expected exactly one completed H4 arm")
    from utils.eval_series_data import load_records
    records = load_records(output / "bundle/manifest.json", source_run=entry["source_run"], inventory_path=output / "checkpoint-inventory.json")
    if len(records) != 1:
        raise ValueError("H4 bundle must normalize to exactly one arm")
    record = records[0]
    validate_spec(entry, index, record)
    if record["checkpoint"] != {"step": entry["step"], "sha256": entry["sha256"]}:
        raise ValueError("H4 checkpoint changed")
    if sorted(ep["seed"] for ep in record["episodes"]) != prior.SEEDS or any(
            ep["length"] != 500 or not ep.get("truncated") or ep.get("terminated") or ep.get("truncated_by_evaluator")
            for ep in record["episodes"]):
        raise ValueError("H4 result lacks five complete environment-limited episodes")
    traces = set(manifest["runs"][0].get("trace_files", []))
    record["provenance"]["remote_trace_files"] = [{"path": path, "sha256": series._file_digest(path)}
        for name, path in record["artifact_files"].items() if name in traces]
    record["artifact_files"] = {name: path for name, path in record["artifact_files"].items() if name not in traces}
    record["artifact_files"].update({name: str(output / name) for name in ("results.json", "receipt.json")})
    controls, _ = reference_records(entry, index, reference_root, cadence_root, pins)
    return add_horizon_delta(prior.paired_record(record, controls["prior"]), controls["h3"])


def install_workspace(receipt, path):
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr
    path = Path(path)
    previous = series._read(path) if path.exists() else {}
    if previous and previous.get("campaign_id") != ATTEMPT:
        raise ValueError("Workspace belongs to another campaign")
    rows = list(cadence.layout_curves(receipt).values())
    def bar(title, metric):
        return wr.BarPlot(title=title, metrics=[metric], orientation="h", aggregate=False,
            max_runs_to_show=6, max_bars_to_show=6, legend_template="${run:displayName}",
            line_colors={row["run_id"] + ":" + metric: row["color"] for row in rows},
            line_titles={row["run_id"] + ":" + metric: row["name"] for row in rows}, layout=wr.Layout(w=12, h=7))
    markdown = ("**Frozen J8 checkpoints: H4 versus H3 at evaluation K1.** Train K labels each 500k checkpoint's "
        "original training cadence. Only the two H4/K1 evaluations are new. Both prior controls and both H3/K1 "
        "controls are reused from validated completed runs; the K3-trained H3/K1 control comes from the cadence swap. "
        "Mean actions, seeds 101–105, controller seed 55, and 500-decision full episodes are matched. "
        "Paired horizon gain is H4 minus H3 for the same checkpoint and environment seed; positive favors H4. "
        "Both horizons solve afresh at every real decision. Missing pending bars are not zero returns. "
        "Sample SD describes episode variability, not training-seed uncertainty; H4 uses more imagined transitions.")
    sections = [ws.Section(name=PREFIX + "Protocol and progress", is_open=True, panels=[
        wr.MarkdownPanel(markdown=markdown, layout=wr.Layout(w=24, h=8)),
        bar("Completed checkpoints per run (of one)", "progress/checkpoints_complete"),
        bar("Failed validation cells", "progress/failed_cells")]),
        ws.Section(name=PREFIX + "Returns, paired gains and cost", is_open=True,
            panels=[bar(title, metric) for title, metric in METRICS], layout_settings=ws.SectionLayoutSettings(columns=2, rows=3))]
    if previous.get("url"):
        work = ws.Workspace.from_url(previous["url"])
        if work.entity != prior.ENTITY or work.project != prior.PROJECT:
            raise ValueError("H4 workspace has the wrong target")
        replacements = {section.name: section for section in sections}
        work.sections = [replacements.pop(section.name, section) for section in work.sections]
        work.sections.extend(replacements.values())
    else:
        work = ws.Workspace(entity=prior.ENTITY, project=prior.PROJECT,
            name="Frozen J8 checkpoints: H4 versus H3 at K1", sections=sections, auto_generate_panels=False,
            settings=ws.WorkspaceSettings(max_runs=6, remove_legends_from_panels=False),
            runset_settings=ws.RunsetSettings(filters=[expr.Metric("ID").isin([row["run_id"] for row in rows])], groupby=[],
                run_settings={row["run_id"]: ws.RunSettings(color=row["color"], disabled=False) for row in rows},
                pinned_columns=["run:displayName", "summary:progress/checkpoints_complete", "summary:eval/return_mean"]))
    work.save()
    saved = {**previous, "campaign_id": ATTEMPT, "url": work.url, "updated_at": prior.now(),
        "run_ids": [row["run_id"] for row in rows], "panel_count": 9, "api_verified": False}
    saved.setdefault("browser_verified", False)
    series._atomic_json(path, saved)
    readback = ws.Workspace.from_url(work.url)
    panels = [panel for section in readback.sections if section.name.startswith(PREFIX) for panel in section.panels]
    wire = str([getattr(panel, "metrics", []) for panel in panels])
    if len(panels) != 9 or any(metric not in wire for _, metric in METRICS) or any(
            row["run_id"] not in readback.runset_settings.run_settings for row in rows):
        raise ValueError("H4 workspace failed native API read-back")
    saved["api_verified"] = True
    series._atomic_json(path, saved)
    return saved


def verify_workspace(receipt, path, wandb_module=None):
    path = Path(path)
    if not path.is_file():
        raise ValueError("Install H4 workspace panels before publication")
    saved = series._read(path)
    wanted = {row["run_id"] for row in cadence.layout_curves(receipt).values()}
    parsed = urlparse(saved["url"])
    view = parse_qs(parsed.query).get("nw", [None])[0]
    if (saved.get("campaign_id") != ATTEMPT or saved.get("api_verified") is not True or set(saved.get("run_ids", [])) != wanted
            or parsed.hostname != "wandb.ai" or parsed.path.rstrip("/") != f"/{prior.ENTITY}/{prior.PROJECT}" or not view):
        raise ValueError("H4 workspace must contain the six selected runs")
    if wandb_module is None:
        import wandb as wandb_module
    try:
        gql = importlib.import_module("wandb_gql").gql
    except ImportError:
        gql = importlib.import_module("wandb.vendor.gql").gql
    query = gql('query View($entityName:String,$name:String,$viewName:String) { project(name:$name,entityName:$entityName) { allViews(viewType:"project-view",viewName:$viewName) { edges { node { spec } } } } }')
    data = wandb_module.Api(timeout=30).client.execute(query, variable_values={"entityName": prior.ENTITY,
        "name": prior.PROJECT, "viewName": "nw-" + view + "-v"})
    edges = (data.get("project") or {}).get("allViews", {}).get("edges", [])
    if len(edges) != 1 or any(token not in json.dumps(edges) for token in [*wanted,
            "progress/checkpoints_complete", "progress/failed_cells", *[metric for _, metric in METRICS]]):
        raise ValueError("H4 workspace lost expected native panels or run IDs")
    return saved


def publish_once(inventory, root, reference_root, cadence_root, *, wandb_module=None, workspace_verifier=verify_workspace):
    root, reference_root, cadence_root = map(lambda p: Path(p).resolve(), (root, reference_root, cadence_root))
    publication = root / "publication"
    with series._lock(publication / ".publisher.lock", blocking=False):
        receipt = series._read(publication / "publication-map.json")
        if (receipt["inventory_sha256"] != series._digest(inventory) or receipt["reference_root"] != str(reference_root)
                or receipt["cadence_root"] != str(cadence_root) or receipt["campaign_id"] != ATTEMPT):
            raise ValueError("H4 inventory/reference identity changed")
        workspace = workspace_verifier(receipt, publication / "workspace.json", wandb_module)
        failures = {}
        for index, entry in enumerate(inventory["entries"][:2]):
            try:
                record = completed_record(entry, index, root / f"production/cell-{index}", reference_root, cadence_root, receipt["references"][str(index)])
                if record is not None:
                    series.stage_record(receipt["curves"][f"cell-{index}_h4"]["run_dir"], record)
            except Exception as exc:
                failures[str(index)] = str(exc)
        for row in prior.progress_rows(receipt):
            with series.Publisher(row["run_dir"], owner=prior.OWNER, wandb_module=wandb_module) as publisher:
                publisher.run.name = row["name"]
                publisher.run.config.update({"campaign_id": ATTEMPT, "source_K": row["source_K"], "evaluation_K": 1, "evaluation_H": 4, "J": 8})
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
            raise ValueError("H4 validation failed: " + json.dumps(failures))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "layout", "publish", "watch"))
    for name in ("inventory", "root", "reference-root", "cadence-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--compute-jobs")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    try:
        inventory = prior.load_inventory(args.inventory)
        if args.command == "prepare":
            result = prepare(inventory, args.root, args.reference_root, args.cadence_root, args.prepared_root)
        elif args.command == "layout":
            receipt = series._read(args.root / "publication/publication-map.json")
            if receipt["inventory_sha256"] != series._digest(inventory):
                raise ValueError("H4 inventory changed")
            result = install_workspace(receipt, args.root / "publication/workspace.json")
        else:
            if args.command == "watch" and (not args.compute_jobs or not re.fullmatch(r"\d+(,\d+)*", args.compute_jobs)):
                raise ValueError("watch requires explicit numeric compute job IDs")
            if not 1 <= args.poll_seconds <= 60:
                raise ValueError("Polling must be between one and 60 seconds")
            while True:
                result = publish_once(inventory, args.root, args.reference_root, args.cadence_root)
                if args.command != "watch":
                    break
                if not subprocess.check_output(["squeue", "-h", "-j", args.compute_jobs], text=True).strip():
                    result = publish_once(inventory, args.root, args.reference_root, args.cadence_root)
                    if any(row["published"] != 1 for row in result["runs"]):
                        raise ValueError("Compute ended before both H4 results were published")
                    break
                time.sleep(args.poll_seconds)
        print(json.dumps(result, allow_nan=False))
    except Exception as exc:
        series._atomic_json(args.root / "publication/failure.json", {"at": prior.now(), "command": args.command, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
