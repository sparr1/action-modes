"""Publish two cadence swaps while reusing the four original comparison runs."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm import ambi_return_prior_publish as prior
from utils import eval_series as series

ATTEMPT = "return-prior-cadence-swap-20261026"
PREFIX = "Frozen cadence swap / "
TARGETS = (3, 1)
_REFERENCE_CACHE = {}
METRICS = [
    ("Episode return: mean of five seeds", "eval/return_mean"),
    ("Paired improvement over the same checkpoint's prior", "eval/paired_gain_mean"),
    ("Swap minus original SAC: paired mean", "eval/cadence_delta_mean"),
    ("Swap minus original SAC: paired sample SD", "eval/cadence_delta_sample_std"),
    ("Controller seconds per decision", "runtime/control_seconds_per_decision"),
    ("Episode return: sample standard deviation", "eval/return_sample_std"),
]


def validate_spec(entry, index, spec):
    if index not in (0, 1) or entry["J"] != 8 or entry["step"] != 500000 or entry["K"] != (1, 3)[index]:
        raise ValueError("Cadence comparison requires the two pinned J8 500k checkpoints")
    if spec.get("selector") != f"cadence/k{TARGETS[index]}":
        raise ValueError("Unexpected cadence selector")
    adapted = {**spec, "selector": "comparison/sac"}
    prior.validate_spec({**entry, "K": TARGETS[index]}, "sac", adapted)


def reference_records(entry, index, reference_root, pins=None):
    output = Path(reference_root) / "production" / f"cell-{index}"
    observed = {name: series._file_digest(output / name) for name in
                ("receipt.json", "results.json", "bundle/manifest.json")}
    if pins is not None and observed != pins:
        raise ValueError("Previously validated original reference changed")
    # Recheck compact immutable files each pass; hash large traces only once.
    key = (str(output.resolve()), series._digest(entry), series._digest(observed))
    if key not in _REFERENCE_CACHE:
        record = prior.completed_records(entry, output)
        if record is None:
            raise ValueError("Original prior/SAC comparison is not complete")
        _REFERENCE_CACHE[key] = record
    return copy.deepcopy(_REFERENCE_CACHE[key]), observed


def prepare(inventory, root, reference_root, prepared_root=None):
    root, reference_root = Path(root).resolve(), Path(reference_root).resolve()
    prepared_root = Path(prepared_root).resolve() if prepared_root else root / "prepare"
    publication = root / "publication"
    publication.mkdir(parents=True, exist_ok=True)
    original = series._read(reference_root / "publication/publication-map.json")
    if original["inventory_sha256"] != series._digest(inventory):
        raise ValueError("Original publication map belongs to a different inventory")
    path = publication / "publication-map.json"
    with series._lock(publication / ".prepare.lock"):
        receipt = series._read(path) if path.exists() else {
            "schema_version": 1, "campaign_id": ATTEMPT, "inventory_sha256": series._digest(inventory),
            "reference_root": str(reference_root), "owner": prior.OWNER,
            "attempt_label": ATTEMPT, "curves": {}, "context_curves": {}, "references": {}, "cell_run_maps": {}}
        if receipt["inventory_sha256"] != series._digest(inventory) or receipt["reference_root"] != str(reference_root):
            raise ValueError("Prepared inventory or reference owner changed")
        for index, entry in enumerate(inventory["entries"][:2]):
            _, pins = reference_records(entry, index, reference_root, receipt["references"].get(str(index)))
            receipt["references"][str(index)] = pins
            for arm in prior.ARMS:
                key = f"cell-{index}_{arm}"
                curve = original["curves"][key]
                journal = series._read(Path(curve["run_dir"]) / "publication.json")
                if len(journal["records"]) != 1 or any(row["status"] != "published" for row in journal["records"].values()):
                    raise ValueError("Reused original curve is not fully published")
                name = f"Train K{entry['K']} @ 500k | " + ("Prior only" if arm == "prior" else f"SAC eval K{entry['K']}")
                context = {**curve, "name": name}
                if key in receipt["context_curves"] and receipt["context_curves"][key] != context:
                    raise ValueError("Original curve identity changed")
                receipt["context_curves"][key] = context
            selector = f"cadence/k{TARGETS[index]}"
            spec = series._read(prepared_root / f"cell-{index}" / "specs" / (selector.replace("/", "__") + ".json"))
            validate_spec(entry, index, spec)
            key = f"cell-{index}_swap"
            owned = list((publication / "runs" / key).glob("*/run.json"))
            if key in receipt["curves"]:
                registry = series.validate_identity(receipt["curves"][key]["run_dir"], spec["identity"])
            elif owned:
                if len(owned) != 1:
                    raise ValueError("Ambiguous interrupted cadence-run allocation")
                registry = series.validate_identity(owned[0].parent, spec["identity"])
            else:
                registry = series.create_run(publication / "runs" / key, spec, ATTEMPT, prior.PROJECT, prior.ENTITY, prior.OWNER)
            if registry["attempt_label"] != ATTEMPT or registry["owner"] != prior.OWNER:
                raise ValueError("Cadence run owner or attempt changed")
            receipt["curves"][key] = {
                "run_id": registry["run_id"], "run_dir": registry["run_dir"], "index": index,
                "source_run": entry["source_run"], "source_K": entry["K"], "K": TARGETS[index],
                "name": f"Train K{entry['K']} @ 500k | SAC eval K{TARGETS[index]}",
                "color": ("#14B8A6", "#D946EF")[index], "step": entry["step"]}
            series._atomic_json(path, receipt)
            map_path = publication / f"cell-{index}" / "eval-run-map.json"
            series._atomic_json(map_path, {selector: registry["run_dir"]})
            receipt["cell_run_maps"][str(index)] = str(map_path)
        series._atomic_json(path, receipt)
    return receipt


def add_cadence_delta(record, original_sac):
    if (record["checkpoint"] != original_sac["checkpoint"]
            or record["identity"]["backbone"] != original_sac["identity"]["backbone"]
            or record["identity"]["protocol"] != original_sac["identity"]["protocol"]):
        raise ValueError("Cadence delta needs the same checkpoint and episode protocol")
    if original_sac["identity"]["planner"]["type"] != "sac":
        raise ValueError("Original cadence control is not SAC")
    baseline = {episode["seed"]: episode["return"] for episode in original_sac["episodes"]}
    if len(baseline) != 5 or sorted(baseline) != prior.SEEDS or sorted(ep["seed"] for ep in record["episodes"]) != prior.SEEDS:
        raise ValueError("Cadence comparison requires the same five seeds")
    result = copy.deepcopy(record)
    deltas = []
    for episode in result["episodes"]:
        delta = episode["return"] - baseline[episode["seed"]]
        episode["cadence_return_delta"] = delta
        deltas.append(delta)
    result["metrics"].update({"eval/cadence_delta_mean": statistics.mean(deltas),
        "eval/cadence_delta_sample_std": statistics.stdev(deltas),
        "eval/cadence_win_fraction": sum(delta > 0 for delta in deltas) / 5})
    result["provenance"]["original_sac_reference"] = {"record_id": original_sac["record_id"],
        "identity": original_sac["identity"], "checkpoint": original_sac["checkpoint"],
        "episodes": [{"seed": seed, "return": baseline[seed]} for seed in sorted(baseline)]}
    result["record_id"] = series._digest({"paired_record": record["record_id"], "original_sac": original_sac["record_id"]})
    result["artifact_files"].update({"original-sac/" + name: source for name, source in original_sac["artifact_files"].items()})
    return result


def completed_record(entry, index, output, reference_root, pins):
    output = Path(output)
    if not (output / "receipt.json").exists():
        return None
    receipt = series._read(output / "receipt.json")
    expected = {"validated": True, "mode": "run", "campaign_id": ATTEMPT, "source_run": entry["source_run"],
        "step": entry["step"], "J": 8, "source_K": entry["K"], "eval_K": TARGETS[index],
        "selector": f"cadence/k{TARGETS[index]}",
        "checkpoint_sha256": entry["sha256"], "metadata_sha256": entry["metadata_sha256"]}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Cadence receipt disagrees with authorized checkpoint/controller")
    reference = receipt.get("reference", {})
    if any(reference.get(field) != pins[name] for name, field in (
            ("receipt.json", "receipt_sha256"), ("results.json", "results_sha256"),
            ("bundle/manifest.json", "manifest_sha256"))):
        raise ValueError("Cadence receipt used a different original reference")
    files = {"results_sha256": output / "results.json", "manifest_sha256": output / "bundle/manifest.json"}
    if any(series._file_digest(path) != receipt.get(key) for key, path in files.items()):
        raise ValueError("Cadence result changed after validation")
    manifest = series._read(files["manifest_sha256"])
    if manifest.get("status") != "complete" or len(manifest.get("runs", [])) != 1 or manifest["runs"][0].get("status") != "complete":
        raise ValueError("Expected one complete cadence arm")
    from utils.eval_series_data import load_records
    records = load_records(files["manifest_sha256"], source_run=entry["source_run"], inventory_path=output / "checkpoint-inventory.json")
    if len(records) != 1:
        raise ValueError("Cadence bundle must normalize to exactly one arm")
    record = records[0]
    validate_spec(entry, index, record)
    if record["checkpoint"] != {"step": entry["step"], "sha256": entry["sha256"]}:
        raise ValueError("Cadence result checkpoint changed")
    if sorted(ep["seed"] for ep in record["episodes"]) != prior.SEEDS or any(
            ep["length"] != 500 or not ep.get("truncated") or ep.get("terminated") or ep.get("truncated_by_evaluator")
            for ep in record["episodes"]):
        raise ValueError("Cadence result lacks five complete environment-limited episodes")
    trace_names = set(manifest["runs"][0].get("trace_files", []))
    record["provenance"]["remote_trace_files"] = [{"path": path, "sha256": series._file_digest(path)}
        for name, path in record["artifact_files"].items() if name in trace_names]
    record["artifact_files"] = {name: path for name, path in record["artifact_files"].items() if name not in trace_names}
    record["artifact_files"].update({"results.json": str(output / "results.json"), "receipt.json": str(output / "receipt.json")})
    original, _ = reference_records(entry, index, reference_root, pins)
    return add_cadence_delta(prior.paired_record(record, original["prior"]), original["sac"])


def layout_curves(receipt):
    return {**receipt["context_curves"], **receipt["curves"]}


def install_workspace(receipt, path):
    """A separate six-run view; original run metadata and workspace stay intact."""
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr
    path = Path(path)
    previous = series._read(path) if path.exists() else {}
    if previous and previous.get("campaign_id") != ATTEMPT:
        raise ValueError("Workspace receipt belongs to another campaign")
    rows = list(layout_curves(receipt).values())
    def bar(title, metric):
        return wr.BarPlot(title=title, metrics=[metric], orientation="h", aggregate=False,
            max_runs_to_show=6, max_bars_to_show=6, legend_template="${run:displayName}",
            line_colors={row["run_id"] + ":" + metric: row["color"] for row in rows},
            line_titles={row["run_id"] + ":" + metric: row["name"] for row in rows}, layout=wr.Layout(w=12, h=7))
    markdown = ("**Frozen-checkpoint cadence crossover.** Both 500k checkpoints use J8 and H3. "
        "Train K identifies the data-collection cadence used to learn the checkpoint; eval K is the real-decision "
        "interval between fresh inner solves. Each checkpoint is now evaluated at K1 and K3. "
        "The four original prior/SAC runs are reused without reevaluation or modification. "
        "All arms use mean actions, seeds 101–105, controller seed55 and 500-decision full episodes. "
        "Paired prior gain subtracts the same checkpoint's prior; cadence delta subtracts its original SAC setting "
        "seed by seed. A positive cadence delta favors the swapped setting. "
        "Pending bars are absent, not zero returns. SD describes episode variability, not training-seed uncertainty.")
    sections = [ws.Section(name=PREFIX + "Protocol and progress", is_open=True, panels=[
        wr.MarkdownPanel(markdown=markdown, layout=wr.Layout(w=24, h=8)),
        bar("Completed checkpoints per run (of one)", "progress/checkpoints_complete"),
        bar("Failed validation cells", "progress/failed_cells")]),
        ws.Section(name=PREFIX + "Returns, paired gains and cost", is_open=True,
            panels=[bar(title, metric) for title, metric in METRICS],
            layout_settings=ws.SectionLayoutSettings(columns=2, rows=3))]
    if previous.get("url"):
        work = ws.Workspace.from_url(previous["url"])
        if work.entity != prior.ENTITY or work.project != prior.PROJECT:
            raise ValueError("Cadence workspace has the wrong target")
        replacements = {section.name: section for section in sections}
        work.sections = [replacements.pop(section.name, section) for section in work.sections]
        work.sections.extend(replacements.values())
    else:
        work = ws.Workspace(entity=prior.ENTITY, project=prior.PROJECT, name="Frozen J8 checkpoints: train cadence vs evaluation cadence",
            sections=sections, auto_generate_panels=False,
            settings=ws.WorkspaceSettings(max_runs=6, remove_legends_from_panels=False),
            runset_settings=ws.RunsetSettings(filters=[expr.Metric("ID").isin([row["run_id"] for row in rows])],
                groupby=[], run_settings={row["run_id"]: ws.RunSettings(color=row["color"], disabled=False) for row in rows},
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
        raise ValueError("Cadence workspace failed native API read-back")
    saved["api_verified"] = True
    series._atomic_json(path, saved)
    return saved


def verify_workspace(receipt, path, wandb_module=None):
    path = Path(path)
    if not path.is_file():
        raise ValueError("Install cadence workspace panels before publication")
    saved = series._read(path)
    wanted = {row["run_id"] for row in layout_curves(receipt).values()}
    if saved.get("campaign_id") != ATTEMPT or saved.get("api_verified") is not True or set(saved.get("run_ids", [])) != wanted:
        raise ValueError("Cadence workspace receipt differs from the six selected runs")
    parsed = urlparse(saved["url"])
    view = parse_qs(parsed.query).get("nw", [None])[0]
    if parsed.hostname != "wandb.ai" or parsed.path.rstrip("/") != f"/{prior.ENTITY}/{prior.PROJECT}" or not view:
        raise ValueError("A dedicated cadence workspace URL is required")
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
    wire = json.dumps(edges)
    if len(edges) != 1 or any(token not in wire for token in [*wanted, "progress/checkpoints_complete", "progress/failed_cells", *[metric for _, metric in METRICS]]):
        raise ValueError("Cadence workspace lost expected run IDs or panels")
    return saved


def publish_once(inventory, root, reference_root, *, wandb_module=None, workspace_verifier=verify_workspace):
    root, reference_root = Path(root).resolve(), Path(reference_root).resolve()
    publication = root / "publication"
    with series._lock(publication / ".publisher.lock", blocking=False):
        receipt = series._read(publication / "publication-map.json")
        if receipt["inventory_sha256"] != series._digest(inventory) or receipt["reference_root"] != str(reference_root):
            raise ValueError("Cadence inventory/reference identity changed")
        workspace = workspace_verifier(receipt, publication / "workspace.json", wandb_module)
        failures = {}
        for index, entry in enumerate(inventory["entries"][:2]):
            try:
                record = completed_record(entry, index, root / "production" / f"cell-{index}", reference_root, receipt["references"][str(index)])
                if record is not None:
                    series.stage_record(receipt["curves"][f"cell-{index}_swap"]["run_dir"], record)
            except Exception as exc:
                failures[str(index)] = str(exc)
        for row in prior.progress_rows(receipt):
            with series.Publisher(row["run_dir"], owner=prior.OWNER, wandb_module=wandb_module) as publisher:
                publisher.run.name = row["name"]
                publisher.run.config.update({"campaign_id": ATTEMPT, "source_K": row["source_K"], "evaluation_K": row["K"], "J": 8})
                for _, metric in METRICS:
                    publisher.run.define_metric(metric, step_metric=series.X_AXIS, step_sync=False, hidden=False)
                publisher.publish_pending()
                publisher.run.summary.update({"progress/checkpoints_complete": row["complete"], "progress/checkpoints_expected": 1,
                    "progress/failed_cells": int(str(row["index"]) in failures),
                    "progress/status": "failed validation" if str(row["index"]) in failures else ("complete" if row["complete"] else "pending results")})
        result = {"campaign_id": ATTEMPT, "updated_at": prior.now(), "runs": prior.progress_rows(receipt),
            "failures": failures, "workspace_url": workspace["url"]}
        series._atomic_json(publication / "status.json", result)
        if failures:
            raise ValueError("Cadence validation failed: " + json.dumps(failures))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "layout", "publish", "watch"))
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--compute-jobs")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    try:
        inventory = prior.load_inventory(args.inventory)
        if args.command == "prepare":
            result = prepare(inventory, args.root, args.reference_root, args.prepared_root)
        elif args.command == "layout":
            receipt = series._read(args.root / "publication/publication-map.json")
            if receipt["inventory_sha256"] != series._digest(inventory):
                raise ValueError("Cadence inventory changed")
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
                        raise ValueError("Compute ended before both cadence results were published")
                    break
                time.sleep(args.poll_seconds)
        print(json.dumps(result, allow_nan=False))
    except Exception as exc:
        series._atomic_json(args.root / "publication/failure.json", {"at": prior.now(), "command": args.command, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
