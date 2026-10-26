"""Publish the three selected return-critic checkpoints and their matched priors.

GPU workers remain offline. One CPU process owns all six durable run registries.
The ``layout`` command uses the separate wandb-workspaces runtime; ``publish``
uses the locked evaluator runtime with WANDB__REQUIRE_CORE=true. Layout and
publication failures are written to publication/failure.json and raised.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
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
from utils import eval_series as series

ENTITY = "rwgao_b-brown-university"
PROJECT = "ambi-inner-bench"
ATTEMPT = "return-prior-selected-20261026"
OWNER = "oscar-rgao48"
ARMS = ("prior", "sac")
SEEDS = [101, 102, 103, 104, 105]
PREFIX = "Selected return priors / "
METRICS = {
    "Return and paired gain": [
        ("Episode return: mean of five seeds", "eval/return_mean"),
        ("Inner SAC improvement over matching prior", "eval/paired_gain_mean"),
        ("Episode return: sample standard deviation", "eval/return_sample_std"),
        ("Paired gain: sample standard deviation", "eval/paired_gain_sample_std"),
    ],
    "Runtime and pairing": [
        ("Controller seconds per decision", "runtime/control_seconds_per_decision"),
        ("Fraction of seeds improved", "eval/paired_win_fraction"),
        ("Completed episodes", "eval/episodes"),
        ("Matched prior episodes", "eval/paired_episodes"),
    ],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def load_inventory(path):
    data = series._read(path)
    entries = data.get("entries", [])
    if data.get("campaign_id") != ATTEMPT or len(entries) != 3:
        raise ValueError("Expected the explicitly selected three-checkpoint campaign")
    if len({entry["source_run"] for entry in entries}) != 3:
        raise ValueError("Each selected checkpoint must identify a distinct source run")
    for entry in entries:
        if not re.fullmatch(r"[\w.-]+/[\w.-]+/[\w.-]+", entry["source_run"]):
            raise ValueError("Checkpoint source must be an entity/project/run identity")
        for key in ("sha256", "metadata_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", entry[key]):
                raise ValueError("Checkpoint and metadata require SHA256 pins")
        if any(type(entry[key]) is not int or entry[key] <= 0 for key in ("step", "J", "K")):
            raise ValueError("Checkpoint step and original J/K must be positive integers")
    return data


def label(entry):
    step = str(entry["step"] // 1000) + "k" if entry["step"] % 1000 == 0 else str(entry["step"])
    return f"J{entry['J']} K{entry['K']} @ {step}"


def validate_spec(entry, arm, spec):
    identity = spec["identity"]
    if spec["selector"] != "comparison/" + arm or identity["backbone"] != entry["source_run"]:
        raise ValueError("Specification source or selector differs from the selected checkpoint")
    planner, protocol = identity["planner"], identity["protocol"]
    if planner.get("type") != ("prior" if arm == "prior" else "sac"):
        raise ValueError("Specification planner differs from the requested arm")
    if (planner.get("action_rule") != "tanh_mean" or protocol.get("action_rule") != "tanh_mean"
            or protocol.get("environment_seeds") != SEEDS
            or protocol.get("controller_seed") != 55 or protocol.get("max_steps") != 500):
        raise ValueError("Specification must use five paired full episodes and mean actions")
    if arm == "sac":
        settings = planner.get("settings", {})
        if (settings.get("inner_rounds") != entry["J"] or settings.get("inner_rollout_horizon") != 3
                or settings.get("inner_solve_interval", 1) != entry["K"]):
            raise ValueError("Specification changed the selected run's original J/K")


def prepare(inventory, root, prepared_root=None):
    """Allocate New attempts once, recovering an interrupted local allocation."""
    root = Path(root).resolve()
    prepared_root = Path(prepared_root).resolve() if prepared_root else root / "prepared"
    publication = root / "publication"
    publication.mkdir(parents=True, exist_ok=True)
    path = publication / "publication-map.json"
    with series._lock(publication / ".prepare.lock"):
        receipt = series._read(path) if path.exists() else {
            "schema_version": 1, "campaign_id": ATTEMPT, "inventory_sha256": series._digest(inventory),
            "owner": OWNER, "attempt_label": ATTEMPT, "curves": {}, "cell_run_maps": {}}
        if receipt["inventory_sha256"] != series._digest(inventory):
            raise ValueError("Selected inventory changed after publication preparation")
        for index, entry in enumerate(inventory["entries"]):
            run_map = {}
            for arm in ARMS:
                key = f"cell-{index}_{arm}"
                spec = series._read(prepared_root / f"cell-{index}" / "specs" / f"comparison__{arm}.json")
                validate_spec(entry, arm, spec)
                owned = list((publication / "runs" / key).glob("*/run.json"))
                if key in receipt["curves"]:
                    registry = series.validate_identity(receipt["curves"][key]["run_dir"], spec["identity"])
                elif owned:
                    if len(owned) != 1:
                        raise ValueError("Ambiguous interrupted run allocation")
                    registry = series.validate_identity(owned[0].parent, spec["identity"])
                else:
                    registry = series.create_run(publication / "runs" / key, spec, ATTEMPT, PROJECT, ENTITY, OWNER)
                if registry["owner"] != OWNER or registry["attempt_label"] != ATTEMPT:
                    raise ValueError("Registry owner or attempt differs")
                color = ("#64748B", "#2563EB") if index == 0 else (("#A8A29E", "#EA580C") if index == 1 else ("#94A3B8", "#7C3AED"))
                receipt["curves"][key] = {
                    "run_id": registry["run_id"], "run_dir": registry["run_dir"],
                    "name": label(entry) + (" | Prior only" if arm == "prior" else " | Inner SAC"),
                    "color": color[arm == "sac"], "index": index, "arm": arm,
                    "source_run": entry["source_run"], "step": entry["step"],
                }
                run_map["comparison/" + arm] = registry["run_dir"]
                series._atomic_json(path, receipt)
            map_path = publication / f"cell-{index}" / "eval-run-map.json"
            series._atomic_json(map_path, run_map)
            receipt["cell_run_maps"][str(index)] = str(map_path)
        series._atomic_json(path, receipt)
    return receipt


def paired_record(inner, prior):
    """Pair by environment seed while retaining controller-specific solver RNG."""
    if inner["checkpoint"] != prior["checkpoint"] or inner["identity"]["backbone"] != prior["identity"]["backbone"]:
        raise ValueError("Paired controls have different checkpoint/source identities")
    keys = ("environment", "env_wrapper", "env_wrappers", "observation", "action_rule", "max_steps", "environment_seeds", "controller_seed")
    if any(inner["identity"]["protocol"].get(key) != prior["identity"]["protocol"].get(key) for key in keys):
        raise ValueError("Paired environment/action/seed protocol differs")
    if prior["identity"]["planner"]["type"] != "prior":
        raise ValueError("Matched control is not an unadapted prior")
    baseline = {episode["seed"]: episode for episode in prior["episodes"]}
    if (len(baseline) != 5 or sorted(baseline) != SEEDS
            or sorted(ep["seed"] for ep in inner["episodes"]) != SEEDS):
        raise ValueError("Pairing requires five unique matching environment seeds")
    record = copy.deepcopy(inner)
    gains = []
    for episode in record["episodes"]:
        gain = episode["return"] - baseline[episode["seed"]]["return"]
        episode["paired_return_delta"] = gain
        gains.append(gain)
    record["metrics"].update({"eval/paired_gain_mean": statistics.mean(gains),
        "eval/paired_gain_sample_std": statistics.stdev(gains), "eval/paired_episodes": 5,
        "eval/paired_win_fraction": sum(gain > 0 for gain in gains) / 5,
        "eval/base_return_mean": prior["metrics"]["eval/return_mean"]})
    record["provenance"]["paired_baseline"] = {"record_id": prior["record_id"],
        "identity": prior["identity"], "checkpoint": prior["checkpoint"],
        "episodes": [{key: ep[key] for key in ("seed", "solver_seed", "return", "length") if key in ep}
                     for ep in prior["episodes"]]}
    record["record_id"] = series._digest({"result": inner["record_id"], "paired_baseline": prior["record_id"]})
    record["artifact_files"].update({"paired-prior/" + key: value for key, value in prior["artifact_files"].items()})
    return record


def completed_records(entry, output):
    """Only accepted full production receipts may become published points."""
    output = Path(output)
    receipt_path = output / "receipt.json"
    if not receipt_path.exists():
        return None
    receipt = series._read(receipt_path)
    if receipt.get("validated") is not True or receipt.get("mode") != "run":
        raise ValueError("Production receipt failed validation or is a smoke result")
    for key, expected in {"source_run": entry["source_run"], "step": entry["step"], "J": entry["J"], "K": entry["K"],
                          "checkpoint_sha256": entry["sha256"], "metadata_sha256": entry["metadata_sha256"]}.items():
        if receipt.get(key) != expected:
            raise ValueError("Receipt differs from selected checkpoint: " + key)
    paths = {"results_sha256": output / "results.json", "manifest_sha256": output / "bundle" / "manifest.json"}
    if any(series._file_digest(path) != receipt.get(key) for key, path in paths.items()):
        raise ValueError("Validated result files changed after receipt creation")
    manifest = series._read(paths["manifest_sha256"])
    if manifest.get("status") != "complete" or any(run.get("status") != "complete" for run in manifest.get("runs", [])):
        raise ValueError("Receipt refers to an incomplete bundle")
    from utils.eval_series_data import load_records
    records = load_records(paths["manifest_sha256"], source_run=entry["source_run"],
                           inventory_path=output / "checkpoint-inventory.json")
    if len(records) != 2 or {record["selector"] for record in records} != {"comparison/prior", "comparison/sac"}:
        raise ValueError("Bundle must contain exactly the two authorized controllers")
    by_arm = {record["selector"].split("/")[1]: record for record in records}
    for arm, record in by_arm.items():
        validate_spec(entry, arm, record)
        if record["checkpoint"] != {"step": entry["step"], "sha256": entry["sha256"]}:
            raise ValueError("Normalized result has a different checkpoint")
        if sorted(ep["seed"] for ep in record["episodes"]) != SEEDS:
            raise ValueError("Result lacks the five complete seeds")
        if any(ep["length"] != 500 or not ep.get("truncated") or ep.get("truncated_by_evaluator") for ep in record["episodes"]):
            raise ValueError("Result contains a shortened episode")
        # Full traces stay on scratch. Hash their immutable locations for audit,
        # while uploading the complete summary, receipt and manifest artifacts.
        trace_names = {name for run in manifest["runs"] for name in run.get("trace_files", [])}
        traces = {name: source for name, source in record["artifact_files"].items() if name in trace_names}
        record["provenance"]["remote_trace_files"] = [{"path": source, "sha256": series._file_digest(source)} for source in traces.values()]
        record["artifact_files"] = {name: source for name, source in record["artifact_files"].items() if name not in traces}
        record["artifact_files"].update({"results.json": str(paths["results_sha256"]), "receipt.json": str(receipt_path)})
    by_arm["sac"] = paired_record(by_arm["sac"], by_arm["prior"])
    return by_arm


def progress_rows(receipt):
    rows = []
    for key, curve in receipt["curves"].items():
        entries = list(series._read(Path(curve["run_dir"]) / "publication.json")["records"].values())
        rows.append({**curve, "curve": key, "complete": len(entries),
                     "published": sum(entry["status"] == "published" for entry in entries), "expected": 1})
    return rows


def install_workspace(inventory, receipt, path):
    """Create a dedicated view; later calls preserve other sections and filters."""
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr
    path = Path(path)
    previous = series._read(path) if path.exists() else {}
    if previous and previous.get("campaign_id") != ATTEMPT:
        raise ValueError("Workspace receipt belongs to another campaign")
    rows = list(receipt["curves"].values())
    def bar(title, metric):
        return wr.BarPlot(title=title, metrics=[metric], orientation="h", aggregate=False,
            max_runs_to_show=6, max_bars_to_show=6, legend_template="${run:displayName}",
            line_colors={row["run_id"] + ":" + metric: row["color"] for row in rows},
            line_titles={row["run_id"] + ":" + metric: row["name"] for row in rows},
            layout=wr.Layout(w=12, h=7))
    markdown = ("**Selected return-only critic checkpoints: policy prior versus inner SAC.**\n\n"
        + "; ".join(label(entry) for entry in inventory["entries"]) + ". "
        "Each pair uses the same frozen checkpoint, horizon H=3, original J inner rounds and K real-decision solve interval, mean actions, environment seeds 101–105, "
        "controller seed 55, and 500 decisions per episode. No outer learning occurs. "
        "Gain is the per-seed SAC return minus its matching prior return. SD describes episode variability, "
        "not a confidence interval or variation across training seeds. "
        "Empty bars are pending measurements, never zero returns. One completed checkpoint is expected per run.")
    sections = [ws.Section(name=PREFIX + "Protocol and progress", is_open=True, panels=[
        wr.MarkdownPanel(markdown=markdown, layout=wr.Layout(w=24, h=7)),
        bar("Completed checkpoints per run (of one)", "progress/checkpoints_complete"),
        bar("Failed validation cells", "progress/failed_cells")])]
    sections += [ws.Section(name=PREFIX + title, is_open=True,
        panels=[bar(name, metric) for name, metric in metrics],
        layout_settings=ws.SectionLayoutSettings(columns=2, rows=2)) for title, metrics in METRICS.items()]
    if previous.get("url"):
        work = ws.Workspace.from_url(previous["url"])
        if work.entity != ENTITY or work.project != PROJECT:
            raise ValueError("Workspace target differs from campaign")
        replacements = {section.name: section for section in sections}
        work.sections = [replacements.pop(section.name, section) for section in work.sections]
        work.sections.extend(replacements.values())
        for row in rows:
            setting = work.runset_settings.run_settings.get(row["run_id"], ws.RunSettings())
            setting.color = row["color"]
            work.runset_settings.run_settings[row["run_id"]] = setting
    else:
        work = ws.Workspace(entity=ENTITY, project=PROJECT,
            name="Selected successes: prior vs original inner SAC", sections=sections,
            auto_generate_panels=False,
            settings=ws.WorkspaceSettings(max_runs=6, remove_legends_from_panels=False),
            runset_settings=ws.RunsetSettings(filters=[expr.Metric("ID").isin([row["run_id"] for row in rows])],
                groupby=[], run_settings={row["run_id"]: ws.RunSettings(color=row["color"], disabled=False) for row in rows},
                pinned_columns=["run:displayName", "summary:progress/checkpoints_complete", "summary:eval/return_mean"]))
    work.save()
    saved = {**previous, "campaign_id": ATTEMPT, "url": work.url, "updated_at": now(),
             "api_verified": False, "run_ids": [row["run_id"] for row in rows], "panel_count": 11}
    saved.setdefault("browser_verified", False)
    series._atomic_json(path, saved)
    readback = ws.Workspace.from_url(work.url)
    panels = [panel for section in readback.sections if section.name in {section.name for section in sections} for panel in section.panels]
    if len(panels) != 11 or any(panel.aggregate for panel in panels if isinstance(panel, wr.BarPlot)):
        raise ValueError("Native workspace panels failed read-back")
    if any(readback.runset_settings.run_settings[row["run_id"]].color != row["color"] for row in rows):
        raise ValueError("Native run colors failed read-back")
    saved["api_verified"] = True
    series._atomic_json(path, saved)
    return saved


def verify_workspace(receipt, path, wandb_module=None):
    path = Path(path)
    if not path.is_file():
        raise ValueError("Install native workspace panels before publication")
    saved = series._read(path)
    wanted = {curve["run_id"] for curve in receipt["curves"].values()}
    if saved.get("campaign_id") != ATTEMPT or saved.get("api_verified") is not True or set(saved.get("run_ids", [])) != wanted:
        raise ValueError("Workspace receipt does not verify this campaign's six runs")
    parsed = urlparse(saved["url"])
    if parsed.hostname != "wandb.ai" or parsed.path.rstrip("/") != f"/{ENTITY}/{PROJECT}":
        raise ValueError("Workspace URL differs from campaign project")
    view = parse_qs(parsed.query).get("nw", [None])[0]
    if not view:
        raise ValueError("A dedicated named workspace URL is required")
    if wandb_module is None:
        import wandb as wandb_module
    try:
        gql = importlib.import_module("wandb_gql").gql
    except ImportError:
        gql = importlib.import_module("wandb.vendor.gql").gql
    query = gql('query View($entityName:String,$name:String,$viewName:String) { project(name:$name,entityName:$entityName) { allViews(viewType:"project-view",viewName:$viewName) { edges { node { id spec } } } } }')
    response = wandb_module.Api(timeout=30).client.execute(query, variable_values={
        "entityName": ENTITY, "name": PROJECT, "viewName": "nw-" + view + "-v"})
    edges = (response.get("project") or {}).get("allViews", {}).get("edges", [])
    if len(edges) != 1:
        raise ValueError("Dedicated workspace is missing")
    raw = edges[0]["node"]["spec"]
    wire = json.dumps(json.loads(raw) if isinstance(raw, str) else raw, sort_keys=True)
    required = [*wanted, "progress/checkpoints_complete", "progress/failed_cells",
                *[metric for metrics in METRICS.values() for _, metric in metrics]]
    if any(token not in wire for token in required):
        raise ValueError("Dedicated workspace lost expected runs or panels")
    return saved


def publish_once(inventory, root, *, wandb_module=None, workspace_verifier=verify_workspace):
    root = Path(root).resolve()
    publication = root / "publication"
    with series._lock(publication / ".publisher.lock", blocking=False):
        receipt = series._read(publication / "publication-map.json")
        if receipt["inventory_sha256"] != series._digest(inventory):
            raise ValueError("Inventory changed after preparation")
        workspace = workspace_verifier(receipt, publication / "workspace.json", wandb_module)
        failures = {}
        for index, entry in enumerate(inventory["entries"]):
            try:
                records = completed_records(entry, root / "production" / f"cell-{index}")
                if records:
                    for arm in ARMS:
                        series.stage_record(receipt["curves"][f"cell-{index}_{arm}"]["run_dir"], records[arm])
            except Exception as exc:
                failures[str(index)] = str(exc)
        for row in progress_rows(receipt):
            with series.Publisher(row["run_dir"], owner=OWNER, wandb_module=wandb_module) as publisher:
                publisher.run.name = row["name"]
                publisher.run.config.update({"campaign_id": ATTEMPT, "selected_checkpoint_step": row["step"],
                    "curve_role": row["arm"], "source_run": row["source_run"]})
                for metrics in METRICS.values():
                    for _, metric in metrics:
                        publisher.run.define_metric(metric, step_metric=series.X_AXIS, step_sync=False, hidden=False)
                publisher.publish_pending()
                publisher.run.summary.update({"progress/checkpoints_complete": row["complete"],
                    "progress/checkpoints_expected": 1, "progress/failed_cells": int(str(row["index"]) in failures),
                    "progress/status": "failed validation" if str(row["index"]) in failures else ("complete" if row["complete"] else "pending results")})
        result = {"campaign_id": ATTEMPT, "updated_at": now(), "runs": progress_rows(receipt),
                  "failures": failures, "workspace_url": workspace["url"]}
        series._atomic_json(publication / "status.json", result)
        if failures:
            raise ValueError("Evaluation validation failed: " + json.dumps(failures, sort_keys=True))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "layout", "publish", "watch"))
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path, help="Campaign root containing prepared, production and publication")
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--compute-jobs", help="Comma-separated Slurm job IDs; required for watch")
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    try:
        inventory = load_inventory(args.inventory)
        publication = args.root / "publication"
        if args.command == "prepare":
            result = prepare(inventory, args.root, args.prepared_root)
        elif args.command == "layout":
            receipt = series._read(publication / "publication-map.json")
            if receipt["inventory_sha256"] != series._digest(inventory):
                raise ValueError("Inventory changed after preparation")
            result = install_workspace(inventory, receipt, publication / "workspace.json")
        else:
            if args.command == "watch" and (not args.compute_jobs or not re.fullmatch(r"\d+(,\d+)*", args.compute_jobs)):
                raise ValueError("watch requires explicit numeric --compute-jobs")
            if not 1 <= args.poll_seconds <= 60:
                raise ValueError("Polling interval must be between 1 and 60 seconds")
            while True:
                result = publish_once(inventory, args.root)
                if args.command != "watch":
                    break
                active = subprocess.check_output(["squeue", "-h", "-j", args.compute_jobs], text=True).strip()
                if not active:
                    result = publish_once(inventory, args.root)
                    if any(row["published"] != 1 for row in result["runs"]):
                        raise RuntimeError("Compute ended with incomplete evaluation publication")
                    break
                time.sleep(args.poll_seconds)
        print(json.dumps(result, allow_nan=False))
    except Exception as exc:
        series._atomic_json(args.root / "publication" / "failure.json", {"at": now(), "command": args.command, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
