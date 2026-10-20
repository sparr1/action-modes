"""Validate, stage and publish prior-KL episodes into a dedicated native view.

Layout uses a separate wandb==0.28.1 / wandb-workspaces==0.4.5 runtime.
Publication uses the existing eval-series SDK adapter and oscar-rgao48 owner.
GPU workers do not stage or upload data. Existing entropy runs are read-only.
Use publish --once --require-complete only after all worker jobs have terminated;
missing validated results then become visible failures instead of pending cells.
"""
import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import eval_series as series

ENTITY, PROJECT, OWNER = "rwgao_b-brown-university", "ambi-inner-bench", "oscar-rgao48"
PREFIX = "Prior KL / "
EXISTING = [
    dict(run_id="9edcea495bf243a5ab9a6e139e603b19", name="MPPI-trained | entropy H3 J8 | sampled", color="#14B8A6"),
    dict(run_id="566a5d7c358b4cbfa056bc22ec715929", name="SAC-trained | entropy H3 J8 | sampled", color="#A855F7"),
]
CURVES = {"sac_aux_100k": ("SAC-trained | prior KL H3 J8 | sampled", "#E87924"),
          "mppi_aux_400k": ("MPPI-trained | prior KL H3 J8 | sampled", "#2563EB")}
METRICS = [
    ("Episode return (five-seed mean)", "eval/return_mean"),
    ("Return variability (sample SD)", "eval/return_sample_std"),
    ("Improvement over paired sampled prior", "eval/paired_gain_mean"),
    ("Sampled prior return at paired checkpoint", "reference/return_mean"),
    ("Controller seconds per decision", "runtime/control_seconds_per_decision"),
    ("Inner entropy (update-time batches)", "diagnostics/inner_actor_entropy_mean"),
    ("Forward KL to inherited prior", "diagnostics/inner_outer_policy_kl_mean"),
    ("Temperature optimizer steps", "work/temperature_updates"),
]


class PublicationError(ValueError):
    """Local, credential-free validation error safe to report to the user."""


def now():
    return datetime.now(timezone.utc).isoformat()


def load_campaign(path):
    campaign = series._read(path)
    if (campaign.get("campaign_id") != "prior-kl-episodes-20261020"
            or campaign.get("seeds") != [101, 102, 103, 104, 105]
            or campaign.get("controller_seed") != 55 or campaign.get("max_steps") != 500
            or campaign.get("execution_action") != "sample"):
        raise PublicationError("Expected the authorized prior-KL five-seed sampled episode campaign")
    cells = campaign.get("cells", [])
    if len(cells) != 2 or {c["checkpoint_id"] for c in cells} != set(CURVES):
        raise PublicationError("Expected SAC100k and MPPI400k exactly once")
    commit = campaign.get("source_commit", "")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise PublicationError("Campaign requires an exact committed source SHA")
    if len({c["run_id"] for c in cells}) != 2 or len({c["output_dir"] for c in cells}) != 2:
        raise PublicationError("Each checkpoint requires its own run and result directory")
    for cell in cells:
        expected_step = 100000 if cell["checkpoint_id"] == "sac_aux_100k" else 400000
        if cell["step"] != expected_step or cell["selector"] != "prior_kl/" + cell["checkpoint_id"]:
            raise PublicationError("Unexpected checkpoint step or prior-KL selector")
    return campaign


def curve_rows(campaign):
    return [dict(run_id=c["run_id"], run_dir=c["run_dir"], checkpoint_id=c["checkpoint_id"],
                 name=CURVES[c["checkpoint_id"]][0], color=CURVES[c["checkpoint_id"]][1])
            for c in campaign["cells"]]


def stage_completed(campaign):
    from utils.eval_series_data import load_records
    status = {}
    for cell in campaign["cells"]:
        key, output = cell["checkpoint_id"], Path(cell["output_dir"])
        receipt_path, manifest = output / "receipt.json", output / "bundle/manifest.json"
        if not receipt_path.exists():
            status[key] = "failed" if (output / "failure.json").exists() else "pending"
            continue
        receipt = series._read(receipt_path)
        if (receipt.get("validated") is not True or receipt.get("smoke") is not False
                or receipt.get("checkpoint_id") != key or receipt.get("source_commit") != campaign["source_commit"]):
            raise PublicationError("Only validated production receipts from the campaign source can be published")
        results = output / "results.json"
        if (receipt.get("manifest_sha256") != series._file_digest(manifest)
                or receipt.get("results_sha256") != series._file_digest(results)):
            raise PublicationError("Validated result or manifest changed")
        registry = series.load_run(cell["run_dir"])
        if (registry["run_id"] != cell["run_id"] or registry["owner"] != OWNER
                or registry["entity"] != ENTITY or registry["project"] != PROJECT):
            raise PublicationError("Allocated publication identity or owner changed")
        journal = series._read(Path(cell["run_dir"]) / "publication.json")
        old = [r for r in journal["records"].values() if r["checkpoint_step"] == cell["step"]]
        if old:
            files = old[0]["artifact_sha256"]
            if (len(old) != 1 or old[0]["checkpoint_sha256"] != cell["checkpoint_sha256"]
                    or files.get("manifest.json") != receipt["manifest_sha256"]
                    or files.get("results.json") != receipt["results_sha256"]
                    or files.get("validation-receipt.json") != series._file_digest(receipt_path)):
                raise PublicationError("Previously accepted result changed")
            status[key] = "complete"
            continue
        for kind in ("reference", "baseline"):
            if series._file_digest(Path(cell[kind + "_bundle"]) / "manifest.json") != cell[kind + "_manifest_sha256"]:
                raise PublicationError("Pinned prior or entropy reference changed")
        selected = [r for r in series._read(manifest)["runs"] if r["selector"] == cell["selector"]]
        if len(selected) != 1 or selected[0].get("status") != "complete":
            raise PublicationError("Validated bundle lacks its complete selected run")
        records = [r for r in load_records(manifest, source_run=cell["source_run"], inventory_path=cell["inventory_path"])
                   if r["selector"] == cell["selector"]]
        if len(records) != 1:
            raise PublicationError("Bundle must normalize to one exact planner record")
        record = records[0]
        if record["checkpoint"] != {"step": cell["step"], "sha256": cell["checkpoint_sha256"]}:
            raise PublicationError("Record checkpoint differs from the campaign")
        if sorted(e["seed"] for e in record["episodes"]) != campaign["seeds"]:
            raise PublicationError("Record must contain all five paired seeds")
        if any(e.get("paired_return_delta") is None for e in record["episodes"]):
            raise PublicationError("A validated sampled-prior reference is required")
        priors = [e["return"] - e["paired_return_delta"] for e in record["episodes"]]
        record["metrics"]["reference/return_mean"] = statistics.mean(priors)
        record["metrics"]["reference/return_sample_std"] = statistics.stdev(priors)
        trace_names, retained, remote = set(selected[0].get("trace_files", [])), {}, []
        if not trace_names.issubset(record["artifact_files"]):
            raise PublicationError("Full trace files are missing; cannot retain their hashed remote index")
        for name, path in record["artifact_files"].items():
            if name in trace_names:
                remote.append(dict(name=name, path=path, sha256=series._file_digest(path), size_bytes=Path(path).stat().st_size))
            else:
                retained[name] = path
        trace_index = Path(cell["run_dir"]) / f"remote-traces-{cell['step']}.json"
        series._atomic_json(trace_index, dict(policy="full traces retained on Oscar", files=remote))
        retained.update({"results.json": str(results), "validation-receipt.json": str(receipt_path), "remote-traces.json": str(trace_index)})
        if sum(Path(p).stat().st_size for p in retained.values()) > 16 * 1024 * 1024:
            raise PublicationError("Summary artifact exceeds 16 MiB; refusing a large upload")
        record["artifact_files"] = retained
        record["provenance"].update(artifact_policy="summary-only; full traces retained on Oscar", remote_trace_files=remote)
        series.stage_record(cell["run_dir"], record)
        status[key] = "complete"
    return status


def progress_rows(campaign, status):
    rows = curve_rows(campaign)
    for row in rows:
        journal = series._read(Path(row["run_dir"]) / "publication.json")
        published = sum(r["status"] == "published" for r in journal["records"].values())
        complete = int(status.get(row["checkpoint_id"]) == "complete")
        failed = int(status.get(row["checkpoint_id"]) == "failed")
        row.update(complete=complete, published=published, expected=1, failed=failed,
                   status="complete" if published == 1 else "failed" if failed else "pending results")
    return rows


def install_workspace(campaign, path):
    for package, expected in (("wandb", "0.28.1"), ("wandb-workspaces", "0.4.5")):
        if importlib.metadata.version(package) != expected:
            raise PublicationError(f"layout requires separate runtime {package}=={expected}")
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr
    rows, path = EXISTING + curve_rows(campaign), Path(path)
    previous = series._read(path) if path.exists() else {}
    if previous and previous.get("campaign_sha256") != series._digest(campaign):
        raise PublicationError("Workspace receipt belongs to a different campaign identity")
    markdown = """**Prior-KL versus entropy, sampled-action frozen H3/J8 evaluation.**

New paired checkpoints: SAC-trained 100k and MPPI-trained 400k. Each new KL point
uses seeds 101–105, controller seed 55 and 500 decisions. The KL coefficient is
the inherited outer alpha held fixed; entropy and temperature learning are off.
Existing entropy runs retain other checkpoint points as context, not additional
paired KL comparisons. No outer learning occurs. X is source-training decisions.

Pending results have no return point and are never represented as zero. Progress
scalars show one expected checkpoint per new KL run. Native curves have distinct
colors and hover highlighting, with no aggregation. SD describes environment-seed
variability. Paired gain and reference return use the exact sampled prior at the
same checkpoint; those metrics are present on the new KL runs. Prior values are
reference measurements, not another KL return. Inner entropy is an update-batch
statistic, not final-root entropy. Timing comes from actual episode control.
"""
    def plot(title, metric):
        return wr.LinePlot(title=title, x=series.X_AXIS, y=[metric], aggregate=False,
            smoothing_type="none", legend_template="${run:name}", max_runs_to_show=len(rows),
            title_x="Source training decisions", layout=wr.Layout(w=12, h=8),
            line_colors={r["run_id"] + ":" + metric: r["color"] for r in rows},
            line_marks={r["run_id"] + ":" + metric: "points" for r in rows if "checkpoint_id" in r},
            line_titles={r["run_id"] + ":" + metric: r["name"] for r in rows})
    sections = [ws.Section(name=PREFIX + "Protocol and progress", is_open=True, panels=[
        wr.MarkdownPanel(markdown=markdown, layout=wr.Layout(w=24, h=10)),
        wr.ScalarChart(title="Completed new KL checkpoints (one per run)", metric="progress/checkpoints_complete", layout=wr.Layout(w=12, h=6)),
        wr.ScalarChart(title="Failed new KL cells", metric="progress/failed_cells", layout=wr.Layout(w=12, h=6))]),
        ws.Section(name=PREFIX + "Return and improvement", is_open=True, panels=[plot(*m) for m in METRICS[:4]]),
        ws.Section(name=PREFIX + "Runtime and policy diagnostics", is_open=True, panels=[plot(*m) for m in METRICS[4:]])]
    if previous.get("url"):
        work = ws.Workspace.from_url(previous["url"])
        if work.entity != ENTITY or work.project != PROJECT:
            raise PublicationError("Workspace target mismatch")
        replacement = {s.name: s for s in sections}
        work.sections = [replacement.pop(s.name, s) for s in work.sections] + list(replacement.values())
        for row in rows:
            work.runset_settings.run_settings[row["run_id"]] = ws.RunSettings(color=row["color"], disabled=False)
    else:
        work = ws.Workspace(entity=ENTITY, project=PROJECT, name="Prior KL: sampled H3 J8, SAC100k and MPPI400k",
            sections=sections, auto_generate_panels=False,
            settings=ws.WorkspaceSettings(x_axis=series.X_AXIS, max_runs=len(rows), tooltip_number_of_runs="all_runs", tooltip_color_run_names=True),
            runset_settings=ws.RunsetSettings(filters=[expr.Metric("ID").isin([r["run_id"] for r in rows])], groupby=[],
                run_settings={r["run_id"]: ws.RunSettings(color=r["color"], disabled=False) for r in rows}))
    work.save()
    saved = dict(campaign_id=campaign["campaign_id"], campaign_sha256=series._digest(campaign), url=work.url,
                 updated_at=now(), api_verified=False, runs=rows)
    series._atomic_json(path, saved)
    check = ws.Workspace.from_url(work.url)
    panels = [p for s in check.sections if s.name.startswith(PREFIX) for p in s.panels]
    if len(panels) != 11 or any(check.runset_settings.run_settings[r["run_id"]].color != r["color"] for r in rows):
        raise PublicationError("Native workspace panel/color read-back failed")
    if any(p.aggregate or getattr(p.x, "name", p.x) != series.X_AXIS for p in panels if isinstance(p, wr.LinePlot)):
        raise PublicationError("Native workspace lost independent checkpoint-axis curves")
    saved.update(api_verified=True, panel_count=len(panels))
    series._atomic_json(path, saved)
    return saved


def verify_workspace(campaign, path, wandb_module=None):
    path = Path(path)
    if not path.exists():
        raise PublicationError("Install and verify the dedicated native layout before publication")
    saved = series._read(path)
    wanted = EXISTING + curve_rows(campaign)
    if (saved.get("campaign_sha256") != series._digest(campaign) or not saved.get("api_verified")
            or {r["run_id"] for r in saved.get("runs", [])} != {r["run_id"] for r in wanted}):
        raise PublicationError("Native workspace receipt does not match the campaign")
    parsed = urlparse(saved["url"])
    if parsed.hostname != "wandb.ai" or parsed.path.rstrip("/") != f"/{ENTITY}/{PROJECT}":
        raise PublicationError("Dedicated view must be in the intended W&B project")
    name = parse_qs(parsed.query).get("nw", [None])[0]
    if not name:
        raise PublicationError("Dedicated native view ID is missing")
    if wandb_module is None:
        import wandb as wandb_module
    try:
        gql = importlib.import_module("wandb_gql").gql
    except ImportError:
        gql = importlib.import_module("wandb.vendor.gql").gql
    query = gql('query View($entityName:String,$name:String,$viewName:String){project(name:$name,entityName:$entityName){allViews(viewType:"project-view",viewName:$viewName){edges{node{id spec}}}}}')
    response = wandb_module.Api(timeout=30).client.execute(query, variable_values={"entityName": ENTITY, "name": PROJECT, "viewName": "nw-" + name + "-v"})
    edges = (response.get("project") or {}).get("allViews", {}).get("edges", [])
    if len(edges) != 1:
        raise PublicationError("Verified native workspace is missing remotely")
    wire = json.dumps(edges[0]["node"]["spec"])
    tokens = [series.X_AXIS, "progress/checkpoints_complete", *[metric for _, metric in METRICS],
              *[r["run_id"] for r in wanted], *[r["color"] for r in wanted]]
    if any(token not in wire for token in tokens):
        raise PublicationError("Native workspace lost required panels, runs or colors")
    return saved


def publish_once(campaign, root, *, wandb_module=None, workspace_verifier=verify_workspace,
                 require_complete=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with series._lock(root / ".publisher.lock", blocking=False):
        workspace = workspace_verifier(campaign, root / "workspace.json", wandb_module)
        statuses = stage_completed(campaign)
        missing = {key for key, status in statuses.items() if require_complete and status == "pending"}
        statuses.update({key: "failed" for key in missing})
        rows = progress_rows(campaign, statuses)
        for row in rows:
            with series.Publisher(row["run_dir"], owner=OWNER, wandb_module=wandb_module) as publisher:
                publisher.run.name = row["name"]
                publisher.run.config.update(dict(campaign_id=campaign["campaign_id"], source_commit=campaign["source_commit"], execution_action="sample", regularizer="forward prior KL", checkpoint_id=row["checkpoint_id"]))
                for _, metric in METRICS:
                    publisher.run.define_metric(metric, step_metric=series.X_AXIS, step_sync=False, hidden=False)
                publisher.publish_pending()
                publisher.run.summary.update({"progress/checkpoints_complete": row["complete"], "progress/checkpoints_expected": 1,
                    "progress/failed_cells": row["failed"], "progress/status": "complete" if row["complete"] else row["status"]})
                if row["checkpoint_id"] in missing:
                    publisher.run.summary["progress/failure_reason"] = "Worker terminated without a validated production receipt"
        rows = progress_rows(campaign, statuses)
        result = dict(updated_at=now(), campaign_id=campaign["campaign_id"], cells=statuses, runs=rows, workspace_url=workspace["url"],
                      require_complete=require_complete,
                      all_complete=all(r["published"] == 1 for r in rows))
        series._atomic_json(root / "status.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("layout", "publish"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--require-complete", action="store_true",
                        help="Final --once after all workers terminate: publish missing cells as failed and exit nonzero")
    parser.add_argument("--jobs", nargs="+", help="Slurm jobs to follow; bounded by max-seconds")
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    campaign = load_campaign(args.campaign)
    args.root.mkdir(parents=True, exist_ok=True)
    try:
        if args.command == "layout":
            result = install_workspace(campaign, args.root / "workspace.json")
        else:
            if args.require_complete and not args.once:
                raise PublicationError("--require-complete requires --once after all worker jobs terminate")
            if not args.once and not args.jobs:
                raise PublicationError("Use --once or explicit --jobs for bounded publication")
            if args.max_seconds < 1 or not 1 <= args.poll_seconds <= 60:
                raise PublicationError("Use positive bounded polling, at most 60 seconds per wait")
            deadline = time.monotonic() + args.max_seconds
            while True:
                result = publish_once(campaign, args.root, require_complete=args.require_complete)
                print(series._json(result), flush=True)
                if args.require_complete and not result["all_complete"]:
                    raise PublicationError("Final publication is incomplete; missing worker results are marked failed")
                if args.once or result["all_complete"]:
                    break
                states = subprocess.check_output(["squeue", "-h", "-j", ",".join(args.jobs), "-o", "%T"], text=True).strip()
                if not states or time.monotonic() >= deadline:
                    raise PublicationError("Workers stopped or publisher deadline reached with unpublished results")
                time.sleep(min(args.poll_seconds, max(0, deadline-time.monotonic())))
    except Exception as error:
        message = str(error) if isinstance(error, PublicationError) else "Inspect validated receipts and the publication journal."
        series._atomic_json(args.root / "publication-failure.json", dict(stage=args.command, error_type=type(error).__name__, message=message, at=now()))
        raise SystemExit(f"{args.command} failed ({type(error).__name__}): {message}") from None
    print(series._json(result), flush=True)


if __name__ == "__main__":
    main()
