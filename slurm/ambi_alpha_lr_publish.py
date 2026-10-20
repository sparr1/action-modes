"""Validate, stage and publish inner-temperature learning-rate episodes into a dedicated native view.

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
PREFIX = "Inner alpha LR / "
CHECKPOINTS = {"sac_aux_100k": (100000, "SAC-trained"), "mppi_aux_400k": (400000, "MPPI-trained")}
RATES = {"0p003": 0.003, "0p03": 0.03, "0p3": 0.3}
COLORS = {"sac_aux_100k": ("#D97706", "#DC2626", "#7C3AED"),
          "mppi_aux_400k": ("#059669", "#2563EB", "#DB2777")}
EXISTING = [
    dict(run_id="9edcea495bf243a5ab9a6e139e603b19", name="MPPI-trained | alpha LR 0.0003 | H3 J8 sampled", color="#0F766E"),
    dict(run_id="566a5d7c358b4cbfa056bc22ec715929", name="SAC-trained | alpha LR 0.0003 | H3 J8 sampled", color="#6B7280"),
]
METRICS = [
    ("Episode return (five-seed mean)", "eval/return_mean"),
    ("Return variability (sample SD)", "eval/return_sample_std"),
    ("Improvement over paired sampled prior", "eval/paired_gain_mean"),
    ("Sampled prior return at paired checkpoint", "reference/return_mean"),
    ("Controller seconds per decision", "runtime/control_seconds_per_decision"),
    ("Inner entropy (update-time batches)", "diagnostics/inner_actor_entropy_mean"),
    ("Initial inner alpha", "diagnostics/inner_alpha_initial_mean"),
    ("Final inner alpha", "diagnostics/inner_alpha_final_mean"),
    ("Inner alpha change", "diagnostics/inner_alpha_delta_mean"),
    ("Temperature optimizer steps", "work/temperature_updates"),
]
PANEL_COUNT = 3 + len(METRICS)


class PublicationError(ValueError):
    """Local, credential-free validation error safe to report to the user."""


def now():
    return datetime.now(timezone.utc).isoformat()


def _hex_digest(value, length=64):
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def load_campaign(path):
    campaign = series._read(path)
    if (campaign.get("campaign_id") != "alpha-lr-episodes-20261020"
            or campaign.get("seeds") != [101, 102, 103, 104, 105]
            or campaign.get("controller_seed") != 55 or campaign.get("max_steps") != 500
            or campaign.get("execution_action") != "sample"):
        raise PublicationError("Expected the authorized six-cell alpha-LR sampled episode campaign")
    expected = {f"{checkpoint}__lr{token}" for checkpoint in CHECKPOINTS for token in RATES}
    cells = campaign.get("cells", [])
    if len(cells) != 6 or {c.get("cell_id") for c in cells} != expected:
        raise PublicationError("Expected SAC100k and MPPI400k at each of the three new learning rates")
    if not _hex_digest(campaign.get("source_commit"), 40):
        raise PublicationError("Campaign requires an exact committed source SHA")
    if any(not _hex_digest(campaign.get(key)) for key in ("matrix_sha256", "profile_sha256")):
        raise PublicationError("Campaign requires pinned matrix and profile hashes")
    if any(len({c[key] for c in cells}) != 6 for key in ("run_id", "run_dir", "output_dir")):
        raise PublicationError("Each cell requires its own run and result directory")
    for cell in cells:
        checkpoint = cell["checkpoint_id"]
        if checkpoint not in CHECKPOINTS:
            raise PublicationError("Unexpected checkpoint identity")
        token = cell["cell_id"].removeprefix(checkpoint + "__lr")
        if (token not in RATES or cell["temperature_lr"] != RATES[token]
                or cell["step"] != CHECKPOINTS[checkpoint][0]
                or cell["selector"] != "alpha_lr/lr_" + token):
            raise PublicationError("Cell checkpoint, learning rate or selector mismatch")
        if any(not _hex_digest(cell.get(key)) for key in
               ("checkpoint_sha256", "reference_manifest_sha256", "baseline_manifest_sha256")):
            raise PublicationError("Cell requires pinned checkpoint and reference hashes")
    for checkpoint in CHECKPOINTS:
        selected = [c for c in cells if c["checkpoint_id"] == checkpoint]
        for key in ("checkpoint_sha256", "source_run", "reference_manifest_sha256", "baseline_manifest_sha256"):
            if len({c[key] for c in selected}) != 1:
                raise PublicationError("Learning-rate cells must share the same checkpoint and references")
    return campaign


def curve_rows(campaign):
    rows = []
    for cell in campaign["cells"]:
        checkpoint, rate = cell["checkpoint_id"], cell["temperature_lr"]
        index = list(RATES.values()).index(rate)
        rows.append(dict(run_id=cell["run_id"], run_dir=cell["run_dir"], checkpoint_id=checkpoint,
                         cell_id=cell["cell_id"], temperature_lr=rate,
                         name=f"{CHECKPOINTS[checkpoint][1]} | alpha LR {rate:g} | H3 J8 sampled",
                         color=COLORS[checkpoint][index]))
    return rows


def validate_registry(cell):
    registry = series.load_run(cell["run_dir"])
    if (registry["run_id"] != cell["run_id"] or registry["owner"] != OWNER
            or registry["entity"] != ENTITY or registry["project"] != PROJECT
            or registry["identity"]["backbone"] != cell["source_run"]):
        raise PublicationError("Allocated publication identity or owner changed")
    planner = registry["identity"]["planner"]
    expected = dict(inner_rollout_horizon=3, inner_rounds=8, inner_rollouts_per_round=128,
                    inner_batch_size=256, inner_critic_updates_per_round=16,
                    inner_actor_updates_per_round=4, inner_entropy_enabled=True,
                    inner_temperature_mode="auto", inner_temperature_initialization="inherit_outer",
                    inner_temperature_lr=cell["temperature_lr"], inner_actor_lr=0.0003,
                    inner_critic_lr=0.0003)
    settings = planner.get("settings", {})
    if (planner.get("type") != "sac" or planner.get("action_rule") != "tanh_sample"
            or any(settings.get(k) != v for k, v in expected.items())
            or settings.get("inner_outer_policy_kl_coef", 0) != 0):
        raise PublicationError("Allocated run does not match the authorized alpha-LR controller")
    return registry


def stage_completed(campaign):
    from utils.eval_series_data import load_records
    status = {}
    for cell in campaign["cells"]:
        validate_registry(cell)
        key, output = cell["cell_id"], Path(cell["output_dir"])
        receipt_path, manifest = output / "receipt.json", output / "bundle/manifest.json"
        if not receipt_path.exists():
            status[key] = "failed" if (output / "failure.json").exists() else "pending"
            continue
        receipt = series._read(receipt_path)
        if (receipt.get("validated") is not True or receipt.get("smoke") is not False
                or receipt.get("checkpoint_id") != cell["checkpoint_id"] or receipt.get("cell_id") != key
                or receipt.get("temperature_lr") != cell["temperature_lr"]
                or receipt.get("source_commit") != campaign["source_commit"]):
            raise PublicationError("Only validated production receipts from the campaign source can be published")
        results = output / "results.json"
        if (receipt.get("manifest_sha256") != series._file_digest(manifest)
                or receipt.get("results_sha256") != series._file_digest(results)):
            raise PublicationError("Validated result or manifest changed")
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
        metrics = record["metrics"]
        if (any(metric not in metrics for _, metric in METRICS)
                or metrics.get("eval/frozen_state_unchanged") is not True
                or metrics.get("work/temperature_updates") != 32 * 500 * len(campaign["seeds"])):
            raise PublicationError("Record lacks complete alpha/entropy metrics or the expected frozen 32-temperature-step solves")
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
        complete = int(status.get(row["cell_id"]) == "complete")
        failed = int(status.get(row["cell_id"]) == "failed")
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
    markdown = """**Inner temperature learning rate, sampled-action frozen H3/J8 evaluation.**

Six new cells: SAC-trained 100k and MPPI-trained 400k, each at alpha learning
rates 0.003, 0.03 and 0.3. The default 0.0003 results are reused. Each new point
uses seeds 101–105, controller seed 55 and 500 decisions. Auto entropy and its
32 temperature updates remain enabled; only inner temperature learning rate
changes. Outer learning is frozen, actor/critic learning rates are unchanged,
and KL regularization is off. Initial alpha is inherited from each checkpoint.

X is source-training decisions. Existing baseline points at other checkpoints
are context only. New rates use point markers, readable backbone/rate labels,
distinct colors, native hover and no aggregation. Compare rates within the same
checkpoint. Pending results never appear as zero returns. Progress shows one
expected result per new run; a final failed worker is explicitly marked failed.

Return SD is environment-seed variability. Paired gain and reference return use
the exact sampled prior at the same checkpoint. Entropy is an update-batch
statistic, not final-root entropy. Alpha initial/final/change are means over real
decisions, with a fresh solve at each decision. Timing is episode controller time.
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
        wr.ScalarChart(title="Completed alpha-LR cells (one per run)", metric="progress/checkpoints_complete", layout=wr.Layout(w=12, h=6)),
        wr.ScalarChart(title="Failed alpha-LR cells", metric="progress/failed_cells", layout=wr.Layout(w=12, h=6))]),
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
        work = ws.Workspace(entity=ENTITY, project=PROJECT, name="Inner alpha LR: sampled H3 J8, SAC100k and MPPI400k",
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
    if len(panels) != PANEL_COUNT or any(check.runset_settings.run_settings[r["run_id"]].color != r["color"] for r in rows):
        raise PublicationError("Native workspace panel/color read-back failed")
    for panel in (p for p in panels if isinstance(p, wr.LinePlot)):
        if panel.aggregate or getattr(panel.x, "name", panel.x) != series.X_AXIS:
            raise PublicationError("Native workspace lost independent checkpoint-axis curves")
        metric = getattr(panel.y[0], "name", panel.y[0])
        if any(panel.line_titles.get(r["run_id"] + ":" + metric) != r["name"] for r in rows):
            raise PublicationError("Native workspace lost readable curve aliases")
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
    query = 'query View($entityName:String,$name:String,$viewName:String){project(name:$name,entityName:$entityName){allViews(viewType:"project-view",viewName:$viewName){edges{node{id spec}}}}}'
    variables = {"entityName": ENTITY, "name": PROJECT, "viewName": "nw-" + name + "-v"}
    api = wandb_module.Api(timeout=30)
    if hasattr(api, "_service_api"):
        response = api._service_api.execute_graphql(query, variables=variables)
    else:
        try:
            gql = importlib.import_module("wandb_gql").gql
        except ImportError:
            gql = importlib.import_module("wandb.vendor.gql").gql
        response = api.client.execute(gql(query), variable_values=variables)
    edges = (response.get("project") or {}).get("allViews", {}).get("edges", [])
    if len(edges) != 1:
        raise PublicationError("Verified native workspace is missing remotely")
    wire = json.dumps(edges[0]["node"]["spec"])
    tokens = [series.X_AXIS, "progress/checkpoints_complete", *[metric for _, metric in METRICS],
              *[r["run_id"] for r in wanted], *[r["color"] for r in wanted], *[r["name"] for r in wanted]]
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
                publisher.run.config.update(dict(campaign_id=campaign["campaign_id"], source_commit=campaign["source_commit"], execution_action="sample", regularizer="entropy", checkpoint_id=row["checkpoint_id"], cell_id=row["cell_id"], inner_temperature_lr=row["temperature_lr"]))
                for _, metric in METRICS:
                    publisher.run.define_metric(metric, step_metric=series.X_AXIS, step_sync=False, hidden=False)
                publisher.publish_pending()
                publisher.run.summary.update({"progress/checkpoints_complete": row["complete"], "progress/checkpoints_expected": 1,
                    "progress/failed_cells": row["failed"], "progress/status": "complete" if row["complete"] else row["status"]})
                if row["cell_id"] in missing:
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
