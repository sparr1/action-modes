"""Publish the four matched MPPI-trained 500k objective comparisons.

GPU workers remain offline. Install the dedicated layout in the separate pinned
workspace runtime, then copy its receipt to this CPU publisher. Existing 400k
context runs are read-only; four new curve identities own the new results.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import eval_series as series
from utils.aux_return_identity import AUX_RETURN_SOURCE_DEFAULTS
from slurm.ambi_entropy_kl_publish import PublicationError, now, _hex_digest

ENTITY, PROJECT, OWNER = "rwgao_b-brown-university", "ambi-inner-bench", "oscar-rgao48"
PREFIX = "MPPI 500k objectives / "
METHODS = {
    "prior": ("MPPI-trained 500k | sampled prior (no inner)", "#6B7280"),
    "entropy": ("MPPI-trained 500k | entropy only | H3 J8 sampled", "#0F766E"),
    "kl": ("MPPI-trained 500k | KL only | H3 J8 sampled", "#DB2777"),
    "both": ("MPPI-trained 500k | entropy + KL | H3 J8 sampled", "#2563EB"),
}
EXISTING = [
    dict(run_id="9edcea495bf243a5ab9a6e139e603b19", name="Context: MPPI-trained earlier checkpoints | entropy only",
         color="#6BAFA5", line_mark="solid"),
    dict(run_id="db24d8d0c7a245189b2ccc896d1e8cc6", name="Context: MPPI-trained 400k | KL only",
         color="#E8A0C2", line_mark="points"),
    dict(run_id="44c306e7c69c4625a1b1b7dce156fde3", name="Context: MPPI-trained 400k | entropy + KL, beta=outer alpha",
         color="#9BAEEB", line_mark="points"),
]
METRICS = [
    ("Episode return (five-seed mean)", "eval/return_mean"),
    ("Return variability (sample SD)", "eval/return_sample_std"),
    ("Improvement over paired sampled prior", "eval/paired_gain_mean"),
    ("Paired gain variability (sample SD)", "eval/paired_gain_sample_std"),
    ("Sampled prior return", "reference/return_mean"),
    ("Controller seconds per decision", "runtime/control_seconds_per_decision"),
    ("Inner entropy (training-update batches)", "diagnostics/inner_actor_entropy_mean"),
    ("Mean per-solve maximum batch entropy", "diagnostics/inner_actor_entropy_max_mean"),
    ("Final KL(inner || frozen prior)", "diagnostics/inner_final_outer_policy_kl_mean"),
    ("Initial inner alpha (zero when entropy is disabled)", "diagnostics/inner_alpha_initial_mean"),
    ("Final inner alpha", "diagnostics/inner_alpha_final_mean"),
    ("Inner alpha change", "diagnostics/inner_alpha_delta_mean"),
    ("Temperature optimizer steps", "work/temperature_updates"),
]
PANEL_COUNT = len(METRICS) + 3


def _line_color(value):
    # W&B expands a saved hex color into {color, transparentColor} on read-back.
    return value.get("color") if isinstance(value, dict) else value


def load_campaign(path):
    """Validate the driver's exact scientific pins without reading model tensors."""
    from slurm import ambi_entropy_kl_mppi500k as driver
    campaign = series._read(path)
    if (campaign.get("profile_sha256") != series._file_digest(driver.REPO / driver.PROFILE)
            or campaign.get("matrix_sha256") != series._file_digest(driver.REPO / driver.MATRIX)):
        raise PublicationError("Campaign profile or matrix changed from the pinned source")
    driver.validate_campaign(campaign)
    cells = campaign["cells"]
    if (len(cells) != 4 or {cell.get("mode") for cell in cells} != set(METHODS)
            or campaign.get("seeds") != [101, 102, 103, 104, 105]
            or campaign.get("controller_seed") != 55 or campaign.get("max_steps") != 500
            or campaign.get("execution_action") != "sample"):
        raise PublicationError("Expected the four matched sampled MPPI500k objective cells")
    if not _hex_digest(campaign.get("source_commit"), 40):
        raise PublicationError("Campaign requires an exact committed source SHA")
    for key in ("run_id", "run_dir", "output_dir", "cell_id"):
        if len({cell[key] for cell in cells}) != 4:
            raise PublicationError("Each objective requires its own run and output directory")
    if {cell["run_id"] for cell in cells} & {row["run_id"] for row in EXISTING}:
        raise PublicationError("New objectives cannot reuse existing context run IDs")
    return campaign


def curve_rows(campaign):
    return [{**cell, "name": METHODS[cell["mode"]][0], "color": METHODS[cell["mode"]][1],
             "line_mark": "points"} for cell in campaign["cells"]]


def validate_registry(campaign, cell):
    registry = series.load_run(cell["run_dir"])
    if (registry["run_id"] != cell["run_id"] or registry["owner"] != OWNER
            or registry["entity"] != ENTITY or registry["project"] != PROJECT
            or registry["identity"]["backbone"] != cell["source_run"]):
        raise PublicationError("Allocated publication identity or owner changed")
    protocol = registry["identity"]["protocol"]
    if any(protocol.get(key) != value for key, value in {
            "action_rule": "tanh_sample", "environment_seeds": campaign["seeds"],
            "max_steps": 500, "controller_seed": 55}.items()):
        raise PublicationError("Allocated run has the wrong sampled episode protocol")
    planner = registry["identity"]["planner"]
    method = cell["mode"]
    if method == "prior":
        if planner != {"type": "prior", "action_rule": "tanh_sample"}:
            raise PublicationError("The prior must execute the inherited primary actor without an inner solver")
    else:
        entropy = method in ("entropy", "both")
        expected = dict(inner_rollout_horizon=3, inner_rounds=8, inner_rollouts_per_round=128,
            inner_batch_size=256, inner_critic_updates_per_round=16, inner_actor_updates_per_round=4,
            inner_entropy_enabled=entropy, inner_temperature_mode="auto" if entropy else "inherit_outer",
            inner_temperature_initialization="inherit_outer",
            inner_actor_lr=0.0003, inner_critic_lr=0.0003, inner_outer_policy_kl_coef=cell["beta"],
            inner_actor_source="sac", inner_horizon_actor_source="sac",
            inner_critic_source="aux_return", inner_horizon_critic_source="aux_return",
            inner_sac_critic_target="reward_only")
        # The canonical identity omits the inactive temperature optimizer's LR.
        # The raw receipt still pins it for every mode; active alpha arms must
        # retain the actual LR in their normalized publication identity.
        if entropy:
            expected["inner_temperature_lr"] = 0.0003
        settings = planner.get("settings", {})
        if (planner.get("type") != "sac" or planner.get("action_rule") != "tanh_sample"
                or any(settings.get(key, AUX_RETURN_SOURCE_DEFAULTS.get(key)) != value
                       for key, value in expected.items())):
            raise PublicationError("Allocated run has the wrong entropy/KL objective or update recipe")
    return registry


def _finite(metrics, keys):
    return all(isinstance(metrics.get(key), (int, float)) and not isinstance(metrics.get(key), bool)
               and math.isfinite(metrics[key]) for key in keys)


def validate_record(campaign, cell, record):
    if record["checkpoint"] != {"step": 500000, "sha256": cell["checkpoint_sha256"]}:
        raise PublicationError("Normalized record checkpoint differs from the pinned 500k model")
    if sorted(episode["seed"] for episode in record["episodes"]) != campaign["seeds"]:
        raise PublicationError("Record must contain all five paired seeds")
    metrics, method = record["metrics"], cell["mode"]
    if (metrics.get("eval/frozen_state_unchanged") is not True
            or metrics.get("eval/episodes") != 5 or metrics.get("work/environment_decisions") != 2500):
        raise PublicationError("Record must contain complete frozen full episodes")
    counts = {"actor": 0 if method == "prior" else 80000,
              "critic": 0 if method == "prior" else 320000,
              "temperature": 80000 if method in ("entropy", "both") else 0}
    if any(metrics.get(f"work/{key}_updates") != count for key, count in counts.items()):
        raise PublicationError("Unexpected optimizer work for the selected objective")
    required = ["eval/return_mean", "eval/return_sample_std", "runtime/control_seconds_per_decision"]
    if method != "prior":
        required += [metric for _, metric in METRICS[6:12]]
        expected_alpha = cell["outer_alpha"] if method in ("entropy", "both") else 0.0
        if (not _finite(metrics, required)
                or not math.isclose(metrics["diagnostics/inner_alpha_initial_mean"], expected_alpha,
                                    rel_tol=1e-6, abs_tol=1e-10)):
            raise PublicationError("Missing policy diagnostics or incorrect initial alpha")
        if method == "kl" and any(metrics[key] != 0.0 for key in
                ("diagnostics/inner_alpha_final_mean", "diagnostics/inner_alpha_delta_mean")):
            raise PublicationError("Entropy-disabled KL-only must keep actual alpha at zero")
    elif not _finite(metrics, required):
        raise PublicationError("Missing prior return or controller timing")
    return counts


def _paired_prior(campaign, cell, record, receipt):
    """Add a self-reference for the prior; require real matched references otherwise."""
    if cell["mode"] == "prior":
        priors = [ep["return"] for ep in record["episodes"]]
        for ep in record["episodes"]:
            ep["paired_return_delta"] = 0.0
        record["metrics"].update({"eval/paired_gain_mean": 0.0, "eval/paired_gain_sample_std": 0.0,
                                  "eval/paired_episodes": 5})
        # A no-inner prior has no meaningful inner entropy, KL or temperature readout.
        record["metrics"] = {k: v for k, v in record["metrics"].items() if not k.startswith("diagnostics/")}
    else:
        if any(ep.get("paired_return_delta") is None for ep in record["episodes"]):
            raise PublicationError("Adapted results require the exact paired sampled-prior reference")
        prior_cell = next(c for c in campaign["cells"] if c["mode"] == "prior")
        prior_output = Path(prior_cell["output_dir"])
        prior_receipt = series._read(prior_output / "receipt.json")
        prior_manifest = prior_output / "bundle/manifest.json"
        prior_results = prior_output / "results.json"
        if (prior_receipt.get("validated") is not True or prior_receipt.get("smoke") is not False
                or prior_receipt.get("source_commit") != campaign["source_commit"]
                or receipt.get("prior_receipt_sha256") != series._file_digest(prior_output / "receipt.json")
                or receipt.get("prior_manifest_sha256") != series._file_digest(prior_manifest)
                or receipt.get("prior_results_sha256") != series._file_digest(prior_results)
                or receipt.get("prior_trace_sha256") != prior_receipt.get("trace_sha256")
                or prior_receipt.get("manifest_sha256") != series._file_digest(prior_manifest)
                or prior_receipt.get("results_sha256") != series._file_digest(prior_results)):
            raise PublicationError("Validated paired prior changed or is not pinned by the adapted receipt")
        prior_payload = series._read(prior_results)
        if prior_payload.get("checkpoint_sha256") != cell["checkpoint_sha256"]:
            raise PublicationError("Paired prior uses a different checkpoint")
        prior = prior_payload["results"][0]
        if (prior.get("outer_state_unchanged") is not True or prior.get("execution_action") != "sample"
                or prior.get("selector") != prior_cell["selector"]):
            raise PublicationError("Paired prior must be the frozen sampled primary actor")
        prior_episodes = {ep["seed"]: ep for ep in prior["episodes"]}
        if sorted(prior_episodes) != campaign["seeds"]:
            raise PublicationError("Paired prior has different seed coverage")
        priors = []
        for ep in record["episodes"]:
            base = prior_episodes[ep["seed"]]
            if (base.get("length") != 500 or base.get("truncated_by_evaluator") is not False
                    or ep.get("solver_seed") != base.get("solver_seed")
                    or not math.isclose(ep["paired_return_delta"], ep["return"] - base["return"],
                                        rel_tol=1e-8, abs_tol=1e-8)):
                raise PublicationError("Paired gains disagree with the exact sampled-prior episodes")
            priors.append(base["return"])
        if not _finite(record["metrics"], ["eval/paired_gain_mean", "eval/paired_gain_sample_std"]):
            raise PublicationError("Missing paired improvement metrics")
    record["metrics"]["reference/return_mean"] = statistics.mean(priors)
    record["metrics"]["reference/return_sample_std"] = statistics.stdev(priors)


def _summary_artifacts(cell, output, selected, record, receipt):
    names = selected.get("trace_files", [])
    if len(names) != len(set(names)) or (cell["mode"] != "prior" and not names):
        raise PublicationError("Missing or duplicate full traces")
    if set(names) != set(receipt.get("trace_sha256", {})):
        raise PublicationError("Receipt does not pin the complete trace list")
    retained, remote = {}, []
    for name in names:
        if name not in record["artifact_files"]:
            raise PublicationError("Full trace file is missing from the normalized record")
        path = Path(record["artifact_files"][name]).resolve(strict=True)
        if not path.is_relative_to((output / "bundle").resolve()) or not path.name.endswith(".jsonl.gz"):
            raise PublicationError("Full trace path escapes the validated bundle")
        digest = series._file_digest(path)
        if receipt["trace_sha256"][name] != digest:
            raise PublicationError("Validated full trace changed")
        remote.append(dict(name=name, path=str(path), sha256=digest, size_bytes=path.stat().st_size))
    retained = {name: path for name, path in record["artifact_files"].items() if name not in names}
    index = Path(cell["run_dir"]) / "remote-traces-500000.json"
    series._atomic_json(index, dict(policy="full traces retained on Oscar", files=remote))
    retained.update({"results.json": str(output / "results.json"),
                     "validation-receipt.json": str(output / "receipt.json"), "remote-traces.json": str(index)})
    if sum(Path(path).stat().st_size for path in retained.values()) > 16 * 1024 * 1024:
        raise PublicationError("Summary artifact exceeds 16 MiB")
    record["artifact_files"] = retained
    record["provenance"].update(artifact_policy="summary-only; full traces retained on Oscar", remote_trace_files=remote)


def stage_completed(campaign):
    from utils.eval_series_data import load_records
    statuses = {}
    for cell in campaign["cells"]:
        validate_registry(campaign, cell)
        if series._file_digest(cell["inventory_path"]) != cell["inventory_sha256"]:
            raise PublicationError("Pinned inventory changed")
        output, key = Path(cell["output_dir"]), cell["cell_id"]
        receipt_path, manifest_path = output / "receipt.json", output / "bundle/manifest.json"
        if not receipt_path.exists():
            statuses[key] = "failed" if (output / "failure.json").exists() else "pending"
            continue
        receipt = series._read(receipt_path)
        expected = {"validated": True, "smoke": False, "source_commit": campaign["source_commit"],
                    **{k: cell[k] for k in ("checkpoint_id", "cell_id", "selector", "mode", "beta", "outer_alpha",
                                              "entropy_enabled", "expected_temperature_steps", "temperature_lr")}}
        if (receipt.get("validated") is not True or receipt.get("smoke") is not False
                or any(receipt.get(k) != v for k, v in expected.items())):
            raise PublicationError("Only exact validated production receipts may be published")
        if (receipt.get("manifest_sha256") != series._file_digest(manifest_path)
                or receipt.get("results_sha256") != series._file_digest(output / "results.json")):
            raise PublicationError("Validated result or manifest changed")
        journal = series._read(Path(cell["run_dir"]) / "publication.json")
        old = list(journal["records"].values())
        if old:
            hashes = old[0]["artifact_sha256"]
            if (len(old) != 1 or old[0]["checkpoint_step"] != 500000
                    or old[0]["checkpoint_sha256"] != cell["checkpoint_sha256"]
                    or hashes.get("manifest.json") != receipt["manifest_sha256"]
                    or hashes.get("results.json") != receipt["results_sha256"]
                    or hashes.get("validation-receipt.json") != series._file_digest(receipt_path)):
                raise PublicationError("Previously accepted result changed")
            statuses[key] = "complete"
            continue
        manifest = series._read(manifest_path)
        if (manifest.get("status") != "complete" or manifest["code"].get("dirty") is not False
                or manifest["code"].get("commit") != campaign["source_commit"]):
            raise PublicationError("Bundle is incomplete or has different scientific source")
        selected = [r for r in manifest["runs"] if r["selector"] == cell["selector"]]
        if len(selected) != 1 or selected[0].get("status") != "complete":
            raise PublicationError("Bundle lacks its complete selected controller")
        records = [r for r in load_records(manifest_path, source_run=cell["source_run"],
                        inventory_path=cell["inventory_path"]) if r["selector"] == cell["selector"]]
        if len(records) != 1:
            raise PublicationError("Bundle must normalize to one exact controller record")
        record = records[0]
        validate_record(campaign, cell, record)
        _paired_prior(campaign, cell, record, receipt)
        _summary_artifacts(cell, output, selected[0], record, receipt)
        series.stage_record(cell["run_dir"], record)
        statuses[key] = "complete"
    return statuses


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
    for package, version in (("wandb", "0.28.1"), ("wandb-workspaces", "0.4.5")):
        if importlib.metadata.version(package) != version:
            raise PublicationError(f"Layout requires separate runtime {package}=={version}")
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr
    path = Path(path)
    rows = EXISTING + curve_rows(campaign)
    previous = series._read(path) if path.exists() else {}
    if previous and previous.get("campaign_sha256") != series._digest(campaign):
        raise PublicationError("Workspace receipt belongs to another campaign")
    markdown = """**MPPI-trained 500k checkpoint: four paired sampled-action controllers.**

The sampled prior, entropy-only inner SAC, KL-only inner SAC and joint
entropy-plus-KL inner SAC each run five full 500-decision episodes with seeds
101–105 and controller seed 55. The checkpoint and outer state are frozen.
All adapted controllers use H3/J8/C16/A4/N128/B256 and auxiliary return Q for
critic initialization and the horizon tail. KL beta equals this checkpoint's
saved outer alpha and remains fixed. Entropy-only and joint refinement retain
automatic alpha at learning rate 0.0003; KL-only disables entropy and temperature
updates. Other learning settings are unchanged. Prior has no inner updates.

Each new curve owns one 500k point. Earlier checkpoint curves are context only;
they retain connected lines where multiple checkpoints exist. No artificial
line joins independently identified runs. Native hover/legend and distinct
colors identify every curve. Existing context runs are not modified. Pending
cells are shown in the progress panels, never as zero returns.

Returns and paired gains use the same sampled prior and matched episode seeds.
SD is variability across five environment seeds, not uncertainty across training
seeds. Entropy is the average of actor-update batches; its maximum is the mean
of each solve's maximum batch entropy. Final KL is the final inner-to-prior
joint-action KL averaged across real decisions. Alpha readouts are actual engine
alpha: zero for entropy-disabled KL-only. Prior inner diagnostics are omitted.
Timing is measured episode controller time. Full traces remain on Oscar;
uploaded summaries are bounded to 16 MiB per result.
"""
    def plot(title, metric):
        return wr.LinePlot(title=title, x=series.X_AXIS, y=[metric], aggregate=False,
            smoothing_type="none", legend_template="${run:name}", max_runs_to_show=len(rows),
            title_x="Source training decisions", layout=wr.Layout(w=12, h=8),
            line_colors={r["run_id"] + ":" + metric: r["color"] for r in rows},
            line_marks={r["run_id"] + ":" + metric: r["line_mark"] for r in rows},
            line_titles={r["run_id"] + ":" + metric: r["name"] for r in rows})
    sections = [ws.Section(name=PREFIX + "Protocol and progress", is_open=True, panels=[
        wr.MarkdownPanel(markdown=markdown, layout=wr.Layout(w=24, h=10)),
        wr.ScalarChart(title="Completed 500k cells (one per new run)", metric="progress/checkpoints_complete", layout=wr.Layout(w=12, h=6)),
        wr.ScalarChart(title="Failed 500k cells", metric="progress/failed_cells", layout=wr.Layout(w=12, h=6))]),
        ws.Section(name=PREFIX + "Return and improvement", is_open=True, panels=[plot(*item) for item in METRICS[:5]]),
        ws.Section(name=PREFIX + "Runtime and policy diagnostics", is_open=True, panels=[plot(*item) for item in METRICS[5:]])]
    runset = ws.RunsetSettings(filters=[expr.Metric("ID").isin([r["run_id"] for r in rows])], groupby=[],
        run_settings={r["run_id"]: ws.RunSettings(color=r["color"], disabled=False) for r in rows})
    if previous.get("url"):
        work = ws.Workspace.from_url(previous["url"])
        if work.entity != ENTITY or work.project != PROJECT:
            raise PublicationError("Workspace target mismatch")
        replacement = {s.name: s for s in sections}
        work.sections = [replacement.pop(s.name, s) for s in work.sections] + list(replacement.values())
        work.runset_settings = runset
    else:
        work = ws.Workspace(entity=ENTITY, project=PROJECT, name="MPPI-trained 500k: prior, entropy, KL and joint refinement",
            sections=sections, auto_generate_panels=False, runset_settings=runset,
            settings=ws.WorkspaceSettings(x_axis=series.X_AXIS, max_runs=len(rows),
                tooltip_number_of_runs="all_runs", tooltip_color_run_names=True))
    work.save()
    saved = dict(campaign_id=campaign["campaign_id"], campaign_sha256=series._digest(campaign),
                 url=work.url, updated_at=now(), api_verified=False, runs=rows)
    series._atomic_json(path, saved)
    check = ws.Workspace.from_url(work.url)
    panels = [p for s in check.sections if s.name.startswith(PREFIX) for p in s.panels]
    if len(panels) != PANEL_COUNT or any(check.runset_settings.run_settings[r["run_id"]].color != r["color"] for r in rows):
        raise PublicationError("Native workspace panel/color read-back failed")
    for panel in (p for p in panels if isinstance(p, wr.LinePlot)):
        if panel.aggregate or getattr(panel.x, "name", panel.x) != series.X_AXIS:
            raise PublicationError("Native workspace lost independent checkpoint-axis curves")
        metric = getattr(panel.y[0], "name", panel.y[0])
        if any(panel.line_titles.get(r["run_id"] + ":" + metric) != r["name"]
               or panel.line_marks.get(r["run_id"] + ":" + metric) != r["line_mark"]
               or _line_color(panel.line_colors.get(r["run_id"] + ":" + metric)) != r["color"] for r in rows):
            raise PublicationError("Native workspace lost readable aliases, colors or context lines")
    saved.update(api_verified=True, panel_count=len(panels))
    series._atomic_json(path, saved)
    return saved


def verify_workspace(campaign, path, wandb_module=None):
    if not Path(path).exists():
        raise PublicationError("Install and verify the dedicated native layout before publication")
    saved = series._read(path)
    wanted = EXISTING + curve_rows(campaign)
    if (saved.get("campaign_sha256") != series._digest(campaign) or not saved.get("api_verified")
            or {r["run_id"] for r in saved.get("runs", [])} != {r["run_id"] for r in wanted}):
        raise PublicationError("Native workspace receipt does not match the campaign")
    parsed = urlparse(saved["url"])
    if parsed.hostname != "wandb.ai" or parsed.path.rstrip("/") != f"/{ENTITY}/{PROJECT}":
        raise PublicationError("Dedicated view must belong to the intended W&B project")
    name = parse_qs(parsed.query).get("nw", [None])[0]
    if not name:
        raise PublicationError("Native view ID is missing")
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
    tokens = [series.X_AXIS, "progress/checkpoints_complete", "progress/failed_cells",
              *[metric for _, metric in METRICS], *[r["run_id"] for r in wanted],
              *[r["color"] for r in wanted], *[r["name"] for r in wanted]]
    if any(token not in wire for token in tokens):
        raise PublicationError("Native workspace lost required panels, runs or colors")
    return saved


def publish_once(campaign, root, *, wandb_module=None, workspace_verifier=verify_workspace, require_complete=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with series._lock(root / ".publisher.lock", blocking=False):
        workspace = workspace_verifier(campaign, root / "workspace.json", wandb_module)
        statuses = stage_completed(campaign)
        missing = {key for key, status in statuses.items() if require_complete and status == "pending"}
        statuses.update({key: "failed" for key in missing})
        for row in progress_rows(campaign, statuses):
            with series.Publisher(row["run_dir"], owner=OWNER, wandb_module=wandb_module) as publisher:
                publisher.run.name = row["name"]
                publisher.run.config.update(dict(campaign_id=campaign["campaign_id"], source_commit=campaign["source_commit"],
                    execution_action="sample", objective_mode=row["mode"], checkpoint_id=row["checkpoint_id"],
                    cell_id=row["cell_id"], inner_temperature_lr=row["temperature_lr"],
                    inner_entropy_enabled=row["entropy_enabled"], inner_outer_policy_kl_coef=row["beta"],
                    checkpoint_outer_alpha=row["outer_alpha"]))
                for _, metric in METRICS:
                    publisher.run.define_metric(metric, step_metric=series.X_AXIS, step_sync=False, hidden=False)
                publisher.publish_pending()
                publisher.run.summary.update({"progress/checkpoints_complete": row["complete"],
                    "progress/checkpoints_expected": 1, "progress/failed_cells": row["failed"],
                    "progress/status": "complete" if row["complete"] else row["status"]})
                if row["cell_id"] in missing:
                    publisher.run.summary["progress/failure_reason"] = "Worker terminated without a validated production receipt"
        rows = progress_rows(campaign, statuses)
        result = dict(updated_at=now(), campaign_id=campaign["campaign_id"], cells=statuses, runs=rows,
                      workspace_url=workspace["url"], require_complete=require_complete,
                      all_complete=all(r["published"] == 1 for r in rows))
        series._atomic_json(root / "status.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("layout", "publish"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--jobs", nargs="+")
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
                raise PublicationError("--require-complete requires --once after all workers terminate")
            if not args.once and not args.jobs:
                raise PublicationError("Use --once or explicit jobs for bounded publication")
            if args.max_seconds < 1 or not 1 <= args.poll_seconds <= 60:
                raise PublicationError("Use positive bounded polling, at most 60 seconds per wait")
            deadline = time.monotonic() + args.max_seconds
            while True:
                result = publish_once(campaign, args.root, require_complete=args.require_complete)
                print(series._json(result), flush=True)
                if args.require_complete and not result["all_complete"]:
                    raise PublicationError("Final publication incomplete; missing worker results are marked failed")
                if args.once or result["all_complete"]:
                    break
                states = subprocess.check_output(["squeue", "-h", "-j", ",".join(args.jobs), "-o", "%T"], text=True).strip()
                if not states or time.monotonic() >= deadline:
                    raise PublicationError("Workers stopped or publisher deadline reached with unpublished results")
                time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
    except Exception as error:
        message = str(error) if isinstance(error, PublicationError) else "Inspect validated receipts and the publication journal."
        series._atomic_json(args.root / "publication-failure.json",
                            dict(stage=args.command, error_type=type(error).__name__, message=message, at=now()))
        raise SystemExit(f"{args.command} failed ({type(error).__name__}): {message}") from None
    print(series._json(result), flush=True)


if __name__ == "__main__":
    main()
