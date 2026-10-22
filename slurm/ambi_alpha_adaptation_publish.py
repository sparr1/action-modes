"""Publish alpha-adaptation sanity results and separate derived decision curves.

Canonical eval-series runs retain one immutable result per checkpoint. Six
explicitly derived display runs provide native decision-axis curves, including
the two reused reset baselines. Old results and workspaces are never modified.
Full traces stay on Oscar; only compact summaries enter W&B artifacts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import eval_series as series
from slurm.ambi_alpha_lr_publish import PublicationError, CHECKPOINTS, ENTITY, PROJECT, OWNER

CAMPAIGN_ID = "alpha-adaptation-episodes-20261021"
PREFIX = "Alpha adaptation / "
AXIS = "episode/decision"
METHODS = {"reset": ("Reset alpha each solve", "#6B7280"),
           "repeat100": ("100 temperature steps per actor update", "#D97706"),
           "carry": ("Carry alpha and Adam within episode", "#2563EB")}
CURVE_METRICS = [("Update-batch entropy", "entropy"),
                 ("Entropy target", "target_entropy"),
                 ("Entropy minus target", "entropy_error"),
                 ("Alpha at solve start", "alpha_initial"),
                 ("Alpha at solve end", "alpha_final"),
                 ("Log alpha at solve end", "log_alpha_final"),
                 ("Cumulative environment reward", "cumulative_reward"),
                 ("Temperature Adam step at solve end", "optimizer_step_final")]
SUMMARY_METRICS = [("Episode return", "return_mean"), ("Return sample SD", "return_sample_std"),
                   ("Paired improvement over sampled prior", "prior_gain_mean"),
                   ("Paired improvement over reset baseline", "baseline_gain_mean"),
                   ("Paired gain sample SD versus baseline", "baseline_gain_sample_std"),
                   ("Controller seconds per decision", "control_seconds_per_decision")]


def now():
    return datetime.now(timezone.utc).isoformat()


def load_campaign(path):
    from slurm import ambi_alpha_adaptation_episodes as driver
    campaign = series._read(path)
    driver.validate_campaign(campaign)
    if (campaign.get("campaign_id") != CAMPAIGN_ID or campaign.get("seeds") != [101, 102, 103, 104, 105]
            or campaign.get("max_steps") != 500 or campaign.get("execution_action") != "sample"):
        raise PublicationError("Expected the authorized sampled alpha-adaptation sanity campaign")
    expected = {(checkpoint, mode) for checkpoint in CHECKPOINTS for mode in ("repeat100", "carry")}
    cells = campaign["cells"]
    if len(cells) != 4 or {(c["checkpoint_id"], c["mode"]) for c in cells} != expected:
        raise PublicationError("Expected two checkpoints and exactly two new mechanisms")
    for key in ("cell_id", "run_id", "run_dir", "output_dir"):
        if len({cell[key] for cell in cells}) != 4:
            raise PublicationError("Every new evaluation requires independent result ownership")
    return campaign


def display_rows(campaign):
    """Stable display identities are explicitly distinct from evaluation identities."""
    result = []
    for checkpoint in CHECKPOINTS:
        for mode, (label, color) in METHODS.items():
            cell = next(c for c in campaign["cells"] if c["checkpoint_id"] == checkpoint
                        and (mode == "reset" or c["mode"] == mode))
            identity = dict(kind="derived-alpha-decision-display-v1", campaign_sha256=series._digest(campaign),
                            checkpoint_id=checkpoint, mode=mode)
            result.append(dict(run_id=series._digest(identity)[:32], checkpoint_id=checkpoint, mode=mode,
                name=f"{CHECKPOINTS[checkpoint][1]} {CHECKPOINTS[checkpoint][0]//1000}k | {label}",
                color=color, identity=identity, cell=cell, reused_result=mode == "reset"))
    return result


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def reduce_decisions(summary, campaign):
    """Equal seed weights at each observed real decision; no missing-data imputation."""
    rows = summary.get("rows", [])
    expected = {(seed, decision) for seed in campaign["seeds"] for decision in range(campaign["max_steps"])}
    coordinates = [(row.get("seed"), row.get("decision_index")) for row in rows]
    if len(coordinates) != len(set(coordinates)) or set(coordinates) != expected:
        raise PublicationError("Decision summaries require every paired seed and decision exactly once")
    by_decision = {decision: [] for decision in range(campaign["max_steps"])}
    for original in rows:
        row = dict(original)
        for key in ("alpha_initial", "alpha_final", "entropy", "target_entropy"):
            if not _finite(row.get(key)):
                raise PublicationError("Decision summary lacks finite " + key)
        if row["alpha_initial"] <= 0 or row["alpha_final"] <= 0:
            raise PublicationError("Enabled temperature must remain positive")
        row["log_alpha_final"] = math.log(row["alpha_final"])
        row["entropy_error"] = row["entropy"] - row["target_entropy"]
        by_decision[row["decision_index"]].append(row)
    reduced = []
    for decision, samples in sorted(by_decision.items()):
        result = {AXIS: decision + 1, "seeds": len(samples)}
        for _, key in CURVE_METRICS:
            values = [sample.get(key) for sample in samples]
            if all(value is None for value in values):
                continue
            if not all(_finite(value) for value in values):
                raise PublicationError("Partial or nonfinite diagnostic coverage: " + key)
            result[key] = statistics.mean(values)
            result[key + "_sample_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        reduced.append(result)
    return reduced


def _episodes(summary, campaign):
    episodes = summary.get("episodes", [])
    values = {episode["seed"]: episode for episode in episodes}
    if (len(episodes) != len(values) or set(values) != set(campaign["seeds"])
            or any(episode.get("length") != campaign["max_steps"] or not _finite(episode.get("return"))
                   for episode in values.values())):
        raise PublicationError("Only complete paired full episodes may be displayed")
    return values


def summarize_episodes(summary, baseline, prior, campaign):
    episodes, bases, priors = (_episodes(value, campaign) for value in (summary, baseline, prior))
    returns = [episodes[seed]["return"] for seed in campaign["seeds"]]
    baseline_gain = [episodes[seed]["return"] - bases[seed]["return"] for seed in campaign["seeds"]]
    prior_gain = [episodes[seed]["return"] - priors[seed]["return"] for seed in campaign["seeds"]]
    for seed in campaign["seeds"]:
        if any(episodes[seed].get("solver_seed") != other[seed].get("solver_seed") for other in (bases, priors)):
            raise PublicationError("Paired summaries have mismatched controller seeds")
    values = dict(return_mean=statistics.mean(returns), return_sample_std=statistics.stdev(returns),
        prior_gain_mean=statistics.mean(prior_gain), prior_gain_sample_std=statistics.stdev(prior_gain),
        baseline_gain_mean=statistics.mean(baseline_gain), baseline_gain_sample_std=statistics.stdev(baseline_gain))
    timings = [row.get("control_seconds") for row in summary["rows"]]
    if all(_finite(value) for value in timings):
        values["control_seconds_per_decision"] = statistics.mean(timings)
    return values


def _reference_episodes(cell):
    manifest = Path(cell["reference_bundle"]) / "manifest.json"
    if series._file_digest(manifest) != cell["reference_manifest_sha256"]:
        raise PublicationError("Pinned sampled-prior reference changed")
    runs = [run for run in series._read(manifest)["runs"] if run.get("status") == "complete"]
    if len(runs) != 1:
        raise PublicationError("Sampled-prior reference must contain one complete run")
    return {"episodes": runs[0].get("result", runs[0])["episodes"]}


def validate_registry(campaign, cell):
    registry = series.load_run(cell["run_dir"])
    if (registry["run_id"] != cell["run_id"] or registry["owner"] != OWNER
            or registry["entity"] != ENTITY or registry["project"] != PROJECT
            or registry["identity"]["backbone"] != cell["source_run"]):
        raise PublicationError("Allocated evaluation ownership changed")
    settings = registry["identity"]["planner"].get("settings", {})
    expected = dict(inner_rollout_horizon=3, inner_rounds=8, inner_rollouts_per_round=128,
        inner_batch_size=256, inner_critic_updates_per_round=16, inner_actor_updates_per_round=4,
        inner_temperature_mode="auto", inner_temperature_lr=0.0003, inner_entropy_enabled=True,
        inner_actor_lr=0.0003, inner_critic_lr=0.0003, inner_outer_policy_kl_coef=0.0,
        inner_temperature_updates_per_actor=100 if cell["mode"] == "repeat100" else 1,
        inner_temperature_scope="episode" if cell["mode"] == "carry" else "action",
        inner_temperature_optimizer_scope="episode" if cell["mode"] == "carry" else "action")
    if any(settings.get(key, 1 if key == "inner_temperature_updates_per_actor" else None) != value
           for key, value in expected.items()):
        raise PublicationError("Evaluation identity changed the temperature mechanism or update budget")
    return registry


def stage_completed(campaign):
    """Stage validated scientific records once, excluding full trace files."""
    from utils.eval_series_data import load_records
    from slurm import ambi_alpha_adaptation_episodes as driver
    statuses = {}
    for cell in campaign["cells"]:
        validate_registry(campaign, cell)
        pins = driver.reference_pins(cell)
        if (any(cell.get(key) != value for key, value in pins.items())
                or series._file_digest(cell["inventory_path"]) != cell["inventory_sha256"]):
            raise PublicationError("Pinned sampled prior, reset baseline or inventory changed")
        output = Path(cell["output_dir"])
        receipt_path = output / "receipt.json"
        if not receipt_path.exists():
            statuses[cell["cell_id"]] = "failed" if (output / "failure.json").exists() else "pending"
            continue
        receipt = series._read(receipt_path)
        if (receipt.get("validated") is not True or receipt.get("smoke") is not False
                or receipt.get("cell_id") != cell["cell_id"] or receipt.get("source_commit") != campaign["source_commit"]
                or any(receipt.get(key) != value for key, value in pins.items())):
            raise PublicationError("Only validated production results from the pinned source may publish")
        manifest, results, compact = output / "bundle/manifest.json", output / "results.json", output / "decision-summary.json"
        for key, path in (("manifest_sha256", manifest), ("results_sha256", results), ("decision_summary_sha256", compact)):
            if receipt.get(key) != series._file_digest(path):
                raise PublicationError("Validated result changed: " + key)
        compact_payload = series._read(compact)
        if compact_payload.get("mode") != cell["mode"] or compact_payload.get("checkpoint_id") != cell["checkpoint_id"]:
            raise PublicationError("Decision summary belongs to another mechanism or checkpoint")
        driver.validate_decisions(compact_payload, cell, campaign["seeds"], campaign["max_steps"])
        journal = series._read(Path(cell["run_dir"]) / "publication.json")
        old = list(journal["records"].values())
        if old:
            expected_files = {"manifest.json": series._file_digest(manifest), "results.json": series._file_digest(results),
                              "validation-receipt.json": series._file_digest(receipt_path),
                              "decision-summary.json": series._file_digest(compact)}
            if len(old) != 1 or any(old[0]["artifact_sha256"].get(key) != value for key, value in expected_files.items()):
                raise PublicationError("Previously accepted scientific result changed")
            statuses[cell["cell_id"]] = "complete"
            continue
        selected = [run for run in series._read(manifest)["runs"] if run["selector"] == cell["selector"]]
        records = [record for record in load_records(manifest, source_run=cell["source_run"], inventory_path=cell["inventory_path"])
                   if record["selector"] == cell["selector"]]
        if len(selected) != 1 or len(records) != 1 or selected[0].get("status") != "complete":
            raise PublicationError("Result must select exactly one complete scientific record")
        record = records[0]
        if (record["checkpoint"] != {"step": cell["step"], "sha256": cell["checkpoint_sha256"]}
                or sorted(ep["seed"] for ep in record["episodes"]) != campaign["seeds"]
                or record["metrics"].get("eval/frozen_state_unchanged") is not True):
            raise PublicationError("Checkpoint, seed coverage, or frozen-state validation changed")
        counts = {"actor": 32, "critic": 128, "temperature": 3200 if cell["mode"] == "repeat100" else 32}
        if any(record["metrics"].get("work/" + key + "_updates") != count * 2500 for key, count in counts.items()):
            raise PublicationError("Scientific record has the wrong optimizer update counts")
        remote = []
        for name in selected[0].get("trace_files", []):
            path = Path(record["artifact_files"].pop(name)).resolve(strict=True)
            if not path.is_relative_to((output / "bundle").resolve()):
                raise PublicationError("Trace path escapes the validated bundle")
            digest = series._file_digest(path)
            if receipt.get("trace_sha256", {}).get(name) != digest:
                raise PublicationError("Validated trace hash changed")
            remote.append(dict(name=name, path=str(path), sha256=digest, size_bytes=path.stat().st_size))
        index_path = Path(cell["run_dir"]) / "remote-traces.json"
        series._atomic_json(index_path, dict(policy="Full traces retained on Oscar", files=remote))
        record["artifact_files"].update({"results.json": str(results), "validation-receipt.json": str(receipt_path),
            "decision-summary.json": str(compact), "remote-traces.json": str(index_path)})
        if sum(Path(path).stat().st_size for path in record["artifact_files"].values()) > 16 * 1024 * 1024:
            raise PublicationError("Summary artifact exceeds 16 MiB")
        record["provenance"].update(remote_trace_files=remote, artifact_policy="compact summaries only")
        series.stage_record(cell["run_dir"], record)
        statuses[cell["cell_id"]] = "complete"
    return statuses


def prepare_displays(campaign, statuses):
    from slurm import ambi_alpha_adaptation_episodes as driver
    displays = []
    for row in display_rows(campaign):
        cell = row["cell"]
        baseline_path = Path(cell["baseline_decision_summary"])
        state = "pending" if row["mode"] == "reset" else statuses[cell["cell_id"]]
        summary = baseline = None
        if baseline_path.exists():
            baseline = driver.load_baseline_decision_summary(cell)
        if row["mode"] == "reset" and baseline is not None:
            summary, state = baseline, "complete"
        elif state == "complete":
            if baseline is None:
                raise PublicationError("Complete new result requires its verified reused baseline")
            summary = series._read(Path(cell["output_dir"]) / "decision-summary.json")
        display = {key: value for key, value in row.items() if key != "cell"}
        display.update(status=state, history=[], summary={}, provenance={
            "source_run": cell["source_run"], "checkpoint_sha256": cell["checkpoint_sha256"],
            "baseline_manifest_sha256": cell["baseline_manifest_sha256"],
            "reference_manifest_sha256": cell["reference_manifest_sha256"],
            "canonical_evaluation_run": None if row["mode"] == "reset" else cell["run_id"],
            "baseline_bundle": cell["baseline_bundle"]})
        if summary is not None:
            display["history"] = reduce_decisions(summary, campaign)
            display["summary"] = summarize_episodes(summary, baseline, _reference_episodes(cell), campaign)
            display["provenance"]["decision_summary_sha256"] = series._digest(summary)
        displays.append(display)
    return displays


def _wire_history(display):
    prefix = "decision/" + display["checkpoint_id"] + "/"
    result = []
    for row in display["history"]:
        wire = {AXIS: row[AXIS], **{prefix + key: value for key, value in row.items() if key != AXIS}}
        wire["derived/row_sha256"] = series._digest(wire)
        result.append(wire)
    return result


def _verify_prefix(actual, expected):
    """Require an exact prefix before resuming; never silently duplicate a row."""
    if len(actual) > len(expected):
        raise PublicationError("Derived history contains unexpected rows")
    for index, row in enumerate(actual):
        if row.get("_step") != index or any(row.get(key) != value for key, value in expected[index].items()):
            raise PublicationError("Derived history changed or is not a contiguous prefix")
    return len(actual)


def publish_display(display, root, wandb_module):
    directory = Path(root) / "displays" / display["run_id"]
    directory.mkdir(parents=True, exist_ok=True)
    journal_path = directory / "publication.json"
    old = series._read(journal_path) if journal_path.exists() else {}
    identity_hash = series._digest(display["identity"])
    payload_hash = series._digest(display)
    if old and old.get("identity_sha256") != identity_hash:
        raise PublicationError("Derived display identity changed")
    if old.get("status") == "complete":
        if old.get("payload_sha256") != payload_hash:
            raise PublicationError("Published derived result changed")
        return old
    expected = _wire_history(display)
    actual = []
    remote_exists = bool(old.get("initialized"))
    if remote_exists or old.get("init_started"):
        remote_path = f"{ENTITY}/{PROJECT}/{display['run_id']}"
        try:
            remote = wandb_module.Api(timeout=30).run(remote_path)
        except Exception as error:
            # In the pinned SDK, a successful lookup with a null run raises
            # ValueError, wrapped by CommError.exc. Network/auth failures must
            # never be interpreted as proof that an uncertain run is absent.
            cause = getattr(error, "exc", error)
            missing = (isinstance(cause, ValueError)
                       and str(cause).startswith("Could not find run <Run " + remote_path + " ("))
            if old.get("initialized") or not missing:
                raise
            remote_exists = False
        else:
            remote_exists = True
            if remote.config.get("derived_identity_sha256") != identity_hash:
                raise PublicationError("Remote derived run identity changed")
            actual = list(remote.scan_history(page_size=1000))
    prefix = _verify_prefix(actual, expected)
    journal = dict(identity_sha256=identity_hash, payload_sha256=payload_hash,
                   init_started=True, initialized=remote_exists, status="inflight")
    series._atomic_json(journal_path, journal)
    run = wandb_module.init(id=display["run_id"], entity=ENTITY, project=PROJECT, name=display["name"],
        job_type="derived-decision-curves", tags=[CAMPAIGN_ID, "derived-display", display["mode"]],
        resume="must" if remote_exists else "allow", dir=str(directory), reinit=True,
        config=dict(derived_identity_sha256=identity_hash, derived_identity=display["identity"],
                    derived_display=True, reused_result=display["reused_result"], provenance=display["provenance"]))
    try:
        if getattr(run, "disabled", False) or getattr(getattr(run, "settings", None), "mode", "online") != "online":
            raise PublicationError("Derived publication requires online W&B")
        journal["initialized"] = True
        series._atomic_json(journal_path, journal)
        if getattr(run, "step", prefix) != prefix:
            raise PublicationError("SDK history position differs from verified prefix; no rows resent")
        run.define_metric(AXIS)
        run.define_metric("decision/*", step_metric=AXIS, step_sync=False)
        for index in range(prefix, len(expected)):
            run.log(expected[index], step=index, commit=True)
        run.summary.update({"progress/status": display["status"], "progress/complete": int(display["status"] == "complete"),
            "progress/pending": int(display["status"] == "pending"), "progress/failed": int(display["status"] == "failed"),
            **{"result/" + display["checkpoint_id"] + "/" + key: value for key, value in display["summary"].items()}})
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise
    if expected:
        deadline = time.monotonic() + 30
        while True:
            actual = list(wandb_module.Api(timeout=30).run(f"{ENTITY}/{PROJECT}/{display['run_id']}").scan_history(page_size=1000))
            count = _verify_prefix(actual, expected)
            if count == len(expected):
                break
            if time.monotonic() >= deadline:
                raise PublicationError("W&B has not acknowledged derived rows; safe resume will reconcile")
            time.sleep(1)
    journal.update(status="complete" if display["status"] == "complete" else "pending", rows=len(expected), updated_at=now())
    series._atomic_json(journal_path, journal)
    return journal


def install_workspace(campaign, path, *, backend=None):
    """Create only the dedicated view; subsequent calls replace owned sections."""
    from slurm.ambi_entropy_kl_curves_layout import _color, _filter_model
    if backend is None:
        for package, version in (("wandb", "0.28.1"), ("wandb-workspaces", "0.4.5")):
            if importlib.metadata.version(package) != version:
                raise PublicationError(f"Layout requires isolated {package}=={version}")
        import wandb_workspaces.workspaces as ws
        import wandb_workspaces.reports.v2 as wr
        from wandb_workspaces import expr
    else:
        ws, wr, expr = backend
    rows, path = display_rows(campaign), Path(path)
    previous = series._read(path) if path.exists() else {}
    campaign_hash = series._digest(campaign)
    if previous and previous.get("campaign_sha256") != campaign_hash:
        raise PublicationError("Workspace receipt belongs to another campaign")
    markdown = """**Alpha adaptation: sampled-action frozen H3/J8 full episodes.**

Gray: reset alpha and Adam every solve (reused measured baseline). Orange: 100
temperature Adam steps per actor update, still resetting each solve. Blue:
carry alpha and Adam between decisions, resetting both at each episode start.
Actor/critic/replay remain fresh per decision; outer learning is frozen. Five
paired seeds, 500 decisions, no KL. Entropy is the actor-update batch average,
not final-root entropy. The target is plotted separately. Lines connect actual
per-decision means, without smoothing; sample SD is available in run history.

Each checkpoint has separate panels. Native legend names and colors identify
methods. These are explicitly derived display runs linked to immutable source
results; they are not additional evaluations. Missing results stay pending,
never zero. Episode return variability is across evaluation seeds, not training
seeds. Timing is measured controller time. Full traces remain on Oscar.
"""
    sections = [ws.Section(name=PREFIX + "Protocol and progress", is_open=True, panels=[
        wr.MarkdownPanel(markdown=markdown, layout=wr.Layout(w=24, h=10)),
        *[wr.ScalarChart(title=label, metric="progress/" + key, layout=wr.Layout(w=8, h=5))
          for label, key in (("Complete display curves", "complete"), ("Pending display curves", "pending"), ("Failed display curves", "failed"))]])]
    for checkpoint in CHECKPOINTS:
        chosen = [row for row in rows if row["checkpoint_id"] == checkpoint]
        panels = []
        for title, metric in CURVE_METRICS:
            key = "decision/" + checkpoint + "/" + metric
            panels.append(wr.LinePlot(title=title, x=AXIS, y=[key], aggregate=False,
                smoothing_type="none", legend_template="${run:name}", max_runs_to_show=6,
                title_x="Real environment decision within episode", layout=wr.Layout(w=12, h=8),
                line_colors={r["run_id"] + ":" + key: r["color"] for r in chosen},
                line_marks={r["run_id"] + ":" + key: "solid" for r in chosen},
                line_titles={r["run_id"] + ":" + key: r["name"] for r in chosen}))
        panels += [wr.ScalarChart(title=title, metric="result/" + checkpoint + "/" + key,
                                 layout=wr.Layout(w=8, h=6)) for title, key in SUMMARY_METRICS]
        sections.append(ws.Section(name=PREFIX + checkpoint, is_open=True, panels=panels))
    wanted_filter = expr.Metric("ID").isin([row["run_id"] for row in rows])
    if previous.get("url"):
        work = ws.Workspace.from_url(previous["url"])
        if work.entity != ENTITY or work.project != PROJECT:
            raise PublicationError("Dedicated workspace target changed")
        work.sections = [section for section in work.sections if not section.name.startswith(PREFIX)] + sections
    else:
        work = ws.Workspace(entity=ENTITY, project=PROJECT, name="Alpha adaptation sanity: reset, extra updates, carry",
                            sections=sections, auto_generate_panels=False)
    work.settings = ws.WorkspaceSettings(x_axis=AXIS, max_runs=6, tooltip_number_of_runs="all_runs",
                                         tooltip_color_run_names=True, remove_legends_from_panels=False)
    work.runset_settings = ws.RunsetSettings(filters=[wanted_filter], groupby=[],
        run_settings={row["run_id"]: ws.RunSettings(color=row["color"], disabled=False) for row in rows})
    work.save()
    receipt = dict(campaign_sha256=campaign_hash, url=work.url, api_verified=False,
                   display_runs=[{k: v for k, v in row.items() if k != "cell"} for row in rows])
    series._atomic_json(path, receipt)
    saved = ws.Workspace.from_url(work.url)
    owned = [section for section in saved.sections if section.name.startswith(PREFIX)]
    panels = [panel for section in owned for panel in section.panels]
    if (len(owned) != 3 or len(panels) != 4 + 2 * (len(CURVE_METRICS) + len(SUMMARY_METRICS))
            or saved.runset_settings.groupby or _filter_model(saved.runset_settings.filters, expr) != _filter_model(wanted_filter)
            or any(_color(saved.runset_settings.run_settings[row["run_id"]].color) != row["color"] for row in rows)):
        raise PublicationError("Saved workspace lost panels, filters or exact colors")
    for panel in (panel for panel in panels if isinstance(panel, wr.LinePlot)):
        if (panel.aggregate or getattr(panel.x, "name", panel.x) != AXIS or panel.smoothing_type != "none"
                or set(panel.line_marks.values()) != {"solid"} or panel.legend_template != "${run:name}"):
            raise PublicationError("Saved workspace lost independent connected native curves")
        metric = getattr(panel.y[0], "name", panel.y[0])
        chosen = [row for row in rows if metric.startswith("decision/" + row["checkpoint_id"] + "/")]
        for row in chosen:
            key = row["run_id"] + ":" + metric
            if (panel.line_titles.get(key) != row["name"]
                    or _color(panel.line_colors.get(key)) != row["color"]
                    or panel.line_marks.get(key) != "solid"):
                raise PublicationError("Saved native curve lost readable legend, color or solid line")
    receipt.update(api_verified=True, panel_count=len(panels), updated_at=now())
    series._atomic_json(path, receipt)
    return receipt


def verify_workspace(campaign, path, wandb_module=None):
    """Publication refuses a missing/unverified layout before any W&B mutation."""
    path = Path(path)
    if not path.exists():
        raise PublicationError("Install the dedicated native workspace before publishing")
    receipt = series._read(path)
    if not receipt.get("api_verified") or receipt.get("campaign_sha256") != series._digest(campaign):
        raise PublicationError("Native workspace receipt does not match campaign")
    # The pinned layout runtime verifies every panel on installation. Inspect the
    # remote view through the stable public API as well, in the CPU publisher.
    from urllib.parse import parse_qs, urlparse
    import importlib
    if wandb_module is None:
        import wandb as wandb_module
    view = parse_qs(urlparse(receipt["url"]).query).get("nw", [None])[0]
    if not view:
        raise PublicationError("Dedicated workspace ID is missing")
    query = 'query View($entityName:String,$name:String,$viewName:String){project(name:$name,entityName:$entityName){allViews(viewType:"project-view",viewName:$viewName){edges{node{id spec}}}}}'
    variables = dict(entityName=ENTITY, name=PROJECT, viewName="nw-" + view + "-v")
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
    wire = json.dumps(edges[0]["node"]["spec"]) if len(edges) == 1 else ""
    tokens = [AXIS, "progress/pending", *[row["run_id"] for row in display_rows(campaign)],
              *["decision/" + checkpoint + "/" + metric for checkpoint in CHECKPOINTS for _, metric in CURVE_METRICS]]
    if any(token not in wire for token in tokens):
        raise PublicationError("Dedicated workspace is missing required remote panels or curves")
    return receipt


def render_report(campaign, displays, path):
    payload = dict(campaign_id=campaign["campaign_id"], source_commit=campaign["source_commit"], displays=displays,
                   curve_metrics=CURVE_METRICS, summary_metrics=SUMMARY_METRICS)
    encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    html = _HTML.replace("__DATA__", encoded)
    Path(path).write_text(html)
    return payload


def publish_once(campaign, root, *, wandb_module=None, workspace_verifier=verify_workspace, require_complete=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if wandb_module is None:
        import wandb as wandb_module
    with series._lock(root / ".publisher.lock", blocking=False):
        workspace = workspace_verifier(campaign, root / "workspace.json", wandb_module)
        statuses = stage_completed(campaign)
        if require_complete:
            statuses = {key: "failed" if value == "pending" else value for key, value in statuses.items()}
        for cell in campaign["cells"]:
            with series.Publisher(cell["run_dir"], owner=OWNER, wandb_module=wandb_module) as publisher:
                publisher.publish_pending()
                publisher.run.summary.update({"progress/status": statuses[cell["cell_id"]],
                    "progress/checkpoints_complete": int(statuses[cell["cell_id"]] == "complete"),
                    "progress/checkpoints_pending": int(statuses[cell["cell_id"]] == "pending"),
                    "progress/failed_cells": int(statuses[cell["cell_id"]] == "failed")})
        displays = prepare_displays(campaign, statuses)
        render_report(campaign, displays, root / "report.html")
        for display in displays:
            publish_display(display, root, wandb_module)
        result = dict(campaign_id=campaign["campaign_id"], cells=statuses, updated_at=now(), workspace_url=workspace["url"],
                      all_complete=all(value == "complete" for value in statuses.values()))
        series._atomic_json(root / "status.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("layout", "publish", "report"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--once", action="store_true", help="Compatibility flag; each invocation publishes once")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    campaign = load_campaign(args.campaign)
    args.root.mkdir(parents=True, exist_ok=True)
    try:
        if args.command == "layout":
            result = install_workspace(campaign, args.root / "workspace.json")
        elif args.command == "report":
            statuses = stage_completed(campaign)
            render_report(campaign, prepare_displays(campaign, statuses), args.root / "report.html")
            result = {"report": str(args.root / "report.html"), "cells": statuses}
        else:
            result = publish_once(campaign, args.root, require_complete=args.require_complete)
            if args.require_complete and not result["all_complete"]:
                raise PublicationError("Final publication incomplete; missing results marked failed")
    except Exception as error:
        message = str(error) if isinstance(error, PublicationError) else "Inspect campaign receipts and publication journals"
        series._atomic_json(args.root / "publication-failure.json", dict(error_type=type(error).__name__, message=message, at=now()))
        raise SystemExit(f"Publication failed ({type(error).__name__}): {message}") from None
    print(series._json(result))


_HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Alpha adaptation sanity</title><style>body{font:15px system-ui;margin:30px auto;max-width:1250px;padding:0 24px;background:#f6f8fb;color:#183044}section{background:white;padding:20px;margin:20px 0;border:1px solid #dde4ec;border-radius:10px}select{padding:8px}svg{width:100%;height:360px}button{background:white;border:1px solid #ddd;border-radius:5px;margin:4px;padding:8px;cursor:pointer}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:8px;border-bottom:1px solid #eee}.muted{color:#607386}#tooltip{min-height:25px}</style><h1>Alpha adaptation sanity check</h1><p>Frozen H3/J8 · sampled execution · five paired seeds · 500 decisions. Baselines are reused, not reevaluated. Curves connect measured decision means without smoothing. Entropy is measured on actor-update batches.</p><section><label>Checkpoint <select id="checkpoint"></select></label> <label>Metric <select id="metric"></select></label><div id="legend"></div><svg id="plot" viewBox="0 0 1180 360" aria-label="Connected decision curves"></svg><div id="tooltip"></div><p class="muted">Hover curves to highlight a method; click legend entries to hide or show. Hover labels show between-seed sample SD. Missing lifecycle diagnostics in historical baselines remain absent.</p></section><section><h2>Full-episode results and progress</h2><div id="table"></div></section><section><h2>Provenance</h2><pre id="provenance" style="overflow:auto;max-height:300px"></pre></section><script id="alpha-data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('alpha-data').textContent),$=id=>document.getElementById(id),ns='http://www.w3.org/2000/svg',hidden=new Set();function option(p,v,t){const e=document.createElement('option');e.value=v;e.textContent=t;p.append(e)}Array.from(new Set(data.displays.map(r=>r.checkpoint_id))).forEach(v=>option($('checkpoint'),v,v));data.curve_metrics.forEach(([t,k])=>option($('metric'),k,t));function add(tag,attrs){const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));$('plot').append(e);return e}function draw(){const checkpoint=$('checkpoint').value,metric=$('metric').value,rows=data.displays.filter(r=>r.checkpoint_id===checkpoint);$('plot').replaceChildren();$('legend').replaceChildren();$('tooltip').textContent='';rows.forEach(r=>{const b=document.createElement('button');b.textContent=r.name.split(' | ')[1]+' · '+r.status;b.style.color=r.color;b.style.opacity=hidden.has(r.run_id)?.4:1;b.onclick=()=>{hidden.has(r.run_id)?hidden.delete(r.run_id):hidden.add(r.run_id);draw()};$('legend').append(b)});const available=rows.filter(r=>!hidden.has(r.run_id)),values=available.flatMap(r=>r.history.filter(p=>Number.isFinite(p[metric])).map(p=>p[metric]));if(!values.length){add('text',{x:90,y:160,fill:'#607386'}).textContent='Pending measured results';return}let lo=Math.min(...values),hi=Math.max(...values),pad=Math.max((hi-lo)*.1,.01);lo-=pad;hi+=pad;const x=d=>70+(d-1)/499*1060,y=v=>305-(v-lo)/(hi-lo)*265;for(let i=0;i<=5;i++){const v=lo+(hi-lo)*i/5;add('line',{x1:70,x2:1130,y1:y(v),y2:y(v),stroke:'#e4eaf0'});add('text',{x:4,y:y(v)+4,'font-size':12}).textContent=v.toPrecision(4)}[1,100,200,300,400,500].forEach(d=>add('text',{x:x(d),y:330,'text-anchor':'middle','font-size':12}).textContent=d);available.forEach(r=>{const pts=r.history.filter(p=>Number.isFinite(p[metric]));if(!pts.length)return;const line=add('polyline',{points:pts.map(p=>`${x(p['episode/decision'])},${y(p[metric])}`).join(' '),fill:'none',stroke:r.color,'stroke-width':2.5,'data-run':r.run_id});line.onmouseenter=()=>{$('plot').querySelectorAll('polyline').forEach(e=>e.style.opacity=e===line?1:.18);line.setAttribute('stroke-width',4)};line.onmouseleave=()=>{$('plot').querySelectorAll('polyline').forEach(e=>e.style.opacity=1);line.setAttribute('stroke-width',2.5)};line.onmousemove=e=>{const point=$('plot').createSVGPoint();point.x=e.clientX;point.y=e.clientY;const local=point.matrixTransform($('plot').getScreenCTM().inverse()),d=Math.max(1,Math.min(500,Math.round(1+(local.x-70)/1060*499))),p=pts.reduce((a,b)=>Math.abs(a['episode/decision']-d)<Math.abs(b['episode/decision']-d)?a:b);$('tooltip').textContent=`${r.name} · decision ${p['episode/decision']}: ${p[metric].toPrecision(5)} ± ${p[metric+'_sample_std']?.toPrecision(4)??'—'} sample SD`}})}['checkpoint','metric'].forEach(id=>$(id).onchange=draw);const t=document.createElement('table'),h=t.insertRow();['Checkpoint','Method','Status',...data.summary_metrics.map(m=>m[0])].forEach(v=>{const th=document.createElement('th');th.textContent=v;h.append(th)});data.displays.forEach(r=>{const tr=t.insertRow();[r.checkpoint_id,r.name.split(' | ')[1],r.status,...data.summary_metrics.map(([_,k])=>Number.isFinite(r.summary[k])?r.summary[k].toFixed(4):'—')].forEach(v=>tr.insertCell().textContent=v);tr.cells[1].style.color=r.color});$('table').append(t);$('provenance').textContent=JSON.stringify({campaign:data.campaign_id,source_commit:data.source_commit,displays:data.displays.map(({history,summary,...rest})=>rest)},null,2);draw();</script></html>'''


if __name__ == "__main__":
    main()
