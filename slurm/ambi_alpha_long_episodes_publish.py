"""Ten sampled episodes: newly measured temperature reset and carry histories.

Five paired histories are the sampling units; episode 10 is primary. Historical
three-episode campaigns supply provenance only, never invented ten-point controls.
"""
from __future__ import annotations
import argparse
import importlib
import importlib.metadata
import json
import math
import statistics
from pathlib import Path
import time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import eval_series as series
from slurm.ambi_alpha_carry_chain_publish import (
    PublicationError, _finite, _mean_sd, now, hardware_provenance, history_rows,
    AXIS, EPISODE_AXIS, ENTITY, PROJECT,
)
from slurm.ambi_alpha_adaptation_publish import _verify_prefix

CHECKPOINTS = {"mppi_aux_500k": (500000, "MPPI-trained")}
PREFIX = "Alpha ten sampled episodes MPPI500k / "
EPISODES, STEPS, HISTORIES = 10, 500, 5
METHODS = {
    "episode_reset": ("Reset alpha + Adam between episodes · sampled actions", "#0F766E"),
    "run_carry": ("Carry alpha + Adam across ten episodes · sampled actions", "#2563EB"),
}
DECISION_METRICS = [
    ("Update-batch entropy", "entropy"),
    ("Update-batch entropy minus target", "entropy_error"),
    ("Alpha at solve start", "alpha_initial"),
    ("Alpha at solve end", "alpha_final"),
    ("Log alpha at solve end", "log_alpha_final"),
    ("Temperature Adam step at solve end", "optimizer_step_final"),
    ("Alpha discontinuity at episode boundary", "boundary_alpha_jump"),
    ("Reward accumulated within current episode", "episode_cumulative_reward"),
    ("Reward accumulated across ten-episode history", "chain_cumulative_reward"),
]
EPISODE_METRICS = [
    ("Episode return · episode 10 is primary", "return_mean"),
    ("Return sample SD across five histories", "return_sample_std"),
    ("Carry minus episode-reset return · positive means carry helps", "carry_reset_gain_mean"),
    ("Paired carry-minus-reset sample SD", "carry_reset_gain_sample_std"),
    ("Fraction of paired histories where carry helps", "carry_reset_win_fraction"),
    ("Worst paired carry-minus-reset return", "carry_reset_worst_gain"),
    ("Episode mean update-batch entropy", "entropy_mean"),
    ("Update-batch entropy error RMSE across decisions", "entropy_error_rmse"),
    ("First 100 decisions · update-batch entropy", "early100_entropy_mean"),
    ("Last 100 decisions · update-batch entropy", "late100_entropy_mean"),
    ("First 100 decisions · accumulated real reward", "early100_return_mean"),
    ("Last 100 decisions · accumulated real reward", "late100_return_mean"),
    ("Alpha at episode end · mean across five histories", "alpha_final_mean"),
    ("Controller seconds per decision · consult hardware", "control_seconds_per_decision"),
]


def stream_seeds(stream):
    return [base + ((stream + episode) % 5) for base in (101, 106) for episode in range(5)]


def validate_summary(summary):
    expected = {(stream, episode, decision) for stream in range(HISTORIES)
                for episode in range(EPISODES) for decision in range(STEPS)}
    rows = summary.get("rows", [])
    coordinates = [(row.get("stream_index"), row.get("episode_index"), row.get("decision_index")) for row in rows]
    if len(coordinates) != len(set(coordinates)) or set(coordinates) != expected:
        raise PublicationError("Exactly five measured ten-episode histories are required; no duplicate or missing decisions")
    _episode_index(summary)
    for row in rows:
        stream, episode, decision = row["stream_index"], row["episode_index"], row["decision_index"]
        if row.get("seed") != stream_seeds(stream)[episode] or row.get("global_decision_index") != episode * STEPS + decision:
            raise PublicationError("Ten-episode seed order or global decision index changed")
        for key in ("entropy", "target_entropy", "alpha_initial", "alpha_final", "reward", "control_seconds",
                    "optimizer_step_initial", "optimizer_step_final"):
            if not _finite(row.get(key)):
                raise PublicationError("Incomplete decision metric: " + key)
        if min(row["alpha_initial"], row["alpha_final"]) <= 0:
            raise PublicationError("Learned temperature must remain positive")
    return summary


def merge_streams(streams):
    if len(streams) != HISTORIES:
        raise PublicationError("Exactly five newly measured histories are required")
    summary = {"rows": [row for stream in streams for row in stream["rows"]],
               "episodes": [episode for stream in streams for episode in stream["episodes"]]}
    return validate_summary(summary)


def _episode_index(summary):
    values = summary["episodes"]
    indexed = {(ep["stream_index"], ep["episode_index"]): ep for ep in values}
    wanted = {(stream, episode) for stream in range(HISTORIES) for episode in range(EPISODES)}
    if len(values) != HISTORIES * EPISODES or set(indexed) != wanted:
        raise PublicationError("Each arm needs exactly five measured histories of ten episodes")
    for (stream, episode), row in indexed.items():
        if (row.get("seed") != stream_seeds(stream)[episode] or row.get("length") != STEPS
                or not _finite(row.get("return")) or not isinstance(row.get("solver_seed"), int)):
            raise PublicationError("Episode identity or result is incomplete")
    return indexed


def _effect_statistics(gains):
    mean, sd = _mean_sd(gains)
    return dict(carry_reset_gain_mean=mean, carry_reset_gain_sample_std=sd,
        carry_reset_wins=sum(gain > 0 for gain in gains), carry_reset_losses=sum(gain < 0 for gain in gains),
        carry_reset_ties=sum(gain == 0 for gain in gains), carry_reset_win_fraction=sum(gain > 0 for gain in gains)/HISTORIES,
        carry_reset_worst_gain=min(gains), carry_reset_best_gain=max(gains))


def reduce_decisions(summary):
    validate_summary(summary)
    groups = {decision: [] for decision in range(EPISODES * STEPS)}
    accumulated, previous, episode_totals = {stream: 0. for stream in range(HISTORIES)}, {}, {}
    for original in sorted(summary["rows"], key=lambda row: (row["stream_index"], row["global_decision_index"])):
        row = dict(original)
        stream, episode, decision = row["stream_index"], row["episode_index"], row["decision_index"]
        accumulated[stream] += row["reward"]
        episode_totals[stream, episode] = episode_totals.get((stream, episode), 0.) + row["reward"]
        row.update(episode_cumulative_reward=episode_totals[stream, episode], chain_cumulative_reward=accumulated[stream],
                   entropy_error=row["entropy"]-row["target_entropy"], log_alpha_final=math.log(row["alpha_final"]))
        row["boundary_alpha_jump"] = row["alpha_initial"] - previous[stream] if decision == 0 and episode > 0 else None
        previous[stream] = row["alpha_final"]
        groups[row["global_decision_index"]].append(row)
    for episode in summary["episodes"]:
        actual = episode_totals[episode["stream_index"], episode["episode_index"]]
        if not math.isclose(actual, episode["return"], rel_tol=1e-8, abs_tol=1e-7):
            raise PublicationError("Decision rewards do not sum to the saved episode return")
    points = []
    for decision, rows in groups.items():
        point = {AXIS: decision + 1, EPISODE_AXIS: decision // STEPS + 1, "streams": HISTORIES}
        for _, metric in DECISION_METRICS:
            values = [row.get(metric) for row in rows]
            if all(value is None for value in values):
                continue
            if any(value is None for value in values):
                raise PublicationError("Partial history coverage for " + metric)
            point[metric], point[metric + "_sample_std"] = _mean_sd(values)
        points.append(point)
    return points


def reduce_episodes(summary, method, summaries):
    indexed = _episode_index(summary)
    other = _episode_index(summaries["episode_reset"]) if method == "run_carry" else None
    points, comparisons, stream_effects = [], [], {stream: [] for stream in range(HISTORIES)}
    for episode in range(EPISODES):
        selected = [indexed[stream, episode] for stream in range(HISTORIES)]
        mean, sd = _mean_sd([row["return"] for row in selected])
        point = {EPISODE_AXIS: episode + 1, "streams": HISTORIES, "return_mean": mean, "return_sample_std": sd}
        times = [row.get("control_seconds") for row in selected]
        if all(_finite(value) for value in times):
            point["control_seconds_per_decision"] = statistics.mean(times) / STEPS
        decisions = [row for row in summary["rows"] if row["episode_index"] == episode]
        point["entropy_mean"] = statistics.mean(row["entropy"] for row in decisions)
        point["entropy_error_rmse"] = math.sqrt(statistics.mean((row["entropy"] - row["target_entropy"])**2 for row in decisions))
        point["alpha_final_mean"] = statistics.mean(row["alpha_final"] for row in decisions if row["decision_index"] == STEPS-1)
        for label, start, stop in (("early100", 0, 100), ("late100", 400, 500)):
            windows = [[row for row in decisions if row["stream_index"] == stream
                        and start <= row["decision_index"] < stop] for stream in range(HISTORIES)]
            if any(len(window) != 100 for window in windows):
                raise PublicationError("Entropy windows require 100 decisions from all five histories")
            values = dict(entropy=[statistics.mean(row["entropy"] for row in window) for window in windows],
                entropy_error=[statistics.mean(row["entropy"] - row["target_entropy"] for row in window) for window in windows],
                **{"return": [sum(row["reward"] for row in window) for window in windows]})
            for metric, samples in values.items():
                point[label + "_" + metric + "_mean"], point[label + "_" + metric + "_sample_std"] = _mean_sd(samples)
        if method == "run_carry":
            pairs = []
            for stream, carry in enumerate(selected):
                reset = other[stream, episode]
                if (carry["seed"], carry["solver_seed"]) != (reset["seed"], reset["solver_seed"]):
                    raise PublicationError("Carry/reset pairing requires matching environment and controller seeds")
                delta = carry["return"] - reset["return"]
                stream_effects[stream].append(delta)
                pairs.append(dict(stream_index=stream, episode_index=episode, seed=carry["seed"], solver_seed=carry["solver_seed"],
                    returns=dict(run_carry=carry["return"], episode_reset=reset["return"]), difference=delta))
            point.update(_effect_statistics([row["difference"] for row in pairs]))
            comparisons.append(dict(episode=episode + 1, comparison="carry minus episode reset", pairs=pairs))
        points.append(point)
    stream_means = [statistics.mean(indexed[stream, episode]["return"] for episode in range(EPISODES)) for stream in range(HISTORIES)]
    mean, sd = _mean_sd(stream_means)
    overall = dict(streams=HISTORIES, episodes_per_stream=EPISODES, primary_episode=EPISODES,
        stream_mean_return=mean, stream_mean_return_sample_std=sd,
        statistical_unit="five paired initialized histories; fifty episodes are not fifty independent trials",
        interpretation="descriptive matched carry/reset comparison; no prespecified significance or noninferiority threshold",
        first_episode_interpretation="negative control: both scopes start from the same alpha and Adam state; numerical/backend variation can amplify through closed-loop execution",
        **{"primary_" + key: value for key, value in points[-1].items() if key != EPISODE_AXIS})
    if method == "run_carry":
        contrasts = [statistics.mean(stream_effects[stream][5:]) - statistics.mean(stream_effects[stream][:5])
                     for stream in range(HISTORIES)]
        overall["last5_minus_first5_carry_effect_mean"], overall["last5_minus_first5_carry_effect_sample_std"] = _mean_sd(contrasts)
        overall["last5_minus_first5_carry_effect_pairs"] = [dict(stream_index=stream,
            first5_mean=statistics.mean(stream_effects[stream][:5]), last5_mean=statistics.mean(stream_effects[stream][5:]),
            difference=contrasts[stream]) for stream in range(HISTORIES)]
        overall["window_contrast_interpretation"] = "descriptive change in paired carry effect; positions 1-5 and 6-10 use different seed blocks, so it is not an isolated causal effect of adaptation time"
    return points, overall, comparisons


def build_payload(checkpoint_id, method, summary, summaries, provenance):
    episode_points, overall, comparisons = reduce_episodes(summary, method, summaries)
    return dict(checkpoint_id=checkpoint_id, method=method, provenance=provenance,
        decision_points=reduce_decisions(summary), episode_points=episode_points, overall=overall,
        paired_comparisons=comparisons,
        diagnostic_availability=dict(final_root_entropy="not recorded by this protocol",
            entropy_definition="pre-update actor minibatch entropy averaged across inner updates; not executed-policy root entropy"))


def load_campaign(path):
    from slurm import ambi_alpha_long_episodes as driver
    campaign = series._read(path)
    driver.validate_campaign(campaign)
    return campaign


def display_rows(campaign):
    rows = []
    for checkpoint in CHECKPOINTS:
        for method, (label, color) in METHODS.items():
            scope = "run" if method == "run_carry" else "episode"
            identity = dict(kind="alpha-long-episodes-publication-v1", campaign_sha256=series._digest(campaign),
                checkpoint_id=checkpoint, method=method, protocol="alpha-long-episodes-v1", reference_reuse=False,
                histories=HISTORIES, episodes_per_history=EPISODES, primary_episode=EPISODES, execution_action="sample",
                method_temperature_scope=scope, method_temperature_optimizer_scope=scope,
                seed_orders=[stream_seeds(stream) for stream in range(HISTORIES)],
                aggregation="five paired histories at each of ten episode positions; historical results are provenance only")
            rows.append(dict(run_id=series._digest(identity)[:32], checkpoint_id=checkpoint, method=method,
                name=f"{CHECKPOINTS[checkpoint][1]} {CHECKPOINTS[checkpoint][0]//1000}k | {label}",
                color=color, identity=identity))
    return rows


def collect(campaign, *, require_complete=False):
    from slurm import ambi_alpha_long_episodes as driver
    snapshot = driver.publication_snapshot(campaign)
    entries = []
    for checkpoint_id in CHECKPOINTS:
        checkpoint = snapshot["checkpoints"][checkpoint_id]
        streams = checkpoint["streams"]
        ready = all(len(streams.get(method, [])) == HISTORIES for method in METHODS)
        summaries = {method: merge_streams(streams[method]) for method in METHODS} if ready else {}
        for display in (row for row in display_rows(campaign) if row["checkpoint_id"] == checkpoint_id):
            method = display["method"]
            own_status = [row for row in checkpoint.get("status", []) if row.get("arm", row.get("mode")) == method]
            complete = len(streams.get(method, []))
            failed = sum(row["state"] in {"failed", "validation_failed"} for row in own_status)
            if not 0 <= complete <= HISTORIES or not 0 <= failed <= HISTORIES - complete:
                raise PublicationError("Invalid long-history completion counts")
            pending = HISTORIES - complete - failed
            if require_complete:
                failed += pending
                pending = 0
            status = "complete" if ready else "failed" if failed or require_complete else "pending"
            provenance = dict(checkpoint=campaign.get("checkpoints", {}).get(checkpoint_id, {}),
                source_commit=campaign["source_commit"], science_id=checkpoint.get("science_id"),
                source_receipts=checkpoint.get("status", []),
                publication_policy="both arms are newly measured; paired curves publish only after all ten cells validate; no historical outcomes repeated")
            if ready:
                filtered = {**campaign, "cells": [cell for cell in campaign.get("cells", []) if cell.get("arm", cell.get("mode")) == method]}
                provenance["production_hardware"] = hardware_provenance(filtered, checkpoint_id)
            summary = summaries.get(method)
            payload = build_payload(checkpoint_id, method, summary, summaries, provenance) if summary else None
            entries.append({**display, "status": status, "payload": payload,
                "progress": dict(complete=complete, pending=pending, failed=failed, expected=HISTORIES,
                                 paired_comparison_ready=int(ready))})
    return entries


def plotted_rows(rows, metric):
    return [row for row in rows if row["method"] == "run_carry"] if metric.startswith("carry_reset_") else rows


def publish_entry(entry, root, wandb_module):
    """Resume only a verified exact prefix; uncertain writes never duplicate rows."""
    directory = Path(root) / "runs" / entry["run_id"]
    directory.mkdir(parents=True, exist_ok=True)
    journal_path = directory / "publication.json"
    previous = series._read(journal_path) if journal_path.exists() else {}
    identity_hash, payload_hash = series._digest(entry["identity"]), series._digest(entry["payload"])
    if previous and previous.get("identity_sha256") != identity_hash:
        raise PublicationError("Chain publication identity changed")
    if previous.get("status") == "complete":
        if previous.get("payload_sha256") != payload_hash:
            raise PublicationError("Completed chain result changed")
        return previous
    expected, actual = history_rows(entry), []
    remote_exists = bool(previous.get("initialized"))
    remote_path = f"{ENTITY}/{PROJECT}/{entry['run_id']}"
    if remote_exists or previous.get("init_started"):
        try:
            remote = wandb_module.Api(timeout=30).run(remote_path)
        except Exception as error:
            cause = getattr(error, "exc", error)
            missing = isinstance(cause, ValueError) and str(cause).startswith("Could not find run <Run " + remote_path + " (")
            if previous.get("initialized") or not missing:
                raise
            remote_exists = False
        else:
            remote_exists = True
            if remote.config.get("chain_identity_sha256") != identity_hash:
                raise PublicationError("Remote chain identity changed")
            actual = list(remote.scan_history(page_size=2000))
    prefix = _verify_prefix(actual, expected)
    journal = dict(identity_sha256=identity_hash, payload_sha256=payload_hash,
                   initialized=remote_exists, init_started=True, status="inflight")
    series._atomic_json(journal_path, journal)
    if entry["payload"] is not None:
        result_path = directory / "result.json"
        if result_path.exists() and series._digest(series._read(result_path)) != payload_hash:
            raise PublicationError("Saved chain publication payload changed")
        if not result_path.exists():
            series._atomic_json(result_path, entry["payload"])
        if result_path.stat().st_size > 16 * 1024 * 1024:
            raise PublicationError("Compact publication exceeds 16 MiB")
    run = wandb_module.init(entity=ENTITY, project=PROJECT, id=entry["run_id"], name=entry["name"],
        job_type="temperature-long-episodes-evaluation",
        resume="must" if remote_exists else "allow", reinit=True, dir=str(directory),
        tags=["alpha-long-episodes", entry["method"]], config=dict(chain_identity_sha256=identity_hash,
            chain_identity=entry["identity"], reused_reference=False,
            statistical_unit="initialized temperature history; balanced cyclic seed orders"))
    try:
        if getattr(run, "disabled", False) or getattr(getattr(run, "settings", None), "mode", "online") != "online":
            raise PublicationError("Chain publication requires online W&B")
        journal["initialized"] = True
        series._atomic_json(journal_path, journal)
        if getattr(run, "step", prefix) != prefix:
            raise PublicationError("SDK step differs from verified chain prefix")
        run.define_metric(AXIS)
        run.define_metric(EPISODE_AXIS)
        run.define_metric("decision/*", step_metric=AXIS, step_sync=False)
        run.define_metric("episode/*", step_metric=EPISODE_AXIS, step_sync=False)
        for index in range(prefix, len(expected)):
            run.log(expected[index], step=index, commit=True)
        run.summary.update({"publication/status": entry["status"], "publication/payload_sha256": payload_hash,
            **{"progress/histories_" + key: value for key, value in entry["progress"].items()}})
        if entry["payload"]:
            run.summary.update({"aggregate/" + key: value for key, value in entry["payload"]["overall"].items()})
            run.summary["publication/provenance"] = entry["payload"]["provenance"]
            run.summary["comparison/paired_episode_results"] = entry["payload"]["paired_comparisons"]
            run.summary["diagnostic/availability"] = entry["payload"]["diagnostic_availability"]
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise
    if expected:
        deadline = time.monotonic() + 30
        while True:
            remote = wandb_module.Api(timeout=30).run(remote_path)
            count = _verify_prefix(list(remote.scan_history(page_size=2000)), expected)
            if count == len(expected):
                break
            if time.monotonic() >= deadline:
                raise PublicationError("W&B has not acknowledged chain rows; retry reconciles their exact prefix")
            time.sleep(1)
    journal.update(status="complete" if entry["status"] == "complete" else "pending", rows=len(expected), updated_at=now())
    series._atomic_json(journal_path, journal)
    return journal


def install_workspace(campaign, path, *, backend=None):
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
    old = series._read(path) if path.exists() else {}
    if old and old.get("campaign_sha256") != series._digest(campaign):
        raise PublicationError("Workspace belongs to another chain campaign")
    markdown = """**MPPI500k: ten sampled episodes with matched temperature carry/reset controls. Episode 10 is primary.**

Both actual methods are NEW: teal resets alpha and Adam between episodes; blue
carries both across all ten episodes. H3/J8, sampled real actions, alpha LR
0.0003, no KL, frozen outer checkpoint. Actor, critic and replay reset every real
decision. Historical three-episode outcomes are provenance only; neither curve
repeats or relabels old measurements as ten new episode positions.

Five paired histories per arm each contain ten 500-decision episodes. Episode
positions 1–5 balance seeds 101–105; positions 6–10 balance seeds 106–110. Carry
and reset use the same ordered environment/controller seeds within each pair.
Mean and sample SD use five paired histories at each position, not fifty
independent episodes. Primary outcome is episode-10 carry-minus-reset return;
positive means carry helps. All per-history paired returns, wins/losses/ties,
and worst differences are retained. The last-five minus first-five change in
carry effect is descriptive: time and environment seed blocks both change.
No formal significance or noninferiority threshold is assumed. Episode 1 is a
negative control because both temperature lifetimes start identically; small
backend/numerical differences can still amplify through closed-loop execution.

The complete trajectory has 5,000 decisions per history. Episode boundaries
occur every 500 decisions; episode reward resets while history reward does not.
Entropy is the actor-update minibatch statistic, not final-root executed-policy
entropy; target is −10.5. First/last 100-decision summaries separate transients
from late-episode entropy and accumulated real reward. Root entropy is unavailable.
Both curves await all ten validated cells to keep paired native histories immutable;
per-arm pending/failure counts remain explicit. Distinct colors, solid connected
curves, native hover and no smoothing. Compact aggregates only; full traces stay
on Oscar. Both methods in each pair share a GPU allocation; inspect timing provenance.
"""
    sections = [ws.Section(name=PREFIX + "Protocol and progress", is_open=True, panels=[
        wr.MarkdownPanel(markdown=markdown, layout=wr.Layout(w=24, h=12)),
        *[wr.ScalarChart(title=title, metric="progress/histories_" + key, layout=wr.Layout(w=8, h=5))
          for title, key in (("Validated histories per checkpoint", "complete"),
                             ("Pending histories per checkpoint", "pending"), ("Failed histories per checkpoint", "failed"))]])]
    for checkpoint in CHECKPOINTS:
        chosen = [row for row in rows if row["checkpoint_id"] == checkpoint]
        def plot(title, metric, axis, prefix):
            key = prefix + "/" + checkpoint + "/" + metric
            plotted = plotted_rows(chosen, metric)
            return wr.LinePlot(title=title, x=axis, y=[key], aggregate=False, smoothing_type="none",
                legend_template="${run:name}", max_runs_to_show=8, layout=wr.Layout(w=12, h=8),
                title_x="Episode position" if prefix == "episode" else "Global decision (new episode every 500 decisions)",
                line_colors={row["run_id"] + ":" + key: row["color"] for row in plotted},
                line_titles={row["run_id"] + ":" + key: row["name"] for row in plotted},
                line_marks={row["run_id"] + ":" + key: "solid" for row in plotted})
        sections.append(ws.Section(name=PREFIX + checkpoint + " episode outcomes", is_open=True,
            panels=[plot(title, metric, EPISODE_AXIS, "episode") for title, metric in EPISODE_METRICS]))
        sections.append(ws.Section(name=PREFIX + checkpoint + " temperature dynamics", is_open=True,
            panels=[plot(title, metric, AXIS, "decision") for title, metric in DECISION_METRICS]))
    if old.get("url"):
        work = ws.Workspace.from_url(old["url"])
        if work.entity != ENTITY or work.project != PROJECT:
            raise PublicationError("Workspace target changed")
        work.sections = [section for section in work.sections if not section.name.startswith(PREFIX)] + sections
    else:
        work = ws.Workspace(entity=ENTITY, project=PROJECT, name="MPPI500k alpha carry/reset · ten sampled episodes", sections=sections,
                            auto_generate_panels=False)
    wanted = expr.Metric("ID").isin([row["run_id"] for row in rows])
    work.settings = ws.WorkspaceSettings(x_axis=AXIS, max_runs=8, tooltip_number_of_runs="all_runs",
        tooltip_color_run_names=True, remove_legends_from_panels=False)
    work.runset_settings = ws.RunsetSettings(filters=[wanted], groupby=[],
        run_settings={row["run_id"]: ws.RunSettings(color=row["color"], disabled=False) for row in rows})
    work.save()
    receipt = dict(campaign_sha256=series._digest(campaign), url=work.url, api_verified=False, runs=rows)
    series._atomic_json(path, receipt)
    saved = ws.Workspace.from_url(work.url)
    owned = [section for section in saved.sections if section.name.startswith(PREFIX)]
    panels = [panel for section in owned for panel in section.panels]
    if (saved.entity != ENTITY or saved.project != PROJECT or saved.url != work.url
            or len(owned) != 3 or len(panels) != 4 + len(EPISODE_METRICS) + len(DECISION_METRICS)
            or saved.runset_settings.groupby or _filter_model(saved.runset_settings.filters, expr) != _filter_model(wanted)):
        raise PublicationError("Saved chain workspace lost its exact target, panels or run filter")
    for row in rows:
        settings = saved.runset_settings.run_settings[row["run_id"]]
        if settings.disabled or _color(settings.color) != row["color"]:
            raise PublicationError("Saved chain run visibility or color changed")
    actual_metrics = set()
    for panel in (panel for panel in panels if isinstance(panel, wr.LinePlot)):
        metric = getattr(panel.y[0], "name", panel.y[0])
        axis = EPISODE_AXIS if metric.startswith("episode/") else AXIS
        if (panel.aggregate or panel.smoothing_type != "none" or panel.legend_template != "${run:name}"
                or getattr(panel.x, "name", panel.x) != axis or metric in actual_metrics):
            raise PublicationError("Saved chain panel lost its independent connected curves")
        actual_metrics.add(metric)
        checkpoint = metric.split("/")[1]
        for row in rows:
            if row["checkpoint_id"] != checkpoint or row not in plotted_rows(rows, metric.split("/")[-1]):
                continue
            key = row["run_id"] + ":" + metric
            if (panel.line_titles.get(key) != row["name"] or _color(panel.line_colors.get(key)) != row["color"]
                    or panel.line_marks.get(key) != "solid"):
                raise PublicationError("Saved chain curve lost a name, color or solid line")
    expected_metrics = {prefix + "/" + checkpoint + "/" + metric for checkpoint in CHECKPOINTS
                        for prefix, definitions in (("episode", EPISODE_METRICS), ("decision", DECISION_METRICS))
                        for _, metric in definitions}
    if actual_metrics != expected_metrics:
        raise PublicationError("Saved chain workspace metric set changed")
    receipt.update(api_verified=True, panel_count=len(panels), updated_at=now())
    series._atomic_json(path, receipt)
    return receipt


def verify_workspace(campaign, path, wandb_module):
    from urllib.parse import parse_qs, urlparse
    path = Path(path)
    if not path.exists():
        raise PublicationError("Install and verify the dedicated chain workspace before publication")
    receipt = series._read(path)
    if not receipt.get("api_verified") or receipt.get("campaign_sha256") != series._digest(campaign):
        raise PublicationError("Dedicated chain workspace receipt changed")
    parsed = urlparse(receipt["url"])
    if parsed.hostname != "wandb.ai" or parsed.path.rstrip("/") != f"/{ENTITY}/{PROJECT}":
        raise PublicationError("Workspace URL is outside the intended project")
    view = parse_qs(parsed.query).get("nw", [None])[0]
    if not view:
        raise PublicationError("Named chain workspace ID is missing")
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
    tokens = [AXIS, EPISODE_AXIS, "progress/histories_pending", *[row["run_id"] for row in display_rows(campaign)],
        *[prefix + "/" + checkpoint + "/" + metric for checkpoint in CHECKPOINTS
          for prefix, definitions in (("episode", EPISODE_METRICS), ("decision", DECISION_METRICS)) for _, metric in definitions]]
    if any(token not in wire for token in tokens):
        raise PublicationError("Saved remote chain workspace lacks required panels or runs")
    return receipt


def render_report(campaign, entries, output):
    value = dict(campaign_id=campaign.get("campaign_id"), source_commit=campaign["source_commit"], entries=entries,
                 decision_metrics=DECISION_METRICS, episode_metrics=EPISODE_METRICS,
                 episodes=EPISODES, steps_per_episode=STEPS, global_decisions=EPISODES*STEPS)
    encoded = json.dumps(value, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    Path(output).write_text(_HTML.replace("__DATA__", encoded))
    return value


def publish_once(campaign, root, *, wandb_module=None, require_complete=False, workspace_verifier=verify_workspace):
    if wandb_module is None:
        import wandb as wandb_module
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with series._lock(root / ".publisher.lock", blocking=False):
        workspace = workspace_verifier(campaign, root / "workspace.json", wandb_module)
        entries = collect(campaign, require_complete=require_complete)
        render_report(campaign, entries, root / "report.html")
        for entry in entries:
            publish_entry(entry, root, wandb_module)
        progress = {entry["checkpoint_id"] + "/" + entry["method"]: {"status": entry["status"], **entry["progress"]}
                    for entry in entries}
        result = dict(updated_at=now(), workspace_url=workspace["url"], checkpoints=progress,
                      all_complete=all(row["status"] == "complete" for row in progress.values()))
        series._atomic_json(root / "status.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("layout", "publish", "report"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    campaign = load_campaign(args.campaign)
    args.root.mkdir(parents=True, exist_ok=True)
    try:
        if args.command == "layout":
            result = install_workspace(campaign, args.root / "workspace.json")
        elif args.command == "report":
            entries = collect(campaign)
            render_report(campaign, entries, args.root / "report.html")
            result = {"report": str(args.root / "report.html")}
        else:
            result = publish_once(campaign, args.root, require_complete=args.require_complete)
            if args.require_complete and not result["all_complete"]:
                raise PublicationError("Missing final histories were marked failed; publication remains incomplete")
    except Exception as error:
        message = str(error) if isinstance(error, PublicationError) else "Inspect ten-episode receipts and publication journals"
        series._atomic_json(args.root / "publication-failure.json", dict(error_type=type(error).__name__, message=message, at=now()))
        raise SystemExit(f"Ten-episode publication failed ({type(error).__name__}): {message}") from None
    print(series._json(result))


_HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MPPI500k: ten sampled episodes</title><style>body{font:15px system-ui;margin:30px auto;max-width:1250px;padding:0 24px;color:#183044;background:#f6f8fb}section{background:white;padding:20px;margin:18px 0;border:1px solid #dde4ec;border-radius:10px}select{padding:7px}svg{width:100%;height:360px}button{background:white;border:1px solid #ddd;padding:8px;margin:4px;cursor:pointer}table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:left;padding:7px;border-bottom:1px solid #eee}.muted{color:#607386}#tooltip{min-height:27px}</style><h1>MPPI500k: ten sampled episodes · carry versus reset</h1><p>Both methods are newly measured, five paired histories each. Episode 10 is primary. Every history contains ten 500-decision episodes with sampled actions; only the temperature lifetime differs.</p><p class="muted">Paired carry-minus-reset effects use five histories at each position. Fifty episodes are not fifty independent trials. The first/last five contrast is descriptive because both episode position and seed block change. Entropy measures actor-update batches; first/last 100-decision summaries distinguish transients from late-episode behavior. No historical outcomes are relabeled as new episode controls.</p><section><label>Checkpoint <select id="checkpoint"></select></label> <label>Axis <select id="axis"><option value="episode">Episode position</option><option value="decision">Global environment decision</option></select></label> <label>Metric <select id="metric"></select></label><div id="legend"></div><svg id="plot" viewBox="0 0 1180 360"></svg><div id="tooltip"></div><p class="muted">Solid lines connect measured means without smoothing. Dashed vertical lines mark episode boundaries. Hover a curve to highlight it; click legend labels to hide/show. Episode reward accumulation resets; chain reward does not.</p></section><section><h2>Progress</h2><div id="progress"></div></section><section><h2>Episode results</h2><div id="results" style="overflow:auto"></div></section><section><h2>Paired histories</h2><div id="paired" style="overflow:auto"></div></section><section><h2>Provenance</h2><pre id="provenance" style="overflow:auto;max-height:300px"></pre></section><script id="chain-data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('chain-data').textContent),$=id=>document.getElementById(id),ns='http://www.w3.org/2000/svg',hidden=new Set();function option(p,v,t){const e=document.createElement('option');e.value=v;e.textContent=t;p.append(e)}Array.from(new Set(data.entries.map(e=>e.checkpoint_id))).forEach(c=>option($('checkpoint'),c,c));function metricOptions(){$('metric').replaceChildren();data[$('axis').value+'_metrics'].forEach(([t,k])=>option($('metric'),k,t));draw()}function add(tag,attrs){const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));$('plot').append(e);return e}function draw(){const axis=$('axis').value,m=$('metric').value,ak=axis==='decision'?'chain/global_decision':'chain/episode',entries=data.entries.filter(e=>e.checkpoint_id===$('checkpoint').value);$('plot').replaceChildren();$('legend').replaceChildren();$('tooltip').textContent='';entries.forEach(e=>{const b=document.createElement('button');b.textContent=e.name.split(' | ')[1]+' · '+e.status;b.style.color=e.color;b.style.opacity=hidden.has(e.run_id)?.4:1;b.onclick=()=>{hidden.has(e.run_id)?hidden.delete(e.run_id):hidden.add(e.run_id);draw()};$('legend').append(b)});const available=entries.filter(e=>e.payload&&!hidden.has(e.run_id)),points=e=>e.payload[axis+'_points'].filter(p=>Number.isFinite(p[m])),vals=available.flatMap(e=>points(e).map(p=>p[m]));if(!vals.length){add('text',{x:80,y:160,fill:'#607386'}).textContent='Pending complete five-history results';return}let lo=Math.min(...vals),hi=Math.max(...vals),pad=Math.max((hi-lo)*.1,.01);lo-=pad;hi+=pad;const xmax=axis==='decision'?data.global_decisions:data.episodes,x=d=>70+(d-1)/(xmax-1)*1060,y=v=>305-(v-lo)/(hi-lo)*265;for(let i=0;i<=5;i++){const v=lo+(hi-lo)*i/5;add('line',{x1:70,x2:1130,y1:y(v),y2:y(v),stroke:'#e5eaf0'});add('text',{x:2,y:y(v)+4,'font-size':12}).textContent=v.toPrecision(4)}(axis==='decision'?[1,...Array.from({length:data.episodes},(_,i)=>(i+1)*data.steps_per_episode)]:Array.from({length:data.episodes},(_,i)=>i+1)).forEach(d=>add('text',{x:x(d),y:330,'text-anchor':'middle','font-size':12}).textContent=d);if(axis==='decision')Array.from({length:data.episodes-1},(_,i)=>(i+1)*data.steps_per_episode+.5).forEach(d=>{add('line',{x1:x(d),x2:x(d),y1:30,y2:305,stroke:'#8b98a5','stroke-dasharray':'5 5'});add('text',{x:x(d)+5,y:23,'font-size':11}).textContent='Episode '+(Math.floor(d/data.steps_per_episode)+1)});available.forEach(e=>{const ps=points(e);if(!ps.length)return;const line=add('polyline',{points:ps.map(p=>`${x(p[ak])},${y(p[m])}`).join(' '),fill:'none',stroke:e.color,'stroke-width':2.6});line.onmouseenter=()=>{$('plot').querySelectorAll('polyline').forEach(p=>p.style.opacity=p===line?1:.18);line.setAttribute('stroke-width',4)};line.onmouseleave=()=>{$('plot').querySelectorAll('polyline').forEach(p=>p.style.opacity=1);line.setAttribute('stroke-width',2.6)};line.onmousemove=event=>{const p=$('plot').createSVGPoint();p.x=event.clientX;p.y=event.clientY;const q=p.matrixTransform($('plot').getScreenCTM().inverse()),d=Math.max(1,Math.min(xmax,Math.round(1+(q.x-70)/1060*(xmax-1)))),nearest=ps.reduce((a,b)=>Math.abs(a[ak]-d)<Math.abs(b[ak]-d)?a:b),sdk=m.endsWith('_mean')?m.slice(0,-5)+'_sample_std':m+'_sample_std';$('tooltip').textContent=`${e.name} · ${axis} ${nearest[ak]} · ${nearest[m].toPrecision(5)}${Number.isFinite(nearest[sdk])?' ± '+nearest[sdk].toPrecision(4)+' sample SD':''}`}})}function table(parent,heads,rows){const t=document.createElement('table'),h=t.insertRow();heads.forEach(v=>{const e=document.createElement('th');e.textContent=v;h.append(e)});rows.forEach(row=>{const tr=t.insertRow();row.forEach(v=>tr.insertCell().textContent=Number.isFinite(v)?v.toFixed(4):String(v??'—'))});parent.append(t)}table($('progress'),['Checkpoint','Method','Status','Complete histories','Pending','Failed'],data.entries.map(e=>[e.checkpoint_id,e.method,e.status,e.progress.complete,e.progress.pending,e.progress.failed]));table($('results'),['Checkpoint','Method','Episode (10 primary)','Return mean','Return SD','Carry minus reset','Paired effect SD','Update-batch entropy','First100 entropy','Last100 entropy','Last100 target error'],data.entries.flatMap(e=>(e.payload?.episode_points??[]).map(p=>[e.checkpoint_id,e.name.split(' | ')[1],p['chain/episode'],p.return_mean,p.return_sample_std,p.carry_reset_gain_mean,p.carry_reset_gain_sample_std,p.entropy_mean,p.early100_entropy_mean,p.late100_entropy_mean,p.late100_entropy_error_mean])));table($('paired'),['Method','Comparison','Episode','History','Environment seed','Controller seed','Paired difference','Returns'],data.entries.flatMap(e=>(e.payload?.paired_comparisons??[]).flatMap(c=>c.pairs.map(p=>[e.method,c.comparison,c.episode,p.stream_index,p.seed,p.solver_seed,p.difference,JSON.stringify(p.returns)]))));$('provenance').textContent=JSON.stringify({campaign:data.campaign_id,source_commit:data.source_commit,entries:data.entries.map(e=>({...e,payload:e.payload?{overall:e.payload.overall,provenance:e.payload.provenance,diagnostic_availability:e.payload.diagnostic_availability,paired_comparisons:e.payload.paired_comparisons}:null}))},null,2);$('axis').onchange=metricOptions;$('checkpoint').onchange=draw;$('metric').onchange=draw;metricOptions();</script></html>'''


if __name__ == "__main__":
    main()
