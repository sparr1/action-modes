"""Install native per-run curves for the full-training alpha-carry factorial.

Use an isolated wandb==0.28.1 / wandb-workspaces==0.4.5 environment. Credentials
come from the environment or normal SDK authentication; this file stores none.
The receipt passed with --output owns one dedicated named view. Re-running
refreshes startup status and colors after pending runs receive their run IDs.
--dry-run writes the complete planned metric catalog without network access.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ENTITY = "rwgao_b-brown-university"
PROJECT = "ambi"
GROUP = "ambi-alpha-carry-full-20261023"
NAME = "Full AMBI H3 J8: alpha carry, representation gradients and entropy targets"
PREFIX = "Alpha carry full training / "
MANIFEST = ROOT / "configs/dmcontrol/experiments/ambi_alpha_carry_training.json"
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7",
          "#56B4E9", "#E69F00", "#8A5A44", "#6A3D9A")
STEP = "env_step"
DIAGNOSTIC_STEP = "train/inner_diagnostics_step"

# Each metric is emitted by AMBITDMPC2/TDMPC2 and the interval accumulators.
# Entropy min/max summarize update-batch means, not individual-action entropy.
METRICS = [
    ("Learning and cost", "Training episode return (sampled actions)", "episode/return", STEP),
    ("Learning and cost", "Decisions completed versus elapsed seconds", "time/total_timesteps", "time/time_elapsed"),
    ("Learning and cost", "End-to-end decisions per second", "time/fps", STEP),
    ("Learning and cost", "Controller seconds per decision", "time/inner_seconds_per_action", STEP),
    ("Entropy and temperature", "Inner alpha before solve (window mean)", "train/inner_alpha_initial", STEP),
    ("Entropy and temperature", "Inner alpha after solve (window mean)", "train/inner_alpha_final", STEP),
    ("Entropy and temperature", "Outer alpha", "train/ent_coef", STEP),
    ("Entropy and temperature", "Inner actor-update entropy mean", "train/inner_actor_entropy", STEP),
    ("Entropy and temperature", "Inner actor-update entropy maximum", "train/inner_actor_entropy_max", STEP),
    ("Entropy and temperature", "Inner actor-update entropy minimum", "train/inner_actor_entropy_min", STEP),
    ("Entropy and temperature", "Outer actor-update entropy mean", "train/actor_entropy", STEP),
    ("Entropy and temperature", "Outer actor-update entropy maximum", "train/actor_entropy_max", STEP),
    ("Entropy and temperature", "Outer actor-update entropy minimum", "train/actor_entropy_min", STEP),
    ("Entropy and temperature", "Inner temperature entropy residual (entropy minus target)", "train/inner_temperature_entropy_residual", STEP),
    ("Entropy and temperature", "Selected target entropy (both SAC actors)", "train/inner_target_entropy", STEP),
    ("Root policy diagnostics", "Root entropy before adaptation", "train/inner_root_entropy_initial", DIAGNOSTIC_STEP),
    ("Root policy diagnostics", "Root entropy after adaptation", "train/inner_root_entropy_final", DIAGNOSTIC_STEP),
    ("Root policy diagnostics", "Root entropy change: final minus initial", "train/inner_root_entropy_delta", DIAGNOSTIC_STEP),
    ("Root policy diagnostics", "Root entropy change Monte Carlo standard error", "train/inner_root_entropy_delta_mc_se", DIAGNOSTIC_STEP),
    ("Root policy diagnostics", "Final adapted-policy KL from prior", "train/inner_final_outer_policy_kl", DIAGNOSTIC_STEP),
    ("Temperature carry", "Inner alpha Adam step at solve start (window mean)", "train/inner_temperature_optimizer_step_initial", STEP),
    ("Temperature carry", "Inner alpha Adam step at solve end (window mean)", "train/inner_temperature_optimizer_step_final", STEP),
    ("Temperature carry", "Inner alpha lifetime updates at solve end (window mean)", "train/inner_temperature_lifetime_steps_final", STEP),
    ("Temperature carry", "Fraction of solves resetting alpha at an episode boundary", "train/inner_temperature_episode_reset", STEP),
    ("Temperature carry", "Fraction of solves initializing run-carried alpha", "train/inner_temperature_chain_reset", STEP),
    ("Temperature carry", "Inner temperature loss", "train/inner_temperature_loss", STEP),
    ("Temperature carry", "Inner temperature gradient norm", "train/inner_temperature_grad_norm", STEP),
    ("Optimization", "Primary soft critic loss", "train/critic_loss", STEP),
    ("Optimization", "Auxiliary return critic loss", "train/aux_return_critic_loss", STEP),
    ("Optimization", "Inner reward-only critic loss", "train/inner_critic_loss", STEP),
    ("Optimization", "Outer actor loss", "train/actor_loss", STEP),
    ("Optimization", "Inner actor loss", "train/inner_actor_loss", STEP),
    ("Optimization", "Inner strict compilation fallback (must remain zero)", "train/inner_compile_fallback", STEP),
]


class WorkspaceError(ValueError):
    """The saved view or run identities do not match the requested campaign."""


def _write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _view_id(url):
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if parsed.hostname == "forge.coreweave.com" and path.startswith("/wandb/"):
        path = path[len("/wandb"):]
    ids = parse_qs(parsed.query).get("nw", [])
    if (parsed.scheme != "https" or parsed.hostname not in {"wandb.ai", "forge.coreweave.com"}
            or path != f"/{ENTITY}/{PROJECT}" or len(ids) != 1
            or not re.fullmatch(r"[A-Za-z0-9_-]+", ids[0])):
        raise WorkspaceError("Receipt URL must identify this campaign's named project view")
    return ids[0]


def _color(value):
    return value.get("color") if isinstance(value, dict) else value


def _filter_wire(value, expr):
    if isinstance(value, str):
        model = expr.expr_to_filters(value)
    else:
        if isinstance(value, list):
            if len(value) != 1:
                raise WorkspaceError("Expected exactly one campaign group filter")
            value = value[0]
        model = value.to_model()
    wire = model.model_dump(exclude_none=True) if hasattr(model, "model_dump") else model.dict(exclude_none=True)

    def normalize(node):
        if isinstance(node, dict):
            node = {key: normalize(item) for key, item in node.items()
                    if not (key == "disabled" and item is False)}
            if node.get("op") in ("AND", "OR") and len(node.get("filters", [])) == 1 and "key" not in node:
                return node["filters"][0]
        elif isinstance(node, list):
            node = [normalize(item) for item in node]
        return node
    return normalize(wire)


def expected_rows():
    manifest = json.loads(MANIFEST.read_text())
    if len(manifest["configs"]) != len(COLORS):
        raise WorkspaceError("Expected exactly eight campaign configurations")
    rows = []
    for name, color in zip(manifest["configs"], COLORS):
        config = json.loads((ROOT / "configs/dmcontrol/algs" / f"{name}.json").read_text())
        params = config["alg_params"]
        if (params["wandb_group"], params["wandb_project"], params["wandb_entity"]) != (GROUP, PROJECT, ENTITY):
            raise WorkspaceError("Configuration publishes outside the expected campaign")
        rows.append(dict(config=name, name=params["wandb_run_name"], color=color,
                         run_id=None, state="pending startup", decisions=None, last_return=None))
    if len({row["name"] for row in rows}) != 8:
        raise WorkspaceError("Each factorial cell requires a unique display name")
    return rows


def discover_rows(api):
    rows = expected_rows()
    by_name = {row["name"]: row for row in rows}
    for run in api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}, per_page=20):
        if run.name not in by_name:
            raise WorkspaceError(f"Unexpected run name in campaign: {run.name}")
        row = by_name[run.name]
        if row["run_id"] is not None:
            raise WorkspaceError(f"Multiple run IDs claim the same factorial cell: {run.name}")
        summary = dict(run.summary)
        row.update(run_id=run.id, state=run.state, decisions=summary.get("env_step"),
                   last_return=summary.get("episode/return"))
    return rows


def _markdown(rows, updated):
    status = "\n".join(
        f"| {row['name'].replace(' | ', ' / ')} | {row['state']} | "
        f"{row['decisions'] if row['decisions'] is not None else 'pending'} |"
        for row in rows)
    return f"""**Eight full-training AMBI runs: H3/J8; alpha episode/run carry × auxiliary shared/detached gradients × entropy target −10.5/−21.**

Both temperature modes carry alpha and its Adam state across decisions. Episode mode resets both at episode boundaries to the current outer alpha; run mode continues across episodes. Both SAC actors use the selected target, with independently learned temperatures. Detachment blocks auxiliary return TD gradients into the representation; the auxiliary critic still trains.

**1M decisions = 2M raw environment steps; sampled training actions; seed55; C16/A4/N128/B256.** All checkpoints every25k decisions are retained. Actor, critic and imagined replay stay action-local. No KL regularization, Bernoulli or Retrace. Training return is stochastic collection return, not a paired base-policy improvement estimate.

Native lines represent separate runs, with exact factor labels and eight distinct colors. Hover over a run's name or curve to identify it; no cross-run averaging or smoothing is applied. Root entropy uses128 paired samples at the diagnostic cadence and its sampled-decision axis. Actor-update entropy min/max are extrema of update-batch mean entropy, not maxima/minima over individual actions. Entropy residual is **entropy minus target**. Reset panels are fractions of solves in each logging window. One seed per setting does not estimate training-seed variance.

**Startup/progress snapshot: {updated}.** Re-run this installer to refresh the table and bind newly registered run IDs to their exact colors. Pending/empty panels mean no uploaded observations yet, never zero performance. Run status and metrics update independently of this snapshot.

| Setting | Status | Uploaded decisions |
|---|---|---|
{status}
"""


def _sections(rows, updated, ws, wr):
    active = [row for row in rows if row["run_id"] is not None]
    sections = [ws.Section(name=PREFIX + "Protocol and startup", is_open=True,
                           panels=[wr.MarkdownPanel(markdown=_markdown(rows, updated), layout=wr.Layout(w=24, h=14))])]
    for section in dict.fromkeys(item[0] for item in METRICS):
        panels = []
        for category, title, metric, axis in METRICS:
            if category != section:
                continue
            panels.append(wr.LinePlot(title=title, x=axis, y=[metric], aggregate=False,
                smoothing_type="none", legend_template="${run:name}", max_runs_to_show=8,
                title_x=("Elapsed seconds" if axis == "time/time_elapsed" else
                         "Sampled decision" if axis == DIAGNOSTIC_STEP else "Training decisions"),
                line_colors={row["run_id"] + ":" + metric: row["color"] for row in active},
                line_titles={row["run_id"] + ":" + metric: row["name"] for row in active},
                line_marks={row["run_id"] + ":" + metric: "solid" for row in active},
                layout=wr.Layout(w=12, h=8)))
        sections.append(ws.Section(name=PREFIX + section, is_open=True, panels=panels))
    return sections


def _verify(work, rows, url, expected_filter, wr, expr):
    if ((work.entity, work.project, work.name) != (ENTITY, PROJECT, NAME)
            or _view_id(work.url) != _view_id(url)):
        raise WorkspaceError("Saved workspace identity differs from the receipt")
    owned = [section for section in work.sections if section.name.startswith(PREFIX)]
    if len(owned) != 1 + len({item[0] for item in METRICS}):
        raise WorkspaceError("Saved workspace is missing or duplicating campaign sections")
    if sum(len(section.panels) for section in owned) != len(METRICS) + 1:
        raise WorkspaceError("Saved workspace panel count differs from the planned layout")
    settings = work.runset_settings
    if settings.groupby or settings.query or _filter_wire(settings.filters, expr) != expected_filter:
        raise WorkspaceError("Saved workspace lost the independent campaign run filter")
    if (work.settings.x_axis != STEP or work.settings.remove_legends_from_panels
            or not work.settings.tooltip_color_run_names
            or work.settings.tooltip_number_of_runs != "all_runs"):
        raise WorkspaceError("Saved workspace lost native hover/legend settings")
    active = [row for row in rows if row["run_id"] is not None]
    for row in active:
        run = settings.run_settings.get(row["run_id"])
        if run is None or run.disabled or _color(run.color) != row["color"]:
            raise WorkspaceError("Saved workspace lost run visibility or exact colors")
    expected = {metric: (title, axis) for _, title, metric, axis in METRICS}
    seen = set()
    for section in owned:
        for panel in section.panels:
            if not isinstance(panel, wr.LinePlot):
                continue
            metric = getattr(panel.y[0], "name", panel.y[0]) if len(panel.y) == 1 else None
            if metric not in expected or metric in seen:
                raise WorkspaceError("Saved panel metric is missing or duplicated")
            seen.add(metric)
            title, axis = expected[metric]
            if (panel.title != title or getattr(panel.x, "name", panel.x) != axis
                    or panel.aggregate or panel.smoothing_type != "none"
                    or panel.legend_template != "${run:name}" or panel.max_runs_to_show < 8):
                raise WorkspaceError("Saved panel lost its metric axis or independent line semantics")
            for row in active:
                key = row["run_id"] + ":" + metric
                if (panel.line_titles.get(key) != row["name"] or panel.line_marks.get(key) != "solid"
                        or _color(panel.line_colors.get(key)) != row["color"]):
                    raise WorkspaceError("Saved panel lost readable names, solid lines or exact colors")
    if seen != set(expected):
        raise WorkspaceError("Saved workspace lacks required training metrics")


def install(output, *, api=None, backend=None):
    output = Path(output)
    previous = json.loads(output.read_text()) if output.exists() else {}
    if previous and (previous.get("entity"), previous.get("project"), previous.get("group")) != (ENTITY, PROJECT, GROUP):
        raise WorkspaceError("Refusing to replace a receipt belonging to another campaign")
    url = previous.get("url")
    if url:
        _view_id(url)
    if backend is None:
        for package, version in (("wandb", "0.28.1"), ("wandb-workspaces", "0.4.5")):
            if importlib.metadata.version(package) != version:
                raise WorkspaceError(f"Workspace installation requires isolated {package}=={version}")
        import wandb
        import wandb_workspaces.workspaces as ws
        import wandb_workspaces.reports.v2 as wr
        from wandb_workspaces import expr
        api = api or wandb.Api(timeout=30)
    else:
        ws, wr, expr = backend
    rows = discover_rows(api)
    updated = datetime.now(timezone.utc).isoformat()
    receipt = dict(entity=ENTITY, project=PROJECT, group=GROUP, name=NAME, url=url,
                   api_verified=False, rows=deepcopy(rows), updated_at=updated,
                   panel_count=len(METRICS) + 1, metrics=METRICS)
    _write(output, receipt)
    try:
        sections = _sections(rows, updated, ws, wr)
        if url:
            work = ws.Workspace.from_url(url)
            if (work.entity, work.project, work.name) != (ENTITY, PROJECT, NAME):
                raise WorkspaceError("Receipt points to an unrelated named workspace")
            # User-added sections in this dedicated view are also preserved.
            replacement, inserted = [], False
            for section in work.sections:
                if section.name.startswith(PREFIX):
                    if not inserted:
                        replacement.extend(sections)
                        inserted = True
                else:
                    replacement.append(section)
            work.sections = replacement if inserted else replacement + sections
        else:
            work = ws.Workspace(entity=ENTITY, project=PROJECT, name=NAME,
                                sections=sections, auto_generate_panels=False)
        wanted = expr.Metric("Group") == GROUP
        work.runset_settings = ws.RunsetSettings(filters=[wanted], groupby=[],
            run_settings={row["run_id"]: ws.RunSettings(color=row["color"], disabled=False)
                          for row in rows if row["run_id"] is not None},
            pinned_columns=["run:displayName", "summary:env_step", "summary:episode/return"])
        work.settings = ws.WorkspaceSettings(x_axis=STEP, max_runs=8,
            tooltip_number_of_runs="all_runs", tooltip_color_run_names=True, remove_legends_from_panels=False)
        work._auto_generate_panels = False
        work.save()
        receipt["url"] = work.url
        receipt["view_id"] = _view_id(work.url)
        _write(output, receipt)  # Preserve the view identity even if read-back fails.
        _verify(ws.Workspace.from_url(work.url), rows, work.url, _filter_wire(wanted, expr), wr, expr)
        receipt["api_verified"] = True
    except Exception as exc:
        receipt.update(error_type=type(exc).__name__, error=str(exc))
        _write(output, receipt)
        raise
    _write(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        if args.output.exists():
            raise WorkspaceError("Dry-run output must be new; do not overwrite a live workspace receipt")
        result = dict(entity=ENTITY, project=PROJECT, group=GROUP, name=NAME,
                      url=None, api_verified=False, dry_run=True, rows=expected_rows(),
                      metrics=METRICS, panel_count=len(METRICS) + 1)
        _write(args.output, result)
    else:
        result = install(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
