"""Install the dedicated native K1-versus-K3 alpha-carry training workspace.

Requires isolated wandb==0.28.1 / wandb-workspaces==0.4.5 and normal SDK
authentication. The --output receipt owns the view; reruns update that view,
preserve user sections, and fail visibly if server readback loses the layout.
The new --run-id may be pending registration; no missing data is shown as zero.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_alpha_carry_workspace import (  # noqa: E402
    ENTITY, PROJECT, STEP, DIAGNOSTIC_STEP, WorkspaceError,
    _color, _filter_wire, _view_id, _write,
)

SOURCE_RUN = "be85c916fcd94d5a813d7857d0ef5efb"
NAME = "Alpha run carry: K1 versus K3 / H3 J8 detached target -10.5 seed55"
PREFIX = "Alpha carry K1 vs K3 / "
METRICS = [
    ("Learning and progress", "Training episode return (sampled actions)", "episode/return", STEP),
    ("Learning and progress", "Uploaded training decisions", "time/total_timesteps", STEP),
    ("Learning and progress", "End-to-end decisions per second", "time/fps", STEP),
    ("Learning and progress", "Controller seconds per decision", "time/inner_seconds_per_action", STEP),
    ("Temperature and entropy", "Inner alpha before solve (window mean)", "train/inner_alpha_initial", STEP),
    ("Temperature and entropy", "Inner alpha after solve (window mean)", "train/inner_alpha_final", STEP),
    ("Temperature and entropy", "Outer alpha", "train/ent_coef", STEP),
    ("Temperature and entropy", "Inner actor-update entropy", "train/inner_actor_entropy", STEP),
    ("Temperature and entropy", "Outer actor-update entropy", "train/actor_entropy", STEP),
    ("Root diagnostics", "Root entropy before adaptation", "train/inner_root_entropy_initial", DIAGNOSTIC_STEP),
    ("Root diagnostics", "Root entropy after adaptation", "train/inner_root_entropy_final", DIAGNOSTIC_STEP),
    ("Root diagnostics", "Root entropy change (final minus initial)", "train/inner_root_entropy_delta", DIAGNOSTIC_STEP),
    ("Solve and hold", "Fresh solves per logging window", "train/inner_solve_performed", STEP),
    ("Solve and hold", "Held-policy decisions per logging window", "train/inner_policy_held", STEP),
    ("Solve and hold", "Policy age in decisions (window mean)", "train/inner_action_age", STEP),
    ("Solve and hold", "Configured decisions per solve", "train/inner_solve_interval", STEP),
    ("Solve and hold", "Imagined transitions per logging window", "train/inner_model_steps", STEP),
    ("Solve and hold", "Strict compilation fallback (must remain zero)", "train/inner_compile_fallback", STEP),
]


def expected_rows(run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or run_id == SOURCE_RUN:
        raise WorkspaceError("--run-id must be a distinct valid K3 run ID")
    return [dict(run_id=value, k=k, name=f"K={k} | H3 J8 | alpha run | aux detached | target -10.5 | seed55",
                 color=color, state="pending registration", decisions=None)
            for value, k, color in ((SOURCE_RUN, 1, "#0072B2"), (run_id, 3, "#D55E00"))]


def discover_rows(api, run_id):
    rows = expected_rows(run_id)
    by_id = {row["run_id"]: row for row in rows}
    seen = set()
    for run in api.runs(f"{ENTITY}/{PROJECT}", filters={"name": {"$in": list(by_id)}}, per_page=2):
        if run.id not in by_id or run.id in seen:
            raise WorkspaceError("Run lookup returned an unexpected or duplicate identity")
        seen.add(run.id)
        by_id[run.id].update(state=run.state, decisions=dict(run.summary).get(STEP))
    if SOURCE_RUN not in seen:
        raise WorkspaceError("The original K1 run could not be read")
    return rows


def _markdown(rows, updated):
    status = "\n".join(f"| K={row['k']} | {row['state']} | "
                       f"{row['decisions'] if row['decisions'] is not None else 'pending'} | `{row['run_id']}` |"
                       for row in rows)
    return f"""**K=1 versus K=3 full-training AMBI: decisions between fresh inner solves.**

K=1 adapts at every real decision. K=3 adapts once, then samples the same adapted actor on the next two current observations using that solve's frozen encoder. Episode boundaries discard held control. Actor, critic and imagined replay are reinitialized from the current prior at each fresh solve. Alpha and its Adam state carry across solves and episodes; held decisions do not perform temperature updates. Outer training continues at every decision.

Both runs: **Humanoid Walk state observations; H3/J8; C16/A4/N128/B256 per solve; auxiliary return critic with detached representation; target entropy −10.5; sampled training actions; seed55; 1M decisions = 2M raw environment steps.** Retain full replay checkpoints every25k decisions. The K3 variant changes the solve interval while retaining per-solve update dose, so inner compute per real decision decreases. Training return is stochastic collection return; one seed per setting does not estimate training-seed variance.

Each native curve is one exact run ID, without averaging or smoothing. Main panels use `env_step`; root entropy uses128 paired samples and the actual sampled diagnostic decision axis. K3 diagnostics run on the first fresh solve at or after each1000-decision cadence boundary, so held decisions cannot suppress diagnostics. Solve/hold counts are sums within each logging window, not lifetime totals. Original K1 data may lack newly introduced solve/hold counters; missing curves do not mean zero.

**Startup/progress snapshot: {updated}.** Curves update with uploaded history. Rerun this installer to refresh the table; pending/empty panels mean no observations are uploaded yet.

| Setting | State at refresh | Uploaded decisions | Run ID |
|---|---|---|---|
{status}
"""


def _sections(rows, updated, ws, wr):
    sections = [ws.Section(name=PREFIX + "Protocol and progress", is_open=True,
                           panels=[wr.MarkdownPanel(markdown=_markdown(rows, updated), layout=wr.Layout(w=24, h=14))])]
    for category in dict.fromkeys(item[0] for item in METRICS):
        panels = [wr.LinePlot(title=title, x=axis, y=[metric], aggregate=False,
                             smoothing_type="none", legend_template="${run:name}", max_runs_to_show=2,
                             title_x="Sampled diagnostic decision" if axis == DIAGNOSTIC_STEP else "Training decisions",
                             line_colors={row["run_id"] + ":" + metric: row["color"] for row in rows},
                             line_titles={row["run_id"] + ":" + metric: row["name"] for row in rows},
                             line_marks={row["run_id"] + ":" + metric: "solid" for row in rows},
                             layout=wr.Layout(w=12, h=8))
                  for section, title, metric, axis in METRICS if section == category]
        sections.append(ws.Section(name=PREFIX + category, is_open=True, panels=panels))
    return sections


def _verify(work, rows, url, wanted, wr, expr):
    if ((work.entity, work.project, work.name) != (ENTITY, PROJECT, NAME)
            or _view_id(work.url) != _view_id(url)):
        raise WorkspaceError("Saved workspace identity differs from the receipt")
    owned = [section for section in work.sections if section.name.startswith(PREFIX)]
    if (len(owned) != 1 + len({item[0] for item in METRICS})
            or sum(len(section.panels) for section in owned) != len(METRICS) + 1):
        raise WorkspaceError("Saved workspace lost required sections or panels")
    settings = work.runset_settings
    if settings.groupby or settings.query or _filter_wire(settings.filters, expr) != _filter_wire(wanted, expr):
        raise WorkspaceError("Saved workspace lost its exact two-run filter")
    if (work.settings.x_axis != STEP or work.settings.remove_legends_from_panels
            or not work.settings.tooltip_color_run_names or work.settings.tooltip_number_of_runs != "all_runs"):
        raise WorkspaceError("Saved workspace lost native hover and legend settings")
    for row in rows:
        run = settings.run_settings.get(row["run_id"])
        if run is None or run.disabled or _color(run.color) != row["color"]:
            raise WorkspaceError("Saved workspace lost run visibility or color")
    expected = {metric: (title, axis) for _, title, metric, axis in METRICS}
    seen = set()
    for section in owned:
        for panel in section.panels:
            if not isinstance(panel, wr.LinePlot):
                continue
            metric = getattr(panel.y[0], "name", panel.y[0]) if len(panel.y) == 1 else None
            if metric not in expected or metric in seen:
                raise WorkspaceError("Saved panel metric is unexpected or duplicated")
            seen.add(metric)
            title, axis = expected[metric]
            if (panel.title != title or getattr(panel.x, "name", panel.x) != axis
                    or panel.aggregate or panel.smoothing_type != "none"
                    or panel.legend_template != "${run:name}" or panel.max_runs_to_show < 2):
                raise WorkspaceError("Saved panel lost its axis or independent line semantics")
            for row in rows:
                key = row["run_id"] + ":" + metric
                if (panel.line_titles.get(key) != row["name"] or panel.line_marks.get(key) != "solid"
                        or _color(panel.line_colors.get(key)) != row["color"]):
                    raise WorkspaceError("Saved panel lost readable K labels or colors")
    if seen != set(expected):
        raise WorkspaceError("Saved workspace lacks required metrics")


def install(output, run_id, *, api=None, backend=None):
    expected_rows(run_id)
    output = Path(output)
    previous = json.loads(output.read_text()) if output.exists() else {}
    identity = dict(entity=ENTITY, project=PROJECT, source_run_id=SOURCE_RUN, run_id=run_id, name=NAME)
    if previous and any(previous.get(key) != value for key, value in identity.items()):
        raise WorkspaceError("Refusing to replace another comparison's receipt")
    url = previous.get("url")
    if url:
        _view_id(url)
    receipt = dict(**identity, url=url, api_verified=False, panel_count=len(METRICS) + 1, metrics=METRICS)
    _write(output, receipt)
    try:
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
        rows = discover_rows(api, run_id)
        updated = datetime.now(timezone.utc).isoformat()
        receipt.update(rows=rows, updated_at=updated)
        sections = _sections(rows, updated, ws, wr)
        if url:
            work = ws.Workspace.from_url(url)
            if (work.entity, work.project, work.name) != (ENTITY, PROJECT, NAME):
                raise WorkspaceError("Receipt points to an unrelated named workspace")
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
        wanted = expr.Metric("ID").isin([row["run_id"] for row in rows])
        work.runset_settings = ws.RunsetSettings(filters=[wanted], groupby=[],
            run_settings={row["run_id"]: ws.RunSettings(color=row["color"], disabled=False) for row in rows},
            pinned_columns=["run:displayName", "summary:env_step", "summary:episode/return"])
        work.settings = ws.WorkspaceSettings(x_axis=STEP, max_runs=2, tooltip_number_of_runs="all_runs",
            tooltip_color_run_names=True, remove_legends_from_panels=False)
        work._auto_generate_panels = False
        work.save()
        receipt.update(url=work.url, view_id=_view_id(work.url))
        _write(output, receipt)
        _verify(ws.Workspace.from_url(work.url), rows, work.url, wanted, wr, expr)
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
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.output, args.run_id), indent=2))


if __name__ == "__main__":
    main()
