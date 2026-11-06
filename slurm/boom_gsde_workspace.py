"""Install and verify the BOOM x0.01 / gSDE campaign's owned W&B workspace.

Usage: AMBI_WORKSPACE_PYTHON=/path/to/python <python> -m
slurm.boom_gsde_workspace /absolute/campaign/submission.json

Training creates runs and metrics. This publisher creates only a dedicated view;
its receipt and file lock make repeat calls safe across concurrent launchers.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import sys

OWNER = "boom-gsde-oscar-20261106"
ENTITY, PROJECT = "rwgao_b-brown-university", "ambi"
PREFIX = "BOOM + gSDE / "
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#000000")
METRICS = {
    "Return and progress": [
        ("Training episode return", "episode/return"),
        ("Rolling 20-episode mean", "episode/return_window20_mean"),
        ("Rolling mean drawdown", "episode/return_window20_drawdown"),
        ("End-to-end decisions per second", "time/fps"),
    ],
    "BOOM alignment and gating": [
        ("Shared Q scale S (gate activates above 2)", "train/actor_loss_scale"),
        ("BOOM gate active", "train/boom_gate_active"),
        ("Weighted alignment NLL before scaling", "train/boom_alignment_nll"),
        ("BOOM loss after gate, S and coefficient", "train/boom_alignment_loss"),
        ("Effective Q-weighted samples", "train/boom_effective_samples"),
        ("Valid saved behavior targets", "train/boom_valid_fraction"),
    ],
    "Gradients and policy": [
        ("Total outer actor gradient norm (all actor losses)", "train/actor_grad_norm"),
        ("Inner actor gradient norm", "train/inner_actor_grad_norm"),
        ("Joint post-tanh entropy", "train/actor_post_tanh_entropy"),
        ("Sampled actor saturation at |a| >= 0.99", "train/actor_action_abs_ge_0p99_fraction"),
        ("Executed within-episode action change RMS", "train/executed_action_delta_rms"),
        ("Critic fitting loss", "train/critic_loss"),
    ],
    "Acceleration": [
        ("Whole-inner CUDA graph used", "train/inner_cuda_graph_used"),
        ("Inner solve seconds per decision", "time/inner_seconds_per_action"),
        ("Inner compilation fallback (must stay zero)", "train/inner_compile_fallback"),
        ("BOOM actor compilation fallback (must stay zero)", "train/compile_outer_actor_fallback"),
    ],
    "Critic geometry": [
        ("Primary critic action gradient RMS", "train/critic_geometry_primary_action_gradient_rms"),
        ("Primary action smoothness loss", "train/critic_geometry_primary_action_smoothness_loss"),
        ("Primary MPR loss", "train/critic_geometry_primary_mpr_loss"),
        ("Action-gradient direction agreement", "train/critic_geometry_primary_action_smoothness_cosine_similarity"),
    ],
}


def atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def finite(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def specs_from_state(state):
    runs = state["runs"]
    if len(runs) != 3:
        raise ValueError("The BOOM/gSDE campaign requires exactly three new runs.")
    specs = []
    for index, source in enumerate(runs + state.get("controls", [])):
        spec = dict(source)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", spec["id"]):
            raise ValueError("Invalid W&B run ID.")
        spec["label"] = spec.get("label", spec.get("name", spec["id"]))
        spec["color"] = spec.get("color", COLORS[index % len(COLORS)])
        specs.append(spec)
    if len({spec["id"] for spec in specs}) != len(specs):
        raise ValueError("Duplicate run IDs in campaign state.")
    if not re.fullmatch(r"[0-9a-f]{40}", state["commit"]):
        raise ValueError("An exact source commit is required.")
    return specs


def rows_from_api(api, specs):
    ids = [spec["id"] for spec in specs]
    existing = {run.id: run for run in api.runs(
        f"{ENTITY}/{PROJECT}", filters={"name": {"$in": ids}}, per_page=len(ids),
    )}
    rows = []
    for spec in specs:
        run = existing.get(spec["id"])
        summary = {} if run is None else dict(run.summary)
        status = ("submitted / awaiting training start" if spec.get("job") else
                  "pending GPU validation / training submission")
        rows.append({**spec, "state": status if run is None else run.state,
                     "decisions": finite(summary.get("env_step")),
                     "last_return": finite(summary.get("episode/return"))})
    return rows


def markdown(state, rows, updated):
    def cell(value):
        return str(value if value is not None else "pending").replace("|", "/")
    status = "\n".join(
        f"| {cell(row['label'])} | {cell(row.get('job'))} | {cell(row['state'])} | {cell(row['decisions'])} |"
        for row in rows)
    return f"""**BOOM x0.01 with gSDE: three fresh Humanoid Walk runs on Oscar.**

All three use outer adaptation, H3/J8/K1/N128/C16/A4/B256, seed55 and
**1,000,000 real decisions**, with every25k portable checkpoint retained.
They differ only in imagined gSDE matrix persistence: `real_steps`, `solve`, or
`round`. Real-step collection and independent real-action execution matrices
refresh every8 decisions. Both prior and cloned inner actors use actor-feature
gSDE, initial log-scale -3 and mean clip2.

The BOOM coefficient is **0.0042** (0.01 times0.42), with Q-weighting,
standard-deviation floor0.05, and the original strict **S > 2** gate. Alignment
uses the saved adapted policy mean; persistent exploration noise is not the
alignment target. Post-tanh entropy is retained with fixed alpha0.0021,
reward-only primary critics and no auxiliary critic.

Strict compilation, vmap and whole-inner CUDA graphs are enabled. GPU validation
and CPU sizing benchmarks publish no W&B runs. Critic geometry is observed
every1000 outer updates. The gradient plot is the **total actor gradient**;
these runs do not separately log BOOM-only parameter-gradient norms.

Source `{state['commit']}`. Outputs `{state.get('output', 'see submission receipt')}`.
Curves are stochastic training returns with no smoothing or across-seed aggregation.
2500 random warmup decisions,2500 pretraining updates and compilation precede
normal learning. Empty curves mean pending measurements, never zero performance.

| Arm | Slurm job | State at {updated} | Uploaded decisions |
|---|---|---|---|
{status}

This table is a timestamped upload snapshot; native charts update as training
uploads data. Existing controls, when listed, are references and are not relaunched.
"""


def verify_filter(work, ids, expr):
    filters = work.runset_settings.filters
    if isinstance(filters, str):
        node = expr.expr_to_filters(filters)
    elif isinstance(filters, (list, tuple)):
        if len(filters) != 1:
            raise ValueError("Workspace no longer has its exact campaign filter.")
        node = filters[0].to_model()
    else:
        node = filters.to_model()
    while node.op in ("AND", "OR") and len(node.filters or []) == 1:
        node = node.filters[0]
    if (node.op != "IN" or node.key is None or node.key.section != "run"
            or node.key.name != "name" or not isinstance(node.value, list)
            or set(node.value) != set(ids) or len(node.value) != len(ids) or node.disabled):
        raise ValueError("Workspace filter changed; preserving it and refusing to overwrite.")
    if work.runset_settings.query or work.runset_settings.groupby:
        raise ValueError("Workspace has user-added search/grouping; refusing to overwrite.")


def verify(work, specs, ws, wr, expr):
    ids = [spec["id"] for spec in specs]
    if (work.entity, work.project) != (ENTITY, PROJECT):
        raise ValueError("Workspace points to a different project.")
    verify_filter(work, ids, expr)
    if work.settings.x_axis != "env_step" or work.settings.max_runs != len(ids):
        raise ValueError("Workspace axes or run limit do not match the campaign.")
    owned = {section.name: section for section in work.sections if section.name.startswith(PREFIX)}
    status = owned.get(PREFIX + "Protocol and progress")
    if status is None or len(status.panels) != 1 or not isinstance(status.panels[0], wr.MarkdownPanel):
        raise ValueError("Missing pending/progress panel.")
    for section, metrics in METRICS.items():
        panels = owned.get(PREFIX + section)
        if panels is None or len(panels.panels) != len(metrics):
            raise ValueError("Missing required campaign charts.")
        for panel, (title, metric) in zip(panels.panels, metrics):
            name = lambda value: getattr(value, "name", value)
            if (not isinstance(panel, wr.LinePlot) or panel.title != title
                    or name(panel.x) != "env_step" or [name(value) for value in panel.y] != [metric]
                    or panel.aggregate or panel.smoothing_type != "none"
                    or panel.max_runs_to_show != len(ids)):
                raise ValueError("Campaign chart differs from its required metric/axis.")
    for spec in specs:
        setting = work.runset_settings.run_settings.get(spec["id"])
        if setting is None or setting.disabled:
            raise ValueError("Campaign run is missing or disabled in the saved view.")
    return 1 + sum(len(metrics) for metrics in METRICS.values())


def install(state_path):
    import wandb
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr

    state = json.loads(state_path.read_text())
    specs = specs_from_state(state)
    ids = [spec["id"] for spec in specs]
    rows = rows_from_api(wandb.Api(), specs)
    updated = datetime.now(timezone.utc).isoformat()
    sections = [ws.Section(name=PREFIX + "Protocol and progress", is_open=True, panels=[
        wr.MarkdownPanel(markdown=markdown(state, rows, updated), layout=wr.Layout(w=24, h=15))])]
    for name, metrics in METRICS.items():
        panels = [wr.LinePlot(title=title, x="env_step", y=[metric], title_x="Agent decisions",
            aggregate=False, smoothing_type="none", legend_template="${run:displayName}",
            max_runs_to_show=len(ids), layout=wr.Layout(w=12, h=8),
            line_colors={spec["id"] + ":" + metric: spec["color"] for spec in specs},
            line_titles={spec["id"] + ":" + metric: spec["label"] for spec in specs})
            for title, metric in metrics]
        sections.append(ws.Section(name=PREFIX + name, is_open=True, panels=panels,
            layout_settings=ws.SectionLayoutSettings(columns=2, rows=2)))
    receipt_path = state_path.parent / "workspace.json"
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    if previous:
        if (previous.get("owner") != OWNER or previous.get("run_ids") != ids
                or previous.get("commit") != state["commit"]):
            raise ValueError("Workspace receipt belongs to a different campaign.")
        work = ws.Workspace.from_url(previous["url"])
        verify(work, specs, ws, wr, expr)
        replacements = {section.name: section for section in sections}
        work.sections = [replacements.pop(section.name, section) for section in work.sections]
        work.sections.extend(replacements.values())
    else:
        work = ws.Workspace(entity=ENTITY, project=PROJECT,
            name="BOOM x0.01 with gSDE - Oscar", auto_generate_panels=False, sections=sections,
            settings=ws.WorkspaceSettings(x_axis="env_step", max_runs=len(ids),
                tooltip_number_of_runs="all_runs", tooltip_color_run_names=True),
            runset_settings=ws.RunsetSettings(filters=[expr.Metric("ID").isin(ids)], groupby=[],
                run_settings={spec["id"]: ws.RunSettings(color=spec["color"], disabled=False) for spec in specs},
                pinned_columns=["run:displayName", "summary:env_step", "summary:episode/return"]))
    work.save()
    receipt = {"owner": OWNER, "commit": state["commit"], "url": work.url,
               "run_ids": ids, "updated_at": updated, "api_verified": False,
               "browser_verified": previous.get("browser_verified", False), "runs": rows}
    atomic_json(receipt_path, receipt)
    receipt.update(panel_count=verify(ws.Workspace.from_url(work.url), specs, ws, wr, expr),
                   api_verified=True)
    atomic_json(receipt_path, receipt)
    return receipt


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m slurm.boom_gsde_workspace /absolute/campaign/submission.json")
    state_path = Path(sys.argv[1]).resolve(strict=True)
    with (state_path.parent / "workspace.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            receipt = install(state_path)
        except Exception as error:
            atomic_json(state_path.parent / "workspace.failure.json", {
                "success": False, "error_type": type(error).__name__,
                "at": datetime.now(timezone.utc).isoformat(),
            })
            raise SystemExit(f"Workspace setup failed: {type(error).__name__}; no success claimed") from None
        print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
