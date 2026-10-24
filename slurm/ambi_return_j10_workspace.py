"""Idempotent native J8/J10 comparison in its own named W&B workspace.

Use isolated wandb==0.28.1 and wandb-workspaces==0.4.5. ``install(output,
api=authenticated_api)`` supports local credential injection. This publisher
only edits its receipt-owned view; training owns runs, configuration and data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse


ENTITY = "rwgao_b-brown-university"
PROJECT = "ambi"
GROUP = "ambi-return-only-j10-training-20261024"
BASELINE = "333ac0c3631e46699146eb291c5377c0"
NAME = "Return-only H3 K3: J8 vs J10"
PREFIX = "Return-only J10 / "
COLORS = {8: "#0072B2", 10: "#D55E00"}
LABELS = {8: "H3 K3 J8 | return-only | seed55", 10: "H3 K3 J10 | return-only | seed55"}
METRICS = [
    ("Return and cost", "Training episode return (sampled actions)", "episode/return", "env_step"),
    ("Return and cost", "Controller seconds per real decision", "time/inner_seconds_per_action", "env_step"),
    ("Return and cost", "End-to-end decisions per second", "time/fps", "env_step"),
    ("Return and cost", "Training decisions versus elapsed time", "time/total_timesteps", "time/time_elapsed"),
    ("Entropy and temperature", "Outer actor-update entropy", "train/actor_entropy", "env_step"),
    ("Entropy and temperature", "Inner actor-update entropy (solve decisions)", "train/inner_actor_entropy", "env_step"),
    ("Entropy and temperature", "Outer temperature", "train/ent_coef", "env_step"),
    ("Entropy and temperature", "Inner temperature before solve", "train/inner_alpha_initial", "env_step"),
    ("Entropy and temperature", "Inner temperature after solve", "train/inner_alpha_final", "env_step"),
    ("Learning losses", "Outer reward-return critic loss", "train/critic_loss", "env_step"),
    ("Learning losses", "Inner reward-return critic loss", "train/inner_critic_loss", "env_step"),
    ("Learning losses", "Outer actor loss", "train/actor_loss", "env_step"),
    ("Learning losses", "Inner actor loss", "train/inner_actor_loss", "env_step"),
    ("Learning losses", "Latent consistency loss", "train/consistency_loss", "env_step"),
    ("Learning losses", "Reward-model loss", "train/reward_loss", "env_step"),
    ("Solve and held-action cadence", "Solves per logging window", "train/inner_solve_performed", "env_step"),
    ("Solve and held-action cadence", "Held-policy decisions per logging window", "train/inner_policy_held", "env_step"),
    ("Solve and held-action cadence", "Solve seconds per logging window", "time/inner_solve_seconds", "env_step"),
    ("Solve and held-action cadence", "Held-action seconds per logging window", "time/inner_held_seconds", "env_step"),
]


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _view_id(url):
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if parsed.hostname == "forge.coreweave.com" and path.startswith("/wandb/"):
        path = path[len("/wandb"):]
    identifiers = parse_qs(parsed.query).get("nw", [])
    if (parsed.scheme != "https" or parsed.hostname not in {"wandb.ai", "forge.coreweave.com"}
            or path != f"/{ENTITY}/{PROJECT}" or len(identifiers) != 1
            or not re.fullmatch(r"[A-Za-z0-9_-]+", identifiers[0])):
        raise ValueError("Receipt must identify this project's named workspace.")
    return identifiers[0]


def _validate_run(run, rounds):
    config = dict(run.config)
    params = {**config.get("alg_params", {}), **config.get("config", {})}
    expected = {"inner_operator": "sac", "inner_rollout_horizon": 3,
                "inner_rounds": rounds, "inner_solve_interval": 3,
                "aux_return_mode": "off", "outer_critic_target": "reward_only",
                "inner_sac_critic_target": "reward_only"}
    for key, value in expected.items():
        if params.get(key) != value:
            raise ValueError(f"Run {run.id} does not match its J{rounds} label: {key}.")
    seed = config.get("run_params", {}).get("seed", params.get("seed"))
    if seed != 55:
        raise ValueError(f"Run {run.id} does not have seed55.")


def discover_rows(api):
    baseline = api.run(f"{ENTITY}/{PROJECT}/{BASELINE}")
    if baseline.id != BASELINE:
        raise ValueError("Reference run ID differs from the authorized baseline.")
    _validate_run(baseline, 8)
    candidates = []
    for run in api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}, per_page=2):
        candidates.append(run)
        if len(candidates) > 1:
            raise ValueError("Multiple runs claim the J10 campaign; refusing an ambiguous comparison.")
    if candidates:
        _validate_run(candidates[0], 10)
        if candidates[0].id == BASELINE:
            raise ValueError("J8 and J10 must have distinct run identities.")
    rows = []
    for rounds, run in ((8, baseline), (10, candidates[0] if candidates else None)):
        summary = dict(run.summary) if run else {}
        rows.append({"rounds": rounds, "run_id": run.id if run else None,
                     "name": LABELS[rounds], "color": COLORS[rounds],
                     "state": run.state if run else "pending W&B startup",
                     "decisions": summary.get("env_step", summary.get("time/total_timesteps")),
                     "last_return": summary.get("episode/return")})
    return rows


def _markdown(rows, updated):
    table = "\n".join(f"| {row['name'].replace(' | ', ' / ')} | {row['state']} | "
                      f"{row['decisions'] if row['decisions'] is not None else 'pending'} |" for row in rows)
    return f"""**Fresh return-only training: H3/K3, J8 versus J10, seed55.**

Blue is the existing J8 reference; orange is the new J10 run. Both use one primary
reward-return critic, no auxiliary critic, and automatic actor temperatures with
target entropy −10.5. Outer and inner critic targets are reward-only. The new run
changes the solve rounds from 8 to 10; remaining algorithm settings match the reference.
The imagined replay capacity increases from 3,072 to 3,840 as required by J × N × H
so both retain every imagined transition. Both target 500,000 decisions
(1,000,000 raw environment steps at action repeat 2).

K3 solves at episode decisions 0, 3, 6, … and holds the adapted actor and solve-time
encoder for the next two decisions. Held actions use the new observation and a
fresh action sample. Outer collection and training continue every decision.
Return is actual stochastic training collection return, not a separate evaluation
or a paired base-policy improvement estimate. These are single-seed results.

Curves are separate native runs with no smoothing or cross-run averaging. Hover
over a curve or its legend to identify it. Controller cost includes solve and
held decisions; inner losses and entropy summarize solve updates. Cadence counts
and times are totals per logging window. There are no MPPI curves.

**Upload snapshot: {updated}.** Empty plots mean pending observations, never zero
performance. Startup includes warmup, pretraining and compilation; exact-resume
metrics upload at committed boundaries. Refresh this installer after J10 starts
to bind its run ID, color and label and update this timestamped table.

| Run | Status | Uploaded decisions |
|---|---|---|
{table}
"""


def _filter_wire(value, expr):
    if isinstance(value, str):
        model = expr.expr_to_filters(value)
    else:
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError("Expected one comparison filter.")
            value = value[0]
        model = value.to_model()
    wire = model.model_dump(exclude_none=True) if hasattr(model, "model_dump") else model.dict(exclude_none=True)

    def normalize(node):
        if isinstance(node, list):
            return [normalize(item) for item in node]
        if isinstance(node, dict):
            node = {key: normalize(item) for key, item in node.items() if not (key == "disabled" and item is False)}
            if node.get("op") in {"AND", "OR"} and len(node.get("filters", [])) == 1 and "key" not in node:
                return node["filters"][0]
        return node
    return normalize(wire)


def install(output, *, api=None, backend=None):
    output = Path(output)
    previous = json.loads(output.read_text()) if output.exists() else {}
    if previous and (previous.get("entity"), previous.get("project"), previous.get("group"), previous.get("baseline")) != (ENTITY, PROJECT, GROUP, BASELINE):
        raise ValueError("Receipt belongs to another comparison.")
    if previous.get("url"):
        _view_id(previous["url"])
    if backend is None:
        for package, version in (("wandb", "0.28.1"), ("wandb-workspaces", "0.4.5")):
            if importlib.metadata.version(package) != version:
                raise RuntimeError(f"Use isolated {package}=={version}.")
        import wandb
        import wandb_workspaces.workspaces as ws
        import wandb_workspaces.reports.v2 as wr
        from wandb_workspaces import expr
        api = api or wandb.Api(timeout=30)
    else:
        ws, wr, expr = backend
    rows = discover_rows(api)
    old_ids = {row["rounds"]: row["run_id"] for row in previous.get("rows", []) if row["run_id"]}
    if any(old_ids.get(row["rounds"], row["run_id"]) != row["run_id"] for row in rows):
        raise ValueError("A previously published run disappeared or changed identity.")
    active = [row for row in rows if row["run_id"]]
    updated = datetime.now(timezone.utc).isoformat()
    receipt = {"entity": ENTITY, "project": PROJECT, "group": GROUP, "baseline": BASELINE,
               "name": NAME, "url": previous.get("url"), "rows": rows,
               "api_verified": False, "browser_verified": False,
               "updated_at": updated, "panel_count": len(METRICS) + 1}
    sections = [ws.Section(name=PREFIX + "Protocol and upload status", is_open=True,
        panels=[wr.MarkdownPanel(markdown=_markdown(rows, updated), layout=wr.Layout(w=24, h=14))])]
    for category in dict.fromkeys(item[0] for item in METRICS):
        panels = [wr.LinePlot(title=title, x=axis, y=[metric], aggregate=False,
            smoothing_type="none", legend_template="${run:name}", max_runs_to_show=2,
            title_x="Elapsed seconds" if axis == "time/time_elapsed" else "Training decisions",
            line_colors={row["run_id"] + ":" + metric: row["color"] for row in active},
            line_titles={row["run_id"] + ":" + metric: row["name"] for row in active},
            line_marks={row["run_id"] + ":" + metric: "solid" for row in active},
            layout=wr.Layout(w=12, h=8)) for section, title, metric, axis in METRICS if section == category]
        sections.append(ws.Section(name=PREFIX + category, is_open=True, panels=panels))
    work = ws.Workspace.from_url(receipt["url"]) if receipt["url"] else ws.Workspace(
        entity=ENTITY, project=PROJECT, name=NAME, auto_generate_panels=False)
    if (work.entity, work.project, work.name) != (ENTITY, PROJECT, NAME):
        raise ValueError("Saved URL points to an unrelated workspace.")
    work.sections = [section for section in work.sections if not section.name.startswith(PREFIX)] + sections
    wanted = expr.Metric("ID").isin([row["run_id"] for row in active])
    work.runset_settings = ws.RunsetSettings(filters=[wanted], groupby=[],
        run_settings={row["run_id"]: ws.RunSettings(color=row["color"], disabled=False) for row in active},
        pinned_columns=["run:displayName", "summary:env_step", "summary:episode/return"])
    work.settings = ws.WorkspaceSettings(x_axis="env_step", max_runs=2,
        tooltip_number_of_runs="all_runs", tooltip_color_run_names=True, remove_legends_from_panels=False)
    work._auto_generate_panels = False
    _write(output, receipt)
    try:
        work.save()
        receipt["url"], receipt["view_id"] = work.url, _view_id(work.url)
        _write(output, receipt)
        saved = ws.Workspace.from_url(work.url)
        owned = [section for section in saved.sections if section.name.startswith(PREFIX)]
        if (saved.entity, saved.project, saved.name) != (ENTITY, PROJECT, NAME) or _view_id(saved.url) != receipt["view_id"]:
            raise RuntimeError("Workspace read-back changed its identity.")
        if len(owned) != len(sections) or sum(len(section.panels) for section in owned) != len(METRICS) + 1:
            raise RuntimeError("Workspace read-back lost comparison panels.")
        if saved.runset_settings.groupby or saved.runset_settings.query or _filter_wire(saved.runset_settings.filters, expr) != _filter_wire(wanted, expr):
            raise RuntimeError("Workspace read-back changed the exact independent run selection.")
        if saved.settings.remove_legends_from_panels or not saved.settings.tooltip_color_run_names or saved.settings.tooltip_number_of_runs != "all_runs":
            raise RuntimeError("Workspace read-back lost native legend/hover settings.")
        expected = {metric: (title, axis) for _, title, metric, axis in METRICS}
        seen = set()
        for section in owned:
            for panel in section.panels:
                if not isinstance(panel, wr.LinePlot):
                    continue
                metric = getattr(panel.y[0], "name", panel.y[0]) if len(panel.y) == 1 else None
                if metric not in expected or metric in seen or panel.aggregate or panel.smoothing_type != "none":
                    raise RuntimeError("Workspace read-back changed native individual curve semantics.")
                seen.add(metric)
                if (panel.title, getattr(panel.x, "name", panel.x)) != expected[metric]:
                    raise RuntimeError("Workspace read-back changed a metric or axis.")
                for row in active:
                    key = row["run_id"] + ":" + metric
                    color = panel.line_colors.get(key)
                    color = color.get("color") if isinstance(color, dict) else color
                    setting = saved.runset_settings.run_settings.get(row["run_id"])
                    if (color != row["color"] or panel.line_titles.get(key) != row["name"]
                            or panel.line_marks.get(key) != "solid" or setting is None
                            or setting.disabled or setting.color != row["color"]):
                        raise RuntimeError("Workspace read-back lost an exact label, color or visible line.")
        if seen != set(expected):
            raise RuntimeError("Workspace read-back is missing required metrics.")
        receipt["api_verified"] = True
    except Exception as error:
        receipt["error_type"] = type(error).__name__
        _write(output, receipt)
        raise
    _write(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        if args.output.exists():
            raise ValueError("Dry-run output must be new; preserve live workspace receipts.")
        result = {"entity": ENTITY, "project": PROJECT, "group": GROUP, "baseline": BASELINE,
                  "name": NAME, "url": None, "api_verified": False, "dry_run": True,
                  "metrics": METRICS, "panel_count": len(METRICS) + 1}
        _write(args.output, result)
    else:
        result = install(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
