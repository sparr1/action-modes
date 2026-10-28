"""Idempotent native J/H scaling comparison in its own named W&B workspace.

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
GROUP = "ambi-return-jh-scaling-20261028"
BASELINE = "de37a4eec8094c67a2a691db04e47a2b"
NAME = "Return-only training: J12 and H4/H5 scaling"
PREFIX = "Return-only JH scaling / "
VARIANTS = {
    "baseline": {"horizon": 3, "rounds": 10, "interval": 3, "color": "#0072B2"},
    "h3_j12_k3": {"horizon": 3, "rounds": 12, "interval": 3, "color": "#D55E00"},
    "h4_j10_k4": {"horizon": 4, "rounds": 10, "interval": 4, "color": "#009E73"},
    "h5_j10_k5": {"horizon": 5, "rounds": 10, "interval": 5, "color": "#CC79A7"},
}
for _spec in VARIANTS.values():
    _spec["label"] = (f"H{_spec['horizon']} J{_spec['rounds']} K{_spec['interval']}"
                      " | reward-only | seed55")

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


def _validate_run(run, variant):
    spec = VARIANTS[variant]
    config = dict(run.config)
    params = {**config.get("alg_params", {}), **config.get("config", {})}
    expected = {
        "inner_operator": "sac", "inner_rollout_horizon": spec["horizon"],
        "inner_rounds": spec["rounds"], "inner_solve_interval": spec["interval"],
        "train_unroll_horizon": 3, "outer_planning_horizon": 3,
        "inner_rollouts_per_round": 128, "inner_critic_updates_per_round": 16,
        "inner_actor_updates_per_round": 4, "inner_batch_size": 256,
        "inner_replay_capacity": spec["horizon"] * spec["rounds"] * 128,
        "aux_return_mode": "off", "outer_critic_target": "reward_only",
        "inner_sac_critic_target": "reward_only", "ent_coef": "auto_1.0",
        "target_entropy": -10.5, "inner_temperature_mode": "auto",
        "inner_target_entropy": "inherit_outer", "inner_actor_adaptation": "clone",
        "inner_critic_adaptation": "clone", "inner_hold_encoder": "solve_snapshot",
        "inner_actor_initialization": "prior", "inner_critic_initialization": "prior",
        "inner_execution_action": "policy_sample", "inner_behavior_action": "policy_sample",
    }
    for key, value in expected.items():
        if params.get(key) != value:
            raise ValueError(f"Run {run.id} does not match {variant}: {key}.")
    run_params = config.get("run_params", {})
    if run_params.get("seed", params.get("seed")) != 55:
        raise ValueError(f"Run {run.id} does not have seed55.")
    if run_params.get("total_steps", params.get("steps")) != 1000000:
        raise ValueError(f"Run {run.id} does not target one million decisions.")
    if variant != "baseline" and params.get("wandb_group") != GROUP:
        raise ValueError(f"Run {run.id} has the wrong campaign group.")


def _expected_ids(value):
    if value is None:
        return {}
    if not isinstance(value, dict) or set(value) != set(VARIANTS) - {"baseline"}:
        raise ValueError("Expected IDs must identify exactly the three new variants.")
    ids = list(value.values())
    if (len(set(ids)) != 3 or BASELINE in ids or any(
            not isinstance(item, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", item) for item in ids)):
        raise ValueError("Expected IDs must be distinct valid run identifiers.")
    return dict(value)


def discover_rows(api, *, expected_ids=None):
    expected_ids = _expected_ids(expected_ids) if expected_ids else {}
    baseline = api.run(f"{ENTITY}/{PROJECT}/{BASELINE}")
    if baseline.id != BASELINE:
        raise ValueError("Reference run ID differs from the authorized baseline.")
    _validate_run(baseline, "baseline")
    matched = {"baseline": baseline}
    identities = {BASELINE}
    for index, run in enumerate(api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}, per_page=4)):
        if index >= 3:
            raise ValueError("More than three runs claim this campaign; refusing an ambiguous comparison.")
        config = dict(run.config)
        params = {**config.get("alg_params", {}), **config.get("config", {})}
        shape = (params.get("inner_rollout_horizon"), params.get("inner_rounds"),
                 params.get("inner_solve_interval"))
        variants = [key for key, spec in VARIANTS.items() if key != "baseline" and
                    shape == (spec["horizon"], spec["rounds"], spec["interval"])]
        if len(variants) != 1:
            raise ValueError(f"Run {run.id} claims the campaign with an unapproved H/J/K tuple.")
        variant = variants[0]
        _validate_run(run, variant)
        if variant in matched or run.id in identities:
            raise ValueError("Duplicate variant or run identity in the campaign.")
        if expected_ids and expected_ids[variant] != run.id:
            raise ValueError(f"Run {run.id} differs from the launch identity for {variant}.")
        matched[variant] = run
        identities.add(run.id)
    rows = []
    for variant, spec in VARIANTS.items():
        run = matched.get(variant)
        summary = dict(run.summary) if run else {}
        rows.append({"variant": variant, "horizon": spec["horizon"], "rounds": spec["rounds"],
                     "interval": spec["interval"], "run_id": run.id if run else None,
                     "name": spec["label"], "color": spec["color"],
                     "state": run.state if run else "pending W&B startup",
                     "decisions": summary.get("env_step", summary.get("time/total_timesteps")),
                     "last_return": summary.get("episode/return")})
    return rows


def _markdown(rows, updated):
    table = "\n".join(f"| {row['name'].replace(' | ', ' / ')} | {row['state']} | "
                      f"{row['decisions'] if row['decisions'] is not None else 'pending'} |" for row in rows)
    return f"""**Reward-only training: J and inner-rollout-horizon scaling, seed55.**

Blue is the existing H3/J10/K3 reference ({BASELINE}); orange is H3/J12/K3,
green is H4/J10/K4, and purple is H5/J10/K5. The three new runs start fresh
and target 1,000,000 decisions (2,000,000 raw environment steps at action repeat 2).
The baseline is its existing continued run. Each uses C16/A4/N128/B256,
one primary reward-return critic, no auxiliary critic, fresh prior-initialized
inner actor/critic clones, and automatic actor temperatures with target entropy −10.5.
Inner and outer critic targets are reward-only. These are single-seed comparisons.

**H denotes only the inner imagined rollout horizon.** Model training unroll and
outer planning horizons remain 3 in every run. K equals H: the adapted feedback
policy and solve-time encoder are held between solves while processing new
observations and sampled actions. This couples longer rollout horizon with longer
policy holding and fewer optimizer updates per real decision. At J10, nominal model
transitions per decision stay 1,280, but critic updates per decision fall from
160/3 to 160/4 to 160/5. Replay retains every solve transition: 3,840 baseline,
4,608 J12/H3, 5,120 J10/H4, and 6,400 J10/H5. J12 increases both model and update work.

**These are stochastic training episode returns, not K1 checkpoint evaluations or
paired policy-prior improvement estimates.** The four runs remain separate native
curves with no smoothing or cross-run averaging. Hover or use the legend to identify
each curve. Controller costs include solve and held decisions; inner losses and
entropy summarize solve updates. Cadence counts/times are logging-window totals.

**Upload snapshot: {updated}.** Missing runs are explicitly pending below; empty
plots mean pending observations, never zero performance. Startup includes warmup,
pretraining and compilation. Exact-resume metrics upload at committed boundaries.
Refresh the installer after startup to bind exact run IDs, colors and labels.

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


def install(output, *, api=None, backend=None, expected_ids=None):
    output = Path(output)
    previous = json.loads(output.read_text()) if output.exists() else {}
    if previous and (previous.get("entity"), previous.get("project"), previous.get("group"), previous.get("baseline")) != (ENTITY, PROJECT, GROUP, BASELINE):
        raise ValueError("Receipt belongs to another comparison.")
    if previous.get("url"):
        _view_id(previous["url"])
    elif previous.get("creation_intent"):
        raise RuntimeError("A prior workspace creation has an unknown outcome; recover its URL before retrying.")
    expected_ids = _expected_ids(expected_ids) if expected_ids is not None else previous.get("expected_ids", {})
    if previous.get("expected_ids") and expected_ids != previous["expected_ids"]:
        raise ValueError("Expected launch identities changed from the previous receipt.")
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
    rows = discover_rows(api, expected_ids=expected_ids)
    old_ids = {row["variant"]: row["run_id"] for row in previous.get("rows", []) if row["run_id"]}
    if any(old_ids.get(row["variant"], row["run_id"]) != row["run_id"] for row in rows):
        raise ValueError("A previously published run disappeared or changed identity.")
    active = [row for row in rows if row["run_id"]]
    updated = datetime.now(timezone.utc).isoformat()
    receipt = {"entity": ENTITY, "project": PROJECT, "group": GROUP, "baseline": BASELINE,
               "name": NAME, "url": previous.get("url"), "rows": rows,
               "expected_ids": expected_ids, "creation_intent": not bool(previous.get("url")),
               "api_verified": False, "browser_verified": False,
               "updated_at": updated, "panel_count": len(METRICS) + 1}
    sections = [ws.Section(name=PREFIX + "Protocol and upload status", is_open=True,
        panels=[wr.MarkdownPanel(markdown=_markdown(rows, updated), layout=wr.Layout(w=24, h=20))])]
    for category in dict.fromkeys(item[0] for item in METRICS):
        panels = [wr.LinePlot(title=title, x=axis, y=[metric], aggregate=False,
            smoothing_type="none", legend_template="${run:name}", max_runs_to_show=4,
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
    work.settings = ws.WorkspaceSettings(x_axis="env_step", max_runs=4,
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
        if (saved.settings.max_runs != 4 or saved.settings.remove_legends_from_panels
                or not saved.settings.tooltip_color_run_names
                or saved.settings.tooltip_number_of_runs != "all_runs"):
            raise RuntimeError("Workspace read-back lost native legend/hover settings.")
        expected = {metric: (title, axis) for _, title, metric, axis in METRICS}
        seen = set()
        for section in owned:
            for panel in section.panels:
                if not isinstance(panel, wr.LinePlot):
                    continue
                metric = getattr(panel.y[0], "name", panel.y[0]) if len(panel.y) == 1 else None
                if (metric not in expected or metric in seen or panel.aggregate
                        or panel.smoothing_type != "none" or panel.max_runs_to_show != 4):
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
    parser.add_argument("--run-ids", type=Path, help="JSON mapping of the three variant keys to exact launch run IDs.")
    args = parser.parse_args()
    if args.dry_run:
        if args.output.exists():
            raise ValueError("Dry-run output must be new; preserve live workspace receipts.")
        result = {"entity": ENTITY, "project": PROJECT, "group": GROUP, "baseline": BASELINE,
                  "name": NAME, "url": None, "api_verified": False, "dry_run": True,
                  "metrics": METRICS, "panel_count": len(METRICS) + 1}
        _write(args.output, result)
    else:
        result = install(args.output, expected_ids=json.loads(args.run_ids.read_text()) if args.run_ids else None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
