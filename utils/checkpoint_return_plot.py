"""W&B checkpoint returns with individual evaluation episodes and mean lines."""

import hashlib
import json
import math
from pathlib import Path
import statistics

from utils.evaluation_episodes import validate_consecutive_episodes

TABLE_KEY = "eval/episode_returns_table"
COLUMNS = ["training_decisions", "controller", "environment_seed", "episode_index", "episode_return", "episode_count"]
TITLE = "Full-episode return · individual episodes and mean"
LABELS = {"controller/prior": "Prior", "controller/graph": "H3/J8 inner SAC"}


def episode_return_rows(records):
    """Retain each measured episode; reject incomplete or duplicate cohorts."""
    rows = []
    seen = set()
    cohort = None
    for record in sorted(records, key=lambda r: r["checkpoint"]["step"]):
        protocol = record["identity"]["protocol"]
        sampling = protocol.get("episode_sampling")
        if sampling is not None:
            episodes = record["episodes"]
            validate_consecutive_episodes(episodes, sampling, protocol["controller_seed"])
            current_cohort = sampling
        else:
            episodes = sorted(record["episodes"], key=lambda e: e["seed"])
            expected = protocol["environment_seeds"]
            if not expected or [e["seed"] for e in episodes] != sorted(set(expected)):
                raise ValueError("Return plot requires all paired episode seeds")
            current_cohort = sorted(expected)
        if cohort is not None and current_cohort != cohort:
            raise ValueError("Return plot cannot mix episode sampling protocols")
        cohort = current_cohort
        values = [e["return"] for e in episodes]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Episode returns must be finite")
        if not math.isclose(statistics.mean(values), record["metrics"]["eval/return_mean"], abs_tol=1e-8):
            raise ValueError("Episode returns disagree with the published mean")
        for index, episode in enumerate(episodes, start=1):
            step = record["checkpoint"]["step"]
            seed = episode["environment_seed"] if sampling is not None else episode["seed"]
            label = LABELS[record["selector"]]
            key = (step, label, index)
            if key in seen:
                raise ValueError("Duplicate checkpoint episode")
            seen.add(key)
            rows.append([step, label, seed, index, episode["return"], len(episodes)])
    return rows


def publish_episode_returns(run, run_dir):
    """Add a summary table without inserting or rewriting scalar history rows."""
    import wandb
    records = [json.loads(p.read_text()) for p in (Path(run_dir) / "records").glob("*.json")]
    rows = episode_return_rows(records)
    digest = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    if run.summary.get("evaluation/episode_table_sha256") != digest:
        run.summary[TABLE_KEY] = wandb.Table(columns=COLUMNS, data=rows)
        run.summary["evaluation/episode_table_sha256"] = digest
    return len(rows)


def return_episode_spec(runs):
    colors = {r["selector"]: r["color"] for r in runs}
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": TITLE,
        "description": "Dots are individual episode returns; lines average all episodes. Horizontal offsets only separate overlapping dots.",
        "data": {"name": "wandb"},
        "transform": [{"calculate": "(datum['${field:episode}'] - (datum['${field:count}'] + 1) / 2) * 12 / max(1, datum['${field:count}'] - 1)", "as": "episode_offset"}],
        "encoding": {
            "x": {"field": "${field:x}", "type": "quantitative",
                  "title": "Training decisions at checkpoint", "axis": {"format": "~s"}},
            "color": {"field": "${field:controller}", "type": "nominal", "title": None,
                      "scale": {"domain": list(LABELS.values()),
                                "range": [colors.get("controller/prior", "#6B7280"),
                                          colors.get("controller/graph", "#0F766E")]},
                      "legend": {"orient": "bottom"}},
        },
        "layer": [
            {"mark": {"type": "line", "strokeWidth": 2},
             "encoding": {
                 "y": {"field": "${field:return}", "aggregate": "mean", "type": "quantitative",
                       "title": "Episode return", "scale": {"domain": [0, 1000], "nice": False}},
                 "order": {"field": "${field:x}", "type": "quantitative"},
                 "tooltip": [
                     {"field": "${field:controller}", "type": "nominal", "title": "Controller"},
                     {"field": "${field:x}", "type": "quantitative", "title": "Training decisions", "format": ","},
                     {"field": "${field:return}", "aggregate": "mean", "type": "quantitative", "title": "Mean episode return", "format": ".2f"},
                 ]}},
            {"mark": {"type": "point", "filled": True, "size": 42, "opacity": 0.65,
                      "xOffset": {"expr": "datum.episode_offset"}},
             "encoding": {
                 "y": {"field": "${field:return}", "type": "quantitative", "title": "Episode return"},
                 "tooltip": [
                     {"field": "${field:controller}", "type": "nominal", "title": "Controller"},
                     {"field": "${field:x}", "type": "quantitative", "title": "Training decisions", "format": ","},
                     {"field": "${field:seed}", "type": "ordinal", "title": "Environment / stream seed"},
                     {"field": "${field:episode}", "type": "ordinal", "title": "Episode"},
                     {"field": "${field:return}", "type": "quantitative", "title": "Episode return", "format": ".2f"},
                 ]}},
        ],
    }


def return_episode_panel(runs, entity, cached=None):
    """Create a reusable private preset backed by each run's episode table."""
    import wandb
    import wandb_workspaces.reports.v2 as wr

    spec = return_episode_spec(runs)
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    receipt = cached if cached and cached.get("spec_sha256") == digest else None
    if receipt is None:
        chart_id = wandb.Api().create_custom_chart(
            entity=entity, name="checkpoint-return-episodes-" + digest[:16],
            display_name=TITLE, spec_type="vega2", access="private", spec=spec)
        receipt = {"chart_id": chart_id, "spec_sha256": digest}
    panel = wr.CustomChart(
        query={"summaryTable": {"tableKey": TABLE_KEY}}, chart_name=receipt["chart_id"],
        chart_fields=dict(zip(["x", "controller", "seed", "episode", "return", "count"], COLUMNS)),
        layout=wr.Layout(w=24, h=10))
    return panel, receipt
