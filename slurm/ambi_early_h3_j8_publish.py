"""Durable publication for the early-checkpoint H3 J8 evaluation.

Prepare once from clean-checkout evaluator specifications, then run ``publish
--once`` periodically from a single CPU publisher. GPU workers remain offline.
The remote publisher supports the locked W&B SDK with WANDB__REQUIRE_CORE=true.
The one-time ``layout`` command uses wandb==0.28.1 and wandb-workspaces==0.4.5
in a separate runtime; copy its verified workspace.json receipt to the remote
publisher root before publication. It does not change the locked environment.
Authentication comes from
the process environment or the normal SDK configuration, never campaign files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import re
import statistics
import sys
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import eval_series as series

ENTITY = "rwgao_b-brown-university"
PROJECT = "ambi-inner-bench"
BACKBONES = {
    "mppi_aux": "rwgao_b-brown-university/ambi/mppi7530927x0",
    "sac_aux": "rwgao_b-brown-university/ambi/full7391933x4",
}
STEPS = [100000, 150000, 200000, 400000]
CURVES = {
    "mppi_aux_mean": ("MPPI-trained backbone | H3 J8 | mean", "#2563EB"),
    "mppi_aux_sample": ("MPPI-trained backbone | H3 J8 | sample", "#14B8A6"),
    "sac_aux_mean": ("SAC-trained backbone | H3 J8 | mean", "#E87924"),
    "sac_aux_sample": ("SAC-trained backbone | H3 J8 | sample", "#A855F7"),
}
PREFIX = "Early H3 J8 / "
METRICS = {
    "Return and runtime": [
        ("Episode return — mean of five seeds", "eval/return_mean"),
        ("Episode return — sample standard deviation", "eval/return_sample_std"),
        ("Controller seconds per decision", "runtime/control_seconds_per_decision"),
        ("Completed evaluation episodes", "eval/episodes"),
    ],
    "Inner diagnostics": [
        ("Inner critic loss", "diagnostics/inner_critic_loss_mean"),
        ("Inner actor entropy", "diagnostics/inner_actor_entropy_mean"),
        ("Inner final temperature", "diagnostics/inner_alpha_final_mean"),
        ("Critic optimizer steps across five episodes", "work/critic_updates"),
    ],
    "Predicted model probes": [
        ("Predicted return gain vs initialization — final round", "diagnostics/inner_togo_return_gain_vs_initial_mean"),
        ("Predicted return gain vs outer policy — final round", "diagnostics/inner_togo_return_gain_vs_outer_mean"),
        ("Predicted discounted reward component — final round", "diagnostics/inner_togo_reward_mean_mean"),
        ("Predicted terminal bootstrap component — final round", "diagnostics/inner_togo_bootstrap_mean_mean"),
    ],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def curve_key(cell):
    return cell.get("curve_key", cell["backbone_id"] + "_" + cell["execution"])


def curve_definitions(campaign):
    """Derive display identity from the pinned manifest, never from run names."""
    sources = campaign.get("sources", {key: {"source_run": source} for key, source in BACKBONES.items()})
    result = {}
    palette = [("#2563EB", "#14B8A6"), ("#E87924", "#A855F7")]
    for number, (backbone, source) in enumerate(sources.items()):
        label = source.get("label", backbone.replace("_", " "))
        for execution, color in zip(("mean", "sample"), palette[number % len(palette)]):
            key = backbone + "_" + execution
            legacy = CURVES.get(key) if "sources" not in campaign else None
            result[key] = {"name": legacy[0] if legacy else f"{label} | H3 J8 | {execution}",
                           "color": legacy[1] if legacy else color, "role": "inner",
                           "backbone_id": backbone, "source_run": source["source_run"], "execution": execution}
            if campaign.get("baseline_root"):
                result[backbone + "_base_" + execution] = {
                    "name": f"{label} | Base policy | {execution}",
                    "color": "#E87924" if execution == "mean" else "#A855F7", "role": "base",
                    "backbone_id": backbone, "source_run": source["source_run"], "execution": execution}
    return result


def baseline_cell(campaign, cell):
    return {**cell, "role": "base", "curve_key": cell["backbone_id"] + "_base_" + cell["execution"],
            "selector": "early_prior/" + cell["execution"],
            "output_dir": str(Path(campaign["baseline_root"]) / f'cell-{cell["index"]:02d}')}


def publication_cells(campaign):
    cells = campaign["cells"]
    return cells + [baseline_cell(campaign, cell) for cell in cells] if campaign.get("baseline_root") else cells


def metric_sections(campaign):
    result = {key: list(value) for key, value in METRICS.items()}
    if campaign.get("baseline_root"):
        result["Paired improvement over matching base"] = [
            ("Paired return improvement — mean of five seeds", "eval/paired_gain_mean"),
            ("Paired improvement — sample standard deviation", "eval/paired_gain_sample_std"),
            ("Seeds improved over base — fraction", "eval/paired_win_fraction"),
            ("Matched baseline episodes", "eval/paired_episodes"),
        ]
    return result


def load_campaign(path):
    campaign = series._read(path)
    if campaign.get("selection"):
        from slurm.ambi_early_h3_j8_campaign import validate_campaign_selection
        validate_campaign_selection(campaign)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", campaign.get("campaign_id", "")):
        raise ValueError("Campaign requires a stable publication identifier")
    if campaign.get("seeds") != [101, 102, 103, 104, 105] or campaign.get("controller_seed") != 55 or campaign.get("max_steps") != 500:
        raise ValueError("Campaign must preserve the authorized five-seed episode protocol")
    if not isinstance(campaign.get("source_commit"), str) or len(campaign["source_commit"]) != 40:
        raise ValueError("A full committed source SHA is required")
    cells = campaign.get("cells", [])
    sources = campaign.get("sources", {key: {"source_run": source} for key, source in BACKBONES.items()})
    steps = campaign.get("steps", STEPS)
    if not sources or not steps or len(set(steps)) != len(steps) or any(type(step) is not int or step < 0 for step in steps):
        raise ValueError("Manifest must declare distinct checkpoint steps and source backbones")
    wanted = {(backbone, execution, step) for backbone in sources
              for execution in ("mean", "sample") for step in steps}
    actual = {(cell["backbone_id"], cell["execution"], cell["step"]) for cell in cells}
    if len(cells) != len(wanted) or actual != wanted or len({cell["index"] for cell in cells}) != len(cells):
        raise ValueError("Evaluation cells must exactly cover the declared source/action/checkpoint matrix")
    for cell in cells:
        if cell["source_run"] != sources[cell["backbone_id"]]["source_run"] or cell["selector"] != "early_h3_j8/" + cell["execution"]:
            raise ValueError("Cell source or selector differs from the authorized comparison")
        for key in ("checkpoint_sha256", "metadata_sha256"):
            value = cell[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Each checkpoint and sidecar requires its SHA256")
        for key in ("checkpoint", "metadata", "output_dir"):
            if not Path(cell[key]).is_absolute():
                raise ValueError("Campaign paths must be absolute")
    if len({cell["output_dir"] for cell in cells}) != len(cells):
        raise ValueError("Each evaluation cell must own a distinct output directory")
    if campaign.get("baseline_root") and not Path(campaign["baseline_root"]).is_absolute():
        raise ValueError("Baseline root must be absolute")
    return campaign


def validate_spec(campaign, cell, spec):
    identity = spec["identity"]
    base = cell.get("role") == "base"
    if (not base and spec.get("selector") != cell["selector"]) or identity["backbone"] != cell["source_run"]:
        raise ValueError("Specification source/selector does not match its campaign cell")
    action = "tanh_" + cell["execution"]
    protocol = identity["protocol"]
    if identity["planner"].get("action_rule") != action or protocol.get("action_rule") != action:
        raise ValueError("Specification does not describe the actual execution action")
    if protocol.get("environment_seeds") != campaign["seeds"] or protocol.get("max_steps") != 500 or protocol.get("controller_seed") != 55:
        raise ValueError("Specification episode protocol does not match campaign")
    if base:
        if identity["planner"].get("type") != "prior":
            raise ValueError("Baseline specification must be a no-adaptation prior")
        # Reused controls retain their audited historical scientific identity.
        return
    from utils.eval_series_data import scientific_identity
    # The generic identity contains an executable-source digest, not the Git SHA.
    science = identity["science"]
    expected_science = scientific_identity(science["algorithm"], "sac", campaign["source_commit"])
    if science != expected_science:
        raise ValueError("Specification was not generated from the campaign source commit")
    settings = identity["planner"].get("settings", {})
    if settings.get("inner_rollout_horizon") != 3 or settings.get("inner_rounds") != 8:
        raise ValueError("Specification must be H3 J8")


def publication_prepare(campaign, root, spec_root):
    """Allocate manifest-defined generic registries; never contact W&B here.

    ``spec_root/<backbone>_<execution>/<selector with / replaced by __>.json``
    is written by the evaluator's metadata-only --eval-series-spec-dir mode.
    A worker reads ``cell_run_dirs[str(index)]`` from publication-map.json.
    """
    root, spec_root = Path(root).resolve(), Path(spec_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "publication-map.json"
    with series._lock(root / ".prepare.lock"):
        receipt = series._read(receipt_path) if receipt_path.exists() else {
            "schema_version": 1, "campaign_id": campaign["campaign_id"],
            "campaign_sha256": series._digest(campaign), "curves": {}, "cell_run_dirs": {},
        }
        if receipt["campaign_sha256"] != series._digest(campaign):
            raise ValueError("Prepared campaign changed; do not reuse its publication identity")
        for key, definition in curve_definitions(campaign).items():
            cell = next(cell for cell in publication_cells(campaign) if curve_key(cell) == key)
            candidates = list((spec_root / key).glob("*.json"))
            if cell.get("role") == "base" and not candidates:
                record, _ = completed_record(campaign, cell)
                if record is None:
                    raise ValueError("Base curve needs a specification or a validated reused baseline")
                spec = {"identity": record["identity"], "selector": record["selector"], "label": record["label"]}
            else:
                expected = spec_root / key / (cell["selector"].replace("/", "__") + ".json")
                spec = series._read(candidates[0] if cell.get("role") == "base" and len(candidates) == 1 else expected)
            validate_spec(campaign, cell, spec)
            if key in receipt["curves"]:
                registry = series.validate_identity(receipt["curves"][key]["run_dir"], spec["identity"])
            else:
                # Recover an allocation interrupted before the map was saved.
                owned = list((root / "runs" / key).glob("*/run.json"))
                if len(owned) > 1:
                    raise ValueError("Ambiguous interrupted curve allocation")
                if owned:
                    registry = series.validate_identity(owned[0].parent, spec["identity"])
                else:
                    registry = series.create_run(root / "runs" / key, spec, campaign["campaign_id"],
                                                 PROJECT, ENTITY, campaign["campaign_id"])
                receipt["curves"][key] = {"run_dir": registry["run_dir"], "run_id": registry["run_id"],
                    **definition, "spec_sha256": series._digest(spec)}
                series._atomic_json(receipt_path, receipt)
        receipt["cell_run_dirs"] = {str(cell["index"]): receipt["curves"][curve_key(cell)]["run_dir"]
                                    for cell in campaign["cells"]}
        if campaign.get("baseline_root"):
            receipt["baseline_run_dirs"] = {str(cell["index"]): receipt["curves"][curve_key(baseline_cell(campaign, cell))]["run_dir"]
                                             for cell in campaign["cells"]}
        series._atomic_json(receipt_path, receipt)
    return receipt


def find_manifest(cell):
    output = Path(cell["output_dir"])
    matches = [path for path in (output / "manifest.json", output / "bundle" / "manifest.json") if path.is_file()]
    if len(matches) > 1:
        raise ValueError("Ambiguous cell bundle location")
    return matches[0] if matches else None


def validated_files(cell):
    path = find_manifest(cell)
    receipt_path = Path(cell["output_dir"]) / "receipt.json"
    if path is None or not receipt_path.is_file():
        return None
    validation = series._read(receipt_path)
    if (validation.get("validated") is not True or validation.get("smoke") is not False
            or validation.get("cell_index") != cell["index"]):
        raise ValueError("Only explicitly validated production cell receipts can be published")
    results = Path(cell["output_dir"]) / "results.json"
    if (validation.get("manifest_sha256") != series._file_digest(path)
            or validation.get("results_sha256") != series._file_digest(results)):
        raise ValueError("Validated cell files changed before publication")
    return {"manifest.json": str(path), "results.json": str(results), "validation-receipt.json": str(receipt_path)}


def completed_record(campaign, cell, *, cached=None):
    """Load a validated result, preserving historical baseline identity verbatim."""
    from utils.eval_series_data import load_records
    files = validated_files(cell)
    if files is None:
        return None, "pending"
    if cached is not None:
        if cached["checkpoint"] != {"step": cell["step"], "sha256": cell["checkpoint_sha256"]}:
            raise ValueError("Previously accepted cell changed")
        for name, source in files.items():
            if series._file_digest(source) != cached["_artifact_sha256"].get(name):
                raise ValueError("Previously accepted cell changed")
        return {key: value for key, value in cached.items() if key != "_artifact_sha256"}, "complete"
    path = Path(files["manifest.json"])
    manifest = series._read(path)
    base = cell.get("role") == "base"
    selected = [run for run in manifest["runs"] if base or run["selector"] == cell["selector"]]
    if not selected or (not base and len(selected) != 1):
        raise ValueError("Cell bundle must contain its exact selected planner")
    if not all(run.get("status") == "complete" for run in selected):
        return None, "pending"
    records = load_records(path, source_run=cell["source_run"], inventory_path=cell.get("inventory_path"))
    if base:
        records = [record for record in records if record["identity"]["planner"].get("type") == "prior"
                   and record["identity"]["planner"].get("action_rule") == "tanh_" + cell["execution"]]
    else:
        records = [record for record in records if record.get("selector") == cell["selector"]]
    if len(records) != 1:
        raise ValueError("Completed bundle must normalize to one selected record")
    record = records[0]
    if record["checkpoint"] != {"step": cell["step"], "sha256": cell["checkpoint_sha256"]}:
        raise ValueError("Completed result checkpoint does not match campaign")
    if sorted(ep["seed"] for ep in record["episodes"]) != campaign["seeds"]:
        raise ValueError("Completed result does not contain all five seeds")
    if record["identity"]["backbone"] != cell["source_run"]:
        raise ValueError("Completed result source does not match campaign")
    selected = [run for run in manifest["runs"] if run["selector"] == record["selector"]]
    if len(selected) != 1:
        raise ValueError("Normalized record does not identify exactly one bundle run")
    trace_names = set(selected[0].get("trace_files", []))
    retained, remote_traces = {}, []
    for name, source in record["artifact_files"].items():
        if name in trace_names:
            remote_traces.append({"name": name, "path": source, "sha256": series._file_digest(source),
                                  "size_bytes": Path(source).stat().st_size})
        else:
            retained[name] = source
    retained.update(files)
    record["artifact_files"] = retained
    record["provenance"]["artifact_policy"] = "summary-only; full traces retained on Oscar"
    record["provenance"]["remote_trace_files"] = remote_traces
    return record, "complete"


def pair_with_baseline(record, baseline):
    """Pair environment seeds only; solver streams remain controller-specific."""
    if record["checkpoint"] != baseline["checkpoint"] or record["identity"]["backbone"] != baseline["identity"]["backbone"]:
        raise ValueError("Paired baseline has a different checkpoint or source")
    left, right = record["identity"]["protocol"], baseline["identity"]["protocol"]
    keys = ("environment", "env_wrapper", "env_wrappers", "observation", "action_rule",
            "max_steps", "environment_seeds", "controller_seed")
    if any(left.get(key) != right.get(key) for key in keys):
        raise ValueError("Paired baseline environment/action/seed protocol differs")
    if baseline["identity"]["planner"].get("type") != "prior":
        raise ValueError("Paired baseline must be the frozen base policy")
    base_episodes = {episode["seed"]: episode for episode in baseline["episodes"]}
    if len(base_episodes) != len(baseline["episodes"]) or set(base_episodes) != {ep["seed"] for ep in record["episodes"]}:
        raise ValueError("Paired baseline seed coverage differs")
    gains = []
    for episode in record["episodes"]:
        delta = episode["return"] - base_episodes[episode["seed"]]["return"]
        episode["paired_return_delta"] = delta
        gains.append(delta)
    record["metrics"].update({"eval/paired_gain_mean": statistics.mean(gains),
        "eval/paired_gain_sample_std": statistics.stdev(gains) if len(gains) > 1 else None,
        "eval/paired_win_fraction": sum(gain > 0 for gain in gains) / len(gains),
        "eval/paired_episodes": len(gains), "eval/base_return_mean": baseline["metrics"]["eval/return_mean"]})
    record["provenance"]["paired_baseline"] = {"record_id": baseline["record_id"],
        "identity": baseline["identity"], "checkpoint": baseline["checkpoint"],
        "source_result_path": baseline["source_result_path"],
        "episodes": [{key: ep[key] for key in ("seed", "solver_seed", "return", "length") if key in ep}
                     for ep in baseline["episodes"]],
        "pairing": "same environment seeds and action mode; controller-specific solver streams"}
    record["record_id"] = series._digest({"result": record["record_id"], "paired_baseline": baseline["record_id"]})
    record["artifact_files"].update({"paired-base/" + name: source for name, source in baseline["artifact_files"].items()})
    return record


def status_key(cell):
    return ("base:" if cell.get("role") == "base" else "") + str(cell["index"])


def stage_completed(campaign, receipt):
    """Stage immutable points only after their complete matched controls exist."""
    status, baseline_records = {}, {}
    cells = publication_cells(campaign)
    cells = sorted(cells, key=lambda cell: cell.get("role") != "base")
    for cell in cells:
        base = cell.get("role") == "base"
        mapping = receipt["baseline_run_dirs"] if base else receipt["cell_run_dirs"]
        run_dir = Path(mapping[str(cell["index"])])
        journal = series._read(run_dir / "publication.json")
        entries = [(rid, entry) for rid, entry in journal["records"].items() if entry["checkpoint_step"] == cell["step"]]
        cached = None
        if entries:
            rid, entry = entries[0]
            cached = {**series._read(run_dir / "records" / (rid + ".json")), "_artifact_sha256": entry["artifact_sha256"]}
        record, state = completed_record(campaign, cell, cached=cached)
        if record is None:
            status[status_key(cell)] = state
            continue
        if base:
            baseline_records[cell["index"]] = record
        elif campaign.get("baseline_root"):
            baseline = baseline_records.get(cell["index"])
            if baseline is None:
                status[status_key(cell)] = "waiting for matched base"
                continue
            if cached:
                expected = cached["provenance"].get("paired_baseline", {}).get("record_id")
                if expected != baseline["record_id"]:
                    raise ValueError("Previously accepted paired baseline changed")
                for name, source in baseline["artifact_files"].items():
                    if cached["_artifact_sha256"].get("paired-base/" + name) != series._file_digest(source):
                        raise ValueError("Previously accepted paired baseline artifacts changed")
            else:
                record = pair_with_baseline(record, baseline)
        if not cached:
            if sum(Path(source).stat().st_size for source in record["artifact_files"].values()) > 16 * 1024 * 1024:
                raise ValueError("Summary artifact unexpectedly exceeds 16 MiB; refusing a large upload")
            series.stage_record(run_dir, record)
        status[status_key(cell)] = "complete"
    return status


def progress_rows(campaign, receipt, statuses):
    rows = []
    for key, spec in receipt["curves"].items():
        cells = [cell for cell in publication_cells(campaign) if curve_key(cell) == key]
        journal = series._read(Path(spec["run_dir"]) / "publication.json")
        uploaded = sum(entry["status"] == "published" for entry in journal["records"].values())
        complete = sum(statuses.get(status_key(cell)) == "complete" for cell in cells)
        failed = sum(statuses.get(status_key(cell)) in ("failed", "error") for cell in cells)
        waiting = sum(statuses.get(status_key(cell)) == "waiting for matched base" for cell in cells)
        rows.append({**spec, "curve": key, "complete": complete, "published": uploaded,
                     "expected": len(cells), "failed": failed,
                     "status": "complete" if uploaded == len(cells) else "failed cells" if failed else
                               "waiting for matched base" if waiting else "pending results"})
    return rows


def install_workspace(campaign, receipt, rows, path):
    """Idempotently update only this campaign's dedicated native view."""
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr
    path = Path(path)
    previous = series._read(path) if path.exists() else {}
    if previous and previous.get("campaign_id") != campaign["campaign_id"]:
        raise ValueError("Workspace receipt belongs to a different campaign")
    paired = bool(campaign.get("baseline_root"))
    metrics = metric_sections(campaign)
    title = campaign.get("workspace_title", "Original backbone: H3 J8 vs matched base, mean and sample"
                         if paired else "Early backbones: H3 J8 fresh SAC, mean vs sample")
    comparison = ("H3 J8 curves evaluate fresh auxiliary-return inner SAC. Base-policy curves execute "
                  "the same checkpoint's actor without adaptation. Mean and sampled execution each "
                  "have their own matched base control." if paired else
                  "Source labels identify how each frozen backbone was trained; all curves evaluate "
                  "the same fresh H3 J8 auxiliary-return inner SAC controller.")
    pairing = ("Paired improvement subtracts the matching base return separately for each environment seed. "
               "Controls have the same checkpoint, action mode and episode conditions. Controller RNG "
               "streams remain distinct. Reused controls retain their original scientific provenance. "
               "An improved point waits for its validated base so its gain is published atomically."
               if paired else "There is no paired-prior improvement curve in this campaign.")
    markdown = f"""**Fresh H3 J8 inner SAC on early frozen backbones.**

{comparison} Mean and sampled execution are separate curves, with seeds 101–105,
controller seed 55 and 500 decisions per episode. No outer learning occurs.

The live progress panels and run table show completed checkpoints, out of {len(campaign.get('steps', STEPS))}
per curve. Empty result panels mean pending or unavailable measurements;
pending checkpoints are never represented as zero-return points. Return is the
mean of five episodes; the separate SD panel shows seed variability, not a
confidence interval. Timing includes the actual evaluator's controller cost.
The x-axis is source-training decisions, not raw environment steps. {pairing}
Diagnostic losses are not returns.
Model-probe gains and components are predictions at sampled decision roots after
the final inner round; they are not measured environment improvement. Probes use
paired noise and a common frozen outer critic within each source checkpoint.

Source commit: `{campaign['source_commit']}`. Colors and legends identify individual
runs, with native hover highlighting and no aggregation.
"""
    def plot(title, metric):
        return wr.LinePlot(title=title, x=series.X_AXIS, y=[metric],
            title_x="Source training decisions", aggregate=False, smoothing_type="none",
            legend_template="${run:displayName}", max_runs_to_show=len(rows),
            line_colors={row["run_id"] + ":" + metric: row["color"] for row in rows},
            line_titles={row["run_id"] + ":" + metric: row["name"] for row in rows},
            layout=wr.Layout(w=12, h=8))
    def progress_plot(title, metric):
        return wr.BarPlot(title=title, metrics=[metric], orientation="h",
            aggregate=False, max_runs_to_show=len(rows), max_bars_to_show=len(rows),
            legend_template="${run:displayName}",
            line_colors={row["run_id"] + ":" + metric: row["color"] for row in rows},
            line_titles={row["run_id"] + ":" + metric: row["name"] for row in rows},
            layout=wr.Layout(w=12, h=6))
    sections = [ws.Section(name=PREFIX + "Protocol and progress", is_open=True,
        panels=[wr.MarkdownPanel(markdown=markdown, layout=wr.Layout(w=24, h=10)),
                progress_plot(f"Completed checkpoints per curve (of {len(campaign.get('steps', STEPS))})", "progress/checkpoints_complete"),
                progress_plot("Failed validation cells", "progress/failed_cells")])]
    sections += [ws.Section(name=PREFIX + name, is_open=True,
        panels=[plot(title, metric) for title, metric in specs],
        layout_settings=ws.SectionLayoutSettings(columns=2, rows=2)) for name, specs in metrics.items()]
    if previous.get("url"):
        work = ws.Workspace.from_url(previous["url"])
        if work.entity != ENTITY or work.project != PROJECT:
            raise ValueError("Saved workspace target does not match campaign")
        replacements = {section.name: section for section in sections}
        work.sections = [replacements.pop(section.name, section) for section in work.sections]
        work.sections.extend(replacements.values())
        # Preserve all existing filters and user sections on later updates.
        for row in rows:
            setting = work.runset_settings.run_settings.get(row["run_id"], ws.RunSettings())
            setting.color = row["color"]
            work.runset_settings.run_settings[row["run_id"]] = setting
    else:
        work = ws.Workspace(entity=ENTITY, project=PROJECT,
            name=title, sections=sections,
            auto_generate_panels=False,
            settings=ws.WorkspaceSettings(x_axis=series.X_AXIS, max_runs=len(rows),
                tooltip_number_of_runs="all_runs", tooltip_color_run_names=True,
                remove_legends_from_panels=False),
            runset_settings=ws.RunsetSettings(filters=[expr.Metric("ID").isin([r["run_id"] for r in rows])],
                groupby=[], run_settings={r["run_id"]: ws.RunSettings(color=r["color"], disabled=False) for r in rows},
                pinned_columns=["run:displayName", "summary:progress/checkpoints_complete", "summary:progress/checkpoints_expected"]))
    work.save()
    saved = {**previous, "campaign_id": campaign["campaign_id"], "url": work.url,
             "updated_at": now(), "api_verified": False, "runs": rows}
    saved.setdefault("browser_verified", False)
    series._atomic_json(path, saved)
    readback = ws.Workspace.from_url(work.url)
    owned = [section for section in readback.sections if section.name in {s.name for s in sections}]
    panels = [panel for section in owned for panel in section.panels]
    if len(owned) != len(sections) or len(panels) != 3 + sum(map(len, metrics.values())):
        raise RuntimeError("Workspace panel read-back failed")
    for row in rows:
        if readback.runset_settings.run_settings[row["run_id"]].color != row["color"]:
            raise RuntimeError("Workspace color read-back failed")
    if any(p.aggregate or getattr(p.x, "name", p.x) != series.X_AXIS for p in panels if isinstance(p, wr.LinePlot)):
        raise RuntimeError("Workspace curves must be individual checkpoint-axis runs")
    saved.update(api_verified=True, panel_count=len(panels))
    series._atomic_json(path, saved)
    return saved


def verify_workspace(campaign, receipt, path, wandb_module=None):
    """Read-only live view check; no workspace package is needed on Oscar."""
    path = Path(path)
    if not path.is_file():
        raise ValueError("Install native panels with layout and copy workspace.json before publishing")
    saved = series._read(path)
    if saved.get("campaign_id") != campaign["campaign_id"] or saved.get("api_verified") is not True:
        raise ValueError("Native workspace installation was not verified")
    wanted = {row["run_id"] for row in receipt["curves"].values()}
    if {row["run_id"] for row in saved.get("runs", [])} != wanted:
        raise ValueError("Workspace receipt contains different evaluation run IDs")
    parsed = urlparse(saved["url"])
    if parsed.hostname != "wandb.ai" or parsed.path.rstrip("/") != f"/{ENTITY}/{PROJECT}":
        raise ValueError("Workspace URL is outside the campaign project")
    view_name = parse_qs(parsed.query).get("nw", [None])[0]
    if not view_name:
        raise ValueError("A dedicated workspace view URL is required")
    if wandb_module is None:
        import wandb as wandb_module
    try:
        gql = importlib.import_module("wandb_gql").gql
    except ImportError:
        gql = importlib.import_module("wandb.vendor.gql").gql
    query = gql("""query View($entityName:String,$name:String,$viewName:String) {
      project(name:$name,entityName:$entityName) {
        allViews(viewType:"project-view",viewName:$viewName) { edges { node { id spec } } }
      }
    }""")
    response = wandb_module.Api(timeout=30).client.execute(query, variable_values={
        "entityName": ENTITY, "name": PROJECT, "viewName": "nw-" + view_name + "-v"})
    edges = (response.get("project") or {}).get("allViews", {}).get("edges", [])
    if len(edges) != 1:
        raise ValueError("Installed campaign workspace was not found")
    raw = edges[0]["node"]["spec"]
    wire = json.dumps(json.loads(raw) if isinstance(raw, str) else raw, sort_keys=True)
    expected_metrics = [metric for specs in metric_sections(campaign).values() for _, metric in specs]
    if any(token not in wire for token in [*wanted, series.X_AXIS, *expected_metrics, "progress/checkpoints_complete",
                                          *[row["color"] for row in receipt["curves"].values()]]):
        raise ValueError("Installed workspace lost expected campaign panels or run colors")
    return saved


def publish_once(campaign, root, *, wandb_module=None, workspace_verifier=verify_workspace):
    root = Path(root).resolve()
    with series._lock(root / ".campaign-publisher.lock", blocking=False):
        receipt = series._read(root / "publication-map.json")
        if receipt["campaign_sha256"] != series._digest(campaign):
            raise ValueError("Campaign changed after publication preparation")
        workspace = workspace_verifier(campaign, receipt, root / "workspace.json", wandb_module)
        statuses = stage_completed(campaign, receipt)
        rows = progress_rows(campaign, receipt, statuses)
        for row in rows:
            cells = [cell for cell in publication_cells(campaign) if curve_key(cell) == row["curve"]]
            evaluation_commit = campaign["source_commit"]
            if cells[0].get("role") == "base":
                baseline_receipt = Path(cells[0]["output_dir"]) / "receipt.json"
                if baseline_receipt.is_file():
                    evaluation_commit = series._read(baseline_receipt).get("source_commit", evaluation_commit)
            with series.Publisher(row["run_dir"], owner=campaign["campaign_id"], wandb_module=wandb_module) as publisher:
                publisher.run.name = row["name"]
                publisher.run.config.update({"campaign_id": campaign["campaign_id"],
                    "campaign_sha256": receipt["campaign_sha256"], "source_commit": evaluation_commit,
                    "campaign_source_commit": campaign["source_commit"],
                    "selection_sha256": campaign.get("selection", {}).get("sha256"),
                    "source_backbone": cells[0]["backbone_id"], "execution_action": cells[0]["execution"],
                    "curve_role": cells[0].get("role", "inner"),
                    "checkpoint_pins": [{k: cell[k] for k in ("step", "checkpoint_sha256", "metadata_sha256")} for cell in cells]})
                for specs in metric_sections(campaign).values():
                    for _, metric in specs:
                        publisher.run.define_metric(metric, step_metric=series.X_AXIS, step_sync=False, hidden=False)
                publisher.publish_pending()
                # Summary-only progress never creates an invented checkpoint row.
                publisher.run.summary.update({"progress/checkpoints_complete": row["complete"],
                    "progress/checkpoints_expected": row["expected"], "progress/failed_cells": row["failed"],
                    "progress/status": "complete" if row["complete"] == row["expected"] else row["status"]})
        rows = progress_rows(campaign, receipt, statuses)
        result = {"updated_at": now(), "campaign_id": campaign["campaign_id"], "cells": statuses,
                  "runs": rows, "workspace_url": workspace["url"], "workspace_api_verified": workspace["api_verified"]}
        series._atomic_json(root / "status.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "layout", "publish"))
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--spec-root", type=Path)
    parser.add_argument("--once", action="store_true", help="Publish one batch; scheduler owns polling")
    args = parser.parse_args()
    try:
        campaign = load_campaign(args.campaign)
        if args.command == "prepare":
            if args.spec_root is None:
                parser.error("prepare requires --spec-root")
            result = publication_prepare(campaign, args.root, args.spec_root)
        elif args.command == "layout":
            receipt = series._read(args.root / "publication-map.json")
            if receipt["campaign_sha256"] != series._digest(campaign):
                raise ValueError("Campaign changed after publication preparation")
            rows = [{**spec, "curve": key, "complete": 0, "published": 0,
                     "expected": len(campaign.get("steps", STEPS)),
                     "failed": 0, "status": "pending results"} for key, spec in receipt["curves"].items()]
            result = install_workspace(campaign, receipt, rows, args.root / "workspace.json")
        else:
            result = publish_once(campaign, args.root)
    except Exception as exc:
        series._atomic_json(args.root / "publication-failure.json", {
            "stage": args.command, "error_type": type(exc).__name__, "at": now()})
        # Deliberately omit remote exception text, which can contain credentials.
        raise SystemExit(f"{args.command} failed ({type(exc).__name__}); see retained publication journal") from None
    print(series._json(result))


if __name__ == "__main__":
    main()
