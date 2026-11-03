"""Pinned, return-only CUDA graph/prior checkpoint campaigns on Oscar.

GPU workers never initialize W&B. Each curve has one CPU publisher and a durable
eval-series registry. Layout updates are serialized and preserve the saved view.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import eval_series as series

MATRIX = ROOT / "configs/research/ambi_checkpoint_graph_eval.json"
OWNER = "oscar-rgao48"
ENTITY, PROJECT = "rwgao_b-brown-university", "ambi-inner-bench"
SELECTORS = ["controller/prior", "controller/graph"]
COLORS = {SELECTORS[0]: "#6B7280", SELECTORS[1]: "#0F766E"}


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def now():
    return datetime.now(timezone.utc).isoformat()


def controller_details(config):
    h, j, k, n, c, a, b = (config[key] for key in (
        "inner_rollout_horizon", "inner_rounds", "inner_solve_interval",
        "inner_rollouts_per_round", "inner_critic_updates_per_round",
        "inner_actor_updates_per_round", "inner_batch_size"))
    scale = {("action", "per_action"): "outer Q scale frozen within each solve",
             ("action", "per_update"): "outer Q scale adapted within each solve",
             ("run", "per_update"): "adaptive Q scale carried between decisions, reset to outer at each independent episode"}[
                 config["inner_actor_loss_scale_scope"], config["inner_actor_loss_scale_update"]]
    return {
        "labels": {SELECTORS[0]: "Frozen prior mean", SELECTORS[1]: f"H{h} J{j} K{k} refinement mean · CUDA graph"},
        "description": (f"Refinement retains H{h}/J{j}/K{k}, N{n}/C{c}/A{a}/B{b}, "
                        f"{config['inner_actor_entropy_mode']} entropy, "
                        f"temperature mode {config['inner_temperature_mode']}, and {scale}."),
        "expected_metrics": {"inner_cuda_graph_used": 1,
                             "inner_critic_optimizer_steps": config["inner_critic_updates_per_action"],
                             "inner_actor_optimizer_steps": config["inner_actor_updates_per_action"],
                             "inner_model_steps": config["inner_model_step_budget"],
                             "inner_compile_fallback": 0},
    }


def prepare(args):
    from evaluate_ambi_checkpoint import evaluate_matrix
    from types import SimpleNamespace
    from utils.ambi_research import resolve_preset
    from utils.checkpoint_context import load_checkpoint_context
    from utils.eval_series_data import resolved_checkpoint_config
    from RL.tdmpc2_core.whole_inner_graph import validate_training_graph
    import torch
    if git("status", "--porcelain"):
        raise ValueError("Preparation requires a clean committed source checkout")
    if args.campaign.exists():
        raise FileExistsError("Campaign exists; select its existing registry instead of creating duplicate runs")
    rows = []
    for metadata in sorted(args.training_root.glob("*/*/*.metadata.json")):
        weight = Path(str(metadata).removesuffix(".metadata.json"))
        if not weight.is_file() or not weight.name.startswith("model:"):
            continue
        data = series._read(metadata)
        step = data["checkpoint"]["step"]
        if step < 50000 or step > args.max_step or step % 50000:
            continue
        trial = data["trial_run_params"]
        if trial["name"] != args.config_name:
            raise ValueError("Checkpoint inventory includes a different training configuration")
        rows.append(dict(step=step, path=str(weight.resolve()), sha256=series._file_digest(weight),
                         metadata_path=str(metadata.resolve()), metadata_sha256=series._file_digest(metadata)))
    rows.sort(key=lambda row: row["step"])
    if not rows or len({row["step"] for row in rows}) != len(rows):
        raise ValueError("The present-checkpoint inventory is empty or duplicated")
    context = load_checkpoint_context(rows[0]["path"])
    resolved = resolve_preset(MATRIX, SELECTORS[1], checkpoint_context=context)
    config = resolved_checkpoint_config({"metadata": series._read(rows[0]["metadata_path"])}, resolved)
    validate_training_graph(SimpleNamespace(cfg=SimpleNamespace(**config), device=torch.device("cuda"),
                                            _active_trace=None, _retrace_enabled=False))
    details = controller_details(config)
    args.campaign.mkdir(parents=True)
    inventory_path = args.campaign / "checkpoint-inventory.json"
    series._atomic_json(inventory_path, {"source_run": args.source_run, "checkpoints": rows,
                                       "training_root": str(args.training_root), "inventoried_at": now()})
    specs = evaluate_matrix(MATRIX, rows[0]["path"], source_run=args.source_run,
                            checkpoint_inventory=inventory_path,
                            eval_series_spec_dir=args.campaign / "specs")
    runs = []
    for selector, path in specs["specs"].items():
        registry = series.create_run(args.registry, series._read(path), args.campaign.name,
                                     PROJECT, ENTITY, OWNER)
        runs.append({"selector": selector, "id": registry["run_id"], "run_dir": registry["run_dir"],
                     "label": details["labels"][selector], "color": COLORS[selector]})
    run_map = {run["selector"]: run["run_dir"] for run in runs}
    series._atomic_json(args.campaign / "run-map.json", run_map)
    campaign = dict(created_at=now(), source_run=args.source_run, source_name=args.source_name,
                    controller_description=details["description"], expected_metrics=details["expected_metrics"],
                    source_commit=git("rev-parse", "HEAD"),
                    source_tree=git("rev-parse", "HEAD^{tree}"), matrix_sha256=series._file_digest(MATRIX),
                    cutoff_step=args.max_step, checkpoints=rows, runs=runs,
                    seeds=[101, 102, 103, 104, 105], controller_seed=55, max_steps=500,
                    execution_action="mean", recording="returns", root=str(args.campaign.resolve()))
    series._atomic_json(args.campaign / "campaign.json", campaign)
    print(json.dumps(campaign), flush=True)


def load_campaign(path):
    campaign = series._read(path / "campaign.json")
    if git("rev-parse", "HEAD") != campaign["source_commit"] or git("status", "--porcelain"):
        raise ValueError("Campaign must execute from its exact clean source commit")
    if series._file_digest(MATRIX) != campaign["matrix_sha256"]:
        raise ValueError("Evaluation matrix changed")
    return campaign


def validate_result(payload, expected_steps, seeds, expected_metrics):
    if len(payload["results"]) != 2:
        raise ValueError("Both prior and graph results are required")
    for result in payload["results"]:
        if not result["outer_state_unchanged"] or result["outer_updates_before"] != result["outer_updates_after"]:
            raise ValueError("Outer checkpoint state changed")
        if result["environment_seeds"] != seeds or any(ep["length"] != expected_steps for ep in result["episodes"]):
            raise ValueError("Episode coverage differs from the requested paired protocol")
        if result["selector"] == "controller/graph":
            if result.get("graph_replays") != expected_steps * len(seeds):
                raise ValueError("Missing whole-solve graph replay")
            for episode in result["episodes"]:
                metrics = episode["model_metrics"]
                for key, value in expected_metrics.items():
                    if metrics.get(key) != value:
                        raise ValueError(f"Unexpected graph workload: {key}={metrics.get(key)}")
    return payload


def worker(args):
    import torch
    from evaluate_ambi_checkpoint import evaluate_matrix
    if os.environ.get("WANDB_MODE") != "disabled" or not torch.cuda.is_available():
        raise ValueError("GPU evaluation requires CUDA and WANDB_MODE=disabled")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    campaign = load_campaign(args.campaign)
    row = campaign["checkpoints"][args.index]
    if any(series._file_digest(row[path]) != row[digest] for path, digest in
           (("path", "sha256"), ("metadata_path", "metadata_sha256"))):
        raise ValueError("Checkpoint or metadata changed after inventory")
    output = args.campaign / ("smoke" if args.smoke else "production") / str(row["step"])
    output.mkdir(parents=True, exist_ok=False)
    seeds = [101, 102] if args.smoke else campaign["seeds"]
    steps = 8 if args.smoke else 500
    if args.smoke:
        subprocess.run([sys.executable, "-m", "pytest", "-q",
                        "tests/test_frozen_graph_evaluation.py", "-k", "cuda"], check=True)
    started = time.perf_counter()
    payload = evaluate_matrix(MATRIX, row["path"], seeds=seeds, max_steps=steps, device="cuda",
                              bundle_dir=output / "bundle", source_run=campaign["source_run"],
                              checkpoint_inventory=args.campaign / "checkpoint-inventory.json",
                              eval_run_map=None if args.smoke else args.campaign / "run-map.json",
                              stage_results=False)
    validate_result(payload, steps, seeds, campaign["expected_metrics"])
    series._atomic_json(output / "results.json", payload)
    receipt = dict(passed=True, checkpoint_step=row["step"], seconds=time.perf_counter()-started,
                   source_commit=campaign["source_commit"], gpu=torch.cuda.get_device_name(),
                   cpus=1, tf32="high", wandb=False, seeds=seeds, max_steps=steps,
                   results=[{k:r[k] for k in ("selector", "return", "outer_state_unchanged")}
                            for r in payload["results"]])
    series._atomic_json(output / "receipt.json", receipt)
    if not args.smoke:
        for run in campaign["runs"]:
            series.stage_result(run["run_dir"], output / "bundle/manifest.json", selector=run["selector"],
                                source_run=campaign["source_run"], inventory_path=args.campaign / "checkpoint-inventory.json")
    print(json.dumps(receipt), flush=True)


def progress(campaign):
    rows = []
    for cp in campaign["checkpoints"]:
        path = Path(campaign["root"]) / "production" / str(cp["step"]) / "bundle/manifest.json"
        manifest = series._read(path) if path.exists() else {}
        states = {run["selector"]: run["status"] for run in manifest.get("runs", [])}
        rows.append({"step": cp["step"], **{selector: states.get(selector, "pending") for selector in SELECTORS}})
    return rows


def layout(args):
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr
    campaign = load_campaign(args.campaign)
    with series._lock(args.campaign / ".layout.lock"):
        rows = progress(campaign)
        table = "\n".join(f"| {r['step']:,} | {r[SELECTORS[0]]} | {r[SELECTORS[1]]} |" for r in rows)
        markdown = (f"**{campaign['source_name']} · checkpoint evaluation.** Source `{campaign['source_run']}`.\n\n"
                    "Five paired full Humanoid Walk episodes per checkpoint; environment seeds 101–105, "
                    "controller seed 55, 500 decisions. Both execute actor means. "
                    f"{campaign['controller_description']} "
                    "CUDA graph RNG is isolated per episode; outer checkpoint state stays frozen. "
                    "Return-only logging omits inner traces/model probes. Error bars are represented by the separate episode SD panel.\n\n"
                    f"Present-checkpoint cutoff: {campaign['cutoff_step']:,}. Status at {now()}. Missing points remain pending.\n\n"
                    "| Training decisions | Prior | Refinement |\n|---:|---|---|\n" + table)
        panels = [wr.MarkdownPanel(markdown=markdown, layout=wr.Layout(w=24, h=14))]
        for title, metric in (("Full-episode return · five-seed mean", "eval/return_mean"),
                              ("Paired improvement over prior", "eval/paired_gain_mean"),
                              ("Episode return variability · sample SD", "eval/return_sample_std"),
                              ("Controller seconds per decision", "runtime/control_seconds_per_decision")):
            panels.append(wr.LinePlot(title=title, x=series.X_AXIS, y=[metric], aggregate=False,
                                     smoothing_type="none", title_x="Training decisions at checkpoint",
                                     legend_template="${run:displayName}", max_runs_to_show=2,
                                     line_titles={r["id"]+":"+metric:r["label"] for r in campaign["runs"]},
                                     line_colors={r["id"]+":"+metric:r["color"] for r in campaign["runs"]},
                                     layout=wr.Layout(w=12, h=8)))
        section = ws.Section(name="Checkpoint evaluation / Results and progress", is_open=True, panels=panels)
        receipt_path = args.campaign / "workspace.json"
        if receipt_path.exists():
            work = ws.Workspace.from_url(series._read(receipt_path)["url"])
            found = any(s.name == section.name for s in work.sections)
            work.sections = [section if s.name == section.name else s for s in work.sections]
            if not found:
                work.sections.append(section)
        else:
            work = ws.Workspace(entity=ENTITY, project=PROJECT, name=campaign['source_name'] + " · checkpoint evaluation",
                                auto_generate_panels=False, sections=[section],
                                runset_settings=ws.RunsetSettings(filters=[expr.Metric("ID").isin([r["id"] for r in campaign["runs"]])],
                                                               groupby=[], run_settings={r["id"]:ws.RunSettings(color=r["color"],disabled=False) for r in campaign["runs"]}))
        work.save()
        verified = ws.Workspace.from_url(work.url)
        if not any(s.name == section.name and len(s.panels) == 5 for s in verified.sections):
            raise RuntimeError("W&B result panel verification failed")
        series._atomic_json(receipt_path, {"url": work.url, "api_verified": True, "updated_at": now(), "rows": rows})
        print(json.dumps({"workspace_url": work.url}), flush=True)


def publish(args):
    campaign = load_campaign(args.campaign)
    run = next(r for r in campaign["runs"] if r["selector"] == args.selector)
    command = [args.workspace_python, str(Path(__file__).resolve()), "layout", "--campaign", str(args.campaign)]
    subprocess.run(command, check=True)
    expected = len(campaign["checkpoints"])
    with series.Publisher(run["run_dir"], owner=OWNER) as publisher:
        publisher.run.name = campaign["source_name"] + " · " + run["label"]
        publisher.run.config.update({"curve_label": run["label"]}, allow_val_change=True)
        last = None
        while True:
            status = publisher.publish_pending()
            count = status["accepted"]
            publisher.run.summary.update({"evaluation/checkpoints_expected": expected,
                                          "evaluation/checkpoints_accepted": count,
                                          "evaluation/status": "complete" if count == expected else "pending"})
            state = progress(campaign)
            signature = json.dumps(state)
            if signature != last:
                subprocess.run(command, check=True)
                last = signature
            if count == expected:
                break
            active = subprocess.check_output(["squeue", "-h", "-j", args.compute_job, "-o", "%i"], text=True).strip()
            if not active:
                # GPU workers stage after their final result writes; recheck once.
                status = publisher.publish_pending()
                if status["accepted"] != expected:
                    publisher.run.summary["evaluation/status"] = "incomplete_or_failed"
                    raise RuntimeError(f"Compute ended with {status['accepted']}/{expected} complete checkpoints")
                break
            time.sleep(15)
    receipt = series._read(Path(run["run_dir"]) / "publication.json")
    if len(receipt["records"]) != expected or any(r["status"] != "published" for r in receipt["records"].values()):
        raise RuntimeError("Final W&B publication acknowledgement is incomplete")
    subprocess.run(command, check=True)
    series._atomic_json(args.campaign / (args.selector.split("/")[-1] + "-publication.json"),
                        {"passed": True, "run_id": run["id"], "published": expected, "finished_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "worker", "layout", "publish"):
        p = sub.add_parser(name)
        p.add_argument("--campaign", type=Path, required=True)
        if name == "prepare":
            p.add_argument("--training-root", type=Path, required=True)
            p.add_argument("--source-run", required=True)
            p.add_argument("--source-name", required=True)
            p.add_argument("--config-name", required=True)
            p.add_argument("--max-step", type=int, required=True)
            p.add_argument("--registry", type=Path, required=True)
        if name == "worker":
            p.add_argument("--index", type=int, required=True)
            p.add_argument("--smoke", action="store_true")
        if name == "publish":
            p.add_argument("--selector", choices=SELECTORS, required=True)
            p.add_argument("--compute-job", required=True)
            p.add_argument("--workspace-python", required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
