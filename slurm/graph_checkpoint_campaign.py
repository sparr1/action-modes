"""Pinned, return-only CUDA graph/prior checkpoint campaigns on Oscar.

GPU workers never initialize W&B. Each curve has one CPU publisher and a durable
eval-series registry. Layout updates are serialized and preserve the saved view.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import eval_series as series
from utils.evaluation_episodes import consecutive_protocol, validate_consecutive_episodes, validate_paired_starts

MATRIX = ROOT / "configs/research/ambi_checkpoint_graph_eval.json"
CONSECUTIVE_MATRIX = ROOT / "configs/research/ambi_checkpoint_graph_eval_consecutive.json"
OWNER = "oscar-rgao48"
ENTITY, PROJECT = "rwgao_b-brown-university", "ambi-inner-bench"
SELECTORS = ["controller/prior", "controller/graph"]
LABELS = {SELECTORS[0]: "Frozen prior mean", SELECTORS[1]: "H3 J8 K1 refinement mean · CUDA graph"}
COLORS = {SELECTORS[0]: "#6B7280", SELECTORS[1]: "#0F766E"}


def evaluation_settings(mode):
    if mode not in {"independent_seeds", "consecutive"}:
        raise ValueError("Unknown episode sampling mode")
    matrix = CONSECUTIVE_MATRIX if mode == "consecutive" else MATRIX
    evaluation = series._read(matrix)["evaluation"]
    sampling = consecutive_protocol(evaluation)
    return dict(matrix=str(matrix.relative_to(ROOT)), matrix_sha256=series._file_digest(matrix),
                controller_seed=evaluation["controller_seed"], max_steps=evaluation["max_steps"],
                execution_action=evaluation["execution_action"], recording=evaluation["recording"],
                **({"episode_sampling": sampling} if sampling else {"seeds": evaluation["seeds"]}))


def campaign_matrix(campaign):
    matrix = ROOT / campaign.get("matrix", str(MATRIX.relative_to(ROOT)))
    if matrix not in {MATRIX, CONSECUTIVE_MATRIX}:
        raise ValueError("Unknown checkpoint campaign matrix")
    return matrix


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def now():
    return datetime.now(timezone.utc).isoformat()


def checkpoint_steps(max_step, every=50000, descending=False):
    if max_step <= 0 or every <= 0 or max_step % every:
        raise ValueError("The checkpoint interval must divide the positive training budget")
    return sorted(range(every, max_step + 1, every), reverse=descending)


def inventory(training_root, config_name, steps, *, allow_missing=False, known=()):
    pinned = {row["step"]: row for row in known if "path" in row}
    rows = {}
    for metadata in sorted(training_root.glob("*/*/*.metadata.json")):
        weight = Path(str(metadata).removesuffix(".metadata.json"))
        if not weight.is_file() or not weight.name.startswith("model:"):
            continue
        data = series._read(metadata)
        step = data["checkpoint"]["step"]
        if step not in steps:
            continue
        if data["trial_run_params"]["name"] != config_name:
            raise ValueError("Checkpoint inventory includes a different training configuration")
        if step in rows:
            raise ValueError("The requested checkpoint grid contains duplicates")
        metadata_sha256 = series._file_digest(metadata)
        previous = pinned.get(step)
        if previous is not None:
            if (previous["path"] != str(weight.resolve()) or
                    previous["metadata_path"] != str(metadata.resolve()) or
                    previous["metadata_sha256"] != metadata_sha256):
                raise ValueError("An inventoried checkpoint changed")
            # Workers verify the full weight hash again immediately before use.
            # Do not reread all existing weights on every discovery poll.
            rows[step] = previous
        else:
            rows[step] = dict(step=step, path=str(weight.resolve()), sha256=series._file_digest(weight),
                              metadata_path=str(metadata.resolve()), metadata_sha256=metadata_sha256)
    if not set(pinned).issubset(rows):
        raise ValueError("An inventoried checkpoint disappeared")
    if not allow_missing and set(rows) != set(steps):
        raise ValueError("The requested checkpoint grid is incomplete")
    return [rows[step] for step in steps if step in rows]


def complete_grid(steps, rows):
    available = {row["step"]: row for row in rows}
    return [available.get(step, {"step": step}) for step in steps]


def write_inventory(path, campaign, rows):
    series._atomic_json(path, {"source_run": campaign["source_run"], "checkpoints": rows,
                              "training_root": campaign.get("training_root"), "inventoried_at": now()})


def refresh_inventory(path):
    with series._lock(path / ".inventory.lock"):
        campaign = load_campaign(path)
        steps = [row["step"] for row in campaign["checkpoints"]]
        rows = inventory(Path(campaign["training_root"]), campaign["config_name"], steps,
                         allow_missing=True, known=campaign["checkpoints"])
        grid = complete_grid(steps, rows)
        if grid != campaign["checkpoints"]:
            write_inventory(path / "checkpoint-inventory.json", campaign, rows)
            campaign["checkpoints"] = grid
            series._atomic_json(path / "campaign.json", campaign)
        return campaign


def plan(args):
    """Show the future workload without inventing checkpoints or result runs."""
    if git("status", "--porcelain"):
        raise ValueError("Planning requires a clean committed source checkout")
    args.campaign.mkdir(parents=True, exist_ok=False)
    steps = checkpoint_steps(args.max_step, args.checkpoint_every, args.descending)
    settings = evaluation_settings(getattr(args, "episode_sampling", "independent_seeds"))
    series._atomic_json(args.campaign / "campaign.json", dict(
        planned=True, created_at=now(), source_run=args.source_run,
        source_commit=git("rev-parse", "HEAD"), source_tree=git("rev-parse", "HEAD^{tree}"),
        **settings, cutoff_step=args.max_step,
        checkpoints=[{"step": step} for step in steps], runs=[],
        training_root=str(args.training_root), config_name=args.config_name,
        root=str(args.campaign.resolve())))


def prepare(args):
    from evaluate_ambi_checkpoint import evaluate_matrix
    if git("status", "--porcelain"):
        raise ValueError("Preparation requires a clean committed source checkout")
    previous = load_campaign(args.campaign) if args.campaign.exists() else None
    if previous is not None and not previous.get("planned"):
        raise FileExistsError("Campaign exists; select its existing registry instead of creating duplicate runs")
    steps = checkpoint_steps(args.max_step, args.checkpoint_every, args.descending)
    settings = evaluation_settings(getattr(args, "episode_sampling", "independent_seeds"))
    if previous is not None:
        if (previous["source_run"] != args.source_run or previous["training_root"] != str(args.training_root)
                or previous["config_name"] != args.config_name
                or [row["step"] for row in previous["checkpoints"]] != steps
                or previous["matrix_sha256"] != settings["matrix_sha256"]):
            raise ValueError("Preparation differs from the queued evaluation plan")
    rows = inventory(args.training_root, args.config_name, steps, allow_missing=args.available_only)
    if not rows:
        raise ValueError("At least one saved checkpoint is required to prepare evaluation")
    args.campaign.mkdir(parents=True, exist_ok=True)
    inventory_path = args.campaign / "checkpoint-inventory.json"
    series._atomic_json(inventory_path, {"source_run": args.source_run, "checkpoints": rows,
                                       "training_root": str(args.training_root), "inventoried_at": now()})
    specs = evaluate_matrix(campaign_matrix(settings), rows[0]["path"], source_run=args.source_run,
                            checkpoint_inventory=inventory_path,
                            eval_series_spec_dir=args.campaign / "specs")
    runs = []
    for selector, path in specs["specs"].items():
        registry = series.create_run(args.registry, series._read(path), args.campaign.name,
                                     PROJECT, ENTITY, OWNER)
        runs.append({"selector": selector, "id": registry["run_id"], "run_dir": registry["run_dir"],
                     "label": LABELS[selector], "color": COLORS[selector]})
    run_map = {run["selector"]: run["run_dir"] for run in runs}
    series._atomic_json(args.campaign / "run-map.json", run_map)
    campaign = dict(planned=False, created_at=now(), source_run=args.source_run, source_commit=git("rev-parse", "HEAD"),
                    source_tree=git("rev-parse", "HEAD^{tree}"), **settings,
                    cutoff_step=args.max_step, checkpoints=complete_grid(steps, rows), runs=runs,
                    training_root=str(args.training_root), config_name=args.config_name,
                    streaming=args.available_only,
                    root=str(args.campaign.resolve()))
    series._atomic_json(args.campaign / "campaign.json", campaign)
    print(json.dumps(campaign), flush=True)


def load_campaign(path):
    campaign = series._read(path / "campaign.json")
    if git("rev-parse", "HEAD") != campaign["source_commit"] or git("status", "--porcelain"):
        raise ValueError("Campaign must execute from its exact clean source commit")
    if series._file_digest(campaign_matrix(campaign)) != campaign["matrix_sha256"]:
        raise ValueError("Evaluation matrix changed")
    return campaign


def release_available(campaign, compute_job, released, save):
    """Release ready tasks newest first; future tasks retain their user hold."""
    for index, row in enumerate(campaign["checkpoints"]):
        if "path" not in row or index in released:
            continue
        job = f"{compute_job}_{index}"
        state = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True)
        if "Reason=JobHeldUser " in state:
            subprocess.run(["scontrol", "release", job], check=True)
        elif not any(f"JobState={value} " in state for value in ("RUNNING", "COMPLETING", "COMPLETED")):
            if "JobState=PENDING " not in state or "Priority=0 " in state or "Dependency=(null) " not in state:
                raise RuntimeError(f"Evaluation task {job} is not releasable: {state.strip()}")
        released.add(index)
        save()
        print(json.dumps({"released": job, "checkpoint_step": row["step"]}), flush=True)


def release(args):
    """Finite CPU controller; no GPU allocation waits for an unwritten checkpoint."""
    if args.timeout_seconds <= 0:
        raise ValueError("Release controller requires a positive finite deadline")
    receipt = args.campaign / "release-state.json"
    started = time.monotonic()
    released = set()
    def save(**extra):
        series._atomic_json(receipt, dict(compute_job=args.compute_job, training_job=args.training_job,
                                         released=sorted(released), updated_at=now(), **extra))
    def terminated(signum, frame):
        raise RuntimeError(f"Checkpoint release controller received signal {signum}")
    signal.signal(signal.SIGTERM, terminated)
    with series._lock(args.campaign / ".release.lock", blocking=False):
        if receipt.exists():
            previous = series._read(receipt)
            if previous["compute_job"] != args.compute_job or previous["training_job"] != args.training_job:
                raise ValueError("Release receipt belongs to different Slurm jobs")
            released.update(previous["released"])
        try:
            while True:
                campaign = refresh_inventory(args.campaign)
                release_available(campaign, args.compute_job, released, save)
                if len(released) == len(campaign["checkpoints"]):
                    save(complete=True)
                    return
                active = subprocess.check_output(
                    ["squeue", "-h", "-j", args.training_job, "-o", "%i"], text=True).strip()
                if not active:
                    # The final save can land between the inventory and queue reads.
                    campaign = refresh_inventory(args.campaign)
                    release_available(campaign, args.compute_job, released, save)
                    if len(released) == len(campaign["checkpoints"]):
                        save(complete=True)
                        return
                    raise RuntimeError("Training ended before every requested checkpoint was inventoried")
                if time.monotonic() - started >= args.timeout_seconds:
                    raise TimeoutError("Checkpoint release controller reached its finite deadline")
                time.sleep(60)
        except BaseException as exc:
            # Preserve every running/finished evaluation. Cancel only future
            # tasks still on our original user hold, so publishers can finish.
            for index in range(len(load_campaign(args.campaign)["checkpoints"])):
                if index in released:
                    continue
                job = f"{args.compute_job}_{index}"
                result = subprocess.run(["scontrol", "show", "job", job, "-o"], text=True, capture_output=True)
                if result.returncode == 0 and "Reason=JobHeldUser " in result.stdout:
                    subprocess.run(["scancel", job], check=True)
            save(complete=False, error=str(exc))
            raise


def validate_result(payload, expected_steps, seeds=None, *, sampling=None, controller_seed=55):
    if len(payload["results"]) != 2:
        raise ValueError("Both prior and graph results are required")
    expected_count = sampling["num_episodes"] if sampling else len(seeds)
    if {r["selector"] for r in payload["results"]} != set(SELECTORS):
        raise ValueError("Both prior and graph selectors are required")
    for result in payload["results"]:
        if not result["outer_state_unchanged"] or result["outer_updates_before"] != result["outer_updates_after"]:
            raise ValueError("Outer checkpoint state changed")
        if sampling is not None:
            if result.get("episode_sampling") != sampling:
                raise ValueError("Consecutive episode protocol changed")
            validate_consecutive_episodes(result["episodes"], sampling, controller_seed)
        elif (result["environment_seeds"] != seeds
              or [ep["seed"] for ep in result["episodes"]] != seeds):
            raise ValueError("Episode seeds differ from the requested paired protocol")
        if len(result["episodes"]) != expected_count or any(ep["length"] != expected_steps for ep in result["episodes"]):
            raise ValueError("Episode coverage differs from the requested paired protocol")
        if result["selector"] == "controller/graph":
            if result.get("graph_replays") != expected_steps * expected_count:
                raise ValueError("Missing whole-solve graph replay")
            for episode in result["episodes"]:
                metrics = episode["model_metrics"]
                for key, value in {"inner_cuda_graph_used": 1, "inner_critic_optimizer_steps": 128,
                                   "inner_actor_optimizer_steps": 32, "inner_model_steps": 3072,
                                   "inner_compile_fallback": 0}.items():
                    if metrics.get(key) != value:
                        raise ValueError(f"Unexpected graph workload: {key}={metrics.get(key)}")
    if sampling is not None:
        validate_paired_starts(payload["results"][0]["episodes"], payload["results"][1]["episodes"])
    return payload


def worker(args):
    import torch
    from evaluate_ambi_checkpoint import evaluate_matrix
    if os.environ.get("WANDB_MODE") != "disabled" or not torch.cuda.is_available():
        raise ValueError("GPU evaluation requires CUDA and WANDB_MODE=disabled")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    campaign = load_campaign(args.campaign)
    row = campaign["checkpoints"][args.index]
    if any(series._file_digest(row[path]) != row[digest] for path, digest in
           (("path", "sha256"), ("metadata_path", "metadata_sha256"))):
        raise ValueError("Checkpoint or metadata changed after inventory")
    output = args.campaign / ("smoke" if args.smoke else "production") / str(row["step"])
    output.mkdir(parents=True, exist_ok=False)
    # Publication fingerprints include this file; later discovery must never
    # mutate the provenance of a completed checkpoint result.
    inventory_path = output / "checkpoint-inventory.json"
    write_inventory(inventory_path, campaign, [row])
    sampling = campaign.get("episode_sampling")
    if sampling is not None:
        sampling = {**sampling, "num_episodes": 2 if args.smoke else sampling["num_episodes"]}
        seeds = None
        episode_options = {"num_episodes": sampling["num_episodes"]}
    else:
        seeds = campaign["seeds"][:2] if args.smoke else campaign["seeds"]
        episode_options = {"seeds": seeds}
    steps = 8 if args.smoke else campaign["max_steps"]
    if args.smoke:
        subprocess.run([sys.executable, "-m", "pytest", "-q",
                        "tests/test_frozen_graph_evaluation.py::test_cuda_mean_graph_matches_control_and_resets_episode_rng"], check=True)
    started = time.perf_counter()
    payload = evaluate_matrix(campaign_matrix(campaign), row["path"], **episode_options, max_steps=steps, device="cuda",
                              bundle_dir=output / "bundle", source_run=campaign["source_run"],
                              checkpoint_inventory=inventory_path,
                              eval_run_map=None if args.smoke else args.campaign / "run-map.json",
                              stage_results=False)
    validate_result(payload, steps, seeds, sampling=sampling, controller_seed=campaign["controller_seed"])
    series._atomic_json(output / "results.json", payload)
    receipt = dict(passed=True, checkpoint_step=row["step"], seconds=time.perf_counter()-started,
                   source_commit=campaign["source_commit"], gpu=torch.cuda.get_device_name(),
                   cpus=torch.get_num_threads(), tf32="high", wandb=False, max_steps=steps,
                   **({"episode_sampling": sampling} if sampling else {"seeds": seeds}),
                   results=[{k:r[k] for k in ("selector", "return", "outer_state_unchanged")}
                            for r in payload["results"]])
    series._atomic_json(output / "receipt.json", receipt)
    if not args.smoke:
        for run in campaign["runs"]:
            series.stage_result(run["run_dir"], output / "bundle/manifest.json", selector=run["selector"],
                                source_run=campaign["source_run"], inventory_path=inventory_path)
    print(json.dumps(receipt), flush=True)


def progress(campaign):
    rows = []
    for cp in campaign["checkpoints"]:
        path = Path(campaign["root"]) / "production" / str(cp["step"]) / "bundle/manifest.json"
        manifest = series._read(path) if path.exists() else {}
        states = {run["selector"]: run["status"] for run in manifest.get("runs", [])}
        pending = "queued" if "path" in cp else "awaiting checkpoint"
        rows.append({"step": cp["step"], **{selector: states.get(selector, pending) for selector in SELECTORS}})
    return rows


def layout(args):
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces import expr
    from utils.checkpoint_return_plot import return_episode_panel
    campaign = load_campaign(args.campaign)
    with series._lock(args.campaign / ".layout.lock"):
        rows = progress(campaign)
        table = "\n".join(f"| {r['step']:,} | {r[SELECTORS[0]]} | {r[SELECTORS[1]]} |" for r in rows)
        sampling = campaign.get("episode_sampling")
        episode_description = (
            f"{sampling['num_episodes']} paired consecutive full Humanoid Walk episodes per checkpoint; "
            f"environment stream seeded once with {sampling['environment_seed']}, then ordinary resets. "
            "Controller and CUDA graph RNG advance across episodes; both streams restart at each checkpoint. "
            if sampling else
            f"{len(campaign['seeds'])} paired full Humanoid Walk episodes per checkpoint; "
            f"environment seeds {campaign['seeds']}. CUDA graph RNG is isolated per episode. "
        )
        markdown = (f"**Post-tanh / outer-adapt checkpoint evaluation.** Source `{campaign['source_run']}`.\n\n"
                    + episode_description +
                    f"Controller seed {campaign['controller_seed']}, {campaign['max_steps']} decisions per episode. "
                    "Both execute actor means. Refinement retains H3/J8/K1, "
                    "N128/C16/A4/B256, inherited fixed entropy and action-local adaptive Q scaling. "
                    "Outer checkpoint state stays frozen. "
                    "Return-only logging omits inner traces/model probes. Dots show all episode returns "
                    "at each checkpoint; lines show their means. Small horizontal offsets separate overlapping dots.\n\n"
                    f"Checkpoint order: {', '.join(str(cp['step']) for cp in campaign['checkpoints'])}. "
                    f"Status at {now()}. Missing points remain pending. "
                    "Available checkpoints are evaluated newest first while training continues. "
                    "Future checkpoints are released as their weights and metadata become available.\n\n"
                    "| Training decisions | Prior | Refinement |\n|---:|---|---|\n" + table)
        panels = [wr.MarkdownPanel(markdown=markdown, layout=wr.Layout(w=24, h=14))]
        preset_path = args.campaign / "return-episodes-preset.json"
        return_panel, preset = return_episode_panel(campaign["runs"], ENTITY,
            series._read(preset_path) if preset_path.exists() else None)
        series._atomic_json(preset_path, preset)
        panels.append(return_panel)
        for title, metric in (("Paired improvement over prior", "eval/paired_gain_mean"),
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
            work = ws.Workspace(entity=ENTITY, project=PROJECT, name="Posttanh outer adapt · checkpoint evaluation",
                                auto_generate_panels=False, sections=[section],
                                runset_settings=ws.RunsetSettings(filters=[expr.Metric("ID").isin([r["id"] for r in campaign["runs"]])],
                                                               groupby=[], run_settings={r["id"]:ws.RunSettings(color=r["color"],disabled=False) for r in campaign["runs"]}))
        # A planned campaign has no result IDs yet. Keep unrelated runs hidden;
        # preparation replaces this empty selection with the two real series.
        work.runset_settings.filters = [expr.Metric("ID").isin(
            [r["id"] for r in campaign["runs"]] or ["pending-" + args.campaign.name])]
        work.runset_settings.run_settings = {r["id"]: ws.RunSettings(color=r["color"], disabled=False)
                                             for r in campaign["runs"]}
        work.save()
        verified = ws.Workspace.from_url(work.url)
        if not any(s.name == section.name and len(s.panels) == 4 and
                   any(isinstance(p, wr.CustomChart) and p.chart_name == preset["chart_id"]
                       and p.chart_fields == return_panel.chart_fields for p in s.panels)
                   for s in verified.sections):
            raise RuntimeError("W&B result panel verification failed")
        series._atomic_json(receipt_path, {"url": work.url, "api_verified": True, "updated_at": now(), "rows": rows})
        print(json.dumps({"workspace_url": work.url}), flush=True)


def publish(args):
    from utils.checkpoint_return_plot import publish_episode_returns
    campaign = load_campaign(args.campaign)
    run = next(r for r in campaign["runs"] if r["selector"] == args.selector)
    command = [args.workspace_python, str(Path(__file__).resolve()), "layout", "--campaign", str(args.campaign)]
    subprocess.run(command, check=True)
    expected = len(campaign["checkpoints"])
    with series.Publisher(run["run_dir"], owner=OWNER) as publisher:
        publisher.run.name = "Posttanh outer-adapt · " + run["label"]
        publisher.run.config.update({"curve_label": run["label"]}, allow_val_change=True)
        last = None
        table_count = None
        while True:
            status = publisher.publish_pending()
            count = status["accepted"]
            if count != table_count:
                publish_episode_returns(publisher.run, run["run_dir"])
                table_count = count
            publisher.run.summary.update({"evaluation/checkpoints_expected": expected,
                                          "evaluation/checkpoints_accepted": count,
                                          "evaluation/status": "complete" if count == expected else "pending"})
            state = progress(load_campaign(args.campaign))
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
        publish_episode_returns(publisher.run, run["run_dir"])
    receipt = series._read(Path(run["run_dir"]) / "publication.json")
    if len(receipt["records"]) != expected or any(r["status"] != "published" for r in receipt["records"].values()):
        raise RuntimeError("Final W&B publication acknowledgement is incomplete")
    subprocess.run(command, check=True)
    series._atomic_json(args.campaign / (args.selector.split("/")[-1] + "-publication.json"),
                        {"passed": True, "run_id": run["id"], "published": expected, "finished_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "prepare", "worker", "layout", "publish", "release"):
        p = sub.add_parser(name)
        p.add_argument("--campaign", type=Path, required=True)
        if name in {"plan", "prepare"}:
            p.add_argument("--training-root", type=Path, required=True)
            p.add_argument("--source-run", required=True)
            p.add_argument("--config-name", required=True)
            p.add_argument("--max-step", type=int, required=True)
            p.add_argument("--checkpoint-every", type=int, default=50000)
            p.add_argument("--descending", action="store_true")
            p.add_argument("--episode-sampling", choices=("independent_seeds", "consecutive"),
                           default="independent_seeds", help="Consecutive selects ten episodes from seed 55.")
            p.add_argument("--registry", type=Path, required=True)
        if name == "prepare":
            p.add_argument("--available-only", action="store_true")
        if name == "worker":
            p.add_argument("--index", type=int, required=True)
            p.add_argument("--smoke", action="store_true")
        if name == "publish":
            p.add_argument("--selector", choices=SELECTORS, required=True)
            p.add_argument("--compute-job", required=True)
            p.add_argument("--workspace-python", required=True)
        if name == "release":
            p.add_argument("--compute-job", required=True)
            p.add_argument("--training-job", required=True)
            p.add_argument("--timeout-seconds", type=int, required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
