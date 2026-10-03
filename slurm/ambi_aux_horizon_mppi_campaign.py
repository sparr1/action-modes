"""Prepare and execute the frozen backbone-by-MPPI horizon comparison.

This orchestrator reuses the existing evaluator and curve publisher. It does
not implement planning, training, controller randomness, or W&B networking.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RECIPE = ROOT / "configs/research/ambi_aux_horizon_mppi_campaign.json"
BASELINE = ROOT / "configs/dmcontrol/algs/ambi_aux_return_sac_clip_target10p5_shared.json"
HORIZONS = (1, 3, 5, 7)
SEEDS = [101, 102, 103, 104, 105]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    from utils.ambi_benchmark import atomic_json
    atomic_json(path, value)


def source_commit():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    require(not git("status", "--porcelain", "--untracked-files=all"), "Source checkout must be clean")
    return git("rev-parse", "HEAD")


def model_steps(horizon):
    require(type(horizon) is int and horizon in HORIZONS, "Unsupported planning horizon")
    return 8 * 512 * horizon + 24 * (horizon - 1)


def matrix_for(source_run, planning_horizon=None):
    """Keep the previously tested scientific preset, changing only source and H."""
    filename = "ambi_aux_return_prior_reference.json" if planning_horizon is None else "ambi_aux_return_mppi.json"
    matrix = read(ROOT / "configs/research" / filename)
    matrix["source_run"] = source_run
    matrix["description"] = "Frozen shared auxiliary-return SAC backbone: training horizon versus planning horizon."
    if planning_horizon is not None:
        matrix["shared_alg_params"]["inner_rollout_horizon"] = planning_horizon
        matrix["shared_alg_params"]["inner_model_step_budget"] = model_steps(planning_horizon)
    return matrix


def scientific_recipe(params):
    return {key: value for key, value in params.items()
            if key not in {"train_unroll_horizon", "wandb_group", "wandb_tags"}}


def validate_checkpoint(backbone, row, *, hash_weights=True):
    """Bind training horizon, training launch, exact weights and sidecar settings."""
    from utils.checkpoint_context import load_checkpoint_context
    horizon = backbone["train_horizon"]
    require(type(horizon) is int and horizon in HORIZONS, "Unsupported training horizon")
    path = Path(row["path"])
    require(path.is_absolute() and path.is_file(), "Checkpoint must be an existing absolute path")
    require(type(row["step"]) is int and 0 < row["step"] <= 2_000_000 and row["step"] % 25_000 == 0,
            "Checkpoint must be on the trained 25K grid through 2M")
    for key in ("sha256", "metadata_sha256"):
        require(re.fullmatch(r"[0-9a-f]{64}", row[key]) is not None, "Invalid checkpoint digest")
    if hash_weights:
        require(digest(path) == row["sha256"], "Checkpoint weights changed")
    metadata_path = row.get("metadata_path", str(path) + ".metadata.json")
    require(digest(metadata_path) == row["metadata_sha256"], "Checkpoint sidecar changed")
    context = load_checkpoint_context(path, metadata_path=metadata_path)
    trial, params = context.trial_run_params, context.trial_run_params["alg_params"]
    require(context.metadata["checkpoint"]["step"] == row["step"], "Checkpoint step disagrees with sidecar")
    require(params["train_unroll_horizon"] == horizon, "Checkpoint training horizon differs")
    require(scientific_recipe(params) == scientific_recipe(read(BASELINE)["alg_params"]),
            "Checkpoint does not match the shared target -10.5 SAC backbone recipe")
    require((trial["alg"], trial["env"], trial["seed"], trial["total_steps"]) ==
            ("AMBITDMPC2/AMBITDMPC2", "DMControl-v0", 55, 2_000_000), "Training identity differs")
    require(context.experiment_params["env_params"] ==
            {"task": "humanoid-walk", "obs": "state", "render_mode": None}, "Environment differs")
    launch = read(backbone["training_launch"])
    require(launch["wandb"]["path"] == backbone["source_run"], "Training launch source run differs")
    require(re.fullmatch(r"[0-9a-f]{40}", launch["binding"]["source_commit"]) is not None,
            "Training launch lacks exact source commit")
    require(launch["run_params"]["alg_params"] == params, "Checkpoint recipe differs from training launch")
    if "training_launch_sha256" in backbone:
        require(digest(backbone["training_launch"]) == backbone["training_launch_sha256"], "Training launch changed")
    if "training_source_commit" in backbone:
        require(launch["binding"]["source_commit"] == backbone["training_source_commit"], "Training source changed")
    return context


def validate_input(inventory, *, hash_weights=True):
    require(inventory.get("schema_version") == 1, "Unsupported campaign inventory schema")
    backbones = inventory["backbones"]
    require(sorted(b["train_horizon"] for b in backbones) == list(HORIZONS), "Supply exactly training H=1,3,5,7")
    require(len({b["source_run"] for b in backbones}) == 4, "Training source runs must be distinct")
    grids = []
    for backbone in backbones:
        require(len(backbone["source_run"].split("/")) == 3, "Source run must be entity/project/id")
        require(Path(backbone["training_launch"]).is_absolute(), "Training launch path must be absolute")
        rows = backbone["checkpoints"]
        steps = [row["step"] for row in rows]
        require(steps and len(steps) == len(set(steps)), "Checkpoint grid is empty or duplicated")
        grids.append(sorted(steps))
        for row in rows:
            validate_checkpoint(backbone, row, hash_weights=hash_weights)
        for step, bundle in backbone.get("prior_bundles", {}).items():
            require(int(step) in steps and Path(bundle).is_absolute(), "Prior reuse is outside the selected grid")
        for step, bundles in backbone.get("mppi_bundles", {}).items():
            require(int(step) in steps, "MPPI reuse is outside the selected grid")
            for horizon, bundle in bundles.items():
                require(int(horizon) in HORIZONS and Path(bundle).is_absolute(), "Invalid MPPI reuse entry")
    require(all(grid == grids[0] for grid in grids), "All four backbones must use the same selected checkpoint grid")
    return sorted(backbones, key=lambda b: b["train_horizon"])


def records_for(bundle, inventory):
    from utils.eval_series_data import load_records
    from utils.eval_series import validate_record
    records = load_records(bundle, checkpoint_inventory=inventory)
    return [validate_record(record) for record in records]


def validate_records(records, row, run_map):
    from utils.eval_series import load_run, validate_identity
    require(len(records) == len(run_map), "Unexpected number of completed controllers")
    require({r["selector"] for r in records} == set(run_map), "Controller selectors differ")
    for record in records:
        require(record["checkpoint"] == {"step": row["step"], "sha256": row["sha256"]}, "Result checkpoint differs")
        validate_identity(load_run(run_map[record["selector"]]), record["identity"])
        require(record["metrics"].get("eval/frozen_state_unchanged") is True, "Frozen-state verification failed")


def prepare(inventory_path, output_root, registry_root, attempt_label):
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.eval_series import create_run, stage_record
    commit = source_commit()
    inventory = read(inventory_path)
    inventory = deepcopy(inventory)
    for backbone in inventory["backbones"]:
        for row in backbone["checkpoints"]:
            actual_sha256 = digest(row["path"])
            require(row.get("sha256", actual_sha256) == actual_sha256, "Checkpoint weights changed")
            row["sha256"] = actual_sha256
            if "metadata_sha256" not in row:
                row["metadata_sha256"] = digest(row.get("metadata_path", row["path"] + ".metadata.json"))
            if "bytes" in row:
                require(Path(row["path"]).stat().st_size == row["bytes"], "Checkpoint size changed")
    backbones = validate_input(inventory, hash_weights=False)
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    prepared = {"schema_version": 1, "source_commit": commit, "recipe_sha256": digest(RECIPE),
                "input_inventory_sha256": digest(inventory_path), "attempt_label": attempt_label,
                "root": str(output_root), "backbones": [], "prior_tasks": [], "mppi_tasks": [], "curves": [],
                "reused": []}
    for original in backbones:
        backbone = deepcopy(original)
        horizon = backbone["train_horizon"]
        home = output_root / f"train_h{horizon}"
        launch = read(backbone["training_launch"])
        backbone["training_launch_sha256"] = digest(backbone["training_launch"])
        backbone["training_source_commit"] = launch["binding"]["source_commit"]
        backbone["core_recipe_sha256"] = hashlib.sha256(json.dumps(scientific_recipe(
            launch["run_params"]["alg_params"]), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        checkpoint_inventory = home / "checkpoint-manifest.json"
        write(checkpoint_inventory, {"source_run": backbone["source_run"], "checkpoints": backbone["checkpoints"]})
        backbone["checkpoint_inventory"] = str(checkpoint_inventory)
        backbone["checkpoint_inventory_sha256"] = digest(checkpoint_inventory)
        backbone["matrices"], backbone["run_maps"] = {}, {}
        representative = backbone["checkpoints"][0]
        for planning_horizon in (None, *HORIZONS):
            key = "prior" if planning_horizon is None else f"mppi_h{planning_horizon}"
            matrix_path = home / "matrices" / (key + ".json")
            write(matrix_path, matrix_for(backbone["source_run"], planning_horizon))
            backbone["matrices"][key] = {"path": str(matrix_path), "sha256": digest(matrix_path)}
            specifications = evaluate_matrix(matrix_path, representative["path"], device="cpu",
                checkpoint_inventory=checkpoint_inventory, metadata_path=representative.get("metadata_path"),
                eval_series_spec_dir=home / "specs" / key)
            run_map = {}
            for selector, spec_path in specifications["specs"].items():
                registry = create_run(registry_root, read(spec_path),
                    attempt_label=f"{attempt_label} / backbone train H={horizon}",
                    project="ambi-inner-bench", entity="rwgao_b-brown-university", owner="oscar-rgao48")
                run_map[selector] = registry["run_dir"]
                prepared["curves"].append({"train_horizon": horizon, "planning_horizon": planning_horizon,
                    "selector": selector, "run_dir": registry["run_dir"], "run_id": registry["run_id"]})
            backbone["run_maps"][key] = run_map
            for row in sorted(backbone["checkpoints"], key=lambda r: r["step"]):
                step = str(row["step"])
                bundle = (backbone.get("prior_bundles", {}).get(step) if planning_horizon is None else
                          backbone.get("mppi_bundles", {}).get(step, {}).get(str(planning_horizon)))
                if bundle:
                    records = records_for(bundle, checkpoint_inventory)
                    validate_records(records, row, run_map)
                    for record in records:
                        stage_record(run_map[record["selector"]], record)
                    prepared["reused"].append({"train_horizon": horizon, "planning_horizon": planning_horizon,
                        "step": row["step"], "bundle": bundle})
                else:
                    kind = "prior" if planning_horizon is None else "mppi"
                    tasks = prepared[kind + "_tasks"]
                    task = {"index": len(tasks), "train_horizon": horizon, "planning_horizon": planning_horizon,
                            "step": row["step"], "output": str(home / "production" / key / f"step_{step}")}
                    tasks.append(task)
                    if kind == "prior":
                        backbone.setdefault("prior_bundles", {})[step] = str(Path(task["output"]) / "bundle")
        prepared["backbones"].append(backbone)
    write(output_root / "preparation.json", prepared)
    return prepared


def validated_campaign(path):
    campaign = read(path)
    require(campaign["schema_version"] == 1, "Unsupported campaign schema")
    require(source_commit() == campaign["source_commit"], "Source commit changed")
    require(digest(RECIPE) == campaign["recipe_sha256"], "Campaign recipe changed")
    for backbone in campaign["backbones"]:
        require(digest(backbone["checkpoint_inventory"]) == backbone["checkpoint_inventory_sha256"], "Checkpoint inventory changed")
        for matrix in backbone["matrices"].values():
            require(digest(matrix["path"]) == matrix["sha256"], "Evaluation matrix changed")
    return campaign


def validate_result(payload, task, *, smoke=False, device="cuda"):
    expected_seeds = SEEDS[:2] if smoke else SEEDS
    expected_count = 1 if task["planning_horizon"] is None else 2
    require(len(payload["results"]) == expected_count, "Missing evaluation arm")
    for result in payload["results"]:
        require(result["outer_state_unchanged"] is True and not result["nonfinite_model_metrics"]
                and not result["nonfinite_trace_metrics"], "Frozen/finite checks failed")
        require(result["outer_updates_before"] == result["outer_updates_after"], "Outer optimizer updates occurred")
        require(result["resolved_device"].startswith(device), "Unexpected evaluation device")
        cfg = result["resolved_config"]
        require(cfg["train_unroll_horizon"] == task["train_horizon"], "Evaluator changed the training horizon")
        require(cfg["inner_actor_source"] == cfg["inner_horizon_actor_source"] == "sac", "Actor route changed")
        require([e["seed"] for e in result["episodes"]] == expected_seeds, "Episode pairing differs")
        for episode in result["episodes"]:
            require(math.isfinite(episode["return"]), "Nonfinite episode return")
            require(episode["length"] > 0, "Empty evaluation episode")
            if smoke:
                require(episode["length"] == 3, "Smoke must execute all three decisions")
            else:
                require(episode["length"] <= 500 and (episode["terminated"] or episode["truncated"])
                        and not episode.get("truncated_by_evaluator") and not episode.get("capped"), "Incomplete full episode")
        if task["planning_horizon"] is not None:
            horizon = task["planning_horizon"]
            require(cfg["inner_rollout_horizon"] == horizon, "Planning horizon changed")
            require(cfg["inner_mppi_num_samples"] == 512 and cfg["inner_mppi_iterations"] == 8, "MPPI sample budget changed")
            require(cfg["mppi_terminal_q_reduction"] == "mean_pair", "Terminal Q reduction changed")
            require(result["model_metrics"]["inner_model_steps"]["mean"] == model_steps(horizon), "Model-step budget changed")
            for component in ("actor", "critic", "temperature"):
                require(result["model_metrics"][f"inner_{component}_optimizer_steps"]["mean"] == 0, "Inner optimizer updated")
    if expected_count == 2:
        require([r["value_routing"]["inner_horizon_critic_source"] for r in payload["results"]] ==
                ["sac", "aux_return"], "Expected soft-Q and auxiliary return-Q terminal routes")


def worker(campaign_path, kind, index, *, smoke=False):
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.eval_series import stage_record
    from utils.eval_series_paired import paired_measurements
    campaign = validated_campaign(campaign_path)
    require(kind in {"prior", "mppi"}, "Unknown task kind")
    tasks = campaign[kind + "_tasks"]
    require(type(index) is int and 0 <= index < len(tasks), "Task index is out of range")
    task = tasks[index]
    backbone = next(b for b in campaign["backbones"] if b["train_horizon"] == task["train_horizon"])
    row = next(r for r in backbone["checkpoints"] if r["step"] == task["step"])
    validate_checkpoint(backbone, row)
    key = "prior" if kind == "prior" else f"mppi_h{task['planning_horizon']}"
    output = Path(campaign["root"]) / "smoke" / kind / f"task_{index}" if smoke else Path(task["output"])
    output.mkdir(parents=True, exist_ok=False)
    run_map = backbone["run_maps"][key]
    payload = evaluate_matrix(backbone["matrices"][key]["path"], row["path"], device="cuda",
        seeds=SEEDS[:2] if smoke else SEEDS, max_steps=3 if smoke else 500,
        bundle_dir=output / "bundle", metadata_path=row.get("metadata_path"),
        checkpoint_inventory=backbone["checkpoint_inventory"],
        eval_run_map=None if smoke else run_map, stage_results=False)
    write(output / "results.json", payload)
    validate_result(payload, task, smoke=smoke)
    if not smoke:
        records = records_for(output / "bundle", backbone["checkpoint_inventory"])
        validate_records(records, row, run_map)
        if kind == "mppi":
            prior, = records_for(backbone["prior_bundles"][str(row["step"])], backbone["checkpoint_inventory"])
            for record in records:
                paired_measurements(record, prior)
        for record in records:
            stage_record(run_map[record["selector"]], record)
    write(output / "validation.json", {"passed": True, "smoke": smoke, "task": task,
          "checkpoint_sha256": row["sha256"], "source_commit": campaign["source_commit"]})
    return {"output": str(output), "passed": True, "smoke": smoke}


def pair(campaign_path, train_horizon=None):
    """Stage existing paired-reference supplements after base rows are published."""
    from utils.eval_series import load_run
    from utils.eval_series_paired import stage_paired_reference
    campaign = validated_campaign(campaign_path)
    result = []
    for backbone in campaign["backbones"]:
        if train_horizon is not None and backbone["train_horizon"] != train_horizon:
            continue
        priors = {}
        for row in backbone["checkpoints"]:
            records = records_for(backbone["prior_bundles"][str(row["step"])], backbone["checkpoint_inventory"])
            prior, = records
            validate_records(records, row, backbone["run_maps"]["prior"])
            priors[row["step"]] = prior
        for horizon in HORIZONS:
            for run_dir in backbone["run_maps"][f"mppi_h{horizon}"].values():
                registry = load_run(run_dir)
                entries = read(Path(run_dir) / "publication.json")["records"]
                original = [(rid, entry) for rid, entry in entries.items() if not entry.get("record_kind")]
                require(sorted(e["checkpoint_step"] for _, e in original) == sorted(priors), "Curve checkpoint grid is incomplete")
                for rid, entry in original:
                    result.append({"run_id": registry["run_id"], **stage_paired_reference(
                        run_dir, rid, priors[entry["checkpoint_step"]])})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--registry-root", type=Path, required=True)
    p.add_argument("--attempt-label", required=True)
    w = commands.add_parser("worker")
    w.add_argument("--campaign", type=Path, required=True)
    w.add_argument("--kind", choices=("prior", "mppi"), required=True)
    w.add_argument("--index", type=int, required=True)
    w.add_argument("--smoke", action="store_true")
    q = commands.add_parser("pair")
    q.add_argument("--campaign", type=Path, required=True)
    q.add_argument("--train-horizon", type=int, choices=HORIZONS)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.inventory, args.output_root, args.registry_root, args.attempt_label)
        result = {"preparation": str(args.output_root / "preparation.json"), "curves": len(result["curves"]),
                  "prior_tasks": len(result["prior_tasks"]), "mppi_tasks": len(result["mppi_tasks"]), "reused": len(result["reused"])}
    elif args.command == "worker":
        result = worker(args.campaign, args.kind, args.index, smoke=args.smoke)
    else:
        result = pair(args.campaign, args.train_horizon)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
