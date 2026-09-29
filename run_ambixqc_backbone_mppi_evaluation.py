"""Paired frozen prior/soft-tail MPPI evaluation of the six 500k XQC banks.

The inventory is cell-major, contains all 120 checkpoints, and keeps paths
relative to its training root so an unchanged inventory can accompany a copy.
No training replay is loaded or used by this evaluation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

from run_ambixqc_mppi_evaluation import MATRIX, file_sha256, validate_bundle

ROOT = Path(__file__).resolve().parent
TRAINING_SOURCE_SHA = "5ac28350b273200fb268609313f937eee7a32c22"
INVENTORY_SCHEMA = "ambixqc-backbone-mppi-inventory-v1"
RUN_MAP_SCHEMA = "ambixqc-backbone-mppi-run-map-v1"
CELLS = tuple(f"{arm}_utd{ratio}" for ratio in (1, 2)
              for arm in ("baseline", "aux_shared", "aux_detached"))
STEPS = tuple(range(25_000, 500_001, 25_000))
SELECTORS = ("controller/prior", "controller/mppi")
SOURCE_RUNS = {
    cell: f"rwgao_b-brown-university/ambi/axqc-500k-{cell}-5ac28350-{job}-{index}"
    for index, (cell, job) in enumerate(zip(CELLS, (39079, 39080, 39081, 39082, 39083, 39078)))
}
SOURCE_CONTRACT = {
    "algorithm": "AMBIXQC/AMBIXQC", "environment": "DMControl-v0", "task": "humanoid-walk",
    "observation": "state", "observation_shape": [67], "action_dim": 21,
    "seed": 55, "total_decisions": 500_000, "checkpoint_every": 25_000,
    "replay_capacity": 1_000_000, "world_utd": 1, "inner_operator": "none",
    "cells": list(CELLS), "checkpoint_steps": list(STEPS),
}


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _relative_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Inventory paths must be nonempty relative paths.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
        raise ValueError("Inventory paths must be canonical relative paths without traversal.")
    return path


def _child(root, relative):
    path = (root / _relative_path(relative)).resolve()
    if root not in path.parents:
        raise ValueError("Inventory path escapes its checkpoint root.")
    if not path.is_file():
        raise ValueError(f"Missing inventory file: {path}")
    return path


def _check_hash(path, expected, label):
    if not _hash(expected) or file_sha256(path) != expected:
        raise ValueError(f"{label} hash differs from the inventory: {path}")


def load_inventory(path, *, checkpoint_root=None):
    path = Path(path).resolve()
    inventory = read_json(path)
    if (inventory.get("schema") != INVENTORY_SCHEMA
            or inventory.get("training_source_sha") != TRAINING_SOURCE_SHA
            or inventory.get("source_contract") != SOURCE_CONTRACT):
        raise ValueError("Inventory does not identify the six completed 500k training banks.")
    rows = inventory.get("checkpoints", [])
    expected = [(cell, step) for cell in CELLS for step in STEPS]
    if (not isinstance(rows, list) or len(rows) != len(expected)
            or any(not isinstance(row, dict) or type(row.get("step")) is not int for row in rows)
            or [(row.get("cell"), row.get("step")) for row in rows] != expected):
        raise ValueError("Expected 120 cell-major checkpoints, 25k through 500k for each of six cells.")
    seen = set()
    for row in rows:
        relative = str(_relative_path(row.get("path")))
        if relative in seen:
            raise ValueError("Checkpoint paths must be unique across all six banks.")
        seen.add(relative)
        if (row.get("source_run") != SOURCE_RUNS[row["cell"]]
                or not _hash(row.get("sha256")) or not _hash(row.get("metadata_sha256"))):
            raise ValueError("Checkpoint source run or content hash is invalid.")
    evidence = inventory.get("training_validations", {})
    if not isinstance(evidence, dict) or set(evidence) != set(CELLS):
        raise ValueError("Inventory must pin all six training validation files.")
    for record in evidence.values():
        if not isinstance(record, dict) or not _hash(record.get("sha256")):
            raise ValueError("Invalid training validation hash.")
        _relative_path(record.get("path"))
    location = checkpoint_root if checkpoint_root is not None else inventory.get("checkpoint_root")
    if not isinstance(location, (str, Path)) or not str(location):
        raise ValueError("Inventory requires a checkpoint root or an explicit override.")
    root = Path(location)
    if not root.is_absolute():
        root = path.parent / root
    return inventory, root.resolve()


def _validate_training_record(record, cell):
    ratio = int(cell[-1])
    if (record.get("schema") != "ambixqc-backbone-replay-500k-validation-v1"
            or record.get("source_sha") != TRAINING_SOURCE_SHA or record.get("arm") != cell
            or record.get("mode") != "production" or record.get("total_steps") != 500_000
            or record.get("seed") != 55 or record.get("xqc_utd") != ratio
            or record.get("replay_capacity") != 1_000_000
            or record.get("lr_transition_steps") != 500_000 * ratio
            or record.get("all_finite") is not True
            or record.get("final_raw_replay_matches_training") is not True):
        raise ValueError(f"Training validation violates the completed source contract for {cell}.")
    rows = record.get("checkpoints", [])
    if (len(rows) != 20 or [row.get("step") for row in rows] != list(STEPS)
            or any(row.get("replay_hashes_verified") is not True for row in rows)):
        raise ValueError(f"Training validation has incomplete checkpoint/replay evidence for {cell}.")
    return rows


def _validate_metadata(metadata, row):
    trial = metadata.get("trial_run_params", {})
    experiment = metadata.get("experiment_params", {})
    params = trial.get("alg_params", {})
    expected = read_json(ROOT / "configs/dmcontrol/algs" /
                         f"ambixqc_humanoid_walk_backbone_replay_500k_{row['cell']}.json")
    # Run names are filled by the scheduler; the science settings stay identical.
    for key, value in expected["alg_params"].items():
        if key != "wandb_run_name" and (key not in params or params[key] != value
                or isinstance(value, bool) and type(params[key]) is not bool):
            raise ValueError(f"Checkpoint metadata has incompatible {key} for {row['cell']}.")
    observation = trial.get("resolved_runtime", {}).get("observation", {})
    replay = metadata.get("replay", {})
    if (metadata.get("schema_version") != 1
            or metadata.get("checkpoint", {}).get("step") != row["step"]
            or trial.get("alg") != SOURCE_CONTRACT["algorithm"] or trial.get("env") != "DMControl-v0"
            or type(trial.get("seed")) is not int or trial["seed"] != 55
            or trial.get("total_steps") != 500_000
            or experiment.get("env_params") != {"task": "humanoid-walk", "obs": "state", "render_mode": None}
            or experiment.get("checkpoint_every") != 25_000 or experiment.get("save_strat") != ["all"]
            or experiment.get("save_replay_buffer") is not True
            or observation.get("mode") != "state" or observation.get("shape") != [67]
            or observation.get("dtype") != "float32" or observation.get("action_dim") != 21
            or observation.get("latent_dim") != 512 or observation.get("episode_length") != 500
            or replay.get("schema") != "ambi-replay-reference" or replay.get("version") != 1
            or replay.get("step") != row["step"] or replay.get("checkpoint_sha256") != row["sha256"]):
        raise ValueError("Checkpoint metadata violates the source state/replay contract.")


def select_checkpoint(manifest_path, index, *, checkpoint_root=None):
    inventory, root = load_inventory(manifest_path, checkpoint_root=checkpoint_root)
    if type(index) is not int or not 0 <= index < len(inventory["checkpoints"]):
        raise ValueError("Checkpoint index must be an integer from 0 through 119.")
    row = dict(inventory["checkpoints"][index])
    evidence = inventory["training_validations"][row["cell"]]
    validation_path = _child(root, evidence["path"])
    _check_hash(validation_path, evidence["sha256"], "Training validation")
    training = _validate_training_record(read_json(validation_path), row["cell"])
    if not (validation_path.parent.parent / "PASS").is_file():
        raise ValueError("Training job PASS marker is absent.")
    if training[STEPS.index(row["step"])]["sha256"] != row["sha256"]:
        raise ValueError("Checkpoint hash differs from completed training validation.")
    checkpoint = _child(root, row["path"])
    _check_hash(checkpoint, row["sha256"], "Checkpoint")
    sidecar = _child(root, row["path"] + ".metadata.json")
    _check_hash(sidecar, row["metadata_sha256"], "Checkpoint metadata")
    _validate_metadata(read_json(sidecar), row)
    row["path"] = str(checkpoint)
    return row


def build_inventory(training_root, output):
    """Hash a completed bank using JSON evidence; no Torch, environment or learner."""
    root = Path(training_root).resolve()
    rows, evidence = [], {}
    for cell in CELLS:
        matches = sorted(root.glob(f"production-{cell}-seed55-job*-task*/run-{cell}/validation.json"))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one completed production validation for {cell}.")
        validation_path = matches[0]
        if not (validation_path.parent.parent / "PASS").is_file():
            raise ValueError(f"Training PASS marker is absent for {cell}.")
        records = _validate_training_record(read_json(validation_path), cell)
        evidence[cell] = {"path": str(validation_path.relative_to(root)), "sha256": file_sha256(validation_path)}
        for record in records:
            # Training validation records contain original absolute Hydra paths.
            # Locate by the unique filename under its own run after relocation.
            candidates = list((validation_path.parent / "training").rglob(Path(record["path"]).name))
            if len(candidates) != 1 or not candidates[0].is_file():
                raise ValueError(f"Missing or ambiguous checkpoint {cell}/{record['step']}.")
            checkpoint = candidates[0]
            row = {"cell": cell, "step": record["step"], "path": str(checkpoint.relative_to(root)),
                   "sha256": record["sha256"], "source_run": SOURCE_RUNS[cell]}
            _check_hash(checkpoint, row["sha256"], "Completed training checkpoint")
            sidecar = Path(str(checkpoint) + ".metadata.json")
            if not sidecar.is_file():
                raise ValueError(f"Checkpoint metadata is absent: {sidecar}")
            row["metadata_sha256"] = file_sha256(sidecar)
            _validate_metadata(read_json(sidecar), row)
            rows.append(row)
    payload = {"schema": INVENTORY_SCHEMA, "training_source_sha": TRAINING_SOURCE_SHA,
               "checkpoint_root": str(root), "source_contract": deepcopy(SOURCE_CONTRACT),
               "training_validations": evidence, "checkpoints": rows}
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves old inventories and requires deliberate new paths.
    with output.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return payload


def resolve_cell_run_map(path, cell, *, mode):
    if mode == "smoke":
        if path is not None:
            raise ValueError("Smoke evaluation must not assign or publish evaluation runs.")
        return {}
    if path is None:
        raise ValueError("Production requires an explicit per-cell --eval-run-map.")
    mapping = read_json(path)
    if mapping.get("schema") != RUN_MAP_SCHEMA or set(mapping.get("cells", {})) != set(CELLS):
        raise ValueError("Evaluation run map must assign exactly the six campaign cells.")
    paths = []
    for assignments in mapping["cells"].values():
        if not isinstance(assignments, dict) or set(assignments) != set(SELECTORS):
            raise ValueError("Each cell must assign separate prior and MPPI evaluation runs.")
        for value in assignments.values():
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError("Evaluation run directories must be absolute paths.")
            paths.append(str(Path(value).resolve()))
    if len(set(paths)) != len(paths):
        raise ValueError("All twelve cell/controller curves require distinct evaluation runs.")
    from utils.ambi_benchmark import resolve_eval_run_map
    return resolve_eval_run_map(SELECTORS, run_map=mapping["cells"][cell])


def prepare_specs(manifest_path, index, directory, *, checkpoint_root=None):
    row = select_checkpoint(manifest_path, index, checkpoint_root=checkpoint_root)
    from evaluate_ambi_checkpoint import evaluate_matrix
    return evaluate_matrix(MATRIX, row["path"], selectors=list(SELECTORS),
                           seeds=list(range(101, 106)), controller_seed=12345, max_steps=500,
                           checkpoint_inventory=manifest_path, source_run=row["source_run"],
                           eval_series_spec_dir=directory)


def run(manifest_path, index, result_root, *, mode="production", device="cuda",
        eval_run_map=None, checkpoint_root=None):
    if mode not in {"smoke", "production"}:
        raise ValueError("Mode must be smoke or production.")
    row = select_checkpoint(manifest_path, index, checkpoint_root=checkpoint_root)
    assigned = resolve_cell_run_map(eval_run_map, row["cell"], mode=mode)
    destination = Path(result_root).resolve() / row["cell"] / f"step_{row['step']}"
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("Evaluation results must be outside the source checkout.")
    from evaluate_ambi_checkpoint import evaluate_matrix
    from report_ambi_benchmark import load_bundles, write_report
    from utils.ambi_benchmark import atomic_json, code_identity, stage_completed_bundle

    code = code_identity()
    if code.get("dirty") is not False or not code.get("commit"):
        raise ValueError("Evaluate the campaign from a clean, committed checkout.")
    destination.mkdir(parents=True, exist_ok=False)
    seeds, max_steps = (list(range(101, 106)), 500) if mode == "production" else ([101, 102], 3)
    provenance = {
        "source_run": row["source_run"], "checkpoint": row,
        "source_sha": code["commit"], "training_source_sha": TRAINING_SOURCE_SHA,
        "manifest_sha256": file_sha256(manifest_path), "matrix_sha256": file_sha256(MATRIX),
        "index": index, "cell": row["cell"], "step": row["step"], "mode": mode,
        "seeds": seeds, "max_steps": max_steps, "controller_seed": 12345,
        "terminal_value_source": "online_xqc_twin_mean", "replay_used": False,
        "eval_run_map": assigned,
    }
    atomic_json(destination / "provenance.json", provenance)
    payload = evaluate_matrix(
        MATRIX, row["path"], selectors=list(SELECTORS), seeds=seeds, controller_seed=12345,
        max_steps=max_steps, device=device, bundle_dir=destination / "bundle",
        eval_run_map=assigned or None, stage_results=False,
        checkpoint_inventory=manifest_path, source_run=row["source_run"],
    )
    atomic_json(destination / "paired.json", payload)
    if payload["checkpoint_sha256"] != row["sha256"]:
        raise ValueError("Evaluated checkpoint hash differs from preflight inventory.")
    validation = validate_bundle(destination / "bundle", seeds=seeds, max_steps=max_steps)
    atomic_json(destination / "validation.json", {**provenance, **validation})
    write_report(load_bundles([destination / "bundle"]), destination / "comparison.html",
                 title=f"AMBI-XQC {row['cell']} prior versus MPPI at {row['step']:,} decisions")
    if assigned:
        status = stage_completed_bundle(destination / "bundle", assigned,
                                        source_run=row["source_run"], inventory_path=manifest_path)
        if set(status) != set(SELECTORS) or any(item["status"] != "queued" for item in status.values()):
            raise RuntimeError("Completed evaluation could not stage all curves; preserve results for upload recovery.")
    (destination / "PASS").write_text("PASS\n")
    print(json.dumps({"cell": row["cell"], "step": row["step"], "output": str(destination), **validation}, sort_keys=True))
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-inventory", action="store_true")
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, help="Override the relocatable inventory's checkpoint root.")
    parser.add_argument("--index", type=int)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--mode", choices=("smoke", "production"), default="production")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-run-map", type=Path)
    parser.add_argument("--eval-series-spec-dir", type=Path)
    args = parser.parse_args(argv)
    if args.build_inventory:
        if not args.training_root or not args.output:
            parser.error("--build-inventory requires --training-root and --output")
        build_inventory(args.training_root, args.output)
    else:
        if args.manifest is None or args.index is None:
            parser.error("Evaluation/specification requires --manifest and --index")
        if args.eval_series_spec_dir:
            if args.eval_run_map or args.result_root:
                parser.error("Prepare specifications separately from result output or run assignments")
            print(json.dumps(prepare_specs(args.manifest, args.index, args.eval_series_spec_dir,
                                           checkpoint_root=args.checkpoint_root), sort_keys=True))
        else:
            if args.result_root is None:
                parser.error("Evaluation requires --result-root")
            run(args.manifest, args.index, args.result_root, mode=args.mode, device=args.device,
                eval_run_map=args.eval_run_map, checkpoint_root=args.checkpoint_root)


if __name__ == "__main__":
    main()
