"""Guarded command assembly for a matrix-selected frozen-checkpoint Oscar screen.

This module never submits jobs. GPU workers save results and queue checkpoint
records; the separate CPU publisher uploads only completed diagnostic bundles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.ambi_research import load_preset_matrix


STEPS = (100000, 125000, 150000, 200000, 300000, 500000, 2000000)
SOURCE_RUN = "rwgao_b-brown-university/ambi/mey3rxj8"
MATRIX = "configs/research/ambi_scratch_takeoff_h1.json"
SELECTOR = "initialization/scratch"


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_campaign(matrix=MATRIX):
    """Require an explicit, hash-pinned panel and exactly one selected preset."""
    matrix = Path(matrix)
    if not matrix.is_file():
        raise ValueError(f"Research matrix must be an existing file: {matrix}")
    config = load_preset_matrix(matrix)
    if config.get("source_run") != SOURCE_RUN or config.get("base_alg_config") != "checkpoint":
        raise ValueError("The campaign requires the checkpoint-based mey3rxj8 backbone.")
    checkpoints = config.get("checkpoint_contract", {}).get("checkpoints", [])
    if not isinstance(checkpoints, list) or not checkpoints or any(not isinstance(row, dict) for row in checkpoints):
        raise ValueError("The matrix must pin a nonempty checkpoint panel.")
    steps = tuple(row.get("step") for row in checkpoints)
    if any(type(step) is not int or step not in STEPS for step in steps) or len(set(steps)) != len(steps):
        raise ValueError("The matrix must select distinct supported checkpoint steps.")
    if any(not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64
           or any(char not in "0123456789abcdef" for char in row["sha256"]) for row in checkpoints):
        raise ValueError("Each matrix checkpoint must pin a SHA256.")
    selectors = config.get("evaluation", {}).get("default_presets", [])
    if len(selectors) != 1:
        raise ValueError("The campaign requires exactly one default preset.")
    return matrix, steps, selectors[0]


def load_inventory(path, steps=STEPS):
    inventory = json.loads(Path(path).read_text())
    if inventory.get("source_run") != SOURCE_RUN:
        raise ValueError("The takeoff screen requires the mey3rxj8 backbone.")
    rows = inventory.get("checkpoints", [])
    if [row.get("step") for row in rows] != list(steps):
        raise ValueError("Inventory must contain exactly the matrix-selected ordered checkpoints.")
    return rows


def verify_checkpoint(row):
    checkpoint = Path(row["path"])
    if not checkpoint.is_absolute():
        raise ValueError("Checkpoint paths must be absolute.")
    metadata = Path(str(checkpoint) + ".metadata.json")
    for path, key in ((checkpoint, "sha256"), (metadata, "metadata_sha256")):
        if _hash(path) != row[key]:
            raise ValueError(f"Verified inventory hash mismatch: {path}")
    if json.loads(metadata.read_text()).get("checkpoint", {}).get("step") != row["step"]:
        raise ValueError("Checkpoint metadata step differs from the inventory.")
    return checkpoint


def task_cell(task_index, steps=STEPS):
    if not 0 <= task_index < 2 * len(steps):
        raise ValueError(f"Task index must be between 0 and {2 * len(steps) - 1}.")
    return ("episodes" if task_index < len(steps) else "real", task_index % len(steps))


def _run(arguments, *, env=None):
    command = [sys.executable, *map(str, arguments)]
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, check=True, env=env)


def run_worker(inventory_path, output_root, attempt_label, task_index, *, smoke=False, eval_run_map=None,
               matrix=MATRIX):
    if not attempt_label.strip():
        raise ValueError("Choose an explicit nonempty attempt label.")
    matrix, steps, selector = load_campaign(matrix)
    mode, checkpoint_index = task_cell(task_index, steps)
    row = load_inventory(inventory_path, steps)[checkpoint_index]
    checkpoint = verify_checkpoint(row)
    output_root = Path(output_root)
    if not output_root.is_absolute():
        raise ValueError("Output root must be absolute.")
    if not smoke and mode == "episodes":
        if eval_run_map is None or not Path(eval_run_map).is_file():
            raise ValueError("Production episodes require an explicitly created evaluation run map.")
        reference = Path(row.get("prior_reference_bundle", ""))
        if not reference.is_absolute() or not (reference / "manifest.json").is_file():
            raise ValueError("Production episodes require a verified prior_reference_bundle.")
        if _hash(reference / "manifest.json") != row.get("prior_reference_manifest_sha256"):
            raise ValueError("Prior reference manifest differs from the verified inventory.")
    real_root_bank = real_reference_cache = None
    reuse_keys = ("real_root_bank", "real_root_bank_sha256", "real_reference_cache")
    if not smoke and mode == "real" and any(key in row for key in reuse_keys):
        if not all(row.get(key) for key in reuse_keys):
            raise ValueError("Real reference reuse requires a root bank, its hash, and a reference cache.")
        real_root_bank = Path(row["real_root_bank"])
        real_reference_cache = Path(row["real_reference_cache"])
        if not real_root_bank.is_absolute() or not real_root_bank.is_file():
            raise ValueError("Real root bank must be an existing absolute file.")
        if _hash(real_root_bank) != row["real_root_bank_sha256"]:
            raise ValueError("Real root bank differs from the verified inventory.")
        if not real_reference_cache.is_absolute() or not real_reference_cache.is_dir():
            raise ValueError("Real reference cache must be an existing absolute directory.")
    output = output_root / ("smoke" if smoke else "production") / f"step_{row['step']}" / mode
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    # An unavailable GPU must fail this smoke rather than become a CUDA skip.
    _run(["-c", "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
          "print({'device': torch.cuda.get_device_name(0), 'torch': torch.__version__})"])
    if smoke:
        environment = {**os.environ, "AMBI_RUN_REAL_DMCONTROL_TESTS": "1"}
        _run(["-m", "pytest", "-q", "tests/test_ambi_togo_trace.py",
              "tests/test_ambi_real_calibration.py", "tests/test_ambi_calibration_cli.py",
              "tests/test_ambi_inner_decoupling.py::test_cuda_act_preserves_all_global_rng_streams_and_outer_state"],
             env=environment)
    common = ["--matrix", matrix, "--preset", selector, "--checkpoint", checkpoint, "--device", "cuda"]
    if mode == "episodes":
        arguments = ["evaluate_ambi_checkpoint.py", *common, "--bundle-dir", output / "bundle",
                     "--output", output / "results.json"]
        if smoke:
            arguments += ["--seeds", "101", "--max-steps", "2"]
        else:
            arguments += ["--checkpoint-inventory", inventory_path, "--eval-run-map", eval_run_map,
                          "--reference-bundle", row["prior_reference_bundle"]]
        _run(arguments)
        _run(["report_ambi_benchmark.py", "--bundle", output / "bundle", "--output", output / "report.html"])
        _run(["evaluate_ambi_calibration.py", "export-model", "--bundle", output / "bundle",
              "--selector", selector, "--attempt-label", attempt_label,
              "--output", output / "model-series"])
    else:
        arguments = ["evaluate_ambi_calibration.py", "run", *common,
                     "--bundle-dir", output / "bundle", "--attempt-label", attempt_label,
                     "--benchmark-repetitions", "7"]
        if real_root_bank is None:
            arguments += ["--save-root-bank", output / "simulator-roots.json",
                          "--reference-cache", output / "prior-continuations"]
        else:
            # The evaluator independently checks runtime, science, protocol,
            # and per-reference content identity before accepting cached data.
            arguments += ["--root-bank", real_root_bank, "--reference-cache", real_reference_cache]
        if smoke:
            arguments += ["--seeds", "101", "--decisions", "0", "--max-steps", "500",
                          "--solver-repetitions", "1", "--rollout-repetitions", "4", "--tail-steps", "1000"]
        _run(arguments)
    receipt = {"status": "complete", "step": row["step"], "mode": mode, "smoke": smoke,
               "attempt_label": attempt_label, "checkpoint_sha256": row["sha256"],
               "matrix": str(matrix), "matrix_sha256": _hash(matrix), "selector": selector,
               "checkpoint_steps": list(steps),
               "worker_elapsed_seconds": time.perf_counter() - started}
    if real_root_bank is not None:
        receipt.update({key: row[key] for key in reuse_keys})
    (output / "worker-completion.json").write_text(json.dumps(receipt, indent=2) + "\n")


def publish_completed(output_root, *, entity, project, task_index=None, matrix=MATRIX):
    """Drain after compute ends; retain completed uploads when another cell failed."""
    matrix, steps, selector = load_campaign(matrix)
    output_root = Path(output_root)
    if not output_root.is_absolute() or not output_root.is_dir():
        raise ValueError("Publication requires the existing absolute campaign root.")
    results = []
    indices = range(2 * len(steps)) if task_index is None else [task_index]
    for index in indices:
        mode, checkpoint_index = task_cell(index, steps)
        output = output_root / "production" / f"step_{steps[checkpoint_index]}" / mode
        bundle = output / ("model-series" if mode == "episodes" else "bundle")
        result = {"step": steps[checkpoint_index], "mode": mode, "bundle": str(bundle)}
        try:
            manifest = json.loads((bundle / "manifest.json").read_text())
            if manifest.get("status") != "complete":
                result.update(status="skipped", reason="Incomplete diagnostic bundle")
            else:
                _run(["evaluate_ambi_calibration.py", "publish", "--bundle", bundle,
                      "--mode", "online", "--entity", entity, "--project", project])
                result["status"] = "complete"
        except FileNotFoundError:
            result.update(status="skipped", reason="Missing diagnostic bundle")
        except (subprocess.CalledProcessError, ValueError) as exc:
            result.update(status="failed", reason=str(exc))
        results.append(result)
    summary = {"results": results, "matrix": str(matrix), "matrix_sha256": _hash(matrix),
               "selector": selector, "checkpoint_steps": list(steps),
               "status": "complete" if all(x["status"] == "complete" for x in results) else "incomplete"}
    receipt = output_root / f"publication-summary-{time.time_ns()}.json"
    receipt.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0 if summary["status"] == "complete" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--inventory", type=Path, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--attempt-label", required=True)
    worker.add_argument("--task-index", type=int, required=True)
    worker.add_argument("--smoke", action="store_true")
    worker.add_argument("--eval-run-map", type=Path)
    worker.add_argument("--matrix", type=Path, default=os.environ.get("AMBI_TAKEOFF_MATRIX", MATRIX))
    publisher = commands.add_parser("publish")
    publisher.add_argument("--output-root", type=Path, required=True)
    publisher.add_argument("--entity", default="rwgao_b-brown-university")
    publisher.add_argument("--project", default="ambi-inner-bench")
    publisher.add_argument("--task-index", type=int)
    publisher.add_argument("--matrix", type=Path, default=os.environ.get("AMBI_TAKEOFF_MATRIX", MATRIX))
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "worker":
        args["inventory_path"] = args.pop("inventory")
        run_worker(**args)
        return 0
    return publish_completed(**args)


if __name__ == "__main__":
    raise SystemExit(main())
