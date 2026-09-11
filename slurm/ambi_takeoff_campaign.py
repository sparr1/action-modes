"""Guarded command assembly for the frozen seven-checkpoint Oscar screen.

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


def load_inventory(path):
    inventory = json.loads(Path(path).read_text())
    if inventory.get("source_run") != SOURCE_RUN:
        raise ValueError("The takeoff screen requires the mey3rxj8 backbone.")
    rows = inventory.get("checkpoints", [])
    if [row.get("step") for row in rows] != list(STEPS):
        raise ValueError("Inventory must contain exactly the seven ordered takeoff checkpoints.")
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


def task_cell(task_index):
    if not 0 <= task_index < 2 * len(STEPS):
        raise ValueError("Task index must be between 0 and 13.")
    return ("episodes" if task_index < len(STEPS) else "real", task_index % len(STEPS))


def _run(arguments, *, env=None):
    command = [sys.executable, *map(str, arguments)]
    print(json.dumps({"command": command}), flush=True)
    subprocess.run(command, check=True, env=env)


def run_worker(inventory_path, output_root, attempt_label, task_index, *, smoke=False, eval_run_map=None):
    if not attempt_label.strip():
        raise ValueError("Choose an explicit nonempty attempt label.")
    mode, checkpoint_index = task_cell(task_index)
    row = load_inventory(inventory_path)[checkpoint_index]
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
    common = ["--matrix", MATRIX, "--preset", SELECTOR, "--checkpoint", checkpoint, "--device", "cuda"]
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
              "--selector", SELECTOR, "--attempt-label", attempt_label,
              "--output", output / "model-series"])
    else:
        arguments = ["evaluate_ambi_calibration.py", "run", *common,
                     "--bundle-dir", output / "bundle", "--attempt-label", attempt_label,
                     "--save-root-bank", output / "simulator-roots.json",
                     "--reference-cache", output / "prior-continuations",
                     "--benchmark-repetitions", "7"]
        if smoke:
            arguments += ["--seeds", "101", "--decisions", "0", "--max-steps", "500",
                          "--solver-repetitions", "1", "--rollout-repetitions", "4", "--tail-steps", "1000"]
        _run(arguments)
    receipt = {"status": "complete", "step": row["step"], "mode": mode, "smoke": smoke,
               "attempt_label": attempt_label, "checkpoint_sha256": row["sha256"],
               "worker_elapsed_seconds": time.perf_counter() - started}
    (output / "worker-completion.json").write_text(json.dumps(receipt, indent=2) + "\n")


def publish_completed(output_root, *, entity, project, task_index=None):
    """Drain after compute ends; retain completed uploads when another cell failed."""
    output_root = Path(output_root)
    if not output_root.is_absolute() or not output_root.is_dir():
        raise ValueError("Publication requires the existing absolute campaign root.")
    results = []
    indices = range(2 * len(STEPS)) if task_index is None else [task_index]
    for index in indices:
        mode, checkpoint_index = task_cell(index)
        output = output_root / "production" / f"step_{STEPS[checkpoint_index]}" / mode
        bundle = output / ("model-series" if mode == "episodes" else "bundle")
        result = {"step": STEPS[checkpoint_index], "mode": mode, "bundle": str(bundle)}
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
    summary = {"results": results, "status": "complete" if all(x["status"] == "complete" for x in results) else "incomplete"}
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
    publisher = commands.add_parser("publish")
    publisher.add_argument("--output-root", type=Path, required=True)
    publisher.add_argument("--entity", default="rwgao_b-brown-university")
    publisher.add_argument("--project", default="ambi-inner-bench")
    publisher.add_argument("--task-index", type=int)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "worker":
        args["inventory_path"] = args.pop("inventory")
        run_worker(**args)
        return 0
    return publish_completed(**args)


if __name__ == "__main__":
    raise SystemExit(main())
