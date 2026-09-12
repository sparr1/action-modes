"""Guarded, seed-sharded execution of the mean-prefix measurement experiment.

Workers own disjoint outputs and never publish. A CPU merge must validate the
entire seed panel before checkpoint or diagnostic publication is enabled.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm.ambi_takeoff_campaign import _hash, _run, load_campaign, load_inventory, verify_checkpoint
from utils.ambi_research import load_preset_matrix

MATRIX = "configs/research/ambi_prior_mean_prefix_h1_20seeds.json"


def campaign(matrix=MATRIX):
    path, steps, selector = load_campaign(matrix)
    config = load_preset_matrix(path)
    seeds = config["evaluation"]["seeds"]
    shards = config.get("oscar_seed_shards")
    if (not isinstance(shards, list) or not shards
            or any(not isinstance(s, list) or not s for s in shards)
            or [seed for shard in shards for seed in shard] != seeds):
        raise ValueError("Seed shards must partition the ordered evaluation seeds exactly.")
    if config.get("real_calibration", {}).get("prefix_action_rule") != "mean":
        raise ValueError("This campaign requires an explicit mean prefix.")
    if selector != "initialization/inherited":
        raise ValueError("This campaign requires the inherited initialization preset.")
    return path, steps, selector, shards


def cell(task_index, steps, shards):
    count = len(steps) * len(shards)
    if type(task_index) is not int or not 0 <= task_index < 2 * count:
        raise ValueError(f"Task index must be in [0, {2 * count}).")
    local = task_index % count
    return ("episodes" if task_index < count else "real", local // len(shards), local % len(shards))


def _output_root(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("Campaign output root must be absolute.")
    return path


def run_worker(inventory, output_root, attempt_label, task_index, *, matrix=MATRIX, smoke=False):
    if not attempt_label.strip():
        raise ValueError("Choose an explicit new attempt label.")
    matrix, steps, selector, shards = campaign(matrix)
    mode, checkpoint_index, shard_index = cell(task_index, steps, shards)
    row = load_inventory(inventory, steps)[checkpoint_index]
    checkpoint = verify_checkpoint(row)
    seeds = shards[shard_index][:1] if smoke else shards[shard_index]
    output = (_output_root(output_root) / ("smoke-shards" if smoke else "shards")
              / f"step_{row['step']}" / mode / f"shard_{shard_index}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    _run(["-c", "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
          "print({'device':torch.cuda.get_device_name(0),'torch':torch.__version__})"])
    if smoke:
        _run(["-m", "pytest", "-q", "tests/test_ambi_togo_trace.py",
              "tests/test_ambi_real_calibration.py", "tests/test_ambi_calibration_cli.py",
              "tests/test_ambi_inner_decoupling.py::test_cuda_act_preserves_all_global_rng_streams_and_outer_state"],
             env={**os.environ, "AMBI_RUN_REAL_DMCONTROL_TESTS": "1"})
    common = ["--matrix", matrix, "--checkpoint", checkpoint, "--device", "cuda"]
    if mode == "episodes":
        args = ["evaluate_ambi_checkpoint.py", *common, "--preset", "initialization/prior",
                "--preset", selector, "--bundle-dir", output / "bundle",
                "--output", output / "results.json", "--checkpoint-inventory", inventory,
                "--seeds", *seeds]
        if smoke:
            args += ["--max-steps", "2"]
        _run(args)
        _run(["merge_ambi_seed_shards.py", "seal-episodes", "--bundle", output / "bundle"])
    else:
        args = ["evaluate_ambi_calibration.py", "run", *common, "--preset", selector,
                "--bundle-dir", output / "bundle", "--attempt-label", attempt_label,
                "--save-root-bank", output / "simulator-roots.json",
                "--reference-cache", output / "prior-continuations", "--seeds", *seeds]
        if smoke:
            args += ["--decisions", "0", "--max-steps", "500", "--solver-repetitions", "1",
                     "--rollout-repetitions", "4", "--tail-steps", "1000", "--benchmark-repetitions", "7"]
        _run(args)
    receipt = dict(status="complete", step=row["step"], mode=mode, shard_index=shard_index,
                   task_index=task_index, seeds=seeds, smoke=smoke, attempt_label=attempt_label,
                   checkpoint_sha256=row["sha256"], matrix=str(matrix), matrix_sha256=_hash(matrix),
                   selector=selector, worker_elapsed_seconds=time.perf_counter() - started)
    (output / "worker-completion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


def finalize_checkpoint(inventory, output_root, attempt_label, checkpoint_index, *, matrix=MATRIX,
                        smoke=False, shard_indices=None, eval_run_map=None):
    matrix, steps, selector, shards = campaign(matrix)
    if type(checkpoint_index) is not int or not 0 <= checkpoint_index < len(steps):
        raise ValueError("Invalid checkpoint index.")
    indices = list(range(len(shards))) if shard_indices is None else shard_indices
    if (not indices or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < len(shards) for i in indices)
            or (not smoke and indices != list(range(len(shards))))):
        raise ValueError("Production must merge every seed shard exactly once.")
    expected_seeds = [s for i in indices for s in (shards[i][:1] if smoke else shards[i])]
    row = load_inventory(inventory, steps)[checkpoint_index]
    run_map = None
    if not smoke:
        if eval_run_map is None:
            raise ValueError("Production requires the explicitly created full-panel run identity.")
        run_map = json.loads(Path(eval_run_map).read_text())
        if set(run_map) != {selector} or not Path(run_map[selector]).is_absolute():
            raise ValueError("Unexpected publication mapping.")
    root = _output_root(output_root)
    output = root / ("smoke" if smoke else "production") / f"step_{row['step']}"
    started = time.perf_counter()
    all_sources = {}
    for mode in ("episodes", "real"):
        source_dirs = [root / ("smoke-shards" if smoke else "shards") / f"step_{row['step']}"
                       / mode / f"shard_{i}" for i in indices]
        for i, source in zip(indices, source_dirs):
            receipt = json.loads((source / "worker-completion.json").read_text())
            expected = dict(status="complete", step=row["step"], mode=mode, shard_index=i,
                            seeds=shards[i][:1] if smoke else shards[i], smoke=smoke,
                            attempt_label=attempt_label, checkpoint_sha256=row["sha256"],
                            matrix_sha256=_hash(matrix), selector=selector)
            if any(receipt.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Worker receipt differs from the selected campaign: {source}")
        all_sources[mode] = source_dirs
    for mode, source_dirs in all_sources.items():
        _run(["merge_ambi_seed_shards.py", mode, "--sources", *[p / "bundle" for p in source_dirs],
              "--output", output / mode / "bundle", "--seeds", *expected_seeds])
    _run(["report_ambi_benchmark.py", "--bundle", output / "episodes/bundle",
          "--output", output / "episodes/report.html"])
    _run(["evaluate_ambi_calibration.py", "export-model", "--bundle", output / "episodes/bundle",
          "--selector", selector, "--attempt-label", attempt_label,
          "--output", output / "episodes/model-series"])
    if not smoke:
        _run(["eval_series.py", "append", run_map[selector], output / "episodes/bundle",
              "--selector", selector, "--checkpoint-inventory", inventory])
    receipt = dict(status="complete", step=row["step"], seeds=expected_seeds, smoke=smoke,
                   source_shards=indices, matrix_sha256=_hash(matrix), attempt_label=attempt_label,
                   elapsed_seconds=time.perf_counter() - started, staged=not smoke)
    (output / "merge-completion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("worker")
    worker.add_argument("--task-index", type=int, required=True)
    merge = commands.add_parser("merge")
    merge.add_argument("--checkpoint-index", type=int, required=True)
    merge.add_argument("--shard-indices", nargs="+", type=int)
    merge.add_argument("--eval-run-map", type=Path)
    for command in (worker, merge):
        command.add_argument("--inventory", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--attempt-label", required=True)
        command.add_argument("--matrix", type=Path, default=MATRIX)
        command.add_argument("--smoke", action="store_true")
    args = vars(parser.parse_args(argv))
    (run_worker if args.pop("command") == "worker" else finalize_checkpoint)(**args)


if __name__ == "__main__":
    main()
