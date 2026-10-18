"""Pinned, offline H3/J8 checkpoint evaluations; publication is a CPU task."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
STEPS = [100000, 150000, 200000, 400000]
SEEDS = [101, 102, 103, 104, 105]
SOURCES = {
    "mppi_aux": "rwgao_b-brown-university/ambi/mppi7530927x0",
    "sac_aux": "rwgao_b-brown-university/ambi/full7391933x4",
}
MATRIX = "configs/research/ambi_early_h3_j8.json"


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def verify_source(campaign=None):
    sha = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise RuntimeError("Evaluation requires a clean, committed source checkout")
    if campaign and (sha != campaign["source_commit"] or
                     sha256(REPO / MATRIX) != campaign["matrix_sha256"]):
        raise RuntimeError("Campaign source/configuration changed")
    return sha


def verify_checkpoint(row):
    for path_key, hash_key in (("checkpoint", "checkpoint_sha256"),
                               ("metadata", "metadata_sha256")):
        if sha256(row[path_key]) != row[hash_key]:
            raise RuntimeError(f"Changed {path_key}: {row[path_key]}")
    metadata = read(row["metadata"])
    if metadata["checkpoint"]["step"] != row["step"]:
        raise ValueError("Checkpoint step and sidecar disagree")
    params = metadata["trial_run_params"]["alg_params"]
    expected_operator = "mppi" if row["backbone_id"] == "mppi_aux" else "sac"
    if params["inner_operator"] != expected_operator or params["aux_return_mode"] != "sac":
        raise ValueError("Unexpected source backbone")


def prepare(inventory_path, root):
    root = Path(root).resolve()
    sha = verify_source()
    rows = read(inventory_path)["checkpoints"]
    expected = {(backbone, step) for backbone in SOURCES for step in STEPS}
    if len(rows) != len(expected) or {(r["backbone_id"], r["step"]) for r in rows} != expected:
        raise ValueError("Inventory must contain exactly the selected eight checkpoints")
    for row in rows:
        if row["source_run"] != SOURCES[row["backbone_id"]]:
            raise ValueError("Wrong checkpoint source run")
        verify_checkpoint(row)
    campaign = dict(schema="ambi-early-h3-j8-v1", campaign_id="early-h3-j8-20261018",
                    source_commit=sha, source_tree=git("rev-parse", "HEAD^{tree}"),
                    matrix=MATRIX, matrix_sha256=sha256(REPO / MATRIX),
                    seeds=SEEDS, controller_seed=55, max_steps=500, cells=[])
    for backbone in SOURCES:
        inventory = root / "inputs" / f"{backbone}.json"
        selected = sorted((r for r in rows if r["backbone_id"] == backbone), key=lambda r: r["step"])
        write_new(inventory, {"source_run": SOURCES[backbone], "checkpoints": [
            {**row, "path": row["checkpoint"], "sha256": row["checkpoint_sha256"]} for row in selected]})
        for row in selected:
            for execution in ("mean", "sample"):
                index = len(campaign["cells"])
                campaign["cells"].append({**row, "index": index, "execution": execution,
                    "selector": f"early_h3_j8/{execution}", "inventory_path": str(inventory),
                    "output_dir": str(root / "production" / f"cell-{index:02d}")})
    write_new(root / "campaign.json", campaign)
    return campaign


def evaluator_command(campaign, cell, output, *, smoke_steps=None, eval_run_dir=None):
    seeds = [101] if smoke_steps is not None else campaign["seeds"]
    max_steps = smoke_steps if smoke_steps is not None else campaign["max_steps"]
    command = [sys.executable, str(REPO / "evaluate_ambi_checkpoint.py"),
               "--matrix", str(REPO / campaign["matrix"]),
               "--checkpoint", cell["checkpoint"], "--metadata", cell["metadata"],
               "--preset", cell["selector"], "--device", "cuda",
               "--seeds", *map(str, seeds), "--controller-seed", str(campaign["controller_seed"]),
               "--max-steps", str(max_steps), "--output", str(output / "results.json"),
               "--bundle-dir", str(output / "bundle"),
               "--checkpoint-inventory", cell["inventory_path"]]
    if eval_run_dir:
        command += ["--eval-run-dir", str(eval_run_dir)]
    return command, seeds, max_steps


def run_cell(campaign_path, index, smoke_steps=None):
    campaign_path = Path(campaign_path).resolve()
    campaign = read(campaign_path)
    verify_source(campaign)
    cell = campaign["cells"][index]
    assert cell["index"] == index
    verify_checkpoint(cell)
    cpu = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    output = (campaign_path.parent / "smoke" / f"cell-{index:02d}-cpu{cpu}-steps{smoke_steps}"
              if smoke_steps is not None else Path(cell["output_dir"]))
    if output.exists():
        raise FileExistsError(f"Refusing to repeat/overwrite {output}")
    output.mkdir(parents=True)
    if smoke_steps is None:
        publication = read(campaign_path.parent / "publication" / "publication-map.json")
        # Registry allocation must precede compute, but only the CPU publisher
        # stages records after receipt validation. The evaluator's auto-stage
        # path would expose a complete bundle before this validation finishes.
        if not publication["cell_run_dirs"].get(str(index)):
            raise RuntimeError("No publication identity allocated for cell")
    command, seeds, max_steps = evaluator_command(campaign, cell, output,
                                                 smoke_steps=smoke_steps)
    write_new(output / "started.json", {"cell": cell, "command": command,
                                        "job_id": os.environ.get("SLURM_JOB_ID"), "cpus": cpu})
    started = time.perf_counter()
    subprocess.run(command, cwd=REPO, check=True)
    from slurm.ambi_early_h3_j8_validation import validate_result
    validation = validate_result(output / "results.json", output / "bundle",
                                 expected_seeds=seeds, max_steps=max_steps)
    receipt = {"cell_index": index, "smoke": smoke_steps is not None, "validated": True,
               "elapsed_seconds": time.perf_counter() - started,
               "peak_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
               "cpus": cpu, "validation": validation,
               "results_sha256": sha256(output / "results.json"),
               "manifest_sha256": sha256(output / "bundle" / "manifest.json")}
    write_new(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--inventory", required=True, type=Path)
    p.add_argument("--root", required=True, type=Path)
    p = sub.add_parser("run")
    p.add_argument("--campaign", required=True, type=Path)
    p.add_argument("--index", required=True, type=int)
    p.add_argument("--smoke-steps", type=int, choices=(32, 500))
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.inventory, args.root)
        print(json.dumps({"cells": len(result["cells"]), "root": str(args.root)}))
    else:
        print(json.dumps(run_cell(args.campaign, args.index, args.smoke_steps), allow_nan=False))


if __name__ == "__main__":
    main()
