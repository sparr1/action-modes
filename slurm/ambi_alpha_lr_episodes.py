"""Pinned sampled-action inner-temperature learning-rate sweep.

One worker owns all five seeds at one checkpoint. Baselines are reused from
the compatible early H3/J8 campaign; GPU workers never initialize W&B.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from slurm.ambi_early_h3_j8_campaign import read, write_new, sha256, verify_checkpoint

MATRIX = "configs/research/ambi_alpha_lr_episodes.json"
PROFILE = "configs/research/ambi_entropy_probe.json"
REFERENCE_ROOT = Path("/oscar/scratch/rgao48/ambi/early-h3-j8-eval/20261018-a0d73c82")
CELLS = {"sac_aux_100k": "09", "mppi_aux_400k": "07"}
SEEDS = [101, 102, 103, 104, 105]
RATES = {"0p003": 0.003, "0p03": 0.03, "0p3": 0.3}


def source():
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
        raise RuntimeError("Use a clean committed source checkout")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def prepare(root):
    from utils import eval_series as series
    root = Path(root).resolve()
    if (root / "campaign.json").exists():
        raise FileExistsError("An episode campaign already exists here")
    profile, matrix = read(REPO / PROFILE), read(REPO / MATRIX)
    campaign = {"schema_version": 1, "campaign_id": "alpha-lr-episodes-20261020",
                "source_commit": source(), "matrix_sha256": sha256(REPO / MATRIX),
                "profile_sha256": sha256(REPO / PROFILE), "seeds": SEEDS,
                "controller_seed": 55, "max_steps": 500, "execution_action": "sample", "cells": []}
    for name, reference_cell in CELLS.items():
        for token, temperature_lr in RATES.items():
            row = dict(profile["checkpoints"][name])
            verify_checkpoint(row)
            selector = "alpha_lr/lr_" + token
            cell_id = name + "__lr" + token
            configured = matrix["comparisons"]["alpha_lr"]["variants"]["lr_" + token]["alg_params"]
            if configured != {"inner_temperature_lr": temperature_lr}:
                raise RuntimeError("Only the inner temperature learning rate may change")
            prior = REFERENCE_ROOT / "baselines-8401ce40" / f"cell-{reference_cell}" / "bundle"
            baseline = REFERENCE_ROOT / "production" / f"cell-{reference_cell}" / "bundle"
            cell = {**row, "checkpoint_id": name, "cell_id": cell_id,
                    "index": len(campaign["cells"]), "selector": selector, "temperature_lr": temperature_lr,
                    "reference_bundle": str(prior), "baseline_bundle": str(baseline),
                    "reference_manifest_sha256": sha256(prior / "manifest.json"),
                    "baseline_manifest_sha256": sha256(baseline / "manifest.json"),
                    "inventory_path": str(REFERENCE_ROOT / "inputs" / f"{row['backbone_id']}.json"),
                    "output_dir": str(root / "production" / cell_id)}
            spec_dir = root / "specs" / cell_id
            command = evaluator_command(cell, root / "specification-only" / cell_id)
            command += ["--eval-series-spec-dir", str(spec_dir)]
            subprocess.run(command, cwd=REPO, check=True)
            spec = read(spec_dir / (selector.replace("/", "__") + ".json"))
            registry = series.create_run(root / "publication" / "runs", spec,
                                         campaign["campaign_id"], "ambi-inner-bench",
                                         "rwgao_b-brown-university", "oscar-rgao48")
            cell.update(run_dir=registry["run_dir"], run_id=registry["run_id"])
            campaign["cells"].append(cell)
    write_new(root / "campaign.json", campaign)
    return campaign


def evaluator_command(cell, output, smoke_steps=None):
    command = [sys.executable, str(REPO / "evaluate_ambi_checkpoint.py"),
               "--matrix", str(REPO / MATRIX), "--checkpoint", cell["checkpoint"],
               "--metadata", cell["metadata"], "--preset", cell["selector"], "--device", "cuda",
               "--seeds", *map(str, [101] if smoke_steps else SEEDS), "--controller-seed", "55",
               "--max-steps", str(smoke_steps or 500), "--output", str(output / "results.json"),
               "--bundle-dir", str(output / "bundle"), "--checkpoint-inventory", cell["inventory_path"]]
    if smoke_steps is None:
        command += ["--reference-bundle", cell["reference_bundle"]]
    return command


def run(campaign_path, index, smoke_steps=None):
    from slurm.ambi_early_h3_j8_validation import validate_result
    campaign_path = Path(campaign_path).resolve()
    campaign = read(campaign_path)
    if (source() != campaign["source_commit"]
            or sha256(REPO / MATRIX) != campaign["matrix_sha256"]
            or sha256(REPO / PROFILE) != campaign["profile_sha256"]):
        raise RuntimeError("Campaign source or configuration changed")
    cell = campaign["cells"][index]
    verify_checkpoint(cell)
    for kind in ("reference", "baseline"):
        if sha256(Path(cell[kind + "_bundle"]) / "manifest.json") != cell[kind + "_manifest_sha256"]:
            raise RuntimeError("A reused reference changed")
    output = (campaign_path.parent / "smoke" / cell["cell_id"]
              if smoke_steps else Path(cell["output_dir"]))
    output.mkdir(parents=True, exist_ok=False)
    command = evaluator_command(cell, output, smoke_steps)
    write_new(output / "started.json", {"command": command, "job_id": os.environ.get("SLURM_JOB_ID")})
    started = time.perf_counter()
    try:
        subprocess.run(command, cwd=REPO, check=True)
        validation = validate_result(output / "results.json", output / "bundle",
            expected_seeds=[101] if smoke_steps else SEEDS, max_steps=smoke_steps or 500,
            expected_temperature_steps=32,
            config_overrides={"inner_temperature_lr": cell["temperature_lr"]})
        if source() != campaign["source_commit"]:
            raise RuntimeError("Source changed during evaluation")
        receipt = {"validated": True, "smoke": smoke_steps is not None, "checkpoint_id": cell["checkpoint_id"],
                   "cell_id": cell["cell_id"], "temperature_lr": cell["temperature_lr"],
                   "source_commit": campaign["source_commit"], "validation": validation,
                   "elapsed_seconds": time.perf_counter() - started,
                   "results_sha256": sha256(output / "results.json"),
                   "manifest_sha256": sha256(output / "bundle" / "manifest.json")}
        write_new(output / "receipt.json", receipt)
        return receipt
    except BaseException as exc:
        write_new(output / "failure.json", {"complete": False, "cell_id": cell["cell_id"],
                  "exception_type": type(exc).__name__, "message": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--index", type=int, choices=range(6), required=True)
    p.add_argument("--smoke-steps", type=int, choices=(8, 32))
    args = parser.parse_args()
    result = prepare(args.root) if args.command == "prepare" else run(args.campaign, args.index, args.smoke_steps)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
