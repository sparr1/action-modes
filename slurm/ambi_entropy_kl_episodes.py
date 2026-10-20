"""Pinned sampled episodes with automatic entropy plus a fixed prior-KL penalty.

One worker owns all five seeds at one checkpoint/coefficient. Compatible prior
and entropy baseline episodes are reused; GPU workers never initialize W&B.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from slurm.ambi_early_h3_j8_campaign import read, write_new, sha256, verify_checkpoint

MATRIX = "configs/research/ambi_entropy_kl_episodes.json"
PROFILE = "configs/research/ambi_entropy_probe.json"
BASELINE_MATRIX = "configs/research/ambi_early_h3_j8.json"
REFERENCE_ROOT = Path("/oscar/scratch/rgao48/ambi/early-h3-j8-eval/20261018-a0d73c82")
CAMPAIGN_ID = "entropy-kl-episodes-20261020"
CELLS = {"sac_aux_100k": "09", "mppi_aux_400k": "07"}
OUTER_ALPHAS = {"sac_aux_100k": 0.003365497337654233,
                "mppi_aux_400k": 0.013081684708595276}
BETA_MULTIPLIERS = (1, 3)
TEMPERATURE_LR = 0.0003
SEEDS = [101, 102, 103, 104, 105]


def source():
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip():
        raise RuntimeError("Use a clean committed source checkout")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def intervention(name, multiplier):
    if name not in CELLS or type(multiplier) is not int or multiplier not in BETA_MULTIPLIERS:
        raise ValueError("Unknown checkpoint or beta multiplier")
    return {"checkpoint_id": name, "cell_id": f"{name}__beta{multiplier}",
            "selector": f"entropy_kl/{name}_beta{multiplier}",
            "beta_multiplier": multiplier, "beta": OUTER_ALPHAS[name] * multiplier,
            "temperature_lr": TEMPERATURE_LR}


def validate_matrix(matrix, baseline):
    """Reject any second scientific intervention before creating run identities."""
    if matrix["shared_alg_params"] != baseline["shared_alg_params"]:
        raise ValueError("Shared settings must equal the existing entropy baseline")
    if matrix.get("base_alg_config") != "checkpoint":
        raise ValueError("The episode matrix must derive its networks from the checkpoint")
    for key in ("seeds", "controller_seed", "max_steps", "togo_return_rollouts", "togo_return_every"):
        if matrix["evaluation"][key] != baseline["evaluation"][key]:
            raise ValueError(f"Changed evaluation setting: {key}")
    variants = matrix["comparisons"]["entropy_kl"]["variants"]
    expected = {}
    for name in CELLS:
        for multiplier in BETA_MULTIPLIERS:
            cell = intervention(name, multiplier)
            expected[cell["selector"].split("/")[1]] = cell["beta"]
    if set(variants) != set(expected) or set(matrix["comparisons"]) != {"entropy_kl"}:
        raise ValueError("Expected exactly the four pinned entropy-plus-KL cells")
    for key, beta in expected.items():
        if (variants[key]["alg_params"] != {"inner_outer_policy_kl_coef": beta}
                or variants[key].get("evaluation") != {"execution_action": "sample"}):
            raise ValueError("Only the fixed prior-KL coefficient may change")
    if (matrix["evaluation"].get("execution_action") != "sample"
            or matrix["evaluation"]["default_presets"] != ["entropy_kl/" + key for key in expected]):
        raise ValueError("Episode selectors must use paired sampled execution")


def validate_campaign(campaign, profile):
    expected_top = {"schema_version": 1, "campaign_id": CAMPAIGN_ID, "seeds": SEEDS,
                    "controller_seed": 55, "max_steps": 500, "execution_action": "sample"}
    if any(campaign.get(key) != value for key, value in expected_top.items()):
        raise ValueError("Campaign protocol changed")
    expected = [intervention(name, multiplier) for name in CELLS for multiplier in BETA_MULTIPLIERS]
    if len(campaign["cells"]) != len(expected):
        raise ValueError("Expected exactly four episode cells")
    for index, (cell, spec) in enumerate(zip(campaign["cells"], expected)):
        pinned = {**profile["checkpoints"][spec["checkpoint_id"]], **spec, "index": index}
        if any(cell.get(key) != value for key, value in pinned.items()):
            raise ValueError("Cell checkpoint, coefficient, selector or order changed")


def prepare(root):
    from utils import eval_series as series
    root = Path(root).resolve()
    if (root / "campaign.json").exists():
        raise FileExistsError("An episode campaign already exists here")
    profile, matrix = read(REPO / PROFILE), read(REPO / MATRIX)
    validate_matrix(matrix, read(REPO / BASELINE_MATRIX))
    campaign = {"schema_version": 1, "campaign_id": CAMPAIGN_ID,
                "source_commit": source(), "matrix_sha256": sha256(REPO / MATRIX),
                "profile_sha256": sha256(REPO / PROFILE), "seeds": SEEDS,
                "controller_seed": 55, "max_steps": 500, "execution_action": "sample", "cells": []}
    for name, reference_cell in CELLS.items():
        row = dict(profile["checkpoints"][name])
        verify_checkpoint(row)
        prior = REFERENCE_ROOT / "baselines-8401ce40" / f"cell-{reference_cell}" / "bundle"
        baseline = REFERENCE_ROOT / "production" / f"cell-{reference_cell}" / "bundle"
        inventory = REFERENCE_ROOT / "inputs" / f"{row['backbone_id']}.json"
        for multiplier in BETA_MULTIPLIERS:
            spec = intervention(name, multiplier)
            cell_id = spec["cell_id"]
            cell = {**row, **spec, "index": len(campaign["cells"]),
                    "reference_bundle": str(prior), "baseline_bundle": str(baseline),
                    "reference_manifest_sha256": sha256(prior / "manifest.json"),
                    "baseline_manifest_sha256": sha256(baseline / "manifest.json"),
                    "inventory_path": str(inventory), "inventory_sha256": sha256(inventory),
                    "output_dir": str(root / "production" / cell_id)}
            spec_dir = root / "specs" / cell_id
            command = evaluator_command(cell, root / "specification-only" / cell_id)
            command += ["--eval-series-spec-dir", str(spec_dir)]
            subprocess.run(command, cwd=REPO, check=True)
            registry_spec = read(spec_dir / (cell["selector"].replace("/", "__") + ".json"))
            registry = series.create_run(root / "publication" / "runs", registry_spec,
                                         CAMPAIGN_ID, "ambi-inner-bench",
                                         "rwgao_b-brown-university", "oscar-rgao48")
            cell.update(run_dir=registry["run_dir"], run_id=registry["run_id"])
            campaign["cells"].append(cell)
    validate_campaign(campaign, profile)
    if source() != campaign["source_commit"]:
        raise RuntimeError("Source changed during campaign preparation")
    write_new(root / "campaign.json", campaign)
    return campaign


def evaluator_command(cell, output, smoke_steps=None):
    if smoke_steps is not None and (type(smoke_steps) is not int or smoke_steps not in (8, 32)):
        raise ValueError("Smoke must contain 8 or 32 complete decisions")
    smoke = smoke_steps is not None
    command = [sys.executable, str(REPO / "evaluate_ambi_checkpoint.py"),
               "--matrix", str(REPO / MATRIX), "--checkpoint", cell["checkpoint"],
               "--metadata", cell["metadata"], "--preset", cell["selector"], "--device", "cuda",
               "--seeds", *map(str, [101] if smoke else SEEDS), "--controller-seed", "55",
               "--max-steps", str(smoke_steps if smoke else 500), "--output", str(output / "results.json"),
               "--bundle-dir", str(output / "bundle"), "--checkpoint-inventory", cell["inventory_path"]]
    if not smoke:
        command += ["--reference-bundle", cell["reference_bundle"]]
    return command


def validate_episode_result(output, cell, smoke_steps=None):
    from slurm.ambi_early_h3_j8_validation import validate_result
    seeds = [101] if smoke_steps is not None else SEEDS
    steps = smoke_steps if smoke_steps is not None else 500
    validation = validate_result(output / "results.json", output / "bundle",
        expected_seeds=seeds, max_steps=steps, expected_temperature_steps=32,
        config_overrides={"inner_entropy_enabled": True, "inner_temperature_mode": "auto",
                          "inner_temperature_lr": TEMPERATURE_LR,
                          "inner_outer_policy_kl_coef": cell["beta"]})
    if validation["selector"] != cell["selector"] or validation["execution_action"] != "sample":
        raise ValueError("Wrong selector or action execution mode")
    result = read(output / "results.json")["results"][0]
    for metric in ("inner_final_outer_policy_kl", "inner_outer_policy_kl", "inner_actor_entropy",
                   "inner_alpha_initial", "inner_alpha_final"):
        aggregate = result["model_metrics"].get(metric, {})
        if (aggregate.get("count") != len(seeds) * steps
                or any(not isinstance(aggregate.get(key), (int, float))
                       or not math.isfinite(aggregate[key]) for key in ("mean", "min", "max"))):
            raise ValueError(f"Incomplete joint-objective diagnostic: {metric}")
    return validation


def trace_hashes(bundle):
    bundle = Path(bundle).resolve(strict=True)
    runs = read(bundle / "manifest.json")["runs"]
    if len(runs) != 1:
        raise ValueError("Expected one run in the trace bundle")
    names = runs[0]["trace_files"]
    if not names or len(names) != len(set(names)):
        raise ValueError("Missing or duplicate trace files")
    hashes = {}
    for name in names:
        path = (bundle / name).resolve(strict=True)
        if not path.is_relative_to(bundle) or not path.name.endswith(".jsonl.gz"):
            raise ValueError("Invalid trace path")
        hashes[name] = sha256(path)
    return hashes


def run(campaign_path, index, smoke_steps=None):
    campaign_path = Path(campaign_path).resolve()
    campaign = read(campaign_path)
    if (source() != campaign["source_commit"]
            or sha256(REPO / MATRIX) != campaign["matrix_sha256"]
            or sha256(REPO / PROFILE) != campaign["profile_sha256"]):
        raise RuntimeError("Campaign source or configuration changed")
    validate_matrix(read(REPO / MATRIX), read(REPO / BASELINE_MATRIX))
    validate_campaign(campaign, read(REPO / PROFILE))
    if type(index) is not int or index not in range(4):
        raise ValueError("Episode cell index must be in [0, 3]")
    cell = campaign["cells"][index]
    verify_checkpoint(cell)
    for kind in ("reference", "baseline"):
        if sha256(Path(cell[kind + "_bundle"]) / "manifest.json") != cell[kind + "_manifest_sha256"]:
            raise RuntimeError("A reused reference changed")
    if sha256(cell["inventory_path"]) != cell["inventory_sha256"]:
        raise RuntimeError("Checkpoint inventory changed")
    output = (campaign_path.parent / "smoke" / cell["cell_id"]
              if smoke_steps is not None else Path(cell["output_dir"]))
    command = evaluator_command(cell, output, smoke_steps)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "started.json", {"command": command, "job_id": os.environ.get("SLURM_JOB_ID")})
    started = time.perf_counter()
    try:
        subprocess.run(command, cwd=REPO, check=True)
        validation = validate_episode_result(output, cell, smoke_steps)
        if source() != campaign["source_commit"]:
            raise RuntimeError("Source changed during evaluation")
        receipt = {"validated": True, "smoke": smoke_steps is not None,
                   **{key: cell[key] for key in ("checkpoint_id", "cell_id", "beta_multiplier",
                                                "beta", "temperature_lr")},
                   "source_commit": campaign["source_commit"], "validation": validation,
                   "elapsed_seconds": time.perf_counter() - started,
                   "results_sha256": sha256(output / "results.json"),
                   "manifest_sha256": sha256(output / "bundle" / "manifest.json"),
                   "trace_sha256": trace_hashes(output / "bundle")}
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
    p.add_argument("--index", type=int, choices=range(4), required=True)
    p.add_argument("--smoke-steps", type=int, choices=(8, 32))
    args = parser.parse_args()
    result = prepare(args.root) if args.command == "prepare" else run(args.campaign, args.index, args.smoke_steps)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
