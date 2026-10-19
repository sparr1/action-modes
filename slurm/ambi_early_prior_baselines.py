"""Measure frozen actor references for the existing early H3/J8 campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from slurm.ambi_early_h3_j8_campaign import read, sha256, verify_checkpoint, verify_source, write_new
from slurm.ambi_early_h3_j8_publish import load_campaign
from utils.ambi_benchmark import solver_seed

MATRIX = "configs/research/ambi_early_prior_baselines.json"


def validate_result(result, manifest, cell, campaign):
    if result["checkpoint_sha256"] != cell["checkpoint_sha256"]:
        raise ValueError("Reference checkpoint does not match adapted checkpoint")
    rows = result["results"]
    if len(rows) != 1:
        raise ValueError("Reference cell must contain exactly one execution mode")
    row = rows[0]
    if row["selector"] != "early_prior/" + cell["execution"] or row["execution_action"] != cell["execution"]:
        raise ValueError("Reference execution mode does not match adapted execution")
    if row["resolved_config"]["inner_operator"] != "none":
        raise ValueError("Reference must execute the unadapted actor")
    if not row["outer_state_unchanged"] or row["outer_updates_before"] != row["outer_updates_after"]:
        raise ValueError("Reference altered outer state")
    episodes = row["episodes"]
    if len(episodes) != 5 or sorted(ep["seed"] for ep in episodes) != campaign["seeds"]:
        raise ValueError("Reference requires all five paired episode seeds")
    for episode in episodes:
        if episode["solver_seed"] != solver_seed(campaign["controller_seed"], "episode", episode["seed"]):
            raise ValueError("Reference controller seed differs")
        if episode["length"] != 500 or not episode["truncated"] or episode["truncated_by_evaluator"]:
            raise ValueError("Reference requires complete environment-limited episodes")
        for metric in ("inner_actor_optimizer_steps", "inner_critic_optimizer_steps",
                       "inner_temperature_optimizer_steps", "inner_model_steps"):
            if episode["model_metrics"][metric] != 0 or row["model_metrics"][metric]["max"] != 0:
                raise ValueError("Reference performed inner learning or imagination")
    if manifest["status"] != "complete" or manifest["protocol"]["action_rule"] != "tanh_" + cell["execution"]:
        raise ValueError("Reference bundle is incomplete or has a different action protocol")
    return {"passed": True, "seeds": campaign["seeds"], "execution": cell["execution"],
            "episodes": len(episodes), "mean_return": row["return"]["mean"],
            "outer_state_unchanged": True, "inner_updates": 0}


def run(campaign_path, index, output_root):
    campaign = load_campaign(campaign_path)
    cell = campaign["cells"][index]
    source_commit = verify_source()
    verify_checkpoint(cell)
    output = Path(output_root).resolve() / f"cell-{index:02d}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    command = [sys.executable, str(REPO / "evaluate_ambi_checkpoint.py"),
        "--matrix", str(REPO / MATRIX), "--checkpoint", cell["checkpoint"],
        "--metadata", cell["metadata"], "--preset", "early_prior/" + cell["execution"],
        "--device", "cuda", "--seeds", *map(str, campaign["seeds"]),
        "--controller-seed", str(campaign["controller_seed"]), "--max-steps", "500",
        "--output", str(output / "results.json"), "--bundle-dir", str(output / "bundle"),
        "--checkpoint-inventory", cell["inventory_path"]]
    write_new(output / "started.json", {"cell": cell, "source_commit": source_commit,
        "adapted_source_commit": campaign["source_commit"], "command": command})
    started = time.perf_counter()
    subprocess.run(command, cwd=REPO, check=True)
    result = read(output / "results.json")
    manifest = read(output / "bundle" / "manifest.json")
    validation = validate_result(result, manifest, cell, campaign)
    receipt = {"validated": True, "cell_index": index, "validation": validation,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "source_commit": source_commit, "source_run": cell["source_run"],
        "checkpoint_sha256": cell["checkpoint_sha256"],
        "metadata_sha256": cell["metadata_sha256"], "matrix_sha256": sha256(REPO / MATRIX),
        "results_sha256": sha256(output / "results.json"),
        "manifest_sha256": sha256(output / "bundle" / "manifest.json")}
    write_new(output / "receipt.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    run(args.campaign, args.index, args.output_root)
