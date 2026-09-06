"""Evaluate one immutable checkpoint from the AMBI-XQC 50k MPPI campaign."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MATRIX = ROOT / "configs/research/ambixqc_humanoid_mppi_benchmark.json"
SOURCE_RUN = "rwgao_b-brown-university/ambi/axqc-prior-92441d99-5959199"
STEPS = tuple(range(50_000, 1_500_001, 50_000))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_checkpoint(manifest_path, index):
    """Validate the grid, content hashes and source contract before creating output."""
    manifest = json.loads(Path(manifest_path).read_text())
    rows = manifest.get("checkpoints", [])
    if manifest.get("source_run") != SOURCE_RUN:
        raise ValueError("Expected the seed-55 prior-only AMBI-XQC source run.")
    if (not all(type(row.get("step")) is int for row in rows)
            or [row.get("step") for row in rows] != list(STEPS)
            or len({row.get("path") for row in rows}) != len(rows)):
        raise ValueError("Expected all 30 ordered checkpoints from 50k through 1.5M.")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(rows):
        raise ValueError("Checkpoint index is outside the manifest.")
    row = rows[index]
    checkpoint = Path(row["path"])
    if not checkpoint.is_absolute() or not checkpoint.is_file():
        raise ValueError("Checkpoint must be an existing absolute file.")
    if file_sha256(checkpoint) != row["sha256"]:
        raise ValueError("Checkpoint hash differs from the manifest.")
    sidecar = Path(str(checkpoint) + ".metadata.json")
    if file_sha256(sidecar) != row["metadata_sha256"]:
        raise ValueError("Checkpoint metadata hash differs from the manifest.")
    metadata = json.loads(sidecar.read_text())
    trial, experiment = metadata["trial_run_params"], metadata["experiment_params"]
    params = trial["alg_params"]
    if (metadata["checkpoint"]["step"] != row["step"]
            or trial["alg"] != "AMBIXQC/AMBIXQC"
            or trial["env"] != "DMControl-v0" or trial["seed"] != 55
            or trial["total_steps"] != 1_500_000
            or params["inner_operator"] != "none"
            or params["obs"] != "state" or params["model_size"] != 5
            or params["train_unroll_horizon"] != 3 or params["eval_freq"] is not None
            or experiment["env_params"]["task"] != "humanoid-walk"
            or experiment["env_params"]["obs"] != "state"):
        raise ValueError("Checkpoint metadata violates the source training contract.")
    return row


def validate_bundle(bundle_path, *, seeds, max_steps):
    manifest = json.loads((Path(bundle_path) / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Evaluation bundle did not complete.")
    runs = {run["selector"]: run for run in manifest["runs"]}
    if set(runs) != {"controller/prior", "controller/mppi"}:
        raise ValueError("Expected exactly the paired prior and MPPI controllers.")
    counts = {}
    for selector, run in runs.items():
        if run["status"] != "complete" or not run["result"]["outer_state_unchanged"]:
            raise ValueError("Evaluation failed or changed frozen outer state.")
        episodes = run["episodes"]
        if [episode["seed"] for episode in episodes] != list(seeds):
            raise ValueError("Episode seeds differ from the requested protocol.")
        if any(episode["length"] != max_steps for episode in episodes):
            raise ValueError("Humanoid evaluation did not finish the requested episode lengths.")
        expected_steps = 12_336 if selector == "controller/mppi" else 0
        decisions = 0
        for relative in run["trace_files"]:
            with gzip.open(Path(bundle_path) / relative, "rt") as stream:
                for line in stream:
                    event = json.loads(line)
                    if event["phase"] != "decision":
                        raise ValueError("Only per-decision diagnostics are expected.")
                    if any(event[f"{component}_updates"] != 0
                           for component in ("critic", "actor", "temperature")):
                        raise ValueError("MPPI/prior evaluation performed an optimizer update.")
                    metrics = event["metrics"]
                    if metrics["decision/inner_model_steps"] != expected_steps:
                        raise ValueError("Planner model-step count differs from Humanoid defaults.")
                    if selector == "controller/mppi" and metrics["decision/inner_mppi_iterations"] != 8:
                        raise ValueError("Expected eight effective Humanoid MPPI iterations.")
                    decisions += 1
        if decisions != len(seeds) * max_steps:
            raise ValueError("Per-decision diagnostics are missing or duplicated.")
        counts[selector] = decisions
    prior_returns = {episode["seed"]: episode["return"]
                     for episode in runs["controller/prior"]["episodes"]}
    for episode in runs["controller/mppi"]["episodes"]:
        if abs(episode["paired_return_delta"] - (episode["return"] - prior_returns[episode["seed"]])) > 1e-9:
            raise ValueError("Paired return delta does not match the reference episode.")
    return {"outer_state_unchanged": True, "decision_counts": counts,
            "optimizer_updates": 0, "mppi_model_steps_per_decision": 12_336}


def run(manifest_path, index, result_root, *, mode="production", device="cuda", wandb=False):
    if mode not in {"smoke", "production"}:
        raise ValueError("Mode must be smoke or production.")
    if wandb and mode == "smoke":
        raise ValueError("Smoke evaluation must not publish to W&B.")
    row = select_checkpoint(manifest_path, index)
    destination = Path(result_root).resolve() / f"step_{row['step']}"
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("Evaluation results must be outside the source checkout.")
    destination.mkdir(parents=True, exist_ok=False)
    from evaluate_ambi_checkpoint import evaluate_matrix
    from report_ambi_benchmark import load_bundles, write_report
    from utils.ambi_benchmark import atomic_json

    seeds, max_steps = (list(range(101, 106)), 500) if mode == "production" else ([101, 102], 3)
    atomic_json(destination / "provenance.json", {
        "source_run": SOURCE_RUN, "checkpoint": row,
        "checkpoint_manifest_sha256": file_sha256(manifest_path),
        "matrix_sha256": file_sha256(MATRIX), "mode": mode,
        "seeds": seeds, "max_steps": max_steps, "controller_seed": 12345,
    })
    payload = evaluate_matrix(
        MATRIX, row["path"], selectors=["controller/prior", "controller/mppi"],
        seeds=seeds, controller_seed=12345, max_steps=max_steps, device=device,
        bundle_dir=destination / "bundle",
        wandb_options={"project": "ambi-inner-bench", "entity": "rwgao_b-brown-university",
                       "mode": "online"} if wandb else None,
    )
    # Preserve completed episode results even if an acceptance check/report fails.
    atomic_json(destination / "paired.json", payload)
    if payload["checkpoint_sha256"] != row["sha256"]:
        raise ValueError("Evaluated checkpoint hash differs from the preflight manifest.")
    validation = validate_bundle(destination / "bundle", seeds=seeds, max_steps=max_steps)
    atomic_json(destination / "validation.json", {"step": row["step"], "mode": mode, **validation})
    write_report(load_bundles([destination / "bundle"]), destination / "comparison.html",
                 title=f"AMBI-XQC prior versus MPPI at {row['step']:,} decisions")
    print(json.dumps({"step": row["step"], "output": str(destination), **validation}, sort_keys=True))
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "production"), default="production")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args(argv)
    run(args.manifest, args.index, args.result_root, mode=args.mode, device=args.device, wandb=args.wandb)


if __name__ == "__main__":
    main()
