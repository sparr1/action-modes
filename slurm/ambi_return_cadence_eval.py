"""Evaluate one cadence swap on each selected J8 checkpoint; reuse its prior."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm import ambi_return_prior_eval as original
from slurm.ambi_return_prior_eval import (
    ROOT, SEEDS, file_hash, read_json, require, runtime_info, verify_entry, write_json,
)
from utils.ambi_benchmark import protocol_for, reference_returns, solver_seed
from utils.ambi_research import load_preset_matrix, resolve_preset
from utils.checkpoint_context import load_checkpoint_context

MATRIX = ROOT / "configs/research/return_prior_cadence_swap.json"
CAMPAIGN_ID = "return-prior-cadence-swap-20261026"


def selection(index):
    require(index in (0, 1), "Only the two selected J8 checkpoints may be swapped")
    interval = (3, 1)[index]
    return f"cadence/k{interval}", interval


def verify_reference(root, index, entry):
    cell = Path(root) / "production" / f"cell-{index}"
    receipt = read_json(cell / "receipt.json")
    require(receipt.get("validated") is True and receipt.get("mode") == "run",
            "Reference requires validated production results")
    for name, field in (("results.json", "results_sha256"),
                        ("bundle/manifest.json", "manifest_sha256")):
        require(file_hash(cell / name) == receipt[field], "Reference file hash changed")
    original.validate_results(read_json(cell / "results.json"), entry)
    selector, _ = selection(index)
    resolved = resolve_preset(MATRIX, selector, load_preset_matrix(MATRIX),
                             checkpoint_context=load_checkpoint_context(entry["checkpoint"], entry["metadata"]))
    values = reference_returns(cell / "bundle", entry["sha256"], protocol_for(resolved, 55, 500))
    require(set(values) == set(SEEDS), "Reference seed coverage changed")
    return {"bundle": str((cell / "bundle").resolve()),
            "manifest_sha256": receipt["manifest_sha256"],
            "results_sha256": receipt["results_sha256"],
            "receipt_sha256": file_hash(cell / "receipt.json")}


def validate_results(payload, entry, index, *, smoke=False, smoke_steps=6):
    selector, interval = selection(index)
    seeds, steps = ([101], smoke_steps) if smoke else (SEEDS, 500)
    require(payload.get("frozen_outer_learning") is True and
            payload.get("deterministic_execution") is True, "Expected frozen mean-action evaluation")
    require(payload.get("checkpoint_sha256") == entry["sha256"], "Result checkpoint mismatch")
    results = payload.get("results", [])
    require(len(results) == 1 and results[0]["selector"] == selector,
            "Expected only the requested cadence swap; do not rerun the prior")
    result = results[0]
    require(result.get("outer_state_unchanged") is True and
            result["outer_updates_before"] == result["outer_updates_after"], "Outer state changed")
    require(result.get("controller_seed") == 55 and result.get("environment_seeds") == seeds,
            "Evaluation seed mismatch")
    episodes = result.get("episodes", [])
    require([ep["seed"] for ep in episodes] == seeds and all(ep["length"] == steps for ep in episodes),
            "Incomplete full-episode coverage")
    require(all(ep.get("solver_seed") == solver_seed(55, "episode", ep["seed"]) for ep in episodes),
            "Episode solver seed changed")
    if not smoke:
        require(all(ep.get("truncated") is True and ep.get("terminated") is False and
                    ep.get("truncated_by_evaluator") is False for ep in episodes),
                "Production episodes must end at the native environment time limit")
    require(all(math.isfinite(ep["return"]) for ep in episodes) and
            not result.get("nonfinite_model_metrics") and not result.get("nonfinite_trace_metrics"),
            "Nonfinite evaluation metrics")
    require(not result.get("togo_return_probe"), "No to-go probes were requested")
    saved = dict(read_json(entry["metadata"])["trial_run_params"]["alg_params"])
    saved["inner_solve_interval"] = interval
    actual = dict(result["alg_params"])
    for runtime_key in ("device", "wandb"):
        saved.pop(runtime_key, None)
        actual.pop(runtime_key, None)
    require(actual == saved, "Controller settings changed beyond the requested cadence")
    cfg = result["resolved_config"]
    require((cfg["inner_operator"], cfg["inner_rounds"], cfg["inner_solve_interval"],
             cfg["inner_rollout_horizon"]) == ("sac", 8, interval, 3), "Resolved controller changed")
    require(cfg["inner_temperature_mode"] == "auto" and
            cfg["inner_temperature_initialization"] == "inherit_outer", "Inner alpha changed")
    if interval == 3:
        require(cfg.get("inner_hold_encoder") == "solve_snapshot", "Held encoder changed")
    per_episode_solves = math.ceil(steps / interval)
    # Check each episode so a missing reset cannot be hidden by a matching grand total.
    work = {"inner_model_steps": 3072, "inner_critic_optimizer_steps": 128,
            "inner_actor_optimizer_steps": 32, "inner_temperature_optimizer_steps": 32}
    if interval == 3:
        work["inner_solve_performed"] = 1
    # Episode metrics store flat means; result metrics store count/sum statistics.
    for episode in episodes:
        for key, dose in work.items():
            mean = episode["model_metrics"].get(key)
            require(isinstance(mean, (int, float)) and
                    math.isclose(mean, dose * per_episode_solves / steps, rel_tol=1e-9, abs_tol=1e-9),
                    f"Incorrect solve cadence/work counter: {key}")
    for key, dose in work.items():
        metric = result["model_metrics"].get(key, {})
        require(metric.get("count") == len(seeds) * steps and
                metric.get("sum") == dose * per_episode_solves * len(seeds),
                f"Incorrect solve cadence/work counter: {key}")
    return {"episodes": len(seeds), "decisions": len(seeds) * steps,
            "solves": len(seeds) * per_episode_solves}


def run(args):
    selector, interval = selection(args.index)
    entry = verify_entry(read_json(args.inventory), args.index)
    require(entry["step"] == 500000 and entry["J"] == 8, "Expected the selected J8 checkpoints at 500k")
    require(1 <= args.smoke_steps <= 6, "Smoke must use at most six decisions")
    if args.expected_commit:
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        require(actual == args.expected_commit, "Source commit mismatch")
    reference = None
    if args.mode != "smoke":
        require(args.reference_root is not None, "Prepare and production require the completed reference root")
        reference = verify_reference(args.reference_root, args.index, entry)
    run_map = None
    if args.mode == "run":
        require(args.eval_run_map is not None, "Production requires a prepared publication identity")
        run_map = read_json(args.eval_run_map)
        require(set(run_map) == {selector}, "Publication map must contain only the requested swap")
        require((Path(run_map[selector]) / "run.json").is_file(), "Missing publication registry")
    else:
        require(args.eval_run_map is None, "Prepare and smoke cannot publish")
    output = args.output.resolve()
    require(not output.is_relative_to(ROOT), "Campaign outputs must live outside the source checkout")
    output.mkdir(parents=True, exist_ok=False)
    native_inventory = output / "checkpoint-inventory.json"
    write_json(native_inventory, {"source_run": entry["source_run"], "checkpoints": [{
        "step": entry["step"], "sha256": entry["sha256"], "metadata_sha256": entry["metadata_sha256"]}]})
    command = [sys.executable, str(ROOT / "evaluate_ambi_checkpoint.py"),
               "--matrix", str(MATRIX), "--preset", selector, "--checkpoint", entry["checkpoint"],
               "--metadata", entry["metadata"], "--checkpoint-inventory", str(native_inventory),
               "--controller-seed", "55", "--device", "cuda"]
    started = time.perf_counter()
    if run_map is not None:
        subprocess.run(command + ["--eval-series-spec-dir", str(output / "preflight-specs"),
                                  "--output", str(output / "preflight.json")], cwd=ROOT, check=True)
        spec = read_json(output / "preflight-specs" / (selector.replace("/", "__") + ".json"))
        require(spec["identity"] == read_json(Path(run_map[selector]) / "run.json")["identity"],
                "Prepared evaluation identity changed")
    if args.mode == "prepare":
        command += ["--eval-series-spec-dir", str(output / "specs"), "--output", str(output / "prepared.json")]
    else:
        command += ["--bundle-dir", str(output / "bundle"), "--output", str(output / "results.json")]
        if args.mode == "smoke":
            command += ["--seeds", "101", "--max-steps", str(args.smoke_steps)]
        else:
            command += ["--reference-bundle", reference["bundle"]]
    # Only the CPU publisher normalizes and stages completed results.
    write_json(output / "command.json", command)
    subprocess.run(command, cwd=ROOT, check=True)
    receipt = {"validated": True, "campaign_id": CAMPAIGN_ID, "mode": args.mode, "source_run": entry["source_run"],
               "step": entry["step"], "J": 8, "source_K": entry["K"], "eval_K": interval,
               "selector": selector, "checkpoint_sha256": entry["sha256"],
               "metadata_sha256": entry["metadata_sha256"], "matrix_sha256": file_hash(MATRIX),
               "elapsed_seconds": time.perf_counter() - started,
               "runtime": runtime_info(gpu=args.mode != "prepare")}
    if reference is not None:
        require(reference == verify_reference(args.reference_root, args.index, entry), "Reference changed during evaluation")
        receipt["reference"] = reference
    if args.mode != "prepare":
        results, manifest_path = output / "results.json", output / "bundle/manifest.json"
        receipt.update(validate_results(read_json(results), entry, args.index,
                                        smoke=args.mode == "smoke", smoke_steps=args.smoke_steps))
        manifest = read_json(manifest_path)
        require(manifest.get("status") == "complete" and len(manifest.get("runs", [])) == 1,
                "Expected one completed benchmark arm")
        if reference is not None:
            require(manifest.get("reference", {}).get("manifest_sha256") == reference["manifest_sha256"],
                    "Bundle used a different prior reference")
        receipt.update(results=str(results), bundle=str(output / "bundle"),
                       results_sha256=file_hash(results), manifest_sha256=file_hash(manifest_path))
    if run_map is not None:
        receipt.update(eval_run_map=str(args.eval_run_map.resolve()), eval_run_map_sha256=file_hash(args.eval_run_map))
    write_json(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "smoke"))
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--eval-run-map", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--smoke-steps", type=int, default=6)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
