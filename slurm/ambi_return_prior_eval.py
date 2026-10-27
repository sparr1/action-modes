"""Run the three selected frozen prior/SAC comparisons; never train or publish."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/return_prior_selected_eval.json"
SELECTORS = ("comparison/prior", "comparison/sac")
SEEDS = [101, 102, 103, 104, 105]
SOURCES = (
    ("e83a27546bcd4ffb92650dade99e619f", 8, 1),
    ("333ac0c3631e46699146eb291c5377c0", 8, 3),
    ("de37a4eec8094c67a2a691db04e47a2b", 10, 3),
)


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def runtime_info(*, gpu):
    from importlib.metadata import version

    info = {"hostname": socket.gethostname(), "python": platform.python_version(),
            "pytorch": version("torch"), "gpu_model": None}
    if gpu:
        import torch

        info["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_model"] = torch.cuda.get_device_name(0)
    return info


def verify_entry(inventory, index):
    require(inventory.get("schema_version") == 1, "Unsupported inventory schema")
    entries = inventory.get("entries", [])
    require(len(entries) == 3 and index in range(3), "Expected exactly three ordered entries")
    for row, (run_id, rounds, interval) in zip(entries, SOURCES):
        require((row.get("source_run_id"), row.get("J"), row.get("K")) ==
                (run_id, rounds, interval), "Source run/order/J/K mismatch")
        require(row.get("source_run") == f"rwgao_b-brown-university/ambi/{run_id}",
                "Source run path mismatch")
        require(isinstance(row.get("source_run_name"), str) and row["source_run_name"],
                "Missing source run name")
    entry = entries[index]
    for field, hash_field in (("checkpoint", "sha256"), ("metadata", "metadata_sha256")):
        path = Path(entry[field])
        require(path.is_absolute() and path.is_file(), f"Missing absolute {field} path")
        require(file_hash(path) == entry[hash_field], f"{field} SHA256 mismatch")
    metadata = read_json(entry["metadata"])
    step = metadata["checkpoint"]["step"]
    require(type(step) is int and 0 < step <= 500000 and step == entry["step"],
            "Checkpoint step must match the selected available step at or below 500k")
    require(step == min(500000, entry.get("latest_at_selection", step)),
            "Checkpoint does not follow the frozen selection rule")
    run = metadata["trial_run_params"]
    params = run["alg_params"]
    require(run.get("seed") == 55 and run.get("env") == "DMControl-v0" and
            run.get("alg") == "AMBITDMPC2/AMBITDMPC2", "Source algorithm/environment/seed mismatch")
    env = metadata["experiment_params"]["env_params"]
    require(env.get("task") == "humanoid-walk", "Expected Humanoid Walk")
    expected = {
        "obs": "state", "mpc": False, "inner_operator": "sac",
        "inner_rollout_horizon": 3, "inner_rounds": entry["J"],
        "inner_solve_interval": entry["K"], "inner_rollouts_per_round": 128,
        "inner_critic_updates_per_round": 16, "inner_actor_updates_per_round": 4,
        "inner_batch_size": 256, "outer_critic_target": "reward_only",
        "inner_sac_critic_target": "reward_only", "aux_return_mode": "off",
        "inner_temperature_mode": "auto", "inner_temperature_initialization": "inherit_outer",
        "inner_target_entropy": "inherit_outer", "target_entropy": -10.5,
        "inner_entropy_enabled": True, "inner_critic_bernoulli_p": 0.0,
        "inner_actor_writeback_coef": 0.0, "inner_critic_writeback_coef": 0.0,
        "inner_outer_replay_fraction": 0.0, "inner_diagnostic_rollouts": 0,
    }
    for key, value in expected.items():
        actual = params.get(key, 1) if key == "inner_solve_interval" else params.get(key)
        require(actual == value, f"Saved {key} mismatch: {actual!r}")
    require(params.get("wandb_run_name") == entry["source_run_name"], "Saved source run name mismatch")
    return entry


def validate_results(payload, entry, *, smoke=False, smoke_steps=6):
    seeds, steps = ([101], smoke_steps) if smoke else (SEEDS, 500)
    require(payload.get("frozen_outer_learning") is True and
            payload.get("deterministic_execution") is True, "Expected frozen mean-action evaluation")
    require(payload.get("checkpoint_sha256") == entry["sha256"], "Result checkpoint mismatch")
    results = payload.get("results", [])
    require(len(results) == 2 and {r["selector"] for r in results} == set(SELECTORS),
            "Expected both prior and SAC results")
    for result in results:
        require(result.get("outer_state_unchanged") is True and
                result["outer_updates_before"] == result["outer_updates_after"], "Outer state changed")
        require(result.get("controller_seed") == 55 and result.get("environment_seeds") == seeds,
                "Evaluation seed mismatch")
        episodes = result.get("episodes", [])
        require([ep["seed"] for ep in episodes] == seeds and all(ep["length"] == steps for ep in episodes),
                "Incomplete full-episode coverage")
        if not smoke:
            require(all(ep.get("truncated") is True and ep.get("terminated") is False and
                        ep.get("truncated_by_evaluator") is False for ep in episodes),
                    "Production episodes must end at the native environment time limit")
        require(all(math.isfinite(ep["return"]) for ep in episodes), "Nonfinite episode return")
        require(not result.get("nonfinite_model_metrics") and not result.get("nonfinite_trace_metrics"),
                "Nonfinite evaluation metrics")
        require(not result.get("togo_return_probe"), "This comparison does not enable to-go probes")
        cfg = result["resolved_config"]
        if result["selector"] == SELECTORS[1]:
            require((cfg["inner_operator"], cfg["inner_rounds"], cfg["inner_solve_interval"],
                     cfg["inner_rollout_horizon"]) == ("sac", entry["J"], entry["K"], 3),
                    "SAC controller changed")
            require(cfg["inner_temperature_mode"] == "auto" and
                    cfg["inner_temperature_initialization"] == "inherit_outer", "Inner alpha changed")
            if entry["K"] > 1:
                require(cfg.get("inner_hold_encoder") == "solve_snapshot", "Held encoder changed")
        else:
            require(cfg["inner_operator"] == "none" and cfg["inner_rounds"] == 0,
                    "Prior performed inner work")
            for key in ("inner_model_steps", "inner_actor_optimizer_steps",
                        "inner_critic_optimizer_steps", "inner_temperature_optimizer_steps"):
                metric = result.get("model_metrics", {}).get(key, {})
                require(metric.get("count") == len(seeds) * steps and
                        all(metric.get(field) == 0 for field in ("sum", "min", "max")),
                        f"Prior has missing or nonzero work counter: {key}")
    return {"episodes": 2 * len(seeds), "decisions": 2 * len(seeds) * steps}


def run(args, *, receipt_fields=None, validator=validate_results):
    inventory = read_json(args.inventory)
    entry = verify_entry(inventory, args.index)
    if args.expected_commit:
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        require(actual == args.expected_commit, "Source commit mismatch")
    require(1 <= args.smoke_steps <= 6, "Smoke must use at most six decisions")
    run_map = None
    if args.mode == "run":
        require(args.eval_run_map is not None, "Production requires prepared publication identities")
        run_map = read_json(args.eval_run_map)
        require(set(run_map) == set(SELECTORS) and len(set(run_map.values())) == 2,
                "Publication map must identify both distinct controllers")
        require(all((Path(value) / "run.json").is_file() for value in run_map.values()),
                "Missing publication registry")
    else:
        require(args.eval_run_map is None, "Prepare and smoke cannot publish")
    output = args.output.resolve()
    require(not output.is_relative_to(ROOT), "Campaign outputs must live outside the source checkout")
    output.mkdir(parents=True, exist_ok=False)
    native_inventory = output / "checkpoint-inventory.json"
    write_json(native_inventory, {"source_run": entry["source_run"], "checkpoints": [{
        "step": entry["step"], "sha256": entry["sha256"],
        "metadata_sha256": entry["metadata_sha256"],
    }]})
    command = [sys.executable, str(ROOT / "evaluate_ambi_checkpoint.py"),
               "--matrix", str(MATRIX), "--checkpoint", entry["checkpoint"],
               "--metadata", entry["metadata"], "--checkpoint-inventory", str(native_inventory),
               "--controller-seed", "55", "--device", "cuda"]
    started = time.perf_counter()
    if run_map is not None:
        # Resolve actual checkpoint semantics against the prepared identities
        # before constructing any environment or running a GPU solve.
        preflight = command + ["--eval-series-spec-dir", str(output / "preflight-specs"),
                               "--output", str(output / "preflight.json")]
        subprocess.run(preflight, cwd=ROOT, check=True)
        for selector, registry_dir in run_map.items():
            spec = read_json(output / "preflight-specs" / (selector.replace("/", "__") + ".json"))
            require(spec["identity"] == read_json(Path(registry_dir) / "run.json")["identity"],
                    f"Prepared evaluation identity changed for {selector}")
    if args.mode == "prepare":
        command += ["--eval-series-spec-dir", str(output / "specs"),
                    "--output", str(output / "prepared.json")]
    else:
        command += ["--bundle-dir", str(output / "bundle"), "--output", str(output / "results.json")]
        if args.mode == "smoke":
            command += ["--seeds", "101", "--max-steps", str(args.smoke_steps)]
    # GPU workers save and validate paired results. The CPU publisher alone
    # normalizes, stages and publishes them after checking this receipt.
    write_json(output / "command.json", command)
    subprocess.run(command, cwd=ROOT, check=True)
    receipt = {"validated": True, "mode": args.mode, "source_run": entry["source_run"],
               "step": entry["step"], "J": entry["J"], "K": entry["K"],
               "checkpoint_sha256": entry["sha256"], "metadata_sha256": entry["metadata_sha256"],
               "elapsed_seconds": time.perf_counter() - started,
               "matrix_sha256": file_hash(MATRIX), "runtime": runtime_info(gpu=args.mode != "prepare")}
    receipt.update(receipt_fields or {})
    if args.mode != "prepare":
        results, manifest = output / "results.json", output / "bundle/manifest.json"
        receipt.update(validator(read_json(results), entry, smoke=args.mode == "smoke",
                                 smoke_steps=args.smoke_steps))
        require(read_json(manifest).get("status") == "complete", "Incomplete benchmark bundle")
        receipt.update(results=str(results), bundle=str(output / "bundle"),
                       results_sha256=file_hash(results), manifest_sha256=file_hash(manifest))
    if run_map is not None:
        receipt.update(eval_run_map=str(args.eval_run_map.resolve()),
                       eval_run_map_sha256=file_hash(args.eval_run_map))
    write_json(output / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "smoke"))
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-run-map", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--smoke-steps", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
