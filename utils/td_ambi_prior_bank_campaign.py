"""Hash the four TD-AMBI prior banks and validate their short evaluation smoke.

Run hashing on a scheduled CPU worker for each bank::

    python -m utils.td_ambi_prior_bank_campaign --bank-inventory inventory.json \
        --bank-index 0 --output-dir inventories
    python -m utils.td_ambi_prior_bank_campaign smoke-check \
        --results smoke.json --bundle smoke-bundle

Only checkpoint bytes and metadata are read; no torch import or model load is
needed. Output inventories are published atomically and never overwritten.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import urlparse


FAMILIES = ("reward_qscale", "entropy_qscale", "entropy_autotemp", "reward_autotemp")
STEPS = tuple(range(25_000, 2_000_001, 25_000))
SELECTORS = ("controller/prior", "controller/mppi", "inner/fixed", "inner/adaptive")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
             f"File changed while hashing: {path}")
    _require(after.st_size > 0, f"Empty checkpoint or metadata: {path}")
    return digest.hexdigest()


def _create_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link creates the destination atomically and fails if it exists.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def prepare_inventory(bank_inventory, bank_index, output_dir, *, steps=None):
    inventory = _read(bank_inventory)
    _require(type(bank_index) is int and 0 <= bank_index < 4, "bank-index must be 0 through 3")
    banks = inventory.get("banks", [])
    _require(len(banks) == 4, "Expected exactly four prior banks")
    bank = banks[bank_index]
    family = FAMILIES[bank_index]
    config = f"td_ambi_prior_{family}"
    _require(bank.get("config") == config, f"Unexpected bank at index {bank_index}")
    source_commit = inventory.get("source_commit")
    _require(re.fullmatch(r"[0-9a-f]{40}", str(source_commit)), "Invalid source_commit provenance")
    parsed = urlparse(bank.get("wandb_run_url", ""))
    parts = parsed.path.strip("/").split("/")
    _require(parsed.scheme == "https" and parsed.netloc == "wandb.ai"
             and len(parts) == 4 and parts[2] == "runs" and all(parts),
             "Expected an authoritative https://wandb.ai/entity/project/runs/id URL")
    source_run = "/".join((parts[0], parts[1], parts[3]))
    selected = STEPS if steps is None else tuple(sorted(steps))
    _require(selected and len(set(selected)) == len(selected)
             and all(type(step) is int and step in STEPS for step in selected),
             "steps must be unique 25k checkpoints between 25k and 2M")
    output = Path(output_dir).resolve() / f"{config}.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite inventory: {output}")
    directory = Path(bank["models_directory"]).resolve()
    prefix = bank["filename_prefix"]
    _require(prefix == f"model:{config}_0_", "Unexpected checkpoint filename prefix")
    expected_target = "reward_only" if family.startswith("reward") else "entropy_augmented"
    expected_scale = "tdmpc2_percentile_range" if family.endswith("qscale") else "none"
    rows = []
    for step in selected:
        checkpoint = directory / f"{prefix}{step}"
        metadata_path = Path(str(checkpoint) + ".metadata.json")
        metadata = _read(metadata_path)
        trial = metadata.get("trial_run_params", {})
        params = trial.get("alg_params", {})
        expected = {
            "schema_version": (metadata.get("schema_version"), 1),
            "name": (trial.get("name"), config),
            "alg": (trial.get("alg"), "AMBITDMPC2/AMBITDMPC2"),
            "seed": (trial.get("seed"), 55),
            "step": (metadata.get("checkpoint", {}).get("step"), step),
            "kind": (metadata.get("checkpoint", {}).get("kind"), "periodic"),
            "outer_critic_target": (params.get("outer_critic_target"), expected_target),
            "sac_actor_loss_scale_mode": (params.get("sac_actor_loss_scale_mode"), expected_scale),
            "outer_actor_entropy_mode": (params.get("outer_actor_entropy_mode"), "tdmpc2_scaled"),
            "inner_operator": (params.get("inner_operator"), "none"),
        }
        for key, (actual, wanted) in expected.items():
            _require(actual == wanted, f"{metadata_path}: {key}={actual!r}, expected {wanted!r}")
        rows.append({"step": step, "path": str(checkpoint), "source_run": source_run,
                     "sha256": _sha256(checkpoint),
                     "metadata_sha256": _sha256(metadata_path)})
    payload = {"schema_version": 1, "config": config, "source_run": source_run,
               "source_commit": source_commit, "seed": 55,
               "checkpoints": rows}
    _create_json(output, payload)
    return output


def validate_smoke(results_path, bundle_dir):
    evaluation = _read(results_path)
    bundle = Path(bundle_dir).resolve()
    manifest = _read(bundle / "manifest.json")
    results = evaluation.get("results", [])
    runs = manifest.get("runs", [])
    _require(manifest.get("status") == "complete", "Smoke bundle is incomplete")
    for rows, name in ((results, "results"), (runs, "bundle")):
        _require(len(rows) == 4 and {row.get("selector") for row in rows} == set(SELECTORS),
                 f"Smoke {name} must contain exactly all four selectors")
    by_selector = {row["selector"]: row for row in results}
    paired_seeds = None
    for run in runs:
        selector = run["selector"]
        result = by_selector[selector]
        _require(run.get("status") == "complete", f"{selector}: incomplete run")
        _require(result.get("outer_state_unchanged") is True
                 and result.get("outer_updates_before") == result.get("outer_updates_after"),
                 f"{selector}: frozen outer state changed")
        _require(not result.get("nonfinite_model_metrics")
                 and not result.get("nonfinite_trace_metrics"), f"{selector}: nonfinite metrics")
        episodes = result.get("episodes", [])
        _require(len(episodes) == 2 and all(row.get("length") == 3 for row in episodes),
                 f"{selector}: expected two episodes with three decisions each")
        seeds = [row["seed"] for row in episodes]
        _require(len(set(seeds)) == 2, f"{selector}: duplicate episode seeds")
        paired_seeds = seeds if paired_seeds is None else paired_seeds
        _require(seeds == paired_seeds, f"{selector}: unpaired episode seeds")
        _require(all(math.isfinite(row["return"]) for row in episodes),
                 f"{selector}: nonfinite returns")
        cfg = result.get("resolved_config", {})
        sac = selector.startswith("inner/")
        adaptive_alpha = selector == "inner/adaptive" and cfg.get("sac_actor_loss_scale_mode") == "none"
        expected = {
            "inner_model_steps": 7680 if sac else 12336 if selector == "controller/mppi" else 0,
            "inner_critic_optimizer_steps": 15 if sac else 0,
            "inner_actor_optimizer_steps": 15 if sac else 0,
            "inner_temperature_optimizer_steps": 15 if adaptive_alpha else 0,
        }
        if sac:
            _require(cfg.get("inner_temperature_mode") == ("auto" if adaptive_alpha else "inherit_outer"),
                     f"{selector}: wrong temperature mode")
        if adaptive_alpha:
            _require(cfg.get("inner_target_entropy") == -441, f"{selector}: wrong entropy target")
        decisions = []
        for relative in run.get("trace_files", []):
            path = (bundle / relative).resolve()
            _require(path.is_relative_to(bundle), "Trace file escapes bundle")
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                decisions.extend(row for row in map(json.loads, stream) if row.get("phase") == "decision")
        _require(len(decisions) == 6, f"{selector}: expected six decision traces")
        identities = {(row.get("episode_id"), row.get("decision_index")) for row in decisions}
        _require(identities == {(f"seed-{seed}", index) for seed in seeds for index in range(3)},
                 f"{selector}: duplicate or missing decision traces")
        for key, value in expected.items():
            summary = result.get("model_metrics", {}).get(key, {})
            _require(summary.get("count") == 6 and summary.get("min") == value
                     and summary.get("max") == value, f"{selector}: wrong {key} aggregate")
            _require(all(row.get("metrics", {}).get(f"decision/{key}") == value for row in decisions),
                     f"{selector}: wrong {key} trace")
    return {"status": "passed", "selectors": list(SELECTORS), "episodes_per_selector": 2,
            "decisions_per_episode": 3, "outer_state_unchanged": True}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "smoke-check":
        parser = argparse.ArgumentParser(description="Validate the four-selector GPU smoke")
        parser.add_argument("--results", type=Path, required=True)
        parser.add_argument("--bundle", type=Path, required=True)
        args = parser.parse_args(argv[1:])
        print(json.dumps(validate_smoke(args.results, args.bundle), sort_keys=True))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--bank-inventory", type=Path, required=True)
        parser.add_argument("--bank-index", type=int, choices=range(4), required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--steps", help="Optional comma-separated subset, e.g. 25000,2000000")
        args = parser.parse_args(argv)
        steps = None if args.steps is None else [int(value) for value in args.steps.split(",")]
        print(prepare_inventory(args.bank_inventory, args.bank_index, args.output_dir, steps=steps))


if __name__ == "__main__":
    main()
