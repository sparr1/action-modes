"""Validate and stage saved prior/MPPI evaluations for one W&B run per curve.

No evaluation is performed here. A campaign consists of step_N/paired.json and
step_N/provenance.json files. The latter pins the source run, checkpoint hash,
and evaluation code SHA/tree before evaluation starts.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile


STEPS = tuple(range(100_000, 1_500_001, 50_000))
EVALUATION_SOURCE_PATHS = (
    "evaluate_tdmpc2_mppi_checkpoint.py", "render_checkpoint.py", "RL", "domains", "utils",
    "environments/dmcontrol/pyproject.toml", "environments/dmcontrol/uv.lock",
)
SOURCE_NAME = "TDMPC2-humanoid-walk-prior-only-no-MPPI-checkpoint-bank-1p5m-seed55"
PLANNER = {
    "configured_iterations": 6, "effective_iterations": 8,
    "num_samples": 512, "num_elites": 64, "num_pi_trajs": 24,
    "planning_horizon": 3, "model_transitions_per_action": 12336,
}
METRICS = ("policy_prior_return_mean", "native_mppi_return_mean", "paired_return_delta_mean")


def _read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value), "Missing or nonfinite numeric measurement")
    return float(value)


@lru_cache(maxsize=128)
def evaluation_source_fingerprint(code_sha, code_tree, repository=None):
    """Verify pinned Git identity and hash all evaluator/runtime source inputs.

    Launcher, publisher, tests and documentation outside these paths do not
    affect the controller experiment. Git is required only to verify new
    provenance or combine results from different commits, including legacy
    bundles that predate the fingerprint field.
    """
    repository = repository or Path(__file__).resolve().parent
    for value in (code_sha, code_tree):
        _require(bool(re.fullmatch(r"[0-9a-f]{40}", value)), "Invalid evaluation Git identity")
    try:
        actual_tree = subprocess.check_output(
            ["git", "rev-parse", f"{code_sha}^{{tree}}"], cwd=repository, stderr=subprocess.PIPE,
        ).decode().strip()
        entries = subprocess.check_output(
            ["git", "ls-tree", "-rz", code_sha, "--", *EVALUATION_SOURCE_PATHS],
            cwd=repository, stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Pinned evaluation source is unavailable in Git; cannot verify mixed-code results") from error
    _require(actual_tree == code_tree, "Evaluation code SHA and tree do not match")
    names = [entry.split(b"\t", 1)[1].decode() for entry in entries.split(b"\0") if entry]
    _require(all(any(name == path or name.startswith(path + "/") for name in names)
                 for path in EVALUATION_SOURCE_PATHS), "Pinned evaluation source is incomplete")
    return hashlib.sha256(entries).hexdigest()


def _expected_steps(expected_max_step):
    _require(expected_max_step in STEPS, "Expected maximum must be 100k–1.5M on the 50k grid")
    return tuple(step for step in STEPS if step <= expected_max_step)


def load_result(path, source_run, campaign):
    """Validate one complete result before admitting it to a comparison."""
    path = Path(path).resolve()
    match = re.fullmatch(r"step_(\d+)", path.parent.name)
    _require(match is not None, "Result must be inside a step_N directory")
    step = int(match[1])
    _require(step in STEPS, "Checkpoint is outside the 100k–1.5M campaign grid")
    result = _read(path)
    provenance = _read(path.parent / "provenance.json")
    _require(provenance.get("source_run") == source_run
             and provenance.get("campaign") == campaign, "Foreign source run or campaign")
    for key in ("code_sha", "code_tree"):
        _require(bool(re.fullmatch(r"[0-9a-f]{40}", provenance.get(key, ""))),
                 f"Missing evaluation {key}")
    checkpoint_hash = result.get("checkpoint_sha256", "")
    _require(bool(re.fullmatch(r"[0-9a-f]{64}", checkpoint_hash))
             and checkpoint_hash == provenance.get("checkpoint_sha256"),
             "Checkpoint hash differs from pinned provenance")
    _require(result.get("schema_version") == 1
             and result.get("algorithm") == "TDMPC2/TDMPC2Baseline"
             and result.get("environment") == "DMControl-v0", "Unexpected result schema or backend")
    _require(result.get("checkpoint_metadata", {}).get("step") == step,
             "Checkpoint metadata step differs from result directory")
    metadata_path = path.parent / "checkpoint.metadata.json"
    if not metadata_path.exists():
        metadata_path = Path(result["configuration_source"])
    metadata = _read(metadata_path)
    if provenance.get("metadata_sha256"):
        _require(_hash(metadata_path) == provenance["metadata_sha256"], "Metadata hash mismatch")
    trial = metadata["trial_run_params"]
    settings = trial["alg_params"]
    env = metadata["experiment_params"]["env_params"]
    _require(metadata["checkpoint"]["step"] == step
             and trial.get("seed") == 55 and settings.get("mpc") is False
             and settings.get("wandb_run_name") == SOURCE_NAME
             and env.get("task") == "humanoid-walk" and env.get("obs") == "state",
             "Sidecar does not identify the prior-only Humanoid seed55 backbone")
    for key, value in {"iterations": 6, "num_samples": 512, "num_elites": 64,
                       "num_pi_trajs": 24, "outer_planning_horizon": 3,
                       "min_std": .05, "max_std": 2, "temperature": .5}.items():
        _require(settings.get(key) == value, f"Nondefault MPPI setting: {key}")
    planner = dict(result.get("planner", {}))
    warm_start = planner.pop("warm_start", "shift_previous_mean_within_episode")
    _require(warm_start in {"none", "shift_previous_mean_within_episode"}, "Unknown MPPI warm start")
    _require(planner == PLANNER, "Nondefault resolved MPPI planner")
    if warm_start == "none":
        _require(all(step.get("planner", {}).get("planner_warm_start_used") == 0.0
                     for episode in result["episodes"] for step in episode["native_mppi"]["steps"]),
                 "Missing per-decision cold-start evidence")
    frozen = result["frozen_state"]
    _require(frozen.get("unchanged") is True
             and bool(frozen.get("model_digest_before"))
             and frozen.get("model_digest_before") == frozen.get("model_digest_after")
             and frozen.get("num_updates_before") is not None
             and frozen.get("num_updates_before") == frozen.get("num_updates_after"),
             "Evaluation did not preserve frozen state")
    protocol = result["protocol"]
    episodes = result["episodes"]
    _require(protocol.get("controllers") == ["policy_prior_mean", "native_mppi"],
             "Unexpected controller action protocol")
    seeds = list(range(protocol["environment_seed_first"], protocol["environment_seed_last"] + 1))
    _require(bool(episodes) and [item["environment_seed"] for item in episodes] == seeds,
             "Incomplete or unpaired episode seeds")
    rows = []
    for episode in episodes:
        arms = [episode[controller] for controller in protocol["controllers"]]
        for arm in arms:
            _require(arm["length"] > 0 and arm["length"] == len(arm["steps"])
                     and (arm["terminated"] or arm["truncated"] or arm["capped"]),
                     "Incomplete episode")
        prior, mppi = [_number(arm["return"]) for arm in arms]
        delta = _number(episode["return_delta"])
        _require(math.isclose(delta, mppi - prior, rel_tol=1e-9, abs_tol=1e-8),
                 "Incorrect paired episode return delta")
        rows.append([episode["environment_seed"], prior, mppi, delta,
                     *[_number(arm["seconds"]) for arm in arms]])
    _require(result["summary"]["paired_episodes"] == len(rows), "Incorrect episode count")
    for index, key in enumerate(METRICS, 1):
        _require(math.isclose(_number(result["summary"][key]),
                             statistics.fmean(row[index] for row in rows),
                             rel_tol=1e-9, abs_tol=1e-8), "Incorrect return summary")
    return {"path": path, "step": step, "result": result, "metadata": metadata,
            "metadata_path": metadata_path,
            "provenance": provenance, "rows": rows}


def _run_id(source_run, campaign, suffix):
    return "td2" + hashlib.sha256(f"{source_run}\0{campaign}\0{suffix}".encode()).hexdigest()[:20]


def campaign_data(root, source_run, campaign, expected_max_step=1_500_000):
    """Rebuild from atomic completed JSONs; never infer an absent checkpoint."""
    loaded = [load_result(path, source_run, campaign)
              for path in sorted(Path(root).glob("step_*/paired.json"))]
    _require(bool(loaded), "No completed checkpoint evaluations")
    steps = _expected_steps(expected_max_step)
    _require(all(item["step"] in steps for item in loaded), "Completed checkpoint exceeds expected maximum")
    first = loaded[0]
    signature = lambda item: (item["result"]["protocol"], item["result"]["planner"],
                              item["metadata"]["trial_run_params"],
                              item["metadata"]["experiment_params"])
    _require(all(signature(item) == signature(first) for item in loaded),
             "Cannot combine differing environment, model, code or evaluation protocols")
    identities = {(item["provenance"]["code_sha"], item["provenance"]["code_tree"]) for item in loaded}
    if len(identities) > 1:
        fingerprints = set()
        for item in loaded:
            provenance = item["provenance"]
            fingerprint = evaluation_source_fingerprint(provenance["code_sha"], provenance["code_tree"])
            _require(provenance.get("evaluation_source_sha256", fingerprint) == fingerprint,
                     "Stored evaluation source fingerprint differs from Git")
            fingerprints.add(fingerprint)
        _require(len(fingerprints) == 1, "Cannot combine differing evaluation source code")
    by_step = {item["step"]: item for item in loaded}
    _require(len(by_step) == len(loaded), "Duplicate checkpoint steps")
    curves = [[by_step[step]["result"]["summary"][key] if step in by_step else None
               for step in steps] for key in METRICS]
    return loaded, curves


def publish(path, *, source_run, campaign, project="ambi-inner-bench",
            entity="rwgao_b-brown-university", expected_max_step=1_500_000,
            eval_run_map=None, wandb_module=None):
    """Validate and stage a paired result for its two preselected curve runs.

    The historical command name is retained, but GPU workers never initialize
    W&B. Run ``eval_series.py publish RUN_DIR`` on the publication owner.
    """
    _expected_steps(expected_max_step)
    item = load_result(path, source_run, campaign)
    _require(item["step"] <= expected_max_step, "Completed checkpoint exceeds expected maximum")
    path = item["path"]
    record_path = path.parent / ".series-staging.json"
    record = {"result_sha256": _hash(path), "checkpoint_step": item["step"],
              "source_run": source_run, "campaign": campaign, "controllers": {}}
    if record_path.exists():
        previous = _read(record_path)
        _require(previous.get("result_sha256") == record["result_sha256"],
                 "Existing staging belongs to a different result")
    # Preserve exact checkpoint sidecar bytes for publication on another host.
    copied = path.parent / "checkpoint.metadata.json"
    if item["metadata_path"].resolve() != copied:
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".metadata.")
        os.close(descriptor)
        try:
            shutil.copyfile(item["metadata_path"], temporary)
            os.replace(temporary, copied)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    if os.environ.get("WANDB_MODE", "").lower() == "disabled" and eval_run_map is None:
        record["status"] = "validated_upload_disabled"
    else:
        if isinstance(eval_run_map, (str, Path)):
            eval_run_map = _read(eval_run_map)
        required_controllers = {"native_mppi"} if item["result"].get("prior_reference", {}).get("reused") else {"policy_prior", "native_mppi"}
        _require(isinstance(eval_run_map, dict) and set(eval_run_map) == required_controllers,
                 "Choose explicit New/Append runs before evaluation and supply --eval-run-map with policy_prior/native_mppi directories")
        _require(len(set(map(str, eval_run_map.values()))) == len(required_controllers),
                 "Prior and MPPI require distinct evaluation run directories")
        from utils.eval_series import load_run, stage_result
        for controller, run_dir in eval_run_map.items():
            load_run(run_dir)
            try:
                stage_result(run_dir, path, selector=controller, format="tdmpc2-paired",
                             source_run=source_run)
                record["controllers"][controller] = {"status": "queued", "run_dir": str(run_dir)}
            except Exception as error:
                record["controllers"][controller] = {"status": "failed", "run_dir": str(run_dir),
                                                     "error": f"{type(error).__name__}: {error}"}
        record["status"] = "queued" if all(row["status"] == "queued" for row in record["controllers"].values()) else "failed"
    _atomic_json(record_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--eval-run-map", type=Path, help="Explicit policy_prior/native_mppi to existing-run-directory mapping.")
    parser.add_argument("--project", default="ambi-inner-bench")
    parser.add_argument("--entity", default="rwgao_b-brown-university")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--expected-max-step", type=int, default=1_500_000,
                        help="Last expected 50k-grid checkpoint (default: 1500000).")
    args = parser.parse_args()
    record = publish(args.result, project=args.project, entity=args.entity,
                     source_run=args.source_run, campaign=args.campaign,
                     expected_max_step=args.expected_max_step, eval_run_map=args.eval_run_map)
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
