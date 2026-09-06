"""Publish saved prior/MPPI evaluations and refresh native W&B campaign curves.

No evaluation is performed here. A campaign consists of step_N/paired.json and
step_N/provenance.json files. The latter pins the source run, checkpoint hash,
and evaluation code SHA/tree before evaluation starts.
"""

from __future__ import annotations

import argparse
import fcntl
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
    _require(result.get("planner") == PLANNER, "Nondefault resolved MPPI planner")
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
            entity="rwgao_b-brown-university", expected_max_step=1_500_000, wandb_module=None):
    steps = _expected_steps(expected_max_step)
    item = load_result(path, source_run, campaign)
    path = item["path"]
    root = path.parent.parent
    record_path = path.parent / ".publication.json"
    record = {**item["provenance"], "result_sha256": _hash(path),
              "checkpoint_step": item["step"], "project": project, "entity": entity,
              "checkpoint_run_id": _run_id(source_run, campaign, item["step"]),
              "campaign_run_id": _run_id(source_run, campaign, "curves")}
    if record_path.exists():
        previous = _read(record_path)
        _require(all(previous.get(key) == record[key] for key in
                     ("source_run", "campaign", "checkpoint_sha256", "result_sha256")),
                 "Existing publication belongs to a different result")
    # Copy the sidecar into the result bundle so publication can be repeated off-cluster.
    copied_metadata = path.parent / "checkpoint.metadata.json"
    if item["metadata_path"].resolve() != copied_metadata:
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".metadata.")
        os.close(descriptor)
        try:
            shutil.copyfile(item["metadata_path"], temporary)
            os.replace(temporary, copied_metadata)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    disabled = os.environ.get("WANDB_MODE", "").lower() == "disabled"
    with (root / ".wandb-publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        loaded, curves = campaign_data(root, source_run, campaign, expected_max_step)
        if disabled:
            record["status"] = "validated_upload_disabled"
        else:
            if wandb_module is None:
                import wandb as wandb_module
            wandb = wandb_module
            shared = {"entity": entity, "project": project, "group": campaign,
                      "resume": "allow", "reinit": True,
                      "tags": ["tdmpc2", "humanoid-walk", "prior-only-backbone", "paper-default-mppi"]}
            config = {**item["provenance"], "checkpoint_step": item["step"],
                      "protocol": item["result"]["protocol"], "planner": PLANNER}
            with wandb.init(**shared, id=record["checkpoint_run_id"], job_type="checkpoint-evaluation",
                            name=f"TD-MPC2 prior vs MPPI | ckpt {item['step']//1000}k | H3 J8 N512 Pi24 | seed55",
                            config=config) as run:
                run.summary.update({f"eval/{key}": value for key, value in item["result"]["summary"].items()})
                run.summary.update({"eval/frozen_state_unchanged": True,
                                    "checkpoint/training_decisions": item["step"],
                                    "runtime/prior_seconds": sum(row[4] for row in item["rows"]),
                                    "runtime/mppi_seconds": sum(row[5] for row in item["rows"])})
                run.log({"eval/episode_table": wandb.Table(columns=["environment_seed", "prior_return",
                         "mppi_return", "paired_gain", "prior_seconds", "mppi_seconds"], data=item["rows"])})
                artifact = wandb.Artifact(f"tdmpc2-paired-{record['checkpoint_run_id']}", type="evaluation",
                                         metadata=config)
                for filename in ("paired.json", "checkpoint.metadata.json", "provenance.json"):
                    artifact.add_file(str(path.parent / filename), name=filename)
                run.log_artifact(artifact)
            with wandb.init(**shared, id=record["campaign_run_id"], job_type="checkpoint-comparison",
                            name=f"TD-MPC2 prior vs MPPI | checkpoint curves | {campaign}",
                            allow_val_change=True,
                            config={"source_run": source_run, "campaign": campaign,
                                    "expected_checkpoint_steps": list(steps), "planner": PLANNER,
                                    "protocol": item["result"]["protocol"]}) as run:
                run.log({"comparison/return_curves": wandb.plot.line_series(
                    xs=list(steps), ys=curves[:2], keys=["Policy prior mean", "Paper MPPI (H3, J8)"],
                    title=f"Mean episode return — {len(loaded)}/{len(steps)} checkpoints complete",
                    xname="Training checkpoint (agent decisions)"),
                    "comparison/paired_gain_curve": wandb.plot.line_series(
                    xs=list(steps), ys=curves[2:], keys=["MPPI − policy prior"],
                    title="Mean paired return gain", xname="Training checkpoint (agent decisions)")})
                run.summary.update({"completed_checkpoints": len(loaded),
                                    "completed_checkpoint_steps": sorted(entry["step"] for entry in loaded),
                                    "missing_checkpoint_steps": sorted(set(steps) - {entry["step"] for entry in loaded}),
                                    "status": "complete" if len(loaded) == len(steps) else "partial"})
            record["status"] = "published"
        record["completed_checkpoint_steps"] = sorted(entry["step"] for entry in loaded)
        _atomic_json(record_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--project", default="ambi-inner-bench")
    parser.add_argument("--entity", default="rwgao_b-brown-university")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--expected-max-step", type=int, default=1_500_000,
                        help="Last expected 50k-grid checkpoint (default: 1500000).")
    args = parser.parse_args()
    record = publish(args.result, project=args.project, entity=args.entity,
                     source_run=args.source_run, campaign=args.campaign,
                     expected_max_step=args.expected_max_step)
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
