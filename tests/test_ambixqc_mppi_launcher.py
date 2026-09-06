import gzip
import json
from pathlib import Path
import subprocess

import pytest

from run_ambixqc_mppi_evaluation import (
    SOURCE_RUN, STEPS, file_sha256, run, select_checkpoint, validate_bundle,
)


def checkpoint_manifest(tmp_path):
    rows = []
    for step in STEPS:
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.write_bytes(f"weights-{step}".encode())
        sidecar = Path(str(checkpoint) + ".metadata.json")
        sidecar.write_text(json.dumps({
            "checkpoint": {"step": step},
            "trial_run_params": {"alg": "AMBIXQC/AMBIXQC", "env": "DMControl-v0",
                                 "seed": 55, "total_steps": 1_500_000,
                                 "alg_params": {"inner_operator": "none", "obs": "state",
                                                "model_size": 5, "train_unroll_horizon": 3,
                                                "eval_freq": None}},
            "experiment_params": {"env_params": {"task": "humanoid-walk", "obs": "state"}},
        }))
        rows.append({"step": step, "path": str(checkpoint), "sha256": file_sha256(checkpoint),
                     "metadata_sha256": file_sha256(sidecar)})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"source_run": SOURCE_RUN, "checkpoints": rows}))
    return path, rows


def test_selects_both_endpoints_with_verified_weights_and_metadata(tmp_path):
    manifest, rows = checkpoint_manifest(tmp_path)
    assert select_checkpoint(manifest, 0) == rows[0]
    assert select_checkpoint(manifest, 29) == rows[-1]


@pytest.mark.parametrize("failure", ["source", "grid", "duplicate_path", "weights", "metadata"])
def test_invalid_input_fails_before_creating_results(tmp_path, failure):
    manifest, rows = checkpoint_manifest(tmp_path)
    data = json.loads(manifest.read_text())
    if failure == "source":
        data["source_run"] = "foreign/run"
    elif failure == "grid":
        data["checkpoints"].pop()
    elif failure == "duplicate_path":
        data["checkpoints"][1]["path"] = rows[0]["path"]
    elif failure == "weights":
        Path(rows[0]["path"]).write_bytes(b"changed")
    else:
        Path(rows[0]["path"] + ".metadata.json").write_text("{}")
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        run(manifest, 0, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def bundle(tmp_path):
    runs = []
    for selector, returns, model_steps in (("prior", [1., 2.], 0), ("mppi", [3., 1.], 12336)):
        events, episodes = [], []
        for seed, reward, prior in zip([101, 102], returns, [1., 2.]):
            episodes.append({"seed": seed, "return": reward, "length": 3,
                             "paired_return_delta": reward - prior})
            for decision in range(3):
                events.append({"phase": "decision", "decision_index": decision,
                               "critic_updates": 0, "actor_updates": 0, "temperature_updates": 0,
                               "metrics": {"decision/inner_model_steps": model_steps,
                                           "decision/inner_mppi_iterations": 8 if model_steps else 0}})
        trace = f"{selector}.jsonl.gz"
        with gzip.open(tmp_path / trace, "wt") as stream:
            stream.write("\n".join(json.dumps(event) for event in events) + "\n")
        runs.append({"selector": f"controller/{selector}", "status": "complete",
                     "result": {"outer_state_unchanged": True}, "episodes": episodes,
                     "trace_files": [trace]})
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "complete", "runs": runs}))
    return runs


def test_acceptance_counts_actual_decisions_and_paired_gains(tmp_path):
    bundle(tmp_path)
    result = validate_bundle(tmp_path, seeds=[101, 102], max_steps=3)
    assert result["decision_counts"] == {"controller/prior": 6, "controller/mppi": 6}
    assert result["optimizer_updates"] == 0


@pytest.mark.parametrize("field,value", [("actor_updates", 1), ("decision/inner_model_steps", 0)])
def test_acceptance_rejects_optimizer_or_search_budget_mismatch(tmp_path, field, value):
    bundle(tmp_path)
    trace = tmp_path / "mppi.jsonl.gz"
    with gzip.open(trace, "rt") as stream:
        events = [json.loads(line) for line in stream]
    target = events[0]["metrics"] if field.startswith("decision/") else events[0]
    target[field] = value
    with gzip.open(trace, "wt") as stream:
        stream.write("\n".join(json.dumps(event) for event in events) + "\n")
    with pytest.raises(ValueError):
        validate_bundle(tmp_path, seeds=[101, 102], max_steps=3)


def test_oscar_launcher_preserves_science_and_cluster_guards():
    path = Path(__file__).resolve().parents[1] / "slurm/run_ambixqc_mppi_eval_oscar.sbatch"
    content = path.read_text()
    for guard in ("#SBATCH --gres=gpu:l40s:1", "#SBATCH --cpus-per-task=6",
                  "#SBATCH --mem=32G", "#SBATCH --time=00:30:00", "#SBATCH --no-requeue",
                  "EXPECTED_ACTION_MODES_SHA", "--untracked-files=all", "LOCK_SHA",
                  "CHECKPOINT_MANIFEST", "run_ambixqc_mppi_evaluation.py", "--wandb",
                  "tests/test_xqc_mppi.py", "WANDB_CACHE_DIR"):
        assert guard in content
    assert "pip install" not in content and "uv sync" not in content
    subprocess.run(["/bin/bash", "-n", str(path)], check=True, close_fds=False)
