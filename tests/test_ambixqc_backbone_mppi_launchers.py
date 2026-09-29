"""The cluster launch contract and integrity of its six-checkpoint smoke gate."""
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "backbone_mppi_smoke_gate", ROOT / "tests/benchmarks/ambixqc_backbone_mppi_smoke_gate.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
SOURCE_SHA = "b" * 40
TRAINING_SHA = "c" * 40
CELLS = ("baseline_utd1", "aux_shared_utd1", "aux_detached_utd1",
         "baseline_utd2", "aux_shared_utd2", "aux_detached_utd2")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def smoke_campaign(tmp_path):
    rows = [{"cell": cell, "step": step, "sha256": "a" * 64,
             "metadata_sha256": "d" * 64}
            for cell in CELLS for step in range(25_000, 500_001, 25_000)]
    manifest = tmp_path / "inventory.json"
    write_json(manifest, {"schema": "ambixqc-backbone-mppi-inventory-v1",
                         "training_source_sha": TRAINING_SHA, "checkpoints": rows})
    smoke_root = tmp_path / "smoke"
    paths = {}
    for index in gate.SMOKE_INDICES:
        row = rows[index]
        job = smoke_root / f"job1-task{index}"
        destination = job / row["cell"] / f"step_{row['step']}"
        bundle = destination / "bundle"
        bundle.mkdir(parents=True)
        runs = []
        for selector, model_steps, rewards in (
            ("controller/prior", 0, [1.0, 2.0]),
            ("controller/mppi", 12_336, [3.0, 1.0]),
        ):
            trace = selector.split("/")[-1] + ".jsonl.gz"
            events = [{"phase": "decision", "critic_updates": 0, "actor_updates": 0,
                       "temperature_updates": 0,
                       "metrics": {"decision/inner_model_steps": model_steps,
                                   "decision/inner_mppi_iterations": 8 if model_steps else 0}}
                      for _ in range(6)]
            with gzip.open(bundle / trace, "wt") as stream:
                for event in events:
                    stream.write(json.dumps(event) + "\n")
            runs.append({"selector": selector, "status": "complete",
                         "result": {"outer_state_unchanged": True}, "trace_files": [trace],
                         "episodes": [{"seed": seed, "length": 3, "return": reward,
                                       "paired_return_delta": reward - prior}
                                      for seed, reward, prior in zip([101, 102], rewards, [1., 2.])]})
        write_json(bundle / "manifest.json", {"status": "complete", "runs": runs,
                   "code": {"commit": SOURCE_SHA, "dirty": False},
                   "checkpoint": {"sha256": row["sha256"], "metadata_sha256": row["metadata_sha256"]}})
        validation = {"source_sha": SOURCE_SHA, "training_source_sha": TRAINING_SHA,
                      "manifest_sha256": gate.sha256(manifest), "matrix_sha256": gate.sha256(gate.MATRIX),
                      "index": index, "mode": "smoke", "cell": row["cell"], "step": row["step"],
                      "outer_state_unchanged": True, "optimizer_updates": 0,
                      "mppi_model_steps_per_decision": 12_336,
                      "decision_counts": {"controller/prior": 6, "controller/mppi": 6}}
        write_json(destination / "validation.json", validation)
        write_json(destination / "paired.json", {"test": "paired"})
        write_json(destination / "provenance.json", {"mode": "smoke"})
        (destination / "PASS").write_text("PASS\n")
        (job / "PASS").write_text("PASS\n")
        (job / "runtime.json").write_text('{"gpu":"L40S"}')
        (job / "gpu.txt").write_text("NVIDIA L40S\n")
        if index == 0:
            (job / "pytest.log").write_text("183 passed in 12.3s\n")
        paths[index] = destination / "validation.json"
    return {"root": smoke_root, "manifest": manifest, "paths": paths,
            "output": tmp_path / "smoke-gate.json"}


def create(campaign):
    return gate.create_gate(campaign["root"], campaign["manifest"], SOURCE_SHA, campaign["output"])


def test_gate_certifies_only_six_actual_smokes_and_round_trips(smoke_campaign):
    payload = create(smoke_campaign)
    assert payload["checkpoint_count_validated"] == 6
    assert payload["production_checkpoint_count"] == 120
    assert payload["smoke_indices"] == [0, 39, 40, 79, 80, 119]
    assert len({receipt["cell"] for receipt in payload["receipts"]}) == 6
    assert gate.verify_gate(smoke_campaign["output"], smoke_campaign["manifest"], SOURCE_SHA) == payload
    with pytest.raises(FileExistsError):
        create(smoke_campaign)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "foreign_source", "wrong_manifest", "wrong_mode",
                                    "updated_model", "optimizer", "short", "bundle_hash", "dirty_source",
                                    "failed_job", "no_pass", "no_regression"])
def test_gate_rejects_incomplete_or_incompatible_smokes(smoke_campaign, failure):
    path = smoke_campaign["paths"][0]
    validation = json.loads(path.read_text())
    if failure == "missing":
        shutil.rmtree(path.parent.parent.parent)
    elif failure == "duplicate":
        shutil.copytree(path.parent.parent.parent, smoke_campaign["root"] / "duplicate")
    elif failure == "foreign_source":
        validation["source_sha"] = "e" * 40
    elif failure == "wrong_manifest":
        validation["manifest_sha256"] = "e" * 64
    elif failure == "wrong_mode":
        validation["mode"] = "production"
    elif failure == "updated_model":
        validation["outer_state_unchanged"] = False
    elif failure == "optimizer":
        validation["optimizer_updates"] = 1
    elif failure == "short":
        validation["decision_counts"]["controller/mppi"] = 5
    elif failure in {"bundle_hash", "dirty_source"}:
        bundle_path = path.parent / "bundle/manifest.json"
        bundle = json.loads(bundle_path.read_text())
        if failure == "bundle_hash":
            bundle["checkpoint"]["sha256"] = "e" * 64
        else:
            bundle["code"]["dirty"] = True
        write_json(bundle_path, bundle)
    elif failure == "failed_job":
        (path.parent.parent.parent / "FAILED").write_text("failure")
    elif failure == "no_pass":
        (path.parent / "PASS").unlink()
    elif failure == "no_regression":
        (path.parent.parent.parent / "pytest.log").write_text("FAILED")
    if path.exists():
        write_json(path, validation)
    with pytest.raises((ValueError, FileNotFoundError)):
        create(smoke_campaign)
    assert not smoke_campaign["output"].exists()


@pytest.mark.parametrize("mutation", ["trace", "paired", "validation", "source", "manifest", "regression"])
def test_published_gate_detects_later_artifact_or_identity_changes(smoke_campaign, mutation):
    create(smoke_campaign)
    path = smoke_campaign["paths"][0]
    source = SOURCE_SHA
    if mutation == "trace":
        with gzip.open(path.parent / "bundle/mppi.jsonl.gz", "at") as stream:
            stream.write("{}\n")
    elif mutation == "paired":
        write_json(path.parent / "paired.json", {"changed": True})
    elif mutation == "validation":
        data = json.loads(path.read_text())
        data["extra"] = True
        write_json(path, data)
    elif mutation == "source":
        source = "f" * 40
    elif mutation == "manifest":
        data = json.loads(smoke_campaign["manifest"].read_text())
        data["extra"] = True
        write_json(smoke_campaign["manifest"], data)
    else:
        (path.parent.parent.parent / "pytest.log").write_text("184 passed in 14.1s\n")
    with pytest.raises((ValueError, KeyError)):
        gate.verify_gate(smoke_campaign["output"], smoke_campaign["manifest"], source)


@pytest.mark.parametrize("cluster", ["oscar", "hydra"])
def test_launchers_preserve_protocol_and_allocation_guards(cluster):
    path = ROOT / f"slurm/run_ambixqc_backbone_mppi_eval_{cluster}.sbatch"
    text = path.read_text()
    for guard in ("#SBATCH --cpus-per-task=6", "#SBATCH --mem=32G", "#SBATCH --time=00:30:00",
                  "EXPECTED_ACTION_MODES_SHA", "--untracked-files=all", "LOCK_SHA",
                  "AMBIXQC_CHECKPOINT_MANIFEST", "AMBIXQC_SMOKE_GATE", "EVAL_RUN_MAP",
                  "run_ambixqc_backbone_mppi_evaluation.py", "--checkpoint-root", "--eval-run-map",
                  "WANDB_MODE=disabled WANDB_DISABLED=true", "CUBLAS_WORKSPACE_CONFIG=:4096:8",
                  '"$SLURM_ARRAY_TASK_ID" == 0', "0|39|40|79|80|119", "refusing to overwrite",
                  '"$RESULTS_ROOT/$MODE/job${ARRAY_ID}-task${SLURM_ARRAY_TASK_ID}"'):
        assert guard in text
    assert "WANDB_MODE=online" not in text
    assert "pip install" not in text and "uv sync" not in text
    if cluster == "hydra":
        assert "#SBATCH --nodelist=gpu2501" in text
        assert '"$(hostname -s)" == gpu2501' in text
        assert "#SBATCH --partition=gpus" in text
    else:
        assert "#SBATCH --gres=gpu:l40s:1" in text
        assert "#SBATCH --partition=gpu" in text
    subprocess.run(["/bin/bash", "-n", str(path)], check=True, close_fds=False)


def test_python_path_canonicalization_preserves_venv_and_accepts_home_alias(tmp_path):
    project = tmp_path / "actual/environments/dmcontrol"
    python = project / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to("/usr/bin/python3")
    alias = tmp_path / "home-alias"
    alias.symlink_to(tmp_path / "actual", target_is_directory=True)
    launcher = (ROOT / "slurm/run_ambixqc_backbone_mppi_eval_hydra.sbatch").read_text()
    start = launcher.index('readonly PYTHON="')
    end = launcher.index('case "$RESULTS_ROOT/"', start)
    script = "fail() { exit 2; }\n" + launcher[start:end] + '\nprintf "%s\\n" "$PYTHON"\n'
    result = subprocess.run(["/bin/bash", "-c", script], check=True, capture_output=True, text=True,
                            env={"PATH": "/usr/bin:/bin", "AMBIXQC_PYTHON": str(alias / "environments/dmcontrol/.venv/bin/python")},
                            close_fds=False)
    assert Path(result.stdout.strip()) == python
