"""Exercise the actual Oscar worker with hash checks and recorded model calls."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm/run_td_ambi_prior_bank_eval_oscar.sbatch"
SOURCE = "test-owner/ambi/bank"


@pytest.fixture
def launch_env(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    git = binaries / "git"
    git.write_text('#!/bin/sh\ncase "$1" in rev-parse) echo "${TEST_GIT_SHA:-test-sha}";; status) printf "%s" "${TEST_GIT_DIRTY:-}";; *) exit 1;; esac\n')
    git.chmod(0o755)
    python = binaries / "record-python"
    python.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "if sys.argv[1] == '-':\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        "with open(os.environ['TEST_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps({'args': sys.argv[1:], 'env': dict(os.environ)}) + '\\n')\n"
    )
    python.chmod(0o755)
    rows = []
    for step in range(25000, 2000001, 25000):
        checkpoint = tmp_path / f"checkpoint with spaces_{step}"
        checkpoint.write_bytes(f"weights-{step}".encode())
        metadata = Path(str(checkpoint) + ".metadata.json")
        metadata.write_text(json.dumps({"step": step}))
        rows.append({
            "step": step, "path": str(checkpoint), "source_run": SOURCE,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        })
    manifest = tmp_path / "inventory.json"
    manifest.write_text(json.dumps({"checkpoints": rows}))
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({"source_run": SOURCE}))
    run_map = tmp_path / "run-map.json"
    run_map.write_text("{}")
    return {
        **os.environ, "PATH": f"{binaries}:{os.environ['PATH']}",
        "EXPECTED_ACTION_MODES_SHA": "test-sha", "BANK_MATRIX": str(matrix),
        "CHECKPOINT_MANIFEST": str(manifest),
        "AMBI_BENCHMARK_OUTPUT_ROOT": str(tmp_path / "output"),
        "AMBI_DMC_PYTHON": str(python), "EVAL_RUN_MAP": str(run_map),
        "SLURM_ARRAY_TASK_ID": "0", "SLURM_JOB_ID": "test-bank",
        "SLURM_SUBMIT_DIR": str(tmp_path), "SLURM_CPUS_PER_TASK": "6",
        "SLURM_TMPDIR": str(tmp_path / "scratch"),
        "TEST_CALLS": str(tmp_path / "calls.jsonl"),
    }


def _calls(env):
    return [json.loads(line) for line in Path(env["TEST_CALLS"]).read_text().splitlines()]


def _prior(env, step):
    prior = Path(env["AMBI_BENCHMARK_OUTPUT_ROOT"]) / f"mppi/step_{step}/bundle"
    prior.mkdir(parents=True)
    (prior / "manifest.json").write_text("{}")
    return prior


@pytest.mark.parametrize("index", [0, 39, 79])
@pytest.mark.parametrize("mode", ["mppi", "fixed", "adaptive"])
def test_production_routes_checkpoint_planner_and_existing_prior(launch_env, index, mode):
    env = launch_env
    env.update(SLURM_ARRAY_TASK_ID=str(index), TD_AMBI_EVAL_MODE=mode)
    step = (index + 1) * 25000
    if mode != "mppi":
        prior = _prior(env, step)
    subprocess.run(["/bin/bash", str(LAUNCHER)], env=env, check=True, capture_output=True, close_fds=False, text=True)
    recorded = _calls(env)
    evaluation, report = [item["args"] for item in recorded]
    assert evaluation[0] == "evaluate_ambi_checkpoint.py"
    assert evaluation[evaluation.index("--checkpoint") + 1].endswith(f"checkpoint with spaces_{step}")
    assert evaluation[evaluation.index("--matrix") + 1] == env["BANK_MATRIX"]
    assert evaluation[evaluation.index("--checkpoint-inventory") + 1] == env["CHECKPOINT_MANIFEST"]
    assert evaluation[evaluation.index("--eval-run-map") + 1] == env["EVAL_RUN_MAP"]
    assert evaluation[evaluation.index("--seeds") + 1:evaluation.index("--seeds") + 6] == ["101", "102", "103", "104", "105"]
    assert evaluation[evaluation.index("--max-steps") + 1] == "500"
    expected = ["controller/prior", "controller/mppi"] if mode == "mppi" else [f"inner/{mode}"]
    assert [evaluation[i + 1] for i, arg in enumerate(evaluation) if arg == "--preset"] == expected
    output = Path(env["AMBI_BENCHMARK_OUTPUT_ROOT"]) / f"{mode}/step_{step}"
    assert evaluation[evaluation.index("--bundle-dir") + 1] == str(output / "bundle")
    assert evaluation[evaluation.index("--output") + 1] == str(output / "results.json")
    assert report[0] == "report_ambi_benchmark.py"
    assert report[report.index("--output") + 1] == str(output / "comparison.html")
    if mode == "mppi":
        assert "--reference-bundle" not in evaluation
        bundles = [str(output / "bundle")]
    else:
        assert evaluation[evaluation.index("--reference-bundle") + 1] == str(prior)
        bundles = [str(prior), str(output / "bundle")]
    assert [report[i + 1] for i, arg in enumerate(report) if arg == "--bundle"] == bundles
    assert not {"--wandb", "--overwrite", "--allow-nonfinite-metrics"} & set(evaluation)
    child_env = recorded[0]["env"]
    assert child_env["MUJOCO_GL"] == "egl" and child_env["MUJOCO_EGL_DEVICE_ID"] == "0"
    assert child_env["OMP_NUM_THREADS"] == child_env["MKL_NUM_THREADS"] == "6"
    assert child_env["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
    repeated = subprocess.run(["/bin/bash", str(LAUNCHER)], env=env, capture_output=True, close_fds=False, text=True)
    assert repeated.returncode != 0
    assert len(_calls(env)) == 2  # Never replace an existing task output.


def test_smoke_runs_all_four_selectors_without_registry_or_existing_references(launch_env):
    env = launch_env
    env.pop("EVAL_RUN_MAP")
    env["BANK_MATRIX"] = "matrix.json"  # Repo-relative matrices are supported.
    subprocess.run(["/bin/bash", str(LAUNCHER), "--smoke"], env=env, check=True, capture_output=True, close_fds=False, text=True)
    evaluation, report = [item["args"] for item in _calls(env)]
    assert [evaluation[i + 1] for i, arg in enumerate(evaluation) if arg == "--preset"] == [
        "controller/prior", "controller/mppi", "inner/fixed", "inner/adaptive",
    ]
    assert evaluation[evaluation.index("--seeds") + 1:evaluation.index("--seeds") + 3] == ["101", "102"]
    assert evaluation[evaluation.index("--max-steps") + 1] == "3"
    assert not {"--eval-run-map", "--reference-bundle", "--wandb"} & set(evaluation)
    assert evaluation[evaluation.index("--bundle-dir") + 1].endswith("/smoke/step_25000/bundle")
    assert report.count("--bundle") == 1


@pytest.mark.parametrize("fault", [
    "wrong_commit", "dirty", "invalid_index", "missing_map", "missing_prior",
    "wrong_hash", "wrong_metadata", "wrong_source", "wrong_grid", "relative_output", "unknown_mode",
])
def test_preflight_rejects_mismatches_before_any_evaluation(launch_env, fault):
    env = launch_env
    if fault == "wrong_commit":
        env["TEST_GIT_SHA"] = "other-sha"
    elif fault == "dirty":
        env["TEST_GIT_DIRTY"] = " M unrelated.py"
    elif fault == "invalid_index":
        env["SLURM_ARRAY_TASK_ID"] = "80"
    elif fault == "missing_map":
        env.pop("EVAL_RUN_MAP")
    elif fault == "missing_prior":
        env["TD_AMBI_EVAL_MODE"] = "fixed"
    elif fault in {"wrong_hash", "wrong_metadata", "wrong_source", "wrong_grid"}:
        path = Path(env["CHECKPOINT_MANIFEST"])
        manifest = json.loads(path.read_text())
        if fault == "wrong_grid":
            manifest["checkpoints"].reverse()
        else:
            key = {"wrong_hash": "sha256", "wrong_metadata": "metadata_sha256", "wrong_source": "source_run"}[fault]
            manifest["checkpoints"][0][key] = "other/source/run" if fault == "wrong_source" else "0" * 64
        path.write_text(json.dumps(manifest))
    elif fault == "relative_output":
        env["AMBI_BENCHMARK_OUTPUT_ROOT"] = "relative-output"
    else:
        env["TD_AMBI_EVAL_MODE"] = "unknown"
    result = subprocess.run(["/bin/bash", str(LAUNCHER)], env=env, capture_output=True, close_fds=False, text=True)
    assert result.returncode != 0
    assert not Path(env["TEST_CALLS"]).exists()


def test_resource_defaults_allow_submission_to_set_live_concurrency():
    subprocess.run(["/bin/bash", "-n", str(LAUNCHER)], check=True, capture_output=True, close_fds=False)
    directives = [line for line in LAUNCHER.read_text().splitlines() if line.startswith("#SBATCH ")]
    for expected in ("--partition=gpu", "--gres=gpu:l40s:1", "--cpus-per-task=6",
                     "--mem=32G", "--time=04:00:00", "--array=0-79", "--no-requeue"):
        assert f"#SBATCH {expected}" in directives
    assert not any("--account" in line or "--qos" in line or "--exclude" in line for line in directives)
    assert not any("%" in line for line in directives if "--array" in line)
