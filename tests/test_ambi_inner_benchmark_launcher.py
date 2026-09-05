"""Exercise launch modes without submitting jobs or evaluating a model."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "slurm/run_ambi_inner_benchmark_hydra.sbatch"


@pytest.mark.parametrize("mode", ["--bank-only", "--bank-smoke"])
def test_bank_launch_reuses_roots_and_never_runs_episodes(tmp_path, mode):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    git = binaries / "git"
    git.write_text('#!/bin/sh\ncase "$1" in rev-parse) echo test-sha;; status) ;; *) exit 1;; esac\n')
    git.chmod(0o755)
    python = binaries / "record-python"
    python.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "with open(os.environ['TEST_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    python.chmod(0o755)
    checkpoint = tmp_path / "checkpoint_300000"
    checkpoint.touch()
    checkpoint.with_suffix(".metadata.json").touch()
    reference = tmp_path / "reference"
    roots = reference / "step_300000/roots.json"
    roots.parent.mkdir(parents=True)
    roots.write_text("{}")
    calls = tmp_path / "calls.jsonl"
    env = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "EXPECTED_ACTION_MODES_SHA": "test-sha",
        "AMBI_CHECKPOINT_PREFIX": str(tmp_path / "checkpoint_"),
        "AMBI_BENCHMARK_OUTPUT_ROOT": str(tmp_path / "output"),
        "AMBI_BENCHMARK_REFERENCE_ROOT": str(reference),
        "AMBI_BENCHMARK_MATRIX": "configs/research/ambi_humanoid_inner_critic_sweep.json",
        "AMBI_BENCHMARK_PRESETS": "critic_budget/inner_target_c6 critic_budget/inner_target_c12",
        "AMBI_DMC_PYTHON": str(python),
        "SLURM_ARRAY_TASK_ID": "3",
        "SLURM_JOB_ID": "test-bank",
        "SLURM_SUBMIT_DIR": str(tmp_path),
        "SLURM_TMPDIR": str(tmp_path / "scratch"),
        "TEST_CALLS": str(calls),
    }
    subprocess.run(["bash", str(LAUNCHER), mode], env=env, check=True, capture_output=True, text=True)
    evaluation, report = [json.loads(line) for line in calls.read_text().splitlines()]
    assert evaluation[0] == "evaluate_ambi_checkpoint.py"
    assert evaluation[evaluation.index("--matrix") + 1] == env["AMBI_BENCHMARK_MATRIX"]
    assert [evaluation[i + 1] for i, arg in enumerate(evaluation) if arg == "--preset"] == [
        "critic_budget/inner_target_c6", "critic_budget/inner_target_c12",
    ]
    assert evaluation[evaluation.index("--root-bank") + 1] == str(roots)
    assert "--bank-only" in evaluation
    assert not {"--save-root-bank", "--reference-bundle", "--seeds", "--max-steps"} & set(evaluation)
    if mode == "--bank-smoke":
        assert evaluation[evaluation.index("--bank-repetitions") + 1] == "1"
        assert "--wandb" not in evaluation
    else:
        assert "--bank-repetitions" not in evaluation  # Matrix default: three.
        assert "--wandb" in evaluation
    assert report[0] == "report_ambi_benchmark.py"
    assert report[report.index("--bundle") + 1] == str(tmp_path / "output/step_300000/bank")
    # Reusing an output directory must fail before another evaluation starts.
    repeated = subprocess.run(["bash", str(LAUNCHER), mode], env=env, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert len(calls.read_text().splitlines()) == 2
