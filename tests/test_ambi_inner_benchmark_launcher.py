"""Exercise launch modes without submitting jobs or evaluating a model."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "slurm/run_ambi_inner_benchmark_hydra.sbatch"


@pytest.fixture
def launch_env(tmp_path):
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
    reference = tmp_path / "reference"
    for step in (100000, 200000, 300000, 400000, 500000):
        checkpoint = tmp_path / f"checkpoint_{step}"
        checkpoint.touch()
        checkpoint.with_suffix(".metadata.json").touch()
        prior = reference / f"step_{step}/prior"
        prior.mkdir(parents=True)
        (prior / "manifest.json").write_text("{}")
        (prior.parent / "roots.json").write_text("{}")
    calls = tmp_path / "calls.jsonl"
    return {
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


@pytest.mark.parametrize("mode", ["--bank-only", "--bank-smoke"])
def test_bank_launch_reuses_roots_and_never_runs_episodes(tmp_path, launch_env, mode):
    env = launch_env
    roots = Path(env["AMBI_BENCHMARK_REFERENCE_ROOT"]) / "step_300000/roots.json"
    calls = Path(env["TEST_CALLS"])
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


@pytest.mark.parametrize("step_index", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("bootstrap", ["inner_target", "outer_target"])
def test_full_launch_evaluates_both_budgets_and_reuses_prior(launch_env, step_index, bootstrap):
    env = launch_env
    env["SLURM_ARRAY_TASK_ID"] = str(step_index)
    presets = [f"critic_budget/{bootstrap}_c6", f"critic_budget/{bootstrap}_c12"]
    env["AMBI_BENCHMARK_PRESETS"] = " ".join(presets)
    subprocess.run(["bash", str(LAUNCHER)], env=env, check=True, capture_output=True, text=True)
    evaluation, report = [json.loads(line) for line in Path(env["TEST_CALLS"]).read_text().splitlines()]
    assert evaluation[0] == "evaluate_ambi_checkpoint.py"
    assert [evaluation[i + 1] for i, arg in enumerate(evaluation) if arg == "--preset"] == presets
    assert evaluation[evaluation.index("--checkpoint") + 1] == f"{env['AMBI_CHECKPOINT_PREFIX']}{step_index * 100000}"
    assert evaluation[evaluation.index("--seeds") + 1:evaluation.index("--seeds") + 6] == ["101", "102", "103", "104", "105"]
    assert evaluation[evaluation.index("--max-steps") + 1] == "500"
    prior = f"{env['AMBI_BENCHMARK_REFERENCE_ROOT']}/step_{step_index * 100000}/prior"
    assert evaluation[evaluation.index("--reference-bundle") + 1] == prior
    assert "--wandb" in evaluation
    assert not {"--bank-only", "--root-bank", "--save-root-bank", "--bank-repetitions"} & set(evaluation)
    assert report[0] == "report_ambi_benchmark.py"
    assert [report[i + 1] for i, arg in enumerate(report) if arg == "--bundle"] == [
        prior, f"{env['AMBI_BENCHMARK_OUTPUT_ROOT']}/step_{step_index * 100000}/inner",
    ]


@pytest.mark.parametrize("explicit_single", [False, True])
def test_existing_single_preset_fallback_remains_compatible(launch_env, explicit_single):
    env = launch_env
    env.pop("AMBI_BENCHMARK_PRESETS")
    preset = "named_run/d512_4_j6_outer_target" if explicit_single else "named_run/d512_4_j6"
    if explicit_single:
        env["AMBI_BENCHMARK_PRESET"] = preset
    else:
        env.pop("AMBI_BENCHMARK_PRESET", None)
    subprocess.run(["bash", str(LAUNCHER)], env=env, check=True, capture_output=True, text=True)
    evaluation, _ = [json.loads(line) for line in Path(env["TEST_CALLS"]).read_text().splitlines()]
    assert [evaluation[i + 1] for i, arg in enumerate(evaluation) if arg == "--preset"] == [preset]


def test_full_smoke_retains_short_protocol_and_multiple_presets(launch_env):
    env = launch_env
    subprocess.run(["bash", str(LAUNCHER), "--smoke"], env=env, check=True, capture_output=True, text=True)
    evaluation, _ = [json.loads(line) for line in Path(env["TEST_CALLS"]).read_text().splitlines()]
    assert evaluation[evaluation.index("--seeds") + 1:evaluation.index("--seeds") + 3] == ["101", "--max-steps"]
    assert evaluation[evaluation.index("--max-steps") + 1] == "3"
    assert evaluation.count("--preset") == 2
    assert "--reference-bundle" in evaluation
    assert not {"--wandb", "--bank-only", "--root-bank"} & set(evaluation)


def test_original_full_workflow_still_creates_prior_when_reference_omitted(launch_env):
    env = launch_env
    for key in ("AMBI_BENCHMARK_REFERENCE_ROOT", "AMBI_BENCHMARK_PRESETS",
                "AMBI_BENCHMARK_PRESET", "AMBI_BENCHMARK_MATRIX"):
        env.pop(key, None)
    subprocess.run(["bash", str(LAUNCHER)], env=env, check=True, capture_output=True, text=True)
    prior, evaluation, report = [json.loads(line) for line in Path(env["TEST_CALLS"]).read_text().splitlines()]
    assert prior[prior.index("--preset") + 1] == "named_run/prior"
    assert "--save-root-bank" in prior
    assert evaluation[evaluation.index("--preset") + 1] == "named_run/d512_4_j6"
    assert report[0] == "report_ambi_benchmark.py"
