"""Guard and task-mapping tests for the twelve-cell Oscar launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT / "slurm/run_ambi_search_humanoid_walk_literature12_oscar.sbatch"
)
EXPECTED_SHA = "a" * 40
CAMPAIGN_ROOT = "configs/research/ambi_search_humanoid_walk_literature12"
ALGORITHM_DIR = f"{CAMPAIGN_ROOT}/algs"
STEMS = [
    "00_full_suffix__shared_a1",
    "01_full_suffix__depth_a1",
    "02_full_suffix__stage_a1",
    "03_full_suffix__shared_a4",
    "04_full_suffix__depth_a4",
    "05_lambda_online__shared_a4",
    "06_lambda_online__depth_a4",
    "07_lambda_online__shared_a12",
    "08_vtrace__shared_a4",
    "09_vtrace__depth_a4",
    "10_hard_propagation__stage_td0",
    "11_polyak_ablation__depth_lambda_tau005",
]
LAYOUTS = [
    "shared",
    "depth_conditioned",
    "stage_heads",
    "shared",
    "depth_conditioned",
    "shared",
    "depth_conditioned",
    "shared",
    "shared",
    "depth_conditioned",
    "stage_heads",
    "depth_conditioned",
]
ESTIMATORS = [
    "full_suffix",
    "full_suffix",
    "full_suffix",
    "full_suffix",
    "full_suffix",
    "lambda_return",
    "lambda_return",
    "lambda_return",
    "vtrace",
    "vtrace",
    "td0",
    "lambda_return",
]
CRITIC_UPDATES = [3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 3]
ACTOR_UPDATES = [1, 1, 1, 4, 4, 4, 4, 12, 4, 4, 1, 4]


def _executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = tmp_path / "dmcontrol-python"
    bash_args = tmp_path / "bash-args"
    exported_env = tmp_path / "exported-env"

    _executable(
        fake_bin / "git",
        """#!/bin/bash
set -eu
if [[ "$1" == "rev-parse" && "$2" == "--show-toplevel" ]]; then
  printf '%s\n' "$FAKE_GIT_ROOT"
elif [[ "$1" == "status" ]]; then
  printf '%s' "${FAKE_GIT_STATUS:-}"
elif [[ "$1" == "rev-parse" && "$2" == "HEAD" ]]; then
  printf '%s\n' "$FAKE_HEAD_COMMIT"
elif [[ "$1" == "rev-parse" && "$2" == "origin/ambisearch^{commit}" ]]; then
  printf '%s\n' "$FAKE_ORIGIN_COMMIT"
else
  exit 4
fi
""",
    )
    _executable(
        fake_bin / "bash",
        """#!/bin/bash
set -eu
printf '%s\n' "$@" > "$FAKE_BASH_ARGS"
{
  printf 'AMBI_RUN_CONFIG=%s\n' "$AMBI_RUN_CONFIG"
  printf 'AMBI_ALG_DIR=%s\n' "$AMBI_ALG_DIR"
  printf 'AMBI_PYTHON=%s\n' "$AMBI_PYTHON"
  printf 'AMBI_DURABLE_ROOT=%s\n' "$AMBI_DURABLE_ROOT"
  printf 'AMBI_LINEAGE_DIR=%s\n' "$AMBI_LINEAGE_DIR"
  printf 'AMBI_DURABLE_QUOTA_LABEL=%s\n' "$AMBI_DURABLE_QUOTA_LABEL"
  printf 'AMBI_DURABLE_QUOTA_PATH=%s\n' "$AMBI_DURABLE_QUOTA_PATH"
  printf 'WANDB_MODE=%s\n' "$WANDB_MODE"
  printf 'WANDB_DIR=%s\n' "$WANDB_DIR"
} > "$FAKE_EXPORTED_ENV"
""",
    )
    _executable(fake_python, "#!/usr/bin/env bash\nexit 0\n")

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "SLURM_ARRAY_JOB_ID": "24680",
            "SLURM_JOB_ID": "24680_0",
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_NODELIST": "fake-gpu",
            "SLURM_RESTART_COUNT": "2",
            "SLURM_SUBMIT_DIR": str(ROOT),
            "SLURM_TMPDIR": str(tmp_path / "slurm-tmp"),
            "AMBI_EXPECTED_COMMIT": EXPECTED_SHA,
            "AMBI_SEARCH_CAMPAIGN": "lit12-test",
            "AMBI_DMCONTROL_PYTHON": str(fake_python),
            "FAKE_GIT_ROOT": str(ROOT),
            "FAKE_GIT_STATUS": "",
            "FAKE_HEAD_COMMIT": EXPECTED_SHA,
            "FAKE_ORIGIN_COMMIT": EXPECTED_SHA,
            "FAKE_BASH_ARGS": str(bash_args),
            "FAKE_EXPORTED_ENV": str(exported_env),
        }
    )
    return env, bash_args, exported_env


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_literature12_launcher_syntax_resources_and_guards():
    syntax = subprocess.run(
        ["/bin/bash", "-n", str(LAUNCHER)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr
    contents = LAUNCHER.read_text(encoding="utf-8")
    for directive in (
        "#SBATCH --partition=gpu",
        "#SBATCH --gres=gpu:l40s:1",
        "#SBATCH --cpus-per-task=6",
        "#SBATCH --mem=32G",
        "#SBATCH --time=96:00:00",
        "#SBATCH --array=0-11%12",
        "#SBATCH --requeue",
        "#SBATCH --signal=USR1@3600",
        "#SBATCH --output=slurm/%x-%A_%a.out",
        "#SBATCH --error=slurm/%x-%A_%a.err",
    ):
        assert directive in contents
    assert "--nodelist" not in contents
    assert "AMBI_EXPECTED_COMMIT:?" in contents
    assert "AMBI_SEARCH_CAMPAIGN:?" in contents
    assert "origin/ambisearch^{commit}" in contents
    assert "production launch requires a clean checkout" in contents
    assert 'approved_durable_root="/oscar/scratch/rgao48/ambi-durable"' in contents
    assert 'export AMBI_DURABLE_QUOTA_PATH="/oscar/scratch"' in contents
    assert 'exec bash "$project_dir/run_ambi_oscar.sh"' in contents
    assert "srun --unbuffered" not in contents


def test_literature12_launcher_maps_every_task_to_one_guarded_lineage(tmp_path):
    base_env, _, _ = _environment(tmp_path)
    for task_id, stem in enumerate(STEMS):
        cell = tmp_path / f"cell-{task_id}"
        cell.mkdir()
        bash_args = cell / "bash-args"
        exported_env = cell / "exported-env"
        slurm_tmp = cell / "slurm-tmp"
        env = base_env.copy()
        env.update(
            {
                "SLURM_JOB_ID": f"24680_{task_id}",
                "SLURM_ARRAY_TASK_ID": str(task_id),
                "SLURM_TMPDIR": str(slurm_tmp),
                "FAKE_BASH_ARGS": str(bash_args),
                "FAKE_EXPORTED_ENV": str(exported_env),
            }
        )
        result = _run(env)
        assert result.returncode == 0, result.stderr
        assert bash_args.read_text(encoding="utf-8").splitlines() == [
            str(ROOT / "run_ambi_oscar.sh")
        ]
        exported = dict(
            line.split("=", 1)
            for line in exported_env.read_text(encoding="utf-8").splitlines()
        )
        assert exported["AMBI_RUN_CONFIG"] == (
            f"{CAMPAIGN_ROOT}/experiments/{stem}.json"
        )
        assert exported["AMBI_ALG_DIR"] == ALGORITHM_DIR
        assert exported["AMBI_PYTHON"] == env["AMBI_DMCONTROL_PYTHON"]
        assert exported["AMBI_DURABLE_ROOT"] == (
            "/oscar/scratch/rgao48/ambi-durable"
        )
        assert exported["AMBI_LINEAGE_DIR"] == (
            "/oscar/scratch/rgao48/ambi-durable/ambisearch-literature12/"
            f"lit12-test/task_{task_id}"
        )
        assert exported["AMBI_DURABLE_QUOTA_LABEL"] == "rgao48"
        assert exported["AMBI_DURABLE_QUOTA_PATH"] == "/oscar/scratch"
        assert exported["WANDB_MODE"] == "online"
        assert exported["WANDB_DIR"] == str(
            slurm_tmp / f"ambi-search-lit12-wandb-24680-{task_id}-2"
        )
        assert f"Algorithm config: {stem}" in result.stdout
        assert f"Critic layout: {LAYOUTS[task_id]}" in result.stdout
        assert f"Return estimator: {ESTIMATORS[task_id]}" in result.stdout
        assert f"Critic updates per round: {CRITIC_UPDATES[task_id]}" in result.stdout
        assert f"Actor updates per round: {ACTOR_UPDATES[task_id]}" in result.stdout
        assert "Model transitions per real action: 12288" in result.stdout
        assert "Inner policy-to-prior KL coefficient: 0.0" in result.stdout
        assert "Outer behavior-policy KL schedule: none" in result.stdout
        assert "Outer behavior-policy KL coefficient: 0.0" in result.stdout
        assert f"Source commit: {EXPECTED_SHA}" in result.stdout
        assert f"origin/ambisearch: {EXPECTED_SHA}" in result.stdout


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_expected", "AMBI_EXPECTED_COMMIT"),
        ("malformed_expected", "full lowercase Git SHA"),
        ("missing_campaign", "AMBI_SEARCH_CAMPAIGN"),
        ("unsafe_campaign", "filesystem-safe basename"),
        ("bad_task", "configured range 0--11"),
        ("dirty", "clean checkout"),
        ("head_mismatch", "HEAD does not match"),
        ("origin_mismatch", "origin/ambisearch does not match"),
        ("missing_python", "missing locked DMControl interpreter"),
    ),
)
def test_literature12_launcher_guards_fail_before_delegation(
    tmp_path, mutation, message
):
    env, bash_args, _ = _environment(tmp_path)
    if mutation == "missing_expected":
        del env["AMBI_EXPECTED_COMMIT"]
    elif mutation == "malformed_expected":
        env["AMBI_EXPECTED_COMMIT"] = "abc123"
    elif mutation == "missing_campaign":
        del env["AMBI_SEARCH_CAMPAIGN"]
    elif mutation == "unsafe_campaign":
        env["AMBI_SEARCH_CAMPAIGN"] = "../escape"
    elif mutation == "bad_task":
        env["SLURM_ARRAY_TASK_ID"] = "12"
    elif mutation == "dirty":
        env["FAKE_GIT_STATUS"] = "?? untracked-output"
    elif mutation == "head_mismatch":
        env["FAKE_HEAD_COMMIT"] = "b" * 40
    elif mutation == "origin_mismatch":
        env["FAKE_ORIGIN_COMMIT"] = "b" * 40
    elif mutation == "missing_python":
        env["AMBI_DMCONTROL_PYTHON"] = str(tmp_path / "missing-python")
    else:  # pragma: no cover
        raise AssertionError(mutation)
    result = _run(env)
    assert result.returncode != 0
    assert message in result.stderr
    assert not bash_args.exists()
