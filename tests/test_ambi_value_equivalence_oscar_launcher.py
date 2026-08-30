import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT
    / "slurm/run_ambi_humanoid_walk_value_equivalence_loss_oscar.sbatch"
)
EXPECTED_SHA = "a" * 40
ALGORITHM_CONFIG = (
    "ambi_humanoid_walk_q5_pair2_ve_loss_c0p1_mc4_1m_seed55"
)
MANIFEST = f"configs/dmcontrol/experiments/{ALGORITHM_CONFIG}.json"


def _executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = tmp_path / "dmcontrol-python"
    bash_args = tmp_path / "bash-args"
    exported_env = tmp_path / "exported-env"
    slurm_tmp = tmp_path / "slurm-tmp"

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
elif [[ "$1" == "rev-parse" && "$2" == "origin/ambi-inner-loop^{commit}" ]]; then
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
  printf 'MUJOCO_GL=%s\n' "$MUJOCO_GL"
  printf 'PYTHONDONTWRITEBYTECODE=%s\n' "$PYTHONDONTWRITEBYTECODE"
  printf 'PYTHONUNBUFFERED=%s\n' "$PYTHONUNBUFFERED"
  printf 'WANDB_MODE=%s\n' "$WANDB_MODE"
  printf 'WANDB_DIR=%s\n' "$WANDB_DIR"
  printf 'WANDB_CACHE_DIR=%s\n' "$WANDB_CACHE_DIR"
  printf 'WANDB_DATA_DIR=%s\n' "$WANDB_DATA_DIR"
  printf 'WANDB_ARTIFACT_DIR=%s\n' "$WANDB_ARTIFACT_DIR"
  printf 'WANDB_DISABLE_CODE=%s\n' "$WANDB_DISABLE_CODE"
} > "$FAKE_EXPORTED_ENV"
""",
    )
    _executable(fake_python, "#!/bin/bash\nexit 0\n")

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "SLURM_JOB_ID": "24680",
            "SLURM_NODELIST": "fake-gpu",
            "SLURM_RESTART_COUNT": "2",
            "SLURM_SUBMIT_DIR": str(ROOT),
            "SLURM_TMPDIR": str(slurm_tmp),
            "AMBI_EXPECTED_COMMIT": EXPECTED_SHA,
            "AMBI_VALUE_EQUIVALENCE_CAMPAIGN": "ve-loss-test",
            "AMBI_DMCONTROL_PYTHON": str(fake_python),
            "FAKE_GIT_ROOT": str(ROOT),
            "FAKE_GIT_STATUS": "",
            "FAKE_HEAD_COMMIT": EXPECTED_SHA,
            "FAKE_ORIGIN_COMMIT": EXPECTED_SHA,
            "FAKE_BASH_ARGS": str(bash_args),
            "FAKE_EXPORTED_ENV": str(exported_env),
        }
    )
    return env, bash_args, exported_env, slurm_tmp


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


def test_launcher_syntax_and_single_job_resource_contract():
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
        "#SBATCH --requeue",
        "#SBATCH --signal=USR1@3600",
        "#SBATCH --output=slurm/%x-%j.out",
        "#SBATCH --error=slurm/%x-%j.err",
    ):
        assert directive in contents
    assert "#SBATCH --array" not in contents
    assert "--nodelist" not in contents
    assert "srun " not in contents
    assert "AMBI_EXPECTED_COMMIT:?" in contents
    assert "AMBI_VALUE_EQUIVALENCE_CAMPAIGN:?" in contents
    assert "git status --porcelain --untracked-files=all" in contents
    assert "origin/ambi-inner-loop^{commit}" in contents
    assert 'exec bash "$project_dir/run_ambi_oscar.sh"' in contents
    assert ALGORITHM_CONFIG in contents
    assert 'manifest="configs/dmcontrol/experiments/${algorithm_config}.json"' in contents


def test_launcher_exports_one_guarded_cell_and_delegates(tmp_path):
    env, bash_args, exported_env, slurm_tmp = _environment(tmp_path)

    result = _run(env)

    assert result.returncode == 0, result.stderr
    assert bash_args.read_text(encoding="utf-8").splitlines() == [
        str(ROOT / "run_ambi_oscar.sh")
    ]
    exported = dict(
        line.split("=", 1)
        for line in exported_env.read_text(encoding="utf-8").splitlines()
    )
    wandb_root = slurm_tmp / "ambi-ve-loss-wandb-24680-2"
    assert exported == {
        "AMBI_RUN_CONFIG": MANIFEST,
        "AMBI_ALG_DIR": "configs/dmcontrol/algs",
        "AMBI_PYTHON": env["AMBI_DMCONTROL_PYTHON"],
        "AMBI_DURABLE_ROOT": "/oscar/home/rgao48/ambi-durable",
        "AMBI_LINEAGE_DIR": (
            "/oscar/home/rgao48/ambi-durable/value-equivalence-loss/"
            "ve-loss-test/seed_55"
        ),
        "AMBI_DURABLE_QUOTA_LABEL": "rgao48",
        "AMBI_DURABLE_QUOTA_PATH": "/oscar/home",
        "MUJOCO_GL": "egl",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "WANDB_MODE": "online",
        "WANDB_DIR": str(wandb_root),
        "WANDB_CACHE_DIR": str(wandb_root / "cache"),
        "WANDB_DATA_DIR": str(wandb_root / "data"),
        "WANDB_ARTIFACT_DIR": str(wandb_root / "artifacts"),
        "WANDB_DISABLE_CODE": "true",
    }
    for path in (
        wandb_root,
        wandb_root / "cache",
        wandb_root / "data",
        wandb_root / "artifacts",
    ):
        assert path.is_dir()
    assert "Algorithm config: " + ALGORITHM_CONFIG in result.stdout
    assert "Agent decisions: 1000000" in result.stdout
    assert "Value-equivalence loss coefficient: 0.1" in result.stdout
    assert f"Source commit: {EXPECTED_SHA}" in result.stdout


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_expected", "AMBI_EXPECTED_COMMIT"),
        ("malformed_expected", "full lowercase Git SHA"),
        ("missing_campaign", "AMBI_VALUE_EQUIVALENCE_CAMPAIGN"),
        ("unsafe_campaign", "filesystem-safe basename"),
        ("array", "must not run as a Slurm array"),
        ("dirty", "clean checkout"),
        ("head_mismatch", "HEAD does not match"),
        ("origin_mismatch", "origin/ambi-inner-loop does not match"),
        ("bad_restart", "SLURM_RESTART_COUNT must be non-negative"),
        ("missing_python", "missing locked DMControl interpreter"),
    ),
)
def test_launcher_guards_fail_before_staging_or_delegation(
    tmp_path, mutation, message
):
    env, bash_args, _, slurm_tmp = _environment(tmp_path)
    if mutation == "missing_expected":
        del env["AMBI_EXPECTED_COMMIT"]
    elif mutation == "malformed_expected":
        env["AMBI_EXPECTED_COMMIT"] = "abc123"
    elif mutation == "missing_campaign":
        del env["AMBI_VALUE_EQUIVALENCE_CAMPAIGN"]
    elif mutation == "unsafe_campaign":
        env["AMBI_VALUE_EQUIVALENCE_CAMPAIGN"] = "../escape"
    elif mutation == "array":
        env["SLURM_ARRAY_JOB_ID"] = env["SLURM_JOB_ID"]
        env["SLURM_ARRAY_TASK_ID"] = "0"
    elif mutation == "dirty":
        env["FAKE_GIT_STATUS"] = "?? unexpected-output"
    elif mutation == "head_mismatch":
        env["FAKE_HEAD_COMMIT"] = "b" * 40
    elif mutation == "origin_mismatch":
        env["FAKE_ORIGIN_COMMIT"] = "b" * 40
    elif mutation == "bad_restart":
        env["SLURM_RESTART_COUNT"] = "not-an-integer"
    elif mutation == "missing_python":
        env["AMBI_DMCONTROL_PYTHON"] = str(tmp_path / "missing-python")
    else:  # pragma: no cover - protects additions to the parameter table
        raise AssertionError(f"unknown mutation: {mutation}")

    result = _run(env)

    assert result.returncode != 0
    assert message in result.stderr
    assert not bash_args.exists()
    assert not slurm_tmp.exists()


def test_launcher_rejects_non_root_submit_directory_before_staging(tmp_path):
    env, bash_args, _, slurm_tmp = _environment(tmp_path)
    other_root = tmp_path / "other-root"
    other_root.mkdir()
    env["FAKE_GIT_ROOT"] = str(other_root)

    result = _run(env)

    assert result.returncode != 0
    assert "submit directory is not the Git root" in result.stderr
    assert not bash_args.exists()
    assert not slurm_tmp.exists()
