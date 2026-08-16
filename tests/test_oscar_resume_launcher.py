import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from utils.oscar_resume_launcher import (
    DoneVerificationError,
    HandoffVerificationError,
    QuotaPreflightError,
    StoragePreflightError,
    verify_done,
    verify_durable_storage,
    verify_handoff,
    verify_quota_output,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_ambi_oscar.sh"


def _write_lineage(
    root: Path,
    *,
    generation: str = "generation-000001",
    target: int = 1000,
    job_id: str = "98765",
    segment_id: str = "98765.2",
    done: bool = False,
) -> None:
    root.mkdir()
    (root / "LINEAGE.json").write_text(
        json.dumps({"schema_version": 1, "metadata": {"total_steps": target}}),
        encoding="utf-8",
    )
    (root / "LATEST").write_text(f"{generation}\n", encoding="ascii")
    generation_dir = root / "generations" / generation
    generation_dir.mkdir(parents=True)
    (generation_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generation_id": generation,
                "metadata": {
                    "segment_id": segment_id,
                    "global_step": target,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "HANDOFF.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "slurm_job_id": job_id,
                "segment_id": segment_id,
                "generation_id": generation,
            }
        ),
        encoding="utf-8",
    )
    if done:
        (root / "DONE").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "segment_id": segment_id,
                    "generation_id": generation,
                    "global_step": target,
                }
            ),
            encoding="utf-8",
        )


def _executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _launcher_environment(
    tmp_path: Path, *, status: int, restart: int = 0, quota_state: str = "OK"
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    srun_args = tmp_path / "srun-args"
    scontrol_calls = tmp_path / "scontrol-calls"
    _executable(
        fake_bin / "git",
        """#!/bin/sh
case "$1" in
  rev-parse) printf '%s\n' "$FAKE_GIT_ROOT" ;;
  status) printf '%s' "${FAKE_GIT_STATUS:-}" ;;
  *) exit 4 ;;
esac
""",
    )
    _executable(fake_bin / "module", "#!/bin/sh\nexit 0\n")
    _executable(fake_bin / "checkquota", "#!/bin/sh\nprintf '%s\n' \"$FAKE_QUOTA\"\n")
    _executable(
        fake_bin / "srun",
        """#!/bin/sh
printf '%s\n' "$@" > "$FAKE_SRUN_ARGS"
if [ -n "${FAKE_COMPLETION_SOURCE:-}" ]; then
  if [ -d "$AMBI_LINEAGE_DIR" ]; then
    cp -R "$FAKE_COMPLETION_SOURCE"/. "$AMBI_LINEAGE_DIR"/
  else
    mv "$FAKE_COMPLETION_SOURCE" "$AMBI_LINEAGE_DIR"
  fi
fi
exit "$FAKE_STATUS"
""",
    )
    _executable(
        fake_bin / "scontrol",
        """#!/bin/sh
printf '%s\n' "$*" >> "$FAKE_SCONTROL_CALLS"
exit "${FAKE_SCONTROL_STATUS:-0}"
""",
    )
    mamba = tmp_path / "mamba"
    profile = mamba / "etc" / "profile.d"
    profile.mkdir(parents=True)
    (profile / "conda.sh").write_text(
        """if [ -n "${FAKE_PREFLIGHT_SECONDS:-}" ]; then SECONDS="$FAKE_PREFLIGHT_SECONDS"; fi
conda() { return 0; }
""",
        encoding="utf-8",
    )
    durable = tmp_path / "durable"
    durable.mkdir()
    lineage = durable / "lineage"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "SLURM_JOB_ID": "98765",
            "SLURM_RESTART_COUNT": str(restart),
            "SLURM_SUBMIT_DIR": str(ROOT),
            "AMBI_DURABLE_ROOT": str(durable),
            "AMBI_LINEAGE_DIR": str(lineage),
            "AMBI_PYTHON": sys.executable,
            "MAMBA_ROOT_PREFIX": str(mamba),
            "FAKE_GIT_ROOT": str(ROOT),
            "FAKE_QUOTA": f"data+rbalestr 1 TB 2 TB {quota_state}",
            "FAKE_STATUS": str(status),
            "FAKE_SRUN_ARGS": str(srun_args),
            "FAKE_SCONTROL_CALLS": str(scontrol_calls),
        }
    )
    return env, lineage, srun_args, scontrol_calls


def _run(env):
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_resources_and_foreground_signal_contract_are_explicit():
    contents = LAUNCHER.read_text(encoding="utf-8")
    for directive in (
        "#SBATCH --partition=gpu",
        "#SBATCH --gres=gpu:l40s:1",
        "#SBATCH --time=96:00:00",
        "#SBATCH --mem=32G",
        "#SBATCH --cpus-per-task=6",
        "#SBATCH --requeue",
        "#SBATCH --signal=USR1@3600",
    ):
        assert directive in contents
    assert "srun --unbuffered --kill-on-bad-exit=1" in contents
    assert "AMBI_WANDB_REWIND_VERIFIED" not in contents
    assert "AMBI_OSCAR_RESUME_CANARY_VERIFIED" not in contents
    assert "AMBI_OSCAR_DURABLE_PREFIX" not in contents
    assert "AMBI_OSCAR_TRANSIENT_ROOTS" not in contents


def test_quota_requires_one_row_ending_in_exact_ok():
    assert verify_quota_output(
        "data+rbalestr 1 TB 2 TB OK\n", "data+rbalestr"
    ).endswith(" OK")
    for output in (
        "data+rbalestr 1 TB 2 TB GRACE_EXPIRED\n",
        "data+rbalestr 1 TB 2 TB NOT_OK\n",
        "home 1 TB 2 TB OK\n",
        "data+rbalestr a OK\ndata+rbalestr b OK\n",
    ):
        with pytest.raises(QuotaPreflightError):
            verify_quota_output(output, "data+rbalestr")


def test_quota_can_select_exact_filesystem_path_for_shared_allocation():
    output = (
        "rgao48 /oscar/scratch 1 TB 2 TB GRACE_EXPIRED\n"
        "rgao48 /oscar/home 80 GB 100 GB OK\n"
    )
    with pytest.raises(QuotaPreflightError, match="found 2"):
        verify_quota_output(output, "rgao48")
    assert (
        verify_quota_output(output, "rgao48", "/oscar/home")
        == "rgao48 /oscar/home 80 GB 100 GB OK"
    )
    with pytest.raises(QuotaPreflightError, match="not exactly OK"):
        verify_quota_output(output, "rgao48", "/oscar/scratch")
    with pytest.raises(QuotaPreflightError, match="found 0"):
        verify_quota_output(output, "rgao48", "/oscar/hom")
    with pytest.raises(QuotaPreflightError, match="one non-empty field"):
        verify_quota_output(output, "rgao48", "/oscar/home path")


def test_launcher_passes_optional_quota_filesystem_path(tmp_path):
    env, _, args_path, scontrol = _launcher_environment(tmp_path, status=23)
    env.update(
        {
            "AMBI_DURABLE_QUOTA_LABEL": "rgao48",
            "AMBI_DURABLE_QUOTA_PATH": "/oscar/home",
            "FAKE_QUOTA": (
                "rgao48 /oscar/scratch 1 TB 2 TB GRACE_EXPIRED\n"
                "rgao48 /oscar/home 80 GB 100 GB OK"
            ),
        }
    )
    result = _run(env)
    assert result.returncode == 23, result.stderr
    assert args_path.exists()
    assert not scontrol.exists()


def test_storage_requires_child_path_and_performs_fsync_probe(tmp_path, monkeypatch):
    root = tmp_path / "durable"
    root.mkdir()
    calls = []
    real_fsync = os.fsync

    def observed_fsync(descriptor):
        calls.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", observed_fsync)
    metrics = {}
    assert verify_durable_storage(
        durable_root=root, lineage_dir=root / "lineage", metrics=metrics
    ) == (root.resolve(), (root / "lineage").resolve())
    assert len(calls) == 3  # file, directory after create, directory after remove
    assert set(metrics) == {
        "file_fsync_seconds",
        "directory_fsync_seconds",
        "total_seconds",
    }
    assert not any(path.name.startswith(".ambi-fsync-probe-") for path in root.iterdir())
    with pytest.raises(StoragePreflightError, match="below"):
        verify_durable_storage(durable_root=root, lineage_dir=root)
    with pytest.raises(StoragePreflightError, match="outside"):
        verify_durable_storage(durable_root=root, lineage_dir=tmp_path / "other")


def test_control_records_check_scheduler_facts_small_manifest_and_target(tmp_path):
    lineage = tmp_path / "lineage"
    _write_lineage(lineage, done=True)
    # Large payload/checksum validation is deliberately left to Python restore;
    # the launcher reads only the small manifest metadata that owns the record.
    assert not (lineage / "generations" / "generation-000001" / "trainer.pt").exists()
    assert verify_handoff(
        lineage_dir=lineage, slurm_job_id="98765", segment_id="98765.2"
    ) == "generation-000001"
    assert verify_done(lineage_dir=lineage) == "generation-000001"

    json_path = lineage / "HANDOFF.json"
    record = json.loads(json_path.read_text(encoding="utf-8"))
    record["segment_id"] = "stale"
    json_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(HandoffVerificationError, match="stale segment"):
        verify_handoff(
            lineage_dir=lineage, slurm_job_id="98765", segment_id="98765.2"
        )

    done = json.loads((lineage / "DONE").read_text(encoding="utf-8"))
    done["global_step"] = 999
    (lineage / "DONE").write_text(json.dumps(done), encoding="utf-8")
    with pytest.raises(DoneVerificationError, match="LATEST generation"):
        verify_done(lineage_dir=lineage)


def test_status_zero_requires_done_and_never_requeues(tmp_path):
    env, _, args_path, scontrol = _launcher_environment(tmp_path, status=0)
    completion = tmp_path / "completion"
    _write_lineage(completion, segment_id="98765.0", done=True)
    env["FAKE_COMPLETION_SOURCE"] = str(completion)
    env["FAKE_PREFLIGHT_SECONDS"] = "125"
    result = _run(env)
    assert result.returncode == 0, result.stderr
    args = args_path.read_text(encoding="utf-8").splitlines()
    assert args[args.index("--resume-mode") + 1] == "new"
    remaining = int(args[args.index("--drain-after-seconds") + 1])
    # Bash SECONDS has whole-second resolution; process startup may cross the
    # next tick in addition to the mocked 125-second preflight.
    assert 341700 - 127 <= remaining <= 341700 - 125
    assert not scontrol.exists()

    missing_tmp = tmp_path / "missing"
    missing_tmp.mkdir()
    missing_env, _, _, missing_scontrol = _launcher_environment(missing_tmp, status=0)
    missing = _run(missing_env)
    assert missing.returncode != 0
    assert "cannot read valid DONE" in missing.stderr
    assert not missing_scontrol.exists()


def test_existing_valid_done_is_noop(tmp_path):
    env, lineage, args_path, scontrol = _launcher_environment(
        tmp_path, status=23, restart=4
    )
    _write_lineage(lineage, done=True)
    result = _run(env)
    assert result.returncode == 0, result.stderr
    assert "already complete" in result.stdout
    assert not args_path.exists()
    assert not scontrol.exists()


def test_restart_requires_resume_and_requeues_exactly_once(tmp_path):
    env, lineage, args_path, scontrol = _launcher_environment(
        tmp_path, status=75, restart=2
    )
    _write_lineage(lineage, segment_id="98765.1")
    completion = tmp_path / "completion"
    _write_lineage(completion, segment_id="98765.2")
    env["FAKE_COMPLETION_SOURCE"] = str(completion)
    env["AMBI_RESUME_MODE"] = "new"
    result = _run(env)
    assert result.returncode == 0, result.stderr
    args = args_path.read_text(encoding="utf-8").splitlines()
    assert args[args.index("--resume-mode") + 1] == "required"
    assert scontrol.read_text(encoding="utf-8").splitlines() == ["requeue 98765"]


def test_external_restart_without_prior_clean_handoff_never_starts(tmp_path):
    env, lineage, args_path, scontrol = _launcher_environment(
        tmp_path, status=23, restart=1
    )
    _write_lineage(lineage, segment_id="unrelated")
    result = _run(env)
    assert result.returncode != 0
    assert "stale segment" in result.stderr
    assert not args_path.exists()
    assert not scontrol.exists()


@pytest.mark.parametrize("status", [1, 23, 74])
def test_training_failure_never_requeues(tmp_path, status):
    env, _, _, scontrol = _launcher_environment(tmp_path, status=status)
    result = _run(env)
    assert result.returncode == status
    assert "automatic requeue suppressed" in result.stderr
    assert not scontrol.exists()


def test_stale_handoff_and_quota_failure_never_requeue(tmp_path):
    env, lineage, _, scontrol = _launcher_environment(tmp_path, status=75, restart=2)
    _write_lineage(lineage, segment_id="98765.1")
    result = _run(env)
    assert result.returncode != 0
    assert "stale segment" in result.stderr
    assert not scontrol.exists()

    quota_tmp = tmp_path / "quota"
    quota_tmp.mkdir()
    quota_env, _, quota_args, quota_scontrol = _launcher_environment(
        quota_tmp, status=0, quota_state="GRACE_EXPIRED"
    )
    quota_result = _run(quota_env)
    assert quota_result.returncode != 0
    assert not quota_args.exists()
    assert not quota_scontrol.exists()


def test_requeue_failure_propagates_and_rollback_is_consumed_once(tmp_path):
    env, lineage, args_path, scontrol = _launcher_environment(
        tmp_path, status=75, restart=0
    )
    _write_lineage(lineage, segment_id="98765.0")
    env["AMBI_RESUME_MODE"] = "required"
    env["AMBI_RESUME_GENERATION"] = "operator-rollback"
    env["FAKE_SCONTROL_STATUS"] = "9"
    result = _run(env)
    assert result.returncode == 9
    args = args_path.read_text(encoding="utf-8").splitlines()
    assert args[args.index("--resume-generation") + 1] == "operator-rollback"
    assert scontrol.read_text(encoding="utf-8").splitlines() == ["requeue 98765"]

    restart_tmp = tmp_path / "restart"
    restart_tmp.mkdir()
    restart_env, restart_lineage, restart_args, _ = _launcher_environment(
        restart_tmp, status=23, restart=1
    )
    _write_lineage(restart_lineage, segment_id="98765.0")
    restart_env["AMBI_RESUME_GENERATION"] = "operator-rollback"
    restart_result = _run(restart_env)
    assert restart_result.returncode == 23
    assert "--resume-generation" not in restart_args.read_text(encoding="utf-8").splitlines()


def test_launcher_accepts_explicit_run_and_algorithm_config_paths(tmp_path):
    env, _, args_path, _ = _launcher_environment(tmp_path, status=23)
    env["AMBI_RUN_CONFIG"] = "configs/ambi/experiments/ambi_horizon_h1.json"
    env["AMBI_ALG_DIR"] = "configs/ambi/algs-custom"
    result = _run(env)
    assert result.returncode == 23
    args = args_path.read_text(encoding="utf-8").splitlines()
    assert args[args.index("--run") + 1] == env["AMBI_RUN_CONFIG"]
    assert args[args.index("--alg-dir") + 1] == env["AMBI_ALG_DIR"]


def test_startup_that_exhausts_drain_budget_never_starts_training(tmp_path):
    env, _, args_path, scontrol = _launcher_environment(tmp_path, status=0)
    env["AMBI_DRAIN_AFTER_SECONDS"] = "1"
    env["FAKE_PREFLIGHT_SECONDS"] = "2"
    result = _run(env)
    assert result.returncode != 0
    assert "drain budget expired during startup" in result.stderr
    assert not args_path.exists()
    assert not scontrol.exists()
