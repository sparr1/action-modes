"""Exercise the continuation shell against real storage/handoff verification."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm/run_ambi_exact_continuation_oscar.sbatch"


def _executable(path, contents):
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _record(lineage, *, segment="123.0", done=False):
    generation = "generation-000001"
    (lineage / "generations" / generation).mkdir(parents=True, exist_ok=True)
    (lineage / "LINEAGE.json").write_text(json.dumps({
        "schema_version": 1, "metadata": {"total_steps": 1000000},
    }))
    (lineage / "LATEST").write_text(generation + "\n")
    (lineage / "generations" / generation / "manifest.json").write_text(json.dumps({
        "generation_id": generation,
        "metadata": {"segment_id": segment, "global_step": 1000000 if done else 500000},
    }))
    record = {
        "schema_version": 1, "slurm_job_id": segment.split(".")[0],
        "segment_id": segment, "generation_id": generation,
    }
    (lineage / "HANDOFF.json").write_text(json.dumps(record))
    if done:
        record.update(global_step=1000000, target_step=1000000)
        (lineage / "DONE").write_text(json.dumps(record))


def _environment(tmp_path, *, status=75, restart=0):
    # Training lives in a different, genuinely clean Git checkout from the
    # tooling script; no mocked Git command can hide a changed source tree.
    source = tmp_path / "original-source"
    (source / "utils").mkdir(parents=True)
    (source / "utils/__init__.py").touch()
    shutil.copyfile(ROOT / "utils/oscar_resume_launcher.py", source / "utils/oscar_resume_launcher.py")
    (source / "main.py").write_text("raise AssertionError('srun must be mocked')\n")
    (source / "run.json").write_text("{}\n")
    (source / "algs").mkdir()
    (source / "algs/algorithm.json").write_text("{}\n")
    for command in (
        ["git", "init", "-q"],
        ["git", "add", "main.py", "run.json", "algs", "utils"],
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "original"],
    ):
        subprocess.run(command, cwd=source, check=True, capture_output=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(bin_dir / "checkquota", """#!/bin/sh
sleep "${FAKE_PREFLIGHT_SECONDS:-0}"
printf '%s\n' "rgao48 /oscar/scratch 1 TB 2 TB ${FAKE_QUOTA_STATE:-OK} None"
""")
    _executable(bin_dir / "srun", """#!/bin/sh
printf '%s\n' "$@" > "$FAKE_SRUN_ARGS"
printf '%s\n' "$PWD" "$AMBI_SEGMENT_ID" "$WANDB__REQUIRE_CORE" "$WANDB_DIR" \
  "${WANDB_RUN_ID:-unset}" "${WANDB_RESUME:-unset}" "$OMP_NUM_THREADS" > "$FAKE_SRUN_ENV"
if [ -n "${FAKE_RESULT:-}" ]; then cp -R "$FAKE_RESULT"/. "$AMBI_LINEAGE_DIR"/; fi
exit "$FAKE_STATUS"
""")
    _executable(bin_dir / "scontrol", """#!/bin/sh
printf '%s\n' "$*" >> "$FAKE_SCONTROL_CALLS"
exit "${FAKE_SCONTROL_STATUS:-0}"
""")
    output = tmp_path / "scratch-output"
    output.mkdir()
    lineage = output / "lineage"
    _record(lineage, segment=f"123.{restart - 1}" if restart else "oldjob.0")
    result = tmp_path / "result"
    _record(result, segment=f"123.{restart}", done=(status == 0))
    local_tmp = tmp_path / "local-tmp"
    local_tmp.mkdir()
    env = os.environ.copy()
    for name in ("SLURM_ARRAY_TASK_ID", "PYTHONPATH", "AMBI_RESUME_GENERATION"):
        env.pop(name, None)
    env.update({
        "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
        "SLURM_JOB_ID": "123", "SLURM_RESTART_COUNT": str(restart),
        "SLURM_CPUS_PER_TASK": "1", "SLURM_TMPDIR": str(local_tmp),
        "AMBI_PROJECT_DIR": str(source), "AMBI_EXPECTED_COMMIT": commit,
        "AMBI_PYTHON": sys.executable, "AMBI_OUTPUT_ROOT": str(output),
        "AMBI_LINEAGE_DIR": str(lineage), "AMBI_RUN_CONFIG": "run.json",
        "AMBI_ALG_DIR": "algs", "AMBI_QUOTA_PATH": "/oscar/scratch",
        "AMBI_DRAIN_AFTER_SECONDS": "3600", "FAKE_STATUS": str(status),
        "FAKE_RESULT": str(result), "FAKE_SRUN_ARGS": str(tmp_path / "srun-args"),
        "FAKE_SRUN_ENV": str(tmp_path / "srun-env"),
        "FAKE_SCONTROL_CALLS": str(tmp_path / "scontrol-calls"),
        "WANDB_RUN_ID": "wrong-inherited-id", "WANDB_RESUME": "never",
    })
    return env


def _run(env, launcher=LAUNCHER):
    return subprocess.run(["bash", str(launcher)], cwd=ROOT, env=env, text=True,
                          capture_output=True, check=False)


def test_spooled_tooling_launcher_preserves_original_source_and_requeues(tmp_path):
    env = _environment(tmp_path)
    spool = tmp_path / "slurm-script"
    shutil.copyfile(LAUNCHER, spool)
    result = _run(env, spool)
    assert result.returncode == 0, result.stderr
    args = Path(env["FAKE_SRUN_ARGS"]).read_text().splitlines()
    for key, value in {
        "--run": "run.json", "--alg-dir": "algs", "--num-runs": "1",
        "--lineage-dir": env["AMBI_LINEAGE_DIR"], "--resume-mode": "required",
        "--resume-wandb-mode": "online", "--resume-checkpoint-minutes": "15",
    }.items():
        assert args[args.index(key) + 1] == value
    assert args[args.index("main.py") - 1] == sys.executable
    assert 0 < int(args[args.index("--drain-after-seconds") + 1]) <= 3600
    assert Path(env["FAKE_SRUN_ENV"]).read_text().splitlines() == [
        env["AMBI_PROJECT_DIR"], "123.0", "true",
        env["AMBI_OUTPUT_ROOT"] + "/wandb/123.0", "unset", "unset", "1",
    ]
    assert "--output=" + env["AMBI_OUTPUT_ROOT"] + "/segment-console/123.0/stdout.log" in args
    assert Path(env["FAKE_SCONTROL_CALLS"]).read_text() == "requeue 123\n"
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=env["AMBI_PROJECT_DIR"])


def test_resources_are_explicit_and_signal_reaches_training():
    contents = LAUNCHER.read_text()
    assert "#SBATCH --requeue" in contents
    assert "#SBATCH --signal=USR1@1800" in contents
    for resource in ("time", "gres", "mem", "cpus-per-task", "output", "error"):
        assert f"#SBATCH --{resource}=" not in contents


def test_absolute_configuration_paths_are_preserved(tmp_path):
    env = _environment(tmp_path)
    for name in ("AMBI_RUN_CONFIG", "AMBI_ALG_DIR"):
        env[name] = str(Path(env["AMBI_PROJECT_DIR"]) / env[name])
    result = _run(env)
    assert result.returncode == 0, result.stderr
    args = Path(env["FAKE_SRUN_ARGS"]).read_text().splitlines()
    assert args[args.index("--run") + 1] == env["AMBI_RUN_CONFIG"]
    assert args[args.index("--alg-dir") + 1] == env["AMBI_ALG_DIR"]


@pytest.mark.parametrize("restart", [0, 2])
def test_success_requires_verified_done_without_requeue(tmp_path, restart):
    env = _environment(tmp_path, status=0, restart=restart)
    result = _run(env)
    assert result.returncode == 0, result.stderr
    assert "DONE verified" in result.stdout
    assert not Path(env["FAKE_SCONTROL_CALLS"]).exists()


def test_already_completed_lineage_verifies_done_without_starting(tmp_path):
    env = _environment(tmp_path)
    _record(Path(env["AMBI_LINEAGE_DIR"]), done=True)
    result = _run(env)
    assert result.returncode == 0, result.stderr
    assert not Path(env["FAKE_SRUN_ARGS"]).exists()


@pytest.mark.parametrize("status", [0, 75, 23])
def test_unverified_completion_and_failure_never_requeue(tmp_path, status):
    env = _environment(tmp_path, status=status)
    env.pop("FAKE_RESULT")
    result = _run(env)
    assert result.returncode != 0
    assert not Path(env["FAKE_SCONTROL_CALLS"]).exists()


def test_restart_checks_previous_segment_and_creates_new_identity(tmp_path):
    env = _environment(tmp_path, restart=2)
    result = _run(env)
    assert result.returncode == 0, result.stderr
    assert Path(env["FAKE_SRUN_ENV"]).read_text().splitlines()[1] == "123.2"


def test_restart_rejects_stale_prior_handoff_before_starting(tmp_path):
    env = _environment(tmp_path, restart=2)
    _record(Path(env["AMBI_LINEAGE_DIR"]), segment="123.0")
    result = _run(env)
    assert result.returncode != 0
    assert "stale segment" in result.stderr
    assert not Path(env["FAKE_SRUN_ARGS"]).exists()


@pytest.mark.parametrize("condition", ["commit", "dirty", "output", "quota", "missing-lineage"])
def test_preflight_rejects_unsafe_source_and_storage(tmp_path, condition):
    env = _environment(tmp_path)
    if condition == "commit":
        env["AMBI_EXPECTED_COMMIT"] = "0" * 40
    elif condition == "dirty":
        (Path(env["AMBI_PROJECT_DIR"]) / "main.py").write_text("changed\n")
    elif condition == "output":
        env["AMBI_OUTPUT_ROOT"] = env["AMBI_PROJECT_DIR"]
    elif condition == "quota":
        env["FAKE_QUOTA_STATE"] = "OVER"
    else:
        env["AMBI_LINEAGE_DIR"] += "-missing"
    result = _run(env)
    assert result.returncode != 0
    assert not Path(env["FAKE_SRUN_ARGS"]).exists()
    assert not Path(env["FAKE_SCONTROL_CALLS"]).exists()


def test_startup_time_consumes_drain_budget(tmp_path):
    env = _environment(tmp_path)
    env.update(AMBI_DRAIN_AFTER_SECONDS="1", FAKE_PREFLIGHT_SECONDS="2")
    result = _run(env)
    assert result.returncode != 0
    assert "expired during startup" in result.stderr
    assert not Path(env["FAKE_SRUN_ARGS"]).exists()
