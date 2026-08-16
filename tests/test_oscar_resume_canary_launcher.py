import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_ambi_oscar_canary.sh"


def _executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _environment(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    srun_calls = tmp_path / "srun-calls"
    python_calls = tmp_path / "python-calls"
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
    _executable(
        fake_bin / "checkquota",
        "#!/bin/sh\nprintf '%s\n' 'data+rbalestr 1 TB 2 TB OK'\n",
    )
    _executable(
        fake_bin / "python",
        """#!/bin/sh
printf '%s\n' "$*" >> "$FAKE_PYTHON_CALLS"
case "$*" in
  *"utils.oscar_resume_launcher quota"*) exit 0 ;;
  *"utils.oscar_resume_launcher storage"*) printf '%s\n' '{"total_seconds":0.1}' ;;
  *"utils.oscar_resume_launcher handoff"*) exit 0 ;;
  *"utils.oscar_resume_canary verify-lineage"*) exit "${FAKE_VERIFY_STATUS:-0}" ;;
  *) exit 91 ;;
esac
""",
    )
    _executable(
        fake_bin / "srun",
        """#!/bin/sh
{
  printf '%s\n' "BEGIN:${AMBI_SEGMENT_ID:-benchmark}"
  printf '%s\n' "$@"
  printf '%s\n' END
} >> "$FAKE_SRUN_CALLS"
case "$*" in
  *"utils.oscar_resume_canary"*)
    previous=''
    for argument in "$@"; do
      if [ "$previous" = '--output' ]; then
        printf '%s\n' '{"verified":true}' > "$argument"
      fi
      previous="$argument"
    done
    exit 0
    ;;
esac
exit "${FAKE_SEGMENT_STATUS:-75}"
""",
    )
    mamba = tmp_path / "mamba"
    profile = mamba / "etc" / "profile.d"
    profile.mkdir(parents=True)
    (profile / "conda.sh").write_text("conda() { return 0; }\n", encoding="utf-8")
    durable = tmp_path / "durable"
    durable.mkdir()
    lineage = durable / "lineage"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "SLURM_JOB_ID": "24680",
            "SLURM_RESTART_COUNT": "0",
            "SLURM_SUBMIT_DIR": str(ROOT),
            "AMBI_RUN_OSCAR_RESUME_CANARY": "1",
            "AMBI_DURABLE_ROOT": str(durable),
            "AMBI_LINEAGE_DIR": str(lineage),
            "AMBI_PYTHON": str(fake_bin / "python"),
            "MAMBA_ROOT_PREFIX": str(mamba),
            "FAKE_GIT_ROOT": str(ROOT),
            "FAKE_SRUN_CALLS": str(srun_calls),
            "FAKE_PYTHON_CALLS": str(python_calls),
        }
    )
    return env, lineage, srun_calls, python_calls


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


def _calls(path: Path):
    result = []
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("BEGIN:"):
            current = [line]
        elif line == "END":
            result.append(current)
            current = None
        else:
            current.append(line)
    return result


def test_canary_has_bounded_gpu_debug_resources_and_no_requeue():
    contents = LAUNCHER.read_text(encoding="utf-8")
    for directive in (
        "#SBATCH --partition=gpu-debug",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --time=01:00:00",
        "#SBATCH --mem=32G",
        "#SBATCH --cpus-per-task=6",
    ):
        assert directive in contents
    assert "#SBATCH --requeue" not in contents
    assert "scontrol" not in contents
    assert "AMBI_WANDB_REWIND_VERIFIED" not in contents
    assert "utils.oscar_resume_canary verify-lineage" in contents
    assert "utils.oscar_resume_canary benchmark-replay" in contents
    assert "--shard-rows 100000" in contents
    assert "--maximum-estimated-bytes 4000000000" in contents


def test_canary_runs_new_then_required_and_real_replay_benchmark(tmp_path):
    env, _, srun_path, python_path = _environment(tmp_path)
    env["AMBI_DURABLE_QUOTA_PATH"] = "/oscar/home"
    result = _run(env)
    assert result.returncode == 0, result.stderr
    calls = _calls(srun_path)
    assert len(calls) == 3
    assert calls[0][0] == "BEGIN:24680.canary.0"
    assert calls[1][0] == "BEGIN:24680.canary.1"
    for call, mode in zip(calls[:2], ("new", "required"), strict=True):
        assert call[call.index("--resume-mode") + 1] == mode
        assert call[call.index("--resume-wandb-mode") + 1] == "online"
        assert call[call.index("--drain-after-seconds") + 1] == "300"
        assert call[call.index("--run") + 1].endswith(
            "AntAMBITDMPC2ResumeCanary.json"
        )
        assert call[call.index("--alg-dir") + 1].endswith("configs/algs")
    assert "utils.oscar_resume_canary" in calls[2]
    assert "benchmark-replay" in calls[2]
    assert calls[2][calls[2].index("--run") + 1].endswith("ambi_anchor.json")
    assert calls[2][calls[2].index("--algorithm") + 1].endswith("ambi_anchor.json")
    python_calls = python_path.read_text(encoding="utf-8").splitlines()
    quota_call = next(
        call for call in python_calls if "utils.oscar_resume_launcher quota" in call
    )
    assert "--filesystem-path /oscar/home" in quota_call
    assert sum("utils.oscar_resume_launcher handoff" in call for call in python_calls) == 2
    assert sum("utils.oscar_resume_canary verify-lineage" in call for call in python_calls) == 1
    assert any("utils.oscar_resume_launcher storage" in call for call in python_calls)
    result_path = (
        Path(env["AMBI_DURABLE_ROOT"])
        / "resume-canary"
        / "24680"
        / "REPLAY_BENCHMARK.json"
    )
    assert result_path.read_text(encoding="utf-8").strip() == '{"verified":true}'


def test_canary_fails_before_srun_without_authorization_or_clean_checkout(tmp_path):
    env, _, srun_path, _ = _environment(tmp_path)
    del env["AMBI_RUN_OSCAR_RESUME_CANARY"]
    result = _run(env)
    assert result.returncode != 0
    assert not srun_path.exists()

    dirty_tmp = tmp_path / "dirty"
    dirty_tmp.mkdir()
    dirty_env, _, dirty_srun, _ = _environment(dirty_tmp)
    dirty_env["FAKE_GIT_STATUS"] = " M main.py"
    dirty = _run(dirty_env)
    assert dirty.returncode != 0
    assert "clean checkout" in dirty.stderr
    assert not dirty_srun.exists()


def test_canary_rejects_existing_lineage_and_segment_failure(tmp_path):
    env, lineage, srun_path, _ = _environment(tmp_path)
    lineage.mkdir()
    result = _run(env)
    assert result.returncode != 0
    assert "must not exist" in result.stderr
    assert not srun_path.exists()

    failure_tmp = tmp_path / "failure"
    failure_tmp.mkdir()
    failure_env, _, failure_srun, _ = _environment(failure_tmp)
    failure_env["FAKE_SEGMENT_STATUS"] = "23"
    failure = _run(failure_env)
    assert failure.returncode != 0
    assert "instead of clean handoff 75" in failure.stderr
    assert len(_calls(failure_srun)) == 1


def test_canary_stops_before_replay_when_resumed_progress_is_unverified(tmp_path):
    env, _, srun_path, _ = _environment(tmp_path)
    env["FAKE_VERIFY_STATUS"] = "23"
    result = _run(env)
    assert result.returncode != 0
    calls = _calls(srun_path)
    assert len(calls) == 2
    assert all("benchmark-replay" not in call for call in calls)
