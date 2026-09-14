"""Execute the production launcher against local fake Slurm/Git/CUDA commands."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm/run_ambi_prior_reward_study_hydra.sbatch"
MANIFEST = "configs/dmcontrol/experiments/ambi_prior_reward_critic_study.json"
SHA = "a" * 40
CAPTURED_ENV_KEYS = (
    "WANDB_MODE", "WANDB_DISABLE_CODE", "MUJOCO_GL",
    "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "PYTHONNOUSERSITE",
    "OMP_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS",
    "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME",
    "WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_ARTIFACT_DIR",
    "LAUNCHER_TEST_MULTILINE",
)


def _executable(path, text):
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def harness(tmp_path):
    project = tmp_path / "checkout"
    project.mkdir()
    manifest = json.loads((ROOT / MANIFEST).read_text())
    files = [MANIFEST] + [
        f"configs/dmcontrol/algs/{name}.json" for name in manifest["configs"]
    ]
    for relative in files:
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _executable(fake_bin / "git", """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "$FAKE_GIT_CALLS"
case "$*" in
  'rev-parse --show-toplevel') printf '%s\\n' "$FAKE_GIT_ROOT" ;;
  'rev-parse HEAD') printf '%s\\n' "$FAKE_GIT_SHA" ;;
  'status --porcelain --untracked-files=all') printf '%s' "${FAKE_GIT_STATUS:-}" ;;
  *) exit 17 ;;
esac
""")
    _executable(fake_bin / "df", """#!/usr/bin/env bash
set -eu
printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'
printf 'fake 100000000 0 %s 0%% /\\n' "$FAKE_FREE_KB"
""")
    _executable(fake_bin / "srun", """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$@" > "$FAKE_SRUN_ARGS"
# Record only fields checked by this fixture, never the caller's full environment.
# NUL delimiters preserve multiline values (including exported module functions).
: > "$FAKE_RUN_ENV"
for key in CAPTURED_ENV_KEYS_PLACEHOLDER; do
  printf '%s=%s\\0' "$key" "${!key-}" >> "$FAKE_RUN_ENV"
done
exit "${FAKE_SRUN_EXIT:-0}"
""".replace("CAPTURED_ENV_KEYS_PLACEHOLDER", " ".join(CAPTURED_ENV_KEYS)))
    python = project / "environments/dmcontrol/.venv/bin/python"
    python.parent.mkdir(parents=True)
    _executable(python, """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$@" >> "$FAKE_PYTHON_CALLS"
exit "${FAKE_PYTHON_EXIT:-0}"
""")
    local_tmp = tmp_path / "node-local"
    local_tmp.mkdir()
    env = os.environ.copy()
    # Do not inherit a caller's launcher overrides or fake-command settings.
    for key in list(env):
        if key.startswith(("AMBI_", "SLURM_", "FAKE_")):
            env.pop(key)
    env.update({
        "PATH": str(fake_bin) + os.pathsep + env["PATH"],
        "SLURM_JOB_ID": "1234_0", "SLURM_ARRAY_JOB_ID": "1234",
        "SLURM_ARRAY_TASK_ID": "0", "SLURM_SUBMIT_DIR": str(project),
        "SLURM_NODELIST": "fake-l40s", "SLURM_TMPDIR": str(local_tmp),
        "SLURM_CPUS_PER_TASK": "8", "AMBI_EXPECTED_COMMIT": SHA,
        "AMBI_STUDY_OUTPUT_ROOT": str(tmp_path / "output"),
        "AMBI_DMCONTROL_PYTHON": str(python),
        "FAKE_GIT_ROOT": str(project), "FAKE_GIT_SHA": SHA,
        "FAKE_FREE_KB": str(60 * 1024 * 1024),
        "FAKE_GIT_CALLS": str(tmp_path / "git-calls"),
        "FAKE_PYTHON_CALLS": str(tmp_path / "python-calls"),
        "FAKE_SRUN_ARGS": str(tmp_path / "srun-args"),
        "FAKE_RUN_ENV": str(tmp_path / "run-env"),
    })
    return {"env": env, "project": project, "python": python,
            "configs": manifest["configs"], "tmp": tmp_path}


def _run(harness):
    # The launcher itself enters SLURM_SUBMIT_DIR. Avoid a redundant cwd and
    # allow posix_spawn, so collecting PyTorch tests in the same process does
    # not invoke macOS OpenMP's unsafe post-fork initialization before exec.
    return subprocess.run(
        ["/bin/bash", str(LAUNCHER)], env=harness["env"],
        capture_output=True, text=True, check=False, close_fds=False,
    )


def _after(arguments, option):
    return arguments[arguments.index(option) + 1]


def _captured_env(path):
    raw = Path(path).read_bytes()
    assert raw.endswith(b"\0")
    return dict(record.decode("utf-8").split("=", 1)
                for record in raw[:-1].split(b"\0"))


def test_hydra_launcher_scheduler_contract():
    source = LAUNCHER.read_text()
    for directive in (
        "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--partition=gpus",
        "--constraint=l40s", "--gres=gpu:1", "--mem=32G",
        "--time=7-00:00:00", "--array=0-5", "--no-requeue",
    ):
        assert f"#SBATCH {directive}\n" in source
    assert "#SBATCH --requeue" not in source
    assert "#SBATCH --nodelist" not in source
    assert "#SBATCH --array=0-5%" not in source
    syntax = subprocess.run(
        ["/bin/bash", "-n", str(LAUNCHER)], capture_output=True, close_fds=False,
    )
    assert syntax.returncode == 0, syntax.stderr


@pytest.mark.parametrize("task", range(6))
def test_hydra_launcher_maps_one_seeded_cell_and_isolates_outputs(harness, task):
    env = harness["env"]
    env.update(SLURM_ARRAY_TASK_ID=str(task), SLURM_JOB_ID=f"1234_{task}")
    result = _run(harness)
    assert result.returncode == 0, result.stderr
    arguments = Path(env["FAKE_SRUN_ARGS"]).read_text().splitlines()
    alg_index, trial_index = task, 0
    config = harness["configs"][alg_index]
    seed = 55 + trial_index
    output = Path(env["AMBI_STUDY_OUTPUT_ROOT"]) / config / f"seed{seed}" / f"job_1234_{task}"
    assert arguments[:3] == ["--kill-on-bad-exit=1", str(harness["python"]), "main.py"]
    assert _after(arguments, "--run") == MANIFEST
    assert _after(arguments, "--alg-dir") == "configs/dmcontrol/algs"
    assert _after(arguments, "--alg-index") == str(alg_index)
    assert _after(arguments, "--trial-index") == str(trial_index)
    assert _after(arguments, "--num-runs") == "1"
    assert _after(arguments, "--log-dir") == str(output)
    assert output.is_dir()
    assert f"Configuration: {config}; seed: {seed}; decisions: 2000000" in result.stdout
    assert f"Source commit: {SHA}" in result.stdout
    assert Path(env["FAKE_GIT_CALLS"]).read_text().splitlines() == [
        "rev-parse --show-toplevel", "status --porcelain --untracked-files=all", "rev-parse HEAD",
    ]
    python_calls = Path(env["FAKE_PYTHON_CALLS"]).read_text().splitlines()
    assert len(python_calls) == 2 and python_calls[0] == "-c"
    assert "torch.cuda.get_device_name()" in python_calls[1]
    actual_env = _captured_env(env["FAKE_RUN_ENV"])
    for key, value in {
        "WANDB_MODE": "online", "WANDB_DISABLE_CODE": "true", "MUJOCO_GL": "egl",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "8", "TORCHINDUCTOR_COMPILE_THREADS": "1",
    }.items():
        assert actual_env[key] == value
    cache_parents = set()
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "XDG_CACHE_HOME",
                "WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_ARTIFACT_DIR"):
        path = Path(actual_env[key])
        assert path.is_dir() and path.is_relative_to(Path(env["SLURM_TMPDIR"]))
        assert not path.is_relative_to(harness["project"])
        assert not path.is_relative_to(output)
        cache_parents.add(path.parent)
    assert len(cache_parents) == 1


@pytest.mark.parametrize("key", [
    "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
    "AMBI_EXPECTED_COMMIT", "AMBI_STUDY_OUTPUT_ROOT", "AMBI_DMCONTROL_PYTHON",
])
def test_hydra_launcher_requires_launch_identity_before_creating_outputs(harness, key):
    output = Path(harness["env"]["AMBI_STUDY_OUTPUT_ROOT"])
    harness["env"].pop(key)
    result = _run(harness)
    assert result.returncode != 0 and key in result.stderr
    assert not output.exists()
    assert not Path(harness["env"]["FAKE_PYTHON_CALLS"]).exists()
    assert not Path(harness["env"]["FAKE_SRUN_ARGS"]).exists()


@pytest.mark.parametrize("setting,value,error", [
    ("SLURM_ARRAY_TASK_ID", "8", "array task must be in 0..5"),
    ("SLURM_ARRAY_TASK_ID", "-1", "array task must be in 0..5"),
    ("SLURM_ARRAY_TASK_ID", "00", "array task must be in 0..5"),
    ("SLURM_ARRAY_TASK_ID", "1+1", "array task must be in 0..5"),
    ("AMBI_EXPECTED_COMMIT", "abc", "invalid expected commit SHA"),
    ("SLURM_RESTART_COUNT", "1", "cannot resume after requeue"),
    ("AMBI_STUDY_OUTPUT_ROOT", "relative-output", "output root must be absolute"),
    ("FAKE_GIT_STATUS", " M main.py\n", "checkout is dirty"),
    ("FAKE_GIT_STATUS", "?? surprise.py\n", "checkout is dirty"),
    ("FAKE_GIT_SHA", "b" * 40, "HEAD differs from the intended commit"),
    ("AMBI_DMCONTROL_PYTHON", "/nonexistent/locked/python", "missing locked DMControl interpreter"),
    ("FAKE_FREE_KB", str(6 * 1024 * 1024 - 1), "less than 6 GiB free"),
    ("FAKE_FREE_KB", "unknown", "could not determine output space"),
])
def test_hydra_launcher_rejects_unsafe_or_mismatched_launches(harness, setting, value, error):
    harness["env"][setting] = value
    result = _run(harness)
    assert result.returncode != 0 and error in result.stderr
    assert not Path(harness["env"]["FAKE_PYTHON_CALLS"]).exists()
    assert not Path(harness["env"]["FAKE_SRUN_ARGS"]).exists()


@pytest.mark.parametrize("case,error", [
    ("nonroot", "submit from the repository root"),
    ("missing-config", "missing study configuration"),
    ("missing-manifest", "missing study configuration"),
    ("source-output", "checkpoint output must be outside the source checkout"),
    ("symlink-output", "checkpoint output must be outside the source checkout"),
])
def test_hydra_launcher_validates_real_paths_and_study_files(harness, case, error):
    project = harness["project"]
    if case == "nonroot":
        harness["env"]["FAKE_GIT_ROOT"] = str(project.parent)
    elif case == "missing-config":
        (project / f"configs/dmcontrol/algs/{harness['configs'][0]}.json").unlink()
    elif case == "missing-manifest":
        (project / MANIFEST).unlink()
    elif case == "source-output":
        harness["env"]["AMBI_STUDY_OUTPUT_ROOT"] = str(project / "outputs")
    else:
        output = harness["tmp"] / "link-output"
        output.symlink_to(project, target_is_directory=True)
        harness["env"]["AMBI_STUDY_OUTPUT_ROOT"] = str(output)
    result = _run(harness)
    assert result.returncode != 0 and error in result.stderr
    assert not Path(harness["env"]["FAKE_PYTHON_CALLS"]).exists()
    assert not Path(harness["env"]["FAKE_SRUN_ARGS"]).exists()


def test_hydra_launcher_refuses_duplicate_output_without_reexecuting(harness):
    assert _run(harness).returncode == 0
    calls = Path(harness["env"]["FAKE_PYTHON_CALLS"])
    before = calls.read_bytes()
    result = _run(harness)
    assert result.returncode != 0 and "run output already exists" in result.stderr
    assert calls.read_bytes() == before


@pytest.mark.parametrize("process", ["PYTHON", "SRUN"])
def test_hydra_launcher_propagates_runtime_failure(harness, process):
    harness["env"][f"FAKE_{process}_EXIT"] = "19"
    result = _run(harness)
    assert result.returncode == 19
    if process == "PYTHON":
        assert not Path(harness["env"]["FAKE_SRUN_ARGS"]).exists()


def test_postlaunch_smoke_is_independent_and_requires_live_cuda():
    smoke = ROOT / "slurm/validate_ambi_prior_reward_hydra.sbatch"
    source = smoke.read_text()
    for directive in ("--partition=gpus", "--constraint=l40s", "--gres=gpu:1",
                      "--cpus-per-task=8", "--mem=32G", "--time=01:00:00", "--no-requeue"):
        assert f"#SBATCH {directive}\n" in source
    assert "--dependency" not in LAUNCHER.read_text()
    assert "assert torch.cuda.is_available()" in source
    assert "tests/test_ambi_prior_reward_cuda_gate.py" in source
    assert "WANDB_MODE=offline" in source
    syntax = subprocess.run(["/bin/bash", "-n", str(smoke)], capture_output=True, close_fds=False)
    assert syntax.returncode == 0, syntax.stderr


def test_fake_srun_preserves_multiline_values_without_dumping_other_environment(harness):
    value = "first=one\nsecond line without equals\nthird=two=three\r\n"
    harness["env"]["LAUNCHER_TEST_MULTILINE"] = value
    harness["env"]["UNRELATED_INHERITED_VALUE"] = "not captured"
    result = _run(harness)
    assert result.returncode == 0, result.stderr
    actual_env = _captured_env(harness["env"]["FAKE_RUN_ENV"])
    assert actual_env["LAUNCHER_TEST_MULTILINE"] == value
    assert actual_env.keys() == set(CAPTURED_ENV_KEYS)
    assert "UNRELATED_INHERITED_VALUE" not in actual_env
