import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm/run_td_ambi_prior_study_oscar.sbatch"
CONFIGS = (
    "td_ambi_prior_reward_qscale",
    "td_ambi_prior_entropy_qscale",
    "td_ambi_prior_entropy_autotemp",
    "td_ambi_prior_reward_autotemp",
)


def _executable(path, text):
    path.write_text(text)
    path.chmod(0o755)


@pytest.fixture
def launch_env(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _executable(fake_bin / "git", """#!/usr/bin/env bash
if [[ "$1 $2" == 'rev-parse --show-toplevel' ]]; then
  printf '%s\n' "$SLURM_SUBMIT_DIR"
elif [[ "$1 $2" == 'rev-parse HEAD' ]]; then
  printf '%s\n' "$FAKE_SHA"
elif [[ "$1" == status ]]; then
  printf '%s' "${FAKE_DIRTY:-}"
else
  exit 2
fi
""")
    _executable(fake_bin / "checkquota", "#!/bin/sh\nprintf 'rgao48 /oscar/scratch 567 GB 1 TB OK None\n'\n")
    _executable(fake_bin / "df", "#!/bin/sh\nprintf 'Filesystem 1024-blocks Used Available Capacity Mounted\nscratch 209715200 0 209715200 0%% /scratch\n'\n")
    _executable(fake_bin / "python", "#!/bin/sh\nexit 0\n")
    _executable(fake_bin / "srun", """#!/usr/bin/env bash
printf '%s\n' "$@" > "$FAKE_SRUN_ARGS"
printf '%s\n' "$WANDB_DIR" "$WANDB_MODE" "$TORCHINDUCTOR_CACHE_DIR" > "$FAKE_RUNTIME_ENV"
""")
    env = os.environ.copy()
    env.update({
        "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
        "SLURM_SUBMIT_DIR": str(ROOT),
        "SLURM_JOB_ID": "1234_0",
        "SLURM_ARRAY_JOB_ID": "1234",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_RESTART_COUNT": "0",
        "SLURM_TMPDIR": str(tmp_path / "local"),
        "EXPECTED_ACTION_MODES_SHA": "a" * 40,
        "FAKE_SHA": "a" * 40,
        "AMBI_DMC_PYTHON": str(fake_bin / "python"),
        "TD_AMBI_OUTPUT_ROOT": str(tmp_path / "scratch"),
        "TD_AMBI_CAMPAIGN": "test-campaign",
        "FAKE_SRUN_ARGS": str(tmp_path / "args"),
        "FAKE_RUNTIME_ENV": str(tmp_path / "runtime"),
    })
    return env


def _run(env):
    return subprocess.run(["bash", str(LAUNCHER)], cwd=ROOT, env=env,
                          text=True, capture_output=True, check=False)


@pytest.mark.parametrize("index,config", enumerate(CONFIGS))
def test_prior_array_runs_one_complete_cell_on_scratch(launch_env, index, config):
    launch_env["SLURM_ARRAY_TASK_ID"] = str(index)
    result = _run(launch_env)
    assert result.returncode == 0, result.stderr
    args = Path(launch_env["FAKE_SRUN_ARGS"]).read_text().splitlines()
    def value(option):
        return args[args.index(option) + 1]
    assert value("--run") == f"configs/dmcontrol/experiments/{config}.json"
    assert value("--alg-dir") == "configs/dmcontrol/algs"
    assert value("--log-dir") == str(Path(launch_env["TD_AMBI_OUTPUT_ROOT"]) / "test-campaign" / f"task_{index}")
    assert value("--num-runs") == "1"
    assert value("--alg-index") == value("--trial-index") == "0"
    assert "--lineage-dir" not in args and "--resume-mode" not in args
    runtime = Path(launch_env["FAKE_RUNTIME_ENV"]).read_text().splitlines()
    assert runtime[0].startswith(launch_env["SLURM_TMPDIR"])
    assert runtime[1] == "online"
    assert runtime[2].startswith(launch_env["SLURM_TMPDIR"])
    manifest = json.loads((ROOT / value("--run")).read_text())
    algorithm = json.loads((ROOT / f"configs/dmcontrol/algs/{config}.json").read_text())
    assert manifest["configs"] == [config] and manifest["trials"] == 1
    assert manifest["overrides_alg"]["total_steps"] == algorithm["total_steps"] == 2_000_000
    assert manifest["checkpoint_every"] == algorithm["checkpoint_every"] == 25_000
    assert manifest["save_strat"] == algorithm["save_strat"] == "all"
    assert algorithm["alg_params"]["inner_operator"] == "none"


@pytest.mark.parametrize("overrides", [
    {"FAKE_SHA": "b" * 40},
    {"FAKE_DIRTY": " M main.py"},
    {"SLURM_RESTART_COUNT": "1"},
    {"SLURM_ARRAY_TASK_ID": "4"},
    {"TD_AMBI_CAMPAIGN": "../wrong-place"},
])
def test_prior_array_rejects_wrong_or_duplicate_execution(launch_env, overrides):
    launch_env.update(overrides)
    result = _run(launch_env)
    assert result.returncode != 0
    assert not Path(launch_env["FAKE_SRUN_ARGS"]).exists()


def test_prior_array_does_not_overwrite_an_existing_cell(launch_env):
    root = Path(launch_env["TD_AMBI_OUTPUT_ROOT"]) / "test-campaign" / "task_0"
    root.mkdir(parents=True)
    sentinel = root / "checkpoint.pt"
    sentinel.write_bytes(b"keep")
    result = _run(launch_env)
    assert result.returncode != 0
    assert not Path(launch_env["FAKE_SRUN_ARGS"]).exists()
    assert sentinel.read_bytes() == b"keep"
