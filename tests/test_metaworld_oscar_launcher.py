"""Exercise array restarts and Slurm handoff routing without starting a learner."""
import os
from pathlib import Path
import subprocess
import time

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "slurm/run_metaworld_oscar.sbatch"


@pytest.fixture
def launch(tmp_path):
    tools = tmp_path / "bin"
    tools.mkdir()
    invocations = tmp_path / "invocations"
    root = tmp_path / "runs"
    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=test", "-c",
                    "user.email=test@example.invalid", "commit", "-qm", "initial",
                    "--allow-empty"], check=True)
    sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    for name, body in {
        "flock": 'exit "${LOCK_STATUS:-0}"',
        "realpath": 'test "$1" = -m && echo "$2"',  # GNU option on Oscar
        "srun": 'echo "train $*" >> "$CALLS"; exit "$TRAIN_STATUS"',
        "scontrol": 'echo "scheduler $*" >> "$CALLS"',
        "runtime": 'echo "verify $*" >> "$CALLS"; '
                   'if [[ "$3" = handoff ]]; then exit "${HANDOFF_STATUS:-0}"; fi',
    }.items():
        command = tools / name
        command.write_text("#!/bin/bash\n" + body + "\n")
        command.chmod(0o755)

    def run(**changes):
        if invocations.exists():
            invocations.unlink()
        env = dict(os.environ, PATH=f"{tools}:{os.environ['PATH']}", CALLS=str(invocations),
                   AMBI_PROJECT_DIR=str(repo), AMBI_EXPECTED_COMMIT=sha,
                   AMBI_METAWORLD_PYTHON=str(tools / "runtime"), AMBI_METAWORLD_ROOT=str(root),
                   SLURM_ARRAY_JOB_ID="741", SLURM_ARRAY_TASK_ID="3", SLURM_JOB_ID="749",
                   SLURM_TMPDIR=str(tmp_path), SLURM_JOB_END_TIME=str(int(time.time()) + 7200),
                   TRAIN_STATUS="75")
        env.update(changes)
        result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
        return result.returncode, invocations.read_text() if invocations.exists() else ""
    return run, root / "stick-pull/ambi_state/seed_55/lineage"


def test_verified_handoff_requeues_only_the_correct_array_element(launch):
    run, lineage = launch
    status, calls = run()
    assert status == 0
    assert f"--lineage-dir {lineage} --resume-mode new" in calls
    assert "--run configs/metaworld/experiments/stick-pull.json" in calls
    assert "--resume-checkpoint-steps 25000" in calls
    assert "handoff" in calls and "--segment-id slurm-749-0" in calls
    assert calls.endswith("scheduler requeue 741_3\n")
    lineage.mkdir()
    (lineage / "LATEST").write_text("committed")
    status, calls = run(SLURM_RESTART_COUNT="1", TRAIN_STATUS="0")
    assert status == 0 and "--resume-mode required" in calls and " done " in calls
    assert "scheduler requeue" not in calls
    (lineage / "DONE").touch()
    status, calls = run(SLURM_RESTART_COUNT="2")
    assert status == 0 and " done " in calls and "train " not in calls


@pytest.mark.parametrize("overrides", [{"HANDOFF_STATUS": "2"}, {"TRAIN_STATUS": "6"},
                                       {"LOCK_STATUS": "1"}])
def test_failed_handoff_or_training_never_requeues(launch, overrides):
    run, _ = launch
    status, calls = run(**overrides)
    assert status != 0 and "scheduler" not in calls


def test_missing_genesis_refuses_to_restart_in_place(launch):
    run, lineage = launch
    lineage.mkdir(parents=True)
    status, calls = run()
    assert status != 0 and "train " not in calls
