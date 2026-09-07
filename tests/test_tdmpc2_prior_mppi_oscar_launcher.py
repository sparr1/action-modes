import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest
import publish_tdmpc2_mppi_eval as publisher


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm/run_tdmpc2_prior_mppi_eval_oscar.sbatch"
ALGORITHM = ROOT / "configs/dmcontrol/algs/tdmpc2_humanoid_walk_state_prior_only_checkpoint_bank_1p5m.json"
EXPERIMENT = ROOT / "configs/dmcontrol/experiments/tdmpc2_humanoid_walk_state_prior_only_checkpoint_bank_1p5m.json"


@pytest.fixture
def launch_input(tmp_path):
    rows = []
    for step in (450000, 500000, 600000, 750000, 1000000, 1150000, 1500000):
        checkpoint = tmp_path / f"checkpoint {step}"
        checkpoint.write_bytes(f"immutable checkpoint {step}".encode())
        metadata = {
            "checkpoint": {"step": step},
            "trial_run_params": json.loads(ALGORITHM.read_text()),
            "experiment_params": json.loads(EXPERIMENT.read_text()),
        }
        sidecar = Path(str(checkpoint) + ".metadata.json")
        sidecar.write_text(json.dumps(metadata))
        rows.append({"step": step, "path": str(checkpoint),
                     "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                     "metadata_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest()})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"checkpoints": rows}))
    binaries = tmp_path / "bin"
    binaries.mkdir()
    git = binaries / "git"
    git.write_text('#!/usr/bin/env bash\nif [[ "$1" == status ]]; then exit 0; fi\n'
                   f'exec {shlex.quote(shutil.which("git"))} "$@"\n')
    git.chmod(0o755)
    python = binaries / "python"
    python.write_text(f"#!{sys.executable}\n" + """
import json, os, sys
from pathlib import Path
if sys.argv[1] == '-':
    os.execv(sys.executable, [sys.executable, '-'])
with open(os.environ['CALL_LOG'], 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1] == 'evaluate_tdmpc2_mppi_checkpoint.py':
    Path(sys.argv[sys.argv.index('--output') + 1]).write_text('{}')
elif sys.argv[1] == 'publish_tdmpc2_mppi_eval.py':
    assert Path(sys.argv[2]).is_file()
else:
    raise SystemExit('Unexpected command')
""")
    python.chmod(0o755)
    env = os.environ.copy()
    env.update({
        "PATH": str(binaries) + os.pathsep + env["PATH"],
        "PYTHON_BIN": str(python), "SLURM_SUBMIT_DIR": str(ROOT),
        "SLURM_ARRAY_TASK_ID": "0", "SLURM_ARRAY_JOB_ID": "test-array",
        "SLURM_JOB_ID": "test-job", "SLURM_TMPDIR": str(tmp_path),
        "CHECKPOINT_MANIFEST": str(manifest), "RESULT_ROOT": str(tmp_path / "results"),
        "CAMPAIGN": "prior-mppi-test", "WANDB_MODE": "disabled",
        "EXPECTED_ACTION_MODES_SHA": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "CALL_LOG": str(tmp_path / "calls.jsonl"),
    })
    env.pop("EPISODES", None)
    env.pop("MAX_STEPS", None)
    env.pop("EXPECTED_MAX_STEP", None)
    return env, rows


def _run(env):
    return subprocess.run(["bash", str(LAUNCHER)], env=env, text=True, capture_output=True)


@pytest.mark.parametrize("index", range(7))
def test_oscar_launches_exact_checkpoint_and_paired_protocol(launch_input, index):
    env, rows = launch_input
    env["SLURM_ARRAY_TASK_ID"] = str(index)
    result = _run(env)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in Path(env["CALL_LOG"]).read_text().splitlines()]
    assert len(calls) == 2
    evaluation, publish = calls
    assert evaluation[:2] == ["evaluate_tdmpc2_mppi_checkpoint.py", rows[index]["path"]]
    options = dict(zip(evaluation[2::2], evaluation[3::2]))
    assert options == {
        "--output": str(Path(env["RESULT_ROOT"]) / f"step_{rows[index]['step']}" / "paired.json"),
        "--episodes": "5", "--max-steps": "500", "--seed": "101",
        "--controller-seed": "12345", "--bootstrap-samples": "20000", "--device": "cuda",
    }
    assert publish == [
        "publish_tdmpc2_mppi_eval.py", options["--output"], "--project", "ambi-inner-bench",
        "--source-run", "rwgao_b-brown-university/ambi/xq3zva9u", "--campaign", "prior-mppi-test",
        "--expected-max-step", "1500000",
    ]
    provenance = json.loads(Path(options["--output"]).with_name("provenance.json").read_text())
    code_sha = env["EXPECTED_ACTION_MODES_SHA"]
    code_tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT).decode().strip()
    assert provenance == {
        "code_sha": code_sha, "code_tree": code_tree, "campaign": "prior-mppi-test",
        "evaluation_source_sha256": publisher.evaluation_source_fingerprint(code_sha, code_tree),
        "source_run": "rwgao_b-brown-university/ambi/xq3zva9u",
        "checkpoint_sha256": rows[index]["sha256"], "metadata_sha256": rows[index]["metadata_sha256"],
    }


def test_oscar_smoke_only_changes_episode_length_and_count(launch_input):
    env, _ = launch_input
    env.update(EPISODES="1", MAX_STEPS="3")
    result = _run(env)
    assert result.returncode == 0, result.stderr
    evaluation = json.loads(Path(env["CALL_LOG"]).read_text().splitlines()[0])
    assert evaluation[evaluation.index("--episodes") + 1] == "1"
    assert evaluation[evaluation.index("--max-steps") + 1] == "3"


@pytest.mark.parametrize("failure", ["hash", "metadata_hash", "planner", "step", "overwrite", "source", "index", "order"])
def test_oscar_rejects_incompatible_inputs_before_evaluation(launch_input, failure):
    env, rows = launch_input
    checkpoint = Path(rows[0]["path"])
    sidecar = Path(str(checkpoint) + ".metadata.json")
    metadata = json.loads(sidecar.read_text())
    if failure == "hash":
        checkpoint.write_bytes(b"replaced")
    elif failure == "metadata_hash":
        sidecar.write_text(sidecar.read_text() + "\n")
    elif failure == "planner":
        metadata["trial_run_params"]["alg_params"]["num_pi_trajs"] = 512
        sidecar.write_text(json.dumps(metadata))
    elif failure == "step":
        metadata["checkpoint"]["step"] = 125000
        sidecar.write_text(json.dumps(metadata))
    elif failure == "overwrite":
        output = Path(env["RESULT_ROOT"]) / f"step_{rows[0]['step']}" / "paired.json"
        output.parent.mkdir(parents=True)
        output.write_text("preserve this result")
    elif failure == "source":
        env["EXPECTED_ACTION_MODES_SHA"] = "different-sha"
    elif failure == "index":
        env["SLURM_ARRAY_TASK_ID"] = "-1"
    elif failure == "order":
        changed = copy.deepcopy(rows)
        changed[0], changed[1] = changed[1], changed[0]
        Path(env["CHECKPOINT_MANIFEST"]).write_text(json.dumps({"checkpoints": changed}))
    if failure in {"planner", "step"}:
        rows[0]["metadata_sha256"] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
        Path(env["CHECKPOINT_MANIFEST"]).write_text(json.dumps({"checkpoints": rows}))
    result = _run(env)
    assert result.returncode != 0
    assert not Path(env["CALL_LOG"]).exists()
    if failure == "overwrite":
        assert output.read_text() == "preserve this result"


def test_oscar_launcher_requests_bounded_resources():
    contents = LAUNCHER.read_text()
    for directive in ("--partition=gpu", "--qos=pri-gpu+", "--gres=gpu:l40s:1",
                      "--cpus-per-task=6", "--mem=32G", "--time=00:15:00",
                      "--array=0", "--no-requeue"):
        assert f"#SBATCH {directive}" in contents
    assert "#SBATCH --account" not in contents
    assert "evaluate_tdmpc2_mppi_action_mc.py" not in contents


def test_oscar_extends_existing_campaign_to_explicit_maximum(launch_input):
    env, rows = launch_input
    Path(env["CHECKPOINT_MANIFEST"]).write_text(json.dumps({"checkpoints": rows[:-1]}))
    env.update(EXPECTED_MAX_STEP="1150000", SLURM_ARRAY_TASK_ID="5")
    result = _run(env)
    assert result.returncode == 0, result.stderr
    publish = json.loads(Path(env["CALL_LOG"]).read_text().splitlines()[1])
    assert publish[-2:] == ["--expected-max-step", "1150000"]
    assert Path(env["RESULT_ROOT"], "step_1150000", "paired.json").is_file()


@pytest.mark.parametrize("steps", [[], [450000, 450000], [450001], [425000], [0], [1550000], [True]])
def test_oscar_rejects_invalid_manifest_grid(launch_input, steps):
    env, rows = launch_input
    changed = [{**rows[0], "step": step} for step in steps]
    Path(env["CHECKPOINT_MANIFEST"]).write_text(json.dumps({"checkpoints": changed}))
    assert _run(env).returncode != 0
    assert not Path(env["CALL_LOG"]).exists()


def test_online_launch_requires_explicit_prior_and_mppi_runs(launch_input):
    env, _ = launch_input
    env["WANDB_MODE"] = "online"
    env.pop("EVAL_RUN_MAP", None)
    result = _run(env)
    assert result.returncode != 0
    assert "New/Append" in result.stderr
    assert not Path(env["CALL_LOG"]).exists()
