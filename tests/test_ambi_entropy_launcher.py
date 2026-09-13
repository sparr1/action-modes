"""Exercise the real shell launchers with stub CUDA, Git and evaluator processes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "slurm/run_ambi_entropy_prior_oscar.sbatch"
PUBLISHER = ROOT / "slurm/run_ambi_entropy_prior_publish_oscar.sbatch"


@pytest.fixture
def harness(tmp_path):
    repo = tmp_path / "checkout with spaces"
    repo.mkdir()
    binaries = tmp_path / "bin"
    binaries.mkdir()
    git = binaries / "git"
    git.write_text("#!/usr/bin/env bash\nif [[ \"$1\" == rev-parse ]]; then printf '%s\\n' \"${FAKE_SHA:-tested}\"; else printf '%s' \"${FAKE_DIRTY:-}\"; fi\n")
    git.chmod(0o755)
    (repo / "torch.py").write_text(
        "import os\n__version__ = 'test'\nthreads = None\n"
        "def set_num_threads(value):\n global threads; threads = value\n"
        "def set_num_interop_threads(value):\n assert value == 1\n"
        "class cuda:\n"
        " @staticmethod\n def is_available(): return os.environ.get('FAKE_CUDA', '1') == '1'\n"
        " @staticmethod\n def get_device_name(index): return 'stub CUDA'\n"
    )
    (repo / "pytest.py").write_text(
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['SMOKE_RECEIPT']).write_text(json.dumps({'argv':sys.argv[1:],"
        "'real_tests':os.environ.get('AMBI_RUN_REAL_DMCONTROL_TESTS')}))\n"
        "sys.exit(int(os.environ.get('FAKE_TEST_EXIT', '0')))\n"
    )
    (repo / "evaluate_ambi_entropy.py").write_text(
        "import json, os, pathlib, sys\n"
        "data = {'argv':sys.argv[1:], 'wandb':os.environ.get('WANDB_MODE'),"
        "'omp':os.environ.get('OMP_NUM_THREADS'), 'mujoco':os.environ.get('MUJOCO_GL')}\n"
        "if sys.argv[1] == 'run':\n import torch\n data['torch_threads'] = torch.threads\n"
        "pathlib.Path(os.environ['EVALUATOR_RECEIPT']).write_text(json.dumps(data))\n"
        "sys.exit(int(os.environ.get('FAKE_EVALUATOR_EXIT', '0')))\n"
    )
    study = tmp_path / "study with spaces.json"
    study.write_text(json.dumps({"checkpoints": [{"id": "older"}, {"id": "current"}]}))
    checkpoint_paths = {}
    for name in ("older", "current"):
        checkpoint = tmp_path / f"{name} checkpoint"
        checkpoint.write_text("weights")
        Path(str(checkpoint) + ".metadata.json").write_text("{}")
        checkpoint_paths[name] = str(checkpoint)
    campaign = tmp_path / "campaign.json"
    payload = {"study": str(study), "checkpoint_paths": checkpoint_paths,
               "tasks": [{"checkpoint_index": index, "episode_seed": seed,
                          "output_dir": str(tmp_path / f"result {index}-{seed}")}
                         for index, seed in ((0, 101), (1, 102))]}
    campaign.write_text(json.dumps(payload))
    env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
               PYTHONPATH=str(repo), SLURM_SUBMIT_DIR=str(repo), SLURM_JOB_ID="test",
               SLURM_TMPDIR=str(tmp_path), SLURM_ARRAY_TASK_ID="1", SLURM_CPUS_PER_TASK="6",
               EXPECTED_ACTION_MODES_SHA="tested", AMBI_ENTROPY_CAMPAIGN=str(campaign),
               AMBI_DMC_PYTHON=sys.executable, AMBI_ENTROPY_SMOKE="0",
               EVALUATOR_RECEIPT=str(tmp_path / "evaluate.json"),
               SMOKE_RECEIPT=str(tmp_path / "smoke.json"))
    for key in ("FAKE_SHA", "FAKE_DIRTY", "FAKE_CUDA", "FAKE_TEST_EXIT", "FAKE_EVALUATOR_EXIT"):
        env.pop(key, None)
    return env, campaign, payload


def launch(path, env):
    return subprocess.run(["bash", str(path)], env=env, capture_output=True, text=True, timeout=20)


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("smoke", [False, True])
def test_worker_selects_exact_campaign_cell_and_keeps_wandb_disabled(harness, index, smoke):
    env, _, payload = harness
    env.update(SLURM_ARRAY_TASK_ID=str(index), AMBI_ENTROPY_SMOKE=str(int(smoke)))
    result = launch(WORKER, env)
    assert result.returncode == 0, result.stderr
    actual = json.loads(Path(env["EVALUATOR_RECEIPT"]).read_text())
    task = payload["tasks"][index]
    expected = ["run", "--study", payload["study"], "--checkpoint",
                payload["checkpoint_paths"]["older" if index == 0 else "current"],
                "--checkpoint-index", str(index), "--episode-seed", str(task["episode_seed"]),
                "--output-dir", task["output_dir"]]
    assert actual["argv"] == expected + (["--smoke"] if smoke else [])
    assert actual["wandb"] == "disabled"
    assert actual["torch_threads"] == 6 and actual["omp"] == "6"
    assert actual["mujoco"] == "egl"
    assert Path(env["SMOKE_RECEIPT"]).exists() == smoke
    if smoke:
        tests = json.loads(Path(env["SMOKE_RECEIPT"]).read_text())
        assert tests["real_tests"] == "1"
        assert tests["argv"] == ["-q", "tests/test_ambi_entropy_probe.py",
                                 "tests/test_ambi_entropy_reporting.py",
                                 "tests/test_ambi_entropy_evaluator.py",
                                 "tests/test_ambi_real_calibration.py"]


@pytest.mark.parametrize("failure", ["wrong_sha", "dirty", "no_cuda", "bad_smoke", "negative_index",
                                     "large_index", "bad_checkpoint_index", "bad_seed", "missing_checkpoint",
                                     "missing_sidecar", "relative_output", "existing_output", "existing_work",
                                     "duplicate_output", "duplicate_cell", "relative_study", "duplicate_key"])
def test_worker_rejects_invalid_dispatch_before_evaluation(harness, failure):
    env, campaign, payload = harness
    if failure == "wrong_sha":
        env["FAKE_SHA"] = "other"
    elif failure == "dirty":
        env["FAKE_DIRTY"] = " M source.py"
    elif failure == "no_cuda":
        env["FAKE_CUDA"] = "0"
    elif failure == "bad_smoke":
        env["AMBI_ENTROPY_SMOKE"] = "yes"
    elif failure == "negative_index":
        env["SLURM_ARRAY_TASK_ID"] = "-1"
    elif failure == "large_index":
        env["SLURM_ARRAY_TASK_ID"] = "2"
    elif failure == "bad_checkpoint_index":
        payload["tasks"][1]["checkpoint_index"] = True
    elif failure == "bad_seed":
        payload["tasks"][1]["episode_seed"] = -1
    elif failure == "missing_checkpoint":
        Path(payload["checkpoint_paths"]["current"]).unlink()
    elif failure == "missing_sidecar":
        Path(payload["checkpoint_paths"]["current"] + ".metadata.json").unlink()
    elif failure == "relative_output":
        payload["tasks"][1]["output_dir"] = "relative"
    elif failure in ("existing_output", "existing_work"):
        Path(payload["tasks"][1]["output_dir"] + (".work" if failure == "existing_work" else "")).mkdir()
    elif failure == "duplicate_output":
        payload["tasks"][1]["output_dir"] = payload["tasks"][0]["output_dir"]
    elif failure == "duplicate_cell":
        payload["tasks"][1].update(checkpoint_index=0, episode_seed=101)
    elif failure == "relative_study":
        payload["study"] = "study.json"
    campaign.write_text(json.dumps(payload) if failure != "duplicate_key" else '{"tasks": [], "tasks": []}')
    result = launch(WORKER, env)
    assert result.returncode != 0
    assert not Path(env["EVALUATOR_RECEIPT"]).exists()
    assert not Path(env["SMOKE_RECEIPT"]).exists()


def test_failed_cuda_smoke_tests_stop_before_experiment(harness):
    env, _, _ = harness
    env.update(AMBI_ENTROPY_SMOKE="1", FAKE_TEST_EXIT="7")
    result = launch(WORKER, env)
    assert result.returncode != 0
    assert Path(env["SMOKE_RECEIPT"]).exists()
    assert not Path(env["EVALUATOR_RECEIPT"]).exists()


@pytest.mark.parametrize("exit_code", [0, 4])
def test_publisher_delegates_coverage_and_preserves_incomplete_result_failure(harness, exit_code):
    env, campaign, _ = harness
    env["FAKE_EVALUATOR_EXIT"] = str(exit_code)
    result = launch(PUBLISHER, env)
    assert result.returncode == exit_code, result.stderr
    receipt = json.loads(Path(env["EVALUATOR_RECEIPT"]).read_text())
    assert receipt["argv"] == ["publish", "--campaign", str(campaign)]
    assert receipt["wandb"] == "online" and receipt["omp"] == "2"
    assert "torch_threads" not in receipt


@pytest.mark.parametrize("path", [WORKER, PUBLISHER])
@pytest.mark.parametrize("failure", ["wrong_sha", "dirty", "relative_campaign", "missing_campaign"])
def test_both_launchers_enforce_git_and_campaign_preflight(harness, path, failure):
    env, _, _ = harness
    if failure == "wrong_sha":
        env["FAKE_SHA"] = "other"
    elif failure == "dirty":
        env["FAKE_DIRTY"] = "?? untracked.py"
    elif failure == "relative_campaign":
        env["AMBI_ENTROPY_CAMPAIGN"] = "campaign.json"
    else:
        env["AMBI_ENTROPY_CAMPAIGN"] += ".missing"
    assert launch(path, env).returncode != 0
    assert not Path(env["EVALUATOR_RECEIPT"]).exists()


def test_slurm_resources_and_shell_syntax_keep_concurrency_and_dependencies_at_submission():
    for path in (WORKER, PUBLISHER):
        subprocess.run(["bash", "-n", str(path)], check=True)
        text = path.read_text()
        assert "#SBATCH --array=" not in text
        assert "#SBATCH --dependency=" not in text
        assert "--no-requeue" in text
    worker = WORKER.read_text()
    assert "--partition=gpu" in worker and "--gres=gpu:l40s:1" in worker
    assert "--cpus-per-task=6" in worker and "--mem=32G" in worker and "--time=04:00:00" in worker
    publisher = PUBLISHER.read_text()
    assert "--partition=batch" in publisher and "--mem=16G" in publisher and "--cpus-per-task=2" in publisher
    assert "--time=04:00:00" in publisher and "--gres" not in publisher
    assert "campaign.compute_job_id" in publisher
