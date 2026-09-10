"""Production checkpoint mapping and early-failure guards without a scheduler."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "slurm/run_tdambi_checkpoint_eval_hydra.sbatch"
SOURCE = "rwgao_b-brown-university/ambi/xq3zva9u"


def launcher_environment(tmp_path, index=2):
    executables = tmp_path / "bin"
    executables.mkdir()
    git = executables / "git"
    git.write_text('#!/bin/bash\nif [[ "$1" == rev-parse ]]; then echo tested; '
                   'else echo "${FAKE_DIRTY:-}"; fi\n')
    git.chmod(0o755)
    python = executables / "python"
    python.write_text(f"#!{sys.executable}\n" +
                      "import json, os, sys\n"
                      "if sys.argv[1] == '-':\n"
                      "    sys.argv = sys.argv[1:]\n"
                      "    exec(compile(sys.stdin.read(), '<launcher-preflight>', 'exec'))\n"
                      "else:\n"
                      "    with open(os.environ['CALL_LOG'], 'a') as handle:\n"
                      "        handle.write(json.dumps(sys.argv[1:]) + '\\n')\n")
    python.chmod(0o755)
    step = int(index) * 50000
    checkpoint_prefix = tmp_path / "checkpoint_"
    checkpoint = Path(f"{checkpoint_prefix}{step}")
    checkpoint.touch()
    Path(f"{checkpoint}.metadata.json").write_text("{}")
    reference_root = tmp_path / "references"
    reference = reference_root / f"step_{step}" / "paired.json"
    reference.parent.mkdir(parents=True)
    reference.write_text("{}")
    registry = tmp_path / "curve"
    registry.mkdir()
    (registry / "run.json").write_text("{}")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"source_run": SOURCE}))
    return {**os.environ, "PATH": str(executables) + os.pathsep + os.environ["PATH"],
            "SLURM_SUBMIT_DIR": str(ROOT), "SLURM_ARRAY_TASK_ID": str(index),
            "EXPECTED_ACTION_MODES_SHA": "tested", "AMBI_DMC_PYTHON": str(python),
            "TDAMBI_CHECKPOINT_PREFIX": str(checkpoint_prefix),
            "TDAMBI_REFERENCE_ROOT": str(reference_root),
            "TDAMBI_OUTPUT_ROOT": str(tmp_path / "output"),
            "CHECKPOINT_INVENTORY": str(inventory), "EVAL_RUN_DIR": str(registry),
            "CALL_LOG": str(tmp_path / "calls.jsonl"), "FAKE_DIRTY": ""}


@pytest.mark.parametrize("index", [2, 3, 6, 9, 10, 19, 20])
def test_production_maps_checkpoint_and_keeps_fixed_evaluation_protocol(tmp_path, index):
    env = launcher_environment(tmp_path, index)
    subprocess.run(["bash", str(SCRIPT)], env=env, check=True, capture_output=True)
    calls = [json.loads(line) for line in Path(env["CALL_LOG"]).read_text().splitlines()]
    assert len(calls) == 3
    preflight, evaluate, report = calls
    assert preflight[0] == "-c"
    assert "assert torch.cuda.is_available()" in preflight[1]
    assert 'device="cuda"' in preflight[1]
    assert evaluate[0] == "evaluate_ambi_checkpoint.py"
    step = index * 50000
    assert evaluate[evaluate.index("--checkpoint") + 1] == env["TDAMBI_CHECKPOINT_PREFIX"] + str(step)
    assert evaluate[evaluate.index("--reference-bundle") + 1] == f"{env['TDAMBI_REFERENCE_ROOT']}/step_{step}/paired.json"
    assert evaluate[evaluate.index("--preset") + 1] == "inner_budget/tdambi_3"
    assert evaluate[evaluate.index("--controller-seed") + 1] == "12345"
    assert evaluate[evaluate.index("--seeds") + 1:evaluate.index("--max-steps")] == ["101", "102", "103", "104", "105"]
    assert evaluate[evaluate.index("--max-steps") + 1] == "500"
    assert evaluate[evaluate.index("--device") + 1] == "cuda"
    assert evaluate[evaluate.index("--eval-run-dir") + 1] == env["EVAL_RUN_DIR"]
    assert evaluate[evaluate.index("--checkpoint-inventory") + 1] == env["CHECKPOINT_INVENTORY"]
    assert evaluate[evaluate.index("--bundle-dir") + 1] == f"{env['TDAMBI_OUTPUT_ROOT']}/step_{step}/bundle"
    assert "--wandb" not in evaluate and "--overwrite" not in evaluate
    assert report[0] == "report_ambi_benchmark.py"
    assert report[report.index("--bundle") + 1] == evaluate[evaluate.index("--bundle-dir") + 1]


@pytest.mark.parametrize("invalid", ["commit", "dirty", "index", "sidecar", "reference", "registry", "source", "existing_output"])
def test_production_rejects_invalid_launch_without_starting_evaluation(tmp_path, invalid):
    env = launcher_environment(tmp_path)
    if invalid == "commit":
        env["EXPECTED_ACTION_MODES_SHA"] = "different"
    elif invalid == "dirty":
        env["FAKE_DIRTY"] = " M source.py"
    elif invalid == "index":
        env["SLURM_ARRAY_TASK_ID"] = "21"
    elif invalid == "sidecar":
        Path(env["TDAMBI_CHECKPOINT_PREFIX"] + "100000.metadata.json").unlink()
    elif invalid == "reference":
        Path(env["TDAMBI_REFERENCE_ROOT"], "step_100000", "paired.json").unlink()
    elif invalid == "registry":
        Path(env["EVAL_RUN_DIR"], "run.json").unlink()
    elif invalid == "source":
        Path(env["CHECKPOINT_INVENTORY"]).write_text(json.dumps({"source_run": "entity/project/wrong"}))
    elif invalid == "existing_output":
        existing = Path(env["TDAMBI_OUTPUT_ROOT"], "step_100000")
        existing.mkdir(parents=True)
        (existing / "keep.txt").write_text("completed data")
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True)
    assert result.returncode != 0
    assert not Path(env["CALL_LOG"]).exists()
    if invalid == "existing_output":
        assert (existing / "keep.txt").read_text() == "completed data"
    else:
        assert not Path(env["TDAMBI_OUTPUT_ROOT"]).exists()


def test_production_defaults_to_fast_gpu_pool_without_array_throttle():
    text = SCRIPT.read_text()
    for directive in ("--partition=gpus", "--constraint=l40s|rtx_a6000", "--gres=gpu:1",
                      "--cpus-per-task=8", "--mem=32G", "--time=06:00:00", "--array=2-20"):
        assert f"#SBATCH {directive}" in text
    assert "#SBATCH --nodelist" not in text
    assert "#SBATCH --array=2-20%" not in text


def test_production_selects_step_preset_without_changing_episode_protocol(tmp_path):
    env = launcher_environment(tmp_path, 20)
    env["TDAMBI_PRESET"] = "update_timing/step_j5_c1_a1"
    subprocess.run(["bash", str(SCRIPT)], env=env, check=True, capture_output=True)
    preflight, evaluate, report = [json.loads(line) for line in Path(env["CALL_LOG"]).read_text().splitlines()]
    assert evaluate[evaluate.index("--preset") + 1] == env["TDAMBI_PRESET"]
    assert evaluate[evaluate.index("--checkpoint") + 1].endswith("1000000")
    assert evaluate[evaluate.index("--device") + 1] == "cuda"
    assert evaluate[evaluate.index("--max-steps") + 1] == "500"
    assert env["TDAMBI_PRESET"] in report[report.index("--title") + 1]


@pytest.mark.parametrize("preset", [f"update_timing/step_j{j}_c1_a1" for j in (1, 3, 7, 10)]
                         + [f"entropy/squashed_eta1e_{e}" for e in (5, 4, 3)])
def test_oscar_production_reuses_protocol_with_requested_preset(tmp_path, preset):
    env = launcher_environment(tmp_path, 20)
    env["TDAMBI_PRESET"] = preset
    oscar = ROOT / "slurm/run_tdambi_checkpoint_eval_oscar.sbatch"
    subprocess.run(["bash", str(oscar)], env=env, check=True, capture_output=True)
    preflight, evaluate, report = [json.loads(line) for line in Path(env["CALL_LOG"]).read_text().splitlines()]
    assert "assert torch.cuda.is_available()" in preflight[1]
    assert evaluate[evaluate.index("--preset") + 1] == env["TDAMBI_PRESET"]
    assert evaluate[evaluate.index("--checkpoint") + 1].endswith("1000000")
    assert evaluate[evaluate.index("--seeds") + 1:evaluate.index("--max-steps")] == ["101", "102", "103", "104", "105"]
    assert evaluate[evaluate.index("--max-steps") + 1] == "500"
    assert evaluate[evaluate.index("--reference-bundle") + 1].endswith("step_1000000/paired.json")
    text = oscar.read_text()
    for directive in ("--partition=gpu", "--qos=pri-gpu+", "--gres=gpu:l40s:1", "--cpus-per-task=6", "--mem=32G"):
        assert f"#SBATCH {directive}" in text
    assert "#SBATCH --array=2-20%" not in text
