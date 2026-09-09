"""Exercise smoke submission guards without a scheduler or GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "slurm/run_tdambi_checkpoint_smoke_oscar.sbatch"


def setup_launcher(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text('#!/bin/bash\nif [[ "$1" == rev-parse ]]; then echo tested; '
                   'else echo "${FAKE_DIRTY:-}"; fi\n')
    git.chmod(0o755)
    python = bin_dir / "python"
    python.write_text(f"#!{sys.executable}\n" +
                      "import json, os, sys\n"
                      "with open(os.environ['CALL_LOG'], 'a') as f:\n"
                      "    f.write(json.dumps({'args': sys.argv[1:], 'matrix': "
                      "os.environ.get('AMBI_TRACE_BENCHMARK_MATRIX')}) + '\\n')\n")
    python.chmod(0o755)
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    Path(str(checkpoint) + ".metadata.json").write_text("{}")
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
           "SLURM_SUBMIT_DIR": str(ROOT), "EXPECTED_ACTION_MODES_SHA": "tested",
           "TDAMBI_CHECKPOINT": str(checkpoint), "SLURM_JOB_ID": "smoke-test",
           "TDAMBI_SMOKE_OUTPUT_ROOT": str(tmp_path / "results"),
           "AMBI_DMC_PYTHON": str(python), "CALL_LOG": str(tmp_path / "calls.jsonl"),
           "TDAMBI_MEASURE_TRACE": "0", "FAKE_DIRTY": ""}
    return env


@pytest.mark.parametrize("measure", [False, True])
def test_smoke_invokes_one_checkpoint_and_optional_timing(tmp_path, measure):
    env = setup_launcher(tmp_path)
    env["TDAMBI_MEASURE_TRACE"] = str(int(measure))
    subprocess.run(["bash", str(SCRIPT)], env=env, check=True, capture_output=True)
    calls = [json.loads(line) for line in Path(env["CALL_LOG"]).read_text().splitlines()]
    assert len(calls) == 4 + int(measure)
    assert calls[0]["args"][0] == "-c"
    assert "assert torch.cuda.is_available()" in calls[0]["args"][1]
    evaluator = calls[2]["args"]
    assert evaluator[0] == "evaluate_ambi_checkpoint.py"
    assert evaluator[evaluator.index("--checkpoint") + 1] == env["TDAMBI_CHECKPOINT"]
    assert evaluator[evaluator.index("--preset") + 1] == "inner_budget/tdambi_3"
    assert evaluator[evaluator.index("--max-steps") + 1] == "3"
    assert "--wandb" not in evaluator and "--reference-bundle" not in evaluator
    if measure:
        assert calls[-1]["args"][-1].endswith("::test_cuda_trace_overhead_measurement")
        assert calls[-1]["matrix"] == str(ROOT / "configs/research/tdambi_humanoid_inner_benchmark.json")


@pytest.mark.parametrize("rounds", [7, 10])
def test_smoke_uses_requested_rounds_preset(tmp_path, rounds):
    env = setup_launcher(tmp_path)
    env["TDAMBI_PRESET"] = f"update_timing/step_j{rounds}_c1_a1"
    subprocess.run(["bash", str(SCRIPT)], env=env, check=True, capture_output=True)
    calls = [json.loads(line) for line in Path(env["CALL_LOG"]).read_text().splitlines()]
    evaluator = calls[2]["args"]
    assert evaluator[evaluator.index("--preset") + 1] == env["TDAMBI_PRESET"]
    assert evaluator[evaluator.index("--max-steps") + 1] == "3"


@pytest.mark.parametrize("invalid", ["commit", "dirty", "existing_output", "missing_sidecar"])
def test_smoke_preflight_fails_before_compute_or_overwrite(tmp_path, invalid):
    env = setup_launcher(tmp_path)
    if invalid == "commit":
        env["EXPECTED_ACTION_MODES_SHA"] = "different"
    elif invalid == "dirty":
        env["FAKE_DIRTY"] = " M source.py"
    elif invalid == "existing_output":
        Path(env["TDAMBI_SMOKE_OUTPUT_ROOT"]).mkdir()
    else:
        Path(env["TDAMBI_CHECKPOINT"] + ".metadata.json").unlink()
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True)
    assert result.returncode != 0
    assert not Path(env["CALL_LOG"]).exists()
