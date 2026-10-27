"""Measured resources, pinned receipts and durable single-submission behavior."""

from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from slurm import submit_ambi_alpha_carry_k3 as scheduler


def receipts():
    runtime = {"gpu": "NVIDIA L40S", "executable": sys.executable}
    gate = {"passed": True, "binding": {"commit": "a" * 40}, "runtime": runtime, "seconds": 80.}
    benchmark = {"passed": True, "binding": gate["binding"], "recommended_cpus_within_5_percent_of_fastest": 1,
                 "cases": [{"passed": True, "binding": gate["binding"], "runtime": runtime, "physical_cores": cpus,
                            "measured": {"seconds_per_decision": seconds}} for cpus, seconds in ((1, .25), (2, .24))]}
    return deepcopy(gate), deepcopy(benchmark)


def test_walltime_uses_selected_cpu_measurement_and_requested_margin():
    gate, benchmark = receipts()
    sizing = scheduler.resources(gate, benchmark)
    assert sizing["cpus"] == 1 and sizing["memory_gib"] == 16 and sizing["gpus"] == 1
    assert sizing["walltime_seconds"] == math.ceil((.25 * 1e6 * 1.15 + 160 + 600) / 3600) * 3600
    benchmark["cases"][1]["measured"]["seconds_per_decision"] = .20
    benchmark["recommended_cpus_within_5_percent_of_fastest"] = 2
    assert scheduler.resources(gate, benchmark)["cpus"] == 2


@pytest.mark.parametrize("corruption", ["binding", "runtime", "missing", "recommended", "nonfinite", "failed"])
def test_corrupt_benchmark_cannot_size(corruption):
    gate, benchmark = receipts()
    if corruption == "binding": benchmark["cases"][0]["binding"] = {"commit": "b" * 40}
    elif corruption == "runtime": benchmark["cases"][0]["runtime"] = {"gpu": "other"}
    elif corruption == "missing": benchmark["cases"].pop()
    elif corruption == "recommended": benchmark["recommended_cpus_within_5_percent_of_fastest"] = 2
    elif corruption == "nonfinite": benchmark["cases"][0]["measured"]["seconds_per_decision"] = float("nan")
    else: benchmark["cases"][0]["passed"] = False
    with pytest.raises(AssertionError): scheduler.resources(gate, benchmark)


def setup(tmp_path, monkeypatch, outcome="12345;oscar\n"):
    source, campaign = tmp_path / "source", tmp_path / "campaign"
    source.mkdir()
    (campaign / "gate").mkdir(parents=True)
    (campaign / "cpu-benchmark").mkdir()
    gate, benchmark = receipts()
    (campaign / "gate/receipt.json").write_text(json.dumps(gate))
    (campaign / "cpu-benchmark/results.json").write_text(json.dumps(benchmark))
    calls = []

    def check_output(command, **kwargs):
        if command[0] == "git":
            return "a" * 40 + "\n" if "rev-parse" in command else ""
        if command[0] == sys.executable:
            assert kwargs["cwd"] == source and "validate_gate" in command[2]
            return json.dumps(gate["binding"])
        assert command[0] == "sbatch"
        calls.append((command, kwargs))
        intent = json.loads((campaign / "submission.json").read_text())
        assert intent["status"] == "submitting"
        if isinstance(outcome, Exception): raise outcome
        return outcome

    monkeypatch.setattr(scheduler.subprocess, "check_output", check_output)
    return source, campaign, calls


def test_submission_exports_measured_request_and_rerun_is_idempotent(tmp_path, monkeypatch):
    source, campaign, calls = setup(tmp_path, monkeypatch)
    result = scheduler.submit(source, campaign, sys.executable, "a" * 40)
    assert result["status"] == "submitted" and result["job_id"] == "12345"
    command, kwargs = calls[0]
    assert "--partition=gpu" in command and "--qos=pri-gpu+" in command
    assert "--gres=gpu:l40s:1" in command and "--cpus-per-task=1" in command and "--mem=16G" in command
    assert command[-1] == str(source / "slurm/run_ambi_alpha_carry_k3_oscar.sbatch")
    env = kwargs["env"]
    assert env["AMBI_PROJECT_DIR"] == str(source) and env["AMBI_ALPHA_CARRY_K3_MODE"] == "train"
    assert env["AMBI_TRAIN_WALLTIME_SECONDS"] == str(result["resources"]["walltime_seconds"])
    assert env["AMBI_RESUME_MODE"] == "new"
    assert scheduler.submit(source, campaign, sys.executable, "a" * 40) == result
    assert len(calls) == 1


@pytest.mark.parametrize("outcome", ["unexpected response", subprocess.CalledProcessError(1, ["sbatch"])])
def test_unknown_outcome_preserves_intent_and_never_resubmits(tmp_path, monkeypatch, outcome):
    source, campaign, calls = setup(tmp_path, monkeypatch, outcome)
    with pytest.raises((AssertionError, subprocess.CalledProcessError)):
        scheduler.submit(source, campaign, sys.executable, "a" * 40)
    with pytest.raises(AssertionError, match="Unknown submission outcome"):
        scheduler.submit(source, campaign, sys.executable, "a" * 40)
    assert len(calls) == 1


def test_wrong_receipt_binding_is_rejected_before_sbatch(tmp_path, monkeypatch):
    source, campaign, calls = setup(tmp_path, monkeypatch)
    path = campaign / "cpu-benchmark/results.json"
    value = json.loads(path.read_text())
    value["binding"] = {"commit": "b" * 40}
    path.write_text(json.dumps(value))
    with pytest.raises(AssertionError): scheduler.submit(source, campaign, sys.executable, "a" * 40)
    assert not calls and not (campaign / "submission.json").exists()
