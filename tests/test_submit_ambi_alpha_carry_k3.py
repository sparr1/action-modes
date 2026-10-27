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
    gate = {"passed": True, "binding": {"commit": "a" * 40, "manifest": scheduler.DEFAULT_MANIFEST},
            "runtime": runtime, "seconds": 80.}
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


def test_measured_production_overhead_precedes_remaining_walltime_margin():
    gate, benchmark = receipts()
    sizing = scheduler.resources(gate, benchmark, production_overhead_factor=1.13)
    assert sizing["production_overhead_factor"] == 1.13
    assert sizing["walltime_seconds"] == math.ceil((.25 * 1.13 * 1e6 * 1.15 + 160 + 600) / 3600) * 3600


@pytest.mark.parametrize("factor", [.99, float("inf"), float("nan")])
def test_invalid_production_overhead_is_rejected(factor):
    with pytest.raises(AssertionError):
        scheduler.resources(*receipts(), production_overhead_factor=factor)


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


def setup(tmp_path, monkeypatch, outcome="12345;oscar\n", manifest=scheduler.DEFAULT_MANIFEST):
    source, campaign = tmp_path / "source", tmp_path / "campaign"
    source.mkdir()
    (campaign / "gate").mkdir(parents=True)
    (campaign / "cpu-benchmark").mkdir()
    gate, benchmark = receipts()
    gate["binding"]["manifest"] = manifest
    benchmark["binding"]["manifest"] = manifest
    for row in benchmark["cases"]:
        row["binding"]["manifest"] = manifest
    (campaign / "gate/receipt.json").write_text(json.dumps(gate))
    (campaign / "cpu-benchmark/results.json").write_text(json.dumps(benchmark))
    calls = []

    def check_output(command, **kwargs):
        if command[0] == "git":
            return "a" * 40 + "\n" if "rev-parse" in command else ""
        if command[0] == sys.executable:
            assert kwargs["cwd"] == source and "validate_gate" in command[2]
            assert kwargs["env"]["AMBI_ALPHA_CARRY_K3_MANIFEST"] == manifest
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
    assert "--job-name=ambi-alpha-k3" in command
    assert "--gres=gpu:l40s:1" in command and "--cpus-per-task=1" in command and "--mem=16G" in command
    assert command[-1] == str(source / "slurm/run_ambi_alpha_carry_k3_oscar.sbatch")
    env = kwargs["env"]
    assert env["AMBI_PROJECT_DIR"] == str(source) and env["AMBI_ALPHA_CARRY_K3_MODE"] == "train"
    assert env["AMBI_TRAIN_WALLTIME_SECONDS"] == str(result["resources"]["walltime_seconds"])
    assert env["AMBI_RESUME_MODE"] == "new"
    assert scheduler.submit(source, campaign, sys.executable, "a" * 40) == result
    assert len(calls) == 1


def test_j10_manifest_is_pinned_in_validator_dispatch_and_receipt(tmp_path, monkeypatch):
    manifest = "configs/dmcontrol/experiments/ambi_alpha_carry_k3_j10_training.json"
    source, campaign, calls = setup(tmp_path, monkeypatch, manifest=manifest)
    result = scheduler.submit(source, campaign, sys.executable, "a" * 40, manifest, 1.13)
    assert result["identity"]["manifest"] == manifest
    assert result["identity"]["production_overhead_factor"] == result["resources"]["production_overhead_factor"] == 1.13
    assert calls[0][1]["env"]["AMBI_ALPHA_CARRY_K3_MANIFEST"] == manifest
    assert "--job-name=ambi-alpha-k3-j10" in calls[0][0]
    with pytest.raises(AssertionError, match="another identity"):
        scheduler.submit(source, campaign, sys.executable, "a" * 40, scheduler.DEFAULT_MANIFEST, 1.13)
    with pytest.raises(AssertionError, match="another identity"):
        scheduler.submit(source, campaign, sys.executable, "a" * 40, manifest, 1.0)
    assert len(calls) == 1


def test_successful_legacy_j8_receipt_defaults_to_no_extra_overhead(tmp_path, monkeypatch):
    source, campaign, calls = setup(tmp_path, monkeypatch)
    result = scheduler.submit(source, campaign, sys.executable, "a" * 40)
    del result["identity"]["manifest"]
    del result["identity"]["production_overhead_factor"]
    (campaign / "submission.json").write_text(json.dumps(result))
    assert scheduler.submit(source, campaign, sys.executable, "a" * 40) == result
    assert len(calls) == 1


def test_nested_submission_removes_parent_allocation_and_temporary_environment(tmp_path, monkeypatch):
    source, campaign, calls = setup(tmp_path, monkeypatch)
    poisoned = {"SLURM_TRES_PER_TASK": "cpu=1", "SLURM_MEM_PER_CPU": "1024",
                "SLURM_TMPDIR": "/parent-node/slurm", "SLURM_JOB_ID": "parent-job",
                "SBATCH_PARTITION": "debug", "SBATCH_EXPORT": "NONE",
                "SRUN_CPUS_PER_TASK": "1", "CUDA_VISIBLE_DEVICES": "",
                "TMPDIR": "/parent-node/tmpdir", "TMP": "/parent-node/tmp",
                "TEMP": "/parent-node/temp"}
    for key, value in poisoned.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("WANDB_API_KEY", "test-only-auth")
    monkeypatch.setenv("USER_SETTING", "preserved")
    scheduler.submit(source, campaign, sys.executable, "a" * 40)
    command, kwargs = calls[0]
    environment = kwargs["env"]
    assert not set(poisoned) & environment.keys()
    assert not any(key.startswith(("SLURM_", "SBATCH_", "SRUN_")) for key in environment)
    assert environment["WANDB_API_KEY"] == "test-only-auth"
    assert environment["USER_SETTING"] == "preserved"
    assert environment["AMBI_PROJECT_DIR"] == str(source)
    assert environment["AMBI_ALPHA_CARRY_K3_MODE"] == "train"
    assert "--partition=gpu" in command and "--qos=pri-gpu+" in command
    assert "--cpus-per-task=1" in command and "--mem=16G" in command
    assert "--gres=gpu:l40s:1" in command and "--export=ALL" in command


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
