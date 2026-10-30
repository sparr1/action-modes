"""Reject stale gates and unsafe completion claims before production launch."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from slurm import ambi_smoothness_launch as launch


@pytest.fixture
def gate():
    binding = {"commit": "a" * 40, "config_sha256": "recipe", "lock_sha256": "lock"}
    runtime = {"python": "3.10.19", "torch": "2.3.1+cu121", "cuda": "12.1",
               "gpu": "NVIDIA L40S", "executable": "/env/bin/python"}
    cases = [{"passed": True, "binding": binding, "runtime": runtime,
              "compile_fallback": False, "pave_sample_count": 768,
              "replay_storage_device": "cpu", "physical_cores": n,
              "measured": {"seconds_per_decision": .8 / n}}
             for n in (1, 2)]
    return {"passed": True, "cases": cases}, binding, runtime


def test_gate_requires_exact_source_runtime_and_measured_cpu(gate):
    data, binding, runtime = gate
    assert launch.validate_gate(data, binding, runtime, 1)["physical_cores"] == 1
    for other_binding, other_runtime, cpus in (({**binding, "commit": "b" * 40}, runtime, 1),
                                              (binding, {**runtime, "torch": "other"}, 1),
                                              (binding, runtime, 8)):
        with pytest.raises(ValueError):
            launch.validate_gate(data, other_binding, other_runtime, cpus)


@pytest.mark.parametrize("key,value", [("passed", False), ("compile_fallback", True),
                                     ("pave_sample_count", 64), ("replay_storage_device", "cuda")])
def test_gate_rejects_failed_or_changed_work(gate, key, value):
    data, binding, runtime = gate
    changed = deepcopy(data)
    changed["cases"][0][key] = value
    with pytest.raises(ValueError):
        launch.validate_gate(changed, binding, runtime, 2)


def test_command_uses_individual_exact_resume_and_separate_snapshot_storage(tmp_path):
    home, scratch = tmp_path / "home", tmp_path / "scratch"
    command = launch.command_for("train", "/env/bin/python", 2, Path("cell2.json"),
                                 home, scratch, home / "gate", 7200, "required")
    def value(flag):
        return command[command.index(flag) + 1]
    assert command[:2] == ["/env/bin/python", "main.py"]
    assert value("--run") == "cell2.json" and "--alg-index" not in command
    assert value("--num-runs") == "1"
    assert value("--lineage-dir") == str(home / "cell-2/lineage")
    assert value("--resume-model-snapshot-root") == str(scratch / "cell-2/snapshot-bank")
    assert value("--resume-mode") == "required"
    assert value("--resume-wandb-mode") == "online"
    assert value("--resume-checkpoint-minutes") == "15"
    assert value("--drain-after-seconds") == "7200"
    gate = launch.command_for("gate", "/env/bin/python", 2, Path("cell2.json"), home, scratch, home / "gate")
    assert gate[1] == "slurm/benchmark_ambi_smoothness.py"
    assert gate[gate.index("--cores") + 1] == "1,2"
    with pytest.raises(ValueError):
        launch.command_for("train", "/env/bin/python", 2, Path("cell2.json"), home, scratch, home / "gate", 0)


def test_manifest_binding_includes_real_files_and_detects_individual_drift(tmp_path):
    root = tmp_path / "source"
    (root / launch.MASTER.parent).mkdir(parents=True)
    (root / launch.ALGS).mkdir(parents=True)
    (root / "environments/dmcontrol").mkdir(parents=True)
    (root / "environments/dmcontrol/uv.lock").write_text("locked")
    master = {"configs": [f"cell{i}" for i in range(4)], "trials": 1, "env_params": {"task": "humanoid-walk"}}
    (root / launch.MASTER).write_text(json.dumps(master))
    path = root / launch.MASTER.parent / "cell2.json"
    individual = {**master, "configs": ["cell2"]}
    path.write_text(json.dumps(individual))
    (root / launch.ALGS / "cell2.json").write_text(json.dumps({"seed": 55, "total_steps": 1_000_000}))
    name, selected, binding = launch.recipe_binding(root, 2, "a" * 40)
    assert name == "cell2" and root / selected == path
    assert binding["manifest_sha256"] == launch.digest(root / launch.MASTER)
    assert binding["lock_sha256"] == launch.digest(root / "environments/dmcontrol/uv.lock")
    path.write_text(json.dumps({**individual, "env_params": {"task": "hopper-hop"}}))
    with pytest.raises(ValueError, match="disagree"):
        launch.recipe_binding(root, 2, "a" * 40)


@pytest.mark.parametrize("value,expected", [("01:30:00", 5400), ("2-01:02:03", 176523)])
def test_allocation_is_read_from_slurm(monkeypatch, value, expected):
    monkeypatch.setattr(launch.subprocess, "check_output", lambda *a, **kw: f"JobId=1 TimeLimit={value} NumCPUs=2")
    assert launch.allocation_seconds({"SLURM_JOB_ID": "1"}) == expected


def test_zero_exit_requires_verified_done_and_fresh_segment(tmp_path, monkeypatch):
    from utils import oscar_resume_launcher as resume
    lineage = tmp_path / "lineage"
    (lineage / "segments/new").mkdir(parents=True)
    (lineage / "DONE").write_text(json.dumps({"segment_id": "new"}))
    calls = []
    monkeypatch.setattr(resume, "verify_done", lambda **kw: calls.append(kw) or "generation")
    result = launch.verify_exit(0, lineage, "42", set())
    assert result["state"] == "complete" and calls == [{"lineage_dir": lineage}]
    with pytest.raises(ValueError, match="new exact-resume"):
        launch.verify_exit(0, lineage, "42", {"new"})
    (lineage / "DONE").write_text(json.dumps({"segment_id": "old"}))
    with pytest.raises(ValueError, match="older"):
        launch.verify_exit(0, lineage, "42", set())


def test_handoff_is_75_and_never_auto_requeues(tmp_path, monkeypatch):
    from utils import oscar_resume_launcher as resume
    lineage = tmp_path / "lineage"
    (lineage / "segments/new").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(resume, "verify_handoff", lambda **kw: calls.append(kw) or "generation")
    result = launch.verify_exit(75, lineage, "42", set())
    assert result["state"] == "clean_handoff" and result["exit_code"] == 75
    assert calls == [{"lineage_dir": lineage, "slurm_job_id": "42", "segment_id": "new"}]
    assert launch.verify_exit(85, lineage, "42", set())["state"] == "failed"
    script = (launch.ROOT / "slurm/run_ambi_smoothness_hydra.sbatch").read_text()
    for required in ("--no-requeue", "--nodelist=gpu2501", "--constraint=l40s", "--gres=gpu:nvidia_l40s:1"):
        assert required in script
    assert "scontrol requeue" not in script


def test_hydra_opts_into_wandb_core_before_starting_the_job():
    script = (launch.ROOT / "slurm/run_ambi_smoothness_hydra.sbatch").read_text()
    for setting in ("WANDB__REQUIRE_CORE=true", "WANDB_INIT_TIMEOUT=90",
                    "WANDB_HTTP_TIMEOUT=30", "WANDB__SERVICE_WAIT=60"):
        assert setting in script
        assert script.index(setting) < script.index("exec srun")
