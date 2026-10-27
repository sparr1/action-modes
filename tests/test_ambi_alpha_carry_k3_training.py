"""One-axis K3 provenance and fail-closed gate/launch checks."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from slurm import ambi_alpha_carry_k3_training as campaign


def test_k3_only_changes_source_cadence_and_publication_labels():
    source = campaign.read(campaign.ROOT / "configs/dmcontrol/algs/ambi_alpha_carry_h3_j8_run_detached_ent10p5.json")
    name, manifest, run = campaign.recipe()
    expected = deepcopy(source)
    expected["alg_params"]["inner_solve_interval"] = 3
    del run["name"]
    for config in (run, expected):
        for key in ("wandb_run_name", "wandb_group", "wandb_tags"):
            config["alg_params"].pop(key)
    assert run == expected
    assert manifest["configs"] == [name] and manifest["trials"] == 1
    assert manifest["source_run_url"].endswith("be85c916fcd94d5a813d7857d0ef5efb")
    original = campaign.read(campaign.ROOT / "configs/dmcontrol/experiments/ambi_alpha_carry_training.json")
    for key in ("overrides_alg", "env_params", "checkpoint_every", "checkpoint_replay", "save_strat", "save_trials", "logs", "log_type"):
        assert manifest[key] == original[key]


def receipt():
    return {"passed": True, "binding": campaign.binding(), "config": campaign.recipe()[0],
            "held_cadence": True, "held_alpha_adam_unchanged": True, "alpha_and_adam_lifecycle": True,
            "checkpoint_roundtrip": True, "checkpoint_replay_roundtrip": True,
            "checkpoint_replay_rng_unchanged": True, "compile_fallback": False,
            "runtime": {"python": "3.10.19", "torch": "2.3.1+cu121", "cuda": "12.1",
                        "gpu": "NVIDIA L40S", "executable": "/locked/python"}}


@pytest.mark.parametrize("field", ["held_cadence", "held_alpha_adam_unchanged", "alpha_and_adam_lifecycle",
                                  "checkpoint_roundtrip", "checkpoint_replay_roundtrip", "checkpoint_replay_rng_unchanged"])
def test_incomplete_gate_cannot_launch(field):
    report = receipt()
    report[field] = False
    with pytest.raises(AssertionError):
        campaign.validate_gate(report, campaign.binding())


def test_changed_commit_or_configuration_cannot_reuse_gate():
    report = receipt()
    report["binding"]["commit"] = "another-commit"
    with pytest.raises(AssertionError):
        campaign.validate_gate(report, campaign.binding())


def test_held_steps_must_have_zero_inner_work():
    metrics = dict.fromkeys(campaign.HELD_ZERO_WORK, 0.)
    metrics.update(inner_solve_performed=0., inner_policy_held=1., inner_action_age=1., inner_policy_evaluations=1.)
    assert campaign.assert_cadence(metrics, 1) is False
    metrics["inner_temperature_optimizer_steps"] = 32.
    with pytest.raises(AssertionError):
        campaign.assert_cadence(metrics, 1)


def test_launch_binds_gate_benchmark_resources_and_single_native_lineage(tmp_path, monkeypatch):
    report = receipt()
    benchmark = {"passed": True, "binding": campaign.binding(),
                 "recommended_cpus_within_5_percent_of_fastest": 1,
                 "cases": [{"passed": True, "binding": campaign.binding(), "runtime": report["runtime"],
                            "physical_cores": cores} for cores in (1, 2)]}
    gate_path, benchmark_path = tmp_path / "gate.json", tmp_path / "benchmark.json"
    gate_path.write_text(json.dumps(report))
    benchmark_path.write_text(json.dumps(benchmark))
    monkeypatch.setattr(campaign, "runtime", lambda: report["runtime"])
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setenv("AMBI_LINEAGE_DIR", str(tmp_path / "lineage"))
    monkeypatch.setenv("AMBI_TRAIN_WALLTIME_SECONDS", "259200")
    monkeypatch.setenv("AMBI_DRAIN_AFTER_SECONDS", "257400")
    calls = []
    monkeypatch.setattr(campaign.os, "execv", lambda executable, command: calls.append(command))
    campaign.launch(tmp_path / "launch", gate_path, benchmark_path)
    command = calls[0]
    assert command[command.index("--resume-checkpoint-minutes") + 1] == "15"
    assert command[command.index("--resume-mode") + 1] == "new"
    assert command[command.index("--drain-after-seconds") + 1] == "257400"
    assert "--alg-index" not in command
    manifest = json.loads(Path(command[command.index("--run") + 1]).read_text())
    assert manifest["configs"] == [campaign.recipe()[0]]
    record = json.loads((tmp_path / "launch/launch.json").read_text())
    assert record["requested_walltime_seconds"] == 259200
    assert record["run_params"]["checkpoint_replay"] is True
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    with pytest.raises(AssertionError):
        campaign.launch(tmp_path / "bad-cpus", gate_path, benchmark_path)
    assert not (tmp_path / "bad-cpus").exists()
