"""One-axis K3 provenance and fail-closed gate/launch checks."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

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


def test_j10_only_changes_rounds_derived_capacity_and_publication_labels(monkeypatch):
    monkeypatch.delenv("AMBI_ALPHA_CARRY_K3_MANIFEST", raising=False)
    _, source_manifest, source = campaign.recipe()
    monkeypatch.setenv("AMBI_ALPHA_CARRY_K3_MANIFEST", campaign.SUPPORTED_MANIFESTS[1])
    name, manifest, run = campaign.recipe()
    source["alg_params"]["inner_rounds"] = 10
    source["alg_params"]["inner_replay_capacity"] = 10 * 128 * 3
    for config in (source, run):
        del config["name"]
        for key in ("wandb_run_name", "wandb_group", "wandb_tags"):
            config["alg_params"].pop(key)
    assert run == source
    assert run["alg_params"]["inner_replay_capacity"] == 3840
    assert manifest["configs"] == [name]
    for key in set(source_manifest) - {"study_note", "study_type", "configs"}:
        assert manifest[key] == source_manifest[key]
    assert campaign.binding()["manifest"] == campaign.SUPPORTED_MANIFESTS[1]


def test_j10_resolves_full_update_dose_and_no_eviction_replay_capacity(monkeypatch):
    from tests.test_ambi_root_local_sac import _build_cfg
    monkeypatch.setenv("AMBI_ALPHA_CARRY_K3_MANIFEST", campaign.SUPPORTED_MANIFESTS[1])
    _, _, run = campaign.recipe()
    cfg = _build_cfg(**run["alg_params"])
    assert (cfg.inner_rounds, cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action) == (10, 160, 40)
    assert cfg.inner_replay_capacity == cfg.inner_model_step_budget == 3840 and cfg.inner_solve_interval == 3
    agent = SimpleNamespace(target_entropy=-10.5, aux_return=SimpleNamespace(has_actor=False),
                            inner_engine=SimpleNamespace(_resolved_inner_target_entropy=lambda: -10.5))
    campaign.assert_recipe(cfg, agent, run)


def test_only_supported_manifests_can_be_selected(monkeypatch):
    monkeypatch.setenv("AMBI_ALPHA_CARRY_K3_MANIFEST", "configs/dmcontrol/experiments/ambi_alpha_carry_training.json")
    with pytest.raises(AssertionError, match="supported K3"):
        campaign.recipe()


def test_j10_cadence_requires_160_40_40_and_no_held_updates():
    metrics = dict(inner_solve_performed=1., inner_policy_held=0., inner_action_age=0.,
                   inner_compile_fallback=0., inner_critic_optimizer_steps=160.,
                   inner_actor_optimizer_steps=40., inner_temperature_optimizer_steps=40.)
    assert campaign.assert_cadence(metrics, 3, expected_rounds=10)
    with pytest.raises(AssertionError):
        campaign.assert_cadence(metrics, 3)
    metrics["inner_temperature_optimizer_steps"] = 32.
    with pytest.raises(AssertionError):
        campaign.assert_cadence(metrics, 3, expected_rounds=10)
    held = dict.fromkeys(campaign.HELD_ZERO_WORK, 0.)
    held.update(inner_solve_performed=0., inner_policy_held=1., inner_action_age=2., inner_policy_evaluations=1.)
    assert campaign.assert_cadence(held, 5, expected_rounds=10) is False


def test_j10_run_temperature_carries_forty_adam_steps_per_solve():
    row = dict(t0=False, outer_alpha=.7, alpha_initial=.7, alpha_final=.69,
               optimizer_step_initial=0., optimizer_step_final=40.,
               episode_reset=0., alpha_identity=1, optimizer_identity=2)
    campaign.shared.check_temperature_transition(None, row, "run", temperature_steps=40)
    following = {**row, "t0": True, "outer_alpha": .8, "alpha_initial": .69,
                 "optimizer_step_initial": 40., "optimizer_step_final": 80.}
    campaign.shared.check_temperature_transition(row, following, "run", temperature_steps=40)
    with pytest.raises(AssertionError):
        campaign.shared.check_temperature_transition(row, following, "run")


def test_benchmark_workers_inherit_selected_j10_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("AMBI_ALPHA_CARRY_K3_MANIFEST", campaign.SUPPORTED_MANIFESTS[1])
    monkeypatch.setenv("SLURM_JOB_ID", "test")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setattr(campaign.os, "sched_getaffinity", lambda _: {0, 1}, raising=False)
    monkeypatch.setattr(campaign, "physical_core_cpus", lambda _: [0, 1])
    expected_binding = campaign.binding()
    monkeypatch.setattr(campaign, "binding", lambda: expected_binding)
    selected = []

    def worker(command, *, cwd, env, **_):
        selected.append(env["AMBI_ALPHA_CARRY_K3_MANIFEST"])
        cpus = len(command[command.index("--worker-cpus") + 1].split(","))
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps({"runtime": {}, "physical_cores": cpus,
                                      "measured": {"seconds_per_decision": 1.}}))

    monkeypatch.setattr(campaign.subprocess, "run", worker)
    campaign.benchmark(SimpleNamespace(output=tmp_path / "benchmark", warmup=24, decisions=96))
    assert selected == [campaign.SUPPORTED_MANIFESTS[1]] * 2


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
