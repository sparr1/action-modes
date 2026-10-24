"""The J10 follow-up preserves the protocol, including complete solve replay."""

from copy import deepcopy

import pytest

from slurm import ambi_reward_training as campaign
from slurm.benchmark_ambi_training_cpu import action_work
from tests.test_ambi_root_local_sac import _build_cfg


GROUP = "ambi-return-only-j10-training-20261024"
NAME = "ambi_full_h3_j10_k3_reward_only"


def test_j10_changes_only_rounds_derived_replay_and_publication_from_reference():
    base_name, base_manifest, base = campaign.validate_campaign(campaign.MANIFESTS["sac"])
    name, manifest, current = campaign.validate_campaign(campaign.J10_MANIFEST)
    expected = deepcopy(base)
    expected["name"] = NAME
    expected["alg_params"]["inner_rounds"] = 10
    expected["alg_params"]["inner_replay_capacity"] = 3840
    for key in ("wandb_tags", "wandb_group", "wandb_run_name"):
        expected["alg_params"][key] = current["alg_params"][key]
    assert current == expected
    expected_manifest = deepcopy(base_manifest)
    expected_manifest.update(configs=[NAME], study_note=manifest["study_note"])
    assert manifest == expected_manifest
    assert name == NAME and base_name == "ambi_full_h3_j8_k3_reward_only"
    p = current["alg_params"]
    assert p["wandb_group"] == GROUP and p["wandb_project"] == "ambi"
    assert p["wandb_run_name"] == "H3 | Fresh SAC | J10 | K3 | reward-only Q | seed55"
    assert "J10" in p["wandb_tags"] and "J8" not in p["wandb_tags"]
    assert GROUP in p["wandb_tags"]
    cfg = _build_cfg(**p)
    assert (cfg.inner_rounds, cfg.inner_rollout_horizon, cfg.inner_solve_interval) == (10, 3, 3)
    assert cfg.inner_hold_encoder == "solve_snapshot"
    assert (cfg.inner_critic_updates_per_action, cfg.inner_actor_updates_per_action,
            cfg.inner_temperature_updates_per_action, cfg.inner_model_step_budget) == (160, 40, 40, 3840)
    assert cfg.inner_replay_capacity == cfg.inner_model_step_budget == 3840
    assert cfg.outer_critic_target == cfg.inner_sac_critic_target == "reward_only"
    assert cfg.aux_return_mode == "off" and cfg.critic_value_mode == "single"
    assert cfg.target_entropy == -10.5 and cfg.inner_target_entropy == "inherit_outer"
    assert cfg.inner_temperature_scope == cfg.inner_temperature_optimizer_scope == "action"
    assert current["total_steps"] == 500000 and current["seed"] == 55
    assert (current["checkpoint_every"], current["save_strat"]) == (25000, "all")


def test_j10_smoke_preserves_solve_protocol():
    _, _, production = campaign.validate_campaign(campaign.J10_MANIFEST)
    _, _, run, overrides = campaign.smoke_recipe(12, 120, campaign.J10_MANIFEST)
    expected = deepcopy(production["alg_params"])
    expected.update(overrides)
    assert run["alg_params"] == expected
    assert run["total_steps"] == 633


@pytest.mark.parametrize("key,value", [("inner_rounds", 8), ("inner_solve_interval", 1),
    ("inner_replay_capacity", 3072), ("inner_critic_updates_per_round", 8),
    ("aux_return_mode", "sac"), ("inner_temperature_scope", "run")])
def test_gate_rejects_changed_j10_protocol(monkeypatch, key, value):
    selected = campaign.recipe(0, campaign.J10_MANIFEST)
    selected[2]["alg_params"][key] = value
    monkeypatch.setattr(campaign, "recipe", lambda *args: selected)
    with pytest.raises(AssertionError):
        campaign.validate_campaign(campaign.J10_MANIFEST)


def test_gate_rejects_unapproved_manifest_and_renamed_cell(monkeypatch):
    unapproved = campaign.ROOT / "configs/dmcontrol/experiments/ambi_held_training.json"
    with pytest.raises(AssertionError, match="Unapproved"):
        campaign.validate_campaign(unapproved)
    _, manifest, run = campaign.recipe(0, campaign.J10_MANIFEST)
    monkeypatch.setattr(campaign, "recipe", lambda *args: ("renamed", manifest, run))
    with pytest.raises(AssertionError, match="identity"):
        campaign.validate_campaign(campaign.J10_MANIFEST)


@pytest.mark.parametrize("rounds", [8, 10])
def test_work_validation_accepts_exact_dose_and_rejects_old_or_incomplete_dose(rounds):
    metrics = {**campaign.sac_work(rounds), "inner_compile_fallback": 0.,
               "inner_solve_performed": 1., "inner_policy_held": 0.,
               "inner_action_age": 0., "inner_episode_decision_index": 0.}
    assert campaign.validate_action(metrics, operator="sac", decision=0, interval=3, rounds=rounds)
    if rounds == 8:
        assert action_work(metrics, decision=0, interval=3, probability=0.)
    else:
        with pytest.raises(AssertionError):
            action_work(metrics, decision=0, interval=3, probability=0.)
    for key in campaign.sac_work(rounds):
        wrong = {**metrics, key: metrics[key] - 1}
        with pytest.raises(AssertionError):
            campaign.validate_action(wrong, operator="sac", decision=0, interval=3, rounds=rounds)


def _receipt(rounds=10):
    binding, runtime = {"commit": "test", "manifest_sha256": "test"}, {"gpu": "NVIDIA L40S"}
    cases = []
    for cores in (1, 2):
        row = dict(passed=True, binding=binding, runtime=runtime, config=NAME,
            operator="sac", inner_solve_interval=3, inner_rounds=rounds,
            inner_work_per_solve=campaign.sac_work(rounds), physical_cores=cores,
            compile_fallback=False, replay_storage_device="cpu", compile_graph_counts={"before": 6, "after": 6},
            auxiliary_critic_present=False, outer_target_entropy=-10.5,
            measured=dict(decisions=120, outer_updates=120, solve_decisions=40,
                          held_decisions=80, seconds_per_decision=.4),
            warmup_active_decisions=12, outer_actor_optimizer_steps=134,
            outer_temperature_optimizer_steps=134)
        for key in ("reward_only_targets", "primary_critic_sources", "outer_actor_entropy_auto",
                    "checkpoint_roundtrip", "checkpoint_sidecars", "boundary_training_state_roundtrip"):
            row[key] = True
        cases.append(row)
    return dict(passed=True, binding=binding, runtime=runtime, cases=cases)


def _validate(receipt, **overrides):
    kwargs = dict(name=NAME, operator="sac", interval=3)
    kwargs.update(overrides)
    campaign.validate_receipt(receipt, receipt["binding"], receipt["runtime"], 1, **kwargs)


def test_receipt_requires_j10_work_for_both_cpu_measurements():
    _validate(_receipt())
    with pytest.raises(AssertionError):
        _validate(_receipt(8))
    for key in campaign.sac_work(10):
        broken = _receipt()
        broken["cases"][1]["inner_work_per_solve"][key] -= 1
        with pytest.raises(AssertionError, match="dose"):
            _validate(broken)
    with pytest.raises(AssertionError, match="protocol"):
        _validate(_receipt(), interval=1)
    with pytest.raises(AssertionError, match="Unapproved"):
        _validate(_receipt(), name="renamed")
