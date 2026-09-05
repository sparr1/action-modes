"""Transition-based inner SAC cadence and new opt-in configuration contracts."""

from copy import deepcopy

import pytest
import torch

from tests.test_ambi_root_local_sac import _build_cfg, _tiny_model
from utils.resume_identity import scientific_trial_parameters
from RL.tdmpc2_core.common.inner_utils import updates_for_transitions


@pytest.mark.parametrize("count,interval,expected", [
    (49, .07, 700), (28, .14, 200), (3, .3, 10),
    (0, .07, 0), (1, 1.01, 0), (10**18, .125, 8 * 10**18),
])
def test_decimal_interval_boundaries_do_not_lose_updates(count, interval, expected):
    assert updates_for_transitions(count, interval) == expected


@pytest.mark.parametrize("interval", [0, -1, True, "2", float("nan"), float("inf")])
def test_steps_per_update_rejects_invalid_intervals(interval):
    with pytest.raises(ValueError, match="inner_steps_per_update"):
        _build_cfg(inner_steps_per_update=interval)


@pytest.mark.parametrize("key,value", [
    ("inner_updates_per_round", 1),
    ("inner_updates_per_round", "auto"),
    ("inner_critic_updates_per_round", 1),
    ("inner_actor_updates_per_round", 2),
    ("inner_updates_per_iteration", 1),
    ("inner_model_step_budget", 8),
    ("inner_actor_updates_per_action", 1),
    ("inner_critic_updates_per_action", 1),
    ("inner_temperature_updates_per_action", 1),
    ("inner_explorer_actor_updates_per_round", 1),
    ("inner_explorer_critic_updates_per_round", 0),
    ("inner_explorer_temperature_updates_per_round", 1),
])
def test_steps_per_update_rejects_competing_update_controls(key, value):
    with pytest.raises(ValueError, match="inner_steps_per_update"):
        _build_cfg(inner_steps_per_update=2.5, **{key: value})


def test_null_gradient_controls_do_not_activate_a_legacy_schedule():
    cfg = _build_cfg(
        inner_steps_per_update=2.5, inner_updates_per_iteration=None,
        inner_updates_per_round=None, inner_model_step_budget=None,
        inner_critic_updates_per_round=None, inner_actor_updates_per_round=None,
    )
    assert cfg.inner_schedule_mode == "canonical"
    assert cfg.inner_steps_per_update == 2.5


@pytest.mark.parametrize("component", ["actor", "critic"])
def test_steps_per_update_requires_actor_and_critic_updates(component):
    with pytest.raises(ValueError, match="inner_steps_per_update"):
        _build_cfg(inner_steps_per_update=2, **{f"inner_{component}_adaptation": "frozen"})


@pytest.mark.parametrize("key,value", [
    ("inner_finite_horizon", True),
    ("inner_outer_replay_fraction", 0.25),
    ("inner_steps_per_update", 2),
])
@pytest.mark.parametrize("operator", ["none", "mppi", "td3"])
def test_new_inner_sac_controls_reject_other_operators(key, value, operator):
    with pytest.raises(ValueError, match=key):
        _build_cfg(inner_operator=operator, **{key: value})


@pytest.mark.parametrize("value", [-0.1, 1.1, True, "0.5", float("nan"), float("inf")])
def test_outer_replay_fraction_is_strict_probability(value):
    with pytest.raises(ValueError, match="inner_outer_replay_fraction"):
        _build_cfg(inner_outer_replay_fraction=value)


@pytest.mark.parametrize("value", [1, "true", None])
def test_finite_horizon_is_strict_boolean(value):
    with pytest.raises(ValueError, match="inner_finite_horizon"):
        _build_cfg(inner_finite_horizon=value)


@pytest.mark.parametrize("interval,expected", [(5, [0, 1, 1]), (2.5, [1, 2, 1]), (0.5, [8, 8, 8])])
def test_steps_carry_between_rounds_reset_between_actions(interval, expected, monkeypatch):
    model = _tiny_model(inner_updates_per_round=None, inner_steps_per_update=interval)
    try:
        engine = model.agent.inner_engine
        counts = []
        original = engine._run_update_counts

        def track(**kwargs):
            counts.append((kwargs["critic_count"], kwargs["actor_count"], kwargs["temperature_count"]))
            return original(**kwargs)

        monkeypatch.setattr(engine, "_run_update_counts", track)
        for _ in range(2):
            model.agent.act(torch.zeros(3), collect_diagnostics=False)
            assert counts[-3:] == [(count, count, count) for count in expected]
            assert model.agent.last_inner_metrics["inner_critic_optimizer_steps"] == sum(expected)
            assert model.agent.last_inner_metrics["inner_requested_update_slots"] == sum(expected)
        assert model.cfg.inner_expected_update_slots == sum(expected)
        assert model.cfg.inner_updates_per_iteration is None
        assert model.cfg.inner_critic_updates_per_action == sum(expected)
    finally:
        model.env.close()


def test_early_termination_counts_only_generated_transitions(monkeypatch):
    model = _tiny_model(inner_updates_per_round=None, inner_steps_per_update=5, episodic=True)
    try:
        engine = model.agent.inner_engine
        monkeypatch.setattr(engine.model, "termination", lambda z: z.new_ones((z.shape[0], 1)))
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        assert model.agent.last_inner_metrics["inner_model_steps"] == 6
        assert model.agent.last_inner_metrics["inner_critic_optimizer_steps"] == 1
        assert model.agent.last_inner_metrics["inner_actor_optimizer_steps"] == 1
    finally:
        model.env.close()


@pytest.mark.parametrize("mode", ["frozen_random", "shared_mixture", "separate_critics"])
def test_symmetric_explorer_inherits_fractional_cadence(mode):
    model = _tiny_model(inner_updates_per_round=None, inner_steps_per_update=5, inner_explorer_mode=mode)
    try:
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        metrics = model.agent.last_inner_metrics
        assert metrics["inner_critic_optimizer_steps"] == 2
        assert metrics["inner_actor_optimizer_steps"] == 2
        assert metrics["inner_explorer_actor_optimizer_steps"] == (0 if mode == "frozen_random" else 2)
        assert metrics["inner_explorer_critic_optimizer_steps"] == (2 if mode == "separate_critics" else 0)
        assert model.cfg.inner_expected_update_slots == 2
        assert model.cfg.inner_explorer_actor_updates_per_action == (0 if mode == "frozen_random" else 2)
    finally:
        model.env.close()


def test_new_controls_are_part_of_resume_identity_with_inactive_defaults():
    base = {"alg": "AMBITDMPC2/AMBITDMPC2", "alg_params": {}}
    defaults = {"inner_finite_horizon": False, "inner_steps_per_update": None, "inner_outer_replay_fraction": 0.0}
    assert scientific_trial_parameters(base) == scientific_trial_parameters({**base, "alg_params": defaults})
    for key, value in (("inner_finite_horizon", True), ("inner_steps_per_update", 2), ("inner_outer_replay_fraction", 0.5)):
        assert scientific_trial_parameters(base) != scientific_trial_parameters({**base, "alg_params": {key: value}})


@pytest.mark.parametrize("option", [
    {"inner_finite_horizon": True},
    {"inner_steps_per_update": 5, "inner_updates_per_round": None},
    {"inner_outer_replay_fraction": .5},
])
def test_exact_resume_rejects_changed_inner_options_before_mutation(option):
    source = _tiny_model(**option)
    target = _tiny_model()
    matching = _tiny_model(**option)
    try:
        source.agent.prepare_training_resume_boundary()
        saved = deepcopy(source.agent.training_state_dict())
        matching.agent.load_training_state_dict(saved)
        assert matching.agent._critic_target_spec() == saved["outer"]["critic_target_spec"]
        before = {name: value.clone() for name, value in target.agent.model.state_dict().items()}
        with pytest.raises(ValueError, match="critic-target specification"):
            target.agent.load_training_state_dict(saved)
        for name, value in target.agent.model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    finally:
        source.env.close()
        target.env.close()
        matching.env.close()
