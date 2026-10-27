"""Held feedback-policy identity, failure boundaries, and exact continuation."""

from copy import deepcopy

import gymnasium as gym
import pytest
import torch

from RL.tdmpc2_core.paired_controller_evaluation import PairedControllerEvaluator
from tests.test_ambi_inner_initialization import _snapshot
from tests.test_ambi_latency_contract import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_component_model
from tests.test_pave_integration import _batch
from utils.eval_series_data import planner_identity
from utils.resume_identity import scientific_trial_parameters


@pytest.fixture
def models():
    opened = []

    def create(**overrides):
        options = dict(
            inner_solve_interval=3, inner_rounds=1, inner_rollout_horizon=3,
            train_unroll_horizon=3,
            inner_critic_updates_per_round=2,
            inner_actor_updates_per_round=1, inner_replay_capacity=6,
            inner_execution_action="policy_sample", inner_critic_bernoulli_p=.5,
        )
        options.update(overrides)
        model = _tiny_component_model(**options)
        opened.append(model)
        return model

    yield create
    for model in opened:
        model.close()
        model.env.close()


def test_cadence_and_encoder_semantics_are_scientific_but_one_is_legacy():
    base = {"alg": "AMBITDMPC2/AMBITDMPC2", "seed": 13,
            "alg_params": {"inner_operator": "sac"}}
    explicit = {**base, "alg_params": {**base["alg_params"], "inner_solve_interval": 1}}
    held = {**base, "alg_params": {**base["alg_params"], "inner_solve_interval": 3}}
    resolved = {**held, "alg_params": {**held["alg_params"], "inner_hold_encoder": "solve_snapshot"}}
    incompatible = {**held, "alg_params": {**held["alg_params"], "inner_hold_encoder": "current"}}
    for identity in (
        scientific_trial_parameters,
        lambda trial: planner_identity(trial["alg_params"], {}, trial["alg"], "tanh_sample"),
    ):
        assert identity(base) == identity(explicit)
        assert identity(held) == identity(resolved)
        assert identity(held) != identity(base)
        assert identity(incompatible) != identity(held)


def test_interval_one_preserves_actions_rng_and_exact_payload(models):
    implicit = models(inner_solve_interval=1)
    explicit = models(inner_solve_interval=1)
    # Omission resolves to the default, rather than adding encoder/cache state.
    del implicit.cfg.inner_solve_interval
    for index in range(4):
        obs = torch.tensor([.1 * index, .2, -.3])
        expected = implicit.agent.act(obs, t0=index == 0, collect_diagnostics=False)
        actual = explicit.agent.act(obs, t0=index == 0, collect_diagnostics=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _assert_tree_equal(_snapshot(explicit.agent.inner_engine), _snapshot(implicit.agent.inner_engine))
    for model in (implicit, explicit):
        model.agent.prepare_training_resume_boundary()
    _assert_tree_equal(explicit.agent.training_state_dict(), implicit.agent.training_state_dict())
    assert "solve_interval" not in explicit.agent._critic_target_spec()["inner_solve"]


@pytest.mark.parametrize("probability", [0., .5])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_boundary_resume_reproduces_held_actions_masks_and_outer_updates(models, probability, auxiliary):
    options = dict(inner_critic_bernoulli_p=probability)
    if auxiliary:
        options.update(aux_return_mode="sac", inner_critic_source="aux_return",
                       inner_horizon_critic_source="aux_return", inner_sac_critic_target="reward_only")
    source = models(**options).agent
    source.act(torch.zeros(3), t0=True, collect_diagnostics=False)
    source._update(*_batch(source))
    source.act(torch.ones(3) * .1, collect_diagnostics=False)
    assert source.inner_engine._held_encoder is not None
    with pytest.raises(RuntimeError, match="boundary|held|cadence|allocation|transfer"):
        source.training_state_dict()
    source.prepare_training_resume_boundary()
    payload = deepcopy(source.training_state_dict())
    assert payload["outer"]["critic_target_spec"]["inner_solve"]["solve_interval"] == 3
    assert payload["outer"]["critic_target_spec"]["inner_solve"]["hold_encoder"] == "solve_snapshot"
    assert source.inner_engine._held_encoder is source.inner_engine._encoder_pool is None

    restored = models(**options).agent
    global_rng = torch.random.get_rng_state().clone()
    restored.load_training_state_dict(payload)
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    _assert_tree_equal(restored.training_state_dict(), payload)
    for agent in (source, restored):
        agent.reset()
    for index in range(7):
        outer_rng = torch.random.get_rng_state().clone()
        for agent in (source, restored):
            torch.random.set_rng_state(outer_rng)
            agent._update(*_batch(agent))
        observation = torch.tensor([.1 * index, .25, -.4])
        expected = source.act(observation, collect_diagnostics=False)
        actual = restored.act(observation, collect_diagnostics=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _assert_tree_equal(_snapshot(source.inner_engine), _snapshot(restored.inner_engine))
        _assert_tree_equal(source.inner_engine._held_encoder.state_dict(),
                           restored.inner_engine._held_encoder.state_dict())
        assert restored.last_inner_metrics["inner_solve_performed"] == float(index % 3 == 0)
        if probability:
            assert restored.last_inner_metrics["inner_critic_bernoulli_applied"] == float(index > 0 and index % 3 == 0)
    for agent in (source, restored):
        agent.prepare_training_resume_boundary()
    _assert_tree_equal(restored.training_state_dict(), source.training_state_dict())


@pytest.mark.parametrize("source_interval,target_interval", [(1, 3), (3, 1), (2, 3)])
@pytest.mark.parametrize("direct", [False, True])
def test_exact_resume_rejects_changed_interval_before_any_mutation(models, source_interval, target_interval, direct):
    source = models(inner_solve_interval=source_interval).agent
    target = models(inner_solve_interval=target_interval).agent
    if direct:
        source, target = source.inner_engine, target.inner_engine
    before = deepcopy(target.training_state_dict())
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="incompatible|does not match|cadence|solve"):
        target.load_training_state_dict(deepcopy(source.training_state_dict()))
    _assert_tree_equal(target.training_state_dict(), before)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


def test_exact_resume_rejects_changed_encoder_semantics_before_mutation(models):
    agent = models().agent
    before = deepcopy(agent.training_state_dict())
    invalid = deepcopy(before)
    invalid["outer"]["critic_target_spec"]["inner_solve"]["hold_encoder"] = "current"
    rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="critic-target specification"):
        agent.load_training_state_dict(invalid)
    _assert_tree_equal(agent.training_state_dict(), before)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["encoder", "missing", "extra", "boolean_version"])
def test_direct_inner_resume_rejects_invalid_hold_spec_before_mutation(models, corruption):
    engine = models().agent.inner_engine
    before = deepcopy(engine.training_state_dict())
    invalid = deepcopy(before)
    if corruption == "encoder":
        invalid["held_policy_spec"]["encoder"] = "current"
    elif corruption == "missing":
        del invalid["held_policy_spec"]
    elif corruption == "extra":
        invalid["held_policy_spec"]["extra"] = True
    else:
        invalid["held_policy_spec"]["version"] = True
    rng = torch.random.get_rng_state().clone()
    with pytest.raises((ValueError, TypeError), match="held-policy"):
        engine.load_training_state_dict(invalid)
    _assert_tree_equal(engine.training_state_dict(), before)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("source_interval,target_interval", [(1, 3), (3, 1), (2, 3)])
def test_portable_load_accepts_different_cadence_and_discards_live_cache(models, source_interval, target_interval):
    source = models(inner_solve_interval=source_interval).agent
    target = models(inner_solve_interval=target_interval).agent
    target.act(torch.zeros(3), t0=True)
    target.load(deepcopy(source.checkpoint_state()))
    _assert_tree_equal(target.model.state_dict(), source.model.state_dict())
    assert target.inner_engine._held_encoder is None
    assert target.inner_engine._encoder_pool is None
    assert target.inner_engine._held_actor is None
    assert not target.inner_engine._critic_transfer_ready
    target.act(torch.ones(3) * .1)
    if target_interval > 1:
        assert target.last_inner_metrics["inner_solve_performed"] == 1.
        assert target.last_inner_metrics["inner_action_age"] == 0.
    assert target.last_inner_metrics["inner_critic_bernoulli_applied"] == 0.


@pytest.mark.parametrize("failure", ["encoding", "materialization"])
def test_agent_failures_discard_held_control_and_donor(models, monkeypatch, failure):
    agent = models().agent
    agent.act(torch.zeros(3), t0=True)
    engine = agent.inner_engine
    assert engine._held_encoder is not None and engine._critic_transfer_ready

    def fail(*args, **kwargs):
        raise RuntimeError("injected held control failure")

    with monkeypatch.context() as patch:
        if failure == "encoding":
            patch.setattr(engine._held_encoder, "forward", fail)
        else:
            patch.setattr(agent, "_materialize_action_metrics", fail)
        with pytest.raises(RuntimeError, match="injected held control failure"):
            agent.act(torch.ones(3))
    assert engine._held_encoder is engine._held_actor is None
    assert not engine._critic_transfer_ready
    agent.act(torch.zeros(3))
    assert agent.last_inner_metrics["inner_solve_performed"] == 1.
    assert agent.last_inner_metrics["inner_critic_bernoulli_applied"] == 0.


@pytest.mark.parametrize("probability", [0., .5])
def test_paired_summary_counts_q_samples_on_solves_and_costs_on_all_actions(models, monkeypatch, probability):
    agent = models(inner_critic_bernoulli_p=probability).agent
    act = agent.act

    def measured_action(*args, **kwargs):
        result = act(*args, **kwargs)
        metrics = agent.last_inner_metrics
        if metrics["inner_solve_performed"]:
            metrics["inner_fixed_target_q_action_gain"] = (
                2. if metrics["inner_episode_decision_index"] == 0 else 4.
            )
            metrics["inner_action_seconds"] = 1.
            metrics["inner_diagnostic_seconds"] = .2
            metrics["inner_encoder_snapshot_seconds"] = .05
        else:
            assert "inner_fixed_target_q_action_gain" not in metrics
            assert "inner_diagnostic_seconds" not in metrics
            metrics["inner_action_seconds"] = .1
        return result

    monkeypatch.setattr(agent, "act", measured_action)
    evaluator = PairedControllerEvaluator(
        agent, lambda: gym.make("Pendulum-v1", max_episode_steps=4),
        lambda obs: torch.as_tensor(obs, dtype=torch.float32), lambda action: action * 2,
        episodes=2, seed=18, device="cpu",
    )
    try:
        metrics = evaluator.evaluate()
    finally:
        evaluator.close()
    stem = "paired_bernoulli_inner" if probability else "paired_fresh_inner"
    assert metrics[f"eval/{stem}_actions"] == 8
    assert metrics[f"eval/{stem}_solve_decisions"] == 4
    assert metrics[f"eval/{stem}_held_decisions"] == 4
    assert metrics[f"eval/{stem}_fixed_target_q_action_gain_count"] == 4
    assert metrics[f"eval/{stem}_fixed_target_q_action_gain_mean"] == 3.
    assert metrics[f"eval/{stem}_model_steps_per_action"] == 3.
    assert metrics[f"time/{stem}_control_seconds_per_action"] == pytest.approx(.475)
    assert metrics[f"time/{stem}_diagnostic_seconds_per_action"] == pytest.approx(.1)
