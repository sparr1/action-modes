"""Cadence mechanics preserve private learner state between actual solves."""

from copy import deepcopy

import pytest
import torch

from tests.test_ambi_bernoulli_transfer import _oracle, _parameters
from tests.test_ambi_inner_initialization import _snapshot
from tests.test_ambi_latency_contract import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_model
from tests.test_pave_integration import _batch


@pytest.fixture
def models():
    opened = []

    def create(**overrides):
        options = dict(inner_rounds=1, inner_updates_per_round=2,
                       inner_solve_interval=3, inner_critic_bernoulli_p=.5,
                       inner_execution_action="policy_sample")
        options.update(overrides)
        model = _tiny_model(**options)
        opened.append(model)
        return model

    yield create
    for model in opened:
        model.close()
        model.env.close()


@pytest.mark.parametrize("probability", [0., .5, 1.])
def test_held_actions_advance_only_execution_rng_and_preserve_learner(models, probability):
    agent = models(inner_critic_bernoulli_p=probability).agent
    engine = agent.inner_engine
    agent.act(torch.zeros(3), t0=True)
    before = _snapshot(engine)
    counters = {name: getattr(engine.state, name)
                for name in engine._ACTION_TRANSIENT_COUNTER_FIELDS}
    global_rng = torch.random.get_rng_state().clone()
    actor = deepcopy(engine._held_actor.state_dict())
    for age in (1, 2):
        agent.act(torch.ones(3) * age / 10, collect_diagnostics=True)
        metrics = agent.last_inner_metrics
        assert metrics["inner_action_age"] == age
        assert metrics["inner_solve_performed"] == 0
        assert metrics["inner_policy_held"] == 1
        assert metrics["inner_model_steps_budget"] == 0
        assert metrics["inner_policy_evaluations"] == 1
        assert not any("alpha" in key or "loss" in key or "probe" in key
                       for key in metrics)
        assert "inner_solve_seconds" not in metrics
        assert "inner_encoder_snapshot_seconds" not in metrics
        assert metrics["inner_held_seconds"] >= 0
    after = _snapshot(engine)
    before_rng, after_rng = before.pop("rng"), after.pop("rng")
    _assert_tree_equal(after, before)
    _assert_tree_equal(engine._held_actor.state_dict(), actor)
    assert counters == {name: getattr(engine.state, name) for name in counters}
    for group in ("streams", "phase_streams"):
        for stream, state in before_rng[group].items():
            if stream == "execution":
                assert not torch.equal(after_rng[group][stream], state)
            else:
                torch.testing.assert_close(after_rng[group][stream], state, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)


@pytest.mark.parametrize("source", ["sac", "aux_return"])
def test_bernoulli_donor_survives_holds_and_uses_latest_prior(models, monkeypatch, source):
    options = {}
    if source == "aux_return":
        options.update(aux_return_mode="sac", inner_critic_source="aux_return",
                       inner_horizon_critic_source="aux_return",
                       inner_sac_critic_target="reward_only")
    agent = models(**options).agent
    engine = agent.inner_engine
    initialized = []
    prepare = engine._prepare_workspace

    def capture(**kwargs):
        prepare(**kwargs)
        initialized.append(_parameters(engine.state.critic))

    monkeypatch.setattr(engine, "_prepare_workspace", capture)
    agent.act(torch.zeros(3), t0=True)
    donor = _parameters(engine._action_pool.critic)
    rng = engine.rng.generator("critic_bernoulli").get_state().clone()
    for index in range(2):
        agent._update(*_batch(agent))
        agent.act(torch.ones(3) * index / 10)
        assert engine._critic_transfer_ready
        assert agent.last_inner_metrics["inner_critic_bernoulli_applied"] == 0
        _assert_tree_equal(_parameters(engine._action_pool.critic), donor)
        torch.testing.assert_close(engine.rng.generator("critic_bernoulli").get_state(), rng,
                                   rtol=0, atol=0)
    oracle_rng = torch.Generator().set_state(rng)
    expected, fraction = _oracle(donor, _parameters(engine._critic_base), .5, oracle_rng)
    old_actor = engine._held_actor
    agent.act(torch.ones(3) * .2)
    assert len(initialized) == 2
    _assert_tree_equal(initialized[1], expected)
    assert engine._held_actor is old_actor
    assert engine._held_actor is not engine._action_pool.actor
    assert all(not p.requires_grad for p in engine._held_actor.parameters())
    assert all(p.requires_grad for p in engine._action_pool.actor.parameters())
    assert agent.last_inner_metrics["inner_critic_bernoulli_applied"] == 1
    assert agent.last_inner_metrics["inner_critic_transfer_outer_updates"] == 2
    assert agent.last_inner_metrics["inner_critic_bernoulli_retained_fraction"] == pytest.approx(fraction)
    torch.testing.assert_close(engine.rng.generator("critic_bernoulli").get_state(),
                               oracle_rng.get_state(), rtol=0, atol=0)


@pytest.mark.parametrize("reset", ["t0", "episode", "evaluation", "clear", "mode", "abort"])
def test_reset_paths_restart_solve_zero_without_old_donor(models, reset):
    agent = models().agent
    engine = agent.inner_engine
    agent.act(torch.zeros(3), t0=True)
    agent.act(torch.zeros(3))
    old_encoder = engine._held_encoder
    if reset == "episode":
        engine.reset_episode()
    elif reset == "evaluation":
        engine.reset_for_evaluation(13, reuse_action_pool=True)
    elif reset == "clear":
        engine.clear_all()
    elif reset == "abort":
        engine.abort_control()
    if reset not in {"t0", "mode"}:
        assert engine._held_actor is engine._held_encoder is None
    agent.act(torch.ones(3), t0=reset == "t0", eval_mode=reset == "mode")
    assert agent.last_inner_metrics["inner_episode_decision_index"] == 0
    assert agent.last_inner_metrics["inner_action_age"] == 0
    assert agent.last_inner_metrics["inner_solve_index"] == 0
    assert agent.last_inner_metrics["inner_solve_performed"] == 1
    assert agent.last_inner_metrics["inner_critic_bernoulli_applied"] == 0
    if reset in {"evaluation", "clear", "abort"}:
        assert engine._held_encoder is not old_encoder


def test_prepare_is_idempotent_and_preserves_encoder_timer(models):
    agent = models().agent
    engine = agent.inner_engine
    assert engine.prepare_control(t0=True) is False
    engine._pending_timers["inner_encoder_snapshot_seconds"] = [.125]
    assert engine.prepare_control(t0=True) is False
    assert engine._pending_timers == {"inner_encoder_snapshot_seconds": [.125]}
    with pytest.raises(RuntimeError, match="preparation changed"):
        engine.prepare_control(t0=False)
    engine.abort_control()
    assert engine._pending_timers == {}
    agent.act(torch.zeros(3), t0=True)
    assert engine._control_prepared is None
    assert agent.last_inner_metrics["inner_encoder_snapshot_seconds"] >= 0
    assert engine.prepare_control() is True
    assert engine.prepare_control() is True
    agent.act(torch.ones(3))
    assert engine._control_prepared is None


def test_partial_solve_failure_invalidates_hold_and_transfer(models, monkeypatch):
    agent = models().agent
    engine = agent.inner_engine
    agent.act(torch.zeros(3), t0=True)
    for _ in range(2):
        agent.act(torch.zeros(3))
    prepare = engine._prepare_workspace

    def fail(**kwargs):
        prepare(**kwargs)
        raise RuntimeError("injected partial solve failure")

    with monkeypatch.context() as patch:
        patch.setattr(engine, "_prepare_workspace", fail)
        with pytest.raises(RuntimeError, match="injected partial solve failure"):
            agent.act(torch.zeros(3))
    assert engine._held_actor is engine._held_encoder is engine._encoder_pool is None
    assert engine._control_prepared is None
    assert not engine._critic_transfer_ready
    agent.act(torch.zeros(3))
    assert agent.last_inner_metrics["inner_solve_performed"] == 1
    assert agent.last_inner_metrics["inner_episode_decision_index"] == 0
    assert agent.last_inner_metrics["inner_critic_bernoulli_applied"] == 0
