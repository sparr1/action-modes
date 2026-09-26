"""Uniform per-decision budgets and the narrow legacy J10 equivalence."""

import pytest
import torch

from tests.test_aux_actor_transfer import _params, _trace
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _model_from_params


@pytest.mark.parametrize("scope", ["action", "episode"])
@pytest.mark.parametrize("horizon,rounds", [(1, 1), (3, 4), (3, 10)])
def test_uniform_budget_applies_to_first_later_and_reset_decisions(scope, horizon, rounds):
    model = _model_from_params(_params(
        inner_actor_scope=scope, inner_rollout_horizon=horizon,
        inner_rounds=rounds, inner_first_action_rounds=None,
        inner_replay_capacity=None,
    ))
    try:
        engine = model.agent.inner_engine
        for episode in range(2):
            engine.reset_for_evaluation(173 + episode, reuse_action_pool=True)
            for decision in range(2):
                trace = _trace(horizon)
                model.agent.act(torch.tensor([1., .2, -.1]), t0=decision == 0,
                                eval_mode=True, trace=trace)
                metrics = model.agent.last_inner_metrics
                transferred = scope == "episode" and decision > 0
                assert metrics["inner_rounds"] == rounds
                assert metrics["inner_first_action_rounds_applied"] == 0
                assert metrics["inner_actor_transferred"] == transferred
                assert metrics["inner_model_steps_budget"] == rounds * 2 * horizon
                assert metrics["inner_critic_optimizer_steps"] == rounds * 2
                assert metrics["inner_actor_optimizer_steps"] == rounds * 2
                initial = trace.events[0]
                assert initial["replay_size"] == 0
                assert initial["metrics"]["inner_actor_lifetime_updates_initial"] == (
                    rounds * 2 if transferred else 0
                )
                for component in ("actor", "critic", "temperature"):
                    assert initial["metrics"][f"{component}_optimizer_steps_initial"] == 0
                if decision == 0:
                    probe = next(event for event in trace.events
                                 if event["phase"] == "transfer_probe")
                    assert probe["metrics"]["transfer_mean_action_delta_l2"] == 0
    finally:
        model.close()


@pytest.mark.parametrize("scope", ["action", "episode"])
def test_uniform_budget_probes_preserve_actions_state_and_training_rng(scope):
    options = dict(inner_actor_scope=scope, inner_first_action_rounds=None,
                   inner_rounds=2, inner_replay_capacity=None)
    plain = _model_from_params(_params(**options))
    traced = _model_from_params(_params(**options))
    try:
        traced.agent.model.load_state_dict(plain.agent.model.state_dict())
        for t0 in (True, False, True):
            observation = torch.tensor([1., -.1, .3])
            expected = plain.agent.act(observation, t0=t0, eval_mode=True)
            actual = traced.agent.act(observation, t0=t0, eval_mode=True, trace=_trace(2))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            left, right = plain.agent.inner_engine, traced.agent.inner_engine
            _assert_tree_equal(left.rng.training_state_dict(), right.rng.training_state_dict())
            for component in ("actor", "critic", "critic_target"):
                a = getattr(left.state, component) or getattr(left._action_pool, component)
                b = getattr(right.state, component) or getattr(right._action_pool, component)
                _assert_tree_equal(a.state_dict(), b.state_dict())
            for component in ("actor_optim", "critic_optim", "temperature_optim"):
                _assert_tree_equal(getattr(left._action_pool, component).state_dict(),
                                   getattr(right._action_pool, component).state_dict())
            _assert_tree_equal(left._action_pool.log_alpha, right._action_pool.log_alpha)
    finally:
        plain.close()
        traced.close()


@pytest.mark.parametrize("scope", ["action", "episode"])
@pytest.mark.parametrize("horizon", [1, 3])
def test_legacy_first_j10_and_uniform_j10_have_identical_control(scope, horizon):
    options = dict(inner_actor_scope=scope, inner_rounds=10,
                   inner_rollout_horizon=horizon, inner_replay_capacity=None,
                   q_representation="distributional", num_q=5)
    uniform = _model_from_params(_params(**options, inner_first_action_rounds=None))
    legacy = _model_from_params(_params(**options, inner_first_action_rounds=10))
    try:
        legacy.agent.model.load_state_dict(uniform.agent.model.state_dict())
        assert uniform.agent.cfg.inner_replay_capacity == legacy.agent.cfg.inner_replay_capacity
        for model in (uniform, legacy):
            model.agent.inner_engine.reset_for_evaluation(177, reuse_action_pool=True)
        outer = _clone_tree(uniform.agent.model.state_dict())
        for t0 in (True, False, True):
            observation = torch.tensor([1., .2, -.1])
            actions = [model.agent.act(observation, t0=t0, eval_mode=True,
                                       trace=_trace(horizon)) for model in (uniform, legacy)]
            torch.testing.assert_close(actions[0], actions[1], rtol=0, atol=0)
            assert uniform.agent.last_inner_metrics["inner_first_action_rounds_applied"] == 0
            assert legacy.agent.last_inner_metrics["inner_first_action_rounds_applied"] == t0
            for key in ("inner_rounds", "inner_model_steps_budget", "inner_actor_transferred",
                        "inner_critic_optimizer_steps", "inner_actor_optimizer_steps"):
                assert uniform.agent.last_inner_metrics[key] == legacy.agent.last_inner_metrics[key]
            left, right = uniform.agent.inner_engine, legacy.agent.inner_engine
            _assert_tree_equal(left.rng.training_state_dict(), right.rng.training_state_dict())
            for component in ("actor", "critic", "critic_target"):
                a = getattr(left.state, component) or getattr(left._action_pool, component)
                b = getattr(right.state, component) or getattr(right._action_pool, component)
                _assert_tree_equal(a.state_dict(), b.state_dict())
            for component in ("actor_optim", "critic_optim", "temperature_optim"):
                _assert_tree_equal(getattr(left._action_pool, component).state_dict(),
                                   getattr(right._action_pool, component).state_dict())
            _assert_tree_equal(left._action_pool.log_alpha, right._action_pool.log_alpha)
            _assert_tree_equal(uniform.agent.model.state_dict(), outer)
            _assert_tree_equal(legacy.agent.model.state_dict(), outer)
    finally:
        uniform.close()
        legacy.close()
