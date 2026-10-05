"""Retain cheap measurements without changing the evaluated controller."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from utils.transfer_campaign import evaluate_episode, selected_metrics


def test_metric_policy_cli_is_explicit_and_defaults_to_legacy():
    from evaluate_ambi_transfer_campaign import parser
    assert parser().parse_args([]).metric_policy == "legacy"
    assert parser().parse_args(["--metric-policy", "all_scalars"]).metric_policy == "all_scalars"
    with pytest.raises(SystemExit):
        parser().parse_args(["--metric-policy", "invented"])


def test_all_scalars_retains_computed_inner_metrics_without_mutation():
    metrics = {
        "inner_actor_loss": torch.tensor(1.5, requires_grad=True),
        "inner_critic_loss": 0.2,
        "inner_actor_q_mean": np.float64(2.5),
        "inner_actor_entropy": -3.,
        "inner_q_target_mean": 3.,
        "inner_td_error_abs_mean": 0.5,
        "inner_td_error_abs_mean_max": 0.7,
        "inner_temperature_optimizer_steps": 4,
        "inner_solve_performed": True,
        "inner_future_scalar": torch.tensor([9.]),
        "inner_vector": torch.tensor([1., 2.]),
        "inner_label": "not numeric",
        "outer_loss": 10.,
    }
    before = deepcopy(metrics)
    result = selected_metrics(metrics, policy="all_scalars")
    assert result == {
        "inner_actor_loss": 1.5, "inner_critic_loss": 0.2,
        "inner_actor_q_mean": 2.5, "inner_actor_entropy": -3.,
        "inner_q_target_mean": 3., "inner_td_error_abs_mean": 0.5,
        "inner_td_error_abs_mean_max": 0.7,
        "inner_temperature_optimizer_steps": 4., "inner_solve_performed": 1.,
        "inner_future_scalar": 9.,
    }
    for key, value in before.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(metrics[key], value, rtol=0, atol=0)
        else:
            assert metrics[key] == value
    assert metrics["inner_actor_loss"].requires_grad
    assert metrics["inner_actor_loss"].grad is None
    assert selected_metrics(metrics) == {
        "inner_actor_loss": 1.5, "inner_critic_loss": 0.2,
        "inner_solve_performed": 1.,
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), torch.tensor(-float("inf"))])
def test_all_scalars_rejects_nonfinite_measurements(value):
    with pytest.raises(ValueError, match="Nonfinite controller metric inner_q_target_mean"):
        selected_metrics({"inner_q_target_mean": value}, policy="all_scalars")


def test_unknown_metric_policy_fails_before_environment_or_learner_use():
    with pytest.raises(ValueError, match="Unknown inner metric policy"):
        selected_metrics({}, policy="invented")
    with pytest.raises(ValueError, match="Unknown inner metric policy"):
        evaluate_episode(None, None, {}, episode_seed=101, controller_seed=55,
                         max_steps=3, metric_policy="invented")


@pytest.mark.parametrize("horizon,arm", [
    (1, {"actor_rho": .5, "critic_rho": 0.}),
    (2, {"actor_rho": 1., "critic_rho": .5}),
    (3, {"actor_rho": 0., "critic_rho": .5}),
])
def test_metric_policy_preserves_actions_rewards_learner_state_and_rng(horizon, arm):
    from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
    from tests.test_ambi_root_local_sac import _model_from_params
    from tests.test_aux_critic_transfer import critic_params

    params = critic_params(inner_critic_scope="action", inner_rounds=2,
                           inner_rollout_horizon=horizon)
    legacy, expanded = (_model_from_params(params) for _ in range(2))
    try:
        expanded.agent.load(_clone_tree(legacy.agent.checkpoint_state()))
        frozen = _clone_tree(legacy.agent.checkpoint_state())
        rows_by_policy = {}
        for model, policy in ((legacy, "legacy"), (expanded, "all_scalars")):
            rows = rows_by_policy[policy] = []
            evaluate_episode(model, model.env, arm, episode_seed=101,
                controller_seed=55, max_steps=3, on_step=rows.append,
                smoke=True, metric_policy=policy)
            _assert_tree_equal(model.agent.checkpoint_state(), frozen)

        for original, extended in zip(rows_by_policy["legacy"], rows_by_policy["all_scalars"]):
            for key in ("action", "reward", "cumulative_reward", "terminated", "truncated"):
                assert original[key] == extended[key]
            assert original["metrics"].items() <= extended["metrics"].items()
            assert {"inner_actor_q_mean", "inner_actor_entropy", "inner_q_mean",
                    "inner_q_target_mean", "inner_td_error_abs_mean",
                    "inner_actor_optimizer_steps", "inner_critic_optimizer_steps",
                    "inner_temperature_optimizer_steps"} <= extended["metrics"].keys()

        left, right = legacy.agent.inner_engine, expanded.agent.inner_engine
        _assert_tree_equal(left.rng.training_state_dict(), right.rng.training_state_dict())
        _assert_tree_equal(left.export_diagnostic_state(include_optimizers=True, include_replay=True),
                          right.export_diagnostic_state(include_optimizers=True, include_replay=True))
    finally:
        legacy.close()
        expanded.close()
