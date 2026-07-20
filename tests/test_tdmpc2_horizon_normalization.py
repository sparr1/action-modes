from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.TDMPC2 import TDMPC2Baseline, _normalize_horizon_params
from RL.tdmpc2_core.agent import TDMPC2
from RL.tdmpc2_core.common import math as td_math
from RL.tdmpc2_core.common.buffer import Buffer


def _ambi_cfg(**params):
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {
        "seed": 3,
        "device": "cpu",
        "env": "test-env",
        "total_steps": 12,
    }
    algorithm.custom_params = params
    try:
        return algorithm._build_cfg({"device": "cpu", **params})
    finally:
        algorithm.env.close()


def _tiny_network_params():
    return {
        "device": "cpu",
        "model_size": None,
        "enc_dim": 16,
        "mlp_dim": 16,
        "latent_dim": 8,
        "num_enc_layers": 2,
        "num_q": 2,
        "simnorm_dim": 4,
        "num_bins": 5,
        "vmin": -5,
        "vmax": 5,
        "batch_size": 2,
        "buffer_size": 32,
        "seed_steps": 2,
        "pretrain_steps": 1,
        "compile": False,
        "episodic": False,
        "discount": 0.99,
        "iterations": 1,
        "num_samples": 4,
        "num_elites": 2,
        "num_pi_trajs": 0,
    }


def test_horizon_resolution_defaults_legacy_mapping_and_ambiguity_rejection():
    defaults = _normalize_horizon_params({})
    assert defaults == {
        "train_unroll_horizon": 3,
        "outer_planning_horizon": 3,
        "inner_rollout_horizon": 3,
    }

    with pytest.warns(FutureWarning, match="deprecated"):
        legacy = _normalize_horizon_params({"horizon": 6})
    assert legacy == {
        "train_unroll_horizon": 6,
        "outer_planning_horizon": 6,
        "inner_rollout_horizon": 3,
    }

    for explicit in (
        "train_unroll_horizon",
        "outer_planning_horizon",
        "inner_rollout_horizon",
    ):
        with pytest.raises(ValueError, match="Cannot combine legacy horizon"):
            _normalize_horizon_params({"horizon": 3, explicit: 3})


def test_temporal_transition_weights_match_anchor_and_fixed_horizon_six_target():
    expected_h6 = torch.tensor(
        [0.248201, 0.173740, 0.121618, 0.085133, 0.059593, 0.041715]
    )
    h6 = td_math.temporal_loss_weights(
        6,
        0.7,
        normalization="reference_weighted_mean",
        reference_horizon=3,
    )
    torch.testing.assert_close(h6, expected_h6, rtol=2e-6, atol=5e-7)
    for horizon in (1, 2, 3, 4, 6):
        weights = td_math.temporal_loss_weights(
            horizon,
            0.7,
            normalization="reference_weighted_mean",
            reference_horizon=3,
        )
        torch.testing.assert_close(
            weights.sum(), torch.tensor(0.73), rtol=0, atol=1e-7
        )

    anchor = td_math.temporal_loss_weights(
        3,
        0.7,
        normalization="reference_weighted_mean",
        reference_horizon=3,
    )
    torch.testing.assert_close(
        anchor,
        torch.pow(torch.tensor(0.7), torch.arange(3)) / 3,
        rtol=0,
        atol=0,
    )


def test_temporal_actor_aggregate_is_constant_and_rho_one_is_safe():
    expected_actor_total = sum(0.7**index for index in range(4)) / 4
    for horizon in (1, 2, 3, 4, 6):
        weights = td_math.temporal_loss_weights(
            horizon,
            0.7,
            normalization="reference_weighted_mean",
            reference_horizon=3,
            include_terminal=True,
        )
        assert float(weights.sum()) == pytest.approx(expected_actor_total, abs=1e-7)

    transition = td_math.temporal_loss_weights(
        6,
        1.0,
        normalization="reference_weighted_mean",
        reference_horizon=3,
    )
    actor = td_math.temporal_loss_weights(
        6,
        1.0,
        normalization="reference_weighted_mean",
        reference_horizon=3,
        include_terminal=True,
    )
    torch.testing.assert_close(transition, torch.full((6,), 1 / 6))
    torch.testing.assert_close(actor, torch.full((7,), 1 / 7))


def _historical_reduce(losses, rho, order):
    if order == "sequential":
        total = 0
        for index, loss in enumerate(losses.unbind(0)):
            total = total + loss * rho**index
        return total / len(losses)
    rho_weights = torch.pow(rho, torch.arange(len(losses)))
    weighted = losses * rho_weights
    if order == "vector_mean":
        return weighted.mean()
    return weighted.sum() / len(losses)


@pytest.mark.parametrize(
    ("order", "include_terminal", "term_count"),
    [
        ("sequential", False, 3),
        ("vector_sum_divide", False, 3),
        ("vector_mean", True, 4),
    ],
)
@pytest.mark.parametrize("normalization", ["reference_weighted_mean", "divide_horizon"])
def test_reference_reducer_exactly_preserves_h3_loss_and_parameter_gradients(
    order,
    include_terminal,
    term_count,
    normalization,
):
    expected_parameter = torch.tensor(
        [0.25, -0.5, 1.25, -1.5][:term_count], requires_grad=True
    )
    actual_parameter = expected_parameter.detach().clone().requires_grad_(True)
    expected_terms = expected_parameter.square() + 0.125 * expected_parameter
    actual_terms = actual_parameter.square() + 0.125 * actual_parameter

    expected = _historical_reduce(expected_terms, 0.7, order)
    actual = td_math.reduce_temporal_loss(
        actual_terms,
        0.7,
        normalization=normalization,
        reference_horizon=3,
        include_terminal=include_terminal,
        legacy_order=order,
    )
    assert torch.equal(actual, expected)

    expected.backward()
    actual.backward()
    assert torch.equal(actual_parameter.grad, expected_parameter.grad)


def test_reference_weighted_reducer_sends_gradient_to_every_depth():
    parameter = torch.linspace(0.5, 1.0, 6, requires_grad=True)
    losses = parameter.square()
    reduced = td_math.reduce_temporal_loss(
        losses,
        0.7,
        normalization="reference_weighted_mean",
        reference_horizon=3,
        weights=td_math.temporal_loss_weights(6, 0.7),
    )
    reduced.backward()
    assert torch.all(parameter.grad != 0)


def test_ambi_train_six_plan_three_inner_three_resolves_independently():
    cfg = _ambi_cfg(
        train_unroll_horizon=6,
        outer_planning_horizon=3,
        inner_rollout_horizon=3,
    )
    assert cfg.train_unroll_horizon == 6
    assert cfg.outer_planning_horizon == 3
    assert cfg.inner_rollout_horizon == 3
    assert cfg.horizon == 6  # read-only one-release compatibility alias


def test_ambi_extrapolation_warns_and_action_local_capacity_cannot_truncate():
    with pytest.warns(UserWarning, match="extrapolating"):
        cfg = _ambi_cfg(
            train_unroll_horizon=3,
            outer_planning_horizon=3,
            inner_rollout_horizon=6,
            inner_rounds=1,
            inner_rollouts_per_round=2,
            inner_updates_per_round=1,
        )
    assert cfg.inner_rollout_horizon == 6

    with pytest.raises(ValueError, match=r"cumulative nominal J\*N\*H"):
        _ambi_cfg(
            train_unroll_horizon=3,
            outer_planning_horizon=3,
            inner_rollout_horizon=3,
            inner_rounds=2,
            inner_rollouts_per_round=4,
            inner_updates_per_round=1,
            inner_replay_capacity=23,
            inner_replay_scope="action",
        )


def test_replay_sampling_uses_train_unroll_horizon_plus_one():
    calls = []

    class Sample:
        def view(self, *shape):
            calls.append(("view", shape))
            return self

        def permute(self, *dims):
            calls.append(("permute", dims))
            return self

    class Replay:
        @staticmethod
        def sample():
            return Sample()

    replay = object.__new__(Buffer)
    replay.cfg = SimpleNamespace(train_unroll_horizon=6)
    replay._buffer = Replay()
    replay._prepare_batch = lambda td: td
    replay.sample()

    assert calls == [("view", (-1, 7)), ("permute", (1, 0))]


def test_standard_value_estimation_uses_outer_planning_horizon():
    class Model:
        def __init__(self):
            self.reward_calls = 0
            self.next_calls = 0

        def reward(self, z, action, task):
            del action, task
            self.reward_calls += 1
            return torch.zeros(z.shape[0], 1)

        def next(self, z, action, task):
            del action, task
            self.next_calls += 1
            return z

        @staticmethod
        def pi(z, task):
            del task
            return torch.zeros(z.shape[0], 1), {}

        @staticmethod
        def Q(z, action, task, return_type):
            del action, task, return_type
            return torch.zeros(z.shape[0], 1)

    agent = object.__new__(TDMPC2)
    torch.nn.Module.__init__(agent)
    agent.cfg = SimpleNamespace(
        num_samples=2,
        outer_planning_horizon=2,
        multitask=False,
        episodic=False,
        num_bins=1,
        vmin=-1,
        vmax=1,
        bin_size=2,
    )
    agent.model = Model()
    agent.discount = 0.99
    actions = torch.zeros(5, 2, 1)

    agent._estimate_value(torch.zeros(2, 3), actions, None)

    assert agent.model.reward_calls == 2
    assert agent.model.next_calls == 2


def test_standard_planner_warm_start_state_uses_outer_planning_horizon():
    env = gym.make("Pendulum-v1", max_episode_steps=8)
    try:
        model = TDMPC2Baseline(
            "TDMPC2Baseline",
            env,
            {
                **_tiny_network_params(),
                "min_std": 0.0,
                "max_std": 0.0,
                "train_unroll_horizon": 6,
                "outer_planning_horizon": 3,
                "inner_rollout_horizon": 3,
            },
            {"seed": 3, "device": "cpu", "env": "test", "total_steps": 8},
            {},
        )
        agent = model.agent
        assert agent._prev_mean.shape == (3, model.cfg.action_dim)
        agent._prev_mean[0].fill_(0.1)
        agent._prev_mean[1].fill_(0.7)
        agent._prev_mean[2].fill_(-0.2)

        action = agent.act(
            torch.zeros(model.cfg.obs_shape["state"]),
            t0=False,
            eval_mode=True,
        )

        torch.testing.assert_close(action, torch.full_like(action, 0.7))

        obs = torch.randn(7, model.cfg.batch_size, 3)
        replay_action = torch.randn(6, model.cfg.batch_size, 1).tanh()
        reward = torch.randn(6, model.cfg.batch_size, 1)
        metrics = agent._update(
            obs,
            replay_action,
            reward,
            torch.zeros_like(reward),
        )
        assert "q_error_depth_6" in metrics
    finally:
        env.close()


def test_ambi_train_six_plan_three_inner_three_outer_update_runs():
    env = gym.make("Pendulum-v1", max_episode_steps=8)
    try:
        model = AMBITDMPC2(
            "AMBITDMPC2",
            env,
            {
                **_tiny_network_params(),
                "train_unroll_horizon": 6,
                "outer_planning_horizon": 3,
                "inner_rollout_horizon": 3,
                "q_representation": "scalar",
                "q_num_bins": 5,
                "q_vmin": -5,
                "q_vmax": 5,
                "inner_rounds": 1,
                "inner_rollouts_per_round": 1,
                "inner_updates_per_round": 0,
                "inner_batch_size": 2,
                "inner_replay_capacity": 3,
            },
            {"seed": 3, "device": "cpu", "env": "test", "total_steps": 8},
            {},
        )
        obs = torch.randn(7, model.cfg.batch_size, 3)
        action = torch.randn(6, model.cfg.batch_size, 1).tanh()
        reward = torch.randn(6, model.cfg.batch_size, 1)

        metrics = model.agent._update(
            obs,
            action,
            reward,
            torch.zeros_like(reward),
        )

        assert "q_error_depth_6" in metrics
        assert model.cfg.outer_planning_horizon == 3
        assert model.cfg.inner_rollout_horizon == 3
    finally:
        env.close()
