from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.TDMPC2 import TDMPC2Baseline, _normalize_horizon_params
from RL.tdmpc2_core.agent import TDMPC2
from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
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


@pytest.mark.parametrize("horizon", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("include_terminal", [False, True])
def test_temporal_weights_use_actual_term_count(horizon, rho, include_terminal):
    count = horizon + int(include_terminal)
    expected = torch.tensor([rho**t / count for t in range(count)])
    actual = td_math.temporal_loss_weights(
        horizon, rho, include_terminal=include_terminal
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.sum().item() == pytest.approx(
        sum(rho**t for t in range(count)) / count
    )


def test_total_weight_follows_tdmpc2_as_horizon_changes():
    horizons = (1, 2, 3, 4, 6)
    for include_terminal in (False, True):
        totals = [
            td_math.temporal_loss_weights(
                horizon, 0.5, include_terminal=include_terminal
            ).sum().item()
            for horizon in horizons
        ]
        assert all(left > right for left, right in zip(totals, totals[1:]))
        assert totals == pytest.approx([
            sum(0.5**t for t in range(horizon + int(include_terminal)))
            / (horizon + int(include_terminal))
            for horizon in horizons
        ])


def _upstream_reduce(losses, rho, order):
    # Spell out upstream arithmetic independently of the shared AMBI helper.
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


@pytest.mark.parametrize("horizon", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    ("order", "include_terminal"),
    [("sequential", False), ("vector_sum_divide", False), ("vector_mean", True)],
)
def test_reducer_preserves_upstream_loss_and_parameter_gradients(
    horizon, rho, order, include_terminal
):
    count = horizon + int(include_terminal)
    expected_parameter = torch.linspace(-1.5, 1.25, count, requires_grad=True)
    actual_parameter = expected_parameter.detach().clone().requires_grad_(True)
    expected_terms = expected_parameter.square() + 0.125 * expected_parameter
    actual_terms = actual_parameter.square() + 0.125 * actual_parameter

    expected = _upstream_reduce(expected_terms, rho, order)
    actual = td_math.reduce_temporal_loss(
        actual_terms, rho, include_terminal=include_terminal, legacy_order=order
    )
    assert torch.equal(actual, expected)
    expected.backward()
    actual.backward()
    assert torch.equal(actual_parameter.grad, expected_parameter.grad)
    if include_terminal:
        assert actual_parameter.grad[-1].item() == pytest.approx(
            rho**horizon * (2 * actual_parameter[-1].item() + 0.125) / count
        )


class _TemporalLossModel(torch.nn.Module):
    """Distinct trainable predictions at each time, sample, and critic head."""

    def __init__(self, horizon, heads, batch=2):
        super().__init__()
        self.latents = torch.nn.Parameter(
            torch.linspace(0.2, 1.4, horizon * batch).reshape(horizon, batch, 1)
        )
        reward_grid = torch.arange(horizon * batch * 3, dtype=torch.float32)
        self.rewards = torch.nn.Parameter(
            (reward_grid.square() / 53).reshape(horizon, batch, 3)
        )
        q_grid = torch.arange(heads * horizon * batch * 3, dtype=torch.float32)
        self.qs = torch.nn.Parameter(
            (torch.sin(q_grid / 4) + q_grid / 13).reshape(heads, horizon, batch, 3)
        )

    def encode(self, obs, task=None):
        return obs

    def next(self, z, action, task=None):
        return self.latents[int(action[0, 0])]

    def joint_input(self, z, action):
        return z

    def reward(self, z, action, task=None):
        return self.rewards

    def reward_from_joint(self, joint):
        return self.rewards

    def Q(self, z, action, task=None, return_type=None):
        assert return_type == "all"
        return self.qs

    def q_predictions_from_joint(self, joint):
        return self.qs

    def critic_loss(self, predictions, targets, reduction):
        assert reduction == "none"
        # Zero target lies exactly on the middle of the three symlog bins.
        return -predictions.log_softmax(-1)[..., 1:2]

    def soft_update_target_Q(self):
        pass


@pytest.mark.parametrize("horizon", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("algorithm", ["ambi", "tdmpc2"])
@pytest.mark.parametrize("heads", [2, 5])
def test_training_losses_and_gradients_use_tdmpc2_time_and_head_averages(
    horizon, rho, algorithm, heads
):
    model = _TemporalLossModel(horizon, heads)
    cfg = SimpleNamespace(
        train_unroll_horizon=horizon, rho=rho, batch_size=2, latent_dim=1,
        num_q=heads, num_bins=3, vmin=-1, vmax=1, bin_size=1,
        episodic=False, consistency_coef=0.3, reward_coef=0.6,
        value_coef=0.9, critic_coef=0.9, termination_coef=1.0,
        grad_clip_norm=1e6,
    )
    agent = SimpleNamespace(
        cfg=cfg, model=model, device=torch.device("cpu"), num_updates=0,
        optim=SimpleNamespace(step=lambda: None, zero_grad=lambda **_: None),
        _td_target=lambda next_z, reward, terminated, task: torch.zeros_like(reward),
        update_pi=lambda zs, task: {},
    )
    obs = torch.zeros(horizon + 1, 2, 1)
    action = torch.arange(horizon).reshape(horizon, 1, 1).expand(-1, 2, -1)
    reward = torch.zeros(horizon, 2, 1)
    terminated = torch.zeros_like(reward)

    reference_parameters = [
        parameter.detach().clone().requires_grad_() for parameter in model.parameters()
    ]
    latent, reward_logits, q_logits = reference_parameters
    # Independent upstream equations: each transition gets rho**t, each model
    # loss divides by H, and the critic additionally divides by every head.
    consistency = sum(
        rho**t * latent[t].square().mean() for t in range(horizon)
    ) / horizon
    reward_loss = sum(
        rho**t * -reward_logits[t].log_softmax(-1)[..., 1].mean()
        for t in range(horizon)
    ) / horizon
    critic_loss = sum(
        rho**t * -q_logits[head, t].log_softmax(-1)[..., 1].mean()
        for t in range(horizon) for head in range(heads)
    ) / (horizon * heads)
    expected_total = 0.3 * consistency + 0.6 * reward_loss + 0.9 * critic_loss
    expected_total.backward()

    if algorithm == "ambi":
        result = AMBITDMPC2Agent._outer_update_kernel(
            agent, obs[0], action, reward, terminated, obs[1:], reward
        )
        actual_losses = result[5:8]
        result[-1].backward()
    else:
        metrics = TDMPC2._update(agent, obs, action, reward, terminated)
        actual_losses = [metrics[key] for key in (
            "consistency_loss", "reward_loss", "value_loss"
        )]
    for actual, expected in zip(actual_losses, (consistency, reward_loss, critic_loss)):
        torch.testing.assert_close(actual, expected.detach())
    for actual, expected in zip(model.parameters(), reference_parameters):
        torch.testing.assert_close(actual.grad, expected.grad)


@pytest.mark.parametrize("horizon", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
def test_tdmpc2_actor_loss_and_gradients_include_terminal_latent(horizon, rho):
    actor = torch.nn.Linear(1, horizon + 1, bias=False)
    with torch.no_grad():
        actor.weight.copy_(torch.linspace(0.2, 1.1, horizon + 1).reshape(-1, 1))
    initial = actor.weight.detach().clone().requires_grad_()

    def pi(zs, task):
        action = actor.weight.reshape(horizon + 1, 1, 1).expand(-1, 2, -1)
        entropy = action.square() + 0.1
        return action, {"entropy": entropy, "scaled_entropy": entropy}

    class Scale:
        value = torch.tensor([2.0])

        def update(self, qs):
            torch.testing.assert_close(qs, (3 * initial[0]).expand(2, 1))

        def __call__(self, qs):
            return qs / self.value

    model = SimpleNamespace(
        pi=pi, _pi=actor,
        Q=lambda zs, action, task, **kwargs: 3 * action,
    )
    agent = SimpleNamespace(
        cfg=SimpleNamespace(rho=rho, entropy_coef=0.2, grad_clip_norm=1e6),
        model=model, scale=Scale(),
        pi_optim=SimpleNamespace(step=lambda: None, zero_grad=lambda **_: None),
    )
    expected = sum(
        rho**t * -(0.2 * (initial[t].square() + 0.1) + 3 * initial[t] / 2)
        for t in range(horizon + 1)
    ).sum() / (horizon + 1)
    expected.backward()
    metrics = TDMPC2.update_pi(agent, torch.zeros(horizon + 1, 2, 1), None)
    torch.testing.assert_close(metrics["pi_loss"], expected.detach())
    torch.testing.assert_close(actor.weight.grad, initial.grad)
    if rho > 0:
        assert actor.weight.grad[-1].abs().item() > 0


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
