import math

import gymnasium as gym
import numpy as np
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.TDMPC2 import TDMPC2Baseline


def _tiny_params(**overrides):
    params = {
        "device": "cpu",
        "model_size": None,
        "enc_dim": 32,
        "mlp_dim": 32,
        "latent_dim": 16,
        "num_enc_layers": 2,
        "num_q": 2,
        "simnorm_dim": 8,
        "num_bins": 11,
        "vmin": -5,
        "vmax": 5,
        "batch_size": 4,
        "horizon": 2,
        "buffer_size": 100,
        "seed_steps": 4,
        "pretrain_steps": 2,
        "utd": 1,
        "compile": False,
        "episodic": False,
        "discount": 0.99,
        "iterations": 2,
        "num_samples": 16,
        "num_elites": 4,
        "num_pi_trajs": 2,
        "inner_adaptation": "clone",
        "inner_iterations": 2,
        "inner_rollouts": 4,
        "inner_horizon": 2,
        "inner_updates_per_iteration": 2,
        "inner_batch_size": 8,
        "inner_tau": 1.0,
        "wandb": False,
        "dropout": 0.0,
    }
    params.update(overrides)
    return params


def _make_model(cls, params=None, total_steps=20):
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    return cls(
        cls.__name__,
        env,
        params or _tiny_params(),
        {"seed": 3, "device": "cpu", "env": "test-env", "total_steps": total_steps},
        {},
    )


def _assert_finite_metrics(metrics):
    assert all(math.isfinite(float(torch.as_tensor(value).mean())) for value in metrics.values())


def test_tdmpc_update_and_final_plan_diagnostics(monkeypatch):
    model = _make_model(TDMPC2Baseline)
    agent = model.agent

    estimate_calls = 0
    original_estimate = agent._estimate_value

    def counted_estimate(*args, **kwargs):
        nonlocal estimate_calls
        estimate_calls += 1
        return original_estimate(*args, **kwargs)

    monkeypatch.setattr(agent, "_estimate_value", counted_estimate)
    agent.act(torch.zeros(model.cfg.obs_shape["state"]), t0=True, eval_mode=True)
    assert estimate_calls == model.cfg.iterations
    assert set(agent.last_plan_metrics) == {
        "planner_value_mean",
        "planner_value_std",
        "planner_value_max",
        "planner_elite_value_mean",
        "planner_elite_value_std",
        "planner_elite_value_max",
        "planner_std_mean",
        "planner_std_min",
        "planner_std_max",
        "planner_action_l2",
        "planner_seconds",
    }
    _assert_finite_metrics(agent.last_plan_metrics)

    obs = torch.randn(model.cfg.horizon + 1, model.cfg.batch_size, model.cfg.obs_shape["state"][0])
    action = torch.randn(model.cfg.horizon, model.cfg.batch_size, model.cfg.action_dim).tanh()
    reward = torch.randn(model.cfg.horizon, model.cfg.batch_size, 1)
    terminated = torch.zeros_like(reward)
    metrics = agent._update(obs, action, reward, terminated)
    assert {
        "q_mean",
        "q_target_mean",
        "td_error_abs_mean",
        "reward_pred_mean",
        "reward_target_mean",
        "pi_q_mean",
        "num_updates",
    } <= set(metrics.keys())
    assert agent.num_updates == 1
    _assert_finite_metrics(metrics)


def test_ambi_outer_and_inner_diagnostics_are_complete():
    model = _make_model(AMBITDMPC2)
    agent = model.agent
    agent.act(torch.zeros(model.cfg.obs_shape["state"]), eval_mode=False)

    expected_inner = {
        "inner_active",
        "inner_actions",
        "inner_rollouts",
        "inner_steps",
        "inner_updates",
        "inner_buffer_size",
        "inner_buffer_capacity",
        "inner_buffer_fill_ratio",
        "inner_return_mean",
        "inner_return_std",
        "inner_return_min",
        "inner_return_max",
        "inner_rollout_len_mean",
        "inner_rollout_len_std",
        "inner_rollout_len_min",
        "inner_rollout_len_max",
        "inner_termination_rate",
        "inner_actor_loss",
        "inner_critic_loss",
        "inner_actor_grad_norm",
        "inner_critic_grad_norm",
        "inner_alpha",
        "inner_q_mean",
        "inner_q_target_mean",
        "inner_actor_q_mean",
        "inner_actor_entropy",
        "inner_td_error_abs_mean",
        "inner_policy_mean_delta_l2",
        "inner_outer_q_gain",
        "inner_action_seconds",
    }
    assert expected_inner <= set(agent.last_inner_metrics)
    assert agent.last_inner_metrics["inner_rollouts"] == 8
    assert agent.last_inner_metrics["inner_steps"] == 16
    assert agent.last_inner_metrics["inner_updates"] == 4
    _assert_finite_metrics(agent.last_inner_metrics)

    obs = torch.randn(model.cfg.horizon + 1, model.cfg.batch_size, model.cfg.obs_shape["state"][0])
    action = torch.randn(model.cfg.horizon, model.cfg.batch_size, model.cfg.action_dim).tanh()
    reward = torch.randn(model.cfg.horizon, model.cfg.batch_size, 1)
    metrics = agent._update(obs, action, reward, torch.zeros_like(reward))
    assert {
        "q_mean",
        "q_target_mean",
        "td_error_abs_mean",
        "reward_pred_mean",
        "reward_target_mean",
        "actor_q_mean",
        "num_updates",
    } <= set(metrics)
    _assert_finite_metrics(metrics)


def test_replay_reports_real_transitions_and_storage_occupancy():
    model = _make_model(TDMPC2Baseline, total_steps=20)
    obs = np.zeros(model.cfg.obs_shape["state"], dtype=np.float32)
    episode = torch.cat(
        [
            model._to_td(obs),
            model._to_td(obs, np.zeros(model.cfg.action_dim), 1.0, False),
            model._to_td(obs, np.zeros(model.cfg.action_dim), 2.0, False),
        ]
    )
    episodes = torch.stack([episode.clone(), episode.clone()])
    model.buffer.add(episode)

    assert model.buffer.num_transitions == 2
    assert model.buffer.total_transitions == 2
    assert model.buffer.size == 3
    assert model.buffer.fill_fraction == pytest.approx(3 / model.buffer.capacity)

    for _ in range(9):
        model.buffer.add(episode.clone())
    assert model.buffer.size == model.buffer.capacity == 20
    assert model.buffer.num_transitions == 14
    assert model.buffer.total_transitions == 20
    assert model.buffer.fill_fraction == 1.0

    load_model = _make_model(TDMPC2Baseline, total_steps=20)
    load_model.buffer.load(episodes)
    assert load_model.buffer.num_transitions == 4
    assert load_model.buffer.total_transitions == 4
    assert load_model.buffer.size == 6
    assert load_model.buffer.fill_fraction == pytest.approx(6 / load_model.buffer.capacity)
