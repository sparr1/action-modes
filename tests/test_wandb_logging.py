from collections import defaultdict
import importlib

import gymnasium as gym
import numpy as np
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.SAC import SAC as NativeSAC
from RL.TDMPC2 import TDMPC2Baseline
from RL.baselines import Baseline


class ConstantEpisodeEnv(gym.Env):
    metadata = {}

    def __init__(self, episode_length=5, fail_at=None):
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.episode_length = int(episode_length)
        self.fail_at = fail_at
        self.total_steps = 0
        self.episode_step = 0
        self.spec = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_step = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        del action
        self.total_steps += 1
        self.episode_step += 1
        if self.fail_at is not None and self.total_steps == self.fail_at:
            raise RuntimeError("intentional environment failure")
        truncated = self.episode_step >= self.episode_length
        info = {"reward_forward": 0.75, "reward_info": {"control cost": -0.25}}
        return np.zeros(3, dtype=np.float32), 1.0, False, truncated, info


class VaryingRewardComponentsEnv(ConstantEpisodeEnv):
    def step(self, action):
        obs, reward, terminated, truncated, _info = super().step(action)
        value = float(self.total_steps)
        info = {
            "reward_forward": value,
            "reward_info": {"control cost": -value},
        }
        return obs, reward, terminated, truncated, info


class ReservedRewardComponentEnv(ConstantEpisodeEnv):
    def step(self, action):
        obs, reward, terminated, truncated, _info = super().step(action)
        return obs, reward, terminated, truncated, {
            "reward_mean": float(self.total_steps),
        }


class OverlappingRewardComponentsEnv(ConstantEpisodeEnv):
    def step(self, action):
        obs, reward, terminated, truncated, _info = super().step(action)
        value = float(self.total_steps)
        return obs, reward, terminated, truncated, {
            "reward_forward": value,
            "reward_forward_mean": 10.0 * value,
        }


class StaggeredOverlappingRewardComponentsEnv(ConstantEpisodeEnv):
    def step(self, action):
        obs, reward, terminated, truncated, _info = super().step(action)
        value = float(self.total_steps)
        info = {"reward_forward_mean": 10.0 * value}
        if self.total_steps > 1:
            info["reward_forward"] = value
        return obs, reward, terminated, truncated, info


class FakeRun:
    def __init__(self):
        self.history = defaultdict(dict)

    def log(self, payload, step):
        self.history[int(step)].update(dict(payload))


def _install_fake_wandb(monkeypatch, module):
    run = FakeRun()
    finished = []
    monkeypatch.setattr(module, "init_wandb", lambda *_args, **_kwargs: run)
    monkeypatch.setattr(module, "finish_wandb", finished.append)
    return run, finished


def _tiny_td_params(**overrides):
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
        "batch_size": 2,
        "train_unroll_horizon": 2,
        "outer_planning_horizon": 2,
        "buffer_size": 100,
        "seed_steps": 2,
        "pretrain_steps": 2,
        "utd": 1,
        "compile": False,
        "episodic": False,
        "discount": 0.99,
        "iterations": 2,
        "num_samples": 8,
        "num_elites": 2,
        "num_pi_trajs": 1,
        "dropout": 0.0,
        "wandb": True,
        "wandb_step_every": 3,
    }
    params.update(overrides)
    return params


def test_native_sac_logs_weighted_update_window_and_forced_final_flush(monkeypatch):
    sac_module = importlib.import_module("RL.SAC")
    run, finished = _install_fake_wandb(monkeypatch, sac_module)
    model = NativeSAC(
        "SAC",
        ConstantEpisodeEnv(episode_length=10),
        {
            "device": "cpu",
            "learning_starts": 0,
            "train_freq": 1,
            "gradient_steps": 1,
            "batch_size": 2,
            "buffer_size": 16,
            "net_arch": [8],
            "wandb": True,
            "wandb_step_every": 10,
        },
        {"seed": 3, "env": "constant"},
        {},
    )

    def fake_update(_replay, gradient_steps, _batch_size):
        model.agent.num_updates += gradient_steps
        return {"actor_loss": float(model.agent.num_updates)}

    model.agent.update = fake_update
    model.learn(total_timesteps=4)

    assert set(run.history) == {4}
    payload = run.history[4]
    assert payload["train/actor_loss"] == pytest.approx(2.5)
    assert payload["train/n_updates"] == 4
    assert payload["train/updates_since_log"] == 4
    assert payload["train/replay_size"] == 4
    assert payload["rollout/reward_mean"] == 1.0
    assert payload["rollout/reward_forward"] == 0.75
    assert payload["rollout/reward_control_cost"] == -0.25
    assert payload["episode/current_return"] == 4.0
    assert payload["episode/current_len"] == 4
    assert finished == [run]
    assert model._wandb_run is None


def test_native_sac_finishes_wandb_on_training_exception(monkeypatch):
    sac_module = importlib.import_module("RL.SAC")
    run, finished = _install_fake_wandb(monkeypatch, sac_module)
    model = NativeSAC(
        "SAC",
        ConstantEpisodeEnv(episode_length=10, fail_at=2),
        {"device": "cpu", "learning_starts": 100, "net_arch": [8], "wandb": True},
        {},
        {},
    )

    with pytest.raises(RuntimeError, match="intentional environment failure"):
        model.learn(total_timesteps=4)

    assert finished == [run]
    assert model._wandb_run is None
    assert 1 in run.history


def test_native_sac_does_not_repeat_stale_losses(monkeypatch):
    sac_module = importlib.import_module("RL.SAC")
    run, _finished = _install_fake_wandb(monkeypatch, sac_module)
    model = NativeSAC(
        "SAC",
        ConstantEpisodeEnv(episode_length=10),
        {
            "device": "cpu",
            "learning_starts": 0,
            "train_freq": 2,
            "gradient_steps": 1,
            "batch_size": 2,
            "buffer_size": 16,
            "net_arch": [8],
            "wandb": True,
            "wandb_step_every": 1,
        },
        {},
        {},
    )

    def fake_update(_replay, gradient_steps, _batch_size):
        model.agent.num_updates += gradient_steps
        return {"actor_loss": 9.0}

    model.agent.update = fake_update
    model.learn(total_timesteps=3)

    assert "train/actor_loss" not in run.history[1]
    assert run.history[2]["train/actor_loss"] == 9.0
    assert "train/actor_loss" not in run.history[3]
    assert run.history[3]["train/updates_since_log"] == 0


def test_tdmpc_terminal_payload_precedes_reset_and_pretrain_is_averaged(monkeypatch):
    td_module = importlib.import_module("RL.TDMPC2")
    run, finished = _install_fake_wandb(monkeypatch, td_module)
    env = ConstantEpisodeEnv(episode_length=3)
    model = TDMPC2Baseline(
        "TDMPC2Baseline",
        env,
        _tiny_td_params(),
        {"seed": 3, "device": "cpu", "env": "constant", "total_steps": 3},
        {},
    )
    values = iter((1.0, 3.0))
    model.agent.update = lambda _buffer: {"total_loss": next(values)}
    model.learn(total_timesteps=3)

    payload = run.history[3]
    assert payload["train/total_loss"] == pytest.approx(2.0)
    assert payload["train/updates_since_log"] == 2
    assert payload["train/replay_size"] == 3
    assert payload["episode/current_return"] == 3.0
    assert payload["episode/current_len"] == 3
    assert payload["episode/return"] == 3.0
    assert payload["episode/len"] == 3
    assert finished == [run]


def test_tdmpc_metric_window_crosses_episode_boundaries_until_cadence(monkeypatch):
    td_module = importlib.import_module("RL.TDMPC2")
    run, _finished = _install_fake_wandb(monkeypatch, td_module)
    model = TDMPC2Baseline(
        "TDMPC2Baseline",
        ConstantEpisodeEnv(episode_length=2),
        _tiny_td_params(
            seed_steps=0,
            pretrain_steps=1,
            utd=1,
            wandb_step_every=5,
        ),
        {"seed": 3, "device": "cpu", "env": "constant", "total_steps": 5},
        {},
    )
    values = iter((1.0, 2.0, 3.0, 4.0))
    model.agent.update = lambda _buffer: {"total_loss": next(values)}

    model.learn(total_timesteps=5)

    assert set(run.history) == {2, 4, 5}
    assert run.history[2]["episode/return"] == 2.0
    assert run.history[4]["episode/return"] == 2.0
    assert "train/total_loss" not in run.history[2]
    assert "train/total_loss" not in run.history[4]
    payload = run.history[5]
    assert payload["train/total_loss"] == pytest.approx(2.5)
    assert payload["train/total_loss_count"] == 4.0
    assert payload["train/total_loss_min"] == 1.0
    assert payload["train/total_loss_max"] == 4.0


def test_tdmpc_reward_components_pool_over_the_logging_window(monkeypatch):
    td_module = importlib.import_module("RL.TDMPC2")
    run, _finished = _install_fake_wandb(monkeypatch, td_module)
    model = TDMPC2Baseline(
        "TDMPC2Baseline",
        VaryingRewardComponentsEnv(episode_length=10),
        _tiny_td_params(seed_steps=100, wandb_step_every=10),
        {"seed": 3, "device": "cpu", "env": "varying", "total_steps": 4},
        {},
    )

    model.learn(total_timesteps=4)

    payload = run.history[4]
    assert payload["rollout/reward_forward"] == pytest.approx(2.5)
    assert payload["rollout/reward_forward_count"] == 4.0
    assert payload["rollout/reward_forward_mean"] == pytest.approx(2.5)
    assert payload["rollout/reward_forward_std"] == pytest.approx(np.sqrt(1.25))
    assert payload["rollout/reward_forward_min"] == 1.0
    assert payload["rollout/reward_forward_max"] == 4.0
    assert payload["rollout/reward_control_cost"] == pytest.approx(-2.5)
    assert payload["rollout/reward_control_cost_count"] == 4.0
    assert payload["rollout/reward_control_cost_min"] == -4.0
    assert payload["rollout/reward_control_cost_max"] == -1.0


def test_tdmpc_episode_events_do_not_mix_raw_and_windowed_reward_components(
    monkeypatch,
):
    td_module = importlib.import_module("RL.TDMPC2")
    run, _finished = _install_fake_wandb(monkeypatch, td_module)
    model = TDMPC2Baseline(
        "TDMPC2Baseline",
        VaryingRewardComponentsEnv(episode_length=2),
        _tiny_td_params(seed_steps=100, wandb_step_every=5),
        {"seed": 3, "device": "cpu", "env": "varying", "total_steps": 5},
        {},
    )

    model.learn(total_timesteps=5)

    assert "rollout/reward_forward" not in run.history[2]
    assert "rollout/reward_forward" not in run.history[4]
    assert run.history[5]["rollout/reward_forward"] == pytest.approx(3.0)
    assert run.history[5]["rollout/reward_forward_count"] == 5.0


def test_tdmpc_reward_component_colliding_with_total_stats_is_aliased(monkeypatch):
    td_module = importlib.import_module("RL.TDMPC2")
    run, _finished = _install_fake_wandb(monkeypatch, td_module)
    model = TDMPC2Baseline(
        "TDMPC2Baseline",
        ReservedRewardComponentEnv(episode_length=10),
        _tiny_td_params(seed_steps=100, wandb_step_every=10),
        {"seed": 3, "device": "cpu", "env": "reserved", "total_steps": 4},
        {},
    )

    model.learn(total_timesteps=4)

    payload = run.history[4]
    assert payload["rollout/reward_mean"] == pytest.approx(1.0)
    assert payload["rollout/reward_mean_component"] == pytest.approx(2.5)
    assert payload["rollout/reward_mean_component_count"] == 4.0
    assert payload["rollout/reward_mean_component_mean"] == pytest.approx(2.5)


def test_tdmpc_overlapping_component_families_keep_one_persistent_alias(monkeypatch):
    td_module = importlib.import_module("RL.TDMPC2")
    run, _finished = _install_fake_wandb(monkeypatch, td_module)
    model = TDMPC2Baseline(
        "TDMPC2Baseline",
        OverlappingRewardComponentsEnv(episode_length=10),
        _tiny_td_params(seed_steps=100, wandb_step_every=2),
        {"seed": 3, "device": "cpu", "env": "overlap", "total_steps": 4},
        {},
    )

    model.learn(total_timesteps=4)

    first, second = run.history[2], run.history[4]
    assert first["rollout/reward_forward"] == pytest.approx(1.5)
    assert first["rollout/reward_forward_mean"] == pytest.approx(1.5)
    assert first["rollout/reward_forward_mean_component"] == pytest.approx(15.0)
    assert first["rollout/reward_forward_mean_component_mean"] == pytest.approx(15.0)
    assert second["rollout/reward_forward"] == pytest.approx(3.5)
    assert second["rollout/reward_forward_mean"] == pytest.approx(3.5)
    assert second["rollout/reward_forward_mean_component"] == pytest.approx(35.0)
    assert second["rollout/reward_forward_mean_component_mean"] == pytest.approx(35.0)
    assert model._reward_component_aliases["rollout/reward_forward_mean"] == (
        "rollout/reward_forward_mean_component"
    )


def test_tdmpc_component_aliases_do_not_depend_on_first_seen_order(monkeypatch):
    td_module = importlib.import_module("RL.TDMPC2")
    run, _finished = _install_fake_wandb(monkeypatch, td_module)
    model = TDMPC2Baseline(
        "TDMPC2Baseline",
        StaggeredOverlappingRewardComponentsEnv(episode_length=10),
        _tiny_td_params(seed_steps=100, wandb_step_every=2),
        {"seed": 3, "device": "cpu", "env": "staggered", "total_steps": 2},
        {},
    )

    model.learn(total_timesteps=2)

    payload = run.history[2]
    assert payload["rollout/reward_forward"] == pytest.approx(2.0)
    assert payload["rollout/reward_forward_mean_component"] == pytest.approx(15.0)
    assert model._reward_component_aliases == {
        "rollout/reward_forward": "rollout/reward_forward",
        "rollout/reward_forward_mean": "rollout/reward_forward_mean_component",
    }


def test_ambi_inner_metrics_survive_matching_episode_and_wandb_boundaries(monkeypatch):
    td_module = importlib.import_module("RL.TDMPC2")
    run, finished = _install_fake_wandb(monkeypatch, td_module)
    env = ConstantEpisodeEnv(episode_length=3)
    params = _tiny_td_params(
        pretrain_steps=1,
        inner_adaptation="clone",
        inner_iterations=1,
        inner_rollouts=2,
        inner_horizon=2,
        inner_updates_per_iteration=1,
        inner_batch_size=2,
        inner_tau=1.0,
        mpc=False,
    )
    model = AMBITDMPC2(
        "AMBITDMPC2",
        env,
        params,
        {"seed": 3, "device": "cpu", "env": "constant", "total_steps": 6},
        {},
    )
    model.agent.update = lambda _buffer: {"total_loss": 1.0}

    def fake_act(_obs, **_kwargs):
        model.agent.last_inner_rollout_lengths = [2, 2]
        model.agent.last_inner_metrics = {
            "inner_active": 1.0,
            "inner_actions": 1.0,
            "inner_rollouts": 2.0,
            "inner_steps": 4.0,
            "inner_updates": 2.0,
            "inner_buffer_size": 4.0,
            "inner_buffer_capacity": 8.0,
            "inner_buffer_fill_ratio": 0.5,
            "inner_return_mean": 2.0,
            "inner_return_std": 1.0,
            "inner_return_min": 1.0,
            "inner_return_max": 3.0,
            "inner_rollout_len_mean": 2.0,
            "inner_rollout_len_std": 0.0,
            "inner_rollout_len_min": 2.0,
            "inner_rollout_len_max": 2.0,
            "inner_termination_rate": 0.0,
            "inner_actor_loss": 5.0,
            "inner_critic_loss": 7.0,
            "inner_alpha": 0.2,
            "inner_q_mean": 1.5,
            "inner_q_target_mean": 2.0,
            "inner_actor_q_mean": 1.75,
            "inner_actor_entropy": 0.4,
            "inner_td_error_abs_mean": 0.5,
            "inner_policy_mean_delta_l2": 0.1,
            "inner_outer_q_gain": 0.25,
        }
        return torch.zeros(model.cfg.action_dim)

    model.agent.act = fake_act
    model.learn(total_timesteps=6)

    seed_payload = run.history[3]
    assert seed_payload["train/inner_active"] == 0.0
    assert seed_payload["train/inner_actions"] == 0.0
    assert "train/inner_actor_loss" not in seed_payload
    payload = run.history[6]
    assert payload["train/inner_active"] == 1.0
    assert payload["train/inner_actions"] == 3.0
    assert payload["train/inner_rollouts"] == 6.0
    assert payload["train/inner_steps"] == 12.0
    assert payload["train/inner_updates"] == 6.0
    assert payload["train/inner_actor_loss"] == 5.0
    assert payload["train/inner_return_mean"] == pytest.approx(2.0)
    assert payload["train/inner_return_std"] == pytest.approx(1.0)
    assert payload["train/inner_steps_total"] == 12
    assert payload["episode/current_return"] == 3.0
    assert payload["episode/current_len"] == 3
    assert finished == [run]


def test_sb3_sac_captures_final_training_burst_and_logger_dump(monkeypatch):
    baselines_module = importlib.import_module("RL.baselines")
    run, finished = _install_fake_wandb(monkeypatch, baselines_module)
    model = Baseline(
        "SAC",
        ConstantEpisodeEnv(episode_length=3),
        {
            "policy": "MlpPolicy",
            "learning_starts": 0,
            "train_freq": 1,
            "gradient_steps": 1,
            "batch_size": 2,
            "buffer_size": 16,
            "policy_kwargs": {"net_arch": [8]},
            "device": "cpu",
            "seed": 7,
            "wandb": True,
            "wandb_step_every": 3,
        },
    )
    model.learn(total_timesteps=4, log_interval=1)

    assert 3 in run.history and 4 in run.history
    assert run.history[3]["episode/return"] == 3.0
    final = run.history[4]
    assert final["train/n_updates"] == 4
    assert final["train/updates_since_log"] >= 1
    assert "train/actor_loss" in final
    assert final["train/replay_size"] == 4
    assert "rollout/ep_rew_mean" in run.history[3]
    assert finished == [run]
