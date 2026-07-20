import json
from pathlib import Path

import numpy as np
import pytest
import torch
import gymnasium as gym

from RL.SAC import SAC
from RL.sac_core import ReplayBuffer, SACAgent, SACConfig
from utils.utils import setup_logs


class ShortEpisodeEnv(gym.Env):
    metadata = {}

    def __init__(self, episode_length=3):
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-2.0, 2.0, shape=(1,), dtype=np.float32)
        self.episode_length = episode_length
        self.total_steps = 0
        self._episode_step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_step = 0
        return self.np_random.normal(size=2).astype(np.float32), {}

    def step(self, action):
        self._episode_step += 1
        self.total_steps += 1
        obs = self.np_random.normal(size=2).astype(np.float32)
        truncated = self._episode_step >= self.episode_length
        return obs, -float(np.square(action).sum()), False, truncated, {}


def test_replay_samples_with_replacement_and_validates_shapes():
    replay = ReplayBuffer(obs_dim=3, action_dim=2, capacity=8)
    replay.add(np.zeros(3), np.zeros(2), 1.0, np.ones(3), False, False)
    batch = replay.sample(batch_size=16, device=torch.device("cpu"))
    assert batch["obs"].shape == (16, 3)
    assert batch["actions"].shape == (16, 2)

    with pytest.raises(ValueError, match="must contain 3 values"):
        replay.add(np.zeros(1), np.zeros(2), 1.0, np.ones(3), False, False)
    with pytest.raises(ValueError, match="empty replay"):
        ReplayBuffer(3, 2, 8).sample(1, torch.device("cpu"))


def test_asymmetric_and_empty_network_architectures_are_preserved():
    env = ShortEpisodeEnv()
    model = SAC(
        "SAC",
        env,
        {
            "device": "cpu",
            "policy_kwargs": {"net_arch": {"pi": [7], "qf": []}},
            "wandb": False,
        },
        {},
        {},
    )
    assert model.cfg.actor_net_arch == (7,)
    assert model.cfg.critic_net_arch == ()
    assert len(model.agent.actor.latent_pi) == 2
    assert len(model.agent.critic.qf1) == 1


def test_reward_logging_preserves_termination_metadata():
    data = setup_logs(
        reward=1.0,
        obs=np.zeros((1, 2), dtype=np.float32),
        action=np.zeros((1, 1), dtype=np.float32),
        dones=[True],
        info=[{"reward_info": {"base": 1.0}, "terminated": False, "truncated": True}],
    )
    assert data["infos"]["base"] == 1.0
    assert data["infos"]["terminated"] is False
    assert data["infos"]["truncated"] is True


def test_episode_train_frequency_and_gradient_steps_minus_one():
    env = ShortEpisodeEnv(episode_length=3)
    model = SAC(
        "SAC",
        env,
        {
            "device": "cpu",
            "seed": 5,
            "learning_starts": 0,
            "batch_size": 64,
            "train_freq": [2, "episode"],
            "gradient_steps": -1,
            "buffer_size": 64,
            "net_arch": [8, 8],
            "wandb": False,
        },
        {"seed": 5, "device": "cpu", "env": "ShortEpisodeEnv"},
        {},
    )
    calls = []

    def record_update(replay, gradient_steps, batch_size):
        calls.append((model.num_timesteps, gradient_steps, batch_size))
        return {"actor_loss": 0.0}

    model.agent.update = record_update
    model.learn(total_timesteps=12)
    assert calls == [(6, 6, 64), (12, 6, 64)]


def test_episode_train_frequency_finishes_sb3_rollout_chunk():
    env = ShortEpisodeEnv(episode_length=3)
    model = SAC(
        "SAC",
        env,
        {
            "device": "cpu",
            "learning_starts": 0,
            "batch_size": 4,
            "train_freq": [2, "episode"],
            "gradient_steps": -1,
            "buffer_size": 64,
            "net_arch": [8],
            "wandb": False,
        },
        {},
        {},
    )
    calls = []

    def record_update(_replay, gradient_steps, _batch_size):
        calls.append((model.num_timesteps, gradient_steps))
        return {}

    model.agent.update = record_update
    model.learn(total_timesteps=4)
    assert env.total_steps == 6
    assert calls == [(6, 6)]


def test_fresh_agents_seed_first_environment_reset_identically():
    def build():
        env = ShortEpisodeEnv()
        model = SAC(
            "SAC",
            env,
            {"device": "cpu", "seed": 17, "net_arch": [8], "wandb": False},
            {"seed": 17, "device": "cpu", "env": "ShortEpisodeEnv"},
            {},
        )
        model.learn(total_timesteps=0)
        return model._last_obs

    np.testing.assert_array_equal(build(), build())


def test_repeated_learn_calls_run_additional_work():
    env = ShortEpisodeEnv()
    model = SAC(
        "SAC",
        env,
        {
            "device": "cpu",
            "seed": 3,
            "learning_starts": 100,
            "buffer_size": 32,
            "net_arch": [8],
            "wandb": False,
        },
        {"seed": 3, "device": "cpu", "env": "ShortEpisodeEnv"},
        {},
    )
    model.learn(total_timesteps=4)
    model.learn(total_timesteps=4)
    assert env.total_steps == 8
    assert model.num_timesteps == 4
    assert model.replay_buffer.size == 8


def test_repeated_learn_calls_reopen_wandb_and_clear_stale_metrics(monkeypatch):
    import importlib

    sac_module = importlib.import_module("RL.SAC")
    runs = []
    finished = []

    class FakeRun:
        def log(self, *_args, **_kwargs):
            pass

    def fake_init(*_args, **_kwargs):
        run = FakeRun()
        runs.append(run)
        return run

    monkeypatch.setattr(sac_module, "init_wandb", fake_init)
    monkeypatch.setattr(sac_module, "finish_wandb", finished.append)

    env = ShortEpisodeEnv()
    model = SAC(
        "SAC",
        env,
        {
            "device": "cpu",
            "learning_starts": 100,
            "buffer_size": 32,
            "net_arch": [8],
            "wandb": True,
        },
        {},
        {},
    )
    model.learn(total_timesteps=1)
    model._last_metrics = {"stale": 1.0}
    model.learn(total_timesteps=1)

    assert len(runs) == 2
    assert finished == runs
    assert model._last_metrics == {}


def test_checkpoint_load_validates_configuration_and_is_not_fake_resume(tmp_path):
    def build(gamma=0.99):
        env = ShortEpisodeEnv()
        return SAC(
            "SAC",
            env,
            {
                "device": "cpu",
                "seed": 3,
                "gamma": gamma,
                "learning_starts": 100,
                "buffer_size": 32,
                "net_arch": [8],
                "wandb": False,
            },
            {"seed": 3, "device": "cpu", "env": "ShortEpisodeEnv"},
            {},
        )

    source = build()
    source.learn(total_timesteps=4)
    source.save(tmp_path, "native")
    checkpoint = tmp_path / "native.pt"
    metadata = json.loads((tmp_path / "native.pt.metadata.json").read_text())
    assert metadata["schema_version"] == 1
    assert metadata["checkpoint"]["kind"] == "trial_final"
    assert metadata["checkpoint"]["step"] == 4
    assert metadata["trial_run_params"]["seed"] == 3

    restored = build()
    restored.load(checkpoint)
    assert restored.num_timesteps == 0
    for key, value in source.agent.actor.state_dict().items():
        torch.testing.assert_close(value, restored.agent.actor.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="cannot safely resume"):
        restored.load(checkpoint, resume=True)
    with pytest.raises(ValueError, match="configuration mismatch"):
        build(gamma=0.5).load(checkpoint)


def test_native_sac_composes_best_and_latest_without_numbered_checkpoints(tmp_path):
    env = ShortEpisodeEnv(episode_length=2)
    run_params = {"seed": 8, "device": "cpu", "env": "ShortEpisodeEnv"}
    experiment_params = {"trials": 1, "save_strat": ["best", "latest"]}
    model = SAC(
        "SAC",
        env,
        {
            "device": "cpu",
            "seed": 8,
            "learning_starts": 100,
            "buffer_size": 32,
            "net_arch": [8],
            "wandb": False,
        },
        run_params,
        experiment_params,
    )
    model.set_checkpointing(
        2,
        tmp_path,
        "model:native_0",
        save_strat=["best", "latest"],
        checkpoint_best_window=2,
    )
    model.learn(total_timesteps=4)

    best = tmp_path / "model:native_0_best.pt"
    latest = tmp_path / "model:native_0_latest.pt"
    assert best.is_file()
    assert latest.is_file()
    assert not list(tmp_path.glob("model:native_0_*_steps.pt"))
    latest_metadata = json.loads(Path(f"{latest}.metadata.json").read_text())
    assert latest_metadata["checkpoint"]["kind"] == "latest"
    assert latest_metadata["checkpoint"]["step"] == 4
    assert latest_metadata["checkpoint"]["episode"] == 2
    assert latest_metadata["trial_run_params"] == run_params
    assert latest_metadata["experiment_params"] == experiment_params


def test_native_sac_exception_does_not_publish_a_false_final_latest(tmp_path):
    class FailingEnv(ShortEpisodeEnv):
        def step(self, action):
            if self.total_steps == 2:
                raise RuntimeError("environment failed")
            return super().step(action)

    model = SAC(
        "SAC",
        FailingEnv(episode_length=2),
        {
            "device": "cpu",
            "seed": 4,
            "learning_starts": 100,
            "buffer_size": 16,
            "net_arch": [8],
            "wandb": False,
        },
        {"seed": 4, "device": "cpu", "env": "FailingEnv"},
        {},
    )
    model.set_checkpointing(
        2,
        tmp_path,
        "model:native_0",
        save_strat="latest",
    )

    with pytest.raises(RuntimeError, match="environment failed"):
        model.learn(total_timesteps=4)

    latest = tmp_path / "model:native_0_latest.pt"
    metadata = json.loads(Path(f"{latest}.metadata.json").read_text())
    assert metadata["checkpoint"]["step"] == 2
    assert metadata["checkpoint"]["episode"] == 1
    assert model.num_timesteps == 2


def test_native_sac_one_step_matches_sb3_232():
    stable_baselines3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.type_aliases import ReplayBufferSamples

    batch_size, obs_dim, action_dim = 16, 3, 1
    env = gym.make("Pendulum-v1")
    sb3 = stable_baselines3.SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1000,
        learning_starts=100,
        batch_size=batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_entropy="auto",
        target_update_interval=1,
        policy_kwargs={"net_arch": [32, 32]},
        seed=7,
        device="cpu",
        verbose=0,
    )
    sb3.set_logger(configure(folder=None, format_strings=[]))
    native = SACAgent(
        obs_dim,
        action_dim,
        SACConfig(net_arch=(32, 32), seed=99, device="cpu", adam_eps=1e-8),
    )

    sb3.actor.load_state_dict(native.actor.state_dict())

    def to_sb3_critic(state):
        return {
            key.replace("qf1.", "qf0.").replace("qf2.", "qf1."): value
            for key, value in state.items()
        }

    sb3.critic.load_state_dict(to_sb3_critic(native.critic.state_dict()))
    sb3.critic_target.load_state_dict(to_sb3_critic(native.critic_target.state_dict()))
    sb3.log_ent_coef.data.copy_(native.log_ent_coef.data)

    generator = torch.Generator().manual_seed(123)
    obs = torch.randn(batch_size, obs_dim, generator=generator)
    actions = torch.tanh(torch.randn(batch_size, action_dim, generator=generator))
    rewards = torch.randn(batch_size, 1, generator=generator)
    next_obs = torch.randn(batch_size, obs_dim, generator=generator)
    dones = (torch.rand(batch_size, 1, generator=generator) < 0.2).float()

    class NativeBatch:
        def sample(self, *_args):
            return {
                "obs": obs,
                "actions": actions,
                "rewards": rewards,
                "next_obs": next_obs,
                "dones": dones,
            }

    class SB3Batch:
        def sample(self, *_args, **_kwargs):
            return ReplayBufferSamples(obs, actions, next_obs, dones, rewards)

    sb3.replay_buffer = SB3Batch()
    torch.manual_seed(2024)
    native.update(NativeBatch(), gradient_steps=1, batch_size=batch_size)
    torch.manual_seed(2024)
    sb3.train(gradient_steps=1, batch_size=batch_size)

    for key, value in native.actor.state_dict().items():
        torch.testing.assert_close(value, sb3.actor.state_dict()[key], rtol=0, atol=1e-7)
    for key, value in native.critic.state_dict().items():
        sb3_key = key.replace("qf1.", "qf0.").replace("qf2.", "qf1.")
        torch.testing.assert_close(value, sb3.critic.state_dict()[sb3_key], rtol=0, atol=1e-7)
    torch.testing.assert_close(native.log_ent_coef, sb3.log_ent_coef, rtol=0, atol=1e-7)
