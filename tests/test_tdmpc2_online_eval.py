import csv
from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch

from RL.TDMPC2 import TDMPC2Baseline


class FakeWandbRun:
    def __init__(self):
        self.history = defaultdict(dict)

    def log(self, payload, step):
        self.history[int(step)].update(dict(payload))


class TwoStepEnv(gym.Env):
    metadata = {}

    def __init__(self):
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, shape=(3,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(1,), dtype=np.float32
        )
        self.spec = None
        self.reset_seeds = []
        self.total_env_steps = 0
        self._episode_step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_seeds.append(seed)
        self._episode_step = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.total_env_steps += 1
        self._episode_step += 1
        return (
            np.zeros(3, dtype=np.float32),
            1.0,
            False,
            self._episode_step == 2,
            {},
        )


def _tiny_params(eval_csv_path):
    return {
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
        "train_unroll_horizon": 1,
        "outer_planning_horizon": 1,
        "inner_rollout_horizon": 1,
        "buffer_size": 100,
        "seed_steps": 100,
        "pretrain_steps": 1,
        "utd": 1,
        "compile": False,
        "episodic": False,
        "discount": 0.99,
        "iterations": 1,
        "num_samples": 4,
        "num_elites": 2,
        "num_pi_trajs": 1,
        "wandb": False,
        "dropout": 0.0,
        "episode_length": 2,
        "eval_freq": 2,
        "eval_episodes": 3,
        "eval_csv_path": str(eval_csv_path),
    }


def test_online_eval_writes_only_official_rows_and_never_counts_eval_steps(
    tmp_path, monkeypatch,
):
    eval_csv = tmp_path / "seed_3.csv"
    env = TwoStepEnv()
    model = TDMPC2Baseline(
        "TDMPC2Baseline",
        env,
        _tiny_params(eval_csv),
        {"seed": 3, "device": "cpu", "env": "test", "total_steps": 4},
        {},
    )

    eval_calls = []

    def deterministic_eval_action(obs_t, *, t0, eval_mode):
        assert eval_mode is True
        eval_calls.append(bool(t0))
        return torch.zeros(model.cfg.action_dim)

    model._act_agent = deterministic_eval_action
    model._random_action_norm = lambda: np.zeros(model.cfg.action_dim, np.float32)
    fake_wandb = FakeWandbRun()
    model._wandb_run = fake_wandb
    monkeypatch.setattr("RL.TDMPC2.finish_wandb", lambda _run: None)
    model.learn(total_timesteps=4)

    with eval_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {"step": "0", "reward": "2.0", "seed": "3"},
        {"step": "2", "reward": "2.0", "seed": "3"},
        {"step": "4", "reward": "2.0", "seed": "3"},
    ]
    assert len(eval_calls) == 3 * 3 * 2
    assert sum(eval_calls) == 3 * 3
    assert model._global_step == 4
    assert model.buffer.num_eps == 2
    assert env.total_env_steps == 4 + 3 * 3 * 2
    assert env.reset_seeds[0] == 3
    assert all(seed is None for seed in env.reset_seeds[1:])
    assert list(tmp_path.iterdir()) == [eval_csv]
    assert [
        fake_wandb.history[step]["eval/episode_reward"]
        for step in (0, 2, 4)
    ] == [2.0, 2.0, 2.0]
    assert all(
        fake_wandb.history[step]["eval/episodes"] == 3
        for step in (0, 2, 4)
    )
