import csv
from collections import defaultdict

import gymnasium as gym
import numpy as np
import pytest

from RL.SAC import SAC


class FakeWandbRun:
    def __init__(self):
        self.history = defaultdict(list)

    def log(self, payload, step):
        self.history[int(step)].append(dict(payload))


class ActionRewardEnv(gym.Env):
    metadata = {}

    def __init__(self, episode_length=2):
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, shape=(2,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -2.0, 2.0, shape=(1,), dtype=np.float32
        )
        self.episode_length = int(episode_length)
        self.total_steps = 0
        self.actions = []
        self.reset_seeds = []
        self._episode_step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_seeds.append(seed)
        self._episode_step = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        action_value = float(np.asarray(action).reshape(-1)[0])
        self.actions.append(action_value)
        self.total_steps += 1
        self._episode_step += 1
        truncated = self._episode_step >= self.episode_length
        return (
            np.zeros(2, dtype=np.float32),
            action_value,
            False,
            truncated,
            {},
        )


def _model(env, **params):
    algorithm = {
        "device": "cpu",
        "seed": 11,
        "learning_starts": 0,
        "batch_size": 2,
        "buffer_size": 64,
        "net_arch": [8],
        "wandb": False,
    }
    algorithm.update(params)
    return SAC(
        "SAC",
        env,
        algorithm,
        {"seed": 11, "device": "cpu", "env": "ActionRewardEnv"},
        {},
    )


def _eval_payloads(run, step):
    return [
        payload
        for payload in run.history[int(step)]
        if "eval/episode_reward" in payload
    ]


def test_online_eval_is_deterministic_and_does_not_contaminate_training(
    tmp_path, monkeypatch
):
    eval_csv = tmp_path / "seed_11.csv"
    env = ActionRewardEnv(episode_length=2)
    model = _model(
        env,
        eval_freq=2,
        eval_episodes=3,
        eval_csv_path=str(eval_csv),
    )

    act_calls = []

    def fixed_action(_obs, deterministic=False):
        act_calls.append(bool(deterministic))
        value = 0.25 if deterministic else -0.75
        return np.array([value], dtype=np.float32)

    model.agent.act = fixed_action
    update_calls = []

    def record_update(*args, **kwargs):
        update_calls.append((args, kwargs))
        return {}

    model.agent.update = record_update
    fake_wandb = FakeWandbRun()
    model._wandb_run = fake_wandb
    monkeypatch.setattr("RL.SAC.finish_wandb", lambda _run: None)

    model.learn(total_timesteps=4)

    with eval_csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {"step": "0", "reward": "1.0", "seed": "11"},
        {"step": "2", "reward": "1.0", "seed": "11"},
        {"step": "4", "reward": "1.0", "seed": "11"},
    ]

    assert act_calls.count(True) == 3 * 3 * 2
    assert act_calls.count(False) == 4
    assert len(update_calls) == 4
    assert model.num_timesteps == 4
    assert model.replay_buffer.size == 4
    np.testing.assert_array_equal(
        model.replay_buffer.actions[:4],
        np.full((4, 1), -0.75, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        model.replay_buffer.rewards[:4],
        np.full((4, 1), -1.5, dtype=np.float32),
    )
    assert env.total_steps == 4 + 3 * 3 * 2
    assert env.reset_seeds[0] == 11
    assert all(seed is None for seed in env.reset_seeds[1:])

    for step in (0, 2, 4):
        assert _eval_payloads(fake_wandb, step) == [
            {
                "eval/episode_reward": 1.0,
                "eval/episodes": 3,
                "env_step": step,
            }
        ]


def test_eval_cadence_is_deferred_to_episode_boundaries():
    env = ActionRewardEnv(episode_length=3)
    model = _model(env, eval_freq=2, eval_episodes=1)
    model.agent.update = lambda *_args, **_kwargs: {}
    evaluation_steps = []

    def record_evaluation(step, *, initial_obs=None):
        evaluation_steps.append(int(step))
        return 0.0

    model._evaluate_policy = record_evaluation
    model.learn(total_timesteps=6)

    assert evaluation_steps == [0, 3, 6]
    assert model.num_timesteps == 6
    assert model.replay_buffer.size == 6
    assert env.total_steps == 6


def test_eval_csv_refuses_an_existing_file_before_environment_reset(tmp_path):
    eval_csv = tmp_path / "existing.csv"
    eval_csv.write_text("sentinel\n")
    env = ActionRewardEnv()
    model = _model(
        env,
        eval_freq=2,
        eval_episodes=1,
        eval_csv_path=eval_csv,
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        model.learn(total_timesteps=0)

    assert eval_csv.read_text() == "sentinel\n"
    assert env.reset_seeds == []
    assert env.total_steps == 0
    assert model.replay_buffer.size == 0


def test_eval_csv_uses_per_job_environment_fallback(tmp_path, monkeypatch):
    eval_csv = tmp_path / "from_environment.csv"
    monkeypatch.setenv("SAC_EVAL_CSV", str(eval_csv))
    model = _model(
        ActionRewardEnv(episode_length=1),
        eval_freq=1,
        eval_episodes=1,
    )
    model.agent.act = lambda _obs, deterministic=False: np.zeros(
        1, dtype=np.float32
    )

    model.learn(total_timesteps=0)

    with eval_csv.open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [
            {"step": "0", "reward": "0.0", "seed": "11"}
        ]


def test_explicit_eval_csv_path_wins_over_environment(tmp_path, monkeypatch):
    environment_csv = tmp_path / "from_environment.csv"
    explicit_csv = tmp_path / "explicit.csv"
    monkeypatch.setenv("SAC_EVAL_CSV", str(environment_csv))
    model = _model(
        ActionRewardEnv(episode_length=1),
        eval_freq=1,
        eval_episodes=1,
        eval_csv_path=explicit_csv,
    )
    model.agent.act = lambda _obs, deterministic=False: np.zeros(
        1, dtype=np.float32
    )

    model.learn(total_timesteps=0)

    assert explicit_csv.exists()
    assert not environment_csv.exists()


def test_absent_eval_frequency_preserves_the_legacy_path(tmp_path):
    untouched = tmp_path / "untouched.csv"
    untouched.write_text("sentinel\n")
    env = ActionRewardEnv(episode_length=2)
    model = _model(
        env,
        eval_episodes=0,
        eval_csv_path=str(untouched),
    )
    deterministic_flags = []

    def training_action(_obs, deterministic=False):
        deterministic_flags.append(bool(deterministic))
        return np.zeros(1, dtype=np.float32)

    model.agent.act = training_action
    model.agent.update = lambda *_args, **_kwargs: {}
    model._evaluate_policy = lambda *_args, **_kwargs: pytest.fail(
        "evaluation must remain disabled when eval_freq is absent"
    )

    model.learn(total_timesteps=2)

    assert deterministic_flags == [False, False]
    assert model.num_timesteps == 2
    assert model.replay_buffer.size == 2
    assert env.total_steps == 2
    assert untouched.read_text() == "sentinel\n"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"eval_freq": 0}, "eval_freq"),
        ({"eval_freq": True}, "eval_freq"),
        ({"eval_freq": 1.5}, "eval_freq"),
        ({"eval_freq": 1, "eval_episodes": 0}, "eval_episodes"),
        ({"eval_freq": 1, "eval_csv_path": ""}, "eval_csv_path"),
    ],
)
def test_enabled_eval_options_are_strictly_validated(overrides, message):
    with pytest.raises(ValueError, match=message):
        _model(ActionRewardEnv(), **overrides)
