from collections import defaultdict
import importlib

import gymnasium as gym
import numpy as np

from RL.XQC import XQC


class AlternatingDoneEnv(gym.Env):
    metadata = {}
    action_repeat = 2

    def __init__(self, *, evaluation=False):
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, shape=(2,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(1,), dtype=np.float32
        )
        self.evaluation = bool(evaluation)
        self.total_steps = 0
        self.episode_index = -1
        self.episode_step = 0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_index += 1
        self.episode_step = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        del action
        self.total_steps += 1
        self.episode_step += 1
        reward = 5.0 if self.evaluation else float(self.episode_step)
        done = self.episode_step == 2
        terminated = done and self.episode_index % 2 == 0
        truncated = done and not terminated
        return (
            np.zeros(2, dtype=np.float32),
            reward,
            terminated,
            truncated,
            {},
        )

    def close(self):
        self.closed = True


class FakeWandbRun:
    def __init__(self):
        self.history = defaultdict(list)
        self.definitions = []
        self.log_steps = []
        self.finished = False

    def define_metric(self, name, **kwargs):
        self.definitions.append((name, kwargs))

    def log(self, payload, step=None):
        self.log_steps.append(step)
        axis = payload["comparison/raw_frame"] if step is None else step
        self.history[int(axis)].append(dict(payload))

    def finish(self):
        self.finished = True


def _tiny_xqc(env):
    return XQC(
        "XQC",
        env,
        {
            "device": "cpu",
            "seed": 1,
            "buffer_size": 32,
            "learning_starts": 100,
            "batch_size": 2,
            "gradient_steps": 1,
            "updates_per_step": 1,
            "num_interactions": 4,
            "actor_net_arch": [8],
            "critic_net_arch": [8],
            "num_atoms": 5,
            "vmin": -2.0,
            "vmax": 2.0,
            "eval_freq": 2,
            "eval_episodes": 1,
            "wandb": True,
            "wandb_step_every": 100,
        },
        {
            "seed": 1,
            "device": "cpu",
            "env": "DMControl-v0",
            "total_steps": 4,
        },
        {
            "env_params": {
                "task": "humanoid-walk",
                "obs": "state",
                "render_mode": None,
            }
        },
    )


def _payloads_with(run, metric):
    return [
        (step, payload)
        for step, payloads in sorted(run.history.items())
        for payload in payloads
        if metric in payload
    ]


def test_port_logs_canonical_train_and_eval_returns_on_raw_frame_axis(
    monkeypatch,
):
    xqc_module = importlib.import_module("RL.XQC")
    run = FakeWandbRun()
    init_calls = []

    def fake_init(params, **kwargs):
        init_calls.append((params, kwargs))
        return run

    monkeypatch.setattr(xqc_module, "init_wandb", fake_init)
    monkeypatch.setenv("XQC_IMPLEMENTATION", "action-pytorch")
    monkeypatch.setenv("XQC_TASK", "humanoid-walk")
    monkeypatch.setenv("XQC_SOURCE_SHA", "a" * 40)
    monkeypatch.setenv("XQC_ACTION_REPEAT", "2")
    monkeypatch.setenv("XQC_COMPARISON_ID", "humanoid-parity")
    monkeypatch.setenv("WANDB_RUN_GROUP", "humanoid-parity-action-pytorch")

    train_env = AlternatingDoneEnv()
    eval_env = AlternatingDoneEnv(evaluation=True)
    model = _tiny_xqc(train_env)
    model._build_evaluation_env = lambda: eval_env
    model.agent.act = lambda _obs, deterministic=False: np.zeros(
        1, dtype=np.float32
    )
    model.learn(total_timesteps=4)

    assert len(init_calls) == 1
    _params, kwargs = init_calls[0]
    assert kwargs["run_name"] == "xqc-action-pytorch-humanoid-walk-seed1"
    assert {
        key: kwargs["config"][key]
        for key in (
            "implementation",
            "seed",
            "task",
            "source_sha",
            "comparison_id",
            "action_repeat",
        )
    } == {
        "implementation": "action-pytorch",
        "seed": 1,
        "task": "humanoid-walk",
        "source_sha": "a" * 40,
        "comparison_id": "humanoid-parity",
        "action_repeat": 2,
    }
    assert run.definitions == [
        ("comparison/raw_frame", {}),
        ("comparison/decision_step", {"step_metric": "comparison/raw_frame"}),
        ("comparison/train_return", {"step_metric": "comparison/raw_frame"}),
        ("comparison/eval_return", {"step_metric": "comparison/raw_frame"}),
    ]

    training = _payloads_with(run, "comparison/train_return")
    assert [(step, payload["comparison/train_return"]) for step, payload in training] == [
        (4, 3.0),
        (8, 3.0),
    ]
    assert [payload["comparison/decision_step"] for _, payload in training] == [2, 4]
    assert [payload["comparison/raw_frame"] for _, payload in training] == [4, 8]
    assert [payload["env_step"] for _, payload in training] == [4, 8]
    assert [payload["episode/return"] for _, payload in training] == [3.0, 3.0]
    assert [payload["train/terminated"] for _, payload in training] == [1, 0]
    assert [payload["train/truncated"] for _, payload in training] == [0, 1]

    evaluations = _payloads_with(run, "comparison/eval_return")
    assert [(step, payload["comparison/eval_return"]) for step, payload in evaluations] == [
        (2, 10.0),
        (4, 10.0),
        (8, 10.0),
    ]
    assert [payload["comparison/decision_step"] for _, payload in evaluations] == [
        1,
        2,
        4,
    ]
    assert [payload["eval/episode_reward"] for _, payload in evaluations] == [
        10.0,
        10.0,
        10.0,
    ]
    assert run.finished is True
    assert eval_env.closed is True
    assert run.log_steps and set(run.log_steps) == {None}
