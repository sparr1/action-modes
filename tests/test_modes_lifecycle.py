import types

import numpy as np
import pytest

from RL.modes import ModalAlg


class _FakeEnv:
    def __init__(self, *, truncate_at=None):
        self.truncate_at = truncate_at
        self.reset_calls = 0
        self.step_calls = 0
        self.close_calls = 0
        self.steps_in_episode = 0

    def reset(self, *, options=None):
        del options
        self.reset_calls += 1
        self.steps_in_episode = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        del action
        self.step_calls += 1
        self.steps_in_episode += 1
        truncated = self.steps_in_episode == self.truncate_at
        return (
            np.array([float(self.steps_in_episode)], dtype=np.float32),
            1.0,
            False,
            truncated,
            {"source": "env"},
        )

    def close(self):
        self.close_calls += 1


class _FakeAgent:
    def __init__(self, *, max_steps):
        self.custom_params = {
            "save_freq": 0,
            "save_dir": "",
            "visualise": False,
            "render_freq": 1,
            "save_frames": False,
            "title": "test",
            "seed": 1,
            "evaluation_episodes": 0,
            "max_steps": max_steps,
        }
        self.start_calls = 0
        self.end_calls = 0
        self.transitions = []

    def start_episode(self):
        self.start_calls += 1

    def end_episode(self):
        self.end_calls += 1

    def step(self, *transition):
        self.transitions.append(transition)


class _FakeLogger:
    def __init__(self):
        self.records = []

    def on_step(self, data):
        self.records.append(data)


def _algorithm(*, max_steps=100, truncate_at=None):
    algorithm = object.__new__(ModalAlg)
    algorithm.env = _FakeEnv(truncate_at=truncate_at)
    agent = _FakeAgent(max_steps=max_steps)
    algorithm.orchestrator = types.SimpleNamespace(
        model=agent,
        process_obs=lambda observation: (np.asarray(observation), 1),
    )
    algorithm.alg_logger = _FakeLogger()

    def predict(observation, modulus=0, old_orch_act=None):
        del modulus, old_orch_act
        parameter = np.array([float(observation[0]) + 0.25], dtype=np.float32)
        return np.array([0.0], dtype=np.float32), {
            "orchestral_action": (0, parameter),
        }

    algorithm.predict = predict
    return algorithm, agent


def test_modal_learn_honors_exact_budget_and_algorithm_episode_cap():
    algorithm, agent = _algorithm(max_steps=2)

    result = algorithm.learn(total_timesteps=3)

    assert result is algorithm
    assert algorithm.env.step_calls == 3
    assert algorithm.env.reset_calls == 2
    assert algorithm.env.close_calls == 0
    assert agent.start_calls == agent.end_calls == 2
    assert [transition[5] for transition in agent.transitions] == [
        False,
        True,
        False,
    ]
    assert algorithm.alg_logger.records[0]["actions"] == [0, 0.25]
    assert algorithm.alg_logger.records[1]["dones"] == [False, True]
    assert algorithm.alg_logger.records[1]["infos"][0] == {
        "source": "env",
        "terminated": False,
        "truncated": True,
        "TimeLimit.truncated": True,
    }


def test_modal_learn_passes_native_truncation_to_agent():
    algorithm, agent = _algorithm(max_steps=100, truncate_at=1)

    algorithm.learn(total_timesteps=1)

    assert len(agent.transitions) == 1
    assert agent.transitions[0][5] is True
    assert algorithm.alg_logger.records[0]["dones"] == [False, True]
    assert algorithm.alg_logger.records[0]["infos"][0] == {
        "source": "env",
        "terminated": False,
        "truncated": True,
        "TimeLimit.truncated": True,
    }


def test_modal_learn_zero_budget_is_a_noop_and_does_not_close_env():
    algorithm, agent = _algorithm()

    assert algorithm.learn(total_timesteps=0) is algorithm
    assert algorithm.env.reset_calls == 0
    assert algorithm.env.step_calls == 0
    assert algorithm.env.close_calls == 0
    assert agent.start_calls == agent.end_calls == 0


def test_modal_learn_rejects_missing_or_negative_budget():
    algorithm, _ = _algorithm()
    with pytest.raises(ValueError, match="requires total_timesteps"):
        algorithm.learn()
    with pytest.raises(ValueError, match="non-negative"):
        algorithm.learn(total_timesteps=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        algorithm.learn(total_timesteps=1.5)
    with pytest.raises(ValueError, match="non-negative integer"):
        algorithm.learn(total_timesteps=True)


@pytest.mark.parametrize("max_steps", (True, 0, -1, 1.5))
def test_modal_learn_rejects_invalid_episode_cap(max_steps):
    algorithm, _ = _algorithm(max_steps=max_steps)
    with pytest.raises(ValueError, match="ModalAlg max_steps"):
        algorithm.learn(total_timesteps=1)


def test_modal_sticky_orchestral_action_lasts_five_executed_steps():
    algorithm, agent = _algorithm(max_steps=100)
    generations = []

    def predict(_observation, modulus=0, old_orch_act=None):
        if modulus != 0 and old_orch_act is not None:
            orchestral_action = old_orch_act
        else:
            generation = len(generations) + 1
            generations.append(generation)
            orchestral_action = (
                0,
                np.array([float(generation)], dtype=np.float32),
            )
        return np.array([0.0], dtype=np.float32), {
            "orchestral_action": orchestral_action,
        }

    algorithm.predict = predict
    algorithm.learn(total_timesteps=6)

    executed_generations = [
        int(transition[1][1][0]) for transition in agent.transitions
    ]
    assert executed_generations == [1, 1, 1, 1, 1, 2]
    assert generations == [1, 2]


def test_modal_save_returns_delegated_checkpoint_path():
    algorithm = object.__new__(ModalAlg)
    algorithm.orchestrator = types.SimpleNamespace(
        model=types.SimpleNamespace(
            save=lambda path, name: f"{path}/{name}.checkpoint"
        )
    )

    assert algorithm.save("models", "best") == "models/best.checkpoint"


@pytest.mark.parametrize(
    ("num_modes", "mode_configs", "message"),
    [
        (0, [], "at least one mode"),
        (2, ["only-one.json"], "must match"),
    ],
)
def test_modal_constructor_rejects_incoherent_mode_counts(
    num_modes, mode_configs, message
):
    with pytest.raises(ValueError, match=message):
        ModalAlg("modal", object(), "orchestrator.json", mode_configs, num_modes)
