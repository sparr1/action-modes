import numpy as np
import pytest

from RL.alg import Random, SimpleAlgorithm, validate_timestep_budget
from RL.AMBI import AMBI as LegacyAMBI


class _CountingActionSpace:
    def __init__(self):
        self.calls = 0

    def sample(self):
        self.calls += 1
        return np.array([self.calls], dtype=np.float32)


class _CountingEnv:
    def __init__(self, episode_length=5):
        self.action_space = _CountingActionSpace()
        self.episode_length = episode_length
        self.episode_step = 0
        self.steps = 0
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.episode_step = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        del action
        self.steps += 1
        self.episode_step += 1
        terminated = self.episode_step == self.episode_length
        return (
            np.array([self.steps], dtype=np.float32),
            1.0,
            terminated,
            False,
            {},
        )


class _ConstantAlgorithm(SimpleAlgorithm):
    def predict(self, observation):
        del observation
        return np.array([0.0], dtype=np.float32), None


def test_simple_algorithm_stops_at_exact_timestep_budget():
    env = _CountingEnv(episode_length=5)
    algorithm = _ConstantAlgorithm("constant", env)

    result = algorithm.learn(total_timesteps=7)

    assert result is algorithm
    assert env.steps == 7
    assert env.resets == 2


def test_simple_algorithm_rejects_negative_timestep_budget():
    algorithm = _ConstantAlgorithm("constant", _CountingEnv())

    with pytest.raises(ValueError, match="non-negative"):
        algorithm.learn(total_timesteps=-1)


@pytest.mark.parametrize("value", [True, 1.5, float("inf"), float("nan"), "nope"])
def test_timestep_budget_rejects_silent_coercions(value):
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_timestep_budget(value)


def test_timestep_budget_accepts_integral_scientific_notation():
    assert validate_timestep_budget(4.5e6) == 4_500_000


@pytest.mark.parametrize("value", [2**53 + 1, str(2**53 + 1)])
def test_timestep_budget_preserves_large_exact_integers(value):
    assert validate_timestep_budget(value) == 2**53 + 1


def test_random_prediction_consumes_exactly_one_action_sample():
    env = _CountingEnv()
    algorithm = Random("random", env)

    action, state = algorithm.predict(np.array([0.0], dtype=np.float32))

    np.testing.assert_array_equal(action, np.array([1.0], dtype=np.float32))
    assert state is None
    assert env.action_space.calls == 1


def test_legacy_ambi_save_returns_delegated_checkpoint_path():
    algorithm = object.__new__(LegacyAMBI)

    class _OuterAgent:
        @staticmethod
        def save(path, name):
            return f"{path}/{name}.zip"

    algorithm.outer_agent = _OuterAgent()

    assert algorithm.save("models", "best") == "models/best.zip"
