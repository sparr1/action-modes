import numpy as np
import click
import pytest
from click.testing import CliRunner

from RL.PAMDP import PAMDP
from RL.pamdp_transition import (
    save_pamdp_returns,
    unpack_pamdp_step,
    with_owned_environment,
)


class _PAMDPEnv:
    def __init__(self, *, truncate_after=1):
        self.closed = False
        self.truncate_after = truncate_after
        self.step_count = 0
        self._episode_step = 0

    def reset(self):
        self._episode_step = 0
        return np.array([1.0, 2.0], dtype=np.float32), {}

    def step(self, action):
        del action
        self.step_count += 1
        self._episode_step += 1
        truncated = (
            self.truncate_after is not None
            and self._episode_step >= self.truncate_after
        )
        return (
            (
                np.array(
                    [1.0 + self._episode_step, 2.0 + self._episode_step],
                    dtype=np.float64,
                ),
                1,
            ),
            0.5,
            False,
            truncated,
            {},
        )

    def close(self):
        self.closed = True

    def get_episode_rewards(self):
        return [0.5]


class _RecordingPAMDPAgent:
    def __init__(self):
        self.observations = []
        self.transitions = []

    def act(self, observation):
        self.observations.append(np.asarray(observation).copy())
        return 0, np.zeros(2, dtype=np.float32), np.zeros(4, dtype=np.float32)

    def step(self, *transition):
        self.transitions.append(transition)

    def start_episode(self):
        pass

    def end_episode(self):
        pass


class _CheckpointPAMDPAgent(_RecordingPAMDPAgent):
    def __init__(self):
        super().__init__()
        self.saved_prefixes = []
        self.loaded_prefixes = []

    def save_models(self, prefix):
        self.saved_prefixes.append(prefix)

    def load_models(self, prefix):
        self.loaded_prefixes.append(prefix)


class _RecordingLogger:
    def __init__(self):
        self.rows = []

    def on_step(self, data):
        self.rows.append(data)


def _params(tmp_path, **overrides):
    params = {
        "save_freq": 0,
        "save_dir": str(tmp_path),
        "visualise": False,
        "render_freq": 1,
        "save_frames": False,
        "title": "test",
        "seed": 7,
        "evaluation_episodes": 0,
    }
    params.update(overrides)
    return params


def test_pamdp_learning_passes_environment_next_state_and_truncation(
    tmp_path, monkeypatch
):
    agent = _RecordingPAMDPAgent()
    monkeypatch.setattr(PAMDP, "get_PAMDP_model", lambda *args, **kwargs: agent)
    env = _PAMDPEnv(truncate_after=1)
    learner = PAMDP("PAMDP", env, _params(tmp_path, evaluation_episodes=1))
    logger = _RecordingLogger()
    learner.set_logger(logger)

    result = learner.learn(total_timesteps=1)

    assert len(agent.transitions) == 1
    state, _, _, next_state, _, terminal, steps = agent.transitions[0]
    np.testing.assert_array_equal(state, np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(next_state, np.array([2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(agent.observations[1], next_state)
    assert terminal is True
    assert steps == 1
    assert env.closed is False
    assert (tmp_path / "test7.npy").is_file()
    assert (tmp_path / "test7e.npy").is_file()
    assert result is learner
    assert len(logger.rows) == 1
    assert logger.rows[0]["dones"] == [False, True]
    assert logger.rows[0]["obs"] == [2.0, 3.0]


def test_pamdp_learning_stops_at_exact_timestep_budget(tmp_path, monkeypatch):
    agent = _RecordingPAMDPAgent()
    monkeypatch.setattr(PAMDP, "get_PAMDP_model", lambda *args, **kwargs: agent)
    env = _PAMDPEnv(truncate_after=None)
    learner = PAMDP("PAMDP", env, _params(tmp_path))

    learner.learn(total_timesteps=3)

    assert env.step_count == 3
    assert len(agent.transitions) == 3
    assert env.closed is False


def test_pamdp_rejects_inexact_timestep_budget_before_training(tmp_path, monkeypatch):
    agent = _RecordingPAMDPAgent()
    monkeypatch.setattr(PAMDP, "get_PAMDP_model", lambda *args, **kwargs: agent)
    env = _PAMDPEnv(truncate_after=None)
    learner = PAMDP("PAMDP", env, _params(tmp_path))

    with np.testing.assert_raises_regex(ValueError, "non-negative integer"):
        learner.learn(total_timesteps=1.5)

    assert env.step_count == 0
    assert agent.transitions == []


def test_pamdp_max_steps_is_a_truncation_but_global_budget_is_not(
    tmp_path, monkeypatch
):
    agent = _RecordingPAMDPAgent()
    monkeypatch.setattr(PAMDP, "get_PAMDP_model", lambda *args, **kwargs: agent)
    env = _PAMDPEnv(truncate_after=None)
    learner = PAMDP("PAMDP", env, _params(tmp_path, max_steps=2))
    logger = _RecordingLogger()
    learner.set_logger(logger)

    learner.learn(total_timesteps=3)

    assert [transition[5] for transition in agent.transitions] == [False, True, False]
    assert [row["dones"] for row in logger.rows] == [
        [False, False],
        [False, True],
        [False, False],
    ]
    assert logger.rows[1]["infos"][0]["TimeLimit.truncated"] is True


def test_pamdp_evaluation_stops_at_configured_max_steps(tmp_path, monkeypatch):
    agent = _RecordingPAMDPAgent()
    monkeypatch.setattr(PAMDP, "get_PAMDP_model", lambda *args, **kwargs: agent)
    env = _PAMDPEnv(truncate_after=None)
    learner = PAMDP("PAMDP", env, _params(tmp_path, max_steps=2))

    returns = learner.evaluate(env, agent, episodes=2)

    np.testing.assert_array_equal(returns, np.array([1.0, 1.0]))
    assert env.step_count == 4


def test_legacy_pamdp_step_normalization_preserves_next_state_and_done():
    result = unpack_pamdp_step(
        (
            np.array([9.0, 8.0], dtype=np.float64),
            1.5,
            True,
            {"steps": 4},
        )
    )

    next_state, reward, terminated, truncated, done, info, steps = result
    np.testing.assert_array_equal(next_state, np.array([9.0, 8.0], np.float32))
    assert (reward, terminated, truncated, done, steps) == (1.5, True, False, True, 4)
    assert info == {
        "steps": 4,
        "terminated": True,
        "truncated": False,
        "TimeLimit.truncated": False,
    }

    simultaneous = unpack_pamdp_step(
        ((np.array([1.0]), 2), 0.0, True, True, {})
    )
    assert simultaneous[2:5] == (True, True, True)
    assert simultaneous[5] == {
        "terminated": True,
        "truncated": True,
        "TimeLimit.truncated": False,
    }

    legacy_timeout = unpack_pamdp_step(
        (np.array([2.0]), 0.0, True, {"TimeLimit.truncated": True})
    )
    assert legacy_timeout[2:5] == (False, True, True)
    assert legacy_timeout[5]["TimeLimit.truncated"] is True


def test_pamdp_return_writer_uses_configured_directory(tmp_path):
    path = save_pamdp_returns(
        tmp_path / "nested", "trial", 4, [1.0, 2.0], evaluation=True
    )

    assert path == str(tmp_path / "nested" / "trial4e.npy")
    np.testing.assert_array_equal(np.load(path), np.array([1.0, 2.0]))


def test_pamdp_save_and_load_follow_mpdqn_prefix_contract(tmp_path, monkeypatch):
    agent = _CheckpointPAMDPAgent()
    monkeypatch.setattr(PAMDP, "get_PAMDP_model", lambda *args, **kwargs: agent)
    learner = PAMDP("PAMDP", _PAMDPEnv(), _params(tmp_path))
    prefix = str(tmp_path / "models" / "best")

    artifacts = learner.save(str(tmp_path / "models"), "best")
    loaded = learner.load(prefix)

    assert artifacts == (prefix + "_actor.pt", prefix + "_actor_param.pt")
    assert agent.saved_prefixes == [prefix]
    assert agent.loaded_prefixes == [prefix]
    assert learner.model is agent
    assert loaded is learner


def test_pamdp_load_fallback_keeps_in_place_custom_backend(tmp_path, monkeypatch):
    class InPlaceBackend(_RecordingPAMDPAgent):
        def __init__(self):
            super().__init__()
            self.loaded = []

        def load(self, path):
            self.loaded.append(path)
            return None

    agent = InPlaceBackend()
    monkeypatch.setattr(PAMDP, "get_PAMDP_model", lambda *args, **kwargs: agent)
    learner = PAMDP("PAMDP", _PAMDPEnv(), _params(tmp_path))

    assert learner.load("legacy") is learner
    assert learner.model is agent
    assert agent.loaded == ["legacy"]


def test_owned_environment_cleanup_preserves_primary_exception():
    class FailingCloseEnv:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True
            raise RuntimeError("cleanup failed")

    env = FailingCloseEnv()

    @with_owned_environment(lambda: env)
    def run(*, _owned_env):
        assert _owned_env is env
        raise ValueError("training failed")

    with pytest.raises(ValueError, match="training failed") as raised:
        run()
    assert env.closed is True
    assert any("cleanup failed" in note for note in raised.value.__notes__)


def test_owned_environment_integrates_with_click_callback_options():
    env = _PAMDPEnv()
    seen = []

    @click.command()
    @click.option("--value", type=int, required=True)
    @with_owned_environment(lambda: env)
    def command(value, _owned_env):
        seen.append((value, _owned_env))

    result = CliRunner().invoke(command, ["--value", "4"])

    assert result.exit_code == 0, result.output
    assert seen == [(4, env)]
    assert env.closed is True
