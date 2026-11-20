"""Exact continuation from an episode boundary on TD-MPC2's pinned simulator."""

import os
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

import domains  # noqa: F401


TASKS = ("assembly", "hand-insert", "push", "stick-pull")


def test_boundary_capture_rejects_mid_episode_and_mismatched_task(monkeypatch):
    from domains import metaworld
    from tests.test_metaworld_env import FakeTask
    from utils.resume_runtime import (
        ResumeRuntimeMismatch, UnsupportedResumeEnvironment, capture_environment_state,
        restore_environment_state, validate_environment_capability,
    )

    class BoundaryTask(FakeTask):
        def __init__(self, seed=None):
            super().__init__(seed=seed)
            self.np_random = np.random.RandomState(seed)
            self.model = SimpleNamespace(cam_pos=np.zeros((3, 3)), body_pos=np.ones((5, 3)),
                                         site_pos=np.ones((3, 3)))
            self._prev_obs = np.ones(18)
            self._last_stable_obs = np.ones(39)
            self.obj_init_pos = np.ones(3)

    monkeypatch.setattr(metaworld, "_load_metaworld_environments", lambda: {
        f"{task}-v2-goal-observable": BoundaryTask for task in ("assembly", "reach")
    })
    env = gym.make("MetaWorld-v0", task="assembly")
    unreviewed = gym.make("MetaWorld-v0", task="reach")
    try:
        with pytest.raises(UnsupportedResumeEnvironment, match="reviewed"):
            validate_environment_capability(unreviewed)
        env.reset(seed=7)
        env.observation_space.seed(7)
        with pytest.raises(ValueError, match="completed episode"):
            capture_environment_state(env)
        for _ in range(100):
            env.step(np.zeros(4))
        state = capture_environment_state(env)
        state["base_state"]["task"] = "hand-insert"
        before = env.unwrapped._env._prev_obs.copy()
        with pytest.raises(ResumeRuntimeMismatch):
            restore_environment_state(env, state)
        np.testing.assert_array_equal(env.unwrapped._env._prev_obs, before)
    finally:
        env.close()
        unreviewed.close()


@pytest.mark.skipif(os.environ.get("AMBI_RUN_REAL_METAWORLD_TESTS") != "1",
                    reason="requires the pinned legacy simulator")
@pytest.mark.parametrize("task", TASKS)
def test_boundary_restore_matches_uninterrupted_trajectory(task):
    from utils.resume_runtime import capture_environment_state, restore_environment_state

    original = gym.make("MetaWorld-v0", task=task)
    resumed = gym.make("MetaWorld-v0", task=task)
    try:
        original.reset(seed=55)
        original.observation_space.seed(55)
        resumed.reset(seed=37)
        resumed.observation_space.seed(37)
        for _ in range(100):
            original.step(original.action_space.sample())
        saved = capture_environment_state(original)
        restore_environment_state(resumed, saved)
        # Check successive goal draws, inherited frames, sampled actions,
        # rewards, success, and timeouts through two complete episodes.
        for _ in range(2):
            np.testing.assert_array_equal(original.reset()[0], resumed.reset()[0])
            for decision in range(100):
                action = original.action_space.sample()
                np.testing.assert_array_equal(resumed.action_space.sample(), action)
                expected = original.step(action)
                actual = resumed.step(action)
                np.testing.assert_array_equal(expected[0], actual[0])
                assert actual[1:] == expected[1:]
                assert actual[2:4] == (False, decision == 99)
    finally:
        original.close()
        resumed.close()
