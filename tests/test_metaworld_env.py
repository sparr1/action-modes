"""Adapter contracts; opt-in real checks step simulators without training."""

import importlib
import os
import subprocess
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest


TASKS = ("assembly", "hand-insert", "push", "stick-pull")


class FakeTask:
    def __init__(self, seed=None):
        self.action_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(39,))
        self.model = SimpleNamespace(cam_pos=np.zeros((3, 3)))
        self._freeze_rand_vec = True
        self.actions = []
        self.closed = False
        self.count = 0
        self.random_draw = np.random.uniform()

    def reset(self):
        self.count = 0
        self.actions = []
        self.random_draw = np.random.uniform()
        return np.full(39, self.random_draw, dtype=np.float64)

    def step(self, action):
        self.count += 1
        self.actions.append(action.copy())
        # Upstream is allowed to mutate its argument; repeats must be copies.
        action[:] = 0
        return (
            np.full(39, self.count, dtype=np.float64),
            float(self.count),
            False,
            {"success": float(self.count % 2 == 0)},
        )

    def close(self):
        self.closed = True


@pytest.fixture
def adapter(monkeypatch):
    module = importlib.import_module("domains.metaworld")
    monkeypatch.setattr(
        module, "_load_metaworld_environments",
        lambda: {f"{task}-v2-goal-observable": FakeTask for task in TASKS},
    )
    return module


def test_lazy_registration():
    import domains  # noqa: F401

    result = subprocess.run(
        [sys.executable, "-c", "import domains, sys; "
         "assert {'mujoco', 'mujoco_py', 'metaworld'}.isdisjoint(sys.modules)"],
        check=True, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert gym.spec("MetaWorld-v0").entry_point == "domains.metaworld:MetaWorldEnv"
    assert gym.spec("MetaWorld-v0").max_episode_steps == 100


@pytest.mark.parametrize("task", TASKS)
def test_task_construction_and_reset_observation(adapter, task):
    env = gym.make("MetaWorld-v0", task=f"mw-{task}")
    try:
        obs, info = env.reset(seed=7)
        raw = env.unwrapped._env
        assert obs.shape == (39,) and obs.dtype == np.float32
        np.testing.assert_array_equal(obs, np.full(39, raw.random_draw, dtype=np.float32))
        assert info == {}
        assert raw.count == 1  # one no-op; return the pre-noop reset state
        assert not raw._freeze_rand_vec
        assert env.unwrapped.evaluation_success_key == "success"
        np.testing.assert_array_equal(raw.actions[0], np.zeros(4))
    finally:
        env.close()


def test_repeated_reward_final_success_and_timeout(adapter):
    env = adapter.MetaWorldEnv("push")
    env.reset(seed=1)
    raw = env._env
    action = np.full(4, 0.25, dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    assert reward == 2 + 3
    assert info["success"] == 0  # raw step 2 succeeded, final raw step 3 did not
    assert not terminated and not truncated
    np.testing.assert_array_equal(observation, np.full(39, 3, dtype=np.float32))
    np.testing.assert_array_equal(action, np.full(4, 0.25, dtype=np.float32))
    np.testing.assert_array_equal(raw.actions[1], raw.actions[2])
    for _ in range(98):
        assert env.step(action)[2:4] == (False, False)
    assert env.step(action)[2:4] == (False, True)
    assert raw.count == 201
    with pytest.raises(RuntimeError, match="reset"):
        env.step(action)
    env.close()
    env.close()
    assert raw.closed


def test_rng_isolation_seed_reproducibility_and_episode_randomization(adapter):
    global_before = np.random.get_state()
    env = adapter.MetaWorldEnv("assembly", seed=9)
    obs1, _ = env.reset(seed=5)
    env.step(np.zeros(4))
    obs2, _ = env.reset()
    obs3, _ = env.reset(seed=5)
    global_after = np.random.get_state()
    assert global_before[0] == global_after[0]
    np.testing.assert_array_equal(global_before[1], global_after[1])
    assert global_before[2:] == global_after[2:]
    np.testing.assert_array_equal(obs1, obs3)
    assert not np.array_equal(obs1, obs2)
    env.close()


@pytest.mark.parametrize("action", [np.zeros(3), [0, 0, 0, np.nan], [0, 0, 0, 1.1]])
def test_bad_actions_fail_before_physics(adapter, action):
    env = adapter.MetaWorldEnv("push")
    env.reset()
    with pytest.raises(ValueError):
        env.step(action)
    assert env._env.count == 1
    env.close()


def test_reset_reports_missing_stable_simulator_state(adapter, monkeypatch):
    env = adapter.MetaWorldEnv("stick-pull")
    try:
        env.reset()
        monkeypatch.setattr(env._env, "step", lambda action: (None, 0.0, False, {}))
        with pytest.raises(RuntimeError, match="no stable state while resetting stick-pull"):
            env.reset()
        with pytest.raises(RuntimeError, match="Call reset"):
            env.step(np.zeros(4))
    finally:
        env.close()


def test_wrong_metaworld_revision_fails_before_simulator_import(monkeypatch):
    from domains import metaworld

    native_source = '{"vcs_info": {"commit_id": "83ac03ca3207c0060112bfc101393ca794ebf1bd"}}'
    installed = SimpleNamespace(read_text=lambda _: native_source)
    monkeypatch.setattr(metaworld.metadata, "distribution", lambda _: installed)
    with pytest.raises(ImportError, match=metaworld.METAWORLD_REVISION):
        metaworld.MetaWorldEnv("push")


def test_rgb_render_uses_upstream_camera(adapter, monkeypatch):
    env = adapter.MetaWorldEnv("push", render_mode="rgb_array")
    raw = env._env
    pixels = np.zeros((384, 384, 3), dtype=np.uint8)

    def render(*, offscreen, resolution, camera_name):
        assert offscreen and resolution == (384, 384) and camera_name == "corner2"
        return pixels

    monkeypatch.setattr(raw, "render", render, raising=False)
    try:
        np.testing.assert_array_equal(raw.model.cam_pos[2], [0.75, 0.075, 0.7])
        frame = env.render()
        np.testing.assert_array_equal(frame, pixels)
        assert not np.shares_memory(frame, pixels)
    finally:
        env.close()


def test_reject_unsupported_task_and_observations(adapter):
    with pytest.raises(ValueError, match="Unknown"):
        adapter.MetaWorldEnv("not-a-task")
    with pytest.raises(ValueError, match="state"):
        adapter.MetaWorldEnv("push", obs="rgb")
    with pytest.raises(ValueError, match="protocol"):
        adapter.MetaWorldEnv("push", protocol="metaworld-native-v2-state-final-success-v1")


@pytest.mark.skipif(
    os.environ.get("AMBI_RUN_REAL_METAWORLD_TESTS") != "1",
    reason="requires the isolated Meta-World runtime; opt in explicitly",
)
@pytest.mark.parametrize("task", TASKS)
def test_real_task_matches_tdmpc2_raw_trajectory(task):
    import domains  # noqa: F401
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

    previous_rng = np.random.get_state()
    env = None
    raw = None
    try:
        seed = 55
        np.random.seed(seed)
        env = gym.make("MetaWorld-v0", task=task)
        raw = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{task}-v2-goal-observable"](seed=seed)
        raw._freeze_rand_vec = False
        raw.model.cam_pos[2] = [0.75, 0.075, 0.7]

        # Follow the official wrapper directly in old Gym's API.
        initial_observation = raw.reset().astype(np.float32)
        raw.step(np.zeros(raw.action_space.shape))
        obs, reset_info = env.reset(seed=seed)
        np.testing.assert_array_equal(obs, initial_observation)
        assert obs.shape == (39,) and obs.dtype == np.float32
        assert reset_info == {}
        rng = np.random.default_rng(21)
        for decision in range(100):
            action = rng.uniform(-1, 1, 4).astype(np.float32)
            _, r1, _, _ = raw.step(action.copy())
            raw_obs, r2, _, raw_info = raw.step(action.copy())
            obs, reward, terminated, truncated, info = env.step(action)
            np.testing.assert_array_equal(obs, raw_obs.astype(np.float32))
            assert reward == r1 + r2
            assert not terminated
            assert truncated == (decision == 99)
            assert info["success"] == raw_info["success"]
        assert env.unwrapped._env.curr_path_length == 201

        # An ordinary reset continues the goal sampler and stacked-frame history.
        next_observation = raw.reset().astype(np.float32)
        np.testing.assert_array_equal(env.reset()[0], next_observation)
        np.testing.assert_array_equal(env.reset(seed=seed)[0], initial_observation)
    finally:
        np.random.set_state(previous_rng)
        if env is not None:
            env.close()
        if raw is not None:
            raw.close()
