from collections import OrderedDict
import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest


class _FakePhysics:
    def __init__(self, env):
        self._env = env
        self.render_calls = []
        self.free_calls = 0

    def render(self, *, height, width, camera_id):
        self.render_calls.append((height, width, camera_id))
        return np.full(
            (height, width, 3),
            self._env.step_count,
            dtype=np.uint8,
        )

    def free(self):
        self.free_calls += 1


class _FakeRawEnv:
    def __init__(self, seed):
        self.seed = seed
        self.step_count = 0
        self.actions = []
        self.close_calls = 0
        self.physics = _FakePhysics(self)
        self._action_spec = SimpleNamespace(
            shape=(2,),
            dtype=np.dtype(np.float64),
            minimum=np.array([-1.0, -1.0], dtype=np.float64),
            maximum=np.array([1.0, 1.0], dtype=np.float64),
        )
        self._observation_spec = OrderedDict(
            (
                ("z_first", SimpleNamespace(shape=(2,))),
                ("a_second", SimpleNamespace(shape=())),
            )
        )

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _observation(self):
        return OrderedDict(
            (
                (
                    "z_first",
                    np.array(
                        [self.step_count, self.step_count + 0.5],
                        dtype=np.float64,
                    ),
                ),
                ("a_second", np.array(self.seed, dtype=np.int64)),
            )
        )

    def reset(self):
        self.step_count = 0
        return SimpleNamespace(observation=self._observation(), reward=None)

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.step_count += 1
        return SimpleNamespace(
            observation=self._observation(),
            reward=float(self.step_count),
            discount=0.0,
        )

    def close(self):
        self.close_calls += 1


class _FakeSuite:
    ALL_TASKS = (
        ("walker", "walk"),
        ("quadruped", "walk"),
        ("ball_in_cup", "catch"),
        ("point_mass", "easy"),
        ("cheetah", "run_backwards"),
    )

    def __init__(self):
        self.load_calls = []
        self.envs = []

    def load(
        self,
        domain,
        task,
        *,
        task_kwargs,
        visualize_reward,
    ):
        self.load_calls.append(
            (domain, task, dict(task_kwargs), visualize_reward)
        )
        env = _FakeRawEnv(task_kwargs["random"])
        self.envs.append(env)
        return env


@pytest.fixture
def fake_dmcontrol(monkeypatch):
    module = importlib.import_module("domains.dmcontrol")
    suite = _FakeSuite()
    scale_calls = []

    class _ActionScale:
        @staticmethod
        def Wrapper(env, *, minimum, maximum):
            scale_calls.append((env, minimum, maximum))
            return env

    monkeypatch.setattr(
        module,
        "_load_dmcontrol_dependencies",
        lambda: (suite, _ActionScale),
    )
    return module, suite, scale_calls


def test_registration_uses_a_lazy_entry_point():
    import domains  # noqa: F401

    spec = gym.spec("DMControl-v0")
    assert spec.entry_point == "domains.dmcontrol:DMControlEnv"
    assert spec.max_episode_steps == 500


def test_importing_domains_does_not_import_dmcontrol():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import domains; "
                "print(int('dm_control' in sys.modules)); "
                "print(int('domains.dmcontrol' in sys.modules))"
            ),
        ],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == ["0", "0"]


def test_state_observation_action_repeat_and_reward_match_tdmpc2(fake_dmcontrol):
    module, suite, scale_calls = fake_dmcontrol
    env = module.DMControlEnv(task="walker-walk")
    try:
        observation, info = env.reset(seed=7)
        assert info == {}
        assert observation.dtype == np.float32
        np.testing.assert_array_equal(observation, [0.0, 0.5, 7.0])
        assert env.observation_space.shape == (3,)
        assert env.observation_type == "state"
        assert env.task_name == "walker-walk"
        assert env.action_repeat == 2
        assert env.frame_stack is None
        assert env.image_size is None

        observation, reward, terminated, truncated, info = env.step(
            np.array([0.25, -0.5], dtype=np.float32)
        )
        np.testing.assert_array_equal(observation, [2.0, 2.5, 7.0])
        assert reward == 3.0
        assert terminated is False
        assert truncated is False
        assert info == {}

        raw = suite.envs[-1]
        assert raw.step_count == 2
        assert len(raw.actions) == 2
        assert all(action.dtype == np.float64 for action in raw.actions)
        assert scale_calls[-1][1:] == (-1.0, 1.0)
        np.testing.assert_array_equal(env.action_space.low, [-1.0, -1.0])
        np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0])
    finally:
        env.close()


def test_explicit_reset_reconstructs_seed_and_unseeded_reset_reuses_env(
    fake_dmcontrol,
):
    module, suite, _ = fake_dmcontrol
    env = module.DMControlEnv(task="walker_walk")
    initial = suite.envs[-1]
    try:
        env.reset(seed=11)
        first_seeded = suite.envs[-1]
        assert initial.close_calls == 1
        assert initial.physics.free_calls == 1

        env.reset()
        assert suite.envs[-1] is first_seeded
        assert first_seeded.close_calls == 0

        env.reset(seed=11)
        second_seeded = suite.envs[-1]
        assert second_seeded is not first_seeded
        assert first_seeded.close_calls == 1
        assert first_seeded.physics.free_calls == 1
        assert [call[2]["random"] for call in suite.load_calls] == [0, 11, 11]
    finally:
        final_env = suite.envs[-1]
        env.close()
        assert final_env.close_calls == 1
        assert final_env.physics.free_calls == 1
        env.close()
        assert final_env.close_calls == 1
        assert final_env.physics.free_calls == 1


def test_rgb_observation_has_exact_upstream_frame_stack(fake_dmcontrol):
    module, suite, _ = fake_dmcontrol
    env = module.DMControlEnv(task="walker-walk", obs="rgb")
    try:
        observation, _ = env.reset(seed=3)
        assert observation.shape == (9, 64, 64)
        assert observation.dtype == np.uint8
        assert observation.flags.c_contiguous
        assert np.all(observation == 0)
        assert env.observation_type == "rgb"
        assert env.frame_stack == 3
        assert env.image_size == 64

        first, *_ = env.step(np.zeros(2, dtype=np.float32))
        assert np.all(first[:6] == 0)
        assert np.all(first[6:] == 2)

        second, *_ = env.step(np.zeros(2, dtype=np.float32))
        assert np.all(second[:3] == 0)
        assert np.all(second[3:6] == 2)
        assert np.all(second[6:] == 4)
        assert suite.envs[-1].physics.render_calls == [
            (64, 64, 0),
            (64, 64, 0),
            (64, 64, 0),
        ]
    finally:
        env.close()


def test_policy_pixels_are_independent_of_public_render_mode(fake_dmcontrol):
    module, suite, _ = fake_dmcontrol
    pixels = module.DMControlEnv(task="quadruped-walk", obs="rgb")
    public = module.DMControlEnv(
        task="walker-walk", render_mode="rgb_array"
    )
    try:
        pixels.reset()
        assert pixels.render() is None
        assert suite.envs[-2].physics.render_calls[-1] == (64, 64, 2)

        public.reset()
        frame = public.render()
        assert frame.shape == (384, 384, 3)
        assert frame.dtype == np.uint8
        assert suite.envs[-1].physics.render_calls[-1] == (384, 384, 0)
    finally:
        pixels.close()
        public.close()


def test_registered_time_limit_truncates_only_at_500_decisions(fake_dmcontrol):
    _, suite, _ = fake_dmcontrol
    env = gym.make("DMControl-v0", task="walker-walk")
    try:
        env.reset(seed=2)
        assert env.get_wrapper_attr("observation_type") == "state"
        for step in range(1, 501):
            _, _, terminated, truncated, _ = env.step(
                np.zeros(2, dtype=np.float64)
            )
            assert terminated is False
            assert truncated is (step == 500)
        assert suite.envs[-1].step_count == 1000
    finally:
        env.close()


@pytest.mark.parametrize("task", ["mt30", "mt80"])
def test_multitask_names_are_rejected(fake_dmcontrol, task):
    module, _, _ = fake_dmcontrol
    with pytest.raises(ValueError, match="one task at a time"):
        module.DMControlEnv(task=task)


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("cup-catch", ("ball_in_cup", "catch", "cup-catch")),
        ("ball_in_cup-catch", ("ball_in_cup", "catch", "cup-catch")),
        ("pointmass-easy", ("point_mass", "easy", "pointmass-easy")),
        ("point_mass-easy", ("point_mass", "easy", "pointmass-easy")),
        (
            "cheetah-run-backwards",
            ("cheetah", "run_backwards", "cheetah-run-backwards"),
        ),
    ],
)
def test_tdmpc2_task_aliases(fake_dmcontrol, task, expected):
    module, suite, _ = fake_dmcontrol
    env = module.DMControlEnv(task=task)
    try:
        assert (
            env.domain_name,
            env.dmcontrol_task_name,
            env.task_name,
        ) == expected
        assert suite.load_calls[-1][:2] == expected[:2]
    finally:
        env.close()


def test_invalid_configuration_fails_before_loading_a_raw_task(fake_dmcontrol):
    module, suite, _ = fake_dmcontrol
    with pytest.raises(ValueError, match="'state' or 'rgb'"):
        module.DMControlEnv(task="walker-walk", obs="pixels")
    with pytest.raises(ValueError, match="render_mode"):
        module.DMControlEnv(task="walker-walk", render_mode="human")
    with pytest.raises(ValueError, match="Unknown DMControl task"):
        module.DMControlEnv(task="walker-fly")
    assert suite.load_calls == []


def test_reset_rejects_nonempty_options(fake_dmcontrol):
    module, _, _ = fake_dmcontrol
    env = module.DMControlEnv(task="walker-walk")
    try:
        env.reset(options={})
        with pytest.raises(ValueError, match="reset options"):
            env.reset(options={"difficulty": "hard"})
    finally:
        env.close()


def test_vendored_custom_task_assets_are_complete():
    task_dir = Path(__file__).parents[1] / "domains" / "dmcontrol_tasks"
    expected_python = {
        "ball_in_cup.py",
        "cheetah.py",
        "fish.py",
        "hopper.py",
        "pendulum.py",
        "reacher.py",
        "walker.py",
    }
    expected_xml = {
        "ball_in_cup.xml",
        "cheetah.xml",
        "fish.xml",
        "hopper.xml",
        "pendulum.xml",
        "reacher_three_links.xml",
        "reacher_four_links.xml",
        "walker.xml",
    }
    assert expected_python <= {path.name for path in task_dir.glob("*.py")}
    assert expected_xml <= {path.name for path in task_dir.glob("*.xml")}
    assert (task_dir / "LICENSE.tdmpc2").is_file()


def test_real_dmcontrol_smoke_and_custom_registry():
    if os.environ.get("AMBI_RUN_REAL_DMCONTROL_TESTS") != "1":
        pytest.skip("set AMBI_RUN_REAL_DMCONTROL_TESTS=1 for rendering-host smoke")
    pytest.importorskip("dm_control")
    from dm_control import suite
    from dm_control.suite.wrappers import action_scale

    import domains  # noqa: F401

    env = gym.make("DMControl-v0", task="walker-walk")
    reference = action_scale.Wrapper(
        suite.load(
            "walker",
            "walk",
            task_kwargs={"random": 0},
            visualize_reward=False,
        ),
        minimum=-1.0,
        maximum=1.0,
    )
    try:
        observation, info = env.reset(seed=0)
        reference_timestep = reference.reset()
        reference_observation = np.concatenate(
            [
                np.asarray(value).reshape(-1)
                for value in reference_timestep.observation.values()
            ]
        ).astype(np.float32, copy=False)
        assert observation.shape == env.observation_space.shape
        assert observation.dtype == np.float32
        np.testing.assert_array_equal(observation, reference_observation)
        assert info == {}

        action = np.linspace(-0.25, 0.25, env.action_space.shape[0])
        for decision in range(1, 501):
            observation, reward, terminated, truncated, _ = env.step(action)
            reference_reward = 0.0
            for _ in range(2):
                reference_timestep = reference.step(
                    action.astype(reference.action_spec().dtype)
                )
                reference_reward += float(reference_timestep.reward)
            reference_observation = np.concatenate(
                [
                    np.asarray(value).reshape(-1)
                    for value in reference_timestep.observation.values()
                ]
            ).astype(np.float32, copy=False)
            np.testing.assert_array_equal(observation, reference_observation)
            assert reward == reference_reward
            assert terminated is False
            assert truncated is (decision == 500)

        expected_custom_tasks = {
            ("ball_in_cup", "spin"),
            ("cheetah", "flip"),
            ("cheetah", "flip_backwards"),
            ("cheetah", "jump"),
            ("cheetah", "legs_up"),
            ("cheetah", "lie_down"),
            ("cheetah", "run_back"),
            ("cheetah", "run_backwards"),
            ("cheetah", "run_front"),
            ("cheetah", "stand_back"),
            ("cheetah", "stand_front"),
            ("fish", "obstacles"),
            ("hopper", "flip"),
            ("hopper", "flip_backwards"),
            ("hopper", "hop_backwards"),
            ("pendulum", "spin"),
            ("reacher", "four_easy"),
            ("reacher", "four_hard"),
            ("reacher", "three_easy"),
            ("reacher", "three_hard"),
            ("walker", "arabesque"),
            ("walker", "backflip"),
            ("walker", "flip"),
            ("walker", "headstand"),
            ("walker", "legs_up"),
            ("walker", "lie_down"),
            ("walker", "run_backwards"),
            ("walker", "walk_backwards"),
        }
        assert expected_custom_tasks <= set(suite.ALL_TASKS)
    finally:
        env.close()
        reference_physics = reference.physics
        reference.close()
        reference_physics.free()

    pixels = gym.make(
        "DMControl-v0",
        task="walker-walk",
        obs="rgb",
        render_mode="rgb_array",
    )
    try:
        initial, _ = pixels.reset(seed=0)
        assert initial.shape == (9, 64, 64)
        assert initial.dtype == np.uint8
        np.testing.assert_array_equal(initial[:3], initial[3:6])
        np.testing.assert_array_equal(initial[3:6], initial[6:])
        following, *_ = pixels.step(np.zeros(pixels.action_space.shape))
        np.testing.assert_array_equal(following[:6], initial[3:])
        public_frame = pixels.render()
        assert public_frame.shape == (384, 384, 3)
        assert public_frame.dtype == np.uint8
    finally:
        pixels.close()
