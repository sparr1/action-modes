import copy
import importlib
import json
import shutil
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from domains.ant_3leg_deadstump_env import Ant3LegDeadStumpEnv
from domains.ant_leg_adaptation import AntLegAdaptationEnv
from domains.ant_variable_legs import AntVariableLegsEnv
import domains.ant_variable_legs as ant_variable_legs_module
from utils.resume_runtime import (
    ResumeRuntimeMismatch,
    capture_environment_state,
    restore_environment_state,
)
from utils.core import build_env


ASSETS = Path(__file__).parents[1] / "domains" / "assets"


@pytest.fixture(autouse=True)
def _supply_unused_imageio_when_dmcontrol_env_is_minimal(monkeypatch):
    try:
        __import__("imageio")
    except ModuleNotFoundError:
        monkeypatch.setitem(sys.modules, "imageio", types.ModuleType("imageio"))


def _asset_snapshot():
    return {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in ASSETS.glob("*.xml")
    }


def _motor_count(path):
    return len(ET.parse(path).getroot().findall(".//actuator/motor"))


def test_custom_ant_instances_use_independent_xml_without_touching_assets():
    before = _asset_snapshot()
    envs = []
    paths = []
    try:
        envs = [
            AntVariableLegsEnv(num_legs=3),
            AntVariableLegsEnv(num_legs=5),
            Ant3LegDeadStumpEnv(),
        ]
        paths = [Path(env.xml_file).resolve() for env in envs]

        assert len(set(paths)) == len(paths)
        assert all(path.is_file() and ASSETS.resolve() not in path.parents for path in paths)
        assert [_motor_count(path) for path in paths] == [6, 10, 6]
        assert [env.model.nu for env in envs] == [6, 10, 6]

        # Constructing the five-leg model must not overwrite the first model.
        assert _motor_count(paths[0]) == 6
        assert _asset_snapshot() == before
    finally:
        for env in envs:
            env.close()

    assert paths and not any(path.exists() for path in paths)
    assert _asset_snapshot() == before


def test_custom_ant_preserves_absolute_static_xml_input(tmp_path):
    source = tmp_path / "renamed-dead-stump.xml"
    shutil.copy2(ASSETS / "ant_3leg_deadstump.xml", source)
    source_bytes = source.read_bytes()
    env = Ant3LegDeadStumpEnv(xml_file=str(source))
    generated = Path(env.xml_file)
    try:
        assert generated.resolve() != source.resolve()
        assert generated.is_file()
        assert env.model.nu == 6
        assert source.read_bytes() == source_bytes
    finally:
        env.close()
    assert not generated.exists()


def test_variable_ant_registration_is_headless_by_default():
    spec = gym.spec("VarLegsAnt-v0")
    assert spec.kwargs["render_mode"] is None


@pytest.mark.parametrize("num_legs", (True, 0, -1, 3.5, "3"))
def test_variable_ant_rejects_non_positive_or_non_integral_leg_counts(num_legs):
    with pytest.raises(ValueError, match="num_legs must be a positive integer"):
        AntVariableLegsEnv(num_legs=num_legs)


@pytest.mark.parametrize("num_legs", (False, 0, 4, 3.0))
def test_dead_stump_rejects_any_leg_count_other_than_integer_three(num_legs):
    with pytest.raises(ValueError, match="requires num_legs=3"):
        Ant3LegDeadStumpEnv(num_legs=num_legs)


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"total_timesteps": True}, "total_timesteps"),
        ({"total_timesteps": 0}, "total_timesteps"),
        ({"total_timesteps": 2.5}, "total_timesteps"),
        ({"total_timesteps": 10, "start_legs": 0}, "start_legs"),
        ({"total_timesteps": 10, "end_legs": 2.5}, "end_legs"),
        (
            {"total_timesteps": 10, "start_legs": 3, "end_legs": 4},
            "cannot exceed start_legs",
        ),
        ({"total_timesteps": 10, "switch_fraction": float("nan")}, "finite"),
        ({"total_timesteps": 10, "switch_fraction": float("inf")}, "finite"),
        ({"total_timesteps": 10, "switch_fraction": -0.1}, r"\[0, 1\]"),
        ({"total_timesteps": 10, "switch_fraction": 1.1}, r"\[0, 1\]"),
    ),
)
def test_leg_adaptation_rejects_invalid_morphology_schedule(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AntLegAdaptationEnv(**kwargs)


def test_zero_fraction_leg_adaptation_switches_on_first_reset():
    env = AntLegAdaptationEnv(
        total_timesteps=10,
        switch_fraction=0.0,
        terminate_when_unhealthy=False,
    )
    try:
        _, info = env.reset(seed=7)
        assert env._switched
        assert not env._pending_switch
        assert env.model.nu == 6
        assert info["num_legs"] == 3
    finally:
        env.close()


def test_variable_ant_constructor_cleans_xml_workspace_on_base_exception(
    monkeypatch, tmp_path
):
    workspace_path = tmp_path / "generated-ant"

    class _Workspace:
        def __init__(self):
            workspace_path.mkdir()
            self.name = str(workspace_path)
            self.cleanup_calls = 0

        def cleanup(self):
            self.cleanup_calls += 1
            shutil.rmtree(workspace_path)

    workspace = _Workspace()
    monkeypatch.setattr(
        ant_variable_legs_module.tempfile,
        "TemporaryDirectory",
        lambda **_kwargs: workspace,
    )

    def _interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt("construction interrupted")

    monkeypatch.setattr(
        ant_variable_legs_module.MujocoEnv,
        "__init__",
        _interrupt,
    )

    with pytest.raises(KeyboardInterrupt, match="construction interrupted"):
        AntVariableLegsEnv(num_legs=4)
    assert workspace.cleanup_calls == 1
    assert not workspace_path.exists()


def test_variable_ant_applies_and_reports_contact_cost():
    env = AntVariableLegsEnv(num_legs=4, terminate_when_unhealthy=False)
    try:
        env.reset(seed=11)
        env.contact_cost = lambda _forces: 2.5
        _, reward, _, _, info = env.step(np.zeros(env.action_space.shape))

        assert info["reward_contact"] == -2.5
        assert reward == pytest.approx(
            info["reward_forward"]
            + info["reward_survive"]
            + info["reward_ctrl"]
            + info["reward_contact"]
        )
    finally:
        env.close()


def test_adaptation_replaces_and_closes_renderer_with_morphology():
    env = AntLegAdaptationEnv(
        total_timesteps=1,
        switch_fraction=1.0,
        terminate_when_unhealthy=False,
    )
    try:
        old_renderer = env.mujoco_renderer
        old_model = env.model
        close_calls = []
        old_renderer.close = lambda: close_calls.append(True)

        env.reset(seed=13)
        env.step(np.zeros(env.action_space.shape))
        env.reset()

        assert close_calls == [True]
        assert env.model is not old_model
        assert env.mujoco_renderer is not old_renderer
        assert env.mujoco_renderer.model is env.model
        assert env.mujoco_renderer.data is env.data
        assert env.mujoco_renderer.model.nu == 6
    finally:
        env.close()


def test_adaptation_preserves_renderer_switch_error_and_notes_cleanup_failure(
    monkeypatch,
):
    env = AntLegAdaptationEnv(
        total_timesteps=1,
        switch_fraction=1.0,
        terminate_when_unhealthy=False,
    )
    old_renderer = env.mujoco_renderer
    original_close = old_renderer.close
    try:
        rendering = importlib.import_module(
            "gymnasium.envs.mujoco.mujoco_rendering"
        )

        class _CleanupFailingRenderer:
            def __init__(self, model, data, default_cam_config=None):
                self.model = model
                self.data = data

            def close(self):
                raise ValueError("replacement cleanup failed")

        monkeypatch.setattr(rendering, "MujocoRenderer", _CleanupFailingRenderer)

        def _old_close_fails():
            raise RuntimeError("old renderer close failed")

        old_renderer.close = _old_close_fails
        env.reset(seed=19)
        env.step(np.zeros(env.action_space.shape))
        with pytest.raises(RuntimeError, match="old renderer close failed") as exc_info:
            env.reset()

        assert env.mujoco_renderer is old_renderer
        assert not env._switched
        assert any(
            "Additional replacement-renderer cleanup failure: "
            "replacement cleanup failed" in note
            for note in getattr(exc_info.value, "__notes__", ())
        )
    finally:
        old_renderer.close = original_close
        env.close()


def test_tracked_antplane_then_subtask_wrapper_order_constructs_and_steps():
    run_params = {
        "env": "Ant-v4",
        "env_wrappers": [
            {
                "name": "AntPlane",
                "wrapper_params": {
                    "random_resets": False,
                    "random_rotation": False,
                },
            },
            {
                "name": "Subtask",
                "wrapper_params": {
                    "task": "domains.Maze:Move",
                    "task_params": {
                        "direction": "X",
                        "desired_velocity_minimum": 1.0,
                        "desired_velocity_maximum": 1.0,
                        "metric": "L1",
                    },
                },
            },
        ],
    }
    env = build_env(
        run_params,
        {
            "env_params": {
                "render_mode": None,
                "exclude_current_positions_from_observation": False,
            }
        },
    )
    try:
        observation, _ = env.reset(seed=17)
        assert observation["desired_goal"].shape == (1,)
        assert observation["desired_goal"].dtype == np.float64
        assert env.observation_space.contains(observation)

        next_observation, reward, terminated, truncated, _ = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        assert env.observation_space.contains(next_observation)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()


def test_subtask_without_antplane_task_info_fails_during_env_construction():
    with pytest.raises(ValueError, match="requires explicit task_info"):
        build_env(
            {
                "env": "Ant-v4",
                "env_wrapper": {
                    "name": "Subtask",
                    "wrapper_params": {
                        "task": "domains.Maze:Move",
                        "task_params": {
                            "desired_velocity_minimum": 1.0,
                            "desired_velocity_maximum": 1.0,
                            "metric": "L1",
                        },
                    },
                },
            },
            {"env_params": {"render_mode": None}},
        )


def test_tracked_legacy_move_manifest_uses_explicit_historical_task_info():
    root = Path(__file__).parents[1]
    experiment = json.loads(
        (root / "configs/experiments/AntPlaneMove2.json").read_text()
    )
    env = build_env(experiment["overrides_alg"], experiment)
    try:
        observation, _ = env.reset(seed=23)
        assert env._task.velocity_coords == [15, 21]
        assert env._task.dir_coords == [3, 7]
        assert env.observation_space.contains(observation)

        next_observation, reward, terminated, truncated, _ = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        assert env.observation_space.contains(next_observation)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()


def test_adaptation_resume_switches_private_models_at_the_same_boundary():
    before = _asset_snapshot()
    envs = []
    try:
        envs = [
            AntLegAdaptationEnv(
                total_timesteps=2,
                switch_fraction=0.5,
                terminate_when_unhealthy=False,
            )
            for _ in range(2)
        ]
        source, restored = envs
        for index, env in enumerate(envs):
            env.spec = types.SimpleNamespace(id="LegAdaptAnt-v0", max_episode_steps=1000)
            env.action_space.seed(10 + index)
            env.observation_space.seed(20 + index)
            env.reset(seed=30 + index)

        source.step(np.zeros(source.action_space.shape, dtype=np.float64))
        assert source._pending_switch and not source._switched

        restore_environment_state(restored, capture_environment_state(source))
        source_obs, _ = source.reset()
        restored_obs, _ = restored.reset()

        assert source._switched and restored._switched
        assert source.model.nu == restored.model.nu == 6
        assert Path(source.xml_file).resolve() != Path(restored.xml_file).resolve()
        np.testing.assert_array_equal(source_obs, restored_obs)
        assert _asset_snapshot() == before
    finally:
        for env in envs:
            env.close()

    assert _asset_snapshot() == before


@pytest.mark.parametrize(
    "corruption",
    (
        "missed-switch",
        "early-pending",
        "early-switched",
        "pending-and-switched",
    ),
)
def test_adaptation_resume_rejects_impossible_switch_states_before_mutation(
    corruption,
):
    envs = []
    try:
        envs = [
            gym.wrappers.TimeLimit(
                AntLegAdaptationEnv(
                    total_timesteps=4,
                    switch_fraction=0.5,
                    terminate_when_unhealthy=False,
                ),
                max_episode_steps=1,
            )
            for _ in range(2)
        ]
        source, target = envs
        for env in envs:
            env.action_space.seed(10)
            env.observation_space.seed(20)
            env.reset(seed=30)
            _, _, _, truncated, _ = env.step(
                np.zeros(env.action_space.shape, dtype=np.float64)
            )
            assert truncated

        invalid = copy.deepcopy(capture_environment_state(source))
        base_state = invalid["base_state"]
        if corruption == "missed-switch":
            base_state["steps_taken"] = base_state["switch_step"]
        elif corruption == "early-pending":
            base_state["pending_switch"] = True
        elif corruption == "early-switched":
            base_state["switched"] = True
            base_state["num_legs"] = base_state["end_legs"]
        else:
            base_state["pending_switch"] = True
            base_state["switched"] = True
            base_state["num_legs"] = base_state["end_legs"]

        before = capture_environment_state(target)
        with pytest.raises(ResumeRuntimeMismatch, match="Base environment"):
            restore_environment_state(target, invalid)
        assert capture_environment_state(target) == before
    finally:
        for env in envs:
            env.close()
