import copy
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
from utils.resume_runtime import (
    ResumeRuntimeMismatch,
    capture_environment_state,
    restore_environment_state,
)


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
