from collections import OrderedDict
import inspect
import json
import types

import gymnasium as gym
import numpy as np
import pytest

import modes.controller as controller_module
from domains.Maze import Change, Move
from domains.mpqdn_wrappers import (
    FlattenedActionWrapper,
    FlattenStateWrapper,
    ModalWrapper,
    QPAMDPScaledParameterisedActionWrapper,
    ScaledParameterisedActionWrapper,
    ScaledStateWrapper,
    TimestepWrapper,
)
from modes.controller import ModalController, OrchestralController
from modes.tasks import Subtask


class _ObservationEnv(gym.Env):
    action_space = gym.spaces.Discrete(2)

    def __init__(self, observation_space, observation):
        self.observation_space = observation_space
        self._observation = observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._observation, {"reset": True}

    def step(self, action):
        return self._observation, float(action), False, True, {"step": True}


class _ActionEnv(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def __init__(self, action_space):
        self.action_space = action_space
        self.actions = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        self.actions.append(action)
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


@pytest.mark.parametrize(
    "space, observation, expected",
    (
        (
            gym.spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32),
            np.array([[1.0, 0.5], [-0.5, -1.0]], dtype=np.float32),
            [1.0, 0.5, -0.5, -1.0],
        ),
        (
            gym.spaces.Tuple(
                (
                    gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                    gym.spaces.Box(0.0, 5.0, shape=(1, 2), dtype=np.float64),
                )
            ),
            (
                np.array([0.25, -0.5], dtype=np.float32),
                np.array([[2.0, 3.0]], dtype=np.float64),
            ),
            [0.25, -0.5, 2.0, 3.0],
        ),
        (
            gym.spaces.Dict(
                OrderedDict(
                    (
                        (
                            "z_first",
                            gym.spaces.Box(0, 10, shape=(2,), dtype=np.int32),
                        ),
                        (
                            "a_second",
                            gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
                        ),
                    )
                )
            ),
            OrderedDict(
                (
                    ("z_first", np.array([2, 3], dtype=np.int32)),
                    ("a_second", np.array([0.5], dtype=np.float32)),
                )
            ),
            [2.0, 3.0, 0.5],
        ),
    ),
)
def test_flatten_state_wrapper_spaces_and_gymnasium_api(space, observation, expected):
    env = FlattenStateWrapper(_ObservationEnv(space, observation))

    reset_observation, reset_info = env.reset(seed=3)
    step_observation, reward, terminated, truncated, step_info = env.step(1)

    np.testing.assert_array_equal(reset_observation, expected)
    np.testing.assert_array_equal(step_observation, expected)
    assert env.observation_space.shape == (len(expected),)
    assert env.observation_space.contains(reset_observation)
    assert reset_info == {"reset": True}
    assert reward == 1.0
    assert not terminated and truncated
    assert step_info == {"step": True}


def test_modal_wrappers_preserve_the_achieved_goal_space():
    achieved = gym.spaces.Box(
        low=np.array([-2.0], dtype=np.float32),
        high=np.array([2.0], dtype=np.float32),
    )
    desired = gym.spaces.Box(
        low=np.array([-10.0], dtype=np.float32),
        high=np.array([10.0], dtype=np.float32),
    )
    observation = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    base = _ObservationEnv(
        gym.spaces.Dict(
            {
                "achieved_goal": achieved,
                "desired_goal": desired,
                "observation": observation,
            }
        ),
        {
            "achieved_goal": np.array([1.5], dtype=np.float32),
            "desired_goal": np.array([8.0], dtype=np.float32),
            "observation": np.zeros(2, dtype=np.float32),
        },
    )

    for wrapper_class in (ModalWrapper, ModalController.ModalWrapper):
        wrapped = wrapper_class(base)
        result, _ = wrapped.reset()
        assert wrapped.observation_space["achieved_goal"] == achieved
        assert wrapped.observation_space.contains(result)


@pytest.mark.parametrize(
    "space",
    (
        gym.spaces.Discrete(2),
        gym.spaces.Tuple(
            (gym.spaces.Box(-1, 1, shape=(1,)), gym.spaces.Discrete(2))
        ),
    ),
)
def test_flatten_state_wrapper_rejects_unsupported_spaces(space):
    with pytest.raises(TypeError, match="requires|only Box"):
        FlattenStateWrapper(_ObservationEnv(space, space.sample()))


def test_timestep_wrapper_emits_gymnasium_contract_inside_declared_space():
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    raw_observation = np.array([0.25, -0.5], dtype=np.float32)
    env = TimestepWrapper(
        _ObservationEnv(observation_space, raw_observation),
        max_steps=1,
    )

    observation, info = env.reset(seed=5)
    next_observation, reward, terminated, truncated, step_info = env.step(1)

    assert observation[1] == next_observation[1] == 1
    assert env.observation_space.contains(observation)
    assert env.observation_space.contains(next_observation)
    assert info == {"reset": True}
    assert reward == 1.0
    assert not terminated and truncated
    assert step_info == {"step": True}


@pytest.mark.parametrize("max_steps", (True, 0, -1, 1.5))
def test_timestep_wrapper_rejects_invalid_max_steps(max_steps):
    env = _ObservationEnv(
        gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="positive integer"):
        TimestepWrapper(env, max_steps=max_steps)


@pytest.mark.parametrize(
    "space, message",
    (
        (
            gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            "finite bounds",
        ),
        (
            gym.spaces.Box(
                low=np.array([0.0, 1.0]),
                high=np.array([0.0, 2.0]),
                dtype=np.float32,
            ),
            "strictly increasing",
        ),
    ),
)
def test_scaled_state_wrapper_fails_fast_on_unsafe_linear_bounds(space, message):
    with pytest.raises(ValueError, match=message):
        ScaledStateWrapper(_ObservationEnv(space, np.zeros(space.shape)))


def test_scaled_state_wrapper_maps_finite_box_and_gymnasium_api():
    space = gym.spaces.Box(
        low=np.array([-2.0, 10.0], dtype=np.float32),
        high=np.array([2.0, 20.0], dtype=np.float32),
    )
    env = ScaledStateWrapper(
        _ObservationEnv(space, np.array([0.0, 15.0], dtype=np.float32))
    )

    observation, info = env.reset()
    next_observation, _, _, truncated, _ = env.step(0)

    np.testing.assert_array_equal(observation, [0.0, 0.0])
    np.testing.assert_array_equal(next_observation, [0.0, 0.0])
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert info == {"reset": True}
    assert truncated


def _nested_parameter_action_space():
    return gym.spaces.Tuple(
        (
            gym.spaces.Discrete(2),
            gym.spaces.Tuple(
                (
                    gym.spaces.Box(-2.0, 2.0, shape=(1,), dtype=np.float32),
                    gym.spaces.Box(
                        low=np.array([10.0, 20.0], dtype=np.float32),
                        high=np.array([20.0, 40.0], dtype=np.float32),
                    ),
                )
            ),
        )
    )


def test_flattened_and_scaled_action_wrappers_reconstruct_nested_action():
    raw_env = _ActionEnv(_nested_parameter_action_space())
    flattened = FlattenedActionWrapper(raw_env)
    env = ScaledParameterisedActionWrapper(flattened)

    env.step(
        (
            1,
            np.array([0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        )
    )

    action_id, parameters = raw_env.actions[-1]
    assert action_id == 1
    assert isinstance(parameters, tuple)
    np.testing.assert_array_equal(parameters[0], [0.0])
    np.testing.assert_array_equal(parameters[1], [15.0, 40.0])

    env.step(
        (
            0,
            [
                np.array([1.0], dtype=np.float32),
                np.array([0.0, 0.0], dtype=np.float32),
            ],
        )
    )
    action_id, parameters = raw_env.actions[-1]
    assert action_id == 0
    np.testing.assert_array_equal(parameters[0], [2.0])


@pytest.mark.parametrize(
    "action, message",
    (
        ((2, np.zeros(1), np.zeros(2)), "outside"),
        ((0, np.zeros(1)), "contain an id"),
        ((0, np.zeros(2), np.zeros(2)), "shape"),
    ),
)
def test_flattened_action_wrapper_rejects_invalid_actions(action, message):
    wrapper = FlattenedActionWrapper(_ActionEnv(_nested_parameter_action_space()))
    with pytest.raises(ValueError, match=message):
        wrapper.action(action)


def test_qpamdp_scaled_action_reconstructs_nested_tuple_without_mutation():
    wrapper = QPAMDPScaledParameterisedActionWrapper(
        _ActionEnv(_nested_parameter_action_space())
    )
    original = (
        0,
        (
            np.array([0.5], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
        ),
    )

    action_id, parameters = wrapper.action(original)

    assert action_id == 0
    assert isinstance(parameters, tuple)
    np.testing.assert_array_equal(parameters[0], [1.0])
    np.testing.assert_array_equal(parameters[1], [0.0, 0.0])
    np.testing.assert_array_equal(original[1][0], [0.5])


@pytest.mark.parametrize("high", (np.inf, 0.0))
def test_parameter_scaling_wrappers_reject_nonfinite_or_zero_width_bounds(high):
    bad_box = gym.spaces.Box(
        low=np.array([0.0], dtype=np.float32),
        high=np.array([high], dtype=np.float32),
        dtype=np.float32,
    )
    flat_space = gym.spaces.Tuple((gym.spaces.Discrete(1), bad_box))
    nested_space = gym.spaces.Tuple(
        (gym.spaces.Discrete(1), gym.spaces.Tuple((bad_box,)))
    )

    with pytest.raises(ValueError, match="finite|strictly increasing"):
        ScaledParameterisedActionWrapper(_ActionEnv(flat_space))
    with pytest.raises(ValueError, match="finite|strictly increasing"):
        QPAMDPScaledParameterisedActionWrapper(_ActionEnv(nested_space))


class _Task:
    def __init__(self, goal=(0.5,), goal_length=1):
        self.goal = goal
        self.goal_length = goal_length
        self.task_info = None

    def set_task_info(self, task_info):
        self.task_info = task_info

    def get_goal_length(self):
        return self.goal_length

    def get_goal(self):
        return self.goal

    def reset(self, seed=None):
        return None

    def get_reward(self, observation, action, contact_forces):
        return 0.0, {}

    def get_termination(self, observation):
        return False


def _box_env():
    return _ObservationEnv(
        gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        np.array([0.25, -0.5], dtype=np.float32),
    )


def test_subtask_requires_task_info_at_construction():
    with pytest.raises(ValueError, match="requires explicit task_info"):
        Subtask(_box_env(), _Task())


def test_subtask_explicit_task_info_and_goal_contract_are_valid():
    task = _Task(goal=(0.5,))
    env = Subtask(_box_env(), task, task_info={"coordinates": (0, 1)})

    observation, _ = env.reset(seed=7)

    assert task.task_info == {"coordinates": (0, 1)}
    assert observation["desired_goal"].dtype == np.float64
    assert observation["desired_goal"].shape == (1,)
    assert env.observation_space.contains(observation)


def test_subtask_rejects_goal_shape_drift_during_construction():
    with pytest.raises(ValueError, match="goal shape changed"):
        Subtask(_box_env(), _Task(goal=(0.5, 0.75), goal_length=1), task_info={})


@pytest.mark.parametrize("goal", ((float("nan"),), (float("inf"),)))
def test_subtask_rejects_nonfinite_goals(goal):
    with pytest.raises(ValueError, match="finite values"):
        Subtask(_box_env(), _Task(goal=goal), task_info={})


def test_subtask_does_not_swallow_contact_force_runtime_failures():
    class _BrokenContactEnv(_ObservationEnv):
        @property
        def contact_forces(self):
            raise RuntimeError("contact sensor failed")

    env = _BrokenContactEnv(
        gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )
    wrapped = Subtask(env, _Task(), task_info={})
    with pytest.raises(RuntimeError, match="contact sensor failed"):
        wrapped.step(0)


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"direction": "sideways"}, "direction"),
        ({"metric": "squared"}, "metric"),
        (
            {"desired_velocity_minimum": 2.0, "desired_velocity_maximum": 1.0},
            "cannot exceed",
        ),
        ({"adaptive_margin_minimum": 0.0}, "adaptive_margin_minimum"),
        ({"metric": "huber", "margin": 0.0}, "huber margin"),
    ),
)
def test_move_rejects_invalid_constructor_contracts(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Move(**kwargs)


def test_move_maximize_goal_and_goal_setter_remain_finite_scalars():
    task = Move(
        desired_velocity_minimum=0.0,
        desired_velocity_maximum=float("inf"),
    )
    task.reset(seed=3)
    assert task.get_goal() == 0.0
    task.set_goal(np.array([1.25], dtype=np.float32))
    assert task.get_goal() == pytest.approx(1.25)
    with pytest.raises(ValueError, match="finite scalar"):
        task.set_goal([1.0, 2.0])


def test_move_and_change_contact_cost_require_available_forces():
    with pytest.raises(RuntimeError, match="Move contact_cost requires"):
        Move(contact_cost=True).contact_cost(None)
    with pytest.raises(RuntimeError, match="Change contact_cost requires"):
        Change(contact_cost=True).contact_cost(None)


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"target_coords": ""}, "target_coords"),
        ({"target_coords": "XX"}, "target_coords"),
        ({"target_coords": "Q"}, "target_coords"),
        ({"metric": "squared"}, "metric"),
        (
            {"desired_coord_minimum": 2.0, "desired_coord_maximum": 1.0},
            "cannot exceed",
        ),
        ({"metric": "huber", "margin": 0.0}, "huber margin"),
    ),
)
def test_change_rejects_invalid_constructor_contracts(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Change(**kwargs)


def test_change_goal_setter_enforces_coordinate_shape_and_numeric_dtype():
    task = Change(target_coords="XZ")
    task.set_goal(np.array([0.4, 0.8], dtype=np.float32))
    assert task.get_goal() == pytest.approx([0.4, 0.8])
    with pytest.raises(ValueError, match="2 finite values"):
        task.set_goal([0.5])


def test_controller_model_loading_preserves_initialize_alg_tuple_contract(
    monkeypatch, tmp_path
):
    config = tmp_path / "algorithm.json"
    config.write_text(json.dumps({"alg": "Fake", "alg_params": {}}))
    fake_model = types.SimpleNamespace(load_calls=[])
    fake_model.load = lambda path: fake_model.load_calls.append(path)
    monkeypatch.setattr(
        controller_module,
        "initialize_alg",
        lambda *_args, **_kwargs: (fake_model, True, "FakeAlgorithm"),
    )

    modal = object.__new__(ModalController)
    assert modal.load_model(str(config), object(), "checkpoint", object()) == (
        fake_model,
        True,
        "FakeAlgorithm",
    )

    orchestral = object.__new__(OrchestralController)
    orchestral.orchestral_action_space = object()
    assert orchestral.load_model(str(config), object(), None) == (
        fake_model,
        "FakeAlgorithm",
        True,
    )
    assert fake_model.load_calls == ["checkpoint"]
    assert (
        inspect.signature(OrchestralController.__init__)
        .parameters["sub_controllers"]
        .default
        is None
    )
