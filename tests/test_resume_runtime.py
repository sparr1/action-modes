import copy
import random

import gymnasium as gym
import numpy as np
import pytest
import torch

from utils.resume_runtime import (
    ResumeRuntimeMismatch,
    UnsupportedResumeEnvironment,
    capture_environment_state,
    capture_global_rng_state,
    environment_contract,
    register_test_resume_environment,
    restore_environment_state,
    restore_global_rng_state,
    validate_environment_capability,
)


class ExplicitBoundaryEnv:
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(3,), dtype=np.float32
        )
        self.action_space.seed(11)
        self.observation_space.seed(12)
        self.reset_counter = 7
        self.load_calls = 0

    def training_resume_state(self):
        return {"schema_version": 1, "reset_counter": self.reset_counter}

    def load_training_resume_state(self, state):
        self.load_calls += 1
        self.validate_training_resume_state(state)
        self.reset_counter = int(state["reset_counter"])

    def validate_training_resume_state(self, state):
        if state.get("schema_version") != 1:
            raise ValueError("bad state")
        int(state["reset_counter"])


register_test_resume_environment(ExplicitBoundaryEnv, episode_steps=5)


class EarlyTerminatingBoundaryEnv(ExplicitBoundaryEnv):
    pass


register_test_resume_environment(
    EarlyTerminatingBoundaryEnv,
    episode_steps=5,
    early_termination=True,
)


def test_environment_protocol_restores_spaces_and_explicit_state():
    env = ExplicitBoundaryEnv()
    state = capture_environment_state(env)

    expected_action = env.action_space.sample()
    expected_observation = env.observation_space.sample()
    env.reset_counter = 99
    env.action_space.seed(100)
    env.observation_space.seed(101)

    restore_environment_state(env, state)

    assert env.reset_counter == 7
    np.testing.assert_array_equal(env.action_space.sample(), expected_action)
    np.testing.assert_array_equal(env.observation_space.sample(), expected_observation)


def test_environment_type_mismatch_fails_before_restore():
    env = ExplicitBoundaryEnv()
    state = capture_environment_state(env)
    state["base_type"] = "different.Environment"

    with pytest.raises(ResumeRuntimeMismatch, match="type changed"):
        restore_environment_state(env, state)


def test_explicit_protocol_requires_a_nonmutating_validator():
    class MissingValidatorEnv(ExplicitBoundaryEnv):
        validate_training_resume_state = None

    register_test_resume_environment(MissingValidatorEnv, episode_steps=5)

    with pytest.raises(UnsupportedResumeEnvironment, match="validate_training_resume_state"):
        capture_environment_state(MissingValidatorEnv())


def test_corrupt_explicit_base_state_fails_before_adapter_loader():
    env = ExplicitBoundaryEnv()
    state = capture_environment_state(env)
    state["base_state"].pop("reset_counter")

    with pytest.raises(ResumeRuntimeMismatch, match="Base environment resume state"):
        restore_environment_state(env, state)

    assert env.load_calls == 0
    assert env.reset_counter == 7


@pytest.mark.parametrize("space_field", ["action_space", "observation_space"])
def test_corrupt_space_rng_fails_before_any_restore_mutation(space_field):
    env = ExplicitBoundaryEnv()
    state = capture_environment_state(env)
    state[space_field]["rng"] = {
        "bit_generator": state[space_field]["rng"]["bit_generator"]
    }
    expected_action_rng = copy.deepcopy(env.action_space._np_random.bit_generator.state)
    expected_observation_rng = copy.deepcopy(
        env.observation_space._np_random.bit_generator.state
    )

    with pytest.raises(ResumeRuntimeMismatch, match=f"{space_field} RNG state"):
        restore_environment_state(env, state)

    assert env.load_calls == 0
    assert env.action_space._np_random.bit_generator.state == expected_action_rng
    assert (
        env.observation_space._np_random.bit_generator.state
        == expected_observation_rng
    )


def test_unknown_environment_is_rejected():
    class Unknown:
        def __init__(self):
            self.action_space = gym.spaces.Discrete(2)
            self.observation_space = gym.spaces.Discrete(2)
            self.action_space.seed(1)
            self.observation_space.seed(2)

    with pytest.raises(UnsupportedResumeEnvironment, match="reviewed"):
        capture_environment_state(Unknown())


def test_unknown_duck_typed_environment_is_rejected_despite_protocol_methods():
    class UnknownExplicit(ExplicitBoundaryEnv):
        pass

    with pytest.raises(UnsupportedResumeEnvironment, match="reviewed"):
        capture_environment_state(UnknownExplicit())


def test_fixed_horizon_contract_matches_trainer_episode_length():
    env = ExplicitBoundaryEnv()
    contract = environment_contract(env, expected_episode_steps=5)

    assert contract["schema_version"] == 2
    assert contract["episode_steps"] == 5
    assert contract["early_termination"] is False
    with pytest.raises(ResumeRuntimeMismatch, match="episode_length"):
        validate_environment_capability(env, expected_episode_steps=4)


def test_environment_declaring_early_termination_is_rejected():
    with pytest.raises(UnsupportedResumeEnvironment, match="early termination"):
        validate_environment_capability(EarlyTerminatingBoundaryEnv())


def test_reviewed_ant_requires_early_termination_disabled():
    fake_ant_type = type(
        "AntEnv",
        (),
        {"__module__": "gymnasium.envs.mujoco.ant_v4"},
    )
    ant = fake_ant_type()
    ant.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    ant.observation_space = gym.spaces.Box(
        -np.inf, np.inf, shape=(3,), dtype=np.float32
    )
    ant.spec = type("Spec", (), {"max_episode_steps": 1000})()
    ant._terminate_when_unhealthy = True

    with pytest.raises(UnsupportedResumeEnvironment, match="early termination"):
        validate_environment_capability(ant, expected_episode_steps=1000)

    ant._terminate_when_unhealthy = False
    assert validate_environment_capability(
        ant, expected_episode_steps=1000
    ) == 1000


def test_global_rng_round_trip_restores_next_draws():
    random.seed(3)
    np.random.seed(4)
    torch.manual_seed(5)
    state = capture_global_rng_state()

    expected = (random.random(), np.random.random(), torch.rand(3))
    for _ in range(10):
        random.random()
        np.random.random()
        torch.rand(3)

    restore_global_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
