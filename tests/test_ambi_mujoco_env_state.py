import copy
import importlib
import sys
import types

import gymnasium as gym
import numpy as np
import pytest

import domains  # noqa: F401  # Register the custom Ant environments.


@pytest.fixture(autouse=True)
def _supply_optional_runtime_modules(monkeypatch):
    try:
        __import__("imageio")
    except ModuleNotFoundError:
        monkeypatch.setitem(sys.modules, "imageio", types.ModuleType("imageio"))

    try:
        __import__("stable_baselines3")
    except ModuleNotFoundError:
        stable_baselines3 = types.ModuleType("stable_baselines3")
        common = types.ModuleType("stable_baselines3.common")
        logger = types.ModuleType("stable_baselines3.common.logger")
        noise = types.ModuleType("stable_baselines3.common.noise")

        logger.configure = lambda *args, **kwargs: None

        class _UnusedActionNoise:
            pass

        noise.NormalActionNoise = _UnusedActionNoise
        noise.OrnsteinUhlenbeckActionNoise = _UnusedActionNoise
        stable_baselines3.common = common
        common.logger = logger
        common.noise = noise
        monkeypatch.setitem(sys.modules, "stable_baselines3", stable_baselines3)
        monkeypatch.setitem(sys.modules, "stable_baselines3.common", common)
        monkeypatch.setitem(sys.modules, "stable_baselines3.common.logger", logger)
        monkeypatch.setitem(sys.modules, "stable_baselines3.common.noise", noise)


@pytest.fixture
def state_helpers():
    module = importlib.import_module("RL.AMBIMujoco")
    return module._snapshot_env_state, module._set_env_state


def _adaptation_env():
    return gym.make(
        "LegAdaptAnt-v0",
        total_timesteps=1,
        switch_fraction=1.0,
        start_legs=4,
        end_legs=3,
        exclude_current_positions_from_observation=False,
        terminate_when_unhealthy=False,
        render_mode=None,
    )


def test_snapshot_restore_synchronizes_leg_adaptation_before_physics(
    state_helpers,
):
    snapshot_env_state, set_env_state = state_helpers
    source = _adaptation_env()
    target = _adaptation_env()
    incompatible = gym.make(
        "Ant3LegDeadStump-v0",
        exclude_current_positions_from_observation=False,
        render_mode=None,
    )
    try:
        source.reset(seed=31)
        target.reset(seed=37)
        incompatible.reset(seed=41)

        source.step(np.zeros(source.action_space.shape, dtype=np.float64))
        assert source.unwrapped._pending_switch
        source.reset()
        assert source.unwrapped._switched
        assert source.unwrapped.model.nu == 6
        assert target.unwrapped.model.nu == 8

        snapshot = snapshot_env_state(source)

        corrupted = copy.deepcopy(snapshot)
        corrupted["mujoco"]["state"] = corrupted["mujoco"]["state"][:-1]
        with pytest.raises(ValueError, match="invalid physics vector"):
            set_env_state(target, corrupted)
        assert not target.unwrapped._switched
        assert target.unwrapped.model.nu == 8

        with pytest.raises(ValueError, match="different base environment"):
            set_env_state(incompatible, snapshot)

        set_env_state(target, snapshot)
        assert target.unwrapped._switched
        assert target.unwrapped.model.nu == source.unwrapped.model.nu == 6
        assert target.unwrapped._steps_taken == source.unwrapped._steps_taken
        assert target.unwrapped._pending_switch == source.unwrapped._pending_switch
        assert target._elapsed_steps == source._elapsed_steps
        np.testing.assert_array_equal(
            target.unwrapped.data.qpos, source.unwrapped.data.qpos
        )
        np.testing.assert_array_equal(
            target.unwrapped.data.qvel, source.unwrapped.data.qvel
        )

        # The explicit state also synchronizes the reset RNG stream.
        source_observation, _ = source.reset()
        target_observation, _ = target.reset()
        np.testing.assert_array_equal(target_observation, source_observation)
    finally:
        source.close()
        target.close()
        incompatible.close()


def test_snapshot_restore_remains_compatible_with_standard_ant(state_helpers):
    snapshot_env_state, set_env_state = state_helpers
    source = gym.make(
        "Ant-v4",
        exclude_current_positions_from_observation=False,
        render_mode=None,
    )
    target = gym.make(
        "Ant-v4",
        exclude_current_positions_from_observation=False,
        render_mode=None,
    )
    try:
        source.reset(seed=11)
        target.reset(seed=17)
        source.step(np.full(source.action_space.shape, 0.1, dtype=np.float32))
        snapshot = snapshot_env_state(source)
        set_env_state(target, snapshot)

        np.testing.assert_array_equal(
            target.unwrapped.data.qpos, source.unwrapped.data.qpos
        )
        np.testing.assert_array_equal(
            target.unwrapped.data.qvel, source.unwrapped.data.qvel
        )
        assert target._elapsed_steps == source._elapsed_steps
    finally:
        source.close()
        target.close()


def test_snapshot_restore_remains_compatible_with_dead_stump_ant(state_helpers):
    snapshot_env_state, set_env_state = state_helpers
    source = gym.make(
        "Ant3LegDeadStump-v0",
        exclude_current_positions_from_observation=False,
        render_mode=None,
    )
    target = gym.make(
        "Ant3LegDeadStump-v0",
        exclude_current_positions_from_observation=False,
        render_mode=None,
    )
    try:
        source.reset(seed=19)
        target.reset(seed=23)
        source.step(np.full(source.action_space.shape, 0.1, dtype=np.float32))
        snapshot = snapshot_env_state(source)
        set_env_state(target, snapshot)

        assert target.unwrapped.model.nu == source.unwrapped.model.nu == 6
        np.testing.assert_array_equal(
            target.unwrapped.data.qpos, source.unwrapped.data.qpos
        )
        np.testing.assert_array_equal(
            target.unwrapped.data.qvel, source.unwrapped.data.qvel
        )

        source_observation, _ = source.reset()
        target_observation, _ = target.reset()
        np.testing.assert_array_equal(target_observation, source_observation)
    finally:
        source.close()
        target.close()


def test_ambi_save_returns_the_delegated_checkpoint_path(state_helpers):
    del state_helpers
    module = importlib.import_module("RL.AMBIMujoco")
    algorithm = object.__new__(module.AMBI)

    class _OuterAgent:
        def save(self, path, name):
            return f"{path}/{name}.zip"

    algorithm.outer_agent = _OuterAgent()
    assert algorithm.save("models", "best") == "models/best.zip"


def test_wandb_setup_finishes_run_when_metric_definition_fails(
    state_helpers, monkeypatch
):
    del state_helpers
    module = importlib.import_module("RL.AMBIMujoco")

    class _Run:
        def __init__(self):
            self.finish_calls = 0

        def finish(self):
            self.finish_calls += 1

    run = _Run()
    monkeypatch.setattr(module.wandb, "init", lambda **kwargs: run)
    monkeypatch.setattr(
        module.wandb,
        "define_metric",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("metric setup failed")
        ),
    )

    with pytest.raises(RuntimeError, match="metric setup failed"):
        module.wandb_setup({}, run_name="test")
    assert run.finish_calls == 1


def test_ambi_constructor_closes_inner_env_when_inner_reset_fails(
    state_helpers, monkeypatch
):
    del state_helpers
    module = importlib.import_module("RL.AMBIMujoco")

    class _InnerEnv:
        def __init__(self):
            self.close_calls = 0

        @staticmethod
        def reset():
            raise RuntimeError("inner reset failed")

        def close(self):
            self.close_calls += 1

    class _OuterEnv:
        def __init__(self):
            self.reset_calls = 0
            self.close_calls = 0

        def reset(self):
            self.reset_calls += 1

        def close(self):
            self.close_calls += 1

    inner_env = _InnerEnv()
    outer_env = _OuterEnv()
    outer_agent = types.SimpleNamespace(model=types.SimpleNamespace(_logger=object()))
    monkeypatch.setattr(
        module.utils_core,
        "initialize_alg",
        lambda *args, **kwargs: (outer_agent, False, "fake"),
    )
    monkeypatch.setattr(module.gym, "make", lambda *args, **kwargs: inner_env)
    params = {
        "outer_alg": "fake/Agent",
        "outer_alg_params": {
            "train_freq": 1,
            "learning_starts": {"steps": 0},
        },
        "inner_alg_params": {
            "train_freq": 1,
            "learning_starts": {"steps": 0},
        },
        "max_episode_steps": 10,
    }

    with pytest.raises(RuntimeError, match="inner reset failed"):
        module.AMBI(
            "AMBI",
            outer_env,
            params,
            run_params={"name": "fake", "seed": 3, "env": "Fake-v0"},
            experiment_params={},
        )

    assert inner_env.close_calls == 1
    assert outer_env.reset_calls == 0
    assert outer_env.close_calls == 0


def test_ambi_close_only_closes_owned_resources_and_is_idempotent(state_helpers):
    del state_helpers
    module = importlib.import_module("RL.AMBIMujoco")
    algorithm = object.__new__(module.AMBI)

    class _Env:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    class _Run:
        def __init__(self):
            self.finish_calls = 0

        def finish(self):
            self.finish_calls += 1

    inner_env = _Env()
    outer_env = _Env()
    run = _Run()
    algorithm.inner_env = inner_env
    algorithm.env = outer_env
    algorithm.run = run

    algorithm.close()
    algorithm.close()

    assert inner_env.close_calls == 1
    assert outer_env.close_calls == 0
    assert run.finish_calls == 1
    assert algorithm.inner_env is None
    assert algorithm.run is None


def test_ambi_close_aggregates_failures_without_masking_run_error(state_helpers):
    del state_helpers
    module = importlib.import_module("RL.AMBIMujoco")
    algorithm = object.__new__(module.AMBI)
    inner_env = types.SimpleNamespace(close_calls=0)

    def close_inner():
        inner_env.close_calls += 1
        raise ValueError("inner close failed")

    inner_env.close = close_inner
    algorithm.inner_env = inner_env
    algorithm.run = types.SimpleNamespace(
        finish=lambda: (_ for _ in ()).throw(RuntimeError("finish failed"))
    )

    with pytest.raises(RuntimeError, match="finish failed") as exc_info:
        algorithm.close()
    assert inner_env.close_calls == 1
    assert algorithm.inner_env is None
    assert algorithm.run is None
    assert any(
        "Additional cleanup failure: inner close failed" in note
        for note in getattr(exc_info.value, "__notes__", ())
    )


@pytest.mark.parametrize(
    ("reinitialize_each_step", "expected_initializations"),
    [(True, 2), (False, 1)],
)
def test_prepare_inner_agent_honors_reinitialization_option(
    state_helpers,
    monkeypatch,
    reinitialize_each_step,
    expected_initializations,
):
    del state_helpers
    module = importlib.import_module("RL.AMBIMujoco")
    algorithm = object.__new__(module.AMBI)
    algorithm.inner_env = object()
    algorithm.inner_agent = None
    algorithm.inner_reinit_every_step = reinitialize_each_step

    agents = []
    learning_start_calls = []
    restored_states = []

    def initialize_inner_agent():
        agent = object()
        agents.append(agent)
        return agent

    algorithm._initialize_inner_agent = initialize_inner_agent
    algorithm._initialize_learning_starts = learning_start_calls.append
    monkeypatch.setattr(
        module,
        "_set_env_state",
        lambda env, state: restored_states.append((env, state)),
    )

    first = algorithm._prepare_inner_agent("snapshot-1")
    second = algorithm._prepare_inner_agent("snapshot-2")

    assert len(agents) == expected_initializations
    assert len(learning_start_calls) == expected_initializations
    assert learning_start_calls == ["inner"] * expected_initializations
    assert restored_states == [
        (algorithm.inner_env, "snapshot-1"),
        (algorithm.inner_env, "snapshot-2"),
    ]
    if reinitialize_each_step:
        assert first is not second
    else:
        assert first is second


def test_collect_inner_rollout_marks_configured_horizon_as_timeout(state_helpers):
    del state_helpers
    module = importlib.import_module("RL.AMBIMujoco")
    algorithm = object.__new__(module.AMBI)

    class _ReplayBuffer:
        def __init__(self):
            self.transitions = []

        def add(self, *transition):
            self.transitions.append(transition)

    class _Policy:
        @staticmethod
        def scale_action(action):
            return action

    class _InnerAgent:
        def __init__(self):
            self.model = types.SimpleNamespace(
                policy=_Policy(), replay_buffer=_ReplayBuffer()
            )

        @staticmethod
        def predict(obs):
            return np.array([0.0], dtype=np.float32), None

    class _InnerEnv:
        def __init__(self):
            self.steps = 0

        def step(self, action):
            del action
            self.steps += 1
            return np.array([self.steps]), 1.0, False, False, {"source": "env"}

    algorithm.inner_agent = _InnerAgent()
    algorithm.inner_env = _InnerEnv()

    _, rollout_return, steps, done = algorithm._collect_inner_rollout(
        np.array([0]), max_steps=2, truncate_on_limit=True
    )

    assert (rollout_return, steps, done) == (2.0, 2, True)
    transitions = algorithm.inner_agent.model.replay_buffer.transitions
    assert not bool(transitions[0][4][0])
    assert bool(transitions[1][4][0])
    assert transitions[1][5][0] == {
        "source": "env",
        "terminated": False,
        "truncated": True,
        "TimeLimit.truncated": True,
    }


def test_timeout_normalization_handles_native_and_simultaneous_endings(state_helpers):
    del state_helpers
    module = importlib.import_module("RL.AMBIMujoco")

    truncated, info = module._apply_configured_time_limit(
        False,
        True,
        {"source": "native-time-limit"},
        next_episode_step=2,
        max_episode_steps=5,
    )
    assert truncated
    assert info["terminated"] is False
    assert info["truncated"] is True
    assert info["TimeLimit.truncated"] is True

    truncated, info = module._apply_configured_time_limit(
        True,
        True,
        {"TimeLimit.truncated": True},
        next_episode_step=5,
        max_episode_steps=5,
    )
    assert truncated
    assert info["terminated"] is True
    assert info["truncated"] is True
    assert info["TimeLimit.truncated"] is False


def test_learning_starts_noise_params_are_reusable_across_action_dims(
    state_helpers, monkeypatch
):
    del state_helpers
    module = importlib.import_module("RL.AMBIMujoco")
    algorithm = object.__new__(module.AMBI)
    configured_noise = {"mean": 0.0, "sigma": 0.2}
    algorithm.max_episode_steps = 1000
    algorithm.inner_agent = object()
    algorithm.inner_env = types.SimpleNamespace(
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
    )
    algorithm.inner_learning_starts_steps = 0
    algorithm.inner_random_actions = False
    algorithm.inner_use_action_noise = True
    algorithm.inner_action_noise_type = "normal"
    algorithm.inner_action_noise_params = configured_noise

    noise_calls = []

    class _Noise:
        def __init__(self, **kwargs):
            noise_calls.append(kwargs)

    monkeypatch.setattr(module, "NormalActionNoise", _Noise)
    monkeypatch.setattr(
        module,
        "_snapshot_env_state",
        lambda env: {"wrappers": {"_elapsed_steps": 0}},
    )
    monkeypatch.setattr(module, "_set_env_state", lambda env, state: None)

    algorithm._initialize_learning_starts("inner")
    algorithm.inner_env.action_space = gym.spaces.Box(
        -1.0, 1.0, shape=(6,), dtype=np.float32
    )
    algorithm._initialize_learning_starts("inner")

    assert configured_noise == {"mean": 0.0, "sigma": 0.2}
    assert noise_calls[0]["mean"].shape == (8,)
    assert noise_calls[0]["sigma"].shape == (8,)
    assert noise_calls[1]["mean"].shape == (6,)
    assert noise_calls[1]["sigma"].shape == (6,)
    np.testing.assert_array_equal(noise_calls[1]["sigma"], np.full(6, 0.2))


@pytest.mark.parametrize("value", (True, 0, -1, 1.5, None))
def test_ambi_positive_frequency_and_horizon_settings_fail_fast(
    state_helpers, value
):
    del state_helpers
    module = importlib.import_module("RL.AMBIMujoco")
    with pytest.raises(ValueError, match="positive|non-negative integer"):
        module._positive_integer_setting(value, "test_setting")


@pytest.mark.parametrize(
    "learning_starts",
    (None, 3, {}, {"steps": True}, {"steps": -1}, {"steps": 1.5}),
)
def test_ambi_learning_starts_settings_require_exact_nonnegative_steps(
    state_helpers, learning_starts
):
    del state_helpers
    module = importlib.import_module("RL.AMBIMujoco")
    with pytest.raises(ValueError, match="learning_starts"):
        module._learning_starts_settings(
            {"learning_starts": learning_starts}, "inner"
        )


@pytest.mark.parametrize("total_timesteps", (True, -1, 1.5, float("inf")))
def test_ambi_learn_rejects_inexact_timestep_budgets_before_work(
    state_helpers, total_timesteps
):
    del state_helpers
    module = importlib.import_module("RL.AMBIMujoco")
    algorithm = object.__new__(module.AMBI)
    with pytest.raises(ValueError, match="non-negative integer"):
        algorithm.learn(total_timesteps=total_timesteps)


def test_learn_enforces_configured_outer_episode_horizon(state_helpers, monkeypatch):
    del state_helpers
    module = importlib.import_module("RL.AMBIMujoco")
    algorithm = object.__new__(module.AMBI)

    class _ReplayBuffer:
        def __init__(self):
            self.transitions = []

        def add(self, *transition):
            self.transitions.append(transition)

    class _Policy:
        @staticmethod
        def scale_action(action):
            return action

    class _OuterAgent:
        def __init__(self):
            self.model = types.SimpleNamespace(
                device="cpu",
                policy=_Policy(),
                replay_buffer=_ReplayBuffer(),
                train=lambda **kwargs: None,
            )

        @staticmethod
        def predict(obs):
            return np.array([0.0], dtype=np.float32), None

    class _OuterEnv:
        def __init__(self):
            self.reset_calls = 0
            self.steps_in_episode = 0

        def reset(self):
            self.reset_calls += 1
            self.steps_in_episode = 0
            return np.array([0]), {}

        def step(self, action):
            del action
            self.steps_in_episode += 1
            return (
                np.array([self.steps_in_episode]),
                1.0,
                False,
                False,
                {"source": "env"},
            )

    algorithm.env = _OuterEnv()
    algorithm.outer_agent = _OuterAgent()
    algorithm.inner_agent = None
    algorithm.inner_rollouts = 0
    algorithm.inner_train_freq = 1
    algorithm.max_episode_steps = 2
    algorithm.outer_train_freq = 100
    algorithm.outer_alg_params = {}
    algorithm.inner_alg_params = {}
    logger_records = []
    algorithm.alg_logger = types.SimpleNamespace(on_step=logger_records.append)
    algorithm.render = False
    algorithm._initialize_learning_starts = lambda layer: None

    class _Run:
        def __init__(self):
            self.finish_calls = 0

        def finish(self):
            self.finish_calls += 1

    run = _Run()
    algorithm.run = run

    monkeypatch.setattr(module, "_snapshot_env_state", lambda env: None)
    monkeypatch.setattr(module, "log_inner_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "log_outer_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "log_outer_episode", lambda *args, **kwargs: None)

    algorithm.learn(total_timesteps=3)
    algorithm.close()

    transitions = algorithm.outer_agent.model.replay_buffer.transitions
    assert algorithm.env.reset_calls == 2
    assert [bool(transition[4][0]) for transition in transitions] == [
        False,
        True,
        False,
    ]
    assert transitions[1][5][0] == {
        "source": "env",
        "terminated": False,
        "truncated": True,
        "TimeLimit.truncated": True,
    }
    assert [record["obs"] for record in logger_records] == [[1], [2], [1]]
    assert run.finish_calls == 1
    assert algorithm.run is None
