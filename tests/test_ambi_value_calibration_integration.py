import random
from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.TDMPC2 import TDMPC2Baseline
from RL.tdmpc2_core.value_calibration import ValueCalibrationEvaluator


class _OneStepEnv(gym.Env):
    metadata = {}

    def __init__(self):
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, shape=(3,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(1,), dtype=np.float32
        )
        self.spec = None
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_calls += 1
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.step_calls += 1
        return np.zeros(3, dtype=np.float32), 2.0, False, True, {}


class _FakeRun:
    def __init__(self):
        self.calls = []

    def log(self, payload, step):
        self.calls.append((int(step), dict(payload)))


def _resolved_ambi_cfg(*, config_env=None, **overrides):
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = config_env or gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {
        "seed": 3,
        "device": "cpu",
        "env": "test-env",
        "total_steps": 10,
    }
    algorithm.custom_params = dict(overrides)
    try:
        return algorithm._build_cfg({"device": "cpu", **overrides})
    finally:
        algorithm.env.close()


def _tiny_tdmpc2_params():
    return {
        "device": "cpu",
        "model_size": None,
        "enc_dim": 32,
        "mlp_dim": 32,
        "latent_dim": 16,
        "num_enc_layers": 2,
        "num_q": 2,
        "simnorm_dim": 8,
        "num_bins": 11,
        "vmin": -5,
        "vmax": 5,
        "batch_size": 2,
        "train_unroll_horizon": 1,
        "outer_planning_horizon": 1,
        "inner_rollout_horizon": 1,
        "buffer_size": 10,
        "seed_steps": 1,
        "pretrain_steps": 1,
        "compile": False,
        "episodic": False,
        "discount": 0.99,
        "iterations": 1,
        "num_samples": 2,
        "num_elites": 1,
        "num_pi_trajs": 1,
        "dropout": 0.0,
        "episode_length": 1,
        "eval_freq": 1,
        "eval_episodes": 1,
    }


def _tiny_ambi_params():
    return {
        **_tiny_tdmpc2_params(),
        "q_representation": "scalar",
        "q_num_bins": 11,
        "q_vmin": -5,
        "q_vmax": 5,
        "q_pair_size": 2,
        "outer_critic_target": "reward_only",
        "inner_sac_critic_target": "reward_only",
        "inner_operator": "sac",
        "inner_model_step_budget": 1,
        "inner_rounds": 1,
        "inner_rollout_horizon": 1,
        "inner_critic_updates_per_action": 1,
        "inner_actor_updates_per_action": 1,
        "inner_temperature_updates_per_action": 0,
        "inner_batch_size": 1,
        "inner_replay_capacity": 2,
        "inner_temperature_mode": "inherit_outer",
        "inner_mppi_num_elites": 1,
        "inner_mppi_num_pi_trajs": 0,
    }


def _clone_tree(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return deepcopy(value)


def _assert_tree_equal(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_tree_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_value_calibration_defaults_are_disabled_and_explicit():
    cfg = _resolved_ambi_cfg()

    assert cfg.eval_value is False
    assert cfg.eval_value_samples == 100
    assert cfg.eval_value_seed == 12345
    assert cfg.eval_value_protocols == [
        "paper_deterministic",
        "stochastic_bellman",
    ]
    assert cfg.eval_inner_comparison is False
    assert cfg.eval_inner_comparison_episodes == 5
    assert cfg.eval_inner_comparison_seed == 12345


def test_value_calibration_valid_reward_only_configuration_resolves():
    cfg = _resolved_ambi_cfg(
        eval_value=True,
        eval_freq=50_000,
        outer_critic_target="reward_only",
        q_pair_size=2,
    )

    assert cfg.eval_value is True
    assert cfg.eval_freq == 50_000


def test_inner_comparison_valid_configuration_resolves():
    cfg = _resolved_ambi_cfg(
        eval_inner_comparison=True,
        eval_inner_comparison_episodes=7,
        eval_inner_comparison_seed=24680,
        eval_freq=50_000,
    )

    assert cfg.eval_inner_comparison is True
    assert cfg.eval_inner_comparison_episodes == 7
    assert cfg.eval_inner_comparison_seed == 24680
    assert cfg.eval_freq == 50_000


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"eval_inner_comparison": 1}, "must be a boolean"),
        ({"eval_inner_comparison_episodes": 0}, "positive integer"),
        ({"eval_inner_comparison_episodes": True}, "positive integer"),
        ({"eval_inner_comparison_seed": -1}, "non-negative integer"),
        ({"eval_inner_comparison_seed": True}, "non-negative integer"),
        ({"eval_inner_comparison": True}, "configured eval_freq"),
        (
            {"eval_inner_comparison": True, "eval_freq": 0},
            "eval_freq",
        ),
        (
            {
                "eval_inner_comparison": True,
                "eval_freq": 1,
                "inner_operator": "td3",
            },
            "inner_operator='sac'",
        ),
        (
            {
                "eval_inner_comparison": True,
                "eval_freq": 1,
                "inner_rounds": 0,
            },
            "inner_rounds>0",
        ),
        (
            {
                "eval_inner_comparison": True,
                "eval_freq": 1,
                "inner_diagnostic_rollouts": 1,
            },
            "inner_diagnostic_rollouts=0",
        ),
    ],
)
def test_inner_comparison_configuration_fails_closed(overrides, message):
    with pytest.raises(ValueError, match=message):
        _resolved_ambi_cfg(**overrides)


@pytest.mark.parametrize(
    "scope",
    [
        "inner_actor_scope",
        "inner_critic_scope",
        "inner_temperature_scope",
        "inner_replay_scope",
        "inner_actor_optimizer_scope",
        "inner_critic_optimizer_scope",
        "inner_temperature_optimizer_scope",
    ],
)
def test_inner_comparison_requires_every_scope_to_be_action_local(scope):
    with pytest.raises(ValueError, match=scope):
        _resolved_ambi_cfg(
            eval_inner_comparison=True,
            eval_freq=1,
            **{scope: "episode"},
        )


def test_inner_comparison_rejects_rgb_observations():
    env = _OneStepEnv()
    env.observation_space = gym.spaces.Box(
        0,
        255,
        shape=(9, 64, 64),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError, match="supports state observations only"):
        _resolved_ambi_cfg(
            config_env=env,
            eval_inner_comparison=True,
            eval_freq=1,
            obs="rgb",
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"eval_value": True}, "eval_freq"),
        (
            {"eval_value": True, "eval_freq": 1},
            "outer_critic_target='reward_only'",
        ),
        (
            {
                "eval_value": True,
                "eval_freq": 1,
                "outer_critic_target": "reward_only",
                "q_pair_size": 3,
            },
            "q_pair_size=2",
        ),
        ({"eval_value_samples": 0}, "eval_value_samples"),
        ({"eval_value_seed": -1}, "eval_value_seed"),
        ({"eval_value_protocols": "paper_deterministic"}, "non-empty list"),
        ({"eval_value_protocols": []}, "must not be empty"),
        (
            {"eval_value_protocols": ["paper_deterministic", "unknown"]},
            "chosen from",
        ),
        (
            {
                "eval_value_protocols": [
                    "paper_deterministic",
                    "paper_deterministic",
                ]
            },
            "duplicate",
        ),
    ],
)
def test_value_calibration_configuration_fails_closed(overrides, message):
    with pytest.raises(ValueError, match=message):
        _resolved_ambi_cfg(**overrides)


def test_base_evaluation_merges_extras_into_one_wandb_event():
    env = _OneStepEnv()
    model = TDMPC2Baseline(
        "TDMPC2Baseline",
        env,
        _tiny_tdmpc2_params(),
        {"seed": 7, "device": "cpu", "env": "test", "total_steps": 1},
        {},
    )
    model._act_agent = lambda *_args, **_kwargs: torch.zeros(1)
    model._evaluation_payload_extras = lambda step: {
        "eval/mc_value": 1.25,
        "eval/q_value": 2.5,
    }
    run = _FakeRun()
    model._wandb_run = run
    observation, _ = env.reset(seed=7)

    result = model._evaluate_policy(12, initial_obs=observation)

    assert result == 2.0
    assert len(run.calls) == 1
    step, payload = run.calls[0]
    assert step == 12
    assert payload == {
        "eval/episode_reward": 2.0,
        "eval/episodes": 1,
        "eval/mc_value": 1.25,
        "eval/q_value": 2.5,
        "env_step": 12,
    }


def test_evaluation_payload_rejects_reserved_metric_collisions():
    model = object.__new__(TDMPC2Baseline)
    model._eval_episodes = 1
    model._episode_idx = 0
    model._eval_csv_path = None
    model._wandb_run = None
    model.cfg = SimpleNamespace(seed=1)

    with pytest.raises(ValueError, match="reserved metrics"):
        model._record_evaluation(0, 1.0, extras={"env_step": 4.0})


def test_ambi_evaluators_are_lazy_merged_reused_and_closed_idempotently():
    class Evaluator:
        def __init__(self, metrics):
            self.metrics = metrics
            self.evaluate_calls = 0
            self.close_calls = 0

        def evaluate(self):
            self.evaluate_calls += 1
            return dict(self.metrics)

        def close(self):
            self.close_calls += 1

    algorithm = object.__new__(AMBITDMPC2)
    algorithm.cfg = SimpleNamespace(
        eval_value=False,
        eval_inner_comparison=False,
    )
    algorithm._value_calibration_evaluator = None
    algorithm._paired_controller_evaluator = None
    algorithm._make_value_calibration_evaluator = lambda: pytest.fail(
        "disabled diagnostics must not construct an evaluator"
    )
    algorithm._make_paired_controller_evaluator = lambda: pytest.fail(
        "disabled diagnostics must not construct an evaluator"
    )
    assert algorithm._evaluation_payload_extras(0) == {}

    value_evaluator = Evaluator({"eval/mc_value": 3.0})
    paired_evaluator = Evaluator({"eval/inner_return_delta": 1.5})
    constructions = []

    def make_value_evaluator():
        constructions.append("value")
        return value_evaluator

    def make_paired_evaluator():
        constructions.append("paired")
        return paired_evaluator

    algorithm.cfg.eval_value = True
    algorithm.cfg.eval_inner_comparison = True
    algorithm._make_value_calibration_evaluator = make_value_evaluator
    algorithm._make_paired_controller_evaluator = make_paired_evaluator
    expected = {
        "eval/mc_value": 3.0,
        "eval/inner_return_delta": 1.5,
    }
    assert algorithm._evaluation_payload_extras(0) == expected
    assert algorithm._evaluation_payload_extras(50_000) == expected
    assert constructions == ["value", "paired"]
    assert value_evaluator.evaluate_calls == 2
    assert paired_evaluator.evaluate_calls == 2

    algorithm.close()
    algorithm.close()
    assert value_evaluator.close_calls == 1
    assert paired_evaluator.close_calls == 1
    assert algorithm._value_calibration_evaluator is None
    assert algorithm._paired_controller_evaluator is None


def test_ambi_evaluation_payload_rejects_duplicate_probe_metrics():
    class Evaluator:
        def evaluate(self):
            return {"eval/shared_metric": 1.0}

        def close(self):
            pass

    algorithm = object.__new__(AMBITDMPC2)
    algorithm.cfg = SimpleNamespace(
        eval_value=True,
        eval_inner_comparison=True,
    )
    algorithm._value_calibration_evaluator = None
    algorithm._paired_controller_evaluator = None
    algorithm._make_value_calibration_evaluator = Evaluator
    algorithm._make_paired_controller_evaluator = Evaluator

    with pytest.raises(RuntimeError, match="duplicate metrics") as error:
        algorithm._evaluation_payload_extras(0)

    assert "eval/shared_metric" in str(error.value)
    algorithm.close()


def test_ambi_close_attempts_both_evaluators_when_the_first_raises():
    class CloseSignal(BaseException):
        pass

    class Evaluator:
        def __init__(self, error=None):
            self.error = error
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            if self.error is not None:
                raise self.error

    value_evaluator = Evaluator(CloseSignal("value close interrupted"))
    paired_evaluator = Evaluator()
    algorithm = object.__new__(AMBITDMPC2)
    algorithm._value_calibration_evaluator = value_evaluator
    algorithm._paired_controller_evaluator = paired_evaluator

    with pytest.raises(CloseSignal, match="value close interrupted"):
        algorithm.close()

    assert value_evaluator.close_calls == 1
    assert paired_evaluator.close_calls == 1
    assert algorithm._value_calibration_evaluator is None
    assert algorithm._paired_controller_evaluator is None
    algorithm.close()


def test_real_ambi_outer_probe_preserves_complete_agent_and_training_state(
    monkeypatch,
):
    training_env = _OneStepEnv()
    algorithm = AMBITDMPC2(
        "AMBITDMPC2",
        training_env,
        _tiny_ambi_params(),
        {"seed": 11, "device": "cpu", "env": "test", "total_steps": 1},
        {},
    )
    auxiliary_env = _OneStepEnv()
    evaluator = ValueCalibrationEvaluator(
        model=algorithm.agent.model,
        env_factory=lambda: auxiliary_env,
        observation_to_tensor=algorithm._obs_to_tensor,
        unscale_action=algorithm._unscale_action,
        discount=0.5,
        samples=2,
        seed=12345,
        protocols=("paper_deterministic", "stochastic_bellman"),
        device="cpu",
    )
    monkeypatch.setattr(
        algorithm.agent,
        "act",
        lambda *_args, **_kwargs: pytest.fail("outer calibration called agent.act"),
    )
    monkeypatch.setattr(
        algorithm.agent.inner_engine,
        "act",
        lambda *_args, **_kwargs: pytest.fail("outer calibration called inner act"),
    )

    agent_before = _clone_tree(algorithm.agent.training_state_dict())
    replay_episodes_before = algorithm.buffer.num_eps
    global_step_before = algorithm._global_step
    training_reset_calls_before = training_env.reset_calls
    training_step_calls_before = training_env.step_calls
    python_rng_before = random.getstate()
    numpy_rng_before = np.random.get_state()
    torch_rng_before = torch.random.get_rng_state().clone()

    metrics = evaluator.evaluate()

    assert metrics["eval/value_samples"] == 2.0
    _assert_tree_equal(algorithm.agent.training_state_dict(), agent_before)
    assert algorithm.buffer.num_eps == replay_episodes_before
    assert algorithm._global_step == global_step_before
    assert training_env.reset_calls == training_reset_calls_before
    assert training_env.step_calls == training_step_calls_before
    assert random.getstate() == python_rng_before
    assert np.array_equal(np.random.get_state()[1], numpy_rng_before[1])
    assert np.random.get_state()[0] == numpy_rng_before[0]
    assert np.random.get_state()[2:] == numpy_rng_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_rng_before)

    evaluator.close()
    training_env.close()
