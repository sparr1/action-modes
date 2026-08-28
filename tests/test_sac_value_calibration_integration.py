from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from RL.SAC import SAC


class _OneStepEnv(gym.Env):
    metadata = {}

    def __init__(self, *, fail_after=None):
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, shape=(3,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -2.0, 2.0, shape=(1,), dtype=np.float32
        )
        self.fail_after = fail_after
        self.step_calls = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.step_calls += 1
        if self.fail_after is not None and self.step_calls > self.fail_after:
            raise RuntimeError("environment failed")
        return np.zeros(3, dtype=np.float32), 2.0, False, True, {}


class _FakeRun:
    def __init__(self):
        self.calls = []

    def log(self, payload, step):
        self.calls.append((int(step), dict(payload)))


class _FakeEvaluator:
    def __init__(self, metrics=None, error=None):
        self.metrics = metrics or {"eval/mc_value": 3.0}
        self.error = error
        self.evaluate_calls = 0
        self.close_calls = 0

    def evaluate(self):
        self.evaluate_calls += 1
        if self.error is not None:
            raise self.error
        return dict(self.metrics)

    def close(self):
        self.close_calls += 1


def _model(env=None, **overrides):
    params = {
        "device": "cpu",
        "seed": 7,
        "learning_starts": 100,
        "batch_size": 2,
        "buffer_size": 32,
        "net_arch": [8],
        "wandb": False,
    }
    params.update(overrides)
    return SAC(
        "SAC",
        env or _OneStepEnv(),
        params,
        {"seed": 7, "device": "cpu", "env": "test", "total_steps": 1},
        {"env_params": {"task": "test", "obs": "state"}},
    )


def test_value_calibration_defaults_are_strict_disabled_wrapper_options():
    model = _model()

    assert model._eval_value is False
    assert model._eval_value_samples == 100
    assert model._eval_value_seed == 12345
    assert model._eval_value_protocols == [
        "paper_deterministic",
        "stochastic_soft_bellman",
    ]
    assert model._value_calibration_evaluator is None
    assert "eval_value" not in vars(model.cfg)


def test_wandb_config_records_environment_metadata_for_plot_validation(monkeypatch):
    model = _model()
    captured = {}

    def fake_init_wandb(params, **kwargs):
        captured["params"] = params
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("RL.SAC.init_wandb", fake_init_wandb)

    assert model._init_wandb() is not None
    assert captured["config"]["run_params"] is model.run_params
    assert captured["config"]["experiment_params"] is model.experiment_params
    assert captured["config"]["experiment_params"]["env_params"] == {
        "task": "test",
        "obs": "state",
    }


def test_valid_vanilla_twin_q_value_calibration_configuration_resolves():
    model = _model(
        eval_freq=50_000,
        eval_episodes=1,
        eval_value=True,
        eval_value_samples=17,
        eval_value_seed=23,
        eval_value_protocols=["stochastic_soft_bellman"],
    )

    assert model._eval_value is True
    assert model._eval_value_samples == 17
    assert model._eval_value_seed == 23
    assert model._eval_value_protocols == ["stochastic_soft_bellman"]
    assert model.cfg.q_representation == "scalar"
    assert model.cfg.num_q == model.cfg.q_pair_size == 2
    assert model.cfg.q_target_reduction == "min_pair"
    assert model.cfg.q_actor_reduction == "min_pair"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"eval_value": "true"}, "eval_value must be a boolean"),
        ({"eval_value_samples": 1.0}, "eval_value_samples"),
        ({"eval_value_samples": 0}, "eval_value_samples"),
        ({"eval_value_seed": True}, "eval_value_seed"),
        ({"eval_value_seed": -1}, "eval_value_seed"),
        ({"eval_value_protocols": "paper_deterministic"}, "non-empty list"),
        ({"eval_value_protocols": []}, "must not be empty"),
        ({"eval_value_protocols": [1]}, "entries must be strings"),
        ({"eval_value_protocols": ["unknown"]}, "chosen from"),
        (
            {
                "eval_value_protocols": [
                    "paper_deterministic",
                    "paper_deterministic",
                ]
            },
            "duplicate",
        ),
        ({"eval_value": True}, "eval_freq"),
        (
            {
                "eval_value": True,
                "eval_freq": 1,
                "q_representation": "distributional",
                "num_q": 2,
            },
            "q_representation='scalar'",
        ),
        (
            {"eval_value": True, "eval_freq": 1, "num_q": 3},
            "num_q=2",
        ),
        (
            {"eval_value": True, "eval_freq": 1, "q_pair_size": 3},
            "q_pair_size=2",
        ),
        (
            {
                "eval_value": True,
                "eval_freq": 1,
                "q_target_reduction": "mean_pair",
            },
            "q_target_reduction='min_pair'",
        ),
        (
            {
                "eval_value": True,
                "eval_freq": 1,
                "q_actor_reduction": "mean_pair",
            },
            "q_actor_reduction='min_pair'",
        ),
    ],
)
def test_value_calibration_configuration_fails_closed(overrides, message):
    with pytest.raises(ValueError, match=message):
        _model(**overrides)


def test_evaluator_constructor_wiring_uses_an_auxiliary_environment(
    monkeypatch,
):
    model = _model(
        eval_freq=50_000,
        eval_value=True,
        eval_value_samples=19,
        eval_value_seed=29,
        eval_value_protocols=[
            "paper_deterministic",
            "stochastic_soft_bellman",
        ],
    )
    captured = {}
    auxiliary_env = object()

    class Evaluator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def build_env(run_params, experiment_params, *, render_mode):
        captured["build_env"] = (run_params, experiment_params, render_mode)
        return auxiliary_env

    monkeypatch.setattr(
        "RL.sac_value_calibration.SACValueCalibrationEvaluator", Evaluator
    )
    monkeypatch.setattr("utils.core.build_env", build_env)

    evaluator = model._make_value_calibration_evaluator()

    assert isinstance(evaluator, Evaluator)
    assert captured["agent"] is model.agent
    assert captured["observation_to_array"].__self__ is model
    assert captured["unscale_action"].__self__ is model
    assert captured["discount"] == model.cfg.gamma
    assert captured["samples"] == 19
    assert captured["seed"] == 29
    assert captured["protocols"] == (
        "paper_deterministic",
        "stochastic_soft_bellman",
    )
    assert captured["device"] == model.agent.device
    assert captured["env_factory"]() is auxiliary_env
    assert captured["build_env"] == (
        model.run_params,
        model.experiment_params,
        None,
    )


def test_value_evaluator_is_lazy_reused_and_closed_idempotently():
    model = _model()
    model._make_value_calibration_evaluator = lambda: pytest.fail(
        "disabled calibration must not construct an evaluator"
    )
    assert model._evaluation_payload_extras(0) == {}

    evaluator = _FakeEvaluator()
    model._eval_value = True
    model._make_value_calibration_evaluator = lambda: evaluator
    assert model._evaluation_payload_extras(0) == {"eval/mc_value": 3.0}
    assert model._evaluation_payload_extras(50_000) == {"eval/mc_value": 3.0}
    assert evaluator.evaluate_calls == 2

    model.close()
    model.close()
    assert evaluator.close_calls == 1
    assert model._value_calibration_evaluator is None


def test_evaluation_merges_numeric_calibration_metrics_into_one_wandb_event():
    env = _OneStepEnv()
    model = _model(env, eval_freq=1, eval_episodes=1, eval_value=True)
    model.agent.act = lambda _obs, deterministic=False: np.zeros(
        1, dtype=np.float32
    )
    model._evaluation_payload_extras = lambda _step: {
        "eval/mc_value": np.float32(1.25),
        "eval/stochastic_soft_q_min_all": 2,
    }
    run = _FakeRun()
    model._wandb_run = run
    observation, _ = env.reset(seed=7)

    result = model._evaluate_policy(12, initial_obs=observation)

    assert result == 2.0
    assert run.calls == [
        (
            12,
            {
                "eval/episode_reward": 2.0,
                "eval/episodes": 1,
                "eval/mc_value": 1.25,
                "eval/stochastic_soft_q_min_all": 2.0,
                "env_step": 12,
            },
        )
    ]


@pytest.mark.parametrize(
    ("extras", "error", "message"),
    [
        ([1.0], TypeError, "must be a mapping"),
        ({"env_step": 1.0}, ValueError, "reserved metrics"),
        ({"": 1.0}, TypeError, "non-empty strings"),
        ({"eval/value": True}, TypeError, "must be numeric"),
        ({"eval/value": "1"}, TypeError, "must be numeric"),
        ({"eval/value": np.inf}, ValueError, "must be finite"),
    ],
)
def test_evaluation_payload_rejects_unsafe_extras(extras, error, message):
    model = _model(eval_freq=1, eval_episodes=1)

    with pytest.raises(error, match=message):
        model._record_evaluation(0, 1.0, extras=extras)


@pytest.mark.parametrize(
    ("fail_after", "evaluator_error", "expected_error"),
    [
        (None, None, None),
        (None, RuntimeError("probe failed"), "probe failed"),
        (1, None, "environment failed"),
    ],
)
def test_learn_closes_the_auxiliary_evaluator_on_every_path(
    fail_after, evaluator_error, expected_error
):
    env = _OneStepEnv(fail_after=fail_after)
    model = _model(env, eval_freq=1, eval_episodes=1, eval_value=True)
    model.agent.act = lambda _obs, deterministic=False: np.zeros(
        1, dtype=np.float32
    )
    evaluator = _FakeEvaluator(error=evaluator_error)
    model._make_value_calibration_evaluator = lambda: evaluator

    if expected_error is None:
        model.learn(total_timesteps=0)
    else:
        total_timesteps = 0 if evaluator_error is not None else 1
        with pytest.raises(RuntimeError, match=expected_error):
            model.learn(total_timesteps=total_timesteps)

    assert evaluator.evaluate_calls == 1
    assert evaluator.close_calls == 1
    assert model._value_calibration_evaluator is None


def test_wandb_initialization_failure_still_closes_a_preexisting_evaluator():
    model = _model()
    evaluator = _FakeEvaluator()
    model._value_calibration_evaluator = evaluator
    model._init_wandb = lambda: (_ for _ in ()).throw(
        RuntimeError("wandb initialization failed")
    )

    with pytest.raises(RuntimeError, match="wandb initialization failed"):
        model.learn(total_timesteps=0)

    assert evaluator.close_calls == 1
    assert model._value_calibration_evaluator is None


def test_scalar_twin_q_configuration_is_visible_to_the_evaluator_contract():
    model = _model(eval_freq=1, eval_value=True)

    contract = SimpleNamespace(
        q_representation=model.cfg.q_representation,
        num_q=model.cfg.num_q,
        q_pair_size=model.cfg.q_pair_size,
        target=model.cfg.q_target_reduction,
        actor=model.cfg.q_actor_reduction,
        device=torch.device(model.agent.device),
    )
    assert contract == SimpleNamespace(
        q_representation="scalar",
        num_q=2,
        q_pair_size=2,
        target="min_pair",
        actor="min_pair",
        device=torch.device("cpu"),
    )
