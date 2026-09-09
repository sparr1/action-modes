import copy
import random
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

import main as training_main
from RL.TDMPC2 import TDMPC2Baseline
from RL.tdmpc2_core.paired_controller_evaluation import (
    TDMPC2PairedControllerEvaluator,
)


class _ConfigEnv(gym.Env):
    metadata = {}

    def __init__(self, observation_type="state"):
        self.observation_type = observation_type
        if observation_type == "rgb":
            self.observation_space = gym.spaces.Box(
                0, 255, shape=(9, 64, 64), dtype=np.uint8
            )
        else:
            self.observation_space = gym.spaces.Box(
                -1.0, 1.0, shape=(3,), dtype=np.float32
            )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(2,), dtype=np.float32
        )
        self.spec = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        return self.observation_space.sample(), {}

    def step(self, action):
        del action
        return self.observation_space.sample(), 0.0, False, True, {}


class _CountingPairedEnv(gym.Env):
    metadata = {}

    def __init__(self, *, fail_step=False):
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, shape=(3,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(2,), dtype=np.float32
        )
        self.spec = None
        self.fail_step = bool(fail_step)
        self.reset_calls = 0
        self.step_calls = 0
        self.close_calls = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        self.reset_calls += 1
        random.random()
        np.random.random()
        torch.rand(())
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.step_calls += 1
        random.random()
        np.random.random()
        torch.rand(())
        if self.fail_step:
            raise RuntimeError("synthetic paired environment failure")
        reward = float(np.asarray(action, dtype=np.float64).sum())
        return np.zeros(3, dtype=np.float32), reward, False, True, {}

    def close(self):
        self.close_calls += 1


class _CountingPairedFactory:
    def __init__(self, *, fail_inner=False):
        self.fail_inner = bool(fail_inner)
        self.envs = []

    def __call__(self):
        env = _CountingPairedEnv(
            fail_step=self.fail_inner and len(self.envs) == 1
        )
        self.envs.append(env)
        return env


def _tiny_real_params(**overrides):
    params = {
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
        "buffer_size": 8,
        "episode_length": 1,
        "seed_steps": 1,
        "pretrain_steps": 1,
        "utd": 1,
        "compile": False,
        "episodic": False,
        "iterations": 1,
        "num_samples": 8,
        "num_elites": 2,
        "num_pi_trajs": 2,
        "eval_freq": 1,
        "eval_inner_comparison": True,
        "eval_inner_comparison_episodes": 1,
        "eval_inner_comparison_seed": 12_345,
        "wandb": False,
        "dropout": 0.0,
    }
    params.update(overrides)
    return params


def _assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
        return
    if isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
        return
    assert actual == expected


def _resolved_cfg(*, observation_type="state", **overrides):
    algorithm = object.__new__(TDMPC2Baseline)
    algorithm.env = _ConfigEnv(observation_type)
    algorithm.run_params = {
        "seed": 3,
        "device": "cpu",
        "env": "toy",
        "total_steps": 8,
    }
    params = {
        "device": "cpu",
        "obs": observation_type,
        "buffer_size": 8,
        "episode_length": 2,
        **overrides,
    }
    algorithm.custom_params = params
    try:
        return algorithm._build_cfg(params)
    finally:
        algorithm.env.close()


def test_tdmpc2_paired_comparison_defaults_are_disabled_and_explicit():
    cfg = _resolved_cfg()

    assert cfg.eval_inner_comparison is False
    assert cfg.eval_inner_comparison_episodes == 5
    assert cfg.eval_inner_comparison_seed == 12_345


def test_tdmpc2_paired_comparison_valid_configuration_resolves():
    cfg = _resolved_cfg(
        eval_freq=100_000,
        eval_inner_comparison=True,
        eval_inner_comparison_episodes=7,
        eval_inner_comparison_seed=24_680,
        mpc=True,
        num_pi_trajs=4,
    )

    assert cfg.eval_inner_comparison is True
    assert cfg.eval_inner_comparison_episodes == 7
    assert cfg.eval_inner_comparison_seed == 24_680
    assert cfg.eval_freq == 100_000


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"eval_inner_comparison": 1}, "must be a boolean"),
        ({"eval_inner_comparison_episodes": 5.0}, "positive integer"),
        ({"eval_inner_comparison_episodes": 0}, "positive integer"),
        ({"eval_inner_comparison_episodes": True}, "positive integer"),
        ({"eval_inner_comparison_seed": -1}, "non-negative integer"),
        ({"eval_inner_comparison_seed": True}, "non-negative integer"),
        ({"eval_inner_comparison": True}, "configured eval_freq"),
        (
            {"eval_inner_comparison": True, "eval_freq": 1, "mpc": False},
            "requires mpc=true",
        ),
        (
            {"eval_inner_comparison": True, "eval_freq": 1, "mpc": 1},
            "requires mpc=true",
        ),
        (
            {
                "eval_inner_comparison": True,
                "eval_freq": 1,
                "num_pi_trajs": 0,
            },
            "requires num_pi_trajs>0",
        ),
    ],
)
def test_tdmpc2_paired_comparison_configuration_fails_closed(
    overrides, message
):
    with pytest.raises(ValueError, match=message):
        _resolved_cfg(**overrides)


def test_tdmpc2_paired_comparison_rejects_rgb_observations():
    with pytest.raises(ValueError, match="state observations only"):
        _resolved_cfg(
            observation_type="rgb",
            eval_freq=1,
            eval_inner_comparison=True,
        )


def test_tdmpc2_paired_evaluator_factory_is_lazy_and_rebuilds_the_env_stack(
    monkeypatch,
):
    captured = {}
    built = []
    sentinel_env = object()

    class Evaluator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def build_env(run_params, experiment_params, *, render_mode):
        built.append((run_params, experiment_params, render_mode))
        return sentinel_env

    monkeypatch.setattr(
        "RL.tdmpc2_core.paired_controller_evaluation."
        "TDMPC2PairedControllerEvaluator",
        Evaluator,
    )
    monkeypatch.setattr("utils.core.build_env", build_env)

    algorithm = object.__new__(TDMPC2Baseline)
    algorithm.agent = SimpleNamespace(device=torch.device("cpu"))
    algorithm.cfg = SimpleNamespace(
        eval_inner_comparison_episodes=5,
        eval_inner_comparison_seed=12_345,
    )
    algorithm.run_params = {"env": "toy"}
    algorithm.experiment_params = {"env_params": {"task": "toy-task"}}

    evaluator = algorithm._make_paired_controller_evaluator()

    assert isinstance(evaluator, Evaluator)
    assert built == []
    assert captured["agent"] is algorithm.agent
    assert captured["episodes"] == 5
    assert captured["seed"] == 12_345
    assert captured["device"] == torch.device("cpu")
    assert captured["observation_to_tensor"].__self__ is algorithm
    assert captured["unscale_action"].__self__ is algorithm
    assert captured["env_factory"]() is sentinel_env
    assert built == [
        (algorithm.run_params, algorithm.experiment_params, None)
    ]


def test_tdmpc2_paired_extras_are_lazy_and_close_idempotently():
    class Evaluator:
        def __init__(self):
            self.evaluate_calls = 0
            self.close_calls = 0

        def evaluate(self):
            self.evaluate_calls += 1
            return {"eval/paired_episodes": 5.0}

        def close(self):
            self.close_calls += 1

    algorithm = object.__new__(TDMPC2Baseline)
    algorithm.cfg = SimpleNamespace(eval_inner_comparison=False)
    algorithm._paired_controller_evaluator = None
    algorithm._make_paired_controller_evaluator = lambda: pytest.fail(
        "disabled diagnostics must not construct an evaluator"
    )
    assert algorithm._evaluation_payload_extras(0) == {}

    evaluator = Evaluator()
    constructions = []

    def make_evaluator():
        constructions.append("paired")
        return evaluator

    algorithm.cfg.eval_inner_comparison = True
    algorithm._make_paired_controller_evaluator = make_evaluator
    assert algorithm._evaluation_payload_extras(0) == {
        "eval/paired_episodes": 5.0
    }
    assert algorithm._evaluation_payload_extras(100_000) == {
        "eval/paired_episodes": 5.0
    }
    assert constructions == ["paired"]
    assert evaluator.evaluate_calls == 2

    algorithm.close()
    algorithm.close()
    assert evaluator.close_calls == 1
    assert algorithm._paired_controller_evaluator is None


def test_tdmpc2_paired_extras_require_a_mapping():
    algorithm = object.__new__(TDMPC2Baseline)
    algorithm.cfg = SimpleNamespace(eval_inner_comparison=True)
    algorithm._paired_controller_evaluator = SimpleNamespace(
        evaluate=lambda: [("eval/paired_episodes", 5.0)]
    )

    with pytest.raises(TypeError, match="must return a mapping"):
        algorithm._evaluation_payload_extras(0)


def test_tdmpc2_paired_extras_merge_into_one_evaluation_event():
    class Run:
        def __init__(self):
            self.calls = []

        def log(self, payload, step):
            self.calls.append((int(step), dict(payload)))

    run = Run()
    algorithm = object.__new__(TDMPC2Baseline)
    algorithm._eval_episodes = 10
    algorithm._eval_csv_path = None
    algorithm._wandb_run = run
    algorithm._resume_enabled = False
    algorithm.cfg = SimpleNamespace(seed=3)

    algorithm._record_evaluation(
        100_000,
        12.5,
        extras={
            "eval/paired_outer_episode_reward": 10.0,
            "eval/paired_fresh_inner_episode_reward": 15.0,
        },
    )

    assert len(run.calls) == 1
    step, payload = run.calls[0]
    assert step == 100_000
    assert payload["eval/episode_reward"] == 12.5
    assert payload["eval/episodes"] == 10
    assert payload["eval/paired_outer_episode_reward"] == 10.0
    assert payload["eval/paired_fresh_inner_episode_reward"] == 15.0


def test_tdmpc2_runtime_metadata_includes_paired_comparison_defaults():
    model = SimpleNamespace(
        cfg=SimpleNamespace(
            eval_freq=100_000,
            eval_episodes=10,
            eval_inner_comparison=False,
            eval_inner_comparison_episodes=5,
            eval_inner_comparison_seed=12_345,
        ),
        agent=SimpleNamespace(),
        env=SimpleNamespace(),
    )

    metadata = training_main._resolved_runtime_metadata(
        model,
        trial_run_params={
            "alg": "TDMPC2/TDMPC2Baseline",
            "seed": 3,
        },
    )

    assert metadata["evaluation"] == {
        "eval_freq": 100_000,
        "eval_episodes": 10,
        "eval_inner_comparison": False,
        "eval_inner_comparison_episodes": 5,
        "eval_inner_comparison_seed": 12_345,
    }


@pytest.mark.parametrize("fail_inner", [False, True])
def test_real_tdmpc2_paired_probe_preserves_all_training_state(fail_inner):
    training_env = _CountingPairedEnv()
    algorithm = TDMPC2Baseline(
        "paired-state-isolation",
        training_env,
        _tiny_real_params(),
        {"seed": 3, "device": "cpu", "env": "toy", "total_steps": 8},
        {},
    )
    factory = _CountingPairedFactory(fail_inner=fail_inner)
    algorithm._paired_controller_evaluator = TDMPC2PairedControllerEvaluator(
        agent=algorithm.agent,
        env_factory=factory,
        observation_to_tensor=algorithm._obs_to_tensor,
        unscale_action=algorithm._unscale_action,
        episodes=1,
        seed=12_345,
        device=algorithm.agent.device,
    )

    algorithm._wandb_train_window.add_sum("train/sentinel", 3.0)
    algorithm._wandb_reward_window.set_last("rollout/sentinel", 4.0)
    algorithm._wandb_update_window.update(
        {"update_sentinel": torch.tensor(5.0)}
    )
    algorithm._global_step = 7
    algorithm._episode_idx = 2
    algorithm._num_updates = 6
    algorithm._predict_t0 = False
    algorithm.agent._prev_mean.fill_(0.125)
    algorithm.agent.last_plan_metrics = {"training": 17.0}
    algorithm.agent._resume_boundary_prepared = True
    algorithm.agent.train(True)
    algorithm.agent.model.train(False)
    next(iter(algorithm.agent.model.modules())).training = True

    random.seed(701)
    np.random.seed(702)
    torch.manual_seed(703)
    agent_state = copy.deepcopy(algorithm.agent.training_state_dict())
    wrapper_state = {
        "global_step": algorithm._global_step,
        "episode_idx": algorithm._episode_idx,
        "num_updates": algorithm._num_updates,
        "predict_t0": algorithm._predict_t0,
        "train_window": copy.deepcopy(
            algorithm._wandb_train_window.__dict__
        ),
        "reward_window": copy.deepcopy(
            algorithm._wandb_reward_window.__dict__
        ),
        "update_window": copy.deepcopy(
            algorithm._wandb_update_window.__dict__
        ),
        "buffer": {
            "num_eps": algorithm.buffer.num_eps,
            "num_transitions": algorithm.buffer.num_transitions,
            "total_transitions": algorithm.buffer.total_transitions,
            "size": algorithm.buffer.size,
            "resident_episode_rows": copy.deepcopy(
                algorithm.buffer._resident_episode_rows
            ),
            "resident_rows": algorithm.buffer._resident_rows,
            "has_storage": hasattr(algorithm.buffer, "_buffer"),
        },
        "module_modes": tuple(
            module.training for module in algorithm.agent.modules()
        ),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state().clone(),
        "training_env": (
            training_env.reset_calls,
            training_env.step_calls,
        ),
    }

    try:
        if fail_inner:
            with pytest.raises(
                RuntimeError, match="synthetic paired environment failure"
            ):
                algorithm._evaluation_payload_extras(0)
        else:
            metrics = algorithm._evaluation_payload_extras(0)
            assert metrics["eval/paired_episodes"] == 1.0
            assert metrics[
                "eval/paired_fresh_inner_fixed_target_q_action_gain_count"
            ] == 1.0

        _assert_nested_equal(
            algorithm.agent.training_state_dict(), agent_state
        )
        _assert_nested_equal(
            algorithm._wandb_train_window.__dict__,
            wrapper_state["train_window"],
        )
        _assert_nested_equal(
            algorithm._wandb_reward_window.__dict__,
            wrapper_state["reward_window"],
        )
        _assert_nested_equal(
            algorithm._wandb_update_window.__dict__,
            wrapper_state["update_window"],
        )
        assert algorithm._global_step == wrapper_state["global_step"]
        assert algorithm._episode_idx == wrapper_state["episode_idx"]
        assert algorithm._num_updates == wrapper_state["num_updates"]
        assert algorithm._predict_t0 is wrapper_state["predict_t0"]
        assert {
            "num_eps": algorithm.buffer.num_eps,
            "num_transitions": algorithm.buffer.num_transitions,
            "total_transitions": algorithm.buffer.total_transitions,
            "size": algorithm.buffer.size,
            "resident_episode_rows": algorithm.buffer._resident_episode_rows,
            "resident_rows": algorithm.buffer._resident_rows,
            "has_storage": hasattr(algorithm.buffer, "_buffer"),
        } == wrapper_state["buffer"]
        assert tuple(
            module.training for module in algorithm.agent.modules()
        ) == wrapper_state["module_modes"]
        assert random.getstate() == wrapper_state["python_rng"]
        _assert_nested_equal(np.random.get_state(), wrapper_state["numpy_rng"])
        torch.testing.assert_close(
            torch.random.get_rng_state(),
            wrapper_state["torch_rng"],
            rtol=0,
            atol=0,
        )
        assert (
            training_env.reset_calls,
            training_env.step_calls,
        ) == wrapper_state["training_env"]
    finally:
        algorithm.close()
        training_env.close()

    assert len(factory.envs) == 2
    assert [env.close_calls for env in factory.envs] == [1, 1]
