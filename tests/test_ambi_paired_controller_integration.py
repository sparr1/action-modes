import random

import gymnasium as gym
import numpy as np
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core.paired_controller_evaluation import (
    PairedControllerEvaluator,
)
from RL.tdmpc2_core.value_calibration import ValueCalibrationEvaluator


class _SeededFiniteEnv(gym.Env):
    metadata = {}

    def __init__(self, *, fail_on_step=False):
        self.observation_space = gym.spaces.Box(
            -1.0, 1.0, shape=(3,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(1,), dtype=np.float32
        )
        self.spec = None
        self.fail_on_step = fail_on_step
        self.reset_calls = 0
        self.step_calls = 0
        self.close_calls = 0
        self.reset_seeds = []
        self.actions = []
        self._state = 0.0

    @staticmethod
    def _consume_global_rng():
        random.random()
        np.random.random()
        torch.rand(())

    def reset(self, *, seed=None, options=None):
        del options
        super().reset(seed=seed)
        self.reset_calls += 1
        self.reset_seeds.append(seed)
        self._state = float(self.np_random.uniform(-0.5, 0.5))
        self._consume_global_rng()
        return np.asarray(
            [self._state, 0.5 * self._state, -self._state],
            dtype=np.float32,
        ), {}

    def step(self, action):
        self.step_calls += 1
        self._consume_global_rng()
        action_value = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        if not np.isfinite(action_value):
            raise ValueError("toy action must be finite")
        self.actions.append(action_value)
        if self.fail_on_step:
            raise RuntimeError("injected auxiliary environment failure")
        reward = 1.0 + 0.05 * action_value + 0.01 * self._state
        observation = np.asarray(
            [self._state, -self._state, 0.25 * self._state],
            dtype=np.float32,
        )
        return observation, reward, False, True, {}

    def close(self):
        self.close_calls += 1


class _AuxEnvFactory:
    def __init__(self, *, fail_inner_step=False):
        self.fail_inner_step = fail_inner_step
        self.envs = []

    def __call__(self):
        env = _SeededFiniteEnv(
            fail_on_step=self.fail_inner_step and len(self.envs) == 1
        )
        self.envs.append(env)
        return env


class _FakeRun:
    def __init__(self):
        self.calls = []

    def log(self, payload, step):
        self.calls.append((int(step), dict(payload)))


def _tiny_ambi_params():
    return {
        "device": "cpu",
        "model_size": None,
        "enc_dim": 16,
        "mlp_dim": 16,
        "latent_dim": 8,
        "num_enc_layers": 2,
        "simnorm_dim": 4,
        "num_bins": 5,
        "vmin": -5,
        "vmax": 5,
        "batch_size": 1,
        "train_unroll_horizon": 1,
        "outer_planning_horizon": 1,
        "buffer_size": 8,
        "seed_steps": 1,
        "pretrain_steps": 1,
        "utd": 1,
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
        "q_representation": "scalar",
        "q_num_bins": 5,
        "q_vmin": -5,
        "q_vmax": 5,
        "q_pair_size": 2,
        "outer_critic_target": "reward_only",
        "inner_sac_critic_target": "reward_only",
        "inner_operator": "sac",
        "inner_rounds": 1,
        "inner_rollouts_per_round": 1,
        "inner_rollout_horizon": 1,
        "inner_updates_per_round": 1,
        "inner_batch_size": 1,
        "inner_replay_capacity": 2,
        "inner_temperature_mode": "inherit_outer",
        "inner_critic_dropout_enabled": False,
        "inner_mppi_num_elites": 1,
        "inner_mppi_num_pi_trajs": 0,
        "inner_diagnostic_rollouts": 0,
        "outer_behavior_policy_kl_schedule": "dual",
        "outer_behavior_policy_kl_min_valid_count": 1,
        "eval_inner_comparison": True,
        "eval_inner_comparison_episodes": 1,
        "eval_inner_comparison_seed": 12345,
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
    return value


def _assert_tree_equal(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0,
            atol=0,
            equal_nan=True,
        )
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


def _assert_numpy_rng_equal(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def _prime_optimizer(optimizer):
    parameter = next(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    )
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _build_algorithm():
    training_env = _SeededFiniteEnv()
    algorithm = AMBITDMPC2(
        "AMBITDMPC2",
        training_env,
        _tiny_ambi_params(),
        {"seed": 11, "device": "cpu", "env": "toy", "total_steps": 8},
        {},
    )

    for optimizer in (
        algorithm.agent.optim,
        algorithm.agent.pi_optim,
        algorithm.agent.ent_coef_optim,
        algorithm.agent.behavior_policy_kl_optim,
    ):
        _prime_optimizer(optimizer)
    algorithm.agent.behavior_policy_kl_dual_updates = 3

    algorithm.buffer.enable_resumable_storage()
    observation = np.zeros(3, dtype=np.float32)
    episode = torch.cat(
        [
            algorithm._to_td(observation),
            algorithm._to_td(
                observation,
                np.zeros(1, dtype=np.float32),
                1.0,
                True,
            ),
        ]
    )
    algorithm.buffer.add(episode)
    algorithm._global_step = 6
    algorithm._wandb_train_window.add_sum("train/sentinel", 3.0)
    algorithm._wandb_update_window.add(
        "update/sentinel", torch.tensor(4.0)
    )
    algorithm._wandb_reward_window.add_weighted(
        "reward/sentinel", 5.0, weight=2.0
    )
    algorithm._wandb_inner_seconds = 1.25
    algorithm._wandb_inner_actions = 2
    algorithm._wandb_inner_steps = 3

    training_env.reset(seed=101)
    training_env.step(np.zeros(1, dtype=np.float32))
    algorithm.agent.last_inner_metrics = {"training/sentinel": 7.0}
    algorithm.agent.last_inner_rollout_lengths = [8]
    return algorithm, training_env


def _set_heterogeneous_module_modes(agent):
    agent.train(True)
    agent.model.eval()
    leaf = list(agent.model.modules())[-1]
    leaf.train(True)
    modes = tuple(bool(module.training) for module in agent.modules())
    assert any(modes) and not all(modes)


def _capture_invariants(algorithm, training_env):
    agent = algorithm.agent
    engine = agent.inner_engine
    agent_state = _clone_tree(agent.training_state_dict())
    outer_state = agent_state["outer"]
    assert outer_state["optim"]["state"]
    assert outer_state["pi_optim"]["state"]
    assert outer_state["ent_coef_optim"]["state"]
    assert outer_state["behavior_policy_kl_state"]["optim"]["state"]
    return {
        "agent_state": agent_state,
        "engine": engine,
        "engine_state_ref": engine.state,
        "engine_rng_ref": engine.rng,
        "engine_state": _clone_tree(engine.training_state_dict()),
        "engine_rng": _clone_tree(engine.rng.training_state_dict()),
        "module_modes": tuple(
            bool(module.training) for module in agent.modules()
        ),
        "last_metrics": agent.last_inner_metrics,
        "last_lengths": agent.last_inner_rollout_lengths,
        "buffer": algorithm.buffer,
        "replay_metadata": _clone_tree(
            algorithm.buffer.training_state_metadata()
        ),
        "replay_shards": _clone_tree(
            list(algorithm.buffer.iter_training_state_shards(max_rows=2))
        ),
        "global_step": algorithm._global_step,
        "wandb_windows": (
            _clone_tree(algorithm._wandb_train_window.snapshot()),
            _clone_tree(algorithm._wandb_update_window.snapshot()),
            _clone_tree(algorithm._wandb_reward_window.snapshot()),
        ),
        "wandb_inner": (
            algorithm._wandb_inner_seconds,
            algorithm._wandb_inner_actions,
            algorithm._wandb_inner_steps,
        ),
        "training_env_counts": (
            training_env.reset_calls,
            training_env.step_calls,
        ),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state().clone(),
    }


def _assert_invariants_preserved(algorithm, training_env, snapshot):
    assert random.getstate() == snapshot["python_rng"]
    _assert_numpy_rng_equal(np.random.get_state(), snapshot["numpy_rng"])
    torch.testing.assert_close(
        torch.random.get_rng_state(),
        snapshot["torch_rng"],
        rtol=0,
        atol=0,
    )

    agent = algorithm.agent
    engine = snapshot["engine"]
    assert agent.inner_engine is engine
    assert engine.state is snapshot["engine_state_ref"]
    assert engine.rng is snapshot["engine_rng_ref"]
    _assert_tree_equal(agent.training_state_dict(), snapshot["agent_state"])
    _assert_tree_equal(engine.training_state_dict(), snapshot["engine_state"])
    _assert_tree_equal(
        engine.rng.training_state_dict(), snapshot["engine_rng"]
    )
    assert tuple(
        bool(module.training) for module in agent.modules()
    ) == snapshot["module_modes"]
    assert agent.last_inner_metrics is snapshot["last_metrics"]
    assert agent.last_inner_rollout_lengths is snapshot["last_lengths"]

    assert algorithm.buffer is snapshot["buffer"]
    _assert_tree_equal(
        algorithm.buffer.training_state_metadata(),
        snapshot["replay_metadata"],
    )
    _assert_tree_equal(
        list(algorithm.buffer.iter_training_state_shards(max_rows=2)),
        snapshot["replay_shards"],
    )
    assert algorithm._global_step == snapshot["global_step"]
    _assert_tree_equal(
        (
            algorithm._wandb_train_window.snapshot(),
            algorithm._wandb_update_window.snapshot(),
            algorithm._wandb_reward_window.snapshot(),
        ),
        snapshot["wandb_windows"],
    )
    assert (
        algorithm._wandb_inner_seconds,
        algorithm._wandb_inner_actions,
        algorithm._wandb_inner_steps,
    ) == snapshot["wandb_inner"]
    assert (
        training_env.reset_calls,
        training_env.step_calls,
    ) == snapshot["training_env_counts"]


def _paired_evaluator(algorithm, factory):
    return PairedControllerEvaluator(
        agent=algorithm.agent,
        env_factory=factory,
        observation_to_tensor=algorithm._obs_to_tensor,
        unscale_action=algorithm._unscale_action,
        episodes=1,
        seed=12345,
        device="cpu",
    )


def test_real_paired_evaluation_is_observational_and_routes_controllers(
    monkeypatch,
):
    algorithm, training_env = _build_algorithm()
    _set_heterogeneous_module_modes(algorithm.agent)
    random.seed(701)
    np.random.seed(702)
    torch.manual_seed(703)
    snapshot = _capture_invariants(algorithm, training_env)
    factory = _AuxEnvFactory()
    evaluator = _paired_evaluator(algorithm, factory)
    _assert_invariants_preserved(algorithm, training_env, snapshot)

    events = []
    outer_calls = []
    inner_calls = []
    engine_calls = []
    original_outer = algorithm.agent.act_outer_policy
    original_inner = algorithm.agent.act
    evaluation_engine = evaluator._evaluation_inner_engine
    original_engine_act = evaluation_engine.act

    def tracked_outer(observation, **kwargs):
        events.append("outer_agent")
        outer_calls.append(dict(kwargs))
        return original_outer(observation, **kwargs)

    def tracked_inner(observation, **kwargs):
        events.append("inner_agent")
        inner_calls.append(dict(kwargs))
        return original_inner(observation, **kwargs)

    def tracked_engine_act(root_z, **kwargs):
        events.append("inner_engine")
        engine_calls.append(dict(kwargs))
        return original_engine_act(root_z, **kwargs)

    monkeypatch.setattr(algorithm.agent, "act_outer_policy", tracked_outer)
    monkeypatch.setattr(algorithm.agent, "act", tracked_inner)
    monkeypatch.setattr(evaluation_engine, "act", tracked_engine_act)

    metrics = evaluator.evaluate()

    assert metrics["eval/paired_episodes"] == 1.0
    assert metrics["eval/paired_fresh_inner_model_steps_per_action"] == 1.0
    assert all(np.isfinite(value) for value in metrics.values())
    assert outer_calls == [{"deterministic": True}]
    assert inner_calls == [
        {
            "t0": True,
            "eval_mode": True,
            "collect_diagnostics": True,
            "apply_inner_writeback": False,
        }
    ]
    assert len(engine_calls) == 1
    assert events == ["outer_agent", "inner_agent", "inner_engine"]
    assert len(factory.envs) == 2
    assert factory.envs[0] is not factory.envs[1]
    assert factory.envs[0].reset_seeds == factory.envs[1].reset_seeds
    _assert_invariants_preserved(algorithm, training_env, snapshot)

    evaluator.close()
    evaluator.close()
    assert [env.close_calls for env in factory.envs] == [1, 1]
    _assert_invariants_preserved(algorithm, training_env, snapshot)
    training_env.close()


@pytest.mark.parametrize("failure", ["environment", "agent"])
def test_real_paired_evaluation_failure_restores_all_training_state(
    monkeypatch,
    failure,
):
    algorithm, training_env = _build_algorithm()
    _set_heterogeneous_module_modes(algorithm.agent)
    random.seed(801)
    np.random.seed(802)
    torch.manual_seed(803)
    snapshot = _capture_invariants(algorithm, training_env)
    factory = _AuxEnvFactory(fail_inner_step=failure == "environment")
    evaluator = _paired_evaluator(algorithm, factory)

    if failure == "agent":
        def fail_agent(*_args, **_kwargs):
            random.random()
            np.random.random()
            torch.rand(())
            algorithm.agent.last_inner_metrics = {"corrupted": 1.0}
            algorithm.agent.last_inner_rollout_lengths = [999]
            raise RuntimeError("injected agent failure")

        monkeypatch.setattr(algorithm.agent, "act", fail_agent)
        message = "injected agent failure"
    else:
        message = "injected auxiliary environment failure"

    with pytest.raises(RuntimeError, match=message):
        evaluator.evaluate()

    _assert_invariants_preserved(algorithm, training_env, snapshot)
    evaluator.close()
    evaluator.close()
    assert [env.close_calls for env in factory.envs] == [1, 1]
    _assert_invariants_preserved(algorithm, training_env, snapshot)
    training_env.close()


def test_real_paired_and_value_extras_emit_one_wandb_event():
    algorithm, training_env = _build_algorithm()
    value_factory = _AuxEnvFactory()
    paired_factory = _AuxEnvFactory()
    algorithm.cfg.eval_value = True
    algorithm._value_calibration_evaluator = ValueCalibrationEvaluator(
        model=algorithm.agent.model,
        env_factory=value_factory,
        observation_to_tensor=algorithm._obs_to_tensor,
        unscale_action=algorithm._unscale_action,
        discount=float(algorithm.cfg.discount),
        samples=1,
        seed=23456,
        protocols=("paper_deterministic",),
        device="cpu",
    )
    algorithm._paired_controller_evaluator = _paired_evaluator(
        algorithm, paired_factory
    )
    run = _FakeRun()
    algorithm._wandb_run = run

    extras = algorithm._evaluation_payload_extras(17)
    algorithm._record_evaluation(17, 2.5, extras=extras)

    assert len(run.calls) == 1
    step, payload = run.calls[0]
    assert step == 17
    assert payload["env_step"] == 17
    assert payload["eval/episode_reward"] == 2.5
    assert payload["eval/value_samples"] == 1.0
    assert payload["eval/paired_episodes"] == 1.0

    algorithm.close()
    algorithm.close()
    assert [env.close_calls for env in value_factory.envs] == [1]
    assert [env.close_calls for env in paired_factory.envs] == [1, 1]
    training_env.close()
