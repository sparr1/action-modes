import random

import numpy as np
import pytest
import torch

from RL.tdmpc2_core.paired_controller_evaluation import (
    PairedControllerEvaluator,
    TDMPC2PairedControllerEvaluator,
    _namespaced_seed,
)


_Q_STEM = "eval/paired_fresh_inner_fixed_target_q_action_gain"


class _ThreeStepEnv:
    def __init__(
        self,
        *,
        fail=False,
        nonfinite_reward=False,
        initial_offset=0.0,
        boundary="truncated",
    ):
        self.fail = bool(fail)
        self.nonfinite_reward = bool(nonfinite_reward)
        self.initial_offset = float(initial_offset)
        self.boundary = boundary
        self.reset_seeds = []
        self.actions = []
        self.close_calls = 0
        self._step = 0

    @staticmethod
    def _consume_global_rng():
        random.random()
        np.random.random()
        torch.rand(())

    def reset(self, *, seed=None):
        self._consume_global_rng()
        self.reset_seeds.append(seed)
        self._step = 0
        return np.array([1.0 + self.initial_offset], dtype=np.float32), {}

    def step(self, action):
        self._consume_global_rng()
        if self.fail:
            raise RuntimeError("synthetic paired environment failure")
        action_value = float(np.asarray(action).reshape(-1)[0])
        self.actions.append(action_value)
        self._step += 1
        reward = float(self._step) + action_value
        if self.nonfinite_reward:
            reward = float("nan")
        done = self._step == 3
        return (
            np.array([1.0 + self._step], dtype=np.float32),
            reward,
            done and self.boundary == "terminated",
            done and self.boundary == "truncated",
            {},
        )

    def close(self):
        self.close_calls += 1


class _Factory:
    def __init__(
        self,
        *,
        fail_inner=False,
        nonfinite_reward=False,
        shared=False,
        mismatch_inner=False,
        boundary="truncated",
    ):
        self.fail_inner = bool(fail_inner)
        self.nonfinite_reward = bool(nonfinite_reward)
        self.shared = bool(shared)
        self.mismatch_inner = bool(mismatch_inner)
        self.boundary = boundary
        self.calls = 0
        self.envs = []

    def __call__(self):
        self.calls += 1
        if self.shared and self.envs:
            return self.envs[0]
        env = _ThreeStepEnv(
            fail=self.fail_inner and self.calls == 2,
            nonfinite_reward=self.nonfinite_reward,
            initial_offset=float(self.mismatch_inner and self.calls == 2),
            boundary=self.boundary,
        )
        self.envs.append(env)
        return env


class _PlannedReturnEnv:
    def __init__(self, rewards):
        self.rewards = list(rewards)
        self.reset_seeds = []
        self.close_calls = 0
        self._episode = -1

    def reset(self, *, seed=None):
        self.reset_seeds.append(seed)
        self._episode += 1
        return np.array([1.0], dtype=np.float32), {}

    def step(self, action):
        del action
        return (
            np.array([2.0], dtype=np.float32),
            self.rewards[self._episode],
            True,
            False,
            {},
        )

    def close(self):
        self.close_calls += 1


class _PlannedReturnFactory:
    def __init__(self):
        self.calls = 0
        self.envs = []

    def __call__(self):
        rewards = ([1.0, 4.0, 7.0], [3.0, 4.0, 5.0])[self.calls]
        self.calls += 1
        env = _PlannedReturnEnv(rewards)
        self.envs.append(env)
        return env


class _FakeAgent(torch.nn.Module):
    def __init__(self, *, invalid_metric=None, fail_mpc=False):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, 2),
            torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(2, 1)),
        )
        self.register_buffer("_prev_mean", torch.full((2, 1), 7.0))
        self.last_plan_metrics = {"training": 17.0}
        self._resume_boundary_prepared = True
        self.invalid_metric = invalid_metric
        self.fail_mpc = bool(fail_mpc)
        self.policy_calls = []
        self.mpc_calls = []
        self.call_modes = []
        self.reset_calls = 0
        self.mpc_rng_draws = []

    def _record_modes(self):
        self.call_modes.append(tuple(module.training for module in self.modules()))

    def reset(self):
        self.reset_calls += 1
        if self._resume_boundary_prepared:
            self._resume_boundary_prepared = False
            return
        self._prev_mean.zero_()
        self.last_plan_metrics = {}

    def act_policy_mean(self, observation):
        random.random()
        np.random.random()
        torch.rand(())
        self._record_modes()
        self.policy_calls.append(observation.detach().clone())
        self.last_plan_metrics = {"outer_touched": True}
        return torch.tensor([0.0])

    def act_mpc(
        self,
        observation,
        *,
        t0,
        eval_mode,
        collect_comparison_diagnostics,
    ):
        if self.fail_mpc:
            raise RuntimeError("synthetic MPC failure")
        self._record_modes()
        draw = (random.random(), float(np.random.random()), float(torch.rand(())))
        self.mpc_rng_draws.append(draw)
        self.mpc_calls.append(
            {
                "observation": observation.detach().clone(),
                "t0": t0,
                "eval_mode": eval_mode,
                "collect_comparison_diagnostics": collect_comparison_diagnostics,
                "prev_mean": self._prev_mean.detach().clone(),
            }
        )
        root = float(observation.reshape(-1)[0])
        self._prev_mean.fill_(root)
        metrics = {
            "inner_fixed_target_q_action_gain": root - 2.0,
            "planner_model_steps": 10.0 * root,
            "planner_seconds": 0.2 * root + 0.1,
            "planner_diagnostic_seconds": 0.1 * root,
        }
        if self.invalid_metric == "missing_q":
            metrics.pop("inner_fixed_target_q_action_gain")
        elif self.invalid_metric == "nonfinite_q":
            metrics["inner_fixed_target_q_action_gain"] = float("nan")
        elif self.invalid_metric == "negative_cost":
            metrics["planner_model_steps"] = -1.0
        elif self.invalid_metric == "nonfinite_timing":
            metrics["planner_seconds"] = float("inf")
        self.last_plan_metrics = metrics
        return torch.tensor([1.0])


def _observation_to_tensor(observation):
    return torch.as_tensor(observation, dtype=torch.float32)


def _evaluator(*, agent=None, factory=None, **overrides):
    arguments = {
        "agent": _FakeAgent() if agent is None else agent,
        "env_factory": _Factory() if factory is None else factory,
        "observation_to_tensor": _observation_to_tensor,
        "unscale_action": lambda action: action,
        "episodes": 2,
        "seed": 12345,
        "device": "cpu",
    }
    arguments.update(overrides)
    return TDMPC2PairedControllerEvaluator(**arguments)


def _assert_numpy_rng_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def _state_snapshot(agent):
    return {
        "previous_mean": agent._prev_mean.detach().clone(),
        "metrics": agent.last_plan_metrics,
        "boundary": agent._resume_boundary_prepared,
        "modes": tuple(module.training for module in agent.modules()),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state().clone(),
        "model": {
            key: value.detach().clone() for key, value in agent.state_dict().items()
        },
    }


def _assert_state_restored(agent, snapshot):
    torch.testing.assert_close(agent._prev_mean, snapshot["previous_mean"], rtol=0, atol=0)
    assert agent.last_plan_metrics is snapshot["metrics"]
    assert agent._resume_boundary_prepared is snapshot["boundary"]
    assert tuple(module.training for module in agent.modules()) == snapshot["modes"]
    assert random.getstate() == snapshot["python"]
    _assert_numpy_rng_equal(np.random.get_state(), snapshot["numpy"])
    torch.testing.assert_close(
        torch.random.get_rng_state(), snapshot["torch"], rtol=0, atol=0
    )
    for key, expected in snapshot["model"].items():
        torch.testing.assert_close(agent.state_dict()[key], expected, rtol=0, atol=0)


def test_baseline_paired_evaluation_exact_metrics_routing_and_restoration():
    factory = _Factory()
    agent = _FakeAgent()
    evaluator = _evaluator(agent=agent, factory=factory)
    assert factory.calls == 0

    agent.train(True)
    agent.model.train(False)
    agent.model[1][1].train(True)
    random.seed(1001)
    np.random.seed(1002)
    torch.manual_seed(1003)
    snapshot = _state_snapshot(agent)

    metrics = evaluator.evaluate()

    expected_keys = {
        "eval/paired_outer_episode_reward",
        "eval/paired_outer_episode_reward_std",
        "eval/paired_fresh_inner_episode_reward",
        "eval/paired_fresh_inner_episode_reward_std",
        "eval/paired_fresh_inner_minus_outer",
        "eval/paired_fresh_inner_minus_outer_std",
        "eval/paired_fresh_inner_win_fraction",
        "eval/paired_episodes",
        _Q_STEM,
        f"{_Q_STEM}_count",
        f"{_Q_STEM}_mean",
        f"{_Q_STEM}_std",
        f"{_Q_STEM}_min",
        f"{_Q_STEM}_p05",
        f"{_Q_STEM}_p25",
        f"{_Q_STEM}_p50",
        f"{_Q_STEM}_p75",
        f"{_Q_STEM}_p95",
        f"{_Q_STEM}_max",
        f"{_Q_STEM}_positive_fraction",
        "eval/paired_fresh_inner_model_steps_per_action",
        "time/paired_fresh_inner_control_seconds_per_action",
        "time/paired_fresh_inner_diagnostic_seconds_per_action",
        "time/paired_inner_comparison_seconds",
    }
    assert set(metrics) == expected_keys
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["eval/paired_outer_episode_reward"] == pytest.approx(6.0)
    assert metrics["eval/paired_outer_episode_reward_std"] == 0.0
    assert metrics["eval/paired_fresh_inner_episode_reward"] == pytest.approx(9.0)
    assert metrics["eval/paired_fresh_inner_episode_reward_std"] == 0.0
    assert metrics["eval/paired_fresh_inner_minus_outer"] == pytest.approx(3.0)
    assert metrics["eval/paired_fresh_inner_minus_outer_std"] == 0.0
    assert metrics["eval/paired_fresh_inner_win_fraction"] == 1.0
    assert metrics["eval/paired_episodes"] == 2.0

    # Six roots contribute [-1, 0, 1, -1, 0, 1]. Population moments and
    # quantiles deliberately treat zero as non-positive.
    assert metrics[_Q_STEM] == pytest.approx(0.0)
    assert metrics[f"{_Q_STEM}_count"] == 6.0
    assert metrics[f"{_Q_STEM}_mean"] == pytest.approx(0.0)
    assert metrics[f"{_Q_STEM}_std"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert metrics[f"{_Q_STEM}_min"] == -1.0
    assert metrics[f"{_Q_STEM}_p05"] == -1.0
    assert metrics[f"{_Q_STEM}_p25"] == pytest.approx(-0.75)
    assert metrics[f"{_Q_STEM}_p50"] == 0.0
    assert metrics[f"{_Q_STEM}_p75"] == pytest.approx(0.75)
    assert metrics[f"{_Q_STEM}_p95"] == 1.0
    assert metrics[f"{_Q_STEM}_max"] == 1.0
    assert metrics[f"{_Q_STEM}_positive_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["eval/paired_fresh_inner_model_steps_per_action"] == 20.0
    assert metrics[
        "time/paired_fresh_inner_control_seconds_per_action"
    ] == pytest.approx(0.3)
    assert metrics[
        "time/paired_fresh_inner_diagnostic_seconds_per_action"
    ] == pytest.approx(0.2)
    assert metrics["time/paired_inner_comparison_seconds"] >= 0.0

    assert factory.calls == 2
    outer_env, inner_env = factory.envs
    assert outer_env is not inner_env
    assert outer_env.reset_seeds == inner_env.reset_seeds
    expected_reset_seeds = [
        _namespaced_seed(
            12345,
            "paired_controller",
            "environment_reset",
            episode,
        )
        & 0xFFFFFFFF
        for episode in range(2)
    ]
    assert outer_env.reset_seeds == expected_reset_seeds
    assert [
        evaluator._seed("environment_reset", episode) for episode in range(2)
    ] == [
        PairedControllerEvaluator._seed(
            evaluator, "environment_reset", episode
        )
        for episode in range(2)
    ]
    assert len(set(outer_env.reset_seeds)) == 2
    assert outer_env.actions == [0.0] * 6
    assert inner_env.actions == [1.0] * 6
    assert len(agent.policy_calls) == 6
    assert [call["t0"] for call in agent.mpc_calls] == [
        True,
        False,
        False,
        True,
        False,
        False,
    ]
    assert all(call["eval_mode"] for call in agent.mpc_calls)
    assert all(
        call["collect_comparison_diagnostics"] for call in agent.mpc_calls
    )
    assert torch.count_nonzero(agent.mpc_calls[0]["prev_mean"]) == 0
    assert torch.count_nonzero(agent.mpc_calls[3]["prev_mean"]) == 0
    assert agent.reset_calls == 2
    assert all(not any(modes) for modes in agent.call_modes)
    _assert_state_restored(agent, snapshot)

    evaluator.close()
    evaluator.close()
    assert outer_env.close_calls == 1
    assert inner_env.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        evaluator.evaluate()


def test_repeated_evaluation_restarts_same_environment_and_mpc_seed_banks():
    factory = _Factory()
    agent = _FakeAgent()
    evaluator = _evaluator(agent=agent, factory=factory)

    first = evaluator.evaluate()
    first_draws = list(agent.mpc_rng_draws)
    first_reset_seeds = [list(env.reset_seeds) for env in factory.envs]
    second = evaluator.evaluate()
    second_draws = agent.mpc_rng_draws[len(first_draws) :]
    second_reset_seeds = [
        env.reset_seeds[len(first_reset_seeds[index]) :]
        for index, env in enumerate(factory.envs)
    ]

    assert first_draws == second_draws
    assert first_draws[:3] != first_draws[3:]
    assert first_reset_seeds == second_reset_seeds
    for key in set(first) - {"time/paired_inner_comparison_seconds"}:
        assert second[key] == pytest.approx(first[key])
    evaluator.close()


@pytest.mark.parametrize("boundary", ["terminated", "truncated"])
def test_both_gymnasium_boundaries_stop_each_baseline_controller(boundary):
    factory = _Factory(boundary=boundary)
    evaluator = _evaluator(factory=factory, episodes=1)

    metrics = evaluator.evaluate()

    assert metrics["eval/paired_outer_episode_reward"] == pytest.approx(6.0)
    assert metrics["eval/paired_fresh_inner_episode_reward"] == pytest.approx(9.0)
    assert [len(env.actions) for env in factory.envs] == [3, 3]
    evaluator.close()


def test_paired_return_delta_population_stats_and_strict_win_rate():
    factory = _PlannedReturnFactory()
    evaluator = _evaluator(factory=factory, episodes=3)

    metrics = evaluator.evaluate()

    # Outer=[1,4,7], MPC=[3,4,5], deltas=[2,0,-2].
    assert metrics["eval/paired_outer_episode_reward"] == pytest.approx(4.0)
    assert metrics["eval/paired_outer_episode_reward_std"] == pytest.approx(
        np.sqrt(6.0)
    )
    assert metrics["eval/paired_fresh_inner_episode_reward"] == pytest.approx(4.0)
    assert metrics["eval/paired_fresh_inner_episode_reward_std"] == pytest.approx(
        np.sqrt(2.0 / 3.0)
    )
    assert metrics["eval/paired_fresh_inner_minus_outer"] == pytest.approx(0.0)
    assert metrics["eval/paired_fresh_inner_minus_outer_std"] == pytest.approx(
        np.sqrt(8.0 / 3.0)
    )
    assert metrics["eval/paired_fresh_inner_win_fraction"] == pytest.approx(1.0 / 3.0)
    assert factory.envs[0].reset_seeds == factory.envs[1].reset_seeds
    evaluator.close()


@pytest.mark.parametrize(
    ("invalid_metric", "factory", "message"),
    [
        ("missing_q", _Factory(), "inner_fixed_target_q_action_gain"),
        ("nonfinite_q", _Factory(), "must be finite"),
        ("negative_cost", _Factory(), "must be non-negative"),
        ("nonfinite_timing", _Factory(), "must be finite"),
        (None, _Factory(nonfinite_reward=True), "environment reward"),
    ],
)
def test_invalid_diagnostics_and_rewards_fail_closed_and_restore(
    invalid_metric, factory, message
):
    agent = _FakeAgent(invalid_metric=invalid_metric)
    evaluator = _evaluator(agent=agent, factory=factory, episodes=1)
    snapshot = _state_snapshot(agent)

    with pytest.raises((KeyError, TypeError, ValueError), match=message):
        evaluator.evaluate()

    _assert_state_restored(agent, snapshot)
    evaluator.close()


@pytest.mark.parametrize("failure", ["environment", "controller"])
def test_runtime_failure_restores_planner_telemetry_modes_and_rng(failure):
    factory = _Factory(fail_inner=failure == "environment")
    agent = _FakeAgent(fail_mpc=failure == "controller")
    agent.model.train(False)
    agent.model[1][1].train(True)
    evaluator = _evaluator(agent=agent, factory=factory, episodes=1)
    random.seed(2001)
    np.random.seed(2002)
    torch.manual_seed(2003)
    snapshot = _state_snapshot(agent)

    with pytest.raises(RuntimeError, match="synthetic"):
        evaluator.evaluate()

    _assert_state_restored(agent, snapshot)
    evaluator.close()
    assert [env.close_calls for env in factory.envs] == [1, 1]


def test_pairing_requires_independent_envs_and_identical_initial_observations():
    shared = _Factory(shared=True)
    shared_evaluator = _evaluator(factory=shared, episodes=1)
    with pytest.raises(ValueError, match="independent"):
        shared_evaluator.evaluate()
    shared_evaluator.close()
    assert shared.envs[0].close_calls == 1

    mismatch = _Factory(mismatch_inner=True)
    mismatch_evaluator = _evaluator(factory=mismatch, episodes=1)
    with pytest.raises(ValueError, match="different initial observations"):
        mismatch_evaluator.evaluate()
    assert mismatch_evaluator.agent.policy_calls == []
    assert mismatch_evaluator.agent.mpc_calls == []
    mismatch_evaluator.close()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"episodes": 0}, "episodes"),
        ({"episodes": True}, "episodes"),
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"env_factory": None}, "env_factory"),
        ({"observation_to_tensor": None}, "observation_to_tensor"),
        ({"unscale_action": None}, "unscale_action"),
        ({"device": "mps"}, "CPU and CUDA"),
    ],
)
def test_constructor_rejects_invalid_controls(overrides, message):
    with pytest.raises((TypeError, ValueError, NotImplementedError), match=message):
        _evaluator(**overrides)


def test_close_before_first_evaluation_preserves_lazy_construction():
    factory = _Factory()
    evaluator = _evaluator(factory=factory)

    evaluator.close()
    evaluator.close()

    assert factory.calls == 0
    with pytest.raises(RuntimeError, match="closed"):
        evaluator.evaluate()
