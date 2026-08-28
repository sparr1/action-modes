import random

import numpy as np
import pytest
import torch

from RL.tdmpc2_core import paired_controller_evaluation
from RL.tdmpc2_core.paired_controller_evaluation import PairedControllerEvaluator


_Q_STEM = "eval/paired_fresh_inner_fixed_target_q_action_gain"


class _EvaluationEngine:
    instances = []

    def __init__(self, agent):
        self.agent = agent
        self.reset_seeds = []
        # Constructor isolation is part of the evaluator's observational
        # contract, even if the production engine currently uses private RNGs.
        random.random()
        np.random.random()
        torch.rand(())
        self.instances.append(self)

    def reset_for_evaluation(self, seed):
        self.reset_seeds.append(seed)
        return self


class _ThreeStepEnv:
    def __init__(
        self,
        *,
        fail=False,
        nonfinite_reward=False,
        initial_offset=0.0,
        boundary="truncated",
    ):
        if boundary not in {"terminated", "truncated"}:
            raise ValueError("boundary must be terminated or truncated")
        self.fail = fail
        self.nonfinite_reward = nonfinite_reward
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
        self.fail_inner = fail_inner
        self.nonfinite_reward = nonfinite_reward
        self.shared = shared
        self.mismatch_inner = mismatch_inner
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
        self.actions = []
        self.close_calls = 0
        self._episode = -1

    def reset(self, *, seed=None):
        self.reset_seeds.append(seed)
        self._episode += 1
        return np.array([1.0], dtype=np.float32), {}

    def step(self, action):
        self.actions.append(float(np.asarray(action).reshape(-1)[0]))
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
    def __init__(self, *, invalid_metric=None):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, 2),
            torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(2, 1)),
        )
        self.device = torch.device("cpu")
        self.inner_engine = object()
        self.last_inner_metrics = {"training": 17.0}
        self.last_inner_rollout_lengths = [91]
        self.invalid_metric = invalid_metric
        self.outer_calls = []
        self.inner_calls = []
        self.call_modes = []

    @staticmethod
    def _consume_global_rng():
        random.random()
        np.random.random()
        torch.rand(())

    def _record_modes(self):
        self.call_modes.append(tuple(module.training for module in self.modules()))

    def act_outer_policy(self, observation, *, deterministic):
        self._consume_global_rng()
        self._record_modes()
        assert deterministic is True
        self.outer_calls.append(
            {
                "observation": observation.detach().clone(),
            }
        )
        self.last_inner_metrics = {}
        self.last_inner_rollout_lengths = []
        return torch.tensor([0.0])

    def act(
        self,
        observation,
        *,
        t0,
        eval_mode,
        collect_diagnostics,
        apply_inner_writeback,
    ):
        self._consume_global_rng()
        self._record_modes()
        assert isinstance(self.inner_engine, _EvaluationEngine)
        self.inner_calls.append(
            {
                "observation": observation.detach().clone(),
                "t0": t0,
                "eval_mode": eval_mode,
                "collect_diagnostics": collect_diagnostics,
                "apply_inner_writeback": apply_inner_writeback,
            }
        )
        root = float(observation.reshape(-1)[0])
        metrics = {
            "inner_fixed_target_q_action_gain": root - 2.0,
            "inner_model_steps": 10.0 * root,
            "inner_action_seconds": 0.2 * root + 0.1,
            "inner_diagnostic_seconds": 0.1 * root,
        }
        if self.invalid_metric == "missing_q":
            metrics.pop("inner_fixed_target_q_action_gain")
        elif self.invalid_metric == "nonfinite_q":
            metrics["inner_fixed_target_q_action_gain"] = float("nan")
        elif self.invalid_metric == "negative_cost":
            metrics["inner_model_steps"] = -1.0
        elif self.invalid_metric == "nonfinite_timing":
            metrics["inner_action_seconds"] = float("inf")
        self.last_inner_metrics = metrics
        self.last_inner_rollout_lengths = [int(root)]
        return torch.tensor([1.0])


def _observation_to_tensor(observation):
    return torch.as_tensor(observation, dtype=torch.float32)


def _evaluator(monkeypatch, *, agent=None, factory=None, **overrides):
    _EvaluationEngine.instances.clear()
    monkeypatch.setattr(
        paired_controller_evaluation,
        "InnerImprovementEngine",
        _EvaluationEngine,
    )
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
    return PairedControllerEvaluator(**arguments)


def _assert_numpy_rng_equal(left, right):
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_paired_evaluation_uses_isolated_seeded_controllers_and_exact_metrics(
    monkeypatch,
):
    factory = _Factory()
    agent = _FakeAgent()

    random.seed(1001)
    np.random.seed(1002)
    torch.manual_seed(1003)
    python_before_constructor = random.getstate()
    numpy_before_constructor = np.random.get_state()
    torch_before_constructor = torch.random.get_rng_state().clone()
    evaluator = _evaluator(monkeypatch, agent=agent, factory=factory)

    assert factory.calls == 0
    assert random.getstate() == python_before_constructor
    _assert_numpy_rng_equal(np.random.get_state(), numpy_before_constructor)
    torch.testing.assert_close(
        torch.random.get_rng_state(), torch_before_constructor, rtol=0, atol=0
    )

    # Deliberately install inconsistent nested modes.  Restoration must retain
    # every module's individual mode rather than only the root flag.
    agent.train(True)
    agent.model.train(False)
    agent.model[1][1].train(True)
    modes_before = tuple(module.training for module in agent.modules())
    live_engine = agent.inner_engine
    telemetry = agent.last_inner_metrics
    rollout_lengths = agent.last_inner_rollout_lengths
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

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

    # Six roots contribute [-1, 0, 1, -1, 0, 1].  Zero is deliberately not a
    # positive gain, matching the strict win convention.
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
    assert len(set(outer_env.reset_seeds)) == 2
    assert outer_env.actions == [0.0] * 6
    assert inner_env.actions == [1.0] * 6
    assert [call["t0"] for call in agent.inner_calls] == [
        True,
        False,
        False,
        True,
        False,
        False,
    ]
    assert all(call["eval_mode"] for call in agent.inner_calls)
    assert all(call["collect_diagnostics"] for call in agent.inner_calls)
    assert all(not call["apply_inner_writeback"] for call in agent.inner_calls)
    assert all(not any(modes) for modes in agent.call_modes)
    assert len(_EvaluationEngine.instances) == 1
    evaluation_engine = _EvaluationEngine.instances[0]
    assert len(evaluation_engine.reset_seeds) == 2
    assert len(set(evaluation_engine.reset_seeds)) == 2
    assert set(outer_env.reset_seeds).isdisjoint(evaluation_engine.reset_seeds)

    assert agent.inner_engine is live_engine
    assert agent.last_inner_metrics is telemetry
    assert agent.last_inner_rollout_lengths is rollout_lengths
    assert tuple(module.training for module in agent.modules()) == modes_before
    assert random.getstate() == python_state
    _assert_numpy_rng_equal(np.random.get_state(), numpy_state)
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)

    evaluator.close()
    evaluator.close()
    assert outer_env.close_calls == 1
    assert inner_env.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        evaluator.evaluate()


def test_repeated_evaluation_restarts_the_same_private_episode_seed_bank(monkeypatch):
    factory = _Factory()
    agent = _FakeAgent()
    evaluator = _evaluator(monkeypatch, agent=agent, factory=factory)

    first = evaluator.evaluate()
    engine = _EvaluationEngine.instances[0]
    first_engine_seeds = list(engine.reset_seeds)
    first_reset_seeds = [list(env.reset_seeds) for env in factory.envs]

    agent.outer_calls.clear()
    second = evaluator.evaluate()
    second_engine_seeds = engine.reset_seeds[len(first_engine_seeds) :]
    second_reset_seeds = [
        env.reset_seeds[len(first_reset_seeds[index]) :]
        for index, env in enumerate(factory.envs)
    ]

    assert len(_EvaluationEngine.instances) == 1
    assert first_engine_seeds == second_engine_seeds
    assert first_reset_seeds == second_reset_seeds
    for key in set(first) - {"time/paired_inner_comparison_seconds"}:
        assert second[key] == pytest.approx(first[key])
    evaluator.close()


@pytest.mark.parametrize("boundary", ["terminated", "truncated"])
def test_both_gymnasium_episode_boundaries_stop_each_controller(monkeypatch, boundary):
    factory = _Factory(boundary=boundary)
    evaluator = _evaluator(monkeypatch, factory=factory, episodes=1)

    metrics = evaluator.evaluate()

    assert metrics["eval/paired_outer_episode_reward"] == pytest.approx(6.0)
    assert metrics["eval/paired_fresh_inner_episode_reward"] == pytest.approx(9.0)
    assert [len(env.actions) for env in factory.envs] == [3, 3]
    evaluator.close()


def test_paired_episode_deltas_use_population_stats_and_strict_mixed_win_rate(
    monkeypatch,
):
    factory = _PlannedReturnFactory()
    evaluator = _evaluator(monkeypatch, factory=factory, episodes=3)

    metrics = evaluator.evaluate()

    # Outer=[1,4,7], inner=[3,4,5], and paired deltas=[2,0,-2].  The tie is
    # deliberately not a win, and each standard deviation uses ddof=0.
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
    assert metrics["eval/paired_episodes"] == 3.0
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
def test_invalid_diagnostics_and_environment_rewards_fail_closed_and_restore(
    monkeypatch,
    invalid_metric,
    factory,
    message,
):
    agent = _FakeAgent(invalid_metric=invalid_metric)
    evaluator = _evaluator(monkeypatch, agent=agent, factory=factory, episodes=1)
    live_engine = agent.inner_engine
    telemetry = agent.last_inner_metrics
    lengths = agent.last_inner_rollout_lengths
    modes = tuple(module.training for module in agent.modules())
    torch_state = torch.random.get_rng_state().clone()

    with pytest.raises((KeyError, TypeError, ValueError), match=message):
        evaluator.evaluate()

    assert agent.inner_engine is live_engine
    assert agent.last_inner_metrics is telemetry
    assert agent.last_inner_rollout_lengths is lengths
    assert tuple(module.training for module in agent.modules()) == modes
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)
    evaluator.close()


def test_environment_failure_restores_live_engine_telemetry_modes_and_rng(monkeypatch):
    factory = _Factory(fail_inner=True)
    agent = _FakeAgent()
    evaluator = _evaluator(monkeypatch, agent=agent, factory=factory, episodes=1)
    agent.model.train(False)
    agent.model[1][1].train(True)
    live_engine = agent.inner_engine
    telemetry = agent.last_inner_metrics
    lengths = agent.last_inner_rollout_lengths
    modes = tuple(module.training for module in agent.modules())
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="synthetic paired environment failure"):
        evaluator.evaluate()

    assert agent.inner_engine is live_engine
    assert agent.last_inner_metrics is telemetry
    assert agent.last_inner_rollout_lengths is lengths
    assert tuple(module.training for module in agent.modules()) == modes
    assert random.getstate() == python_state
    _assert_numpy_rng_equal(np.random.get_state(), numpy_state)
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)
    evaluator.close()
    assert [env.close_calls for env in factory.envs] == [1, 1]


def test_factory_must_create_two_independent_environments(monkeypatch):
    factory = _Factory(shared=True)
    evaluator = _evaluator(monkeypatch, factory=factory, episodes=1)

    with pytest.raises(ValueError, match="independent"):
        evaluator.evaluate()

    evaluator.close()
    assert factory.envs[0].close_calls == 1


def test_paired_initial_observations_must_match_exactly(monkeypatch):
    factory = _Factory(mismatch_inner=True)
    evaluator = _evaluator(monkeypatch, factory=factory, episodes=1)

    with pytest.raises(ValueError, match="different initial observations"):
        evaluator.evaluate()

    # Neither controller is allowed to act after a failed pairing check.
    assert evaluator.agent.outer_calls == []
    assert evaluator.agent.inner_calls == []
    evaluator.close()


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
    ],
)
def test_constructor_rejects_invalid_controls(monkeypatch, overrides, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _evaluator(monkeypatch, **overrides)


def test_close_before_first_evaluation_keeps_environments_lazy(monkeypatch):
    factory = _Factory()
    evaluator = _evaluator(monkeypatch, factory=factory)

    evaluator.close()
    evaluator.close()

    assert factory.calls == 0
    with pytest.raises(RuntimeError, match="closed"):
        evaluator.evaluate()
