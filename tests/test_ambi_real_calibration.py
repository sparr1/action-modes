import json
import os
from dataclasses import FrozenInstanceError

import gymnasium as gym
import numpy as np
import pytest

from utils.ambi_real_calibration import (
    SimulatorSnapshot, capture_simulator_snapshot, enable_continuing_calibration,
    evaluate_real_branches, omitted_tail_bound, restore_simulator_snapshot,
    terminal_q_reductions,
)


class AnalyticEnv(gym.Env):
    """Reward equals the executed action; observation reveals real state."""

    def __init__(self, terminal_at=None, truncated_at=None):
        self.action_space = gym.spaces.Box(-100, 100, (1,), dtype=np.float64)
        self.observation_space = gym.spaces.Box(-100, 100, (1,), dtype=np.float64)
        self.position = 0
        self.actions = []
        self.terminal_at = terminal_at
        self.truncated_at = truncated_at
        self.raw_limit = 5

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0
        self.actions = []
        return np.array([0.0]), {}

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.position += 1
        return (np.array([float(self.position)]), float(action[0]),
                self.position == self.terminal_at,
                self.position == self.truncated_at, {})

    def calibration_state(self):
        return {"position": self.position, "raw_limit": self.raw_limit}

    def load_calibration_state(self, state):
        self.position = state["position"]
        self.raw_limit = state["raw_limit"]
        self.actions = []
        return np.array([float(self.position)])

    def enable_continuing_calibration(self):
        self.raw_limit = None


def _snapshot():
    env = AnalyticEnv()
    env.reset(seed=10)
    return capture_simulator_snapshot(env)


def test_snapshot_json_is_immutable_and_preserves_typed_rng_data():
    state = {"array": np.arange(6, dtype=np.float32).reshape(2, 3),
             "rng": np.random.RandomState(7).get_state()}
    snapshot = SimulatorSnapshot.capture(state)
    state["array"][0, 0] = 100
    restored = SimulatorSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))
    assert restored.sha256 == snapshot.sha256
    assert restored.state()["array"][0, 0] == 0
    assert restored.state()["array"].dtype == np.float32
    rng = np.random.RandomState()
    rng.set_state(restored.state()["rng"])
    np.testing.assert_array_equal(rng.randn(4), np.random.RandomState(7).randn(4))
    decoded = restored.state()
    decoded["array"][0, 0] = 99
    assert restored.state()["array"][0, 0] == 0
    with pytest.raises(FrozenInstanceError):
        restored.payload = b"changed"
    corrupted = snapshot.to_dict()
    corrupted["state"]["extra"] = True
    with pytest.raises(ValueError, match="checksum"):
        SimulatorSnapshot.from_dict(corrupted)


def test_restore_preserves_wrapper_bookkeeping_and_disables_both_clocks_only_on_branch():
    source = gym.wrappers.TimeLimit(gym.wrappers.OrderEnforcing(AnalyticEnv()), 5)
    source.reset(seed=1)
    source.step(np.array([1.0]))
    snapshot = capture_simulator_snapshot(source)
    branch = gym.wrappers.TimeLimit(gym.wrappers.OrderEnforcing(AnalyticEnv()), 5)
    obs = restore_simulator_snapshot(branch, snapshot, continuing=True)
    np.testing.assert_array_equal(obs, [1.0])
    assert branch._elapsed_steps == 1
    assert branch.env._has_reset
    assert branch._max_episode_steps == float("inf")
    assert branch.unwrapped.raw_limit is None
    assert source._max_episode_steps == 5
    assert source.unwrapped.raw_limit == 5
    for _ in range(10):
        assert not branch.step(np.array([1.0]))[3]
    restore_simulator_snapshot(branch, snapshot)
    assert branch._max_episode_steps == 5
    assert branch.unwrapped.raw_limit == 5


def test_snapshot_rejects_unreviewed_wrappers():
    class UnknownWrapper(gym.Wrapper):
        pass

    with pytest.raises(ValueError, match="UnknownWrapper"):
        capture_simulator_snapshot(UnknownWrapper(AnalyticEnv()))
    source = gym.wrappers.TimeLimit(AnalyticEnv(), 5)
    source.reset()
    with pytest.raises(ValueError, match="wrapper identity"):
        restore_simulator_snapshot(AnalyticEnv(), capture_simulator_snapshot(source))


def test_analytic_discounting_exact_tail_action_and_batched_observation_encoding():
    envs = [AnalyticEnv(), AnalyticEnv()]
    prefix_noise = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])
    tail_noise = np.array([[[5.0], [6.0]], [[7.0], [8.0]], [[9.0], [10.0]]])
    observed = []
    q_actions = []

    def actor(obs, noise):
        observed.append(obs.copy())
        return noise + obs

    def q(obs, actions):
        np.testing.assert_array_equal(obs, [[2.0], [2.0]])
        q_actions.append(actions.copy())
        return 10 * actions[:, 0]

    result = evaluate_real_branches(
        envs, _snapshot(), actor, actor, q, prefix_noise, tail_noise,
        horizon=2, discount=0.5, original_remaining_steps=3,
    )
    assert len(observed) == 5
    assert all(obs.shape == (2, 1) for obs in observed)
    np.testing.assert_array_equal(q_actions[0], np.stack([env.actions[2] for env in envs]))
    for index, row in enumerate(result["rows"]):
        actions = np.array(envs[index].actions)[:, 0]
        prefix = actions[0] + 0.5 * actions[1]
        tail = actions[2] + 0.5 * actions[3] + 0.25 * actions[4]
        assert row["real_prefix_reward"] == prefix
        assert row["real_bootstrap"] == 0.25 * 10 * actions[2]
        assert row["real_bootstrapped_return"] == prefix + row["real_bootstrap"]
        assert row["real_tail_return"] == tail
        assert row["real_mc_return"] == prefix + 0.25 * tail
        assert row["episode_cutoff_discounted_return"] == prefix + 0.25 * actions[2]
        assert row["episode_cutoff_undiscounted_return"] == sum(actions[:3])
        assert row["episode_cutoff_complete"]
        assert row["mc_complete"]
        assert row["bootstrap_prediction_error"] == row["real_bootstrap"] - 0.25 * tail
    assert result["work"] == {"simulator_decisions": 10, "policy_rows": 10, "q_rows": 2}
    assert all(value >= 0 for value in result["timing"].values())


def test_termination_masks_bootstrap_and_retains_rollout_noise_indices():
    envs = [AnalyticEnv(terminal_at=1), AnalyticEnv(terminal_at=3), AnalyticEnv()]
    noise = np.arange(6, dtype=float).reshape(2, 3, 1) + 1
    tail = np.arange(6, dtype=float).reshape(2, 3, 1) + 10
    batches = []

    def actor(obs, epsilon):
        batches.append(epsilon.copy())
        return epsilon

    result = evaluate_real_branches(
        envs, _snapshot(), actor, actor, lambda obs, actions: np.full(len(obs), 50),
        noise, tail, horizon=2, discount=0.5, original_remaining_steps=5,
    )
    first, second, third = result["rows"]
    assert first["real_mc_return"] == 1
    assert first["real_bootstrap"] == 0
    assert first["endpoint_action"] is None
    assert first["return_truncation_bound"] == 0
    assert first["episode_cutoff_complete"]
    assert second["real_mc_return"] == 2 + 0.5 * 5 + 0.25 * 11
    assert second["real_bootstrap"] == 0.25 * 50
    assert second["return_truncation_bound"] == 0
    np.testing.assert_array_equal(batches[-1], [[15]])
    assert not third["episode_cutoff_complete"]


def test_unexpected_truncation_marks_incomplete_measurement():
    result = evaluate_real_branches(
        [AnalyticEnv(truncated_at=1)], _snapshot(), lambda obs, e: e,
        lambda obs, e: e, lambda obs, a: np.zeros(len(obs)),
        np.ones((2, 1, 1)), np.ones((2, 1, 1)),
        horizon=2, discount=0.99, original_remaining_steps=5,
    )["rows"][0]
    assert result["truncated"] and not result["mc_complete"]
    assert not result["episode_cutoff_complete"]
    assert result["return_truncation_bound"] is None


def test_no_hidden_mc_bootstrap_and_reference_tail_bound():
    result = evaluate_real_branches(
        [AnalyticEnv()], _snapshot(), lambda obs, e: np.zeros_like(e),
        lambda obs, e: np.zeros_like(e), lambda obs, a: np.full(len(obs), 999),
        np.ones((2, 1, 1)), np.ones((3, 1, 1)),
        horizon=2, discount=0.99, original_remaining_steps=2,
    )["rows"][0]
    assert result["real_mc_return"] == 0
    assert result["real_bootstrapped_return"] == 0.99 ** 2 * 999
    assert omitted_tail_bound(0.99, 1000) == pytest.approx(0.008634249482131572)
    with pytest.raises(ValueError):
        omitted_tail_bound(1.0, 1000)


def test_terminal_head_reductions_match_all_distinct_pair_expectations():
    heads = np.array([[1.0, 2.0, 4.0, 8.0, 16.0], [-5.0, 0.0, 9.0, 2.0, 8.0]])
    result = terminal_q_reductions(heads)
    pair_means = []
    pair_mins = []
    for first in range(heads.shape[1]):
        for second in range(heads.shape[1]):
            if first != second:
                pair_means.append((heads[:, first] + heads[:, second]) / 2)
                pair_mins.append(np.minimum(heads[:, first], heads[:, second]))
    np.testing.assert_allclose(result["expected_mean_pair"], np.mean(pair_means, axis=0))
    np.testing.assert_allclose(result["expected_min_pair"], np.mean(pair_mins, axis=0))
    np.testing.assert_array_equal(result["min_all"], [1.0, -5.0])
    for invalid in (np.array([1, 2]), np.zeros((2, 1)), np.array([[np.nan, 1.0]])):
        with pytest.raises(ValueError, match="Terminal Q heads"):
            terminal_q_reductions(invalid)


def test_optional_heads_use_exact_tail_action_and_do_not_change_default_measurements():
    prefix_noise = np.array([[[1.0], [2.0]], [[3.0], [4.0]]])
    tail_noise = np.array([[[5.0], [6.0]], [[7.0], [8.0]]])
    arguments = dict(horizon=2, discount=0.5, original_remaining_steps=4)
    actor = lambda obs, noise: obs + noise
    q = lambda obs, actions: actions[:, 0] * 10
    plain = evaluate_real_branches(
        [AnalyticEnv(), AnalyticEnv()], _snapshot(), actor, actor, q,
        prefix_noise, tail_noise, **arguments,
    )
    envs = [AnalyticEnv(), AnalyticEnv()]
    calls = []

    def q_heads(obs, actions):
        calls.append((obs.copy(), actions.copy()))
        return actions + np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])

    with_heads = evaluate_real_branches(
        envs, _snapshot(), actor, actor, q, prefix_noise, tail_noise,
        outer_q_heads=q_heads, **arguments,
    )
    np.testing.assert_array_equal(calls[0][1], [env.actions[2] for env in envs])
    for original, enhanced in zip(plain["rows"], with_heads["rows"]):
        assert original == {name: enhanced[name] for name in original}
        action = enhanced["endpoint_action"][0]
        assert enhanced["real_endpoint_q_heads"] == [action + value for value in range(5)]
        assert enhanced["real_endpoint_q_expected_mean_pair"] == action + 2
        assert enhanced["real_endpoint_q_expected_min_pair"] == action + 1
        assert enhanced["real_endpoint_q_min_all"] == action
        for reduction in ("expected_mean_pair", "expected_min_pair", "min_all"):
            contribution = 0.25 * enhanced[f"real_endpoint_q_{reduction}"]
            assert enhanced[f"real_bootstrap_{reduction}"] == contribution
            assert enhanced[f"real_bootstrapped_return_{reduction}"] == enhanced["real_prefix_reward"] + contribution
            assert enhanced[f"bootstrap_prediction_error_{reduction}"] == contribution - enhanced["real_tail_contribution"]


def test_optional_heads_are_masked_after_prefix_termination():
    def forbidden(*args):
        pytest.fail("A terminated branch must not query its terminal Q.")

    row = evaluate_real_branches(
        [AnalyticEnv(terminal_at=1)], _snapshot(), lambda obs, noise: noise,
        lambda obs, noise: noise, forbidden, np.ones((2, 1, 1)), np.ones((2, 1, 1)),
        horizon=2, discount=0.5, original_remaining_steps=4, outer_q_heads=forbidden,
    )["rows"][0]
    assert row["real_endpoint_q_heads"] is None
    for reduction in ("expected_mean_pair", "expected_min_pair", "min_all"):
        assert row[f"real_bootstrap_{reduction}"] == 0
        assert row[f"real_bootstrapped_return_{reduction}"] == 1


def test_mean_prefix_does_not_replace_sampled_prior_tail():
    env = AnalyticEnv()
    row = evaluate_real_branches(
        [env], _snapshot(), lambda obs, noise: np.full_like(noise, 2.0),
        lambda obs, noise: noise, lambda obs, actions: actions[:, 0],
        np.full((2, 1, 1), 999.0), np.array([[[7.0]], [[8.0]]]),
        horizon=2, discount=0.5, original_remaining_steps=4,
    )["rows"][0]
    np.testing.assert_array_equal(np.asarray(env.actions)[:, 0], [2.0, 2.0, 7.0, 8.0])
    assert row["real_endpoint_q"] == 7.0
    assert row["real_mc_return"] == 3.0 + 0.25 * 11.0


@pytest.mark.skipif(os.environ.get("AMBI_RUN_REAL_DMCONTROL_TESTS") != "1",
                    reason="Real MuJoCo integration is opt-in and host-dependent.")
@pytest.mark.parametrize("decision", [0, 499])
def test_real_humanoid_serialized_snapshot_roundtrip_and_continuing_cutoff(decision):
    import domains  # noqa: F401

    source = gym.make("DMControl-v0", task="humanoid-walk", obs="state")
    branch = gym.make("DMControl-v0", task="humanoid-walk", obs="state")
    try:
        source.reset(seed=101)
        rng = np.random.default_rng(7)
        for _ in range(decision):
            source.step(np.zeros(source.action_space.shape))
        before_task_rng = source.unwrapped._env.task.random.get_state()
        snapshot = capture_simulator_snapshot(source)
        snapshot = SimulatorSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))
        after_task_rng = source.unwrapped._env.task.random.get_state()
        assert before_task_rng[0] == after_task_rng[0]
        np.testing.assert_array_equal(before_task_rng[1], after_task_rng[1])
        assert before_task_rng[2:] == after_task_rng[2:]
        restore_simulator_snapshot(branch, snapshot, continuing=True)
        enable_continuing_calibration(source)
        actions = rng.uniform(-0.2, 0.2, (12, *source.action_space.shape))
        for action in actions:
            expected = source.step(action)
            actual = branch.step(action)
            np.testing.assert_array_equal(expected[0], actual[0])
            assert expected[1:] == actual[1:]
            assert not actual[2] and not actual[3]
        if decision == 499:
            assert branch.unwrapped._env._env._step_count > 1000
            assert not branch.unwrapped._env._env._reset_next_step
        restore_simulator_snapshot(branch, snapshot)
        assert branch._max_episode_steps == 500
        if decision == 499:
            assert branch.step(actions[0])[3]
    finally:
        source.close()
        branch.close()
