from copy import deepcopy
import os
import random
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from utils.transfer_diagnostic_real import (
    PolicyOutput, SimulatorSnapshot, capture_simulator_snapshot,
    evaluate_continuation, evaluate_real_prefix, preserve_simulator,
    restore_simulator_snapshot,
)


class Accumulator(gym.Env):
    """Analytic reward equals action; stochastic state exercises RNG restore."""

    def __init__(self, terminate_at=None):
        self.action_space = gym.spaces.Box(-10., 10., (1,), dtype=np.float64)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float64)
        self.terminate_at = terminate_at
        self.reset(seed=8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position, self.steps, self.continuing = 0., 0, False
        return np.array([self.position]), {}

    def step(self, action):
        self.position += float(action[0]) + self.np_random.normal()
        self.steps += 1
        terminal = self.terminate_at is not None and self.steps >= self.terminate_at
        return np.array([self.position]), float(action[0]), terminal, False, {}

    def calibration_state(self):
        return dict(position=self.position, steps=self.steps, continuing=self.continuing,
                    rng=deepcopy(self.np_random.bit_generator.state))

    def load_calibration_state(self, state):
        self.position, self.steps, self.continuing = state["position"], state["steps"], state["continuing"]
        self.np_random.bit_generator.state = deepcopy(state["rng"])
        return np.array([self.position])

    def enable_continuing_calibration(self):
        self.continuing = True


def constant(action, log_prob=-2.):
    def actor(observations, noise, remaining_horizon):
        return PolicyOutput(np.full((len(observations), 1), action), np.full(len(observations), log_prob))
    return actor


def evaluate(env, *, horizon=3, tail_steps=2, **kwargs):
    return evaluate_real_prefix(
        env, capture_simulator_snapshot(env), first_action=np.array([1.]),
        prefix_actor=constant(2.), prior_actor=constant(3.),
        terminal_q=lambda obs, actions: np.array([10.]),
        prefix_noise=np.zeros((horizon, 1)), tail_noise=np.zeros((tail_steps, 1)),
        discount=.5, **kwargs)


@pytest.mark.parametrize("horizon", [1, 2, 3, 5, 8])
def test_arbitrary_horizon_reward_decomposition_and_noninterference(horizon):
    env = gym.wrappers.TimeLimit(Accumulator(), max_episode_steps=2)
    env.reset(seed=9)
    root = capture_simulator_snapshot(env)
    result = evaluate(env, horizon=horizon)
    prefix = 1. + sum(.5 ** step * 2. for step in range(1, horizon))
    tail = 3. + .5 * 3.
    assert result["real_prefix_reward"] == prefix
    assert result["real_bootstrapped_return"] == prefix + .5 ** horizon * 10.
    assert result["real_mc_return"] == prefix + .5 ** horizon * tail
    assert result["terminal_prediction_error"] == .5 ** horizon * (10. - tail)
    assert result["endpoint_action"] == [3.]
    assert capture_simulator_snapshot(env).sha256 == root.sha256
    assert env._max_episode_steps == 2
    assert result["prefix_decisions"] == horizon


def test_soft_prefix_and_soft_boundary_exclude_forced_and_boundary_action_entropy():
    result = evaluate(Accumulator(), horizon=3, tail_steps=3,
                      objective="soft", alpha=.25, terminal_objective="soft", terminal_alpha=.5)
    assert result["real_prefix_entropy"] == .25 * 2. * (.5 + .25)
    assert result["real_tail_entropy"] == .5 * 2. * (.5 + .25)
    assert result["real_mc_return"] == (
        result["real_prefix_reward"] + result["real_prefix_entropy"]
        + .5 ** 3 * (result["real_tail_reward"] + result["real_tail_entropy"]))
    assert not result["first_action_entropy_included"]
    assert not result["boundary_action_entropy_included"]


def test_true_termination_masks_bootstrap_and_later_rewards():
    result = evaluate(Accumulator(terminate_at=2), horizon=5)
    assert result["real_bootstrapped_return"] == 2.
    assert result["real_mc_return"] == 2.
    assert result["real_bootstrap"] == 0.
    assert result["tail_decisions"] == 0
    assert result["prefix_complete"] and result["mc_complete"] and result["terminated"]


def test_outer_boundary_entropy_is_shared_by_bootstrap_and_real_tail():
    kwargs = dict(horizon=2, tail_steps=3, objective="soft", alpha=.25,
                  terminal_objective="soft", terminal_alpha=.5)
    without = evaluate(Accumulator(), **kwargs)
    with_boundary = evaluate(Accumulator(), terminal_first_action_entropy=True, **kwargs)
    contribution = .5 ** 2 * .5 * 2.
    assert with_boundary["real_endpoint_entropy"] == 1.
    assert with_boundary["real_bootstrapped_return"] - without["real_bootstrapped_return"] == contribution
    assert with_boundary["real_mc_return"] - without["real_mc_return"] == contribution
    assert with_boundary["terminal_prediction_error"] == without["terminal_prediction_error"]
    assert with_boundary["boundary_action_entropy_included"]


def test_tail_termination_keeps_endpoint_value_but_stops_mc_tail():
    result = evaluate(Accumulator(terminate_at=3), horizon=2, tail_steps=5)
    assert result["real_mc_return"] == 2. + .25 * 3.
    assert result["real_bootstrapped_return"] == 2. + .25 * 10.
    assert result["tail_decisions"] == 1


def test_time_limit_is_explicit_and_not_bootstrapped_or_reset():
    env = gym.wrappers.TimeLimit(Accumulator(), max_episode_steps=2)
    env.reset(seed=2)
    result = evaluate(env, horizon=4, continuing=False)
    assert result["truncated"] and not result["prefix_complete"] and not result["mc_complete"]
    assert result["real_bootstrapped_return"] is None
    assert result["real_mc_return"] is None
    assert result["real_mc_partial_return"] == 2.
    assert result["prefix_decisions"] == 2
    assert env._elapsed_steps == 0


def test_paired_noise_indexing_and_remaining_horizon():
    env = Accumulator()
    prefix_calls, tail_calls = [], []
    def prefix(obs, noise, remaining):
        prefix_calls.append((noise.copy(), remaining))
        return PolicyOutput(noise)
    def prior(obs, noise, remaining):
        tail_calls.append((noise.copy(), remaining))
        return PolicyOutput(noise)
    result = evaluate_real_prefix(
        env, capture_simulator_snapshot(env), first_action=np.array([1.]),
        prefix_actor=prefix, prior_actor=prior, terminal_q=lambda obs, actions: actions,
        prefix_noise=np.array([[99.], [2.], [3.], [4.]]),
        tail_noise=np.array([[5.], [6.]]), discount=1.)
    assert [r for _, r in prefix_calls] == [3, 2, 1]
    assert [n.item() for n, _ in prefix_calls] == [2., 3., 4.]
    assert [n.item() for n, _ in tail_calls] == [5., 6.]
    assert result["real_bootstrapped_return"] == 15.
    assert result["real_mc_return"] == 21.


def test_failure_preserves_simulator_and_python_numpy_torch_rng():
    env = Accumulator()
    root = capture_simulator_snapshot(env)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="probe failure"):
        with preserve_simulator(env):
            env.step(np.array([2.]))
            random.random()
            np.random.normal()
            torch.randn(5)
            raise RuntimeError("probe failure")
    assert capture_simulator_snapshot(env).sha256 == root.sha256
    assert random.getstate() == python_state
    assert np.random.get_state()[0] == numpy_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_snapshot_roundtrip_immutable_decode_and_restore_rng():
    env = Accumulator()
    root = SimulatorSnapshot.from_dict(capture_simulator_snapshot(env).to_dict())
    state = root.state()
    state["base"]["state"]["position"] = 999.
    first = env.step(np.array([1.]))
    restore_simulator_snapshot(env, root)
    second = env.step(np.array([1.]))
    np.testing.assert_array_equal(first[0], second[0])
    assert root.state()["base"]["state"]["position"] == 0.
    tampered = root.to_dict()
    tampered["state"]["base"]["state"]["position"] = 99.
    with pytest.raises(ValueError, match="checksum"):
        SimulatorSnapshot.from_dict(tampered)


def test_continuation_separates_action_and_memory_and_restores_source():
    env = Accumulator()
    root = capture_simulator_snapshot(env)
    def branch(first, memory, kind):
        return evaluate_continuation(
            env, root, first_action=np.array([first]),
            controller_factory=lambda: lambda obs, offset: np.array([memory]),
            steps=4, discount=.5, intervention=kind,
            controller_identity="fixed-rule", memory_identity=str(memory))
    common = branch(1., 2., "first_action_only")
    changed_action = branch(3., 2., "first_action_only")
    assert changed_action["real_return"] - common["real_return"] == 2.
    changed_memory = branch(1., 3., "memory_only")
    assert changed_memory["first_action"] == common["first_action"]
    assert changed_memory["real_return"] - common["real_return"] == 3.
    assert common["actions"] == [[1.], [2.], [2.], [2.]]
    assert capture_simulator_snapshot(env).sha256 == root.sha256


@pytest.mark.skipif(os.environ.get("AMBI_RUN_REAL_DMCONTROL_TESTS") != "1", reason="Real DMControl tests are opt-in")
def test_real_humanoid_snapshot_reproduces_continuation():
    from domains.dmcontrol import DMControlEnv
    env = gym.wrappers.TimeLimit(DMControlEnv(task="humanoid-walk"), max_episode_steps=500)
    try:
        env.reset(seed=17)
        actions = np.random.default_rng(4).uniform(-.1, .1, (3, *env.action_space.shape))
        env.step(actions[0])
        snapshot = capture_simulator_snapshot(env)
        expected = [env.step(action) for action in actions]
        restore_simulator_snapshot(env, snapshot)
        actual = [env.step(action) for action in actions]
        for before, after in zip(expected, actual):
            np.testing.assert_array_equal(before[0], after[0])
            assert before[1:] == after[1:]
    finally:
        env.close()


def analytic_reference(soft):
    from utils.transfer_diagnostics import Reference

    class Actor(torch.nn.Module):
        def __init__(self, bias):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(bias))

    class Critic(torch.nn.Module):
        def _forward_eager(self, value):
            return value

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._pi, self._Qs = Actor(.1), Critic()
            self.q_backend = SimpleNamespace(pair_size=2)

        def encode(self, observations):
            return observations

        def pi(self, z, policy=None, noise=None, **kwargs):
            policy = self._pi if policy is None else policy
            return torch.tanh(policy.bias + noise), {"log_prob": torch.full((len(z), 1), -2.)}

        def Q(self, z, actions, **kwargs):
            return (7. + actions[:, :1])[None].expand(2, -1, -1)

        def joint_input(self, z, action):
            return z, action

        def reward_from_joint(self, joint):
            return 2. * joint[1]

        def decode_reward(self, reward):
            return reward

        def next_from_joint(self, joint):
            return joint[0] + 2. * joint[1]

    cfg = SimpleNamespace(inner_log_std_mapping="clip", inner_log_std_min=-4., inner_log_std_max=1.,
        log_std_mapping="clip", log_std_min=-5., log_std_max=2.,
        inner_terminal_entropy="outer" if soft else "none",
        inner_horizon_critic_source="sac" if soft else "aux_return",
        inner_sac_critic_target="entropy_augmented" if soft else "reward_only",
        outer_critic_target="entropy_augmented", outer_actor_entropy_mode="squashed",
        mppi_terminal_q_reduction="mean_all", inner_q_actor_reduction="mean_all",
        action_dim=1, episodic=False)
    model = Model()
    engine = SimpleNamespace(_horizon_actor=model._pi, _horizon_critic=model._Qs, _horizon_actor_options={})
    wrapped = SimpleNamespace(cfg=cfg, _scale_action=lambda a: np.asarray(a) / 2.,
        _unscale_action=lambda a: 2. * np.asarray(a),
        agent=SimpleNamespace(model=model, inner_engine=engine, device=torch.device("cpu"), discount=.5,
            alpha=torch.tensor(.4), actor_loss_scale_enabled=True, actor_loss_scale=torch.tensor(2.)))
    actor = Actor(.2)
    final = SimpleNamespace(alpha=.2, actor_loss_scale=1.5,
        policy_bounds={"log_std_mapping": "clip", "log_std_min": -4., "log_std_max": 1.},
        make_module=lambda component, device: deepcopy(actor))
    return Reference(wrapped), final


@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize("horizon", [1, 2, 5])
def test_integrated_real_audit_pairs_model_noise_action_units_and_entropy(soft, horizon):
    from utils.transfer_diagnostics import audit_real
    reference, final = analytic_reference(soft)
    env = Accumulator()
    root = capture_simulator_snapshot(env)
    rows = audit_real(reference, env, root, torch.tensor([[0.]]), reference.model._pi,
        {("natural", "fresh"): (np.array([.7]), final)}, horizon=horizon, seed=9,
        options={"real_rollouts": 3, "real_tail_steps": 4})
    assert len(rows) == 3
    for row in rows:
        assert row["model_prefix_error"] == pytest.approx(0., abs=2e-6)
        assert row["terminal_value_error"] == pytest.approx(row["terminal_prediction_error"])
        assert row["total_prediction_error"] == pytest.approx(row["model_prefix_error"] + row["terminal_value_error"])
        assert row["first_action"] == [.7]
        assert row["boundary_action_entropy_included"] == soft
        if soft:
            assert row["terminal_alpha"] == pytest.approx(.8)
            assert row["real_endpoint_entropy"] == pytest.approx(1.6)
        else:
            assert row["real_tail_entropy"] == 0.
    assert capture_simulator_snapshot(env).sha256 == root.sha256


def test_replanning_audit_pairs_future_streams_and_distinguishes_interventions(monkeypatch):
    import utils.transfer_diagnostics as diagnostics
    class RNG:
        def __init__(self, seed):
            self.value = seed

        def training_state_dict(self):
            return {"value": self.value}

        def load_training_state_dict(self, state):
            self.value = state["value"]

    engine = SimpleNamespace(rng=RNG(44), _new_rng=RNG)
    wrapped = SimpleNamespace(agent=SimpleNamespace(inner_engine=engine, discount=.5))
    reference = SimpleNamespace(wrapped=wrapped)
    calls = []
    def solve(wrapped, observation, rng, *, donor, branch, capture_rounds):
        memory = 0 if branch == "fresh" else donor.marker
        calls.append((branch, memory, rng["value"]))
        engine.rng.value = rng["value"] + 1
        return np.array([memory / 10.]), None, SimpleNamespace(marker=memory + 1)
    monkeypatch.setattr(diagnostics, "solve_fork", solve)
    env = Accumulator()
    snapshot = capture_simulator_snapshot(env)
    finals = {("natural", branch): (np.array([index / 10.]), SimpleNamespace(marker=index))
              for index, branch in enumerate(diagnostics.BRANCHES, start=1)}
    rows = diagnostics.audit_replanning(reference, env, snapshot, finals, {"value": -99},
        horizon=5, seed=18, options={"replan_steps": 3, "replan_repeats": 2})
    assert len(rows) == 24
    assert len(calls) == 48
    for repeat in range(2):
        selected = [row for row in rows if row["future_replicate"] == repeat]
        assert len({row["future_solver_seed"] for row in selected}) == 1
        action_only = [row for row in selected if row["intervention"] == "first_action_only"]
        assert all(row["actions"][1:] == [[0.], [0.]] for row in action_only)
        memory_only = [row for row in selected if row["intervention"] == "memory_only"]
        assert all(row["first_action"] == [.1] for row in memory_only)
        assert all(row["future_component_rule"] == "joint" for row in memory_only)
        assert len({tuple(row["actions"][1]) for row in memory_only}) == 4
        branch_calls = calls[repeat * 24:(repeat + 1) * 24]
        assert len({state for _, _, state in branch_calls[::2]}) == 1
        assert all(right[2] == left[2] + 1 for left, right in zip(branch_calls[::2], branch_calls[1::2]))
    assert rows[0]["future_solver_seed"] != rows[12]["future_solver_seed"]
    assert engine.rng.value == 44
    assert capture_simulator_snapshot(env).sha256 == snapshot.sha256


def test_replanning_callbacks_run_real_inner_sac_forks_without_changing_backbone():
    from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
    from tests.test_ambi_root_local_sac import _model_from_params
    from tests.test_aux_critic_transfer import critic_params
    from utils.transfer_diagnostics import BRANCHES, Reference, audit_replanning, solve_fork

    class ThreeObservationAccumulator(Accumulator):
        def observation(self):
            return np.array([self.position, .2, -.1], dtype=np.float32)

        def step(self, action):
            _, reward, terminal, truncation, info = super().step(action)
            return self.observation(), reward, terminal, truncation, info

        def load_calibration_state(self, state):
            super().load_calibration_state(state)
            return self.observation()

    model = _model_from_params(critic_params("return", inner_critic_scope="action",
        inner_rounds=1, inner_rollout_horizon=1))
    env = ThreeObservationAccumulator()
    try:
        reference = Reference(model)
        rng = model.agent.inner_engine.rng.training_state_dict()
        _, _, donor = solve_fork(model, env.observation(), rng)
        finals = {}
        for branch in BRANCHES:
            action, _, final = solve_fork(model, env.observation(), rng, donor=donor, branch=branch)
            finals[("natural", branch)] = action, final
        frozen = _clone_tree(model.agent.model.state_dict())
        snapshot = capture_simulator_snapshot(env)
        rows = audit_replanning(reference, env, snapshot, finals, rng, horizon=1, seed=17,
            options={"replan_steps": 2, "replan_repeats": 1})
        assert len(rows) == 12
        assert all(row["steps"] == 2 and row["requested_horizon_complete"] for row in rows)
        assert capture_simulator_snapshot(env).sha256 == snapshot.sha256
        _assert_tree_equal(frozen, model.agent.model.state_dict())
    finally:
        model.close()
