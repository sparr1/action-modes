"""To-go return semantics and immutable actor recording at solve boundaries."""

from dataclasses import FrozenInstanceError
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from RL.tdmpc2_core.inner_trace import (
    InnerActionTrace, evaluate_frozen_outer_q, evaluate_outer_tail,
)
from tests.test_ambi_inner_trace import _snapshot
from tests.test_ambi_latency_contract import _assert_tree_equal, _pool_snapshot
from tests.test_ambi_root_local_sac import _tiny_component_model, _tiny_model


class _AnalyticCritic(torch.nn.Module):
    def _forward_eager(self, joint):
        return 7 + joint[:, 1:2]


class _AnalyticModel(torch.nn.Module):
    def __init__(self, *, terminate=False):
        super().__init__()
        self._pi = torch.nn.Linear(1, 1)
        self._Qs = _AnalyticCritic()
        self.terminate = terminate
        self.policy_calls = []
        self.q_calls = []

    def pi(self, z, *, policy=None, noise, **bounds):
        self.policy_calls.append((policy, noise.clone(), bounds))
        return noise, {}

    def joint_input(self, z, action):
        return torch.cat((z, action), -1)

    def reward_from_joint(self, joint):
        return joint[:, 1:2] + 2

    def next_from_joint(self, joint):
        return joint[:, :1] + 1

    def termination(self, z):
        return (z >= 2).float()

    def Q(self, z, action, **kwargs):
        self.q_calls.append((z.clone(), action.clone(), kwargs))
        return kwargs["qs"](self.joint_input(z, action))


@pytest.mark.parametrize("episodic", [False, True])
def test_analytic_outer_tail_uses_raw_discounted_rewards_handoff_and_masks(monkeypatch, episodic):
    model = _AnalyticModel()
    inner = torch.nn.Linear(1, 1)
    inner.eval()
    cfg = SimpleNamespace(action_dim=1, episodic=episodic,
                          inner_termination_threshold=0.5,
                          mppi_terminal_q_reduction="mean_pair")
    engine = SimpleNamespace(model=model, cfg=cfg, agent=SimpleNamespace(discount=0.5))
    noise = torch.tensor([[[0.0], [1.0]], [[1.0], [2.0]],
                          [[2.0], [3.0]], [[3.0], [4.0]]])
    pair = torch.tensor([0, 2])
    bounds = {"log_std_min": -3.0, "log_std_max": 1.0,
              "log_std_mapping": "direct_clamp"}
    monkeypatch.setattr("RL.tdmpc2_core.inner_trace.td_math.two_hot_inv", lambda r, cfg: r)
    rng = torch.random.get_rng_state().clone()
    modes = [m.training for m in model.modules()] + [inner.training]
    result = evaluate_outer_tail(engine, torch.zeros(1, 1), inner, noise,
                                 pair_indices=pair, policy_bounds=bounds)
    expected_reward = torch.tensor([[3.5], [5.0]]) if episodic else torch.tensor([[4.5], [6.25]])
    expected_tail = torch.zeros(2, 1) if episodic else torch.tensor([[1.25], [1.375]])
    torch.testing.assert_close(result["reward"], expected_reward)
    torch.testing.assert_close(result["bootstrap"], expected_tail)
    torch.testing.assert_close(result["total"], expected_reward + expected_tail)
    assert all(call[0] is inner and call[2] == bounds for call in model.policy_calls[:3])
    assert model.policy_calls[-1][0] is None and model.policy_calls[-1][2] == {}
    torch.testing.assert_close(model.q_calls[0][1], noise[-1])
    assert model.q_calls[0][2] == dict(reduction="mean_pair", pair_indices=pair,
                                      trusted_pair_indices=True,
                                      qs=model._Qs._forward_eager)
    assert [m.training for m in model.modules()] + [inner.training] == modes
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    with pytest.raises(ValueError, match="explicit pair_indices"):
        evaluate_outer_tail(engine, torch.zeros(1, 1), inner, noise)


@pytest.mark.parametrize("representation", ["scalar", "distributional", "native_distributional"])
@pytest.mark.parametrize("reduction", ["mean_all", "min_all", "mean_pair", "min_pair"])
def test_frozen_q_preserves_decoding_modes_rng_and_lazy_compile_state(monkeypatch, representation, reduction):
    from evaluate_ambi_calibration import FrozenCallbacks

    heads = 2 if representation == "scalar" else 3
    model = _tiny_model(q_representation="scalar" if representation == "scalar" else "distributional",
                        num_q=heads, dropout=0.2)
    try:
        world = model.agent.model
        if representation == "native_distributional":
            # Q's native TDAMBI decoder intentionally has distinct finite
            # precision; exercise that branch without a separate learner.
            world.cfg.inner_operator = "tdambi"
        world.eval()
        observations = np.zeros((4, 3), dtype=np.float32)
        z = world.encode(torch.from_numpy(observations))
        actions = torch.full((4, model.cfg.action_dim), 0.25)
        pair = torch.tensor([0, heads - 1]) if reduction.endswith("_pair") else None
        expected = world.Q(z, actions, reduction=reduction, pair_indices=pair,
                           trusted_pair_indices=True).detach()
        world._Qs.enable_compile(strict=True)

        def forbidden(*args, **kwargs):
            raise AssertionError("A diagnostic touched a compiled ensemble forward")

        monkeypatch.setattr(torch, "compile", forbidden)
        monkeypatch.setattr(world._Qs, "forward", forbidden)
        sentinel = object()
        world._Qs._compiled_forward = sentinel
        world._Qs.train()
        world._Qs.modules_list[0].eval()
        modes = [module.training for module in world.modules()]
        rng = torch.random.get_rng_state().clone()
        actual = evaluate_frozen_outer_q(world, z, actions, reduction=reduction, pair_indices=pair)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not actual.requires_grad
        assert world._Qs._compiled_forward is sentinel
        assert world._Qs._compile_enabled and not world._Qs._compile_failed
        assert [module.training for module in world.modules()] == modes
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)

        model.cfg.mppi_terminal_q_reduction = reduction
        callbacks = FrozenCallbacks(model, pair_indices=pair)
        np.testing.assert_array_equal(callbacks.q(observations, actions.numpy()), expected.reshape(-1).numpy())
        assert world._Qs._compiled_forward is sentinel
        assert [module.training for module in world.modules()] == modes
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    finally:
        model.env.close()


def _reference_model(**kwargs):
    return _tiny_model(
        train_unroll_horizon=3,
        inner_rounds=5, inner_rollouts_per_round=512, inner_rollout_horizon=3,
        inner_batch_size=512, inner_replay_capacity=7680,
        inner_updates_per_round=None, inner_steps_per_update=512,
        inner_update_timing="step", inner_finite_horizon=True,
        inner_actor_initialization="random", inner_critic_initialization="random",
        inner_critic_target_initialization="online", **kwargs,
    )


def test_h1_critic_first_probe_and_snapshot_boundaries_preserve_the_solve():
    params = dict(train_unroll_horizon=3, inner_rounds=4,
        inner_rollouts_per_round=128, inner_rollout_horizon=1,
        inner_batch_size=256, inner_replay_capacity=2048,
        inner_critic_updates_per_round=32, inner_actor_updates_per_round=4,
        inner_update_timing="round", inner_finite_horizon=True,
        inner_actor_initialization="random", inner_actor_initial_std=0.3,
        inner_critic_initialization="random", inner_critic_target_initialization="online",
        inner_temperature_mode="fixed", inner_temperature=1e-4,
        outer_critic_target="reward_only", inner_sac_critic_target="reward_only",
        q_representation="distributional", num_q=5, dropout=0.01)
    ordinary, observed = _tiny_component_model(**params), _tiny_component_model(**params)
    trace = InnerActionTrace(probes=True, probe_mode="outer_tail", probe_rollouts=32,
        probe_horizon=1, probe_seed=882, capture_actors=True)
    try:
        expected = ordinary.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
        rng = torch.random.get_rng_state().clone()
        actual = observed.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False, trace=trace)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _assert_tree_equal(_pool_snapshot(observed.agent), _pool_snapshot(ordinary.agent))
        _assert_tree_equal(observed.agent.model.state_dict(), ordinary.agent.model.state_dict())
        _assert_tree_equal(observed.agent.inner_engine.rng.training_state_dict(),
                           ordinary.agent.inner_engine.rng.training_state_dict())
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        probes = [event for event in trace.events if event["phase"] == "probe"]
        assert [(event["round_index"], event["critic_updates"], event["actor_updates"])
                for event in probes] == [(0, 0, 0), (1, 32, 4), (2, 64, 8), (3, 96, 12), (4, 128, 16)]
        assert [(actor.round_index, actor.critic_updates, actor.actor_updates)
                for actor in trace.actor_snapshots] == [(0, 0, 0), (1, 32, 4), (2, 64, 8), (3, 96, 12), (4, 128, 16)]
        for round_index in range(1, 5):
            updates = [event for event in trace.events
                       if event["round_index"] == round_index and event["phase"] == "update"]
            assert [event["updated_critic"] for event in updates] == [True] * 32 + [False] * 4
        assert sum(event["metrics"]["probe_model_steps"] for event in probes) == 192
        assert sum(event["metrics"]["probe_policy_evaluations"] for event in probes) == 384
        assert sum(event["metrics"]["probe_q_evaluations"] for event in probes) == 192
    finally:
        ordinary.env.close()
        observed.env.close()


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_reference_round_counts_components_and_immutable_snapshot_noninterference(representation):
    heads = 2 if representation == "scalar" else 3
    ordinary = _reference_model(q_representation=representation, num_q=heads, dropout=0.2)
    observed = _reference_model(q_representation=representation, num_q=heads, dropout=0.2)
    trace = InnerActionTrace(probes=True, probe_mode="outer_tail", probe_rollouts=32,
                             probe_horizon=3, probe_seed=882, capture_actors=True)
    try:
        expected = ordinary.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False)
        global_rng = torch.random.get_rng_state().clone()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        actual = observed.agent.act(torch.zeros(3), t0=True, collect_diagnostics=False, trace=trace)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _assert_tree_equal(_snapshot(observed.agent), _snapshot(ordinary.agent))
        torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
        assert random.getstate() == python_rng
        assert all(np.array_equal(a, b) for a, b in zip(np.random.get_state(), numpy_rng))
        assert [m.training for m in observed.agent.model.modules()] == [m.training for m in ordinary.agent.model.modules()]
        probes = [e for e in trace.events if e["phase"] == "probe"]
        assert [e["round_index"] for e in probes] == list(range(6))
        assert [e["actor_updates"] for e in probes] == [0, 3, 6, 9, 12, 15]
        assert [e["critic_updates"] for e in probes] == [0, 3, 6, 9, 12, 15]
        for event in probes:
            metrics = event["metrics"]
            assert metrics["togo_return_mean"] == pytest.approx(metrics["togo_reward_mean"] + metrics["togo_bootstrap_mean"], abs=1e-5)
            for reference in ("initial", "outer"):
                assert metrics[f"togo_return_{reference}_mean"] == pytest.approx(metrics[f"togo_reward_{reference}_mean"] + metrics[f"togo_bootstrap_{reference}_mean"], abs=1e-5)
                assert metrics[f"togo_return_gain_vs_{reference}"] == pytest.approx(metrics[f"togo_reward_gain_vs_{reference}"] + metrics[f"togo_bootstrap_gain_vs_{reference}"], abs=1e-5)
            assert metrics["probe_seconds"] >= 0
        assert probes[0]["metrics"]["togo_return_gain_vs_initial"] == 0
        assert sum(e["metrics"]["probe_model_steps"] for e in probes) == 672
        assert sum(e["metrics"]["probe_reward_evaluations"] for e in probes) == 672
        assert sum(e["metrics"]["probe_policy_evaluations"] for e in probes) == 896
        assert sum(e["metrics"]["probe_q_evaluations"] for e in probes) == 224
        snapshots = trace.actor_snapshots
        assert [s.actor_updates for s in snapshots] == [0, 3, 6, 9, 12, 15]
        assert [s.round_index for s in snapshots] == list(range(6))
        final_weights = snapshots[-1].make_policy().state_dict()
        _assert_tree_equal(final_weights, observed.agent.inner_engine._action_pool.actor.state_dict())
        initial_weights = snapshots[0].make_policy().state_dict()
        assert any(not torch.equal(initial_weights[k], observed.agent.model._pi.state_dict()[k]) for k in initial_weights)
        assert any(not torch.equal(initial_weights[k], final_weights[k]) for k in initial_weights)
        with pytest.raises(FrozenInstanceError):
            snapshots[0].payload = b"invalid"
        mutable = snapshots[-1].make_policy()
        for p in mutable.parameters():
            p.zero_()
        _assert_tree_equal(final_weights, snapshots[-1].make_policy().state_dict())
        observed.agent.act(torch.ones(3), collect_diagnostics=False)
        _assert_tree_equal(final_weights, snapshots[-1].make_policy().state_dict())
        assert trace._noise is None and trace._togo_initial is None
    finally:
        ordinary.env.close()
        observed.env.close()


def test_selected_actor_snapshots_do_not_require_model_probes():
    model = _tiny_model(inner_rounds=3, inner_updates_per_round=1)
    trace = InnerActionTrace(capture_actors=True, actor_rounds=[0, 2])
    try:
        model.agent.act(torch.zeros(3), collect_diagnostics=False, trace=trace)
        assert [s.round_index for s in trace.actor_snapshots] == [0, 2]
        assert not any(e["phase"] == "probe" for e in trace.events)
        assert all(e["metrics"]["actor_snapshot_seconds"] >= 0 for e in trace.events if e["phase"] == "actor_snapshot")
    finally:
        model.env.close()


def test_prior_only_outer_tail_and_snapshot_have_zero_paired_gain():
    model = _tiny_model(inner_operator="none", inner_rounds=0,
                        inner_rollouts_per_round=0, inner_updates_per_round=0,
                        inner_temperature_mode="inherit_outer", inner_rollout_horizon=3)
    trace = InnerActionTrace(probes=True, probe_mode="outer_tail", capture_actors=True)
    try:
        model.agent.act(torch.zeros(3), trace=trace, collect_diagnostics=False)
        metrics = trace.events[-1]["metrics"]
        assert all(value == 0 for name, value in metrics.items() if "gain_vs" in name)
        assert len(trace.actor_snapshots) == 1 and not trace.actor_snapshots[0].inner
        assert trace.actor_snapshots[0].policy_bounds == {}
        _assert_tree_equal(trace.actor_snapshots[0].make_policy().state_dict(), model.agent.model._pi.state_dict())
    finally:
        model.env.close()


def test_terminated_rows_mask_nan_rewards_and_bootstrap(monkeypatch):
    model = _AnalyticModel()
    inner = torch.nn.Linear(1, 1)
    engine = SimpleNamespace(model=model, cfg=SimpleNamespace(
        action_dim=1, episodic=True, inner_termination_threshold=0.5,
        mppi_terminal_q_reduction="mean_all"), agent=SimpleNamespace(discount=0.5))
    monkeypatch.setattr("RL.tdmpc2_core.inner_trace.td_math.two_hot_inv", lambda r, cfg: r)
    monkeypatch.setattr(model, "reward_from_joint", lambda joint: torch.where(
        joint[:, :1] >= 2, float("nan"), 2.0))
    monkeypatch.setattr(model, "Q", lambda z, action, **kwargs: z.new_full((z.shape[0], 1), float("nan")))
    result = evaluate_outer_tail(engine, torch.zeros(1, 1), inner, torch.zeros(4, 2, 1))
    torch.testing.assert_close(result["total"], torch.full((2, 1), 3.0))
    assert torch.count_nonzero(result["bootstrap"]) == 0


def test_outer_tail_error_restores_modes_and_global_rng(monkeypatch):
    model = _tiny_model(inner_rollout_horizon=3, inner_replay_capacity=18)
    trace = InnerActionTrace(probes=True, probe_mode="outer_tail", capture_actors=True)
    before = torch.random.get_rng_state().clone()
    modes = [module.training for module in model.agent.model.modules()]
    actor_modes = {}
    original_probe = trace.probe
    def observed_probe(engine, root_z, policy, **kwargs):
        actor_modes.update((m, m.training) for m in policy.modules())
        return original_probe(engine, root_z, policy, **kwargs)
    monkeypatch.setattr(trace, "probe", observed_probe)
    def broken(*args, **kwargs):
        raise RuntimeError("tail failed")
    monkeypatch.setattr(model.agent.model, "Q", broken)
    try:
        with pytest.raises(RuntimeError, match="tail failed"):
            model.agent.act(torch.zeros(3), trace=trace)
        assert model.agent.inner_engine._active_trace is None
        assert [module.training for module in model.agent.model.modules()] == modes
        assert all(module.training == mode for module, mode in actor_modes.items())
        assert trace.events == [] and trace.actor_snapshots == []
        assert trace._noise is None and trace._outer_probe is None
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    finally:
        model.env.close()
