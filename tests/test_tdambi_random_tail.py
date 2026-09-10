"""Fresh native inner networks with a frozen TD-MPC2 rollout-tail bootstrap."""

from copy import deepcopy
import json

import pytest
import torch

from RL.tdmpc2_core.common import math as td_math
from RL.tdmpc2_core.inner_trace import InnerActionTrace
from tests.test_ambi_latency_contract import _assert_tree_equal
from tests.test_tdambi_checkpoint import MATRIX, make_tdambi
from tests.test_tdambi_inner import _tdambi_model, _snapshot
from tests.test_tdambi_publication import _config, _identity
from utils.eval_series import _planner_display_label


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]


def _prepare(engine, t0=True):
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=t0)
    return engine.state


@pytest.mark.parametrize("actor", ["prior", "random"])
@pytest.mark.parametrize("critic", ["prior", "random"])
def test_native_wrapper_accepts_independent_initialization_and_outer_tail(actor, critic):
    model = make_tdambi(inner_actor_initialization=actor.upper(),
                        inner_critic_initialization=critic.upper(),
                        inner_finite_horizon=True)
    try:
        assert model.cfg.inner_actor_initialization == actor
        assert model.cfg.inner_critic_initialization == critic
        assert model.cfg.inner_critic_target_initialization == (
            "online" if critic == "random" else "outer_target")
        assert model.cfg.inner_finite_horizon
        assert model.cfg.mppi_terminal_q_reduction == "mean_pair"
        assert model.cfg.inner_operator == "tdambi"
    finally:
        model.env.close()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("actor,critic", [("prior", "prior"), ("random", "prior"),
                                         ("prior", "random"), ("random", "random")])
def test_dense_scratch_reset_and_local_target_are_fresh_on_every_decision(device, actor, critic):
    model = _tdambi_model(device=device, inner_actor_initialization=actor,
                          inner_critic_initialization=critic)
    engine, outer = model.agent.inner_engine, model.agent.model
    before = deepcopy(outer.state_dict())
    global_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    previous, objects = {}, {}
    try:
        for decision in range(3):
            state = _prepare(engine, t0=decision == 0)
            for component, mode, prior in (("actor", actor, outer._pi), ("critic", critic, outer._Qs)):
                local = getattr(state, component)
                assert local is not prior
                values = deepcopy(local.state_dict())
                if decision:
                    assert local is objects[component]
                if mode == "prior":
                    _assert_tree_equal(values, prior.state_dict())
                else:
                    assert any(not torch.equal(value, prior.state_dict()[key]) for key, value in values.items())
                    if decision:
                        assert any(not torch.equal(value, previous[component][key]) for key, value in values.items())
                    for layer in local.modules():
                        if isinstance(layer, torch.nn.LayerNorm):
                            assert torch.all(layer.weight == 1)
                            assert not torch.count_nonzero(layer.bias)
                previous[component], objects[component] = values, local
            expected_target = state.critic if critic == "random" else outer._target_Qs
            _assert_tree_equal(state.critic_target.state_dict(), expected_target.state_dict())
            assert state.critic_target is not expected_target
            assert not state.critic_target.training
            if critic == "random":
                assert all(not torch.count_nonzero(head[-1].weight) for head in state.critic)
                assert any(torch.count_nonzero(head[-1].bias) for head in state.critic)
            engine._clear_expired(t0=False, include_action=True)
        _assert_tree_equal(outer.state_dict(), before)
        assert all(parameter.grad is None for parameter in outer.parameters())
        torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
        for actual, expected in zip(torch.cuda.get_rng_state_all() if cuda_rng else [], cuda_rng):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        model.env.close()


@pytest.mark.parametrize("device", DEVICES)
def test_tail_uses_outer_actor_online_mean_q_and_interior_uses_local_target(monkeypatch, device):
    model = _tdambi_model(device=device, inner_actor_initialization="random",
                          inner_critic_initialization="random", inner_finite_horizon=True)
    engine = model.agent.inner_engine
    _prepare(engine)
    batch = {
        "z": torch.zeros(4, model.cfg.latent_dim, device=device),
        "next_z": torch.ones(4, model.cfg.latent_dim, device=device),
        "action": torch.zeros(4, model.cfg.action_dim, device=device),
        "reward": torch.full((4, 1), 3.0, device=device),
        "terminated": torch.tensor([[0.], [0.], [1.], [1.]], device=device),
        "horizon_end": torch.tensor([[0.], [1.], [0.], [1.]], device=device),
    }
    calls, targets = [], []

    def local_policy(z, **kwargs):
        assert kwargs["policy"] is engine.state.actor
        assert not torch.is_grad_enabled()
        calls.append("local_actor")
        return z.new_full((len(z), model.cfg.action_dim), -0.5)

    def local_values(z, action, **kwargs):
        assert kwargs["qs"] is engine.state.critic_target
        assert torch.all(action == -0.5)
        calls.append("local_target")
        return z.new_full((model.cfg.num_q, len(z), 1), 5.0)

    def outer_policy(z, **kwargs):
        assert set(kwargs) == {"noise"}
        assert not torch.is_grad_enabled() and not engine.model._pi.training
        calls.append("outer_actor")
        return z.new_full((len(z), model.cfg.action_dim), 0.75), {}

    def outer_q(z, action, **kwargs):
        assert kwargs == {"reduction": "mean_pair"}
        assert not torch.is_grad_enabled() and not engine.model._Qs.training
        assert torch.all(action == 0.75)
        calls.append("outer_online_q")
        return z.new_full((len(z), 1), 11.0)

    original_loss = td_math.soft_ce

    def loss(prediction, target, cfg):
        targets.append(target.detach().clone())
        return original_loss(prediction, target, cfg)

    monkeypatch.setattr(engine.model, "pi_action", local_policy)
    monkeypatch.setattr(engine.model, "q_values", local_values)
    monkeypatch.setattr(engine.model, "pi", outer_policy)
    monkeypatch.setattr(engine.model, "Q", outer_q)
    monkeypatch.setattr(td_math, "soft_ce", loss)
    try:
        with engine.rng.action_fork(), engine.rng.fork("bootstrap"):
            metrics = engine._tdambi_critic_step(batch)
        expected = 3 + model.agent.discount * batch["reward"].new_tensor([[5.], [11.], [0.], [0.]])
        assert len(targets) == model.cfg.num_q
        for target in targets:
            torch.testing.assert_close(target, expected)
        assert calls == ["local_actor", "local_target", "outer_actor", "outer_online_q"]
        assert metrics["critic_outer_tail_fraction"] == .5
        assert all(parameter.grad is None for parameter in engine.model.parameters())
    finally:
        model.env.close()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rounds", [1, 5, 10])
def test_sweep_budgets_trace_and_cold_vs_reused_workspace_are_reproducible(device, rounds):
    options = dict(device=device, inner_actor_initialization="random",
                   inner_critic_initialization="random", inner_finite_horizon=True,
                   inner_rounds=rounds, inner_rollouts_per_round=4, inner_rollout_horizon=3,
                   train_unroll_horizon=3,
                   inner_updates_per_round=None, inner_steps_per_update=4,
                   inner_update_timing="step", inner_batch_size=4, inner_replay_capacity=128)
    cold, pooled = [_tdambi_model(**options) for _ in range(2)]
    before = deepcopy(pooled.agent.model.state_dict())
    try:
        for seed in (900, 901):
            cold.agent.inner_engine.reset_for_evaluation(seed)
            pooled.agent.inner_engine.reset_for_evaluation(seed, reuse_action_pool=True)
            for decision in range(2):
                obs = torch.full((3,), decision * .1)
                expected = cold.agent.act(obs, t0=decision == 0, collect_diagnostics=False)
                trace = InnerActionTrace()
                actual = pooled.agent.act(obs, t0=decision == 0, collect_diagnostics=False, trace=trace)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                _assert_tree_equal(_snapshot(pooled), _snapshot(cold))
                metrics = pooled.agent.last_inner_metrics
                assert metrics["inner_actor_optimizer_steps"] == metrics["inner_critic_optimizer_steps"] == rounds * 3
                assert metrics["inner_critic_target_updates"] == rounds * 3
                assert metrics["inner_temperature_optimizer_steps"] == 0
                assert metrics["inner_model_steps"] == metrics["inner_buffer_size"] == rounds * 12
                assert metrics["inner_actor_random_initialization"] == metrics["inner_critic_random_initialization"] == 1
                updates = [event for event in trace.events if event["phase"] == "update"]
                assert len(updates) == rounds * 3
                assert all("critic_outer_tail_fraction" in event["metrics"] for event in updates)
                replay = pooled.agent.inner_engine._action_pool.replay
                assert replay.horizon_end[:replay.size].sum() == rounds * 4
                assert not replay.terminated[:replay.size].any()
        _assert_tree_equal(pooled.agent.model.state_dict(), before)
    finally:
        cold.env.close()
        pooled.env.close()


def test_default_planner_identity_is_preserved_and_random_tail_is_distinct():
    base = _config()
    explicit_prior = dict(base, inner_actor_initialization="prior", inner_critic_initialization="prior",
                          inner_critic_target_initialization="outer_target", inner_finite_horizon=False)
    assert _identity(base) == _identity(explicit_prior)
    identities = set()
    for actor, critic in (("prior", "prior"), ("random", "prior"), ("prior", "random"), ("random", "random")):
        for tail in (False, True):
            planner = _identity(dict(base, inner_actor_initialization=actor, inner_critic_initialization=critic,
                                     inner_finite_horizon=tail, mppi_terminal_q_reduction="mean_pair"))
            identities.add(json.dumps(planner, sort_keys=True))
            if critic == "random":
                assert planner["semantics"]["target_initialization"] == "fresh_random_online_critic"
            if tail:
                assert "outer rollout tail" in _planner_display_label(planner, compact=True)
                assert "frozen_outer_actor_sample_at_horizon" in planner["semantics"]["bootstrap_action"]
    assert len(identities) == 8


def test_versioned_sweep_presets_keep_batch_rollouts_and_horizon_constant():
    matrix = json.loads(MATRIX.read_text())
    variants = matrix["comparisons"]["random_outer_tail"]["variants"]
    for rounds in (1, 5, 10):
        cfg = {**matrix["shared_alg_params"], **variants[f"j{rounds}"]["alg_params"]}
        assert cfg["inner_actor_initialization"] == cfg["inner_critic_initialization"] == "random"
        assert cfg["inner_rounds"] == rounds
        assert cfg["inner_batch_size"] == cfg["inner_rollouts_per_round"] == cfg["inner_steps_per_update"] == 512
        assert cfg["inner_rollout_horizon"] == 3 and cfg["inner_replay_capacity"] == 32768
        assert cfg["inner_updates_per_round"] is None and cfg["inner_update_timing"] == "step"
        assert cfg["inner_finite_horizon"]
