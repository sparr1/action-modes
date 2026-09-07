"""Native TD-MPC2 update oracle and action-local TDAMBI non-interference."""

from copy import deepcopy
from unittest.mock import patch
import random

import numpy as np

import pytest
import torch

from RL.tdmpc2_core.common import math as td_math
from RL.tdmpc2_core.common.inner_utils import InnerRNG
from RL.tdmpc2_core.common.scale import RunningScale
from RL.tdmpc2_core.common.soft_world_model import SoftWorldModel
from RL.tdmpc2_core.common.world_model import WorldModel
from RL.tdmpc2_core.inner_trace import InnerActionTrace, metric_catalog
from tests.test_ambi_latency_contract import (
    _assert_tree_equal, _clone_tree, _optimizer_tensor_pointers, _pool_snapshot,
)
from tests.test_ambi_root_local_sac import _tiny_model


def _tdambi_model(**overrides):
    """Exercise the engine independently of the checkpoint adapter."""
    params = dict(
        q_representation="distributional", num_q=3, q_pair_size=2,
        log_std_mapping="tdmpc2_tanh", log_std_min=-10, log_std_max=2,
        inner_rounds=2, inner_updates_per_round=2,
        inner_rollouts_per_round=3, inner_rollout_horizon=2,
        inner_batch_size=4, inner_replay_capacity=12,
        inner_temperature_mode="fixed", inner_temperature=0.01,
        inner_critic_dropout_enabled=True,
        inner_critic_target_tau=0.03, inner_adam_eps=1e-8,
    )
    params.update(overrides)
    model = _tiny_model(**params)
    cfg = model.cfg
    cfg.inner_operator = "tdambi"
    cfg.tdambi_entropy_coef = 0.001
    cfg.tdambi_value_coef = 0.1
    cfg.tdambi_scale_tau = 0.03
    cfg.inner_actor_adam_eps = 1e-5
    # Nontrivial, different online and saved target critics expose accidental
    # resets from the online Q and ensure actor gradients use learned values.
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(731)
        for head in model.agent.model._Qs:
            head[-1].weight.normal_(std=0.2)
            head[-1].bias.normal_(std=0.2)
        for head in model.agent.model._target_Qs:
            head[-1].weight.normal_(std=0.4)
            head[-1].bias.normal_(std=0.3)
    model.agent.inner_engine.clear_all()
    return model


def _native_world(model):
    native = WorldModel(deepcopy(model.cfg))
    native.load_state_dict(model.agent.model.state_dict(), strict=True)
    native.eval()
    return native


def _same_random_normal(generator):
    def draw(tensor, **kwargs):
        return torch.randn(
            tensor.shape, dtype=tensor.dtype, device=tensor.device,
            generator=generator, **kwargs,
        )
    return draw


def _assert_close_tree(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_close_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_close_tree(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("action_dim", [1, 3])
def test_native_policy_information_and_saved_q_match_without_updates(action_dim):
    holder = _tdambi_model(inner_updates_per_round=0)
    try:
        cfg = deepcopy(holder.cfg)
        cfg.action_dim = action_dim
        soft = SoftWorldModel(cfg).eval()
        native = WorldModel(cfg).eval()
        native.load_state_dict(soft.state_dict(), strict=True)
        z = torch.randn(7, cfg.latent_dim)
        generator = torch.Generator().manual_seed(871)
        noise = torch.randn(7, action_dim, generator=generator)
        with patch("torch.randn_like", lambda tensor: noise):
            expected_action, expected_info = native.pi(z, None)
        actual_action, actual_info = soft.pi_tdmpc2(z, noise=noise)
        torch.testing.assert_close(actual_action, expected_action, rtol=0, atol=0)
        for key in ("mean", "log_std", "entropy", "scaled_entropy"):
            torch.testing.assert_close(actual_info[key], expected_info[key], rtol=0, atol=0)
        torch.testing.assert_close(
            soft.pi_action(z, deterministic=True), expected_info["mean"], rtol=0, atol=0,
        )
        for target in (False, True):
            expected = td_math.two_hot_inv(
                native.Q(z, actual_action, None, return_type="all", target=target), cfg,
            )
            torch.testing.assert_close(soft.q_values(z, actual_action, target=target), expected)
        holder.predict([0.1, 0.2, 0.3], deterministic=True, collect_diagnostics=False)
        assert holder.agent.last_inner_metrics["inner_actor_optimizer_steps"] == 0
        assert holder.agent.last_inner_metrics["inner_tdambi_calibration_samples"] == 0
    finally:
        holder.env.close()


@pytest.mark.parametrize("dropout,num_q", [(0.0, 3), (0.25, 3), (0.25, 2)])
def test_paired_update_matches_native_reference_and_optimizer_state(dropout, num_q):
    holder = _tdambi_model(dropout=dropout, num_q=num_q)
    engine = holder.agent.inner_engine
    cfg = holder.cfg
    try:
        with engine.rng.action_fork():
            engine._prepare_workspace(t0=True)
            with torch.no_grad():
                root = holder.agent.model.encode(torch.tensor([[0.1, 0.2, 0.3]]))
            engine._collect_round(root)
            assert not engine.state.replay._storage.requires_grad
            engine._calibrate_tdambi_scale()
            batch = engine._sample_batch()
        state = engine.state
        native = _native_world(holder)
        native._pi.load_state_dict(state.actor.state_dict())
        native._Qs.load_state_dict(state.critic.state_dict())
        native._target_Qs.load_state_dict(state.critic_target.state_dict())
        native.train()
        native_scale_cfg = deepcopy(cfg)
        native_scale_cfg.tau = cfg.tdambi_scale_tau
        scale = RunningScale(native_scale_cfg)
        scale.value.copy_(state.tdambi_scale)
        critic_optim = torch.optim.Adam(native._Qs.parameters(), lr=cfg.inner_critic_lr, eps=1e-8, foreach=False)
        actor_optim = torch.optim.Adam(native._pi.parameters(), lr=cfg.inner_actor_lr, eps=1e-5, foreach=False)
        oracle_rng = InnerRNG(cfg.seed, "cpu", extra_streams=("tdambi_calibration",))
        oracle_rng.load_training_state_dict(engine.rng.training_state_dict())

        with oracle_rng.action_fork():
            with oracle_rng.fork("bootstrap") as generator:
                with torch.no_grad(), patch("torch.randn_like", _same_random_normal(generator)):
                    next_action, _ = native.pi(batch["next_z"], None)
                    target = batch["reward"] + holder.agent.discount * (1 - batch["terminated"]) * native.Q(
                        batch["next_z"], next_action, None, return_type="min", target=True,
                    )
                logits = native.Q(batch["z"], batch["action"], None, return_type="all")
                critic_loss = cfg.tdambi_value_coef * torch.stack([
                    td_math.soft_ce(head, target, cfg).mean() for head in logits
                ]).mean()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(native._Qs.parameters(), cfg.inner_critic_grad_clip_norm)
                critic_optim.step()
                critic_optim.zero_grad(set_to_none=True)
            with oracle_rng.fork("gradient_policy") as generator:
                with patch("torch.randn_like", _same_random_normal(generator)):
                    action, info = native.pi(batch["z"], None)
                qs = native.Q(batch["z"], action, None, return_type="avg", detach=True)
                scale.update(qs)
                actor_loss = -(scale(qs) + cfg.tdambi_entropy_coef * info["scaled_entropy"]).mean()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(native._pi.parameters(), cfg.inner_actor_grad_clip_norm)
                actor_optim.step()
                actor_optim.zero_grad(set_to_none=True)
            with torch.no_grad():
                for target_parameter, parameter in zip(native._target_Qs.parameters(), native._Qs.parameters()):
                    target_parameter.lerp_(parameter, cfg.inner_critic_target_tau)

        with engine.rng.action_fork():
            with engine.rng.fork("bootstrap"):
                critic_metrics = engine._tdambi_critic_step(batch)
            with engine.rng.fork("gradient_policy"):
                actor_metrics = engine._tdambi_actor_step(batch)
            engine._maybe_update_targets(critic_updated=True, actor_updated=True)
        torch.testing.assert_close(critic_metrics["critic_loss"], critic_loss)
        torch.testing.assert_close(actor_metrics["actor_loss"], actor_loss)
        torch.testing.assert_close(state.tdambi_scale, scale.value)
        _assert_close_tree(state.actor.state_dict(), native._pi.state_dict())
        _assert_close_tree(state.critic.state_dict(), native._Qs.state_dict())
        _assert_close_tree(state.critic_target.state_dict(), native._target_Qs.state_dict())
        _assert_close_tree(state.actor_optim.state_dict(), actor_optim.state_dict())
        _assert_close_tree(state.critic_optim.state_dict(), critic_optim.state_dict())
        _assert_tree_equal(engine.rng.training_state_dict(), oracle_rng.training_state_dict())
        assert state.critic_steps == state.actor_steps == state.critic_target_steps == 1
        assert state.temperature_optim is None
    finally:
        holder.env.close()


def _snapshot(holder):
    engine = holder.agent.inner_engine
    return {
        **_pool_snapshot(holder.agent),
        "scale": _clone_tree(engine._action_pool.tdambi_scale),
        "rng": _clone_tree(engine.rng.training_state_dict()),
        "outer": _clone_tree(holder.agent.model.state_dict()),
    }


@pytest.mark.parametrize("probes", [False, True])
@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable",
    )),
])
def test_trace_non_interference_with_dropout_and_round_fidelity(probes, device):
    ordinary = _tdambi_model(dropout=0.25, device=device)
    observed = _tdambi_model(dropout=0.25, device=device)
    try:
        rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        python_rng, numpy_rng = random.getstate(), repr(np.random.get_state())
        for decision in range(2):
            obs = torch.full((3,), 0.1 * decision)
            expected = ordinary.agent.act(obs, eval_mode=True, collect_diagnostics=False)
            trace = InnerActionTrace(probes=probes, probe_rollouts=2, probe_horizon=2, probe_seed=417)
            actual = observed.agent.act(obs, eval_mode=True, collect_diagnostics=False, trace=trace)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            _assert_tree_equal(_snapshot(observed), _snapshot(ordinary))
            torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
            for actual_rng, expected_rng in zip(torch.cuda.get_rng_state_all() if cuda_rng else [], cuda_rng):
                torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)
            assert random.getstate() == python_rng
            assert repr(np.random.get_state()) == numpy_rng
            assert [m.training for m in observed.agent.model.modules()] == [m.training for m in ordinary.agent.model.modules()]
            updates = [event for event in trace.events if event["phase"] == "update"]
            assert [(event["critic_updates"], event["actor_updates"], event["temperature_updates"]) for event in updates] == [
                (1, 1, 0), (2, 2, 0), (3, 3, 0), (4, 4, 0),
            ]
            assert all(event["measurement"] == "pre_update_minibatch" for event in updates)
            assert len([event for event in trace.events if event["phase"] == "calibration"]) == 1
            assert all(not torch.is_tensor(value) for event in trace.events for value in event["metrics"].values())
            assert all("alpha" not in key and "soft_score" not in key for event in trace.events for key in event["metrics"])
            reconstructed = observed.agent.inner_engine._average_update_metrics([event["metrics"] for event in updates])
            for key, value in reconstructed.items():
                assert observed.agent.last_inner_metrics[key] == pytest.approx(float(value), abs=1e-5)
            assert observed.agent.last_inner_metrics["inner_model_steps"] == 12
            assert observed.agent.last_inner_metrics["inner_tdambi_calibration_samples"] == 4
            assert observed.agent.last_inner_metrics["inner_tdambi_calibration_seconds"] >= 0
            assert all("alpha" not in key for key in observed.agent.last_inner_metrics)
            assert trace._noise is None
        assert metric_catalog()["actor_q_scale_after"]["preferred_axis"] == "actor_updates"
    finally:
        ordinary.env.close()
        observed.env.close()


def test_decision_reset_restores_saved_target_scale_replay_and_adam_allocations():
    holder = _tdambi_model(dropout=0.25)
    try:
        engine = holder.agent.inner_engine
        outer = _clone_tree(holder.agent.model.state_dict())
        holder.agent.act(torch.zeros(3), eval_mode=True, collect_diagnostics=True)
        pool = engine._action_pool
        pointers = (_optimizer_tensor_pointers(pool.actor_optim), _optimizer_tensor_pointers(pool.critic_optim))
        scale_ptr, replay_ptr = pool.tdambi_scale.data_ptr(), pool.replay._storage.data_ptr()
        with engine.rng.action_fork():
            engine._prepare_workspace(t0=False)
            state = engine.state
            assert state.tdambi_scale.data_ptr() == scale_ptr
            assert state.tdambi_scale.item() == 1
            assert not state.tdambi_scale_initialized
            assert state.replay._storage.data_ptr() == replay_ptr
            assert state.replay.size == 0
            assert (_optimizer_tensor_pointers(state.actor_optim), _optimizer_tensor_pointers(state.critic_optim)) == pointers
            for optimizer in (state.actor_optim, state.critic_optim):
                for values in optimizer.state.values():
                    assert all(torch.count_nonzero(value) == 0 for value in values.values() if torch.is_tensor(value))
            _assert_tree_equal(state.actor.state_dict(), holder.agent.model._pi.state_dict())
            _assert_tree_equal(state.critic.state_dict(), holder.agent.model._Qs.state_dict())
            _assert_tree_equal(state.critic_target.state_dict(), holder.agent.model._target_Qs.state_dict())
            assert state.critic.training and not state.critic_target.training
            assert state.log_alpha is state.alpha_fixed is state.temperature_optim is None
        _assert_tree_equal(holder.agent.model.state_dict(), outer)
    finally:
        holder.env.close()


def test_cutoff_bootstraps_and_calibration_rng_is_separate(monkeypatch):
    holder = _tdambi_model()
    try:
        engine = holder.agent.inner_engine
        with engine.rng.action_fork():
            engine._prepare_workspace(t0=True)
            root = holder.agent.model.encode(torch.zeros(1, 3)).detach()
            engine._collect_round(root)
        replay = engine.state.replay
        assert replay.size == 6
        assert torch.count_nonzero(replay.terminated[:replay.size]) == 0
        assert not replay.store_horizon
        before = engine.rng.training_state_dict()
        engine._calibrate_tdambi_scale()
        after = engine.rng.training_state_dict()
        for name in before["streams"]:
            if name != "tdambi_calibration":
                _assert_tree_equal(after["streams"][name], before["streams"][name])
                _assert_tree_equal(after["phase_streams"][name], before["phase_streams"][name])
        calibrated = engine.state.tdambi_scale.clone()
        engine._calibrate_tdambi_scale()
        _assert_tree_equal(engine.rng.training_state_dict(), after)
        torch.testing.assert_close(engine.state.tdambi_scale, calibrated)
        batch = engine._sample_batch()
        batch["reward"] = torch.full_like(batch["reward"], 0.25)
        original = engine.model.q_values
        def fixed_target(z, action, **kwargs):
            if kwargs.get("qs") is engine.state.critic_target:
                return z.new_full((holder.cfg.num_q, z.shape[0], 1), 7.0)
            return original(z, action, **kwargs)
        monkeypatch.setattr(engine.model, "q_values", fixed_target)
        with engine.rng.fork("bootstrap"):
            metrics = engine._tdambi_critic_step(batch)
        assert float(metrics["q_target_mean"]) == pytest.approx(0.25 + 0.99 * 7)
    finally:
        holder.env.close()


def test_raw_trace_reuses_model_forwards_and_legacy_rng_stream_schema(monkeypatch):
    holder = _tdambi_model(dropout=0.25)
    counts = {}
    try:
        for name in ("pi_action", "pi_tdmpc2", "Q", "q_values", "q_predictions", "next_from_joint", "reward_from_joint"):
            original = getattr(holder.agent.model, name)
            def counted(*args, _name=name, _original=original, **kwargs):
                counts[_name] = counts.get(_name, 0) + 1
                return _original(*args, **kwargs)
            monkeypatch.setattr(holder.agent.model, name, counted)
        holder.agent.act(torch.zeros(3), eval_mode=True, collect_diagnostics=False)
        expected = dict(counts)
        counts.clear()
        holder.agent.act(torch.zeros(3), eval_mode=True, collect_diagnostics=False, trace=InnerActionTrace())
        assert counts == expected
        legacy = InnerRNG(13, "cpu").training_state_dict()
        assert "tdambi_calibration" not in legacy["streams"]
        assert tuple(legacy["streams"]) == InnerRNG.STREAMS
    finally:
        holder.env.close()


def test_scale_calibration_uses_frozen_online_prior_and_percentile_range(monkeypatch):
    holder = _tdambi_model(dropout=0.25)
    try:
        engine = holder.agent.inner_engine
        with engine.rng.action_fork():
            engine._prepare_workspace(t0=True)
            root = holder.agent.model.encode(torch.zeros(1, 3)).detach()
            engine._collect_round(root)
        outer = engine.model
        # Mixed module modes must survive the temporary calibration eval mode.
        outer._Qs.train()
        outer._Qs[0][0].eval()
        modes = [module.training for module in outer.modules()]
        calls = []
        original_policy = outer.pi_action
        def prior_action(z, **kwargs):
            assert "policy" not in kwargs
            assert not outer._pi.training
            assert not torch.is_grad_enabled()
            calls.append("prior")
            return original_policy(z, **kwargs)
        def online_values(z, action, **kwargs):
            assert not kwargs  # No target or local critic passed.
            assert not outer._Qs.training
            assert not torch.is_grad_enabled()
            calls.append("online_q")
            rows = torch.arange(z.shape[0], device=z.device).reshape(1, -1, 1) * 10.0
            heads = torch.arange(holder.cfg.num_q, device=z.device).reshape(-1, 1, 1) * 2.0
            return rows + heads
        monkeypatch.setattr(outer, "pi_action", prior_action)
        monkeypatch.setattr(outer, "q_values", online_values)
        engine._calibrate_tdambi_scale()
        assert calls == ["prior", "online_q"]
        # For [0,10,20,30], linear P95-P5 = 28.5-1.5. Head offsets cancel.
        assert engine.state.tdambi_scale.item() == pytest.approx(27.0)
        assert [module.training for module in outer.modules()] == modes
    finally:
        holder.env.close()
