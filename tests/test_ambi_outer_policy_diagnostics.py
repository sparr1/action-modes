"""Outer diagnostics observe the exact learner sample without changing training."""

from copy import deepcopy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.outer_policy_diagnostics import diagnostics_due, policy_diagnostics
from tests.test_ambi_config_decoupling import _build_cfg


def _rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def _equal(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _equal(left, right)
    else:
        assert actual == expected


def test_cadence_disabled_initial_early_and_late_boundaries():
    assert not diagnostics_due(SimpleNamespace(), 0)
    cfg = SimpleNamespace(outer_policy_diagnostics=True)
    for value in (0, 100, 9900, 10000, 11000):
        assert diagnostics_due(cfg, value)
    for value in (1, 99, 9999, 10001, 10100):
        assert not diagnostics_due(cfg, value)
    assert diagnostics_due({"outer_policy_diagnostics": True, "outer_policy_diagnostics_early_every": 2}, 2)
    with pytest.raises(ValueError, match="cadences"):
        diagnostics_due({"outer_policy_diagnostics": True, "outer_policy_diagnostics_early_every": 0}, 1)


def test_analytic_coordinate_metrics_histograms_and_entropy_are_detached():
    mu = torch.tensor([[[0., 10., -3.]], [[0., 0., 0.]]], requires_grad=True)
    log_std = torch.tensor([[[-10., -9.95, 2.]], [[1.95, 0., -4.]]], requires_grad=True)
    action = torch.tensor([[[1., 0., -0.99]], [[0., 0., 0.]]], requires_grad=True)
    log_prob = torch.tensor([[[2.]], [[8.]]], requires_grad=True)
    info = {"mean": mu.tanh(), "pre_tanh_mean": mu, "log_std": log_std, "log_prob": log_prob}
    before = _rng_state()
    packet = policy_diagnostics(info, action, lower=-10, upper=2, rho=0.5)
    _equal(_rng_state(), before)
    metrics = packet["metrics"]
    assert metrics["depth0_mean_action_abs_ge_0p99_fraction"] == pytest.approx(2 / 3)
    assert metrics["depth0_mean_action_exact_saturation_fraction"] == pytest.approx(1 / 3)
    assert metrics["depth0_sample_action_exact_saturation_fraction"] == pytest.approx(1 / 3)
    assert metrics["pooled_sample_action_abs_ge_0p99_fraction"] == pytest.approx(2 / 6)
    assert metrics["pooled_log_std_lower_exact_fraction"] == pytest.approx(1 / 6)
    assert metrics["pooled_log_std_lower_near_0p1_fraction"] == pytest.approx(2 / 6)
    assert metrics["pooled_log_std_upper_exact_fraction"] == pytest.approx(1 / 6)
    assert metrics["pooled_log_std_upper_near_0p1_fraction"] == pytest.approx(2 / 6)
    torch.testing.assert_close(metrics["pooled_std_mean"], log_std.exp().mean())
    torch.testing.assert_close(metrics["pooled_log_std_std"], log_std.std(unbiased=False))
    assert metrics["pooled_coordinate_count"] == 6
    assert metrics["depth0_coordinate_count"] == 3
    assert metrics["entropy_rho_mean"] == pytest.approx(-4.)
    assert metrics["entropy_unweighted_mean"] == pytest.approx(-5.)
    assert metrics["entropy_actor_loss_weighted"] == pytest.approx(-3.)
    for name, expected in (("pooled", 6), ("depth0", 3)):
        histogram = packet["histograms"]["log_std_" + name]
        assert histogram["counts"].shape == (24,)
        assert histogram["edges"].shape == (25,)
        assert histogram["count"] == expected
        assert histogram["counts"].sum() == expected
        assert histogram["edges"][0] == -10
        assert histogram["edges"][-1] == 2
    assert all(value.ndim == 0 and not value.requires_grad for value in metrics.values())
    assert all(value.grad is None for value in (mu, log_std, action, log_prob))
    assert mu.requires_grad and log_std.requires_grad


def test_single_depth_input_matches_explicit_depth_and_rejects_bad_shapes():
    info = {
        "mean": torch.zeros(2, 3), "pre_tanh_mean": torch.zeros(2, 3),
        "log_std": torch.zeros(2, 3), "log_prob": torch.ones(2, 1),
    }
    action = torch.zeros(2, 3)
    observed = policy_diagnostics(info, action, lower=-10, upper=2, rho=0.5)
    expected = policy_diagnostics({key: value[None] for key, value in info.items()}, action[None], lower=-10, upper=2, rho=0.5)
    _equal(observed, expected)
    with pytest.raises(ValueError, match="joint density"):
        policy_diagnostics({**info, "log_prob": torch.ones(2, 3)}, action, lower=-10, upper=2, rho=0.5)


def _agent(device, mapping, target, enabled, *, compiled=False):
    cfg = _build_cfg(
        compile=compiled, compile_strict=compiled, wandb=False, ent_coef="auto_1.0", target_entropy=target,
        train_unroll_horizon=3, outer_critic_target="entropy_augmented",
        outer_q_actor_reduction="mean_pair", outer_q_target_reduction="min_pair",
        outer_actor_entropy_mode="squashed", log_std_mapping=mapping,
        log_std_min=-10, log_std_max=2, inner_operator="none", dropout=0.01,
    )
    cfg.enc_dim = cfg.mlp_dim = 16
    cfg.latent_dim = 16
    cfg.action_dim = 3
    cfg.batch_size = 4
    cfg.device = device
    cfg.outer_policy_diagnostics = enabled
    cfg.outer_policy_diagnostics_early_every = 1
    return AMBITDMPC2Agent(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable: strict compiled diagnostic coverage requires a compatible GPU runtime")
@pytest.mark.parametrize("mapping", ["direct_clamp", "tdmpc2_tanh"])
def test_cuda_compiled_updates_and_bank_probes_preserve_training(tmp_path, mapping):
    from utils.outer_policy_diagnostics import OuterPolicyDiagnostics

    torch.manual_seed(912)
    plain = _agent("cuda", mapping, -21., False, compiled=True)
    torch.manual_seed(912)
    observed = _agent("cuda", mapping, -21., True, compiled=True)
    observed.cfg.outer_policy_diagnostics_states = 32
    observed.cfg.outer_policy_diagnostics_samples = 32
    recorder = OuterPolicyDiagnostics(observed.cfg, tmp_path)
    for index in recorder.indices:
        recorder.observe(torch.zeros(3), index)
    obs = torch.randn(4, 4, 3, device="cuda")
    actions = torch.randn(3, 4, 3, device="cuda").tanh()
    reward = torch.randn(3, 4, 1, device="cuda")
    terminated = torch.zeros_like(reward)

    class Replay:
        draws = 0

        def sample(self):
            self.draws += 1
            return obs, actions, reward, terminated, None

    replay_off, replay_on = Replay(), Replay()
    for completed in (1, 2):
        before = _rng_state()
        expected = deepcopy(plain.update(replay_off))
        expected_rng = _rng_state()
        _restore_rng(before)
        actual = observed.update(replay_on)
        packet = observed.drain_outer_policy_diagnostics()
        assert packet is not None
        recorder.learner(packet, env_step=7, updates=completed, phase="pretraining", run=None)
        recorder.probe(observed, env_step=7, updates=completed, phase="pretraining", run=None)
        _equal(_rng_state(), expected_rng)
        _equal(actual, expected)
        _equal(observed.state_dict(), plain.state_dict())
        for key in ("optim", "pi_optim", "ent_coef_optim"):
            _equal(getattr(observed, key).state_dict(), getattr(plain, key).state_dict())
        for left, right in zip(observed.parameters(), plain.parameters()):
            _equal(left.grad, right.grad)
        assert [module.training for module in observed.modules()] == [module.training for module in plain.modules()]
    assert replay_off.draws == replay_on.draws == 2
    assert observed._outer_update_region.enabled and not observed._outer_update_region.failed


@pytest.mark.parametrize("mapping,target", [("direct_clamp", -21.), ("direct_clamp", -10.5), ("tdmpc2_tanh", -21.), ("tdmpc2_tanh", -10.5)])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable: CPU checks do not establish GPU readiness"))])
def test_full_updates_preserve_rng_forward_counts_weights_optimizers_and_modes(mapping, target, device):
    torch.manual_seed(123)
    plain = _agent(device, mapping, target, False)
    torch.manual_seed(123)
    observed = _agent(device, mapping, target, True)
    captured = {"plain": [], "observed": []}
    hooks = []
    for name, agent in (("plain", plain), ("observed", observed)):
        for component in ("_encoder", "_dynamics", "_reward", "_pi", "_Qs", "_target_Qs"):
            module = getattr(agent.model, component)
            # ModuleDict itself is not called when selecting the state encoder.
            if component == "_encoder":
                module = module["state"]
            hooks.append(module.register_forward_hook(lambda _m, _i, _o, key=name, part=component: captured[key].append(part)))
    obs = torch.randn(4, 4, 3, device=device)
    actions = torch.randn(3, 4, 3, device=device).tanh()
    reward = torch.randn(3, 4, 1, device=device)
    terminated = torch.zeros_like(reward)
    inner_rng = deepcopy(observed.inner_engine.rng.training_state_dict())
    try:
        for completed in (1, 2):
            before_rng = _rng_state()
            expected_metrics = plain._update(obs, actions, reward, terminated)
            expected_rng = _rng_state()
            _restore_rng(before_rng)
            actual_metrics = observed._update(obs, actions, reward, terminated)
            _equal(_rng_state(), expected_rng)
            _equal(actual_metrics, expected_metrics)
            _equal(observed.state_dict(), plain.state_dict())
            for name in ("optim", "pi_optim", "ent_coef_optim"):
                _equal(getattr(observed, name).state_dict(), getattr(plain, name).state_dict())
            _equal(observed.inner_engine.rng.training_state_dict(), inner_rng)
            for left, right in zip(observed.parameters(), plain.parameters()):
                _equal(left.grad, right.grad)
            assert [module.training for module in observed.modules()] == [module.training for module in plain.modules()]
            assert captured["plain"] == captured["observed"]
            packet = observed.drain_outer_policy_diagnostics()
            assert observed.drain_outer_policy_diagnostics() is None
            assert plain.drain_outer_policy_diagnostics() is None
            assert packet["actor_updates_before"] == completed - 1
            assert packet["actor_updates_after"] == completed
            assert packet["policy_snapshot"] == "before_actor_update"
            assert packet["latent_snapshot"] == "before_model_critic_update"
            assert packet["actor_q_snapshot"] == "after_model_critic_update"
            assert packet["diagnostic_collection_seconds"] >= 0
            assert packet["diagnostic_collection_timing"] == "host_enqueue_without_device_synchronization"
            metrics = packet["metrics"]
            torch.testing.assert_close(metrics["alpha_before"], actual_metrics["ent_coef"].reshape(()))
            torch.testing.assert_close(metrics["alpha_after"], observed.alpha.reshape(()))
            torch.testing.assert_close(metrics["temperature_loss"], actual_metrics["ent_coef_loss"])
            torch.testing.assert_close(metrics["entropy_shortfall"], target - metrics["entropy_rho_mean"])
            for key in ("critic_loss", "reward_loss", "consistency_loss", "total_loss", "q_target_mean", "q_mean", "q_abs_mean", "td_error_abs_mean", "q_target_clip_fraction"):
                torch.testing.assert_close(metrics[key], actual_metrics[key])
            torch.testing.assert_close(metrics["world_critic_grad_norm"], actual_metrics["grad_norm"])
            assert all(not value.requires_grad and value.ndim == 0 for value in metrics.values())
    finally:
        for hook in hooks:
            hook.remove()


def test_force_is_single_use_disabled_remains_off_and_failure_clears_packet(monkeypatch):
    agent = _agent("cpu", "direct_clamp", -21., True)
    agent.cfg.outer_policy_diagnostics_early_every = 100
    zs = torch.zeros(4, 4, 16)
    agent._outer_policy_diagnostics_force = True
    agent._update_actor(zs)
    assert not agent._outer_policy_diagnostics_force
    assert agent.drain_outer_policy_diagnostics() is not None
    agent._update_actor(zs)
    assert agent.drain_outer_policy_diagnostics() is None
    agent.cfg.outer_policy_diagnostics = False
    agent._outer_policy_diagnostics_force = True
    agent._update_actor(zs)
    assert agent.drain_outer_policy_diagnostics() is None
    assert not agent._outer_policy_diagnostics_force

    agent.cfg.outer_policy_diagnostics = True
    agent._outer_policy_diagnostics_force = True
    agent._update_actor(zs)
    assert agent._outer_policy_diagnostics_packet is not None
    def fail(*_args, **_kwargs):
        raise RuntimeError("injected policy failure")
    monkeypatch.setattr(agent.model, "pi", fail)
    agent._outer_policy_diagnostics_force = True
    with pytest.raises(RuntimeError, match="injected"):
        agent._update_actor(zs)
    assert agent.drain_outer_policy_diagnostics() is None
    assert not agent._outer_policy_diagnostics_force
