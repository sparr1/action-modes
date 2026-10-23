"""Root entropy observes the adapted controller without changing learning."""
from copy import deepcopy
import math

import numpy as np
import pytest
import torch

from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_component_model


def _model(**overrides):
    return _tiny_component_model(
        inner_critic_updates_per_round=1, inner_actor_updates_per_round=1,
        **overrides,
    )


@pytest.mark.parametrize("value", [-1, 1, True, 2.5, "128"])
def test_invalid_entropy_sample_count(value):
    with pytest.raises(ValueError, match="inner_root_entropy_samples"):
        _build_cfg(inner_root_entropy_samples=value)


def test_default_and_disabled_sample_counts():
    assert _build_cfg().inner_root_entropy_samples == 128
    assert _build_cfg(inner_root_entropy_samples=0).inner_root_entropy_samples == 0


@pytest.mark.parametrize("mean, log_std", [(0., 0.), (1., -.7), (15., -2.)])
def test_root_entropy_matches_independent_gaussian_quadrature(mean, log_std, monkeypatch):
    model = _model()
    try:
        engine = model.agent.inner_engine
        with engine.rng.fork("initialization"):
            engine._prepare_workspace(t0=True)
        # Midpoint normal quantiles give an accurate deterministic integral;
        # the reference uses a separate Gauss-Hermite quadrature rule.
        probabilities = (torch.arange(32768, dtype=torch.float64) + .5) / 32768
        noise = torch.distributions.Normal(0., 1.).icdf(probabilities).reshape(-1, 1, 1)
        monkeypatch.setattr(engine.model, "policy_stats", lambda *args, **kwargs: {
            "pre_tanh_mean": torch.tensor([[mean]], dtype=torch.float64),
            "log_std": torch.tensor([[log_std]], dtype=torch.float64),
        })
        samples, _ = engine._root_entropy_samples(
            torch.zeros(1, model.cfg.latent_dim), engine.state.actor, noise
        )
        nodes, weights = np.polynomial.hermite.hermgauss(100)
        x = mean + math.exp(log_std) * math.sqrt(2) * nodes
        correction = -2 * (np.logaddexp(x, -x) - math.log(2))
        expected = log_std + .5 * math.log(2 * math.pi * math.e)
        expected += float(weights @ correction) / math.sqrt(math.pi)
        assert samples.mean().item() == pytest.approx(expected, abs=2e-4)
        assert torch.isfinite(samples).all()
        assert not samples.requires_grad
    finally:
        model.env.close()


def test_entropy_observation_preserves_actions_updates_and_every_rng_stream():
    disabled = _model(inner_root_entropy_samples=0, dropout=.2)
    observed = _model(inner_root_entropy_samples=128, dropout=.2)
    try:
        outer_before = _clone_tree(observed.agent.model.state_dict())
        global_before = torch.random.get_rng_state().clone()
        for decision in range(2):
            a = disabled.agent.act(torch.zeros(3), t0=decision == 0, eval_mode=False)
            b = observed.agent.act(torch.zeros(3), t0=decision == 0, eval_mode=False)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            left, right = disabled.agent.inner_engine, observed.agent.inner_engine
            for name in left.rng.STREAMS:
                torch.testing.assert_close(left.rng.generator(name).get_state(), right.rng.generator(name).get_state())
                torch.testing.assert_close(left.rng.phase_generators[name].get_state(), right.rng.phase_generators[name].get_state())
            _assert_tree_equal(left._action_pool.actor.state_dict(), right._action_pool.actor.state_dict())
            _assert_tree_equal(left._action_pool.critic.state_dict(), right._action_pool.critic.state_dict())
            _assert_tree_equal(left._action_pool.actor_optim.state_dict(), right._action_pool.actor_optim.state_dict())
            metrics = observed.agent.last_inner_metrics
            assert metrics['inner_root_entropy_samples'] == 128
            assert metrics['inner_root_entropy_delta'] == pytest.approx(
                metrics['inner_root_entropy_final'] - metrics['inner_root_entropy_initial'], abs=1e-6)
            assert metrics['inner_policy_evaluations'] == disabled.agent.last_inner_metrics['inner_policy_evaluations'] + 2
            for key, value in disabled.agent.last_inner_metrics.items():
                if key != 'inner_policy_evaluations' and not key.endswith('seconds'):
                    assert metrics[key] == value, key
        _assert_tree_equal(observed.agent.model.state_dict(), outer_before)
        torch.testing.assert_close(torch.random.get_rng_state(), global_before, rtol=0, atol=0)
    finally:
        disabled.env.close()
        observed.env.close()


def test_snapshots_surround_actual_updates_and_use_root_horizon(monkeypatch):
    model = _model(inner_horizon_conditioning='one_hot', inner_finite_horizon=True)
    try:
        engine = model.agent.inner_engine
        original = engine._root_entropy_samples
        observations = []
        original_stats = engine.model.policy_stats

        def stats(z, *args, **kwargs):
            assert torch.all(kwargs['remaining_horizon'] == model.cfg.inner_rollout_horizon)
            assert not kwargs['policy'].training
            return original_stats(z, *args, **kwargs)

        def capture(root, policy, noise=None):
            modes = [module.training for module in policy.modules()]
            result = original(root, policy, noise)
            assert [module.training for module in policy.modules()] == modes
            observations.append((root.detach().clone(), engine.state.actor_steps, result))
            return result

        monkeypatch.setattr(engine.model, 'policy_stats', stats)
        monkeypatch.setattr(engine, '_root_entropy_samples', capture)
        model.agent.act(torch.zeros(3), t0=True)
        before, after = observations
        torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
        assert before[1] == 0
        assert after[1] == model.cfg.inner_rounds * model.cfg.inner_actor_updates_per_round
        assert before[2][1] is after[2][1]
        metrics = model.agent.last_inner_metrics
        assert metrics['inner_root_entropy_initial'] == pytest.approx(before[2][0].mean().item())
        assert metrics['inner_root_entropy_final'] == pytest.approx(after[2][0].mean().item())
        delta = after[2][0] - before[2][0]
        assert metrics['inner_root_entropy_delta_mc_se'] == pytest.approx(delta.std().item() / math.sqrt(delta.numel()))
    finally:
        model.env.close()


def test_frozen_actor_has_exactly_zero_paired_entropy_change():
    model = _tiny_component_model(inner_actor_adaptation='frozen',
        inner_actor_updates_per_round=0, inner_critic_updates_per_round=1)
    try:
        model.agent.act(torch.zeros(3), t0=True)
        metrics = model.agent.last_inner_metrics
        assert metrics['inner_root_entropy_delta'] == 0
        assert metrics['inner_root_entropy_delta_mc_se'] == 0
    finally:
        model.env.close()


def test_unsampled_and_disabled_decisions_omit_root_entropy():
    model = _model()
    try:
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        assert not any(k.startswith('inner_root_entropy') for k in model.agent.last_inner_metrics)
        model.cfg.inner_root_entropy_samples = 0
        model.agent.act(torch.zeros(3), collect_diagnostics=True)
        assert not any(k.startswith('inner_root_entropy') for k in model.agent.last_inner_metrics)
    finally:
        model.env.close()


def test_training_window_weights_root_entropy_by_observed_decisions():
    model = _model()
    try:
        model.agent.act(torch.zeros(3))
        original = deepcopy(model.agent.last_inner_metrics)
        for initial, final, updates in [(2., 1., 1), (6., 4., 99)]:
            model.agent.last_inner_metrics = {**original,
                'inner_root_entropy_initial': initial, 'inner_root_entropy_final': final,
                'inner_root_entropy_delta': final-initial, 'inner_actor_optimizer_steps': updates}
            model._record_action_metrics(planned=True, action_seconds=0.)
        # A diagnostic-free decision must not create a zero observation.
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        model._record_action_metrics(planned=True, action_seconds=0.)
        payload = model._wandb_train_window.pop()
        assert payload['train/inner_root_entropy_initial'] == 4.
        assert payload['train/inner_root_entropy_final'] == 2.5
        assert payload['train/inner_root_entropy_delta'] == -1.5
        assert payload['train/inner_root_entropy_final_count'] == 2
        assert payload['train/inner_root_entropy_final_std'] == 1.5
        assert payload['train/inner_root_entropy_final_min'] == 1.
        assert payload['train/inner_root_entropy_final_max'] == 4.
        assert 'train/inner_root_entropy_final_mc_se' in payload
    finally:
        model.env.close()
