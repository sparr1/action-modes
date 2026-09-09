"""Literal upstream statistic and preservation of AMBI's original policy API."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from RL.tdmpc2_core.common import math
from RL.tdmpc2_core.common.soft_world_model import SoftWorldModel
from tests.test_ambi_config_decoupling import _build_cfg


def _upstream_reference(mean, log_std, eps):
    # Independent transcription of official 8bbc14e world_model.pi/math.squash.
    ell_g = (-0.5 * eps.pow(2) - log_std - 0.9189385175704956).sum(-1, keepdim=True)
    action = torch.tanh(mean + eps * log_std.exp())
    scaled_log_prob = ell_g * eps.shape[-1]
    ell_td = ell_g - torch.log(torch.relu(1 - action.pow(2)) + 1e-6).sum(-1, keepdim=True)
    entropy_scale = scaled_log_prob / (ell_td + 1e-8)
    return -ell_td * entropy_scale


@pytest.mark.parametrize("action_dim", [1, 3, 21])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scaled_entropy_and_gradients_match_literal_upstream_with_saturation(action_dim, dtype):
    mean = torch.tensor([0., 2., 9., -30.], dtype=dtype).reshape(4, 1).repeat(1, action_dim).requires_grad_()
    log_std = torch.linspace(-4., 1., 4 * action_dim, dtype=dtype).reshape(4, action_dim).requires_grad_()
    eps = torch.linspace(-1.5, 1.7, 4 * action_dim, dtype=dtype).reshape(4, action_dim)
    action = (mean + eps * log_std.exp()).tanh()
    assert (action.abs() == 1).any()
    actual = math.tdmpc2_scaled_entropy(math.gaussian_logprob(eps, log_std), action)
    reference = _upstream_reference(mean, log_std, eps)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    gradients = torch.autograd.grad(actual.sum(), (mean, log_std), retain_graph=True)
    expected = torch.autograd.grad(reference.sum(), (mean, log_std))
    for gradient, oracle in zip(gradients, expected):
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, oracle, rtol=0, atol=0)


def test_ratio_is_not_cancelled_or_detached_near_zero_td_log_probability():
    mean = torch.zeros(1, 1, dtype=torch.float64, requires_grad=True)
    log_std = torch.tensor([[-0.9189385175704956 - torch.log(torch.tensor(1.000001, dtype=torch.float64)).item() - 5e-9]], dtype=torch.float64, requires_grad=True)
    eps = torch.zeros_like(mean)
    ell_g = math.gaussian_logprob(eps, log_std)
    actual = math.tdmpc2_scaled_entropy(ell_g, mean.tanh())
    reference = _upstream_reference(mean, log_std, eps)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert not torch.allclose(actual, -ell_g, rtol=1e-2, atol=0)
    gradient, = torch.autograd.grad(actual.sum(), (log_std,), retain_graph=True)
    oracle, = torch.autograd.grad(reference.sum(), (log_std,))
    torch.testing.assert_close(gradient, oracle, rtol=0, atol=0)
    assert abs(gradient.item() - 1.0) > 1.0


def _world(action_dim):
    cfg = _build_cfg(
        model_size=None, latent_dim=16, enc_dim=32, mlp_dim=32,
        num_enc_layers=2, simnorm_dim=8, num_bins=11,
    )
    cfg.action_dim = action_dim
    return SoftWorldModel(cfg).eval()


def _legacy_pi(world, z):
    # The public pi implementation before adding the optional statistic.
    mean_raw, log_std, eps = world._policy_sample(z, None)
    log_prob = math.gaussian_logprob(eps, log_std)
    pre_tanh_action = mean_raw + eps * log_std.exp()
    mean, action, log_prob = math.squash(mean_raw, pre_tanh_action, log_prob)
    return action, {
        "mean": mean, "pre_tanh_mean": mean_raw, "pre_tanh_action": pre_tanh_action,
        "log_std": log_std, "log_prob": log_prob, "entropy": -log_prob,
    }


@pytest.mark.parametrize("action_dim", [1, 3])
@pytest.mark.parametrize("include_scaled", [None, False, True])
def test_policy_default_values_gradients_and_rng_are_unchanged(action_dim, include_scaled):
    world = _world(action_dim)
    reference_world = deepcopy(world)
    z = torch.randn(3, 4, world.cfg.latent_dim)
    start = torch.random.get_rng_state()
    with patch.object(math, "tdmpc2_scaled_entropy", wraps=math.tdmpc2_scaled_entropy) as scaled:
        kwargs = {} if include_scaled is None else {"include_scaled_entropy": include_scaled}
        action, info = world.pi(z, **kwargs)
        assert scaled.call_count == int(include_scaled is True)
    end = torch.random.get_rng_state()
    torch.random.set_rng_state(start)
    action_ref, info_ref = _legacy_pi(reference_world, z)
    assert torch.equal(end, torch.random.get_rng_state())
    assert set(info) == set(info_ref) | ({"scaled_entropy"} if include_scaled else set())
    torch.testing.assert_close(action, action_ref, rtol=0, atol=0)
    for key in info_ref:
        torch.testing.assert_close(info[key], info_ref[key], rtol=0, atol=0)
    (action.square().mean() + .1 * info["log_prob"].mean()).backward()
    (action_ref.square().mean() + .1 * info_ref["log_prob"].mean()).backward()
    for param, reference in zip(world._pi.parameters(), reference_world._pi.parameters()):
        torch.testing.assert_close(param.grad, reference.grad, rtol=0, atol=0)


def test_policy_scaled_entropy_reuses_supplied_sample_and_matches_oracle():
    world = _world(3)
    z = torch.randn(2, 4, world.cfg.latent_dim)
    noise = torch.randn(2, 4, 3)
    before = torch.random.get_rng_state()
    with patch.object(world, "_policy_sample", wraps=world._policy_sample) as sample:
        _, info = world.pi(z, noise=noise, include_scaled_entropy=True)
    assert sample.call_count == 1
    assert torch.equal(before, torch.random.get_rng_state())
    torch.testing.assert_close(
        info["scaled_entropy"],
        _upstream_reference(info["pre_tanh_mean"], info["log_std"], noise),
        rtol=0, atol=0,
    )
