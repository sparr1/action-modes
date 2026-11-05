"""Independent density and gradient checks for the reference entropy ablation."""
import math

import pytest
import torch

from slurm.boom_entropy_ablation import corrected_log_probability


@pytest.mark.parametrize("dimension", [1, 21])
def test_joint_density_matches_torch_transformed_distribution(dimension):
    generator = torch.Generator().manual_seed(55)
    mean = torch.randn(4, dimension, generator=generator, dtype=torch.float64) * 0.2
    log_std = torch.full_like(mean, -0.5)
    eps = torch.randn(mean.shape, generator=generator, dtype=torch.float64)
    pre_tanh = mean + log_std.exp() * eps
    distribution = torch.distributions.TransformedDistribution(
        torch.distributions.Independent(torch.distributions.Normal(mean, log_std.exp()), 1),
        [torch.distributions.TanhTransform(cache_size=1)],
    )
    expected = distribution.log_prob(pre_tanh.tanh()).unsqueeze(-1)
    actual = corrected_log_probability(eps, log_std, pre_tanh)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mean_value,log_std_value", [(0.0, -4.0), (0.3, 0.0), (30.0, 2.0), (-30.0, 2.0)])
def test_pathwise_gradient_is_exact_and_finite_in_saturation(mean_value, log_std_value):
    mean = torch.full((3, 21), mean_value, dtype=torch.float64, requires_grad=True)
    log_std = torch.full_like(mean, log_std_value, requires_grad=True)
    eps = torch.linspace(-2, 2, 63, dtype=torch.float64).reshape(3, 21)
    pre_tanh = mean + log_std.exp() * eps
    log_prob = corrected_log_probability(eps, log_std, pre_tanh)
    grad_mean, grad_log_std = torch.autograd.grad(log_prob.sum(), (mean, log_std))
    assert torch.isfinite(log_prob).all()
    torch.testing.assert_close(grad_mean, 2 * pre_tanh.tanh(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad_log_std, -1 + 2 * pre_tanh.tanh() * eps * log_std.exp(), rtol=1e-12, atol=1e-12)


def test_dimension_matched_coefficient_preserves_gaussian_pressure():
    dimension, released_alpha = 21, 1e-4
    mean = torch.zeros(2, dimension, dtype=torch.float64)
    log_std = torch.zeros_like(mean, requires_grad=True)
    eps = torch.ones_like(mean) * 0.1
    pre_tanh = mean + log_std.exp() * eps
    gaussian = (-0.5 * eps.square() - log_std - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
    jacobian = (2 * (math.log(2) - pre_tanh - torch.nn.functional.softplus(-2 * pre_tanh))).sum(-1, keepdim=True)
    native = released_alpha * (dimension * gaussian - jacobian)
    corrected = dimension * released_alpha * corrected_log_probability(eps, log_std, pre_tanh)
    # Up to the native Gaussian helper's constant, the sole difference is the
    # stronger exact Jacobian coefficient; Gaussian entropy pressure matches.
    torch.testing.assert_close(corrected - native, -(dimension - 1) * released_alpha * jacobian)
