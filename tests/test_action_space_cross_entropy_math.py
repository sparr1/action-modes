import pytest
import torch

from RL.tdmpc2_core.common import math as td_math


def test_tanh_log_abs_det_jacobian_matches_pytorch_at_extreme_inputs():
    pre_tanh = torch.tensor(
        [[-100.0, -20.0, -3.0, 0.0, 3.0, 20.0, 100.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    transform = torch.distributions.transforms.TanhTransform()
    expected = transform.log_abs_det_jacobian(pre_tanh, torch.tanh(pre_tanh))

    elementwise = td_math.tanh_log_abs_det_jacobian(
        pre_tanh,
        sum_action_dim=False,
    )
    torch.testing.assert_close(elementwise, expected, rtol=0, atol=1e-14)
    assert torch.isfinite(elementwise).all()
    assert elementwise[0, 0] < -190.0
    assert elementwise[0, -1] < -190.0

    summed = td_math.tanh_log_abs_det_jacobian(pre_tanh)
    torch.testing.assert_close(summed, expected.sum(dim=-1, keepdim=True))
    summed.sum().backward()
    torch.testing.assert_close(
        pre_tanh.grad,
        -2.0 * torch.tanh(pre_tanh.detach()),
        rtol=1e-12,
        atol=1e-12,
    )


def test_diagonal_gaussian_cross_entropy_matches_torch_identity_and_broadcasting():
    current_mean = torch.tensor(
        [[[0.2, -0.3, 0.7]], [[-0.5, 0.1, 0.4]]], dtype=torch.float64
    )
    current_log_std = torch.tensor(
        [[[-1.2, 0.3, -0.7]], [[0.2, -0.4, 0.8]]], dtype=torch.float64
    )
    behavior_mean = torch.tensor(
        [
            [[0.0, 0.5, -0.2], [0.4, -0.1, 0.8]],
            [[-0.2, 0.3, 0.1], [0.9, -0.7, 0.0]],
        ],
        dtype=torch.float64,
    )
    behavior_log_std = torch.tensor(
        [
            [[-0.5, 0.1, -1.0], [0.6, -0.8, 0.2]],
            [[-1.1, 0.7, -0.3], [0.0, -0.2, 0.4]],
        ],
        dtype=torch.float64,
    )
    current = torch.distributions.Normal(current_mean, current_log_std.exp())
    behavior = torch.distributions.Normal(
        behavior_mean,
        behavior_log_std.exp(),
    )
    expected = current.entropy() + torch.distributions.kl_divergence(
        current,
        behavior,
    )

    elementwise = td_math.diagonal_gaussian_cross_entropy(
        current_mean,
        current_log_std,
        behavior_mean,
        behavior_log_std,
        sum_action_dim=False,
    )
    torch.testing.assert_close(elementwise, expected)
    assert elementwise.shape == (2, 2, 3)
    torch.testing.assert_close(
        td_math.diagonal_gaussian_cross_entropy(
            current_mean,
            current_log_std,
            behavior_mean,
            behavior_log_std,
        ),
        expected.sum(dim=-1, keepdim=True),
    )


def test_squashed_action_cross_entropy_estimator_matches_transformed_nll():
    dtype = torch.float64
    current_mean = torch.tensor([0.3, -0.4], dtype=dtype)
    current_log_std = torch.tensor([-0.2, 0.1], dtype=dtype)
    behavior_mean = torch.tensor([-0.1, 0.2], dtype=dtype)
    behavior_log_std = torch.tensor([0.15, -0.35], dtype=dtype)
    generator = torch.Generator().manual_seed(741)
    eps = torch.randn(200_000, 2, generator=generator, dtype=dtype)
    pre_tanh = current_mean + eps * current_log_std.exp()
    action = torch.tanh(pre_tanh)

    gaussian_ce = td_math.diagonal_gaussian_cross_entropy(
        current_mean,
        current_log_std,
        behavior_mean,
        behavior_log_std,
    )
    estimate = (
        gaussian_ce
        + td_math.tanh_log_abs_det_jacobian(pre_tanh).mean(dim=0)
    ).squeeze()
    transformed_behavior = torch.distributions.TransformedDistribution(
        torch.distributions.Independent(
            torch.distributions.Normal(
                behavior_mean,
                behavior_log_std.exp(),
            ),
            1,
        ),
        [torch.distributions.transforms.TanhTransform(cache_size=1)],
    )
    reference = -transformed_behavior.log_prob(action).mean()

    torch.testing.assert_close(estimate, reference, rtol=0, atol=1.5e-2)


def test_cross_entropy_detaches_behavior_and_retains_current_gradients():
    current_mean = torch.tensor(
        [[0.4, -0.2, 0.7]], dtype=torch.float64, requires_grad=True
    )
    current_log_std = torch.tensor(
        [[-0.3, 0.5, -1.1]], dtype=torch.float64, requires_grad=True
    )
    behavior_mean = torch.tensor(
        [[-0.1, 0.6, 0.2]], dtype=torch.float64, requires_grad=True
    )
    behavior_log_std = torch.tensor(
        [[0.2, -0.4, -0.7]], dtype=torch.float64, requires_grad=True
    )

    loss = td_math.diagonal_gaussian_cross_entropy(
        current_mean,
        current_log_std,
        behavior_mean,
        behavior_log_std,
    ).sum()
    loss.backward()

    expected_mean_grad = (current_mean.detach() - behavior_mean.detach()) * torch.exp(
        -2.0 * behavior_log_std.detach()
    )
    expected_log_std_grad = torch.exp(
        2.0 * (current_log_std.detach() - behavior_log_std.detach())
    )
    torch.testing.assert_close(current_mean.grad, expected_mean_grad)
    torch.testing.assert_close(current_log_std.grad, expected_log_std_grad)
    assert behavior_mean.grad is None
    assert behavior_log_std.grad is None


def test_cross_entropy_and_jacobian_are_finite_at_policy_extremes():
    current_mean = torch.tensor(
        [[0.25, -0.5, 0.0, 1.0]], dtype=torch.float32, requires_grad=True
    )
    current_log_std = torch.tensor(
        [[2.0, -20.0, 2.0, -20.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    behavior_mean = torch.tensor(
        [[0.0, 0.25, 0.0, 1.0]], dtype=torch.float32, requires_grad=True
    )
    behavior_log_std = torch.tensor(
        [[-20.0, 2.0, 2.0, -20.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    pre_tanh = torch.tensor(
        [[-100.0, -20.0, 20.0, 100.0]],
        dtype=torch.float32,
        requires_grad=True,
    )

    gaussian_ce = td_math.diagonal_gaussian_cross_entropy(
        current_mean,
        current_log_std,
        behavior_mean,
        behavior_log_std,
        sum_action_dim=False,
    )
    log_jacobian = td_math.tanh_log_abs_det_jacobian(
        pre_tanh,
        sum_action_dim=False,
    )
    assert torch.isfinite(gaussian_ce).all()
    assert torch.isfinite(log_jacobian).all()
    assert gaussian_ce[0, 0] > 1e18

    (gaussian_ce.sum() + log_jacobian.sum()).backward()
    assert torch.isfinite(current_mean.grad).all()
    assert torch.isfinite(current_log_std.grad).all()
    assert torch.isfinite(pre_tanh.grad).all()
    assert behavior_mean.grad is None
    assert behavior_log_std.grad is None

    matching_log_std = torch.full((1, 2), -20.0)
    negative_ce = td_math.diagonal_gaussian_cross_entropy(
        torch.zeros_like(matching_log_std),
        matching_log_std,
        torch.zeros_like(matching_log_std),
        matching_log_std,
        sum_action_dim=False,
    )
    assert (negative_ce < 0.0).all()


@pytest.mark.parametrize(
    "helper",
    [
        td_math.diagonal_gaussian_cross_entropy,
        td_math.tanh_log_abs_det_jacobian,
    ],
)
def test_cross_entropy_math_rejects_non_boolean_reduction_flag(helper):
    value = torch.zeros(1, 2)
    args = (value,) if helper is td_math.tanh_log_abs_det_jacobian else (value,) * 4
    with pytest.raises(TypeError, match="sum_action_dim must be bool"):
        helper(*args, sum_action_dim="yes")
