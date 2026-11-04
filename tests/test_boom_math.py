"""Numerical contracts for corrected BOOM's sampled-action alignment."""

import math

import pytest
import torch

from RL.tdmpc2_core.boom import boom_alignment_loss


def _inputs(d=3):
    shape = (3, 4, d)
    count = math.prod(shape)
    return {
        "action": torch.linspace(-0.8, 0.7, count, dtype=torch.float64).reshape(shape).requires_grad_(),
        "actor_log_std": torch.linspace(-5.0, 0.5, count, dtype=torch.float64).reshape(shape),
        "target_action": torch.linspace(0.4, -0.6, count, dtype=torch.float64).reshape(shape),
        "valid": torch.tensor([[True, False, True, True], [False, False, False, False], [True, True, False, True]]),
        "q_values": torch.tensor([[3., -8., 5., 1.], [9., 2., 7., 3.], [-5., 2., 6., 11.]], dtype=torch.float64).unsqueeze(-1),
        "scale": torch.tensor([3.5], dtype=torch.float64),
        "rho": 0.6,
        "coefficient": 0.17,
    }


def _distribution_reference(inputs):
    action = inputs["action"]
    result = action.sum() * 0
    for t in range(action.shape[0]):
        valid = inputs["valid"][t].reshape(-1)
        if not valid.any():
            continue
        target = inputs["target_action"][t, valid].detach()
        std = inputs["actor_log_std"][t, valid].detach().exp().clamp_min(inputs.get("min_std", 0.05))
        nll = -torch.distributions.Normal(target, std).log_prob(action[t, valid]).sum(-1)
        q = inputs["q_values"][t].reshape(-1)[valid].detach()
        weights = (q / inputs["scale"].detach()).softmax(0)
        result = result + inputs["rho"] ** t * (weights * nll).sum()
    return result * inputs["coefficient"] / (action.shape[0] * inputs["scale"].detach().reshape(()))


@pytest.mark.parametrize("action_dim", [1, 2, 6, 38])
def test_matches_normal_distribution_and_action_gradients_without_extra_dimension(action_dim):
    inputs = _inputs(action_dim)
    loss, metrics = boom_alignment_loss(**inputs)
    expected = _distribution_reference(inputs)
    torch.testing.assert_close(loss, expected, rtol=1e-12, atol=1e-12)
    actual_gradient, = torch.autograd.grad(loss, inputs["action"], retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected, inputs["action"])
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    assert all(not value.requires_grad for value in metrics.values())


@pytest.mark.parametrize("scale", [1.0, 2.0, 2.00001, 5.0])
def test_strict_gate_and_scale_apply_to_loss_and_gradient(scale):
    inputs = _inputs()
    inputs["scale"] = torch.tensor(scale, dtype=torch.float64)
    loss, metrics = boom_alignment_loss(**inputs)
    if scale <= 2:
        assert loss.item() == 0
        assert metrics["boom_gate_active"].item() == 0
        gradient, = torch.autograd.grad(loss, inputs["action"])
        assert torch.count_nonzero(gradient) == 0
    else:
        torch.testing.assert_close(loss, _distribution_reference(inputs))
        assert metrics["boom_gate_active"].item() == 1


@pytest.mark.parametrize("rho", [0.0, 1.0, 1.3])
def test_nonnegative_temporal_weights_match_existing_reduction_contract(rho):
    inputs = _inputs()
    inputs["rho"] = rho
    loss, _ = boom_alignment_loss(**inputs)
    expected = _distribution_reference(inputs)
    torch.testing.assert_close(loss, expected, rtol=1e-12, atol=1e-12)
    gradient, = torch.autograd.grad(loss, inputs["action"])
    if rho == 0:
        assert torch.count_nonzero(gradient[0]) > 0
        assert torch.count_nonzero(gradient[1:]) == 0


def test_target_std_q_weights_and_scale_have_no_direct_gradient():
    inputs = _inputs()
    names = ("actor_log_std", "target_action", "q_values", "scale")
    for name in names:
        inputs[name].requires_grad_()
    loss, _ = boom_alignment_loss(**inputs)
    gradients = torch.autograd.grad(loss, (inputs["action"], *(inputs[name] for name in names)), allow_unused=True)
    assert torch.count_nonzero(gradients[0]) > 0
    assert gradients[1:] == (None, None, None, None)


def test_std_still_receives_gradient_through_reparameterized_action():
    mean = torch.tensor([[[0.2, -0.4]]], dtype=torch.float64, requires_grad=True)
    log_std = torch.tensor([[[-4.0, -0.3]]], dtype=torch.float64, requires_grad=True)
    eps = torch.tensor([[[0.7, -0.8]]], dtype=torch.float64)
    action = (mean + log_std.exp() * eps).tanh()
    target = torch.tensor([[[-0.3, 0.5]]], dtype=torch.float64)
    inputs = dict(action=action, actor_log_std=log_std, target_action=target,
                  valid=torch.ones(1, 1, 1, dtype=torch.bool),
                  q_values=torch.ones(1, 1, 1, dtype=torch.float64),
                  scale=torch.tensor(4., dtype=torch.float64), rho=0.5, coefficient=0.2)
    loss, _ = boom_alignment_loss(**inputs)
    mean_grad, std_grad = torch.autograd.grad(loss, (mean, log_std))
    attraction = 0.2 / 4 * (action.detach() - target) / log_std.detach().exp().clamp_min(0.05).square()
    expected_mean = attraction * (1 - action.detach().square())
    torch.testing.assert_close(mean_grad, expected_mean)
    torch.testing.assert_close(std_grad, expected_mean * log_std.detach().exp() * eps)
    assert torch.count_nonzero(std_grad) == 2  # Includes the coordinate below the std floor.


def test_missing_rows_with_nan_placeholders_are_inert_and_empty_depth_is_zero():
    inputs = _inputs()
    expected = _distribution_reference(inputs)
    invalid = ~inputs["valid"]
    for name in ("action", "actor_log_std", "target_action"):
        inputs[name] = inputs[name].detach().masked_fill(invalid.unsqueeze(-1), torch.nan)
    inputs["action"].requires_grad_()
    inputs["q_values"] = inputs["q_values"].masked_fill(invalid.unsqueeze(-1), torch.nan)
    loss, metrics = boom_alignment_loss(**inputs)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(inputs["action"].grad).all()
    assert torch.count_nonzero(inputs["action"].grad[invalid]) == 0
    assert all(torch.isfinite(value) for value in metrics.values())


def test_all_invalid_batch_has_finite_zero_loss_and_gradients():
    inputs = _inputs()
    inputs["valid"].zero_()
    for name in ("action", "actor_log_std", "target_action", "q_values"):
        inputs[name] = torch.full_like(inputs[name], torch.nan)
    inputs["action"].requires_grad_()
    loss, metrics = boom_alignment_loss(**inputs)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(inputs["action"].grad) == 0
    assert all(torch.isfinite(value) for value in metrics.values())
    for name in ("boom_alignment_nll", "boom_valid_fraction", "boom_weight_entropy", "boom_effective_samples"):
        assert metrics[name].item() == 0


def test_softmax_is_per_depth_and_q_shift_invariant():
    inputs = _inputs()
    loss, metrics = boom_alignment_loss(**inputs)
    inputs["q_values"] = inputs["q_values"] + torch.tensor([1000., -999., 47.], dtype=torch.float64).reshape(3, 1, 1)
    shifted, shifted_metrics = boom_alignment_loss(**inputs)
    torch.testing.assert_close(shifted, loss, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(shifted_metrics["boom_weight_entropy"], metrics["boom_weight_entropy"])
    assert 0 < metrics["boom_effective_samples"].item() <= 2  # Two populated depths out of three.


def test_fullgraph_compilation_supports_data_dependent_gate_and_validity():
    inputs = _inputs()
    # A callable backend exercises Dynamo's fullgraph contract without lazy
    # registry imports starting an unrelated Inductor worker pool on macOS.
    compiled = torch.compile(
        boom_alignment_loss,
        backend=lambda graph_module, example_inputs: graph_module.forward,
        fullgraph=True,
    )
    loss, metrics = compiled(**inputs)
    expected, expected_metrics = boom_alignment_loss(**inputs)
    torch.testing.assert_close(loss, expected)
    for name in metrics:
        torch.testing.assert_close(metrics[name], expected_metrics[name])
    inputs["scale"].fill_(2)
    inputs["valid"].zero_()
    gated, metrics = compiled(**inputs)
    assert gated.item() == 0
    assert metrics["boom_gate_active"].item() == 0
    assert metrics["boom_valid_fraction"].item() == 0
