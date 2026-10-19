"""View fusion preserves PAVE geometry, higher derivatives and custom paths."""

from copy import deepcopy
from itertools import product

import pytest
import torch
from torch import nn

from RL.tdmpc2_core import pave
from RL.tdmpc2_core.common import ensemble_vmap
from RL.tdmpc2_core.common.layers import Ensemble, mlp
from RL.tdmpc2_core.common.q_representation import QRepresentation
from tests.pave_reference import separate_critic_pave_loss


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires an allocated CUDA GPU.",
))]
TERMS = [values for values in product((False, True), repeat=3) if any(values)]


def make_case(representation="distributional", *, rows=5, device="cpu", dtype=torch.float64):
    torch.manual_seed(7129)
    scalar = representation == "scalar"
    decoder = QRepresentation(
        "scalar" if scalar else "distributional", num_q=2 if scalar else 5,
        num_bins=7, vmin=-2., vmax=2.,
        codec="symexp_mean" if representation == "symexp_mean" else "symlog",
    )
    critic = Ensemble([
        mlp(7, [12, 12], decoder.output_dim, dropout=.3)
        for _ in range(decoder.num_q)
    ], backend="vmap").to(device=device, dtype=dtype)
    # Noncontiguous inputs exercise view packing without aliasing assumptions.
    inputs = dict(
        z=torch.randn(rows, 8, device=device, dtype=dtype)[:, ::2],
        action=torch.randn(rows, 6, device=device, dtype=dtype)[:, ::2].tanh(),
        noisy_z=torch.randn(rows, 8, device=device, dtype=dtype)[:, ::2],
        next_z=torch.randn(rows, 8, device=device, dtype=dtype)[:, ::2],
        probe=torch.randint(0, 2, (rows, 3), device=device).to(dtype) * 2 - 1,
    )
    return critic, decoder, inputs


def close_tree(actual, expected, *, rtol=2e-8, atol=2e-10):
    if torch.is_tensor(expected):
        assert torch.isfinite(actual).all()
        assert torch.isfinite(expected).all()
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            close_tree(actual[name], expected[name], rtol=rtol, atol=atol)
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            close_tree(left, right, rtol=rtol, atol=atol)
    else:
        assert actual == expected


def _parameter_gradients(loss, critic):
    parameters = tuple(parameter for parameter in critic.parameters() if parameter.requires_grad)
    return torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("representation", ["scalar", "distributional", "symexp_mean"])
@pytest.mark.parametrize("terms", TERMS)
@pytest.mark.parametrize("rows", [1, 5])
def test_all_term_combinations_match_separate_loss_metrics_and_each_parameter_gradient(
    device, representation, terms, rows,
):
    critic, decoder, inputs = make_case(representation, rows=rows, device=device)
    reference = deepcopy(critic)
    for value in inputs.values():
        value.requires_grad_()
    coefficients = dict(zip(("mpr_coef", "vfc_coef", "curvature_coef"),
                            (weight if active else 0. for weight, active in zip((.2, .3, .4), terms))))
    # Unequal modes must be restored even though all geometry disables dropout.
    critic[0][0].dropout.eval()
    modes = [(module, module.training) for module in critic.modules()]
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if device == "cuda" else []
    actual = pave.critic_pave_loss(critic, decoder.decode, **inputs, **coefficients,
                                   curvature_margin=.1, return_components=True)
    expected = separate_critic_pave_loss(reference, decoder.decode, **inputs, **coefficients,
                                         curvature_margin=.1, return_components=True)
    close_tree(actual, expected)
    close_tree(_parameter_gradients(actual[0], critic), _parameter_gradients(expected[0], reference))
    for term in actual[2]:
        close_tree(_parameter_gradients(actual[2][term], critic),
                   _parameter_gradients(expected[2][term], reference))
    assert all(module.training == training for module, training in modes)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if device == "cuda":
        assert all(torch.equal(left, right) for left, right in zip(torch.cuda.get_rng_state_all(), cuda_rng))
    actual[0].backward()
    assert all(value.grad is None for value in inputs.values())


@pytest.mark.parametrize("terms", TERMS)
def test_dense_views_share_parameters_and_keep_curvature_graph_small(monkeypatch, terms):
    critic, decoder, inputs = make_case()
    calls = []
    forward = ensemble_vmap._tensor_mlp_vmap
    stack = ensemble_vmap.stack_parameters
    stacks = []
    parameter_ids = []

    def record(parameters, values, *args, **kwargs):
        calls.append(tuple(values.shape))
        parameter_ids.append(id(parameters))
        return forward(parameters, values, *args, **kwargs)

    def record_stack(*args, **kwargs):
        stacks.append(1)
        return stack(*args, **kwargs)

    monkeypatch.setattr(ensemble_vmap, "_tensor_mlp_vmap", record)
    monkeypatch.setattr(ensemble_vmap, "stack_parameters", record_stack)
    coefficients = dict(zip(("mpr_coef", "vfc_coef", "curvature_coef"), map(float, terms)))
    loss, _ = pave.critic_pave_loss(critic, decoder.decode, **inputs, **coefficients)
    loss.backward()
    other_views = int(terms[0]) + int(terms[1])
    sizes = [5, 5 * other_views] if terms[2] and other_views else [5 * (1 + other_views)]
    assert calls == [(len(critic), rows, 7) for rows in sizes]
    assert stacks == [1]
    assert len(set(parameter_ids)) == 1


@pytest.mark.parametrize("reason", ["loop", "custom", "lora", "head_hook", "ensemble_hook",
                                    "forward", "decoder"])
def test_custom_paths_keep_three_independent_view_calls(monkeypatch, reason):
    critic, decoder, inputs = make_case()
    decode = decoder.decode
    if reason == "loop":
        critic = Ensemble(deepcopy(critic.modules_list), backend="loop")
    elif reason == "custom":
        critic[0].append(nn.Tanh())
    elif reason == "lora":
        from RL.tdmpc2_core.common.lora import make_lora_rl_critic
        critic = make_lora_rl_critic(critic, rank=2)
    elif reason == "head_hook":
        critic[0][0].register_forward_hook(lambda module, args, output: output + .1)
    elif reason == "ensemble_hook":
        critic.register_forward_hook(lambda module, args, output: output)
    elif reason == "forward":
        original = critic.forward_heads_eager
        monkeypatch.setattr(critic, "forward_heads_eager", lambda value: original(value))
    else:
        # This decoder really couples rows: fusing its views would be wrong.
        decode = lambda value: decoder.decode(value).square().sum(dim=1, keepdim=True).expand(-1, 5, -1)

    calls = []
    predictions = pave._head_predictions

    def record(ensemble, latents, actions):
        calls.append(tuple(latents.shape))
        return predictions(ensemble, latents, actions)

    monkeypatch.setattr(pave, "_head_predictions", record)
    options = dict(mpr_coef=.2, vfc_coef=.3, curvature_coef=.4)
    actual = pave.critic_pave_loss(critic, decode, **inputs, **options)
    expected = separate_critic_pave_loss(critic, decode, **inputs, **options)
    close_tree(actual, expected, rtol=0., atol=0.)
    close_tree(_parameter_gradients(actual[0], critic), _parameter_gradients(expected[0], critic),
               rtol=0., atol=0.)
    assert calls == [(5, 4)] * 3


def test_fused_error_restores_modes_and_is_not_retried(monkeypatch):
    critic, decoder, inputs = make_case()
    critic[0].eval()
    modes = [(module, module.training) for module in critic.modules()]
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("intentional fused failure")

    monkeypatch.setattr(ensemble_vmap, "_tensor_mlp_vmap", fail)
    with pytest.raises(RuntimeError, match="intentional fused failure"):
        pave.critic_pave_loss(critic, decoder.decode, **inputs, mpr_coef=.1, vfc_coef=.1)
    assert calls == [1]
    assert all(module.training == training for module, training in modes)
