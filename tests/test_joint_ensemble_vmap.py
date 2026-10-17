"""Joint critic-type execution preserves gradients and independent dropout."""

from contextlib import ExitStack
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch import nn

from RL.tdmpc2_core.common import ensemble_vmap
from RL.tdmpc2_core.common.layers import Ensemble, NormedLinear, mlp
from RL.tdmpc2_core.common.q_representation import QRepresentation


def _critics(representation="scalar", dropout=0.):
    torch.manual_seed(901)
    heads = 2 if representation == "scalar" else 5
    decoder = QRepresentation(representation, num_q=heads, num_bins=7, vmin=-2, vmax=2)
    critics = tuple(
        Ensemble([
            mlp(5, [8, 8], decoder.output_dim, dropout=dropout)
            for _ in range(heads)
        ], backend="vmap").double()
        for _ in range(2)
    )
    references = deepcopy(critics)
    for critic in references:
        critic._backend = "loop"
    return references, critics, decoder


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=3e-9, atol=3e-10)


def _fixed_masks(critics, inputs, specifications):
    result = {}
    heads = sum(len(critic) for critic in critics)
    for index, (_, output_dim, _, norm) in enumerate(specifications):
        if norm is not None and norm[-2] and norm[-1]:
            p = norm[-2]
            result[index] = torch.empty(
                heads, *inputs[0].shape[:-1], output_dim, dtype=inputs[0].dtype,
            ).bernoulli_(1 - p).div_(1 - p)
    return result


def _reference_forwards(critics, inputs, masks=None):
    with ExitStack() as context:
        if masks is not None:
            row = 0
            for critic in critics:
                for head in critic:
                    for index, layer in enumerate(head):
                        if index in masks:
                            mask = masks[index][row]
                            context.enter_context(patch.object(
                                layer.dropout, "forward",
                                lambda value, mask=mask: value * mask,
                            ))
                    row += 1
        return tuple(critic(value) for critic, value in zip(critics, inputs))


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dropout", [0., .3])
def test_outputs_weighted_losses_and_original_parameter_gradients(representation, shared, dropout):
    references, critics, decoder = _critics(representation, dropout)
    # Cover independent masks at multiple depths rather than just mlp's first.
    if dropout:
        for critic in (*references, *critics):
            for head in critic:
                head[1].dropout = nn.Dropout(.2)
    x = torch.randn(3, 6, 5, dtype=torch.float64, requires_grad=True)
    y = x if shared else torch.randn_like(x, requires_grad=True)
    inputs = (x, y)
    spec = ensemble_vmap.grouped_vmap_spec(critics, NormedLinear)
    assert spec is not None
    masks = _fixed_masks(critics, inputs, spec) if dropout else None
    before_rng = torch.random.get_rng_state().clone()
    expected = _reference_forwards(references, inputs, masks)
    actual = ensemble_vmap.grouped_vectorized_forward(critics, inputs, spec, dropout_masks=masks)
    assert torch.equal(before_rng, torch.random.get_rng_state())
    targets = tuple(torch.randn(3, 6, 1, dtype=torch.float64) for _ in critics)

    def loss(predictions):
        losses = tuple(decoder.loss(output, target) for output, target in zip(predictions, targets))
        return .7 * losses[0] + .13 * losses[1]

    expected_loss, actual_loss = loss(expected), loss(actual)
    _close(actual_loss, expected_loss)
    input_parameters = (x,) if shared else (x, y)
    expected_grads = torch.autograd.grad(expected_loss, (
        *input_parameters, *(parameter for critic in references for parameter in critic.parameters()),
    ))
    actual_grads = torch.autograd.grad(actual_loss, (
        *input_parameters, *(parameter for critic in critics for parameter in critic.parameters()),
    ))
    for observed, reference in zip(actual, expected):
        _close(observed, reference)
    for observed, reference in zip(actual_grads, expected_grads):
        _close(observed, reference)


@pytest.mark.parametrize("detached", [False, True])
def test_auxiliary_latent_detachment_preserves_live_action_gradients(detached):
    references, critics, _ = _critics()
    z = torch.randn(3, 6, 3, dtype=torch.float64, requires_grad=True)
    action = torch.randn(3, 6, 2, dtype=torch.float64, requires_grad=True)
    inputs = (
        torch.cat((z, action), dim=-1),
        torch.cat((z.detach() if detached else z, action), dim=-1),
    )
    spec = ensemble_vmap.grouped_vmap_spec(critics, NormedLinear)
    expected = _reference_forwards(references, inputs)
    actual = ensemble_vmap.grouped_vectorized_forward(critics, inputs, spec)
    # Request only the auxiliary loss: no contribution from primary predictions.
    expected_grads = torch.autograd.grad(expected[1].square().sum(), (z, action), allow_unused=True)
    actual_grads = torch.autograd.grad(actual[1].square().sum(), (z, action))
    # The grouped autograd graph can materialize a zero where the separate graph
    # has no route at all. The action remains differentiable in both cases.
    if detached:
        assert expected_grads[0] is None
        assert torch.count_nonzero(actual_grads[0]) == 0
    else:
        _close(actual_grads[0], expected_grads[0])
        assert torch.count_nonzero(actual_grads[0]) > 0
    _close(actual_grads[1], expected_grads[1])
    assert torch.count_nonzero(actual_grads[1]) > 0


def test_grouping_keeps_optimizer_slots_and_state_dict_ownership():
    _, critics, _ = _critics()
    before_keys = tuple(tuple(critic.state_dict()) for critic in critics)
    parameters = tuple(parameter for critic in critics for parameter in critic.parameters())
    identities = tuple(id(parameter) for parameter in parameters)
    optimizer = torch.optim.Adam([
        {"params": critics[0].parameters(), "lr": .003},
        {"params": critics[1].parameters(), "lr": .007},
    ])
    inputs = (torch.randn(6, 5).double(), torch.randn(6, 5).double())
    spec = ensemble_vmap.grouped_vmap_spec(critics, NormedLinear)
    output = ensemble_vmap.grouped_vectorized_forward(critics, inputs, spec)
    (.7 * output[0].square().mean() + .13 * output[1].square().mean()).backward()
    optimizer.step()
    assert len(optimizer.state) == len(parameters)
    assert all(parameter.grad is not None for parameter in parameters)
    assert tuple(id(parameter) for critic in critics for parameter in critic.parameters()) == identities
    assert tuple(tuple(critic.state_dict()) for critic in critics) == before_keys


def test_dropout_masks_are_independent_across_critic_types_and_heads():
    torch.manual_seed(193)
    prototype = nn.Sequential(NormedLinear(1, 64, dropout=.25, act=nn.Identity())).double()
    with torch.no_grad():
        prototype[0].weight.zero_()
        prototype[0].bias.fill_(1.)
    critics = tuple(Ensemble([deepcopy(prototype) for _ in range(2)], backend="vmap") for _ in range(2))
    x = torch.zeros(4096, 1, dtype=torch.float64)
    spec = ensemble_vmap.grouped_vmap_spec(critics, NormedLinear)
    before_rng = torch.random.get_rng_state().clone()
    predictions = ensemble_vmap.grouped_vectorized_forward(critics, (x, x), spec)
    # Positive normalized activations identify retained Bernoulli entries.
    kept = torch.cat(predictions, dim=0) > 0
    for row in kept:
        assert abs(row.double().mean().item() - .75) < .01
    for first in range(4):
        for second in range(first + 1, 4):
            both = (kept[first] & kept[second]).double().mean().item()
            assert abs(both - .75 ** 2) < .01
    torch.random.set_rng_state(before_rng)
    repeated = ensemble_vmap.grouped_vectorized_forward(critics, (x, x), spec)
    for actual, expected in zip(repeated, predictions):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("reason", [
    "loop", "shape", "mode", "head_hook", "wrapper_hook", "wrapper_forward", "custom", "buffer",
])
def test_unsupported_groups_are_rejected_before_rng_or_tensor_work(reason):
    _, critics, _ = _critics(dropout=.2)
    if reason == "loop":
        critics[1]._backend = "loop"
    elif reason == "shape":
        critics[1].modules_list[0][-1] = nn.Linear(8, 3).double()
    elif reason == "mode":
        critics[1].eval()
    elif reason == "head_hook":
        critics[1][0].register_forward_hook(lambda module, args, result: result)
    elif reason == "wrapper_hook":
        critics[1].register_forward_hook(lambda module, args, result: result)
    elif reason == "wrapper_forward":
        critics[1].forward = lambda value: value
    elif reason == "custom":
        critics[1][0].append(nn.Tanh())
    else:
        critics[1].register_buffer("extra", torch.zeros(1))
    before_rng = torch.random.get_rng_state().clone()
    assert ensemble_vmap.grouped_vmap_spec(critics, NormedLinear) is None
    assert torch.equal(before_rng, torch.random.get_rng_state())


def test_shared_inputs_avoid_expansion_and_failures_are_not_retried(monkeypatch):
    _, critics, _ = _critics()
    x = torch.randn(6, 5).double()
    spec = ensemble_vmap.grouped_vmap_spec(critics, NormedLinear)
    calls = []

    def fail(modules, inputs, specifications, **kwargs):
        calls.append((len(modules), inputs, kwargs["head_inputs"]))
        raise RuntimeError("grouped kernel failed")

    monkeypatch.setattr(ensemble_vmap, "vectorized_forward", fail)
    with pytest.raises(RuntimeError, match="grouped kernel failed"):
        ensemble_vmap.grouped_vectorized_forward(critics, (x, x), spec)
    assert len(calls) == 1
    assert calls[0][0] == 4 and calls[0][1] is x and calls[0][2] is False


@pytest.mark.parametrize("malformed", ["missing", "extra", "broadcast"])
def test_fixed_dropout_masks_cannot_silently_change_broadcasting(malformed):
    _, critics, _ = _critics(dropout=.3)
    x = torch.randn(6, 5).double()
    spec = ensemble_vmap.grouped_vmap_spec(critics, NormedLinear)
    masks = _fixed_masks(critics, (x, x), spec)
    if malformed == "missing":
        masks.clear()
    elif malformed == "extra":
        masks[1] = masks[0]
    else:
        masks[0] = masks[0][:, :1]
    before_rng = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="Dropout masks"):
        ensemble_vmap.grouped_vectorized_forward(critics, (x, x), spec, dropout_masks=masks)
    assert torch.equal(before_rng, torch.random.get_rng_state())


@pytest.mark.parametrize("fixed_masks", [False, True])
def test_grouped_strict_aot_compilation_with_independent_inputs(fixed_masks):
    # Import the runtime that installs TensorDict's vmap monkeypatch.
    from tests.test_ambi_root_local_sac import _tiny_model  # noqa: F401

    torch._dynamo.reset()
    try:
        references, critics, _ = _critics("distributional", .3 if fixed_masks else 0.)
        x = torch.randn(3, 6, 5, dtype=torch.float64, requires_grad=True)
        y = torch.randn_like(x, requires_grad=True)
        spec = ensemble_vmap.grouped_vmap_spec(critics, NormedLinear)
        masks = _fixed_masks(critics, (x, y), spec) if fixed_masks else None

        def forward(left, right, dropout_masks):
            return ensemble_vmap.grouped_vectorized_forward(
                critics, (left, right), spec, dropout_masks=dropout_masks,
            )

        compiled = torch.compile(forward, backend="aot_eager", fullgraph=True, dynamic=False)
        expected = _reference_forwards(references, (x, y), masks)
        actual = compiled(x, y, masks)
        expected_grads = torch.autograd.grad(sum(value.square().sum() for value in expected), (
            x, y, *(parameter for critic in references for parameter in critic.parameters()),
        ))
        actual_grads = torch.autograd.grad(sum(value.square().sum() for value in actual), (
            x, y, *(parameter for critic in critics for parameter in critic.parameters()),
        ))
        for observed, reference in zip(actual, expected):
            _close(observed, reference)
        for observed, reference in zip(actual_grads, expected_grads):
            _close(observed, reference)
    finally:
        torch._dynamo.reset()
