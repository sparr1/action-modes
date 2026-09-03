from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.common import layers
from RL.tdmpc2_core.common.horizon_value import (
    HORIZON_MODES,
    HorizonQEnsemble,
    HorizonValue,
    build_horizon_q,
    build_horizon_value,
)


def _outer_qs(*, input_dim=7, output_dim=1, num_q=2):
    torch.manual_seed(19)
    return layers.Ensemble(
        [
            layers.mlp(input_dim, [11, 9], output_dim, dropout=0.0)
            for _ in range(num_q)
        ]
    )


@pytest.mark.parametrize("mode", sorted(HORIZON_MODES))
@pytest.mark.parametrize("output_dim", [1, 13])
def test_horizon_q_initializes_to_outer_at_every_depth(mode, output_dim):
    outer = _outer_qs(output_dim=output_dim, num_q=5 if output_dim > 1 else 2)
    critic = build_horizon_q(outer, 3, mode)
    q_input = torch.randn(6, 7)
    expected = outer(q_input)

    assert isinstance(critic, HorizonQEnsemble)
    assert critic.mode == mode
    assert critic.horizon == 3
    assert critic.output_dim == output_dim
    assert critic.num_q == len(outer)
    for depth in (1, 2, 3):
        torch.testing.assert_close(critic(q_input, depth), expected)
    torch.testing.assert_close(
        critic(q_input, torch.tensor([[1], [2], [3], [1], [2], [3]])),
        expected,
    )

    outer_storage = {parameter.data_ptr() for parameter in outer.parameters()}
    assert outer_storage.isdisjoint(
        parameter.data_ptr() for parameter in critic.parameters()
    )


def test_shared_q_is_a_direct_outer_ensemble_clone():
    outer = _outer_qs()
    critic = build_horizon_q(outer, 3, "shared")

    assert isinstance(critic.ensemble, layers.Ensemble)
    assert isinstance(critic.ensemble[0], torch.nn.Sequential)
    assert critic.ensemble.state_dict().keys() == outer.state_dict().keys()
    for actual, expected in zip(critic.ensemble.parameters(), outer.parameters()):
        torch.testing.assert_close(actual, expected)
        assert actual.data_ptr() != expected.data_ptr()


def test_depth_conditioning_adds_only_zero_input_columns_then_routes_depth():
    outer = _outer_qs()
    critic = build_horizon_q(outer, 3, "depth_conditioned")
    converted = critic.ensemble[0].network[0]
    original = outer[0][0]

    assert converted.in_features == original.in_features + 3
    torch.testing.assert_close(converted.weight[:, : original.in_features], original.weight)
    torch.testing.assert_close(
        converted.weight[:, original.in_features :],
        torch.zeros_like(converted.weight[:, original.in_features :]),
    )

    with torch.no_grad():
        converted.weight[:, -3:] = torch.tensor(
            [[-3.0, 0.0, 3.0]]
        ).expand(converted.out_features, -1)
        # Break the common shift that LayerNorm would otherwise remove.
        converted.weight[0, -3:] = torch.tensor([2.0, -1.0, 4.0])
    q_input = torch.randn(4, 7)
    assert not torch.equal(critic(q_input, 1), critic(q_input, 2))
    assert not torch.equal(critic(q_input, 2), critic(q_input, 3))


def test_stage_heads_are_replicated_then_route_each_row():
    outer = _outer_qs()
    critic = build_horizon_q(outer, 3, "stage_heads")
    member = critic.ensemble[0]
    q_input = torch.randn(3, 7)

    for head in member.heads:
        torch.testing.assert_close(head.weight, outer[0][-1].weight)
        torch.testing.assert_close(head.bias, outer[0][-1].bias)
    with torch.no_grad():
        for index, head in enumerate(member.heads):
            head.bias.fill_(10.0 * (index + 1))
    mixed = critic(q_input, torch.tensor([1, 2, 3]))[0]
    for row, depth in enumerate((1, 2, 3)):
        torch.testing.assert_close(
            mixed[row], critic(q_input[row], depth)[0]
        )


@pytest.mark.parametrize("mode", sorted(HORIZON_MODES))
def test_horizon_q_detached_forward_keeps_only_input_gradients(mode):
    critic = build_horizon_q(_outer_qs(), 3, mode)
    q_input = torch.randn(5, 7, requires_grad=True)
    output = critic.forward_detached(q_input, torch.tensor([1, 2, 3, 2, 1]))

    output.sum().backward()
    assert q_input.grad is not None
    assert torch.isfinite(q_input.grad).all()
    assert all(parameter.grad is None for parameter in critic.parameters())


@pytest.mark.parametrize("mode", sorted(HORIZON_MODES))
def test_horizon_q_reset_from_outer_preserves_parameter_objects(mode):
    outer = _outer_qs(output_dim=5, num_q=3)
    critic = build_horizon_q(outer, 3, mode)
    parameter_ids = tuple(id(parameter) for parameter in critic.parameters())
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.add_(2.0)

    critic.reset_from_outer(outer)
    assert tuple(id(parameter) for parameter in critic.parameters()) == parameter_ids
    q_input = torch.randn(4, 7)
    for depth in (1, 2, 3):
        torch.testing.assert_close(critic(q_input, depth), outer(q_input))


@pytest.mark.parametrize("mode", sorted(HORIZON_MODES))
def test_full_target_hard_and_polyak_updates_are_exact(mode):
    online = build_horizon_q(_outer_qs(), 3, mode)
    target = online.make_target()
    assert not target.training
    assert not any(parameter.requires_grad for parameter in target.parameters())

    with torch.no_grad():
        for parameter in online.parameters():
            parameter.add_(1.0)
    before = [parameter.detach().clone() for parameter in target.parameters()]
    target.polyak_update_from(online, 0.25)
    for old, source, actual in zip(before, online.parameters(), target.parameters()):
        torch.testing.assert_close(actual, old.lerp(source, 0.25))

    target.hard_update_from(online)
    for source, actual in zip(online.parameters(), target.parameters()):
        torch.testing.assert_close(actual, source)


def test_stage_target_update_copies_trunk_and_only_requested_head():
    online = build_horizon_q(_outer_qs(), 3, "stage_heads")
    target = online.make_target()
    before_heads = [
        [head.weight.detach().clone() for head in member.heads]
        for member in target.ensemble
    ]
    with torch.no_grad():
        for member in online.ensemble:
            for parameter in member.trunk.parameters():
                parameter.add_(1.0)
            for index, head in enumerate(member.heads):
                head.weight.add_(10.0 * (index + 1))

    target.hard_update_from(online, remaining_horizon=2)
    for member_index, (source, actual) in enumerate(
        zip(online.ensemble, target.ensemble)
    ):
        for source_parameter, actual_parameter in zip(
            source.trunk.parameters(), actual.trunk.parameters()
        ):
            torch.testing.assert_close(actual_parameter, source_parameter)
        torch.testing.assert_close(actual.heads[1].weight, source.heads[1].weight)
        torch.testing.assert_close(actual.heads[0].weight, before_heads[member_index][0])
        torch.testing.assert_close(actual.heads[2].weight, before_heads[member_index][2])


@pytest.mark.parametrize("mode", sorted(HORIZON_MODES))
def test_scalar_horizon_value_shapes_routing_detach_and_copy(mode):
    torch.manual_seed(23)
    value = build_horizon_value(7, [11, 9], 3, mode)
    assert isinstance(value, HorizonValue)
    latent = torch.randn(6, 7, requires_grad=True)
    depth = torch.tensor([1, 2, 3, 3, 2, 1])

    prediction = value(latent, depth)
    assert prediction.shape == (6, 1)
    value.forward_detached(latent, depth).sum().backward()
    assert latent.grad is not None
    assert all(parameter.grad is None for parameter in value.parameters())

    target = value.make_target()
    with torch.no_grad():
        for parameter in value.parameters():
            parameter.add_(0.5)
    target.reset_from(value)
    torch.testing.assert_close(target(latent.detach(), depth), value(latent.detach(), depth))


def test_value_stage_update_leaves_unselected_heads_unchanged():
    value = build_horizon_value(7, 9, 3, "stage_heads")
    target = value.make_target()
    before = [head.weight.detach().clone() for head in target.value.heads]
    with torch.no_grad():
        for index, head in enumerate(value.value.heads):
            head.weight.add_(index + 1.0)
    target.polyak_update_from(value, 0.5, remaining_horizon=3)

    torch.testing.assert_close(target.value.heads[0].weight, before[0])
    torch.testing.assert_close(target.value.heads[1].weight, before[1])
    torch.testing.assert_close(
        target.value.heads[2].weight, before[2].lerp(value.value.heads[2].weight, 0.5)
    )


@pytest.mark.parametrize("mode", sorted(HORIZON_MODES))
def test_horizon_module_rejects_invalid_depths_and_shapes(mode):
    critic = build_horizon_q(_outer_qs(), 3, mode)
    q_input = torch.randn(4, 7)

    with pytest.raises(ValueError, match=r"\[1, 3\]"):
        critic(q_input, 0)
    with pytest.raises(ValueError, match=r"\[1, 3\]"):
        critic(q_input, 4)
    with pytest.raises(ValueError, match="broadcastable"):
        critic(q_input, torch.ones(3, dtype=torch.long))
    with pytest.raises(ValueError, match="integer depths"):
        critic(q_input, torch.tensor([1.0, 1.5, 2.0, 3.0]))


def test_target_updates_reject_mismatched_layouts_and_invalid_tau():
    outer = _outer_qs()
    shared = build_horizon_q(outer, 3, "shared")
    stages = build_horizon_q(outer, 3, "stage_heads")

    with pytest.raises(ValueError, match="layouts do not match"):
        shared.hard_update_from(stages)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        shared.polyak_update_from(deepcopy(shared), 1.1)
