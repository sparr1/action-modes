from copy import deepcopy
import math

import pytest
import torch
import torch.nn as nn

from RL.tdmpc2_core.common.layers import Ensemble, NormedLinear
from RL.tdmpc2_core.common.lora import (
    LoRARLLinear,
    LoRARLNormedLinear,
    dense_lora_rl_critic,
    lora_rl_parameter_groups,
    make_lora_rl_critic,
    reset_lora_rl_critic_,
    trainable_parameters,
    update_lora_rl_target_,
)


def _head():
    return nn.Sequential(
        NormedLinear(5, 7, dropout=0.25),
        NormedLinear(7, 6),
        nn.Linear(6, 3),
    )


def _adapters(module):
    return {
        path: child for path, child in module.named_modules()
        if isinstance(child, LoRARLLinear)
    }


@pytest.mark.parametrize("placement,indices", [("input_hidden", [0, 1]), ("hidden", [1])])
def test_prior_identity_placement_and_parameter_ownership(placement, indices):
    outer = Ensemble([_head(), _head()]).eval()
    before = deepcopy(outer.state_dict())
    adapted = make_lora_rl_critic(outer, rank=3, placement=placement).eval()
    sample = torch.randn(8, 5)
    torch.testing.assert_close(adapted(sample), outer(sample), rtol=0, atol=0)
    assert {id(value) for value in outer.parameters()}.isdisjoint(
        id(value) for value in adapted.parameters()
    )
    for head in adapted:
        assert [index for index, layer in enumerate(head) if isinstance(layer, LoRARLLinear)] == indices
        assert type(head[-1]) is nn.Linear
        assert all(value.requires_grad for value in head[-1].parameters())
        if placement == "hidden":
            assert all(value.requires_grad for value in head[0].parameters())
        for index in indices:
            adapter = head[index]
            assert not adapter.base.weight.requires_grad
            assert adapter.base.bias.requires_grad
            assert all(value.requires_grad for value in adapter.base.ln.parameters())
            assert not torch.count_nonzero(adapter.lora_B)
    adapted.train()
    assert not outer.training
    assert all(value.requires_grad for value in outer.parameters())
    for name, value in outer.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_normal_initialization_uses_requested_rank_scaling():
    base = nn.Linear(128, 128)
    torch.manual_seed(109)
    adapted = LoRARLLinear(deepcopy(base), rank=96)
    torch.manual_seed(109)
    expected = torch.empty(96, 128).normal_(mean=0, std=1 / math.sqrt(96))
    torch.testing.assert_close(adapted.lora_A, expected, rtol=0, atol=0)
    assert adapted.rank == 96
    assert adapted.scaling == 1
    assert not torch.count_nonzero(adapted.lora_B)


@pytest.mark.parametrize("training", [False, True])
def test_normed_adapter_matches_effective_matrix_before_dropout_and_normalization(training):
    base = NormedLinear(5, 7, dropout=0.25)
    adapted = LoRARLNormedLinear(deepcopy(base), rank=3, scale=0.75)
    with torch.no_grad():
        adapted.lora_B.normal_(std=0.1)
        adapted.base.bias.add_(0.1)
        adapted.base.ln.weight.mul_(1.1)
    merged = dense_lora_rl_critic(adapted)
    adapted.train(training)
    merged.train(training)
    sample = torch.randn(8, 5)
    left_input = sample.clone().requires_grad_(True)
    right_input = sample.clone().requires_grad_(True)
    torch.manual_seed(105)
    left = adapted(left_input)
    torch.manual_seed(105)
    right = merged(right_input)
    torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)
    left.square().sum().backward()
    right.square().sum().backward()
    torch.testing.assert_close(left_input.grad, right_input.grad, rtol=2e-5, atol=2e-6)
    assert adapted.base.weight.grad is None
    assert all(value.grad is not None for value in trainable_parameters(adapted))


def test_detached_critic_preserves_action_gradient_without_parameter_gradients():
    critic = make_lora_rl_critic(Ensemble([_head(), _head()]), rank=3).eval()
    with torch.no_grad():
        for adapter in _adapters(critic).values():
            adapter.lora_B.normal_(std=0.1)
    z_and_action = torch.randn(8, 5, requires_grad=True)
    critic.forward_detached(z_and_action).sum().backward()
    assert z_and_action.grad is not None
    assert torch.count_nonzero(z_and_action.grad[:, -1:])
    assert all(value.grad is None for value in critic.parameters())


def test_dense_target_averages_effective_weights_and_all_auxiliaries():
    prior = Ensemble([_head(), _head()]).eval()
    online = make_lora_rl_critic(prior, rank=3).eval()
    target = dense_lora_rl_critic(online)
    assert not _adapters(target)
    assert not any(value.requires_grad for value in target.parameters())
    assert tuple(target.state_dict()) == tuple(prior.state_dict())
    previous = deepcopy(target.state_dict())
    with torch.no_grad():
        for value in trainable_parameters(online):
            value.add_(0.2)
    effective = dense_lora_rl_critic(online).state_dict()
    update_lora_rl_target_(online, target, tau=0.25)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, previous[name].lerp(effective[name], 0.25))
    previous = deepcopy(target.state_dict())
    with torch.no_grad():
        for adapter in _adapters(online).values():
            adapter.lora_A.mul_(1.3)
            adapter.lora_B.sub_(0.1)
    effective = dense_lora_rl_critic(online).state_dict()
    update_lora_rl_target_(online, target, tau=0.4)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, previous[name].lerp(effective[name], 0.4))
    update_lora_rl_target_(online, target, tau=1.0)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, effective[name], rtol=0, atol=0)


@pytest.mark.parametrize("placement", ["input_hidden", "hidden"])
def test_reset_restores_every_prior_value_and_preserves_allocations(placement):
    prior = Ensemble([_head(), _head()]).eval()
    adapted = make_lora_rl_critic(prior, rank=3, placement=placement).eval()
    parameter_ids = {name: id(value) for name, value in adapted.named_parameters()}
    old_a = {name: value.lora_A.detach().clone() for name, value in _adapters(adapted).items()}
    with torch.no_grad():
        for value in adapted.parameters():
            value.add_(0.7)
        for value in prior.parameters():
            value.add_(0.2)
    reset_lora_rl_critic_(adapted, prior)
    assert parameter_ids == {name: id(value) for name, value in adapted.named_parameters()}
    for name, adapter in _adapters(adapted).items():
        assert not torch.equal(old_a[name], adapter.lora_A)
        assert not torch.count_nonzero(adapter.lora_B)
    merged = dense_lora_rl_critic(adapted)
    for name, value in prior.state_dict().items():
        torch.testing.assert_close(merged.state_dict()[name], value, rtol=0, atol=0)
    sample = torch.randn(5, 5)
    torch.testing.assert_close(adapted(sample), prior(sample), rtol=0, atol=0)


def test_adamw_only_decays_factors_and_covers_all_trainable_parameters_once():
    adapted = make_lora_rl_critic(_head(), rank=3)
    groups = lora_rl_parameter_groups(adapted, weight_decay=2e-4)
    grouped = [value for group in groups for value in group["params"]]
    assert len(grouped) == len({id(value) for value in grouped})
    assert {id(value) for value in grouped} == {id(value) for value in trainable_parameters(adapted)}
    assert [group["weight_decay"] for group in groups] == [2e-4, 0.0]
    optimizer = torch.optim.AdamW(groups, lr=0.1)
    before = {id(value): value.detach().clone() for value in adapted.parameters()}
    for value in grouped:
        value.grad = torch.zeros_like(value)
    optimizer.step()
    factor_ids = {id(value) for value in groups[0]["params"]}
    for value in adapted.parameters():
        multiplier = 1 - 0.1 * 2e-4 if id(value) in factor_ids else 1
        torch.testing.assert_close(value, before[id(value)] * multiplier, rtol=0, atol=0)
        if not value.requires_grad:
            assert value not in optimizer.state


@pytest.mark.parametrize("placement", ["input_hidden", "hidden"])
def test_learning_updates_biases_norms_and_head_but_never_selected_base_weights(placement):
    prior = _head().eval()
    prior_before = deepcopy(prior.state_dict())
    adapted = make_lora_rl_critic(prior, rank=3, placement=placement).eval()
    before = {name: value.detach().clone() for name, value in adapted.named_parameters()}
    optimizer = torch.optim.AdamW(lora_rl_parameter_groups(adapted, 2e-4), lr=0.01)
    sample, labels = torch.randn(12, 5), torch.randn(12, 3)
    # A has zero loss gradient when B=0; subsequent steps exercise both factors.
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        (adapted(sample) - labels).square().mean().backward()
        optimizer.step()
    for name, value in adapted.named_parameters():
        if value.requires_grad:
            assert not torch.equal(value, before[name]), name
        else:
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            assert value.grad is None
    for name, value in prior.state_dict().items():
        torch.testing.assert_close(value, prior_before[name], rtol=0, atol=0)
    assert all(value.grad is None for value in prior.parameters())


def test_weight_space_target_disagrees_with_factor_averaging_after_factors_change():
    online = LoRARLLinear(nn.Linear(3, 3), rank=2)
    with torch.no_grad():
        online.lora_A.fill_(1.0)
    target = dense_lora_rl_critic(online)
    factor_a, factor_b = online.lora_A.detach().clone(), online.lora_B.detach().clone()
    expected = target.weight.detach().clone()
    for a, b in ((2.0, 1.0), (-1.0, 3.0)):
        with torch.no_grad():
            online.lora_A.fill_(a)
            online.lora_B.fill_(b)
            expected.lerp_(online.base.weight + online.lora_B @ online.lora_A, 0.3)
            factor_a.lerp_(online.lora_A, 0.3)
            factor_b.lerp_(online.lora_B, 0.3)
        update_lora_rl_target_(online, target, tau=0.3)
        torch.testing.assert_close(target.weight, expected, rtol=0, atol=0)
        factor_averaged_weight = online.base.weight + factor_b @ factor_a
        assert not torch.allclose(target.weight, factor_averaged_weight)


@pytest.mark.parametrize("rank", [0, -1, 8, 1.5, 3.0, True, "3", float("nan"), float("inf")])
def test_rank_is_validated_without_silent_clipping(rank):
    with pytest.raises(ValueError, match="rank"):
        make_lora_rl_critic(_head(), rank=rank)


def test_target_layout_failure_is_transactional():
    online = make_lora_rl_critic(_head(), rank=3)
    target = nn.Sequential(nn.Linear(5, 7), nn.Linear(7, 6), nn.Linear(6, 3))
    before = deepcopy(target.state_dict())
    with pytest.raises(ValueError, match="layouts"):
        update_lora_rl_target_(online, target, tau=0.5)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
