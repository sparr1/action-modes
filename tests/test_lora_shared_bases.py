from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from RL.tdmpc2_core.common.inner_utils import (
    copy_lora_adapters_,
    lora_uses_shared_bases,
    rebase_lora_base_,
)
from RL.tdmpc2_core.common.layers import NormedLinear
from RL.tdmpc2_core.common.lora import (
    LoRALinear,
    LoRANormedLinear,
    lorafy_copy,
    lorafy_shared,
)


def _network():
    return nn.Sequential(
        NormedLinear(5, 7, dropout=0.25),
        NormedLinear(7, 6),
        nn.Linear(6, 3),
    )


def _adapters(module):
    return {
        path: child
        for path, child in module.named_modules()
        if isinstance(child, (LoRALinear, LoRANormedLinear))
    }


def test_shared_lora_matches_owned_forward_and_gradients():
    torch.manual_seed(11)
    outer = _network()
    outer_state = deepcopy(outer.state_dict())
    outer_requires_grad = [parameter.requires_grad for parameter in outer.parameters()]

    torch.manual_seed(29)
    owned = lorafy_copy(outer, rank=4, scale=0.75, dropout=0.2)
    torch.manual_seed(29)
    shared = lorafy_shared(outer, rank=4, scale=0.75, dropout=0.2)

    owned_adapters = _adapters(owned)
    shared_adapters = _adapters(shared)
    assert owned_adapters.keys() == shared_adapters.keys()
    with torch.no_grad():
        for path in owned_adapters:
            # Exercise a non-zero adapter path, not just identical base output.
            values = torch.linspace(
                -0.1,
                0.1,
                owned_adapters[path].lora_B.numel(),
            ).reshape_as(owned_adapters[path].lora_B)
            owned_adapters[path].lora_B.copy_(values)
            shared_adapters[path].lora_B.copy_(values)

    owned.train()
    shared.train()
    owned_input = torch.randn(8, 5, requires_grad=True)
    shared_input = owned_input.detach().clone().requires_grad_(True)
    torch.manual_seed(101)
    owned_output = owned(owned_input)
    torch.manual_seed(101)
    shared_output = shared(shared_input)
    torch.testing.assert_close(shared_output, owned_output, rtol=0, atol=0)

    owned_output.square().sum().backward()
    shared_output.square().sum().backward()
    torch.testing.assert_close(shared_input.grad, owned_input.grad, rtol=0, atol=0)
    for path in owned_adapters:
        torch.testing.assert_close(
            shared_adapters[path].lora_A.grad,
            owned_adapters[path].lora_A.grad,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            shared_adapters[path].lora_B.grad,
            owned_adapters[path].lora_B.grad,
            rtol=0,
            atol=0,
        )

    assert all(parameter.grad is None for parameter in outer.parameters())
    assert [parameter.requires_grad for parameter in outer.parameters()] == outer_requires_grad
    for key, value in outer.state_dict().items():
        torch.testing.assert_close(value, outer_state[key], rtol=0, atol=0)


def test_shared_lora_does_not_register_or_toggle_outer_bases():
    outer = _network().train()
    outer_parameter_ids = {id(parameter) for parameter in outer.parameters()}
    outer_keys = tuple(outer.state_dict())
    shared = lorafy_shared(outer, rank=3, scale=0.5)

    assert lora_uses_shared_bases(shared)
    assert outer_parameter_ids.isdisjoint(id(parameter) for parameter in shared.parameters())
    assert shared.state_dict()
    assert all(
        key.endswith(("lora_A", "lora_B")) for key in shared.state_dict()
    )
    assert tuple(outer.state_dict()) == outer_keys

    shared.eval()
    assert outer.training
    assert all(child.training for child in outer.modules())
    assert all(parameter.requires_grad for parameter in outer.parameters())

    target = deepcopy(shared).requires_grad_(False)
    target.train()
    assert outer.training
    assert all(parameter.requires_grad for parameter in outer.parameters())
    for path, adapter in _adapters(shared).items():
        target_adapter = target.get_submodule(path)
        assert adapter.base is outer.get_submodule(path)
        assert target_adapter.base is outer.get_submodule(path)
        assert target_adapter.lora_A.data_ptr() != adapter.lora_A.data_ptr()
        assert target_adapter.lora_B.data_ptr() != adapter.lora_B.data_ptr()


def test_shared_lora_tracks_outer_updates_and_rebinds_without_copying_adapters():
    torch.manual_seed(17)
    outer = _network().eval()
    shared = lorafy_shared(outer, rank=2, scale=1.0).eval()
    sample = torch.randn(4, 5)
    before = shared(sample)
    adapter_values = {
        name: value.detach().clone()
        for name, value in shared.named_parameters()
    }

    with torch.no_grad():
        outer[0].weight.add_(0.125)
    after_outer_update = shared(sample)
    assert not torch.equal(after_outer_update, before)

    replacement = deepcopy(outer)
    with torch.no_grad():
        replacement[0].weight.sub_(0.25)
    rebase_lora_base_(shared, replacement)
    for path, adapter in _adapters(shared).items():
        assert adapter.base is replacement.get_submodule(path)
    for name, value in shared.named_parameters():
        torch.testing.assert_close(value, adapter_values[name], rtol=0, atol=0)

    target = deepcopy(shared)
    with torch.no_grad():
        for adapter in _adapters(shared).values():
            adapter.lora_B.add_(0.5)
    copy_lora_adapters_(shared, target, tau=1.0)
    for path, adapter in _adapters(shared).items():
        target_adapter = target.get_submodule(path)
        torch.testing.assert_close(target_adapter.lora_A, adapter.lora_A, rtol=0, atol=0)
        torch.testing.assert_close(target_adapter.lora_B, adapter.lora_B, rtol=0, atol=0)


def test_owned_lora_still_retains_legacy_state_dict_and_snapshot_behavior():
    outer = _network().eval()
    owned = lorafy_copy(outer, rank=2, scale=1.0).eval()
    assert not lora_uses_shared_bases(owned)
    assert any("base.weight" in key for key in owned.state_dict())

    sample = torch.randn(3, 5)
    before = owned(sample)
    with torch.no_grad():
        outer[0].weight.add_(1.0)
    torch.testing.assert_close(owned(sample), before, rtol=0, atol=0)


def test_shared_lora_rejects_incompatible_rebind():
    shared = lorafy_shared(_network(), rank=2, scale=1.0)
    first = next(iter(_adapters(shared).values()))
    with pytest.raises(ValueError, match="shape"):
        first.share_base_(NormedLinear(4, 7))
