"""Critic-only low-rank updates inspired by LoRA-RL (arXiv:2604.18978).

AMBI adapts a learned prior, so B starts at zero instead of perturbing and
renormalizing the prior as in the upstream from-scratch BRC initialization.
Selected kernels are frozen; biases, LayerNorm, and the value head train normally.
"""

from copy import deepcopy
import math
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Ensemble, NormedLinear


class LoRARLLinear(nn.Module):
    """An owned frozen kernel with trainable BA and an ordinary trainable bias."""

    def __init__(self, base, rank, scale=1.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(base)}")
        if isinstance(rank, bool) or not isinstance(rank, Integral) or not 0 < rank <= min(
            base.in_features, base.out_features
        ):
            raise ValueError(
                "LoRA-RL rank must be a positive integer no larger than either "
                f"selected matrix dimension ({base.out_features}, {base.in_features})."
            )
        if not math.isfinite(float(scale)) or float(scale) <= 0:
            raise ValueError("LoRA-RL scale must be finite and positive.")
        self.base = base
        self.base.requires_grad_(True)
        self.base.weight.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(scale)
        self.lora_A = nn.Parameter(base.weight.new_empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_empty(base.out_features, self.rank))
        self.reset_adapters_()

    @torch.no_grad()
    def reset_adapters_(self):
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0 / math.sqrt(self.rank))
        self.lora_B.zero_()

    def effective_weight(self):
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x):
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return F.linear(x, self.base.weight, self.base.bias) + self.scaling * delta


class LoRARLNormedLinear(LoRARLLinear):
    """Insert BA before the inherited dropout, LayerNorm, and activation."""

    def __init__(self, base, rank, scale=1.0):
        if not isinstance(base, NormedLinear):
            raise TypeError(f"Expected NormedLinear, got {type(base)}")
        super().__init__(base, rank, scale)

    def forward(self, x):
        out = super().forward(x)
        if self.base.dropout is not None:
            out = self.base.dropout(out)
        return self.base.act(self.base.ln(out))


def _adapters(module):
    return {
        path: child for path, child in module.named_modules()
        if isinstance(child, LoRARLLinear)
    }


def make_lora_rl_critic(critic, *, rank=96, scale=1.0, placement="input_hidden"):
    """Own a prior copy and adapt selected non-output layers of every Q head.

    ``input_hidden`` adapts input and hidden matrices. ``hidden`` leaves the
    input projection trainable, following the paper's placement more closely.
    Rank is never silently clipped, including on smaller test architectures.
    """
    if placement not in {"input_hidden", "hidden"}:
        raise ValueError("LoRA-RL placement must be 'input_hidden' or 'hidden'.")
    if not isinstance(critic, (Ensemble, nn.Sequential)):
        raise TypeError("LoRA-RL requires a critic Ensemble or Sequential Q head.")
    clone = deepcopy(critic).requires_grad_(True)
    heads = list(clone) if isinstance(clone, Ensemble) else [clone]
    for head in heads:
        if not isinstance(head, nn.Sequential) or len(head) < 3 or not all(
            isinstance(layer, nn.Linear) for layer in head
        ):
            raise ValueError(
                "LoRA-RL requires Sequential Q heads with input, hidden, and output Linear layers."
            )
        start = 0 if placement == "input_hidden" else 1
        for index in range(start, len(head) - 1):
            layer = head[index]
            adapter_type = LoRARLNormedLinear if isinstance(layer, NormedLinear) else LoRARLLinear
            head[index] = adapter_type(layer, rank=rank, scale=scale)
    return clone


def _dense_state(module, *, effective):
    """Return dense-layout state, referencing owned tensors except BA kernels."""
    adapters = _adapters(module)
    if not adapters:
        raise ValueError("No LoRA-RL adapters were found.")
    state = module.state_dict()
    result = {}
    for name, value in state.items():
        matched = False
        for path, adapter in adapters.items():
            prefix = f"{path}." if path else ""
            if name in {prefix + "lora_A", prefix + "lora_B"}:
                matched = True
                break
            if name.startswith(prefix + "base."):
                suffix = name[len(prefix + "base."):]
                result[prefix + suffix] = (
                    adapter.effective_weight().detach()
                    if effective and suffix == "weight" else value
                )
                matched = True
                break
        if not matched:
            result[name] = value
    return result


def _validate_dense_state(source, target):
    if source.keys() != target.keys():
        raise ValueError("LoRA-RL source and dense destination state layouts must match.")
    for name, value in source.items():
        if value.shape != target[name].shape:
            raise ValueError(f"LoRA-RL dense state shape mismatch for {name}.")
        if value.dtype != target[name].dtype or value.device != target[name].device:
            raise ValueError(f"LoRA-RL dense state device/dtype mismatch for {name}.")


@torch.no_grad()
def reset_lora_rl_critic_(adapted, dense_prior):
    """Restore every prior tensor and fresh adapters without replacing parameters."""
    destination = _dense_state(adapted, effective=False)
    source = dense_prior.state_dict()
    _validate_dense_state(source, destination)
    torch._foreach_copy_(
        list(destination.values()), [source[name] for name in destination]
    )
    for adapter in _adapters(adapted).values():
        adapter.reset_adapters_()


@torch.no_grad()
def dense_lora_rl_critic(adapted):
    """Create an independent, frozen dense target with effective online weights."""
    source = _dense_state(adapted, effective=True)
    target = deepcopy(adapted)
    for path, adapter in _adapters(target).items():
        if not path:
            target = adapter.base
        else:
            parent_path, _, name = path.rpartition(".")
            parent = target.get_submodule(parent_path)
            setattr(parent, name, adapter.base)
    _validate_dense_state(source, target.state_dict())
    target.load_state_dict(source, strict=True)
    return target.requires_grad_(False).eval()


@torch.no_grad()
def update_lora_rl_target_(adapted, dense_target, tau):
    """Polyak-average effective kernels and dense auxiliaries in weight space."""
    tau = float(tau)
    if not math.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError("LoRA-RL target tau must be in [0, 1].")
    source = _dense_state(adapted, effective=True)
    destination = dense_target.state_dict()
    _validate_dense_state(source, destination)
    floating_names = [name for name, value in source.items() if value.is_floating_point()]
    target_values = [destination[name] for name in floating_names]
    source_values = [source[name] for name in floating_names]
    if tau == 1:
        torch._foreach_copy_(target_values, source_values)
    else:
        torch._foreach_lerp_(target_values, source_values, tau)
    for name, value in source.items():
        if not value.is_floating_point():
            destination[name].copy_(value)


def lora_rl_parameter_groups(adapted, weight_decay):
    """Use decoupled decay only for adapter factors; retain ordinary Adam elsewhere."""
    weight_decay = float(weight_decay)
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("LoRA-RL adapter weight decay must be finite and nonnegative.")
    adapters = _adapters(adapted)
    if not adapters:
        raise ValueError("No LoRA-RL adapters were found.")
    factors = [value for adapter in adapters.values() for value in (adapter.lora_A, adapter.lora_B)]
    factor_ids = {id(value) for value in factors}
    ordinary = [value for value in trainable_parameters(adapted) if id(value) not in factor_ids]
    return [
        {"params": factors, "weight_decay": weight_decay},
        {"params": ordinary, "weight_decay": 0.0},
    ]


def trainable_parameters(module):
    """Return only parameters that an inner optimizer is allowed to update."""
    return [parameter for parameter in module.parameters() if parameter.requires_grad]
