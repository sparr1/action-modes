"""LoRA adapters that preserve TD-MPC2's NormedLinear computation order."""

from copy import deepcopy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import NormedLinear


class LoRALinear(nn.Module):
    """LoRA around a plain Linear layer: base(x) + scale * B(A(x))."""

    def __init__(self, base, rank, alpha, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(base)}")
        self.base = base
        self.base.requires_grad_(False)
        requested_rank = int(rank)
        self.rank = max(1, min(requested_rank, base.in_features, base.out_features))
        # Keep the user-selected alpha/r scaling even when a narrow output
        # layer forces the effective matrix rank below r.
        self.scaling = float(alpha) / float(requested_rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(base.weight.new_empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        delta = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + self.scaling * delta


class LoRANormedLinear(nn.Module):
    """LoRA inserted before TD-MPC2's dropout -> LayerNorm -> activation."""

    def __init__(self, base, rank, alpha, dropout=0.0):
        super().__init__()
        if not isinstance(base, NormedLinear):
            raise TypeError(f"Expected NormedLinear, got {type(base)}")
        self.base = base
        self.base.requires_grad_(False)
        requested_rank = int(rank)
        self.rank = max(1, min(requested_rank, base.in_features, base.out_features))
        # Keep the user-selected alpha/r scaling even when a narrow output
        # layer forces the effective matrix rank below r.
        self.scaling = float(alpha) / float(requested_rank)
        self.lora_dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(base.weight.new_empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        linear = F.linear(x, self.base.weight, self.base.bias)
        delta = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        out = linear + self.scaling * delta
        if self.base.dropout is not None:
            out = self.base.dropout(out)
        return self.base.act(self.base.ln(out))


def lorafy_copy(module, rank=8, alpha=8.0, dropout=0.0):
    """Deep-copy a module, freeze its base weights, and add trainable LoRA adapters."""
    rank = int(rank)
    if rank <= 0:
        raise ValueError("LoRA rank must be positive.")

    clone = deepcopy(module)
    clone.requires_grad_(False)
    adapted = 0

    def recurse(parent):
        nonlocal adapted
        for name, child in list(parent.named_children()):
            if isinstance(child, NormedLinear):
                setattr(parent, name, LoRANormedLinear(child, rank, alpha, dropout))
                adapted += 1
            elif isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child, rank, alpha, dropout))
                adapted += 1
            else:
                recurse(child)

    recurse(clone)
    if adapted == 0:
        raise ValueError("No Linear layers were found for LoRA adaptation.")
    return clone


def trainable_parameters(module):
    """Return only parameters that an inner optimizer is allowed to update."""
    return [p for p in module.parameters() if p.requires_grad]
