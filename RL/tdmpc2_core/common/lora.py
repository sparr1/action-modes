"""LoRA adapters that preserve TD-MPC2's NormedLinear computation order."""

from copy import deepcopy
import math
import weakref

import torch.nn as nn
import torch.nn.functional as F

from .layers import NormedLinear


class _LoRABaseReference:
    """Support either an owned base module or an unregistered shared one."""

    def __getattr__(self, name):
        if name == "base":
            reference = self.__dict__.get("_shared_base_ref")
            if reference is not None:
                base = reference()
                if base is None:
                    raise RuntimeError("The shared LoRA base module no longer exists.")
                return base
        return super().__getattr__(name)

    @property
    def shares_base(self):
        return "_shared_base_ref" in self.__dict__

    def _set_base(self, base, *, share_base):
        if share_base:
            # Bypass ``nn.Module.__setattr__`` so the outer module does not
            # become a child of the inner adapter. This keeps it out of
            # parameters(), state_dict(), train()/eval(), to(), and deepcopy().
            self.__dict__["_shared_base_ref"] = weakref.ref(base)
        else:
            self.base = base
            self.base.requires_grad_(False)


class LoRALinear(_LoRABaseReference, nn.Module):
    """LoRA around a plain Linear layer: base(x) + scale * B(A(x))."""

    def __init__(self, base, rank, alpha, dropout=0.0, *, share_base=False):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(base)}")
        self._set_base(base, share_base=bool(share_base))
        requested_rank = int(rank)
        self.rank = max(1, min(requested_rank, base.in_features, base.out_features))
        # Keep the user-selected alpha/r scaling even when a narrow output
        # layer forces the effective matrix rank below r.
        self.scaling = float(alpha) / float(requested_rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(base.weight.new_empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def share_base_(self, base):
        """Rebind a shared adapter after an outer module identity change."""
        if not self.shares_base:
            raise RuntimeError("Cannot rebind an owned LoRA base as shared.")
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(base)}")
        if (
            base.in_features != self.lora_A.shape[1]
            or base.out_features != self.lora_B.shape[0]
        ):
            raise ValueError("Shared LoRA base shape does not match its adapters.")
        self.__dict__["_shared_base_ref"] = weakref.ref(base)
        return self

    def forward(self, x):
        delta = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        if self.shares_base:
            base = self.base
            linear = F.linear(
                x,
                base.weight.detach(),
                None if base.bias is None else base.bias.detach(),
            )
        else:
            linear = self.base(x)
        return linear + self.scaling * delta


class LoRANormedLinear(_LoRABaseReference, nn.Module):
    """LoRA inserted before TD-MPC2's dropout -> LayerNorm -> activation."""

    def __init__(self, base, rank, alpha, dropout=0.0, *, share_base=False):
        super().__init__()
        if not isinstance(base, NormedLinear):
            raise TypeError(f"Expected NormedLinear, got {type(base)}")
        self._set_base(base, share_base=bool(share_base))
        requested_rank = int(rank)
        self.rank = max(1, min(requested_rank, base.in_features, base.out_features))
        # Keep the user-selected alpha/r scaling even when a narrow output
        # layer forces the effective matrix rank below r.
        self.scaling = float(alpha) / float(requested_rank)
        self.lora_dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(base.weight.new_empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if self.shares_base:
            # The shared outer module must not be switched between train/eval
            # by the inner lifecycle. Keep only the stateless architecture and
            # a private dropout/activation mode in this adapter.
            self.shared_base_dropout = (
                nn.Dropout(base.dropout.p, inplace=base.dropout.inplace)
                if base.dropout is not None
                else None
            )
            self.shared_base_act = deepcopy(base.act)

    def share_base_(self, base):
        """Rebind a shared adapter after an outer module identity change."""
        if not self.shares_base:
            raise RuntimeError("Cannot rebind an owned LoRA base as shared.")
        if not isinstance(base, NormedLinear):
            raise TypeError(f"Expected NormedLinear, got {type(base)}")
        if (
            base.in_features != self.lora_A.shape[1]
            or base.out_features != self.lora_B.shape[0]
            or tuple(base.ln.normalized_shape) != (self.lora_B.shape[0],)
        ):
            raise ValueError("Shared LoRA base shape does not match its adapters.")
        self.__dict__["_shared_base_ref"] = weakref.ref(base)
        return self

    def forward(self, x):
        base = self.base
        weight = base.weight.detach() if self.shares_base else base.weight
        bias = base.bias
        if bias is not None and self.shares_base:
            bias = bias.detach()
        linear = F.linear(x, weight, bias)
        delta = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        out = linear + self.scaling * delta
        if self.shares_base:
            if self.shared_base_dropout is not None:
                out = self.shared_base_dropout(out)
            layer_norm = base.ln
            out = F.layer_norm(
                out,
                layer_norm.normalized_shape,
                None if layer_norm.weight is None else layer_norm.weight.detach(),
                None if layer_norm.bias is None else layer_norm.bias.detach(),
                layer_norm.eps,
            )
            return self.shared_base_act(out)
        if base.dropout is not None:
            out = base.dropout(out)
        return base.act(base.ln(out))


def _resolve_lora_scale(rank, alpha, scale):
    rank = int(rank)
    if rank <= 0:
        raise ValueError("LoRA rank must be positive.")
    if scale is not None:
        scale = float(scale)
        if scale <= 0:
            raise ValueError("LoRA scale must be positive.")
        alpha = scale * rank
    return rank, alpha


def lorafy_copy(module, rank=8, alpha=8.0, dropout=0.0, *, scale=None):
    """Deep-copy a module and add trainable LoRA adapters.

    ``alpha`` retains the legacy ``alpha / requested_rank`` convention. New
    callers should pass ``scale`` to specify the actual multiplier directly,
    keeping update magnitude independent from adapter rank.
    """
    rank, alpha = _resolve_lora_scale(rank, alpha, scale)

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


def lorafy_shared(module, rank=8, alpha=8.0, dropout=0.0, *, scale=None):
    """Clone module structure while sharing detached immutable base weights.

    Only the LoRA tensors are registered parameters of the returned module.
    Linear and LayerNorm weights stay owned by ``module`` and are referenced
    weakly, so target copies own independent adapters without copying bases.
    """
    rank, alpha = _resolve_lora_scale(rank, alpha, scale)

    # Reuse adaptable leaves while copying the cheap container structure.
    # Every shared leaf is replaced before parameters are inspected or modes
    # are changed, so this temporary registration cannot mutate the source.
    shared_leaves = {
        id(child): child
        for child in module.modules()
        if isinstance(child, (NormedLinear, nn.Linear))
    }
    clone = deepcopy(module, shared_leaves)
    adapted = 0

    def recurse(source_parent, clone_parent):
        nonlocal adapted
        for name, source_child in list(source_parent.named_children()):
            if isinstance(source_child, NormedLinear):
                setattr(
                    clone_parent,
                    name,
                    LoRANormedLinear(
                        source_child,
                        rank,
                        alpha,
                        dropout,
                        share_base=True,
                    ),
                )
                adapted += 1
            elif isinstance(source_child, nn.Linear):
                setattr(
                    clone_parent,
                    name,
                    LoRALinear(
                        source_child,
                        rank,
                        alpha,
                        dropout,
                        share_base=True,
                    ),
                )
                adapted += 1
            else:
                recurse(source_child, getattr(clone_parent, name))

    recurse(module, clone)
    if adapted == 0:
        raise ValueError("No Linear layers were found for LoRA adaptation.")

    source_parameter_ids = {id(parameter) for parameter in module.parameters()}
    if any(id(parameter) in source_parameter_ids for parameter in clone.parameters()):
        raise RuntimeError("A shared LoRA base was accidentally registered as inner state.")

    clone.requires_grad_(False)
    for child in clone.modules():
        if isinstance(child, (LoRALinear, LoRANormedLinear)):
            child.lora_A.requires_grad_(True)
            child.lora_B.requires_grad_(True)
    return clone


def trainable_parameters(module):
    """Return only parameters that an inner optimizer is allowed to update."""
    return [p for p in module.parameters() if p.requires_grad]
