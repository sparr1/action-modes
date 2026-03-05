import math
from typing import Iterable, Optional, Sequence, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def lorafy(
    model,
    r: int = 8,
    lora_alpha: float = 8.0,
    lora_dropout: float = 0.0,
    actor_targets=("latent_pi",),
    critic_targets=("qf",),        # match qf0, qf1, ...
):
    policy = model.policy

    # wrap selected linears
    apply_lora_to_module(policy.actor, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=actor_targets)
    apply_lora_to_module(policy.critic, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=critic_targets)

    # wrap critic_targetso Polyak update sees matching module structure
    if hasattr(policy, "critic_target") and policy.critic_target is not None:
        apply_lora_to_module(policy.critic_target, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=critic_targets)
        for p in policy.critic_target.parameters():
            p.requires_grad_(False)

    # rebuild the optimizers (base linear layer frozen from LoRALinear)
    actor_trainables = [p for p in policy.actor.parameters() if p.requires_grad]
    critic_trainables = [p for p in policy.critic.parameters() if p.requires_grad]

    policy.actor.optimizer = _clone_optimizer(policy.actor.optimizer, actor_trainables)
    policy.critic.optimizer = _clone_optimizer(policy.critic.optimizer, critic_trainables)

    # leave ent_coef_optimizer alone
    return model


def _clone_optimizer(old_opt: torch.optim.Optimizer, params: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
    """
    rebuild optimizer with new params
    """
    opt_cls = type(old_opt)
    defaults = dict(old_opt.defaults)
    defaults.pop("params", None)
    return opt_cls(list(params), **defaults)


def apply_lora_to_module(
    module: nn.Module,
    r: int = 8,
    lora_alpha: float = 8.0,
    lora_dropout: float = 0.0,
    target_modules: Optional[Sequence[str]] = None,
) -> nn.Module:
    """
    Recursively replaces nn.Linear layers with LoRALinear.
    """
    patterns = list(target_modules) if target_modules else []

    def should_apply(path: str) -> bool:
        if not patterns:
            return True
        return any(p in path for p in patterns)

    def check_rank(child: nn.Linear, r: int) -> bool:
        # skip if r > current layer rank
        if min(child.in_features, child.out_features) <= r or r <= 0:
            return False
        return True

    def _recurse(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name

            # avoid double-wrapping
            if isinstance(child, LoRALinear):
                continue

            if isinstance(child, nn.Linear) and should_apply(full_name) and check_rank(child, r):
                setattr(
                    parent,
                    child_name,
                    LoRALinear(
                        base=child,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                    ),
                )
            else:
                _recurse(child, full_name)

    _recurse(module)
    return module


class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a LoRA adapter:
        y = base(x) + (alpha/r) * B(A(dropout(x)))

    - Freezes the base linear layer params.
    - Learns only A (r x in) and B (out x r).
    """
    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(base)}")

        if r < 0:
            raise ValueError("r must be >= 0")

        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features

        # Freeze base layer
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = r
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.r) if self.r > 0 else 0.0
        self.dropout = nn.Dropout(p=float(lora_dropout)) if lora_dropout > 0 else nn.Identity()

        if self.r > 0:
            # Match dtype/device of base weight
            w = self.base.weight
            self.lora_A = nn.Parameter(w.new_zeros((self.r, self.in_features)))   # (r, in)
            self.lora_B = nn.Parameter(w.new_zeros((self.out_features, self.r))) # (out, r)

            # Common init: A ~ Kaiming, B = 0 so initial adapter is a no-op
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r == 0:
            return out

        x_d = self.dropout(x)
        # F.linear: y = x @ W^T + b, with W shaped (out, in)
        lora_out = F.linear(F.linear(x_d, self.lora_A), self.lora_B)  # (..., out)
        return out + lora_out * self.scaling