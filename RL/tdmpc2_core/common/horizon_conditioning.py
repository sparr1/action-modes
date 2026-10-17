"""One-hot remaining-horizon inputs for ordinary dense inner networks.

Only private inner copies are widened. Their first-layer horizon columns start
at zero, preserving the prior function without consuming initialization RNG.
Packing happens before network evaluation so dense and detached compiled paths
continue to receive one ordinary input tensor.
"""

from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from .layers import Ensemble, NormedLinear


def _dense_heads(module):
    if isinstance(module, Ensemble):
        heads = list(module)
    elif isinstance(module, nn.Sequential):
        heads = [module]
    else:
        raise TypeError("Horizon conditioning requires a dense Sequential or Ensemble.")
    if not heads or any(
        not isinstance(head, nn.Sequential)
        or len(head) == 0
        or any(type(layer) not in (nn.Linear, NormedLinear) for layer in head)
        for head in heads
    ):
        raise TypeError("Horizon conditioning requires ordinary dense Linear layers.")
    return heads


@torch.no_grad()
def widen_horizon_module(module, horizon):
    """Append zero horizon columns in place and return the dense inner module.

    This changes first-layer Parameter identities and therefore must run before
    creating optimizers or compiled callables. Later resets preserve identities.
    Every ensemble member owns its own horizon columns.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon <= 0:
        raise ValueError("Horizon conditioning requires a positive integer horizon.")
    horizon = int(horizon)
    heads = _dense_heads(module)
    if any(hasattr(item, "horizon_conditioning_horizon") for item in (module, *heads)):
        raise ValueError("The module is already horizon conditioned.")
    for head in heads:
        first = head[0]
        original = first.weight
        extended = torch.cat(
            (original.detach(), original.new_zeros(first.out_features, horizon)), dim=1,
        )
        first.weight = nn.Parameter(extended, requires_grad=original.requires_grad)
        first.in_features += horizon
        head.horizon_conditioning_horizon = horizon
    module.horizon_conditioning_horizon = horizon
    if isinstance(module, Ensemble):
        module._repr = str(heads[0])
    return module


@torch.no_grad()
def reset_horizon_module(module, source):
    """Restore prior tensors and zero horizon columns without replacing storage."""
    horizon = getattr(module, "horizon_conditioning_horizon", None)
    if horizon is None:
        raise ValueError("The destination module is not horizon conditioned.")
    if hasattr(source, "horizon_conditioning_horizon"):
        raise ValueError("A horizon-conditioned module requires an unconditioned prior.")
    heads, source_heads = _dense_heads(module), _dense_heads(source)
    if len(heads) != len(source_heads):
        raise ValueError("Horizon module and prior must have the same number of heads.")

    copies = []
    # Validate every tensor before changing any destination value.
    for head, source_head in zip(heads, source_heads):
        destination, prior = head.state_dict(), source_head.state_dict()
        first_weight_name = f"{next(iter(head._modules))}.weight"
        if destination.keys() != prior.keys():
            raise ValueError("Horizon module and prior state layouts must match.")
        for name, value in destination.items():
            source_value = prior[name]
            prefix = value[:, :-horizon] if name == first_weight_name else value
            if prefix.shape != source_value.shape:
                raise ValueError(f"Horizon module and prior tensor shapes differ for {name}.")
            if prefix.device != source_value.device or prefix.dtype != source_value.dtype:
                raise ValueError(f"Horizon module and prior device/dtype differ for {name}.")
            copies.append((prefix, source_value))
    for destination, source_value in copies:
        destination.copy_(source_value)
    for head in heads:
        head[0].weight[:, -horizon:].zero_()
    return module


def pack_horizon_input(module, x, remaining_horizon=None):
    """Append e_(h-1) to an input whose leading axes match integer h[..., 1].

    Ordinary prior modules must not receive a horizon. Conditioned modules
    always require one, including when all their horizon columns are zero.
    Explicit one-hot width keeps the tensor path free of device scalar reads.
    """
    horizon = getattr(module, "horizon_conditioning_horizon", None)
    if horizon is None:
        if remaining_horizon is not None:
            raise ValueError("An unconditioned module cannot receive remaining_horizon.")
        return x
    if remaining_horizon is None:
        raise ValueError("A horizon-conditioned module requires remaining_horizon.")
    if not torch.is_tensor(remaining_horizon):
        raise TypeError("remaining_horizon must be an integer tensor with shape [..., 1].")
    if remaining_horizon.shape != (*x.shape[:-1], 1):
        raise ValueError("remaining_horizon must match the input leading axes with width 1.")
    if remaining_horizon.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("remaining_horizon must have an integer dtype.")
    if remaining_horizon.device != x.device:
        raise ValueError("remaining_horizon and input must be on the same device.")
    features = F.one_hot(remaining_horizon.squeeze(-1).long() - 1, num_classes=horizon)
    return torch.cat((x, features.to(dtype=x.dtype)), dim=-1)
