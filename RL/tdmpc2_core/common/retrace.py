"""Detached finite-trajectory Retrace targets for the AMBI inner critic.

This is an independent PyTorch implementation of the backward return in
Munos et al. (2016), with the soft-value construction used by Palenicek et al.
(ICLR 2023). The reviewed reference is ``model_based_rl/{targets,rlax}.py`` at
https://github.com/danielpalenicek/value_expansion/tree/
c7fca6a19062f0e6c3e922b992cb0eb93495ac2b . No JAX code or dependency is vendored.
"""

import torch
from torch import Tensor


@torch.no_grad()
def retrace_targets(
    reward: Tensor,
    discount: Tensor,
    bootstrap_value: Tensor,
    behavior_action_q: Tensor,
    trace_coefficients: Tensor,
    valid: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return every suffix target, effective trace length, and TD correction.

    All inputs have shape ``[batch, horizon, 1]``. ``discount`` already includes
    gamma and the true-termination mask. ``bootstrap_value[:, t]`` is the value
    at the next state of row t: an inner soft value in the interior and the
    caller's frozen outer continuation at an imagined horizon boundary.
    ``behavior_action_q`` and ``trace_coefficients`` refer to the recorded
    action at their *own* row. Thus the correction at t uses Q and c at t+1.
    The first row's Q/c and final row's successor Q/c are never needed.

    A missing/invalid successor cuts the trace but retains that row's supplied
    one-step bootstrap. A zero discount removes both bootstrap and trace.
    Invalid rows return zero, and irrelevant inputs are masked before arithmetic
    so padding, terminal values, and zero-weight Q values may contain NaNs.
    Relevant rewards, discounts, values, and coefficients must be finite; the
    caller constructs coefficients in [0, 1] from the actual policy densities.

    Effective length is 1 + c[t+1] * length[t+1] for a valid continuation,
    without multiplying by gamma. It counts the trace-weighted number of
    participating TD errors, is one for a one-step target, and zero for padding.
    Corrections equal target - (reward + discount * bootstrap_value), with the
    same masking. All returned tensors are detached; inputs are not modified.
    The loop depends only on the fixed horizon and supports torch.compile.
    """
    if reward.ndim != 3 or reward.shape[-1] != 1 or reward.shape[1] < 1:
        raise ValueError("Retrace inputs must have shape [batch, positive horizon, 1].")
    for value in (discount, bootstrap_value, behavior_action_q, trace_coefficients, valid):
        if value.shape != reward.shape:
            raise ValueError("All Retrace inputs must have the same [batch, horizon, 1] shape.")

    valid = valid.to(dtype=torch.bool)
    zero = torch.zeros_like(reward)
    rewards = torch.where(valid, reward, zero)
    discounts = torch.where(valid, discount, zero)
    continuing = valid & (discounts != 0)
    values = torch.where(continuing, bootstrap_value, zero)
    one_step = rewards + discounts * values

    targets = []
    lengths = []
    next_target = torch.zeros_like(reward[:, 0])
    next_length = torch.zeros_like(next_target)
    for t in range(reward.shape[1] - 1, -1, -1):
        target = one_step[:, t]
        length = valid[:, t].to(dtype=reward.dtype)
        if t + 1 < reward.shape[1]:
            successor = continuing[:, t] & valid[:, t + 1]
            coefficient = torch.where(successor, trace_coefficients[:, t + 1], 0.0)
            active = successor & (coefficient != 0)
            # Mask *before* subtracting/multiplying: 0 * NaN is still NaN.
            successor_q = torch.where(active, behavior_action_q[:, t + 1], 0.0)
            successor_target = torch.where(active, next_target, 0.0)
            successor_length = torch.where(active, next_length, 0.0)
            target = target + discounts[:, t] * coefficient * (successor_target - successor_q)
            length = length + coefficient * successor_length
        targets.append(target)
        lengths.append(length)
        next_target, next_length = target, length

    targets = torch.stack(targets[::-1], dim=1)
    effective_trace_lengths = torch.stack(lengths[::-1], dim=1)
    return targets, effective_trace_lengths, targets - one_step
