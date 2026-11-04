"""Corrected Gaussian action-alignment objective adapted from released BOOM.

Reference: molumitu/BOOM_MBRL, commit
``b3b2f09c03533c15b804ced9abc6767e73ff8e1e``, ``boom/boom_alg.py:update_pi``.
This preserves sampled-action alignment, detached actor standard deviations,
actor-Q batch weights, and the shared Q scale/gate. It deliberately removes
the released Gaussian helper's extra action-dimension multiplier. The caller
defines the replay target and owns the Q-scale update and actor's RL loss.
"""

import math

import torch


def boom_alignment_loss(
    action,
    actor_log_std,
    target_action,
    valid,
    q_values,
    scale,
    *,
    rho,
    coefficient,
    min_std=0.05,
):
    """Return BOOM alignment and detached scalar diagnostics.

    Actions, log standard deviations, and replay targets have shape ``[T,B,D]``.
    ``valid`` is boolean ``[T,B]`` or ``[T,B,1]``; reduced, raw actor-action Q
    values have either of those shapes. ``scale`` is a scalar tensor containing
    the already-updated positive Q range. The Q weights, replay targets, scale,
    and floored standard deviations are all constants for actor autograd.

    Each depth uses a softmax of ``Q / scale`` over valid batch rows. Its
    Gaussian negative log score is summed over action coordinates, weighted
    over the batch, and reduced as ``mean(rho**t * loss_t)``. Empty depths
    contribute zero and retain their place in that temporal denominator.
    Alignment is divided by scale and enabled strictly when ``scale > 2``.

    Invalid rows may contain missing-data NaNs. Fixed-shape masking sanitizes
    those rows before arithmetic, including fully invalid minibatches. There
    are no tensor-to-host conversions or data-dependent Python branches.
    """
    if action.ndim != 3 or min(action.shape) < 1:
        raise ValueError("BOOM actions must have nonempty shape [T,B,D].")
    if actor_log_std.shape != action.shape or target_action.shape != action.shape:
        raise ValueError("BOOM action, actor_log_std, and target_action shapes must match.")
    if valid.ndim == 3 and valid.shape[-1] == 1:
        valid = valid.squeeze(-1)
    if q_values.ndim == 3 and q_values.shape[-1] == 1:
        q_values = q_values.squeeze(-1)
    if valid.shape != action.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("BOOM validity must be boolean [T,B] or [T,B,1].")
    if q_values.shape != action.shape[:2]:
        raise ValueError("BOOM Q values must have shape [T,B] or [T,B,1].")
    if not math.isfinite(min_std) or min_std <= 0:
        raise ValueError("BOOM min_std must be finite and positive.")
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("BOOM coefficient must be finite and nonnegative.")
    if not math.isfinite(rho) or rho < 0:
        raise ValueError("BOOM rho must be finite and nonnegative.")
    if scale.numel() != 1:
        raise ValueError("BOOM Q scale must contain one value.")

    scale = scale.detach().to(action).reshape(())
    mask = valid.unsqueeze(-1)
    safe_action = torch.where(mask, action, 0.0)
    target = torch.where(mask, target_action.detach(), 0.0)
    log_std = torch.where(mask, actor_log_std.detach(), 0.0)
    std = log_std.exp().clamp_min(min_std)

    # The Gaussian is an action-space attraction score with a detached scale,
    # not the actor's squashed density at a replay action. Sum once, without the
    # extra dimension multiplier in the released BOOM Gaussian helper.
    residual = (safe_action - target) / std
    nll = (0.5 * residual.square() + std.log() + 0.5 * math.log(2 * math.pi)).sum(-1)

    has_valid = valid.any(dim=1, keepdim=True)
    logits = torch.where(valid, q_values.detach() / scale, -torch.inf)
    # Softmax over an all-masked row would be NaN. Replace only those rows
    # before softmax, then zero their otherwise-uniform weights with the mask.
    logits = torch.where(has_valid, logits, 0.0)
    weights = logits.softmax(dim=1) * valid.to(action.dtype)
    per_depth_nll = (weights * nll).sum(dim=1)
    temporal_weights = rho ** torch.arange(action.shape[0], device=action.device, dtype=action.dtype)
    alignment_nll = (temporal_weights * per_depth_nll).mean()
    gate = scale > 2.0
    scaled_loss = torch.where(gate, alignment_nll / scale, torch.zeros_like(alignment_nll))
    loss = coefficient * scaled_loss

    tiny = torch.finfo(weights.dtype).tiny
    weight_entropy = -(weights * weights.clamp_min(tiny).log()).sum(dim=1)
    effective_samples = torch.where(
        has_valid.squeeze(1),
        weights.square().sum(dim=1).clamp_min(tiny).reciprocal(),
        0.0,
    )
    metrics = {
        "boom_alignment_nll": alignment_nll.detach(),
        "boom_scaled_alignment_loss": scaled_loss.detach(),
        "boom_alignment_loss": loss.detach(),
        "boom_gate_active": gate.to(action.dtype),
        "boom_valid_fraction": valid.to(action.dtype).mean(),
        "boom_weight_entropy": weight_entropy.mean().detach(),
        "boom_effective_samples": effective_samples.mean().detach(),
    }
    return loss, metrics
