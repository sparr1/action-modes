"""Detached diagnostics from an existing policy sample, without sampling again."""

import math

import torch


def policy_sample_diagnostics(action, policy_info, prefix="actor_"):
    """Joint entropies in nats and saturation fractions over action coordinates.

    ``mean`` is the policy's already-squashed deterministic action. Pre-tanh
    entropy is analytic; post-tanh entropy is the Monte Carlo estimate from the
    same sample used by the actor. Missing optional policy fields are omitted.
    """
    action = action.detach()
    metrics = {
        f"{prefix}action_abs_ge_0p99_fraction": (
            action.abs() >= 0.99
        ).to(action.dtype).mean(),
    }
    if "log_std" in policy_info:
        log_std = policy_info["log_std"].detach()
        metrics[f"{prefix}pre_tanh_entropy"] = (
            log_std + 0.5 * math.log(2.0 * math.pi * math.e)
        ).sum(dim=-1).mean()
    if "log_prob" in policy_info:
        metrics[f"{prefix}post_tanh_entropy"] = (
            -policy_info["log_prob"].detach().mean()
        )
    if "mean" in policy_info:
        mean_abs = policy_info["mean"].detach().abs()
        metrics[f"{prefix}mean_action_abs_ge_0p99_fraction"] = (
            mean_abs >= 0.99
        ).to(action.dtype).mean()
        metrics[f"{prefix}mean_action_exact_saturation_fraction"] = (
            mean_abs == 1.0
        ).to(action.dtype).mean()
    return metrics


def entropy_coefficient_diagnostics(
    coefficient, q_scale, entropy_mode, action_dim, prefix="actor_",
):
    """Record the coefficient actually used in ``Q/S + alpha * entropy``.

    Return units multiply alpha by S. Joint units also undo TD-MPC2's action
    dimension multiplier; this is nominal for its epsilon-regularized entropy
    ratio. It does not equate pre- and post-tanh entropy objectives.
    """
    # Native TD-AMBI supplies a Python coefficient and a device-resident scale.
    # Keep diagnostics on that device instead of copying the scale to the host.
    if not torch.is_tensor(coefficient) and torch.is_tensor(q_scale):
        alpha = q_scale.new_tensor(coefficient).detach().reshape(())
    else:
        alpha = torch.as_tensor(coefficient).detach().reshape(())
    scale = (
        torch.ones_like(alpha)
        if q_scale is None
        else torch.as_tensor(q_scale, device=alpha.device, dtype=alpha.dtype)
        .detach().reshape(())
    )
    return_units = alpha * scale
    joint_multiplier = int(action_dim) if entropy_mode == "tdmpc2_scaled" else 1
    return {
        f"{prefix}entropy_coefficient_normalized": alpha.clone(),
        f"{prefix}entropy_coefficient_return_units": return_units,
        f"{prefix}entropy_coefficient_joint_units": return_units * joint_multiplier,
    }
