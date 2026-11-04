"""Fixed-shape kernels for the corrected TD-M(PC)^2 behavior objective."""

import torch

from .common import math as td_math


def squashed_behavior_cross_entropy(
    current_mean,
    current_log_std,
    current_pre_tanh_action,
    behavior_mean,
    behavior_log_std,
    valid,
    temporal_weights,
    min_valid_count,
):
    """Return per-coordinate CE, readiness, and detached scalar diagnostics.

    Inputs have shape H x B x D, validity H x B, and temporal weights H.
    Masking and quantile ranks stay on device, with no dynamic-sized indexing
    or host scalar reads. This pure kernel neither samples nor updates state,
    so compilation and an eager retry share the same actor sample.
    """
    weights = temporal_weights.to(dtype=current_mean.dtype).reshape(-1, 1)
    weighted_valid = weights * valid.to(dtype=current_mean.dtype)
    denominator = weighted_valid.sum()
    valid_count = valid.sum()
    supported = (valid_count > 0) & torch.isfinite(denominator) & (denominator > 0)
    safe_denominator = torch.where(supported, denominator, 1.0)
    weighted_valid = torch.where(supported, weighted_valid, 0.0)
    mask = (valid & supported).unsqueeze(-1)
    current_mean = torch.where(mask, current_mean, 0.0)
    current_log_std = torch.where(mask, current_log_std, 0.0)
    current_pre_tanh_action = torch.where(mask, current_pre_tanh_action, 0.0)
    behavior_mean = torch.where(mask, behavior_mean.detach(), 0.0)
    behavior_log_std = torch.where(mask, behavior_log_std.detach(), 0.0)
    action_dim = current_mean.shape[-1]
    gaussian = td_math.diagonal_gaussian_cross_entropy(
        current_mean, current_log_std, behavior_mean, behavior_log_std,
    ).squeeze(-1) / action_dim
    jacobian = td_math.tanh_log_abs_det_jacobian(
        current_pre_tanh_action,
    ).squeeze(-1) / action_dim
    ce = gaussian + jacobian

    # Mask inputs before arithmetic above, and mask the reductions as well.
    # A batch without weighted support must still have zero actor gradients.
    loss = torch.where(supported, (ce * weighted_valid).sum() / safe_denominator, 0.0)
    ready = supported & (valid_count >= min_valid_count)

    # Both exact, linearly interpolated quantiles share one fixed-size sort.
    # Invalid entries sort after all finite valid entries. Dynamic ranks are
    # tensor gathers rather than boolean indexing or a Python validity branch.
    values = ce.detach()
    ordered = torch.where(valid, values, float("inf")).flatten().sort().values
    last = (valid_count - 1).clamp_min(0).to(dtype=ce.dtype)
    ranks = last * ce.new_tensor([0.5, 0.95])
    lower = ranks.floor().to(dtype=torch.long)
    upper = ranks.ceil().to(dtype=torch.long)
    lows = torch.where(supported, ordered.gather(0, lower), 0.0)
    highs = torch.where(supported, ordered.gather(0, upper), 0.0)
    quantiles = torch.lerp(lows, highs, ranks - lower.to(dtype=ce.dtype))
    # Match torch.quantile's NaN propagation for genuinely invalid model
    # outputs, without propagating NaN placeholders from missing replay labels.
    has_nan = (valid & values.isnan()).any() & supported
    quantiles = torch.where(has_nan, float("nan"), quantiles)

    metrics = {
        "behavior_policy_action_ce": loss.detach(),
        "behavior_policy_action_ce_gaussian": torch.where(
            supported, (gaussian.detach() * weighted_valid).sum() / safe_denominator, 0.0,
        ),
        "behavior_policy_action_ce_log_abs_det_jacobian": torch.where(
            supported, (jacobian.detach() * weighted_valid).sum() / safe_denominator, 0.0,
        ),
        "behavior_policy_action_ce_p50": quantiles[0],
        "behavior_policy_action_ce_p95": quantiles[1],
        "behavior_policy_action_ce_valid_count": valid_count.to(dtype=ce.dtype),
        "behavior_policy_action_ce_valid_fraction": valid_count.to(dtype=ce.dtype) / valid.numel(),
        "behavior_policy_action_ce_ready": ready.to(dtype=ce.dtype),
    }
    return loss, ready, metrics
