"""Observational summaries of existing outer-policy samples.

These helpers never sample, call a model, change module modes, or transfer
statistics to the host. Distributions describe action coordinates, with the
encoded replay root (depth zero) separated from all actor-training depths.
"""

from collections.abc import Mapping
import math

import torch


def _setting(cfg, name, default):
    return cfg.get(name, default) if isinstance(cfg, Mapping) else getattr(cfg, name, default)


def diagnostics_due(cfg, completed_updates):
    """Return whether this completed-update index is a diagnostic milestone."""
    if not bool(_setting(cfg, "outer_policy_diagnostics", False)):
        return False
    completed_updates = int(completed_updates)
    if completed_updates < 0:
        raise ValueError("completed_updates must be nonnegative.")
    early_until = int(_setting(cfg, "outer_policy_diagnostics_early_until", 10000))
    cadence = int(_setting(
        cfg,
        "outer_policy_diagnostics_early_every" if completed_updates <= early_until
        else "outer_policy_diagnostics_every",
        100 if completed_updates <= early_until else 1000,
    ))
    if cadence <= 0 or early_until < 0:
        raise ValueError("Diagnostic cadences must be positive and early_until nonnegative.")
    return completed_updates == 0 or completed_updates % cadence == 0


@torch.no_grad()
def policy_diagnostics(policy_info, action, *, lower, upper, rho):
    """Summarize one existing sample, without changing it or its graph.

    Inputs have shape [depth, batch, action] or [batch, action]. Entropy is the
    joint post-tanh sample -log_pi, in nats. The normalized rho mixture matches
    the outer temperature update; it is distinct from the actor loss's division
    by the number of depths. Histograms have 24 equal-width log-std bins, include
    both endpoints, and retain a coordinate denominator for pooled reporting.
    """
    lower, upper, rho = float(lower), float(upper), float(rho)
    if not math.isfinite(lower) or not math.isfinite(upper) or not lower < upper:
        raise ValueError("Diagnostic log-std bounds must be finite and ordered.")
    if not math.isfinite(rho) or rho < 0:
        raise ValueError("Diagnostic rho must be finite and nonnegative.")

    def depths(value):
        value = value.detach()
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3 or not all(value.shape):
            raise ValueError("Policy diagnostics require nonempty [depth, batch, coordinate] tensors.")
        return value

    actions = depths(action)
    means = depths(policy_info["mean"])
    pre_means = depths(policy_info["pre_tanh_mean"])
    log_stds = depths(policy_info["log_std"])
    if any(value.shape != actions.shape for value in (means, pre_means, log_stds)):
        raise ValueError("Policy action, mean and log-std shapes must match.")
    log_prob = depths(policy_info["log_prob"])
    if log_prob.shape != actions.shape[:-1] + (1,):
        raise ValueError("Policy log_prob must contain one joint density per state.")

    metrics, histograms = {}, {}
    for scope, selection in (("pooled", slice(None)), ("depth0", slice(0, 1))):
        selected_mean = means[selection]
        selected_action = actions[selection]
        selected_pre_mean = pre_means[selection]
        selected_log_std = log_stds[selection]
        prefix = scope + "_"
        metrics[prefix + "coordinate_count"] = actions.new_tensor(selected_action.numel())
        for name, values in (("mean_action", selected_mean), ("sample_action", selected_action)):
            metrics[prefix + name + "_abs_ge_0p99_fraction"] = (values.abs() >= 0.99).float().mean()
            metrics[prefix + name + "_exact_saturation_fraction"] = (values.abs() == 1.0).float().mean()
        metrics[prefix + "pre_tanh_mean_abs_mean"] = selected_pre_mean.abs().mean()
        metrics[prefix + "pre_tanh_mean_abs_max"] = selected_pre_mean.abs().amax()
        for name, values in (("log_std", selected_log_std), ("std", selected_log_std.exp())):
            metrics[prefix + name + "_mean"] = values.mean()
            metrics[prefix + name + "_std"] = values.std(unbiased=False)
            metrics[prefix + name + "_min"] = values.amin()
            metrics[prefix + name + "_max"] = values.amax()
        metrics[prefix + "log_std_lower_exact_fraction"] = (selected_log_std == lower).float().mean()
        metrics[prefix + "log_std_upper_exact_fraction"] = (selected_log_std == upper).float().mean()
        metrics[prefix + "log_std_lower_near_0p1_fraction"] = (selected_log_std <= lower + 0.1).float().mean()
        metrics[prefix + "log_std_upper_near_0p1_fraction"] = (selected_log_std >= upper - 0.1).float().mean()
        metrics[prefix + "log_std_nonfinite_fraction"] = (~torch.isfinite(selected_log_std)).float().mean()
        metrics[prefix + "log_std_outside_bounds_fraction"] = (
            (selected_log_std < lower) | (selected_log_std > upper)
        ).float().mean()
        histograms["log_std_" + scope] = {
            "counts": torch.histc(selected_log_std.float(), bins=24, min=lower, max=upper),
            "edges": torch.linspace(lower, upper, 25, device=log_stds.device, dtype=torch.float32),
            "count": selected_log_std.numel(),
        }

    entropy_per_depth = -log_prob.mean(dim=(1, 2))
    weights = torch.pow(
        log_prob.new_tensor(rho), torch.arange(len(entropy_per_depth), device=log_prob.device)
    )
    normalized_weights = weights / weights.sum()
    metrics["entropy_rho_mean"] = (entropy_per_depth * normalized_weights).sum()
    metrics["entropy_unweighted_mean"] = entropy_per_depth.mean()
    metrics["entropy_depth0"] = entropy_per_depth[0]
    metrics["entropy_actor_loss_weighted"] = (entropy_per_depth * weights).mean()
    return {"metrics": metrics, "histograms": histograms}
