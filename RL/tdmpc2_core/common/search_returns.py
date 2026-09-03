"""Pure finite-horizon return operators used by AMBI search.

All sequence inputs are batch-major ``[batch, time, 1]`` (the trailing
singleton may be omitted).  These helpers intentionally operate on scalar
values, after any distributional critic prediction has been decoded.

For soft-Q returns, callers pass a bootstrap produced by :func:`soft_value`.
This makes the entropy convention explicit: entropy belongs to the *future*
action value and never to the Q target's conditioned current action.  Linked
trajectory log-probabilities are accepted separately for the intermediate
future actions exposed by an n-step expansion.
"""

from __future__ import annotations

from math import isfinite

import torch


def _scalar_sequence(value, name):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor.")
    if value.ndim == 1:
        value = value[:, None, None]
    elif value.ndim == 2:
        value = value.unsqueeze(-1)
    elif value.ndim != 3:
        raise ValueError(f"{name} must have one batch and one time dimension.")
    if value.shape[-1] != 1:
        raise ValueError(f"{name} must have a trailing singleton dimension.")
    return value


def _matching_sequence(value, reference, name):
    value = _scalar_sequence(value, name)
    if value.shape != reference.shape:
        raise ValueError(
            f"{name} must have shape {tuple(reference.shape)}, got {tuple(value.shape)}."
        )
    return value


def _mask_sequence(value, reference, name, default):
    if value is None:
        return torch.full_like(reference, default, dtype=torch.bool)
    value = _matching_sequence(value, reference, name)
    if value.dtype is not torch.bool:
        if not bool(torch.logical_or(value == 0, value == 1).all().item()):
            raise ValueError(f"{name} must be binary.")
    return value.to(dtype=torch.bool)


def _validate_discount(discount):
    if torch.is_tensor(discount):
        if discount.numel() != 1:
            raise ValueError("discount must be scalar.")
        return discount
    discount = float(discount)
    if not isfinite(discount) or discount < 0:
        raise ValueError(f"discount must be finite and non-negative, got {discount}.")
    return discount


def _validate_lambda(trace_lambda):
    trace_lambda = float(trace_lambda)
    if not isfinite(trace_lambda) or not 0 <= trace_lambda <= 1:
        raise ValueError(
            f"trace lambda must lie in [0, 1], got {trace_lambda}."
        )
    return trace_lambda


def _steps_vector(steps, batch_size, max_steps, device):
    if isinstance(steps, bool):
        raise TypeError("return steps must be an integer or integer tensor.")
    if isinstance(steps, int):
        result = torch.full((batch_size,), steps, device=device, dtype=torch.long)
    elif torch.is_tensor(steps):
        if steps.is_floating_point() or steps.dtype is torch.bool:
            raise TypeError("return steps tensor must have an integer dtype.")
        if steps.numel() == 1:
            result = steps.reshape(1).expand(batch_size).to(device=device, dtype=torch.long)
        elif steps.shape in {(batch_size,), (batch_size, 1)}:
            result = steps.reshape(batch_size).to(device=device, dtype=torch.long)
        else:
            raise ValueError("return steps tensor must contain one value per batch row.")
    else:
        raise TypeError("return steps must be an integer or integer tensor.")
    if bool(torch.logical_or(result < 1, result > max_steps).any().item()):
        raise ValueError(f"return steps must lie in [1, {max_steps}].")
    return result


def _column(value, batch_size, reference, name):
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        value = value.reshape(1, 1).expand(batch_size, 1)
    elif value.shape == (batch_size,):
        value = value.unsqueeze(-1)
    elif value.shape == (batch_size, 1, 1):
        value = value[:, 0]
    if value.shape != (batch_size, 1):
        raise ValueError(f"{name} must contain one scalar per batch row.")
    return value


def _sequence_parts(rewards, terminated, valid):
    rewards = _scalar_sequence(rewards, "rewards")
    terminated = _mask_sequence(terminated, rewards, "terminated", False)
    valid = _mask_sequence(valid, rewards, "valid", True)
    continuation = valid & ~terminated
    return rewards, terminated, valid, continuation


def _prefix_before(continuation, dtype):
    continuation = continuation.to(dtype=dtype)
    ones = torch.ones_like(continuation[:, :1])
    if continuation.shape[1] == 1:
        return ones
    return torch.cat((ones, torch.cumprod(continuation[:, :-1], dim=1)), dim=1)


def soft_value(q_value, log_prob=None, entropy_coefficient=0.0):
    """Convert a sampled Q into SAC's future-action value ``Q-alpha log pi``.

    Passing ``log_prob=None`` is the reward-only convention.  No current-action
    entropy is added by any target builder; callers apply this function only to
    the value at the transition successor (including an outer-policy leaf).
    """
    if not torch.is_tensor(q_value):
        raise TypeError("q_value must be a tensor.")
    if log_prob is None:
        if torch.is_tensor(entropy_coefficient):
            if bool((entropy_coefficient != 0).any().item()):
                raise ValueError(
                    "A non-zero entropy coefficient requires a sampled log_prob."
                )
        elif float(entropy_coefficient) != 0:
            raise ValueError(
                "A non-zero entropy coefficient requires a sampled log_prob."
            )
        return q_value
    if not torch.is_tensor(log_prob):
        raise TypeError("log_prob must be a tensor.")
    try:
        return q_value - entropy_coefficient * log_prob
    except RuntimeError as error:
        raise ValueError("q_value and log_prob are not broadcast-compatible.") from error


def importance_sampling_ratios(
    target_log_prob,
    behavior_log_prob,
    *,
    valid=None,
    max_abs_log_ratio=None,
):
    """Compute exact ``pi/mu`` ratios from matching squashed-policy densities.

    The default performs no numerical or algorithmic clipping.  The optional
    ``max_abs_log_ratio`` is solely an explicit numerical guard and must not be
    used for experiments claiming exact PDIS.
    """
    target_log_prob = _scalar_sequence(target_log_prob, "target_log_prob")
    behavior_log_prob = _matching_sequence(
        behavior_log_prob, target_log_prob, "behavior_log_prob"
    )
    log_ratio = target_log_prob - behavior_log_prob
    if max_abs_log_ratio is not None:
        max_abs_log_ratio = float(max_abs_log_ratio)
        if not isfinite(max_abs_log_ratio) or max_abs_log_ratio <= 0:
            raise ValueError("max_abs_log_ratio must be finite and positive.")
        log_ratio = log_ratio.clamp(-max_abs_log_ratio, max_abs_log_ratio)
    ratios = torch.exp(log_ratio)
    if valid is not None:
        valid = _mask_sequence(valid, target_log_prob, "valid", True)
        ratios = torch.where(valid, ratios, torch.ones_like(ratios))
    return ratios


def importance_ratio_diagnostics(ratios, *, valid=None, clip=None):
    """Return ratio moments and (normalized) effective sample size."""
    ratios = _scalar_sequence(ratios, "ratios")
    valid = _mask_sequence(valid, ratios, "valid", True)
    weights = ratios[valid]
    if weights.numel() == 0:
        zero = ratios.new_zeros(())
        return {
            "ratio_mean": zero,
            "ratio_max": zero,
            "ratio_clipped_fraction": zero,
            "ess": zero,
            "normalized_ess": zero,
            "ratio_count": zero,
        }
    weight_sum = weights.sum()
    ess = weight_sum.square() / weights.square().sum().clamp_min(
        torch.finfo(weights.dtype).tiny
    )
    if clip is None:
        clipped_fraction = weights.new_zeros(())
    else:
        clip = float(clip)
        if not isfinite(clip) or clip <= 0:
            raise ValueError("ratio diagnostic clip must be finite and positive.")
        clipped_fraction = (weights > clip).to(weights.dtype).mean()
    count = weights.new_tensor(float(weights.numel()))
    return {
        "ratio_mean": weights.mean(),
        "ratio_max": weights.max(),
        "ratio_clipped_fraction": clipped_fraction,
        "ess": ess,
        "normalized_ess": ess / count,
        "ratio_count": count,
    }


def n_step_target(
    rewards,
    bootstrap_value,
    *,
    steps,
    discount,
    terminated=None,
    valid=None,
    action_log_probs=None,
    entropy_coefficient=0.0,
    importance_ratios=None,
    return_diagnostics=False,
):
    """Build fixed n-step targets, optionally with per-decision IS.

    ``bootstrap_value`` is the value at the successor after ``steps`` and must
    already include that sampled action's entropy term.  ``action_log_probs``
    is aligned with trajectory actions.  Index zero is deliberately ignored;
    indices 1 through ``steps-1`` contribute ``-alpha log pi``.

    When ratios are provided, action zero remains conditioned and therefore has
    unit weight.  Reward/entropy at future index ``j`` uses ``W_j`` while the
    bootstrap uses ``W_(n-1)``.  In particular, a one-step bootstrap is never
    multiplied by the ratio of an action that has not been followed.
    """
    discount = _validate_discount(discount)
    rewards, _, valid, continuation = _sequence_parts(rewards, terminated, valid)
    batch_size, time = rewards.shape[:2]
    steps = _steps_vector(steps, batch_size, time, rewards.device)
    bootstrap_value = _column(
        bootstrap_value, batch_size, rewards, "bootstrap_value"
    )

    index = torch.arange(time, device=rewards.device).reshape(1, time, 1)
    within = index < steps.reshape(batch_size, 1, 1)
    prefix = _prefix_before(continuation, rewards.dtype)
    active = prefix * valid.to(rewards.dtype) * within.to(rewards.dtype)

    if action_log_probs is None:
        if torch.is_tensor(entropy_coefficient):
            nonzero_entropy = bool((entropy_coefficient != 0).any().item())
        else:
            nonzero_entropy = float(entropy_coefficient) != 0
        if nonzero_entropy and time > 1 and bool((steps > 1).any().item()):
            raise ValueError(
                "Multi-step soft targets require linked future action log-probabilities."
            )
        action_log_probs = torch.zeros_like(rewards)
    else:
        action_log_probs = _matching_sequence(
            action_log_probs, rewards, "action_log_probs"
        )
    # Current-action entropy is always absent.  Future entropy is at the same
    # discount and occupancy/importance coefficient as that future reward.
    future = (index > 0).to(rewards.dtype)
    entropy_terms = -entropy_coefficient * action_log_probs * future

    if importance_ratios is None:
        ratios = torch.ones_like(rewards)
    else:
        ratios = _matching_sequence(
            importance_ratios, rewards, "importance_ratios"
        )
        if bool(torch.logical_or(~torch.isfinite(ratios), ratios < 0).any().item()):
            raise ValueError("importance_ratios must be finite and non-negative.")
    ratio_factors = torch.cat(
        (torch.ones_like(ratios[:, :1]), ratios[:, 1:]), dim=1
    )
    cumulative_weights = torch.cumprod(ratio_factors, dim=1)

    gamma = torch.as_tensor(discount, device=rewards.device, dtype=rewards.dtype)
    discount_power = torch.pow(gamma, index.to(rewards.dtype))
    coefficients = discount_power * active * cumulative_weights
    reward_contribution = (coefficients * rewards).sum(dim=1)
    entropy_contribution = (coefficients * entropy_terms).sum(dim=1)

    gather = (steps - 1).reshape(batch_size, 1, 1)
    survival = torch.cumprod(continuation.to(rewards.dtype), dim=1).gather(1, gather)[:, 0]
    bootstrap_weight = cumulative_weights.gather(1, gather)[:, 0]
    bootstrap_discount = torch.pow(gamma, steps.to(rewards.dtype)).unsqueeze(-1)
    bootstrap_contribution = (
        bootstrap_discount * survival * bootstrap_weight * bootstrap_value
    )
    target = reward_contribution + entropy_contribution + bootstrap_contribution
    if not return_diagnostics:
        return target

    ratio_valid = valid & (index > 0) & within
    diagnostics = {
        "reward_contribution": reward_contribution,
        "entropy_contribution": entropy_contribution,
        "bootstrap_contribution": bootstrap_contribution,
        # Full-suffix callers expose this same tensor as their true leaf term.
        "leaf_contribution": bootstrap_contribution,
        "effective_return_length": active.sum(dim=1),
        "cumulative_importance_weight": bootstrap_weight,
        "bootstrap_importance_weight": bootstrap_weight,
        "importance_ratios": ratios,
        "return_steps": steps.unsqueeze(-1),
        "per_step_coefficients": coefficients,
    }
    diagnostics.update(
        importance_ratio_diagnostics(ratios, valid=ratio_valid)
    )
    if importance_ratios is not None:
        # Per-decision ratio moments describe the local policy mismatch, while
        # the final cumulative weight is the actual trajectory-level weight on
        # this sampled target (including the requested bootstrap W_(n-1)).
        # Report both: an ESS over raw per-step ratios is not the PDIS sample
        # ESS used to diagnose weight degeneracy across a minibatch.
        weight_diagnostics = importance_ratio_diagnostics(bootstrap_weight)
        for name in ("mean", "max", "ess", "normalized_ess"):
            diagnostics[f"pdis_weight_{name}"] = weight_diagnostics[
                "ratio_" + name if name in {"mean", "max"} else name
            ]
    return target, diagnostics


def td0_target(
    reward,
    bootstrap_value,
    *,
    discount,
    terminated=None,
    valid=None,
    return_diagnostics=False,
):
    """One-step Bellman target with a precomputed reward-only or soft value."""
    rewards = _scalar_sequence(reward, "reward")
    return n_step_target(
        rewards,
        bootstrap_value,
        steps=1,
        discount=discount,
        terminated=terminated,
        valid=valid,
        return_diagnostics=return_diagnostics,
    )


def full_suffix_target(
    rewards,
    leaf_value,
    *,
    horizon=None,
    discount,
    terminated=None,
    valid=None,
    action_log_probs=None,
    entropy_coefficient=0.0,
    importance_ratios=None,
    return_diagnostics=False,
):
    """Follow every available transition and bootstrap only from the outer leaf."""
    rewards = _scalar_sequence(rewards, "rewards")
    if horizon is None:
        horizon = rewards.shape[1]
    return n_step_target(
        rewards,
        leaf_value,
        steps=horizon,
        discount=discount,
        terminated=terminated,
        valid=valid,
        action_log_probs=action_log_probs,
        entropy_coefficient=entropy_coefficient,
        importance_ratios=importance_ratios,
        return_diagnostics=return_diagnostics,
    )


def resimulated_suffix_target(*args, **kwargs):
    """Target for a suffix newly rolled out under the current inner policy.

    Resimulation changes how the tensors are produced, not the return algebra.
    Importance weights are therefore forbidden on this exact on-policy suffix.
    """
    if kwargs.get("importance_ratios") is not None:
        raise ValueError("Resimulated suffix targets do not use importance ratios.")
    return full_suffix_target(*args, **kwargs)


def lambda_return_target(
    rewards,
    bootstrap_values,
    *,
    horizon=None,
    trace_lambda,
    discount,
    terminated=None,
    valid=None,
    action_log_probs=None,
    entropy_coefficient=0.0,
    importance_ratios=None,
    return_diagnostics=False,
):
    """Finite mixture of one- through h-step returns.

    ``bootstrap_values[:, n-1]`` is the already-soft value after ``n`` linked
    transitions.  Its final in-horizon entry is the outer leaf.  The endpoint
    definitions are exact: lambda=0 is TD(0), and lambda=1 is the full suffix.
    """
    trace_lambda = _validate_lambda(trace_lambda)
    rewards = _scalar_sequence(rewards, "rewards")
    bootstrap_values = _matching_sequence(
        bootstrap_values, rewards, "bootstrap_values"
    )
    batch_size, time = rewards.shape[:2]
    if horizon is None:
        horizon = time
    horizon = _steps_vector(horizon, batch_size, time, rewards.device)

    targets = []
    bootstrap_terms = []
    lengths = []
    for n in range(1, time + 1):
        target, diagnostics = n_step_target(
            rewards,
            bootstrap_values[:, n - 1],
            steps=n,
            discount=discount,
            terminated=terminated,
            valid=valid,
            action_log_probs=action_log_probs,
            entropy_coefficient=entropy_coefficient,
            importance_ratios=importance_ratios,
            return_diagnostics=True,
        )
        targets.append(target)
        bootstrap_terms.append(diagnostics["bootstrap_contribution"])
        lengths.append(diagnostics["effective_return_length"])
    targets = torch.stack(targets, dim=1)
    bootstrap_terms = torch.stack(bootstrap_terms, dim=1)
    lengths = torch.stack(lengths, dim=1)

    n = torch.arange(1, time + 1, device=rewards.device).reshape(1, time)
    h = horizon.reshape(batch_size, 1)
    if trace_lambda == 0.0:
        weights = (n == 1).to(rewards.dtype)
    elif trace_lambda == 1.0:
        weights = (n == h).to(rewards.dtype)
    else:
        lam = rewards.new_tensor(trace_lambda)
        ordinary = (1 - lam) * torch.pow(lam, n - 1)
        final = torch.pow(lam, h - 1)
        weights = torch.where(n < h, ordinary, torch.where(n == h, final, 0.0))
    weights = weights.unsqueeze(-1)
    target = (targets * weights).sum(dim=1)
    if not return_diagnostics:
        return target

    final_index = (horizon - 1).reshape(batch_size, 1, 1)
    leaf_bootstrap = bootstrap_terms.gather(1, final_index)[:, 0]
    leaf_weight = weights.gather(1, final_index)[:, 0]
    ratio_tensor = (
        torch.ones_like(rewards)
        if importance_ratios is None
        else _matching_sequence(importance_ratios, rewards, "importance_ratios")
    )
    index = torch.arange(time, device=rewards.device).reshape(1, time, 1)
    ratio_valid = _mask_sequence(valid, rewards, "valid", True) & (index > 0) & (
        index < h.unsqueeze(-1)
    )
    diagnostics = {
        "mixture_weights": weights,
        "component_targets": targets,
        "bootstrap_contribution": (bootstrap_terms * weights).sum(dim=1),
        "leaf_contribution": leaf_bootstrap * leaf_weight,
        "effective_return_length": (lengths * weights).sum(dim=1),
        "return_steps": horizon.unsqueeze(-1),
    }
    diagnostics.update(
        importance_ratio_diagnostics(ratio_tensor, valid=ratio_valid)
    )
    if importance_ratios is not None:
        ratio_factors = torch.cat(
            (torch.ones_like(ratio_tensor[:, :1]), ratio_tensor[:, 1:]), dim=1
        )
        cumulative_weights = torch.cumprod(ratio_factors, dim=1)
        final_weight = cumulative_weights.gather(1, final_index)[:, 0]
        weight_diagnostics = importance_ratio_diagnostics(final_weight)
        for name in ("mean", "max", "ess", "normalized_ess"):
            diagnostics[f"pdis_weight_{name}"] = weight_diagnostics[
                "ratio_" + name if name in {"mean", "max"} else name
            ]
    return target, diagnostics


def retrace_target(
    rewards,
    current_q,
    next_values,
    log_rhos,
    *,
    discount,
    trace_lambda,
    terminated=None,
    valid=None,
    return_diagnostics=False,
):
    """Compute the Retrace TD-error control variate target for the anchor Q.

    ``current_q`` and ``next_values`` must come from one consistent target
    critic selection/reduction.  ``next_values`` already follows the configured
    soft/reward convention.  The conditioned anchor action has no correction;
    subsequent coefficients use ``c_t=lambda*min(1, pi/mu)``.
    """
    discount = _validate_discount(discount)
    trace_lambda = _validate_lambda(trace_lambda)
    rewards, _, valid, continuation = _sequence_parts(rewards, terminated, valid)
    current_q = _matching_sequence(current_q, rewards, "current_q")
    next_values = _matching_sequence(next_values, rewards, "next_values")
    log_rhos = _matching_sequence(log_rhos, rewards, "log_rhos")
    ratios = torch.exp(log_rhos)
    if bool((~torch.isfinite(ratios)).any().item()):
        raise ValueError("Retrace importance ratios must be finite.")
    c = trace_lambda * torch.minimum(torch.ones_like(ratios), ratios)

    gamma = torch.as_tensor(discount, device=rewards.device, dtype=rewards.dtype)
    td_error = (
        rewards
        + gamma * continuation.to(rewards.dtype) * next_values
        - current_q
    ) * valid.to(rewards.dtype)

    coefficients = [torch.ones_like(td_error[:, 0])]
    target = current_q[:, 0] + td_error[:, 0]
    coefficient = coefficients[0]
    for index in range(1, rewards.shape[1]):
        coefficient = (
            coefficient
            * gamma
            * continuation[:, index - 1].to(rewards.dtype)
            * c[:, index]
        )
        coefficients.append(coefficient)
        target = target + coefficient * td_error[:, index]
    coefficients = torch.stack(coefficients, dim=1)
    if not return_diagnostics:
        return target

    # The final valid delta is the only one whose successor value is the outer
    # leaf.  Report just that direct leaf component, not the accompanying
    # control-variate subtraction.
    horizon = valid.squeeze(-1).sum(dim=1).clamp_min(1).to(torch.long)
    final = (horizon - 1).reshape(-1, 1, 1)
    final_coeff = coefficients.gather(1, final)[:, 0]
    final_cont = continuation.to(rewards.dtype).gather(1, final)[:, 0]
    final_next = next_values.gather(1, final)[:, 0]
    leaf_contribution = final_coeff * gamma * final_cont * final_next
    ratio_valid = valid.clone()
    ratio_valid[:, 0] = False
    diagnostics = {
        "td_error": td_error,
        "trace_coefficients": coefficients,
        "trace_c": c,
        "rho": ratios,
        "leaf_contribution": leaf_contribution,
        "effective_return_length": _prefix_before(
            continuation, rewards.dtype
        ).mul(valid).sum(dim=1),
    }
    diagnostics.update(
        importance_ratio_diagnostics(ratios, valid=ratio_valid, clip=1.0)
    )
    return target, diagnostics


def vtrace_targets(
    rewards,
    values,
    next_values,
    log_rhos,
    *,
    discount,
    trace_lambda,
    rho_clip=1.0,
    c_clip=1.0,
    pg_rho_clip=1.0,
    terminated=None,
    valid=None,
):
    """Canonical V-trace state targets and likelihood-ratio PG advantages.

    ``next_values[:, t]`` is the baseline V at transition ``t``'s successor.
    This explicit representation supports both shared and depth-indexed value
    functions without assuming that adjacent tensor rows use the same head.
    """
    discount = _validate_discount(discount)
    trace_lambda = _validate_lambda(trace_lambda)
    rewards, _, valid, continuation = _sequence_parts(rewards, terminated, valid)
    values = _matching_sequence(values, rewards, "values")
    next_values = _matching_sequence(next_values, rewards, "next_values")
    log_rhos = _matching_sequence(log_rhos, rewards, "log_rhos")
    for name, value in (
        ("rho_clip", rho_clip),
        ("c_clip", c_clip),
        ("pg_rho_clip", pg_rho_clip),
    ):
        value = float(value)
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    rho_clip, c_clip, pg_rho_clip = map(
        float, (rho_clip, c_clip, pg_rho_clip)
    )

    ratios = torch.exp(log_rhos)
    if bool((~torch.isfinite(ratios)).any().item()):
        raise ValueError("V-trace importance ratios must be finite.")
    clipped_rho = ratios.clamp(max=rho_clip)
    clipped_pg_rho = ratios.clamp(max=pg_rho_clip)
    c = trace_lambda * ratios.clamp(max=c_clip)
    gamma = torch.as_tensor(discount, device=rewards.device, dtype=rewards.dtype)
    td_error = (
        rewards
        + gamma * continuation.to(rewards.dtype) * next_values
        - values
    )

    batch_size, time = rewards.shape[:2]
    corrected = [None] * time
    corrected_next = [None] * time
    accumulator = torch.zeros_like(values[:, 0])
    for index in range(time - 1, -1, -1):
        if index + 1 < time:
            has_valid_next = valid[:, index + 1]
            next_target = torch.where(
                has_valid_next, accumulator, next_values[:, index]
            )
        else:
            next_target = next_values[:, index]
        candidate = (
            values[:, index]
            + clipped_rho[:, index] * td_error[:, index]
            + gamma
            * continuation[:, index].to(rewards.dtype)
            * c[:, index]
            * (next_target - next_values[:, index])
        )
        value_target = torch.where(valid[:, index], candidate, values[:, index])
        corrected[index] = value_target
        corrected_next[index] = next_target
        accumulator = torch.where(valid[:, index], value_target, accumulator)

    value_target = torch.stack(corrected, dim=1)
    next_value_target = torch.stack(corrected_next, dim=1)
    pg_advantage = clipped_pg_rho * (
        rewards
        + gamma * continuation.to(rewards.dtype) * next_value_target
        - values
    )
    pg_advantage = pg_advantage * valid.to(rewards.dtype)
    ratio_diagnostics = importance_ratio_diagnostics(
        ratios, valid=valid, clip=rho_clip
    )
    # Contribution of the frozen outer value leaf to the root V-trace label.
    # Complete search trajectories reach that leaf only at their final valid
    # transition.  An early terminal has continuation zero, correctly making
    # the contribution vanish even though its padded suffix has no outer leaf.
    horizon = valid.squeeze(-1).sum(dim=1).clamp_min(1).to(torch.long)
    final = (horizon - 1).reshape(batch_size, 1, 1)
    step_index = torch.arange(time, device=rewards.device).reshape(1, time, 1)
    continuation_prefix = torch.cumprod(
        continuation.to(rewards.dtype), dim=1
    )
    c_before = torch.cat(
        (
            torch.ones_like(c[:, :1]),
            torch.cumprod(c[:, :-1], dim=1),
        ),
        dim=1,
    )
    leaf_coefficients = (
        torch.pow(gamma, step_index.to(rewards.dtype) + 1)
        * continuation_prefix
        * c_before
        * clipped_rho
    )
    leaf_contribution = (
        leaf_coefficients.gather(1, final)[:, 0]
        * next_values.gather(1, final)[:, 0]
    )
    return {
        "value_target": value_target,
        "pg_advantage": pg_advantage,
        "next_value_target": next_value_target,
        "td_error": td_error * valid.to(rewards.dtype),
        "rho": ratios,
        "clipped_rho": clipped_rho,
        "clipped_pg_rho": clipped_pg_rho,
        "trace_c": c,
        "leaf_contribution": leaf_contribution,
        "effective_return_length": _prefix_before(
            continuation, rewards.dtype
        ).mul(valid).sum(dim=1),
        **ratio_diagnostics,
    }


def vtrace_actor_loss(
    target_log_prob,
    pg_advantage,
    *,
    valid=None,
    entropy_log_prob=None,
    entropy_coefficient=0.0,
    return_diagnostics=False,
):
    """Likelihood-ratio V-trace actor loss plus separate entropy regularizer.

    ``pg_advantage`` is expected to include V-trace's clipped policy-gradient
    ratio.  The advantage is detached here so this helper cannot accidentally
    update the value network through the actor loss.
    """
    target_log_prob = _scalar_sequence(target_log_prob, "target_log_prob")
    pg_advantage = _matching_sequence(
        pg_advantage, target_log_prob, "pg_advantage"
    )
    valid = _mask_sequence(valid, target_log_prob, "valid", True)
    if entropy_log_prob is None:
        entropy_log_prob = target_log_prob
    else:
        entropy_log_prob = _matching_sequence(
            entropy_log_prob, target_log_prob, "entropy_log_prob"
        )
    policy_term = -target_log_prob * pg_advantage.detach()
    entropy_term = entropy_coefficient * entropy_log_prob
    count = valid.to(target_log_prob.dtype).sum().clamp_min(1)
    policy_loss = (policy_term * valid).sum() / count
    entropy_loss = (entropy_term * valid).sum() / count
    loss = policy_loss + entropy_loss
    if not return_diagnostics:
        return loss
    return loss, {
        "policy_gradient_loss": policy_loss,
        "entropy_loss": entropy_loss,
        "valid_count": count,
    }


# Plural aliases read naturally at vectorized call sites.
td0_targets = td0_target
n_step_targets = n_step_target
full_suffix_targets = full_suffix_target
lambda_return_targets = lambda_return_target
retrace_targets = retrace_target


__all__ = [
    "soft_value",
    "importance_sampling_ratios",
    "importance_ratio_diagnostics",
    "td0_target",
    "td0_targets",
    "n_step_target",
    "n_step_targets",
    "full_suffix_target",
    "full_suffix_targets",
    "resimulated_suffix_target",
    "lambda_return_target",
    "lambda_return_targets",
    "retrace_target",
    "retrace_targets",
    "vtrace_targets",
    "vtrace_actor_loss",
]
