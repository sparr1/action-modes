"""Dependency-free scientific identity for frozen pathwise evaluation."""

from collections.abc import Mapping
import math


PROTOCOL = "frozen_pathwise_v1"


def _finite_number(value, name, *, minimum=0, maximum=math.inf):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or not minimum < value <= maximum):
        raise ValueError(f"Frozen pathwise {name} must be finite and in ({minimum}, {maximum}].")
    return float(value)


def _log_std_settings(params, prefix, fallback):
    results = {}
    for key in ("mapping", "min", "max"):
        value = params.get(f"{prefix}log_std_{key}")
        results[key] = fallback[key] if value is None else value
    results["mapping"] = str(results["mapping"]).lower()
    if results["mapping"] not in {"direct_clamp", "tdmpc2_tanh"}:
        raise ValueError("Frozen pathwise requires a supported Gaussian log-std mapping.")
    for key in ("min", "max"):
        results[key] = _finite_number(results[key], f"log-std {key}", minimum=-math.inf)
    if results["min"] >= results["max"]:
        raise ValueError("Frozen pathwise log-std minimum must be less than maximum.")
    return results


def pathwise_settings(cfg, *, rollouts_per_update, detach_dynamics):
    """Resolve workload and scientific identity with no checkpoint tensor access."""
    params = cfg if isinstance(cfg, Mapping) else vars(cfg)
    rounds = params["inner_rounds"]
    horizon = params["inner_rollout_horizon"]
    actor_updates = params["inner_actor_updates_per_round"]
    for name, value in (
        ("rounds", rounds), ("horizon", horizon), ("actor_updates_per_round", actor_updates),
        ("rollouts_per_update", rollouts_per_update),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"Frozen pathwise {name} must be a positive integer.")
    if type(detach_dynamics) is not bool:
        raise ValueError("Frozen pathwise detach_dynamics must be boolean.")
    reduction = params.get("mppi_terminal_q_reduction", "mean_pair")
    if reduction not in {"mean_pair", "min_pair", "mean_all", "min_all"}:
        raise ValueError("Frozen pathwise requires an explicit scalar terminal Q reduction.")
    scale_tau = _finite_number(params.get("sac_actor_loss_scale_tau", 0.01), "Q scale tau", maximum=1)
    adam_eps = params.get("inner_actor_adam_eps")
    if adam_eps is None:
        adam_eps = params.get("inner_adam_eps", 1e-8)
    fixed = params.get("discount")
    # TDMPC2's configuration resolver turns an explicit discount into matching
    # min/max clamps before AMBI computes the episode-length-based value.
    discount_rule = {
        "denominator": _finite_number(params.get("discount_denom", 5), "discount denominator"),
        "min": _finite_number(params.get("discount_min", 0.95) if fixed is None else fixed,
                              "discount minimum", maximum=1),
        "max": _finite_number(params.get("discount_max", 0.995) if fixed is None else fixed,
                              "discount maximum", maximum=1),
    }
    if discount_rule["min"] > discount_rule["max"]:
        raise ValueError("Frozen pathwise discount minimum exceeds its maximum.")
    if discount_rule["min"] == discount_rule["max"]:
        discount_rule["effective"] = discount_rule["min"]
    prior_log_std = _log_std_settings(params, "", {"mapping": "direct_clamp", "min": -20, "max": 2})
    return {
        "rounds": rounds,
        "horizon": horizon,
        "actor_updates_per_round": actor_updates,
        "rollouts_per_update": rollouts_per_update,
        "detach_dynamics": detach_dynamics,
        "actor_optimizer_steps": rounds * actor_updates,
        "model_steps": rounds * actor_updates * rollouts_per_update * horizon,
        "actor_learning_rate": _finite_number(params.get("inner_actor_lr", 5e-5), "actor learning rate"),
        "actor_grad_clip_norm": _finite_number(params.get("inner_actor_grad_clip_norm", 20.0), "actor grad clip"),
        "actor_adam_eps": _finite_number(adam_eps, "actor Adam epsilon"),
        "actor_log_std": _log_std_settings(params, "inner_", prior_log_std),
        "terminal_prior_log_std": prior_log_std,
        "discount_rule": discount_rule,
        "terminal_q_reduction": reduction,
        "scale_tau": scale_tau,
        "scale_rule": "action_local_ema_p95_minus_p05_floor_1_before_loss",
        "reward_objective": "discounted_reward_with_sampled_frozen_prior_online_q_tail",
        "root_entropy": "squashed_post_tanh_at_depth_zero_only",
        "rng_protocol": "private_device_generator_per_update_noise_per_decision_q_pair",
    }
