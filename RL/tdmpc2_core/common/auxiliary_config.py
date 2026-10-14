"""Configuration for a separate reward-return learner, never split Q heads."""

import math
from numbers import Real

from .soft_world_model import normalize_log_std_mapping


AUX_ACTOR_KEYS = (
    "actor_lr", "actor_adam_eps", "adam_eps", "grad_clip_norm",
    "log_std_mapping", "log_std_min", "log_std_max",
    "outer_q_actor_reduction", "outer_q_target_reduction",
    "outer_actor_entropy_mode", "ent_coef", "ent_coef_lr", "target_entropy",
    "sac_actor_loss_scale_mode", "sac_actor_loss_scale_tau",
    "outer_behavior_policy_objective", "outer_behavior_policy_kl_schedule",
    "outer_behavior_policy_kl_coef", "outer_behavior_policy_kl_min_valid_count",
    "outer_behavior_policy_kl_ramp_updates", "outer_behavior_policy_kl_q_threshold",
    "outer_behavior_policy_kl_target", "outer_behavior_policy_kl_dual_init",
    "outer_behavior_policy_kl_dual_lr", "outer_behavior_policy_kl_dual_max",
)
SOURCE_CHOICES = {
    "inner_actor_source": {"sac", "return_actor"},
    "inner_critic_source": {"sac", "aux_return"},
    "inner_horizon_actor_source": {"sac", "return_actor"},
    "inner_horizon_critic_source": {"sac", "aux_return"},
}


def _choice(value, name, choices):
    if not isinstance(value, str) or value.lower() not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}.")
    return value.lower()


def _number(value, name, *, minimum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    value = float(value)
    if (not math.isfinite(value) or (positive and value <= 0)
            or (minimum is not None and value < minimum)):
        raise ValueError(f"Invalid {name}: {value!r}.")
    return value


def normalize_auxiliary_params(params):
    """Early structural checks; called before split and scheduling resolution."""
    params = dict(params)
    mode = _choice(params.get("aux_return_mode", "off"), "aux_return_mode",
                   {"off", "sac", "return_actor"})
    params["aux_return_mode"] = mode
    for key, choices in SOURCE_CHOICES.items():
        value = _choice(params.get(key, "sac"), key, choices)
        params[key] = value
        if value == "return_actor" and mode != "return_actor":
            raise ValueError(f"{key}='return_actor' requires aux_return_mode='return_actor'.")
        if value == "aux_return" and mode == "off":
            raise ValueError(f"{key}='aux_return' requires an auxiliary return critic.")
    if mode == "off":
        return params
    if str(params.get("critic_value_mode", "single")).lower() != "single":
        raise ValueError("aux_return_mode requires critic_value_mode='single'; split heads remain separate.")
    operator = str(params.get("inner_operator", "sac")).lower()
    if operator not in {"none", "sac", "mppi"}:
        raise ValueError("aux_return_mode supports inner_operator='none', 'sac', or 'mppi'.")
    if operator == "sac":
        params.setdefault("inner_finite_horizon", True)
        if params["inner_finite_horizon"] is not True:
            raise ValueError("aux_return_mode with inner SAC requires inner_finite_horizon=true.")
    params.setdefault("inner_rebase_persistent", False)
    if params.get("inner_entropy_enabled", True) is False:
        if params.get("inner_sac_critic_target", "reward_only") != "reward_only":
            raise ValueError("inner_entropy_enabled=false requires inner_sac_critic_target='reward_only'.")
        params["inner_sac_critic_target"] = "reward_only"
        params["inner_temperature_mode"] = "inherit_outer"
        params.pop("inner_temperature_updates_per_action", None)
    return params


def resolve_auxiliary_actor_config(cfg):
    """Resolve independent copies of all supported outer actor controls."""
    if cfg.aux_return_mode == "off":
        return
    values = {key: getattr(cfg, f"aux_return_{key}", None) for key in AUX_ACTOR_KEYS}
    for key in values:
        if values[key] is None:
            values[key] = getattr(cfg, key)
    for key in ("actor_lr", "actor_adam_eps", "adam_eps", "grad_clip_norm", "ent_coef_lr"):
        values[key] = _number(values[key], f"aux_return_{key}", positive=True)
    values["log_std_mapping"] = normalize_log_std_mapping(values["log_std_mapping"], "aux_return_log_std_mapping")
    for key in ("log_std_min", "log_std_max"):
        values[key] = _number(values[key], f"aux_return_{key}")
    if values["log_std_min"] >= values["log_std_max"]:
        raise ValueError("aux_return_log_std_min must be less than aux_return_log_std_max.")
    for key in ("outer_q_actor_reduction", "outer_q_target_reduction"):
        values[key] = _choice(values[key], f"aux_return_{key}", {"min_pair", "mean_pair", "min_all", "mean_all"})
    values["outer_actor_entropy_mode"] = _choice(values["outer_actor_entropy_mode"], "aux_return_outer_actor_entropy_mode", {"squashed", "tdmpc2_scaled"})
    values["sac_actor_loss_scale_mode"] = _choice(values["sac_actor_loss_scale_mode"], "aux_return_sac_actor_loss_scale_mode", {"none", "tdmpc2_percentile_range"})
    tau = _number(values["sac_actor_loss_scale_tau"], "aux_return_sac_actor_loss_scale_tau", positive=True)
    if tau > 1:
        raise ValueError("aux_return_sac_actor_loss_scale_tau must be in (0, 1].")
    values["sac_actor_loss_scale_tau"] = tau
    alpha = values["ent_coef"]
    if isinstance(alpha, str):
        alpha = alpha.lower()
        if alpha == "off":
            alpha = 0.0
        elif alpha == "auto":
            pass
        elif alpha.startswith("auto_"):
            try:
                initial = float(alpha[5:])
            except ValueError as exc:
                raise ValueError("Invalid aux_return_ent_coef initial value.") from exc
            _number(initial, "aux_return_ent_coef initial value", positive=True)
        else:
            raise ValueError("aux_return_ent_coef must be off, nonnegative, or auto[_initial].")
    else:
        alpha = _number(alpha, "aux_return_ent_coef", minimum=0)
    values["ent_coef"] = alpha
    target = values["target_entropy"]
    if isinstance(target, str) and target.lower() == "auto":
        target = "auto"
    else:
        target = _number(target, "aux_return_target_entropy")
    values["target_entropy"] = target
    if cfg.aux_return_mode == "return_actor" and isinstance(alpha, str):
        if values["outer_actor_entropy_mode"] == "tdmpc2_scaled" and target == "auto":
            raise ValueError("Automatic scaled auxiliary entropy requires numeric aux_return_target_entropy.")
        if values["sac_actor_loss_scale_mode"] != "none":
            raise ValueError("Auxiliary Q normalization requires a fixed aux_return_ent_coef.")
    objective = _choice(values["outer_behavior_policy_objective"], "aux_return_outer_behavior_policy_objective", {"reverse_kl", "action_space_cross_entropy"})
    schedule = _choice(values["outer_behavior_policy_kl_schedule"], "aux_return_outer_behavior_policy_kl_schedule", {"none", "smooth", "quantile_gate", "dual"})
    values["outer_behavior_policy_objective"] = objective
    values["outer_behavior_policy_kl_schedule"] = schedule
    if objective == "action_space_cross_entropy" and schedule == "dual":
        raise ValueError("Auxiliary action_space_cross_entropy does not support the dual schedule.")
    for key in ("outer_behavior_policy_kl_coef", "outer_behavior_policy_kl_target"):
        values[key] = _number(values[key], f"aux_return_{key}", minimum=0)
    for key in ("outer_behavior_policy_kl_q_threshold", "outer_behavior_policy_kl_dual_init", "outer_behavior_policy_kl_dual_lr", "outer_behavior_policy_kl_dual_max"):
        values[key] = _number(values[key], f"aux_return_{key}", positive=True)
    if values["outer_behavior_policy_kl_dual_init"] > values["outer_behavior_policy_kl_dual_max"]:
        raise ValueError("Auxiliary KL dual initial value exceeds its maximum.")
    for key in ("outer_behavior_policy_kl_ramp_updates", "outer_behavior_policy_kl_min_valid_count"):
        value = values[key]
        if key.endswith("min_valid_count") and value == "auto":
            value = int(cfg.batch_size)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"aux_return_{key} must be a positive integer.")
        values[key] = value
    if schedule in {"smooth", "quantile_gate"} and values["outer_behavior_policy_kl_coef"] <= 0:
        raise ValueError("Auxiliary fixed KL schedule requires a positive coefficient.")
    if cfg.aux_return_mode == "return_actor" and schedule != "none":
        if (cfg.inner_operator != "sac" or cfg.inner_execution_action != "policy_sample"
                or cfg.inner_execution_std_scale <= 0):
            raise ValueError("Auxiliary behavior regularization requires stochastic inner SAC execution.")
    cfg.aux_return_actor_cfg = values
    for key, value in values.items():
        setattr(cfg, f"aux_return_{key}", value)
    cfg.aux_return_critic_coef = _number(cfg.aux_return_critic_coef, "aux_return_critic_coef", minimum=0)
    cfg.aux_return_critic_lr = _number(cfg.critic_lr if cfg.aux_return_critic_lr is None else cfg.aux_return_critic_lr, "aux_return_critic_lr", positive=True)
    if not isinstance(cfg.aux_return_detach_representation, bool):
        raise ValueError("aux_return_detach_representation must be bool.")


def validate_auxiliary_config(cfg):
    if cfg.aux_return_mode == "off":
        return
    if cfg.inner_operator == "sac" and cfg.inner_schedule_mode != "canonical":
        raise ValueError("aux_return_mode requires the canonical inner SAC schedule.")
    requirements = {
        "inner_explorer_mode": "none", "inner_actor_initialization": "prior",
        "inner_critic_initialization": "prior", "inner_bootstrap_source": "inner_target",
        "inner_execution_policy_source": "primary", "inner_actor_writeback_coef": 0.,
        "inner_critic_writeback_coef": 0., "value_equivalence_loss_coef": 0.,
    }
    for component in ("actor", "critic", "temperature", "replay", "actor_optimizer", "critic_optimizer", "temperature_optimizer"):
        requirements[f"inner_{component}_scope"] = "action"
    for key, value in requirements.items():
        if getattr(cfg, key) != value:
            raise ValueError(f"aux_return_mode requires {key}={value!r}.")
    selected = cfg.aux_return_actor_cfg if cfg.inner_critic_source == "aux_return" else vars(cfg)
    if (cfg.inner_operator == "sac" and selected["sac_actor_loss_scale_mode"] != "none"
            and cfg.inner_temperature_mode == "auto" and cfg.inner_entropy_enabled):
        raise ValueError("Selected inner critic normalization requires fixed inner temperature.")
    if cfg.aux_return_mode == "return_actor":
        cfg.store_behavior_policy = cfg.store_behavior_policy or cfg.aux_return_actor_cfg["outer_behavior_policy_kl_schedule"] != "none"
