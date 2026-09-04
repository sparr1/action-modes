"""AMBI: TOLD priors with fresh per-root actor-critic adaptation.

Cloned inner SAC is the canonical operator. Other supported adaptation modes
and operators are auxiliary ablations or comparison methods.
"""

import copy
import hashlib
import math
import warnings
from collections.abc import Mapping

import numpy as np
import torch

from RL.TDMPC2 import TDMPC2Baseline, _normalize_horizon_params
from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.common.soft_world_model import normalize_log_std_mapping
from utils.utils import setup_logs


_Q_REDUCTIONS = {"min_pair", "mean_pair", "min_all", "mean_all"}
_CRITIC_TARGETS = {"entropy_augmented", "reward_only"}
_SAC_ACTOR_LOSS_SCALE_MODES = {"none", "tdmpc2_percentile_range"}
_BEHAVIOR_POLICY_OBJECTIVES = {"reverse_kl", "action_space_cross_entropy"}
_BEHAVIOR_POLICY_KL_SCHEDULES = {"none", "smooth", "quantile_gate", "dual"}
_VALUE_EVAL_PROTOCOLS = {"paper_deterministic", "stochastic_bellman"}
_ADAPTATION_MODES = {"frozen", "clone", "lora"}
_LIFECYCLE_SCOPES = {"action", "episode", "run"}
_SCOPE_RANK = {"action": 0, "episode": 1, "run": 2}
_INNER_EXPLORER_MODES = {
    "none",
    "adaptive_param_noise",
    "frozen_random",
    "shared_mixture",
    "separate_critics",
}
_INNER_MIXTURE_TARGET_ESTIMATORS = {"stratified", "weighted"}
_INNER_EXECUTION_POLICY_SOURCES = {
    "primary",
    "explorer",
    "mixture_sample",
    "outer_q_gate",
    "outer_soft_handoff",
}


_AMBI_DEFAULTS = {
    # Outer latent SAC. Q representation, ensemble reduction, and whether
    # entropy enters policy evaluation are independent controls.
    "mpc": False,
    "q_representation": "distributional",
    "q_num_bins": None,
    "q_vmin": None,
    "q_vmax": None,
    "q_pair_size": 2,
    "outer_q_target_reduction": "min_pair",
    "outer_q_actor_reduction": "min_pair",
    "inner_q_target_reduction": "min_pair",
    "inner_q_actor_reduction": "min_pair",
    "mppi_terminal_q_reduction": "mean_pair",
    "outer_critic_target": "entropy_augmented",
    "inner_sac_critic_target": "entropy_augmented",
    "sac_actor_loss_scale_mode": "none",
    "sac_actor_loss_scale_tau": 0.01,
    # Optional behavior-policy regularizer from the outer actor to the replayed
    # action-generating policy. ``none`` preserves the historical runtime and
    # replay schema; active schedules require stochastic inner SAC execution.
    "outer_behavior_policy_objective": "reverse_kl",
    "outer_behavior_policy_kl_schedule": "none",
    "outer_behavior_policy_kl_coef": 1.0,
    "outer_behavior_policy_kl_min_valid_count": "auto",
    "outer_behavior_policy_kl_ramp_updates": 10_000,
    "outer_behavior_policy_kl_q_threshold": 2.0,
    "outer_behavior_policy_kl_target": 0.1,
    "outer_behavior_policy_kl_dual_init": 0.1,
    "outer_behavior_policy_kl_dual_lr": 3e-4,
    "outer_behavior_policy_kl_dual_max": 10.0,
    "critic_coef": 1.0,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "adam_eps": 1e-8,
    "actor_adam_eps": None,
    "log_std_mapping": "direct_clamp",
    "log_std_min": -20,
    "log_std_max": 2,
    "ent_coef": "auto",
    "ent_coef_lr": 3e-4,
    "target_entropy": "auto",
    "tau": 0.005,
    "target_update_interval": 1,
    "compile_strict": False,

    # Optional trajectory-level intervention used by value-calibration
    # experiments. Eligible training episodes draw once between the canonical
    # AMBI inner behavior and the stochastic outer actor.
    "outer_policy_episode_probability": 0.0,

    # Canonical root-local training schedule. Each round collects N rollouts of
    # length H, appends them to the root-local replay, then runs G joint update
    # slots. ``auto`` keeps critic UTD at one by matching the transitions that
    # were actually generated in that round.
    "inner_operator": "sac",
    "inner_rounds": 4,
    "inner_rollouts_per_round": 64,
    "inner_rollout_horizon": 3,
    "inner_updates_per_round": "auto",
    # Optional canonical component schedule. Both values must be specified;
    # unlike the shared-G schedule, critic phases precede actor phases.
    "inner_critic_updates_per_round": None,
    "inner_actor_updates_per_round": None,

    # Optional action-local random explorer. The prior weight always denotes
    # the primary/prior-initialized policy's share of imagined rollouts.
    "inner_explorer_mode": "none",
    "inner_prior_rollout_weight": 0.5,
    "inner_mixture_target_estimator": "stratified",
    "inner_explorer_actor_updates_per_round": None,
    "inner_explorer_critic_updates_per_round": None,
    "inner_explorer_temperature_updates_per_round": None,

    # Action-local adaptive parameter noise. The actor count is deliberately
    # explicit whenever the mode is active so each configured rollout has an
    # unambiguous policy identity and equal allocation.
    "inner_param_noise_actor_count": None,
    "inner_param_noise_target_action_rms": 0.1,
    "inner_param_noise_sigma_init": 1e-3,
    "inner_param_noise_sigma_min": 1e-6,
    "inner_param_noise_sigma_max": 0.1,
    "inner_param_noise_calibration_directions": 8,
    "inner_param_noise_calibration_batch_size": 32,
    "inner_param_noise_calibration_max_probes": 8,

    # Deprecated total-budget controls. These remain accepted on an isolated
    # legacy schedule path for one release and are resolved to nominal totals
    # for canonical configurations.
    "inner_model_step_budget": None,
    "inner_critic_updates_per_action": None,
    "inner_actor_updates_per_action": None,
    "inner_temperature_updates_per_action": None,
    "inner_batch_size": 128,
    "inner_replay_capacity": None,
    "inner_replay_sampling": "with_replacement",

    # Independently adaptable inner components.
    "inner_actor_adaptation": "clone",
    "inner_critic_adaptation": "clone",
    "inner_critic_dropout_enabled": True,
    # ``None`` is resolved by operator: learned/root-local for SAC and disabled
    # for operators without an entropy objective.
    "inner_temperature_mode": None,
    "inner_temperature_initialization": "inherit_outer",
    "inner_temperature": 1.0,
    "inner_target_entropy": "inherit_outer",
    "inner_bootstrap_source": "inner_target",
    "inner_actor_lr": 5e-5,
    "inner_critic_lr": 5e-5,
    "inner_temperature_lr": 5e-5,
    "inner_adam_eps": 1e-8,
    "inner_critic_target_tau": 0.005,
    "inner_critic_target_update_interval": 1,
    "inner_actor_target_tau": None,
    "inner_actor_target_update_interval": None,
    "inner_actor_grad_clip_norm": 20.0,
    "inner_critic_grad_clip_norm": 20.0,
    "inner_temperature_grad_clip_norm": 20.0,
    "inner_termination_threshold": 0.5,
    "inner_outer_policy_kl_coef": 0.0,
    "inner_outer_action_l2_coef": 0.0,

    # Optional Reptile/Lookahead-style assimilation of the final action-local
    # SAC weights into the persistent control priors. Zero preserves canonical
    # fresh-per-root AMBI exactly; one performs a hard write-back.
    "inner_actor_writeback_coef": 0.0,
    "inner_critic_writeback_coef": 0.0,

    # Exploration during imagined collection is independent of the entropy
    # objective and of noise on the action returned to the real environment.
    "inner_behavior_action": "policy_sample",
    "inner_behavior_std_scale": 1.0,
    "inner_behavior_noise_std": 0.0,
    "inner_execution_action": "policy_sample",
    "inner_execution_std_scale": 1.0,
    "inner_execution_noise_std": 0.0,
    "inner_execution_policy_source": "primary",
    "inner_execution_handoff_samples": 8,
    "inner_log_std_mapping": None,
    "inner_log_std_min": None,
    "inner_log_std_max": None,

    # LoRA rank controls capacity; scale is the actual, rank-independent output
    # multiplier (legacy alpha/r is normalized to this value).
    "inner_actor_lora_rank": 8,
    "inner_actor_lora_scale": 1.0,
    "inner_actor_lora_dropout": 0.0,
    "inner_critic_lora_rank": 8,
    "inner_critic_lora_scale": 1.0,
    "inner_critic_lora_dropout": 0.0,

    # Inner state lifetime. Action-local remains the safe reference behavior.
    "inner_actor_scope": "action",
    "inner_critic_scope": "action",
    "inner_temperature_scope": "action",
    "inner_replay_scope": "action",
    "inner_actor_optimizer_scope": "action",
    "inner_critic_optimizer_scope": "action",
    "inner_temperature_optimizer_scope": "action",
    "inner_rebase_persistent": True,

    # TD3-specific controls.
    "inner_td3_target_noise_std": 0.2,
    "inner_td3_target_noise_clip": 0.5,

    # Auxiliary MPPI-comparator controls. Candidate count is derived from the
    # common model-step budget, including the one-time policy-prior trajectory
    # cost.
    "inner_mppi_iterations": 1,
    "inner_mppi_num_elites": 8,
    "inner_mppi_num_pi_trajs": 0,
    "inner_mppi_temperature": 0.5,
    "inner_mppi_min_std": 0.05,
    "inner_mppi_max_std": 2.0,
    "inner_mppi_warm_start_scope": "action",

    # Diagnostics-only paired model rollouts; excluded from the optimization
    # model-step budget. Zero preserves legacy runtime.
    "inner_diagnostic_rollouts": 0,
    "inner_diagnostics_every": 1,

    # Sparse, observational value-equivalence monitoring on outer replay
    # updates. Disabled by default so it cannot change the reference runtime.
    "value_equivalence_diagnostics": False,
    "value_equivalence_every_updates": 1000,
    "value_equivalence_mc_samples": 4,

    # Optional value-only Bellman-equivalence training for continuing tasks.
    # This is independent of the observational monitor above.
    "value_equivalence_loss_coef": 0.0,
    "value_equivalence_loss_mc_samples": 4,

    # Initial-state outer-Q calibration against real discounted rollouts.
    # Disabled by default because the full Monte Carlo probe is expensive.
    "eval_value": False,
    "eval_value_samples": 100,
    "eval_value_seed": 12345,
    "eval_value_protocols": ["paper_deterministic", "stochastic_bellman"],

    # Optional Figure-2-style paired controller evaluation.  The probe runs
    # on isolated environments and is disabled by default because a fresh
    # inner solve at every evaluation state is intentionally expensive.
    "eval_inner_comparison": False,
    "eval_inner_comparison_episodes": 5,
    "eval_inner_comparison_seed": 12345,
}


_V1_SCHEDULE_KEYS = {
    "inner_iterations",
    "inner_rollouts",
    "inner_horizon",
    "inner_updates_per_iteration",
}
_CANONICAL_SCHEDULE_KEYS = {
    "inner_rounds",
    "inner_rollouts_per_round",
    "inner_rollout_horizon",
    "inner_updates_per_round",
    "inner_critic_updates_per_round",
    "inner_actor_updates_per_round",
}
_COMPONENT_SCHEDULE_KEYS = {
    "inner_critic_updates_per_round",
    "inner_actor_updates_per_round",
}
_TOTAL_SCHEDULE_KEYS = {
    "inner_model_step_budget",
    "inner_critic_updates_per_action",
    "inner_actor_updates_per_action",
    "inner_temperature_updates_per_action",
}

_LEGACY_SCHEDULE_DEFAULTS = {
    "inner_rounds": 1,
    "inner_rollout_horizon": None,
    "inner_model_step_budget": None,
    "inner_critic_updates_per_action": 1,
    "inner_actor_updates_per_action": 1,
    "inner_temperature_updates_per_action": 0,
    "inner_batch_size": 256,
    "inner_actor_lr": 3e-4,
    "inner_critic_lr": 3e-4,
    "inner_temperature_lr": 3e-4,
    "inner_critic_target_tau": 1.0,
    "inner_temperature_mode": "inherit_outer",
    "inner_target_entropy": "auto",
}


def _legacy_warning(keys):
    joined = ", ".join(sorted(keys))
    warnings.warn(
        f"Deprecated AMBI configuration key(s): {joined}. Use the canonical inner-loop controls instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _strict_bool(value, key):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{key} must be a boolean.")
    return bool(value)


def _strict_positive_int(value, key):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{key} must be a positive integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer.")
    return value


def _strict_nonnegative_int(value, key):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{key} must be a non-negative integer.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{key} must be a non-negative integer.")
    return value


def _normalize_eval_value_protocols(value):
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError(
            "eval_value_protocols must be a non-empty list containing only "
            f"{sorted(_VALUE_EVAL_PROTOCOLS)}."
        )
    protocols = []
    for protocol in value:
        if not isinstance(protocol, str):
            raise ValueError(
                "eval_value_protocols entries must be strings from "
                f"{sorted(_VALUE_EVAL_PROTOCOLS)}."
            )
        protocol = protocol.lower()
        if protocol not in _VALUE_EVAL_PROTOCOLS:
            raise ValueError(
                "eval_value_protocols entries must be chosen from "
                f"{sorted(_VALUE_EVAL_PROTOCOLS)}, got {protocol!r}."
            )
        if protocol in protocols:
            raise ValueError(
                f"eval_value_protocols contains duplicate entry {protocol!r}."
            )
        protocols.append(protocol)
    if not protocols:
        raise ValueError("eval_value_protocols must not be empty.")
    return protocols


def _strict_nonnegative_float(value, key):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{key} must be a finite non-negative number.")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{key} must be a finite non-negative number.")
    return value


def _strict_probability(value, key):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{key} must be a finite number in [0, 1].")
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be a finite number in [0, 1].")
    return value


def _finite_float(value, key):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{key} must be a finite number, not a boolean.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite, got {value}.")
    return value


def _normalize_choice(value, key, choices):
    if not isinstance(value, str):
        raise ValueError(f"{key} must be one of {sorted(choices)}, got {value!r}.")
    value = value.lower()
    if value not in choices:
        raise ValueError(f"{key} must be one of {sorted(choices)}, got {value!r}.")
    return value


def _integral_weighted_count(total, weight, *, total_key, weight_key):
    weighted = int(total) * float(weight)
    rounded = round(weighted)
    if not math.isclose(weighted, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"{total_key} * {weight_key} must be integral, got "
            f"{total} * {weight} = {weighted}."
        )
    return int(rounded)


def _resolve_random_explorer_config(cfg):
    """Validate and resolve the action-local two-policy experiment contract."""

    cfg.inner_explorer_mode = _normalize_choice(
        cfg.inner_explorer_mode,
        "inner_explorer_mode",
        _INNER_EXPLORER_MODES,
    )
    cfg.inner_mixture_target_estimator = _normalize_choice(
        cfg.inner_mixture_target_estimator,
        "inner_mixture_target_estimator",
        _INNER_MIXTURE_TARGET_ESTIMATORS,
    )
    cfg.inner_execution_policy_source = _normalize_choice(
        cfg.inner_execution_policy_source,
        "inner_execution_policy_source",
        _INNER_EXECUTION_POLICY_SOURCES,
    )
    cfg.inner_execution_handoff_samples = _strict_positive_int(
        cfg.inner_execution_handoff_samples,
        "inner_execution_handoff_samples",
    )
    cfg.inner_prior_rollout_weight = _strict_probability(
        cfg.inner_prior_rollout_weight,
        "inner_prior_rollout_weight",
    )

    requested_param_noise_actor_count = cfg.inner_param_noise_actor_count
    if requested_param_noise_actor_count is not None:
        requested_param_noise_actor_count = _strict_positive_int(
            requested_param_noise_actor_count,
            "inner_param_noise_actor_count",
        )
    for key in (
        "inner_param_noise_target_action_rms",
        "inner_param_noise_sigma_init",
        "inner_param_noise_sigma_min",
        "inner_param_noise_sigma_max",
    ):
        setattr(cfg, key, _strict_nonnegative_float(getattr(cfg, key), key))
    if not 0.0 < cfg.inner_param_noise_target_action_rms < 2.0:
        raise ValueError(
            "inner_param_noise_target_action_rms must be strictly between "
            "0 and 2 in normalized post-tanh action units."
        )
    if not (
        0.0
        < cfg.inner_param_noise_sigma_min
        <= cfg.inner_param_noise_sigma_init
        <= cfg.inner_param_noise_sigma_max
    ):
        raise ValueError(
            "Adaptive parameter-noise scales must satisfy "
            "0 < inner_param_noise_sigma_min <= inner_param_noise_sigma_init "
            "<= inner_param_noise_sigma_max."
        )
    for key in (
        "inner_param_noise_calibration_directions",
        "inner_param_noise_calibration_batch_size",
        "inner_param_noise_calibration_max_probes",
    ):
        setattr(cfg, key, _strict_positive_int(getattr(cfg, key), key))

    requested_explorer_updates = {}
    for component in ("actor", "critic", "temperature"):
        key = f"inner_explorer_{component}_updates_per_round"
        value = getattr(cfg, key)
        if value is not None:
            value = _strict_nonnegative_int(value, key)
        requested_explorer_updates[component] = value

    rounds = int(cfg.inner_rounds)
    cfg.inner_primary_updates_per_round_is_auto = bool(
        cfg.inner_schedule_mode == "canonical"
        and not cfg.inner_component_update_schedule
        and cfg.inner_updates_per_round == "auto"
    )
    for component in ("actor", "critic", "temperature"):
        total = int(getattr(cfg, f"inner_{component}_updates_per_action"))
        setattr(cfg, f"inner_primary_{component}_updates_per_action", total)
        if rounds > 0:
            quotient, remainder = divmod(total, rounds)
            per_round = quotient if remainder == 0 else total / rounds
        else:
            per_round = 0
        setattr(cfg, f"inner_primary_{component}_updates_per_round", per_round)

    mode = cfg.inner_explorer_mode
    active = mode != "none"
    cfg.inner_explorer_active = active
    cfg.inner_param_noise_active = mode == "adaptive_param_noise"
    cfg.inner_explorer_trainable = mode in {"shared_mixture", "separate_critics"}
    cfg.inner_explorer_has_separate_critic = mode == "separate_critics"

    if mode == "adaptive_param_noise":
        if requested_param_noise_actor_count is None:
            raise ValueError(
                "inner_explorer_mode='adaptive_param_noise' requires an explicit "
                "positive inner_param_noise_actor_count."
            )
        if cfg.inner_behavior_action != "policy_sample":
            raise ValueError(
                "inner_explorer_mode='adaptive_param_noise' requires "
                "inner_behavior_action='policy_sample'."
            )
        if cfg.inner_behavior_std_scale <= 0.0:
            raise ValueError(
                "inner_explorer_mode='adaptive_param_noise' requires a positive "
                "inner_behavior_std_scale so multiple rollouts assigned to one "
                "perturbed actor remain stochastic."
            )
        if cfg.inner_execution_policy_source != "primary":
            raise ValueError(
                "inner_explorer_mode='adaptive_param_noise' requires "
                "inner_execution_policy_source='primary'."
            )
    elif requested_param_noise_actor_count is not None:
        raise ValueError(
            "inner_param_noise_actor_count is only valid when "
            "inner_explorer_mode='adaptive_param_noise'."
        )

    if not active:
        nonzero = {
            component: value
            for component, value in requested_explorer_updates.items()
            if value not in {None, 0}
        }
        if nonzero:
            raise ValueError(
                "inner_explorer_mode='none' cannot use explorer optimizer "
                f"updates: {nonzero}."
            )
        if cfg.inner_execution_policy_source != "primary":
            raise ValueError(
                "inner_execution_policy_source must be 'primary' when "
                "inner_explorer_mode='none'."
            )
    else:
        if cfg.inner_operator != "sac":
            raise ValueError("An active inner explorer requires inner_operator='sac'.")
        if cfg.inner_schedule_mode != "canonical":
            raise ValueError(
                "An active inner explorer requires the canonical per-round schedule."
            )
        if cfg.inner_rounds <= 0 or cfg.inner_rollouts_per_round <= 0:
            raise ValueError(
                "An active inner explorer requires positive inner_rounds and "
                "inner_rollouts_per_round."
            )
        for component in ("actor", "critic"):
            if getattr(cfg, f"inner_{component}_adaptation") != "clone":
                raise ValueError(
                    "An active inner explorer requires clone adaptation; "
                    f"inner_{component}_adaptation must be 'clone'."
                )
        action_scopes = (
            "inner_actor_scope",
            "inner_critic_scope",
            "inner_temperature_scope",
            "inner_replay_scope",
            "inner_actor_optimizer_scope",
            "inner_critic_optimizer_scope",
            "inner_temperature_optimizer_scope",
        )
        for key in action_scopes:
            if getattr(cfg, key) != "action":
                raise ValueError(
                    "An active inner explorer requires action-local state; "
                    f"{key} must be 'action'."
                )
        if cfg.inner_bootstrap_source != "inner_target":
            raise ValueError(
                "An active inner explorer requires "
                "inner_bootstrap_source='inner_target'."
            )
        if not 0.0 < cfg.inner_prior_rollout_weight < 1.0:
            raise ValueError(
                "An active inner explorer requires "
                "inner_prior_rollout_weight strictly between 0 and 1."
            )
        if (
            cfg.inner_actor_writeback_coef != 0.0
            or cfg.inner_critic_writeback_coef != 0.0
        ):
            raise ValueError(
                "An active inner explorer does not support inner actor or critic "
                "writeback."
            )
        if (
            cfg.inner_outer_policy_kl_coef != 0.0
            or cfg.inner_outer_action_l2_coef != 0.0
        ):
            raise ValueError(
                "An active inner explorer requires zero inner outer-policy "
                "regularization coefficients."
            )
        if (
            mode == "shared_mixture"
            and cfg.inner_sac_critic_target != "entropy_augmented"
        ):
            raise ValueError(
                "inner_explorer_mode='shared_mixture' requires "
                "inner_sac_critic_target='entropy_augmented'."
            )
        if (
            cfg.inner_execution_policy_source == "outer_soft_handoff"
            and cfg.outer_critic_target != "entropy_augmented"
        ):
            raise ValueError(
                "inner_execution_policy_source='outer_soft_handoff' requires "
                "outer_critic_target='entropy_augmented'."
            )
        if (
            cfg.outer_behavior_policy_kl_schedule != "none"
            and cfg.inner_execution_policy_source
            in {"mixture_sample", "outer_soft_handoff"}
        ):
            raise ValueError(
                "An active outer behavior-policy regularizer is incompatible "
                "with mixture_sample or outer_soft_handoff execution."
            )

    if active:
        primary_rollouts = _integral_weighted_count(
            cfg.inner_rollouts_per_round,
            cfg.inner_prior_rollout_weight,
            total_key="inner_rollouts_per_round",
            weight_key="inner_prior_rollout_weight",
        )
        explorer_rollouts = cfg.inner_rollouts_per_round - primary_rollouts
        if primary_rollouts <= 0 or explorer_rollouts <= 0:
            raise ValueError(
                "An active inner explorer requires non-empty primary and "
                "explorer rollout populations after exact allocation, got "
                f"primary={primary_rollouts}, explorer={explorer_rollouts}."
            )
    else:
        primary_rollouts = cfg.inner_rollouts_per_round
        explorer_rollouts = 0
    cfg.inner_primary_rollouts_per_round = int(primary_rollouts)
    cfg.inner_explorer_rollouts_per_round = int(explorer_rollouts)
    total_rollouts = (
        cfg.inner_primary_rollouts_per_round
        + cfg.inner_explorer_rollouts_per_round
    )
    cfg.inner_primary_rollout_fraction = (
        cfg.inner_primary_rollouts_per_round / total_rollouts
        if total_rollouts > 0
        else 0.0
    )
    cfg.inner_explorer_rollout_fraction = (
        cfg.inner_explorer_rollouts_per_round / total_rollouts
        if total_rollouts > 0
        else 0.0
    )
    if mode == "adaptive_param_noise":
        actor_count = int(requested_param_noise_actor_count)
        explorer_rollouts = cfg.inner_explorer_rollouts_per_round
        if actor_count > explorer_rollouts:
            raise ValueError(
                "inner_param_noise_actor_count cannot exceed the resolved "
                "inner_explorer_rollouts_per_round: "
                f"{actor_count} > {explorer_rollouts}."
            )
        if explorer_rollouts % actor_count != 0:
            raise ValueError(
                "inner_explorer_rollouts_per_round must be exactly divisible by "
                "inner_param_noise_actor_count, got "
                f"{explorer_rollouts} % {actor_count}."
            )
        cfg.inner_param_noise_actor_count = actor_count
        cfg.inner_param_noise_rollouts_per_actor = explorer_rollouts // actor_count
    else:
        cfg.inner_param_noise_actor_count = None
        cfg.inner_param_noise_rollouts_per_actor = 0

    cfg.inner_primary_target_rows_per_batch = None
    cfg.inner_explorer_target_rows_per_batch = None
    if (
        mode == "shared_mixture"
        and cfg.inner_mixture_target_estimator == "stratified"
    ):
        primary_rows = _integral_weighted_count(
            cfg.inner_batch_size,
            cfg.inner_prior_rollout_weight,
            total_key="inner_batch_size",
            weight_key="inner_prior_rollout_weight",
        )
        cfg.inner_primary_target_rows_per_batch = primary_rows
        cfg.inner_explorer_target_rows_per_batch = cfg.inner_batch_size - primary_rows

    primary_updates = {
        component: getattr(cfg, f"inner_primary_{component}_updates_per_round")
        for component in ("actor", "critic", "temperature")
    }
    if active and any(
        not isinstance(value, int) for value in primary_updates.values()
    ):
        raise ValueError(
            "An active inner explorer requires integral primary optimizer "
            "update counts in every round."
        )

    if mode in {"none", "adaptive_param_noise", "frozen_random"}:
        nonzero = {
            component: value
            for component, value in requested_explorer_updates.items()
            if value not in {None, 0}
        }
        if nonzero:
            raise ValueError(
                f"inner_explorer_mode={mode!r} has no explorer optimizers: {nonzero}."
            )
        explorer_updates = {component: 0 for component in primary_updates}
        explorer_inherits = {component: False for component in primary_updates}
    elif mode == "shared_mixture":
        actor_updates = requested_explorer_updates["actor"]
        if (
            cfg.inner_primary_updates_per_round_is_auto
            and actor_updates is not None
        ):
            raise ValueError(
                "inner_explorer_mode='shared_mixture' with "
                "inner_updates_per_round='auto' requires "
                "inner_explorer_actor_updates_per_round=null so both actors "
                "follow the realized transition count."
            )
        actor_updates = (
            primary_updates["actor"] if actor_updates is None else actor_updates
        )
        if actor_updates != primary_updates["actor"]:
            raise ValueError(
                "inner_explorer_mode='shared_mixture' requires equal primary "
                "and explorer actor update counts."
            )
        for component in ("critic", "temperature"):
            if requested_explorer_updates[component] not in {None, 0}:
                raise ValueError(
                    "inner_explorer_mode='shared_mixture' uses one shared "
                    f"{component}; inner_explorer_{component}_updates_per_round "
                    "must be zero or null."
                )
        explorer_updates = {
            "actor": actor_updates,
            "critic": 0,
            "temperature": 0,
        }
        explorer_inherits = {
            "actor": requested_explorer_updates["actor"] is None,
            "critic": False,
            "temperature": False,
        }
    else:
        explorer_updates = {
            component: (
                primary_updates[component]
                if requested_explorer_updates[component] is None
                else requested_explorer_updates[component]
            )
            for component in primary_updates
        }
        explorer_inherits = {
            component: requested_explorer_updates[component] is None
            for component in primary_updates
        }

    if (
        explorer_updates["temperature"] > 0
        and cfg.inner_temperature_mode != "auto"
    ):
        raise ValueError(
            "Positive inner_explorer_temperature_updates_per_round requires "
            "inner_temperature_mode='auto'."
        )

    for component, per_round in explorer_updates.items():
        setattr(
            cfg,
            f"inner_explorer_{component}_updates_inherit_primary",
            bool(explorer_inherits[component]),
        )
        setattr(
            cfg,
            f"inner_explorer_{component}_updates_per_round",
            int(per_round),
        )
        setattr(
            cfg,
            f"inner_explorer_{component}_updates_per_action",
            int(per_round) * rounds,
        )
    cfg.inner_primary_optimizer_steps_per_action = sum(
        int(getattr(cfg, f"inner_{component}_updates_per_action"))
        for component in ("actor", "critic", "temperature")
    )
    cfg.inner_explorer_optimizer_steps_per_action = sum(
        getattr(cfg, f"inner_explorer_{component}_updates_per_action")
        for component in ("actor", "critic", "temperature")
    )
    cfg.inner_total_optimizer_steps_per_action = (
        cfg.inner_primary_optimizer_steps_per_action
        + cfg.inner_explorer_optimizer_steps_per_action
    )

    # Resolve the actual active-population slot contract. Ordinary canonical
    # scheduling shares one minibatch across every component enabled in a slot;
    # the explicit component schedule has a critic phase followed by an
    # actor/temperature phase. R may have a larger explicit dose than P in the
    # separate-critic experiment, so the P-only estimate is insufficient.
    if active:
        if cfg.inner_component_update_schedule:
            critic_slots = max(
                int(primary_updates["critic"]),
                int(explorer_updates["critic"]),
            )
            actor_slots = max(
                int(primary_updates["actor"]),
                int(primary_updates["temperature"]),
                int(explorer_updates["actor"]),
                int(explorer_updates["temperature"]),
            )
            slots_per_round = critic_slots + actor_slots
        else:
            slots_per_round = max(
                *(int(value) for value in primary_updates.values()),
                *(int(value) for value in explorer_updates.values()),
            )
        cfg.inner_nominal_updates_per_round = int(slots_per_round)
        cfg.inner_expected_update_slots = int(slots_per_round) * rounds

    return cfg


def _outer_policy_hash(seed, episode_start_step, namespace):
    """Return a stable 64-bit episode-local value without touching an RNG."""
    payload = f"{int(seed)}:{int(episode_start_step)}:{namespace}".encode("utf-8")
    digest = hashlib.blake2b(
        payload,
        digest_size=8,
        person=b"AMBI-outer-v1",
    ).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _normalize_legacy_params(params):
    """Normalize schedule aliases and identify canonical versus total scheduling."""
    params = copy.deepcopy(params)

    requested_operator = str(
        params.get("inner_operator", _AMBI_DEFAULTS["inner_operator"])
    ).lower()
    component_schedule = sorted(set(params) & _COMPONENT_SCHEDULE_KEYS)
    if component_schedule and len(component_schedule) != len(
        _COMPONENT_SCHEDULE_KEYS
    ):
        missing = sorted(_COMPONENT_SCHEDULE_KEYS - set(component_schedule))
        raise ValueError(
            "inner_critic_updates_per_round and inner_actor_updates_per_round "
            f"must be specified together; missing={missing}."
        )
    if component_schedule and requested_operator not in {"sac", "td3"}:
        raise ValueError(
            "inner_critic_updates_per_round and inner_actor_updates_per_round "
            "are only valid for inner_operator='sac' or 'td3'."
        )
    v1_schedule = sorted(set(params) & _V1_SCHEDULE_KEYS)
    if v1_schedule:
        conflicts = sorted(
            set(params)
            & (_CANONICAL_SCHEDULE_KEYS | _TOTAL_SCHEDULE_KEYS | {"inner_mppi_iterations"})
        )
        if conflicts:
            raise ValueError(
                "Cannot mix legacy and canonical compute keys: "
                f"legacy={v1_schedule}, canonical={conflicts}."
            )
        _legacy_warning(v1_schedule)
        rounds = int(params.pop("inner_iterations", 1))
        rollouts = int(params.pop("inner_rollouts", 32))
        rollout_horizon = params.pop("inner_horizon", None)
        if rollout_horizon is None:
            rollout_horizon = 3
        updates_per_round = params.pop("inner_updates_per_iteration", 1)
        if requested_operator == "mppi":
            if int(updates_per_round) not in {0, 1}:
                raise ValueError(
                    "Legacy MPPI inner_updates_per_iteration is not an MPPI control."
                )
            params.update(
                inner_mppi_iterations=rounds,
                inner_rollout_horizon=rollout_horizon,
                inner_model_step_budget=rounds * rollouts * int(rollout_horizon),
            )
        else:
            params.update(
                inner_rounds=rounds,
                inner_rollouts_per_round=rollouts,
                inner_rollout_horizon=rollout_horizon,
                inner_updates_per_round=int(updates_per_round),
            )
            # The historical alias did not learn a local temperature.
            params.setdefault("inner_temperature_mode", "inherit_outer")

    if requested_operator in {"sac", "td3"}:
        if component_schedule and "inner_updates_per_round" in params:
            raise ValueError(
                "Cannot combine shared inner_updates_per_round with component "
                "inner_critic_updates_per_round/inner_actor_updates_per_round."
            )
        new_unique = sorted(
            set(params)
            & (
                {"inner_rollouts_per_round", "inner_updates_per_round"}
                | _COMPONENT_SCHEDULE_KEYS
            )
        )
        totals = sorted(set(params) & _TOTAL_SCHEDULE_KEYS)
        if new_unique and totals:
            raise ValueError(
                "Cannot mix canonical J/N/H/G controls with deprecated total-budget "
                f"controls: canonical={new_unique}, deprecated={totals}."
            )
        schedule_mode = "legacy" if totals else "canonical"
        if totals:
            _legacy_warning(totals)
    elif requested_operator == "mppi":
        invalid = sorted(
            set(params)
            & (
                {"inner_rollouts_per_round", "inner_updates_per_round"}
                | _COMPONENT_SCHEDULE_KEYS
            )
        )
        if invalid:
            raise ValueError(f"MPPI does not use SAC/TD3 schedule controls: {invalid}.")
        if "inner_rounds" in params:
            if "inner_mppi_iterations" in params:
                raise ValueError(
                    "Cannot specify both inner_rounds and inner_mppi_iterations for MPPI."
                )
            _legacy_warning(["inner_rounds"])
            params["inner_mppi_iterations"] = params.pop("inner_rounds")
        schedule_mode = "legacy" if v1_schedule else "canonical"
    else:
        schedule_mode = "canonical"

    # Preserve the old learned-temperature initialization only for an explicit
    # pre-migration ``auto`` request. New/default SAC configurations inherit the
    # current outer alpha at every real root.
    if (
        (schedule_mode == "legacy" or bool(v1_schedule))
        and str(params.get("inner_temperature_mode", "")).lower() == "auto"
        and "inner_temperature_initialization" not in params
    ):
        _legacy_warning(["inner_temperature_mode=auto without initialization"])
        params["inner_temperature_initialization"] = "fixed"

    direct_groups = (
        ("inner_buffer_size", ("inner_replay_capacity",), lambda value, _: value),
        ("inner_adaptation", ("inner_actor_adaptation", "inner_critic_adaptation"), lambda value, _: value),
        ("inner_tau", ("inner_critic_target_tau",), lambda value, _: value),
        (
            "inner_target_update_interval",
            ("inner_critic_target_update_interval",),
            lambda value, _: value,
        ),
        (
            "inner_grad_clip_norm",
            ("inner_actor_grad_clip_norm", "inner_critic_grad_clip_norm"),
            lambda value, _: value,
        ),
        ("lora_rank", ("inner_actor_lora_rank", "inner_critic_lora_rank"), lambda value, _: value),
        (
            "lora_dropout",
            ("inner_actor_lora_dropout", "inner_critic_lora_dropout"),
            lambda value, _: value,
        ),
    )
    for legacy, canonical, convert in direct_groups:
        if legacy not in params:
            continue
        conflicts = [key for key in canonical if key in params]
        if conflicts:
            raise ValueError(
                f"Cannot mix legacy {legacy!r} with canonical key(s) {conflicts}."
            )
        _legacy_warning([legacy])
        value = params.pop(legacy)
        for key in canonical:
            params[key] = convert(value, params)

    if "lora_alpha" in params:
        conflicts = [
            key for key in ("inner_actor_lora_scale", "inner_critic_lora_scale")
            if key in params
        ]
        if conflicts:
            raise ValueError(
                f"Cannot mix legacy 'lora_alpha' with canonical key(s) {conflicts}."
            )
        _legacy_warning(["lora_alpha"])
        alpha = float(params.pop("lora_alpha"))
        actor_rank = int(params.get("inner_actor_lora_rank", 8))
        critic_rank = int(params.get("inner_critic_lora_rank", 8))
        if actor_rank <= 0 or critic_rank <= 0:
            raise ValueError("LoRA rank must be positive when converting legacy lora_alpha.")
        params["inner_actor_lora_scale"] = alpha / actor_rank
        params["inner_critic_lora_scale"] = alpha / critic_rank

    if "allow_long_inner_horizon" in params:
        _legacy_warning(["allow_long_inner_horizon"])
        # Long horizons now produce an explicit model-bias warning instead of
        # being gated by an unsafe-override switch.
        params.pop("allow_long_inner_horizon")

    return params, schedule_mode


class AMBITDMPC2(TDMPC2Baseline):
    """TOLD learning with canonical cloned inner SAC and optional ablations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending_behavior_policy = None
        self._value_calibration_evaluator = None
        self._paired_controller_evaluator = None
        self._outer_policy_episode_eligible = False
        self._outer_policy_episode_selected = False
        self._outer_policy_action_generator = None
        self._inner_steps_total = 0
        self._inner_updates_total = 0
        self._wandb_inner_seconds = 0.0
        self._wandb_inner_actions = 0
        self._wandb_inner_steps = 0
        self._wandb_outer_policy_seconds = 0.0
        self._wandb_outer_policy_actions = 0

    @property
    def _stores_behavior_policy(self):
        return bool(getattr(self.cfg, "store_behavior_policy", False))

    def _allocate_episode_staging(self, capacity):
        staging = super()._allocate_episode_staging(capacity)
        if not self._stores_behavior_policy:
            return staging
        tensor_kwargs = {
            "device": "cpu",
            "pin_memory": self._pin_episode_staging,
        }
        staging["behavior_pre_tanh_mean"] = torch.empty(
            (capacity, self.cfg.action_dim),
            dtype=torch.float32,
            **tensor_kwargs,
        )
        staging["behavior_log_std"] = torch.empty(
            (capacity, self.cfg.action_dim),
            dtype=torch.float32,
            **tensor_kwargs,
        )
        staging["behavior_policy_valid"] = torch.empty(
            (capacity,),
            dtype=torch.bool,
            **tensor_kwargs,
        )
        return staging

    def _ensure_episode_staging_capacity(self, required):
        if not self._stores_behavior_policy:
            return super()._ensure_episode_staging_capacity(required)
        if required <= len(self._episode_staging):
            return
        old = self._episode_staging
        replacement = self._allocate_episode_staging(
            max(required, 2 * len(old))
        )
        for key in old.keys():
            replacement[key][: len(old)].copy_(old[key])
        self._episode_staging = replacement

    def _start_episode_staging(self, obs_t):
        rows = super()._start_episode_staging(obs_t)
        self._pending_behavior_policy = None
        if self._stores_behavior_policy:
            self._episode_staging["behavior_pre_tanh_mean"][0].zero_()
            self._episode_staging["behavior_log_std"][0].zero_()
            self._episode_staging["behavior_policy_valid"][0] = False
        return rows

    def _random_action_norm(self):
        self._pending_behavior_policy = None
        return super()._random_action_norm()

    def _stage_transition(self, row, obs_t, action, reward, terminated):
        super()._stage_transition(row, obs_t, action, reward, terminated)
        if not self._stores_behavior_policy:
            return
        behavior_policy = self._pending_behavior_policy
        self._pending_behavior_policy = None
        if behavior_policy is None:
            self._episode_staging["behavior_pre_tanh_mean"][row].zero_()
            self._episode_staging["behavior_log_std"][row].zero_()
            self._episode_staging["behavior_policy_valid"][row] = False
            return
        mean = torch.as_tensor(
            behavior_policy["pre_tanh_mean"], dtype=torch.float32
        ).reshape(self.cfg.action_dim)
        log_std = torch.as_tensor(
            behavior_policy["log_std"], dtype=torch.float32
        ).reshape(self.cfg.action_dim)
        if not bool(torch.isfinite(mean).all()) or not bool(
            torch.isfinite(log_std).all()
        ):
            raise ValueError(
                "Captured behavior-policy parameters must be finite."
            )
        self._episode_staging["behavior_pre_tanh_mean"][row].copy_(mean)
        self._episode_staging["behavior_log_std"][row].copy_(log_std)
        self._episode_staging["behavior_policy_valid"][row] = True

    def _to_td(self, obs, action=None, reward=None, terminated=None):
        td = super()._to_td(obs, action, reward, terminated)
        if self._stores_behavior_policy:
            td["behavior_pre_tanh_mean"] = torch.zeros(
                (1, self.cfg.action_dim), dtype=torch.float32
            )
            td["behavior_log_std"] = torch.zeros(
                (1, self.cfg.action_dim), dtype=torch.float32
            )
            td["behavior_policy_valid"] = torch.zeros((1,), dtype=torch.bool)
        return td

    def _build_cfg(self, params):
        # Resolve the outer legacy alias before translating AMBI's separate
        # legacy inner-loop aliases. This rejects only combinations the caller
        # actually supplied while allowing an all-legacy configuration to make
        # its one-release migration cleanly.
        params = _normalize_horizon_params(params, resolve_defaults=False)
        component_update_schedule = bool(set(params) & _COMPONENT_SCHEDULE_KEYS)
        params, schedule_mode = _normalize_legacy_params(params)
        params = _normalize_horizon_params(params)
        explicit_num_q = params.get("num_q", None)
        requested_operator = str(
            params.get("inner_operator", _AMBI_DEFAULTS["inner_operator"])
        ).lower()
        if requested_operator not in {"none", "sac", "td3", "mppi"}:
            raise ValueError(
                "inner_operator must be one of 'none', 'sac', 'td3', or 'mppi'."
            )
        if requested_operator in {"none", "mppi"}:
            forbidden_updates = {
                key: int(params[key])
                for key in (
                    "inner_critic_updates_per_action",
                    "inner_actor_updates_per_action",
                    "inner_temperature_updates_per_action",
                )
                if key in params and int(params[key]) != 0
            }
            if forbidden_updates:
                raise ValueError(
                    f"inner_operator={requested_operator!r} does not use gradient updates: "
                    f"{forbidden_updates}."
                )
        if requested_operator == "none":
            conflicting_compute = {
                key: params[key]
                for key in (
                    "inner_model_step_budget",
                    "inner_rounds",
                    "inner_rollouts_per_round",
                    "inner_updates_per_round",
                    "inner_mppi_iterations",
                )
                if key in params
                and params[key] not in {None, 0, "0"}
            }
            if conflicting_compute:
                raise ValueError(
                    "inner_operator='none' performs no imagined work; remove or zero "
                    f"these compute controls: {conflicting_compute}."
                )

        if bool(params.get("mpc", False)):
            raise ValueError(
                "AMBI keeps TD-MPC2's outer mpc flag disabled; select inner_operator='mppi' "
                "only for the matched TD-MPC/MPPI comparison ablation."
            )

        merged = dict(_AMBI_DEFAULTS)
        if schedule_mode == "legacy" and requested_operator in {"sac", "td3"}:
            merged.update(_LEGACY_SCHEDULE_DEFAULTS)
        merged.update(params)
        if component_update_schedule:
            # The component schedule has no meaningful shared-G value.
            merged["inner_updates_per_round"] = None
        merged["outer_policy_episode_probability"] = _strict_probability(
            merged["outer_policy_episode_probability"],
            "outer_policy_episode_probability",
        )
        for key in (
            "inner_actor_writeback_coef",
            "inner_critic_writeback_coef",
        ):
            merged[key] = _strict_probability(merged[key], key)
        if merged["outer_policy_episode_probability"] > 0.0:
            requested_obs = merged.get("obs")
            if requested_obs is not None and str(requested_obs).lower() != "state":
                raise ValueError(
                    "A positive outer_policy_episode_probability currently "
                    "requires state observations."
                )
        if merged["inner_temperature_mode"] is None:
            merged["inner_temperature_mode"] = (
                "auto" if requested_operator == "sac" else "inherit_outer"
            )
        if requested_operator == "none":
            merged.update(
                inner_model_step_budget=0,
                inner_rounds=0,
                inner_rollouts_per_round=0,
                inner_updates_per_round=0,
                inner_mppi_iterations=0,
                inner_critic_updates_per_action=0,
                inner_actor_updates_per_action=0,
                inner_temperature_updates_per_action=0,
                inner_temperature_mode="inherit_outer",
            )
        elif requested_operator == "mppi":
            merged.update(
                inner_rounds=0,
                inner_rollouts_per_round=0,
                inner_updates_per_round=0,
                inner_critic_updates_per_action=0,
                inner_actor_updates_per_action=0,
                inner_temperature_updates_per_action=0,
                inner_temperature_mode="inherit_outer",
            )
        cfg = super()._build_cfg(merged)
        cfg.inner_schedule_mode = schedule_mode
        cfg.inner_component_update_schedule = component_update_schedule
        cfg.outer_policy_episode_probability = float(
            cfg.outer_policy_episode_probability
        )
        if cfg.outer_policy_episode_probability > 0.0:
            if cfg.obs != "state":
                raise ValueError(
                    "A positive outer_policy_episode_probability currently "
                    "requires state observations."
                )
            if str(cfg.inner_operator).lower() != "sac":
                raise ValueError(
                    "A positive outer_policy_episode_probability requires "
                    "inner_operator='sac'."
                )

        # ``model_size`` expands architecture defaults inside TD-MPC2. Restore
        # an explicit ensemble size afterwards; scalar SAC remains twin-Q while
        # distributional mode inherits the model-size preset when omitted.
        cfg.q_representation = str(cfg.q_representation).lower()
        if cfg.q_representation not in {"scalar", "distributional"}:
            raise ValueError("q_representation must be 'scalar' or 'distributional'.")
        if explicit_num_q is not None:
            cfg.num_q = int(explicit_num_q)
        elif cfg.q_representation == "scalar":
            cfg.num_q = 2
        else:
            cfg.num_q = int(cfg.num_q)
        if cfg.q_representation == "scalar" and cfg.num_q != 2:
            raise ValueError("Scalar Q representation requires num_q=2.")
        if cfg.q_representation == "distributional" and cfg.num_q < 2:
            raise ValueError("Distributional Q representation requires num_q>=2.")
        cfg.mpc = False

        cfg.q_num_bins = int(cfg.num_bins if cfg.q_num_bins is None else cfg.q_num_bins)
        cfg.q_vmin = _finite_float(
            cfg.vmin if cfg.q_vmin is None else cfg.q_vmin, "q_vmin"
        )
        cfg.q_vmax = _finite_float(
            cfg.vmax if cfg.q_vmax is None else cfg.q_vmax, "q_vmax"
        )
        if cfg.q_num_bins < 2:
            raise ValueError("q_num_bins must be at least 2.")
        if not cfg.q_vmax > cfg.q_vmin:
            raise ValueError("q_vmax must be greater than q_vmin.")
        cfg.q_bin_size = (cfg.q_vmax - cfg.q_vmin) / (cfg.q_num_bins - 1)

        cfg.q_pair_size = int(cfg.q_pair_size)
        if not 1 <= cfg.q_pair_size <= cfg.num_q:
            raise ValueError(f"q_pair_size must be in [1, num_q={cfg.num_q}].")
        for key in (
            "outer_q_target_reduction",
            "outer_q_actor_reduction",
            "inner_q_target_reduction",
            "inner_q_actor_reduction",
            "mppi_terminal_q_reduction",
        ):
            value = str(getattr(cfg, key)).lower()
            if value not in _Q_REDUCTIONS:
                raise ValueError(f"{key} must be one of {sorted(_Q_REDUCTIONS)}, got {value!r}.")
            setattr(cfg, key, value)

        for key in ("outer_critic_target", "inner_sac_critic_target"):
            value = getattr(cfg, key)
            if not isinstance(value, str):
                raise ValueError(
                    f"{key} must be one of {sorted(_CRITIC_TARGETS)}, got {value!r}."
                )
            value = value.lower()
            if value not in _CRITIC_TARGETS:
                raise ValueError(
                    f"{key} must be one of {sorted(_CRITIC_TARGETS)}, got {value!r}."
                )
            setattr(cfg, key, value)

        if not isinstance(cfg.sac_actor_loss_scale_mode, str):
            raise ValueError(
                "sac_actor_loss_scale_mode must be one of "
                f"{sorted(_SAC_ACTOR_LOSS_SCALE_MODES)}, got "
                f"{cfg.sac_actor_loss_scale_mode!r}."
            )
        cfg.sac_actor_loss_scale_mode = cfg.sac_actor_loss_scale_mode.lower()
        if cfg.sac_actor_loss_scale_mode not in _SAC_ACTOR_LOSS_SCALE_MODES:
            raise ValueError(
                "sac_actor_loss_scale_mode must be one of "
                f"{sorted(_SAC_ACTOR_LOSS_SCALE_MODES)}, got "
                f"{cfg.sac_actor_loss_scale_mode!r}."
            )
        if (
            cfg.sac_actor_loss_scale_tau is None
            or isinstance(cfg.sac_actor_loss_scale_tau, bool)
        ):
            raise ValueError(
                "sac_actor_loss_scale_tau must be a finite float in (0, 1]."
            )
        try:
            cfg.sac_actor_loss_scale_tau = _finite_float(
                cfg.sac_actor_loss_scale_tau, "sac_actor_loss_scale_tau"
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "sac_actor_loss_scale_tau must be a finite float in (0, 1]."
            ) from exc
        if not 0.0 < cfg.sac_actor_loss_scale_tau <= 1.0:
            raise ValueError("sac_actor_loss_scale_tau must be in (0, 1].")

        if cfg.inner_operator is None:
            raise ValueError("inner_operator must be a string, not null.")
        cfg.inner_operator = str(cfg.inner_operator).lower()
        if cfg.inner_operator not in {"none", "sac", "td3", "mppi"}:
            raise ValueError("inner_operator must be one of 'none', 'sac', 'td3', or 'mppi'.")

        if cfg.inner_rollout_horizon is None:
            cfg.inner_rollout_horizon = 3
        cfg.inner_rollout_horizon = int(cfg.inner_rollout_horizon)
        if cfg.inner_rollout_horizon <= 0:
            raise ValueError("inner_rollout_horizon must be positive.")
        cfg.inner_horizon_ratio = cfg.inner_rollout_horizon / float(
            cfg.train_unroll_horizon
        )
        if cfg.inner_rollout_horizon > int(cfg.train_unroll_horizon):
            warnings.warn(
                f"inner_rollout_horizon={cfg.inner_rollout_horizon} exceeds "
                f"train_unroll_horizon={cfg.train_unroll_horizon}; the inner "
                "controller is extrapolating beyond recurrent world-model training, "
                "which increases compounding model-bias risk.",
                UserWarning,
                stacklevel=2,
            )

        cfg.inner_mppi_iterations = int(cfg.inner_mppi_iterations)
        if cfg.inner_mppi_iterations < 0:
            raise ValueError("inner_mppi_iterations must be non-negative.")

        if cfg.inner_operator in {"sac", "td3"}:
            cfg.inner_rounds = int(cfg.inner_rounds)
            if cfg.inner_rounds < 0:
                raise ValueError("inner_rounds must be non-negative.")

            if cfg.inner_schedule_mode == "canonical":
                cfg.inner_rollouts_per_round = int(cfg.inner_rollouts_per_round)
                if cfg.inner_rollouts_per_round < 0:
                    raise ValueError("inner_rollouts_per_round must be non-negative.")
                if cfg.inner_component_update_schedule:
                    cfg.inner_critic_updates_per_round = _strict_nonnegative_int(
                        cfg.inner_critic_updates_per_round,
                        "inner_critic_updates_per_round",
                    )
                    cfg.inner_actor_updates_per_round = _strict_nonnegative_int(
                        cfg.inner_actor_updates_per_round,
                        "inner_actor_updates_per_round",
                    )
                    cfg.inner_updates_per_round = None
                    cfg.inner_critic_updates_per_action = (
                        cfg.inner_rounds * cfg.inner_critic_updates_per_round
                    )
                    cfg.inner_actor_updates_per_action = (
                        cfg.inner_rounds * cfg.inner_actor_updates_per_round
                    )
                    cfg.inner_temperature_updates_per_action = (
                        cfg.inner_actor_updates_per_action
                        if cfg.inner_operator == "sac"
                        and str(cfg.inner_temperature_mode).lower() == "auto"
                        else 0
                    )
                else:
                    if isinstance(cfg.inner_updates_per_round, str):
                        cfg.inner_updates_per_round = cfg.inner_updates_per_round.lower()
                        if cfg.inner_updates_per_round != "auto":
                            raise ValueError(
                                "inner_updates_per_round must be a non-negative integer or 'auto'."
                            )
                        nominal_updates_per_round = (
                            cfg.inner_rollouts_per_round * cfg.inner_rollout_horizon
                        )
                    else:
                        cfg.inner_updates_per_round = int(cfg.inner_updates_per_round)
                        if cfg.inner_updates_per_round < 0:
                            raise ValueError("inner_updates_per_round must be non-negative.")
                        nominal_updates_per_round = cfg.inner_updates_per_round

                    nominal_total = cfg.inner_rounds * nominal_updates_per_round
                    actor_enabled = str(cfg.inner_actor_adaptation).lower() != "frozen"
                    critic_enabled = str(cfg.inner_critic_adaptation).lower() != "frozen"
                    temperature_enabled = (
                        cfg.inner_operator == "sac"
                        and str(cfg.inner_temperature_mode).lower() == "auto"
                    )
                    cfg.inner_critic_updates_per_action = (
                        nominal_total if critic_enabled else 0
                    )
                    cfg.inner_actor_updates_per_action = (
                        nominal_total if actor_enabled else 0
                    )
                    cfg.inner_temperature_updates_per_action = (
                        nominal_total if temperature_enabled else 0
                    )

                cfg.inner_model_step_budget = (
                    cfg.inner_rounds
                    * cfg.inner_rollouts_per_round
                    * cfg.inner_rollout_horizon
                )
            else:
                cfg.inner_updates_per_round = None
                if cfg.inner_model_step_budget is None:
                    cfg.inner_model_step_budget = (
                        cfg.inner_rounds * 32 * cfg.inner_rollout_horizon
                    )
                cfg.inner_model_step_budget = int(cfg.inner_model_step_budget)
                if cfg.inner_model_step_budget < 0:
                    raise ValueError("inner_model_step_budget must be non-negative.")
                if cfg.inner_rounds == 0:
                    if cfg.inner_model_step_budget != 0:
                        raise ValueError(
                            "A positive inner_model_step_budget requires inner_rounds>0."
                        )
                    cfg.inner_rollouts_per_round = 0
                else:
                    denominator = cfg.inner_rounds * cfg.inner_rollout_horizon
                    if cfg.inner_model_step_budget % denominator:
                        raise ValueError(
                            "inner_model_step_budget must be divisible by "
                            "inner_rounds * inner_rollout_horizon."
                        )
                    cfg.inner_rollouts_per_round = (
                        cfg.inner_model_step_budget // denominator
                    )
                for key in (
                    "inner_critic_updates_per_action",
                    "inner_actor_updates_per_action",
                    "inner_temperature_updates_per_action",
                ):
                    value = int(getattr(cfg, key))
                    if value < 0:
                        raise ValueError(f"{key} must be non-negative, got {value}.")
                    setattr(cfg, key, value)

            if cfg.inner_rounds == 0 and cfg.inner_model_step_budget != 0:
                raise ValueError(
                    "A positive inner model-step budget requires inner_rounds>0."
                )
            if (
                cfg.inner_rounds > 0
                and cfg.inner_rollouts_per_round == 0
                and cfg.inner_model_step_budget != 0
            ):
                raise ValueError("Positive model compute requires rollouts per round.")
        elif cfg.inner_operator == "mppi":
            cfg.inner_rounds = 0
            cfg.inner_rollouts_per_round = 0
            cfg.inner_updates_per_round = 0
            if cfg.inner_mppi_iterations <= 0:
                raise ValueError("MPPI requires inner_mppi_iterations>0.")
            if cfg.inner_model_step_budget is None:
                policy_prior_steps = int(cfg.inner_mppi_num_pi_trajs) * max(
                    0, cfg.inner_rollout_horizon - 1
                )
                cfg.inner_model_step_budget = (
                    policy_prior_steps
                    + cfg.inner_mppi_iterations * 32 * cfg.inner_rollout_horizon
                )
            cfg.inner_model_step_budget = int(cfg.inner_model_step_budget)
            for key in (
                "inner_critic_updates_per_action",
                "inner_actor_updates_per_action",
                "inner_temperature_updates_per_action",
            ):
                setattr(cfg, key, 0)
        else:
            cfg.inner_rounds = 0
            cfg.inner_rollouts_per_round = 0
            cfg.inner_updates_per_round = 0
            cfg.inner_mppi_iterations = 0
            cfg.inner_model_step_budget = 0
            cfg.inner_critic_updates_per_action = 0
            cfg.inner_actor_updates_per_action = 0
            cfg.inner_temperature_updates_per_action = 0

        cfg.inner_model_step_budget = int(cfg.inner_model_step_budget)
        if cfg.inner_model_step_budget < 0:
            raise ValueError("inner_model_step_budget must be non-negative.")
        nominal_transitions_per_round = (
            cfg.inner_rollouts_per_round * cfg.inner_rollout_horizon
        )
        if cfg.inner_operator in {"sac", "td3"}:
            if cfg.inner_schedule_mode == "canonical":
                if cfg.inner_component_update_schedule:
                    cfg.inner_nominal_updates_per_round = (
                        cfg.inner_critic_updates_per_round
                        + cfg.inner_actor_updates_per_round
                    )
                else:
                    cfg.inner_nominal_updates_per_round = (
                        nominal_transitions_per_round
                        if cfg.inner_updates_per_round == "auto"
                        else int(cfg.inner_updates_per_round)
                    )
            else:
                cfg.inner_nominal_updates_per_round = (
                    max(
                        cfg.inner_critic_updates_per_action,
                        cfg.inner_actor_updates_per_action,
                        cfg.inner_temperature_updates_per_action,
                    )
                    / cfg.inner_rounds
                    if cfg.inner_rounds > 0
                    else 0.0
                )
        else:
            cfg.inner_nominal_updates_per_round = 0
        cfg.inner_nominal_transitions_per_round = nominal_transitions_per_round
        if cfg.inner_component_update_schedule:
            cfg.inner_expected_update_slots = (
                cfg.inner_critic_updates_per_action
                + cfg.inner_actor_updates_per_action
            )
        else:
            cfg.inner_expected_update_slots = max(
                cfg.inner_critic_updates_per_action,
                cfg.inner_actor_updates_per_action,
                cfg.inner_temperature_updates_per_action,
            )
        cfg.inner_nominal_critic_utd = (
            cfg.inner_critic_updates_per_action / cfg.inner_model_step_budget
            if cfg.inner_model_step_budget > 0
            else 0.0
        )

        if (
            cfg.inner_operator in {"sac", "td3"}
            and cfg.inner_model_step_budget == 0
            and any(
                getattr(cfg, key) > 0
                for key in (
                    "inner_critic_updates_per_action",
                    "inner_actor_updates_per_action",
                    "inner_temperature_updates_per_action",
                )
            )
        ):
            raise ValueError("Inner optimizer updates require a positive inner_model_step_budget.")

        integer_positive = (
            "inner_batch_size",
            "inner_critic_target_update_interval",
            "target_update_interval",
        )
        for key in integer_positive:
            value = int(getattr(cfg, key))
            if value <= 0:
                raise ValueError(f"{key} must be positive, got {value}.")
            setattr(cfg, key, value)

        if cfg.inner_actor_target_update_interval is None:
            cfg.inner_actor_target_update_interval = (
                cfg.inner_critic_target_update_interval
            )
        cfg.inner_actor_target_update_interval = int(
            cfg.inner_actor_target_update_interval
        )
        if cfg.inner_actor_target_update_interval <= 0:
            raise ValueError("inner_actor_target_update_interval must be positive.")

        if cfg.inner_replay_capacity is None:
            cfg.inner_replay_capacity = max(1, cfg.inner_model_step_budget)
        cfg.inner_replay_capacity = int(cfg.inner_replay_capacity)
        if cfg.inner_replay_capacity <= 0:
            raise ValueError("inner_replay_capacity must be positive.")
        cfg.inner_replay_sampling = str(cfg.inner_replay_sampling).lower()
        if cfg.inner_replay_sampling not in {"with_replacement", "without_replacement"}:
            raise ValueError(
                "inner_replay_sampling must be 'with_replacement' or 'without_replacement'."
            )
        has_inner_updates = any(
            getattr(cfg, key) > 0
            for key in (
                "inner_critic_updates_per_action",
                "inner_actor_updates_per_action",
                "inner_temperature_updates_per_action",
            )
        )
        if (
            cfg.inner_operator in {"sac", "td3"}
            and cfg.inner_replay_sampling == "without_replacement"
            and has_inner_updates
            and cfg.inner_batch_size > cfg.inner_replay_capacity
        ):
            raise ValueError(
                "without-replacement inner replay requires inner_replay_capacity "
                ">= inner_batch_size."
            )
        if (
            cfg.inner_operator in {"sac", "td3"}
            and cfg.inner_replay_sampling == "without_replacement"
            and has_inner_updates
            and cfg.inner_rounds > 0
            and cfg.inner_batch_size > cfg.inner_model_step_budget // cfg.inner_rounds
        ):
            raise ValueError(
                "without-replacement inner replay cannot fill inner_batch_size before the "
                "first update round; reduce the batch or increase the model-step budget."
            )

        for key in ("inner_actor_adaptation", "inner_critic_adaptation"):
            value = str(getattr(cfg, key)).lower()
            if value not in _ADAPTATION_MODES:
                raise ValueError(f"{key} must be one of {sorted(_ADAPTATION_MODES)}.")
            setattr(cfg, key, value)
        cfg.inner_critic_dropout_enabled = _strict_bool(
            cfg.inner_critic_dropout_enabled,
            "inner_critic_dropout_enabled",
        )
        if (
            cfg.inner_operator in {"sac", "td3"}
            and cfg.inner_actor_updates_per_action > 0
            and cfg.inner_actor_adaptation == "frozen"
        ):
            raise ValueError("Positive actor updates require adaptable inner_actor_adaptation.")
        if (
            cfg.inner_operator in {"sac", "td3"}
            and cfg.inner_critic_updates_per_action > 0
            and cfg.inner_critic_adaptation == "frozen"
        ):
            raise ValueError("Positive critic updates require adaptable inner_critic_adaptation.")

        cfg.inner_temperature_mode = str(cfg.inner_temperature_mode).lower()
        if cfg.inner_temperature_mode not in {"inherit_outer", "fixed", "auto"}:
            raise ValueError(
                "inner_temperature_mode must be 'inherit_outer', 'fixed', or 'auto'."
            )
        if cfg.inner_temperature_updates_per_action > 0:
            if cfg.inner_operator != "sac":
                raise ValueError("Temperature updates are only valid for the SAC inner operator.")
            if cfg.inner_temperature_mode != "auto":
                raise ValueError("Temperature updates require inner_temperature_mode='auto'.")
        if cfg.inner_operator == "td3" and cfg.inner_temperature_mode != "inherit_outer":
            raise ValueError("TD3 has no entropy temperature; use inherit_outer.")
        if cfg.inner_operator in {"none", "mppi"} and cfg.inner_temperature_mode != "inherit_outer":
            raise ValueError(
                f"{cfg.inner_operator} has no learned entropy temperature; use inherit_outer."
            )
        cfg.inner_temperature_initialization = str(
            cfg.inner_temperature_initialization
        ).lower()
        if cfg.inner_temperature_initialization not in {"inherit_outer", "fixed"}:
            raise ValueError(
                "inner_temperature_initialization must be 'inherit_outer' or 'fixed'."
            )
        cfg.inner_temperature = _finite_float(
            cfg.inner_temperature, "inner_temperature"
        )
        if cfg.inner_temperature <= 0.0:
            raise ValueError("inner_temperature must be positive.")
        if isinstance(cfg.inner_target_entropy, str):
            cfg.inner_target_entropy = cfg.inner_target_entropy.lower()
        if cfg.inner_target_entropy not in {"auto", "inherit_outer"}:
            cfg.inner_target_entropy = _finite_float(
                cfg.inner_target_entropy, "inner_target_entropy"
            )

        cfg.inner_bootstrap_source = str(cfg.inner_bootstrap_source).lower()
        if cfg.inner_bootstrap_source not in {"inner_target", "outer_target", "outer_online"}:
            raise ValueError(
                "inner_bootstrap_source must be 'inner_target', 'outer_target', or 'outer_online'."
            )

        for component in ("actor", "critic"):
            rank_key = f"inner_{component}_lora_rank"
            scale_key = f"inner_{component}_lora_scale"
            dropout_key = f"inner_{component}_lora_dropout"
            setattr(cfg, rank_key, int(getattr(cfg, rank_key)))
            setattr(
                cfg,
                scale_key,
                _finite_float(getattr(cfg, scale_key), scale_key),
            )
            setattr(
                cfg,
                dropout_key,
                _finite_float(getattr(cfg, dropout_key), dropout_key),
            )
            if getattr(cfg, rank_key) <= 0:
                raise ValueError(f"{rank_key} must be positive.")
            if getattr(cfg, scale_key) <= 0.0:
                raise ValueError(f"{scale_key} must be positive.")
            if not 0.0 <= getattr(cfg, dropout_key) < 1.0:
                raise ValueError(f"{dropout_key} must be in [0, 1).")

        for key in (
            "inner_actor_lr",
            "inner_critic_lr",
            "inner_temperature_lr",
            "inner_actor_grad_clip_norm",
            "inner_critic_grad_clip_norm",
            "inner_temperature_grad_clip_norm",
        ):
            value = _finite_float(getattr(cfg, key), key)
            if value <= 0.0:
                raise ValueError(f"{key} must be positive.")
            setattr(cfg, key, value)
        if cfg.inner_operator == "sac" and cfg.inner_outer_action_l2_coef != 0.0:
            raise ValueError(
                "inner_outer_action_l2_coef is a TD3-only policy anchor."
            )
        if cfg.inner_operator == "td3" and cfg.inner_outer_policy_kl_coef != 0.0:
            raise ValueError(
                "inner_outer_policy_kl_coef is a SAC-only policy anchor."
            )

        for key in (
            "inner_outer_policy_kl_coef",
            "inner_outer_action_l2_coef",
            "inner_behavior_std_scale",
            "inner_behavior_noise_std",
            "inner_execution_std_scale",
            "inner_execution_noise_std",
            "inner_td3_target_noise_std",
            "inner_td3_target_noise_clip",
        ):
            value = _finite_float(getattr(cfg, key), key)
            if value < 0.0:
                raise ValueError(f"{key} must be non-negative.")
            setattr(cfg, key, value)

        for key in ("inner_behavior_action", "inner_execution_action"):
            value = str(getattr(cfg, key)).lower()
            if value not in {"policy_sample", "mean", "mean_plus_gaussian"}:
                raise ValueError(
                    f"{key} must be 'policy_sample', 'mean', or 'mean_plus_gaussian'."
                )
            setattr(cfg, key, value)
        for prefix in ("inner_behavior", "inner_execution"):
            mode = getattr(cfg, f"{prefix}_action")
            std_scale = getattr(cfg, f"{prefix}_std_scale")
            noise_std = getattr(cfg, f"{prefix}_noise_std")
            if mode != "policy_sample" and std_scale != 1.0:
                raise ValueError(
                    f"{prefix}_std_scale only affects {prefix}_action='policy_sample'."
                )
            if mode != "mean_plus_gaussian" and noise_std != 0.0:
                raise ValueError(
                    f"{prefix}_noise_std requires "
                    f"{prefix}_action='mean_plus_gaussian'."
                )

        objective = cfg.outer_behavior_policy_objective
        if not isinstance(objective, str):
            raise ValueError(
                "outer_behavior_policy_objective must be one of "
                f"{sorted(_BEHAVIOR_POLICY_OBJECTIVES)}, got {objective!r}."
            )
        objective = objective.lower()
        if objective not in _BEHAVIOR_POLICY_OBJECTIVES:
            raise ValueError(
                "outer_behavior_policy_objective must be one of "
                f"{sorted(_BEHAVIOR_POLICY_OBJECTIVES)}, got {objective!r}."
            )
        cfg.outer_behavior_policy_objective = objective

        schedule = cfg.outer_behavior_policy_kl_schedule
        if not isinstance(schedule, str):
            raise ValueError(
                "outer_behavior_policy_kl_schedule must be one of "
                f"{sorted(_BEHAVIOR_POLICY_KL_SCHEDULES)}, got {schedule!r}."
            )
        schedule = schedule.lower()
        if schedule not in _BEHAVIOR_POLICY_KL_SCHEDULES:
            raise ValueError(
                "outer_behavior_policy_kl_schedule must be one of "
                f"{sorted(_BEHAVIOR_POLICY_KL_SCHEDULES)}, got {schedule!r}."
            )
        cfg.outer_behavior_policy_kl_schedule = schedule
        if objective == "action_space_cross_entropy" and schedule == "dual":
            raise ValueError(
                "outer_behavior_policy_objective='action_space_cross_entropy' "
                "does not support outer_behavior_policy_kl_schedule='dual'."
            )

        cfg.outer_behavior_policy_kl_coef = _strict_nonnegative_float(
            cfg.outer_behavior_policy_kl_coef,
            "outer_behavior_policy_kl_coef",
        )
        cfg.outer_behavior_policy_kl_ramp_updates = _strict_positive_int(
            cfg.outer_behavior_policy_kl_ramp_updates,
            "outer_behavior_policy_kl_ramp_updates",
        )
        for key in (
            "outer_behavior_policy_kl_q_threshold",
            "outer_behavior_policy_kl_target",
            "outer_behavior_policy_kl_dual_init",
            "outer_behavior_policy_kl_dual_lr",
            "outer_behavior_policy_kl_dual_max",
        ):
            setattr(cfg, key, _strict_nonnegative_float(getattr(cfg, key), key))
        if cfg.outer_behavior_policy_kl_q_threshold <= 0.0:
            raise ValueError(
                "outer_behavior_policy_kl_q_threshold must be positive."
            )
        for key in (
            "outer_behavior_policy_kl_dual_init",
            "outer_behavior_policy_kl_dual_lr",
            "outer_behavior_policy_kl_dual_max",
        ):
            if getattr(cfg, key) <= 0.0:
                raise ValueError(f"{key} must be positive.")
        if (
            cfg.outer_behavior_policy_kl_dual_init
            > cfg.outer_behavior_policy_kl_dual_max
        ):
            raise ValueError(
                "outer_behavior_policy_kl_dual_init must not exceed "
                "outer_behavior_policy_kl_dual_max."
            )

        min_valid_count = cfg.outer_behavior_policy_kl_min_valid_count
        if isinstance(min_valid_count, str):
            if min_valid_count.lower() != "auto":
                raise ValueError(
                    "outer_behavior_policy_kl_min_valid_count must be 'auto' "
                    "or a positive integer."
                )
            min_valid_count = int(cfg.batch_size)
        else:
            min_valid_count = _strict_positive_int(
                min_valid_count,
                "outer_behavior_policy_kl_min_valid_count",
            )
        cfg.outer_behavior_policy_kl_min_valid_count = min_valid_count

        if schedule in {"smooth", "quantile_gate"}:
            if cfg.outer_behavior_policy_kl_coef <= 0.0:
                raise ValueError(
                    "outer_behavior_policy_kl_coef must be positive for the "
                    f"{schedule!r} schedule."
                )
        if schedule != "none":
            if cfg.inner_operator != "sac":
                raise ValueError(
                    "An active outer behavior-policy regularizer requires "
                    "inner_operator='sac'."
                )
            if (
                cfg.inner_execution_action != "policy_sample"
                or cfg.inner_execution_std_scale <= 0.0
            ):
                raise ValueError(
                    "An active outer behavior-policy regularizer requires "
                    "stochastic inner execution with "
                    "inner_execution_action='policy_sample' and "
                    "inner_execution_std_scale > 0."
                )
        cfg.store_behavior_policy = schedule != "none"
        cfg.log_std_mapping = normalize_log_std_mapping(
            cfg.log_std_mapping, "log_std_mapping"
        )
        cfg.log_std_min = _finite_float(cfg.log_std_min, "log_std_min")
        cfg.log_std_max = _finite_float(cfg.log_std_max, "log_std_max")
        if cfg.log_std_min >= cfg.log_std_max:
            raise ValueError("log_std_min must be less than log_std_max.")
        cfg.inner_log_std_mapping = (
            cfg.log_std_mapping
            if cfg.inner_log_std_mapping is None
            else normalize_log_std_mapping(
                cfg.inner_log_std_mapping, "inner_log_std_mapping"
            )
        )
        cfg.inner_log_std_min = _finite_float(
            cfg.log_std_min if cfg.inner_log_std_min is None else cfg.inner_log_std_min,
            "inner_log_std_min",
        )
        cfg.inner_log_std_max = _finite_float(
            cfg.log_std_max if cfg.inner_log_std_max is None else cfg.inner_log_std_max,
            "inner_log_std_max",
        )
        if cfg.inner_log_std_min >= cfg.inner_log_std_max:
            raise ValueError("inner_log_std_min must be less than inner_log_std_max.")

        scope_keys = (
            "inner_actor_scope",
            "inner_critic_scope",
            "inner_temperature_scope",
            "inner_replay_scope",
            "inner_actor_optimizer_scope",
            "inner_critic_optimizer_scope",
            "inner_temperature_optimizer_scope",
            "inner_mppi_warm_start_scope",
        )
        for key in scope_keys:
            value = str(getattr(cfg, key)).lower()
            if value not in _LIFECYCLE_SCOPES:
                raise ValueError(f"{key} must be one of {sorted(_LIFECYCLE_SCOPES)}.")
            setattr(cfg, key, value)
        if (
            cfg.inner_operator in {"sac", "td3"}
            and cfg.inner_replay_scope == "action"
            and cfg.inner_replay_capacity < cfg.inner_model_step_budget
        ):
            raise ValueError(
                "Action-local inner_replay_capacity must be at least the cumulative "
                "nominal J*N*H transitions for one real action: "
                f"capacity={cfg.inner_replay_capacity}, "
                f"required={cfg.inner_model_step_budget}."
            )
        for component in ("actor", "critic", "temperature"):
            parameter_scope = getattr(cfg, f"inner_{component}_scope")
            optimizer_scope = getattr(cfg, f"inner_{component}_optimizer_scope")
            if _SCOPE_RANK[optimizer_scope] > _SCOPE_RANK[parameter_scope]:
                raise ValueError(
                    f"inner_{component}_optimizer_scope cannot outlive inner_{component}_scope."
                )
        cfg.inner_rebase_persistent = bool(cfg.inner_rebase_persistent)

        writeback_active = bool(
            cfg.inner_actor_writeback_coef > 0.0
            or cfg.inner_critic_writeback_coef > 0.0
        )
        if writeback_active:
            if cfg.inner_schedule_mode != "canonical":
                raise ValueError(
                    "Active inner prior write-back requires the canonical "
                    "J/N/H/G inner schedule."
                )
            if cfg.inner_operator != "sac":
                raise ValueError(
                    "Active inner prior write-back requires inner_operator='sac'."
                )
            for component in ("actor", "critic"):
                if getattr(cfg, f"inner_{component}_adaptation") != "clone":
                    raise ValueError(
                        "Active inner prior write-back requires full-clone actor "
                        "and critic adaptation; "
                        f"inner_{component}_adaptation must be 'clone'."
                    )
            action_scopes = (
                "inner_actor_scope",
                "inner_critic_scope",
                "inner_temperature_scope",
                "inner_replay_scope",
                "inner_actor_optimizer_scope",
                "inner_critic_optimizer_scope",
                "inner_temperature_optimizer_scope",
            )
            for key in action_scopes:
                if getattr(cfg, key) != "action":
                    raise ValueError(
                        "Active inner prior write-back requires the canonical "
                        f"action-local lifecycle; {key} must be 'action'."
                    )
            if cfg.outer_policy_episode_probability != 0.0:
                raise ValueError(
                    "Active inner prior write-back requires "
                    "outer_policy_episode_probability=0 so every planned "
                    "training action uses the inner SAC learner."
                )
            if (
                cfg.inner_actor_writeback_coef > 0.0
                and cfg.inner_actor_updates_per_action <= 0
            ):
                raise ValueError(
                    "A positive inner_actor_writeback_coef requires positive "
                    "inner actor updates per action."
                )
            if (
                cfg.inner_critic_writeback_coef > 0.0
                and cfg.inner_critic_updates_per_action <= 0
            ):
                raise ValueError(
                    "A positive inner_critic_writeback_coef requires positive "
                    "inner critic updates per action."
                )
            if (
                cfg.inner_log_std_mapping != cfg.log_std_mapping
                or cfg.inner_log_std_min != cfg.log_std_min
                or cfg.inner_log_std_max != cfg.log_std_max
            ):
                raise ValueError(
                    "Active inner prior write-back requires identical inner and "
                    "outer actor log-std mappings and bounds."
                )

        _resolve_random_explorer_config(cfg)

        cfg.inner_mppi_num_elites = int(cfg.inner_mppi_num_elites)
        cfg.inner_mppi_num_pi_trajs = int(cfg.inner_mppi_num_pi_trajs)
        if cfg.inner_mppi_num_elites <= 0:
            raise ValueError("inner_mppi_num_elites must be positive.")
        if cfg.inner_mppi_num_pi_trajs < 0:
            raise ValueError("inner_mppi_num_pi_trajs must be non-negative.")
        for key in ("inner_mppi_temperature", "inner_mppi_min_std", "inner_mppi_max_std"):
            value = _finite_float(getattr(cfg, key), key)
            if value <= 0.0:
                raise ValueError(f"{key} must be positive.")
            setattr(cfg, key, value)
        if cfg.inner_mppi_min_std > cfg.inner_mppi_max_std:
            raise ValueError("inner_mppi_min_std cannot exceed inner_mppi_max_std.")
        cfg.inner_mppi_num_samples = 0
        if cfg.inner_operator == "mppi":
            if cfg.inner_mppi_iterations <= 0:
                raise ValueError("MPPI requires inner_mppi_iterations>0.")
            # Policy-prior generation advances the model H-1 times. Candidate
            # evaluation then advances every candidate H times in every MPPI
            # iteration. SAC's collection rounds do not participate here.
            policy_prior_steps = (
                cfg.inner_mppi_num_pi_trajs * max(0, cfg.inner_rollout_horizon - 1)
            )
            optim_budget = cfg.inner_model_step_budget - policy_prior_steps
            denominator = cfg.inner_mppi_iterations * cfg.inner_rollout_horizon
            if optim_budget <= 0 or optim_budget % denominator:
                raise ValueError(
                    "MPPI model-step budget, after policy-prior trajectory overhead, must "
                    "divide evenly across inner_mppi_iterations * inner_rollout_horizon."
                )
            cfg.inner_mppi_num_samples = optim_budget // denominator
            if cfg.inner_mppi_num_pi_trajs > cfg.inner_mppi_num_samples:
                raise ValueError("inner_mppi_num_pi_trajs cannot exceed derived candidate count.")
            if cfg.inner_mppi_num_elites > cfg.inner_mppi_num_samples:
                raise ValueError("inner_mppi_num_elites cannot exceed derived candidate count.")

        cfg.inner_diagnostic_rollouts = int(cfg.inner_diagnostic_rollouts)
        if cfg.inner_diagnostic_rollouts < 0:
            raise ValueError("inner_diagnostic_rollouts must be non-negative.")
        cfg.inner_diagnostics_every = int(cfg.inner_diagnostics_every)
        if cfg.inner_diagnostics_every <= 0:
            raise ValueError("inner_diagnostics_every must be positive.")
        cfg.value_equivalence_diagnostics = _strict_bool(
            cfg.value_equivalence_diagnostics,
            "value_equivalence_diagnostics",
        )
        cfg.value_equivalence_every_updates = _strict_positive_int(
            cfg.value_equivalence_every_updates,
            "value_equivalence_every_updates",
        )
        cfg.value_equivalence_mc_samples = _strict_positive_int(
            cfg.value_equivalence_mc_samples,
            "value_equivalence_mc_samples",
        )
        cfg.value_equivalence_loss_coef = _strict_nonnegative_float(
            cfg.value_equivalence_loss_coef,
            "value_equivalence_loss_coef",
        )
        cfg.value_equivalence_loss_mc_samples = _strict_positive_int(
            cfg.value_equivalence_loss_mc_samples,
            "value_equivalence_loss_mc_samples",
        )
        cfg.eval_value = _strict_bool(cfg.eval_value, "eval_value")
        cfg.eval_value_samples = _strict_positive_int(
            cfg.eval_value_samples,
            "eval_value_samples",
        )
        cfg.eval_value_seed = _strict_nonnegative_int(
            cfg.eval_value_seed,
            "eval_value_seed",
        )
        cfg.eval_value_protocols = _normalize_eval_value_protocols(
            cfg.eval_value_protocols
        )
        cfg.eval_inner_comparison = _strict_bool(
            cfg.eval_inner_comparison,
            "eval_inner_comparison",
        )
        cfg.eval_inner_comparison_episodes = _strict_positive_int(
            cfg.eval_inner_comparison_episodes,
            "eval_inner_comparison_episodes",
        )
        cfg.eval_inner_comparison_seed = _strict_nonnegative_int(
            cfg.eval_inner_comparison_seed,
            "eval_inner_comparison_seed",
        )
        if cfg.value_equivalence_diagnostics and cfg.inner_operator != "sac":
            raise ValueError(
                "value_equivalence_diagnostics requires inner_operator='sac'."
            )
        if cfg.value_equivalence_loss_coef > 0.0:
            if cfg.inner_operator != "sac":
                raise ValueError(
                    "A positive value_equivalence_loss_coef requires "
                    "inner_operator='sac'."
                )
            if bool(cfg.episodic):
                raise ValueError(
                    "A positive value_equivalence_loss_coef requires "
                    "episodic=false; the VE loss is value-only and does not "
                    "model termination."
                )
        if cfg.eval_value:
            if cfg.eval_freq is None:
                raise ValueError("eval_value=true requires a configured eval_freq.")
            if cfg.obs != "state":
                raise ValueError(
                    "eval_value=true currently supports state observations only."
                )
            if cfg.outer_critic_target != "reward_only":
                raise ValueError(
                    "eval_value=true requires outer_critic_target='reward_only' "
                    "so discounted reward rollouts match the critic target."
                )
            if (
                "paper_deterministic" in cfg.eval_value_protocols
                and cfg.q_pair_size != 2
            ):
                raise ValueError(
                    "paper_deterministic value evaluation requires q_pair_size=2."
                )
        if cfg.eval_inner_comparison:
            if cfg.eval_freq is None:
                raise ValueError(
                    "eval_inner_comparison=true requires a configured eval_freq."
                )
            if cfg.obs != "state":
                raise ValueError(
                    "eval_inner_comparison=true currently supports state "
                    "observations only."
                )
            if cfg.inner_operator != "sac":
                raise ValueError(
                    "eval_inner_comparison=true requires inner_operator='sac'."
                )
            if int(cfg.inner_rounds) <= 0:
                raise ValueError(
                    "eval_inner_comparison=true requires an active inner SAC "
                    "solve with inner_rounds>0."
                )
            if cfg.inner_diagnostic_rollouts != 0:
                raise ValueError(
                    "eval_inner_comparison=true requires "
                    "inner_diagnostic_rollouts=0 so its root diagnostic is the "
                    "fixed-target action-Q comparison only."
                )
            comparison_scopes = (
                "inner_actor_scope",
                "inner_critic_scope",
                "inner_temperature_scope",
                "inner_replay_scope",
                "inner_actor_optimizer_scope",
                "inner_critic_optimizer_scope",
                "inner_temperature_optimizer_scope",
            )
            for key in comparison_scopes:
                if getattr(cfg, key) != "action":
                    raise ValueError(
                        "eval_inner_comparison=true requires fresh action-local "
                        f"inner state; {key} must be 'action'."
                    )
        cfg.compile_strict = bool(cfg.compile_strict)

        cfg.inner_termination_threshold = _finite_float(
            cfg.inner_termination_threshold, "inner_termination_threshold"
        )
        if not 0.0 <= cfg.inner_termination_threshold <= 1.0:
            raise ValueError("inner_termination_threshold must be in [0, 1].")
        cfg.tau = _finite_float(cfg.tau, "tau")
        if not 0.0 < cfg.tau <= 1.0:
            raise ValueError("tau must be in (0, 1].")
        cfg.inner_critic_target_tau = _finite_float(
            cfg.inner_critic_target_tau, "inner_critic_target_tau"
        )
        if not 0.0 < cfg.inner_critic_target_tau <= 1.0:
            raise ValueError("inner_critic_target_tau must be in (0, 1].")
        if cfg.inner_actor_target_tau is None:
            cfg.inner_actor_target_tau = cfg.inner_critic_target_tau
        cfg.inner_actor_target_tau = _finite_float(
            cfg.inner_actor_target_tau, "inner_actor_target_tau"
        )
        if not 0.0 < cfg.inner_actor_target_tau <= 1.0:
            raise ValueError("inner_actor_target_tau must be in (0, 1].")
        cfg.adam_eps = _finite_float(cfg.adam_eps, "adam_eps")
        cfg.actor_adam_eps = _finite_float(
            cfg.adam_eps if cfg.actor_adam_eps is None else cfg.actor_adam_eps,
            "actor_adam_eps",
        )
        cfg.inner_adam_eps = _finite_float(cfg.inner_adam_eps, "inner_adam_eps")
        if min(cfg.adam_eps, cfg.actor_adam_eps, cfg.inner_adam_eps) <= 0.0:
            raise ValueError(
                "adam_eps, actor_adam_eps, and inner_adam_eps must be positive."
            )

        for key in ("actor_lr", "critic_lr", "ent_coef_lr"):
            value = _finite_float(getattr(cfg, key), key)
            if value <= 0.0:
                raise ValueError(f"{key} must be positive.")
            setattr(cfg, key, value)
        if isinstance(cfg.target_entropy, str):
            cfg.target_entropy = cfg.target_entropy.lower()
            if cfg.target_entropy != "auto":
                cfg.target_entropy = _finite_float(
                    cfg.target_entropy, "target_entropy"
                )
        else:
            cfg.target_entropy = _finite_float(cfg.target_entropy, "target_entropy")
        if isinstance(cfg.ent_coef, str):
            if not cfg.ent_coef.startswith("auto"):
                raise ValueError("ent_coef must be positive or use 'auto[_initial]'.")
            if "_" in cfg.ent_coef:
                initial_alpha = _finite_float(
                    cfg.ent_coef.split("_", 1)[1], "ent_coef initial value"
                )
                if initial_alpha <= 0.0:
                    raise ValueError("Automatic ent_coef initial value must be positive.")
        else:
            cfg.ent_coef = _finite_float(cfg.ent_coef, "ent_coef")
            if cfg.ent_coef <= 0.0:
                raise ValueError("ent_coef must be positive.")

        # Read-only aliases keep legacy integrations working for one release.
        # Canonical agent code must not use these for scheduling mixed updates.
        cfg.inner_iterations = (
            cfg.inner_mppi_iterations
            if cfg.inner_operator == "mppi"
            else cfg.inner_rounds
        )
        cfg.inner_rollouts = cfg.inner_rollouts_per_round
        cfg.inner_horizon = cfg.inner_rollout_horizon
        cfg.inner_buffer_size = cfg.inner_replay_capacity
        cfg.inner_tau = cfg.inner_critic_target_tau
        cfg.inner_target_update_interval = cfg.inner_critic_target_update_interval
        cfg.inner_adaptation = (
            cfg.inner_actor_adaptation
            if cfg.inner_actor_adaptation == cfg.inner_critic_adaptation
            else "mixed"
        )
        cfg.lora_rank = cfg.inner_actor_lora_rank
        cfg.lora_alpha = cfg.inner_actor_lora_scale * cfg.inner_actor_lora_rank
        cfg.lora_dropout = cfg.inner_actor_lora_dropout
        cfg.inner_grad_clip_norm = max(
            cfg.inner_actor_grad_clip_norm, cfg.inner_critic_grad_clip_norm
        )
        if cfg.inner_component_update_schedule:
            # There is no truthful legacy single-G alias for a phased C/A
            # schedule. Keep it explicitly unset instead of reporting zero
            # updates for a solve that may perform nonzero component steps.
            cfg.inner_updates_per_iteration = None
        elif cfg.inner_schedule_mode == "canonical":
            cfg.inner_updates_per_iteration = cfg.inner_nominal_updates_per_round
        elif (
            cfg.inner_rounds > 0
            and cfg.inner_actor_updates_per_action == cfg.inner_critic_updates_per_action
            and cfg.inner_actor_updates_per_action % cfg.inner_rounds == 0
        ):
            cfg.inner_updates_per_iteration = (
                cfg.inner_actor_updates_per_action // cfg.inner_rounds
            )
        else:
            cfg.inner_updates_per_iteration = 0

        return cfg

    def _make_agent(self, cfg):
        return AMBITDMPC2Agent(cfg)

    def _make_value_calibration_evaluator(self):
        from RL.tdmpc2_core.value_calibration import ValueCalibrationEvaluator
        from utils.core import build_env

        return ValueCalibrationEvaluator(
            model=self.agent.model,
            env_factory=lambda: build_env(
                self.run_params,
                self.experiment_params,
                render_mode=None,
            ),
            observation_to_tensor=self._obs_to_tensor,
            unscale_action=self._unscale_action,
            discount=float(self.cfg.discount),
            samples=int(self.cfg.eval_value_samples),
            seed=int(self.cfg.eval_value_seed),
            protocols=tuple(self.cfg.eval_value_protocols),
            device=self.agent.device,
        )

    def _make_paired_controller_evaluator(self):
        from RL.tdmpc2_core.paired_controller_evaluation import (
            PairedControllerEvaluator,
        )
        from utils.core import build_env

        return PairedControllerEvaluator(
            agent=self.agent,
            env_factory=lambda: build_env(
                self.run_params,
                self.experiment_params,
                render_mode=None,
            ),
            observation_to_tensor=self._obs_to_tensor,
            unscale_action=self._unscale_action,
            episodes=int(self.cfg.eval_inner_comparison_episodes),
            seed=int(self.cfg.eval_inner_comparison_seed),
            device=self.agent.device,
        )

    def _evaluation_payload_extras(self, step):
        del step
        extras = {}
        if bool(getattr(self.cfg, "eval_value", False)):
            if self._value_calibration_evaluator is None:
                self._value_calibration_evaluator = (
                    self._make_value_calibration_evaluator()
                )
            value_extras = self._value_calibration_evaluator.evaluate()
            if not isinstance(value_extras, Mapping):
                raise TypeError(
                    "The value-calibration evaluator must return a mapping."
                )
            extras.update(value_extras)

        if bool(getattr(self.cfg, "eval_inner_comparison", False)):
            if self._paired_controller_evaluator is None:
                self._paired_controller_evaluator = (
                    self._make_paired_controller_evaluator()
                )
            paired_extras = self._paired_controller_evaluator.evaluate()
            if not isinstance(paired_extras, Mapping):
                raise TypeError(
                    "The paired-controller evaluator must return a mapping."
                )
            collisions = set(extras).intersection(paired_extras)
            if collisions:
                raise RuntimeError(
                    "Evaluation probes emitted duplicate metrics: "
                    f"{sorted(collisions)}."
                )
            extras.update(paired_extras)
        return extras

    def close(self):
        value_evaluator = getattr(self, "_value_calibration_evaluator", None)
        paired_evaluator = getattr(self, "_paired_controller_evaluator", None)
        self._value_calibration_evaluator = None
        self._paired_controller_evaluator = None
        first_error = None
        for evaluator in (value_evaluator, paired_evaluator):
            if evaluator is None:
                continue
            try:
                evaluator.close()
            except BaseException as exc:  # Both auxiliary evaluators still close.
                if first_error is None:
                    first_error = exc
                elif hasattr(first_error, "add_note"):
                    first_error.add_note(
                        "A second auxiliary evaluator also failed to close: "
                        f"{exc!r}"
                    )
        if first_error is not None:
            raise first_error

    def _evaluate_policy(self, step, *, initial_obs=None):
        """Evaluate on an isolated copy of AMBI's persistent inner state.

        Evaluation must run the real inner solve, including adaptation within
        and across its own episodes, but those optimizer steps, replay inserts,
        counters, and private RNG draws are not training data. Capture the
        canonical boundary state and restore it after the shared evaluator
        finishes (or raises). At an ordinary post-training-episode call, boundary
        preparation also performs the single lifecycle reset that the subsequent
        environment reset would otherwise perform.
        """

        engine = self.agent.inner_engine
        try:
            inner_state = copy.deepcopy(engine.training_state_dict())
        except RuntimeError:
            self.agent.prepare_training_resume_boundary()
            inner_state = copy.deepcopy(engine.training_state_dict())
        boundary_prepared = self.agent._resume_boundary_prepared
        last_metrics = copy.deepcopy(self.agent.last_inner_metrics)
        last_lengths = list(self.agent.last_inner_rollout_lengths)
        try:
            return super()._evaluate_policy(step, initial_obs=initial_obs)
        finally:
            engine.load_training_state_dict(inner_state)
            self.agent.last_inner_metrics = last_metrics
            self.agent.last_inner_rollout_lengths = last_lengths
            self.agent._resume_boundary_prepared = boundary_prepared

    def _select_outer_policy_episode(self, episode_start_step):
        """Draw the episode intervention from a stateless, namespaced coin."""
        probability = float(self.cfg.outer_policy_episode_probability)
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        coin = _outer_policy_hash(
            self.cfg.seed,
            episode_start_step,
            "episode-coin",
        ) / float(1 << 64)
        return coin < probability

    def _make_outer_policy_action_generator(self, episode_start_step):
        device = torch.device(self.agent.device)
        generator_device = device if device.type == "cuda" else torch.device("cpu")
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(
            _outer_policy_hash(
                self.cfg.seed,
                episode_start_step,
                "action-stream",
            )
            % (2**63 - 1)
        )
        return generator

    def _run_training_episode(self, obs, total_timesteps, *, eval_pending):
        """Fix one behavior source for an entire eligible training episode."""
        if (
            self._outer_policy_episode_eligible
            or self._outer_policy_episode_selected
            or self._outer_policy_action_generator is not None
        ):
            raise RuntimeError("An outer-policy episode intervention is already active.")

        try:
            episode_start_step = int(self._global_step)
            eligible = (
                episode_start_step > int(self.cfg.seed_steps)
                and int(self.buffer.num_eps) > 0
            )
            selected = bool(
                eligible and self._select_outer_policy_episode(episode_start_step)
            )
            self._outer_policy_episode_eligible = bool(eligible)
            self._outer_policy_episode_selected = selected
            if selected:
                self._outer_policy_action_generator = (
                    self._make_outer_policy_action_generator(episode_start_step)
                )
            return super()._run_training_episode(
                obs,
                total_timesteps,
                eval_pending=eval_pending,
            )
        finally:
            self._pending_behavior_policy = None
            self._outer_policy_action_generator = None
            self._outer_policy_episode_selected = False
            self._outer_policy_episode_eligible = False

    def _act_agent(self, obs_t, *, t0, eval_mode):
        if self._outer_policy_episode_selected and not eval_mode:
            generator = self._outer_policy_action_generator
            if generator is None:
                raise RuntimeError(
                    "The selected outer-policy episode lacks its action generator."
                )
            if self._stores_behavior_policy:
                if self._pending_behavior_policy is not None:
                    raise RuntimeError(
                        "Behavior-policy metadata from the previous action was "
                        "not consumed."
                    )
                action, behavior_policy = self.agent.act_outer_policy(
                    obs_t,
                    generator=generator,
                    return_behavior_policy=True,
                )
                self._pending_behavior_policy = behavior_policy
                return action
            return self.agent.act_outer_policy(obs_t, generator=generator)

        # W&B records the resulting environment step, while action selection
        # happens immediately before that step.
        sampling_step = int(self._global_step) + 1
        collect_diagnostics = (
            sampling_step % int(self.cfg.inner_diagnostics_every) == 0
        )
        if self._stores_behavior_policy and not eval_mode:
            if self._pending_behavior_policy is not None:
                raise RuntimeError(
                    "Behavior-policy metadata from the previous action was not "
                    "consumed."
                )
            action, behavior_policy = self.agent.act(
                obs_t,
                t0=t0,
                eval_mode=eval_mode,
                collect_diagnostics=collect_diagnostics,
                return_behavior_policy=True,
                apply_inner_writeback=not eval_mode,
            )
            self._pending_behavior_policy = behavior_policy
        else:
            action = self.agent.act(
                obs_t,
                t0=t0,
                eval_mode=eval_mode,
                collect_diagnostics=collect_diagnostics,
                apply_inner_writeback=not eval_mode,
            )
        # The engine's action index is useful to direct callers, but training
        # telemetry must use the real environment step (including seed/random
        # actions) that researchers see on the W&B x-axis.
        if collect_diagnostics and "inner_diagnostics_step" in self.agent.last_inner_metrics:
            self.agent.last_inner_metrics["inner_diagnostics_step"] = float(
                sampling_step
            )
        return action

    def predict(
        self,
        observation,
        deterministic=True,
        episode_start=None,
        *,
        collect_diagnostics=True,
    ):
        """Select a direct-call action with full diagnostics by default.

        Training uses ``inner_diagnostics_every`` through ``_act_agent``;
        callers of the public prediction API keep the complete historical
        metric contract unless they explicitly opt out.
        """
        t0 = self._predict_t0 if episode_start is None else bool(episode_start)
        if t0 and hasattr(self.agent, "reset"):
            self.agent.reset()
        obs_t = self._obs_to_tensor(observation)
        action_norm = self.agent.act(
            obs_t,
            t0=t0,
            eval_mode=deterministic,
            collect_diagnostics=collect_diagnostics,
        ).numpy()
        self._predict_t0 = False
        return self._unscale_action(action_norm), None

    def _wandb_run_name(self):
        explicit = (self.custom_params or {}).get("wandb_run_name")
        if explicit is not None:
            return explicit
        run_name = self.run_params.get("name")
        if not run_name:
            run_name = self.run_params.get("env", "env")
        return f"AMBITDMPC2-{run_name}-seed{self.cfg.seed}"

    def _episode_payload_extras(self):
        return {
            "rollout/outer_policy_episode": int(
                self._outer_policy_episode_selected
            ),
            "rollout/outer_policy_episode_eligible": int(
                self._outer_policy_episode_eligible
            ),
        }

    def _log_step(self, reward, obs, action, terminated, truncated, info):
        """Use the existing logger while exposing imagined-step accounting."""
        if not self.alg_logger:
            return

        done = bool(terminated or truncated)
        info_for_log = dict(info or {})
        info_for_log.setdefault("terminated", bool(terminated))
        info_for_log.setdefault("truncated", bool(truncated))
        accepts_native_payload = bool(
            getattr(self.alg_logger, "accepts_native_step_payload", False)
        )
        if (
            not accepts_native_payload
            or getattr(self.alg_logger, "retains_trajectories", True)
        ):
            obs_for_log = obs if isinstance(obs, dict) else np.asarray(obs)[None, ...]
            action_for_log = np.asarray(action)[None, ...]
        else:
            obs_for_log = None
            action_for_log = None
        data = setup_logs(
            reward,
            obs_for_log,
            action_for_log,
            [done],
            [info_for_log],
            # AMBITrainingLogger stores one rollout-length list per real step.
            inner_steps=[self.agent.last_inner_rollout_lengths],
            materialize=not accepts_native_payload,
        )
        self.alg_logger.on_step(data)

    def _reset_wandb_window(self):
        super()._reset_wandb_window()
        self._wandb_inner_seconds = 0.0
        self._wandb_inner_actions = 0
        self._wandb_inner_steps = 0
        self._wandb_outer_policy_seconds = 0.0
        self._wandb_outer_policy_actions = 0

    def _training_resume_algorithm_state(self):
        if (
            self._wandb_inner_seconds != 0.0
            or self._wandb_inner_actions != 0
            or self._wandb_inner_steps != 0
            or self._wandb_outer_policy_seconds != 0.0
            or self._wandb_outer_policy_actions != 0
        ):
            raise RuntimeError("AMBI W&B timing counters were not flushed.")
        if (
            self._outer_policy_episode_eligible
            or self._outer_policy_episode_selected
            or self._outer_policy_action_generator is not None
        ):
            raise RuntimeError(
                "AMBI training state requires an outer-policy episode boundary."
            )
        if self._pending_behavior_policy is not None:
            raise RuntimeError(
                "AMBI training state requires consumed behavior-policy metadata."
            )
        return {
            "schema": "ambi-wrapper-training-state",
            "version": 2,
            "inner_steps_total": int(self._inner_steps_total),
            "inner_updates_total": int(self._inner_updates_total),
        }

    def _preflight_training_resume_algorithm_state(self, state):
        expected = {
            "schema",
            "version",
            "inner_steps_total",
            "inner_updates_total",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("AMBI wrapper training-state fields are invalid.")
        if state["schema"] != "ambi-wrapper-training-state" or state["version"] != 2:
            raise ValueError("Unsupported AMBI wrapper training-state version.")
        normalized = dict(state)
        for key in ("inner_steps_total", "inner_updates_total"):
            value = normalized[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"AMBI wrapper {key} must be a non-negative integer.")
        return normalized

    def _load_training_resume_algorithm_state(self, state):
        state = self._preflight_training_resume_algorithm_state(state)
        self._inner_steps_total = state["inner_steps_total"]
        self._inner_updates_total = state["inner_updates_total"]
        self._wandb_inner_seconds = 0.0
        self._wandb_inner_actions = 0
        self._wandb_inner_steps = 0
        self._wandb_outer_policy_seconds = 0.0
        self._wandb_outer_policy_actions = 0
        self._outer_policy_episode_eligible = False
        self._outer_policy_episode_selected = False
        self._outer_policy_action_generator = None
        self._pending_behavior_policy = None

    def _record_action_metrics(self, *, planned, action_seconds):
        outer_policy_action = bool(
            planned and getattr(self, "_outer_policy_episode_selected", False)
        )
        if planned:
            self._wandb_train_window.add_weighted(
                "train/outer_policy_action_fraction",
                float(outer_policy_action),
            )
        self._wandb_train_window.update_sums({
            "train/outer_policy_actions": int(outer_policy_action),
            "train/inner_behavior_actions": int(planned and not outer_policy_action),
        })
        if outer_policy_action:
            self._wandb_outer_policy_seconds += float(action_seconds)
            self._wandb_outer_policy_actions += 1

        # AMBI replaces MPPI, so planned inner-action selection time is
        # adaptation time. Seed/random and direct-outer actions both report no
        # inner work, including when stale diagnostics existed previously.
        if not planned or outer_policy_action:
            self._wandb_train_window.add_weighted("train/inner_active", 0.0)
            self._wandb_train_window.update_sums({
                "train/inner_actions": 0,
                "train/inner_rollouts": 0,
                "train/inner_requested_rollouts": 0,
                "train/inner_steps": 0,
                "train/inner_updates": 0,
                "train/inner_model_steps_budget": 0,
                "train/inner_nominal_model_steps": 0,
                "train/inner_realized_model_steps": 0,
                "train/inner_model_steps": 0,
                "train/inner_total_model_steps": 0,
                "train/inner_update_slots": 0,
                "train/inner_requested_update_slots": 0,
                "train/inner_critic_optimizer_steps": 0,
                "train/inner_actor_optimizer_steps": 0,
                "train/inner_temperature_optimizer_steps": 0,
                "train/inner_target_updates": 0,
                "train/inner_critic_target_updates": 0,
                "train/inner_actor_target_updates": 0,
                "train/inner_policy_evaluations": 0,
                "train/inner_q_evaluations": 0,
                "train/inner_replay_draws": 0,
                "train/inner_diagnostic_model_steps": 0,
                "train/inner_primary_rollouts": 0,
                "train/inner_explorer_rollouts": 0,
                "train/inner_primary_transitions": 0,
                "train/inner_explorer_transitions": 0,
                "train/inner_optimization_model_steps": 0,
                "train/inner_selector_model_steps": 0,
                "train/inner_explorer_actor_optimizer_steps": 0,
                "train/inner_explorer_critic_optimizer_steps": 0,
                "train/inner_explorer_temperature_optimizer_steps": 0,
                "train/inner_explorer_target_updates": 0,
                "train/inner_explorer_critic_target_updates": 0,
                "train/inner_selector_primary_wins": 0,
                "train/inner_selector_explorer_wins": 0,
            })
            return

        metrics = self._metrics_to_floats(dict(self.agent.last_inner_metrics or {}))
        self._wandb_train_window.add_weighted(
            "train/inner_active",
            metrics.get("inner_active", 1.0),
        )
        rollout_count = int(metrics.get(
            "inner_rollouts",
            metrics.get("inner_rollout_count", len(self.agent.last_inner_rollout_lengths)),
        ))
        inner_steps = int(metrics.get("inner_steps", sum(self.agent.last_inner_rollout_lengths)))
        inner_updates = int(metrics.get("inner_updates", 0))

        self._wandb_train_window.update_sums({
            "train/inner_actions": 1,
            "train/inner_rollouts": rollout_count,
            "train/inner_steps": inner_steps,
            "train/inner_updates": inner_updates,
        })
        self._inner_steps_total += inner_steps
        self._inner_updates_total += inner_updates
        self._wandb_inner_seconds += float(action_seconds)
        self._wandb_inner_actions += 1
        self._wandb_inner_steps += inner_steps

        for source, target in (
            ("inner_return", "train/inner_return"),
            ("inner_rollout_len", "train/inner_rollout_len"),
            (
                "inner_behavior_reward_sum",
                "train/inner_behavior_reward_sum",
            ),
            (
                "inner_behavior_discounted_reward",
                "train/inner_behavior_discounted_reward",
            ),
        ):
            mean = metrics.get(f"{source}_mean")
            if rollout_count > 0 and mean is not None:
                self._wandb_train_window.add_stats(
                    target,
                    count=rollout_count,
                    mean=mean,
                    std=metrics.get(f"{source}_std", 0.0),
                    min_value=metrics.get(f"{source}_min", mean),
                    max_value=metrics.get(f"{source}_max", mean),
                )

        # Explicit aggregation registry: work is summed, update diagnostics are
        # weighted by their own optimizer, and action-level gauges are averaged
        # by planned actions. New metrics must be deliberately classified here.
        counter_metrics = {
            "inner_model_steps_budget",
            "inner_nominal_model_steps",
            "inner_realized_model_steps",
            "inner_requested_rollouts",
            "inner_model_steps",
            "inner_total_model_steps",
            "inner_update_slots",
            "inner_requested_update_slots",
            "inner_critic_optimizer_steps",
            "inner_actor_optimizer_steps",
            "inner_temperature_optimizer_steps",
            "inner_actor_writeback_applied",
            "inner_critic_writeback_applied",
            "inner_target_updates",
            "inner_critic_target_updates",
            "inner_actor_target_updates",
            "inner_policy_evaluations",
            "inner_q_evaluations",
            "inner_replay_draws",
            "inner_primary_replay_samples",
            "inner_explorer_replay_samples",
            "inner_diagnostic_model_steps",
            "inner_diagnostics_sample_count",
            "inner_primary_rollouts",
            "inner_explorer_rollouts",
            "inner_primary_transitions",
            "inner_explorer_transitions",
            "inner_optimization_model_steps",
            "inner_selector_model_steps",
            "inner_explorer_actor_optimizer_steps",
            "inner_explorer_critic_optimizer_steps",
            "inner_explorer_temperature_optimizer_steps",
            "inner_explorer_target_updates",
            "inner_explorer_critic_target_updates",
            "inner_selector_primary_wins",
            "inner_selector_explorer_wins",
            "inner_fixed_q_counterfactual_primary_wins",
            "inner_fixed_q_counterfactual_explorer_wins",
            "inner_fixed_q_counterfactual_policy_evaluations",
            "inner_fixed_q_counterfactual_q_evaluations",
            "inner_primary_td_error_abs_count",
            "inner_explorer_td_error_abs_count",
            "inner_explorer_critic_primary_td_error_abs_count",
            "inner_explorer_critic_explorer_td_error_abs_count",
            "inner_param_noise_calibration_probes",
            "inner_param_noise_calibration_policy_evaluations",
            "inner_param_noise_calibration_rounds",
            "inner_param_noise_mean_action_saturation_count",
            "planner_policy_model_steps",
            "planner_candidate_model_steps",
            "planner_model_steps",
            "planner_policy_evaluations",
            "planner_q_evaluations",
        }
        for key in counter_metrics:
            if key in metrics:
                self._wandb_train_window.add_sum(f"train/{key}", metrics[key])

        timing_metrics = {
            "inner_setup_seconds",
            "inner_rollout_seconds",
            "inner_update_seconds",
            "inner_execution_seconds",
            "inner_diagnostic_seconds",
            "inner_mppi_seconds",
            "inner_selector_seconds",
            "inner_param_noise_calibration_seconds",
        }
        for key in timing_metrics:
            if key in metrics:
                self._wandb_train_window.add_sum(f"time/{key}", metrics[key])

        termination_rate = metrics.get(
            "inner_termination_rate",
            metrics.get("inner_rollout_termination_rate"),
        )
        if termination_rate is not None and rollout_count > 0:
            self._wandb_train_window.add_weighted(
                "train/inner_termination_rate",
                termination_rate,
                weight=rollout_count,
            )
            self._wandb_train_window.add_stats(
                "train/inner_termination_rate",
                count=rollout_count,
                mean=termination_rate,
                std=metrics.get("inner_termination_rate_std", 0.0),
                min_value=metrics.get(
                    "inner_termination_rate_min", termination_rate
                ),
                max_value=metrics.get(
                    "inner_termination_rate_max", termination_rate
                ),
            )

        critic_metrics = {
            "inner_critic_loss",
            "inner_critic_grad_norm",
            "inner_q_mean",
            "inner_q_abs_mean",
            "inner_q_target_mean",
            "inner_q_target_clip_fraction",
            "inner_td_error_abs_mean",
        }
        explorer_critic_metrics = {
            "inner_explorer_critic_loss",
            "inner_explorer_critic_grad_norm",
            "inner_explorer_q_mean",
            "inner_explorer_q_abs_mean",
            "inner_explorer_q_target_mean",
            "inner_explorer_q_target_clip_fraction",
            "inner_explorer_critic_td_error_abs_mean",
        }
        actor_metrics = {
            "inner_actor_loss",
            "inner_actor_grad_norm",
            "inner_actor_q_mean",
            "inner_actor_q_mean_all",
            "inner_actor_q_min_all",
            "inner_actor_q_mean_all_minus_min_all",
            "inner_actor_entropy",
            "inner_actor_pre_tanh_abs_mean",
            "inner_actor_pre_tanh_abs_max",
            "inner_actor_pre_tanh_abs_ge_7p6_fraction",
            "inner_actor_action_exact_saturation_fraction",
            "inner_mixture_log_prob",
            "inner_outer_policy_kl",
            "inner_outer_action_l2",
        }
        explorer_actor_metrics = {
            "inner_explorer_actor_loss",
            "inner_explorer_actor_grad_norm",
            "inner_explorer_actor_q_mean",
            "inner_explorer_actor_entropy",
            "inner_explorer_actor_pre_tanh_abs_mean",
            "inner_explorer_actor_pre_tanh_abs_max",
            "inner_explorer_actor_pre_tanh_abs_ge_7p6_fraction",
            "inner_explorer_actor_action_exact_saturation_fraction",
        }
        temperature_metrics = {
            "inner_temperature_loss",
            "inner_temperature_grad_norm",
        }
        explorer_temperature_metrics = {
            "inner_explorer_temperature_loss",
            "inner_explorer_temperature_grad_norm",
        }
        action_gauges = {
            "inner_alpha",
            "inner_alpha_to_abs_q",
            "inner_actor_loss_scale",
            "inner_effective_alpha",
            "inner_buffer_size",
            "inner_buffer_capacity",
            "inner_buffer_fill_ratio",
            "inner_replay_unique_fraction",
            "inner_actor_trainable_params",
            "inner_critic_trainable_params",
            "inner_temperature_trainable_params",
            "inner_explorer_actor_trainable_params",
            "inner_explorer_critic_trainable_params",
            "inner_explorer_temperature_trainable_params",
            "inner_explorer_alpha",
            "inner_explorer_alpha_initial",
            "inner_explorer_alpha_final",
            "inner_explorer_alpha_delta",
            "inner_rounds",
            "inner_mppi_iterations",
            "inner_rollouts_per_round",
            "inner_rollout_horizon",
            "inner_horizon_ratio",
            "inner_nominal_transitions_per_round",
            "inner_nominal_updates_per_round",
            "inner_updates_per_round_realized",
            "inner_nominal_critic_utd",
            "inner_critic_utd",
            "inner_actor_utd",
            "inner_temperature_utd",
            "inner_explorer_actor_utd",
            "inner_explorer_critic_utd",
            "inner_explorer_temperature_utd",
            "inner_primary_rollout_fraction",
            "inner_explorer_rollout_fraction",
            "inner_primary_replay_fraction",
            "inner_explorer_replay_fraction",
            "inner_selector_primary_score",
            "inner_selector_explorer_score",
            "inner_selector_score_margin",
            "inner_selector_score_variance",
            "inner_primary_explorer_action_l2",
            "inner_fixed_q_counterfactual_primary_q",
            "inner_fixed_q_counterfactual_explorer_q",
            "inner_fixed_q_counterfactual_margin",
            "inner_fixed_q_counterfactual_explorer_rate",
            "inner_fixed_q_counterfactual_execution_agreement",
            "inner_fixed_q_counterfactual_action_l2_to_executed",
            "inner_param_noise_actor_count",
            "inner_param_noise_rollouts_per_actor",
            "inner_param_noise_target_action_rms",
            "inner_param_noise_sigma_initial",
            "inner_param_noise_sigma_final",
            "inner_param_noise_sigma_mean",
            "inner_param_noise_sigma_min",
            "inner_param_noise_sigma_max",
            "inner_alpha_initial",
            "inner_alpha_final",
            "inner_alpha_delta",
            "inner_target_entropy",
            "inner_actor_writeback_coef",
            "inner_critic_writeback_coef",
            "inner_policy_mean_delta_l2",
            "inner_final_outer_policy_kl",
            "inner_proposal_mean_delta_l2",
            "inner_fixed_target_q_action_gain",
            "inner_outer_q_gain",
            "inner_fixed_target_q_outer",
            "inner_fixed_target_q_improved",
            "inner_fixed_target_q_abs_mean",
            "inner_fixed_evaluator_alpha",
            "inner_diagnostics_sampled",
            "inner_distributional_q_entropy",
            "inner_distributional_q_edge_mass",
            "inner_compile_rollout_fallback",
            "inner_compile_critic_fallback",
            "inner_compile_actor_fallback",
            "inner_compile_fallback",
            "inner_predicted_j_outer",
            "inner_predicted_j_improved",
            "inner_predicted_j_gain",
            "inner_predicted_soft_j_outer",
            "inner_predicted_soft_j_improved",
            "inner_predicted_soft_j_gain",
            "inner_fixed_alpha_soft_j_outer",
            "inner_fixed_alpha_soft_j_improved",
            "inner_fixed_alpha_soft_j_gain",
            "planner_value_mean",
            "planner_value_std",
            "planner_value_max",
            "planner_elite_value_mean",
            "planner_elite_value_std",
            "planner_elite_value_max",
            "planner_std_mean",
            "planner_std_min",
            "planner_std_max",
            "planner_action_l2",
            "planner_num_samples",
            "planner_num_elites",
            "planner_num_pi_trajs",
            "planner_iterations",
        }
        action_gauges.update(
            key
            for key in metrics
            if key.startswith("inner_fixed_q_counterfactual_action_")
            and key.removeprefix("inner_fixed_q_counterfactual_action_").isdigit()
        )

        # Source-conditioned TD summaries have an exact sampled-row count.
        # Route them separately from ordinary per-optimizer diagnostics so a
        # minibatch that contains one source sparsely (or not at all) cannot
        # bias the interval mean.
        source_td_metrics = {
            "inner_primary_td_error_abs_mean": (
                "inner_primary_td_error_abs_count"
            ),
            "inner_explorer_td_error_abs_mean": (
                "inner_explorer_td_error_abs_count"
            ),
            "inner_explorer_critic_primary_td_error_abs_mean": (
                "inner_explorer_critic_primary_td_error_abs_count"
            ),
            "inner_explorer_critic_explorer_td_error_abs_mean": (
                "inner_explorer_critic_explorer_td_error_abs_count"
            ),
        }
        for key, count_key in source_td_metrics.items():
            count = int(metrics.get(count_key, 0))
            if key in metrics and count > 0:
                self._wandb_train_window.add_weighted(
                    f"train/{key}", metrics[key], weight=count
                )

        replay_sample_total = int(
            metrics.get("inner_primary_replay_samples", 0)
        ) + int(
            metrics.get("inner_explorer_replay_samples", 0)
        )
        if replay_sample_total > 0:
            for source in ("primary", "explorer"):
                key = f"inner_{source}_replay_sample_fraction"
                if key in metrics:
                    self._wandb_train_window.add_weighted(
                        f"train/{key}",
                        metrics[key],
                        weight=replay_sample_total,
                    )
        for source in ("primary", "explorer"):
            # These source-specific summaries are computed over individual
            # imagined transitions (reward and terminal indicator), not over
            # per-rollout aggregates.  Episodic compaction can give the two
            # sources different realized lengths, so transition counts are the
            # only correct aggregation weights.
            source_count = int(metrics.get(f"inner_{source}_transitions", 0))
            if source_count <= 0:
                continue
            for stem in ("reward", "termination_rate"):
                mean_key = f"inner_{source}_{stem}_mean"
                if mean_key not in metrics:
                    continue
                target = f"train/inner_{source}_{stem}"
                mean = metrics[mean_key]
                self._wandb_train_window.add_weighted(
                    target,
                    mean,
                    weight=source_count,
                )
                self._wandb_train_window.add_stats(
                    target,
                    count=source_count,
                    mean=mean,
                    std=metrics.get(f"inner_{source}_{stem}_std", 0.0),
                    min_value=metrics.get(f"inner_{source}_{stem}_min", mean),
                    max_value=metrics.get(f"inner_{source}_{stem}_max", mean),
                )

        for stem, count_key in (
            (
                "inner_param_noise_calibration_action_rms",
                "inner_param_noise_calibration_action_rms_count",
            ),
            (
                "inner_param_noise_behavior_action_rms",
                "inner_param_noise_behavior_action_rms_count",
            ),
        ):
            count = int(metrics.get(count_key, 0))
            mean_key = f"{stem}_mean"
            if count <= 0 or mean_key not in metrics:
                continue
            mean = metrics[mean_key]
            self._wandb_train_window.add_stats(
                f"train/{stem}",
                count=count,
                mean=mean,
                std=metrics.get(f"{stem}_std", 0.0),
                min_value=metrics.get(f"{stem}_min", mean),
                max_value=metrics.get(f"{stem}_max", mean),
            )

        for key, count_key in (
            (
                "inner_param_noise_calibration_target_hit_fraction",
                "inner_param_noise_calibration_rounds",
            ),
            (
                "inner_param_noise_sigma_bound_hit_fraction",
                "inner_param_noise_calibration_probes",
            ),
            (
                "inner_param_noise_mean_action_saturation_fraction",
                "inner_param_noise_mean_action_saturation_count",
            ),
        ):
            count = int(metrics.get(count_key, 0))
            if count > 0 and key in metrics:
                self._wandb_train_window.add_weighted(
                    f"train/{key}", metrics[key], weight=count
                )
        weighted_groups = (
            (critic_metrics, max(1, int(metrics.get("inner_critic_optimizer_steps", 0)))),
            (
                explorer_critic_metrics,
                max(
                    1,
                    int(metrics.get("inner_explorer_critic_optimizer_steps", 0)),
                ),
            ),
            (actor_metrics, max(1, int(metrics.get("inner_actor_optimizer_steps", 0)))),
            (
                explorer_actor_metrics,
                max(
                    1,
                    int(metrics.get("inner_explorer_actor_optimizer_steps", 0)),
                ),
            ),
            (
                temperature_metrics,
                max(1, int(metrics.get("inner_temperature_optimizer_steps", 0))),
            ),
            (
                explorer_temperature_metrics,
                max(
                    1,
                    int(
                        metrics.get(
                            "inner_explorer_temperature_optimizer_steps", 0
                        )
                    ),
                ),
            ),
            (action_gauges, 1),
        )
        for keys, weight in weighted_groups:
            for key in keys:
                if key in metrics:
                    self._wandb_train_window.add_weighted(
                        f"train/{key}", metrics[key], weight=weight
                    )
                    self._wandb_train_window.add_stats(
                        f"train/{key}",
                        count=weight,
                        mean=metrics[key],
                        std=metrics.get(f"{key}_std", 0.0),
                        min_value=metrics.get(f"{key}_min", metrics[key]),
                        max_value=metrics.get(f"{key}_max", metrics[key]),
                    )

        # A sampling step is an event marker, not a continuous gauge.  Keep
        # the most recent sampled step so pooled windows never emit an average
        # step that did not actually occur.
        if "inner_diagnostics_step" in metrics:
            self._wandb_train_window.set_last(
                "train/inner_diagnostics_step", metrics["inner_diagnostics_step"]
            )

    def _timing_wandb_payload(self, updates_since_log):
        outer_update_seconds = float(self._wandb_train_seconds)
        inner_seconds = float(self._wandb_inner_seconds)
        inner_actions = int(self._wandb_inner_actions)
        inner_steps = int(self._wandb_inner_steps)
        outer_policy_seconds = float(self._wandb_outer_policy_seconds)
        outer_policy_actions = int(self._wandb_outer_policy_actions)
        payload = super()._timing_wandb_payload(updates_since_log)
        payload.update({
            "time/outer_update_seconds": outer_update_seconds,
            "time/inner_action_seconds": inner_seconds,
            "time/inner_seconds_per_action": (
                float(inner_seconds / inner_actions) if inner_actions > 0 else 0.0
            ),
            "time/inner_steps_per_second": (
                float(inner_steps / inner_seconds) if inner_seconds > 0 else 0.0
            ),
            "time/outer_policy_action_seconds": outer_policy_seconds,
            "time/outer_policy_seconds_per_action": (
                float(outer_policy_seconds / outer_policy_actions)
                if outer_policy_actions > 0
                else 0.0
            ),
        })
        self._wandb_inner_seconds = 0.0
        self._wandb_inner_actions = 0
        self._wandb_inner_steps = 0
        self._wandb_outer_policy_seconds = 0.0
        self._wandb_outer_policy_actions = 0
        return payload

    def _timing_wandb_metric_keys(self):
        return super()._timing_wandb_metric_keys() | {
            "time/outer_update_seconds",
            "time/inner_action_seconds",
            "time/inner_seconds_per_action",
            "time/inner_steps_per_second",
            "time/outer_policy_action_seconds",
            "time/outer_policy_seconds_per_action",
        }

    def _extra_wandb_payload(self, updates_since_log):
        del updates_since_log
        return {
            "train/inner_steps_total": int(self._inner_steps_total),
            "train/inner_updates_total": int(self._inner_updates_total),
        }
