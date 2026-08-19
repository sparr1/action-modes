"""AMBI with a TD-MPC2 world model and configurable inner improvement."""

import copy
import math
import warnings
from collections.abc import Mapping

import numpy as np

from RL.TDMPC2 import TDMPC2Baseline, _normalize_horizon_params
from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from utils.utils import setup_logs


_Q_REDUCTIONS = {"min_pair", "mean_pair", "min_all", "mean_all"}
_SAC_ACTOR_LOSS_SCALE_MODES = {"none", "tdmpc2_percentile_range"}
_ADAPTATION_MODES = {"frozen", "clone", "lora"}
_LIFECYCLE_SCOPES = {"action", "episode", "run"}
_SCOPE_RANK = {"action": 0, "episode": 1, "run": 2}


_AMBI_DEFAULTS = {
    # Outer latent SAC. Q representation and ensemble reduction are independent
    # of the soft Bellman objective.
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
    "sac_actor_loss_scale_mode": "none",
    "sac_actor_loss_scale_tau": 0.01,
    "critic_coef": 1.0,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "adam_eps": 1e-8,
    "actor_adam_eps": None,
    "log_std_min": -20,
    "log_std_max": 2,
    "ent_coef": "auto",
    "ent_coef_lr": 3e-4,
    "target_entropy": "auto",
    "tau": 0.005,
    "target_update_interval": 1,
    "compile_strict": False,

    # Canonical root-local training schedule. Each round collects N rollouts of
    # length H, appends them to the root-local replay, then runs G joint update
    # slots. ``auto`` keeps critic UTD at one by matching the transitions that
    # were actually generated in that round.
    "inner_operator": "sac",
    "inner_rounds": 4,
    "inner_rollouts_per_round": 64,
    "inner_rollout_horizon": 3,
    "inner_updates_per_round": "auto",

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

    # Exploration during imagined collection is independent of the entropy
    # objective and of noise on the action returned to the real environment.
    "inner_behavior_action": "policy_sample",
    "inner_behavior_std_scale": 1.0,
    "inner_behavior_noise_std": 0.0,
    "inner_execution_action": "policy_sample",
    "inner_execution_std_scale": 1.0,
    "inner_execution_noise_std": 0.0,
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

    # AMBI MPPI controls. Candidate count is derived from the common model-step
    # budget, including the one-time policy-prior trajectory cost.
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


def _finite_float(value, key):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite, got {value}.")
    return value


def _normalize_legacy_params(params):
    """Normalize schedule aliases and identify canonical versus total scheduling."""
    params = copy.deepcopy(params)

    requested_operator = str(
        params.get("inner_operator", _AMBI_DEFAULTS["inner_operator"])
    ).lower()
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
        new_unique = sorted(
            set(params) & {"inner_rollouts_per_round", "inner_updates_per_round"}
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
            set(params) & {"inner_rollouts_per_round", "inner_updates_per_round"}
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
    """TD-MPC2 representation learning with a selectable AMBI inner operator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inner_steps_total = 0
        self._inner_updates_total = 0
        self._wandb_inner_seconds = 0.0
        self._wandb_inner_actions = 0
        self._wandb_inner_steps = 0

    def _build_cfg(self, params):
        # Resolve the outer legacy alias before translating AMBI's separate
        # legacy inner-loop aliases. This rejects only combinations the caller
        # actually supplied while allowing an all-legacy configuration to make
        # its one-release migration cleanly.
        params = _normalize_horizon_params(params, resolve_defaults=False)
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
                "for the matched AMBI planner."
            )

        merged = dict(_AMBI_DEFAULTS)
        if schedule_mode == "legacy" and requested_operator in {"sac", "td3"}:
            merged.update(_LEGACY_SCHEDULE_DEFAULTS)
        merged.update(params)
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

                cfg.inner_model_step_budget = (
                    cfg.inner_rounds
                    * cfg.inner_rollouts_per_round
                    * cfg.inner_rollout_horizon
                )
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
        cfg.log_std_min = _finite_float(cfg.log_std_min, "log_std_min")
        cfg.log_std_max = _finite_float(cfg.log_std_max, "log_std_max")
        if cfg.log_std_min >= cfg.log_std_max:
            raise ValueError("log_std_min must be less than log_std_max.")
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
        if cfg.inner_schedule_mode == "canonical":
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

    def _act_agent(self, obs_t, *, t0, eval_mode):
        # W&B records the resulting environment step, while action selection
        # happens immediately before that step.
        sampling_step = int(self._global_step) + 1
        collect_diagnostics = (
            sampling_step % int(self.cfg.inner_diagnostics_every) == 0
        )
        action = self.agent.act(
            obs_t,
            t0=t0,
            eval_mode=eval_mode,
            collect_diagnostics=collect_diagnostics,
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
        return f"AMBITDMPC2-{self.run_params.get('env', 'env')}-seed{self.cfg.seed}"

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

    def _training_resume_algorithm_state(self):
        if (
            self._wandb_inner_seconds != 0.0
            or self._wandb_inner_actions != 0
            or self._wandb_inner_steps != 0
        ):
            raise RuntimeError("AMBI W&B timing counters were not flushed.")
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

    def _record_action_metrics(self, *, planned, action_seconds):
        # AMBI replaces MPPI, so action selection time is inner-adaptation time.
        if not planned:
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
            "inner_target_updates",
            "inner_critic_target_updates",
            "inner_actor_target_updates",
            "inner_policy_evaluations",
            "inner_q_evaluations",
            "inner_replay_draws",
            "inner_diagnostic_model_steps",
            "inner_diagnostics_sample_count",
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
        actor_metrics = {
            "inner_actor_loss",
            "inner_actor_grad_norm",
            "inner_actor_q_mean",
            "inner_actor_entropy",
            "inner_outer_policy_kl",
            "inner_outer_action_l2",
        }
        temperature_metrics = {
            "inner_temperature_loss",
            "inner_temperature_grad_norm",
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
            "inner_alpha_initial",
            "inner_alpha_final",
            "inner_alpha_delta",
            "inner_target_entropy",
            "inner_policy_mean_delta_l2",
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
        weighted_groups = (
            (critic_metrics, max(1, int(metrics.get("inner_critic_optimizer_steps", 0)))),
            (actor_metrics, max(1, int(metrics.get("inner_actor_optimizer_steps", 0)))),
            (
                temperature_metrics,
                max(1, int(metrics.get("inner_temperature_optimizer_steps", 0))),
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
        })
        self._wandb_inner_seconds = 0.0
        self._wandb_inner_actions = 0
        self._wandb_inner_steps = 0
        return payload

    def _extra_wandb_payload(self, updates_since_log):
        del updates_since_log
        return {
            "train/inner_steps_total": int(self._inner_steps_total),
            "train/inner_updates_total": int(self._inner_updates_total),
        }
