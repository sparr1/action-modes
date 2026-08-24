"""AMBI with persistent XQC priors and fresh per-action inner XQC.

The TOLD encoder, latent dynamics, reward model, termination model, and
recurrent BPTT training remain intact.  XQC supplies the persistent actor and
twin distributional critics, which are copied into a fresh root-local learner
at every real decision.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.TDMPC2 import TDMPC2Baseline
from RL.alg import validate_timestep_budget
from RL.tdmpc2_core.ambixqc_agent import AMBIXQCAgent
from RL.xqc_core import OFFICIAL_XQC_COMMIT


_AMBIXQC_DEFAULTS = {
    # TOLD remains a one-update-per-interaction recurrent learner.  XQC's
    # controller slot is part of that update; UTD=2 must not double BPTT.
    "mpc": False,
    "discount": 0.99,
    "utd": 1,
    "compile": False,
    "compile_strict": False,

    # Released XQC architecture and optimizer semantics.  The prefix keeps
    # the critic support distinct from TOLD's reward-model bins/support.
    "xqc_actor_net_arch": (256, 256, 256, 256),
    "xqc_critic_net_arch": (512, 512, 512, 512),
    "xqc_num_atoms": 101,
    "xqc_vmin": -5.0,
    "xqc_vmax": 5.0,
    "xqc_actor_lr": 3e-4,
    "xqc_critic_lr": 3e-4,
    "xqc_lr_end": 3e-5,
    "xqc_tau": 0.005,
    "xqc_target_update_interval": 1,
    "xqc_policy_delay": 3,
    "xqc_init_temperature": 0.01,
    "xqc_target_entropy": "auto",
    "xqc_adam_eps": 1e-8,
    "xqc_optimizer_backend": "auto",
    "xqc_reward_normalization": True,

    # Canonical action-local AMBI compute budget.  The local schedule counter
    # starts at zero on every real decision; actor and temperature optimizers
    # therefore run at local slots 0, 3, 6, ... by default.
    "inner_rounds": 2,
    "inner_rollouts_per_round": 32,
    "inner_rollout_horizon": 3,
    "inner_updates_per_round": 4,
    "inner_batch_size": 64,
    "inner_replay_capacity": None,
    "inner_replay_sampling": "with_replacement",
    "inner_actor_lr": 5e-5,
    "inner_critic_lr": 5e-5,
    "inner_diagnostics_every": 1000,
}

_PUBLIC_INNER_KEYS = {
    "inner_rounds",
    "inner_rollouts_per_round",
    "inner_rollout_horizon",
    "inner_updates_per_round",
    "inner_batch_size",
    "inner_replay_capacity",
    "inner_replay_sampling",
    "inner_actor_lr",
    "inner_critic_lr",
    "inner_diagnostics_every",
}
_PUBLIC_XQC_KEYS = {key for key in _AMBIXQC_DEFAULTS if key.startswith("xqc_")}


# These settings describe other algorithms or make faithful XQC behavior
# ambiguous.  Fail early instead of accepting an inert option.
_INCOMPATIBLE_EXPLICIT_KEYS = {
    # Standalone XQC aliases. AMBI-XQC uses explicit xqc_* names and TOLD's
    # own replay/warmup loop, so accepting these would silently mislabel runs.
    "learning_rate",
    "lr_end",
    "num_interactions",
    "updates_per_step",
    "learning_starts",
    "train_freq",
    "gradient_steps",
    "gamma",
    "reward_normalization",
    "init_temperature",
    "adam_eps",
    "weight_decay",
    "optimizer_backend",
    "debug_checks",
    "verbose",
    "actor_lr",
    "critic_lr",
    "ent_coef",
    "ent_coef_lr",
    "target_entropy",
    "tau",
    "target_update_interval",
    "policy_delay",
    "actor_net_arch",
    "critic_net_arch",
    "num_atoms",
    "q_representation",
    "q_num_bins",
    "q_vmin",
    "q_vmax",
    "q_pair_size",
    "num_q",
    "outer_q_target_reduction",
    "outer_q_actor_reduction",
    "inner_q_target_reduction",
    "inner_q_actor_reduction",
    "mppi_terminal_q_reduction",
    "outer_critic_target",
    "inner_sac_critic_target",
    "sac_actor_loss_scale_mode",
    "sac_actor_loss_scale_tau",
    "critic_coef",
    "log_std_mapping",
    "log_std_min",
    "log_std_max",
    "inner_log_std_mapping",
    "inner_log_std_min",
    "inner_log_std_max",
    "inner_operator",
    "inner_actor_adaptation",
    "inner_critic_adaptation",
    "inner_temperature_mode",
    "inner_temperature_initialization",
    "inner_temperature",
    "inner_temperature_lr",
    "inner_target_entropy",
    "inner_bootstrap_source",
    "inner_actor_scope",
    "inner_critic_scope",
    "inner_temperature_scope",
    "inner_replay_scope",
    "inner_actor_optimizer_scope",
    "inner_critic_optimizer_scope",
    "inner_temperature_optimizer_scope",
    "inner_rebase_persistent",
    "inner_behavior_action",
    "inner_behavior_std_scale",
    "inner_behavior_noise_std",
    "inner_execution_action",
    "inner_execution_std_scale",
    "inner_execution_noise_std",
    "inner_critic_target_tau",
    "inner_critic_target_update_interval",
    "inner_actor_target_tau",
    "inner_actor_target_update_interval",
    "inner_outer_policy_kl_coef",
    "inner_outer_action_l2_coef",
    "temp_lr",
    "alpha_lr",
    "normalize_last_layer",
    # TD-MPC planning/discount aliases are inert because AMBI-XQC acts through
    # the inner learner and uses one explicit Bellman discount.
    "outer_planning_horizon",
    "iterations",
    "num_samples",
    "num_elites",
    "num_pi_trajs",
    "min_std",
    "max_std",
    "temperature",
    "discount_denom",
    "discount_min",
    "discount_max",
    "entropy_coef",
    "dropout",
    # Derived or fixed inner semantics must not be supplied as apparent knobs.
    "inner_model_step_budget",
    "inner_schedule_mode",
    "inner_expected_update_slots",
    "inner_iterations",
    "inner_rollouts",
    "inner_horizon",
    "inner_updates_per_iteration",
    "inner_buffer_size",
    "inner_adaptation",
    "inner_tau",
    "inner_target_update_interval",
    "inner_grad_clip_norm",
    "inner_critic_dropout_enabled",
    "inner_critic_updates_per_action",
    "inner_actor_updates_per_action",
    "inner_temperature_updates_per_action",
    "inner_adam_eps",
    "inner_actor_grad_clip_norm",
    "inner_critic_grad_clip_norm",
    "inner_temperature_grad_clip_norm",
    "inner_diagnostic_rollouts",
    "inner_termination_threshold",
    "allow_long_inner_horizon",
    "horizon",
}


def _positive_int(value, key):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{key} must be a positive integer.")
    try:
        numeric = float(value)
        resolved = int(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{key} must be a positive integer.") from exc
    if not math.isfinite(numeric) or resolved <= 0 or numeric != resolved:
        raise ValueError(f"{key} must be a positive integer.")
    return resolved


def _finite_float(value, key, *, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{key} must be a finite number, not a boolean.")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{key} must be a finite number.") from exc
    if not math.isfinite(value) or (positive and value <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{key} must be {qualifier}.")
    return value


def _architecture(value, key):
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a sequence of positive widths.")
    try:
        value = tuple(_positive_int(width, key) for width in value)
    except TypeError as exc:
        raise ValueError(f"{key} must be a sequence of positive widths.") from exc
    if not value:
        raise ValueError(f"{key} must contain at least one hidden layer.")
    return value


class AMBIXQC(AMBITDMPC2):
    """TOLD with XQC control priors and fresh action-local XQC adaptation."""

    def _build_cfg(self, params):
        params = dict(params)
        if str(params.get("obs", "state")).lower() != "state":
            raise NotImplementedError(
                "The first AMBI-XQC implementation supports state observations only."
            )
        incompatible = sorted(set(params) & _INCOMPATIBLE_EXPLICIT_KEYS)
        incompatible.extend(
            sorted(
                key
                for key in params
                if (key.startswith("inner_") and key not in _PUBLIC_INNER_KEYS)
                or (key.startswith("xqc_") and key not in _PUBLIC_XQC_KEYS)
                or "lora" in key.lower()
            )
        )
        incompatible = sorted(set(incompatible))
        if incompatible:
            raise ValueError(
                "AMBIXQC has fixed XQC/action-local semantics and does not accept "
                f"these AMBI or standalone-XQC options: {incompatible}."
            )
        if "mpc" in params and params["mpc"] is not False:
            raise ValueError("AMBIXQC does not use MPPI; mpc must be false.")
        if "utd" in params and _positive_int(params["utd"], "utd") != 1:
            raise ValueError(
                "AMBIXQC requires utd=1 so controller updates do not duplicate "
                "TOLD's recurrent BPTT update."
            )
        for key in ("compile", "compile_strict"):
            if key in params and not isinstance(params[key], (bool, np.bool_)):
                raise ValueError(f"{key} must be a boolean.")

        merged = dict(_AMBIXQC_DEFAULTS)
        merged.update(params)
        cfg = TDMPC2Baseline._build_cfg(self, merged)
        if cfg.obs != "state":
            raise NotImplementedError(
                "The first AMBI-XQC implementation supports state observations only."
            )
        cfg.discount = _finite_float(cfg.discount, "discount", positive=True)
        if cfg.discount > 1.0:
            raise ValueError("discount must be in (0, 1].")
        cfg.discount_min = cfg.discount
        cfg.discount_max = cfg.discount
        cfg.mpc = False
        cfg.utd = 1
        cfg.compile = bool(cfg.compile)
        cfg.compile_strict = bool(cfg.compile_strict)
        cfg.value_coef = _finite_float(cfg.value_coef, "value_coef", positive=True)

        cfg.xqc_actor_net_arch = _architecture(
            cfg.xqc_actor_net_arch, "xqc_actor_net_arch"
        )
        cfg.xqc_critic_net_arch = _architecture(
            cfg.xqc_critic_net_arch, "xqc_critic_net_arch"
        )
        cfg.xqc_num_atoms = _positive_int(cfg.xqc_num_atoms, "xqc_num_atoms")
        if cfg.xqc_num_atoms < 2:
            raise ValueError("xqc_num_atoms must be at least two.")
        cfg.xqc_vmin = _finite_float(cfg.xqc_vmin, "xqc_vmin")
        cfg.xqc_vmax = _finite_float(cfg.xqc_vmax, "xqc_vmax")
        if cfg.xqc_vmin >= cfg.xqc_vmax:
            raise ValueError("xqc_vmin must be smaller than xqc_vmax.")
        cfg.xqc_atom_delta = (
            cfg.xqc_vmax - cfg.xqc_vmin
        ) / (cfg.xqc_num_atoms - 1)

        for key in (
            "xqc_actor_lr",
            "xqc_critic_lr",
            "xqc_lr_end",
            "xqc_tau",
            "xqc_init_temperature",
            "xqc_adam_eps",
        ):
            setattr(cfg, key, _finite_float(getattr(cfg, key), key, positive=True))
        if cfg.xqc_tau > 1.0:
            raise ValueError("xqc_tau must be in (0, 1].")
        cfg.xqc_target_update_interval = _positive_int(
            cfg.xqc_target_update_interval, "xqc_target_update_interval"
        )
        cfg.xqc_policy_delay = _positive_int(
            cfg.xqc_policy_delay, "xqc_policy_delay"
        )
        if isinstance(cfg.xqc_target_entropy, str):
            cfg.xqc_target_entropy = cfg.xqc_target_entropy.lower()
            if cfg.xqc_target_entropy != "auto":
                raise ValueError("xqc_target_entropy must be 'auto' or a finite number.")
        else:
            cfg.xqc_target_entropy = _finite_float(
                cfg.xqc_target_entropy, "xqc_target_entropy"
            )
        cfg.xqc_resolved_target_entropy = (
            -cfg.action_dim / 2.0
            if cfg.xqc_target_entropy == "auto"
            else float(cfg.xqc_target_entropy)
        )
        cfg.xqc_optimizer_backend = str(cfg.xqc_optimizer_backend).lower()
        if cfg.xqc_optimizer_backend not in {
            "auto",
            "single_tensor",
            "foreach",
            "fused",
        }:
            raise ValueError(
                "xqc_optimizer_backend must be auto, single_tensor, foreach, or fused."
            )
        if cfg.xqc_reward_normalization is not True:
            raise ValueError(
                "AMBIXQC requires XQC discounted-return reward normalization."
            )
        cfg.xqc_reward_normalization = True
        cfg.reward_normalization = True
        cfg.xqc_official_commit = OFFICIAL_XQC_COMMIT
        # The outer schedule advances once per accepted controller optimizer
        # slot and spans the run's intended TOLD update budget.
        cfg.xqc_lr_transition_steps = int(cfg.steps)

        cfg.inner_rounds = _positive_int(cfg.inner_rounds, "inner_rounds")
        cfg.inner_rollouts_per_round = _positive_int(
            cfg.inner_rollouts_per_round, "inner_rollouts_per_round"
        )
        cfg.inner_rollout_horizon = _positive_int(
            cfg.inner_rollout_horizon, "inner_rollout_horizon"
        )
        cfg.inner_updates_per_round = _positive_int(
            cfg.inner_updates_per_round, "inner_updates_per_round"
        )
        cfg.inner_batch_size = _positive_int(cfg.inner_batch_size, "inner_batch_size")
        cfg.inner_model_step_budget = (
            cfg.inner_rounds
            * cfg.inner_rollouts_per_round
            * cfg.inner_rollout_horizon
        )
        if cfg.inner_replay_capacity is None:
            cfg.inner_replay_capacity = cfg.inner_model_step_budget
        cfg.inner_replay_capacity = _positive_int(
            cfg.inner_replay_capacity, "inner_replay_capacity"
        )
        if cfg.inner_replay_capacity < cfg.inner_model_step_budget:
            raise ValueError(
                "Action-local inner_replay_capacity must hold all imagined "
                f"transitions ({cfg.inner_model_step_budget})."
            )
        cfg.inner_replay_sampling = str(cfg.inner_replay_sampling).lower()
        if cfg.inner_replay_sampling != "with_replacement":
            raise ValueError(
                "AMBI-XQC v1 requires inner_replay_sampling='with_replacement'."
            )
        cfg.inner_actor_lr = _finite_float(
            cfg.inner_actor_lr, "inner_actor_lr", positive=True
        )
        cfg.inner_critic_lr = _finite_float(
            cfg.inner_critic_lr, "inner_critic_lr", positive=True
        )
        cfg.inner_diagnostics_every = _positive_int(
            cfg.inner_diagnostics_every, "inner_diagnostics_every"
        )

        # Fixed AMBI-XQC semantics are recorded on cfg for checkpoint and run
        # metadata, but are intentionally not configurable in the first port.
        cfg.inner_operator = "xqc"
        cfg.inner_schedule_mode = "canonical"
        cfg.inner_actor_adaptation = "clone"
        cfg.inner_critic_adaptation = "clone"
        cfg.inner_temperature_mode = "auto"
        cfg.inner_temperature_initialization = "inherit_outer"
        cfg.inner_target_entropy = cfg.xqc_resolved_target_entropy
        cfg.inner_actor_scope = "action"
        cfg.inner_critic_scope = "action"
        cfg.inner_temperature_scope = "action"
        cfg.inner_replay_scope = "action"
        cfg.inner_actor_optimizer_scope = "action"
        cfg.inner_critic_optimizer_scope = "action"
        cfg.inner_temperature_optimizer_scope = "action"
        cfg.inner_rebase_persistent = False
        cfg.inner_behavior_action = "policy_sample"
        cfg.inner_execution_action = "policy_sample"
        cfg.inner_temperature_lr = cfg.inner_actor_lr
        cfg.inner_critic_target_tau = cfg.xqc_tau
        cfg.inner_critic_target_update_interval = cfg.xqc_target_update_interval
        cfg.inner_critic_dropout_enabled = False
        cfg.inner_diagnostic_rollouts = 0
        cfg.inner_termination_threshold = 0.5
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
        cfg.inner_nominal_transitions_per_round = (
            cfg.inner_rollouts_per_round * cfg.inner_rollout_horizon
        )
        cfg.inner_nominal_updates_per_round = cfg.inner_updates_per_round
        cfg.inner_expected_update_slots = (
            cfg.inner_rounds * cfg.inner_updates_per_round
        )
        cfg.inner_critic_updates_per_action = cfg.inner_expected_update_slots
        accepted_actor_steps = (
            (cfg.inner_expected_update_slots - 1) // cfg.xqc_policy_delay + 1
        )
        cfg.inner_actor_updates_per_action = accepted_actor_steps
        cfg.inner_temperature_updates_per_action = accepted_actor_steps
        cfg.inner_nominal_critic_utd = (
            cfg.inner_expected_update_slots / cfg.inner_model_step_budget
        )
        return cfg

    def _make_agent(self, cfg):
        return AMBIXQCAgent(cfg)

    def learn(self, total_timesteps=10_000, *, resume_session=None):
        total_timesteps = validate_timestep_budget(total_timesteps)
        if total_timesteps != int(self.cfg.steps):
            raise ValueError(
                "AMBIXQC total_timesteps must match the construction-time step "
                "budget so the XQC learning-rate schedule remains exact."
            )
        return super().learn(
            total_timesteps=total_timesteps,
            resume_session=resume_session,
        )

    def _observe_transition(self, reward, terminated, truncated):
        self.agent.observe_reward(reward, terminated, truncated)

    def _wandb_run_name(self):
        env_params = self.experiment_params.get("env_params", {})
        task = env_params.get("task", self.run_params.get("env", "env"))
        return f"AMBIXQC-{task}-seed{self.cfg.seed}"

    def enable_training_resume(self, *, total_timesteps):
        del total_timesteps
        raise NotImplementedError(
            "Exact AMBI-XQC trainer resume is not supported in v1; portable "
            "model checkpoints remain available."
        )


AMBIXQCBaseline = AMBIXQC
