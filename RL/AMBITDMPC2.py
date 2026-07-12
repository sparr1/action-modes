"""AMBI with a TD-MPC2 world model and per-state latent SAC improvement."""

import numpy as np

from RL.TDMPC2 import TDMPC2Baseline
from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from utils.utils import setup_logs


_AMBI_DEFAULTS = {
    # Outer latent SAC. TD-MPC2's encoder/dynamics/reward losses are retained,
    # but its distributional Q/policy-prior control head is replaced by SAC.
    "mpc": False,
    "num_q": 2,
    "critic_coef": 1.0,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "adam_eps": 1e-8,
    "log_std_min": -20,
    "log_std_max": 2,
    "ent_coef": "auto",
    "ent_coef_lr": 3e-4,
    "target_entropy": "auto",
    "tau": 0.005,
    "target_update_interval": 1,

    # Per-state inner SAC. A fresh inner learner is initialized from the outer
    # actor/critic for every real action, then trained only on latent rollouts.
    "inner_adaptation": "clone",
    "inner_iterations": 1,
    "inner_rollouts": 32,
    "inner_horizon": None,
    "inner_updates_per_iteration": 1,
    "inner_batch_size": 256,
    "inner_buffer_size": None,
    "inner_actor_lr": 3e-4,
    "inner_critic_lr": 3e-4,
    "inner_adam_eps": 1e-8,
    # The inner learner is discarded after only a handful of updates. A hard
    # target sync lets later local updates bootstrap through earlier ones;
    # long-run SAC's usual tau=0.005 would leave the target almost fully outer.
    "inner_tau": 1.0,
    "inner_target_update_interval": 1,
    "inner_grad_clip_norm": 20.0,
    "inner_termination_threshold": 0.5,
    "allow_long_inner_horizon": False,

    # Optional low-rank inner adaptation. ``clone`` is the simplest reference
    # implementation; ``lora`` freezes copied outer weights and updates only
    # these adapters.
    "lora_rank": 8,
    "lora_alpha": 8.0,
    "lora_dropout": 0.0,
}


class AMBITDMPC2(TDMPC2Baseline):
    """AMBI algorithm using TD-MPC2 representation learning and latent SAC."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inner_steps_total = 0
        self._inner_updates_total = 0
        self._wandb_inner_seconds = 0.0
        self._wandb_inner_actions = 0
        self._wandb_inner_steps = 0

    def _build_cfg(self, params):
        if "num_q" in params and int(params["num_q"]) != 2:
            raise ValueError("AMBI-TD-MPC2 uses exactly two SAC critics; num_q must be 2.")
        if bool(params.get("mpc", False)):
            raise ValueError("AMBI-TD-MPC2 replaces MPPI; mpc must be false.")

        merged = dict(_AMBI_DEFAULTS)
        merged.update(params)
        cfg = super()._build_cfg(merged)

        # ``model_size`` sets TD-MPC2's original Q-ensemble size. AMBI uses
        # standard SAC twin critics regardless of model width.
        cfg.num_q = 2
        cfg.mpc = False

        if cfg.inner_horizon is None:
            cfg.inner_horizon = cfg.horizon
        if int(cfg.inner_horizon) > int(cfg.horizon) and not bool(cfg.allow_long_inner_horizon):
            raise ValueError(
                "inner_horizon exceeds the horizon used to train the world model. "
                "Increase both horizons or set allow_long_inner_horizon=true for an explicit ablation."
            )

        integer_positive = (
            "inner_rollouts",
            "inner_horizon",
            "inner_batch_size",
            "inner_target_update_interval",
            "target_update_interval",
        )
        for key in integer_positive:
            value = int(getattr(cfg, key))
            if value <= 0:
                raise ValueError(f"{key} must be positive, got {value}.")
            setattr(cfg, key, value)

        for key in ("inner_iterations", "inner_updates_per_iteration"):
            value = int(getattr(cfg, key))
            if value < 0:
                raise ValueError(f"{key} must be non-negative, got {value}.")
            setattr(cfg, key, value)

        if cfg.inner_buffer_size is not None:
            cfg.inner_buffer_size = int(cfg.inner_buffer_size)
            if cfg.inner_buffer_size <= 0:
                raise ValueError("inner_buffer_size must be positive or null.")

        cfg.inner_adaptation = str(cfg.inner_adaptation).lower()
        if cfg.inner_adaptation not in {"clone", "lora"}:
            raise ValueError("inner_adaptation must be 'clone' or 'lora'.")
        if cfg.inner_adaptation == "lora" and int(cfg.lora_rank) <= 0:
            raise ValueError("lora_rank must be positive.")
        if not 0.0 <= float(cfg.inner_termination_threshold) <= 1.0:
            raise ValueError("inner_termination_threshold must be in [0, 1].")
        if not 0.0 < float(cfg.tau) <= 1.0:
            raise ValueError("tau must be in (0, 1].")
        if not 0.0 < float(cfg.inner_tau) <= 1.0:
            raise ValueError("inner_tau must be in (0, 1].")
        if float(cfg.adam_eps) <= 0.0 or float(cfg.inner_adam_eps) <= 0.0:
            raise ValueError("adam_eps and inner_adam_eps must be positive.")

        return cfg

    def _make_agent(self, cfg):
        return AMBITDMPC2Agent(cfg)

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
        obs_for_log = obs if isinstance(obs, dict) else np.asarray(obs)[None, ...]
        action_for_log = np.asarray(action)[None, ...]
        data = setup_logs(
            reward,
            obs_for_log,
            action_for_log,
            [done],
            [info_for_log],
            # AMBITrainingLogger stores one rollout-length list per real step.
            inner_steps=[list(self.agent.last_inner_rollout_lengths)],
        )
        self.alg_logger.on_step(data)

    def _reset_wandb_window(self):
        super()._reset_wandb_window()
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
                "train/inner_steps": 0,
                "train/inner_updates": 0,
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

        aliases = {
            "inner_buffer_fill_fraction": "inner_buffer_fill_ratio",
            "inner_rollout_termination_rate": "inner_termination_rate",
            "inner_policy_action_delta_l2": "inner_policy_mean_delta_l2",
        }
        excluded = {
            "inner_active", "inner_actions", "inner_iterations",
            "inner_rollouts", "inner_rollout_count", "inner_steps", "inner_updates",
            "inner_return_mean", "inner_return_std", "inner_return_min", "inner_return_max",
            "inner_rollout_len_mean", "inner_rollout_len_std",
            "inner_rollout_len_min", "inner_rollout_len_max",
            "inner_termination_rate", "inner_rollout_termination_rate",
        }
        termination_rate = metrics.get(
            "inner_termination_rate",
            metrics.get("inner_rollout_termination_rate"),
        )
        if termination_rate is not None:
            self._wandb_train_window.add_weighted(
                "train/inner_termination_rate",
                termination_rate,
                weight=max(1, rollout_count),
            )
        for key, value in metrics.items():
            if key in excluded or key.endswith("_total") or "_time_" in key or key.endswith("_seconds"):
                continue
            key = aliases.get(key, key)
            update_metric = any(
                token in key
                for token in ("loss", "grad_norm", "_q_", "entropy", "td_error")
            )
            if key in {"inner_outer_q_gain", "inner_policy_mean_delta_l2"}:
                update_metric = False
            weight = max(1, inner_updates) if update_metric else 1
            self._wandb_train_window.add_weighted(f"train/{key}", value, weight=weight)

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
