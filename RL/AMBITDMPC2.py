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
    "inner_tau": 0.005,
    "inner_target_update_interval": 1,
    "inner_grad_clip_norm": 20.0,
    "inner_termination_threshold": 0.5,

    # Optional low-rank inner adaptation. ``clone`` is the simplest reference
    # implementation; ``lora`` freezes copied outer weights and updates only
    # these adapters.
    "lora_rank": 8,
    "lora_alpha": 8.0,
    "lora_dropout": 0.0,
}


class AMBITDMPC2(TDMPC2Baseline):
    """AMBI algorithm using TD-MPC2 representation learning and latent SAC."""

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

    def _log_wandb_step(self, reward, terminated, truncated, metrics=None):
        combined = dict(metrics or {})
        combined.update(self.agent.last_inner_metrics)
        super()._log_wandb_step(reward, terminated, truncated, combined)
