"""Action Modes wrapper for the official state-based XQC learner.

The learner is a PyTorch compatibility port of XQC commit
``9a6832bb742ef01bbe9f1e06153a9338e612dae5``.  This wrapper intentionally
shares SAC's small environment/replay/checkpoint harness while retaining XQC's
one-based warmup and evaluation schedule.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from numbers import Integral

import numpy as np
import torch
from gymnasium.spaces import Box
from gymnasium.spaces import utils as space_utils

from RL.SAC import SAC
from RL.xqc_core import OFFICIAL_XQC_COMMIT, XQCAgent, XQCConfig
from utils.cleanup import add_cleanup_notes
from utils.core import build_env
from utils.wandb_utils import init_wandb


class XQC(SAC):
    """Faithful, state-observation XQC with Action Modes lifecycle support."""

    _display_name = "XQC"
    _eval_file_label = "XQC"
    _eval_csv_environment_variable = "XQC_EVAL_CSV"

    def __init__(
        self,
        name,
        env,
        custom_params=None,
        run_params=None,
        experiment_params=None,
    ):
        params = dict(custom_params or {})
        params.setdefault("eval_freq", 50_000)
        params.setdefault("eval_episodes", 10)
        if params.get("obs", "state") != "state":
            raise ValueError("This XQC port supports state observations only.")
        observation_space = env.observation_space
        if not isinstance(observation_space, Box) or len(observation_space.shape) != 1:
            raise ValueError(
                "This XQC port requires a one-dimensional Box feature vector."
            )
        self._xqc_eval_env = None
        self._xqc_eval_env_seeded = False
        super().__init__(
            name,
            env,
            custom_params=params,
            run_params=run_params,
            experiment_params=experiment_params,
        )

    def _make_config(self) -> XQCConfig:
        misleading_options = {
            key
            for key in ("temp_lr", "alpha_lr", "normalize_last_layer")
            if key in self.params
        }
        if misleading_options:
            raise ValueError(
                "This faithful XQC port deliberately omits ineffective upstream "
                f"options: {sorted(misleading_options)}."
            )
        learning_rate = self.params.get("learning_rate", 3e-4)
        actor_lr = self.params.get("actor_lr", learning_rate)
        critic_lr = self.params.get("critic_lr", learning_rate)

        has_updates = "updates_per_step" in self.params
        has_gradients = "gradient_steps" in self.params
        updates_per_step = self.params.get(
            "updates_per_step", self.params.get("gradient_steps", 2)
        )
        gradient_steps = self.params.get("gradient_steps", updates_per_step)
        if has_updates and has_gradients and updates_per_step != gradient_steps:
            raise ValueError(
                "XQC updates_per_step and gradient_steps must agree when both "
                "are provided."
            )

        train_freq, train_unit = self._parse_train_freq(
            self.params.get("train_freq", 1)
        )
        if train_unit != "step":
            raise ValueError("XQC supports train_freq in environment steps only.")

        actor_arch = self.params.get(
            "actor_net_arch", (256, 256, 256, 256)
        )
        critic_arch = self.params.get(
            "critic_net_arch", (512, 512, 512, 512)
        )
        target_entropy = self.params.get("target_entropy", "auto")
        if self.params.get("reward_normalization", True) is not True:
            raise ValueError(
                "Faithful XQC requires discounted-return reward normalization."
            )

        run_interactions = self.run_params.get("total_steps", 500_000)
        num_interactions = self.params.get("num_interactions", run_interactions)
        device = self.run_params.get("device", self.params.get("device", "auto"))

        config = XQCConfig(
            learning_rate=learning_rate,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            lr_end=self.params.get("lr_end", 3e-5),
            num_interactions=num_interactions,
            updates_per_step=updates_per_step,
            buffer_size=self.params.get("buffer_size", 1_000_000),
            learning_starts=self.params.get("learning_starts", 5_000),
            batch_size=self.params.get("batch_size", 256),
            tau=self.params.get("tau", 0.005),
            gamma=self.params.get("gamma", 0.99),
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=self.params.get("target_update_interval", 1),
            policy_delay=self.params.get("policy_delay", 3),
            actor_net_arch=actor_arch,
            critic_net_arch=critic_arch,
            num_atoms=self.params.get("num_atoms", 101),
            vmin=self.params.get("vmin", -5.0),
            vmax=self.params.get("vmax", 5.0),
            init_temperature=self.params.get("init_temperature", 0.01),
            target_entropy=target_entropy,
            adam_eps=self.params.get("adam_eps", 1e-8),
            weight_decay=self.params.get("weight_decay", 0.0),
            reward_normalization=True,
            seed=self.seed,
            device=device,
            verbose=self.verbose,
        )
        if "num_interactions" in self.params and "total_steps" in self.run_params:
            run_total = self._positive_int(
                self.run_params["total_steps"], "run_params['total_steps']"
            )
            if config.num_interactions != run_total:
                raise ValueError(
                    "XQC num_interactions must equal run_params['total_steps'] so "
                    "the official learning-rate schedule spans the training run."
                )
        return config

    def _make_agent(self):
        return XQCAgent(self.obs_dim, self.action_dim, self.cfg)

    def _init_wandb(self):
        return init_wandb(
            self.params,
            default_project="ambi",
            run_name=(
                f"XQC-{self.run_params.get('env', 'env')}-seed{self.seed}"
            ),
            config={
                "run_params": self.run_params,
                "alg_params": self.params,
                "config": asdict(self.cfg),
                "official_xqc_commit": OFFICIAL_XQC_COMMIT,
            },
        )

    def _observe_transition(self, reward, terminated, truncated):
        self.agent.observe_reward(reward, terminated, truncated)

    def _sample_random_action(self, interaction: int) -> bool:
        # The released XQC loop is one-based: random for i < 5000, policy at
        # i == 5000, and learning after that interaction has been inserted.
        return interaction < self.cfg.learning_starts

    def _should_run_initial_evaluation(self, reset_num_timesteps: bool) -> bool:
        return False

    def _is_evaluation_step(self, step: int) -> bool:
        return self._eval_freq is not None and (
            step == 1 or step % self._eval_freq == 0
        )

    def _evaluation_is_ready(self, episode_done: bool) -> bool:
        # XQC owns a separate evaluation environment, so evaluation does not
        # need to wait for the training episode to finish.
        return True

    @staticmethod
    def _close_env_after_error(env, error):
        try:
            env.close()
        except BaseException as cleanup_error:
            add_cleanup_notes(
                error,
                [cleanup_error],
                prefix="Additional XQC evaluation-environment cleanup failure",
            )

    def _build_evaluation_env(self):
        eval_env = build_env(
            self.run_params,
            self.experiment_params,
            render_mode=None,
        )
        try:
            if not isinstance(eval_env.action_space, Box):
                raise ValueError("XQC evaluation requires a continuous Box action space.")
            if not isinstance(eval_env.observation_space, Box) or len(
                eval_env.observation_space.shape
            ) != 1:
                raise ValueError(
                    "XQC evaluation requires a one-dimensional Box feature vector."
                )
            if int(space_utils.flatdim(eval_env.observation_space)) != self.obs_dim:
                raise ValueError(
                    "XQC training and evaluation observation dimensions differ."
                )
            if tuple(eval_env.action_space.shape) != tuple(self.action_shape):
                raise ValueError("XQC training and evaluation action shapes differ.")
            if not np.array_equal(
                np.asarray(eval_env.action_space.low),
                np.asarray(self.env.action_space.low),
            ) or not np.array_equal(
                np.asarray(eval_env.action_space.high),
                np.asarray(self.env.action_space.high),
            ):
                raise ValueError("XQC training and evaluation action bounds differ.")
            eval_env.action_space.seed(self.seed + 42)
            eval_env.observation_space.seed(self.seed + 42)
        except BaseException as exc:
            self._close_env_after_error(eval_env, exc)
            raise
        return eval_env

    def _get_evaluation_env(self):
        if self._xqc_eval_env is None:
            self._xqc_eval_env = self._build_evaluation_env()
            self._xqc_eval_env_seeded = False
        return self._xqc_eval_env

    def _prepare_evaluation_environment(self):
        # The official training entry point constructs both environments before
        # collecting transitions. Validate the independent eval environment at
        # the same boundary so a construction error cannot partially train.
        self._get_evaluation_env()

    def _reset_evaluation_env(self):
        eval_env = self._get_evaluation_env()
        seed = None if self._xqc_eval_env_seeded else self.seed + 42
        try:
            out = eval_env.reset(seed=seed) if seed is not None else eval_env.reset()
        except TypeError:
            out = eval_env.reset()
        self._xqc_eval_env_seeded = True
        return out[0] if isinstance(out, tuple) else out

    def _close_evaluation_env(self):
        eval_env, self._xqc_eval_env = self._xqc_eval_env, None
        self._xqc_eval_env_seeded = False
        if eval_env is not None:
            eval_env.close()

    def learn(self, total_timesteps=10_000, reset_num_timesteps=True):
        primary_error = None
        try:
            return super().learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=reset_num_timesteps,
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                self._close_evaluation_env()
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                add_cleanup_notes(
                    primary_error,
                    [cleanup_error],
                    prefix="Additional XQC evaluation-environment cleanup failure",
                )

    @staticmethod
    def _semantic_config(config):
        signature = asdict(config)
        # Placement and console verbosity do not change learned state or XQC's
        # update equations and must remain portable across checkpoints.
        for nonsemantic_key in ("device", "verbose"):
            signature.pop(nonsemantic_key, None)
        return signature

    def _checkpoint_state(self):
        return {
            "checkpoint_version": 1,
            "algorithm": "XQC",
            "official_xqc_commit": OFFICIAL_XQC_COMMIT,
            "agent": self.agent.state_dict(),
            "config": asdict(self.cfg),
            "semantic_config": self._semantic_config(self.cfg),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "action_low": np.asarray(
                self.env.action_space.low, dtype=np.float32
            ),
            "action_high": np.asarray(
                self.env.action_space.high, dtype=np.float32
            ),
            "num_timesteps": self.num_timesteps,
            "metrics": self._last_metrics,
            "checkpoint_type": (
                "weights_optimizers_schedules_and_reward_normalizer_without_replay"
            ),
        }

    def load(self, path, *, resume=False, strict_config=True):
        checkpoint = torch.load(
            path, map_location=self.agent.device, weights_only=False
        )
        if not isinstance(checkpoint, Mapping):
            raise ValueError("XQC checkpoint must contain a mapping.")
        if checkpoint.get("algorithm") != "XQC":
            raise ValueError("Checkpoint is not an XQC checkpoint.")
        checkpoint_version = checkpoint.get("checkpoint_version")
        if (
            isinstance(checkpoint_version, bool)
            or not isinstance(checkpoint_version, Integral)
            or int(checkpoint_version) != 1
        ):
            raise ValueError(
                f"Unsupported XQC checkpoint_version={checkpoint_version!r}; "
                "supported version is 1."
            )
        if checkpoint.get("official_xqc_commit") != OFFICIAL_XQC_COMMIT:
            raise ValueError("Checkpoint targets a different official XQC source commit.")
        saved_obs_dim = checkpoint.get("obs_dim")
        if (
            isinstance(saved_obs_dim, bool)
            or not isinstance(saved_obs_dim, Integral)
            or int(saved_obs_dim) != self.obs_dim
        ):
            raise ValueError(
                "Checkpoint observation dimension does not match this environment."
            )
        saved_action_dim = checkpoint.get("action_dim")
        if (
            isinstance(saved_action_dim, bool)
            or not isinstance(saved_action_dim, Integral)
            or int(saved_action_dim) != self.action_dim
        ):
            raise ValueError(
                "Checkpoint action dimension does not match this environment."
            )
        if not np.array_equal(
            np.asarray(checkpoint.get("action_low")),
            np.asarray(self.env.action_space.low, dtype=np.float32),
        ):
            raise ValueError(
                "Checkpoint action lower bounds do not match this environment."
            )
        if not np.array_equal(
            np.asarray(checkpoint.get("action_high")),
            np.asarray(self.env.action_space.high, dtype=np.float32),
        ):
            raise ValueError(
                "Checkpoint action upper bounds do not match this environment."
            )
        if strict_config:
            saved_signature = checkpoint.get("semantic_config")
            if saved_signature is None and "config" in checkpoint:
                if not isinstance(checkpoint["config"], Mapping):
                    raise ValueError("Checkpoint XQC config must be a mapping.")
                saved_signature = dict(checkpoint["config"])
                for nonsemantic_key in ("device", "verbose"):
                    saved_signature.pop(nonsemantic_key, None)
            if not isinstance(saved_signature, Mapping):
                raise ValueError(
                    "Checkpoint XQC semantic_config must be a mapping."
                )
            saved_signature = dict(saved_signature)
            current_signature = self._semantic_config(self.cfg)
            if saved_signature != current_signature:
                saved_signature = saved_signature or {}
                keys = set(saved_signature) | set(current_signature)
                mismatches = {
                    key: (saved_signature.get(key), current_signature.get(key))
                    for key in sorted(keys)
                    if saved_signature.get(key) != current_signature.get(key)
                }
                raise ValueError(f"Checkpoint XQC configuration mismatch: {mismatches}")
        if resume:
            raise ValueError(
                "XQC checkpoints do not include replay or environment state and "
                "cannot safely resume. Load for evaluation/fine-tuning with "
                "resume=False."
            )

        self.agent.load_state_dict(checkpoint["agent"])
        self.num_timesteps = 0
        self._last_metrics = checkpoint.get("metrics", {})
        self._last_obs = None
        self._episode_return = 0.0
        self._episode_len = 0
        self._collected_steps = 0
        self._collected_episodes = 0
        self._eval_pending = False
        return self


XQCBaseline = XQC
