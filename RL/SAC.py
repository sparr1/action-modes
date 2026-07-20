"""AMBI-native SAC algorithm wrapper.

Use this when we need direct access to SAC actor/critic/Q-values without SB3's
VecEnv, callback, and replay-buffer machinery. The learning equations and action
scaling follow Stable-Baselines3 SAC semantics as closely as possible.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import asdict
from numbers import Integral, Real
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from gymnasium.spaces import Box
from gymnasium.spaces import utils as space_utils

from RL.alg import Algorithm
from RL.sac_core import ReplayBuffer, SACAgent, SACConfig
from utils.checkpointing import (
    CheckpointTracker,
    explicit_checkpoint_target,
    publish_checkpoint,
)
from utils.utils import setup_logs
from utils.wandb_utils import (
    WandbAccumulator,
    extract_reward_components,
    finish_wandb,
    init_wandb,
    log_wandb,
)


class SAC(Algorithm):
    supports_composable_checkpointing = True

    def __init__(self, name, env, custom_params=None, run_params=None, experiment_params=None):
        super().__init__(name, env, custom_params=custom_params)
        self.params = custom_params or {}
        self.run_params = run_params or {}
        self.experiment_params = experiment_params or {}
        self.seed = int(self.params.get("seed", self.run_params.get("seed", 0)))
        self.verbose = int(self.params.get("verbose", 1))

        if not isinstance(self.env.action_space, Box):
            raise ValueError("Native SAC supports continuous Box action spaces only.")
        if not np.all(np.isfinite(self.env.action_space.low)) or not np.all(np.isfinite(self.env.action_space.high)):
            raise ValueError("Native SAC requires finite action bounds.")

        self.obs_dim = int(space_utils.flatdim(self.env.observation_space))
        self.action_shape = self.env.action_space.shape
        self.action_dim = int(np.prod(self.action_shape))
        self.cfg = self._make_config()
        self.replay_buffer = ReplayBuffer(self.obs_dim, self.action_dim, self.cfg.buffer_size)
        self.agent = SACAgent(self.obs_dim, self.action_dim, self.cfg)
        self.num_timesteps = 0
        self._last_obs = None
        self._last_metrics = {}
        self._checkpoint = None
        self._episode_idx = 0
        self._episode_return = 0.0
        self._episode_len = 0
        self._collected_steps = 0
        self._collected_episodes = 0
        self._rng_seeded = False
        self._env_seeded = False
        self._wandb_every = max(1, int(self.params.get("wandb_step_every", 1000)))
        self._wandb_run = None
        self._wandb_train_window = WandbAccumulator()
        self._wandb_reward_window = WandbAccumulator()
        self._wandb_start_time = None
        self._wandb_window_start_time = None
        self._wandb_window_start_step = 0
        self._wandb_train_seconds = 0.0
        self._wandb_last_updates = 0
        self._last_wandb_step = None
        self._last_reward = 0.0
        self._last_terminated = False
        self._last_truncated = False
        self._last_info = {}

    def _init_wandb(self):
        return init_wandb(
            self.params,
            default_project="ambi",
            run_name=f"NativeSAC-{self.run_params.get('env', 'env')}-seed{self.seed}",
            config={"run_params": self.run_params, "alg_params": self.params, "config": asdict(self.cfg)},
        )

    def _make_config(self) -> SACConfig:
        unsupported = {
            "policy", "env", "tensorboard_log", "replay_buffer_class", "replay_buffer_kwargs",
            "optimize_memory_usage", "n_steps", "action_noise", "use_sde", "sde_sample_freq",
            "use_sde_at_warmup", "stats_window_size",
        }
        for key in unsupported:
            if key in self.params and self.verbose:
                print(f"Native SAC ignoring SB3-specific parameter: {key}")

        policy_kwargs = self.params.get("policy_kwargs", {}) or {}
        if not isinstance(policy_kwargs, dict):
            raise ValueError("policy_kwargs must be a dictionary when provided.")
        unsupported_policy_kwargs = set(policy_kwargs) - {"net_arch"}
        if unsupported_policy_kwargs:
            raise ValueError(
                "Native SAC only supports policy_kwargs['net_arch']; unsupported keys: "
                f"{sorted(unsupported_policy_kwargs)}"
            )

        net_arch = policy_kwargs.get("net_arch", self.params.get("net_arch", [256, 256]))
        if isinstance(net_arch, dict):
            actor_arch = self._validated_net_arch(net_arch.get("pi", [256, 256]), "pi")
            critic_arch = self._validated_net_arch(net_arch.get("qf", [256, 256]), "qf")
        else:
            actor_arch = critic_arch = self._validated_net_arch(net_arch, "net_arch")

        num_q = int(self.params.get("num_q", 2))
        q_pair_size = int(self.params.get("q_pair_size", 2))
        if num_q != 2 or q_pair_size != 2:
            raise ValueError(
                "Native SAC keeps exactly two critics for a clean scalar-versus-"
                "distributional comparison; num_q and q_pair_size must both be 2."
            )

        device = self.run_params.get("device", self.params.get("device", "auto"))
        return SACConfig(
            learning_rate=float(self.params.get("learning_rate", 3e-4)),
            buffer_size=int(float(self.params.get("buffer_size", 1_000_000))),
            learning_starts=int(float(self.params.get("learning_starts", 100))),
            batch_size=int(float(self.params.get("batch_size", 256))),
            tau=float(self.params.get("tau", 0.005)),
            gamma=float(self.params.get("gamma", 0.99)),
            train_freq=self._parse_train_freq(self.params.get("train_freq", 1))[0],
            gradient_steps=int(float(self.params.get("gradient_steps", 1))),
            ent_coef=self.params.get("ent_coef", "auto"),
            target_entropy=self.params.get("target_entropy", "auto"),
            target_update_interval=int(float(self.params.get("target_update_interval", 1))),
            net_arch=actor_arch,
            actor_net_arch=actor_arch,
            critic_net_arch=critic_arch,
            q_representation=str(self.params.get("q_representation", "scalar")).lower(),
            q_num_bins=int(self.params.get("q_num_bins", 101)),
            q_vmin=float(self.params.get("q_vmin", -10.0)),
            q_vmax=float(self.params.get("q_vmax", 10.0)),
            adam_eps=float(self.params.get("adam_eps", 1e-8)),
            seed=self.seed,
            device=device,
            verbose=self.verbose,
        )

    @staticmethod
    def _validated_net_arch(net_arch, name: str) -> Tuple[int, ...]:
        if not isinstance(net_arch, (list, tuple)):
            raise ValueError(f"{name} must be a list or tuple of positive layer widths.")
        widths = tuple(int(width) for width in net_arch)
        if any(width <= 0 for width in widths):
            raise ValueError(f"{name} must contain only positive layer widths.")
        return widths

    def _parse_train_freq(self, train_freq) -> Tuple[int, str]:
        if isinstance(train_freq, (list, tuple)):
            if len(train_freq) != 2:
                raise ValueError("train_freq tuple/list must be [frequency, 'step'] or [frequency, 'episode'].")
            frequency, unit = train_freq[0], str(train_freq[1])
        else:
            frequency, unit = train_freq, "step"
        if isinstance(frequency, bool) or not isinstance(frequency, (Integral, Real)) or int(frequency) != frequency:
            raise ValueError("train_freq frequency must be an integer.")
        frequency = int(frequency)
        if frequency <= 0:
            raise ValueError("train_freq must be positive.")
        if unit not in ("step", "episode"):
            raise ValueError("Native SAC supports train_freq unit 'step' or 'episode'.")
        return frequency, unit

    def _flatten_obs(self, obs) -> np.ndarray:
        return np.asarray(space_utils.flatten(self.env.observation_space, obs), dtype=np.float32)

    def _scale_action(self, action_env) -> np.ndarray:
        low = self.env.action_space.low.reshape(-1)
        high = self.env.action_space.high.reshape(-1)
        action_env = np.asarray(action_env, dtype=np.float32).reshape(-1)
        action = 2.0 * (action_env - low) / (high - low) - 1.0
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _unscale_action(self, action_norm) -> np.ndarray:
        low = self.env.action_space.low.reshape(-1)
        high = self.env.action_space.high.reshape(-1)
        action_norm = np.asarray(action_norm, dtype=np.float32).reshape(-1)
        action = low + 0.5 * (action_norm + 1.0) * (high - low)
        return np.clip(action, low, high).astype(np.float32).reshape(self.action_shape)

    def _reset_env(self, seed=None):
        try:
            out = self.env.reset(seed=seed) if seed is not None else self.env.reset()
        except TypeError:
            out = self.env.reset()
        return out[0] if isinstance(out, tuple) else out

    def _log_step(self, obs, reward, action_env, terminated, truncated, info):
        if self.alg_logger is None:
            return
        done = bool(terminated or truncated)
        info_for_log = dict(info or {})
        info_for_log.setdefault("terminated", bool(terminated))
        info_for_log.setdefault("truncated", bool(truncated))
        obs_for_log = obs if isinstance(obs, dict) else np.asarray(obs)[None, ...]
        action_for_log = np.asarray(action_env)[None, ...]
        data = setup_logs(reward, obs_for_log, action_for_log, [done], [info_for_log])
        self.alg_logger.on_step(data)

    def _reset_wandb_window(self):
        self._wandb_train_window.clear()
        self._wandb_reward_window.clear()
        now = time.perf_counter()
        self._wandb_start_time = now
        self._wandb_window_start_time = now
        self._wandb_window_start_step = int(self.num_timesteps)
        self._wandb_train_seconds = 0.0
        self._wandb_last_updates = int(self.agent.num_updates)
        self._last_wandb_step = None

    def _replay_wandb_payload(self):
        size = int(self.replay_buffer.size)
        capacity = int(self.replay_buffer.capacity)
        return {
            "train/replay_size": size,
            "train/replay_capacity": capacity,
            "train/replay_fill_ratio": float(size / capacity) if capacity > 0 else 0.0,
        }

    def _timing_wandb_payload(self, updates_since_log):
        now = time.perf_counter()
        start = self._wandb_start_time if self._wandb_start_time is not None else now
        window_start = self._wandb_window_start_time if self._wandb_window_start_time is not None else now
        window_seconds = max(0.0, now - window_start)
        window_steps = max(0, int(self.num_timesteps) - int(self._wandb_window_start_step))
        payload = {
            "time/window_seconds": float(window_seconds),
            "time/time_elapsed": float(max(0.0, now - start)),
            "time/total_timesteps": int(self.num_timesteps),
            "time/fps": float(window_steps / window_seconds) if window_seconds > 0 else 0.0,
            "time/train_seconds": float(self._wandb_train_seconds),
            "time/updates_per_second": (
                float(updates_since_log / self._wandb_train_seconds)
                if self._wandb_train_seconds > 0 else 0.0
            ),
        }
        self._wandb_window_start_time = now
        self._wandb_window_start_step = int(self.num_timesteps)
        self._wandb_train_seconds = 0.0
        return payload

    def _log_wandb_step(self, reward, terminated, truncated, info=None, *, completed_episode=False, force=False):
        if self._wandb_run is None:
            return
        done = bool(terminated or truncated)
        if not force and not done and self.num_timesteps % self._wandb_every != 0:
            return
        if (
            force
            and not self._wandb_train_window
            and not self._wandb_reward_window
            and (
                self._last_wandb_step == self.num_timesteps
                or self.num_timesteps == self._wandb_window_start_step
            )
        ):
            return

        updates_since_log = int(self.agent.num_updates - self._wandb_last_updates)
        payload = {
            "train/reward": float(reward),
            "train/done": int(done),
            "train/terminated": int(bool(terminated)),
            "train/truncated": int(bool(truncated)),
            "train/learning_started": int(self.num_timesteps > self.cfg.learning_starts),
            "train/learning_rate": float(self.cfg.learning_rate),
            "train/n_updates": int(self.agent.num_updates),
            "train/updates_since_log": updates_since_log,
            "episode/current_return": float(self._episode_return),
            "episode/current_len": int(self._episode_len),
        }
        payload.update(self._replay_wandb_payload())
        payload.update(self._wandb_reward_window.pop())
        payload.update(self._wandb_train_window.pop())
        payload.update(extract_reward_components(info or {}))
        payload.update(self._timing_wandb_payload(updates_since_log))
        if completed_episode:
            payload.update({
                "episode/index": int(self._episode_idx),
                "episode/return": float(self._episode_return),
                "episode/len": int(self._episode_len),
            })
        log_wandb(self._wandb_run, payload, step=self.num_timesteps)
        self._last_wandb_step = int(self.num_timesteps)
        self._wandb_last_updates = int(self.agent.num_updates)

    def _maybe_train(self, episode_done: bool = False):
        train_freq, train_unit = self._parse_train_freq(self.params.get("train_freq", 1))
        self._collected_steps += 1
        if episode_done:
            self._collected_episodes += 1

        if train_unit == "step":
            should_train = self._collected_steps >= train_freq
        else:
            should_train = self._collected_episodes >= train_freq
        if not should_train:
            return None

        collected_steps = self._collected_steps
        self._collected_steps = 0
        self._collected_episodes = 0
        if self.num_timesteps <= self.cfg.learning_starts or self.replay_buffer.size == 0:
            return None

        gradient_steps = self.cfg.gradient_steps
        if gradient_steps == -1:
            gradient_steps = collected_steps
        if gradient_steps <= 0:
            return None
        train_start = time.perf_counter()
        self._last_metrics = self.agent.update(self.replay_buffer, gradient_steps, self.cfg.batch_size)
        self._wandb_train_seconds += time.perf_counter() - train_start
        if self._wandb_run is not None:
            for key, value in self._last_metrics.items():
                self._wandb_train_window.add_weighted(f"train/{key}", value, weight=gradient_steps)
        if self.verbose >= 2:
            print(f"Native SAC update @ step {self.num_timesteps}: {self._last_metrics}")
        return self._last_metrics

    def _maybe_checkpoint(self):
        if self._checkpoint is None:
            return
        # Preserve the legacy tuple shape for callers/tests which configure the
        # wrapper by assigning its old private field directly.
        if isinstance(self._checkpoint, tuple):
            save_freq, save_path, name_prefix = self._checkpoint
            if save_freq > 0 and self.num_timesteps > 0 and self.num_timesteps % save_freq == 0:
                self.save(save_path, f"{name_prefix}_{self.num_timesteps}_steps")
            return
        self._publish_checkpoint_targets(self._checkpoint.targets(self.num_timesteps))

    def _final_checkpoint(self):
        if isinstance(self._checkpoint, CheckpointTracker):
            self._publish_checkpoint_targets(
                self._checkpoint.targets(self.num_timesteps, final=True)
            )

    def _checkpoint_state(self):
        return {
            "checkpoint_version": 2,
            "agent": self.agent.state_dict(),
            "config": asdict(self.cfg),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "action_low": np.asarray(self.env.action_space.low, dtype=np.float32),
            "action_high": np.asarray(self.env.action_space.high, dtype=np.float32),
            "num_timesteps": self.num_timesteps,
            "metrics": self._last_metrics,
            "checkpoint_type": "weights_and_optimizers_without_replay",
        }

    def _publish_checkpoint_targets(self, targets):
        targets = tuple(targets)
        if not targets:
            return ()
        state = self._checkpoint_state()
        return publish_checkpoint(
            targets,
            lambda path: torch.save(state, path),
            extension=".pt",
        )

    def _seed_once(self):
        if self._rng_seeded:
            return
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            self.env.action_space.seed(self.seed)
            self.env.observation_space.seed(self.seed)
        except AttributeError:
            pass
        self._rng_seeded = True

    def learn(self, total_timesteps=10000, reset_num_timesteps=True):
        self._seed_once()
        total_timesteps = int(float(total_timesteps))
        if total_timesteps < 0:
            raise ValueError("total_timesteps must be non-negative.")
        if self._wandb_run is None:
            self._wandb_run = self._init_wandb()

        try:
            if reset_num_timesteps:
                self.num_timesteps = 0
                self._episode_idx = 0
                self._episode_return = 0.0
                self._episode_len = 0
                self._collected_steps = 0
                self._collected_episodes = 0
                self._last_metrics = {}
                if isinstance(self._checkpoint, CheckpointTracker):
                    self._checkpoint.reset()
                reset_seed = self.seed if not self._env_seeded else None
                obs = self._reset_env(seed=reset_seed)
                self._env_seeded = True
                target_timesteps = total_timesteps
            else:
                target_timesteps = self.num_timesteps + total_timesteps
                if self._last_obs is None:
                    reset_seed = self.seed if not self._env_seeded else None
                    obs = self._reset_env(seed=reset_seed)
                    self._env_seeded = True
                else:
                    obs = self._last_obs
            self._last_obs = obs
            self._reset_wandb_window()
        except Exception:
            finish_wandb(self._wandb_run)
            self._wandb_run = None
            raise

        try:
            # Match SB3's rollout-chunk semantics: once a train-frequency chunk has
            # started, finish it even if that slightly overshoots total_timesteps.
            while (
                self.num_timesteps < target_timesteps
                or self._collected_steps > 0
                or self._collected_episodes > 0
            ):
                obs_flat = self._flatten_obs(obs)
                if self.num_timesteps < self.cfg.learning_starts:
                    action_env = self.env.action_space.sample()
                    action_norm = self._scale_action(action_env)
                else:
                    action_norm = self.agent.act(obs_flat, deterministic=False)
                    action_env = self._unscale_action(action_norm)

                next_obs, reward, terminated, truncated, info = self.env.step(action_env)
                done = bool(terminated or truncated)
                next_obs_flat = self._flatten_obs(next_obs)
                self.replay_buffer.add(obs_flat, action_norm, reward, next_obs_flat, terminated, truncated)
                self.num_timesteps += 1
                self._episode_return += float(reward)
                self._episode_len += 1
                self._last_reward = float(reward)
                self._last_terminated = bool(terminated)
                self._last_truncated = bool(truncated)
                self._last_info = dict(info or {})
                self._wandb_reward_window.add_stats("rollout/reward", [reward])
                self._log_step(next_obs, reward, action_env, terminated, truncated, info)

                self._maybe_train(episode_done=done)
                self._log_wandb_step(
                    reward,
                    terminated,
                    truncated,
                    info,
                    completed_episode=done,
                )
                if done and isinstance(self._checkpoint, CheckpointTracker):
                    self._checkpoint.record_episode_return(self._episode_return)
                self._maybe_checkpoint()

                if done:
                    self._episode_idx += 1
                    self._episode_return = 0.0
                    self._episode_len = 0
                    obs = self._reset_env()
                else:
                    obs = next_obs
                self._last_obs = obs
            self._final_checkpoint()
            return self
        finally:
            if self._wandb_run is not None:
                self._log_wandb_step(
                    self._last_reward,
                    self._last_terminated,
                    self._last_truncated,
                    self._last_info,
                    force=True,
                )
                finish_wandb(self._wandb_run)
                self._wandb_run = None

    def predict(self, observation, deterministic=False):
        obs_flat = self._flatten_obs(observation)
        action_norm = self.agent.act(obs_flat, deterministic=deterministic)
        return self._unscale_action(action_norm), None

    def get_model(self):
        return self.agent

    def get_q_values(self, observation, action):
        obs_flat = self._flatten_obs(observation)
        action_norm = self._scale_action(action)
        return self.agent.q_values(obs_flat, action_norm)

    def set_checkpointing(
        self,
        save_freq,
        save_path,
        name_prefix,
        save_strat="all",
        checkpoint_best_window=100,
        trial_run_params=None,
        experiment_params=None,
    ):
        self._checkpoint = CheckpointTracker(
            save_freq,
            save_path,
            name_prefix,
            save_strat=save_strat,
            best_window=checkpoint_best_window,
            trial_run_params=self.run_params if trial_run_params is None else trial_run_params,
            experiment_params=(
                self.experiment_params if experiment_params is None else experiment_params
            ),
        )
        return self._checkpoint

    def save(self, path, name):
        os.makedirs(path, exist_ok=True)
        fp = Path(path) / name
        if isinstance(self._checkpoint, CheckpointTracker):
            target = self._checkpoint.explicit_target(
                fp,
                step=self.num_timesteps,
                episode=self._episode_idx,
            )
        else:
            target = explicit_checkpoint_target(
                fp,
                step=self.num_timesteps,
                episode=self._episode_idx,
                trial_run_params=self.run_params,
                experiment_params=self.experiment_params,
            )
        published = self._publish_checkpoint_targets((target,))
        return published[0]

    def load(self, path, *, resume=False, strict_config=True):
        checkpoint = torch.load(path, map_location=self.agent.device, weights_only=False)
        checkpoint_version = int(checkpoint.get("checkpoint_version", 1))
        if checkpoint_version not in {1, 2}:
            raise ValueError(
                f"Unsupported native SAC checkpoint_version={checkpoint_version}; "
                "supported versions are 1 and 2."
            )
        if int(checkpoint.get("obs_dim", self.obs_dim)) != self.obs_dim:
            raise ValueError("Checkpoint observation dimension does not match this environment.")
        if int(checkpoint.get("action_dim", self.action_dim)) != self.action_dim:
            raise ValueError("Checkpoint action dimension does not match this environment.")
        if "action_low" in checkpoint and not np.array_equal(
            np.asarray(checkpoint["action_low"]), np.asarray(self.env.action_space.low, dtype=np.float32)
        ):
            raise ValueError("Checkpoint action lower bounds do not match this environment.")
        if "action_high" in checkpoint and not np.array_equal(
            np.asarray(checkpoint["action_high"]), np.asarray(self.env.action_space.high, dtype=np.float32)
        ):
            raise ValueError("Checkpoint action upper bounds do not match this environment.")

        if strict_config and "config" in checkpoint:
            current = asdict(self.cfg)
            saved = checkpoint["config"]
            critical_keys = [
                "learning_rate", "tau", "gamma", "ent_coef", "target_entropy",
                "target_update_interval", "actor_net_arch", "critic_net_arch", "adam_eps",
            ]
            saved_representation = str(saved.get("q_representation", "scalar")).lower()
            current_representation = str(current["q_representation"]).lower()
            critical_keys.append("q_representation")
            if "distributional" in {saved_representation, current_representation}:
                critical_keys.extend(("q_num_bins", "q_vmin", "q_vmax"))

            legacy_defaults = {
                "q_representation": "scalar",
                "q_num_bins": 101,
                "q_vmin": -10.0,
                "q_vmax": 10.0,
            }
            mismatches = {
                key: (saved.get(key, legacy_defaults.get(key)), current.get(key))
                for key in critical_keys
                if saved.get(key, legacy_defaults.get(key)) != current.get(key)
            }
            if mismatches:
                raise ValueError(f"Checkpoint SAC configuration mismatch: {mismatches}")

        if resume:
            raise ValueError(
                "Native SAC checkpoints do not include replay or environment state and cannot safely resume. "
                "Load for evaluation/fine-tuning with resume=False."
            )
        self.agent.load_state_dict(checkpoint["agent"])
        self.num_timesteps = 0
        self._last_metrics = checkpoint.get("metrics", {})
        self._last_obs = None
        self._episode_return = 0.0
        self._episode_len = 0
        self._collected_steps = 0
        self._collected_episodes = 0
        return self


NativeSAC = SAC
SACBaseline = SAC
