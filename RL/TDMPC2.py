import copy
import os
import random
import time
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.utils import flatdim, flatten
try:
    from tensordict import TensorDict
except ImportError:  # tensordict<newer API compatibility
    from tensordict.tensordict import TensorDict

from RL.alg import Algorithm
from RL.tdmpc2_core.agent import TDMPC2
from RL.tdmpc2_core.common.buffer import Buffer
from RL.tdmpc2_core.common.device import resolve_device
from utils.utils import setup_logs
from utils.wandb_utils import (
    WandbAccumulator,
    extract_reward_components,
    finish_wandb,
    init_wandb,
    log_wandb,
)


_MODEL_SIZE = {
    1: {"enc_dim": 256, "mlp_dim": 384, "latent_dim": 128, "num_enc_layers": 2, "num_q": 2},
    5: {"enc_dim": 256, "mlp_dim": 512, "latent_dim": 512, "num_enc_layers": 2, "num_q": 5},
    19: {"enc_dim": 1024, "mlp_dim": 1024, "latent_dim": 768, "num_enc_layers": 3, "num_q": 5},
    48: {"enc_dim": 1792, "mlp_dim": 1792, "latent_dim": 768, "num_enc_layers": 4, "num_q": 5},
    317: {"enc_dim": 4096, "mlp_dim": 4096, "latent_dim": 1376, "num_enc_layers": 5, "num_q": 8},
}


_DEFAULTS = {
    # training
    "steps": 1_000_000,
    "batch_size": 256,
    "reward_coef": 0.1,
    "value_coef": 0.1,
    "termination_coef": 1.0,
    "consistency_coef": 20.0,
    "rho": 0.7,
    "lr": 3e-4,
    "enc_lr_scale": 0.3,
    "grad_clip_norm": 20.0,
    "tau": 0.01,
    "discount_denom": 5,
    "discount_min": 0.95,
    "discount_max": 0.995,
    "buffer_size": 1_000_000,
    "utd": 1,
    "pretrain_steps": None,

    # planning
    "mpc": True,
    "iterations": 6,
    "num_samples": 512,
    "num_elites": 64,
    "num_pi_trajs": 24,
    "horizon": 3,
    "min_std": 0.05,
    "max_std": 2.0,
    "temperature": 0.5,

    # actor / critic
    "log_std_min": -10,
    "log_std_max": 2,
    "entropy_coef": 1e-4,
    "num_bins": 101,
    "vmin": -10,
    "vmax": 10,

    # architecture, 5M single-task default
    "model_size": 5,
    "num_enc_layers": 2,
    "enc_dim": 256,
    "num_channels": 32,
    "mlp_dim": 512,
    "latent_dim": 512,
    "task_dim": 0,
    "num_q": 5,
    "dropout": 0.01,
    "simnorm_dim": 8,

    # misc
    "obs": "state",
    "episodic": False,
    "compile": False,
    "seed": 1,
    "device": "auto",
}


class _TDMPC2Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class TDMPC2Baseline(Algorithm):
    """
    AMBI wrapper for single-task TD-MPC2.

    This intentionally uses AMBI's already-created Gymnasium environment and
    only adapts the stepping / logging / save interface around the official
    TD-MPC2 agent, world model, and replay buffer.
    """

    def __init__(self, name, env, custom_params=None, run_params=None, experiment_params=None):
        super().__init__(name, env, custom_params)
        self.run_params = run_params or {}
        self.experiment_params = experiment_params or {}
        self.cfg = self._build_cfg(custom_params or {})

        self._set_seed(self.cfg.seed)
        self.agent = self._make_agent(self.cfg)
        self.buffer = Buffer(self.cfg)
        self._predict_t0 = True
        self._checkpointing = None
        self._global_step = 0
        self._episode_idx = 0
        self._episode_return = 0.0
        self._episode_len = 0
        self._pretrained = False
        self._last_train_metrics = None
        self._wandb_every = max(1, int((custom_params or {}).get("wandb_step_every", 1000)))
        self._wandb_run = None
        self._wandb_train_window = WandbAccumulator()
        self._wandb_reward_window = WandbAccumulator()
        self._wandb_start_time = None
        self._wandb_window_start_time = None
        self._wandb_window_start_step = 0
        self._wandb_train_seconds = 0.0
        self._wandb_planner_seconds = 0.0
        self._wandb_last_updates = 0
        self._num_updates = 0
        self._last_wandb_step = None
        self._last_reward = 0.0
        self._last_terminated = False
        self._last_truncated = False
        self._last_info = {}

        print("Architecture:", self.agent.model)

    def _make_agent(self, cfg):
        """Factory hook used by TD-MPC2-derived algorithms."""
        return TDMPC2(cfg)

    def _wandb_run_name(self):
        return f"TDMPC2-{self.run_params.get('env', 'env')}-seed{self.cfg.seed}"

    def _init_wandb(self):
        return init_wandb(
            self.custom_params or {},
            default_project="ambi",
            run_name=self._wandb_run_name(),
            config={
                "algorithm": self.__class__.__name__,
                "run_params": self.run_params,
                "alg_params": self.custom_params or {},
                "config": vars(self.cfg),
            },
        )

    def _build_cfg(self, params):
        cfg = copy.deepcopy(_DEFAULTS)
        cfg.update(copy.deepcopy(params))

        run_device = self.run_params.get("device", None)
        if "device" not in params and run_device is not None:
            cfg["device"] = run_device
        cfg["device"] = str(resolve_device(cfg["device"]))

        cfg["seed"] = int(self.run_params.get("seed", cfg.get("seed", 1)))
        cfg["steps"] = int(float(self.run_params.get("total_steps", cfg.get("steps", 1_000_000))))

        self._obs_space = self.env.observation_space
        if isinstance(self._obs_space, gym.spaces.Dict):
            image_like = [
                key for key, space in self._obs_space.spaces.items()
                if isinstance(space, gym.spaces.Box) and len(space.shape) > 1
            ]
            if image_like:
                raise NotImplementedError(
                    "This TD-MPC2 wrapper supports vector state observations only; "
                    f"image-like Dict entries are not supported: {image_like}."
                )
            obs_dim = flatdim(self._obs_space)
        elif isinstance(self._obs_space, gym.spaces.Box):
            if len(self._obs_space.shape) != 1:
                raise NotImplementedError(
                    "This TD-MPC2 wrapper supports vector state observations only; "
                    f"got observation shape {self._obs_space.shape}."
                )
            obs_dim = int(np.prod(self._obs_space.shape))
        else:
            raise NotImplementedError("TD-MPC2Baseline only supports Box or Dict observation spaces.")

        if not isinstance(self.env.action_space, gym.spaces.Box):
            raise NotImplementedError("TD-MPC2Baseline only supports continuous Box action spaces.")
        self._action_shape = self.env.action_space.shape
        self._action_low = self.env.action_space.low.astype(np.float32).reshape(-1)
        self._action_high = self.env.action_space.high.astype(np.float32).reshape(-1)
        if not np.all(np.isfinite(self._action_low)) or not np.all(np.isfinite(self._action_high)):
            raise ValueError("TD-MPC2Baseline requires finite action-space bounds.")
        action_dim = int(np.prod(self._action_shape))

        model_size = cfg.get("model_size", None)
        if model_size is not None:
            model_size = int(model_size)
            if model_size not in _MODEL_SIZE:
                raise ValueError(f"Invalid TD-MPC2 model_size={model_size}. Expected one of {list(_MODEL_SIZE)}.")
            cfg.update(copy.deepcopy(_MODEL_SIZE[model_size]))
            cfg["model_size"] = model_size

        episode_length = cfg.get("episode_length", None) or self._infer_episode_length()
        cfg["episode_length"] = int(episode_length)
        cfg["episode_lengths"] = [cfg["episode_length"]]

        if cfg.get("seed_steps", None) is None:
            cfg["seed_steps"] = max(5 * cfg["episode_length"], 1000)
        cfg["seed_steps"] = int(cfg["seed_steps"])
        if cfg.get("pretrain_steps", None) is None:
            cfg["pretrain_steps"] = cfg["seed_steps"]
        cfg["pretrain_steps"] = int(cfg["pretrain_steps"])
        cfg["utd"] = int(cfg.get("utd", 1))

        # Allow a simple fixed discount override in alg_params, e.g. "discount": 0.99.
        if "discount" in cfg and cfg["discount"] is not None:
            cfg["discount_min"] = float(cfg["discount"])
            cfg["discount_max"] = float(cfg["discount"])

        cfg["obs"] = "state"
        cfg["obs_shape"] = {"state": (obs_dim,)}
        cfg["action_dim"] = action_dim
        cfg["multitask"] = False
        cfg["tasks"] = [self.run_params.get("env", "ambi-task")]
        cfg["task_dim"] = 0
        cfg["obs_shapes"] = [cfg["obs_shape"]]
        cfg["action_dims"] = [action_dim]
        cfg["bin_size"] = (cfg["vmax"] - cfg["vmin"]) / (cfg["num_bins"] - 1)

        return _TDMPC2Config(**cfg)

    def _infer_episode_length(self):
        if getattr(self.env, "spec", None) is not None and self.env.spec is not None:
            if getattr(self.env.spec, "max_episode_steps", None) is not None:
                return self.env.spec.max_episode_steps
        if hasattr(self.env, "get_wrapper_attr"):
            try:
                return self.env.get_wrapper_attr("_max_episode_steps")
            except Exception:
                pass
        return self.custom_params.get("max_episode_steps", 1000) if self.custom_params else 1000

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if str(self.cfg.device).startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        try:
            self.env.action_space.seed(seed)
            self.env.observation_space.seed(seed)
        except Exception:
            pass

    def _obs_to_numpy(self, obs):
        if isinstance(self._obs_space, gym.spaces.Dict):
            obs = flatten(self._obs_space, obs)
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _obs_to_tensor(self, obs):
        return torch.as_tensor(self._obs_to_numpy(obs), dtype=torch.float32)

    def _scale_action(self, action_env):
        action_env = np.asarray(action_env, dtype=np.float32).reshape(-1)
        action_norm = 2.0 * (action_env - self._action_low) / (self._action_high - self._action_low) - 1.0
        return np.clip(action_norm, -1.0, 1.0).astype(np.float32)

    def _unscale_action(self, action_norm):
        action_norm = np.asarray(action_norm, dtype=np.float32).reshape(-1)
        action_env = self._action_low + 0.5 * (action_norm + 1.0) * (self._action_high - self._action_low)
        action_env = np.clip(action_env, self._action_low, self._action_high).astype(np.float32)
        return action_env.reshape(self._action_shape)

    def _random_action_norm(self):
        return self._scale_action(self.env.action_space.sample())

    def _to_td(self, obs, action=None, reward=None, terminated=None):
        obs_t = self._obs_to_tensor(obs).unsqueeze(0).cpu()
        if action is None:
            action_t = torch.full((self.cfg.action_dim,), float("nan"), dtype=torch.float32)
        else:
            action_t = torch.as_tensor(action, dtype=torch.float32).reshape(self.cfg.action_dim)
        reward_t = torch.tensor(float("nan") if reward is None else float(reward), dtype=torch.float32)
        terminated_t = torch.tensor(float("nan") if terminated is None else float(terminated), dtype=torch.float32)
        return TensorDict(
            {
                "obs": obs_t,
                "action": action_t.unsqueeze(0),
                "reward": reward_t.unsqueeze(0),
                "terminated": terminated_t.unsqueeze(0),
            },
            batch_size=(1,),
        )

    def _reset_env(self, seed=None):
        if seed is None:
            obs, info = self.env.reset()
        else:
            obs, info = self.env.reset(seed=seed)
        if hasattr(self.agent, "reset"):
            self.agent.reset()
        elif hasattr(self.agent, "_prev_mean"):
            self.agent._prev_mean.zero_()
        self._predict_t0 = True
        return obs, info

    def reset(self):
        """Reset episode-local controller state before external evaluation."""
        if hasattr(self.agent, "reset"):
            self.agent.reset()
        elif hasattr(self.agent, "_prev_mean"):
            self.agent._prev_mean.zero_()
        self._predict_t0 = True

    def _log_step(self, reward, obs, action, terminated, truncated, info):
        if not self.alg_logger:
            return

        done = bool(terminated or truncated)
        info_for_log = dict(info or {})
        info_for_log.setdefault("terminated", bool(terminated))
        info_for_log.setdefault("truncated", bool(truncated))

        # AMBI's logger expects a vectorized-env-like leading dimension.
        obs_for_log = obs if isinstance(obs, dict) else np.asarray(obs)[None, ...]
        action_for_log = np.asarray(action)[None, ...]

        data = setup_logs(reward, obs_for_log, action_for_log, [done], [info_for_log])
        self.alg_logger.on_step(data)

    def _metrics_to_floats(self, metrics):
        if metrics is None:
            return {}
        out = {}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                out[key] = float(value.detach().cpu().mean())
            else:
                try:
                    out[key] = float(value)
                except (TypeError, ValueError):
                    pass
        return out

    def _reset_wandb_window(self):
        self._wandb_train_window.clear()
        self._wandb_reward_window.clear()
        now = time.perf_counter()
        self._wandb_start_time = now
        self._wandb_window_start_time = now
        self._wandb_window_start_step = int(self._global_step)
        self._wandb_train_seconds = 0.0
        self._wandb_planner_seconds = 0.0
        self._wandb_last_updates = int(self._num_updates)
        self._last_wandb_step = None

    def _accumulate_train_metrics(self, metrics):
        metrics = self._metrics_to_floats(metrics)
        for key, value in metrics.items():
            if key == "num_updates":
                continue
            self._wandb_train_window.add_weighted(f"train/{key}", value)

    def _record_action_metrics(self, *, planned, action_seconds):
        if not planned or not bool(self.cfg.mpc):
            return
        self._wandb_planner_seconds += float(action_seconds)
        plan_metrics = getattr(self.agent, "last_plan_metrics", {}) or {}
        for key, value in self._metrics_to_floats(plan_metrics).items():
            if key == "planner_seconds":
                continue
            metric_key = key if key.startswith("planner_") else f"planner_{key}"
            self._wandb_train_window.add_weighted(f"train/{metric_key}", value)

    def _replay_wandb_payload(self):
        transitions = int(getattr(self.buffer, "num_transitions", 0))
        capacity = int(getattr(self.buffer, "capacity", 0))
        fill_ratio = float(getattr(self.buffer, "fill_fraction", 0.0))
        return {
            "train/replay_size": transitions,
            "train/replay_capacity": capacity,
            "train/replay_fill_ratio": fill_ratio,
            "train/buffer_episodes": int(self.buffer.num_eps),
        }

    def _timing_wandb_payload(self, updates_since_log):
        now = time.perf_counter()
        start = self._wandb_start_time if self._wandb_start_time is not None else now
        window_start = self._wandb_window_start_time if self._wandb_window_start_time is not None else now
        window_seconds = max(0.0, now - window_start)
        window_steps = max(0, int(self._global_step) - int(self._wandb_window_start_step))
        payload = {
            "time/window_seconds": float(window_seconds),
            "time/time_elapsed": float(max(0.0, now - start)),
            "time/total_timesteps": int(self._global_step),
            "time/fps": float(window_steps / window_seconds) if window_seconds > 0 else 0.0,
            "time/train_seconds": float(self._wandb_train_seconds),
            "time/updates_per_second": (
                float(updates_since_log / self._wandb_train_seconds)
                if self._wandb_train_seconds > 0 else 0.0
            ),
            "time/planner_seconds": float(self._wandb_planner_seconds),
        }
        self._wandb_window_start_time = now
        self._wandb_window_start_step = int(self._global_step)
        self._wandb_train_seconds = 0.0
        self._wandb_planner_seconds = 0.0
        return payload

    def _extra_wandb_payload(self, updates_since_log):
        del updates_since_log
        return {}

    def _log_wandb_step(
        self,
        reward,
        terminated,
        truncated,
        info=None,
        *,
        completed_episode=False,
        force=False,
    ):
        if self._wandb_run is None:
            return
        done = bool(terminated or truncated)
        if not force and (self._global_step % self._wandb_every != 0) and not done:
            return
        if (
            force
            and not self._wandb_train_window
            and not self._wandb_reward_window
            and (
                self._last_wandb_step == self._global_step
                or self._global_step == self._wandb_window_start_step
            )
        ):
            return

        updates_since_log = int(self._num_updates - self._wandb_last_updates)
        payload = {
            "train/reward": float(reward),
            "train/done": int(done),
            "train/terminated": int(bool(terminated)),
            "train/truncated": int(bool(truncated)),
            "train/learning_started": int(self._global_step > self.cfg.seed_steps and self.buffer.num_eps > 0),
            "train/n_updates": int(self._num_updates),
            "train/updates_since_log": updates_since_log,
            "episode/current_return": float(self._episode_return),
            "episode/current_len": int(self._episode_len),
        }
        payload.update(self._replay_wandb_payload())
        payload.update(self._wandb_reward_window.pop())
        payload.update(self._wandb_train_window.pop())
        payload.update(extract_reward_components(info or {}))
        payload.update(self._timing_wandb_payload(updates_since_log))
        payload.update(self._extra_wandb_payload(updates_since_log))
        if completed_episode:
            payload.update({
                "episode/index": int(self._episode_idx),
                "episode/return": float(self._episode_return),
                "episode/len": int(self._episode_len),
            })
        log_wandb(self._wandb_run, payload, step=self._global_step)
        self._last_wandb_step = int(self._global_step)
        self._wandb_last_updates = int(self._num_updates)

    def _maybe_checkpoint(self):
        if not self._checkpointing:
            return
        save_freq, save_path, name_prefix = self._checkpointing
        if self._global_step > 0 and self._global_step % save_freq == 0:
            self.save(save_path, f"{name_prefix}_{self._global_step}")

    def learn(self, total_timesteps=10000):
        total_timesteps = int(float(total_timesteps))
        if total_timesteps < 0:
            raise ValueError("total_timesteps must be non-negative.")
        if total_timesteps != int(self.cfg.steps):
            if self._global_step != 0 or self.buffer.num_eps != 0:
                raise ValueError(
                    "Cannot change total_timesteps after TD-MPC2 replay collection has started; "
                    "construct a new learner with the intended step budget."
                )
            self.cfg.steps = total_timesteps
            self.buffer = Buffer(self.cfg)
        self.cfg.steps = total_timesteps
        if self._wandb_run is None:
            self._wandb_run = self._init_wandb()
        self._reset_wandb_window()

        try:
            obs, _ = self._reset_env(seed=self.cfg.seed)
            episode_tds = [self._to_td(obs)]
            episode_step = 0

            while self._global_step < total_timesteps:
                planned = not (self._global_step <= self.cfg.seed_steps or self.buffer.num_eps == 0)
                action_start = time.perf_counter()
                if not planned:
                    action_norm = self._random_action_norm()
                else:
                    obs_t = self._obs_to_tensor(obs)
                    action_norm = self.agent.act(
                        obs_t,
                        t0=(episode_step == 0),
                        eval_mode=False,
                    ).numpy()
                self._record_action_metrics(
                    planned=planned,
                    action_seconds=time.perf_counter() - action_start,
                )

                action_env = self._unscale_action(action_norm)
                next_obs, reward, terminated, truncated, info = self.env.step(action_env)
                done = bool(terminated or truncated)
                true_terminated = bool(terminated)

                episode_tds.append(self._to_td(next_obs, action_norm, reward, true_terminated))
                self._global_step += 1
                episode_step += 1
                self._episode_return += float(reward)
                self._episode_len += 1
                self._last_reward = float(reward)
                self._last_terminated = bool(terminated)
                self._last_truncated = bool(truncated)
                self._last_info = dict(info or {})
                self._wandb_reward_window.add_stats("rollout/reward", [reward])
                self._log_step(reward, next_obs, action_env, terminated, truncated, info)

                if done:
                    if true_terminated and not self.cfg.episodic:
                        raise ValueError(
                            "TD-MPC2 saw terminated=True while episodic=False. "
                            "Set alg_params.episodic=true or disable true terminations in the env."
                        )
                    self.buffer.add(torch.cat(episode_tds))

                if self._global_step > self.cfg.seed_steps and self.buffer.num_eps > 0:
                    num_updates = self.cfg.pretrain_steps if not self._pretrained else self.cfg.utd
                    if not self._pretrained:
                        print("Pretraining TD-MPC2 on seed data...")
                        self._pretrained = True
                    burst_metrics = WandbAccumulator()
                    train_start = time.perf_counter()
                    for _ in range(num_updates):
                        train_metrics = self.agent.update(self.buffer)
                        self._num_updates += 1
                        metrics_floats = self._metrics_to_floats(train_metrics)
                        burst_metrics.update_weighted(metrics_floats)
                        self._accumulate_train_metrics(metrics_floats)
                    self._wandb_train_seconds += time.perf_counter() - train_start
                    self._last_train_metrics = burst_metrics.snapshot()

                self._log_wandb_step(
                    reward,
                    terminated,
                    truncated,
                    info,
                    completed_episode=done,
                )
                self._maybe_checkpoint()

                if done:
                    self._episode_idx += 1
                    self._episode_return = 0.0
                    self._episode_len = 0
                    obs, _ = self._reset_env()
                    episode_tds = [self._to_td(obs)]
                    episode_step = 0
                else:
                    obs = next_obs
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

    def predict(self, observation, deterministic=True, episode_start=None):
        t0 = self._predict_t0 if episode_start is None else bool(episode_start)
        if t0 and hasattr(self.agent, "reset"):
            self.agent.reset()
        obs_t = self._obs_to_tensor(observation)
        action_norm = self.agent.act(obs_t, t0=t0, eval_mode=deterministic).numpy()
        self._predict_t0 = False
        return self._unscale_action(action_norm), None

    def save(self, path, name):
        os.makedirs(path, exist_ok=True)
        self.agent.save(os.path.join(path, name))

    def load(self, path):
        self.agent.load(path)
        self._num_updates = int(getattr(self.agent, "num_updates", self._num_updates))
        return self

    def set_checkpointing(self, save_freq, save_path, name_prefix):
        self._checkpointing = (int(save_freq), save_path, name_prefix)
