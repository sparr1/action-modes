import copy
import os
import random
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
    "rho": 0.5,
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
    "episodic": True,
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
        self.agent = TDMPC2(self.cfg)
        self.buffer = Buffer(self.cfg)
        self._checkpointing = None
        self._global_step = 0
        self._episode_idx = 0
        self._pretrained = False

        print("Architecture:", self.agent.model)

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
            obs_dim = flatdim(self._obs_space)
        elif isinstance(self._obs_space, gym.spaces.Box):
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
        self.agent._prev_mean.zero_()
        return obs, info

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

        if getattr(self.alg_logger, "_log_info", False):
            data = setup_logs(reward, obs_for_log, action_for_log, [done], [info_for_log])
        else:
            data = setup_logs(reward, obs_for_log, action_for_log, [done])
        self.alg_logger.on_step(data)

    def _maybe_checkpoint(self):
        if not self._checkpointing:
            return
        save_freq, save_path, name_prefix = self._checkpointing
        if self._global_step > 0 and self._global_step % save_freq == 0:
            self.save(save_path, f"{name_prefix}_{self._global_step}")

    def learn(self, total_timesteps=10000):
        total_timesteps = int(float(total_timesteps))
        self.cfg.steps = total_timesteps

        obs, _ = self._reset_env(seed=self.cfg.seed)
        episode_tds = [self._to_td(obs)]
        episode_step = 0

        while self._global_step < total_timesteps:
            if self._global_step < self.cfg.seed_steps or self.buffer.num_eps == 0:
                action_norm = self._random_action_norm()
            else:
                obs_t = self._obs_to_tensor(obs)
                action_norm = self.agent.act(obs_t, t0=(episode_step == 0), eval_mode=False).numpy()

            action_env = self._unscale_action(action_norm)
            next_obs, reward, terminated, truncated, info = self.env.step(action_env)
            done = bool(terminated or truncated)
            true_terminated = bool(terminated)

            episode_tds.append(self._to_td(next_obs, action_norm, reward, true_terminated))
            self._global_step += 1
            episode_step += 1
            self._log_step(reward, next_obs, action_env, terminated, truncated, info)

            if done:
                if true_terminated and not self.cfg.episodic:
                    raise ValueError("TD-MPC2 saw terminated=True while episodic=False. Set alg_params.episodic=true or disable true terminations in the env.")
                self.buffer.add(torch.cat(episode_tds))
                self._episode_idx += 1
                obs, _ = self._reset_env()
                episode_tds = [self._to_td(obs)]
                episode_step = 0
            else:
                obs = next_obs

            if self._global_step >= self.cfg.seed_steps and self.buffer.num_eps > 0:
                num_updates = self.cfg.pretrain_steps if not self._pretrained else self.cfg.utd
                if not self._pretrained:
                    print("Pretraining TD-MPC2 on seed data...")
                    self._pretrained = True
                train_metrics = None
                for _ in range(num_updates):
                    train_metrics = self.agent.update(self.buffer)
                self._last_train_metrics = train_metrics

            self._maybe_checkpoint()

        return self

    def predict(self, observation, deterministic=True):
        obs_t = self._obs_to_tensor(observation)
        action_norm = self.agent.act(obs_t, t0=False, eval_mode=deterministic).numpy()
        return self._unscale_action(action_norm), None

    def save(self, path, name):
        os.makedirs(path, exist_ok=True)
        self.agent.save(os.path.join(path, name))

    def load(self, path):
        self.agent.load(path)
        return self

    def set_checkpointing(self, save_freq, save_path, name_prefix):
        self._checkpointing = (int(save_freq), save_path, name_prefix)
