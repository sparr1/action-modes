import copy
import math
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
from RL.tdmpc2_core import MODEL_SIZE
from RL.tdmpc2_core.agent import TDMPC2
from RL.tdmpc2_core.common.buffer import Buffer
from RL.tdmpc2_core.common.checkpoint import AsyncCheckpointWriter
from RL.tdmpc2_core.common.device import resolve_device
from utils.checkpointing import (
    CheckpointTracker,
    explicit_checkpoint_target,
)
from utils.utils import setup_logs
from utils.wandb_utils import (
    WandbAccumulator,
    extract_reward_components,
    finish_wandb,
    init_wandb,
    log_wandb,
)


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
    **MODEL_SIZE[5],
    "num_channels": 32,
    "task_dim": 0,
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


class _DeviceMeanAccumulator:
    """Pool scalar moments without synchronizing or launching per-key kernels."""

    def __init__(self):
        self.clear()

    def clear(self):
        self._key_order = []
        self._known_keys = set()
        self._python_values = {}
        self._tensor_groups = {}
        self._tensor_key_groups = {}

    def __bool__(self):
        return bool(self._key_order)

    @staticmethod
    def _scalar(value):
        if torch.is_tensor(value):
            value = value.detach()
            if value.numel() == 0 or value.is_complex():
                return None
            if not value.is_floating_point():
                value = value.float()
            elif value.dtype in (torch.float16, torch.bfloat16):
                value = value.float()
            return value.mean() if value.numel() != 1 else value.reshape(())
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return value if math.isfinite(value) else None

    def _register_key(self, key):
        if key not in self._known_keys:
            self._known_keys.add(key)
            self._key_order.append(key)

    @staticmethod
    def _new_tensor_group(keys, *, device, dtype):
        size = len(keys)
        return {
            "keys": list(keys),
            "index": {key: index for index, key in enumerate(keys)},
            "count": torch.zeros(size, dtype=torch.int64, device=device),
            "mean": torch.zeros(size, dtype=dtype, device=device),
            "m2": torch.zeros(size, dtype=dtype, device=device),
            "minimum": torch.full(
                (size,), float("inf"), dtype=dtype, device=device
            ),
            "maximum": torch.full(
                (size,), -float("inf"), dtype=dtype, device=device
            ),
        }

    def _ensure_tensor_keys(self, signature, keys):
        device, dtype = signature
        group = self._tensor_groups.get(signature)
        if group is None:
            group = self._new_tensor_group(keys, device=device, dtype=dtype)
            self._tensor_groups[signature] = group
            for key in keys:
                self._tensor_key_groups[key] = signature
            return group

        new_keys = [key for key in keys if key not in group["index"]]
        if not new_keys:
            return group

        old_size = len(group["keys"])
        group["keys"].extend(new_keys)
        group["index"].update(
            {key: old_size + offset for offset, key in enumerate(new_keys)}
        )
        new_count = len(new_keys)
        group["count"] = torch.cat(
            (
                group["count"],
                torch.zeros(new_count, dtype=torch.int64, device=device),
            )
        )
        for field, fill in (
            ("mean", 0.0),
            ("m2", 0.0),
            ("minimum", float("inf")),
            ("maximum", -float("inf")),
        ):
            group[field] = torch.cat(
                (
                    group[field],
                    torch.full(
                        (new_count,), fill, dtype=dtype, device=device
                    ),
                )
            )
        for key in new_keys:
            self._tensor_key_groups[key] = signature
        return group

    def _promote_python_key(self, key, signature):
        moments = self._python_values.pop(key)
        group = self._ensure_tensor_keys(signature, (key,))
        index = group["index"][key]
        count, mean, m2, minimum, maximum = moments
        group["count"][index] = count
        group["mean"][index] = mean
        group["m2"][index] = m2
        group["minimum"][index] = minimum
        group["maximum"][index] = maximum

    def _add_python(self, key, value):
        self._register_key(key)
        current = self._python_values.get(key)
        if current is None:
            self._python_values[key] = (1, value, 0.0, value, value)
            return
        count, mean, m2, minimum, maximum = current
        new_count = count + 1
        delta = value - mean
        new_mean = mean + delta / new_count
        self._python_values[key] = (
            new_count,
            new_mean,
            m2 + delta * (value - new_mean),
            min(minimum, value),
            max(maximum, value),
        )

    @staticmethod
    def _welford_update(count, mean, m2, values, finite):
        """Update one packed vector of population moments without cancellation."""
        incoming_count = finite.to(dtype=torch.int64)
        new_count = count + incoming_count
        safe_count = new_count.clamp_min(1).to(dtype=mean.dtype)
        safe_values = torch.where(finite, values, torch.zeros_like(values))
        delta = safe_values - mean
        new_mean = mean + torch.where(
            finite,
            delta / safe_count,
            torch.zeros_like(delta),
        )
        new_m2 = m2 + torch.where(
            finite,
            delta * (safe_values - new_mean),
            torch.zeros_like(delta),
        )
        return new_count, new_mean, new_m2

    @staticmethod
    def _update_tensor_group(group, items):
        values = torch.stack([value for _, value in items])
        finite = torch.isfinite(values)
        incoming_minimum = torch.where(
            finite, values, torch.full_like(values, float("inf"))
        )
        incoming_maximum = torch.where(
            finite, values, torch.full_like(values, -float("inf"))
        )
        indices_list = [group["index"][key] for key, _ in items]
        full_group = len(items) == len(group["keys"]) and all(
            index == expected for expected, index in enumerate(indices_list)
        )
        if full_group:
            new_count, new_mean, new_m2 = _DeviceMeanAccumulator._welford_update(
                group["count"], group["mean"], group["m2"], values, finite
            )
            group["count"].copy_(new_count)
            group["mean"].copy_(new_mean)
            group["m2"].copy_(new_m2)
            torch.minimum(
                group["minimum"], incoming_minimum, out=group["minimum"]
            )
            torch.maximum(
                group["maximum"], incoming_maximum, out=group["maximum"]
            )
            return

        indices = torch.as_tensor(
            indices_list, dtype=torch.long, device=values.device
        )
        new_count, new_mean, new_m2 = _DeviceMeanAccumulator._welford_update(
            group["count"].index_select(0, indices),
            group["mean"].index_select(0, indices),
            group["m2"].index_select(0, indices),
            values,
            finite,
        )
        group["count"].index_copy_(0, indices, new_count)
        group["mean"].index_copy_(0, indices, new_mean)
        group["m2"].index_copy_(0, indices, new_m2)
        selected_minimum = torch.minimum(
            group["minimum"].index_select(0, indices), incoming_minimum
        )
        selected_maximum = torch.maximum(
            group["maximum"].index_select(0, indices), incoming_maximum
        )
        group["minimum"].index_copy_(0, indices, selected_minimum)
        group["maximum"].index_copy_(0, indices, selected_maximum)

    def add(self, key, value):
        self.update({key: value})

    def update(self, metrics, *, prefix="", skip=()):
        if metrics is None:
            return
        skip = set(skip)
        tensor_items = {}
        for key, value in metrics.items():
            if key in skip:
                continue
            key = f"{prefix}{key}"
            value = self._scalar(value)
            if value is None:
                continue
            self._register_key(key)
            if torch.is_tensor(value):
                signature = self._tensor_key_groups.get(key)
                if signature is None:
                    signature = (value.device, value.dtype)
                    if key in self._python_values:
                        self._promote_python_key(key, signature)
                else:
                    device, dtype = signature
                    value = value.to(device=device, dtype=dtype)
                tensor_items.setdefault(signature, []).append((key, value))
                continue

            signature = self._tensor_key_groups.get(key)
            if signature is not None:
                device, dtype = signature
                tensor_value = torch.as_tensor(
                    value, device=device, dtype=dtype
                )
                tensor_items.setdefault(signature, []).append(
                    (key, tensor_value)
                )
            else:
                self._add_python(key, value)

        for signature, items in tensor_items.items():
            group = self._ensure_tensor_keys(
                signature, tuple(key for key, _ in items)
            )
            self._update_tensor_group(group, items)

    def snapshot(self):
        return self._payload(include_stats=False)

    def _payload(self, *, include_stats):
        tensor_summaries = {}
        for signature, group in self._tensor_groups.items():
            count = group["count"]
            mean_state = group["mean"]
            safe_count = count.clamp_min(1).to(dtype=mean_state.dtype)
            valid = count > 0
            mean = torch.where(
                valid,
                mean_state,
                torch.full_like(mean_state, float("nan")),
            )
            summary = {"mean": mean}
            if include_stats:
                count_value = count.to(dtype=mean_state.dtype)
                variance = (group["m2"] / safe_count).clamp_min(0)
                summary.update(
                    count=torch.where(
                        valid,
                        count_value,
                        torch.full_like(count_value, float("nan")),
                    ),
                    std=torch.where(
                        valid,
                        variance.sqrt(),
                        torch.full_like(variance, float("nan")),
                    ),
                    minimum=group["minimum"],
                    maximum=group["maximum"],
                )
            tensor_summaries[signature] = summary

        payload = {}
        for key in self._key_order:
            if key in self._python_values:
                count, mean, m2, minimum, maximum = (
                    self._python_values[key]
                )
                payload[key] = mean
                if include_stats:
                    variance = max(0.0, m2 / count)
                    payload.update(
                        {
                            f"{key}_count": float(count),
                            f"{key}_mean": mean,
                            f"{key}_std": math.sqrt(variance),
                            f"{key}_min": minimum,
                            f"{key}_max": maximum,
                        }
                    )
                continue

            signature = self._tensor_key_groups[key]
            group = self._tensor_groups[signature]
            index = group["index"][key]
            summary = tensor_summaries[signature]
            mean = summary["mean"][index]
            payload[key] = mean
            if include_stats:
                payload.update(
                    {
                        f"{key}_count": summary["count"][index],
                        f"{key}_mean": mean,
                        f"{key}_std": summary["std"][index],
                        f"{key}_min": summary["minimum"][index],
                        f"{key}_max": summary["maximum"][index],
                    }
                )
        return payload

    @staticmethod
    def _packed_floats(values):
        output = {}
        tensor_groups = {}
        for key, value in values.items():
            if torch.is_tensor(value):
                tensor_groups.setdefault(
                    (value.device, value.dtype), []
                ).append((key, value))
            else:
                value = float(value)
                if math.isfinite(value):
                    output[key] = value
        for items in tensor_groups.values():
            packed = torch.stack([value for _, value in items]).cpu().tolist()
            for (key, _), value in zip(items, packed):
                if math.isfinite(value):
                    output[key] = value
        return output

    def floats(self, *, clear=False, include_stats=False):
        output = self._packed_floats(
            self._payload(include_stats=include_stats)
        )
        if clear:
            self.clear()
        return output

    def pop_floats(self, *, include_stats=False):
        return self.floats(clear=True, include_stats=include_stats)


class TDMPC2Baseline(Algorithm):
    """
    AMBI wrapper for single-task TD-MPC2.

    This intentionally uses AMBI's already-created Gymnasium environment and
    only adapts the stepping / logging / save interface around the official
    TD-MPC2 agent, world model, and replay buffer.
    """

    supports_composable_checkpointing = True

    def __init__(self, name, env, custom_params=None, run_params=None, experiment_params=None):
        super().__init__(name, env, custom_params)
        self.run_params = run_params or {}
        self.experiment_params = experiment_params or {}
        self.cfg = self._build_cfg(custom_params or {})

        self._set_seed(self.cfg.seed)
        self.agent = self._make_agent(self.cfg)
        self.buffer = Buffer(self.cfg)
        self._pin_episode_staging = (
            str(self.cfg.device).startswith("cuda") and torch.cuda.is_available()
        )
        self._episode_staging = self._allocate_episode_staging(
            max(2, int(self.cfg.episode_length) + 1)
        )
        self._observation_staging = torch.empty(
            int(self.cfg.obs_shape["state"][0]),
            dtype=torch.float32,
            device="cpu",
            pin_memory=self._pin_episode_staging,
        )
        self._predict_t0 = True
        self._checkpointing = None
        self._checkpoint_writer = AsyncCheckpointWriter()
        self._global_step = 0
        self._episode_idx = 0
        self._episode_return = 0.0
        self._episode_len = 0
        self._pretrained = False
        self._last_train_metrics = None
        self._wandb_every = max(1, int((custom_params or {}).get("wandb_step_every", 1000)))
        self._wandb_run = None
        self._wandb_train_window = WandbAccumulator()
        self._wandb_update_window = _DeviceMeanAccumulator()
        self._wandb_reward_window = WandbAccumulator()
        self._reward_component_aliases = {}
        self._reserved_reward_metric_keys = self._reward_metric_family(
            "rollout/reward"
        )
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

    def _act_agent(self, obs_t, *, t0, eval_mode):
        """Action hook for subclasses that add call-scoped options."""
        return self.agent.act(obs_t, t0=t0, eval_mode=eval_mode)

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
        self._action_delta = self._action_high - self._action_low
        self._identity_action_scale = bool(
            np.array_equal(self._action_low, -np.ones_like(self._action_low))
            and np.array_equal(self._action_high, np.ones_like(self._action_high))
        )
        action_dim = int(np.prod(self._action_shape))

        model_size = cfg.get("model_size", None)
        if model_size is not None:
            model_size = int(model_size)
            if model_size not in MODEL_SIZE:
                raise ValueError(f"Invalid TD-MPC2 model_size={model_size}. Expected one of {list(MODEL_SIZE)}.")
            cfg.update(copy.deepcopy(MODEL_SIZE[model_size]))
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

    def _reuse_observation_tensor(self, obs):
        """Copy one environment observation into reusable CPU staging."""
        self._observation_staging.copy_(
            torch.as_tensor(self._obs_to_numpy(obs), dtype=torch.float32)
        )
        return self._observation_staging

    def _scale_action(self, action_env):
        action_env = np.asarray(action_env, dtype=np.float32).reshape(-1)
        if self._identity_action_scale:
            return np.clip(action_env, -1.0, 1.0).astype(np.float32, copy=False)
        action_norm = 2.0 * (action_env - self._action_low) / self._action_delta - 1.0
        return np.clip(action_norm, -1.0, 1.0).astype(np.float32)

    def _unscale_action(self, action_norm):
        action_norm = np.asarray(action_norm, dtype=np.float32).reshape(-1)
        if self._identity_action_scale:
            return np.clip(action_norm, -1.0, 1.0).astype(
                np.float32, copy=False
            ).reshape(self._action_shape)
        action_env = self._action_low + 0.5 * (action_norm + 1.0) * self._action_delta
        action_env = np.clip(action_env, self._action_low, self._action_high).astype(np.float32)
        return action_env.reshape(self._action_shape)

    def _random_action_norm(self):
        if self._identity_action_scale:
            return np.asarray(
                self.env.action_space.sample(), dtype=np.float32
            ).reshape(-1)
        return self._scale_action(self.env.action_space.sample())

    def _allocate_episode_staging(self, capacity):
        obs_dim = int(self.cfg.obs_shape["state"][0])
        empty_kwargs = {
            "dtype": torch.float32,
            "device": "cpu",
            "pin_memory": self._pin_episode_staging,
        }
        return TensorDict(
            {
                "obs": torch.empty((capacity, obs_dim), **empty_kwargs),
                "action": torch.empty((capacity, self.cfg.action_dim), **empty_kwargs),
                "reward": torch.empty((capacity,), **empty_kwargs),
                "terminated": torch.empty((capacity,), **empty_kwargs),
            },
            batch_size=(capacity,),
        )

    def _ensure_episode_staging_capacity(self, required):
        if required <= len(self._episode_staging):
            return
        old = self._episode_staging
        replacement = self._allocate_episode_staging(
            max(required, 2 * len(old))
        )
        for key in ("obs", "action", "reward", "terminated"):
            replacement[key][:len(old)].copy_(old[key])
        self._episode_staging = replacement

    def _start_episode_staging(self, obs_t):
        self._ensure_episode_staging_capacity(1)
        self._episode_staging["obs"][0].copy_(obs_t)
        self._episode_staging["action"][0].fill_(float("nan"))
        self._episode_staging["reward"][0] = float("nan")
        self._episode_staging["terminated"][0] = float("nan")
        return 1

    def _stage_transition(self, row, obs_t, action, reward, terminated):
        self._ensure_episode_staging_capacity(row + 1)
        self._episode_staging["obs"][row].copy_(obs_t)
        self._episode_staging["action"][row].copy_(
            torch.as_tensor(action, dtype=torch.float32).reshape(self.cfg.action_dim)
        )
        self._episode_staging["reward"][row] = float(reward)
        self._episode_staging["terminated"][row] = float(terminated)

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

        accepts_native_payload = bool(
            getattr(self.alg_logger, "accepts_native_step_payload", False)
        )
        # Built-in summary logging never consumes trajectory values. Unknown
        # custom loggers retain the historical fully materialized contract.
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
            materialize=not accepts_native_payload,
        )
        self.alg_logger.on_step(data)

    def _metrics_to_floats(self, metrics):
        accumulator = _DeviceMeanAccumulator()
        accumulator.update(metrics)
        return accumulator.pop_floats()

    def _reset_wandb_window(self):
        self._wandb_train_window.clear()
        self._wandb_update_window.clear()
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
        self._wandb_update_window.update(
            metrics,
            prefix="train/",
            skip=("num_updates",),
        )

    @staticmethod
    def _reward_metric_family(base_key):
        return {base_key} | {
            f"{base_key}_{suffix}"
            for suffix in ("count", "mean", "std", "min", "max")
        }

    def _resolve_reward_components(self, info):
        """Assign stable aliases only when flat W&B metric families overlap."""
        if not hasattr(self, "_reward_component_aliases"):
            # Compatibility for lightweight callers that construct the wrapper
            # without running ``__init__``.
            self._reward_component_aliases = {}
            self._reserved_reward_metric_keys = self._reward_metric_family(
                "rollout/reward"
            )

        components = extract_reward_components(info or {})
        resolved = {}
        for original_key in sorted(components):
            resolved_key = self._reward_component_aliases.get(original_key)
            if resolved_key is None:
                resolved_key = original_key
                # A component ending in a population-stat suffix is inherently
                # ambiguous with the corresponding base component's metric
                # family. Alias it even when that base has not appeared yet so
                # conditional info timing cannot reverse which name is kept.
                suffix_collision = original_key.endswith(
                    ("_count", "_mean", "_std", "_min", "_max")
                )
                if suffix_collision or (
                    self._reward_metric_family(resolved_key)
                    & self._reserved_reward_metric_keys
                ):
                    alias_stem = f"{original_key}_component"
                    resolved_key = alias_stem
                    alias_index = 2
                    while (
                        self._reward_metric_family(resolved_key)
                        & self._reserved_reward_metric_keys
                    ):
                        resolved_key = f"{alias_stem}_{alias_index}"
                        alias_index += 1
                self._reward_component_aliases[original_key] = resolved_key
                self._reserved_reward_metric_keys.update(
                    self._reward_metric_family(resolved_key)
                )
            resolved[resolved_key] = components[original_key]
        return resolved

    def _accumulate_reward_metrics(self, reward, info):
        self._wandb_reward_window.add_stats("rollout/reward", [reward])
        for key, value in self._resolve_reward_components(info).items():
            # Preserve the legacy unsuffixed mean and add complete population
            # moments for analysis over the entire W&B logging window.
            self._wandb_reward_window.add_weighted(key, value)
            self._wandb_reward_window.add_stats(key, [value])

    def _record_action_metrics(self, *, planned, action_seconds):
        if not planned or not bool(self.cfg.mpc):
            return
        self._wandb_planner_seconds += float(action_seconds)
        plan_metrics = getattr(self.agent, "last_plan_metrics", {}) or {}
        packed_metrics = {}
        for key, value in plan_metrics.items():
            if key == "planner_seconds":
                continue
            metric_key = key if key.startswith("planner_") else f"planner_{key}"
            packed_metrics[f"train/{metric_key}"] = value
        self._wandb_update_window.update(packed_metrics)

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
        cadence_due = self._global_step % self._wandb_every == 0
        if not force and not cadence_due:
            if completed_episode:
                # Emit the exact episode result without consuming the running
                # training window. Aggregate moments therefore retain every
                # sample across episode boundaries until the declared cadence.
                episode_payload = {
                    "train/reward": float(reward),
                    "train/done": int(done),
                    "train/terminated": int(bool(terminated)),
                    "train/truncated": int(bool(truncated)),
                    "episode/index": int(self._episode_idx),
                    "episode/return": float(self._episode_return),
                    "episode/len": int(self._episode_len),
                    "episode/current_return": float(self._episode_return),
                    "episode/current_len": int(self._episode_len),
                }
                log_wandb(
                    self._wandb_run,
                    episode_payload,
                    step=self._global_step,
                )
                self._last_wandb_step = int(self._global_step)
            return
        if (
            force
            and not self._wandb_train_window
            and not self._wandb_update_window
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
        # One packed transfer per metric device replaces per-update scalar
        # reads from accelerator memory.
        payload.update(
            self._wandb_update_window.pop_floats(include_stats=True)
        )
        for key, value in self._resolve_reward_components(info).items():
            # Direct callers may not have populated the interval accumulator.
            # Never overwrite a sampled window mean with the final step.
            payload.setdefault(key, value)
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
        if isinstance(self._checkpointing, tuple):
            save_freq, save_path, name_prefix = self._checkpointing
            if not (self._global_step > 0 and self._global_step % save_freq == 0):
                return
            if self.alg_logger is not None and hasattr(self.alg_logger, "flush"):
                self.alg_logger.flush()
            os.makedirs(save_path, exist_ok=True)
            self._checkpoint_writer.enqueue(
                self.agent.checkpoint_state(),
                os.path.join(save_path, f"{name_prefix}_{self._global_step}"),
                signature=self._checkpoint_signature(),
            )
            return

        targets = self._checkpointing.targets(self._global_step)
        if not targets:
            return
        if self.alg_logger is not None and hasattr(self.alg_logger, "flush"):
            self.alg_logger.flush()
        self._checkpoint_writer.enqueue_many(
            self.agent.checkpoint_state(),
            targets,
            signature=self._checkpoint_signature(),
        )

    def _final_checkpoint(self):
        if not isinstance(self._checkpointing, CheckpointTracker):
            return ()
        targets = self._checkpointing.targets(self._global_step, final=True)
        if not targets:
            return ()
        if self.alg_logger is not None and hasattr(self.alg_logger, "flush"):
            self.alg_logger.flush()
        return self._checkpoint_writer.save_many(
            self.agent.checkpoint_state(),
            targets,
            signature=self._checkpoint_signature(),
        )

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
            obs_t = self._reuse_observation_tensor(obs)
            episode_rows = self._start_episode_staging(obs_t)
            episode_step = 0

            while self._global_step < total_timesteps:
                planned = not (self._global_step <= self.cfg.seed_steps or self.buffer.num_eps == 0)
                action_start = time.perf_counter()
                if not planned:
                    action_norm = self._random_action_norm()
                else:
                    action_norm = self._act_agent(
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

                next_obs_t = self._reuse_observation_tensor(next_obs)
                self._stage_transition(
                    episode_rows,
                    next_obs_t,
                    action_norm,
                    reward,
                    true_terminated,
                )
                episode_rows += 1
                self._global_step += 1
                episode_step += 1
                self._episode_return += float(reward)
                self._episode_len += 1
                self._last_reward = float(reward)
                self._last_terminated = bool(terminated)
                self._last_truncated = bool(truncated)
                self._last_info = dict(info or {})
                self._accumulate_reward_metrics(reward, info)
                self._log_step(reward, next_obs, action_env, terminated, truncated, info)

                if done:
                    if true_terminated and not self.cfg.episodic:
                        raise ValueError(
                            "TD-MPC2 saw terminated=True while episodic=False. "
                            "Set alg_params.episodic=true or disable true terminations in the env."
                        )
                    self.buffer.add(self._episode_staging[:episode_rows])

                if self._global_step > self.cfg.seed_steps and self.buffer.num_eps > 0:
                    num_updates = self.cfg.pretrain_steps if not self._pretrained else self.cfg.utd
                    if not self._pretrained:
                        print("Pretraining TD-MPC2 on seed data...")
                        self._pretrained = True
                    burst_metrics = _DeviceMeanAccumulator()
                    train_start = time.perf_counter()
                    for _ in range(num_updates):
                        train_metrics = self.agent.update(self.buffer)
                        self._num_updates += 1
                        burst_metrics.update(train_metrics)
                        self._accumulate_train_metrics(train_metrics)
                    self._wandb_train_seconds += time.perf_counter() - train_start
                    self._last_train_metrics = burst_metrics.snapshot()

                self._log_wandb_step(
                    reward,
                    terminated,
                    truncated,
                    info,
                    completed_episode=done,
                )
                if done and isinstance(self._checkpointing, CheckpointTracker):
                    self._checkpointing.record_episode_return(self._episode_return)
                self._maybe_checkpoint()

                if done:
                    self._episode_idx += 1
                    self._episode_return = 0.0
                    self._episode_len = 0
                    obs, _ = self._reset_env()
                    obs_t = self._reuse_observation_tensor(obs)
                    episode_rows = self._start_episode_staging(obs_t)
                    episode_step = 0
                else:
                    obs = next_obs
                    obs_t = next_obs_t
            self._final_checkpoint()
            return self
        finally:
            try:
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
            finally:
                try:
                    if self.alg_logger is not None and hasattr(self.alg_logger, "flush"):
                        self.alg_logger.flush()
                finally:
                    # Periodic snapshots are exact at enqueue time, but an
                    # exception or normal shutdown must still make the queued
                    # atomic replacement durable before control returns.
                    self._checkpoint_writer.shutdown()

    def predict(self, observation, deterministic=True, episode_start=None):
        t0 = self._predict_t0 if episode_start is None else bool(episode_start)
        if t0 and hasattr(self.agent, "reset"):
            self.agent.reset()
        obs_t = self._obs_to_tensor(observation)
        action_norm = self._act_agent(
            obs_t,
            t0=t0,
            eval_mode=deterministic,
        ).numpy()
        self._predict_t0 = False
        return self._unscale_action(action_norm), None

    def save(self, path, name):
        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, name)
        tracker = getattr(self, "_checkpointing", None)
        if isinstance(tracker, CheckpointTracker):
            target = tracker.explicit_target(
                checkpoint_path,
                step=self._global_step,
                episode=getattr(self, "_episode_idx", tracker.episode_count),
            )
        else:
            target = explicit_checkpoint_target(
                checkpoint_path,
                step=self._global_step,
                episode=getattr(self, "_episode_idx", 0),
                trial_run_params=getattr(self, "run_params", {}),
                experiment_params=getattr(self, "experiment_params", {}),
            )
        return self._checkpoint_writer.save_many(
            self.agent.checkpoint_state(),
            (target,),
            signature=self._checkpoint_signature(),
        )[0]

    def _checkpoint_signature(self):
        """Version the exact outer state represented by native checkpoints."""
        return (
            int(self._global_step),
            int(getattr(self.agent, "num_updates", self._num_updates)),
            int(getattr(self.agent, "outer_version", -1)),
        )

    def flush_checkpoints(self):
        """Wait for any periodic checkpoint and surface background errors."""
        return self._checkpoint_writer.flush()

    def load(self, path):
        self.flush_checkpoints()
        self._checkpoint_writer.invalidate()
        self.agent.load(path)
        self._num_updates = int(getattr(self.agent, "num_updates", self._num_updates))
        return self

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
        self._checkpointing = CheckpointTracker(
            save_freq,
            save_path,
            name_prefix,
            save_strat=save_strat,
            best_window=checkpoint_best_window,
            periodic_step_suffix="",
            trial_run_params=self.run_params if trial_run_params is None else trial_run_params,
            experiment_params=(
                self.experiment_params if experiment_params is None else experiment_params
            ),
        )
        return self._checkpointing
