import copy
import csv
import math
import os
import random
import time
import warnings
from collections.abc import Mapping
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.utils import flatdim, flatten
try:
    from tensordict import TensorDict
except ImportError:  # tensordict<newer API compatibility
    from tensordict.tensordict import TensorDict

from RL.alg import Algorithm, validate_timestep_budget
from RL.tdmpc2_core import MODEL_SIZE
from RL.tdmpc2_core.agent import TDMPC2
from RL.tdmpc2_core.common.buffer import Buffer
from RL.tdmpc2_core.common.checkpoint import AsyncCheckpointWriter
from RL.tdmpc2_core.common.device import resolve_device
from RL.tdmpc2_core.common.math import TEMPORAL_LOSS_NORMALIZATIONS
from utils.checkpointing import (
    CheckpointTracker,
    explicit_checkpoint_target,
)
from utils.cleanup import add_cleanup_notes, raise_cleanup_errors
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

    # evaluation (disabled unless an explicit cadence is configured)
    "eval_freq": None,
    "eval_episodes": 10,
    "eval_csv_path": None,

    # planning
    "mpc": True,
    "iterations": 6,
    "num_samples": 512,
    "num_elites": 64,
    "num_pi_trajs": 24,
    "num_pi_trajs_first_iteration_only": False,
    "train_unroll_horizon": 3,
    "outer_planning_horizon": 3,
    "inner_rollout_horizon": 3,
    "min_std": 0.05,
    "max_std": 2.0,
    "temperature": 0.5,

    # temporal weighting
    "temporal_loss_normalization": "reference_weighted_mean",
    "temporal_loss_reference_horizon": 3,

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


_EXPLICIT_HORIZON_FIELDS = {
    "train_unroll_horizon",
    "outer_planning_horizon",
    "inner_rollout_horizon",
}


_RGB_OBS_SHAPE = (9, 64, 64)
_RGB_REPLAY_WARNING_BYTES = 8 * 1024**3


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


def _nonnegative_int(value, key):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{key} must be a non-negative integer.")
    try:
        numeric = float(value)
        resolved = int(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{key} must be a non-negative integer.") from exc
    if not math.isfinite(numeric) or resolved < 0 or numeric != resolved:
        raise ValueError(f"{key} must be a non-negative integer.")
    return resolved


def _strict_bool(value, key):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{key} must be a boolean.")
    return bool(value)


def _normalize_horizon_params(params, *, resolve_defaults=True):
    """Resolve canonical horizons while retaining one-release legacy support."""
    params = copy.deepcopy(params)
    if "horizon" in params:
        conflicts = sorted(set(params) & _EXPLICIT_HORIZON_FIELDS)
        if conflicts:
            raise ValueError(
                "Cannot combine legacy horizon with explicit horizon "
                f"fields: {conflicts}."
            )
        legacy_horizon = _positive_int(params.pop("horizon"), "horizon")
        warnings.warn(
            "Legacy 'horizon' is deprecated; use train_unroll_horizon and "
            "outer_planning_horizon explicitly.",
            FutureWarning,
            stacklevel=3,
        )
        params["train_unroll_horizon"] = legacy_horizon
        params["outer_planning_horizon"] = legacy_horizon

    for key in _EXPLICIT_HORIZON_FIELDS:
        if resolve_defaults or key in params:
            params[key] = _positive_int(params.get(key, 3), key)
    return params


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
        print(
            "Resolved horizons:",
            f"train_unroll_horizon={self.cfg.train_unroll_horizon},",
            f"outer_planning_horizon={self.cfg.outer_planning_horizon},",
            f"inner_rollout_horizon={self.cfg.inner_rollout_horizon}",
        )
        print(
            "Temporal loss normalization:",
            f"{self.cfg.temporal_loss_normalization} ",
            f"(reference_horizon={self.cfg.temporal_loss_reference_horizon})",
        )

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
            tuple(self.cfg.obs_shape[self.cfg.obs]),
            dtype=self._obs_torch_dtype,
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
        self._wandb_elapsed_offset = 0.0
        self._wandb_window_seconds_offset = 0.0
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
        self._eval_freq = self.cfg.eval_freq
        self._eval_episodes = self.cfg.eval_episodes
        self._eval_csv_path = self.cfg.eval_csv_path or os.environ.get(
            "TDMPC2_EVAL_CSV"
        )
        self._eval_csv_initialized = False

        print("Architecture:", self.agent.model)

    def _make_agent(self, cfg):
        """Factory hook used by TD-MPC2-derived algorithms."""
        return TDMPC2(cfg)

    def _act_agent(self, obs_t, *, t0, eval_mode):
        """Action hook for subclasses that add call-scoped options."""
        return self.agent.act(obs_t, t0=t0, eval_mode=eval_mode)

    def _wandb_run_name(self):
        env_params = self.experiment_params.get("env_params", {})
        task = env_params.get("task", self.run_params.get("env", "env"))
        return f"TDMPC2-{task}-seed{self.cfg.seed}"

    def _init_wandb(self, *, resume_context=None):
        run_config = copy.deepcopy(self.run_params)
        algorithm_config = copy.deepcopy(self.custom_params or {})
        resolved_config = copy.deepcopy(vars(self.cfg))
        if resume_context is not None:
            # Segment destinations are operational and change on every resume.
            # Keep them out of W&B's otherwise-stable run config just as the
            # scientific lineage fingerprint does.
            algorithm_config.pop("eval_csv_path", None)
            resolved_config.pop("eval_csv_path", None)
            nested_algorithm = run_config.get("alg_params")
            if isinstance(nested_algorithm, Mapping):
                nested_algorithm = copy.deepcopy(dict(nested_algorithm))
                nested_algorithm.pop("eval_csv_path", None)
                run_config["alg_params"] = nested_algorithm
        return init_wandb(
            self.custom_params or {},
            default_project="ambi",
            run_name=self._wandb_run_name(),
            config={
                "algorithm": self.__class__.__name__,
                "run_params": run_config,
                "alg_params": algorithm_config,
                "config": resolved_config,
            },
            resume_context=resume_context,
        )

    def _environment_observation_type(self):
        """Return an environment-declared TD-MPC2 observation mode, if any."""
        try:
            observation_type = self.env.get_wrapper_attr("observation_type")
        except AttributeError:
            try:
                # Avoid Wrapper.__getattr__, which emits a deprecation warning
                # for a legitimately absent optional declaration.
                observation_type = object.__getattribute__(
                    self.env, "observation_type"
                )
            except AttributeError:
                observation_type = None
        if observation_type is None:
            return None
        observation_type = str(observation_type).lower()
        if observation_type not in {"state", "rgb"}:
            raise ValueError(
                "Environment observation_type must be 'state' or 'rgb', "
                f"got {observation_type!r}."
            )
        return observation_type

    def _resolve_observation_space(self, params):
        """Resolve and validate the state/RGB contract before model creation."""
        declared_obs = self._environment_observation_type()
        requested_obs = None
        if "obs" in params:
            requested_obs = str(params["obs"]).lower()
            if requested_obs not in {"state", "rgb"}:
                raise ValueError(
                    "alg_params.obs must be 'state' or 'rgb', "
                    f"got {requested_obs!r}."
                )
        if (
            declared_obs is not None
            and requested_obs is not None
            and declared_obs != requested_obs
        ):
            raise ValueError(
                "alg_params.obs does not match the environment's declared "
                "observation_type: "
                f"alg_params.obs={requested_obs!r}, "
                f"observation_type={declared_obs!r}."
            )

        observation_type = declared_obs or requested_obs or "state"
        self._obs_space = self.env.observation_space
        if observation_type == "rgb":
            if not isinstance(self._obs_space, gym.spaces.Box):
                raise NotImplementedError(
                    "TD-MPC2 RGB observations require a Box observation space."
                )
            if tuple(self._obs_space.shape) != _RGB_OBS_SHAPE:
                raise ValueError(
                    "TD-MPC2 RGB observations must use CHW frame stacks with "
                    f"shape {_RGB_OBS_SHAPE}; got {self._obs_space.shape}."
                )
            if np.dtype(self._obs_space.dtype) != np.dtype(np.uint8):
                raise ValueError(
                    "TD-MPC2 RGB observations must use dtype uint8; "
                    f"got {self._obs_space.dtype}."
                )
            if not (
                np.all(self._obs_space.low == 0)
                and np.all(self._obs_space.high == 255)
            ):
                raise ValueError(
                    "TD-MPC2 RGB observation bounds must be exactly [0, 255]."
                )
            self._obs_np_dtype = np.dtype(np.uint8)
            self._obs_torch_dtype = torch.uint8
            return observation_type, _RGB_OBS_SHAPE

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
            raise NotImplementedError(
                "TD-MPC2Baseline only supports Box or Dict observation spaces."
            )
        self._obs_np_dtype = np.dtype(np.float32)
        self._obs_torch_dtype = torch.float32
        return observation_type, (obs_dim,)

    def _build_cfg(self, params):
        params = _normalize_horizon_params(params)
        cfg = copy.deepcopy(_DEFAULTS)
        cfg.update(copy.deepcopy(params))

        run_device = self.run_params.get("device", None)
        if "device" not in params and run_device is not None:
            cfg["device"] = run_device
        cfg["device"] = str(resolve_device(cfg["device"]))

        cfg["seed"] = int(self.run_params.get("seed", cfg.get("seed", 1)))
        cfg["steps"] = int(float(self.run_params.get("total_steps", cfg.get("steps", 1_000_000))))

        observation_type, observation_shape = self._resolve_observation_space(
            params
        )

        if not isinstance(self.env.action_space, gym.spaces.Box):
            raise NotImplementedError("TD-MPC2Baseline only supports continuous Box action spaces.")
        self._action_shape = self.env.action_space.shape
        self._action_low = self.env.action_space.low.astype(np.float32).reshape(-1)
        self._action_high = self.env.action_space.high.astype(np.float32).reshape(-1)
        if not np.all(np.isfinite(self._action_low)) or not np.all(np.isfinite(self._action_high)):
            raise ValueError("TD-MPC2Baseline requires finite action-space bounds.")
        self._action_delta = self._action_high - self._action_low
        if np.any(self._action_delta <= 0.0):
            raise ValueError(
                "TD-MPC2Baseline requires every action dimension to have "
                "strictly increasing bounds."
            )
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
        if observation_type == "rgb" and int(cfg["latent_dim"]) != (
            16 * int(cfg["num_channels"])
        ):
            raise ValueError(
                "TD-MPC2's 64x64 RGB encoder requires "
                "latent_dim == 16 * num_channels; "
                f"got latent_dim={cfg['latent_dim']} and "
                f"num_channels={cfg['num_channels']}."
            )
        if observation_type == "rgb" and bool(
            cfg.get("compile_strict", False)
        ):
            raise ValueError(
                "RGB observations with compile_strict=True are not supported; "
                "use eager execution or compile_strict=False."
            )

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

        eval_freq = cfg.get("eval_freq", None)
        cfg["eval_freq"] = (
            None if eval_freq is None else _positive_int(eval_freq, "eval_freq")
        )
        cfg["eval_episodes"] = _positive_int(
            cfg.get("eval_episodes", 10), "eval_episodes"
        )
        eval_csv_path = cfg.get("eval_csv_path", None)
        if eval_csv_path is not None:
            if not isinstance(eval_csv_path, (str, os.PathLike)):
                raise ValueError("eval_csv_path must be a filesystem path or null.")
            eval_csv_path = os.fspath(eval_csv_path).strip()
            if not eval_csv_path:
                raise ValueError("eval_csv_path cannot be empty.")
        cfg["eval_csv_path"] = eval_csv_path

        cfg["iterations"] = _positive_int(cfg["iterations"], "iterations")
        cfg["num_samples"] = _positive_int(cfg["num_samples"], "num_samples")
        cfg["num_elites"] = _positive_int(cfg["num_elites"], "num_elites")
        cfg["num_pi_trajs"] = _nonnegative_int(
            cfg["num_pi_trajs"], "num_pi_trajs"
        )
        cfg["mpc"] = _strict_bool(cfg["mpc"], "mpc")
        cfg["num_pi_trajs_first_iteration_only"] = _strict_bool(
            cfg["num_pi_trajs_first_iteration_only"],
            "num_pi_trajs_first_iteration_only",
        )
        if cfg["num_elites"] > cfg["num_samples"]:
            raise ValueError("num_elites cannot exceed num_samples.")
        if cfg["num_pi_trajs"] > cfg["num_samples"]:
            raise ValueError("num_pi_trajs cannot exceed num_samples.")
        if cfg["num_pi_trajs_first_iteration_only"]:
            if not cfg["mpc"]:
                raise ValueError(
                    "num_pi_trajs_first_iteration_only=true requires mpc=true."
                )
            if cfg["num_pi_trajs"] == 0:
                raise ValueError(
                    "num_pi_trajs_first_iteration_only=true requires "
                    "num_pi_trajs>0."
                )
            effective_iterations = cfg["iterations"] + 2 * int(action_dim >= 20)
            if effective_iterations < 2:
                raise ValueError(
                    "num_pi_trajs_first_iteration_only=true requires at least "
                    "two effective planning iterations."
                )

        cfg["rho"] = float(cfg["rho"])
        if not math.isfinite(cfg["rho"]):
            raise ValueError("rho must be finite.")
        cfg["temporal_loss_normalization"] = str(
            cfg["temporal_loss_normalization"]
        ).lower()
        if cfg["temporal_loss_normalization"] not in TEMPORAL_LOSS_NORMALIZATIONS:
            raise ValueError(
                "temporal_loss_normalization must be one of "
                f"{sorted(TEMPORAL_LOSS_NORMALIZATIONS)}."
            )
        cfg["temporal_loss_reference_horizon"] = _positive_int(
            cfg["temporal_loss_reference_horizon"],
            "temporal_loss_reference_horizon",
        )
        # Read-only compatibility alias for integrations that still size outer
        # training tensors through cfg.horizon. Core code must use the canonical
        # fields so planning can differ from recurrent training.
        cfg["horizon"] = cfg["train_unroll_horizon"]

        # Allow a simple fixed discount override in alg_params, e.g. "discount": 0.99.
        if "discount" in cfg and cfg["discount"] is not None:
            cfg["discount_min"] = float(cfg["discount"])
            cfg["discount_max"] = float(cfg["discount"])

        cfg["obs"] = observation_type
        cfg["obs_shape"] = {observation_type: observation_shape}
        cfg["obs_dtype"] = self._obs_np_dtype.name
        if observation_type == "rgb":
            replay_rows = min(int(cfg["buffer_size"]), int(cfg["steps"]))
            replay_observation_bytes = (
                replay_rows
                * int(np.prod(observation_shape))
                * self._obs_np_dtype.itemsize
            )
            if replay_observation_bytes >= _RGB_REPLAY_WARNING_BYTES:
                warnings.warn(
                    "RGB replay observations alone are projected to require "
                    f"{replay_observation_bytes / 1e9:.1f} GB for "
                    f"{replay_rows:,} uint8 rows. Replay capacity is unchanged; "
                    "set buffer_size explicitly if this footprint is not intended.",
                    UserWarning,
                    stacklevel=3,
                )
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
        if self.cfg.obs == "rgb":
            obs = np.asarray(obs)
            expected_shape = tuple(self.cfg.obs_shape["rgb"])
            if tuple(obs.shape) != expected_shape:
                raise ValueError(
                    "RGB observation shape changed at runtime: "
                    f"expected {expected_shape}, got {obs.shape}."
                )
            if obs.dtype != self._obs_np_dtype:
                raise ValueError(
                    "RGB observation dtype changed at runtime: "
                    f"expected {self._obs_np_dtype.name}, got {obs.dtype}."
                )
            return np.ascontiguousarray(obs)
        if isinstance(self._obs_space, gym.spaces.Dict):
            obs = flatten(self._obs_space, obs)
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def _obs_to_tensor(self, obs):
        return torch.as_tensor(
            self._obs_to_numpy(obs), dtype=self._obs_torch_dtype
        )

    def _reuse_observation_tensor(self, obs):
        """Copy one environment observation into reusable CPU staging."""
        self._observation_staging.copy_(
            torch.as_tensor(
                self._obs_to_numpy(obs), dtype=self._obs_torch_dtype
            )
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
        obs_shape = tuple(self.cfg.obs_shape[self.cfg.obs])
        empty_kwargs = {
            "dtype": torch.float32,
            "device": "cpu",
            "pin_memory": self._pin_episode_staging,
        }
        return TensorDict(
            {
                "obs": torch.empty(
                    (capacity, *obs_shape),
                    dtype=self._obs_torch_dtype,
                    device="cpu",
                    pin_memory=self._pin_episode_staging,
                ),
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
        self._wandb_elapsed_offset = 0.0
        self._wandb_window_seconds_offset = 0.0
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
        window_seconds = self._wandb_window_seconds_offset + max(
            0.0, now - window_start
        )
        elapsed_seconds = self._wandb_elapsed_offset + max(0.0, now - start)
        window_steps = max(0, int(self._global_step) - int(self._wandb_window_start_step))
        payload = {
            "time/window_seconds": float(window_seconds),
            "time/time_elapsed": float(elapsed_seconds),
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
        self._wandb_window_seconds_offset = 0.0
        self._wandb_window_start_step = int(self._global_step)
        self._wandb_train_seconds = 0.0
        self._wandb_planner_seconds = 0.0
        return payload

    def _timing_wandb_metric_keys(self):
        """Return timing keys reserved by :meth:`_timing_wandb_payload`."""

        return {
            "time/window_seconds",
            "time/time_elapsed",
            "time/total_timesteps",
            "time/fps",
            "time/train_seconds",
            "time/updates_per_second",
            "time/planner_seconds",
        }

    def _commit_resume_timing_checkpoint(self, wandb_state):
        """Continue active timing from the exact pre-serialization boundary.

        Generation serialization/publication, restore/reconciliation, and queue
        residence are operational pauses. Reset the monotonic references after
        those operations so a continuing and restored process use the same
        timing prefix. Required pre-capture log/snapshot flushes remain included.
        """

        now = time.perf_counter()
        self._wandb_elapsed_offset = float(wandb_state["elapsed_seconds"])
        self._wandb_window_seconds_offset = float(wandb_state["window_seconds"])
        self._wandb_start_time = now
        self._wandb_window_start_time = now

    def _extra_wandb_payload(self, updates_since_log):
        del updates_since_log
        return {}

    def _episode_payload_extras(self):
        """Return subclass metrics that are meaningful only at episode end."""
        return {}

    def _validate_episode_payload_extras(self, extras, *, reserved_keys):
        """Validate episode-only metrics without consuming logging windows."""

        if not isinstance(extras, Mapping):
            raise TypeError("Episode payload extras must be a mapping.")
        if not extras:
            return {}
        collisions = (set(reserved_keys) | {"env_step"}).intersection(extras)
        if collisions:
            raise ValueError(
                "Episode payload extras collide with reserved metrics: "
                f"{sorted(collisions)}."
            )
        validated = {}
        for key, value in extras.items():
            if not isinstance(key, str) or not key:
                raise TypeError(
                    "Episode payload extra keys must be non-empty strings."
                )
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise TypeError(f"Episode payload extra {key!r} must be numeric.")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(
                    f"Episode payload extra {key!r} must be finite."
                )
            validated[key] = value
        return validated

    def _merge_episode_payload_extras(self, payload):
        extras = self._validate_episode_payload_extras(
            self._episode_payload_extras(),
            reserved_keys=payload,
        )
        payload.update(extras)
        return payload

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
            if completed_episode and getattr(self, "_resume_enabled", False):
                # Resume boundaries are deterministic scientific boundaries.
                # Emit one complete aggregate+episode event here; the later
                # boundary flush then becomes a no-op.
                force = True
            elif completed_episode:
                # Emit the exact episode result without consuming the running
                # training window. Legacy runs retain it to the cadence.
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
                self._merge_episode_payload_extras(episode_payload)
                log_wandb(
                    self._wandb_run,
                    episode_payload,
                    step=self._global_step,
                )
                self._last_wandb_step = int(self._global_step)
                return
            else:
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
        if getattr(self, "_resume_enabled", False):
            payload["episode/index"] = int(self._episode_idx)
        payload.update(self._replay_wandb_payload())
        # Build a prospective row without consuming interval state. A malformed
        # episode extension must not silently discard pending telemetry.
        payload.update(self._wandb_reward_window.snapshot())
        payload.update(self._wandb_train_window.snapshot())
        # One packed transfer per metric device replaces per-update scalar
        # reads from accelerator memory.
        payload.update(
            self._wandb_update_window.floats(include_stats=True)
        )
        for key, value in self._resolve_reward_components(info).items():
            # Direct callers may not have populated the interval accumulator.
            # Never overwrite a sampled window mean with the final step.
            payload.setdefault(key, value)
        extra_payload = self._extra_wandb_payload(updates_since_log)
        if completed_episode:
            payload.update({
                "episode/index": int(self._episode_idx),
                "episode/return": float(self._episode_return),
                "episode/len": int(self._episode_len),
            })
            episode_extras = self._validate_episode_payload_extras(
                self._episode_payload_extras(),
                reserved_keys=(
                    set(payload)
                    | set(extra_payload)
                    | set(self._timing_wandb_metric_keys())
                ),
            )
        else:
            episode_extras = {}

        # Extension validation succeeded. Commit the interval boundary once,
        # preserving the historical timing/algorithm/episode precedence.
        self._wandb_reward_window.clear()
        self._wandb_train_window.clear()
        self._wandb_update_window.clear()
        payload.update(self._timing_wandb_payload(updates_since_log))
        payload.update(extra_payload)
        payload.update(episode_extras)
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

    def enable_training_resume(self, *, total_timesteps):
        """Enable the explicit boundary-only scientific resume contract."""

        from utils.resume_runtime import validate_environment_capability

        total_timesteps = validate_timestep_budget(total_timesteps)
        if self.cfg.obs != "state":
            raise NotImplementedError("Exact TD-MPC2 resume supports state observations only.")
        if total_timesteps != int(self.cfg.steps):
            raise ValueError("Resume total_timesteps must match the constructed step target.")
        episode_length = int(self.cfg.episode_length)
        validate_environment_capability(
            self.env, expected_episode_steps=episode_length
        )
        if total_timesteps < 0 or total_timesteps % episode_length != 0:
            raise ValueError(
                "Boundary-only resume requires total_steps to be a non-negative "
                f"multiple of episode_length={episode_length}."
            )
        self.buffer.enable_resumable_storage()
        self._resume_enabled = True
        self._resume_phase = "before_initial_seeded_reset"
        self._eval_pending = self._eval_freq is not None
        self._step_zero_eval_done = self._eval_freq is None
        return self

    @staticmethod
    def _resume_nonnegative_int(value, name):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Resume {name} must be a non-negative integer.")
        return int(value)

    def _training_resume_algorithm_state(self):
        """Subclass-specific wrapper state; baseline has no additional fields."""

        return None

    def _preflight_training_resume_algorithm_state(self, state):
        if state is not None:
            raise ValueError("Baseline trainer state must not contain AMBI wrapper state.")
        return None

    def _load_training_resume_algorithm_state(self, state):
        self._preflight_training_resume_algorithm_state(state)

    def training_state_dict(self, *, phase=None):
        """Capture complete learner state at one reviewed episode boundary."""

        if not getattr(self, "_resume_enabled", False):
            raise RuntimeError("Training resume has not been enabled for this learner.")
        phase = self._resume_phase if phase is None else phase
        if phase not in {
            "before_initial_seeded_reset",
            "between_episodes_before_reset",
        }:
            raise ValueError(f"Unsupported resume phase {phase!r}.")
        if self._episode_len != 0 or self._episode_return != 0.0:
            raise RuntimeError("A full trainer checkpoint requires an episode boundary.")
        if not self._predict_t0:
            raise RuntimeError("Episode-local prediction state was not cleared.")
        if int(getattr(self.agent, "num_updates", -1)) != int(self._num_updates):
            raise RuntimeError("Wrapper and agent update counters diverged.")
        if any(
            (
                self._wandb_train_window,
                self._wandb_update_window,
                self._wandb_reward_window,
            )
        ):
            raise RuntimeError(
                "Full checkpoints require an empty W&B metric window; "
                "publish through TrainingResumeSession."
            )
        if self._wandb_train_seconds != 0.0 or self._wandb_planner_seconds != 0.0:
            raise RuntimeError("W&B timing windows were not flushed before checkpointing.")
        if self._wandb_last_updates != self._num_updates:
            raise RuntimeError("W&B and trainer update counters diverged at checkpoint.")
        if self._wandb_window_start_step != self._global_step or (
            self._last_wandb_step is not None
            and self._last_wandb_step != self._global_step
        ):
            raise RuntimeError("W&B metric boundaries diverged from trainer progress.")
        logger_state = None
        if self.alg_logger is not None:
            logger_state = self.alg_logger.resume_state_dict()
            if logger_state["step_count"] != self._global_step:
                raise RuntimeError("Logger and learner environment-step counters diverged.")
            if logger_state["episode_count"] != self._episode_idx:
                raise RuntimeError("Logger and learner episode counters diverged.")
        tracker_state = None
        if self._checkpointing is not None:
            if not isinstance(self._checkpointing, CheckpointTracker):
                raise RuntimeError(
                    "Exact resume requires CheckpointTracker, not legacy tuple checkpointing."
                )
            tracker_state = self._checkpointing.state_dict()
            if tracker_state["episode_count"] != self._episode_idx:
                raise RuntimeError("Checkpoint tracker and learner episodes diverged.")

        timing_now = time.perf_counter()
        elapsed_start = (
            self._wandb_start_time
            if self._wandb_start_time is not None
            else timing_now
        )
        window_start_time = (
            self._wandb_window_start_time
            if self._wandb_window_start_time is not None
            else timing_now
        )
        return {
            "schema": "tdmpc2-wrapper-training-state",
            "version": 4,
            "phase": phase,
            "total_steps": int(self.cfg.steps),
            "global_step": int(self._global_step),
            "completed_episodes": int(self._episode_idx),
            "pretrained": bool(self._pretrained),
            "num_updates": int(self._num_updates),
            "eval_pending": bool(self._eval_pending),
            "step_zero_eval_done": bool(self._step_zero_eval_done),
            "algorithm_state": self._training_resume_algorithm_state(),
            "agent": self.agent.training_state_dict(),
            "logger": logger_state,
            "checkpoint_tracker": tracker_state,
            "wandb": {
                "reward_component_aliases": dict(self._reward_component_aliases),
                "window_start_step": int(self._wandb_window_start_step),
                "elapsed_seconds": float(
                    self._wandb_elapsed_offset
                    + max(0.0, timing_now - elapsed_start)
                ),
                "window_seconds": float(
                    self._wandb_window_seconds_offset
                    + max(0.0, timing_now - window_start_time)
                ),
                "last_updates": int(self._wandb_last_updates),
                "last_step": self._last_wandb_step,
            },
        }

    def preflight_training_state_dict(self, state):
        """Validate a complete trainer state before mutating this learner."""

        expected = {
            "schema",
            "version",
            "phase",
            "total_steps",
            "global_step",
            "completed_episodes",
            "pretrained",
            "num_updates",
            "eval_pending",
            "step_zero_eval_done",
            "algorithm_state",
            "agent",
            "logger",
            "checkpoint_tracker",
            "wandb",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Trainer resume fields do not match the supported schema.")
        if state["schema"] != "tdmpc2-wrapper-training-state" or state["version"] != 4:
            raise ValueError("Unsupported trainer resume schema/version.")
        phase = state["phase"]
        if phase not in {
            "before_initial_seeded_reset",
            "between_episodes_before_reset",
        }:
            raise ValueError(f"Unsupported trainer resume phase {phase!r}.")
        if state["total_steps"] != int(self.cfg.steps):
            raise ValueError("Trainer absolute step target changed across resume.")
        global_step = self._resume_nonnegative_int(state["global_step"], "global_step")
        episodes = self._resume_nonnegative_int(
            state["completed_episodes"], "completed_episodes"
        )
        updates = self._resume_nonnegative_int(state["num_updates"], "num_updates")
        if global_step > int(self.cfg.steps):
            raise ValueError("Trainer global_step exceeds the immutable target.")
        if phase == "before_initial_seeded_reset" and (global_step or episodes or updates):
            raise ValueError("The initial resume phase must have zero progress counters.")
        if (
            phase == "between_episodes_before_reset"
            and global_step != episodes * int(self.cfg.episode_length)
        ):
            raise ValueError("Trainer steps and completed fixed-length episodes differ.")
        for name in ("pretrained", "eval_pending", "step_zero_eval_done"):
            if not isinstance(state[name], bool):
                raise TypeError(f"Trainer {name} must be bool.")
        if self._eval_freq is None and state["eval_pending"]:
            raise ValueError("Evaluation cannot be pending when eval_freq is disabled.")
        if phase == "before_initial_seeded_reset":
            if state["eval_pending"] is not (self._eval_freq is not None):
                raise ValueError(
                    "The genesis checkpoint has inconsistent step-zero evaluation state."
                )
            if state["step_zero_eval_done"] is not (self._eval_freq is None):
                raise ValueError(
                    "The genesis checkpoint has an invalid step-zero evaluation marker."
                )
        elif not state["step_zero_eval_done"]:
            raise ValueError(
                "A between-episode checkpoint cannot precede the canonical "
                "step-zero evaluation."
            )
        algorithm_state = self._preflight_training_resume_algorithm_state(
            state["algorithm_state"]
        )

        agent_preflight = getattr(self.agent, "_preflight_training_state_dict", None)
        agent_commit = getattr(self.agent, "_commit_training_state_candidate", None)
        if not callable(agent_preflight) or not callable(agent_commit):
            raise TypeError(
                "The configured agent lacks exact training-state preflight/commit."
            )
        agent_candidate = agent_preflight(state["agent"])
        boundary_prepared = state["agent"].get("boundary_prepared")
        expected_boundary = phase == "between_episodes_before_reset"
        if boundary_prepared is not expected_boundary:
            raise ValueError(
                "Agent boundary preparation does not match the trainer resume phase."
            )
        if int(state["agent"].get("num_updates", state["agent"].get("outer", {}).get("num_updates", -1))) != updates:
            raise ValueError("Trainer and agent update counters differ in the checkpoint.")

        logger_state = state["logger"]
        if (self.alg_logger is None) != (logger_state is None):
            raise ValueError("Training logger enablement changed across resume.")
        if self.alg_logger is not None:
            normalized_logger = self.alg_logger.validate_resume_state_dict(logger_state)
            if normalized_logger["step_count"] != global_step:
                raise ValueError("Saved logger and learner steps differ.")
            if normalized_logger["episode_count"] != episodes:
                raise ValueError("Saved logger and learner episodes differ.")

        tracker_state = state["checkpoint_tracker"]
        if (self._checkpointing is None) != (tracker_state is None):
            raise ValueError("Model-snapshot checkpoint policy changed across resume.")
        if self._checkpointing is not None:
            if not isinstance(self._checkpointing, CheckpointTracker):
                raise ValueError("Resume requires a CheckpointTracker policy.")
            normalized_tracker = self._checkpointing.validate_state_dict(tracker_state)
            if normalized_tracker["episode_count"] != episodes:
                raise ValueError("Saved checkpoint tracker and learner episodes differ.")

        wandb_state = state["wandb"]
        wandb_fields = {
            "reward_component_aliases",
            "window_start_step",
            "elapsed_seconds",
            "window_seconds",
            "last_updates",
            "last_step",
        }
        if not isinstance(wandb_state, Mapping) or set(wandb_state) != wandb_fields:
            raise ValueError("Trainer W&B boundary fields are invalid.")
        aliases = wandb_state["reward_component_aliases"]
        if (
            not isinstance(aliases, Mapping)
            or any(not isinstance(key, str) or not isinstance(value, str) for key, value in aliases.items())
        ):
            raise ValueError("Trainer reward metric aliases are invalid.")
        window_start = self._resume_nonnegative_int(
            wandb_state["window_start_step"], "W&B window_start_step"
        )
        last_updates = self._resume_nonnegative_int(
            wandb_state["last_updates"], "W&B last_updates"
        )
        if window_start != global_step or last_updates != updates:
            raise ValueError("Saved W&B boundary counters differ from trainer progress.")
        last_step = wandb_state["last_step"]
        if last_step is not None:
            last_step = self._resume_nonnegative_int(last_step, "W&B last_step")
            if last_step != global_step:
                raise ValueError("Saved W&B step differs from trainer progress.")
        for name in ("elapsed_seconds", "window_seconds"):
            value = float(wandb_state[name])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Trainer W&B {name} must be finite and non-negative.")
        if float(wandb_state["window_seconds"]) > float(
            wandb_state["elapsed_seconds"]
        ):
            raise ValueError("Trainer W&B window time exceeds lineage elapsed time.")

        return {
            "phase": phase,
            "global_step": global_step,
            "episodes": episodes,
            "updates": updates,
            "last_step": last_step,
            "algorithm_state": algorithm_state,
            "agent_candidate": agent_candidate,
        }

    def load_training_state_dict(self, state):
        """Restore a fully preflighted trainer snapshot into a fresh learner."""

        if self._global_step != 0 or self.buffer.num_eps != 0:
            raise ValueError("Trainer state must be loaded into a fresh learner.")
        candidate = self.preflight_training_state_dict(state)
        self.agent._commit_training_state_candidate(candidate["agent_candidate"])
        self._global_step = candidate["global_step"]
        self._episode_idx = candidate["episodes"]
        self._episode_return = 0.0
        self._episode_len = 0
        self._pretrained = state["pretrained"]
        self._num_updates = candidate["updates"]
        self._eval_pending = state["eval_pending"]
        self._step_zero_eval_done = state["step_zero_eval_done"]
        self._load_training_resume_algorithm_state(candidate["algorithm_state"])
        self._resume_phase = candidate["phase"]
        if self.alg_logger is not None:
            self.alg_logger.load_resume_state_dict(state["logger"])
        if self._checkpointing is not None:
            self._checkpointing.load_state_dict(state["checkpoint_tracker"])
        self._wandb_train_window.clear()
        self._wandb_reward_window.clear()
        self._wandb_update_window.clear()
        wandb_state = state["wandb"]
        self._reward_component_aliases = dict(wandb_state["reward_component_aliases"])
        self._reserved_reward_metric_keys = self._reward_metric_family(
            "rollout/reward"
        )
        for alias in self._reward_component_aliases.values():
            self._reserved_reward_metric_keys.update(
                self._reward_metric_family(alias)
            )
        self._wandb_window_start_step = int(wandb_state["window_start_step"])
        self._wandb_elapsed_offset = float(wandb_state["elapsed_seconds"])
        self._wandb_window_seconds_offset = float(wandb_state["window_seconds"])
        self._wandb_train_seconds = 0.0
        self._wandb_planner_seconds = 0.0
        self._wandb_last_updates = int(wandb_state["last_updates"])
        self._last_wandb_step = candidate["last_step"]
        now = time.perf_counter()
        self._wandb_start_time = now
        self._wandb_window_start_time = now
        self._last_reward = 0.0
        self._last_terminated = False
        self._last_truncated = False
        self._last_info = {}
        self._last_train_metrics = None
        self._predict_t0 = True
        return self

    def _prepare_eval_csv(self):
        """Create one tiny, official-format evaluation CSV for this seed."""
        if self._eval_csv_initialized or not self._eval_csv_path:
            return
        path = os.path.abspath(os.path.expanduser(self._eval_csv_path))
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            raise FileExistsError(
                f"Refusing to mix TD-MPC2 evaluation rows with existing file: {path}"
            )
        with open(path, "w", newline="") as stream:
            header = (
                ("step", "episode", "reward", "seed")
                if getattr(self, "_resume_enabled", False)
                else ("step", "reward", "seed")
            )
            csv.writer(stream).writerow(header)
            stream.flush()
            os.fsync(stream.fileno())
        self._eval_csv_path = path
        self._eval_csv_initialized = True

    def _evaluation_payload_extras(self, step):
        """Return optional metrics to merge into this evaluation event.

        Subclasses can run observational probes here without adding a second
        evaluation scheduler or a second W&B event at the same environment
        step. The base TD-MPC2 evaluator has no extra metrics.
        """

        return {}

    def _record_evaluation(self, step, reward, *, extras=None):
        reward = float(reward)
        print(
            "TD-MPC2 evaluation:",
            f"step={int(step)},",
            f"reward={reward:.1f},",
            f"episodes={self._eval_episodes},",
            f"seed={self.cfg.seed}",
            flush=True,
        )
        if self._eval_csv_path:
            self._prepare_eval_csv()
            with open(self._eval_csv_path, "a", newline="") as stream:
                row = (
                    (
                        int(step),
                        int(self._episode_idx),
                        f"{reward:.1f}",
                        int(self.cfg.seed),
                    )
                    if getattr(self, "_resume_enabled", False)
                    else (int(step), f"{reward:.1f}", int(self.cfg.seed))
                )
                csv.writer(stream).writerow(row)
                stream.flush()
                os.fsync(stream.fileno())
        payload = {
            "eval/episode_reward": reward,
            "eval/episodes": int(self._eval_episodes),
        }
        if getattr(self, "_resume_enabled", False):
            payload["episode/index"] = int(self._episode_idx)
        if extras is not None:
            if not isinstance(extras, Mapping):
                raise TypeError("Evaluation payload extras must be a mapping.")
            reserved = {
                "eval/episode_reward",
                "eval/episodes",
                "episode/index",
                "env_step",
            }
            collisions = reserved.intersection(extras)
            if collisions:
                raise ValueError(
                    "Evaluation payload extras collide with reserved metrics: "
                    f"{sorted(collisions)}."
                )
            for key, value in extras.items():
                if not isinstance(key, str) or not key:
                    raise TypeError(
                        "Evaluation payload extra keys must be non-empty strings."
                    )
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    raise TypeError(
                        f"Evaluation payload extra {key!r} must be numeric."
                    )
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError(
                        f"Evaluation payload extra {key!r} must be finite."
                    )
                payload[key] = numeric
        log_wandb(self._wandb_run, payload, step=int(step))

    @torch.no_grad()
    def _evaluate_policy(self, step, *, initial_obs=None):
        """Run the official ten-episode deterministic-policy evaluation."""
        episode_returns = []
        for episode in range(self._eval_episodes):
            if episode == 0 and initial_obs is not None:
                obs = initial_obs
                self.reset()
            else:
                obs, _ = self._reset_env()

            episode_return = 0.0
            episode_step = 0
            done = False
            while not done:
                obs_t = self._reuse_observation_tensor(obs)
                action_norm = self._act_agent(
                    obs_t,
                    t0=(episode_step == 0),
                    eval_mode=True,
                ).numpy()
                action_env = self._unscale_action(action_norm)
                obs, reward, terminated, truncated, _ = self.env.step(action_env)
                episode_return += float(reward)
                episode_step += 1
                done = bool(terminated or truncated)
            episode_returns.append(episode_return)

        mean_return = float(np.nanmean(episode_returns))
        extras = self._evaluation_payload_extras(int(step))
        if extras:
            self._record_evaluation(step, mean_return, extras=extras)
        else:
            # Preserve the historical two-argument seam used by downstream
            # evaluators and tests when no observational extension is active.
            self._record_evaluation(step, mean_return)
        return mean_return

    def _prepare_resume_boundary(self):
        # Resume-mode metric events must depend on the scientific episode
        # trajectory, never on where Slurm happened to split the process.
        self._force_resume_metric_boundary()
        prepare = getattr(self.agent, "prepare_training_resume_boundary", None)
        if not callable(prepare):
            raise TypeError("The agent lacks episode-boundary resume preparation.")
        prepare()
        self._predict_t0 = True
        self._resume_phase = "between_episodes_before_reset"

    def _force_resume_metric_boundary(self):
        self._log_wandb_step(
            self._last_reward,
            self._last_terminated,
            self._last_truncated,
            self._last_info,
            force=True,
        )

    def _resume_exit_at_boundary(self, resume_session, *, reason, complete=False):
        generation = resume_session.publish(self, reason=reason)
        if complete:
            return resume_session.complete(self, generation)
        return resume_session.clean_handoff(self, generation)

    def _checkpoint_after_resume_evaluation(self, resume_session, *, label):
        """Honor timers/signals that became due while evaluation was active."""

        if resume_session.drain_requested():
            return self._resume_exit_at_boundary(
                resume_session, reason=f"drain-after-{label}"
            )
        if not resume_session.checkpoint_due():
            return None
        generation = resume_session.publish(
            self, reason=f"hourly-after-{label}"
        )
        if resume_session.drain_requested():
            generation = resume_session.publish(
                self, reason=f"drain-during-{label}-checkpoint"
            )
            return resume_session.clean_handoff(self, generation)
        return None

    def _run_training_episode(self, obs, total_timesteps, *, eval_pending):
        """Run the one shared environment-step/update loop to a safe boundary."""

        obs_t = self._reuse_observation_tensor(obs)
        episode_rows = self._start_episode_staging(obs_t)
        episode_step = 0
        while self._global_step < total_timesteps:
            planned = not (
                self._global_step <= self.cfg.seed_steps
                or self.buffer.num_eps == 0
            )
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
            next_obs, reward, terminated, truncated, info = self.env.step(
                action_env
            )
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
            if (
                self._eval_freq is not None
                and self._global_step % self._eval_freq == 0
            ):
                eval_pending = True
            episode_step += 1
            self._episode_return += float(reward)
            self._episode_len += 1
            self._last_reward = float(reward)
            self._last_terminated = bool(terminated)
            self._last_truncated = bool(truncated)
            self._last_info = dict(info or {})
            self._accumulate_reward_metrics(reward, info)
            self._log_step(
                reward, next_obs, action_env, terminated, truncated, info
            )

            if done:
                if true_terminated and not self.cfg.episodic:
                    raise ValueError(
                        "TD-MPC2 saw terminated=True while episodic=False. "
                        "Set alg_params.episodic=true or disable true terminations in the env."
                    )
                self.buffer.add(self._episode_staging[:episode_rows])

            if self._global_step > self.cfg.seed_steps and self.buffer.num_eps > 0:
                num_updates = (
                    self.cfg.pretrain_steps if not self._pretrained else self.cfg.utd
                )
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
                return True, eval_pending
            obs = next_obs
            obs_t = next_obs_t
        return False, eval_pending

    def _start_resumable_training(self, total_timesteps, resume_session):
        """Restore/start one segment and stop before the shared step loop."""

        from utils.resume_training import ResumeIncompatibilityError

        resume_session.prepare_learner(self)
        if self._global_step == total_timesteps and not (
            self._resume_phase == "before_initial_seeded_reset"
            and self._eval_pending
        ):
            generation = (
                resume_session.publish(self, reason="target-recovery")
                if resume_session.mode == "required"
                else resume_session.last_generation
            )
            return None, resume_session.complete(self, generation)

        phase = self._resume_phase
        if phase == "before_initial_seeded_reset":
            if resume_session.drain_requested():
                return None, self._resume_exit_at_boundary(
                    resume_session, reason="drain-before-initial-reset"
                )
            obs, _ = self._reset_env(seed=self.cfg.seed)
            if self._eval_pending:
                self._prepare_eval_csv()
                self._evaluate_policy(0, initial_obs=obs)
                self._eval_pending = False
                self._step_zero_eval_done = True
                self._prepare_resume_boundary()
                if self._global_step == total_timesteps:
                    generation = resume_session.publish(
                        self, reason="target-after-step-zero-eval"
                    )
                    return None, resume_session.complete(self, generation)
                result = self._checkpoint_after_resume_evaluation(
                    resume_session, label="step-zero-eval"
                )
                if result is not None:
                    return None, result
                obs, _ = self._reset_env()
        elif phase == "between_episodes_before_reset":
            if resume_session.drain_requested():
                return None, self._resume_exit_at_boundary(
                    resume_session, reason="drain-before-pending-eval"
                )
            if self._eval_pending:
                self._evaluate_policy(self._global_step)
                self._eval_pending = False
                self._prepare_resume_boundary()
                result = self._checkpoint_after_resume_evaluation(
                    resume_session, label="pending-eval"
                )
                if result is not None:
                    return None, result
            obs, _ = self._reset_env()
        else:
            raise ResumeIncompatibilityError(
                f"Unsupported resumable learner phase {phase!r}."
            )
        self._resume_phase = "in_episode"
        return obs, None

    def _finish_resumable_episode(self, total_timesteps, resume_session):
        """Commit/evaluate one completed episode before the next reset."""

        if self._global_step == total_timesteps:
            self._final_checkpoint()
        self._episode_idx += 1
        self._episode_return = 0.0
        self._episode_len = 0
        self._prepare_resume_boundary()

        if self._global_step == total_timesteps:
            if self._eval_pending:
                self._evaluate_policy(self._global_step)
                self._eval_pending = False
                self._prepare_resume_boundary()
            generation = resume_session.publish(self, reason="target")
            return None, resume_session.complete(self, generation)

        drain = resume_session.drain_requested()
        due = resume_session.checkpoint_due()
        generation = None
        if due or drain:
            generation = resume_session.publish(
                self, reason="drain" if drain else "hourly"
            )

        # One newer generation covers a signal delivered during an ordinary
        # hourly publication; repeated signals during that drain publication
        # remain idempotent.
        if resume_session.drain_requested():
            if not drain:
                generation = resume_session.publish(
                    self, reason="drain-during-checkpoint"
                )
            return None, resume_session.clean_handoff(self, generation)

        if self._eval_pending:
            self._evaluate_policy(self._global_step)
            self._eval_pending = False
            self._prepare_resume_boundary()
            result = self._checkpoint_after_resume_evaluation(
                resume_session, label="pending-eval"
            )
            if result is not None:
                return None, result

        obs, _ = self._reset_env()
        self._resume_phase = "in_episode"
        return obs, None

    def _learn_resumable(self, total_timesteps, resume_session):
        """Train through exact episode boundaries under a durable lineage."""

        from utils.resume_training import ResumeIncompatibilityError

        if not getattr(self, "_resume_enabled", False):
            raise RuntimeError("enable_training_resume() must precede resumable learn().")
        if total_timesteps != int(self.cfg.steps):
            raise ResumeIncompatibilityError(
                "The resumable absolute target differs from the constructed learner."
            )
        failed = False
        try:
            obs, result = self._start_resumable_training(
                total_timesteps, resume_session
            )
            if result is not None:
                return result
            while self._global_step < total_timesteps:
                completed, self._eval_pending = self._run_training_episode(
                    obs,
                    total_timesteps,
                    eval_pending=self._eval_pending,
                )
                if not completed:
                    raise ResumeIncompatibilityError(
                        "The absolute target was reached inside an active episode; "
                        "partial-episode physics snapshots are unsupported."
                    )
                obs, result = self._finish_resumable_episode(
                    total_timesteps, resume_session
                )
                if result is not None:
                    return result
            raise AssertionError("Resumable loop exited without a boundary result.")
        except BaseException as primary_error:
            failed = True
            resume_session.abort_wandb(self, primary_error)
            try:
                self._checkpoint_writer.shutdown()
            except BaseException as cleanup_error:
                add_cleanup_notes(
                    primary_error,
                    (cleanup_error,),
                    prefix=(
                        "Additional model-snapshot cleanup failure after segmented "
                        "training stopped"
                    ),
                )
            raise
        finally:
            if not failed:
                self._checkpoint_writer.shutdown()

    def learn(self, total_timesteps=10000, *, resume_session=None):
        total_timesteps = validate_timestep_budget(total_timesteps)
        if resume_session is not None:
            return self._learn_resumable(total_timesteps, resume_session)
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

        primary_error = None
        cleanup_errors = []
        try:
            self._reset_wandb_window()
            obs, _ = self._reset_env(seed=self.cfg.seed)
            if self._eval_freq is not None:
                self._prepare_eval_csv()
                # Reuse the authoritative seeded reset as evaluation episode
                # one. This matches upstream, whose environment is constructed
                # with cfg.seed immediately before its step-zero evaluation.
                self._evaluate_policy(0, initial_obs=obs)
                obs, _ = self._reset_env()
            eval_pending = False

            while self._global_step < total_timesteps:
                completed, eval_pending = self._run_training_episode(
                    obs,
                    total_timesteps,
                    eval_pending=eval_pending,
                )
                if not completed:
                    break
                if eval_pending:
                    self._evaluate_policy(self._global_step)
                    eval_pending = False
                self._episode_idx += 1
                self._episode_return = 0.0
                self._episode_len = 0
                obs, _ = self._reset_env()
            self._final_checkpoint()
            return self
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            if self._wandb_run is not None:
                try:
                    self._log_wandb_step(
                        self._last_reward,
                        self._last_terminated,
                        self._last_truncated,
                        self._last_info,
                        force=True,
                    )
                except BaseException as exc:
                    cleanup_errors.append(exc)
                try:
                    finish_wandb(self._wandb_run)
                except BaseException as exc:
                    cleanup_errors.append(exc)
                finally:
                    self._wandb_run = None
            if self.alg_logger is not None and hasattr(self.alg_logger, "flush"):
                try:
                    self.alg_logger.flush()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            # Periodic snapshots are exact at enqueue time, but an exception
            # or normal shutdown must still make the queued atomic replacement
            # durable before control returns.
            try:
                self._checkpoint_writer.shutdown()
            except BaseException as exc:
                cleanup_errors.append(exc)
            if cleanup_errors:
                if primary_error is not None:
                    add_cleanup_notes(
                        primary_error,
                        cleanup_errors,
                        prefix="Additional learner cleanup failure",
                    )
                else:
                    raise_cleanup_errors(cleanup_errors)

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
