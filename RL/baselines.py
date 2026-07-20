import importlib
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter
from RL.alg import Algorithm
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
    wandb_enabled,
)
#from stable_baselines3 import PPO, DQN, TD3, SAC, DDPG, A2C
module_name = "stable_baselines3" #for dynamic importing

WANDB_PARAM_KEYS = {
    "wandb",
    "wandb_project",
    "wandb_entity",
    "wandb_run_name",
    "wandb_mode",
    "wandb_tags",
    "wandb_step_every",
}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_float(value, default=0.0):
    values = _as_list(value)
    if values:
        value = values[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value):
    values = _as_list(value)
    if not values:
        return False
    return bool(values[0])


def _first_info(infos):
    infos = _as_list(infos)
    return infos[0] if infos and isinstance(infos[0], dict) else {}


def _wandb_clean_params(params):
    return {k: v for k, v in (params or {}).items() if k not in WANDB_PARAM_KEYS}

#TODO: I'm using params everywhere to refer to _input_ parameters. but in a machine learning context,
# theres a hash collision on "params", since it usually refers to tunable parameters of the model itself...
#wrapper on stable baselines
class Baseline(Algorithm):
    supports_composable_checkpointing = True

    def __init__(
        self,
        name,
        env,
        params=None,
        run_params=None,
        experiment_params=None,
    ):
        super().__init__(name, env, custom_params=params)
        self.params = params or {}
        self.run_params = run_params or {}
        self.experiment_params = experiment_params or {}
        self.model = self.get_baseline_model(self.name, self.env, self.params)
        self.callback = []
        self._checkpoint_callback = None
        if wandb_enabled(self.params):
            self.callback.append(WandbBaselineCallback(self.name, self.params, env))

    def learn(self, **kwargs):
        try:
            return self.model.learn(callback=self.callback, **kwargs)  # pass this env into ambi
        finally:
            # SB3 does not call ``on_training_end`` when learning raises. Keep W&B
            # lifecycle cleanup idempotent so failed runs are not left open.
            for callback in self.callback:
                if isinstance(callback, WandbBaselineCallback):
                    callback.finish()

    def predict(self, observation):
        return self.model.predict(observation)

    def save(self, path, name):
        step = int(getattr(self.model, "num_timesteps", 0))
        episode = int(getattr(self.model, "_episode_num", 0))
        target_path = Path(path) / name
        if self._checkpoint_callback is not None:
            tracker = self._checkpoint_callback.tracker
            target = tracker.explicit_target(
                target_path,
                step=step,
                episode=tracker.episode_count,
            )
        else:
            target = explicit_checkpoint_target(
                target_path,
                step=step,
                episode=episode,
                trial_run_params=self.run_params,
                experiment_params=self.experiment_params,
            )
        published = publish_checkpoint(
            (target,),
            lambda staging: self.model.save(staging),
            extension=".zip",
        )
        return published[0]

    def get_model(self):
        return self.model

    def set_logger(self, logger):
        super().set_logger(logger)
        self.callback.append(TrajectoryLoggerCallback(self.alg_logger))

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
        if self._checkpoint_callback is not None:
            try:
                self.callback.remove(self._checkpoint_callback)
            except ValueError:
                pass
        tracker = CheckpointTracker(
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
        self._checkpoint_callback = ComposableCheckpointCallback(tracker)
        self.callback.append(self._checkpoint_callback)
        return tracker

    def delete_model(self):
        del self.model

    def load(self, path):#requires retrieving the correct baseline model before loading weights
        self.model = self.model.load(path)

    def get_baseline_model(self, name, env, params = None):
        p = {}

        model = None

        #needed to handle gymnasium robotics observed_goal structure properly
        if not params or "policy" not in params:
            if type(env.observation_space) == gym.spaces.dict.Dict:
                p["policy"] = "MultiInputPolicy"
            else:
                p["policy"] = "MlpPolicy" #just a guess.
                # print("Trying MlpPolicy as none given.")

        p["env"] = env

        if params:
            p.update(_wandb_clean_params(params))

        if "verbose" not in p.keys():
            p["verbose"] = 1
        try:
            module = importlib.import_module(module_name)
            model_class = getattr(module,name) #grab the specific algorithm

            model = model_class(**p) #make sure the json elements actually correspond to parameters of the relevant stable_baseline alg!

            return model

        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Could not find model class '{name}' in module '{module_name}': {e}")

class TrajectoryLoggerCallback(BaseCallback):
    def __init__(self, logger, verbose=0):
        super(TrajectoryLoggerCallback, self).__init__(verbose)
        self.traj_logger = logger

    def _on_episode(self) -> None:
        self.traj_logger.on_episode()

    def _on_step(self) -> bool:
        if self.traj_logger._log_info:
            data = setup_logs(self.locals["rewards"],
                              self.locals["new_obs"],
                          self.locals["actions"],
                            self.locals["dones"],
                            self.locals["infos"])
        else:
            data = setup_logs(self.locals["rewards"],
                              self.locals["new_obs"],
                          self.locals["actions"],
                            self.locals["dones"])

        self.traj_logger.on_step(data = data)
        return True


class ComposableCheckpointCallback(BaseCallback):
    """SB3 callback for numbered, rolling-best, and latest aliases."""

    def __init__(self, tracker, verbose=0):
        super().__init__(verbose)
        self.tracker = tracker
        self._episode_returns = None

    def _on_training_start(self) -> None:
        n_envs = int(getattr(self.model, "n_envs", 1))
        if n_envs <= 0:
            raise ValueError("SB3 checkpointing requires at least one environment.")
        reset = bool(self.locals.get("reset_num_timesteps", True))
        if reset:
            self.tracker.reset()
        if self._episode_returns is None or reset or len(self._episode_returns) != n_envs:
            self._episode_returns = np.zeros(n_envs, dtype=np.float64)

    def _publish(self, targets):
        targets = tuple(targets)
        if not targets:
            return ()
        return publish_checkpoint(
            targets,
            lambda staging: self.model.save(staging),
            extension=".zip",
        )

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", 0.0), dtype=np.float64).reshape(-1)
        dones = np.asarray(self.locals.get("dones", False), dtype=bool).reshape(-1)
        if self._episode_returns is None:
            self._episode_returns = np.zeros(len(rewards), dtype=np.float64)
        if len(rewards) != len(self._episode_returns) or len(dones) != len(rewards):
            raise ValueError("SB3 checkpoint callback received inconsistent vector-env outputs.")
        self._episode_returns += rewards
        for index in np.flatnonzero(dones):
            self.tracker.record_episode_return(self._episode_returns[index])
            self._episode_returns[index] = 0.0
        self._publish(self.tracker.targets(int(self.num_timesteps)))
        return True

    def _on_training_end(self) -> None:
        # SB3 only invokes this hook after a clean learn() completion. Exceptions
        # therefore retain the last cadence save without publishing a false final
        # "latest" alias.
        self._publish(self.tracker.targets(int(self.num_timesteps), final=True))


class _WandbKVWriter(KVWriter):
    """Capture scalar values at SB3's logger-dump boundary before it clears them."""

    def __init__(self, callback):
        self.callback = callback

    def write(self, key_values, key_excluded, step=0):
        del key_excluded
        payload = {}
        for key, value in key_values.items():
            if not isinstance(key, str) or not key.startswith(("rollout/", "time/")):
                continue
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(scalar):
                payload[key] = scalar
        if payload:
            self.callback._log_dump_payload(payload, int(step))

    def close(self):
        return None


class WandbBaselineCallback(BaseCallback):
    """Optional W&B logging for Stable-Baselines3 baselines.

    This callback is intentionally read-only: it observes SB3 locals/logger values
    and never changes rollout, replay-buffer, or optimization behavior.
    """

    def __init__(self, alg_name, params, env, verbose=0):
        super().__init__(verbose)
        self.alg_name = alg_name
        self.params = params or {}
        self.env_id = getattr(getattr(env, "spec", None), "id", "env")
        self.seed = self.params.get("seed", None)
        self._wandb_every = max(1, int(self.params.get("wandb_step_every", 1000)))
        self._wandb_run = None
        self._episode_idx = 0
        self._episode_return = 0.0
        self._episode_len = 0
        self._train_window = WandbAccumulator()
        self._reward_window = WandbAccumulator()
        self._last_seen_updates = 0
        self._last_log_step = None
        self._last_reward = 0.0
        self._last_done = False
        self._last_terminated = False
        self._last_truncated = False
        self._last_info = {}
        self._start_time = None
        self._window_start_time = None
        self._window_start_step = 0
        self._kv_writer = None

    def _on_training_start(self) -> None:
        if int(getattr(self.model, "n_envs", 1)) != 1:
            raise ValueError("W&B baseline logging currently requires exactly one SB3 environment.")
        if bool(self.locals.get("reset_num_timesteps", True)):
            self._episode_idx = 0
            self._episode_return = 0.0
            self._episode_len = 0
        self._train_window.clear()
        self._reward_window.clear()
        self._last_seen_updates = int(getattr(self.model, "_n_updates", 0))
        self._last_log_step = None
        self._start_time = time.perf_counter()
        self._window_start_time = self._start_time
        self._window_start_step = int(self.num_timesteps)
        run_name = f"SB3{self.alg_name}-{self.env_id}"
        if self.seed is not None:
            run_name += f"-seed{self.seed}"
        self._wandb_run = init_wandb(
            self.params,
            default_project="ambi",
            run_name=run_name,
            config={
                "algorithm": f"SB3/{self.alg_name}",
                "env": self.env_id,
                "sb3_params": _wandb_clean_params(self.params),
            },
        )
        self._kv_writer = _WandbKVWriter(self)
        self.model.logger.output_formats.append(self._kv_writer)

    def _on_rollout_start(self) -> None:
        self._capture_train_metrics()
        # If optimization happened immediately after an already-logged rollout
        # boundary, append it to that same explicit W&B step.
        step = int(self.num_timesteps)
        if self._wandb_run is not None and self._last_log_step == step and self._train_window:
            payload = self._train_window.pop()
            payload.update(self._replay_payload())
            payload["train/n_updates"] = int(getattr(self.model, "_n_updates", 0))
            log_wandb(self._wandb_run, payload, step=step)

    def _capture_train_metrics(self):
        values = getattr(getattr(self.model, "logger", None), "name_to_value", {})
        current_updates = int(values.get("train/n_updates", getattr(self.model, "_n_updates", 0)))
        delta_updates = current_updates - self._last_seen_updates
        if delta_updates <= 0:
            return
        for key, value in values.items():
            if not isinstance(key, str) or not key.startswith("train/") or key == "train/n_updates":
                continue
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(scalar):
                self._train_window.add_weighted(key, scalar, weight=delta_updates)
        self._train_window.add_sum("train/updates_since_log", delta_updates)
        self._last_seen_updates = current_updates

    def _replay_payload(self):
        replay = getattr(self.model, "replay_buffer", None)
        if replay is None:
            return {}
        n_envs = int(getattr(replay, "n_envs", 1))
        try:
            size = int(replay.size()) * n_envs
        except (AttributeError, TypeError):
            return {}
        capacity = int(getattr(replay, "buffer_size", 0)) * n_envs
        return {
            "train/replay_size": size,
            "train/replay_capacity": capacity,
            "train/replay_fill_ratio": float(size / capacity) if capacity > 0 else 0.0,
        }

    def _timing_payload(self, step):
        now = time.perf_counter()
        start = self._start_time if self._start_time is not None else now
        window_start = self._window_start_time if self._window_start_time is not None else now
        window_seconds = max(0.0, now - window_start)
        window_steps = max(0, int(step) - int(self._window_start_step))
        payload = {
            "time/window_seconds": float(window_seconds),
            "time/time_elapsed": float(max(0.0, now - start)),
            "time/total_timesteps": int(step),
            "time/fps": float(window_steps / window_seconds) if window_seconds > 0 else 0.0,
        }
        self._window_start_time = now
        self._window_start_step = int(step)
        return payload

    def _log_dump_payload(self, payload, step):
        if self._wandb_run is not None:
            log_wandb(self._wandb_run, payload, step=int(step))

    def _emit(self, step, *, completed_episode=False, force=False):
        if self._wandb_run is None:
            return
        if not force and not completed_episode and step % self._wandb_every != 0:
            return

        payload = {
            "train/reward": float(self._last_reward),
            "train/done": int(self._last_done),
            "train/terminated": int(self._last_terminated),
            "train/truncated": int(self._last_truncated),
            "train/learning_started": int(step > int(getattr(self.model, "learning_starts", 0))),
            "train/n_updates": int(getattr(self.model, "_n_updates", 0)),
            "episode/current_return": float(self._episode_return),
            "episode/current_len": int(self._episode_len),
        }
        payload.setdefault("train/updates_since_log", 0)
        payload.update(self._reward_window.pop())
        train_payload = self._train_window.pop()
        if train_payload:
            payload.update(train_payload)
        payload.update(self._replay_payload())
        payload.update(extract_reward_components(self._last_info))
        payload.update(self._timing_payload(step))
        if completed_episode:
            payload.update({
                "episode/index": int(self._episode_idx),
                "episode/return": float(self._episode_return),
                "episode/len": int(self._episode_len),
            })
        log_wandb(self._wandb_run, payload, step=step)
        self._last_log_step = int(step)

    def _on_step(self) -> bool:
        reward = _as_float(self.locals.get("rewards", 0.0))
        done = _as_bool(self.locals.get("dones", False))
        info = _first_info(self.locals.get("infos", []))
        truncated = bool(info.get("TimeLimit.truncated", info.get("truncated", False)))
        terminated = bool(info.get("terminated", done and not truncated))

        self._episode_return += reward
        self._episode_len += 1
        self._last_reward = reward
        self._last_done = done
        self._last_terminated = terminated
        self._last_truncated = truncated
        self._last_info = info
        self._reward_window.add_stats("rollout/reward", [reward])

        step = int(self.num_timesteps)
        self._emit(step, completed_episode=done)
        if done:
            self._episode_idx += 1
            self._episode_return = 0.0
            self._episode_len = 0
        return True

    def _on_training_end(self) -> None:
        self._capture_train_metrics()
        step = int(self.num_timesteps)
        if self._wandb_run is not None:
            if self._last_log_step == step and self._train_window:
                payload = self._train_window.pop()
                payload.update(self._replay_payload())
                payload["train/n_updates"] = int(getattr(self.model, "_n_updates", 0))
                log_wandb(self._wandb_run, payload, step=step)
            elif (
                step != self._window_start_step
                or self._train_window
                or self._reward_window
            ):
                self._emit(step, force=True)
        self.finish()

    def finish(self):
        if self._kv_writer is not None and getattr(self, "model", None) is not None:
            output_formats = getattr(getattr(self.model, "logger", None), "output_formats", [])
            if self._kv_writer in output_formats:
                output_formats.remove(self._kv_writer)
            self._kv_writer = None
        if self._wandb_run is not None:
            finish_wandb(self._wandb_run)
            self._wandb_run = None
