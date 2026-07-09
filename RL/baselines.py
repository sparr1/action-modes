import numpy as np
import gymnasium as gym
import importlib, json, os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback as BaselineCheckpointCallback
from RL.alg import Algorithm
from utils.utils import setup_logs
from utils.wandb_utils import finish_wandb, init_wandb, log_wandb, wandb_enabled
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
    def __init__(self, name, env, params = None):
        super().__init__(name, env, custom_params=params)
        self.params = params or {}
        self.model = self.get_baseline_model(self.name, self.env, self.params)
        self.callback = []
        if wandb_enabled(self.params):
            self.callback.append(WandbBaselineCallback(self.name, self.params, env))

    def learn(self, **kwargs):
        return self.model.learn(callback = self.callback, **kwargs) # pass this env into ambi

    def predict(self, observation):
        return self.model.predict(observation)

    def save(self, path, name):
        self.model.save(os.path.join(path, name))

    def get_model(self):
        return self.model

    def set_logger(self, logger):
        super().set_logger(logger)
        self.callback.append(TrajectoryLoggerCallback(self.alg_logger))

    def set_checkpointing(self, save_freq, save_path, name_prefix):
        self.callback.append(BaselineCheckpointCallback(save_freq, save_path, name_prefix, False, False, 2))

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
        self._wandb_every = int(self.params.get("wandb_step_every", 1000))
        self._wandb_run = None
        self._episode_idx = 0
        self._episode_return = 0.0
        self._episode_len = 0

    def _on_training_start(self) -> None:
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

    def _logger_payload(self):
        payload = {}
        values = getattr(getattr(self.model, "logger", None), "name_to_value", {})
        for key, value in values.items():
            if not isinstance(key, str) or not key.startswith(("train/", "rollout/", "time/")):
                continue
            try:
                payload[key] = float(value)
            except (TypeError, ValueError):
                pass
        return payload

    def _on_step(self) -> bool:
        reward = _as_float(self.locals.get("rewards", 0.0))
        done = _as_bool(self.locals.get("dones", False))
        info = _first_info(self.locals.get("infos", []))
        truncated = bool(info.get("TimeLimit.truncated", info.get("truncated", False)))
        terminated = bool(info.get("terminated", done and not truncated))

        self._episode_return += reward
        self._episode_len += 1

        step = int(self.num_timesteps)
        if self._wandb_run is not None and (done or self._wandb_every <= 1 or step % self._wandb_every == 0):
            payload = {
                "train/reward": reward,
                "train/done": int(done),
                "train/terminated": int(terminated),
                "train/truncated": int(truncated),
                "episode/current_return": float(self._episode_return),
                "episode/current_len": int(self._episode_len),
            }
            payload.update(self._logger_payload())
            log_wandb(self._wandb_run, payload, step=step)

        if done:
            if self._wandb_run is not None:
                log_wandb(
                    self._wandb_run,
                    {
                        "episode/index": int(self._episode_idx),
                        "episode/return": float(self._episode_return),
                        "episode/len": int(self._episode_len),
                    },
                    step=step,
                )
            self._episode_idx += 1
            self._episode_return = 0.0
            self._episode_len = 0
        return True

    def _on_training_end(self) -> None:
        finish_wandb(self._wandb_run)
        self._wandb_run = None
