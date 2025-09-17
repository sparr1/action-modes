import numpy as np
import gymnasium as gym
import importlib, json, os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback as BaselineCheckpointCallback
from RL.alg import Algorithm
from utils.utils import setup_logs
#from stable_baselines3 import PPO, DQN, TD3, SAC, DDPG, A2C
module_name = "stable_baselines3" #for dynamic importing

#TODO: I'm using params everywhere to refer to _input_ parameters. but in a machine learning context, 
# theres a hash collision on "params", since it usually refers to tunable parameters of the model itself...
#wrapper on stable baselines
class Baseline(Algorithm):
    def __init__(self, name, env, params = None):
        super().__init__(name, env, custom_params=params)
        self.model = self.get_baseline_model(self.name, self.env, self.custom_params)
        self.callback = []

    def learn(self, **kwargs):
        return self.model.learn(callback = self.callback, **kwargs)
    
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
                print("Trying MlpPolicy as none given.")

        p["env"] = env

        if params:
            p.update(params)

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
    