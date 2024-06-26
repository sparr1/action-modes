import numpy as np
import gymnasium as gym
import importlib, json, os
from stable_baselines3.common.callbacks import BaseCallback

#from stable_baselines3 import PPO, DQN, TD3, SAC, DDPG, A2C
module_name = "stable_baselines3" #for dynamic importing

#wrapper on stable baselines
class Baseline():
    def __init__(self, name, env, params = None):
        self.name = name
        self.env = env
        self.custom_params = params
        self.model = self._get_baseline_model(name, env, params)
    
    def get_model(self):
        return self.model
    def learn(self):
        pass
    def predict(self):
        pass
    def vec_env(self):
        pass

    def _get_baseline_model(self, name, env, params = None):
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
    import json

class TrajectoryLoggerCallback(BaseCallback):

    def __init__(self, log_dir, verbose=0):
        super(TrajectoryLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_observations = []
        self.episode_actions = []
        self.log_dir = log_dir
        self.summary_file = os.path.join(log_dir, 'stats.txt')
        self.train_episodes_dir = os.path.join(log_dir, 'train_episodes')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.train_episodes_dir, exist_ok=True)

        self.episode_count = 0

    def _on_step(self) -> bool:
        # print('rewards', type(self.locals['rewards']), self.locals['rewards'].shape)
        # print('new_obs', type(self.locals['new_obs']), self.locals['new_obs'].shape)
        # print('actions', type(self.locals['actions']), self.locals['actions'].shape)
        # print(self.locals['rewards'])
        # print(self.locals['rewards'].tolist())
        self.episode_rewards.extend(self.locals['rewards'].tolist())
        self.episode_observations.extend(self.locals['new_obs'].tolist())
        self.episode_actions.extend(self.locals['actions'].tolist())
        return True

    def _on_rollout_end(self) -> None:
        trajectory = {
            "rewards": self.episode_rewards,
            "observations": self.episode_observations,
            "actions": self.episode_actions
        }
        with open(os.path.join(self.train_episodes_dir, f"episode_{self.episode_count}.json"), 'w') as f:
            json.dump(trajectory, f)

        # Calculate high-level summary
        print(self.episode_rewards)
        total_reward = sum(self.episode_rewards)
        print(total_reward)
        avg_reward = np.mean(self.episode_rewards)
        num_steps = len(self.episode_rewards)

        # Write summary to a file
        with open(self.summary_file, 'a') as f:
            f.write(f"episode_{self.episode_count}: Total Reward = {total_reward}, Average Reward = {avg_reward}, Steps = {num_steps}\n")

        self.episode_rewards = []
        self.episode_observations = []
        self.episode_actions = []
        self.episode_count += 1