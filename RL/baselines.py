import numpy as np
import gymnasium as gym
import importlib, json, os
from stable_baselines3.common.callbacks import BaseCallback
from RL.alg import Algorithm
#from stable_baselines3 import PPO, DQN, TD3, SAC, DDPG, A2C
module_name = "stable_baselines3" #for dynamic importing

#TODO: I'm using params everywhere to refer to _input_ parameters. but in a machine learning context, 
# theres a hash collision on "params", since it usually refers to tunable parameters of the model itself...
#wrapper on stable baselines
class Baseline(Algorithm):
    def __init__(self, name, env, params = None):
        super().__init__(name, env, custom_params=params)
        self.model = self.get_baseline_model(self.name, self.env, self.custom_params)
    
    def get_model(self):
        return self.model

    def learn(self, **kwargs):
        return self.model.learn(**kwargs)
    
    def save(self, path, name):
        self.model.save(os.path.join(path, name))

    def delete_model(self): #requires retrieving a baseline model before loading
        del self.model

    def load_model(self, path):
        pass

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
        
#we will have a few different settings for logs: "none", "overwrite", "warn", and "timestamp"
#the first just does not save logs, the second will ALWAYS delete any folders under the name of the experiment prior to running, the third will refuse to run if there are folders, 
#and the fourth will prepend a timestamp in order to always create new folders each time it has run repeatedly. For serious experiments, I recommend "warn" or "timestamp".
#The default is "warn", since this will let you know that some configuration is required to get the behavior you want.
#For less serious experiments, testing, etc., I recommend "none" or "overwrite", depending on whether you are testing a capability that uses the logs or not.
class TrajectoryLoggerCallback(BaseCallback):

    def __init__(self, log_dir, log_setting = "warn", verbose=0):
        super(TrajectoryLoggerCallback, self).__init__(verbose)
        if log_setting not in ["overwrite", "warn", "timestamp"]:
            raise Exception("unsupported logging setting. Try none, overwrite, warn, or timestamp.")
        self.episode_rewards = []
        self.episode_observations = []
        self.episode_actions = []
        self.log_dir = log_dir
        self.summary_file = os.path.join(log_dir, 'stats.txt')
        self.train_episodes_dir = os.path.join(log_dir, 'train_episodes')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.train_episodes_dir, exist_ok=True)
        self.step_count = 0
        self.rollout_count = 0
        self.training_count = 0 #surely this should end up 1.
        self.episode_count = 0
        self.episode_step_count = 0
        print("episode", self.episode_count, "with total timesteps", self.step_count)

    def _on_episode(self) -> None:
        self.episode_count += 1

        trajectory = {
            "rewards": self.episode_rewards,
            "observations": self.episode_observations,
            "actions": self.episode_actions
        }
        with open(os.path.join(self.train_episodes_dir, f"episode_{self.episode_count}.json"), 'w') as f:
            json.dump(trajectory, f)

        # Calculate high-level summary
        # print(self.episode_rewards)
        total_reward = sum(self.episode_rewards)
        # print(total_reward)
        avg_reward = np.mean(self.episode_rewards)
        num_steps = len(self.episode_rewards)

        # Write summary to a file
        with open(self.summary_file, 'a') as f:
            f.write(f"episode_{self.episode_count}: Total Reward = {total_reward}, Average Reward = {avg_reward}, Steps = {num_steps}\n")

        self.episode_rewards = []
        self.episode_observations = []
        self.episode_actions = []
        self.episode_step_count = 0

    def _on_step(self) -> bool:
        # Check that the `dones` local variable is defined
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
        self.step_count += 1
        self.episode_step_count += 1
        self.episode_rewards.extend(self.locals['rewards'].tolist())
        # print(self.locals['new_obs'])
        if isinstance(self.locals['new_obs'], dict): #to handle ordered and unordered dicts
            obs = [{k:v.tolist() for k,v in self.locals['new_obs'].items()},]
            # obs = json.dumps(self.locals['new_obs'])
        else:
            obs = self.locals['new_obs'].tolist()
        self.episode_observations.extend(obs)
        self.episode_actions.extend(self.locals['actions'].tolist())
        if np.sum(self.locals["dones"]).item() > 0:
          self._on_episode()

        return True

    def _on_rollout_end(self) -> None:
        self.rollout_count += 1
    
    def _on_training_end(self) -> None:
        self.training_count+=1