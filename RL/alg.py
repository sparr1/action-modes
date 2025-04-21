from utils import setup_logs
import numpy as np
#the goal here is to wrap the baselines AND our own custom algorithms to a common interface. a little ambitious, maybe.
class Algorithm():
    def __init__(self, name, env, custom_params = None):
        self.name = name
        self.env = env
        self.custom_params = custom_params
        self.alg_logger = None

    def learn(self, **kwargs):
        pass
    def predict(self, observation):
        pass

    def set_logger(self, logger):
        self.alg_logger = logger

    def save(self, path, name):
        pass
    def load(self, path):
        pass
    # def vec_env(self):
    #     pass

#just take actions randomly...

class SimpleAlgorithm(Algorithm):
    def __init__(self, name, env, custom_params = None):
        super().__init__(name, env, custom_params)

    def learn(self, total_timesteps=0): #simple algorithms simply don't learn anything
        t = 0
        while t < total_timesteps:
            terminated = False
            truncated = False
            observation, info = self.env.reset()
            data = {}
            while not (terminated or truncated):
                action, _ = self.predict(observation)
                observation, reward, terminated, truncated, info = self.env.step(action)
                # data = setup_logs(reward, observation, action, [terminated, truncated], [info,])
                # print(data)
                if self.alg_logger:
                    self.alg_logger.on_step(data)
                t+=1
    
    def predict(self,observation):
        pass

class Random(SimpleAlgorithm):
    def __init__(self, name, env, custom_params = None):
        super().__init__(name, env, custom_params)

    def predict(self, observation):
        return self.env.action_space.sample(), None

    
    
#just return all zeros for the action... 
class Stationary(SimpleAlgorithm):
    def __init__(self, name, env, custom_params = None):
        super().__init__(name, env, custom_params)

    def predict(self, observation):
        return np.zeros(self.env.action_space.shape), None

