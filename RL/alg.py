from decimal import Decimal, InvalidOperation
from numbers import Integral

from utils.utils import setup_logs
import numpy as np


def validate_timestep_budget(value, *, name="total_timesteps"):
    """Return an exact non-negative integer step budget.

    JSON scientific notation such as ``4.5e6`` remains valid, while booleans,
    fractions, infinities, and otherwise non-numeric values fail before a
    learner can silently round or run an unintended number of steps.
    """

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    if isinstance(value, Integral):
        if value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
        return int(value)
    try:
        # Decimal(str(...)) preserves exact integral strings and large integer
        # values instead of rounding them through binary64 first.
        numeric = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if (
        not numeric.is_finite()
        or numeric < 0
        or numeric != numeric.to_integral_value()
    ):
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(numeric)


#the goal here is to wrap the baselines AND our own custom algorithms to a common interface. a little ambitious, maybe.
class Algorithm():
    def __init__(self, name, env, custom_params = None):
        self.name = name
        self.env = env
        self.custom_params = custom_params
        self.alg_logger = None

    def get_env(self):
        return self.env
    
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
        total_timesteps = validate_timestep_budget(total_timesteps)
        t = 0
        while t < total_timesteps:
            terminated = False
            truncated = False
            observation, info = self.env.reset()
            data = {}
            while t < total_timesteps and not (terminated or truncated):
                action, _ = self.predict(observation)
                observation, reward, terminated, truncated, info = self.env.step(action)
                if self.alg_logger:
                    data = setup_logs(reward, observation, action, [terminated, truncated], [info,])
                    # print(data)
                    self.alg_logger.on_step(data)
                t+=1
        return self
    
    def predict(self,observation):
        pass

    def set_checkpointing(self, save_freq, save_path, name_prefix):
        pass

class Random(SimpleAlgorithm):
    def __init__(self, name, env, custom_params = None):
        super().__init__(name, env, custom_params)

    def predict(self, observation):
        # print(self.env.action_space)
        sample = self.env.action_space.sample()
        # print(type(sample))
        # print(sample)
        return sample, None
    
#just return all zeros for the action... 
class Stationary(SimpleAlgorithm):
    def __init__(self, name, env, custom_params = None):
        super().__init__(name, env, custom_params)

    def predict(self, observation):
        return np.zeros(self.env.action_space.shape), None
