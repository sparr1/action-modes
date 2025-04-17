import numpy as np
import gymnasium as gym
import importlib, json, os
from stable_baselines3.common.callbacks import BaseCallback
from RL.alg import Algorithm, SimpleAlgorithm
from utils import setup_logs
#from stable_baselines3 import PPO, DQN, TD3, SAC, DDPG, A2C
module_name = "mpdqn" #for dynamic importing

#TODO: loading in code from the MPDQN codebase 

class RandomPAMDP(SimpleAlgorithm):
    def __init__(self, name, env, custom_params = None):
        super().__init__(name, env, custom_params)

    def predict(self, observation):
        # print(self.env.action_space)
        sample = self.env.action_space.sample()
        return (sample[0],list(sample[1:])), None #annoying, but I think this needed to be here
    
class PAMDP(Algorithm):
    def __init__(self, name, env, params = None):
        super().__init__(name, env, custom_params=params)
        self.model = self.get_PAMDP_model(self.name, self.env, self.custom_params)
        self.callback = None

    def learn(self, **kwargs):
        return self.model.learn(**kwargs)
    
    def predict(self, observation):
        return self.model.predict(observation)
    
    def save(self, path, name):
        self.model.save(os.path.join(path, name))

    def get_model(self):
        return self.model

    # def set_logger(self, logger):
    #     super().set_logger(logger)
    #     self.callback = TrajectoryLoggerCallback(self.alg_logger)
        
    def delete_model(self): 
        del self.model

    def load(self, path):#requires retrieving the correct baseline model before loading weights
        self.model = self.model.load(path)

#
# from https://github.com/cycraig/MP-DQN/
#


    def get_PAMDP_model(self, name, env, params = None):
        p = {}
        env_name = env.unwrapped.spec.id
        agent = None

        if 'Goal' in env_name:
            from gym_goal.envs.config import GOAL_WIDTH, PITCH_LENGTH, PITCH_WIDTH


            if params['scale_actions']:
                kickto_weights = np.array([[-0.375, 0.5, 0, 0.0625, 0],
                                        [0, 0, 0.8333333333333333333, 0, 0.111111111111111111111111]])
                shoot_goal_left_weights = np.array([0.857346647646219686, 0])
                shoot_goal_right_weights = np.array([-0.857346647646219686, 0])
            else:
                xfear = 50.0 / PITCH_LENGTH
                yfear = 50.0 / PITCH_WIDTH
                caution = 5.0 / PITCH_WIDTH
                kickto_weights = np.array([[2.5, 1, 0, xfear, 0], [0, 0, 1 - caution, 0, yfear]])
                shoot_goal_left_weights = np.array([GOAL_WIDTH / 2 - 1, 0])
                shoot_goal_right_weights = np.array([-GOAL_WIDTH / 2 + 1, 0])

            initial_weights = np.zeros((4, 17))
            initial_weights[0, [10, 11, 14, 15]] = kickto_weights[0, 1:]
            initial_weights[1, [10, 11, 14, 15]] = kickto_weights[1, 1:]
            initial_weights[2, 16] = shoot_goal_left_weights[1]
            initial_weights[3, 16] = shoot_goal_right_weights[1]

            initial_bias = np.zeros((4,))
            initial_bias[0] = kickto_weights[0, 0]
            initial_bias[1] = kickto_weights[1, 0]
            initial_bias[2] = shoot_goal_left_weights[0]
            initial_bias[3] = shoot_goal_right_weights[0]

            #if this becomes necessary, we can use this
            # partially_unwrapped_env = env #we will search for the relevant observation wrapper in the wrapper stack
            # while (not issubclass(partially_unwrapped_env, gym.ObservationWrapper)) or hasattr(partially_unwrapped_env, "scale_state"):
            #     partially_unwrapped_env = partially_unwrapped_env.env

            if not params['scale_actions']:
                # rescale initial action-parameters for a scaled state space
                for a in range(env.action_space.spaces[0].n):
                    mid = (env.observation_space.spaces[0].high + env.observation_space.spaces[0].low) / 2.
                    initial_bias[a] += np.sum(initial_weights[a] * mid)
                initial_weights[a] = initial_weights[a]*env.observation_space.spaces[0].high - initial_weights[a] * mid

            # env.seed(seed)
            # np.random.seed(seed)

            #END FIRST PART OF PDQN INITIALIZATION
            assert not (params['split'] and params['multipass'])
            name = "PDQNAgent"
            if params['split']:
                name = "SplitPDQNAgent"
            elif params['multipass']:
                name = "MultiPassPDQNAgent"
    
            try:
                module = importlib.import_module(module_name)
                agent_class = getattr(module,name) #grab the specific algorithm
            
                agent = agent_class(
                                observation_space=env.observation_space.spaces[0], action_space=env.action_space,
                                batch_size=params['batch_size'],
                                learning_rate_actor=params['learning_rate_actor'],  # 0.0001
                                learning_rate_actor_param=params['learning_rate_actor_param'],  # 0.001
                                epsilon_steps=params['epsilon_steps'],
                                epsilon_final=params['epsilon_final'],
                                gamma=params['gamma'],
                                clip_grad=params['clip_grad'],
                                indexed=params['indexed'],
                                average=params['average'],
                                random_weighted=params['random_weighted'],
                                tau_actor=params['tau_actor'],
                                weighted=params['weighted'],
                                tau_actor_param=params['tau_actor_param'],
                                initial_memory_threshold=params['initial_memory_threshold'],
                                use_ornstein_noise=params['use_ornstein_noise'],
                                replay_memory_size=params['replay_memory_size'],
                                inverting_gradients=params['inverting_gradients'],
                                actor_kwargs={'hidden_layers': params['layers'], 'output_layer_init_std': 1e-5, #dubious
                                                'action_input_layer': params['action_input_layer'],},
                                actor_param_kwargs={'hidden_layers': params['layers'], 'output_layer_init_std': 1e-5, #dubious
                                                    'squashing_function': False}, #dubious
                                zero_index_gradients=params['zero_index_gradients'],
                                seed=params['seed'])
            except (ModuleNotFoundError, AttributeError) as e:
                raise ValueError(f"Could not find model class '{name}' in module '{module_name}': {e}")
        elif 'Platform' in env_name: #NOW DO PLATFORM

            initial_params_ = [3., 10., 400.]
            unwrapped_env = env.unwrapped
            if params['scale_actions']:
                for a in range(unwrapped_env.action_space.spaces[0].n): 
                    initial_params_[a] = 2. * (initial_params_[a] - unwrapped_env.action_space.spaces[1].spaces[a].low) / (
                                unwrapped_env.action_space.spaces[1].spaces[a].high - unwrapped_env.action_space.spaces[1].spaces[a].low) - 1.

            print("obs space", env.observation_space)
            print("act space", env.action_space)

            assert not (params['split'] and params['multipass'])
            name = "PDQNAgent"
            if params['split']:
                name = "SplitPDQNAgent"
            elif params['multipass']:
                name = "MultiPassPDQNAgent"
    

            try:
                module = importlib.import_module(module_name)
                agent_class = getattr(module,name) #grab the specific algorithm
            
                agent = agent_class(
                                observation_space=env.observation_space.spaces[0], action_space=env.action_space,
                                batch_size=params['batch_size'],
                                learning_rate_actor=params['learning_rate_actor'],  # 0.0001
                                learning_rate_actor_param=params['learning_rate_actor_param'],  # 0.001
                                epsilon_steps=params['epsilon_steps'],
                                epsilon_final=params['epsilon_final'],
                                gamma=params['gamma'],
                                clip_grad=params['clip_grad'],
                                indexed=params['indexed'],
                                average=params['average'],
                                random_weighted=params['random_weighted'],
                                tau_actor=params['tau_actor'],
                                weighted=params['weighted'],
                                tau_actor_param=params['tau_actor_param'],
                                initial_memory_threshold=params['initial_memory_threshold'],
                                use_ornstein_noise=params['use_ornstein_noise'],
                                replay_memory_size=params['replay_memory_size'],
                                inverting_gradients=params['inverting_gradients'],
                                actor_kwargs={'hidden_layers': params['layers'], #'output_layer_init_std': 1e-5, #dubious
                                                'action_input_layer': params['action_input_layer'],},
                                actor_param_kwargs={'hidden_layers': params['layers'], 'output_layer_init_std': 1e-4, #dubious
                                                    'squashing_function': False}, #dubious
                                zero_index_gradients=params['zero_index_gradients'],
                                seed=params['seed'])
            except (ModuleNotFoundError, AttributeError) as e:
                raise ValueError(f"Could not find model class '{name}' in module '{module_name}': {e}")

            if params['initialise_params']:
                initial_weights = np.zeros((env.action_space.spaces[0].n, env.observation_space.spaces[0].shape[0]))
                initial_bias = np.zeros(env.action_space.spaces[0].n)
                for a in range(env.action_space.spaces[0].n):
                    initial_bias[a] = initial_params_[a]
                agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)
            print(agent)
        return agent
        #TODO NEEDS TESTING T.T