import gymnasium as gym
import numpy as np
import math
#A task in this case can essentially be thought of as a reward function

class Task():
    def __init__(self):
        pass
    def get_reward(self, observation)  -> float:
        pass
    def get_termination(self, observation) -> bool:
        pass
    def reset(self, seed = 32) -> None:
        pass
    def get_goal(self) -> float:
        pass

#TODO: this should be refactored to be an _average reward_ task. To handle this correctly, I need to do some reading...
class EternalTask(Task):
    def __init__(self): 
        super().__init__()
    
    def get_termination(self, observation):
        return False
    
#this takes an object of type Task and wraps an env with the conditions from the task.
class Subtask(gym.Wrapper):
    def __init__(self, env, task):
        super().__init__(env)
        self.last_action = None
        self.reward_info = None
        self._task = task # if None, just use the reward which comes from the environment
        goal_length = self._task.get_goal_length()
        self.observation_space.spaces['desired_goal'] = gym.spaces.Box(-math.inf, math.inf, (goal_length,), np.float64)

    def reset(self, seed = 32): #TODO actually implement seeding properly
        self._task.reset() #also reset the task. this will resample a new subgoal.
        old_return = super().reset()
        return (self.observation(old_return[0]),old_return[1])
    
    def observation(self, obs):
        new_obs = obs.copy()
        new_obs['desired_goal'] = np.array([self._task.get_goal()])
        # print(new_obs)
        return new_obs



    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.last_action = action
        self.contact_forces = self.env.unwrapped.ant_env.contact_forces
        desired_goal = observation['desired_goal']
        new_observation = self.observation(observation)
        new_reward = self.reward(observation)
        new_termination = self.termination(observation)
        info.update({"reward_info":self.reward_info, "old_reward": reward, "old_termination": terminated, "old_goal":desired_goal})
        
        return new_observation, new_reward, new_termination, truncated, info

    def reward(self, observation):
        reward, self.reward_info = self._task.get_reward(observation, self.last_action, self.contact_forces)
        return reward
    
    def termination(self, observation):
        return self._task.get_termination(observation)
    

#we will take an initial task with an action space of Box(-1, 1, (8,), float32), 
#combine it with a few (initially just 2) different lower dimensional controllers action space of Box(-1, 1, (1,), float32),
#and then the new action space will be Dict(Discrete(2), Box(-1, 1, (1,), float32), Box(-1, 1, (1,), float32)), where the discrete action just switches which lower dimensional action space we use.
#In general, we will have M possible modes, which we will winnow into k supported modes for any given state, then project the latent "within-mode" actions to the original high-dimensional state space.

#for now, we will forget about the initial winnowing, and just assume the support of each mode covers the state space. 
class ModalTask(gym.Wrapper):
    def __init__(self, env, mode_controllers):
        #each mode_controller is going to turn a latent "action-within-mode" z into a base action in env. 
        self.num_modes = len(mode_controllers)
        #need to set up a discrete action space corresponding to the controllers
        self.base_action_space = self.action_space
        self.controllers = mode_controllers
        self.action_space = gym.spaces.Discrete(self.num_new_actions)
    def step(self, action):
        print(action)

#I believe I need to create some kind of "AbstractTask", which takes a collection of lower level controllers, and wraps the original environment, using the controllers as actions