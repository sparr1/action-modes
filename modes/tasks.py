import gymnasium as gym
import numpy as np
import math
import traceback, sys
from utils.utils import listify
#A task in this case is a specification which includes a reward function. 
#the policy derived from optimizing this reward function will become a modal controller once paired with a support classifier.

class Task():
    def __init__(self):
        pass
    def get_reward(self, observation)  -> float:
        pass
    def get_termination(self, observation) -> bool:
        pass
    def reset(self, seed = 32) -> None:
        pass
    def get_goal(self) -> float: #in theory, this should be greater than 1-d
        pass
    def set_goal(self) -> float:
        pass

#TODO: this should be refactored to be an _average reward_ task. To handle this correctly, I need to do some reading...
class EternalTask(Task):
    def __init__(self): 
        super().__init__()
    
    def get_termination(self, observation):
        return False
    
#this takes an object of type Task and wraps an env with the conditions from the task.
class Subtask(gym.Wrapper):
    def __init__(self, env, task, task_info = {}):
        super().__init__(env)
        self.last_action = None
        self.reward_info = None
        self._task = task # if None, just use the reward which comes from the environment
        if task_info:
            task.set_task_info(task_info)
        else:
            try:
                # task.set_task_info(env.get_task_info())
                task.set_task_info(env.get_wrapper_attr('get_task_info')())
            except Exception as e:
                print("Exception Message:")
                print(e)
                print()
                print("\nStack Trace:")
                traceback.print_exc(file=sys.stdout)
                print("Have you implemented get_task_info for the domain?")

        goal_length = self._task.get_goal_length()
        
        spaces = {'desired_goal': gym.spaces.Box(-math.inf, math.inf, (goal_length,), np.float64),
                  'observation': self.observation_space}
        #self.observation_space.spaces['desired_goal'] = gym.spaces.Box(-math.inf, math.inf, (goal_length,), np.float64)
        self.observation_space = gym.spaces.Dict(spaces)

    def reset(self, seed = 32): #TODO actually implement seeding properly
        self._task.reset(seed=seed) #also reset the task. this will resample a new subgoal.
        old_return = super().reset(seed=seed)
        return (self.observation(old_return[0]),old_return[1])
    
    def observation(self, obs):
        new_obs = {'observation': obs.copy()}

        new_obs['desired_goal'] = np.array(listify(self._task.get_goal()))
        # print(new_obs)
        return new_obs

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.last_action = action
        try:
            self.contact_forces = self.env.unwrapped.contact_forces
        except:
            self.contact_forces = None
        #self.contact_forces = self.env.unwrapped.ant_env.contact_forces
        # desired_goal = observation['desired_goal']
        new_observation = self.observation(observation)
        new_reward = self.reward(observation)
        new_termination = self.termination(observation)
        # info.update({"reward_info":self.reward_info, "old_reward": reward, "old_termination": terminated, "old_goal":desired_goal})
        info.update({"reward_info":self.reward_info, "old_reward": reward, "old_termination": terminated})

        return new_observation, new_reward, new_termination, truncated, info

    def reward(self, observation):
        reward, self.reward_info = self._task.get_reward(observation, self.last_action, self.contact_forces)
        return reward
    
    def termination(self, observation):
        return self._task.get_termination(observation)
    
    def set_goal(self, new_goal):
        self._task.set_goal(new_goal)
