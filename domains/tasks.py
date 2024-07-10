import gymnasium as gym
#A task in this case can essentially be thought of as a reward function

class Task():
    def __init__(self):
        pass
    def get_reward(self, observation)  -> float:
        pass
    def get_termination(self, observation) -> bool:
        pass

class EternalTask(Task):
    def __init__(self):
        super().__init__()
    
    def get_termination(self, observation):
        return False
    
#this takes an object of type Task and wraps an env with the conditions from the task.
class Subtask(gym.Wrapper):
    def __init__(self, env, task):
        super().__init__(env)
        self._task = task # if None, just use the reward which comes from the environment
    
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info.update({"old_reward": reward, "old_termination": terminated})
        return observation, self.reward(observation), False, truncated, info

    def reward(self, observation):
        return self._task.get_reward(observation)
    
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