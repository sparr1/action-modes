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