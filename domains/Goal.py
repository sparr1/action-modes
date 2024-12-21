import gymnasium as gym
import numpy as np
import random, math

class Goal(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)