#
# from https://github.com/cycraig/MP-DQN/
#


import numpy as np
import gymnasium as gym

class AntFlattenedActionWrapper(gym.ActionWrapper):
    """
    Changes the format of the parameterised action space to conform to that of Goal-v0 and Platform-v0
    """
    def __init__(self, env):
        super(AntFlattenedActionWrapper, self).__init__(env)
        # print("PLATFORM FLATTENED ACTION WRAPPER")
        old_as = env.action_space
        num_actions = old_as.spaces[0].n
        # print("old action space", old_as)
        # print(num_actions)
        self.action_space = gym.spaces.Tuple((
            old_as.spaces[0],  # actions
            *(gym.spaces.Box(old_as.spaces[1].spaces[i].low, old_as.spaces[1].spaces[i].high, dtype=np.float32)
              for i in range(0, num_actions))
        ))
        # print("new action space", self.action_space)

    def action(self, action):
        return action
