#
# from https://github.com/cycraig/MP-DQN/
#


import numpy as np
import gymnasium as gym
import gymnasium_platform

from .mpqdn_wrappers import _parameter_action_index, _validate_parameter_values

class PlatformFlattenedActionWrapper(gym.ActionWrapper):
    """
    Changes the format of the parameterised action space to conform to that of Goal-v0 and Platform-v0
    """
    def __init__(self, env):
        super(PlatformFlattenedActionWrapper, self).__init__(env)
        # print("PLATFORM FLATTENED ACTION WRAPPER")
        old_as = env.action_space
        if (
            not isinstance(old_as, gym.spaces.Tuple)
            or len(old_as.spaces) != 2
            or not isinstance(old_as.spaces[0], gym.spaces.Discrete)
            or not isinstance(old_as.spaces[1], gym.spaces.Tuple)
        ):
            raise TypeError(
                "PlatformFlattenedActionWrapper requires "
                "Tuple(Discrete, Tuple(Box, ...))."
            )
        num_actions = old_as.spaces[0].n
        self.num_actions = num_actions
        self.parameter_spaces = tuple(old_as.spaces[1].spaces)
        if len(self.parameter_spaces) != num_actions or not all(
            isinstance(space, gym.spaces.Box) for space in self.parameter_spaces
        ):
            raise ValueError(
                "PlatformFlattenedActionWrapper requires one Box parameter "
                "space per action."
            )
        # print("old action space", old_as)
        # print(num_actions)
        self.action_space = gym.spaces.Tuple((
            old_as.spaces[0],  # actions
            *(gym.spaces.Box(old_as.spaces[1].spaces[i].low, old_as.spaces[1].spaces[i].high, dtype=np.float32)
              for i in range(0, num_actions))
        ))
        # print("new action space", self.action_space)

    def action(self, action):
        if not isinstance(action, (tuple, list)) or len(action) != self.num_actions + 1:
            raise ValueError(
                "Flattened parameterized action must contain an id followed by "
                f"{self.num_actions} parameter arrays."
            )
        action_id = _parameter_action_index(action[0], self.num_actions)
        parameters = _validate_parameter_values(
            action[1:], self.parameter_spaces, scaled=False
        )
        return action_id, tuple(parameters)
