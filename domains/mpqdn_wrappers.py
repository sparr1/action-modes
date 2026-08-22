#
# from https://github.com/cycraig/MP-DQN/
#


from numbers import Integral

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Tuple, Box


def _validate_scaling_box(space, *, name):
    if not isinstance(space, Box):
        raise TypeError(f"{name} must be a Box space.")
    if not np.isfinite(space.low).all() or not np.isfinite(space.high).all():
        raise ValueError(f"{name} must have finite bounds for linear scaling.")
    if not np.all(space.high > space.low):
        raise ValueError(
            f"{name} must have strictly increasing bounds for linear scaling."
        )


def _parameter_action_index(value, num_actions):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("Parameterized action id must be an integer.")
    index = int(value)
    if not 0 <= index < num_actions:
        raise ValueError(
            f"Parameterized action id {index} is outside [0, {num_actions})."
        )
    return index


def _validate_parameter_values(values, spaces, *, scaled):
    if len(values) != len(spaces):
        raise ValueError(
            f"Expected {len(spaces)} parameter arrays, received {len(values)}."
        )
    normalized = []
    for index, (value, space) in enumerate(zip(values, spaces)):
        array = np.asarray(value)
        if array.shape != space.shape:
            raise ValueError(
                f"Action parameter {index} must have shape {space.shape}, "
                f"received {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError(f"Action parameter {index} must be finite.")
        if scaled and (np.any(array < -1.0) or np.any(array > 1.0)):
            raise ValueError(
                f"Scaled action parameter {index} must be within [-1, 1]."
            )
        normalized.append(array)
    return normalized


class OrchestralWrapper(gym.ActionWrapper):
    def __init__(self, env, orchestral_action_space):
        super().__init__(env)
        self.action_space = orchestral_action_space

class ModalWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        #simply remove desired_goal
        self.observation_space = gym.spaces.Dict({'achieved_goal': env.observation_space['achieved_goal'], 'observation':env.observation_space['observation']})
    def observation(self, obs):
        return {'achieved_goal':obs['achieved_goal'], 'observation': obs['observation']}

class FlattenedActionWrapper(gym.ActionWrapper):
    """
    Changes the format of the parameterised action space to conform to that of Goal-v0 and Platform-v0
    """
    def __init__(self, env):
        super(FlattenedActionWrapper, self).__init__(env)
        # print("FLATTENED ACTION WRAPPER")
        old_as = env.action_space
        if (
            not isinstance(old_as, Tuple)
            or len(old_as.spaces) != 2
            or not isinstance(old_as.spaces[0], gym.spaces.Discrete)
            or not isinstance(old_as.spaces[1], Tuple)
        ):
            raise TypeError(
                "FlattenedActionWrapper requires Tuple(Discrete, Tuple(Box, ...))."
            )
        num_actions = old_as.spaces[0].n
        parameter_spaces = tuple(old_as.spaces[1].spaces)
        if len(parameter_spaces) != num_actions or not all(
            isinstance(space, Box) for space in parameter_spaces
        ):
            raise ValueError(
                "FlattenedActionWrapper requires one Box parameter space per action."
            )
        self.num_actions = num_actions
        self.parameter_spaces = parameter_spaces
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


class FlattenStateWrapper(gym.ObservationWrapper):
    """
    Flattens the observation space to a Box
    """

    def __init__(self, env):
        super(FlattenStateWrapper, self).__init__(env)
        obs = env.observation_space
        if isinstance(obs, gym.spaces.Box):
            self._observation_kind = "box"
            self._observation_keys = None
            component_spaces = (obs,)
        elif isinstance(obs, Tuple):
            self._observation_kind = "tuple"
            self._observation_keys = None
            component_spaces = tuple(obs.spaces)
        elif isinstance(obs, gym.spaces.Dict):
            self._observation_kind = "dict"
            self._observation_keys = tuple(obs.spaces)
            component_spaces = tuple(obs.spaces.values())
        else:
            raise TypeError(
                "FlattenStateWrapper requires a Box, Tuple of Box spaces, "
                "or Dict of Box spaces."
            )

        if not component_spaces or not all(
            isinstance(space, gym.spaces.Box) for space in component_spaces
        ):
            raise TypeError(
                "FlattenStateWrapper compound observations must contain only "
                "Box spaces."
            )
        dtype = np.result_type(*(space.dtype for space in component_spaces))
        concatenated_low = np.concatenate(
            [np.asarray(space.low).reshape(-1) for space in component_spaces]
        ).astype(dtype, copy=False)
        concatenated_high = np.concatenate(
            [np.asarray(space.high).reshape(-1) for space in component_spaces]
        ).astype(dtype, copy=False)
        self.observation_space = gym.spaces.Box(
            low=concatenated_low,
            high=concatenated_high,
            dtype=dtype,
        )

    def observation(self, obs):
        if self._observation_kind == "dict":
            values = [obs[key] for key in self._observation_keys]
        elif self._observation_kind == "tuple":
            values = list(obs)
        else:
            values = [obs]
        return np.ascontiguousarray(
            np.concatenate([np.asarray(value).reshape(-1) for value in values]),
            dtype=self.observation_space.dtype,
        )


class ScaledStateWrapper(gym.ObservationWrapper):
    """
    Scales the observation space to [-1,1]
    """

    def __init__(self, env):
        super(ScaledStateWrapper, self).__init__(env)
        obs = env.observation_space
        self.compound = False
        self.low = None
        self.high = None
        # print(type(obs))
        # print(obs)
        if isinstance(obs, gym.spaces.Box):
            self.low = env.observation_space.low
            self.high = env.observation_space.high
            _validate_scaling_box(obs, name="ScaledStateWrapper observation space")
            self.observation_space = gym.spaces.Box(low=-np.ones(self.low.shape), high=np.ones(self.high.shape),
                                                    dtype=np.float32)
        elif isinstance(obs, Tuple):
            if len(obs.spaces) != 2 or not isinstance(
                obs.spaces[0], Box
            ) or not isinstance(
                obs.spaces[1], gym.spaces.Discrete
            ):
                raise TypeError(
                    "ScaledStateWrapper Tuple observations must be "
                    "Tuple(Box, Discrete)."
                )
            self.low = obs.spaces[0].low
            self.high = obs.spaces[0].high
            _validate_scaling_box(
                obs.spaces[0], name="ScaledStateWrapper observation space"
            )
            self.observation_space = Tuple(
                (gym.spaces.Box(low=-np.ones(self.low.shape), high=np.ones(self.high.shape),
                                dtype=np.float32),
                 obs.spaces[1]))
            self.compound = True
        else:
            raise TypeError(
                "ScaledStateWrapper requires Box or Tuple(Box, Discrete) "
                f"observations, received {type(obs).__name__}."
            )

    def scale_state(self, state):
        state = 2. * (np.asarray(state) - self.low) / (self.high - self.low) - 1.
        return state.astype(np.float32, copy=False)

    def _unscale_state(self, scaled_state):
        scaled_state = np.asarray(scaled_state)
        state = (self.high - self.low) * (scaled_state + 1.) / 2. + self.low
        return state

    def observation(self, obs):
        # print("ScaledStateWrapper")

        if self.compound:
            state, steps = obs
            ret = (self.scale_state(state), steps)
        else:
            ret = self.scale_state(obs)
        return ret


class TimestepWrapper(gym.Wrapper):
    """
    Adds a timestep return to an environment for compatibility reasons.
    """
    def __init__(self, env, max_steps):
        super(TimestepWrapper, self).__init__(env)
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, Integral)
            or max_steps <= 0
        ):
            raise ValueError("TimestepWrapper max_steps must be a positive integer.")
        self.max_steps = int(max_steps)
        # The compatibility value is one elapsed step, so Discrete(max_steps)
        # was off by one when max_steps=1 and did not contain emitted values.
        self.observation_space = gym.spaces.Tuple(
            (env.observation_space, gym.spaces.Discrete(self.max_steps + 1))
        )

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        return (state, 1), info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        obs = (state, 1)
        return obs, reward, terminated, truncated, info


class ScaledParameterisedActionWrapper(gym.ActionWrapper):
    """
    Changes the scale of the continuous action parameters to [-1,1].
    Parameter space must be flattened!

    Tuple((
        Discrete(n),
        Box(c_1),
        Box(c_2),
        ...
        Box(c_n)
        )
    """

    def __init__(self, env):
        super(ScaledParameterisedActionWrapper, self).__init__(env)
        self.old_as = env.action_space
        if (
            not isinstance(self.old_as, Tuple)
            or not self.old_as.spaces
            or not isinstance(self.old_as.spaces[0], gym.spaces.Discrete)
        ):
            raise TypeError(
                "ScaledParameterisedActionWrapper requires "
                "Tuple(Discrete, Box, ...)."
            )
        self.num_actions = self.old_as.spaces[0].n
        if len(self.old_as.spaces) != self.num_actions + 1 or not all(
            isinstance(space, Box) for space in self.old_as.spaces[1:]
        ):
            raise ValueError(
                "ScaledParameterisedActionWrapper requires one Box parameter "
                "space per action."
            )
        self.parameter_spaces = tuple(self.old_as.spaces[1:])
        for index, space in enumerate(self.parameter_spaces):
            _validate_scaling_box(space, name=f"action parameter space {index}")
        self.high = [self.old_as.spaces[i].high for i in range(1, self.num_actions + 1)]
        self.low = [self.old_as.spaces[i].low for i in range(1, self.num_actions + 1)]
        self.range = [self.old_as.spaces[i].high - self.old_as.spaces[i].low for i in range(1, self.num_actions + 1)]
        new_params = [  # parameters
            Box(-np.ones(self.old_as.spaces[i].low.shape), np.ones(self.old_as.spaces[i].high.shape), dtype=np.float32)
            for i in range(1, self.num_actions + 1)
        ]
        self.action_space = Tuple((
            self.old_as.spaces[0],  # actions
            *new_params,
        ))

    def action(self, action):
        """
        Rescale from [-1,1] to original action-parameter range.

        :param action:
        :return:
        """
        # print('before copy', action)
        if not isinstance(action, (tuple, list)):
            raise ValueError("Scaled parameterized action must be a tuple or list.")
        p = _parameter_action_index(action[0], self.num_actions)
        # Historical PAMDP.pad_action emits the nested representation even
        # though this wrapper advertises the flattened action space. Accept it
        # at this boundary and normalize to the flattened representation that
        # the inner FlattenedActionWrapper consumes.
        if (
            len(action) == 2
            and isinstance(action[1], (tuple, list))
        ):
            parameter_values = action[1]
        else:
            parameter_values = action[1:]
        parameters = _validate_parameter_values(
            parameter_values, self.parameter_spaces, scaled=True
        )
        # print(p)
        # print('type action', type(action))
        # print('action', action)
        # print('range', self.range)
        # print('low', self.low)
        # v = np.squeeze(action[1][p])
        #I HAVE NO IDEA WHY THIS IS BREAKING. some kind of numpy change?
        parameters[p] = (
            self.range[p] * (parameters[p] + 1) / 2. + self.low[p]
        ).astype(self.parameter_spaces[p].dtype, copy=False)
        return (p, *parameters)


class QPAMDPScaledParameterisedActionWrapper(gym.ActionWrapper):
    """
    Changes the scale of the continuous action parameters to [-1,1].
    Parameter space not flattened in this case

    Tuple((
        Discrete(n),
        Tuple((
            Box(c_1),
            Box(c_2),
            ...
            Box(c_n)
            ))
        )
    """

    def __init__(self, env):
        super(QPAMDPScaledParameterisedActionWrapper, self).__init__(env)
        self.old_as = env.action_space
        if (
            not isinstance(self.old_as, Tuple)
            or len(self.old_as.spaces) != 2
            or not isinstance(self.old_as.spaces[0], gym.spaces.Discrete)
            or not isinstance(self.old_as.spaces[1], Tuple)
        ):
            raise TypeError(
                "QPAMDPScaledParameterisedActionWrapper requires "
                "Tuple(Discrete, Tuple(Box, ...))."
            )
        self.num_actions = self.old_as.spaces[0].n
        self.parameter_spaces = tuple(self.old_as.spaces[1].spaces)
        if len(self.parameter_spaces) != self.num_actions or not all(
            isinstance(space, Box) for space in self.parameter_spaces
        ):
            raise ValueError(
                "QPAMDPScaledParameterisedActionWrapper requires one Box "
                "parameter space per action."
            )
        for index, space in enumerate(self.parameter_spaces):
            _validate_scaling_box(space, name=f"action parameter space {index}")
        self.high = [self.old_as.spaces[1][i].high for i in range(self.num_actions)]
        self.low = [self.old_as.spaces[1][i].low for i in range(self.num_actions)]
        self.range = [self.old_as.spaces[1][i].high - self.old_as.spaces[1][i].low for i in range(self.num_actions)]
        new_params = [  # parameters
            gym.spaces.Box(-np.ones(self.old_as.spaces[1][i].low.shape), np.ones(self.old_as.spaces[1][i].high.shape),
                           dtype=np.float32)
            for i in range(self.num_actions)
        ]
        self.action_space = gym.spaces.Tuple((
            self.old_as.spaces[0],  # actions
            gym.spaces.Tuple(tuple(new_params)),
        ))

    def action(self, action):
        """
        Rescale from [-1,1] to original action-parameter range.

        :param action:
        :return:
        """
        if (
            not isinstance(action, (tuple, list))
            or len(action) != 2
            or not isinstance(action[1], (tuple, list))
        ):
            raise ValueError(
                "Nested scaled action must be (action_id, parameter_tuple)."
            )
        p = _parameter_action_index(action[0], self.num_actions)
        parameters = _validate_parameter_values(
            action[1], self.parameter_spaces, scaled=True
        )
        parameters[p] = (
            self.range[p] * (parameters[p] + 1) / 2. + self.low[p]
        ).astype(self.parameter_spaces[p].dtype, copy=False)
        return p, tuple(parameters)
