import gymnasium as gym
import numpy as np
import random, math
# from modes.modes import ModalWrapper
# from utils import q_rotate
#
# Code heavily adapted to match the random resetting in Gymnasium-Robotics and Antv4 as closely as possible for transfer
# https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/maze/maze_v4.py
# https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant_v4.py

class AntPlane(gym.Wrapper):
    def __init__(self, env, random_resets = False, reset_map=[[1]*5]*5, map_scale = 4, position_noise_range = 0.25, num_legs = 4):
        super().__init__(env)
        self.base_env = self.env.unwrapped
        self.random_resets = random_resets #boolean for macro-level randomness on the map level
        self.set_reset_locations(reset_map)
        self._map_scaling = map_scale
        self._map_length = len(reset_map)
        self._map_width = len(reset_map[0]) #rectangular reset map assumption
        self.x_map_center = self._map_width / 2 * self._map_scaling
        self.y_map_center = self._map_length / 2 * self._map_scaling
        self.position_noise_range = position_noise_range
        self.base_env.reset_model = self.reset_model #monkey patch the reset function to include randomly initialized position
        self.include_xy = True
        self.num_legs = num_legs

    def reset_model(self):
        noise_low = -self.base_env._reset_noise_scale
        noise_high = self.base_env._reset_noise_scale

        qpos = self.base_env.init_qpos + self.base_env.np_random.uniform(
            low=noise_low, high=noise_high, size=self.base_env.model.nq
        )
        qvel = (
            self.base_env.init_qvel
            + self.base_env._reset_noise_scale * self.base_env.np_random.standard_normal(self.base_env.model.nv)
        )

        # print("old qpos", qpos, qpos.shape)
        # print("old qvel", qvel, qvel.shape)
        if self.random_resets:
            reset_rowcol = self.generate_reset_pos()
            # print("reset_rowcol", reset_rowcol)
            reset_coord = self.cell_rowcol_to_xy(reset_rowcol)
            # print("reset_coord", reset_coord)
            self.reset_pos = self.add_xy_position_noise(reset_coord)
            # print("self.reset_pos", self.reset_pos)
            re_init_pos = qpos.copy()
            re_init_pos[:2] = self.reset_pos
            # print("re_init_pos", re_init_pos)
            qpos = re_init_pos

        self.env.unwrapped.set_state(qpos, qvel)
        observation = self.env.unwrapped._get_obs()

        return observation

    
    def cell_rowcol_to_xy(self, rowcol_pos: tuple) -> np.ndarray:
        # print("x center", self.x_map_center)
        # print("y center", self.y_map_center)
        x = (float(rowcol_pos[1]) + 0.5) * self._map_scaling - self.x_map_center
        y = self.y_map_center - (float(rowcol_pos[0]) + 0.5) * self._map_scaling

        return np.array([x, y])
    
    def cell_xy_to_rowcol(self, xy_pos: np.ndarray) -> np.ndarray:
        i = (self.y_map_center - xy_pos[1].item()) // self._map_scaling
        j = (xy_pos[0].item() + self.x_map_center) // self._map_scaling
        return (i, j)

    def set_reset_locations(self, reset_map):
        self.reset_map = reset_map
        self.reset_locations = [(i,j) for i in range(len(self.reset_map)) for j in range(len(self.reset_map[i])) if self.reset_map[i][j]]

    def generate_reset_pos(self):
        return random.choice(self.reset_locations)

    def add_xy_position_noise(self, xy_pos: np.ndarray) -> np.ndarray:
            noise_x = (
                self.np_random.uniform(
                    low=-self.position_noise_range, high=self.position_noise_range
                )
                * self._map_scaling
            )
            noise_y = (
                self.np_random.uniform(
                    low=-self.position_noise_range, high=self.position_noise_range
                )
                * self._map_scaling
            )
            xy_pos[0] += noise_x
            xy_pos[1] += noise_y

            return xy_pos
    
    #TODO move relative velocity calculations here?
    def get_task_info(self, incl_xy = True):
        offset = 2 if incl_xy else 0
        return {"velocity_coords": (13+offset, 19+offset),
                "dir_coords": (1+offset, 5+offset)}

    #probably not going to need this! keeping it here for posterity
        
    # class AntPlaneModalWrapper(ModalWrapper):
    #     def __init__(self, env, walking_controller, rotating_controller):
    #         super().__init__(env, [walking_controller, rotating_controller])
    #         self.walking_controller = walking_controller
    #         self.rotating_controller = rotating_controller