import numpy as np
import os
import shutil
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import xml.etree.ElementTree as ET


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

def set_num_legs(xml_template, xml_file, num_legs):
        # Reset XML file from template
        shutil.copy2(xml_template, xml_file)
        
        # Parse the XML
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Update number of legs in custom data
        for numeric in root.findall(".//custom/numeric"):
            if numeric.get("name") == "num_legs":
                numeric.set("data", str(num_legs))
        
        # Get the torso body element
        torso = root.find(".//body[@name='torso']")
        
        # Remove all existing leg elements (1-4) to replace them with evenly distributed ones
        for i in range(1, 5):
            for leg in torso.findall(f".//body[@name='leg_{i}']"):
                torso.remove(leg)
        
        # Remove existing actuators related to legs
        actuators = root.find("actuator")
        for i in range(1, 5):
            for motor in actuators.findall(f".//motor[@joint='hip_{i}']"):
                actuators.remove(motor)
            for motor in actuators.findall(f".//motor[@joint='ankle_{i}']"):
                actuators.remove(motor)
        
        # Add all legs with even distribution around the torso
        for i in range(1, num_legs + 1):
            # Calculate angle for even distribution (in radians)
            angle = 2 * np.pi * (i - 1) / num_legs
            
            # Get the normalized direction vector
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            # First segment length - match original ant's sqrt(0.2^2 + 0.2^2) = 0.2*sqrt(2)
            # In original ant, first segment goes 0.2 units in x and 0.2 units in y
            first_segment_length = 0.2 * np.sqrt(2)
            x = first_segment_length * dx
            y = first_segment_length * dy
            
            # Determine joint axis and ranges based on quadrant (following original design)
            # Original leg configuration:
            # Leg 1 (top-right): ankle axis (-1,1,0), range (30,70)
            # Leg 2 (top-left): ankle axis (1,1,0), range (-70,-30)
            # Leg 3 (bottom-left): ankle axis (-1,1,0), range (-70,-30)
            # Leg 4 (bottom-right): ankle axis (1,1,0), range (30,70)
            
            # Determine which quadrant this leg is in
            if x >= 0 and y >= 0:  # First quadrant (top-right)
                ankle_axis = "-1 1 0"
                ankle_range_min = 30
                ankle_range_max = 70
            elif x < 0 and y >= 0:  # Second quadrant (top-left)
                ankle_axis = "1 1 0"
                ankle_range_min = -70
                ankle_range_max = -30
            elif x < 0 and y < 0:  # Third quadrant (bottom-left)
                ankle_axis = "-1 1 0"
                ankle_range_min = -70
                ankle_range_max = -30
            else:  # Fourth quadrant (bottom-right)
                ankle_axis = "1 1 0"
                ankle_range_min = 30
                ankle_range_max = 70
            
            # Second segment (between joints) - same length as first segment
            mid_x = x
            mid_y = y
            
            # Third segment (ankle) - match original ant's sqrt(0.4^2 + 0.4^2) = 0.4*sqrt(2)
            # In original ant, third segment goes 0.4 units in x and 0.4 units in y
            ankle_segment_length = 0.4 * np.sqrt(2)
            ankle_x = ankle_segment_length * dx
            ankle_y = ankle_segment_length * dy
            
            # Create the leg XML (following original structure)
            leg_xml = f"""
            <body name="leg_{i}" pos="0 0 0">
                <geom fromto="0.0 0.0 0.0 {x:.4f} {y:.4f} 0.0" name="aux_{i}_geom" size="0.08" type="capsule"/>
                <body name="aux_{i}" pos="{x:.4f} {y:.4f} 0">
                    <joint axis="0 0 1" name="hip_{i}" pos="0.0 0.0 0.0" range="-30 30" type="hinge"/>
                    <geom fromto="0.0 0.0 0.0 {mid_x:.4f} {mid_y:.4f} 0.0" name="leg_{i}_geom" size="0.08" type="capsule"/>
                    <body pos="{mid_x:.4f} {mid_y:.4f} 0">
                        <joint axis="{ankle_axis}" name="ankle_{i}" pos="0.0 0.0 0.0" range="{ankle_range_min} {ankle_range_max}" type="hinge"/>
                        <geom fromto="0.0 0.0 0.0 {ankle_x:.4f} {ankle_y:.4f} 0.0" name="ankle_{i}_geom" size="0.08" type="capsule"/>
                    </body>
                </body>
            </body>
            """
            
            leg = ET.fromstring(leg_xml)
            torso.append(leg)
            
            # Add actuators for the leg
            hip_motor = ET.fromstring(f'<motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_{i}" gear="150"/>')
            ankle_motor = ET.fromstring(f'<motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_{i}" gear="150"/>')
            actuators.append(hip_motor)
            actuators.append(ankle_motor)
        
        # Save the modified XML
        tree.write(xml_file)

class AntVariableLegsEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        xml_file="ant_variable_legs.xml",
        ctrl_cost_weight=0.5,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.3,2.5),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        num_legs=4,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            ctrl_cost_weight,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            num_legs,
            **kwargs
        )

        self.num_legs = num_legs
        
        # Adjust control cost weight based on number of legs
        # For the original ant with 4 legs, keep the same control cost weight
        # For more legs, scale it down to balance the increased action space
        if num_legs > 4:
            self._ctrl_cost_weight = ctrl_cost_weight * 4 / num_legs
        else:
            self._ctrl_cost_weight = ctrl_cost_weight
            
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        
        # Store paths for XML files
        # self.xml_template = os.path.join(os.path.dirname(__file__), "assets", "ant_variable_legs_template.xml")
        self.xml_file = os.path.join(os.path.dirname(__file__), "assets", xml_file)
        
        # Create template if it doesn't exist
        # if not os.path.exists(self.xml_template):
        #     shutil.copy2(self.xml_file, self.xml_template)
        
        # Set up the environment with the specified number of legs
        # self._set_num_legs(num_legs)
        
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self._get_obs_dim(),), dtype=np.float64
        )
        
        MujocoEnv.__init__(
            self,
            self.xml_file,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs
        )

    def _get_obs_dim(self):
        # 27 base dimensions + additional dimensions for each leg beyond 4
        if self._exclude_current_positions_from_observation:
            return 27 + (self.num_legs - 4) * 4
        else:
            return 29 + (self.num_legs - 4) * 4


    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def contact_cost(self, contact_forces):
        contact_forces = np.clip(contact_forces, self._contact_force_range[0], self._contact_force_range[1])
        contact_cost = self._contact_cost_weight * np.sum(np.square(contact_forces))
        return contact_cost

    @property
    def healthy_reward(self):
        return float(self.is_healthy or self._terminate_when_unhealthy) * self._healthy_reward

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        costs = ctrl_cost = self.control_cost(action)

        terminated = self.terminated
        observation = self._get_obs()
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        reward = rewards - costs

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity))

        return observations

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value) 