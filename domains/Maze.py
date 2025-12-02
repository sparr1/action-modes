import numpy as np
import random as rnd
import math

from modes.tasks import Task
from modes.classifier import SupportClassifier
from utils.utils import q_rotate

class Move(Task):
    def __init__(self,desired_velocity_minimum = -3.0,
                 desired_velocity_maximum = 3.0,
                 ctrl_cost = False,
                 contact_cost = False,
                 healthy_z_range = (0.3,2.5),
                 survival_bonus = 1.0,
                 include_xy = True,
                 modify_obs = True,
                 adaptive_margin = True,
                 adaptive_margin_minimum = 0.25,
                 categorical = False,
                 rotation_starting_weight=0.4,
                 margin = 2.0,
                 loss_at_margin = 1/3,
                 slope = 1.0,
                 direction = "X", 
                 metric = "L1"):
        super().__init__()
        self.desired_velocity_minimum = desired_velocity_minimum #TODO refactor to range so as to have consistent API
        self.desired_velocity_maximum = desired_velocity_maximum
        #If you would like to avoid randomly sampling, just set minimum = maximum
        self.task_info = None
        self.velocity_coords = None
        self.dir_coords = None
        #hypothesis: forwards is simply "x_velocity"
        if self.desired_velocity_maximum == math.inf:
            self.maximize = True
        else:
            self.maximize = False
        self.max_desired_speed = max(abs(self.desired_velocity_minimum), abs(self.desired_velocity_maximum))
        # self.min_norm_reward = 0.1
        self.adaptive_margin_minimum = adaptive_margin_minimum
        self.adaptive_margin_setting = adaptive_margin
        self.adaptive_slope_maximum = 1 / self.adaptive_margin_minimum
        self.adaptive_slope_setting = self.adaptive_margin_setting
        # self.adaptive_margin = None
        self.loss_at_margin = loss_at_margin
    
        # self.value_at_margin = 0.5
        if survival_bonus:
            self.survival_bonus = survival_bonus #float
        else:
            self.survival_bonus = 0.0

        self.ctrl_cost = ctrl_cost
        self.cnt_cost = contact_cost #not currently implemented
        self.categorical = categorical
        if healthy_z_range: #this should be a range
            self.min_z, self.max_z = healthy_z_range
        else:
            self.min_z = -math.inf
            self.max_z = math.inf
        self.include_xy = include_xy
        self.modify_obs = modify_obs
        directions = ["X", "Y", "Z", "XR", "YR", "ZR"]
        self.rotation_starting_weight = rotation_starting_weight
        self.starting_weights = np.concatenate((np.ones(shape=(3,)), np.full(shape=(3,), fill_value = rotation_starting_weight)))
        if direction == "F":
            self.relative = True
            self.relative_direction_start = None
        else:
            self.relative = False
        
        self.direction = np.zeros(6,dtype = float)
        if direction in directions:
            self.direction[directions.index(direction)] = 1.0
            self.starting_weights[directions.index(direction)] = 1.0
        elif direction == 'F':
            pass
        else:
            direction = self.direction #better be a (6,) array!

        print(self.direction)
        print(self.starting_weights)

        self.margin = margin
        self.slope = slope
        self.metric = metric
        self.reset() #to sample the desired velocity from between the minimum and maximum
        print(self.tuned_weights)
    
    #from the Gymnasium robotics AntEnv codebase
    
    def control_cost(self, action):
        if self.ctrl_cost:
            return .1*np.sqrt(np.sum(np.square(action))) #hardcoded a param at .25
        else: 
            return 0.0
        
    def contact_cost(self, contact_forces):
        if self.cnt_cost:
            return .05*np.sqrt(np.sum(np.square(contact_forces)))
        else:
            return 0.0
    
    def healthy(self, obs):
        z_coord = obs[2] if self.include_xy else obs[0]
        return 1.0 if (self.min_z <= z_coord <= self.max_z) else 0.0

    def set_velocity_coords(self, env):
        self.velocity_coords = env.get_velocity_coords()

    def set_relative_start(self, obs):
        self.relative_direction_start = obs[self.velocity_coords[0]:self.velocity_coords[1]]

    #checking state feature for velocity in the desired direction. if direction is None, we'll simply take the magnitude of the velocity vector in any direction.
    def get_reward(self, observation, last_action, contact_forces):
        # print(self.desired_velocity, type(self.desired_velocity))
        if type(observation)==dict:
            obs = observation["observation"]
        else:
            obs = observation #does this make sense? TODO check
        velocity_vec = obs[self.velocity_coords[0]:self.velocity_coords[1]]
        if self.relative:
            # print(self.relative_direction_start)
            q_rot = obs[self.dir_coords[0]:self.dir_coords[1]]
            # print(q_rot) #W is first
            if self.relative_direction_start is None:
                self.set_relative_start(obs)
            self.direction = q_rotate(self.relative_direction_start, q_rot)
            # print(self.direction)
            self.planar_direction = self.direction[:2]
            self.max_heading = np.argmax(np.abs(self.planar_direction))
            self.max_heading_sign = self.planar_direction[self.max_heading]/np.abs(self.planar_direction[self.max_heading])
            new_direction = np.zeros_like(self.direction)
            new_direction[self.max_heading] = self.max_heading_sign
            self.direction = np.concatenate((new_direction, [0.0, 0.0, 0.0]),axis=0)
            # self.direction = np.zeros_like(self.direction)[self.max_heading] = self.max_heading_sign
            # self.direction = np.concatenate((self.direction, [0.0, 0.0, 0.0]),axis = 0)
            # print(self.direction)

        # print(velocity_vec)
        achieved_velocity = np.dot(velocity_vec, self.direction)/np.dot(self.direction, self.direction) #project onto the desired direction 
        # print("achieved forwards velocity", achieved_velocity)
        # print("X,Y,Z velocity", velocity_vec[:3])
        # print("rotation quaternion", q_rot)
        deviation = np.sum(self.tuned_weights*((velocity_vec - self.direction*self.desired_velocity))**2)
        # dist_to_vec = np.sqrt(deviation) if deviation > 1 else deviation
        dist_to_vec = np.sqrt(deviation)
        # squared_margin = 
        if not self.maximize:
            if self.metric == "L2":
                if dist_to_vec < self.margin:
                    base_reward = 0.0
                else:
                    dist_to_ball = dist_to_vec - self.margin

                    base_reward =  -min(dist_to_ball*self.slope,self.survival_bonus) #0 at (or within) margin, less than 0 outside of it, never less than survival bonus
            elif self.metric == "L1":
                base_reward = -1*np.sum(np.abs(velocity_vec - self.direction*self.desired_velocity))
            elif self.metric == "huber":
                if dist_to_vec < self.margin:
                    base_reward = -self.loss_at_margin*(dist_to_vec/self.margin)**2 #normalized quadratic region
                else:
                    base_reward =  2*self.loss_at_margin*(dist_to_vec/self.margin) - self.loss_at_margin #normalized linear region
                    base_reward = -min(base_reward, self.survival_bonus)

        else:
            base_reward = self.achieved_velocity #for now, just linear in velocity.
        healthy = self.healthy(obs)
        healthy_bonus = healthy*self.survival_bonus
        ctrl_cost = self.control_cost(last_action)
        contact_cost = self.contact_cost(contact_forces)
        total_cost = ctrl_cost + contact_cost
        # min_norm = max(self.min_norm_reward, abs(self.desired_velocity))
        # min_norm = self.max_desired_speed
        velocity_bonus = base_reward + healthy_bonus + 0.001
        
        reward = velocity_bonus - total_cost  #bonus is simply zero if this is not desired
        #TODO ctrl cost, contact cost, and termination for healthy z range....
        reward_info = {'desired_velocity': self.desired_velocity, 'achieved_velocity': achieved_velocity, 'base': base_reward, 'healthy_bonus': healthy_bonus, 'control cost': -ctrl_cost, 'contact cost': -contact_cost}
        # print(reward_info)
        return reward, reward_info #TODO how to do average reward?
    
    # def distance_lo
    def get_termination(self, obs):
        return self.healthy(obs) == 0.0
    
    def get_goal(self):
        return self.desired_velocity
    
    def set_goal(self, new_goal):
        self.desired_velocity = new_goal
    
    
    def set_task_info(self, task_info):
        self.task_info = task_info
        self.velocity_coords = task_info["velocity_coords"]
        self.dir_coords = task_info["dir_coords"]

    def get_goal_length(self):
        return 1

    def reset(self, seed = 32):
        if self.desired_velocity_maximum == self.desired_velocity_minimum:
            self.desired_velocity = self.desired_velocity_minimum
        else:
            if self.categorical:
                self.desired_velocity = rnd.sample([self.desired_velocity_minimum, self.desired_velocity_maximum],k=1)[0]
            else:
                self.desired_velocity = rnd.uniform(self.desired_velocity_minimum, self.desired_velocity_maximum)
            # print(self.desired_velocity, type(self.desired_velocity))
        
        if self.adaptive_margin_setting:
            self.margin = max(self.adaptive_margin_minimum, (abs(self.desired_velocity) / 2))
        #    self.margin = abs(self.desired_velocity)
            self.slope = 1/self.margin
        #    self.tuning_margin_correction = min(self.margin/10, 0.5)
        #    self.tuning_margin_correction = 0.5
        #    self.tuned_weights = (1 - self.tuning_margin_correction)*self.direction*self.starting_weights + self.tuning_margin_correction*self.starting_weights
        #    self.squared_margin = self.margin**2
            self.tuned_weights = self.starting_weights
        #    print("margin", self.margin)
        #    print("slope", self.slope)
        #    print("tuning_margin_correction", self.tuning_margin_correction)
        #    print("starting weights", self.starting_weights)
        #    print("tuned weights", self.tuned_weights)


        # parent_return = super().reset()
        # print(parent_return)
        return super().reset()
    

class Change(Task):
    def __init__(self, desired_coord_minimum = 0.3,
                desired_coord_maximum = 2.5,
                ctrl_cost = False,
                contact_cost = False,
                healthy_z_range = (0.3,2.5),
                survival_bonus = 1.0,
                modify_obs = True,
                categorical = False,
                margin = 0.25,
                loss_at_margin = 1/3,
                slope = 1.0,
                target_coords = "X",
                metric = "L1"):
        super().__init__()
        self.include_xy = True
        self.desired_coord_minimum = desired_coord_minimum #TODO refactor to range so as to have consistent API
        self.desired_coord_maximum = desired_coord_maximum
        #If you would like to avoid randomly sampling, just set minimum = maximum
        self.task_info = None
        self.velocity_coords = None
        self.dir_coords = None

        # self.min_norm_reward = 0.1
        # self.adaptive_margin = None
        self.loss_at_margin = loss_at_margin
    
        # self.value_at_margin = 0.5
        if survival_bonus:
            self.survival_bonus = survival_bonus #float
        else:
            self.survival_bonus = 0.0

        self.ctrl_cost = ctrl_cost
        self.cnt_cost = contact_cost #not currently implemented
        self.categorical = categorical
        if healthy_z_range: #this should be a range
            self.min_z, self.max_z = healthy_z_range
        else:
            self.min_z = -math.inf
            self.max_z = math.inf

        self.modify_obs = modify_obs
        coords = ["X", "Y", "Z"]
        self.target_coord_inds = [coords.index(letter) for letter in target_coords]
        self.num_target_coords = len(self.target_coord_inds)

        self.margin = margin
        self.slope = slope
        self.metric = metric
        self.reset() #to sample the desired velocity from between the minimum and maximum

#from the Gymnasium robotics AntEnv codebase

    def control_cost(self, action):
        if self.ctrl_cost:
            return .1*np.sqrt(np.sum(np.square(action))) #hardcoded a param at .25
        else: 
            return 0.0
        
    def contact_cost(self, contact_forces):
        if self.cnt_cost:
            return .05*np.sqrt(np.sum(np.square(contact_forces)))
        else:
            return 0.0
    
    def healthy(self, obs):
        z_coord = obs[2] if self.include_xy else obs[0]
        return 1.0 if (self.min_z <= z_coord <= self.max_z) else 0.0

    def set_position_coords(self, env):
        self.position_coords = env.get_position_coords()

    #checking state feature for velocity in the desired direction. if direction is None, we'll simply take the magnitude of the velocity vector in any direction.
    def get_reward(self, observation, last_action, contact_forces):
        if type(observation)==dict:
            obs = observation["observation"]
        else:
            obs = observation #does this make sense? TODO check
        position_vec = obs[self.position_coords[0]:self.position_coords[1]]

        achieved_position = obs[self.position_coords[0]:self.position_coords[1]]
        target_position =  achieved_position.copy() #in theory, this could be set in a more advanced manner. 
        for i,t in enumerate(self.target_coord_inds):
            target_position[t] = self.desired_coords[i] 

        deviation = np.sum((position_vec - target_position)**2)
        # dist_to_vec = np.sqrt(deviation) if deviation > 1 else deviation
        dist_to_vec = np.sqrt(deviation)

        if self.metric == "L2":
            if dist_to_vec < self.margin:
                base_reward = 0.0
            else:
                dist_to_ball = dist_to_vec - self.margin

                base_reward =  -min(dist_to_ball*self.slope,self.survival_bonus) #0 at (or within) margin, less than 0 outside of it, never less than survival bonus
        elif self.metric == "L1":
            base_reward = -1*np.sum(np.abs(position_vec - target_position))
        elif self.metric == "huber":
            if dist_to_vec < self.margin:
                base_reward = -self.loss_at_margin*(dist_to_vec/self.margin)**2 #normalized quadratic region
            else:
                base_reward =  2*self.loss_at_margin*(dist_to_vec/self.margin) - self.loss_at_margin #normalized linear region
                base_reward = -min(base_reward, self.survival_bonus)

        healthy = self.healthy(obs)
        healthy_bonus = healthy*self.survival_bonus
        ctrl_cost = self.control_cost(last_action)
        contact_cost = self.contact_cost(contact_forces)
        total_cost = ctrl_cost + contact_cost
        # min_norm = max(self.min_norm_reward, abs(self.desired_velocity))
        # min_norm = self.max_desired_speed
        position_bonus = base_reward + healthy_bonus + 0.001
        
        reward = position_bonus - total_cost  #bonus is simply zero if this is not desired
        #TODO ctrl cost, contact cost, and termination for healthy z range....
        reward_info = {'desired_position': self.desired_coords, 'achieved_position': achieved_position, 'base': base_reward, 'healthy_bonus': healthy_bonus, 'control cost': -ctrl_cost, 'contact cost': -contact_cost}
        # print(reward_info)
        return reward, reward_info #TODO how to do average reward?
    
    # def distance_lo
    def get_termination(self, obs):
        return self.healthy(obs) == 0.0
    
    def get_goal(self):
        return self.desired_coords
    
    def set_goal(self, new_goal):
        self.desired_coords = new_goal
    
    
    def set_task_info(self, task_info):
        self.task_info = task_info
        self.position_coords = task_info["position_coords"]

    def get_goal_length(self):
        return 1

    def reset(self, seed = 32):
        if self.desired_coord_minimum == self.desired_coord_maximum:
            self.desired_coords = [self.desired_coord_minimum,]*self.num_target_coords
        else:
            if self.categorical:
                self.desired_coords = rnd.sample([self.desired_coord_minimum, self.desired_coord_maximum],k=self.num_target_coords)
            else:
                self.desired_coords = [rnd.uniform(self.desired_coord_minimum, self.desired_coord_maximum) for _ in range(self.num_target_coords)]
            # print(self.desired_coords)

        # parent_return = super().reset()
        # print(parent_return)
        return super().reset()
    
def check_range(value, range):
    return range[0] < value < range[1]

def get_speed(observation, vel_offset):
    return np.sqrt(np.sum(observation[13 + vel_offset:19+vel_offset]**2))

def check_z_range(observation, z_range, z_coord):
    return check_range(observation[z_coord], z_range)

def check_speed(observation, speed_range, vel_offset):
    return check_range(get_speed(observation, vel_offset), speed_range)
    
class MoveForwardSupportClassifier(SupportClassifier):
    def __init__(self, z_range_minimum = 0.3, z_range_maximum = 1.0, num_legs = 4, include_xy = False, include_target = False):
        self.z_range = [z_range_minimum, z_range_maximum]
        self.num_legs = num_legs
        self.include_xy = include_xy
        self.include_target = include_target

        self.pos_offset = 2 if self.include_xy else 0
        self.target_offset = 2 if self.include_target else 0
        self.z_coord = self.pos_offset+self.target_offset
    
        self.vel_offset = self.target_offset+self.pos_offset + (self.num_legs - 4)*2

    def rule(self, observation):
        return check_range(observation[self.pos_offset], self.z_range)

class MoveRotateSupportClassifier(SupportClassifier):
    def __init__(self, z_range_minimum = 0.3, z_range_maximum = 0.65, speed_minimum = -1.0, speed_maximum=2.0, num_legs = 4, include_xy = False, include_target = False):
        self.z_range = [z_range_minimum, z_range_maximum]
        self.speed_range = [speed_minimum, speed_maximum]
        self.num_legs = num_legs
        self.include_xy = include_xy
        self.include_target = include_target

        self.pos_offset = 2 if self.include_xy else 0
        self.target_offset = 2 if self.include_target else 0
        self.z_coord = self.pos_offset+self.target_offset
    
        self.vel_offset = self.target_offset+self.pos_offset + (self.num_legs - 4)*2

    def rule(self, observation):
        valid_z = check_z_range(observation, self.z_range, self.z_coord)
        valid_speed = check_speed(observation, self.speed_range, self.vel_offset)
        print("xyz",observation[-4:-2], observation[0])
        print("goal",observation[-2:])
        print("valid_z", valid_z)
        print("speed", get_speed(observation, self.vel_offset))
        print("valid_speed", valid_speed)
        print(len(observation))
        print(np.sum(observation))
        print(observation)
        return valid_z and valid_speed 

class ChangeZSupportClassifier(SupportClassifier):
    def __init__(self, speed_minimum = -1.0, speed_maximum = 1.0, num_legs = 4, include_xy = True):
        self.speed_range = [speed_minimum, speed_maximum]
        self.num_legs = num_legs
        self.include_xy = include_xy
        self.pos_offset = 2 if self.include_xy else 0
        self.vel_offset = self.pos_offset + (self.num_legs - 4)*2

    def rule(self, observation):
        return check_speed(observation, self.speed_range)
