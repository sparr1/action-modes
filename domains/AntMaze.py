import numpy as np
import random as rnd
import math

from domains.tasks import EternalTask, Task
from utils import q_rotate

#the solution to each of these "Eternal" tasks is simply the projection function which takes a real-valued variable (desired_velocity, usually) to a high dimensional policy 
#whose aim is to maintain some "fitness" indefinitely. you could think concretely of this fitness as a linear or nonlinear function of state. High fitness is finding a cycle which 
#has a high average reward as integrated over that cycle. 

#re-implementing move task

#for now, we will not specify direction, just to get the infrastructure for custom tasks finished first
#desired speed should be either a single value in m/s, or a range to sample from.
#hypothesis: we don't care about tolerance right now (check with group on thursday)
#hypothesis: coordinate system is local
#TODO check hypotheses 
#If you would like to avoid randomly sampling, just set minimum = maximum
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

        #hypothesis: forwards is simply "x_velocity". Courtesy of Rafa
        if self.desired_velocity_maximum == math.inf: #not supporting this simultaneously with range for now
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

        self.ctrl_cost = ctrl_cost #just boolean flags
        self.cnt_cost = contact_cost #not currently implemented
        self.categorical = categorical
        if healthy_z_range: #this should be a range
            self.min_z, self.max_z = healthy_z_range
        else:
            self.min_z = -math.inf
            self.max_z = math.inf
        self.include_xy = include_xy #boolean
        self.modify_obs = modify_obs #boolean
        directions = ["X", "Y", "Z", "XR", "YR", "ZR"]
        self.rotation_starting_weight = rotation_starting_weight
        starting_rotation_weights = np.array([self.rotation_starting_weight]*3)
        self.starting_weights = np.concatenate(np.ones(shape=(3,)), np.full(starting_rotation_weights, shape=(3,)))
        if direction == "F":
            self.relative = True
            self.relative_direction_start = np.array([1,0,0]) #X,Y,Z, forwards starts at X.
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


    #checking state feature for velocity in the desired direction. if direction is None, we'll simply take the magnitude of the velocity vector in any direction.
    def get_reward(self, observation, last_action, contact_forces):
        # print(self.desired_velocity, type(self.desired_velocity))
        if type(observation)==dict:
            obs = observation["observation"]
        else:
            obs = observation #does this make sense? TODO check
        offset = 0 if not self.include_xy else 2
        velocity_vec = obs[13+offset:19+offset]
        if self.relative:
            # print(self.relative_direction_start)
            q_rot = obs[1+offset:5+offset]
            # print(q_rot) #W is first
            self.direction = q_rotate(self.relative_direction_start, q_rot)
            # print(self.direction)
            self.direction = np.concatenate((self.direction, [0.0, 0.0, 0.0]),axis = 0)
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
        

#Rotation subtask. specify an angular velocity (negative or positive scalar) and we try to match it.
class Rotate(EternalTask):
    def __init__(self, desired_velocity_minimum = -1.0, desired_velocity_maximum = 1.0, direction = "Z", metric = "L2"):
        super().__init__()
        self.desired_velocity_minimum = desired_velocity_minimum
        self.desired_velocity_maximum = desired_velocity_maximum
        if direction == "X":
            self.direction = np.array((1,0,0), dtype = float)
        elif direction == "Y":
            self.direction = np.array((0,1,0), dtype = float)
        elif direction == "Z":
            self.direction = np.array((0,0,1), dtype = float)
        else:
            direction = self.direction #better be a (3,) array!

        self.metric = metric

    def get_reward(self, state):
        obs = state["observation"]
        angular_velocity = np.array((obs[16], obs[17], obs[18]),dtype = float) #only really going to work for AntMaze for now
        achieved_angular_velocity = np.dot(angular_velocity, self.direction) 
        discrepancy = np.abs(self.desired_velocity - achieved_angular_velocity)
        if self.metric == "L2":
            reward = -0.5*discrepancy**2
        elif self.metric == "L1":
            reward = -discrepancy
            
        return reward
    
    def reset(self, seed = 32):
        if self.desired_velocity_maximum == self.desired_velocity_minimum:
            self.desired_velocity = self.desired_velocity_minimum
        else:
            self.desired_velocity = rnd.uniform(self.desired_velocity_minimum, self.desired_velocity_maximum)

        return super().reset()