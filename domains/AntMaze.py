import numpy as np
import random as rnd
import math

from domains.tasks import EternalTask


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
class Move(EternalTask):
    def __init__(self,desired_velocity_minimum = -1.0,
                 desired_velocity_maximum = 1.0,
                 ctrl_cost = True,
                 contact_cost = True,
                 healthy_z_range = (0.3,1.0),
                 survival_bonus = 1.0,
                 modify_obs = True,
                 direction = "X", 
                 metric = "L2"):
        super().__init__()
        self.desired_velocity_minimum = desired_velocity_minimum #TODO refactor to range so as to have consistent API
        self.desired_velocity_maximum = desired_velocity_maximum
        #hypothesis: forwards is simply "x_velocity". Courtesy of Rafa
        if self.desired_velocity_maximum == math.inf: #not supporting this simultaneously with range for now
            self.maximize = True
        else:
            self.maximize = False

        if survival_bonus:
            self.survival_bonus = survival_bonus #float
        else:
            self.survival_bonus = 0.0

        self.ctrl_cost = ctrl_cost #just boolean flags
        self.cnt_cost = contact_cost #not currently implemented

        if healthy_z_range: #this should be a range
            self.min_z, self.max_z = healthy_z_range
        else:
            self.min_z = -math.inf
            self.max_z = math.inf

        self.modify_obs = modify_obs #boolean

        if direction == "X":
            self.direction = np.array((1,0,0), dtype = float)
        elif direction == "Y":
            self.direction = np.array((0,1,0), dtype = float)
        elif direction == "Z":
            self.direction = np.array((0,0,1), dtype = float)
        else:
            direction = self.direction #better be a (3,) array!

        self.metric = metric
        self.reset() #to sample the desired velocity from between the minimum and maximum
    
    #from the Gymnasium robotics AntEnv codebase
    
    def control_cost(self, action):
        if self.ctrl_cost:
            return .25*np.sum(np.square(action)) #hardcoded a param at .25
        else: 
            return 0.0
        
    def contact_cost(self, contact_forces):
        if self.cnt_cost:
            return 5e-4 * np.sum(np.square(contact_forces))
        else:
            return 0.0
    
    def healthy(self, obs):
        return 1.0 if (self.min_z <= obs[0] <= self.max_z) else 0.0

    #checking state feature for velocity in the desired direction. if direction is None, we'll simply take the magnitude of the velocity vector in any direction.
    def get_reward(self, observation, last_action, contact_forces):
        if type(observation)==dict:
            obs = observation["observation"]
        else:
            obs = observation #does this make sense? TODO check
        velocity = np.array((obs[13], obs[14], obs[15]), dtype = float) 
        achieved_velocity = np.dot(velocity, self.direction) #for now, this just selects one of the three axes. 
        if not self.maximize:
            discrepancy = np.abs(self.desired_velocity - achieved_velocity)
            if self.metric == "L2":
                base_reward = -0.5*discrepancy**2 #do we want a margin? TODO add margin. pretty convinced we do, because the velocities for a walking ant have a wide variation.
            elif self.metric == "L1":
                base_reward = -1*discrepancy
        else:
            base_reward = self.achieved_velocity #for now, just linear in velocity.
        healthy = self.healthy(obs)
        unhealthy_cost = (1.0 - healthy)*self.survival_bonus
        ctrl_cost = self.control_cost(last_action)
        contact_cost = self.contact_cost(contact_forces)
        total_cost = unhealthy_cost + ctrl_cost + contact_cost
        reward = base_reward - total_cost  #bonus is simply zero if this is not desired
        #TODO ctrl cost, contact cost, and termination for healthy z range....
        reward_info = {'base': base_reward, 'unhealthy cost': -unhealthy_cost, 'control cost': -ctrl_cost, 'contact cost': -contact_cost}
        # print(reward_info)
        return reward, reward_info #TODO how to do average reward?
    
    def get_goal(self):
        return self.desired_velocity
    
    def get_goal_length(self):
        return 1

    def reset(self, seed = 32):
        if self.desired_velocity_maximum == self.desired_velocity_minimum:
            self.desired_velocity = self.desired_velocity_minimum
        else:
            self.desired_velocity = rnd.uniform(self.desired_velocity_minimum, self.desired_velocity_maximum)
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