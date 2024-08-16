import numpy as np
import random as rnd

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
    def __init__(self,desired_velocity_minimum = -1.0, desired_velocity_maximum = 1.0, direction = "X", metric = "L2"):
        super().__init__()
        self.desired_velocity_minimum = desired_velocity_minimum
        self.desired_velocity_maximum = desired_velocity_maximum
        #hypothesis: forwards is simply "x_velocity". Courtesy of Rafa
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

    #checking state feature for velocity in the desired direction. if direction is None, we'll simply take the magnitude of the velocity vector in any direction.
    def get_reward(self, state):
        obs = state["observation"]
        velocity = np.array((obs[13], obs[14], obs[15]), dtype = float) 
        achieved_velocity = np.dot(velocity, self.direction) #for now, this just selects one of the three axes. 

        discrepancy = np.abs(self.desired_velocity - achieved_velocity)
        if self.metric == "L2":
            reward = -0.5*discrepancy**2 #do we want a margin?
        elif self.metric == "L1":
            reward = -1*discrepancy

        return reward #TODO this needs to be in line with average reward RL theory.
    
    def get_goal(self):
        return self.desired_velocity

    def reset(self, seed = 32):
        if self.desired_velocity_maximum == self.desired_velocity_minimum:
            self.desired_velocity = self.desired_velocity_minimum
        else:
            self.desired_velocity = rnd.uniform(self.desired_velocity_minimum, self.desired_velocity_maximum)
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