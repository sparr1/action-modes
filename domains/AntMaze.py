import numpy as np
import random as rnd

from domains.tasks import EternalTask

#re-implementing move task

#for now, we will not specify direction, just to get the infrastructure for custom tasks finished first
#desired speed should be either a single value in m/s, or a range to sample from.
#hypothesis: we don't care about tolerance right now (check with group on thursday)
#hypothesis: coordinate system is local
#TODO check hypotheses 
class Move(EternalTask):
    def __init__(self, desired_velocity = None, direction = "X", metric = "L2"):
        super().__init__()
        # self.direction = direction #TODO: figure out which direction is forwards :)
        #hypothesis: take in a quaternion in the format (w,x,y,z) and convert to yaw angle?
        #hypothesis: the answer is simply "x_velocity". Courtesy of Rafa
        if direction == "X":
            self.direction = np.array((1,0,0), dtype = float)
        elif direction == "Y":
            self.direction = np.array((0,1,0), dtype = float)
        elif direction == "Z":
            self.direction = np.array((0,0,1), dtype = float)
        else:
            direction = self.direction #better be a (3,) array!

        self.metric = metric

        if desired_velocity and type(desired_velocity) == list and len(desired_velocity) == 2:
            self.desired_velocity = rnd.uniform(self.desired_velocity[0], self.desired_velocity[1])
        elif  type(desired_velocity) == list and len(desired_velocity) > 2:
            raise Exception("need at most two values to specify a range for velocity goals")
        elif desired_velocity:
            self.desired_velocity = desired_velocity
        else: 
            self.desired_velocity = 1.0 #default to 1 m/s just to see what happens

    #checking state feature for velocity in the desired direction. if direction is None, we'll simply take the magnitude of the velocity vector in any direction.
    def get_reward(self, state):
        obs = state["observation"]
        velocity = np.array((obs[13], obs[14], obs[15]),dtype = float) 
        achieved_velocity = np.dot(velocity, self.direction) #for now, this just selects one of the three axes. 

        discrepancy = np.abs(self.desired_velocity - achieved_velocity)
        if self.metric == "L2":
            reward = -0.5*discrepancy**2 #do we want a margin?
        elif self.metric == "L1":
            reward = -1*discrepancy

        return reward

#Rotation subtask. specify an angular velocity (negative or positive scalar) and we try to match it.
class Rotate(EternalTask):
    def __init__(self, desired_velocity = None, direction = "Z", metric = "L2"):
        super().__init__()

        if direction == "X":
            self.direction = np.array((1,0,0), dtype = float)
        elif direction == "Y":
            self.direction = np.array((0,1,0), dtype = float)
        elif direction == "Z":
            self.direction = np.array((0,0,1), dtype = float)
        else:
            direction = self.direction #better be a (3,) array!

        self.metric = metric
        if desired_velocity and type(desired_velocity) == list and len(desired_velocity) == 2:
            self.desired_velocity = rnd.uniform(self.desired_velocity[0], self.desired_velocity[1])
        elif  type(desired_velocity) == list and len(desired_velocity) > 2:
            raise Exception("need at most two values to specify a range for angular velocity goals")
        elif desired_velocity:
            self.desired_velocity = desired_velocity
        else: 
            self.desired_velocity = 1.0 #default to 1 rad/s just to see what happens

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
    