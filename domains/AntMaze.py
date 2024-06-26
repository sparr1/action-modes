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
    def __init__(self, desired_velocity = None, direction = None):
        super().__init__()
        self.direction = direction #TODO: figure out which direction is forwards :)
        #hypothesis: take in a quaternion in the format (w,x,y,z) and convert to yaw angle?
        #hypothesis: the answer is simply "x_velocity". Courtesy of Rafa
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
        velocity = np.array((obs[13], obs[14], obs[15])) #only really going to work for AntMaze for now
        if not self.direction:
            achieved_velocity = velocity
        else:
            pass #TODO: project velocity vector onto self.direction and set achieved velocity to that
        achieved_velocity = np.sqrt(np.dot(achieved_velocity,achieved_velocity)) #for now, we take the magnitude
        reward = -(np.sum(np.abs(achieved_velocity - self.desired_velocity))) #do we want a margin?

        return reward

#Rotation subtask. specify an angular velocity (negative or positive scalar) and we try to match it.
class Rotate(EternalTask):
    def __init__(self, desired_velocity = None):
        super().__init__()
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
        achieved_angular_velocity = obs[18] #only really going to work for AntMaze for now
        reward = -np.abs(achieved_angular_velocity- self.desired_velocity) #do we want a margin?
        return reward
    