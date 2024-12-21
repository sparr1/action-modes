#We need to implement DQN over the discrete parameter selections, and do PPO/TD3/SAC over the continuous parameters. 
#hypothesis: we can use stable baselines for the DQN and continuous action RL algorithms.
#I don't currently know how we're going to stitch these together. A little bit worried that it's going to cause a lot of trouble.
#hypothesis: this won't be bad at all! we just need to make sure the model.predict()s are paired up somehow. 
#hypothesis: we will need to fork stable baselines and do it on that side
#I need to handle both the nested algorithm side, and the nested task side. Seeing as I haven't done the latter yet...
from RL.baselines import Baseline
from RL.alg import Algorithm

class QPAMDP(Algorithm):

    #takes in an algorithm which is charged with producing a policy over modes, and an algorithm which is charged with producing a policy within modes
    #custom params should be a dictionary with keys the name, and values any parameters set for the sub-algorithms.
    def __init__(self, name, env, custom_params):
        super().__init__(name, env, custom_params=custom_params)
        self.alg_over_modes_name, self.alg_within_modes_name = custom_params.keys() #should be only two keys 
        self.alg_over_modes_params, self.alg_within_modes_params = custom_params.values()
        self.alg_over_modes = Baseline(self.alg_over_modes_name, env, params = self.alg_over_modes_params)
        self.alg_within_modes = Baseline(self.alg_within_modes_name, env, params = self.alg_within_modes_params)

    def learn(self):
        pass