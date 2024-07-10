#the goal here is to wrap the baselines AND our own custom algorithms to a common interface. a little ambitious, maybe.
class Algorithm():
    def __init__(self, name, env, custom_params = None):
        self.name = name
        self.env = env
        self.custom_params = custom_params
        
    def learn(self):
        pass
    def predict(self):
        pass
    def vec_env(self):
        pass
