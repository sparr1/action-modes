import gymnasium as gym
import importlib
#from stable_baselines3 import PPO, DQN, TD3, SAC, DDPG, A2C
module_name = "stable_baselines3" #for dynamic importing

def get_baseline_model(name, env, params = None):
    p = {}

    model = None

    #needed to handle gymnasium robotics observed_goal structure properly
    if "policy" not in params:
        if type(env.observation_space) == gym.spaces.dict.Dict:
            p["policy"] = "MultiInputPolicy"
        else:
            p["policy"] = "MlpPolicy" #just a guess.
            print("Trying MlpPolicy as none given.")

    p["env"] = env

    if params:
        p.update(params)

    if "verbose" not in p.keys():
        p["verbose"] = 1
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module,name) #grab the specific algorithm
    
        model = model_class(**p) #make sure the json elements actually correspond to parameters of the relevant stable_baseline alg!
        
        return model
        
    except (ModuleNotFoundError, AttributeError) as e:
        raise ValueError(f"Could not find model class '{name}' in module '{module_name}': {e}")

#TODO handle baseline params