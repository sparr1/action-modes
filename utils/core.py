from RL.baselines import Baseline
from modes.tasks import Subtask
from domains.AntPlane import AntPlane
import importlib, json

SUPPORTED_WRAPPERS = ("Subtask", "AntPlane", "ScaledStateWrapper", "PlatformFlattenedActionWrapper", "ScaledParameterisedActionWrapper")
SUPPORTED_LOG_SETTINGS = ("overwrite", "warn", "timestamp", "overwrite-safe")
SUPPORTED_LOG_TYPES = ("detailed", "summary")

def initialize_alg(alg_string, alg_params, domain, custom_action_space = None):
    baseline = False
    if '/' in alg_string:
        file_name, alg_name = "".join(alg_string.split('/')[:-1]), alg_string.split('/')[-1]
        print(file_name)
        print(alg_name)
        if "baselines" in file_name:
            baseline = True
            try:
                model = Baseline(alg_name, domain, alg_params)
            except Exception as e:
                print(e)
                return #if we cannot run this baseline, we just try another
        elif file_name == "PAMDP":
            try: 
                module = importlib.import_module("RL.PAMDP")
                alg_class = getattr(module, alg_name)
                model = alg_class(alg_name,domain, alg_params, custom_action_space = custom_action_space)
            except Exception as e:
                print(e)
                return #if we cannot run this baseline, we just try another.
        else:
            try:
                module = importlib.import_module("RL."+file_name.replace('/','.')) #last ditch, just try to load it!
                alg_class = getattr(module, alg_name)
                model = alg_class(alg_name,domain, alg_params)
            except Exception as e:
                print(e)
                return #if we cannot run this baseline, we just try another.
    else:
        try:
            module = importlib.import_module("RL.alg")
            alg_class = getattr(module, alg_string)
            model = alg_class(alg_string, domain, alg_params)
            alg_name = alg_string
        except Exception as e:
            print(e)
            return #if we cannot run this baseline, we just try another.
    return model, baseline, alg_name

#TODO: currently does nothing. either add functionality or delete
def handle_settings(path): #processes the settings.json file
    with open(path) as f:
        contents = json.load(f)
    print(contents) 

def setup_wrapper(domain, wrapper_name, wrapper_params):
    if wrapper_name == 'Subtask':      
        try:
            print(wrapper_params["task"])
            module_name,task_name = wrapper_params["task"].split(':') 
            # print(module)
            module = importlib.import_module(module_name)
            task_class = getattr(module,task_name) #grab the specific task
            p = wrapper_params["task_params"]
            task = task_class(**p)  
            domain = Subtask(domain, task) #replace the reward function and termination conditions based on task, then return the new wrapped domain.
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Could not find model class '{task_name}' in module '{module_name}': {e}")
    elif wrapper_name == 'AntPlane':
        domain = AntPlane(domain, **wrapper_params)
    else:
        print("setting up default wrapper ", wrapper_name, "with params", wrapper_params)
        module_name,raw_wrapper_name = wrapper_name.split(':') #this is likely to error out
        module = importlib.import_module(module_name)
        wrapper_class = getattr(module, raw_wrapper_name)
        domain = wrapper_class(domain, **wrapper_params)
        print("wrapping appears to have been successful.")

    return domain
