import torch
import numpy as np
import json
from utils.core import initialize_alg
from utils.core import setup_wrapper
from domains.mpqdn_wrappers import *
from domains.mpqdn_ant_domain import *

class Controller():
    def __init__(self, config, domain):
        self.domain = domain
        with open(config) as c:
            self.controller_settings = json.load(c)
            print(self.controller_settings)
        self.alg_config = self.controller_settings["alg_config_path"]
        self.model_path = self.controller_settings["model_path"]
        self.name = self.controller_settings["name"]

    # def take_action(self, **kwargs):
    #     pass

class ModalController(Controller):
    def __init__(self, controller_config, domain, custom_action_space):
        super().__init__(controller_config, domain)
        self.orig_domain = domain
        self.task_wrapper_settings = self.controller_settings["env_wrapper"]
        self.observation_process_queue = []
        wrapped_env = domain
        print("pre-wrapper obs space", wrapped_env.observation_space)
        print("pre-wrapper act space", wrapped_env.action_space)
        wrapped_env = self.ModalWrapper(wrapped_env)
        self.observation_process_queue.append(wrapped_env.observation)

        print("mode-wrapped obs space", wrapped_env.observation_space)
        print("mode-wrapped act space", wrapped_env.action_space)
        wrapped_env = FlattenStateWrapper(wrapped_env)
        self.observation_process_queue.append(wrapped_env.observation)

        print("flat-state obs space", wrapped_env.observation_space)
        print("flat-state act space", wrapped_env.action_space)
        wrapped_env = self.load_task_wrapper(wrapped_env, self.task_wrapper_settings) #should set self.task_wrapped_env to be a wrapped version of orig_domain
        # self.observation_process_queue.append(wrapped_env.observation)
        print("task-wrapped obs space", wrapped_env.observation_space)
        print("task-wrapped act space", wrapped_env.action_space)
        self.domain = wrapped_env
        self.modal_action_space = custom_action_space
        #will want to make a new action space wrapper for this. 
        # self.task_wrapped_env.action_space = self.modal_action_space
        # self.modal_action_space = self.controller_settings["modal_action_space"]
        # self.base_action_space = self.controller_settings["base_action_space"]
        # self.base_observation_space = self.controller_settings["base_observation_space"]
        # self.domain.action_space = self.modal_action_space

        model, is_baseline, name = self.load_model(self.alg_config, self.domain, self.model_path, custom_action_space=self.modal_action_space)
        self.model = model
        self.alg_name = name
        self.is_baseline = is_baseline
        
    def load_model(self, alg_config, domain, model_path, custom_action_space):
            #will this need some kind of error handling? feels like brittle
        with open(alg_config) as c:
            alg_params = json.load(c)
            initial_model, alg_name, is_baseline = initialize_alg(alg_params["alg"], alg_params["alg_params"], domain, custom_action_space=custom_action_space)
        initial_model.load(model_path)

        return initial_model, alg_name, is_baseline

    def load_task_wrapper(self, env, task_wrapper_settings):
        wrapper_name = task_wrapper_settings['name']
        wrapper_params = task_wrapper_settings['wrapper_params']
        return setup_wrapper(env, wrapper_name, wrapper_params)
    
    def process_obs(self, observation, action_within_mode):
        obs = observation
        for obs_process in self.observation_process_queue:
            obs = obs_process(obs)

        new_obs = {'observation': obs.copy()}
        new_obs['desired_goal'] = np.array((action_within_mode,), dtype = np.float64) #manually apply the task_wrapper observation transformation
        return new_obs
    
    def predict(self, observation):
        return self.model.predict(observation)
    
    # discard the desired/achieved goals of the orchestrator, as it was not present during the training of the modal controllers.
    # This may be Gymnasium Robotics specific! Think about how to architect this for more generality later
    class ModalWrapper(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            #simply remove desired_goal
            self.observation_space = gym.spaces.Dict({'achieved_goal': env.observation_space['desired_goal'], 'observation':env.observation_space['observation']})
        def observation(self, obs):
            # old_obs = obs["observation"]
            # new_obs = obs['observation'].copy()
            # old_w = old_obs[1]
            # old_x = old_obs[2]
            # old_y = old_obs[3]
            # old_z = old_obs[4]
            # new_obs[1] = old_x
            # new_obs[2] = old_y
            # new_obs[3] = old_z
            # new_obs[4] = old_w
            return {'achieved_goal':obs['achieved_goal'], 'observation': obs['observation']}

class OrchestralController(Controller):
    def __init__(self, controller_config, domain, orchestral_action_space, sub_controllers = []):
        super().__init__(controller_config, domain)
        self.sub_controllers = sub_controllers
        #will this need to handle PAMDP wrapper stuff?
        self.orchestral_action_space = orchestral_action_space
        # self.base_action_space = self.controller_settings["base_action_space"]
        # self.base_observation_space = self.controller_settings["base_observation_space"]
        self.observation_process_queue = []
        # action_process_queue = []
        wrapped_env = domain
        wrapped_env = self.OrchestralWrapper(wrapped_env, orchestral_action_space)
        wrapped_env = FlattenStateWrapper(wrapped_env)
        self.observation_process_queue.append(wrapped_env.observation)
        print("mid-wrapper obs space", wrapped_env.observation_space)
        print("mid-wrapper act space", wrapped_env.action_space)
        # wrapped_env = ScaledStateWrapper(wrapped_env)
        # self.observation_process_queue.append(wrapped_env.observation)

        # print("post-scale wrapper obs space", wrapped_env.observation_space)
        # print("post-scale wrapper act space", wrapped_env.action_space)
        wrapped_env = TimestepWrapper(wrapped_env, 1) #TODO should come from config. Also, this is going to cause an issue!
        self.observation_process_queue.append(lambda x: (x,1)) #just hacking around the timestep wrapper for now
        # observation_process_queue.append(wrapped_env.observation)
        # print("time-step wrapper obs space", wrapped_env.observation_space)
        # print("post-step wrapper act space", wrapped_env.action_space)
        wrapped_env = AntFlattenedActionWrapper(wrapped_env) #TODO check if domain specific
        print("post-flattened action wrapper", wrapped_env.observation_space)
        print("post-flattened action wrapper", wrapped_env.action_space)
        wrapped_env = ScaledParameterisedActionWrapper(wrapped_env)
        print("post-scale action wrapper obs", wrapped_env.observation_space)
        print("post-scale action wrapper act", wrapped_env.action_space)
        self.domain = wrapped_env

        with open(controller_config) as c:
            controller_params = json.load(c)
            print(controller_params)
            self.model, self.name, _ = self.load_model(controller_params["alg_config_path"], self.domain, None)

    def load_model(self, alg_config, domain, model_path):
            #will this need some kind of error handling? feels like brittle
        with open(alg_config) as c:
            alg_params = json.load(c)
            initial_model, alg_name, is_baseline = initialize_alg(alg_params["alg"], alg_params["alg_params"], domain, custom_action_space=self.orchestral_action_space)
        if model_path:
            initial_model.load(model_path)

        return initial_model, alg_name, is_baseline
    
    def add_controllers(self, controllers):
        self.sub_controllers.extend(controllers)

    def process_obs(self, observation):
        obs = observation
        for obs_process in self.observation_process_queue:
            obs = obs_process(obs)
        return (obs[0].astype(np.float32), obs[1]) #should this be somewhere else?
    
    def process_act(self, action):
        pass

    def predict(self, observation):
        action, action_parameters, all_action_parameters = self.model.predict(observation[0])
        return (action, all_action_parameters), None
    
    class OrchestralWrapper(gym.ActionWrapper):
        def __init__(self, env, orchestral_action_space):
            super().__init__(env)
            self.action_space = orchestral_action_space
    


#what is left to do? connect everything into one pipeline and then debug
#I need to 
#3)debug any kinks!


#Hm, there seems to be a minor conceptual issue/architectural choice. Am I wrapping the environment with the new controllers?
#or, am I going to deliver an end-to-end version? I think my plan was to start with the wrapping, but it's just not as satisfying or convincing.

#nah, I think the wrapper idea is bad. I want to maintain clean separation between environment and algorithm.

#So the modal algorithm will work with the original base action space. 

#I will need to incorporate task wrapping into the use of the controllers, since this does modify the observation space.
