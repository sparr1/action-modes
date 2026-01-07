import torch
import numpy as np
import json
from utils.core import initialize_alg
from utils.core import setup_wrapper
from domains.mpqdn_wrappers import *
from domains.mpqdn_ant_domain import *
from modes.classifier import UniversalSupport
import importlib

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
        if "support_classifier_path" in self.controller_settings:
            self.support_func = self.load_classifier(self.controller_settings["support_classifier_path"])()
        else:
            self.support_func = UniversalSupport()
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

    def load_classifier(self, classifier_path):
        module_name, classifier_name = classifier_path.split(':')
        try:
            module = importlib.import_module(module_name)
            classifier_func = getattr(module, classifier_name)
        except:
            pass
        return classifier_func


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
    #TODO This may be Gymnasium Robotics specific!
    class ModalWrapper(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            #simply remove desired_goal
            self.observation_space = gym.spaces.Dict({'achieved_goal': env.observation_space['desired_goal'], 'observation':env.observation_space['observation']})
        def observation(self, obs):
            return {'achieved_goal':obs['achieved_goal'], 'observation': obs['observation']}

class OrchestralController(Controller):
    def __init__(self, controller_config, domain, orchestral_action_space):
        super().__init__(controller_config, domain)
        self.sub_controllers = []
        self.supports = []
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
        #cannot use ScaledStateWrapper for domains with unbounded state
        # wrapped_env = ScaledStateWrapper(wrapped_env)
        # self.observation_process_queue.append(wrapped_env.observation)

        # print("post-scale wrapper obs space", wrapped_env.observation_space)
        # print("post-scale wrapper act space", wrapped_env.action_space)
        wrapped_env = TimestepWrapper(wrapped_env, 1)
        self.observation_process_queue.append(lambda x: (x,1)) #just hacking around the timestep wrapper for now
        # observation_process_queue.append(wrapped_env.observation)
        # print("time-step wrapper obs space", wrapped_env.observation_space)
        # print("post-step wrapper act space", wrapped_env.action_space)
        wrapped_env = FlattenedActionWrapper(wrapped_env)
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
        self.supports.extend([c.support_func for c in controllers])

    def process_obs(self, observation):
        obs = observation
        for obs_process in self.observation_process_queue:
            obs = obs_process(obs)
        return (obs[0].astype(np.float32), obs[1]) #should this be somewhere else?
    
    #TODO is this necessary?
    # def process_act(self, action):
    #     pass

    def predict(self, observation):
        mask = np.array([sup(observation[0]) for sup in self.supports],dtype = int)

        action, action_parameters, all_action_parameters = self.model.predict(observation[0], mask=mask)
        return (action, all_action_parameters), None
    
    class OrchestralWrapper(gym.ActionWrapper):
        def __init__(self, env, orchestral_action_space):
            super().__init__(env)
            self.action_space = orchestral_action_space
