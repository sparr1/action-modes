import random, time, argparse, json, os, shutil, math, importlib

import numpy as np
import seaborn as sns
import pandas as pd

import gymnasium as gym
# from RL.alg import *
from RL.baselines import Baseline, TrajectoryLoggerCallback
from log import TrainingLogger
from domains.tasks import Subtask
from domains.AntPlane import AntPlane
from utils import *
def main():
    parser = argparse.ArgumentParser(description="action mode learning experiments")
    parser.add_argument('-r', '--run', help='config file for a run', required=True)
    config = vars(parser.parse_args())['run']

    with open(config) as f:
        experiment_params = json.load(f)
        experiment_name = config.split('/')[-1][:-5] #cut out everything before the / and the extension .json
    # print(experiment_params)

    #TODO: make a default configuration so I can cut all this code!
    if "logs" in experiment_params:
        log_setting = experiment_params["logs"]
    else:
        log_setting = "warn" #just manually enforcing an annoying default because I love my users and I don't want them deleting their logs or causing clutter by default :)

    if "save_trials" in experiment_params:
        save_trials_setting = experiment_params["save_trials"]
    else:
        save_trials_setting = "first"
           
    #there are three save settings: save_num---which denotes the number of model saves during a trial,
    #save_strat---which denotes the behavior within a trial as to which saves (out of the total save_num) are kept around: do we keep none, all, last, or best?
    #and save_trials---which denotes the saving behavior across trials: none, first, all, or best?
    #TODO: implement save_num and save_strat. for now we will just save after the end of the trial, and worry about save_trial behavior.

    num_algs = len(experiment_params["configs"])
    runtime_params = [dict() for _ in range(num_algs)]
    print("Experiment testing")
    for i, alg_config in enumerate(experiment_params["configs"]):
        print("---------")
        print(alg_config)
        runtime_params[i]["name"] = alg_config
        print("verifying config exists and is proper:")
        with open("configs/algs/"+alg_config+".json") as f:
            run_default_params = json.load(f)
        runtime_params[i].update(run_default_params.copy())
        print("config found. Replacing settings based on experiment configs.")
        for override_key, override_value in experiment_params["overrides_alg"].items():
            if override_key in run_default_params:
                print("setting of",override_key,"currently at",run_default_params[override_key], "overriden to", override_value,".")
            else:
                print("override key of", override_key, "not found in run params, setting it to value", override_value, "anyway.")
            runtime_params[i][override_key] = override_value
        print("full runtime alg configuration settings:")
        print(runtime_params[i])
        print("----------")

    # seed = experiment_params["seed"]

    if log_setting != "none":
        experiment_log_dir = f'./logs/{experiment_name}/'
        if log_setting == "warn":
            if os.path.exists(experiment_log_dir):
                print("WARNING: experiment has been run before. Check the files, and delete or change setting from \'warn\'.")
                quit()
        elif log_setting == "overwrite":
            if(os.path.exists(experiment_log_dir)):
                shutil.rmtree(experiment_log_dir) #delete the old experiment! be careful with this setting. 
        elif log_setting =="timestamp":
            experiment_log_dir=experiment_log_dir[:-1]+'_'+datetime_stamp()+'/'

        os.mkdir(experiment_log_dir)
        with open(experiment_log_dir+"settings.json", "w") as f:
            json.dump(experiment_params, f, indent=2) #put the experiment json params next to the data which resulted from a run with those parameters

        if experiment_params["save_trials"] != None:
            model_save_dir = experiment_log_dir +"models/"
            if not os.path.exists(model_save_dir):
                os.mkdir(model_save_dir)

        if log_setting not in SUPPORTED_LOG_SETTINGS:
            raise Exception("unsupported logging setting. Try none, overwrite, warn, or timestamp.")
        training_logger = TrainingLogger()
        


    for i, run_params in enumerate(runtime_params):
        alg_config = run_params["name"]
        print(run_params)
        
        random.seed(run_params['seed']) #is it really correct to do this in the loop with the outer experiment seed?
        np.random.seed(run_params['seed'])
        if "env_params" in experiment_params.keys():
            domain = gym.make(run_params['env'], **experiment_params["env_params"])
        else:
            domain = gym.make(run_params['env']) #often overriden by experiment for consistency

        #handle custom wrappers:
        if "env_wrappers" in run_params:
            for env_wrapper in run_params["env_wrappers"]: #wrappers will be applied first to last in the order of the list
                if 'name' not in env_wrapper or env_wrapper['name'] not in SUPPORTED_WRAPPERS:
                    raise Exception("wrappers misconfigured, or otherwise not currently supported")
                wrapper_name = env_wrapper['name']
                wrapper_params = env_wrapper['wrapper_params']
                try:
                    domain = setup_wrapper(domain, wrapper_name, wrapper_params)
                except Exception as e:
                    continue
                
        if "env_wrapper" in run_params:
            if 'name' not in run_params['env_wrapper'] or run_params['env_wrapper']['name'] not in SUPPORTED_WRAPPERS:
                raise Exception("wrapper misconfigured, or otherwise not currently supported")
            wrapper_name = run_params['env_wrapper']['name']
            wrapper_params = run_params['env_wrapper']["wrapper_params"]
        
            try:
                domain = setup_wrapper(domain, wrapper_name, wrapper_params)
            except Exception as e:
                continue

        alg_name = run_params["alg"]
        
        if save_trials_setting == "best": #this won't take into account old saved models if you're running "best".
            best_trial = -1
            best_score = -math.inf

        for t in range(experiment_params["trials"]):
            if "baselines" in run_params["alg"]: #currently all that is supported. TODO support non-baselines also
                alg_name = run_params["alg"].split('/')[-1]
                try:
                    model = Baseline(alg_name, domain, run_params["alg_params"])

                except Exception as e:
                    print(e)
                    break #if we cannot run this baseline, we just try another.

            else:
                try:
                    module = importlib.import_module("RL.alg")
                    alg_class = getattr(module, alg_name)
                    model = alg_class(alg_name,domain, run_params["alg_params"])
                except Exception as e:
                    print(e)
                    break #if we cannot run this baseline, we just try another.
            print(alg_name)

           
            if log_setting == "none":
                model.learn(total_timesteps=run_params["total_steps"]) #simply don't log anything
            else:
                trial_log_dir = experiment_log_dir + f'{alg_config}_{t}'
                if not os.path.exists(trial_log_dir):
                    os.mkdir(trial_log_dir)
                with open(trial_log_dir+"/alg_settings.json", "w") as f:
                    json.dump(run_params, f, indent=2) #put the algorithm parameters next to the data which resulted from a trial using those params
                training_logger.set_log_dir(trial_log_dir)
                model.set_logger(training_logger)
                model.learn(total_timesteps=run_params["total_steps"])

                if t == 0 and save_trials_setting == "first":
                    model.save(model_save_dir,f'model:{alg_config}_{t}')
                elif save_trials_setting == "all":
                    model.save(model_save_dir,f'model:{alg_config}_{t}')
                elif save_trials_setting == "best":
                    _ , trial_contents = handle_trial(trial_log_dir)
                    rewards = trial_contents["rewards"]
                    score = np.average(rewards) #take a simple average over the whole trial!
                    old_best_score = best_score
                    old_best_trial = best_trial
                    best_score = best_score if score < best_score else score
                    if best_score != old_best_score:
                        model.save(model_save_dir, f'model:{alg_config}_{t}') #save the new model
                        os.system(f'rm -f {model_save_dir}/model:{alg_config}_{old_best_trial}*') #delete the old saved model (but only after the new one is saved)
                        best_trial = t
                        
                    with open(model_save_dir+"/scores.txt", "a") as f:
                        f.write(f'{alg_config}_{t}'+ ":" + str(score) + '\n')
                    
                    # with open(best_trial_score,"r") as f:
                    #     old_best_score = int(f.read())
                    #TODO: finish best trial score implementation
                    training_logger.reset()

                # print("training count", callback.training_count)
                # print("episode count", callback.episode_count)
                # print("rollout count", callback.rollout_count)
                # print("n_calls", callback.n_calls)
                # print("num_timesteps", callback.num_timesteps)
    
if __name__ == '__main__':
   main()