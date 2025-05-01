import random, time, argparse, json, os, shutil, math, importlib

import numpy as np
import seaborn as sns
import pandas as pd

import gymnasium as gym
# from RL.alg import *
from RL.baselines import Baseline, TrajectoryLoggerCallback
from log import TrainingLogger
from domains.tasks import *
from domains.AntPlane import *
#from domains.mpqdn_goal_domain import *
#from domains.mpqdn_platform_domain import *
#from domains.mpqdn_wrappers import *
#import gymnasium_goal
#import gymnasium_platform

from utils import *
def main():
    parser = argparse.ArgumentParser(description="action mode learning experiments")
    parser.add_argument('-r', '--run', help='config file for a run', required=True,)
    parser.add_argument('--alg-dir', help='location of alg configs', default = os.path.join("configs","algs", ""))
    parser.add_argument('--log-dir', help='desired location for logging', default = os.path.join(".","logs", ""))
    parser.add_argument('--num-runs', help='number of consecutive trials to run', default = -1, type = int)
    parser.add_argument('--alg-index', help='which algorithm to start running first',default = 0, type=int)
    parser.add_argument('--trial-index', help='which trial index to start from', default = 0, type = int)
    args = vars(parser.parse_args())
    print(args)
    config, alg_dir, log_dir, num_runs, alg_ind, trial_ind = args['run'], args['alg_dir'], args['log_dir'], args['num_runs'], args['alg_index'], args['trial_index']


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

    if "log_info" in experiment_params:
        log_info_setting = experiment_params["log_info"]
    else:
        log_info_setting = True #backwards compatibility

    if "log_type" in experiment_params:
        log_type_setting = experiment_params["log_type"]
    else:
        log_type_setting = "detailed" #backwards compatibility

    if "checkpoint_every" in experiment_params:
        checkpoint_every = experiment_params["checkpoint_every"]
    else:
        checkpoint_every = None #unfortunately, this is the backwards compatible default!

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
        with open(os.path.join(alg_dir,alg_config+".json")) as f:
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
        experiment_log_dir = os.path.join(log_dir, f'{experiment_name}', '')
        if log_setting == "warn":
            if os.path.exists(experiment_log_dir):
                print("WARNING: experiment has been run before. Check the files, and delete or change setting from \'warn\'.")
                quit()
        elif log_setting == "overwrite":
            if(os.path.exists(experiment_log_dir)):
                shutil.rmtree(experiment_log_dir) #delete the old experiment! be careful with this setting. 
            skip = False
        elif log_setting =="timestamp":
            experiment_log_dir=os.path.join(experiment_log_dir[:-1]+'_'+datetime_stamp(),"") #is there a way to do this with os path?
            skip = False
        elif log_setting == "overwrite-safe":
            print("this run started at", datetime_stamp())
            if os.path.exists(experiment_log_dir):
                print("Experiment folder already exists. Trials will only proceed if not run before.")
                skip = True
            else:
                skip = False
        if not skip:
            os.mkdir(experiment_log_dir)
            with open(experiment_log_dir+"settings.json", "w") as f:
                json.dump(experiment_params, f, indent=2) #put the experiment json params next to the data which resulted from a run with those parameters
        else:
            try: 
                with open(experiment_log_dir+"settings.json", "r") as f:
                    existing_settings = json.load(f)
                    print("Found existing settings:", existing_settings)
            except:
                print("WARNING: experiment folder existed, but there was an issue reading the settings file. Proceeding with caution under the current settings.")
                

        if experiment_params["save_trials"] != None:
            model_save_dir = os.path.join(experiment_log_dir, "models", "")
            if not os.path.exists(model_save_dir):
                os.mkdir(model_save_dir)

        if log_setting not in SUPPORTED_LOG_SETTINGS:
            raise Exception("unsupported logging setting. Try none, overwrite, warn, or timestamp.")
        training_logger = TrainingLogger(log_info=log_info_setting, log_type = log_type_setting)
    
    i
    ran_so_far = 0

    for i, run_params in enumerate(runtime_params, start = alg_ind):
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
                if 'name' not in env_wrapper or env_wrapper['name'].split(':')[-1] not in SUPPORTED_WRAPPERS:
                    raise Exception("wrappers misconfigured, or otherwise not currently supported")
                wrapper_name = env_wrapper['name']
                wrapper_params = env_wrapper['wrapper_params']
                domain = setup_wrapper(domain, wrapper_name, wrapper_params)
                
                
        if "env_wrapper" in run_params:
            if 'name' not in run_params['env_wrapper'] or run_params['env_wrapper']['name'].split(':')[-1] not in SUPPORTED_WRAPPERS:
                raise Exception("wrapper misconfigured, or otherwise not currently supported")
            wrapper_name = run_params['env_wrapper']['name']
            wrapper_params = run_params['env_wrapper']["wrapper_params"]
        
            domain = setup_wrapper(domain, wrapper_name, wrapper_params)
            

        alg_name = run_params["alg"]
        
        if save_trials_setting == "best": #this won't take into account old saved models if you're running "best".
            best_trial = -1
            best_score = -math.inf

        for t in range(trial_ind, experiment_params["trials"]):
            if ran_so_far == num_runs:
                print("completed running", num_runs, "trials!")
                quit()
            baseline = False
            if '/' in run_params["alg"]:
                file_name, alg_name = "".join(run_params["alg"].split('/')[:-1]), run_params["alg"].split('/')[-1]
                print(file_name)
                print(alg_name)
                if "baselines" in file_name: #currently all that is supported. TODO support non-baselines also
                    baseline = True
                    try:
                        model = Baseline(alg_name, domain, run_params["alg_params"])
                    except Exception as e:
                        print(e)
                        break #if we cannot run this baseline, we just try another
                elif file_name == "PAMDP":
                    try: 
                        module = importlib.import_module("RL.PAMDP")
                        alg_class = getattr(module, alg_name)
                        model = alg_class(alg_name,domain, run_params["alg_params"])
                    except Exception as e:
                        print(e)
                        break #if we cannot run this baseline, we just try another.
                else:
                    try:
                        module = importlib.import_module("RL."+file_name.replace('/','.')) #last ditch, just try to load it!
                        alg_class = getattr(module, alg_name)
                        model = alg_class(alg_name,domain, run_params["alg_params"])
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
                if os.path.exists(trial_log_dir):
                    print("WARNING: trial log dir already existed!")
                    if log_setting == "overwrite-safe":
                        print("quitting, as to continue would risk an overwrite.")
                        quit()
                else:
                    os.mkdir(trial_log_dir)
                with open(os.path.join(trial_log_dir,"alg_settings.json"), "w") as f:
                    json.dump(run_params, f, indent=2) #put the algorithm parameters next to the data which resulted from a trial using those params
                training_logger.set_log_dir(trial_log_dir)
                model.set_logger(training_logger)
                if checkpoint_every and baseline: #for now, we are only supporting checkpointing the baselines...
                    model.set_checkpointing(save_freq=checkpoint_every, save_path=model_save_dir, name_prefix=f'model:{alg_config}_{t}')
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
                        old_model_filename = os.path.join(model_save_dir,f'model:{alg_config}_{old_best_trial}*')
                        os.system(f'rm -f {old_model_filename}') #delete the old saved model (but only after the new one is saved)
                        best_trial = t
                        
                    with open(os.path.join(model_save_dir,"scores.txt"), "a") as f:
                        f.write(f'{alg_config}_{t}'+ ":" + str(score) + '\n')
                    
                    # with open(best_trial_score,"r") as f:
                    #     old_best_score = int(f.read())
                    #TODO: finish best trial score implementation
                    training_logger.reset()
                ran_so_far  += 1

                # print("training count", callback.training_count)
                # print("episode count", callback.episode_count)
                # print("rollout count", callback.rollout_count)
                # print("n_calls", callback.n_calls)
                # print("num_timesteps", callback.num_timesteps)
    
if __name__ == '__main__':
   main()
