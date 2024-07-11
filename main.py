import random, time, argparse, json, os, shutil

import numpy as np
import seaborn as sns
import pandas as pd

import gymnasium as gym

from RL.baselines import Baseline, TrajectoryLoggerCallback
from utils import datetime_stamp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="action mode learning experiments")
    parser.add_argument('-r', '--run', help='config file for a run', required=True)
    config = vars(parser.parse_args())['run']

    with open(config) as f:
        experiment_params = json.load(f)
        experiment_name = config.split('/')[-1][:-5] #cut out everything before the / and the extension .json
    # print(experiment_params)

    if "logs" in experiment_params:
        log_setting = experiment_params["logs"]
    else:
        log_setting = "warn" #just manually enforcing an annoying default because I love my users and I don't want them deleting their logs or causing clutter by default :)

    if "save" in experiment_params:
        save_setting = experiment_params["save"]
    else:
        save_setting = "first" #again, not the best setting, but one which causes little harm. For a small number of trials, I recommend "every". For large, I recommend "best".
    #these save settings involve choosing which models to save for any given trial or run. In general, we have a lot of options, and the code is not the most straightforward thing in the world
    #what's the MVP for saving models? I'm nervous about saving "best" as is usual in ML, because RL has no easy validation set from which to make an unbiased selection. 
    #furthermore, there are some difficulties involved in model selection within each trial, and across trials! I want to run a lot of trials, but I don't want to save a lot of models...
    #my solution for this for now is to simply save models from only the first trial... 

    num_algs = len(experiment_params["configs"])

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
            else:
                pass #do nothing on none
            os.mkdir(experiment_log_dir)
            with open(experiment_log_dir+"settings.json", "w") as f:
                json.dump(experiment_params, f, indent=2) #put the experiment json params next to the data which resulted from a run with those parameters


    for i, alg_config in enumerate(experiment_params["configs"]):
        with open("configs/algs/"+alg_config+".json") as f:
            run_params = json.load(f)
        print(run_params)
        
        random.seed(experiment_params['seed']) #is it really correct to do this in the loop with the outer experiment seed?
        np.random.seed(experiment_params['seed'])
        
        if(run_params['env'] != experiment_params['env']):
            raise Exception(f"Mismatch of algorithm and experiment environment configurations: algorithm expects {run_params['env']} while experiment expects {experiment_params['env']}")
        
        domain = gym.make(run_params['env'])
        alg_name = run_params["alg"]
        
        for t in range(experiment_params["trials"]):
            if "baselines" in run_params["alg"]: #currently all that is supported. TODO support non-baselines also
                alg_name = run_params["alg"].split('/')[-1]
            print(alg_name)

            try:
                model = Baseline(alg_name, domain, run_params["alg_params"]).get_model()
            except Exception as e:
                print(e)
                break #if we cannot run this baseline, we just try another.

            if log_setting == "none":
                model.learn(total_timesteps=run_params["total_steps"]) #simply don't log anything
            else:
                trial_log_dir = experiment_log_dir + f'{alg_config}_{t}'
                if not os.path.exists(trial_log_dir):
                    os.mkdir(trial_log_dir)
                with open(trial_log_dir+"/alg_settings.json", "w") as f:
                    json.dump(run_params, f, indent=2) #put the algorithm parameters next to the data which resulted from a trial using those params

                callback = TrajectoryLoggerCallback(trial_log_dir, log_setting=log_setting) #the logger will handle log settings. duh!
                model.learn(total_timesteps=run_params["total_steps"],callback=callback)
                print("training count", callback.training_count)
                print("episode count", callback.episode_count)
                print("rollout count", callback.rollout_count)
                print("n_calls", callback.n_calls)
                print("num_timesteps", callback.num_timesteps)






        
        