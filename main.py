import random, time, argparse, json
import numpy as np
import seaborn as sns
import pandas as pd

import gymnasium as gym
from RL.baselines import Baseline, TrajectoryLoggerCallback


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="action mode learning experiments")
    parser.add_argument('-r', '--run', help='config file for a run', required=True)
    config = vars(parser.parse_args())['run']

    with open(config) as f:
        experiment_params = json.load(f)
        experiment_name = config.split('/')[-1][:-5] #cut out everything before the / and the extension .json
    print(experiment_params)


    alg_list = []
    num_algs = len(experiment_params["configs"])

    r_eps     = np.zeros((num_algs, experiment_params["trials"], experiment_params['episodes']))
    #TODO: store obs also
    time_eps  = np.zeros_like(r_eps)
    steps_eps = np.zeros_like(r_eps)

    # seed = experiment_params["seed"]

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
        if "baselines" in run_params["alg"]: #currently all that is supported. TODO support non-baselines also
            alg_name = run_params["alg"].split('/')[-1]
            print(alg_name)
            model = Baseline(alg_name, domain, run_params["alg_params"]).get_model()

        for t in range(experiment_params["trials"]):
            # alg_state = {}      
            steps = 0
            e = 0
            all_data = np.zeros((experiment_params["episodes"],))

            log_dir = f'./logs/{experiment_name}/{alg_config}_{i}'
            callback = TrajectoryLoggerCallback(log_dir)
            model.learn(total_timesteps=run_params["total_steps"],callback=callback)
            print("training count", callback.training_count)
            print("episode count", callback.episode_count)
            print("rollout count", callback.rollout_count)
            print("n_calls", callback.n_calls)
            print("num_timesteps", callback.num_timesteps)
            # model.learn(total_timesteps=run_params["total_steps"],callback=None)







        
        