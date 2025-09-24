import numpy as np
import os, json, glob
from utils.utils import get_files, get_dirs, load_episode_log, convert_dict_obs_to_arr
from utils.core import handle_settings
import math


def tolerant_stats(arrs, rolling_window = 1): #averaging performance values between trials of varying lengths: [np.arr(1..n),...,np.arr(1..m)]
    # lens = [len(i) for i in arrs]
    # arr = np.ma.empty((np.max(lens),len(arrs)))
    # arr.mask = True
    # for idx, l in enumerate(arrs):
    #     arr[:len(l),idx] = l
    arr = tolerant_rolling_average(arrs, window_size=rolling_window)
    return np.array(arr.mean(axis = -1)), np.array(arr.std(axis=-1)), np.array(arr.max(axis=-1)), np.array(arr.min(axis=-1))

def tolerant_rolling_average(arrs, window_size = 1):
    lens = [len(i) for i in arrs]
    n_cols = len(arrs)
    max_len = np.max(lens)
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l

    data = arr.filled(0.0)
    valid = (np.invert(arr.mask)).astype(np.int64)

    # Cumulative sums to get sliding window sums/counts in O(n)
    # Prepend a zero row to make window [i-w+1, i] as diff of cumsums
    pad0 = np.zeros((1, n_cols), dtype=data.dtype)
    pad0i = np.zeros((1, n_cols), dtype=np.int64)

    csum = np.vstack([pad0, np.cumsum(data, axis=0)])
    cval = np.vstack([pad0i, np.cumsum(valid, axis=0)])

    # For each row i, window sum/count over last `window_size` rows
    # Indices in cumsum space: end = i+1, start = i+1-window_size
    end_idx = np.arange(window_size, max_len + 1)  # cumsum indices with full window
    start_idx = end_idx - window_size

    w_sum = csum[end_idx, :] - csum[start_idx, :]
    w_cnt = cval[end_idx, :] - cval[start_idx, :]

    # Rows where EVERY column has a full window of valid entries
    full_rows = np.all(w_cnt == window_size, axis=1)  # length max_len - window_size + 1

    # Prepare output (fully masked by default)
    out = np.ma.empty((max_len, n_cols), dtype=float)
    out.mask = True

    # Map the valid means back to the correct output rows: i = window_size-1 ... max_len-1
    row_indices = np.arange(window_size - 1, max_len)
    emit_rows = row_indices[full_rows]
    if emit_rows.size:
        means = w_sum[full_rows, :] / float(window_size)
        out[emit_rows, :] = means
    return out

def handle_stats_line(line_segments, keyword):
    return [x for x in line_segments if keyword in x][0].split(' ')[-1]

def handle_trial(path, incl_reward = True, incl_obs = False, incl_act = False, incl_base = False, incl_goal = False, max_steps = 1e6): #processes a trial directory
    alg_name = os.path.basename(path).split('_')[0]

    if not (incl_reward or incl_obs or incl_act or incl_base):
        return None
    
    episodes = []
    rewards = []
    base_rewards = []
    observations = []
    actions = []
    steps = []
    goals = [] #for use only with stats for now
    files = get_files(path)
    dirs = get_dirs(path)
    print("reading files", files)
    print("reading dirs", dirs)
    if not (incl_obs or incl_act):
        for item in files:
            if os.path.basename(item) == 'stats.txt': #we don't use maxsteps here because it's unnecessary, also we want full graphs
                print("reading stats.txt for trial")
                with open(item) as f:
                    content = f.readlines()
                    for l in content:
                        ep_name, stats = l.split(':')
                        ep_number = int(ep_name.split('_')[1])
                        episodes.append(ep_number) #TODO: have we verified this is happening in order?
                        segments = stats.split(',')
                        
                        steps.append(int(handle_stats_line(segments, 'Steps')))

                        if incl_reward:
                            rewards.append(float(handle_stats_line(segments, 'Total Reward')))

                        if incl_base:
                            base_rewards.append(float(handle_stats_line(segments, 'Total Base')))

                        if incl_goal:
                            goals.append(float(handle_stats_line(segments, 'Goal')))

        print(len(episodes), episodes[-1], len(rewards), len(base_rewards))
        
        return alg_name, {'rewards': np.array(rewards), 'base rewards': np.array(base_rewards), 'timesteps': np.array(steps), 'goals': np.array(goals)}
    else: 
        for dir in dirs:
            steps = 0 #we use max_steps here because we are reading the actual logs. it can get very expensive
            if os.path.basename(dir) == 'train_episodes':
                print("reading training episodes for trial")
                items = glob.glob(os.path.join(dir, 'episode_*.json')) #TODO: are we set up to handle the fact that this reads out of order?
                for item in items:
                    log_contents = load_episode_log(item)
                    steps += len(log_contents['rewards'])
                    if incl_reward:
                        rewards.append(log_contents['rewards'])
                    if incl_obs:
                        observations.append(log_contents['observations'])
                    if incl_act:
                        actions.append(log_contents['actions'])
                    if incl_base:
                        base_rewards.append([item['base'] for item in log_contents['info']]) #TODO TEST, & add steps also!

                    if steps > max_steps:
                        print("steps exceeded max! Only grabbing first", steps)
                        break
        trial_contents = {}
        if incl_reward:
            trial_contents['rewards'] = rewards
        if incl_obs:
            trial_contents['observations'] = observations
        if incl_act:
            trial_contents['actions'] = actions
        return alg_name, trial_contents

def retrieve_trial(alg_name, trial_data, keyword, stat_dict):
    if alg_name not in stat_dict:
        stat_dict[alg_name] = []
    stat_dict[alg_name].append(trial_data[keyword])

def compute_stat(alg, stat_trials_data, stat_results_dict, goal_range = (-math.inf,math.inf), goal_trials_data = None, rolling_window = 30):

    if not goal_trials_data:
        avg, std, smax, smin = tolerant_stats(stat_trials_data, rolling_window=rolling_window) 
        stat_results_dict[alg] = {"means": avg, "stds": std, "max": smax, "min": smin}
    else:
        goal_aggr_stat_trials_data = []
        goal_aggr_indices = []
        # print(len(goal_trials_data))
        for i, trial in enumerate(goal_trials_data):
            print("trial length", len(trial))
            trial_goal_aggr_stat_data = []
            trial_goal_aggr_indices = []
            # print("stat_trials_data length", len(stat_trials_data))
            for j, goal in enumerate(trial):
                # print(goal)
                if goal_range[0] <= goal <= goal_range[1]: #this relies on goal not changing during the episode! be careful
                    trial_goal_aggr_stat_data.append(stat_trials_data[i][j])
                    trial_goal_aggr_indices.append(j)
            goal_aggr_stat_trials_data.append(np.array(trial_goal_aggr_stat_data))
            goal_aggr_indices.append(trial_goal_aggr_indices)

        for aggr in goal_aggr_stat_trials_data:
            print(aggr.size)
        avg, std, smax, smin = tolerant_stats(goal_aggr_stat_trials_data)
        # print(stat_results_dict.keys())
        if alg not in stat_results_dict:
            stat_results_dict[alg] = {}
        stat_results_dict[alg][str(goal_range)] = {"means": avg, "stds": std, "max": smax, "min": smin}


def _compute_stats(experiment_name, rewards = True, observations = True, actions = True, base_rewards = False, num_goal_buckets = 2, max_steps = 1e6):
    experiment_dir = f'./logs/{experiment_name}'
    
    files = get_files(experiment_dir)
    dirs =  get_dirs(experiment_dir)

    for item in files:
        if os.path.basename(item) == 'settings.json':
            print("reading settings.json")
            try:
                handle_settings(item) #this actually does nothing! TODO actually use the information in settings.json to improve the graphs...
            except:
                print(f"unexpected issue handling file/directory: {item}")
    num_trials = 0

    trial_rewards = {}
    trial_base_rewards = {}
    trial_base_steps = {}
    trial_goals = {}
    trial_observations = {}
    trial_actions = {}
    goals = True if num_goal_buckets else False
    for item in dirs:
        if "_" not in os.path.basename(item):
            continue
        print(f"processing trial: {item}")
        try:
            alg_name, trial_data = (handle_trial(item,incl_reward=rewards, incl_obs=observations, incl_act=actions, incl_base=base_rewards, incl_goal = goals))
            print(alg_name, len(trial_data))
            num_trials+=1

            if rewards:
                retrieve_trial(alg_name, trial_data, 'rewards', trial_rewards)
            if base_rewards:
                retrieve_trial(alg_name, trial_data, 'base rewards', trial_base_rewards)
                retrieve_trial(alg_name, trial_data, 'timesteps', trial_base_steps)
            if goals:
                retrieve_trial(alg_name, trial_data, 'goals', trial_goals)
            if observations:
                retrieve_trial(alg_name, trial_data, 'observations', trial_observations)
            if actions:
                retrieve_trial(alg_name, trial_data, 'actions', trial_actions)


        except:
            print(f"unexpected issue handling file/directory: {item}") 
    results = {}
    goal_buckets = [(-math.inf, math.inf)] #even in the case of no goals or goal-bucketing, using this in some places makes the code shorter
    if goals:
        goal_stats = {}

        for alg, goal_trials in trial_goals.items():
            compute_stat(alg, goal_trials, goal_stats)
        goal_min = min(goal_stats[alg]["min"])
        goal_max = max(goal_stats[alg]["max"])
        goal_range = goal_max - goal_min
        goal_bucket_size = goal_range/num_goal_buckets
        for i in range(num_goal_buckets):
            goal_bucket_lower = goal_min + i*goal_bucket_size
            goal_bucket_upper = goal_min + (i+1)*goal_bucket_size
            # print(goal_bucket_lower, goal_bucket_upper)
            goal_buckets.append((goal_bucket_lower, goal_bucket_upper))
        print("goal buckets", goal_buckets)

    reward_results = {}
    base_reward_results = {}
    steps_results = {}
    base_avg_results = {}
    observation_results = {}

    for bucket in goal_buckets:
        print("HANDLING BUCKET", bucket)
        #handle rewards
        if rewards:
            # print(len(trial_rewards.keys()))
            # print({key:len(value) for key,value in trial_rewards.items()})
            for alg, trials in trial_rewards.items():
                print("alg", alg)
                print("trials", len(trials))
                if goals:
                    goals_trials = trial_goals[alg]
                    goal_range = bucket
                else:
                    goals_trials = goal_range = None
                    # print(trials)
                compute_stat(alg, trials, reward_results, goal_range = bucket, goal_trials_data = goals_trials)

            results["rewards"] = reward_results

        if base_rewards:
            # print(len(trial_rewards.keys()))
            # print({key:len(value) for key,value in trial_rewards.items()})
            
            for alg in trial_base_rewards.keys():
                # goals_trials = trial_goals[alg]
                base_reward_trials = trial_base_rewards[alg]
                steps_trials = trial_base_steps[alg]
                base_avg_trials = [(trial_base_rewards[alg][i] / trial_base_steps[alg][i]).tolist() for i in range(len(base_reward_trials))]
                # base_avg_trials = (np.array(trial_base_rewards[alg]) / np.array(trial_base_steps[alg])).tolist()
                if goals:
                    goals_trials = trial_goals[alg]
                    goal_range = bucket
                else:
                    goals_trials = goal_range = None
                compute_stat(alg, base_reward_trials, base_reward_results, goal_range = bucket, goal_trials_data = goals_trials)
                compute_stat(alg, steps_trials, steps_results, goal_range = bucket, goal_trials_data = goals_trials)
                compute_stat(alg, base_avg_trials, base_avg_results, goal_range = bucket, goal_trials_data = goals_trials)

            results["base rewards"] = base_reward_results            
            results["steps"] = steps_results
            results["base average"] = base_avg_results

        
        #TODO handle observations (is this done? TODO test) not implemented with buckets
        if observations:
            print(len(trial_observations.keys()))
            for alg, trials in trial_observations.items():
                collect_obs = []
                coll_keys = None
                for trial in trials:
                    for episode in trial:
                        for obs in episode:
                            coll_keys, obs = convert_dict_obs_to_arr(obs)
                            collect_obs.append(obs)
                obs_matrix = np.stack(collect_obs,axis = 0)

                obs_avg = np.mean(obs_matrix,axis = 0)
                obs_std = np.std(obs_matrix,axis = 0)
                obs_max = np.max(obs_matrix,axis = 0)
                obs_min = np.min(obs_matrix,axis = 0)

                observation_results[alg] = {"means": obs_avg, "stds": obs_std, "max": obs_max, "min": obs_min}
            print("observation key order (if applicable):", coll_keys)
            results["observations"] = observation_results

        #TODO handle actions
    # if actions:
    #     print(len(trial_rewards.keys()))
    #     print({key:len(value) for key,value in trial_rewards.items()})
    #     reward_results = {}
    #     for alg,trials in trial_rewards.items():
    #         reward_avg, reward_std, reward_max, reward_min = tolerant_stats(trials)
    #         print(alg, len(reward_avg), len(reward_std))
    #         reward_results[alg] = {"means": reward_avg, "stds": reward_std, "max": reward_max, "min": reward_min}
    #     results["rewards"] = reward_results

    return results

def compute_rewards(experiment_name, base = False, num_goal_buckets = None):
    return _compute_stats(experiment_name, True, False, False, base_rewards=base, num_goal_buckets = num_goal_buckets)

def compute_observations(experiment_name):
    return _compute_stats(experiment_name,False, True, False)['observations']

def compute_actions(experiment_name):
    return _compute_stats(experiment_name,False, False, True)['actions']

def compute_all(experiment_name):
    return _compute_stats(experiment_name)
