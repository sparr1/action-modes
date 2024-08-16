import datetime
import numpy as np
import glob, os, json, sys


def datetime_stamp():
    now = datetime.datetime.now()
    datetime_stamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    return datetime_stamp

def tolerant_stats(arrs): #averaging performance values between trials of varying lengths: [np.arr(1..n),...,np.arr(1..m)]
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return np.array(arr.mean(axis = -1)), np.array(arr.std(axis=-1)), np.array(arr.max(axis=-1)), np.array(arr.min(axis=-1))

# def bound_observations(arrs):

def get_files(dir): #get the names of all the files in a directory
    items = glob.glob(os.path.join(dir, '*'))
    return [item for item in items if os.path.isfile(item)]

def get_dirs(dir): #get the names of all the subdirectories in a directory
    items = glob.glob(os.path.join(dir, '*'))
    return [item for item in items if os.path.isdir(item)]

#An experiment directory consists of a settings.json file, and a series of directories labeled "algname_n" for the nth run of algname


#TODO: currently does nothing. either add functionality or delete
def handle_settings(path): #processes the settings.json file
    with open(path) as f:
        contents = json.load(f)
    print(contents) 

def handle_trial(path, incl_reward = True, incl_obs = False, incl_act = False,max_steps = 1e6): #processes a trial directory
    alg_name = os.path.basename(path).split('_')[0]

    if not (incl_reward or incl_obs or incl_act):
        return None
    
    episodes = []
    rewards = []
    observations = []
    actions = []
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
                        total_reward_segment = stats.split(',')[0]
                        total_reward = total_reward_segment.split(' ')[-1]
                        rewards.append(float(total_reward))
        print(len(episodes), episodes[-1], len(rewards))
        return alg_name, {'rewards': np.array(rewards)}
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

def convert_dict_obs_to_arr(obs):
    coll_obs = []
    coll_keys = []
    for k,v in obs.items():
        coll_keys.append(k)
        # print(type(v))
        coll_obs.append(np.array(v))
        # print(coll_keys)
    # for o in coll_obs:
    #     print(o.shape)
    return coll_keys, np.concatenate(coll_obs, axis = None)

def _compute_stats(experiment_name, rewards = True, observations = True, actions = True, max_steps = 1e6):
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
    trial_observations = {}
    trial_actions = {}

    for item in dirs:
        if "_" not in os.path.basename(item):
            continue
        print(f"processing trial: {item}")
        try:
            alg_name, trial_data = (handle_trial(item,incl_reward=rewards, incl_obs=observations, incl_act=actions))
            print(alg_name, len(trial_data))
            num_trials+=1

            if rewards:
                if alg_name not in trial_rewards:
                    trial_rewards[alg_name] = []
                trial_rewards[alg_name].append(trial_data['rewards'])
            if observations:
                if alg_name not in trial_observations:
                    trial_observations[alg_name] = []
                trial_observations[alg_name].append(trial_data['observations'])
            if actions:
                if alg_name not in trial_actions:
                    trial_actions[alg_name] = []
                trial_actions[alg_name].append(trial_data['actions'])

        except: 
            print(f"unexpected issue handling file/directory: {item}") 
    #handle rewards
    results = {}
    if rewards:
        # print(len(trial_rewards.keys()))
        # print({key:len(value) for key,value in trial_rewards.items()})
        reward_results = {}
        for alg, trials in trial_rewards.items():
            reward_avg, reward_std, reward_max, reward_min = tolerant_stats(trials)
            print(alg, len(reward_avg), len(reward_std))
            reward_results[alg] = {"means": reward_avg, "stds": reward_std, "max": reward_max, "min": reward_min}
        results["rewards"] = reward_results

    #TODO handle observations
    if observations:
        print(len(trial_observations.keys()))
        # print({key:len(value) for key,value in trial_observations.items()})
        observation_results = {}
        for alg, trials in trial_observations.items():
            # observation_results[alg] = {}
            # print(type(trials))
            # print(type(trials[0]))
            # #sometimes, "trials" is a list of list of lists...
            # for trial in trials:
            #     for episode in trial:
            #         for timestep in episode:
            #             print(type(timestep))
            collect_obs = []
            coll_keys = None
            for trial in trials:
                for episode in trial:
                    for obs in episode:
                        # if type(obs) == dict:
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


def compute_rewards(experiment_name):
    return _compute_stats(experiment_name, True, False, False)['rewards']

def compute_observations(experiment_name):
    return _compute_stats(experiment_name,False, True, False)['observations']

def compute_actions(experiment_name):
    return _compute_stats(experiment_name,False, False, True)['actions']

def compute_all(experiment_name):
    return _compute_stats(experiment_name)

def load_episode_log(path):
    with open(path) as f:
        contents = json.load(f)
    
    return {'rewards': contents['rewards'], 'observations':contents['observations'], 'actions': contents['actions']}

if __name__ == "__main__":

    # results = compute_rewards("DQNShort_old")
    # print(results.keys())
    # DQNCartSHORT2 = results['DQNCartSHORT2']
    # DQNCartSHORT = results['DQNCartSHORT']
    # means, stds, maxes, mins = DQNCartSHORT['means'],DQNCartSHORT['stds'], DQNCartSHORT['max'], DQNCartSHORT['min']
    # print(np.min(mins))
    # print(np.max(maxes))

    results = compute_observations("AntMazeSparse_2024-07-23_16-51-17")
    np.set_printoptions(threshold=sys.maxsize)
    labels = ["x velocity", "y velocity", "z velocity", "x angular velocty", "y angular velocity", "z angular velocity"]
    for alg, stats in results.items():
        print(alg)
        
        for k,v in stats.items():
            print(k+":")
            for i, label in enumerate(labels):
                print(label+":",v[17+i])
