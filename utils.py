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

def handle_trial(path, incl_reward = True, incl_obs = False, incl_act = False, incl_base = False, max_steps = 1e6): #processes a trial directory
    alg_name = os.path.basename(path).split('_')[0]

    if not (incl_reward or incl_obs or incl_act or incl_base):
        return None
    
    episodes = []
    rewards = []
    base_rewards = []
    observations = []
    actions = []
    steps = []
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
                        episodes.append(ep_number) #TODO: have we verified this is happening in order?\
                        segments = stats.split(',')
                        steps_segment = [x for x in segments if 'Steps' in x][0]
                        # print(steps_segment)
                        timesteps = steps_segment.split(' ')[-1]
                        steps.append(int(timesteps))
                        if incl_reward:
                            total_reward_segment = [x for x in segments if 'Total Reward' in x][0] 
                            # print(total_reward_segment)
                            total_reward = total_reward_segment.split(' ')[-1]
                            rewards.append(float(total_reward))

                        if incl_base:
                            base_reward_segment = [x for x in segments if 'Total Base' in x][0]
                            print(base_reward_segment)
                            base_reward = base_reward_segment.split(' ')[-1]
                            base_rewards.append(float(base_reward))

        print(len(episodes), episodes[-1], len(rewards), len(base_rewards))
        
        return alg_name, {'rewards': np.array(rewards), 'base rewards': np.array(base_rewards), 'timesteps': np.array(steps)}
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

def _compute_stats(experiment_name, rewards = True, observations = True, actions = True, base_rewards = False, max_steps = 1e6):
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
    trial_observations = {}
    trial_actions = {}

    for item in dirs:
        if "_" not in os.path.basename(item):
            continue
        print(f"processing trial: {item}")
        try:
            alg_name, trial_data = (handle_trial(item,incl_reward=rewards, incl_obs=observations, incl_act=actions, incl_base=base_rewards))
            print(alg_name, len(trial_data))
            num_trials+=1

            if rewards:
                if alg_name not in trial_rewards:
                    trial_rewards[alg_name] = []
                trial_rewards[alg_name].append(trial_data['rewards'])
            if base_rewards:
                if alg_name not in trial_base_rewards:
                    trial_base_rewards[alg_name] = []
                    trial_base_steps[alg_name] = []
                trial_base_rewards[alg_name].append(trial_data['base rewards'])
                trial_base_steps[alg_name].append(trial_data['timesteps'])
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
    results = {}

    #handle rewards
    if rewards:
        # print(len(trial_rewards.keys()))
        # print({key:len(value) for key,value in trial_rewards.items()})
        reward_results = {}
        for alg, trials in trial_rewards.items():
            reward_avg, reward_std, reward_max, reward_min = tolerant_stats(trials)
            print(alg, len(reward_avg), len(reward_std))
            reward_results[alg] = {"means": reward_avg, "stds": reward_std, "max": reward_max, "min": reward_min}
        results["rewards"] = reward_results

    if base_rewards:
        # print(len(trial_rewards.keys()))
        # print({key:len(value) for key,value in trial_rewards.items()})
        base_reward_results = {}
        steps_results = {}
        base_avg_results = {}
        for alg in trial_base_rewards.keys():
            base_reward_trials = trial_base_rewards[alg]
            steps_trials = trial_base_steps[alg]

            base_avg_trials = [(trial_base_rewards[alg][i] / trial_base_steps[alg][i]).tolist() for i in range(len(base_reward_trials))]
            # base_avg_trials = (np.array(trial_base_rewards[alg]) / np.array(trial_base_steps[alg])).tolist()

            reward_avg, reward_std, reward_max, reward_min = tolerant_stats(base_reward_trials)
            print(alg, 'base reward lengths:',len(reward_avg), len(reward_std))
            base_reward_results[alg] = {"means": reward_avg, "stds": reward_std, "max": reward_max, "min": reward_min}
            
            steps_avg, steps_std, steps_max, steps_min = tolerant_stats(steps_trials)
            print(alg, 'step lengths:', len(steps_avg), len(steps_std))
            steps_results[alg] = {"means": steps_avg, "stds": steps_std, "max": steps_max, "min": steps_min}

            base_avg_avg, base_avg_std, base_avg_max, base_avg_min = tolerant_stats(base_avg_trials)
            print(alg, "base avg lengths:", len(base_avg_avg), len(base_avg_std))
            base_avg_results[alg] =  {"means": base_avg_avg, "stds": base_avg_std, "max": base_avg_max, "min": base_avg_min}

        results["base rewards"] = base_reward_results            
        results["steps"] = steps_results
        results["base average"] = base_avg_results

    
    #TODO handle observations (is this done? TODO test)
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


def compute_rewards(experiment_name, base = False):
    return _compute_stats(experiment_name, True, False, False, base_rewards=base)

def compute_observations(experiment_name):
    return _compute_stats(experiment_name,False, True, False)['observations']

def compute_actions(experiment_name):
    return _compute_stats(experiment_name,False, False, True)['actions']

def compute_all(experiment_name):
    return _compute_stats(experiment_name)

def setup_logs(reward, obs, action, dones, info = None):
    data = {}
    if type(reward) is np.ndarray:
        data['rewards'] = reward.tolist()
    else:
        data['rewards'] = [reward,]
    if isinstance(obs, dict): #to handle ordered and unordered dicts
        new_obs = [{k:v.tolist() for k,v in obs.items()},]
        # obs = json.dumps(self.locals['new_obs'])
    else:
        new_obs = obs.tolist()
    data["obs"] = new_obs
    data["actions"] = action.tolist()
    data["dones"] = dones
    if info:
        data["infos"] = info[0]["reward_info"]
    return data

def load_episode_log(path):
    with open(path) as f:
        contents = json.load(f)
    logs = {'rewards': contents['rewards'], 'observations':contents['observations'], 'actions': contents['actions']}
    if 'info' in logs:
        logs['info'] = contents['info']
    return logs

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
