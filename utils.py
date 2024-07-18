import datetime
import numpy as np
import glob, os, json

def datetime_stamp():
    now = datetime.datetime.now()
    datetime_stamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    return datetime_stamp

def tolerant_mean(arrs): #averaging performance values between trials of varying lengths
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return np.array(arr.mean(axis = -1)), np.array(arr.std(axis=-1))


def handle_settings(path):
    with open(path) as f:
        contents = json.load(f)
    print(contents)

def handle_trial(path):
    episodes = []
    rewards = []
    items = glob.glob(os.path.join(path, '*'))
    alg_name = os.path.basename(path).split('_')[0]
    files = [item for item in items if os.path.isfile(item)]
    for item in files:
        if os.path.basename(item) == 'stats.txt':
            print("reading stats.txt for trial")
            with open(item) as f:
                content = f.readlines()
                for l in content:
                    ep_name, stats = l.split(':')
                    ep_number = int(ep_name.split('_')[1])
                    episodes.append(ep_number)
                    total_reward_segment = stats.split(',')[0]
                    total_reward = total_reward_segment.split(' ')[-1]
                    rewards.append(float(total_reward))
    print(len(episodes), episodes[-1], len(rewards))
    return alg_name, np.array(rewards)

def compute_rewards(experiment_name):
    experiment_dir = f'./logs/{experiment_name}'
    items = glob.glob(os.path.join(experiment_dir, '*'))
    files = [item for item in items if os.path.isfile(item)]
    dirs =  [item for item in items if os.path.isdir(item)]
    for item in files:
        if os.path.basename(item) == 'settings.json':
            print("reading settings.json")
            try:
                handle_settings(item) #this actually does nothing! TODO actually use the information in settings.json to improve the graphs...
            except:
                print(f"unexpected issue handling file/directory: {item}")
    num_trials = 0
    trial_rewards = {}
    
    for item in dirs:
        print(f"processing trial: {item}")
        try:
            alg_name, trial_data = (handle_trial(item))
            print(alg_name, len(trial_data))
            num_trials+=1
            if alg_name not in trial_rewards:
                trial_rewards[alg_name] = []
            trial_rewards[alg_name].append(trial_data)
        except: 
            print(f"unexpected issue handling file/directory: {item}") 
    print(len(trial_rewards.keys()))
    print({key:len(value) for key,value in trial_rewards.items()})
    results = {}
    for alg,trials in trial_rewards.items():
        reward_avg, reward_std = tolerant_mean(trials)
        print(alg, len(reward_avg), len(reward_std))
        results[alg] = {"means": reward_avg, "stds": reward_std}
    return results