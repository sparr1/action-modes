import datetime
import numpy as np
import glob, os, json

def datetime_stamp():
    now = datetime.datetime.now()
    datetime_stamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    return datetime_stamp

def get_files(dir): #get the names of all the files in a directory
    items = glob.glob(os.path.join(dir, '*'))
    return [item for item in items if os.path.isfile(item)]

def get_dirs(dir): #get the names of all the subdirectories in a directory
    items = glob.glob(os.path.join(dir, '*'))
    return [item for item in items if os.path.isdir(item)]

#An experiment directory consists of a settings.json file, and a series of directories labeled "algname_n" for the nth run of algname

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

def listify(obj, wrap = True):
    if isinstance(obj, int) or \
       isinstance(obj, bool) or \
       isinstance(obj, float) or \
       isinstance(obj, complex) or \
       isinstance(obj, str):
         return [obj,] if wrap else obj
    elif isinstance(obj,np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict): #to handle ordered and unordered dicts
        return [{k:listify(v, False) for k,v in obj.items()},]
    elif isinstance(obj, set) or isinstance(obj, tuple) or isinstance(obj, list):
        return [listify(x, False) for x in obj]
    elif isinstance(obj,np.generic):
        return [obj.tolist(),] if wrap else obj.tolist()

def setup_logs(reward, obs, action, dones, info = None):
    data = {}
    # print(reward)
    # print(type(reward))
    # new_rewards = listify(reward)
    # print(new_rewards)
    # print(type(new_rewards))
    data['rewards'] = listify(reward)
    data["obs"]     = listify(obs)
    data["actions"] = listify(action)

    data["dones"] = dones
    # print(info)
    if info and "reward_info" in info[0]:
        data["infos"] = info[0]["reward_info"]
    else:
        data["infos"] = info
    return data

def load_episode_log(path):
    with open(path) as f:
        contents = json.load(f)
    logs = {'rewards': contents['rewards'], 'observations':contents['observations'], 'actions': contents['actions']}
    if 'info' in logs:
        logs['info'] = contents['info']
    return logs

def q_prod(q1, q2):
    q3 = np.array([0.0,0.0,0.0,0.0])
    q3[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    q3[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    q3[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    q3[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    return q3

def q_conj(q):
    q2 = -q.copy()
    q2[0] = q[0]
    return q2

def q_rotate(d, q):
    return q_prod(q_prod(q, np.append(0, d)), q_conj(q))[1:]

if __name__ == "__main__":

    # results = compute_rewards("DQNShort_old")['rewards']
    # print(results.keys())
    # DQNCartSHORT2 = results['DQNCartSHORT2']
    # DQNCartSHORT = results['DQNCartSHORT']
    # means, stds, maxes, mins = DQNCartSHORT['means'],DQNCartSHORT['stds'], DQNCartSHORT['max'], DQNCartSHORT['min']
    # print(np.min(mins))
    # print(np.max(maxes))

    # results = compute_observations("AntMazeSparse_2024-07-23_16-51-17")
    # np.set_printoptions(threshold=sys.maxsize)
    # labels = ["x velocity", "y velocity", "z velocity", "x angular velocty", "y angular velocity", "z angular velocity"]
    # for alg, stats in results.items():
    #     print(alg)
        
    #     for k,v in stats.items():
    #         print(k+":")
    #         for i, label in enumerate(labels):
    #             print(label+":",v[17+i])
    pass
