import gymnasium as gym
import numpy as np
from RL.baselines import Baseline
from RL.alg import Random, Stationary
from domains.tasks import Subtask
from domains.AntMaze import Move, Rotate
from domains.AntPlane import AntPlane
domains = {
            "ant_plane": "ant_plane",
            "ant": "Ant-v4",
            "ant_dense": "AntMaze_UMazeDense-v4",
            "ant_sparse": "AntMaze_UMaze-v4",
            "point_dense": "PointMaze_UMazeDense-v3",
            "point_sparse": "PointMaze_UMaze-v3",
            "humanoid_plane": "humanoid_plane",
            "humanoid": "Humanoid-v4"
}

train_env_params = {
                    "id": domains["ant"],
                    # "exclude_current_positions_from_observation":False,
                    "max_episode_steps":600,
                    "render_mode": "human"
                    }

train_objective_params = {
                    "direction": "F",
                    "desired_velocity_minimum":-2.0,
                    "desired_velocity_maximum": 2.0,
                    "survival_bonus": 1.0,
                    "adaptive_margin":True,
                    "adaptive_margin_minimum":0.01,
                    "categorical":False,
                    "metric": "L2"
                    }

test_env_params = train_env_params.copy()
test_objective_params = train_objective_params.copy()
test_env_params["render_mode"] = "human"

if train_env_params["id"] == "ant_plane":
    train_env_params["id"] = domains["ant"]
    train_base_domain = AntPlane(gym.make(**train_env_params))
elif train_env_params["id"] == "humanoid_plane":
    train_env_params["id"] = domains["humanoid"]
    train_base_domain = AntPlane(gym.make(**train_env_params))
else:
    train_base_domain = gym.make(**train_env_params)
    

train_env = Subtask(train_base_domain, Move(**train_objective_params))


print(train_env.observation_space)
params={"learning_starts":int(1e4), "buffer_size":int(1e6)}
model_move = Baseline("SAC", train_env, params = params)
model = Random("random", train_env)
# # model = Stationary("stationary", train_env)
# model_move.load("logs/AntPlaneMoveFinal_2024-11-13_10-22-56/models/model:AntSAC_1")

model_rotate = Baseline("SAC", train_env, params = params)
# model_rotate.load("logs/AntPlaneRotateNew_2024-10-10_16-22-19/models/model:AntSAC_0")
# model.load("logs/AntPlaneRotateNew_2024-10-09_12-24-24/models/model:AntSAC_1")
# model.load("logs/AntPlaneMoveNew6.0_2024-10-03_09-30-15/models/model:AntSAC_2")
# model.load("logs/AntPlaneMove5_2024-10-11_21-28-31/models/model:AntSAC_0")

# model.load("logs/AntPlaneRotateNew_2024-10-10_16-22-19/models/model:AntSAC_0")

model.learn(total_timesteps=1500000)
# model.save("./", "test")
# vec_env = model.get_env()
train_env.reset()
train_env.close()
#switch to test_env
if test_env_params["id"] == "ant_plane":
    test_env_params["id"] = domains["ant"]
    test_base_domain = AntPlane(gym.make(**test_env_params))
elif train_env_params["id"] == "humanoid_plane":
    train_env_params["id"] = domains["humanoid"]
    train_base_domain = AntPlane(gym.make(**train_env_params))
    test_base_domain = gym.make(**test_env_params)

# test_base_domain = gym.make(**test_env_params)
test_env = Subtask(test_base_domain, Move(**test_objective_params))
# model
# model = Baseline("SAC", test_env)
# model.load("logs/AntPlaneMove2_2024-10-01_13-06-09/models/model:AntSAC_0")
observation, info = test_env.reset(seed=42)
# desired_vel = info['reward_info']['desired_velocity']
# labels = ["x velocity", "y velocity", "z velocity", "x angular velocty", "y angular velocity", "z angular velocity"]
ep_step_count = 0
for _ in range(150000):
    if ep_step_count % 2 == 0:
        model = model_rotate
    else:
        model = model_move
    action, _states = model.predict(observation)
    print(np.sum(action))
    observation, reward, terminated, truncated, info = test_env.step(action)
    ep_step_count+=1
    print(observation)
    desired_vel = info['reward_info']['desired_velocity']
    achieved_vel = info['reward_info']['achieved_velocity']
    print('----------')
    #    print("info", info)
    print('achieved velocity: ',achieved_vel, 'desired velocity: ', desired_vel)
    #    print("target velocity:", test_env._task.desired_velocity)
    #    for i,label in enumerate(labels):
    #        print(label+":", observation["observation"][13+i])
    #    vec_env.render() not needed for render_mode = human
    print('----------')
    if terminated or truncated:
        observation, info = test_env.reset()
        ep_step_count = 0
        #    desired_vel = info['reward_info']['desired_velocity']
        print("EPISODE RESET")
test_env.close()
