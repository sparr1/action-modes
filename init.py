import gymnasium as gym
import numpy as np
import traceback, glfw
from RL.baselines import Baseline
from RL.alg import Random, Stationary
from domains.ant_variable_legs import AntVariableLegsEnv
from domains.tasks import Subtask
from domains.Maze import Move
from domains.AntPlane import AntPlane
from domains.HumanoidPlane import HumanoidPlane
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
                    "id": domains["humanoid_plane"],
                    "exclude_current_positions_from_observation":False,
                    "max_episode_steps":400,
                    "render_mode": "human"
                    }

train_objective_params = {
                    "direction": "X",
                    "desired_velocity_minimum":-3.0,
                    "desired_velocity_maximum": 3.0,
                    "survival_bonus": 5.0,
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
elif train_env_params["id"] == "variable_ant":
    train_base_domain = AntVariableLegsEnv(exclude_current_positions_from_observation=False,num_legs = 6, contact_cost_weight=0.0, render_mode = "human")
elif train_env_params["id"] == "humanoid_plane":
    train_env_params["id"] = domains["humanoid"]
    train_base_domain = HumanoidPlane(gym.make(**train_env_params))
else:
    
    train_base_domain = gym.make(**train_env_params)
    
train_env = train_base_domain
train_env = Subtask(train_base_domain, Move(**train_objective_params))


print(train_env.observation_space)
print(train_env.action_space)
params={"learning_starts":int(1e4), "buffer_size":int(1e6)}
model = Baseline("SAC", train_env, params = params)
# model.set_checkpointing(10000, ".", name_prefix="init-test")
# model = Random("random", train_env)
# # model = Stationary("stationary", train_env)
#model.load("logs/AntPlaneMoveFinal_2024-11-13_10-22-56/models/model:AntSAC_1")
# model_move.load("logs/AntPlaneMoveFinal_2024-11-13_10-22-56/models/model:AntSAC_1")
#model_rotate = Baseline("SAC", train_env, params = params)
# model_rotate.load("logs/AntPlaneRotateNew_2024-10-10_16-22-19/models/model:AntSAC_0")
# model.load("logs/AntPlaneRotateNew_2024-10-09_12-24-24/models/model:AntSAC_1")
# model.load("logs/AntPlaneMoveNew6.0_2024-10-03_09-30-15/models/model:AntSAC_2")
# model.load("logs/AntPlaneMove5_2024-10-11_21-28-31/models/model:AntSAC_0")

# model.load("logs/AntPlaneRotateNew_2024-10-10_16-22-19/models/model:AntSAC_0")
# model.load("models/HumanoidMoveResets5xSR1.0/model:HumanoidSAC-B1M_0")
model.load("models/HumanoidMoveResets1.75xTR3.0/model:HumanoidSAC-B1M_0")
# model.load("models/HumanoidBasic/model:HumanoidSAC-B2M_0")
# model.load("model:HumanoidSAC-B1M_0")
# try:
# model.learn(total_timesteps=1500000)
# except Exception as e:
#     print("GLFW initialized? ", glfw._initialized)
#     traceback.print_exc()    # ← shows you the file & line triggering the error
#     raise

# model.learn(total_timesteps=1500000)
# model.save("./", "test")
# vec_env = model.get_env()
# train_env.reset()
# train_env.close()
#switch to test_env
# if test_env_params["id"] == "ant_plane":
#     test_env_params["id"] = domains["ant"]
#     test_base_domain = AntPlane(gym.make(**test_env_params))
# elif train_env_params["id"] == "humanoid_plane":
#     train_env_params["id"] = domains["humanoid"]
#     train_base_domain = AntPlane(gym.make(**train_env_params))
#     test_base_domain = gym.make(**test_env_params)

# # test_base_domain = gym.make(**test_env_params)
# test_env = Subtask(test_base_domain, Move(**test_objective_params))
# # model
# # model = Baseline("SAC", test_env)
# # model.load("logs/AntPlaneMove2_2024-10-01_13-06-09/models/model:AntSAC_0")
test_env = train_env
observation, info = test_env.reset(seed=42)
print(info)
# desired_vel = info['reward_info']['desired_velocity']
labels = ["x velocity", "y velocity", "z velocity", "x angular velocty", "y angular velocity", "z angular velocity"]
start_vel_coords, _ = train_env.get_wrapper_attr("get_task_info")()["velocity_coords"]
ep_step_count = 0
for _ in range(150000):
    # if ep_step_count % 2 == 0:
    #     model = model_rotate
    # else:
    #     model = model_move
    action, _states = model.predict(observation)
    print(np.sum(action))
    observation, reward, terminated, truncated, info = test_env.step(action)
    ep_step_count+=1
    print(observation)
    # desired_vel = info['reward_info']['desired_velocity']
    # achieved_vel = info['reward_info']['achieved_velocity']
    # print('----------')
    # print("info", info)
    # print('achieved velocity: ',achieved_vel, 'desired velocity: ', desired_vel)
    # print("target velocity:", test_env._task.desired_velocity)
    for i,label in enumerate(labels):
        print(label+":", observation["observation"][start_vel_coords+i])
        # print(label+":", observation[start_vel_ceoords+i])

    #    vec_env.render() not needed for render_mode = human
    print('----------')
    if terminated or truncated:
        observation, info = test_env.reset()
        ep_step_count = 0
        # desired_vel = info['reward_info']['desired_velocity']
        print("EPISODE RESET")
test_env.close()
