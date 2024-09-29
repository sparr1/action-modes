import gymnasium as gym
import numpy as np
from RL.baselines import Baseline
from RL.alg import Random, Stationary
from domains.tasks import Subtask
from domains.AntMaze import Move, Rotate
domains = \
{"ant": "Ant-v4",
"ant_dense": "AntMaze_UMazeDense-v4",
"ant_sparse": "AntMaze_UMaze-v4",
"point_dense": "PointMaze_UMazeDense-v3",
"point_sparse": "PointMaze_UMaze-v3"}

selected_domain = domains["ant"]
max_episode_steps = 1000 #TODO: figure out why we're rendering 5x the number of episode step frames
categorical = True
des_vel_min = 1.0
des_vel_max = 4.0
survival_bonus = 1.0
loss_at_margin = 1/3
margin = 0.5
metric = "huber"
render_train = True
render_test = True
direction = "X"
if render_train:
    train_mode = "human"
else:
    train_mode = None
if render_test:
    test_mode = "human"
else:
    test_mode = None
train_env = gym.make(selected_domain,exclude_current_positions_from_observation=False, max_episode_steps=max_episode_steps, render_mode = train_mode) #do not render training steps. god
objective = Move(desired_velocity_minimum=des_vel_min, desired_velocity_maximum=des_vel_max, survival_bonus=survival_bonus, direction=direction,loss_at_margin=loss_at_margin, margin=margin, categorical = categorical, metric=metric)
train_env = Subtask(train_env, objective)
print(train_env.observation_space)
# params={"learning_rate":3e-4, "gamma":1.0}
model = Baseline("SAC", train_env)
# # model = Random("random", train_env)
# # model = Stationary("stationary", train_env)
# model = model.load("logs/AntPlaneRotate_2024-09-18_12-23-13/models/model:AntPPO_0")
# model.load("logs/AntPlaneMove_2024-09-12_20-21-21/models/model:AntSAC_4")

model.learn(total_timesteps=100000)
# model.save("./", "test")
# vec_env = model.get_env()
train_env.reset()
train_env.close()
#switch to test_env
test_env = gym.make(selected_domain, exclude_current_positions_from_observation=False, max_episode_steps=max_episode_steps, render_mode=test_mode) #please render the test steps!
objective = Move(desired_velocity_minimum=des_vel_min, desired_velocity_maximum=des_vel_max, survival_bonus=survival_bonus, direction=direction,loss_at_margin=loss_at_margin, margin=margin, categorical = categorical, metric=metric)
test_env = Subtask(test_env, objective)
# model
# model = Baseline("SAC", test_env)
# model.load("logs/AntPlaneMoveNew_2024-09-27_12-18-34/models/model:AntSAC_0")
observation, info = test_env.reset(seed=42)
# desired_vel = info['reward_info']['desired_velocity']
# labels = ["x velocity", "y velocity", "z velocity", "x angular velocty", "y angular velocity", "z angular velocity"]
for _ in range(100000):
   action, _states = model.predict(observation)
   print(np.sum(action))
   observation, reward, terminated, truncated, info = test_env.step(action)
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
    #    desired_vel = info['reward_info']['desired_velocity']
       print("EPISODE RESET")
test_env.close()
