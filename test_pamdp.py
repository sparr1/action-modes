import gymnasium as gym
import numpy as np
# from RL.baselines import Baseline
# from RL.alg import Random, Stationary
# from domains.tasks import Subtask
# from domains.AntMaze import Move, Rotate
# from domains.AntPlane import AntPlane
from domains.mpqdn_platform_domain import *
from domains.mpqdn_wrappers import *
from RL.PAMDP import PAMDP

import gymnasium_platform
# domains = {
#             "ant_plane": "ant_plane",
#             "ant": "Ant-v4",
#             "ant_dense": "AntMaze_UMazeDense-v4",
#             "ant_sparse": "AntMaze_UMaze-v4",
#             "point_dense": "PointMaze_UMazeDense-v3",
#             "point_sparse": "PointMaze_UMaze-v3"
# }

# train_env_params = {
#                     "id": domains["ant_sparse"],
#                     # "exclude_current_positions_from_observation":False,
#                     "max_episode_steps":600,
#                     "render_mode": "human"
#                     }

# train_objective_params = {
#                     "direction": "F",
#                     "desired_velocity_minimum":-2.0,
#                     "desired_velocity_maximum": 2.0,
#                     "survival_bonus": 1.0,
#                     "adaptive_margin":True,
#                     "adaptive_margin_minimum":0.01,
#                     "categorical":False,
#                     "metric": "L2"
#                     }

# test_env_params = train_env_params.copy()
# test_objective_params = train_objective_params.copy()
# test_env_params["render_mode"] = "human"

# if train_env_params["id"] == "ant_plane":
#     train_env_params["id"] = domains["ant"]
#     train_base_domain = AntPlane(gym.make(**train_env_params))
# else:
#     train_base_domain = gym.make(**train_env_params)
    

# train_env = Subtask(train_base_domain, Move(**train_objective_params))



params = {'seed': 1, 
          'evaluation_episodes': 1000,
          'batch_size': 128,
          'gamma': 0.9,
          'inverting_gradients': True,
          'initial_memory_threshold': 500,
          'use_ornstein_noise': True,
          'replay_memory_size': 10000,
          'epsilon_steps': 1000,
          'epsilon_final': 0.01,
          'tau_actor': 0.1,
          'tau_actor_param':0.001,
          'learning_rate_actor':0.001,
          'learning_rate_actor_param':0.0001,
          'scale_actions': True,
          'initialise_params': True,
          'clip_grad': 10.0,
          'split': False,
          'multipass': True,
          'indexed': False,
          'weighted': False,
          'average': False,
          'random_weighted': False,
          'zero_index_gradients': False,
          'action_input_layer': 0,
          'layers': '[128,]',
          'save_freq': 0,
          'save_dir': "results/platform",
          'render_freq': 100,
          'save_frames': False,
          'visualise': True,
          'title': 'PDDQN'
          }


# @click.option('--seed', default=1, help='Random seed.', type=int)
# @click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
# @click.option('--episodes', default=20000, help='Number of epsiodes.', type=int)
# @click.option('--batch-size', default=128, help='Minibatch size.', type=int)
# @click.option('--gamma', default=0.9, help='Discount factor.', type=float)
# @click.option('--inverting-gradients', default=True,
#               help='Use inverting gradients scheme instead of squashing function.', type=bool)
# @click.option('--initial-memory-threshold', default=500, help='Number of transitions required to start learning.',
#               type=int)  # may have been running with 500??
# @click.option('--use-ornstein-noise', default=True,
#               help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
# @click.option('--replay-memory-size', default=10000, help='Replay memory size in transitions.', type=int)
# @click.option('--epsilon-steps', default=1000, help='Number of episodes over which to linearly anneal epsilon.', type=int)
# @click.option('--epsilon-final', default=0.01, help='Final epsilon value.', type=float)
# @click.option('--tau-actor', default=0.1, help='Soft target network update averaging factor.', type=float)
# @click.option('--tau-actor-param', default=0.001, help='Soft target network update averaging factor.', type=float)  # 0.001
# @click.option('--learning-rate-actor', default=0.001, help="Actor network learning rate.", type=float) # 0.001/0.0001 learns faster but tableaus faster too
# @click.option('--learning-rate-actor-param', default=0.0001, help="Critic network learning rate.", type=float)  # 0.00001
# @click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
# @click.option('--initialise-params', default=True, help='Initialise action parameters.', type=bool)
# @click.option('--clip-grad', default=10., help="Parameter gradient clipping limit.", type=float)
# @click.option('--split', default=False, help='Separate action-parameter inputs.', type=bool)
# @click.option('--multipass', default=True, help='Separate action-parameter inputs using multiple Q-network passes.', type=bool)
# @click.option('--indexed', default=False, help='Indexed loss function.', type=bool)
# @click.option('--weighted', default=False, help='Naive weighted loss function.', type=bool)
# @click.option('--average', default=False, help='Average weighted loss function.', type=bool)
# @click.option('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
# @click.option('--zero-index-gradients', default=False, help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.", type=bool)
# @click.option('--action-input-layer', default=0, help='Which layer to input action parameters.', type=int)
# @click.option('--layers', default='[128,]', help='Duplicate action-parameter inputs.', cls=ClickPythonLiteralOption)
# @click.option('--save-freq', default=0, help='How often to save models (0 = never).', type=int)
# @click.option('--save-dir', default="results/platform", help='Output directory.', type=str)
# @click.option('--render-freq', default=100, help='How often to render / save frames of an episode.', type=int)
# @click.option('--save-frames', default=False, help="Save render frames from the environment. Incompatible with visualise.", type=bool)
# @click.option('--visualise', default=True, help="Render game states. Incompatible with save-frames.", type=bool)
# @click.option('--title', default="PDDQN", help="Prefix of output files", type=str)

env = gym.make('Platform-v0')
env = ScaledStateWrapper(env)
env = PlatformFlattenedActionWrapper(env)
env = ScaledParameterisedActionWrapper(env)
train_env = env

print(train_env.observation_space)
# params={"learning_starts":int(1e4), "buffer_size":int(1e6)}
model_platform = PAMDP("MPDQN", train_env, params=params)
model = model_platform
# model_move = Baseline("SAC", train_env, params = params)
# model = Random("random", train_env)
# # model = Stationary("stationary", train_env)
# model_move.load("logs/AntPlaneMoveFinal_2024-11-13_10-22-56/models/model:AntSAC_1")

# model_rotate = Baseline("SAC", train_env, params = params)
# model_rotate.load("logs/AntPlaneRotateNew_2024-10-10_16-22-19/models/model:AntSAC_0")
# model.load("logs/AntPlaneRotateNew_2024-10-09_12-24-24/models/model:AntSAC_1")
# model.load("logs/AntPlaneMoveNew6.0_2024-10-03_09-30-15/models/model:AntSAC_2")
# model.load("logs/AntPlaneMove5_2024-10-11_21-28-31/models/model:AntSAC_0")

# model.load("logs/AntPlaneRotateNew_2024-10-10_16-22-19/models/model:AntSAC_0")

model.learn(total_timesteps=10000)
# model.save("./", "test")
# vec_env = model.get_env()
train_env.reset()
train_env.close()
#switch to test_env
# if test_env_params["id"] == "ant_plane":
#     test_env_params["id"] = domains["ant"]
#     test_base_domain = AntPlane(gym.make(**test_env_params))
# else:
#     test_base_domain = gym.make(**test_env_params)
# test_base_domain = gym.make(**test_env_params)
# test_env = Subtask(test_base_domain, Move(**test_objective_params))
# model
# model = Baseline("SAC", test_env)
# model.load("logs/AntPlaneMove2_2024-10-01_13-06-09/models/model:AntSAC_0")
observation, info = test_env.reset(seed=42)
# desired_vel = info['reward_info']['desired_velocity']
# labels = ["x velocity", "y velocity", "z velocity", "x angular velocty", "y angular velocity", "z angular velocity"]
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
