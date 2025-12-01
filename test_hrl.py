from modes.tasks import Subtask
from domains.Maze import Move
from RL.baselines import Baseline
from domains.AntPlane import AntPlane
from RL.PAMDP import RandomPAMDP, PAMDP
import gymnasium as gym
import gymnasium_goal
import gymnasium_platform
from domains.mpqdn_wrappers import *
from domains.mpqdn_ant_domain import *

from RL.modes import *
from modes.controller import *
from modes.tasks import *

#list of Maze Environments:
# Ant-v4
# AntMaze_UMazeDense-v4
# AntMaze_UMaze-v4
#
# PointMaze_UMaze-v4
#

if __name__=="__main__":

    train_env_params = {
                        "id": "AntMaze_UMazeDense-v4",
                        # "exclude_current_positions_from_observation":False,
                        "max_episode_steps":600,
                        "render_mode": "human",
                        "maze_map": [
                                [1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1],
                                ]
                        }
    
    env = gym.make(**train_env_params)
    env = AntPlane(env, random_resets= True,
                            reset_map = [[0, 0, 0, 0, 0],
                                         [0, 1, 1, 1, 0],
                                         [0, 1, 1, 1, 0],
                                         [0, 1, 1, 1, 0],
                                         [0, 0, 0, 0, 0]],                
        map_scale = 4,
        position_noise_range = 0.25)
    # print("test")
    # # print(gym.envs.registry.keys())
    # env = gym.make("Platform-v0")
    # move = Move()
    # # rotate = Rotate(desired_velocity_maximum = -1.0, desired_velocity_minimum = -1.0)
    # model = RandomPAMDP("rand", env)

#     reset_options={"goal_cell": np.array((3,1)), "reset_cell": np.array((1, 1))}

#     run_params = {"reset_options": reset_options}
    # model = PAMDP("MP_DQN", env, params=params)

    modal_alg_params = { 
                         "num_modes": 2,
                         "orchestrator_config": "configs/controllers/AntOrchestrator.json",
                         "mode_configs": ["configs/controllers/AntWalker.json",
                                          "configs/controllers/AntRotator.json"]
                        }
    
    model = ModalAlg("modes", env, **modal_alg_params)

    # maze = gym.make("Ant-v4",exclude_current_positions_from_observation=False, max_episode_steps=1000)
    # env = Subtask(maze, move)
    # baseline = Baseline("SAC", env) #does not yet support baseline.learn() etc. 
    # model = baseline.get_model() #extract the actual stable baselines model from the Baseline object

#     model.learn(total_timesteps=4000000, run_params = run_params)
    model.learn(total_timesteps=4000000)
    # model.save()
    # test_env = gym.make(**train_env_params)
    # test_env = AntPlane(test_env)

    # observation, info = test_env.reset(options = reset_options)
    # for _ in range(500000):
    #     # print(observation)
    #     action, _states = model.predict(observation)
    #     observation, reward, terminated, truncated, info = test_env.step(action)
    #     print(action, observation, reward)
    #     # test_env.render()
    #     if terminated or truncated:
    #         observation, info = test_env.reset(options = reset_options)
    # env.close()