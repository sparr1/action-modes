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

from modes.modes import *
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
                        "max_episode_steps":200,
                        "render_mode": "human",
                        "maze_map": [
                                [1, 1, 1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 0, 0, 1],
                                [1, 0, 'R', 0, 'G', 0, 1],
                                [1, 0, 0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1],
                                ]
                        }
    
    env = gym.make(**train_env_params)
    env = AntPlane(env)
    # print("test")
    # # print(gym.envs.registry.keys())
    # env = gym.make("Platform-v0")
    # move = Move()
    # # rotate = Rotate(desired_velocity_maximum = -1.0, desired_velocity_minimum = -1.0)
    # model = RandomPAMDP("rand", env)

    reset_options={"goal_cell": np.array((2,2)), "reset_cell": np.array((2, 3))}

    run_params = {"reset_options": reset_options}
    # model = PAMDP("MP_DQN", env, params=params)

    num_modes = 2

    cont_action_spaces = [gym.spaces.Box(low=-1.0, high=1.0, shape = (1,), dtype=np.float32) for i in range(num_modes)]
    orchestrator_action_space =  gym.spaces.Tuple([gym.spaces.Discrete(num_modes), gym.spaces.Tuple(cont_action_spaces)])

    orchestrator_config = "configs/controllers/AntOrchestrator.json"
    rotator_config = "configs/controllers/AntRotator.json"
    walker_config = "configs/controllers/AntWalker.json"

    model = ModalAlg("modes", env, orchestrator_config, orchestrator_action_space, [rotator_config, walker_config],cont_action_spaces)

    # maze = gym.make("Ant-v4",exclude_current_positions_from_observation=False, max_episode_steps=1000)
    # env = Subtask(maze, move)
    # baseline = Baseline("SAC", env) #does not yet support baseline.learn() etc. 
    # model = baseline.get_model() #extract the actual stable baselines model from the Baseline object

    model.learn(total_timesteps=4000000, run_params = run_params)
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
