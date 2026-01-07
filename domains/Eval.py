import gymnasium as gym
import AntPlane
import numpy as np
#I need to implement an evaluation function where all random goals and starts are sampled twice.
def MazeEvaluation(maze_map, model, max_episode_steps, n=2):
    maze_map = [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
                ]
    eval_env_params = {
                        "id": "AntMaze_UMaze-v4",
                        # "exclude_current_positions_from_observation":False,
                        "max_episode_steps":max_episode_steps,
                        "render_mode": "human",
                        "maze_map": maze_map
                        }
    
    env = gym.make(**eval_env_params)
    env = AntPlane(env, random_resets= True,
                reset_map =    [[0, 0, 0, 0, 0],
                                [0, 1, 1, 1, 0],
                                [0, 0, 0, 1, 0],
                                [0, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0]],                
    map_scale = 4,
    position_noise_range = 0.25)

    empty_coords = [(i,j) for i in range(len(maze_map)) for j in range(maze_map[i]) if maze_map[i][j]==0]
    score = 0
    total_score_possible = 0
    for i in range(n):
        for s in empty_coords:
            for g in empty_coords:
                total_score_possible+=1
                reset_options={"goal_cell": np.array(g), "reset_cell": np.array(s)}
                # run_params = {"reset_options": reset_options}
                        # test_env = gym.make(**train_env_params)
                # test_env = AntPlane(test_env)
                max_reward = 0
                observation, info = env.reset(options = reset_options)
                for _ in range(max_episode_steps):
                    # print(observation)
                    action, _states = model.predict(observation)
                    observation, reward, terminated, truncated, info = env.step(action)
                    if reward > max_reward:
                        score+=1
                        max_reward = reward
                        break #we can break early after goal has been reached
                    # print(action, observation, reward)
                    # test_env.render()
                    if terminated or truncated:
                        break
                env.close()
    return score/total_score_possible #out of n*len(empty_coords)**2


    
if __name__=="__main__":
    pass
