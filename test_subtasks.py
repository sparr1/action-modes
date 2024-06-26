from domains.tasks import Subtask
from domains.AntMaze import Move, Rotate
from RL.baselines import Baseline
import gymnasium as gym

#list of Maze Environments:
#
# AntMaze_UMazeDense-v4
# AntMaze_UMaze-v4
#
# PointMaze_UMaze-v4
#
#

if __name__=="__main__":
    print("test")
    Move(1.0)
    Rotate(-1.0)

    maze = gym.make("AntMaze_UMazeDense-v4",max_episode_steps=300)
    env = Subtask(maze, Rotate(-1))
    baseline = Baseline("PPO", env) #does not yet support baseline.learn() etc. 
    model = baseline.get_model() #extract the actual stable baselines model from the Baseline object

    model.learn(total_timesteps=1000000)
    vec_env = model.get_env()
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        vec_env.render()
        if terminated or truncated:
            observation, info = env.reset()
    env.close()
