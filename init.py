import gymnasium as gym
from stable_baselines3 import PPO
# import dmc2gym

from RL.baselines import Baseline
# env = dmc2gym.make()
env = gym.make("AntMaze_UMazeDense-v4")
print("normal actions", env.action_space)
print("normal observations", env.observation_space)
test_env = gym.make("AntMaze_UMazeDense-v4",render_mode="rgb_array")
print("human actions", test_env.action_space)
print("human observations", test_env.observation_space)

# env = gym.make("PointMaze_UMazeDense-v3", render_mode="human")

# print(env.step())
# print(type(env.observation_space))
model = Baseline("PPO", env).get_model()
# model = PPO("MlpPolicy", env, verbose=1)
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
