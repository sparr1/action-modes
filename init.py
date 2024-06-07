import gymnasium as gym
from stable_baselines3 import PPO
from RL.baselines import get_baseline_model
env = gym.make("AntMaze_UMazeDense-v4", render_mode="human")
# env = gym.make("PointMaze_UMazeDense-v3", render_mode="human")
print(type(env.observation_space))
model = get_baseline_model("PPO", env)
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
