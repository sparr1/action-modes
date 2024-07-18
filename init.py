import gymnasium as gym
# from stable_baselines3 import PPO
# import dmc2gym

from RL.baselines import Baseline
# env = dmc2gym.make()

domains = \
{"ant_dense": "AntMaze_UMazeDense-v4",
"ant_sparse": "AntMaze_UMaze-v4",
"point_dense": "PointMaze_UMazeDense-v3",
"point_sparse": "PointMaze_UMaze-v3"}

selected_domain = domains["ant_sparse"]
max_episode_steps = 700 #TODO: figure out why we're rendering 5x the number of episode step frames
render_train = False
render_test = True
if render_train:
    train_mode = "human"
else:
    train_mode = None
if render_test:
    test_mode = "human"
else:
    test_mode = None
train_env = gym.make(selected_domain, max_episode_steps=max_episode_steps, render_mode = train_mode) #do not render training steps. god
model = Baseline("SAC", train_env).get_model()
# model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("test")
# vec_env = model.get_env()
train_env.reset()
train_env.close()
#switch to test_env
test_env = gym.make(selected_domain, max_episode_steps=max_episode_steps, render_mode=test_mode) #please render the test steps!

observation, info = test_env.reset(seed=42)
for _ in range(1000):
   action, _states = model.predict(observation)
   observation, reward, terminated, truncated, info = test_env.step(action)
#    vec_env.render() not needed for render_mode = human
   if terminated or truncated:
       observation, info = test_env.reset()
test_env.close()
