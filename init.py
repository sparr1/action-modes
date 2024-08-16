import gymnasium as gym

from RL.baselines import Baseline
from domains.tasks import Subtask
from domains.AntMaze import Move, Rotate
domains = \
{"ant_dense": "AntMaze_UMazeDense-v4",
"ant_sparse": "AntMaze_UMaze-v4",
"point_dense": "PointMaze_UMazeDense-v3",
"point_sparse": "PointMaze_UMaze-v3"}

selected_domain = domains["ant_sparse"]
max_episode_steps = 300 #TODO: figure out why we're rendering 5x the number of episode step frames
des_vel_min = 1.0
des_vel_max = 1.0
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
train_env = Subtask(train_env, Rotate(des_vel_min, des_vel_max))
# params={"learning_rate":3e-4, "gamma":.999}
model = Baseline("PPO", train_env,params={"learning_rate":3e-4, "gamma":.999}).get_model()

model.learn(total_timesteps=1000000)
model.save("test")
# vec_env = model.get_env()
train_env.reset()
train_env.close()
#switch to test_env
test_env = gym.make(selected_domain, max_episode_steps=max_episode_steps, render_mode=test_mode) #please render the test steps!
test_env = Subtask(test_env, Rotate(des_vel_min,des_vel_max))
observation, info = test_env.reset(seed=42)
labels = ["x velocity", "y velocity", "z velocity", "x angular velocty", "y angular velocity", "z angular velocity"]
for _ in range(100000):
   action, _states = model.predict(observation)
   observation, reward, terminated, truncated, info = test_env.step(action)
   print('----------')
   print("reward", reward)
   print("target velocity:", test_env._task.desired_velocity)
   for i,label in enumerate(labels):
       print(label+":", observation["observation"][13+i])
#    vec_env.render() not needed for render_mode = human
   print('----------')
   if terminated or truncated:
       observation, info = test_env.reset()
test_env.close()
