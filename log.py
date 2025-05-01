import os, json
import numpy as np

#we will have a few different settings for logs: "none", "overwrite", "warn", and "timestamp"
#the first just does not save logs, the second will ALWAYS delete any folders under the name of the experiment prior to running, the third will refuse to run if there are folders, 
#and the fourth will prepend a timestamp in order to always create new folders each time it has run repeatedly. For serious experiments, I recommend "warn" or "timestamp".
#The default is "warn", since this will let you know that some configuration is required to get the behavior you want.
#For less serious experiments, testing, etc., I recommend "none" or "overwrite", depending on whether you are testing a capability that uses the logs or not.
class TrainingLogger():

    def __init__(self, log_dir = None, log_info = True, log_type = 'detailed'):

        self._log_info = log_info
        self._log_type = log_type
        if log_dir: #if not, should set_log_dir before calling on episode or on step.
            self.set_log_dir(log_dir) 
        
        self.reset()

    def reset(self):
        self.reset_episode()
        self.episode_count = 0
        self.step_count = 0

    def reset_episode(self):
        self.episode_rewards = []
        self.episode_observations = []
        self.episode_actions = []
        self.episode_info = []
        self.episode_step_count = 0

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir
        self.summary_file = os.path.join(log_dir, 'stats.txt')
        self.train_episodes_dir = os.path.join(log_dir, 'train_episodes')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.train_episodes_dir, exist_ok=True)
        print("log directory set to ", log_dir)

        
    def on_episode(self) -> None:
        self.episode_count += 1

        trajectory = {
            "rewards": self.episode_rewards,
            "observations": self.episode_observations,
            "actions": self.episode_actions,
            "info": self.episode_info
        }
        if self._log_type == 'detailed':
            with open(os.path.join(self.train_episodes_dir, f"episode_{self.episode_count}.json"), 'w') as f:
                json.dump(trajectory, f)

        # Calculate high-level summary
        # print(self.episode_rewards)
        total_reward = sum(self.episode_rewards)
        num_steps = len(self.episode_rewards)

        # print(total_reward)
        # avg_reward = np.mean(self.episode_rewards)
        if self._log_info:
            goal = self.episode_info[0]["desired_velocity"]
            base_reward = sum([item["base"] for item in self.episode_info])
            healthy_bonus = sum([item["healthy_bonus"] for item in self.episode_info])
            control_cost = sum([item["control cost"] for item in self.episode_info])
            contact_cost = sum([item["contact cost"] for item in self.episode_info])
            # Write summary to a file
            with open(self.summary_file, 'a') as f:
                f.write(f"episode_{self.episode_count}: "
                        f"Total Reward = {total_reward}, "
                        f"Total Base = {base_reward}, "
                        f"Total Healthy = {healthy_bonus}, "
                        f"Total Control = {control_cost}, "
                        f"Total Contact = {contact_cost}, "
                        f"Goal = {goal}, Steps = {num_steps},\n")
        else:
             with open(self.summary_file, 'a') as f:
                f.write(f"episode_{self.episode_count}: "
                        f"Total Reward = {total_reward}, "
                        f"Steps = {num_steps},\n")
        self.reset_episode()

    def on_step(self, data) -> bool:
        # Check that the `dones` local variable is defined
        assert "dones" in data #used for episode termination in the logger
        self.step_count += 1
        self.episode_step_count += 1
        self.episode_rewards.extend(data['rewards'])
        # print(self.locals['new_obs'])
        self.episode_observations.extend(data['obs'])
        self.episode_actions.extend(data['actions'])
        if self._log_info:
            self.episode_info.append(data['infos'])
        if np.sum(data["dones"]).item() > 0:
            self.on_episode()
