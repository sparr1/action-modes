import numpy as np
import gymnasium as gym
import importlib, json, os, ast, time
from stable_baselines3.common.callbacks import BaseCallback
from RL.alg import Algorithm, SimpleAlgorithm
from utils.utils import setup_logs
#from stable_baselines3 import PPO, DQN, TD3, SAC, DDPG, A2C
module_name = "mpdqn" #for dynamic importing

#TODO: loading in code from the MPDQN codebase 

class RandomPAMDP(SimpleAlgorithm):
    def __init__(self, name, env, custom_params = None):
        super().__init__(name, env, custom_params)

    def predict(self, observation):
        # print(self.env.action_space)
        sample = self.env.action_space.sample()
        return (sample[0],list(sample[1:])), None #annoying, but I think this needed to be here
    
class PAMDP(Algorithm):
    def __init__(self, name, env, params = None, custom_action_space = None):
        super().__init__(name, env, custom_params=params)
        self.model = self.get_PAMDP_model(self.name, self.env, self.custom_params, custom_action_space = custom_action_space)
        self.callback = None

    def learn(self, **kwargs): #only use if training a PAMDP algorithm non-hierarchically.
        if "total_timesteps" in kwargs:
            total_timesteps = kwargs["total_timesteps"]
        else:
            total_timesteps = -1

        save_freq = self.custom_params["save_freq"]
        save_dir = self.custom_params["save_dir"]
        visualise = self.custom_params["visualise"]
        render_freq = self.custom_params["render_freq"]
        save_frames = self.custom_params["save_frames"]
        title = self.custom_params["title"]
        seed = self.custom_params["seed"] #TODO check if this is the right place to get seed
        evaluation_episodes = self.custom_params["evaluation_episodes"]

        if save_freq > 0 and save_dir:
            save_dir = os.path.join(save_dir, title + "{}".format(str(seed)))
            os.makedirs(save_dir, exist_ok=True)
        assert not (save_frames and visualise)
        if visualise:
            assert render_freq > 0
        if save_frames:
            assert render_freq > 0
            vidir = os.path.join(save_dir, "frames")
            os.makedirs(vidir, exist_ok=True)

        max_steps = 250 #TODO this should come from Experiment JSON/environment settings
        total_reward = 0.
        returns = []
        start_time = time.time()
        video_index = 0
        # agent.epsilon_final = 0.
        # agent.epsilon = 0.
        # agent.noise = None
        env = self.env
        agent = self.model
        t_so_far = 0
        i = 0
        ret, info = env.reset()
        while t_so_far < total_timesteps:
            if save_freq > 0 and save_dir and i % save_freq == 0:
                self.model.save_models(os.path.join(save_dir, str(i)))
            
            ret, info = env.reset()
            # print(reset_ret)
            if isinstance(ret, tuple):
                state, _ = ret
            else:
                state = ret
            # state = np.array(state, dtype=np.float32, copy=False)
            state = np.asarray(state, dtype = np.float32)

            # if visualise and i % render_freq == 0:
            #     env.render()

            act, act_param, all_action_parameters = agent.act(state)
            action = self.pad_action(act, act_param)

            episode_reward = 0.
            agent.start_episode()

            for j in range(max_steps):
                t_so_far+=1
                ret = env.step(action)
                if(len(ret)==5):
                    (next_state, steps), reward, terminal, _, _ = ret
                elif(len(ret)==4):
                    (next_state, steps), reward, terminal, _ = ret

                # (next_state, steps), reward, terminal, _, _ = ret
                # next_state = np.array(next_state, dtype=np.float32, copy=False)
                next_state = np.asarray(state, dtype = np.float32)

                next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
                next_action = self.pad_action(next_act, next_act_param)
                agent.step(state, (act, all_action_parameters), reward, next_state,
                        (next_act, next_all_action_parameters), terminal, steps)
                act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
                action = next_action
                state = next_state

                episode_reward += reward
                if visualise and i % render_freq == 0:
                    env.render()

                if terminal:
                    break
            i+=1
            agent.end_episode()

            if save_frames and i % render_freq == 0:
                video_index = env.unwrapped.save_render_states(vidir, title, video_index)

            returns.append(episode_reward)
            total_reward += episode_reward
            if i % 100 == 0:
                print('{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(i), total_reward / (i + 1), np.array(returns[-100:]).mean()))
            
        end_time = time.time()
        print("Took %.2f seconds" % (end_time - start_time))
        env.close()
        if save_freq > 0 and save_dir:
            agent.save_models(os.path.join(save_dir, str(i)))

        returns = env.get_episode_rewards()
        print("Ave. return =", sum(returns) / len(returns))
        print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)

        np.save(os.path.join(dir, title + "{}".format(str(seed))),returns)

        if evaluation_episodes > 0:
            print("Evaluating agent over {} episodes".format(evaluation_episodes))
            agent.epsilon_final = 0.
            agent.epsilon = 0.
            agent.noise = None
            evaluation_returns = self.evaluate(env, agent, evaluation_episodes)
            print("Ave. evaluation return =", sum(evaluation_returns) / len(evaluation_returns))
            np.save(os.path.join(dir, title + "{}e".format(str(seed))), evaluation_returns)
    
    def evaluate(self, env, agent, episodes=1000):
        returns = []
        timesteps = []
        for _ in range(episodes):
            ret, info = env.reset()
            state, _ = ret
            # state, _ = env.reset()
            terminal = False
            t = 0
            total_reward = 0.
            while not terminal:
                t += 1
                # state = np.array(state, dtype=np.float32, copy=False)
                state = np.asarray(state, dtype = np.float32)
                act, act_param, all_action_parameters = agent.act(state)
                action = self.pad_action(act, act_param)
                (state, _), reward, terminal, _ = env.step(action)
                total_reward += reward
            timesteps.append(t)
            returns.append(total_reward)
        # return np.column_stack((returns, timesteps))
        return np.array(returns)

    def step(self, state, action, reward, next_state, next_action, terminal, steps):
        self.model.step(state, action, reward, next_state, next_action, terminal, steps)

    def start_episode(self):
        self.model.start_episode()
    
    def end_episode(self):
        self.model.end_episode()

    def predict(self, observation):
        return self.model.act(observation)
    
    def save(self, path, name):
        self.model.save(os.path.join(path, name))

    def get_model(self):
        return self.model

    # def set_logger(self, logger):
    #     super().set_logger(logger)
    #     self.callback = TrajectoryLoggerCallback(self.alg_logger)
        
    def delete_model(self): 
        del self.model

    def load(self, path):#requires retrieving the correct baseline model before loading weights
        self.model = self.model.load(path)


#
# from https://github.com/cycraig/MP-DQN/
#
    def pad_action(self, act, act_param):
        params = [np.zeros((2,)), np.zeros((1,)), np.zeros((1,))]
        params[act] = act_param
        return (act, params)

    def get_PAMDP_model(self, name, env, params = None, custom_action_space = None):
        p = {}
        env_name = env.unwrapped.spec.id
        agent = None

        if 'Goal' in env_name:
            from gym_goal.envs.config import GOAL_WIDTH, PITCH_LENGTH, PITCH_WIDTH


            # if params['scale_actions']:
            #     kickto_weights = np.array([[-0.375, 0.5, 0, 0.0625, 0],
            #                             [0, 0, 0.8333333333333333333, 0, 0.111111111111111111111111]])
            #     shoot_goal_left_weights = np.array([0.857346647646219686, 0])
            #     shoot_goal_right_weights = np.array([-0.857346647646219686, 0])
            # else:
            #     xfear = 50.0 / PITCH_LENGTH
            #     yfear = 50.0 / PITCH_WIDTH
            #     caution = 5.0 / PITCH_WIDTH
            #     kickto_weights = np.array([[2.5, 1, 0, xfear, 0], [0, 0, 1 - caution, 0, yfear]])
            #     shoot_goal_left_weights = np.array([GOAL_WIDTH / 2 - 1, 0])
            #     shoot_goal_right_weights = np.array([-GOAL_WIDTH / 2 + 1, 0])

            # initial_weights = np.zeros((4, 17))
            # initial_weights[0, [10, 11, 14, 15]] = kickto_weights[0, 1:]
            # initial_weights[1, [10, 11, 14, 15]] = kickto_weights[1, 1:]
            # initial_weights[2, 16] = shoot_goal_left_weights[1]
            # initial_weights[3, 16] = shoot_goal_right_weights[1]

            # initial_bias = np.zeros((4,))
            # initial_bias[0] = kickto_weights[0, 0]
            # initial_bias[1] = kickto_weights[1, 0]
            # initial_bias[2] = shoot_goal_left_weights[0]
            # initial_bias[3] = shoot_goal_right_weights[0]

            # #if this becomes necessary, we can use this
            # # partially_unwrapped_env = env #we will search for the relevant observation wrapper in the wrapper stack
            # # while (not issubclass(partially_unwrapped_env, gym.ObservationWrapper)) or hasattr(partially_unwrapped_env, "scale_state"):
            # #     partially_unwrapped_env = partially_unwrapped_env.env

            # if not params['scale_actions']:
            #     # rescale initial action-parameters for a scaled state space
            #     for a in range(env.action_space.spaces[0].n):
            #         mid = (env.observation_space.spaces[0].high + env.observation_space.spaces[0].low) / 2.
            #         initial_bias[a] += np.sum(initial_weights[a] * mid)
            #     initial_weights[a] = initial_weights[a]*env.observation_space.spaces[0].high - initial_weights[a] * mid

            # # env.seed(seed)
            # # np.random.seed(seed)

            # #END FIRST PART OF PDQN INITIALIZATION
            # assert not (params['split'] and params['multipass'])
            # name = "PDQNAgent"
            # final_module_name = module_name + ".agents.pdqn"
            # if params['split']:
            #     name = "SplitPDQNAgent"
            #     final_module_name += "_split"
            # elif params['multipass']:
            #     name = "MultiPassPDQNAgent"
            #     final_module_name += "_multipass"
            # try:
            #     module = importlib.import_module(module_name)
            #     agent_class = getattr(module,name) #grab the specific algorithm
            #     layers = ast.literal_eval(params['layers'])
            #     print(layers)
            #     agent = agent_class(
            #                     observation_space=env.observation_space.spaces[0], action_space=env.action_space,
            #                     batch_size=params['batch_size'],
            #                     learning_rate_actor=params['learning_rate_actor'],  # 0.0001
            #                     learning_rate_actor_param=params['learning_rate_actor_param'],  # 0.001
            #                     epsilon_steps=params['epsilon_steps'],
            #                     epsilon_final=params['epsilon_final'],
            #                     gamma=params['gamma'],
            #                     clip_grad=params['clip_grad'],
            #                     indexed=params['indexed'],
            #                     average=params['average'],
            #                     random_weighted=params['random_weighted'],
            #                     tau_actor=params['tau_actor'],
            #                     weighted=params['weighted'],
            #                     tau_actor_param=params['tau_actor_param'],
            #                     initial_memory_threshold=params['initial_memory_threshold'],
            #                     use_ornstein_noise=params['use_ornstein_noise'],
            #                     replay_memory_size=params['replay_memory_size'],
            #                     inverting_gradients=params['inverting_gradients'],
            #                     actor_kwargs={'hidden_layers': layers, 'output_layer_init_std': 1e-5, #dubious
            #                                     'action_input_layer': params['action_input_layer'],},
            #                     actor_param_kwargs={'hidden_layers': layers, 'output_layer_init_std': 1e-5, #dubious
            #                                         'squashing_function': False}, #dubious
            #                     zero_index_gradients=params['zero_index_gradients'],
            #                     seed=params['seed'])
            # except (ModuleNotFoundError, AttributeError) as e:
            #     raise ValueError(f"Could not find model class '{name}' in module '{module_name}': {e}")
        elif 'Platform' in env_name: #NOW DO PLATFORM

            initial_params_ = [3., 10., 400.]
            # unwrapped_env = env.unwrapped
            # if params['scale_actions']:
            #     for a in range(unwrapped_env.action_space.spaces[0].n): 
            #         initial_params_[a] = 2. * (initial_params_[a] - unwrapped_env.action_space.spaces[1].spaces[a].low) / (
            #                     unwrapped_env.action_space.spaces[1].spaces[a].high - unwrapped_env.action_space.spaces[1].spaces[a].low) - 1.

            # print("obs space", env.observation_space)
            # print("act space", env.action_space)

            # assert not (params['split'] and params['multipass'])
            # name = "PDQNAgent"
            # final_module_name = module_name + ".agents.pdqn"
            # if params['split']:
            #     name = "SplitPDQNAgent"
            #     final_module_name += "_split"
            # elif params['multipass']:
            #     name = "MultiPassPDQNAgent"
            #     final_module_name += "_multipass"
            # try:
            #     module = importlib.import_module(module_name)
            #     agent_class = getattr(module,name) #grab the specific algorithm
            #     layers = ast.literal_eval(params['layers'])
            #     print(layers)
            #     agent = agent_class(
            #                     observation_space=env.observation_space.spaces[0], action_space=env.action_space,
            #                     batch_size=params['batch_size'],
            #                     learning_rate_actor=params['learning_rate_actor'],  # 0.0001
            #                     learning_rate_actor_param=params['learning_rate_actor_param'],  # 0.001
            #                     epsilon_steps=params['epsilon_steps'],
            #                     epsilon_final=params['epsilon_final'],
            #                     gamma=params['gamma'],
            #                     clip_grad=params['clip_grad'],
            #                     indexed=params['indexed'],
            #                     average=params['average'],
            #                     random_weighted=params['random_weighted'],
            #                     tau_actor=params['tau_actor'],
            #                     weighted=params['weighted'],
            #                     tau_actor_param=params['tau_actor_param'],
            #                     initial_memory_threshold=params['initial_memory_threshold'],
            #                     use_ornstein_noise=params['use_ornstein_noise'],
            #                     replay_memory_size=params['replay_memory_size'],
            #                     inverting_gradients=params['inverting_gradients'],
            #                     actor_kwargs={'hidden_layers': layers, #'output_layer_init_std': 1e-5, #dubious
            #                                     'action_input_layer': params['action_input_layer'],},
            #                     actor_param_kwargs={'hidden_layers': layers, 'output_layer_init_std': 1e-4, #dubious
            #                                         'squashing_function': False}, #dubious
            #                     zero_index_gradients=params['zero_index_gradients'],
            #                     seed=params['seed'])
            # except (ModuleNotFoundError, AttributeError) as e:
            #     raise ValueError(f"Could not find model class '{name}' in module '{module_name}': {e}")

            # if params['initialise_params']:
            #     initial_weights = np.zeros((env.action_space.spaces[0].n, env.observation_space.spaces[0].shape[0]))
            #     initial_bias = np.zeros(env.action_space.spaces[0].n)
            #     for a in range(env.action_space.spaces[0].n):
            #         initial_bias[a] = initial_params_[a]
            #     agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)
            # print(agent)
        elif 'Ant' in env_name: 
            initial_params_ = [0., 0.]
            # unwrapped_env = env.unwrapped
            # action_space = custom_action_space #The ant environment only provides the base action space. so we will always be using custom actions

            print("obs space", env.observation_space)
            print("act space", env.action_space)

            assert not (params['split'] and params['multipass'])
            name = "PDQNAgent"
            final_module_name = module_name + ".agents.pdqn"
            if params['split']:
                name = "SplitPDQNAgent"
                final_module_name += "_split"
            elif params['multipass']:
                name = "MultiPassPDQNAgent"
                final_module_name += "_multipass"
            try:
                module = importlib.import_module(final_module_name)
                agent_class = getattr(module,name) #grab the specific algorithm
                layers = ast.literal_eval(params['layers'])
                print(layers)
                agent = agent_class(
                                env.observation_space[0], env.action_space,
                                batch_size=params['batch_size'],
                                learning_rate_actor=params['learning_rate_actor'],  # 0.0001
                                learning_rate_actor_param=params['learning_rate_actor_param'],  # 0.001
                                epsilon_steps=params['epsilon_steps'],
                                epsilon_final=params['epsilon_final'],
                                gamma=params['gamma'],
                                clip_grad=params['clip_grad'],
                                indexed=params['indexed'],
                                average=params['average'],
                                random_weighted=params['random_weighted'],
                                tau_actor=params['tau_actor'],
                                weighted=params['weighted'],
                                tau_actor_param=params['tau_actor_param'],
                                initial_memory_threshold=params['initial_memory_threshold'],
                                use_ornstein_noise=params['use_ornstein_noise'],
                                replay_memory_size=params['replay_memory_size'],
                                inverting_gradients=params['inverting_gradients'],
                                actor_kwargs={'hidden_layers': layers, 'output_layer_init_std': 0.5, #dubious
                                                'action_input_layer': params['action_input_layer'],},
                                actor_param_kwargs={'hidden_layers': layers, 'output_layer_init_std': 2e-3, #dubious
                                                    'squashing_function': False}, #dubious
                                zero_index_gradients=params['zero_index_gradients'],
                                seed=params['seed'])
            except (ModuleNotFoundError, AttributeError) as e:
                raise ValueError(f"Could not find model class '{name}' in module '{module_name}': {e}")

            if params['initialise_params']:
                initial_weights = np.zeros((env.action_space.spaces[0].n, env.observation_space.spaces[0].shape[0]))
                initial_bias = np.zeros(env.action_space.spaces[0].n)
                for a in range(env.action_space.spaces[0].n):
                    initial_bias[a] = initial_params_[a]
                agent.set_action_parameter_passthrough_weights(initial_weights, initial_bias)
            print(agent)

        return agent
