import gymnasium as gym
import numpy as np
from RL.alg import Algorithm
from modes.controller import OrchestralController, ModalController
import os, time, math
import random
#we will take an initial task with an action space of Box(-1, 1, (8,), float32), 
#combine it with a few (initially just 2) different lower dimensional controllers action space of Box(-1, 1, (1,), float32),
#and then the new action space will be Dict(Discrete(2), Box(-1, 1, (1,), float32), Box(-1, 1, (1,), float32)), where the discrete action just switches which lower dimensional action space we use.
#In general, we will have M possible modes, which we will winnow into k supported modes for any given state, then project the latent "within-mode" actions to the original high-dimensional state space.

#for now, we will forget about the initial winnowing, and just assume the support of each mode covers the state space. 

#one wrinkle is that the mode_controllers expect the environment to be wrapped with a task wrapper. 
class ModalAlg(Algorithm):
    def __init__(self, name, env, orchestrator_config, orchestrator_action_space, mode_configs, mode_action_spaces):
        super().__init__(name, env)
        self.base_env = self.env.unwrapped
        #each mode_controller is going to turn a latent "action-within-mode" z into a base action in env. 
        self.num_modes = len(mode_configs)
        #need to set up a discrete action space corresponding to the controllers
        # self.base_action_space = self.action_space
        self.orchestrator = OrchestralController(orchestrator_config, self.env, orchestrator_action_space)
        self.controllers  = [ModalController(c, self.env, mode_action_spaces[i]) for i,c in enumerate(mode_configs)]
        self.orchestrator.add_controllers(self.controllers)
    
    
    def predict(self, observation, modulus = 0, old_orch_act = None):
        if modulus != 0 and old_orch_act:
            orchestral_action = old_orch_act 
            # print(orchestral_action)
            orch_states = None
        else:
            orch_obs = self.orchestrator.process_obs(observation)
            orchestral_action, orch_states = self.orchestrator.predict(orch_obs)
            # print(orchestral_action)
            orch_action_space = self.orchestrator.domain.action_space
            # print(orchestral_action)
            # orchestral_action = orch_action_space.sample()
            cont_action = np.concatenate(orchestral_action[1:])
            cont_action = np.clip(cont_action, a_min = -2.0, a_max = 2.0)
            orchestral_action = orchestral_action[0], cont_action
            # orchestral_action[1] = orchestral_action[1]*5.0
            # orchestral_action[2] = orchestral_action[2]*5.0
            # print(orchestral_action)
        # print(orchestral_action)
        ind, params = orchestral_action
        controller_action_param = params[ind]
        # print(controller_action_param)
        selected_controller = self.controllers[ind]
        # print(selected_controller.name)
        contr_obs = selected_controller.process_obs(observation, controller_action_param)
        controller_action, contr_states = selected_controller.predict(contr_obs)

        return controller_action, {
            "orchestral_action": orchestral_action, 
            "orchestral_states": orch_states,
            "controller_states": contr_states}
    

    def learn(self, **kwargs): #for now, this is just learning the orchestrator given pretrained controllers
        if "total_timesteps" in kwargs:
            total_timesteps = kwargs["total_timesteps"]
        else:
            total_timesteps = -1
        reset_options = {}
        if "run_params" in kwargs:
            run_params = kwargs["run_params"]
            if "reset_options" in run_params:
                reset_options = run_params["reset_options"]
        else:
            run_params = {}

        custom_params = self.orchestrator.model.custom_params
        save_freq = custom_params["save_freq"]
        save_dir = custom_params["save_dir"]
        visualise = custom_params["visualise"]
        render_freq = custom_params["render_freq"]
        save_frames = custom_params["save_frames"]
        title = custom_params["title"]
        seed = custom_params["seed"] #TODO check if this is the right place to get seed
        evaluation_episodes = custom_params["evaluation_episodes"]

        # if save_freq > 0 and save_dir:
        #     save_dir = os.path.join(save_dir, title + "{}".format(str(seed)))
        #     os.makedirs(save_dir, exist_ok=True)
        # assert not (save_frames and visualise)
        # if visualise:
        #     assert render_freq > 0
        # if save_frames:
        #     assert render_freq > 0
        #     vidir = os.path.join(save_dir, "frames")
        #     os.makedirs(vidir, exist_ok=True)

        max_steps = 10000 #TODO this needs to be bigger than environment max steps
        total_reward = 0.
        returns = []
        start_time = time.time()
        video_index = 0
        # agent.epsilon_final = 0.
        # agent.epsilon = 0.
        # agent.noise = None
        env = self.env
        agent = self.orchestrator.model
        t_so_far = 0
        i = 0

        sticky_orch_value = 5

        ret, info = env.reset(options = reset_options)
        while t_so_far < total_timesteps:
            # if save_freq > 0 and save_dir and i % save_freq == 0:
            #     self.orchestrator.model.save_models(os.path.join(save_dir, str(i)))
            
            ret, info = env.reset(options = reset_options)
            # print(reset_ret)
            raw_state = ret
            processed_state = self.orchestrator.process_obs(raw_state)[0]
            state = processed_state
            # if isinstance(ret, tuple):
            #     state, _ = ret
            # else:
            #     state = ret
            # # state = np.array(state, dtype=np.float32, copy=False)
            # state = np.asarray(state, dtype = np.float32)

            # if visualise and i % render_freq == 0:
            #     env.render()
            predict_ret = self.predict(raw_state)
            # act, act_param, all_action_parameters = agent.act(state)
            base_action, action_data = predict_ret
            act, all_action_parameters = action_data["orchestral_action"]
            old_orch_act = (act, all_action_parameters)

            act_param =  all_action_parameters[act]

            # action = self.pad_action(act, act_param)

            episode_reward = 0.
            agent.start_episode()

            for j in range(max_steps):
                t_so_far+=1
                env_ret = env.step(base_action) #which comes from the controllers now
                # if(len(ret)==5):
                #     (next_state, steps), reward, terminal, _, _ = ret
                # elif(len(ret)==4):
                #     (next_state, steps), reward, terminal, _ = ret
                next_raw_state, reward, terminal, truncated, info = env_ret

                # (next_state, steps), reward, terminal, _, _ = ret
                # next_state = np.array(next_state, dtype=np.float32, copy=False)
                # next_state = np.asarray(state, dtype = np.float32)
                next_predict_ret = self.predict(next_raw_state, j%sticky_orch_value, old_orch_act)
                # act, act_param, all_action_parameters = agent.act(state)
                # act_param =  all_action_parameters[act]

                # next_predict_ret = self.predict(next_raw_state)
                next_processed_state = self.orchestrator.process_obs(next_raw_state)[0]
                next_state = next_processed_state
                next_base_action, next_action_data = next_predict_ret
                next_act, next_all_action_parameters = next_action_data["orchestral_action"]
                old_orch_act = (next_act, next_all_action_parameters)

                next_act_param = next_all_action_parameters[next_act]

                # next_act, next_act_param, next_all_action_parameters = agent.act(next_state)

                # next_action = self.pad_action(next_act, next_act_param)

                agent.step(state, (act, all_action_parameters), reward, next_state,
                        (next_act, next_all_action_parameters), terminal, 1)
                act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
                # action = next_action
                base_action = next_base_action
                # print(base_action)
                raw_state = next_raw_state
                state = next_state

                episode_reward += reward
                # if visualise and i % render_freq == 0:
                #     env.render()

                if terminal or truncated:
                    break
            i+=1
            agent.end_episode()

            # if save_frames and i % render_freq == 0:
            #     video_index = env.unwrapped.save_render_states(vidir, title, video_index)

            returns.append(episode_reward)
            total_reward += episode_reward
            if i % 100 == 0:
                print('{0:5s} R:{1:.4f} r100:{2:.4f}'.format(str(t_so_far), total_reward / (i + 1), np.array(returns[-100:]).mean()))
            
        end_time = time.time()
        print("Took %.2f seconds" % (end_time - start_time))
        env.close()
        # if save_freq > 0 and save_dir:
        #     agent.save_models(os.path.join(save_dir, str(i)))

        # returns = env.get_episode_rewards()
        print("Ave. return =", sum(returns) / len(returns))
        print("Ave. last 100 episode return =", sum(returns[-100:]) / 100.)

        # np.save(os.path.join(dir, title + "{}".format(str(seed))),returns)
        # self.orchestrator.model.learn(**kwargs)
    def save(self, path, name):
        self.orchestrator.model.save(path, name)