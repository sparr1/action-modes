import copy, time
import numpy as np
import torch.nn as nn
from RL.alg import Algorithm
from RL.lora import (
    apply_lora_to_model,
    get_lora_parameters,
    replace_optimizer_params_with_lora,
)
from utils import core as utils_core
from utils.utils import setup_logs


def _snapshot_env_state(env):
    try:
        sim = getattr(env, "sim", None)
        if sim is not None:
            return sim.get_state()
    except Exception:
        pass
    for fn in ("get_state", "_get_state", "state_vector"):
        if hasattr(env, fn):
            try:
                return getattr(env, fn)()
            except Exception:
                pass
    return None


def _restore_env_state(env, state):
    if state is None:
        return False
    try:
        sim = getattr(env, "sim", None)
        if sim is not None:
            sim.set_state(state)
            if hasattr(sim, "forward"):
                try:
                    sim.forward()
                except Exception:
                    pass
            return True
    except Exception:
        pass
    for fn in ("set_state", "_set_state", "set_state_vector"):
        if hasattr(env, fn):
            try:
                getattr(env, fn)(state)
                return True
            except Exception:
                pass
    return False


class AMBI(Algorithm):
    def __init__(self, name, env, custom_params=None):
        super().__init__(name, env, custom_params=custom_params)
        cp = custom_params or {}

        # Outer and inner algorithm configuration
        self.outer_alg_str = cp.get("outer_alg", cp.get("alg", None))
        self.outer_alg_params = cp.get("outer_alg_params", {})
        self.inner_alg_str = cp.get("inner_alg", self.outer_alg_str)
        self.inner_alg_params = cp.get("inner_alg_params", {})

        # LoRA configuration for efficient inner agent adaptation
        self.use_lora = bool(cp.get("use_lora", True))
        self.lora_rank = int(cp.get("lora_rank", 4))
        self.lora_alpha = float(cp.get("lora_alpha", 1.0))
        self.lora_target_modules = cp.get("lora_target_modules", None)

        # Inner loop: imagined rollouts from current state
        self.inner_rollouts = int(cp.get("inner_rollouts", 6))
        self.inner_horizon = int(cp.get("inner_horizon", 32))
        self.inner_reinit_every_step = bool(cp.get("inner_reinit_every_step", True))
        self.inner_updates_per_rollout = int(cp.get("inner_updates_per_rollout", 1))

        # Outer loop: real environment interaction
        self.max_episode_steps = int(cp.get("max_episode_steps", 250))
        self.seed_episodes = int(cp.get("seed_episodes", 0))
        self.outer_update_frequency = int(cp.get("outer_update_frequency", 1))
        self.render = bool(cp.get("render", False))

        if self.outer_alg_str is None:
            raise ValueError("AMBI requires 'outer_alg' string in custom_params")

        self.outer_agent = self.get_outer_model(self.outer_alg_str, env, self.outer_alg_params)
        
        # Replay buffers: outer for real experience, inner for imagined experience
        self.outer_buffer = []
        self.inner_buffer = []
        
        if self.seed_episodes > 0:
            self._initialize_seed_episodes()
        
        self.alg_logger = None
        self._persistent_inner = None

    def get_model(self):
        return self.outer_agent.get_model() if hasattr(self.outer_agent, 'get_model') else self.outer_agent

    def get_outer_model(self, alg_str, env, alg_params):
        _outer = utils_core.initialize_alg(alg_str, alg_params, env)
        if isinstance(_outer, tuple) or isinstance(_outer, list):
            try:
                return _outer[0]
            except Exception:
                return _outer
        else:
            return _outer

    def get_inner_model(self, alg_str, env, alg_params):
        _inner = utils_core.initialize_alg(alg_str, alg_params, env)
        if isinstance(_inner, tuple) or isinstance(_inner, list):
            try:
                return _inner[0]
            except Exception:
                return _inner
        else:
            return _inner

    def _initialize_seed_episodes(self):
        print(f"Initializing {self.seed_episodes} seed episodes for outer buffer...")
        for _ in range(self.seed_episodes):
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs, _ = obs
            episode_data = []
            for _ in range(self.max_episode_steps):
                action = self.env.action_space.sample()
                ret = self.env.step(action)
                if len(ret) == 5:
                    obs_next, reward, terminated, truncated, info = ret
                    done = bool(terminated or truncated)
                else:
                    obs_next, reward, done, info = ret
                episode_data.append((obs, action, reward, obs_next, done))
                obs = obs_next
                if done:
                    break
            self.outer_buffer.extend(episode_data)
        print(f"Initialized outer buffer with {len(self.outer_buffer)} transitions from {self.seed_episodes} seed episodes.")

    def set_logger(self, logger):
        self.alg_logger = logger
        try:
            self.outer_agent.set_logger(logger)
        except Exception:
            pass

    def predict(self, obs):
        return self.outer_agent.predict(obs)

    def save(self, save_path, name):
        try:
            self.outer_agent.save(save_path, name)
        except Exception:
            pass

    def load(self, load_path):
        try:
            self.outer_agent.load(load_path)
        except Exception:
            pass

    def _run_inner_rollout_once(self, env_copy, inner_agent, init_obs):
        obs = init_obs
        cum_reward = 0.0
        rollout_data = []
        for step in range(self.inner_horizon):
            action, _ = inner_agent.predict(obs)
            ret = env_copy.step(action)
            if len(ret) == 5:
                obs_next, reward, terminated, truncated, info = ret
                done = bool(terminated or truncated)
            else:
                obs_next, reward, done, info = ret
            self.inner_buffer.append((obs, action, reward, obs_next, done))
            obs = obs_next
            cum_reward += float(reward)
            if done:
                ret_reset = env_copy.reset()
                if isinstance(ret_reset, tuple):
                    obs = ret_reset[0]
                else:
                    obs = ret_reset
        return obs, cum_reward

    def _get_model_from_agent(self, agent):
        if hasattr(agent, "model"):
            return agent.model
        if hasattr(agent, "policy"):
            return agent.policy
        if hasattr(agent, "actor"):
            return agent.actor
        if hasattr(agent, "q_network") or hasattr(agent, "qf"):
            return getattr(agent, "q_network", None) or agent.qf
        if hasattr(agent, "network"):
            return agent.network
        if hasattr(agent, "get_model"):
            try:
                return agent.get_model()
            except Exception:
                pass
        return agent

    def _copy_model_weights(self, source_agent, target_agent):
        source_model = self._get_model_from_agent(source_agent)
        target_model = self._get_model_from_agent(target_agent)
        if source_model is None or target_model is None:
            return False
        try:
            if hasattr(source_model, "state_dict") and hasattr(
                target_model, "load_state_dict"
            ):
                target_model.load_state_dict(source_model.state_dict())
                return True
        except Exception:
            pass
        try:
            source_params = dict(source_model.named_parameters())
            target_params = dict(target_model.named_parameters())
            for name, param in target_params.items():
                if name in source_params:
                    param.data.copy_(source_params[name].data)
            return True
        except Exception:
            pass
        return False

    def _build_inner_agent(self, agent_env):
        # Initialize inner agent with same architecture as outer agent
        inner_agent = self.get_inner_model(
            self.inner_alg_str, agent_env, self.inner_alg_params
        )

        # Copy weights from outer agent to initialize inner agent
        weights_copied = self._copy_model_weights(self.outer_agent, inner_agent)

        # Optionally apply LoRA for parameter-efficient fine-tuning
        if self.use_lora:
            inner_model = self._get_model_from_agent(inner_agent)

            if inner_model is not None and isinstance(inner_model, nn.Module):
                inner_model = apply_lora_to_model(
                    inner_model,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    target_modules=self.lora_target_modules,
                )

                lora_params = get_lora_parameters(inner_model)

                if lora_params:
                    # Configure optimizer to train only LoRA parameters
                    replaced = replace_optimizer_params_with_lora(
                        inner_agent,
                        lora_params,
                        base_lr=self.inner_alg_params.get("learning_rate", 3e-4),
                    )
        return inner_agent

    def _update_inner_agent(self, inner_agent):
        if not self.inner_buffer:
            return False

        updated = False

        for _ in range(self.inner_updates_per_rollout):
            # Try multiple update strategies for compatibility with different RL frameworks
            # Strategy 1: Direct update() method (custom implementations)
            try:
                if hasattr(inner_agent, "update") and callable(inner_agent.update):
                    inner_agent.update(self.inner_buffer)
                    updated = True
                    break
            except Exception:
                pass

            # Strategy 2: learn() method (stable-baselines3 style)
            try:
                if hasattr(inner_agent, "learn") and callable(inner_agent.learn):
                    inner_agent.learn(total_timesteps=len(self.inner_buffer))
                    updated = True
                    break
            except Exception:
                pass

            # Strategy 3: Replay buffer + train() (stable-baselines3)
            try:
                if hasattr(inner_agent, "replay_buffer") and hasattr(
                    inner_agent, "train"
                ):
                    for obs, action, reward, next_obs, done in self.inner_buffer:
                        try:
                            inner_agent.replay_buffer.add(
                                obs, next_obs, action, reward, done, [{}]
                            )
                        except Exception:
                            try:
                                inner_agent.replay_buffer.add(
                                    obs, action, reward, next_obs, done
                                )
                            except Exception:
                                pass

                    if len(inner_agent.replay_buffer) > 0:
                        inner_agent.train(gradient_steps=self.inner_updates_per_rollout)
                        updated = True
                        break
            except Exception:
                pass

            # Strategy 4: Step-by-step updates (custom implementations)
            try:
                if hasattr(inner_agent, "step") and callable(inner_agent.step):
                    for obs, action, reward, next_obs, done in self.inner_buffer:
                        try:
                            inner_agent.step(obs, action, reward, next_obs, None, done)
                        except Exception:
                            try:
                                inner_agent.step(obs, action, reward, next_obs, done)
                            except Exception:
                                pass
                    updated = True
                    break
            except Exception:
                pass

        if not updated:
            print(
                "  Warning: Could not update inner agent - no compatible update method found"
            )

        return updated

    def _update_outer_agent(self):
        if not self.outer_buffer:
            return False

        updated = False

        # Try multiple update strategies for compatibility with different RL frameworks
        # Strategy 1: Direct update() method
        try:
            if hasattr(self.outer_agent, "update") and callable(self.outer_agent.update):
                self.outer_agent.update(self.outer_buffer)
                updated = True
        except Exception:
            pass

        # Strategy 2: learn() method (stable-baselines3 style)
        if not updated:
            try:
                if hasattr(self.outer_agent, "learn") and callable(self.outer_agent.learn):
                    self.outer_agent.learn(total_timesteps=0)
                    updated = True
            except Exception:
                pass

        # Strategy 3: Replay buffer + train() (stable-baselines3)
        if not updated:
            try:
                if hasattr(self.outer_agent, "replay_buffer") and hasattr(
                    self.outer_agent, "train"
                ):
                    for obs, action, reward, next_obs, done in self.outer_buffer:
                        try:
                            self.outer_agent.replay_buffer.add(
                                obs, next_obs, action, reward, done, [{}]
                            )
                        except Exception:
                            try:
                                self.outer_agent.replay_buffer.add(
                                    obs, action, reward, next_obs, done
                                )
                            except Exception:
                                pass

                    if len(self.outer_agent.replay_buffer) > 0:
                        self.outer_agent.train(gradient_steps=len(self.outer_buffer))
                        updated = True
            except Exception:
                pass

        if not updated:
            print(
                "  Warning: Could not update outer agent - no compatible update method found"
            )

        return updated

    def _make_model_env_copy(self):
        try:
            env_copy = copy.deepcopy(self.env)
            return env_copy, False, None
        except Exception:
            snapshot = _snapshot_env_state(self.env)
            return self.env, True, snapshot

    def learn(self, total_timesteps=10000):
        t = 0
        episodes = 0
        episode_buffer = []
        
        # Initialize environment and get initial observation
        reset_return = self.env.reset()
        if isinstance(reset_return, tuple):
            outer_obs, _info = reset_return
        else:
            outer_obs = reset_return
        
        start_time = time.time()
        
        while t < total_timesteps:
            # Create environment copy for inner imagined rollouts
            env_model_copy, uses_snapshot, snapshot = self._make_model_env_copy()
            
            # Initialize or reuse inner agent (reinitializes each step by default)
            if self.inner_reinit_every_step:
                inner_agent = self._build_inner_agent(env_model_copy)
            else:
                if self._persistent_inner is None:
                    self._persistent_inner = self._build_inner_agent(env_model_copy)
                inner_agent = self._persistent_inner
            
            # Clear inner buffer for new imagined rollouts
            self.inner_buffer = []
            
            # Run multiple imagined rollouts from current state
            for b in range(self.inner_rollouts):
                # Reset inner environment to current outer environment state
                if uses_snapshot:
                    _restore_env_state(env_model_copy, snapshot)
                    reset_return = env_model_copy.reset()
                    if isinstance(reset_return, tuple):
                        obs_model = reset_return[0]
                    else:
                        obs_model = reset_return
                    obs_model = outer_obs
                else:
                    try:
                        # Snapshot current outer env state and restore in inner env
                        outer_snapshot = _snapshot_env_state(self.env)
                        if outer_snapshot is not None:
                            _restore_env_state(env_model_copy, outer_snapshot)
                            reset_return = env_model_copy.reset()
                            if isinstance(reset_return, tuple):
                                obs_model = reset_return[0]
                            else:
                                obs_model = reset_return
                            obs_model = outer_obs
                        else:
                            obs_model = outer_obs
                    except Exception:
                        obs_model = outer_obs
                
                # Run imagined rollout and collect experience
                final_obs, cum = self._run_inner_rollout_once(
                    env_model_copy, inner_agent, obs_model
                )
                
                # Update inner agent on imagined experience
                self._update_inner_agent(inner_agent)
            
            # Use inner agent to select action for real environment
            try:
                real_action, _ = inner_agent.predict(outer_obs)
            except Exception:
                # Fallback to outer agent if inner agent fails
                real_action, _ = self.outer_agent.predict(outer_obs)
            
            # Execute action in real environment
            ret = self.env.step(real_action)
            if len(ret) == 5:
                outer_obs_next, reward, terminated, truncated, info = ret
                done = bool(terminated or truncated)
            else:
                outer_obs_next, reward, done, info = ret
            
            # Log step data if logger is configured
            if self.alg_logger:
                data = setup_logs(
                    reward,
                    outer_obs,
                    real_action,
                    [terminated if "terminated" in locals() else False, done],
                    [info],
                )
                self.alg_logger.on_step(data)
            
            # Store transition in episode buffer
            episode_buffer.append((outer_obs, real_action, reward, outer_obs_next, done))
            
            t += 1
            outer_obs = outer_obs_next
            
            # Handle episode termination
            if done:
                episodes += 1
                
                # Add episode transitions to outer buffer
                self.outer_buffer.extend(episode_buffer)
                episode_buffer = []
                
                # Update outer agent periodically
                if episodes % self.outer_update_frequency == 0:
                    self._update_outer_agent()
                
                # Reset environment for next episode
                reset_return = self.env.reset()
                if isinstance(reset_return, tuple):
                    outer_obs, _info = reset_return
                else:
                    outer_obs = reset_return
            
            # Periodic outer updates during long episodes
            if t % (self.outer_update_frequency * self.max_episode_steps) == 0:
                if episode_buffer:
                    self.outer_buffer.extend(episode_buffer)
                    episode_buffer = []
                self._update_outer_agent()
            
            if self.render:
                try:
                    self.env.render()
                except Exception:
                    pass
        
        # Final update with any remaining data
        if episode_buffer:
            self.outer_buffer.extend(episode_buffer)
            self._update_outer_agent()
        
        total_time = time.time() - start_time
        print(
            f"AMBI Training completed in {total_time:.2f} seconds over {episodes} episodes and {t} timesteps."
        )
        return self.outer_agent
