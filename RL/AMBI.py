import copy, time
import numpy as np
import torch.nn as nn
from RL.alg import Algorithm
from utils import core as utils_core
from utils.utils import setup_logs


def _snapshot_env_state(env):
    """Capture the current state of the environment for later restoration."""
    try:
        sim = getattr(env, "sim", None)
        if sim is not None:
            return sim.get_state()
    except (AttributeError, RuntimeError):
        pass
    for fn in ("get_state", "_get_state", "state_vector"):
        if hasattr(env, fn):
            try:
                return getattr(env, fn)()
            except (AttributeError, RuntimeError):
                pass
    return None

def _restore_env_state(env, state):
    """Restore environment to a previously captured state."""
    if state is None:
        return False
    try:
        sim = getattr(env, "sim", None)
        if sim is not None:
            sim.set_state(state)
            if hasattr(sim, "forward"):
                try:
                    sim.forward()
                except (AttributeError, RuntimeError):
                    pass
            return True
    except (AttributeError, RuntimeError):
        pass
    for fn in ("set_state", "_set_state", "set_state_vector"):
        if hasattr(env, fn):
            try:
                getattr(env, fn)(state)
                return True
            except (AttributeError, RuntimeError):
                pass
    return False


class AMBI(Algorithm):
    """
    Adaptive Model-Based Imagined trajectories (AMBI) Algorithm.

    AMBI uses a two-loop structure:
    - Inner loop: Runs imagined rollouts from the current state to improve action selection
    - Outer loop: Takes real actions in the environment and updates the policy

    Args:
        name: Algorithm name
        env: Gym environment
        custom_params: Dictionary of hyperparameters including:
            - outer_alg: Algorithm for outer agent (e.g., "baselines/SAC")
            - inner_alg: Algorithm for inner agent (defaults to outer_alg)
            - inner_rollouts: Number of imagined rollouts per step (default: 6)
            - inner_reinit_every_step: Whether to reinitialize inner agent each step (default: True)
            - max_episode_steps: Maximum steps per episode (default: 250)
            - seed_episodes: Number of random episodes to initialize buffer (default: 0)
    """
    def __init__(self, name, env, custom_params=None):
        super().__init__(name, env, custom_params=custom_params)
        cp = custom_params or {}

        # Outer and inner algorithm configuration
        self.outer_alg_str = cp.get("outer_alg", cp.get("alg", None))
        self.outer_alg_params = cp.get("outer_alg_params", {})
        self.inner_alg_str = cp.get("inner_alg", self.outer_alg_str)
        self.inner_alg_params = cp.get("inner_alg_params", {})

        # Inner loop: imagined rollouts from current state
        self.inner_rollouts = int(cp.get("inner_rollouts", 6))
        self.inner_reinit_every_step = bool(cp.get("inner_reinit_every_step", True))
        self.inner_updates_per_rollout = int(cp.get("inner_updates_per_rollout", 1))

        # Outer loop: real environment interaction
        self.max_episode_steps = int(cp.get("max_episode_steps", 250))
        self.seed_episodes = int(cp.get("seed_episodes", 0))
        self.render = bool(cp.get("render", False))

        if self.outer_alg_str is None:
            raise ValueError("AMBI requires 'outer_alg' string in custom_params")

        self.outer_agent = self.get_outer_model(self.outer_alg_str, env, self.outer_alg_params)

        # Inner buffer for imagined experience (outer agent manages its own replay buffer)
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
        """Initialize outer agent replay buffer with random exploration."""
        print(f"Initializing {self.seed_episodes} seed episodes...")
        total_transitions = 0
        for _ in range(self.seed_episodes):
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs, _ = obs
            for _ in range(self.max_episode_steps):
                action = self.env.action_space.sample()
                ret = self.env.step(action)
                if len(ret) == 5:
                    obs_next, reward, terminated, truncated, _ = ret
                    done = bool(terminated or truncated)
                else:
                    obs_next, reward, done, _ = ret

                self._add_to_replay_buffer(self.outer_agent, obs, action, reward, obs_next, done)
                total_transitions += 1

                obs = obs_next
                if done:
                    break
        print(f"✓ Buffer initialized with {total_transitions} transitions")

    def set_logger(self, logger):
        self.alg_logger = logger
        if hasattr(self.outer_agent, 'set_logger'):
            self.outer_agent.set_logger(logger)

    def predict(self, obs):
        return self.outer_agent.predict(obs)

    def save(self, save_path, name):
        if hasattr(self.outer_agent, 'save'):
            self.outer_agent.save(save_path, name)

    def load(self, load_path):
        if hasattr(self.outer_agent, 'load'):
            self.outer_agent.load(load_path)

    def _run_inner_rollout_once(self, env_copy, inner_agent, init_obs, current_step):
        """Run imagined rollout from current step to end of episode (T).

        Args:
            env_copy: Copy of the environment for imagination
            inner_agent: Inner agent to use for predictions
            init_obs: Initial observation (current outer observation)
            current_step: Current timestep in outer episode (t)
        """
        obs = init_obs
        cum_reward = 0.0
        # Run from current step t to end of episode T
        steps_remaining = self.max_episode_steps - current_step
        for step in range(steps_remaining):
            action, _ = inner_agent.predict(obs)
            ret = env_copy.step(action)
            if len(ret) == 5:
                obs_next, reward, terminated, truncated, _ = ret
                done = bool(terminated or truncated)
            else:
                obs_next, reward, done, _ = ret
            self.inner_buffer.append((obs, action, reward, obs_next, done))
            obs = obs_next
            cum_reward += float(reward)
            if done:
                # Episode ended early in imagination
                break
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

        return inner_agent

    def _update_inner_agent(self, inner_agent):
        """Update inner agent using collected imagined experience."""
        if not self.inner_buffer:
            return False

        for _ in range(self.inner_updates_per_rollout):
            # Strategy 1: Replay buffer + train() (stable-baselines3)
            if hasattr(inner_agent, "replay_buffer") and hasattr(inner_agent, "train"):
                for obs, action, reward, next_obs, done in self.inner_buffer:
                    try:
                        inner_agent.replay_buffer.add(obs, next_obs, action, reward, done, [{}])
                    except (TypeError, ValueError):
                        inner_agent.replay_buffer.add(obs, action, reward, next_obs, done)

                if len(inner_agent.replay_buffer) > 0:
                    inner_agent.train(gradient_steps=self.inner_updates_per_rollout)
                    return True

            # Strategy 2: Direct update() method (custom implementations)
            if hasattr(inner_agent, "update") and callable(inner_agent.update):
                inner_agent.update(self.inner_buffer)
                return True

            # Strategy 3: Step-by-step updates (custom implementations)
            if hasattr(inner_agent, "step") and callable(inner_agent.step):
                for obs, action, reward, next_obs, done in self.inner_buffer:
                    try:
                        inner_agent.step(obs, action, reward, next_obs, None, done)
                    except TypeError:
                        inner_agent.step(obs, action, reward, next_obs, done)
                return True

        print("Warning: Could not update inner agent - no compatible update method found")
        return False

    def _add_to_replay_buffer(self, agent, obs, action, reward, next_obs, done):
        """Add a transition directly to the agent's replay buffer if it exists."""
        if hasattr(agent, "replay_buffer"):
            try:
                agent.replay_buffer.add(obs, next_obs, action, reward, done, [{}])
            except (TypeError, ValueError):
                agent.replay_buffer.add(obs, action, reward, next_obs, done)

    def _update_outer_agent(self):
        """Update outer agent using real environment experience from replay buffer."""
        # Strategy 1: Replay buffer + train() (stable-baselines3)
        if hasattr(self.outer_agent, "replay_buffer") and hasattr(self.outer_agent, "train"):
            if len(self.outer_agent.replay_buffer) > 0:
                gradient_steps = min(50, len(self.outer_agent.replay_buffer))
                self.outer_agent.train(gradient_steps=gradient_steps)
                return True

        # Strategy 2: Direct update() method (custom implementations)
        if hasattr(self.outer_agent, "update") and callable(self.outer_agent.update):
            self.outer_agent.update([])
            return True

        print("Warning: Could not update outer agent - no compatible update method found")
        return False

    def _make_model_env_copy(self):
        """Create a copy of the environment for imagined rollouts."""
        try:
            env_copy = copy.deepcopy(self.env)
            # Disable rendering for imagined rollouts
            if hasattr(env_copy, 'render_mode'):
                env_copy.render_mode = None
            # Handle wrapped environments
            unwrapped = env_copy
            while hasattr(unwrapped, 'env'):
                if hasattr(unwrapped, 'render_mode'):
                    unwrapped.render_mode = None
                unwrapped = unwrapped.env
            if hasattr(unwrapped, 'render_mode'):
                unwrapped.render_mode = None
            return env_copy, False, None
        except (TypeError, AttributeError):
            # Fallback: use snapshot/restore if deepcopy fails
            snapshot = _snapshot_env_state(self.env)
            return self.env, True, snapshot

    def learn(self, total_timesteps=10000):
        """
        Train the AMBI agent.

        Uses inner loop imagination to improve action selection and outer loop
        real experience to train the policy.

        Args:
            total_timesteps: Total number of timesteps to train for
        """
        t = 0
        episodes = 0
        episode_step = 0
        episode_reward = 0.0

        print(f"\n{'='*60}")
        print("AMBI TRAINING")
        print(f"Timesteps: {total_timesteps:,} | Inner rollouts: {self.inner_rollouts} | Max episode steps: {self.max_episode_steps}")
        print(f"{'='*60}\n")

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
                    env_model_copy, inner_agent, obs_model, episode_step
                )

                # Update inner agent on imagined experience AFTER EACH ROLLOUT
                # This allows the inner agent to improve across rollouts b=1..B
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
                    info,
                )
                self.alg_logger.on_step(data)
            
            # Add transition directly to outer agent's replay buffer
            self._add_to_replay_buffer(self.outer_agent, outer_obs, real_action, reward, outer_obs_next, done)

            t += 1
            episode_step += 1
            episode_reward += reward

            # Print progress every 100 timesteps
            if t % 100 == 0:
                elapsed = time.time() - start_time
                steps_per_sec = t / elapsed if elapsed > 0 else 0
                progress = 100 * t / total_timesteps
                eta_min = (total_timesteps - t) / steps_per_sec / 60 if steps_per_sec > 0 else 0
                print(f"[{t:,}/{total_timesteps:,}] {progress:.1f}% | Episodes: {episodes} | {steps_per_sec:.1f} steps/s | ETA: {eta_min:.1f}m")

            outer_obs = outer_obs_next

            # Handle episode termination
            if done:
                episodes += 1
                print(f"Episode {episodes} complete | Steps: {episode_step} | Return: {episode_reward:.2f} | Timestep: {t:,}")

                episode_step = 0
                episode_reward = 0.0

                # Update outer agent after each episode
                self._update_outer_agent()

                # Reset environment for next episode
                reset_return = self.env.reset()
                if isinstance(reset_return, tuple):
                    outer_obs, _info = reset_return
                else:
                    outer_obs = reset_return

            if self.render:
                self.env.render()

        # Final update
        self._update_outer_agent()

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete | {total_time:.1f}s | {episodes} episodes | {t:,} timesteps")
        print(f"{'='*60}\n")
        return self.outer_agent
