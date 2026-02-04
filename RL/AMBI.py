import copy, time
from RL.alg import Algorithm
from utils import core as utils_core
from utils.utils import setup_logs


def _snapshot_env_state(env):
    """Capture the current MuJoCo state (qpos, qvel) for later restoration."""
    # gymnasium MuJoCo (data.qpos / data.qvel) — forwards through wrapper chain via __getattr__
    try:
        if hasattr(env, "data") and hasattr(env.data, "qpos") and hasattr(env.data, "qvel"):
            return (env.data.qpos.copy(), env.data.qvel.copy())
    except (AttributeError, RuntimeError):
        pass
    # old mujoco-py (sim.get_state)
    try:
        sim = getattr(env, "sim", None)
        if sim is not None:
            return sim.get_state()
    except (AttributeError, RuntimeError):
        pass
    return None


def _restore_env_state(env, state):
    """Restore environment to a previously captured MuJoCo state."""
    if state is None:
        return False
    # gymnasium MuJoCo: (qpos, qvel) tuple — set_state calls mj_forward internally
    try:
        if hasattr(state, "__len__") and len(state) == 2 and hasattr(env, "set_state"):
            env.set_state(state[0], state[1])
            return True
    except (AttributeError, RuntimeError, TypeError):
        pass
    # old mujoco-py
    try:
        sim = getattr(env, "sim", None)
        if sim is not None:
            sim.set_state(state)
            if hasattr(sim, "forward"):
                sim.forward()
            return True
    except (AttributeError, RuntimeError):
        pass
    return False


def _reset_elapsed_steps(env, step_count=0):
    """Reset the TimeLimit wrapper's _elapsed_steps counter in the env chain.
    Needed after state restore to prevent premature truncation on rollouts b=2..B."""
    current = env
    while current is not None:
        if hasattr(current, '_elapsed_steps'):
            current._elapsed_steps = step_count
            return True
        if hasattr(current, 'env'):
            current = current.env
        else:
            break
    return False


def _get_sb3_model(agent):
    """Get the underlying SB3 model from an agent (Baseline wraps it as .model)."""
    return agent.model if hasattr(agent, "model") else agent


class AMBI(Algorithm):
    """Adaptive Model-Based Imagined trajectories (AMBI).
    Implements Algorithm 2: Anytime model-based pi-improvement w/ env as model."""

    def __init__(self, name, env, custom_params=None):
        super().__init__(name, env, custom_params=custom_params)
        cp = custom_params or {}

        self.outer_alg_str = cp.get("outer_alg", cp.get("alg", None))
        self.outer_alg_params = cp.get("outer_alg_params", {})
        self.inner_alg_str = cp.get("inner_alg", self.outer_alg_str)
        self.inner_alg_params = cp.get("inner_alg_params", {})

        self.inner_rollouts = int(cp.get("inner_rollouts", 6))
        self.inner_reinit_every_step = bool(cp.get("inner_reinit_every_step", True))
        self.inner_updates_per_rollout = int(cp.get("inner_updates_per_rollout", 1))

        self.max_episode_steps = int(cp.get("max_episode_steps", 250))
        self.seed_episodes = int(cp.get("seed_episodes", 0))
        self._render = bool(cp.get("render", False))

        if self.outer_alg_str is None:
            raise ValueError("AMBI requires 'outer_alg' string in custom_params")

        # Alg line 6: RL_alg_o <- SAC.init()
        self.outer_agent = self._init_agent(self.outer_alg_str, env, self.outer_alg_params)

        # Alg line 4: Initialize inner env env_i (deepcopy once, re-synced per episode)
        self.env_i = copy.deepcopy(self.env)

        # inner agent created once on first use, re-initialized per timestep
        self._inner_agent = None

        # Alg line 1: Initialize D^o with S seed episodes
        if self.seed_episodes > 0:
            self._initialize_seed_episodes()

    def _init_agent(self, alg_str, env, alg_params):
        result = utils_core.initialize_alg(alg_str, alg_params, env)
        if result is None:
            raise ValueError(f"Failed to initialize algorithm '{alg_str}'")
        if isinstance(result, (tuple, list)):
            return result[0]
        return result

    def get_model(self):
        return self.outer_agent.get_model() if hasattr(self.outer_agent, 'get_model') else self.outer_agent

    def predict(self, observation):
        return self.outer_agent.predict(observation)

    def save(self, path, name):
        if hasattr(self.outer_agent, 'save'):
            self.outer_agent.save(path, name)

    def load(self, path):
        if hasattr(self.outer_agent, 'load'):
            self.outer_agent.load(path)

    def set_logger(self, logger):
        self.alg_logger = logger
        if hasattr(self.outer_agent, 'set_logger'):
            self.outer_agent.set_logger(logger)

    def _initialize_seed_episodes(self):
        print(f"Initializing {self.seed_episodes} seed episodes...")
        total_transitions = 0
        for _ in range(self.seed_episodes):
            observation, info = self.env.reset()
            for _ in range(self.max_episode_steps):
                action = self.env.action_space.sample()
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self._add_to_replay_buffer(
                    self.outer_agent, observation, action, reward, next_obs,
                    terminated or truncated
                )
                total_transitions += 1
                observation = next_obs
                if terminated or truncated:
                    break
        print(f"Buffer initialized with {total_transitions} transitions")

    # ------------------------------------------------------------------
    # Inner agent management
    # ------------------------------------------------------------------

    def _reinit_inner_agent(self):
        """RL_alg_i <- SAC.init() (Alg line 11).
        Reset all parameters in-place instead of allocating a new model."""
        model = _get_sb3_model(self._inner_agent)

        if hasattr(model, "policy"):
            for module in model.policy.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
            # match SB3 init: copy critic -> critic_target
            if hasattr(model, 'critic_target') and hasattr(model, 'critic'):
                model.critic_target.load_state_dict(model.critic.state_dict())
            # clear optimizer momentum / adaptive state
            if hasattr(model, 'actor') and hasattr(model.actor, 'optimizer'):
                model.actor.optimizer.state.clear()
            if hasattr(model, 'critic') and hasattr(model.critic, 'optimizer'):
                model.critic.optimizer.state.clear()
            # reset entropy coefficient (SAC auto-tuning)
            if hasattr(model, 'log_ent_coef'):
                model.log_ent_coef.data.zero_()
            if hasattr(model, 'ent_coef_optimizer'):
                model.ent_coef_optimizer.state.clear()

        # clear replay buffer (D_i empty)
        if hasattr(model, "replay_buffer"):
            model.replay_buffer.reset()

    # ------------------------------------------------------------------
    # Replay buffer helpers
    # ------------------------------------------------------------------

    def _add_to_replay_buffer(self, agent, obs, action, reward, next_obs, done):
        model = _get_sb3_model(agent)
        if hasattr(model, "replay_buffer"):
            try:
                model.replay_buffer.add(obs, next_obs, action, reward, done, [{}])
            except (TypeError, ValueError):
                model.replay_buffer.add(obs, action, reward, next_obs, done)

    def _add_transitions_to_buffer(self, agent, transitions):
        for obs, action, reward, next_obs, done in transitions:
            self._add_to_replay_buffer(agent, obs, action, reward, next_obs, done)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _train_agent(self, agent, gradient_steps=1):
        model = _get_sb3_model(agent)
        if hasattr(model, "replay_buffer") and hasattr(model, "train"):
            buf_size = (model.replay_buffer.size()
                        if hasattr(model.replay_buffer, 'size')
                        else getattr(model.replay_buffer, 'pos', 0))
            batch_size = getattr(model, 'batch_size', 256)
            if buf_size >= batch_size:
                model.train(gradient_steps=gradient_steps)
                return True
        return False

    def _update_outer_agent(self, num_transitions):
        """Alg line 26: theta <- RL_alg_o.update(D_o)"""
        model = _get_sb3_model(self.outer_agent)
        if hasattr(model, "replay_buffer") and hasattr(model, "train"):
            buf_size = (model.replay_buffer.size()
                        if hasattr(model.replay_buffer, 'size')
                        else getattr(model.replay_buffer, 'pos', 0))
            learning_starts = getattr(model, 'learning_starts', 0)
            if buf_size >= learning_starts and buf_size > 0:
                gradient_steps = max(1, num_transitions)
                model.train(gradient_steps=gradient_steps)
                return True
        return False

    # ------------------------------------------------------------------
    # Main training loop — Algorithm 2
    # ------------------------------------------------------------------

    def learn(self, **kwargs):
        if "total_timesteps" in kwargs:
            total_timesteps = kwargs["total_timesteps"]
        else:
            total_timesteps = 10000

        t_so_far = 0
        episodes = 0
        start_time = time.time()

        print(f"\n{'='*60}")
        print("AMBI TRAINING")
        print(f"Timesteps: {total_timesteps:,} | B: {self.inner_rollouts} "
              f"| T: {self.max_episode_steps}")
        print(f"{'='*60}\n")

        # while not converged (Alg line 7)
        while t_so_far < total_timesteps:

            # o_1^o <- env_o.reset() (Alg line 8)
            observation, info = self.env.reset()

            # sync inner env at episode boundary (task goals, wrapper state, etc.)
            self.env_i = copy.deepcopy(self.env)

            # create inner agent on first use
            if self._inner_agent is None:
                self._inner_agent = self._init_agent(
                    self.inner_alg_str, self.env_i, self.inner_alg_params
                )

            episode_reward = 0.
            episode_transitions = []

            # for timestep t = 1..T (Alg line 9)
            for episode_step in range(self.max_episode_steps):

                # d_t <- env_o.sim.data (Alg line 10)
                snapshot = _snapshot_env_state(self.env)
                if snapshot is None:
                    raise RuntimeError(
                        "Cannot snapshot environment state — "
                        "make sure the env exposes MuJoCo data (qpos/qvel)."
                    )

                # RL_alg_i <- SAC.init() (Alg line 11)
                if self.inner_reinit_every_step:
                    self._reinit_inner_agent()

                # for rollout b = 1..B (Alg line 12)
                inner_step_start = time.time()
                total_imagined_steps = 0
                for _ in range(self.inner_rollouts):

                    # env_i.sim.data <- d_t (Alg line 13)
                    _restore_env_state(self.env_i, snapshot)
                    _reset_elapsed_steps(self.env_i, episode_step)

                    # o_1^i <- o_t^o (Alg line 14)
                    obs_i = copy.deepcopy(observation)

                    # for imagined_time i = t..T (Alg lines 15-18)
                    rollout_transitions = []
                    steps_remaining = self.max_episode_steps - episode_step
                    for _ in range(steps_remaining):
                        action_i, _ = self._inner_agent.predict(obs_i)
                        obs_next_i, reward_i, term_i, trunc_i, _ = self.env_i.step(action_i)
                        done_i = term_i or trunc_i
                        rollout_transitions.append(
                            (obs_i, action_i, reward_i, obs_next_i, done_i)
                        )
                        obs_i = obs_next_i
                        if done_i:
                            break

                    total_imagined_steps += len(rollout_transitions)

                    # D_i <- D_i + rollout (Alg line 19)
                    self._add_transitions_to_buffer(self._inner_agent, rollout_transitions)

                    # theta <- RL_alg_i.update(D_i) (Alg line 20)
                    self._train_agent(
                        self._inner_agent,
                        gradient_steps=self.inner_updates_per_rollout
                    )

                inner_step_elapsed = time.time() - inner_step_start

                # a_t^o <- RL_alg_i.predict(o_t^o) (Alg line 22)
                real_action, _ = self._inner_agent.predict(observation)

                # r_t^o, o_{t+1}^o <- env_o.step(a_t^o) (Alg line 23)
                next_obs, reward, terminated, truncated, info = self.env.step(real_action)

                if self.alg_logger:
                    data = setup_logs(reward, observation, real_action,
                                      [terminated, truncated], [info,])
                    self.alg_logger.on_step(data)

                episode_transitions.append(
                    (observation, real_action, reward, next_obs, terminated or truncated)
                )

                t_so_far += 1
                episode_reward += reward
                observation = next_obs

                if t_so_far % 100 == 0:
                    elapsed = time.time() - start_time
                    sps = t_so_far / elapsed if elapsed > 0 else 0
                    pct = 100 * t_so_far / total_timesteps
                    print('ep:{0:4s} t:{1:7s} R:{2:8.2f} | {3:5.1f}% | {4:.1f} sps | '
                          'inner: {5}img_steps in {6:.2f}s'.format(
                        str(episodes), str(t_so_far), episode_reward, pct, sps,
                        total_imagined_steps, inner_step_elapsed))

                if self._render:
                    self.env.render()

                if terminated or truncated or t_so_far >= total_timesteps:
                    break

            # D_o <- D_o + episode (Alg line 25)
            for trans in episode_transitions:
                self._add_to_replay_buffer(self.outer_agent, *trans)

            # theta <- RL_alg_o.update(D_o) (Alg line 26)
            outer_updated = self._update_outer_agent(len(episode_transitions))

            episodes += 1
            elapsed = time.time() - start_time
            outer_buf = 0
            outer_model = _get_sb3_model(self.outer_agent)
            if hasattr(outer_model, "replay_buffer"):
                outer_buf = (outer_model.replay_buffer.size()
                             if hasattr(outer_model.replay_buffer, 'size')
                             else getattr(outer_model.replay_buffer, 'pos', 0))
            print(f"\n--- Episode {episodes} ---")
            print(f"  Steps: {len(episode_transitions)} | Return: {episode_reward:.2f}")
            print(f"  Total timesteps: {t_so_far:,} / {total_timesteps:,} "
                  f"({100*t_so_far/total_timesteps:.1f}%)")
            print(f"  Outer buffer: {outer_buf:,} | Outer updated: {outer_updated}")
            print(f"  Wall time: {elapsed:.0f}s | "
                  f"SPS: {t_so_far/elapsed if elapsed > 0 else 0:.1f}\n")

        end_time = time.time()
        print("Took %.2f seconds" % (end_time - start_time))
        print(f"Training complete | {episodes} episodes | {t_so_far:,} timesteps")
        return self.outer_agent
