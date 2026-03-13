import copy, time, json, os, random
import numpy as np
from typing import Any, Mapping, Iterable

from RL.lora import lorafy
from RL.alg import Algorithm
from utils import core as utils_core
from utils.utils import setup_logs
from utils.core import *
import utils.ambi_debug as utils_ambi_debug

import torch.nn as nn
import torch
import mujoco
import gymnasium as gym
import wandb
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


def _snapshot_env_state(env):
    """Capture the current state of the environment for later restoration."""
    unwrapped = env.unwrapped
    model, data = unwrapped.model, unwrapped.data

    # Get the full physics state of the environment (qpos, qvel, etc.)
    full_physics_spec = int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
    n = mujoco.mj_stateSize(model, full_physics_spec)
    x = np.empty(n, dtype=np.float64)
    mujoco.mj_getState(model, data, x, full_physics_spec)
    mujoco_state = x.copy()    


    # Retrieve wrapper attributes
    elapsed = env.get_wrapper_attr("_elapsed_steps") if hasattr(env, "get_wrapper_attr") else None
    has_reset = env.get_wrapper_attr("_has_reset") if hasattr(env, "get_wrapper_attr") else True

    # PassiveEnvChecker flags are optional; mainly for overhead :contentReference[oaicite:8]{index=8}
    checked_step  = env.get_wrapper_attr("checked_step")  if hasattr(env, "get_wrapper_attr") else True
    checked_reset = env.get_wrapper_attr("checked_reset") if hasattr(env, "get_wrapper_attr") else True
    checked_render = env.get_wrapper_attr("checked_render") if hasattr(env, "get_wrapper_attr") else True

    return {
        "mujoco": (mujoco_state, full_physics_spec),
        "wrappers": {
            "_elapsed_steps": elapsed,
            "_has_reset": has_reset,
            "checked_step": checked_step,
            "checked_reset": checked_reset,
            "checked_render": checked_render,
        }
    }


def _set_env_state(env, state):
    """Set environment to a previously captured state."""
    unwrapped = env.unwrapped
    model, data = unwrapped.model, unwrapped.data

    mujoco_state, full_physics_spec = state["mujoco"]
    mujoco.mj_setState(model, data, mujoco_state, int(full_physics_spec))
    mujoco.mj_forward(model, data)

    # Restore wrappers
    w = state["wrappers"]

    if w.get("_elapsed_steps", None) is not None:
        ok = _set_first_wrapper_attr_direct(env, "_elapsed_steps", int(w["_elapsed_steps"]))
        if not ok:
            raise RuntimeError("Could not restore _elapsed_steps on inner env wrapper chain")

    if w.get("_has_reset", None) is not None:
        _set_first_wrapper_attr_direct(env, "_has_reset", bool(w["_has_reset"]))

    for k in ("checked_step", "checked_reset", "checked_render"):
        if k in w:
            _set_first_wrapper_attr_direct(env, k, bool(w[k]))


def _set_first_wrapper_attr_direct(env, name, value):
    """
    Walk wrapper chain and set `name` on the first wrapper/object that has it.
    Returns True if set, False otherwise.
    """
    cur = env
    seen = set()
    while True:
        if hasattr(cur, name):
            setattr(cur, name, value)
            return True
        if not hasattr(cur, "env"):
            return False
        nxt = cur.env
        if id(nxt) in seen:
            return False
        seen.add(id(nxt))
        cur = nxt


def wandb_setup(cfg: dict, project="ambi", run_name=None):
    run = wandb.init(
        project=project,
        name=run_name,
        config=cfg,
    )

    # Make outer timestep the shared x-axis
    wandb.define_metric("outer/step")
    wandb.define_metric("outer/*", step_metric="outer/step")
    wandb.define_metric("inner/*", step_metric="outer/step")
    wandb.define_metric("time/*",  step_metric="outer/step")
    return run


def log_outer_step(t, reward, done, info=None, every=50, *, outer_step_sec=None):
    if t % every != 0 and not done:
        return

    payload = {
        "outer/step": t,
        "outer/reward": float(reward),
        "outer/done": int(done),
    }

    if outer_step_sec is not None:
        payload["time/outer_step_sec"] = float(outer_step_sec)
        if outer_step_sec > 0:
            payload["time/outer_steps_per_sec"] = float(1.0 / outer_step_sec)

    # Ant reward terms
    if isinstance(info, dict):
        for k, v in info.items():
            if isinstance(v, (int, float, np.number)) and str(k).startswith("reward_"):
                payload[f"outer/{k}"] = float(v)

    wandb.log(payload, step=t)


def log_outer_episode(t, ep_idx, ep_return, ep_len, *, episode_sec=None):
    payload = {
        "outer/step": t,
        "outer/episode": ep_idx,
        "outer/episode_return": float(ep_return),
        "outer/episode_len": int(ep_len),
    }
    if episode_sec is not None:
        payload["time/episode_sec"] = float(episode_sec)
    wandb.log(payload, step=t)


def log_inner_summary(t, rollout_returns, rollout_lengths=None, *, inner_time_sec=None):
    r = np.asarray(rollout_returns, dtype=np.float64)
    payload = {
        "outer/step": t,
        "inner/rollouts": int(len(r)),
        "inner/return_mean": float(r.mean()) if len(r) else 0.0,
        "inner/return_std": float(r.std()) if len(r) else 0.0,
        "inner/return_max": float(r.max()) if len(r) else 0.0,
        "inner/return_min": float(r.min()) if len(r) else 0.0,
    }

    if rollout_lengths is not None and len(rollout_lengths):
        L = np.asarray(rollout_lengths, dtype=np.float64)
        payload["inner/len_mean"] = float(L.mean())
        payload["inner/len_max"] = float(L.max())
        payload["inner/sim_steps"] = int(L.sum())
        if inner_time_sec is not None and inner_time_sec > 0:
            payload["time/inner_sec"] = float(inner_time_sec)
            payload["time/inner_steps_per_sec"] = float(L.sum() / inner_time_sec)

    elif inner_time_sec is not None:
        payload["time/inner_sec"] = float(inner_time_sec)

    wandb.log(payload, step=t)


class AMBI(Algorithm):
    """
    AMBI Algorithm.

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
    """

    def __init__(self, name, env, custom_params=None, run_params=None, experiment_params=None):
        super().__init__(name, env, custom_params=custom_params)
        cp = custom_params or {}

        # Outer and inner algorithm configuration
        self.outer_alg_str = cp.get("outer_alg", cp.get("alg", None))
        self.outer_alg_params = cp.get("outer_alg_params", {})
        self.inner_alg_str = cp.get("inner_alg", self.outer_alg_str)
        self.inner_alg_params = cp.get("inner_alg_params", {})

        assert self.outer_alg_str is not None, "AMBI requires 'outer_alg' string in custom_params"

        # Inner loop: imagined rollouts from current state
        self.inner_rollouts = int(cp.get("inner_rollouts", 6))
        self.inner_reinit_every_step = bool(cp.get("inner_reinit_every_step", True))
        self.inner_train_freq = int(self.inner_alg_params.get("train_freq"))
        self.inner_updates_per_rollout = int(cp.get("inner_updates_per_rollout", 1))
        self.use_lora = bool(cp.get("use_lora", False))
        self.lora_params = cp.get("lora_params", {})
        # learning starts for inner agent
        self.inner_learning_starts = self.inner_alg_params.get("learning_starts")
        self.inner_learning_starts_steps = self.inner_learning_starts.get("steps")
        self.inner_random_actions = bool(self.inner_learning_starts.get("random_actions", False))
        self.inner_use_action_noise = bool(self.inner_learning_starts.get("use_action_noise", False))
        self.inner_action_noise_type = self.inner_learning_starts.get("action_noise_type", "normal")
        self.inner_action_noise_params = self.inner_learning_starts.get("action_noise_params", {})

        # Outer loop: real environment interaction
        self.max_episode_steps = int(cp.get("max_episode_steps", 250))
        self.outer_train_freq = int(self.outer_alg_params.get("train_freq"))
        # learning starts for outer agent
        self.outer_learning_starts = self.outer_alg_params.get("learning_starts")
        self.outer_learning_starts_steps = self.outer_learning_starts.get("steps")
        self.outer_random_actions = bool(self.outer_learning_starts.get("random_actions", False))
        self.outer_use_action_noise = bool(self.outer_learning_starts.get("use_action_noise", False))
        self.outer_action_noise_type = self.outer_learning_starts.get("action_noise_type", "normal")
        self.outer_action_noise_params = self.outer_learning_starts.get("action_noise_params", {})
        self.render = bool(cp.get("render", False))

        # initialize outer agent
        self.outer_agent, _, _ = utils_core.initialize_alg(self.outer_alg_str, self.outer_alg_params, env)
        if not hasattr(self.outer_agent.model, "_logger"):
            self.outer_agent.model._logger = configure(folder=None, format_strings=[])

        # initialize inner env according to the outer env based on main.py
        print("Initializing AMBI inner env")
        alg_config = run_params["name"]        
        random.seed(run_params['seed']) #is it really correct to do this in the loop with the outer experiment seed?
        np.random.seed(run_params['seed'])
        if "env_params" in experiment_params.keys():
            self.inner_env = gym.make(run_params['env'], **experiment_params["env_params"])
        else:
            self.inner_env = gym.make(run_params['env']) #often overriden by experiment for consistency

        # handle custom wrappers:
        if "env_wrappers" in run_params:
            for env_wrapper in run_params["env_wrappers"]: #wrappers will be applied first to last in the order of the list
                if 'name' not in env_wrapper or env_wrapper['name'].split(':')[-1] not in SUPPORTED_WRAPPERS:
                    raise Exception("wrappers misconfigured, or otherwise not currently supported")
                wrapper_name = env_wrapper['name']
                wrapper_params = env_wrapper['wrapper_params']
                self.inner_env = setup_wrapper(self.inner_env, wrapper_name, wrapper_params)
        
        if "env_wrapper" in run_params:
            if 'name' not in run_params['env_wrapper'] or run_params['env_wrapper']['name'].split(':')[-1] not in SUPPORTED_WRAPPERS:
                raise Exception("wrapper misconfigured, or otherwise not currently supported")
            wrapper_name = run_params['env_wrapper']['name']
            wrapper_params = run_params['env_wrapper']["wrapper_params"]
        
            self.inner_env = setup_wrapper(self.inner_env, wrapper_name, wrapper_params)

        # reset inner and outer envs
        self.inner_env.reset()
        self.env.reset()
        
        self.alg_logger = None
        self.run = wandb_setup(custom_params, project="ambi_ant", run_name=f"AntAMBI-seed{custom_params.get('seed', 'NA')}")

    def get_model(self):
        return (
            self.outer_agent.get_model()
            if hasattr(self.outer_agent, "get_model")
            else self.outer_agent
        )

    def set_logger(self, logger):
        self.alg_logger = logger
        if hasattr(self.outer_agent, "set_logger"):
            self.outer_agent.set_logger(logger)

    def predict(self, obs):
        return self.outer_agent.predict(obs)

    def save(self, save_path, name):
        if hasattr(self.outer_agent, "save"):
            self.outer_agent.save(save_path, name)

    def load(self, load_path):
        if hasattr(self.outer_agent, "load"):
            self.outer_agent.load(load_path)

    @property
    def outer_env(self):
        return self.env

    def _initialize_learning_starts(self, layer="inner"):
        """Initialize replay buffer of the given layer with exploration."""
        assert layer in ["outer", "inner"], "Layer must be either 'outer' or 'inner'"

        agent = getattr(self, f"{layer}_agent")
        env = getattr(self, f"{layer}_env")
        total_steps = getattr(self, f"{layer}_learning_starts_steps")
        env_snapshot = _snapshot_env_state(env)

        # action parameters
        random_actions = getattr(self, f"{layer}_random_actions")
        use_action_noise = getattr(self, f"{layer}_use_action_noise")
        action_noise_type = getattr(self, f"{layer}_action_noise_type")
        action_noise_params = getattr(self, f"{layer}_action_noise_params")

        if use_action_noise:
            action_dim = env.action_space.shape[0]
            action_noise_params["mean"] = np.full(action_dim, action_noise_params["mean"])
            action_noise_params["sigma"] = np.full(action_dim, action_noise_params["sigma"])
            if action_noise_type == "normal":
                action_noise = NormalActionNoise(**action_noise_params)
            elif action_noise_type == "ornstein_uhlenbeck":
                action_noise = OrnsteinUhlenbeckActionNoise(**action_noise_params)
            else:
                raise ValueError(f"Invalid action noise type: {action_noise_type}")
        else:
            action_noise = None

        steps = 0
        while steps < total_steps:
            _set_env_state(env, env_snapshot)
            if action_noise is not None:
                action_noise.reset()
            obs = env.unwrapped._get_obs().copy()
            terminated = truncated = False
            while not (terminated or truncated) and steps < total_steps:
                if random_actions:
                    action = env.action_space.sample()
                else:
                    action, _ = agent.predict(obs)

                scaled_action = agent.model.policy.scale_action(action)
                if action_noise is not None:
                    scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
                    action = agent.model.policy.unscale_action(scaled_action)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                rew = np.array([reward], dtype=np.float32)
                dn = np.array([done], dtype=bool)
                agent.model.replay_buffer.add(obs, next_obs, scaled_action, rew, dn, [info]) # important to add scaled action to replay buffer
                obs = next_obs
                steps += 1
        # restore env state
        _set_env_state(env, env_snapshot)

    def _collect_inner_rollout(self, init_obs, max_steps):
        obs = init_obs
        cum_reward = 0.0
        terminated = truncated = False
        steps = 0

        while not (terminated or truncated) and steps < max_steps:
            action, _ = self.inner_agent.predict(obs)            
            next_obs, reward, terminated, truncated, info = self.inner_env.step(action)
            done = bool(terminated or truncated)
            rew = np.array([reward], dtype=np.float32)
            dn  = np.array([done], dtype=bool)
            buffer_action = self.inner_agent.model.policy.scale_action(action)
            self.inner_agent.model.replay_buffer.add(obs, next_obs, buffer_action, rew, dn, [info]) # important to add scaled action to replay buffer
            cum_reward += float(reward)
            obs = next_obs
            steps += 1

        return obs, cum_reward, steps, (terminated or truncated)

    def _initialize_inner_agent(self):
        inner_agent, _, _ = utils_core.initialize_alg(self.inner_alg_str, self.inner_alg_params, self.inner_env)
        inner_agent.model.policy.load_state_dict(self.outer_agent.model.policy.state_dict())
        if hasattr(inner_agent.model, "log_ent_coef") and hasattr(self.outer_agent.model, "log_ent_coef"):
            inner_agent.model.log_ent_coef.data.copy_(self.outer_agent.model.log_ent_coef.data)

        if self.use_lora:
            lorafy(inner_agent.model, **self.lora_params)

        if not hasattr(inner_agent.model, "_logger"):
            inner_agent.model._logger = configure(folder=None, format_strings=[])

        # for debugging
        # assert_sb3_weights_copied(self.outer_agent, inner_agent)

        return inner_agent


    def learn(self, total_timesteps: int = 10000):
        print(f"\n{'='*60}")
        print("AMBI TRAINING")
        print(
            f"Timesteps: {total_timesteps:,} | Inner rollouts: {self.inner_rollouts} | Max episode steps: {self.max_episode_steps}"
        )
        print(f"{'='*60}\n")
        print(f"Outer agent device: {self.outer_agent.model.device}")

        # wandb logging intervals
        wandb_step_every = int(self.outer_alg_params.get("wandb_step_every", 1)) if isinstance(self.outer_alg_params, dict) else 1
        wandb_inner_every = int(self.inner_alg_params.get("wandb_inner_every", wandb_step_every)) if isinstance(self.inner_alg_params, dict) else wandb_step_every

        start_time = time.time()
        it = episodes = 0

        # prepopulate outer replay buffer
        self._initialize_learning_starts("outer")

        while it < total_timesteps:
            # start new outer episode
            terminated = truncated = False
            outer_obs, outer_info = self.env.reset()
            episode_reward = 0.0
            episode_len = 0
            episode_start = time.time()

            # outer loop within episode
            while not (terminated or truncated) and it < total_timesteps:
                outer_step_start = time.time()

                # snapshot outer env at current real state
                outer_snapshot = _snapshot_env_state(self.env)

                # initialize inner agent and learning starts
                self.inner_agent = self._initialize_inner_agent()
                _set_env_state(self.inner_env, outer_snapshot) # need to set here so learning starts collects from correct time step
                self._initialize_learning_starts("inner")

                # logging purposes
                inner_returns = []
                inner_steps = []
                inner_total_start = time.time()
                inner_train_sec = 0.0

                # Run multiple imagined rollouts from current state
                inner_step_counter = 0 # counter for train_freq (needs to carry over across rollouts)
                for b in range(self.inner_rollouts):
                    # reset inner env to the outer env state
                    _set_env_state(self.inner_env, outer_snapshot)

                    # collect imagined rollout into inner replay buffer
                    rollout_return = 0.0
                    rollout_steps = 0
                    done = False
                    inner_obs = self.inner_env.unwrapped._get_obs().copy() # use inner obs readout from restored state

                    while not done: # finish one full rollout
                        step_to_collect = self.inner_train_freq - inner_step_counter
                        inner_obs, cum_reward, steps, done = self._collect_inner_rollout(inner_obs, max_steps=step_to_collect)
                        inner_step_counter += steps
                        rollout_return += float(cum_reward)
                        rollout_steps += int(steps)

                        # train inner
                        if inner_step_counter % self.inner_train_freq == 0:
                            gradient_steps = int(self.inner_alg_params.get("gradient_steps", 1))
                            batch_size = int(self.inner_alg_params.get("batch_size", 64))
                            self.inner_agent.model.train(
                                gradient_steps=gradient_steps,
                                batch_size=batch_size,
                            )
                            inner_step_counter = 0

                    inner_returns.append(rollout_return)
                    inner_steps.append(rollout_steps)

                    # for debugging
                    # before_train = utils_ambi_debug._snapshot_policy_tensors(self.inner_agent.model.policy)

                    ts = time.time()
                    inner_train_sec += (time.time() - ts)

                    # utils_ambi_debug.print_policy_update_report(
                    #     self.inner_agent.model.policy,
                    #     before_train,
                    #     tag=f"outer_it={it} rollout={b} gradient_steps={gradient_steps}",
                    #     atol=0.0,
                    #     rtol=0.0,
                    #     ignore_prefixes=(),
                    # )

                inner_total_sec = time.time() - inner_total_start
                inner_sim_steps = int(sum(inner_steps))

                # choose action for real env from inner policy
                outer_action, _ = self.inner_agent.predict(outer_obs)
                next_outer_obs, reward, terminated, truncated, info = self.env.step(outer_action)
                done = bool(terminated or truncated)
                rew = np.array([reward], dtype=np.float32)
                dn  = np.array([done], dtype=bool)
                buffer_action = self.outer_agent.model.policy.scale_action(outer_action)
                # store real transition into outer replay buffer
                self.outer_agent.model.replay_buffer.add(
                    outer_obs, next_outer_obs, buffer_action, rew, dn, [info]
                )

                # optional existing logger hook
                if self.alg_logger:
                    data = setup_logs(
                        reward,
                        outer_obs,
                        outer_action,
                        [done],
                        [info],
                        inner_steps=[inner_steps]
                    )
                    self.alg_logger.on_step(data)

                # wandb logging
                log_now_outer = (it % wandb_step_every == 0) or done
                log_now_inner = (it % wandb_inner_every == 0) or done
                if log_now_inner:
                    log_inner_summary(it, inner_returns, inner_steps if len(inner_steps) else None, inner_time_sec=inner_total_sec)

                if log_now_outer:
                    outer_step_sec = time.time() - outer_step_start
                    log_outer_step(it, reward, done, info=info, every=1, outer_step_sec=outer_step_sec)  # we control cadence above

                # progress print + global throughput
                if it % 100 == 0 and it > 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = it / elapsed if elapsed > 0 else 0.0
                    progress = 100.0 * it / float(total_timesteps)
                    eta_min = ((total_timesteps - it) / steps_per_sec / 60.0) if steps_per_sec > 0 else 0.0
                    print(
                        f"[{it:,}/{total_timesteps:,}] {progress:.1f}% | Episodes: {episodes} | "
                        f"{steps_per_sec:.1f} steps/s | ETA: {eta_min:.1f}m"
                    )

                # advance outer loop
                it += 1
                episode_len += 1
                episode_reward += float(reward)
                outer_obs = next_outer_obs

                if it % self.outer_train_freq == 0:
                    outer_gradient_steps = int(self.outer_alg_params.get("gradient_steps"))
                    outer_batch_size = int(self.outer_alg_params.get("batch_size"))
                    self.outer_agent.model.train(
                        gradient_steps=outer_gradient_steps,
                        batch_size=outer_batch_size,
                    )
                
            episode_sec = time.time() - episode_start

            # episode-level logging
            log_outer_episode(it, episodes, episode_reward, episode_len, episode_sec=episode_sec)

            print(
                f"Episode {episodes} complete | Steps: {episode_len} | Return: {episode_reward:.2f} | Timestep: {it:,}"
            )

            episodes += 1

            if self.render:
                self.env.render()

        print(f"\n{'='*60}")
        print(f"Training complete | {episodes} episodes | {it:,} timesteps")
        print(f"{'='*60}\n")

        wandb.finish()
        return self.outer_agent
