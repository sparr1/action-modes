import copy, time
import numpy as np
from stable_baselines3.common.logger import configure

import torch.nn as nn
from RL.alg import Algorithm
from utils import core as utils_core
from utils.utils import setup_logs
from utils.core import *
import torch
import mujoco
import gymnasium as gym
import random

######################################################## Testing code ########################################################

def assert_sb3_weights_copied(
    outer_agent,
    inner_agent,
    obs_sample=None,
    *,
    check_action=False,
    deterministic=True,
    atol=0.0,
    rtol=0.0,
):
    def _unwrap_sb3_algo(agent):
        # Handles your Baseline wrapper and raw SB3 objects
        return agent.model if hasattr(agent, "model") else agent

    def _get_policy(agent):
        algo = _unwrap_sb3_algo(agent)
        if not hasattr(algo, "policy"):
            raise TypeError(
                f"Expected an SB3 algorithm with `.policy`, got: {type(algo)}"
            )
        return algo.policy

    outer_algo = _unwrap_sb3_algo(outer_agent)
    inner_algo = _unwrap_sb3_algo(inner_agent)
    outer_policy = _get_policy(outer_agent)
    inner_policy = _get_policy(inner_agent)

    errs = []

    # 1) Compare policy state_dict keys
    sd_out = outer_policy.state_dict()
    sd_in = inner_policy.state_dict()

    keys_out = list(sd_out.keys())
    keys_in = list(sd_in.keys())

    if keys_out != keys_in:
        missing_in = [k for k in keys_out if k not in sd_in]
        extra_in = [k for k in keys_in if k not in sd_out]
        raise AssertionError(
            "Policy state_dict key mismatch.\n"
            f"Missing in inner: {missing_in[:20]}\n"
            f"Extra in inner: {extra_in[:20]}"
        )

    # 2) Compare each tensor/buffer
    for k in keys_out:
        a = sd_out[k].detach().cpu()
        b = sd_in[k].detach().cpu()

        if a.shape != b.shape:
            errs.append(f"{k}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
            continue

        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            diff = (a - b).abs()
            max_diff = diff.max().item()
            idx = int(diff.view(-1).argmax().item())
            av = a.view(-1)[idx].item()
            bv = b.view(-1)[idx].item()
            errs.append(
                f"{k}: max_abs_diff={max_diff:.3e} at flat_idx={idx} "
                f"(outer={av:.9g}, inner={bv:.9g})"
            )

    # 3) Optional: compare predicted actions on same obs
    if check_action:
        if obs_sample is None:
            raise ValueError("obs_sample must be provided when check_action=True")

        # convert torch obs to numpy if needed
        if torch.is_tensor(obs_sample):
            obs_np = obs_sample.detach().cpu().numpy()
        else:
            obs_np = np.array(obs_sample, copy=False)

        # IMPORTANT: call predict on the SB3 algo object, not your wrapper (wrapper may not accept kwargs)
        act_out, _ = outer_algo.predict(obs_np, deterministic=deterministic)
        act_in, _ = inner_algo.predict(obs_np, deterministic=deterministic)

        if not np.allclose(act_out, act_in, atol=max(atol, 1e-7), rtol=max(rtol, 1e-6)):
            d = np.abs(act_out - act_in)
            idx = int(np.argmax(d))
            errs.append(
                "predict() action mismatch: "
                f"max_abs_diff={d.flat[idx]:.3e} at flat_idx={idx} "
                f"(outer={act_out.flat[idx]:.9g}, inner={act_in.flat[idx]:.9g})"
            )

    if errs:
        raise AssertionError(
            "Outer/inner SB3 weights do NOT match after copy:\n- " + "\n- ".join(errs[:50])
        )

    return True

def _debug_wrapper_type_chain(env):
    """Return wrapper -> ... -> base env type names."""
    types = []
    cur = env
    seen = set()
    while True:
        types.append(type(cur).__name__)
        if not hasattr(cur, "env"):
            break
        nxt = cur.env
        if id(nxt) in seen:  # just in case of a weird cycle
            types.append("<cycle>")
            break
        seen.add(id(nxt))
        cur = nxt
    return types


def _debug_get_wrapper_attr(env, name, default=None):
    try:
        if hasattr(env, "get_wrapper_attr"):
            return env.get_wrapper_attr(name)
    except Exception:
        pass
    return default


def _debug_get_full_mujoco_state(env):
    """Flattened MuJoCo FULLPHYSICS state as float64."""
    uw = env.unwrapped
    model, data = uw.model, uw.data
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    n = mujoco.mj_stateSize(model, spec)
    x = np.empty(n, dtype=np.float64)
    mujoco.mj_getState(model, data, x, spec)
    return x


def assert_envs_match_after_copy(
    outer_env,
    inner_env,
    outer_obs=None,
    *,
    check_wrapper_stack=False,
    check_outer_obs_vs_inner_raw=False,
    atol=1e-6,
    rtol=1e-5,
):
    errs = []

    if check_wrapper_stack:
        outer_chain = _debug_wrapper_type_chain(outer_env)
        inner_chain = _debug_wrapper_type_chain(inner_env)
        if outer_chain != inner_chain:
            errs.append(
                "Wrapper stack mismatch:\n"
                f"  outer={outer_chain}\n"
                f"  inner={inner_chain}"
            )

    try:
        if outer_env.observation_space.shape != inner_env.observation_space.shape:
            errs.append(
                f"Observation space shape mismatch: "
                f"{outer_env.observation_space.shape} vs {inner_env.observation_space.shape}"
            )
    except Exception as e:
        errs.append(f"Could not compare observation spaces: {e}")

    try:
        if outer_env.action_space.shape != inner_env.action_space.shape:
            errs.append(
                f"Action space shape mismatch: "
                f"{outer_env.action_space.shape} vs {inner_env.action_space.shape}"
            )
    except Exception as e:
        errs.append(f"Could not compare action spaces: {e}")

    try:
        s_outer = _debug_get_full_mujoco_state(outer_env)
        s_inner = _debug_get_full_mujoco_state(inner_env)
        if s_outer.shape != s_inner.shape:
            errs.append(f"MuJoCo state shape mismatch: {s_outer.shape} vs {s_inner.shape}")
        else:
            if not np.allclose(s_outer, s_inner, atol=atol, rtol=rtol):
                diff = np.abs(s_outer - s_inner)
                idx = int(np.argmax(diff))
                errs.append(
                    "MuJoCo FULLPHYSICS mismatch: "
                    f"max_abs_diff={diff[idx]:.3e} at index {idx} "
                    f"(outer={s_outer[idx]:.9g}, inner={s_inner[idx]:.9g})"
                )
    except Exception as e:
        errs.append(f"Could not compare MuJoCo full state: {e}")

    try:
        qo = np.array(outer_env.unwrapped.data.qpos, copy=True)
        qi = np.array(inner_env.unwrapped.data.qpos, copy=True)
        if not np.allclose(qo, qi, atol=atol, rtol=rtol):
            d = np.abs(qo - qi)
            idx = int(np.argmax(d))
            errs.append(
                f"qpos mismatch: max_abs_diff={d[idx]:.3e} at index {idx} "
                f"(outer={qo[idx]:.9g}, inner={qi[idx]:.9g})"
            )
    except Exception as e:
        errs.append(f"Could not compare qpos: {e}")

    try:
        vo = np.array(outer_env.unwrapped.data.qvel, copy=True)
        vi = np.array(inner_env.unwrapped.data.qvel, copy=True)
        if not np.allclose(vo, vi, atol=atol, rtol=rtol):
            d = np.abs(vo - vi)
            idx = int(np.argmax(d))
            errs.append(
                f"qvel mismatch: max_abs_diff={d[idx]:.3e} at index {idx} "
                f"(outer={vo[idx]:.9g}, inner={vi[idx]:.9g})"
            )
    except Exception as e:
        errs.append(f"Could not compare qvel: {e}")

    for k in ("_elapsed_steps", "_has_reset", "checked_step", "checked_reset", "checked_render"):
        ov = _debug_get_wrapper_attr(outer_env, k, default="<missing>")
        iv = _debug_get_wrapper_attr(inner_env, k, default="<missing>")
        if ov != iv:
            errs.append(f"Wrapper attr mismatch for {k}: outer={ov!r}, inner={iv!r}")

    try:
        if hasattr(outer_env.unwrapped, "_get_obs") and hasattr(inner_env.unwrapped, "_get_obs"):
            raw_outer = np.array(outer_env.unwrapped._get_obs(), copy=True)
            raw_inner = np.array(inner_env.unwrapped._get_obs(), copy=True)
            if raw_outer.shape != raw_inner.shape:
                errs.append(f"raw _get_obs shape mismatch: {raw_outer.shape} vs {raw_inner.shape}")
            elif not np.allclose(raw_outer, raw_inner, atol=atol, rtol=rtol):
                d = np.abs(raw_outer - raw_inner)
                idx = int(np.argmax(d))
                errs.append(
                    f"raw _get_obs mismatch: max_abs_diff={d[idx]:.3e} at index {idx} "
                    f"(outer={raw_outer[idx]:.9g}, inner={raw_inner[idx]:.9g})"
                )

            if outer_obs is not None and check_outer_obs_vs_inner_raw:
                outer_obs_arr = np.array(outer_obs, copy=False)
                if outer_obs_arr.shape != raw_inner.shape:
                    errs.append(
                        f"outer_obs vs inner raw obs shape mismatch: "
                        f"{outer_obs_arr.shape} vs {raw_inner.shape}"
                    )
                elif not np.allclose(outer_obs_arr, raw_inner, atol=atol, rtol=rtol):
                    d = np.abs(outer_obs_arr - raw_inner)
                    idx = int(np.argmax(d))
                    errs.append(
                        f"outer_obs vs inner raw obs mismatch: max_abs_diff={d[idx]:.3e} at index {idx} "
                        f"(outer_obs={outer_obs_arr[idx]:.9g}, inner_raw={raw_inner[idx]:.9g})"
                    )
    except Exception as e:
        errs.append(f"Could not compare raw observations: {e}")

    if errs:
        raise AssertionError("Env copy mismatch after _set_env_state:\n- " + "\n- ".join(errs))

######################################################## End testing code ########################################################

def _snapshot_env_state(env):
    """Capture the current state of the environment for later restoration."""
    unwrapped = env.unwrapped
    model, data = unwrapped.model, unwrapped.data

    # Get the full physics state of the environment (qpos, qvel, etc.)
    full_physics_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
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
    checked_render= env.get_wrapper_attr("checked_render") if hasattr(env, "get_wrapper_attr") else True

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
    mujoco.mj_setState(model, data, mujoco_state, full_physics_spec)
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
            - seed_episodes: Number of random episodes to initialize buffer (default: 0)
    """

    def __init__(self, name, env, custom_params=None, run_params=None, experiment_params=None):
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
        self.inner_env.reset()

        if self.seed_episodes > 0:
            self._initialize_seed_episodes()

        self.alg_logger = None

    def get_model(self):
        return (
            self.outer_agent.get_model()
            if hasattr(self.outer_agent, "get_model")
            else self.outer_agent
        )

    def _initialize_seed_episodes(self):
        """Initialize outer agent replay buffer with random exploration."""
        print(f"Initializing {self.seed_episodes} seed episodes...")
        total_transitions = 0
        for _ in range(self.seed_episodes):
            obs, _ = self.env.reset()
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = self.outer_agent.predict(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = bool(terminated or truncated)
                self.outer_agent.model.replay_buffer.add(obs, next_obs, action, reward, done, [info])
                obs = next_obs
                total_transitions += 1
        print(f"Buffer initialized with {total_transitions} transitions")

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

    def _collect_inner_rollout(self, init_obs):
        obs = init_obs
        cum_reward = 0.0
        terminated = truncated = False
        counter = 0
        while not (terminated or truncated):
            action, _ = self.inner_agent.predict(obs)            
            next_obs, reward, terminated, truncated, info = self.inner_env.step(action)
            done = bool(terminated or truncated)
            self.inner_agent.model.replay_buffer.add(
                obs, 
                next_obs, 
                action, 
                reward, 
                done, 
                [info]
            )
            cum_reward += float(reward)
            obs = next_obs
        
        return obs, cum_reward

    def _initialize_inner_agent(self):
        inner_agent, _, _ = utils_core.initialize_alg(self.inner_alg_str, self.inner_alg_params, self.inner_env)
        inner_agent.model.policy.load_state_dict(self.outer_agent.model.policy.state_dict())

        if not hasattr(inner_agent.model, "_logger"):
            inner_agent.model._logger = configure(folder=None, format_strings=[])

        # for debugging
        # assert_sb3_weights_copied(self.outer_agent, inner_agent)

        return inner_agent


    def learn(self, total_timesteps=10000):
        """
        Train the AMBI agent.

        Uses inner loop imagination to improve action selection and outer loop
        real experience to train the policy.

        Args:
            total_timesteps: Total number of timesteps to train for
        """
        print(f"\n{'='*60}")
        print("AMBI TRAINING")
        print(
            f"Timesteps: {total_timesteps:,} | Inner rollouts: {self.inner_rollouts} | Max episode steps: {self.max_episode_steps}"
        )
        print(f"{'='*60}\n")

        iter = 0
        episodes = 0
        episode_reward = 0.0

        start_time = time.time()
        # while not converged
        while iter < total_timesteps:
            terminated = truncated = False
            outer_obs, outer_info = self.env.reset()

            # outer loop t = 1...T
            t = 0
            while not (terminated or truncated):
                # get outer env state
                outer_snapshot = _snapshot_env_state(self.env)

                # initialize inner agent
                self.inner_agent = self._initialize_inner_agent()

                # Run multiple imagined rollouts from current state
                for b in range(self.inner_rollouts):
                    # set inner env to the outer env state
                    _set_env_state(self.inner_env, outer_snapshot)

                    # for debugging
                    # assert_envs_match_after_copy(
                    #     self.env,
                    #     self.inner_env,
                    #     outer_obs=outer_obs,
                    #     check_wrapper_stack=(t == 0 and b == 0),
                    #     check_outer_obs_vs_inner_raw=True,               
                    # )

                    # this should match outer obs, using inner for safety
                    inner_obs0 = self.inner_env.unwrapped._get_obs().copy()

                    # Run imagined rollout and collect experience
                    final_obs, cum_reward = self._collect_inner_rollout(inner_obs0)

                    # Update inner agent on imagined experience 
                    gradient_steps = self.inner_alg_params.get("gradient_steps", 1)
                    batch_size = self.inner_alg_params.get("batch_size", 64)
                    self.inner_agent.model.train(gradient_steps=gradient_steps, batch_size=batch_size)

                # take action in outer env
                outer_action, _ = self.inner_agent.predict(outer_obs)
                next_outer_obs, reward, terminated, truncated, info = self.env.step(outer_action)
                done = bool(terminated or truncated)
                self.outer_agent.model.replay_buffer.add(outer_obs, next_outer_obs, outer_action, reward, done, [info])

                # log step data
                if self.alg_logger:
                    data = setup_logs(
                        reward,
                        outer_obs,
                        outer_action,
                        [done],
                        [info,],
                    )

                    self.alg_logger.on_step(data)

                # Print progress every 100 timesteps
                if iter % 100 == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = iter / elapsed if elapsed > 0 else 0
                    progress = 100 * iter / total_timesteps
                    eta_min = (
                        (total_timesteps - iter) / steps_per_sec / 60
                        if steps_per_sec > 0
                        else 0
                    )
                    print(
                        f"[{iter:,}/{total_timesteps:,}] {progress:.1f}% | Episodes: {episodes} | {steps_per_sec:.1f} steps/s | ETA: {eta_min:.1f}m"
                    )

                iter += 1
                t += 1
                episode_reward += reward
                outer_obs = next_outer_obs

            # update outer agent
            gradient_steps = self.outer_alg_params.get("gradient_steps", 1)
            batch_size = self.outer_alg_params.get("batch_size", 64)
            self.outer_agent.model.train(gradient_steps=gradient_steps, batch_size=batch_size)

            print(
                f"Episode {episodes} complete | Steps: {iter} | Return: {episode_reward:.2f} | Timestep: {iter:,}"
            )

            episodes += 1
            episode_reward = 0.0

            if self.render:
                self.env.render()


        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(
            f"Training complete | {total_time:.1f}s | {episodes} episodes | {t:,} timesteps"
        )
        print(f"{'='*60}\n")
        return self.outer_agent
