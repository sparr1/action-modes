import copy, time, json, os, random
import numpy as np
from typing import Any, Mapping, Iterable

from RL.lora import lorafy
from RL.alg import Algorithm
from utils import core as utils_core
from utils.utils import setup_logs
from utils.core import *

import torch.nn as nn
import torch
import mujoco
import gymnasium as gym
import wandb
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


######################################################## Testing code ########################################################

def state_dicts_equal(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    exact: bool = False,
    ignore_keys: Iterable[str] = (),
) -> bool:
    ignore = set(ignore_keys)
    a_keys = set(a.keys()) - ignore
    b_keys = set(b.keys()) - ignore
    if a_keys != b_keys:
        return False

    for k in a_keys:
        va, vb = a[k], b[k]
        if torch.is_tensor(va) and torch.is_tensor(vb):
            xa = va.detach()
            xb = vb.detach()
            if xa.shape != xb.shape or xa.dtype != xb.dtype:
                return False
            if exact:
                if not torch.equal(xa, xb):
                    return False
            else:
                if not torch.allclose(xa, xb, rtol=rtol, atol=atol, equal_nan=True):
                    return False
        else:
            if va != vb:
                return False
    return True

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

        assert self.outer_alg_str is not None, "AMBI requires 'outer_alg' string in custom_params"

        # Inner loop: imagined rollouts from current state
        self.inner_rollouts = int(cp.get("inner_rollouts", 6))
        self.inner_reinit_every_step = bool(cp.get("inner_reinit_every_step", True))
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

    def _collect_inner_rollout(self, init_obs):
        obs = init_obs
        cum_reward = 0.0
        terminated = truncated = False
        steps = 0

        while not (terminated or truncated):
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
        return obs, cum_reward, steps

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
                for b in range(self.inner_rollouts):
                    # reset inner env to the outer env state
                    _set_env_state(self.inner_env, outer_snapshot)

                    # use inner obs readout from restored state
                    inner_obs0 = self.inner_env.unwrapped._get_obs().copy()

                    # collect imagined rollout into inner replay buffer
                    _, cum_reward, steps = self._collect_inner_rollout(inner_obs0)

                    inner_returns.append(float(cum_reward))
                    inner_steps.append(int(steps))

                    # train inner after each rollout 
                    gradient_steps = int(self.inner_alg_params.get("gradient_steps", 1))
                    batch_size = int(self.inner_alg_params.get("batch_size", 64))

                    ts = time.time()
                    self.inner_agent.model.train(
                        gradient_steps=gradient_steps,
                        batch_size=batch_size,
                    )
                    inner_train_sec += (time.time() - ts)

                inner_total_sec = time.time() - inner_total_start
                inner_sim_steps = int(sum(inner_steps))

                # choose action for real env from inner policy
                outer_action, _ = self.inner_agent.predict(outer_obs)
                next_outer_obs, reward, terminated, truncated, info = self.env.step(outer_action)
                done = bool(terminated or truncated)
                rew = np.array([reward], dtype=np.float32)
                dn  = np.array([done], dtype=bool)
                # store real transition into outer replay buffer
                self.outer_agent.model.replay_buffer.add(
                    outer_obs, next_outer_obs, outer_action, rew, dn, [info]
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

            # end of outer episode, train outer agent
            outer_train_start = time.time()
            outer_gradient_steps = int(self.outer_alg_params.get("gradient_steps", 1))
            outer_batch_size = int(self.outer_alg_params.get("batch_size", 64))

            self.outer_agent.model.train(
                gradient_steps=outer_gradient_steps,
                batch_size=outer_batch_size,
            )
            outer_train_sec = time.time() - outer_train_start
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
