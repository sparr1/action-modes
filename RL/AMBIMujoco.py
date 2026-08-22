import copy, time, json, os, random
import numpy as np
from typing import Any, Mapping, Iterable

from RL.lora import lorafy
from RL.alg import Algorithm, validate_timestep_budget
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


def _add_cleanup_notes(primary_error, cleanup_errors):
    """Attach secondary cleanup failures without replacing the primary error."""
    for cleanup_error in cleanup_errors:
        note = f"Additional cleanup failure: {cleanup_error}"
        add_note = getattr(primary_error, "add_note", None)
        if callable(add_note):
            add_note(note)
        else:
            notes = list(getattr(primary_error, "__notes__", ()))
            notes.append(note)
            primary_error.__notes__ = notes


def _snapshot_env_state(env):
    """Capture the current state of the environment for later restoration."""
    unwrapped = env.unwrapped
    base_type = f"{type(unwrapped).__module__}.{type(unwrapped).__qualname__}"
    explicit_methods = (
        "training_resume_state",
        "validate_training_resume_state",
        "load_training_resume_state",
    )
    explicit_support = [
        callable(getattr(unwrapped, method, None)) for method in explicit_methods
    ]
    if any(explicit_support) and not all(explicit_support):
        raise TypeError(
            f"{base_type} only partially implements the environment-state protocol."
        )

    base_state = None
    if all(explicit_support):
        base_state = copy.deepcopy(unwrapped.training_resume_state())
        # Reject an incoherent source snapshot before copying its physics.
        unwrapped.validate_training_resume_state(copy.deepcopy(base_state))

    model, data = unwrapped.model, unwrapped.data

    # Get the full physics state of the environment (qpos, qvel, etc.)
    full_physics_spec = int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
    n = mujoco.mj_stateSize(model, full_physics_spec)
    x = np.empty(n, dtype=np.float64)
    mujoco.mj_getState(model, data, x, full_physics_spec)
    mujoco_state = x.copy()    


    wrapper_types = []
    current = env
    seen = set()
    while current is not unwrapped:
        if id(current) in seen or not hasattr(current, "env"):
            raise TypeError("Environment wrapper chain is cyclic or malformed.")
        seen.add(id(current))
        wrapper_types.append(
            f"{type(current).__module__}.{type(current).__qualname__}"
        )
        current = current.env

    # Read wrapper-owned fields directly. Gymnasium wrappers delegate missing
    # attributes, so getattr/hasattr on an outer wrapper can otherwise make a
    # field look as if it is owned by the wrong wrapper.
    elapsed = _get_first_wrapper_attr_direct(env, "_elapsed_steps")
    has_reset = _get_first_wrapper_attr_direct(env, "_has_reset")
    checked_step = _get_first_wrapper_attr_direct(env, "checked_step")
    checked_reset = _get_first_wrapper_attr_direct(env, "checked_reset")
    checked_render = _get_first_wrapper_attr_direct(env, "checked_render")

    return {
        "schema_version": 1,
        "base_type": base_type,
        "base_state": base_state,
        "wrapper_types": wrapper_types,
        "mujoco": {
            "state": mujoco_state,
            "spec": full_physics_spec,
            "size": int(n),
            "model_shape": {
                "nq": int(model.nq),
                "nv": int(model.nv),
                "na": int(model.na),
                "nu": int(model.nu),
            },
        },
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
    if not isinstance(state, Mapping) or state.get("schema_version") != 1:
        raise ValueError("Unsupported AMBI MuJoCo environment-state snapshot.")

    base_type = f"{type(unwrapped).__module__}.{type(unwrapped).__qualname__}"
    if state.get("base_type") != base_type:
        raise ValueError(
            "Cannot restore MuJoCo state into a different base environment: "
            f"saved={state.get('base_type')!r}, live={base_type!r}."
        )

    wrapper_types = []
    current = env
    seen = set()
    while current is not unwrapped:
        if id(current) in seen or not hasattr(current, "env"):
            raise TypeError("Environment wrapper chain is cyclic or malformed.")
        seen.add(id(current))
        wrapper_types.append(
            f"{type(current).__module__}.{type(current).__qualname__}"
        )
        current = current.env
    if state.get("wrapper_types") != wrapper_types:
        raise ValueError(
            "Cannot restore MuJoCo state across different wrapper stacks: "
            f"saved={state.get('wrapper_types')!r}, live={wrapper_types!r}."
        )

    mujoco_payload = state.get("mujoco")
    if not isinstance(mujoco_payload, Mapping):
        raise ValueError("MuJoCo environment snapshot is missing its physics payload.")
    mujoco_state = mujoco_payload.get("state")
    full_physics_spec = mujoco_payload.get("spec")
    saved_size = mujoco_payload.get("size")
    if (
        not isinstance(mujoco_state, np.ndarray)
        or mujoco_state.dtype != np.float64
        or mujoco_state.ndim != 1
        or isinstance(saved_size, bool)
        or not isinstance(saved_size, (int, np.integer))
        or int(saved_size) != mujoco_state.size
        or full_physics_spec != int(mujoco.mjtState.mjSTATE_FULLPHYSICS)
    ):
        raise ValueError("MuJoCo environment snapshot has an invalid physics vector.")

    base_state = state.get("base_state")
    explicit_methods = (
        "training_resume_state",
        "validate_training_resume_state",
        "load_training_resume_state",
    )
    explicit_support = [
        callable(getattr(unwrapped, method, None)) for method in explicit_methods
    ]
    if any(explicit_support) and not all(explicit_support):
        raise TypeError(
            f"{base_type} only partially implements the environment-state protocol."
        )
    if all(explicit_support):
        if base_state is None:
            raise ValueError(
                f"Snapshot for explicit-state environment {base_type} lacks base state."
            )
        # AntLegAdaptationEnv may rebuild model/data here. This must happen
        # before resolving either object or validating the physics-vector size.
        unwrapped.validate_training_resume_state(copy.deepcopy(base_state))
        unwrapped.load_training_resume_state(copy.deepcopy(base_state))
    elif base_state is not None:
        raise ValueError(
            f"Snapshot requires explicit-state support that {base_type} lacks."
        )

    model, data = unwrapped.model, unwrapped.data

    expected_size = int(mujoco.mj_stateSize(model, int(full_physics_spec)))
    if expected_size != int(saved_size):
        raise ValueError(
            "MuJoCo physics state is incompatible with the live model after "
            f"environment-state synchronization: saved={int(saved_size)}, "
            f"live={expected_size}."
        )
    live_shape = {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "na": int(model.na),
        "nu": int(model.nu),
    }
    if mujoco_payload.get("model_shape") != live_shape:
        raise ValueError(
            "MuJoCo model shape is incompatible after environment-state "
            f"synchronization: saved={mujoco_payload.get('model_shape')!r}, "
            f"live={live_shape!r}."
        )
    mujoco.mj_setState(model, data, mujoco_state, int(full_physics_spec))
    mujoco.mj_forward(model, data)

    # Restore wrappers
    w = state.get("wrappers")
    if not isinstance(w, Mapping):
        raise ValueError("MuJoCo environment snapshot lacks wrapper state.")

    if w.get("_elapsed_steps", None) is not None:
        ok = _set_first_wrapper_attr_direct(env, "_elapsed_steps", int(w["_elapsed_steps"]))
        if not ok:
            raise RuntimeError("Could not restore _elapsed_steps on inner env wrapper chain")

    if w.get("_has_reset", None) is not None:
        if not _set_first_wrapper_attr_direct(env, "_has_reset", bool(w["_has_reset"])):
            raise RuntimeError("Could not restore _has_reset on inner env wrapper chain")

    for k in ("checked_step", "checked_reset", "checked_render"):
        if w.get(k) is not None and not _set_first_wrapper_attr_direct(
            env, k, bool(w[k])
        ):
            raise RuntimeError(f"Could not restore {k} on inner env wrapper chain")


def _get_first_wrapper_attr_direct(env, name):
    """Return a field from its actual wrapper owner without delegated lookup."""
    cur = env
    seen = set()
    while True:
        if name in vars(cur):
            return vars(cur)[name]
        if not hasattr(cur, "env"):
            return None
        nxt = cur.env
        if id(nxt) in seen:
            raise TypeError("Environment wrapper chain is cyclic.")
        seen.add(id(nxt))
        cur = nxt


def _set_first_wrapper_attr_direct(env, name, value):
    """
    Walk wrapper chain and set `name` on the first wrapper/object that has it.
    Returns True if set, False otherwise.
    """
    cur = env
    seen = set()
    while True:
        if name in vars(cur):
            setattr(cur, name, value)
            return True
        if not hasattr(cur, "env"):
            return False
        nxt = cur.env
        if id(nxt) in seen:
            return False
        seen.add(id(nxt))
        cur = nxt


def _apply_configured_time_limit(
    terminated, truncated, info, *, next_episode_step, max_episode_steps
):
    """Apply AMBI's episode cap and preserve timeout semantics in replay data."""
    terminated = bool(terminated)
    truncated = bool(truncated)
    if next_episode_step >= max_episode_steps and not terminated:
        truncated = True

    normalized_info = dict(info or {})
    normalized_info["terminated"] = terminated
    normalized_info["truncated"] = truncated
    # SB3 uses this compatibility key to bootstrap pure timeouts. A
    # simultaneous task termination remains a true terminal transition.
    normalized_info["TimeLimit.truncated"] = truncated and not terminated
    return truncated, normalized_info


def _normalized_action_noise_params(params, action_dim):
    """Return per-action noise vectors without mutating configured parameters."""
    if not isinstance(params, Mapping):
        raise TypeError("action_noise_params must be a mapping")

    normalized = copy.deepcopy(dict(params))
    for name in ("mean", "sigma"):
        if name not in normalized:
            raise ValueError(f"action_noise_params must define {name!r}")
        value = np.asarray(normalized[name])
        if value.ndim == 0 or value.size == 1:
            normalized[name] = np.full(action_dim, value.reshape(-1)[0])
        elif value.ndim == 1 and value.shape[0] == action_dim:
            normalized[name] = value.copy()
        else:
            raise ValueError(
                f"action_noise_params[{name!r}] must be scalar or have shape "
                f"({action_dim},), got {value.shape}."
            )
    return normalized


def _positive_integer_setting(value, name):
    value = validate_timestep_budget(value, name=name)
    if value == 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _learning_starts_settings(algorithm_params, layer):
    learning_starts = algorithm_params.get("learning_starts")
    if not isinstance(learning_starts, Mapping):
        raise ValueError(f"{layer}_alg_params.learning_starts must be a mapping.")
    steps = validate_timestep_budget(
        learning_starts.get("steps"),
        name=f"{layer}_alg_params.learning_starts.steps",
    )
    return learning_starts, steps


def wandb_setup(cfg: dict, project="ambi", run_name=None):
    entity = cfg.get("wandb_entity", "rwgao_b-brown-university") if isinstance(cfg, dict) else "rwgao_b-brown-university"
    project = cfg.get("wandb_project", project) if isinstance(cfg, dict) else project
    mode = cfg.get("wandb_mode", None) if isinstance(cfg, dict) else None
    tags = cfg.get("wandb_tags", None) if isinstance(cfg, dict) else None
    kwargs = {
        "entity": entity,
        "project": project,
        "name": run_name,
        "config": cfg,
    }
    if mode:
        kwargs["mode"] = mode
    if tags:
        kwargs["tags"] = tags
    run = wandb.init(**kwargs)
    try:
        # Make outer timestep the shared x-axis
        wandb.define_metric("outer/step")
        wandb.define_metric("outer/*", step_metric="outer/step")
        wandb.define_metric("inner/*", step_metric="outer/step")
        wandb.define_metric("time/*", step_metric="outer/step")
    except BaseException as exc:
        cleanup_errors = []
        try:
            finish = getattr(run, "finish", None)
            if callable(finish):
                finish()
            else:
                wandb.finish()
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        _add_cleanup_notes(exc, cleanup_errors)
        raise
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

        for layer, params in (
            ("outer", self.outer_alg_params),
            ("inner", self.inner_alg_params),
        ):
            if not isinstance(params, Mapping):
                raise ValueError(f"{layer}_alg_params must be a mapping.")

        assert self.outer_alg_str is not None, "AMBI requires 'outer_alg' string in custom_params"

        # Inner loop: imagined rollouts from current state
        self.inner_rollouts = validate_timestep_budget(
            cp.get("inner_rollouts", 6), name="inner_rollouts"
        )
        self.inner_reinit_every_step = bool(cp.get("inner_reinit_every_step", True))
        self.inner_train_freq = _positive_integer_setting(
            self.inner_alg_params.get("train_freq"),
            "inner_alg_params.train_freq",
        )
        self.inner_gradient_steps = validate_timestep_budget(
            self.inner_alg_params.get("gradient_steps", 1),
            name="inner_alg_params.gradient_steps",
        )
        self.inner_batch_size = _positive_integer_setting(
            self.inner_alg_params.get("batch_size", 64),
            "inner_alg_params.batch_size",
        )
        self.inner_updates_per_rollout = validate_timestep_budget(
            cp.get("inner_updates_per_rollout", 1),
            name="inner_updates_per_rollout",
        )
        self.use_lora = bool(cp.get("use_lora", False))
        self.lora_params = cp.get("lora_params", {})
        # learning starts for inner agent
        (
            self.inner_learning_starts,
            self.inner_learning_starts_steps,
        ) = _learning_starts_settings(self.inner_alg_params, "inner")
        self.inner_random_actions = bool(self.inner_learning_starts.get("random_actions", False))
        self.inner_use_action_noise = bool(self.inner_learning_starts.get("use_action_noise", False))
        self.inner_action_noise_type = self.inner_learning_starts.get("action_noise_type", "normal")
        self.inner_action_noise_params = self.inner_learning_starts.get("action_noise_params", {})

        # Outer loop: real environment interaction
        self.max_episode_steps = _positive_integer_setting(
            cp.get("max_episode_steps", 250), "max_episode_steps"
        )
        self.outer_train_freq = _positive_integer_setting(
            self.outer_alg_params.get("train_freq"),
            "outer_alg_params.train_freq",
        )
        self.outer_gradient_steps = validate_timestep_budget(
            self.outer_alg_params.get("gradient_steps", 1),
            name="outer_alg_params.gradient_steps",
        )
        self.outer_batch_size = _positive_integer_setting(
            self.outer_alg_params.get("batch_size", 64),
            "outer_alg_params.batch_size",
        )
        # learning starts for outer agent
        (
            self.outer_learning_starts,
            self.outer_learning_starts_steps,
        ) = _learning_starts_settings(self.outer_alg_params, "outer")
        self.outer_random_actions = bool(self.outer_learning_starts.get("random_actions", False))
        self.outer_use_action_noise = bool(self.outer_learning_starts.get("use_action_noise", False))
        self.outer_action_noise_type = self.outer_learning_starts.get("action_noise_type", "normal")
        self.outer_action_noise_params = self.outer_learning_starts.get("action_noise_params", {})
        self.render = bool(cp.get("render", False))

        # initialize outer agent
        self.outer_agent, _, _ = utils_core.initialize_alg(self.outer_alg_str, self.outer_alg_params, env)
        if not hasattr(self.outer_agent.model, "_logger"):
            self.outer_agent.model._logger = configure(folder=None, format_strings=[])

        self.inner_env = None
        self.inner_agent = None
        self.alg_logger = None
        self.run = None
        try:
            # initialize inner env according to the outer env based on main.py
            print("Initializing AMBI inner env")
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
            self.run = wandb_setup(
                cp,
                project="ambi",
                run_name=f"AntAMBI-seed{cp.get('seed', 'NA')}",
            )
        except BaseException as exc:
            cleanup_errors = []
            try:
                self.close()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            _add_cleanup_notes(exc, cleanup_errors)
            raise

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
            return self.outer_agent.save(save_path, name)
        return None

    def load(self, load_path):
        if hasattr(self.outer_agent, "load"):
            self.outer_agent.load(load_path)

    @property
    def outer_env(self):
        return self.env

    def _finish_run(self):
        run = getattr(self, "run", None)
        if run is None:
            return
        self.run = None
        finish = getattr(run, "finish", None)
        if callable(finish):
            finish()
        else:
            wandb.finish()

    def close(self):
        """Close independently-owned W&B and planning resources exactly once."""
        inner_env = getattr(self, "inner_env", None)
        self.inner_env = None
        cleanup_errors = []
        try:
            self._finish_run()
        except BaseException as exc:
            cleanup_errors.append(exc)
        if inner_env is not None:
            try:
                inner_env.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            primary_error = cleanup_errors[0]
            _add_cleanup_notes(primary_error, cleanup_errors[1:])
            raise primary_error

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
        action_noise_params = copy.deepcopy(
            getattr(self, f"{layer}_action_noise_params")
        )

        if use_action_noise:
            action_dim = env.action_space.shape[0]
            action_noise_params = _normalized_action_noise_params(
                action_noise_params, action_dim
            )
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
            rollout_steps = 0
            snapshot_elapsed = env_snapshot["wrappers"].get("_elapsed_steps")
            snapshot_elapsed = int(snapshot_elapsed or 0)
            rollout_limit = self.max_episode_steps - snapshot_elapsed
            if rollout_limit <= 0:
                raise ValueError(
                    "Cannot initialize replay from an environment snapshot at or "
                    "beyond max_episode_steps."
                )
            while (
                not (terminated or truncated)
                and steps < total_steps
                and rollout_steps < rollout_limit
            ):
                if random_actions:
                    action = env.action_space.sample()
                else:
                    action, _ = agent.predict(obs)

                scaled_action = agent.model.policy.scale_action(action)
                if action_noise is not None:
                    scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
                    action = agent.model.policy.unscale_action(scaled_action)

                next_obs, reward, terminated, truncated, info = env.step(action)
                rollout_steps += 1
                truncated, info = _apply_configured_time_limit(
                    terminated,
                    truncated,
                    info,
                    next_episode_step=rollout_steps,
                    max_episode_steps=rollout_limit,
                )
                done = bool(terminated or truncated)
                rew = np.array([reward], dtype=np.float32)
                dn = np.array([done], dtype=bool)
                agent.model.replay_buffer.add(obs, next_obs, scaled_action, rew, dn, [info]) # important to add scaled action to replay buffer
                obs = next_obs
                steps += 1
        # restore env state
        _set_env_state(env, env_snapshot)

    def _collect_inner_rollout(self, init_obs, max_steps, *, truncate_on_limit=False):
        obs = init_obs
        cum_reward = 0.0
        terminated = truncated = False
        steps = 0

        while not (terminated or truncated) and steps < max_steps:
            action, _ = self.inner_agent.predict(obs)            
            next_obs, reward, terminated, truncated, info = self.inner_env.step(action)
            if truncate_on_limit:
                truncated, info = _apply_configured_time_limit(
                    terminated,
                    truncated,
                    info,
                    next_episode_step=steps + 1,
                    max_episode_steps=max_steps,
                )
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

    def _prepare_inner_agent(self, outer_snapshot):
        """Create or reuse the inner agent, then align its planning state."""
        reinitialize = self.inner_agent is None or self.inner_reinit_every_step
        if reinitialize:
            self.inner_agent = self._initialize_inner_agent()

        _set_env_state(self.inner_env, outer_snapshot)
        if reinitialize:
            self._initialize_learning_starts("inner")
        return self.inner_agent


    def learn(self, total_timesteps: int = 10000):
        total_timesteps = validate_timestep_budget(total_timesteps)
        print(f"\n{'='*60}")
        print("AMBI TRAINING")
        print(
            f"Timesteps: {total_timesteps:,} | Inner rollouts: {self.inner_rollouts} | Max episode steps: {self.max_episode_steps}"
        )
        print(f"{'='*60}\n")
        print(f"Outer agent device: {self.outer_agent.model.device}")

        # wandb logging intervals
        wandb_step_every = _positive_integer_setting(
            self.outer_alg_params.get("wandb_step_every", 1),
            "outer_alg_params.wandb_step_every",
        )
        wandb_inner_every = _positive_integer_setting(
            self.inner_alg_params.get("wandb_inner_every", wandb_step_every),
            "inner_alg_params.wandb_inner_every",
        )

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
                if self.inner_rollouts > 0:
                    self._prepare_inner_agent(outer_snapshot)
                else:
                    self.inner_agent = None

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
                    rollout_limit = self.max_episode_steps - episode_len

                    while not done and rollout_steps < rollout_limit: # finish one full rollout
                        rollout_remaining = rollout_limit - rollout_steps
                        step_to_collect = min(
                            self.inner_train_freq - inner_step_counter,
                            rollout_remaining,
                        )
                        inner_obs, cum_reward, steps, done = self._collect_inner_rollout(
                            inner_obs,
                            max_steps=step_to_collect,
                            truncate_on_limit=step_to_collect == rollout_remaining,
                        )
                        inner_step_counter += steps
                        rollout_return += float(cum_reward)
                        rollout_steps += int(steps)

                        # train inner
                        if inner_step_counter % self.inner_train_freq == 0:
                            self.inner_agent.model.train(
                                gradient_steps=self.inner_gradient_steps,
                                batch_size=self.inner_batch_size,
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
                if self.inner_rollouts > 0:
                    outer_action, _ = self.inner_agent.predict(outer_obs)
                else:
                    outer_action, _ = self.outer_agent.predict(outer_obs)
                next_outer_obs, reward, terminated, truncated, info = self.env.step(outer_action)
                truncated, info = _apply_configured_time_limit(
                    terminated,
                    truncated,
                    info,
                    next_episode_step=episode_len + 1,
                    max_episode_steps=self.max_episode_steps,
                )
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
                        next_outer_obs,
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
                    self.outer_agent.model.train(
                        gradient_steps=self.outer_gradient_steps,
                        batch_size=self.outer_batch_size,
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

        self._finish_run()
        return self.outer_agent
