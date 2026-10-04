"""Restorable real-simulator branches for inner-SAC transfer diagnostics.

These helpers do not optimize policies. Actor callbacks receive batched real
observations, explicit Gaussian noise, and remaining prefix horizon. They must
encode each new observation and return actions in the environment's units.
The forced first action carries no entropy term, matching Q(s, a). A soft
prefix includes entropy on its subsequent actions only. The terminal critic's
reward/soft convention and optional entropy on its first action are separately
specified to match the inner learner's configured frozen-Q boundary.

Built-in exact simulator capture is reviewed for Humanoid Walk state
observations. Other environments may implement the explicit calibration_state /
load_calibration_state / enable_continuing_calibration protocol. This is a
diagnostic utility, not a change to normal environment episode semantics.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import platform
from typing import Any, Callable, Iterator, Literal, Mapping, Protocol

import gymnasium as gym
import numpy as np


def _encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError("Object arrays cannot be simulator state.")
        return {"__array__": base64.b64encode(array.tobytes()).decode("ascii"),
                "dtype": array.dtype.str, "shape": list(array.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _encode(item) for key, item in value.items()}
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        if "__array__" in value:
            dtype = np.dtype(value["dtype"])
            if dtype.hasobject:
                raise ValueError("Object arrays cannot be simulator state.")
            return np.frombuffer(base64.b64decode(value["__array__"], validate=True),
                                 dtype=dtype).reshape(value["shape"]).copy()
        if "__tuple__" in value:
            return tuple(_decode(item) for item in value["__tuple__"])
        return {key: _decode(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class SimulatorSnapshot:
    """Immutable JSON bytes with fresh decoded arrays on every access."""

    payload: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", bytes(self.payload))

    @classmethod
    def capture(cls, state: Mapping[str, Any]) -> "SimulatorSnapshot":
        return cls(json.dumps(_encode(state), sort_keys=True, separators=(",", ":"),
                              allow_nan=False).encode())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    def state(self) -> dict[str, Any]:
        return _decode(json.loads(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "sha256": self.sha256,
                "state": json.loads(self.payload)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SimulatorSnapshot":
        if value.get("schema_version") != 1:
            raise ValueError("Unsupported simulator snapshot schema.")
        result = cls(json.dumps(value["state"], sort_keys=True, separators=(",", ":"),
                                allow_nan=False).encode())
        if result.sha256 != value.get("sha256"):
            raise ValueError("Simulator snapshot checksum mismatch.")
        result.state()
        return result


_WRAPPER_FIELDS = {
    gym.wrappers.TimeLimit: ("_elapsed_steps", "_max_episode_steps"),
    gym.wrappers.OrderEnforcing: ("_has_reset", "_disable_render_order_enforcing"),
    gym.wrappers.PassiveEnvChecker: (
        "checked_reset", "checked_step", "checked_render", "close_called"),
}


def _name(obj: Any) -> str:
    return f"{type(obj).__module__}.{type(obj).__qualname__}"


def _chain(env: Any) -> tuple[list[Any], Any]:
    wrappers = []
    while isinstance(env, gym.Wrapper):
        if type(env) not in _WRAPPER_FIELDS:
            raise ValueError(f"Unreviewed simulator wrapper: {_name(env)}")
        wrappers.append(env)
        env = env.env
    return wrappers, env


def _rng(owner: Any) -> Any:
    generator = owner.__dict__.get("_np_random")
    return None if generator is None else deepcopy(generator.bit_generator.state)


def _load_rng(owner: Any, state: Any) -> None:
    if state is None:
        owner._np_random = None
        return
    kind = getattr(np.random, state["bit_generator"], None)
    if not isinstance(kind, type) or not issubclass(kind, np.random.BitGenerator):
        raise ValueError("Invalid NumPy bit generator in simulator snapshot.")
    generator = kind(0)
    generator.state = deepcopy(state)
    owner._np_random = np.random.Generator(generator)


def _require_humanoid(base: Any) -> None:
    from domains.dmcontrol import DMControlEnv
    if type(base) is not DMControlEnv or base.task_name != "humanoid-walk" or base.observation_type != "state":
        raise ValueError("Exact simulator capture supports Humanoid Walk state or an explicit calibration protocol.")


def _runtime(base: Any) -> dict[str, Any]:
    import mujoco
    physics = base._env.physics
    return {"dm_control": importlib.metadata.version("dm-control"),
            "mujoco": mujoco.__version__, "gymnasium": gym.__version__,
            "numpy": np.__version__, "python": platform.python_version(),
            "platform": platform.system(), "machine": platform.machine(),
            "model_sha256": hashlib.sha256(physics.model.to_bytes()).hexdigest(),
            "physics_class": _name(physics), "legacy_step": bool(physics.legacy_step),
            "action_repeat": base.action_repeat,
            "control_timestep": base._effective_control_timestep}


def _capture_base(base: Any) -> dict[str, Any]:
    if callable(getattr(base, "calibration_state", None)):
        return {"protocol": "explicit", "state": base.calibration_state()}
    _require_humanoid(base)
    import mujoco
    raw, physics = base._env._env, base._env.physics
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    integration = np.empty(mujoco.mj_stateSize(physics.model.ptr, spec))
    mujoco.mj_getState(physics.model.ptr, physics.data.ptr, integration, spec)
    return {"protocol": "humanoid-integration-v1", "state": {
        "runtime": _runtime(base), "integration": integration,
        "observation": base._state_observation(raw.task.get_observation(physics)),
        "task_random_state": deepcopy(raw.task.random.get_state()),
        "step_count": raw._step_count, "reset_next_step": raw._reset_next_step,
        "step_limit": None if np.isinf(raw._step_limit) else raw._step_limit,
        "generators": [_rng(owner) for owner in (base, base.action_space, base.observation_space)]}}


def capture_simulator_snapshot(env: Any) -> SimulatorSnapshot:
    wrappers, base = _chain(env)
    bookkeeping = []
    for wrapper in wrappers:
        fields = {name: getattr(wrapper, name) for name in _WRAPPER_FIELDS[type(wrapper)]}
        if fields.get("_max_episode_steps") == float("inf"):
            fields["_max_episode_steps"] = None
        bookkeeping.append({"class": _name(wrapper), "fields": fields})
    return SimulatorSnapshot.capture({"schema_version": 1, "base_class": _name(base),
                                      "base": _capture_base(base), "wrappers": bookkeeping})


def _restore_base(base: Any, record: Mapping[str, Any]) -> np.ndarray:
    state = record["state"]
    if record["protocol"] == "explicit":
        return np.asarray(base.load_calibration_state(deepcopy(state))).copy()
    if record["protocol"] != "humanoid-integration-v1":
        raise ValueError("Unknown simulator capture protocol.")
    _require_humanoid(base)
    import mujoco
    if state["runtime"] != _runtime(base):
        raise ValueError("Simulator runtime/model identity differs from the snapshot.")
    raw, physics = base._env._env, base._env.physics
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    integration = np.asarray(state["integration"], dtype=np.float64)
    if integration.shape != (mujoco.mj_stateSize(physics.model.ptr, spec),) or not np.isfinite(integration).all():
        raise ValueError("Invalid simulator integration state.")
    if state["runtime"]["legacy_step"] is not True:
        raise ValueError("Simulator restore is reviewed only for the legacy-step boundary.")
    mujoco.mj_setState(physics.model.ptr, physics.data.ptr, integration, spec)
    # mj_forward would overwrite acceleration-dependent and warm-start fields.
    mujoco.mj_step1(physics.model.ptr, physics.data.ptr)
    raw.task.random.set_state(deepcopy(state["task_random_state"]))
    raw._step_count, raw._reset_next_step = state["step_count"], state["reset_next_step"]
    raw._step_limit = float("inf") if state["step_limit"] is None else state["step_limit"]
    for owner, rng in zip((base, base.action_space, base.observation_space), state["generators"]):
        _load_rng(owner, rng)
    observation = base._state_observation(raw.task.get_observation(physics))
    if not np.array_equal(observation, state["observation"]):
        raise RuntimeError("Simulator restoration did not reproduce the saved observation.")
    return observation


def restore_simulator_snapshot(env: Any, snapshot: SimulatorSnapshot, *, continuing: bool = False) -> np.ndarray:
    wrappers, base = _chain(env)
    state = snapshot.state()
    if state.get("schema_version") != 1 or state.get("base_class") != _name(base):
        raise ValueError("Simulator snapshot environment identity differs.")
    if [_name(item) for item in wrappers] != [item["class"] for item in state["wrappers"]]:
        raise ValueError("Simulator snapshot wrapper identity differs.")
    for wrapper, record in zip(wrappers, state["wrappers"]):
        if set(record["fields"]) != set(_WRAPPER_FIELDS[type(wrapper)]):
            raise ValueError("Simulator snapshot wrapper bookkeeping differs.")
    observation = _restore_base(base, state["base"])
    for wrapper, record in zip(wrappers, state["wrappers"]):
        for name, value in record["fields"].items():
            setattr(wrapper, name, float("inf") if name == "_max_episode_steps" and value is None else value)
        wrapper._cached_spec = None
    if continuing:
        if state["base"]["protocol"] == "explicit":
            base.enable_continuing_calibration()
        else:
            raw = base._env._env
            if raw._reset_next_step and raw._step_count < raw._step_limit:
                raise ValueError("Cannot continue a reset-pending simulator root.")
            if raw._step_count >= raw._step_limit:
                raw._reset_next_step = False
            raw._step_limit = float("inf")
        for wrapper in wrappers:
            if type(wrapper) is gym.wrappers.TimeLimit:
                wrapper._max_episode_steps = float("inf")
    return observation


@contextmanager
def preserve_simulator(env: Any) -> Iterator[None]:
    """Restore the caller's exact simulator and process RNG even after failure."""
    from utils.resume_runtime import capture_global_rng_state, restore_global_rng_state
    snapshot, rng = capture_simulator_snapshot(env), capture_global_rng_state()
    try:
        yield
    finally:
        try:
            restore_simulator_snapshot(env, snapshot)
        finally:
            restore_global_rng_state(rng)


@dataclass(frozen=True)
class PolicyOutput:
    actions: np.ndarray
    log_probs: np.ndarray | None = None


class ActorCallback(Protocol):
    def __call__(self, observations: np.ndarray, noise: np.ndarray, remaining_horizon: int) -> PolicyOutput: ...


def _action(actor: ActorCallback, observation: np.ndarray, noise: np.ndarray, remaining: int,
            *, entropy: bool) -> tuple[np.ndarray, float]:
    value = actor(np.asarray(observation)[None].copy(), noise[None].copy(), remaining)
    action = np.asarray(value.actions)
    if action.shape != (1, noise.size) or not np.isfinite(action).all():
        raise ValueError("Actor actions must be finite [1, action_dim].")
    log_prob = 0.0
    if entropy:
        if value.log_probs is None:
            raise ValueError("Soft return evaluation requires actor log probabilities.")
        logs = np.asarray(value.log_probs)
        if logs.size != 1 or not np.isfinite(logs).all():
            raise ValueError("Actor log probabilities must contain one finite scalar.")
        log_prob = float(logs.reshape(-1)[0])
    return action[0].copy(), log_prob


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return int(value)


def evaluate_real_prefix(
    env: Any, snapshot: SimulatorSnapshot, *, first_action: np.ndarray,
    prefix_actor: ActorCallback, prior_actor: ActorCallback,
    terminal_q: Callable[[np.ndarray, np.ndarray], np.ndarray],
    prefix_noise: np.ndarray, tail_noise: np.ndarray, discount: float,
    objective: Literal["reward", "soft"] = "reward", alpha: float = 0.0,
    terminal_objective: Literal["reward", "soft"] = "reward", terminal_alpha: float = 0.0,
    terminal_first_action_entropy: bool = False,
    continuing: bool = True,
) -> dict[str, Any]:
    """Forced-action H-prefix with frozen-Q and sampled-real-tail alternatives.

    H is inferred from ``prefix_noise[H,A]`` and may be any positive integer.
    Noise at prefix step zero is reserved but unused because that action is
    forced. Terminal Q is evaluated on the exact first sampled prior-tail action.
    ``terminal_first_action_entropy`` adds the first prior action's entropy to
    both the Q bootstrap and the measured prior tail. This matches the explicit
    outer-entropy boundary option; it is separate from the terminal Q's own
    reward/soft convention (Q excludes its conditioned first action's entropy).
    The sampled tail has a finite cutoff and no appended Q: it is not claimed to
    be an infinite-horizon ground truth. A time-limit truncation makes the
    affected estimate unavailable; true termination masks future contributions.
    Callback-owned model/controller state must be frozen or independently cloned.
    """
    prefix_noise, tail_noise = np.asarray(prefix_noise), np.asarray(tail_noise)
    action = np.asarray(first_action)
    if (prefix_noise.ndim != 2 or prefix_noise.shape[0] < 1 or prefix_noise.shape[1] < 1
            or tail_noise.ndim != 2 or tail_noise.shape[0] < 1 or tail_noise.shape[1] != prefix_noise.shape[1]
            or action.shape != (prefix_noise.shape[1],)
            or not all(np.isfinite(item).all() for item in (prefix_noise, tail_noise, action))):
        raise ValueError("Actions/noise must be finite, with shapes [A], [positive H,A], [positive T,A].")
    if objective not in {"reward", "soft"} or terminal_objective not in {"reward", "soft"}:
        raise ValueError("Objective conventions must be reward or soft.")
    if not np.isfinite([discount, alpha, terminal_alpha]).all() or not 0 <= discount <= 1 or min(alpha, terminal_alpha) < 0:
        raise ValueError("Discount must be in [0,1] and entropy coefficients nonnegative and finite.")
    horizon, tail_steps = len(prefix_noise), len(tail_noise)
    prefix_reward = prefix_entropy = tail_reward = tail_entropy = 0.0
    endpoint_q = endpoint_entropy = 0.0
    endpoint_action = None
    terminated = truncated = False
    prefix_decisions = tail_decisions = 0
    with preserve_simulator(env):
        observation = restore_simulator_snapshot(env, snapshot, continuing=continuing)
        for step in range(horizon):
            log_prob = 0.0
            if step:
                action, log_prob = _action(prefix_actor, observation, prefix_noise[step], horizon - step,
                                          entropy=objective == "soft" and alpha > 0)
            observation, reward, terminated, truncated, _ = env.step(action.copy())
            if not np.isfinite(reward):
                raise ValueError("Simulator emitted a nonfinite reward.")
            prefix_reward += discount ** step * float(reward)
            prefix_entropy += discount ** step * (-alpha * log_prob if objective == "soft" else 0.0)
            prefix_decisions += 1
            if terminated or truncated:
                break
        if not (terminated or truncated):
            for step in range(tail_steps):
                include_entropy = terminal_first_action_entropy if step == 0 else terminal_objective == "soft"
                action, log_prob = _action(prior_actor, observation, tail_noise[step], 0,
                                          entropy=include_entropy and terminal_alpha > 0)
                if step == 0:
                    q = np.asarray(terminal_q(np.asarray(observation)[None].copy(), action[None].copy()))
                    if q.size != 1 or not np.isfinite(q).all():
                        raise ValueError("Terminal Q must contain one finite scalar.")
                    endpoint_q, endpoint_action = float(q.reshape(-1)[0]), action.tolist()
                    endpoint_entropy = -terminal_alpha * log_prob if terminal_first_action_entropy else 0.0
                observation, reward, terminated, truncated, _ = env.step(action.copy())
                if not np.isfinite(reward):
                    raise ValueError("Simulator emitted a nonfinite reward.")
                tail_reward += discount ** step * float(reward)
                tail_entropy += discount ** step * (-terminal_alpha * log_prob if include_entropy else 0.0)
                tail_decisions += 1
                if terminated or truncated:
                    break
    prefix_complete = prefix_decisions == horizon or bool(terminated)
    bootstrapped_complete = prefix_complete and not (truncated and tail_decisions == 0)
    prefix_return = prefix_reward + prefix_entropy
    bootstrap = discount ** horizon * (endpoint_q + endpoint_entropy)
    tail_contribution = discount ** horizon * (tail_reward + tail_entropy)
    return {
        "horizon": horizon, "tail_steps_requested": tail_steps, "discount": float(discount),
        "objective": objective, "alpha": float(alpha), "terminal_objective": terminal_objective,
        "terminal_alpha": float(terminal_alpha), "first_action": np.asarray(first_action).tolist(),
        "first_action_entropy_included": False,
        "boundary_action_entropy_included": bool(terminal_first_action_entropy),
        "real_prefix_reward": float(prefix_reward), "real_prefix_entropy": float(prefix_entropy),
        "real_prefix_return": float(prefix_return), "real_endpoint_q": float(endpoint_q),
        "real_endpoint_entropy": float(endpoint_entropy),
        "real_bootstrap": float(bootstrap),
        "real_bootstrapped_return": float(prefix_return + bootstrap) if bootstrapped_complete else None,
        "real_tail_reward": float(tail_reward), "real_tail_entropy": float(tail_entropy),
        "real_tail_contribution": float(tail_contribution),
        "real_mc_return": float(prefix_return + tail_contribution) if not truncated else None,
        "real_mc_partial_return": float(prefix_return + tail_contribution),
        "terminal_prediction_error": float(bootstrap - tail_contribution) if bootstrapped_complete and not truncated else None,
        "prefix_decisions": prefix_decisions, "tail_decisions": tail_decisions,
        "prefix_complete": prefix_complete, "mc_complete": not bool(truncated),
        "terminated": bool(terminated), "truncated": bool(truncated),
        "endpoint_action": endpoint_action, "continuing": bool(continuing),
        "return_semantics": "fixed_actor_forced_action_prefix_then_finite_sampled_prior_tail",
        "mc_tail_has_bootstrap": False,
    }


def evaluate_continuation(
    env: Any, snapshot: SimulatorSnapshot, *, first_action: np.ndarray,
    controller_factory: Callable[[], Callable[[np.ndarray, int], np.ndarray]],
    steps: int, discount: float, intervention: Literal["first_action_only", "memory_only", "full"],
    controller_identity: str, memory_identity: str,
) -> dict[str, Any]:
    """Evaluate a forced first action followed by an isolated replanning controller.

    For first_action_only, callers use identical controller/memory identities
    and factories across branches. For memory_only, callers use an identical
    first action and vary the memory installed by the factory. Full branches
    vary both. The factory must create independent mutable state and pair its
    private solver RNG across compared branches; a shared live controller would
    invalidate the intervention. Normal environment truncation is retained.
    """
    steps = _positive_integer(steps, "Continuation steps")
    if intervention not in {"first_action_only", "memory_only", "full"}:
        raise ValueError("Unknown continuation intervention.")
    if not controller_identity or not memory_identity:
        raise ValueError("Controller and memory identities are required.")
    if not np.isfinite(discount) or not 0 <= discount <= 1:
        raise ValueError("Discount must be finite in [0,1].")
    action = np.asarray(first_action)
    if action.ndim != 1 or action.size == 0 or not np.isfinite(action).all():
        raise ValueError("First action must be a finite action vector.")
    rewards, actions = [], []
    terminated = truncated = False
    with preserve_simulator(env):
        controller = controller_factory()
        observation = restore_simulator_snapshot(env, snapshot)
        for offset in range(steps):
            if offset:
                candidate = np.asarray(controller(np.asarray(observation).copy(), offset))
                if candidate.shape != action.shape or not np.isfinite(candidate).all():
                    raise ValueError("Continuation controller emitted an invalid action.")
                action = candidate
            observation, reward, terminated, truncated, _ = env.step(action.copy())
            if not np.isfinite(reward):
                raise ValueError("Simulator emitted a nonfinite reward.")
            rewards.append(float(reward))
            actions.append(action.tolist())
            if terminated or truncated:
                break
    return {"intervention": intervention, "controller_identity": controller_identity,
            "memory_identity": memory_identity, "first_action": np.asarray(first_action).tolist(),
            "rewards": rewards, "actions": actions, "steps": len(rewards), "steps_requested": steps,
            "real_return": float(sum(rewards)),
            "real_discounted_return": float(sum(discount ** index * reward for index, reward in enumerate(rewards))),
            "terminated": bool(terminated), "truncated": bool(truncated),
            "requested_horizon_complete": bool(terminated or len(rewards) == steps),
            "return_semantics": "forced_first_action_then_replanning_to_requested_cutoff",
            "discount": float(discount)}


__all__ = ["SimulatorSnapshot", "capture_simulator_snapshot", "restore_simulator_snapshot",
           "preserve_simulator", "PolicyOutput", "ActorCallback", "evaluate_real_prefix",
           "evaluate_continuation"]
