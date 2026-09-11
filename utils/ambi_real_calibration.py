"""Immutable simulator roots and paired, batched real-return diagnostics.

Actors accept ``(observations[B,...], standard_normal_noise[B,A])`` and return
actions ``[B,A]``. They must encode each new observation, use the supplied noise,
and never optimize. This NumPy boundary keeps the simulator independent of a
particular policy implementation and allows frozen policy inference to batch
across all active branches.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Callable, Mapping, Sequence

import gymnasium as gym
import numpy as np


def _encode(value):
    if isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value)
        if value.dtype.hasobject:
            raise TypeError("Object arrays cannot be simulator snapshot data.")
        return {"__array__": base64.b64encode(value.tobytes()).decode("ascii"),
                "dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(item) for item in value]}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _encode(item) for key, item in value.items()}
    return value


def _decode(value):
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if isinstance(value, dict):
        if "__array__" in value:
            dtype = np.dtype(value["dtype"])
            if dtype.hasobject:
                raise ValueError("Object arrays cannot be simulator snapshot data.")
            return np.frombuffer(base64.b64decode(value["__array__"], validate=True),
                                 dtype=dtype).reshape(value["shape"]).copy()
        if "__tuple__" in value:
            return tuple(_decode(item) for item in value["__tuple__"])
        return {key: _decode(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class SimulatorSnapshot:
    """Self-contained immutable JSON bytes; decoding always returns fresh data."""

    payload: bytes

    def __post_init__(self):
        object.__setattr__(self, "payload", bytes(self.payload))

    @classmethod
    def capture(cls, state: Mapping[str, Any]) -> "SimulatorSnapshot":
        return cls(json.dumps(_encode(state), sort_keys=True, separators=(",", ":"),
                              allow_nan=False).encode("utf-8"))

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
        payload = json.dumps(value["state"], sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")
        result = cls(payload)
        if result.sha256 != value.get("sha256"):
            raise ValueError("Simulator snapshot checksum mismatch.")
        result.state()  # Validate array encodings before any environment changes.
        return result


_WRAPPER_FIELDS = {
    gym.wrappers.TimeLimit: ("_elapsed_steps", "_max_episode_steps"),
    gym.wrappers.OrderEnforcing: ("_has_reset", "_disable_render_order_enforcing"),
    gym.wrappers.PassiveEnvChecker: (
        "checked_reset", "checked_step", "checked_render", "close_called"),
}


def _wrapper_chain(env):
    wrappers = []
    while isinstance(env, gym.Wrapper):
        if type(env) not in _WRAPPER_FIELDS:
            raise ValueError(f"Simulator snapshots do not support wrapper {type(env).__name__}.")
        wrappers.append(env)
        env = env.env
    return wrappers, env


def capture_simulator_snapshot(env) -> SimulatorSnapshot:
    """Capture an initialized Humanoid state environment and reviewed wrappers."""
    wrappers, base = _wrapper_chain(env)
    state = base.calibration_state()
    bookkeeping = []
    for wrapper in wrappers:
        fields = {name: getattr(wrapper, name) for name in _WRAPPER_FIELDS[type(wrapper)]}
        if fields.get("_max_episode_steps") == float("inf"):
            fields["_max_episode_steps"] = None
        bookkeeping.append({"class": f"{type(wrapper).__module__}.{type(wrapper).__qualname__}",
                            "fields": fields})
    return SimulatorSnapshot.capture({"schema_version": 1,
                                      "environment": state, "wrappers": bookkeeping})


def enable_continuing_calibration(env):
    """Disable both clocks on this dedicated environment; retain wrapper state."""
    wrappers, base = _wrapper_chain(env)
    base.enable_continuing_calibration()
    for wrapper in wrappers:
        if isinstance(wrapper, gym.wrappers.TimeLimit):
            wrapper._max_episode_steps = float("inf")
    return env


def restore_simulator_snapshot(env, snapshot: SimulatorSnapshot, *, continuing=False):
    wrappers, base = _wrapper_chain(env)
    state = snapshot.state()
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported simulator root schema.")
    saved = state["wrappers"]
    current = [f"{type(wrapper).__module__}.{type(wrapper).__qualname__}" for wrapper in wrappers]
    if current != [wrapper["class"] for wrapper in saved]:
        raise ValueError("Simulator snapshot Gym wrapper identity differs.")
    for wrapper, record in zip(wrappers, saved):
        if set(record["fields"]) != set(_WRAPPER_FIELDS[type(wrapper)]):
            raise ValueError("Simulator snapshot wrapper bookkeeping differs.")
    observation = base.load_calibration_state(state["environment"])
    for wrapper, record in zip(wrappers, saved):
        for name, value in record["fields"].items():
            if name == "_max_episode_steps" and value is None:
                value = float("inf")
            setattr(wrapper, name, value)
        wrapper._cached_spec = None
    if continuing:
        enable_continuing_calibration(env)
    return observation


def make_continuing_env():
    """Construct the reference diagnostic environment without a Gym time limit."""
    from domains.dmcontrol import DMControlEnv

    return DMControlEnv(task="humanoid-walk", obs="state", continuing_calibration=True)


def omitted_tail_bound(discount: float, steps: int, reward_bound: float = 2.0) -> float:
    """Absolute discounted omitted reward after a finite MC tail, at its root."""
    if (not 0 <= discount < 1 or isinstance(steps, bool)
            or not isinstance(steps, (int, np.integer)) or steps < 0
            or not np.isfinite(reward_bound) or reward_bound < 0):
        raise ValueError("Tail bound requires 0 <= discount < 1 and nonnegative steps/rewards.")
    return float(reward_bound * discount ** steps / (1.0 - discount))


def _finite_array(value, shape, name):
    result = np.asarray(value)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite and have shape {shape}; got {result.shape}.")
    return result


def evaluate_real_branches(
    envs: Sequence[Any], snapshot: SimulatorSnapshot,
    prefix_actor: Callable, prior_actor: Callable, outer_q: Callable,
    prefix_noise: np.ndarray, tail_noise: np.ndarray, *, discount: float,
    horizon: int, original_remaining_steps: int, reward_bound: float = 2.0,
    restore: Callable = restore_simulator_snapshot,
) -> dict[str, Any]:
    """Evaluate frozen actor prefixes, followed by measured sampled prior tails.

    Each environment restores the identical root. Noise is indexed by global
    rollout even when other branches terminate, and endpoint Q receives the
    exact action executed at tail step zero. True terminations mask all later
    reward and bootstrap; a truncation is retained explicitly, never reset.
    No Q value is appended at the Monte Carlo cutoff.
    """
    started = time.perf_counter()
    count = len(envs)
    if not count or horizon < 1 or original_remaining_steps < 0:
        raise ValueError("Branches require environments, positive horizon and a nonnegative cutoff.")
    if len({id(env) for env in envs}) != count:
        raise ValueError("Each simultaneous branch needs its own environment.")
    prefix_noise = np.asarray(prefix_noise)
    tail_noise = np.asarray(tail_noise)
    if prefix_noise.ndim != 3 or prefix_noise.shape[:2] != (horizon, count):
        raise ValueError("prefix_noise must have shape [horizon, rollouts, action_dim].")
    action_dim = prefix_noise.shape[2]
    if tail_noise.ndim != 3 or tail_noise.shape[0] < 1 or tail_noise.shape[1:] != (count, action_dim):
        raise ValueError("tail_noise must have shape [positive tail steps, rollouts, action_dim].")
    if not np.isfinite(prefix_noise).all() or not np.isfinite(tail_noise).all():
        raise ValueError("Policy noise must be finite.")
    tail_steps = len(tail_noise)
    bound = omitted_tail_bound(discount, tail_steps, reward_bound)
    serial_start = time.perf_counter()
    observations = [restore(env, snapshot, continuing=True) for env in envs]
    serialization_seconds = time.perf_counter() - serial_start
    alive = np.ones(count, dtype=bool)
    terminated = np.zeros(count, dtype=bool)
    truncated = np.zeros(count, dtype=bool)
    prefix_return = np.zeros(count)
    tail_return = np.zeros(count)
    endpoint_q = np.zeros(count)
    endpoint_action = [None] * count
    cutoff_discounted = np.zeros(count)
    cutoff_undiscounted = np.zeros(count)
    decisions = np.zeros(count, dtype=int)
    timing = dict(simulator_seconds=0.0, policy_seconds=0.0, q_seconds=0.0,
                  serialization_seconds=serialization_seconds)
    work = dict(simulator_decisions=0, policy_rows=0, q_rows=0)

    for step in range(horizon + tail_steps):
        active = np.flatnonzero(alive)
        if not len(active):
            break
        is_prefix = step < horizon
        actor = prefix_actor if is_prefix else prior_actor
        noise = prefix_noise[step, active] if is_prefix else tail_noise[step - horizon, active]
        obs_batch = np.stack([observations[index] for index in active])
        before = time.perf_counter()
        actions = _finite_array(actor(obs_batch, noise.copy()), (len(active), action_dim), "Policy actions")
        timing["policy_seconds"] += time.perf_counter() - before
        work["policy_rows"] += len(active)
        if step == horizon:
            before = time.perf_counter()
            q = np.asarray(outer_q(obs_batch, actions.copy())).reshape(-1)
            endpoint_q[active] = _finite_array(q, (len(active),), "Endpoint Q")
            timing["q_seconds"] += time.perf_counter() - before
            work["q_rows"] += len(active)
            for index, action in zip(active, actions):
                endpoint_action[index] = action.tolist()
        before = time.perf_counter()
        for index, action in zip(active, actions):
            obs, reward, terminal, truncation, _ = envs[index].step(action.copy())
            reward = float(reward)
            if not np.isfinite(reward):
                raise ValueError("Simulator emitted a nonfinite reward.")
            observations[index] = obs
            decisions[index] += 1
            if is_prefix:
                prefix_return[index] += discount ** step * reward
            else:
                tail_return[index] += discount ** (step - horizon) * reward
            if step < original_remaining_steps:
                cutoff_discounted[index] += discount ** step * reward
                cutoff_undiscounted[index] += reward
            terminated[index] |= bool(terminal)
            truncated[index] |= bool(truncation)
            alive[index] = not (terminal or truncation)
        timing["simulator_seconds"] += time.perf_counter() - before
        work["simulator_decisions"] += len(active)

    rows = []
    for index in range(count):
        bootstrap = discount ** horizon * endpoint_q[index]
        tail_contribution = discount ** horizon * tail_return[index]
        rows.append({
            "rollout_index": index,
            "real_prefix_reward": float(prefix_return[index]),
            "real_endpoint_q": float(endpoint_q[index]),
            "real_bootstrap": float(bootstrap),
            "real_bootstrapped_return": float(prefix_return[index] + bootstrap),
            "real_tail_return": float(tail_return[index]),
            "real_tail_contribution": float(tail_contribution),
            "real_mc_return": float(prefix_return[index] + tail_contribution),
            "bootstrap_prediction_error": float(bootstrap - tail_contribution),
            "episode_cutoff_discounted_return": float(cutoff_discounted[index]),
            "episode_cutoff_undiscounted_return": float(cutoff_undiscounted[index]),
            "episode_cutoff_complete": bool(terminated[index] or decisions[index] >= original_remaining_steps),
            "simulator_decisions": int(decisions[index]),
            "terminated": bool(terminated[index]), "truncated": bool(truncated[index]),
            "mc_complete": not bool(truncated[index]),
            "endpoint_action": endpoint_action[index],
            "tail_truncation_bound": None if truncated[index] else float(bound if alive[index] else 0.0),
            "return_truncation_bound": None if truncated[index] else float(discount ** horizon * bound if alive[index] else 0.0),
        })
    timing["total_seconds"] = time.perf_counter() - started
    return {"rows": rows, "timing": timing, "work": work}


__all__ = ["SimulatorSnapshot", "capture_simulator_snapshot", "restore_simulator_snapshot",
           "enable_continuing_calibration", "make_continuing_env", "omitted_tail_bound",
           "evaluate_real_branches"]
