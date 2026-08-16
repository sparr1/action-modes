"""Boundary-only environment and RNG state for exact training resume."""

from __future__ import annotations

import copy
import hashlib
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


class ResumeRuntimeError(RuntimeError):
    pass


class UnsupportedResumeEnvironment(ResumeRuntimeError):
    pass


class ResumeRuntimeMismatch(ResumeRuntimeError):
    pass


@dataclass(frozen=True)
class _Capability:
    protocol: str
    fixed_horizon: int | None = None
    state_only: bool = False


_CAPABILITIES = {
    "gymnasium.envs.mujoco.ant_v4.AntEnv": _Capability("ant"),
    "gymnasium.envs.mujoco.ant_v5.AntEnv": _Capability("ant"),
    "domains.ant_variable_legs.AntVariableLegsEnv": _Capability("explicit"),
    "domains.ant_leg_adaptation.AntLegAdaptationEnv": _Capability("explicit"),
    "domains.ant_3leg_deadstump_env.Ant3LegDeadStumpEnv": _Capability("explicit"),
    "domains.dmcontrol.DMControlEnv": _Capability(
        "explicit", fixed_horizon=500, state_only=True
    ),
}
_ALLOWED_WRAPPERS = {
    "gymnasium.wrappers.time_limit.TimeLimit",
    "gymnasium.wrappers.order_enforcing.OrderEnforcing",
    "gymnasium.wrappers.env_checker.PassiveEnvChecker",
    "domains.AntPlane.AntPlane",
}
_TEST_CAPABILITIES: dict[type, _Capability] = {}


def _type_name(value: Any) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}"


def register_test_resume_environment(
    environment_type: type, *, episode_steps: int, early_termination: bool = False
) -> None:
    if not isinstance(environment_type, type) or not (
        environment_type.__module__.startswith("tests.")
        or environment_type.__module__.startswith("test_")
    ):
        raise ValueError("Only test-defined environment classes may be registered here.")
    if (
        isinstance(episode_steps, bool)
        or not isinstance(episode_steps, int)
        or episode_steps <= 0
        or not isinstance(early_termination, bool)
    ):
        raise ValueError("A test environment needs a positive fixed horizon.")
    if early_termination:
        capability = _Capability("unsupported-early-termination", episode_steps)
    else:
        capability = _Capability("explicit", episode_steps)
    previous = _TEST_CAPABILITIES.get(environment_type)
    if previous is not None and previous != capability:
        raise ValueError("Test environment was registered inconsistently.")
    _TEST_CAPABILITIES[environment_type] = capability


def _stack(env):
    wrappers = []
    current = env
    seen = set()
    while hasattr(current, "env"):
        if id(current) in seen:
            raise UnsupportedResumeEnvironment("Environment wrapper cycle detected.")
        seen.add(id(current))
        wrappers.append(current)
        current = current.env
    return wrappers, current


def _capability(base) -> _Capability:
    capability = _TEST_CAPABILITIES.get(type(base)) or _CAPABILITIES.get(
        _type_name(base)
    )
    if capability is None:
        raise UnsupportedResumeEnvironment(
            f"Unsupported reviewed exact-resume environment {_type_name(base)!r}."
        )
    return capability


def _positive_horizon(value, owner: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise UnsupportedResumeEnvironment(f"{owner} has no fixed integer horizon.")
    value = int(value)
    if value <= 0:
        raise UnsupportedResumeEnvironment(f"{owner} horizon must be positive.")
    return value


def _review(env):
    wrappers, base = _stack(env)
    unknown = [_type_name(item) for item in wrappers if _type_name(item) not in _ALLOWED_WRAPPERS]
    if unknown:
        raise UnsupportedResumeEnvironment(
            "Unsupported resume wrapper stack: " + ", ".join(unknown)
        )
    capability = _capability(base)
    if capability.protocol == "unsupported-early-termination":
        raise UnsupportedResumeEnvironment(
            "Boundary-only resume does not support early termination."
        )
    if capability.protocol == "explicit":
        missing = [
            name
            for name in (
                "training_resume_state",
                "validate_training_resume_state",
                "load_training_resume_state",
            )
            if not callable(getattr(base, name, None))
        ]
        if missing:
            raise UnsupportedResumeEnvironment(
                f"{_type_name(base)} is missing {', '.join(missing)}."
            )
    elif capability.protocol != "ant":
        raise UnsupportedResumeEnvironment("Unknown environment resume protocol.")
    if capability.state_only and getattr(base, "observation_type", None) != "state":
        raise UnsupportedResumeEnvironment("Exact DMControl resume is state-only.")
    unhealthy = getattr(base, "_terminate_when_unhealthy", False)
    if not isinstance(unhealthy, bool) or unhealthy:
        raise UnsupportedResumeEnvironment(
            "Boundary-only resume does not support early termination; "
            "terminate_when_unhealthy must be false."
        )

    horizons = []
    if capability.fixed_horizon is not None:
        horizons.append(_positive_horizon(capability.fixed_horizon, "adapter"))
    for wrapper in wrappers:
        if _type_name(wrapper) == "gymnasium.wrappers.time_limit.TimeLimit":
            horizons.append(
                _positive_horizon(wrapper._max_episode_steps, "TimeLimit")
            )
    spec = getattr(env, "spec", None)
    if spec is not None and getattr(spec, "max_episode_steps", None) is not None:
        horizons.append(
            _positive_horizon(spec.max_episode_steps, "environment spec")
        )
    if not horizons or len(set(horizons)) != 1:
        raise UnsupportedResumeEnvironment(
            f"Environment horizon is missing or inconsistent: {horizons}."
        )
    return wrappers, base, capability, horizons[0]


def validate_environment_capability(
    env, *, expected_episode_steps: int | None = None
) -> int:
    _, _, _, horizon = _review(env)
    if expected_episode_steps is not None and horizon != _positive_horizon(
        expected_episode_steps, "trainer"
    ):
        raise ResumeRuntimeMismatch(
            "Trainer episode_length differs from the reviewed environment horizon: "
            f"{expected_episode_steps} != {horizon}."
        )
    return horizon


def _space_contract(space):
    result = {"type": _type_name(space)}
    children = getattr(space, "spaces", None)
    if isinstance(children, Mapping):
        result["spaces"] = {str(key): _space_contract(value) for key, value in children.items()}
        return result
    if getattr(space, "shape", None) is not None:
        result["shape"] = [int(item) for item in space.shape]
    if getattr(space, "dtype", None) is not None:
        result["dtype"] = str(np.dtype(space.dtype))
    for name in ("low", "high"):
        value = getattr(space, name, None)
        if value is not None:
            result[f"{name}_sha256"] = hashlib.sha256(
                np.ascontiguousarray(value).tobytes()
            ).hexdigest()
    return result


def environment_contract(
    env, *, expected_episode_steps: int | None = None
) -> dict[str, Any]:
    wrappers, base, _, horizon = _review(env)
    if expected_episode_steps is not None and horizon != int(expected_episode_steps):
        raise ResumeRuntimeMismatch("Configured and live episode horizons differ.")
    spec = getattr(env, "spec", None)
    return {
        "schema_version": 2,
        "wrappers": [_type_name(item) for item in wrappers],
        "base": _type_name(base),
        "spec_id": None if spec is None else getattr(spec, "id", None),
        "episode_steps": horizon,
        "early_termination": False,
        "action_space": _space_contract(env.action_space),
        "observation_space": _space_contract(env.observation_space),
    }


def _generator_state(generator, owner: str):
    if not isinstance(generator, np.random.Generator):
        raise UnsupportedResumeEnvironment(f"{owner} was not seeded with Generator.")
    return copy.deepcopy(generator.bit_generator.state)


def _validate_generator_state(generator, state, owner: str) -> None:
    if not isinstance(generator, np.random.Generator) or not isinstance(state, Mapping):
        raise ResumeRuntimeMismatch(f"{owner} RNG state is incompatible.")
    if generator.bit_generator.state.get("bit_generator") != state.get("bit_generator"):
        raise ResumeRuntimeMismatch(f"{owner} bit generator changed.")
    try:
        probe = copy.deepcopy(generator)
        probe.bit_generator.state = copy.deepcopy(dict(state))
    except (KeyError, TypeError, ValueError) as exc:
        raise ResumeRuntimeMismatch(f"{owner} RNG state is invalid.") from exc


def _load_validated_generator(generator, state) -> None:
    generator.bit_generator.state = copy.deepcopy(dict(state))


def _space_state(space, owner: str):
    generator = getattr(space, "_np_random", None)
    return {"type": _type_name(space), "rng": _generator_state(generator, owner)}


def _validate_space_state(space, state, owner: str) -> None:
    if (
        not isinstance(state, Mapping)
        or set(state) != {"type", "rng"}
        or state.get("type") != _type_name(space)
    ):
        raise ResumeRuntimeMismatch(f"{owner} type changed.")
    _validate_generator_state(getattr(space, "_np_random", None), state.get("rng"), owner)


def _load_validated_space(space, state) -> None:
    _load_validated_generator(space._np_random, state["rng"])


def _validate_explicit_base_state(base, state) -> None:
    if not isinstance(state, Mapping):
        raise ResumeRuntimeMismatch("Base environment resume state is invalid.")
    try:
        base.validate_training_resume_state(copy.deepcopy(state))
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ResumeRuntimeMismatch("Base environment resume state is invalid.") from exc


def capture_environment_state(env) -> dict[str, Any]:
    wrappers, base, capability, _ = _review(env)
    if capability.protocol == "explicit":
        base_state = copy.deepcopy(base.training_resume_state())
        _validate_explicit_base_state(base, base_state)
    else:
        base_state = {"np_random": _generator_state(base.np_random, "environment")}
    return {
        "schema_version": 1,
        "boundary": "between_episodes_before_reset",
        "wrappers": [_type_name(item) for item in wrappers],
        "base_type": _type_name(base),
        "protocol": capability.protocol,
        "base_state": base_state,
        "action_space": _space_state(env.action_space, "action_space"),
        "observation_space": _space_state(env.observation_space, "observation_space"),
    }


def validate_environment_state(env, state: Mapping[str, Any]) -> None:
    wrappers, base, capability, _ = _review(env)
    if not isinstance(state, Mapping) or state.get("schema_version") != 1:
        raise ResumeRuntimeMismatch("Unsupported environment-state schema.")
    expected = {
        "schema_version",
        "boundary",
        "wrappers",
        "base_type",
        "protocol",
        "base_state",
        "action_space",
        "observation_space",
    }
    if set(state) != expected or state["boundary"] != "between_episodes_before_reset":
        raise ResumeRuntimeMismatch("Checkpoint is not a supported environment boundary.")
    if state["wrappers"] != [_type_name(item) for item in wrappers]:
        raise ResumeRuntimeMismatch("Environment wrapper stack changed.")
    if state["base_type"] != _type_name(base) or state["protocol"] != capability.protocol:
        raise ResumeRuntimeMismatch("Base environment type changed across resume.")
    if capability.protocol == "explicit":
        _validate_explicit_base_state(base, state["base_state"])
    else:
        base_state = state["base_state"]
        if not isinstance(base_state, Mapping) or set(base_state) != {"np_random"}:
            raise ResumeRuntimeMismatch("Base environment RNG state is incompatible.")
        _validate_generator_state(base.np_random, base_state["np_random"], "environment")
    for space, saved, owner in (
        (env.action_space, state["action_space"], "action_space"),
        (env.observation_space, state["observation_space"], "observation_space"),
    ):
        _validate_space_state(space, saved, owner)


def restore_environment_state(env, state: Mapping[str, Any]) -> None:
    validate_environment_state(env, state)
    _, base, capability, _ = _review(env)
    if capability.protocol == "explicit":
        base.load_training_resume_state(copy.deepcopy(state["base_state"]))
    else:
        _load_validated_generator(base.np_random, state["base_state"]["np_random"])
    _load_validated_space(env.action_space, state["action_space"])
    _load_validated_space(env.observation_space, state["observation_space"])


def capture_global_rng_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [item.cpu() for item in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def validate_global_rng_state(state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping) or set(state) != {
        "schema_version",
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    } or state.get("schema_version") != 1:
        raise ResumeRuntimeMismatch("Unsupported process RNG state.")
    try:
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(state["numpy"])
        torch.Generator(device="cpu").set_state(state["torch_cpu"])
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ResumeRuntimeMismatch("Saved process RNG state is invalid.") from exc
    cuda = list(state["torch_cuda"])
    live_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(cuda) != live_count:
        raise ResumeRuntimeMismatch(
            f"Visible CUDA count changed: saved={len(cuda)}, live={live_count}."
        )
    for index, item in enumerate(cuda):
        try:
            torch.Generator(device=f"cuda:{index}").set_state(item.cpu())
        except (AttributeError, RuntimeError) as exc:
            raise ResumeRuntimeMismatch(f"CUDA RNG state {index} is invalid.") from exc


def restore_global_rng_state(state: Mapping[str, Any]) -> None:
    validate_global_rng_state(state)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


__all__ = [
    "ResumeRuntimeError",
    "ResumeRuntimeMismatch",
    "UnsupportedResumeEnvironment",
    "capture_environment_state",
    "capture_global_rng_state",
    "environment_contract",
    "register_test_resume_environment",
    "restore_environment_state",
    "restore_global_rng_state",
    "validate_environment_capability",
    "validate_environment_state",
    "validate_global_rng_state",
]
