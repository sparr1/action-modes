"""Shared Gym/Gymnasium utilities for parameterized-action RL."""

from functools import wraps
import os

import numpy as np

from utils.cleanup import add_cleanup_notes


def with_owned_environment(factory):
    """Inject and reliably close an entry point's environment.

    The decorated function receives the created environment as
    ``_owned_env``. If both the run and cleanup fail, retain the run exception;
    cleanup failures still surface after an otherwise successful run.
    """

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            env = factory()
            try:
                result = function(*args, _owned_env=env, **kwargs)
            except BaseException as primary_error:
                try:
                    env.close()
                except BaseException as cleanup_error:
                    add_cleanup_notes(
                        primary_error,
                        (cleanup_error,),
                        prefix=(
                            "Additional environment close failure after the run "
                            "stopped"
                        ),
                    )
                raise
            env.close()
            return result

        return wrapped

    return decorate


def normalize_pamdp_info(info, terminated, truncated):
    """Attach authoritative Gymnasium termination metadata to ``info``."""

    info = dict(info or {})
    terminated = bool(terminated)
    truncated = bool(truncated)
    info["terminated"] = terminated
    info["truncated"] = truncated
    info["TimeLimit.truncated"] = bool(truncated and not terminated)
    return info


def unpack_pamdp_step(transition):
    """Return a normalized MP-DQN transition without discarding time limits."""

    if len(transition) == 5:
        observation, reward, terminated, truncated, info = transition
        terminated = bool(terminated)
        truncated = bool(truncated)
    elif len(transition) == 4:
        observation, reward, done, info = transition
        legacy_timeout = bool(
            isinstance(info, dict) and info.get("TimeLimit.truncated", False)
        )
        truncated = bool(done) and legacy_timeout
        terminated = bool(done) and not truncated
    else:
        raise ValueError(
            "PAMDP environments must return a 4- or 5-item step tuple."
        )

    if isinstance(observation, tuple) and len(observation) == 2:
        next_state, steps = observation
    else:
        next_state = observation
        steps = info.get("steps", 1) if isinstance(info, dict) else 1
    next_state = np.asarray(next_state, dtype=np.float32)
    done = bool(terminated or truncated)
    info = normalize_pamdp_info(info, terminated, truncated)
    return next_state, reward, terminated, truncated, done, info, steps


def save_pamdp_returns(directory, title, seed, returns, *, evaluation=False):
    """Persist returns below the configured output directory, if enabled."""

    if not directory:
        return None
    os.makedirs(directory, exist_ok=True)
    suffix = "e" if evaluation else ""
    path = os.path.join(directory, f"{title}{seed}{suffix}.npy")
    np.save(path, returns)
    return path
