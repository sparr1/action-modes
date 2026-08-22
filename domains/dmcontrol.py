"""Gymnasium adapter for TD-MPC2's single-task DMControl setup.

DMControl is intentionally imported only while constructing this environment.
This keeps all existing environments usable from installations that do not
contain the dedicated DMControl dependency set.
"""

from __future__ import annotations

from collections import deque
import copy
import importlib
from typing import Any, Mapping

import gymnasium as gym
import numpy as np


_ACTION_REPEAT = 2
_EPISODE_LENGTH = 500
_FRAME_STACK = 3
_IMAGE_SIZE = 64
_RENDER_SIZE = 384


def _load_dmcontrol_dependencies():
    """Import DMControl and register TD-MPC2's custom tasks lazily."""
    try:
        from dm_control import suite
        from dm_control.suite.wrappers import action_scale
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".", 1)[0] in {
            "dm_control",
            "dm_env",
            "mujoco",
        }:
            raise ImportError(
                "DMControl-v0 requires the dedicated DMControl environment. "
                "Create it from environments/dmcontrol and run with that "
                "environment's Python interpreter."
            ) from exc
        raise

    # Importing this package runs the upstream SUITE.add('custom') decorators.
    importlib.import_module("domains.dmcontrol_tasks")

    # TD-MPC2 refreshes these suite-level lookup tables after importing its
    # custom tasks. Merge rather than append so repeated environment creation
    # cannot duplicate registrations.
    custom_tasks = tuple(suite._get_tasks("custom"))
    suite.ALL_TASKS = tuple(dict.fromkeys((*suite.ALL_TASKS, *custom_tasks)))
    suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)
    return suite, action_scale


def _parse_task(task: str) -> tuple[str, str, str]:
    if not isinstance(task, str) or not task.strip():
        raise ValueError("DMControl-v0 requires a non-empty single-task name.")
    task = task.strip()
    if task.lower() in {"mt30", "mt80"}:
        raise ValueError(
            "DMControl-v0 supports one task at a time; TD-MPC2 multitask "
            f"environment {task!r} is not supported."
        )

    normalized = task.replace("-", "_")
    # TD-MPC2 publicly abbreviates these two domains, but accepting the full
    # DMControl spellings is useful for callers that already have suite IDs.
    full_domain_prefixes = ("ball_in_cup_", "point_mass_")
    matched_prefix = next(
        (prefix for prefix in full_domain_prefixes if normalized.startswith(prefix)),
        None,
    )
    if matched_prefix is not None:
        domain = matched_prefix[:-1]
        task_name = normalized[len(matched_prefix):]
    else:
        try:
            domain, task_name = normalized.split("_", 1)
        except ValueError as exc:
            raise ValueError(
                "DMControl task names must include a domain and task, for example "
                "'walker-walk'."
            ) from exc
    if not domain or not task_name:
        raise ValueError(
            "DMControl task names must include a domain and task, for example "
            "'walker-walk'."
        )

    domain = {"cup": "ball_in_cup", "pointmass": "point_mass"}.get(
        domain, domain
    )
    public_domain = {
        "ball_in_cup": "cup",
        "point_mass": "pointmass",
    }.get(domain, domain)
    canonical_name = f"{public_domain}-{task_name.replace('_', '-')}"
    return domain, task_name, canonical_name


def _state_size(observation_spec: Mapping[str, Any]) -> int:
    return int(
        sum(int(np.prod(spec.shape)) if spec.shape else 1 for spec in observation_spec.values())
    )


class DMControlEnv(gym.Env):
    """Single-task DMControl with TD-MPC2 state and pixel semantics."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(
        self,
        *,
        task: str,
        obs: str = "state",
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if obs not in {"state", "rgb"}:
            raise ValueError(
                "DMControl-v0 'obs' must be either 'state' or 'rgb'; "
                f"received {obs!r}."
            )
        if render_mode not in {None, "rgb_array"}:
            raise ValueError(
                "DMControl-v0 supports render_mode=None or 'rgb_array'; "
                f"received {render_mode!r}."
            )

        self._suite, self._action_scale = _load_dmcontrol_dependencies()
        self.domain_name, self.dmcontrol_task_name, self.task_name = _parse_task(task)
        if (self.domain_name, self.dmcontrol_task_name) not in self._suite.ALL_TASKS:
            raise ValueError(
                f"Unknown DMControl task {task!r} (resolved to "
                f"{self.domain_name!r}, {self.dmcontrol_task_name!r})."
            )

        self.observation_type = obs
        self.render_mode = render_mode
        self.action_repeat = _ACTION_REPEAT
        self.frame_stack = _FRAME_STACK if obs == "rgb" else None
        self.image_size = _IMAGE_SIZE if obs == "rgb" else None
        self.camera_id = 2 if self.domain_name == "quadruped" else 0
        self._frames: deque[np.ndarray] = deque(maxlen=_FRAME_STACK)
        self._env = self._make_raw_env(seed=0)
        self._effective_control_timestep = self._get_effective_control_timestep(
            self._env
        )
        self.metadata = dict(type(self).metadata)
        if self._effective_control_timestep is not None:
            self.metadata["render_fps"] = max(
                1, int(round(1.0 / self._effective_control_timestep))
            )

        action_spec = self._env.action_spec()
        action_low = np.broadcast_to(
            np.asarray(action_spec.minimum, dtype=action_spec.dtype),
            action_spec.shape,
        ).copy()
        action_high = np.broadcast_to(
            np.asarray(action_spec.maximum, dtype=action_spec.dtype),
            action_spec.shape,
        ).copy()
        self.action_space = gym.spaces.Box(
            low=action_low,
            high=action_high,
            dtype=action_spec.dtype,
        )
        self._action_spec_shape = tuple(action_spec.shape)
        self._action_spec_dtype = np.dtype(action_spec.dtype)

        self._state_observation_size = _state_size(self._env.observation_spec())
        if obs == "state":
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self._state_observation_size,),
                dtype=np.float32,
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(_FRAME_STACK * 3, _IMAGE_SIZE, _IMAGE_SIZE),
                dtype=np.uint8,
            )

    def _make_raw_env(self, *, seed: int):
        env = self._suite.load(
            self.domain_name,
            self.dmcontrol_task_name,
            task_kwargs={"random": int(seed)},
            visualize_reward=False,
        )
        return self._action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)

    @staticmethod
    def _get_effective_control_timestep(env) -> float | None:
        control_timestep = getattr(env, "control_timestep", None)
        if not callable(control_timestep):
            return None
        timestep = float(control_timestep()) * _ACTION_REPEAT
        if not np.isfinite(timestep) or timestep <= 0:
            raise ValueError(
                "DMControl returned an invalid effective control timestep."
            )
        return timestep

    @staticmethod
    def _close_raw_env(env) -> None:
        try:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        finally:
            # dm_env.Environment.close() is a no-op in the pinned DMControl
            # release. Explicitly free Physics so repeated seeded resets and
            # pixel environments release native MuJoCo and rendering contexts.
            physics = getattr(env, "physics", None)
            free = getattr(physics, "free", None)
            if callable(free):
                free()

    def _validate_rebuilt_specs(self, env) -> None:
        action_spec = env.action_spec()
        if (
            tuple(action_spec.shape) != self._action_spec_shape
            or np.dtype(action_spec.dtype) != self._action_spec_dtype
            or _state_size(env.observation_spec()) != self._state_observation_size
        ):
            raise RuntimeError(
                "DMControl task spaces changed after a seeded reconstruction."
            )

        expected_low = np.broadcast_to(action_spec.minimum, action_spec.shape)
        expected_high = np.broadcast_to(action_spec.maximum, action_spec.shape)
        if not (
            np.array_equal(expected_low, self.action_space.low)
            and np.array_equal(expected_high, self.action_space.high)
        ):
            raise RuntimeError(
                "DMControl action bounds changed after a seeded reconstruction."
            )
        if self._get_effective_control_timestep(env) != self._effective_control_timestep:
            raise RuntimeError(
                "DMControl control timestep changed after a seeded reconstruction."
            )

    def _rebuild_for_seed(self, seed: int) -> None:
        replacement = self._make_raw_env(seed=seed)
        try:
            self._validate_rebuilt_specs(replacement)
        except BaseException:
            self._close_raw_env(replacement)
            raise

        previous = self._env
        self._env = replacement
        self._close_raw_env(previous)

    @staticmethod
    def _state_observation(observation: Mapping[str, Any]) -> np.ndarray:
        # Mapping insertion order is part of TD-MPC2's observation contract.
        values = [np.asarray(value).reshape(-1) for value in observation.values()]
        return np.ascontiguousarray(np.concatenate(values).astype(np.float32, copy=False))

    def _render_frame(self, size: int) -> np.ndarray:
        frame = np.asarray(
            self._env.physics.render(
                height=size,
                width=size,
                camera_id=self.camera_id,
            )
        )
        expected_shape = (size, size, 3)
        if frame.shape != expected_shape or frame.dtype != np.uint8:
            raise RuntimeError(
                "DMControl returned an unexpected RGB frame: expected "
                f"uint8 {expected_shape}, received {frame.dtype} {frame.shape}."
            )
        return np.ascontiguousarray(frame)

    def _pixel_observation(self, *, is_reset: bool) -> np.ndarray:
        frame = self._render_frame(_IMAGE_SIZE).transpose(2, 0, 1)
        if is_reset:
            self._frames.clear()
            for _ in range(_FRAME_STACK):
                self._frames.append(frame)
        else:
            self._frames.append(frame)
        return np.ascontiguousarray(np.concatenate(tuple(self._frames), axis=0))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options:
            raise ValueError("DMControl-v0 does not support non-empty reset options.")
        if seed is not None:
            self._rebuild_for_seed(int(seed))

        timestep = self._env.reset()
        if self.observation_type == "rgb":
            observation = self._pixel_observation(is_reset=True)
        else:
            observation = self._state_observation(timestep.observation)
        return observation, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        native_action = np.asarray(action, dtype=self._action_spec_dtype)
        reward = 0.0
        timestep = None
        for _ in range(_ACTION_REPEAT):
            timestep = self._env.step(native_action)
            reward += float(timestep.reward)
        assert timestep is not None

        if self.observation_type == "rgb":
            observation = self._pixel_observation(is_reset=False)
        else:
            observation = self._state_observation(timestep.observation)
        # Supported DMC tasks are fixed-horizon. Gymnasium's registered
        # TimeLimit supplies truncation on the 500th agent decision.
        return observation, reward, False, False, {}

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        return self._render_frame(_RENDER_SIZE)

    def close(self) -> None:
        self._frames.clear()
        env, self._env = getattr(self, "_env", None), None
        if env is not None:
            self._close_raw_env(env)

    def training_resume_state(self) -> dict[str, Any]:
        """Return the reviewed state-only, between-episode resume contract."""
        if self.observation_type != "state":
            raise ValueError("Training resume does not support DMControl RGB state.")
        task_random = self._env.task.random
        if not isinstance(task_random, np.random.RandomState):
            raise TypeError("DMControl task.random is not numpy.random.RandomState.")
        return {
            "schema_version": 1,
            "task": self.task_name,
            "observation_type": self.observation_type,
            "task_random_state": copy.deepcopy(task_random.get_state()),
        }

    def load_training_resume_state(self, state: Mapping[str, Any]) -> None:
        """Restore reset RNG for the next state-observation episode."""
        self.validate_training_resume_state(state)
        task_random = self._env.task.random
        task_random.set_state(copy.deepcopy(state["task_random_state"]))

    def validate_training_resume_state(self, state: Mapping[str, Any]) -> None:
        """Validate the explicit reset-state contract without mutating it."""
        expected = {
            "schema_version",
            "task",
            "observation_type",
            "task_random_state",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) != expected
            or state.get("schema_version") != 1
        ):
            raise ValueError("Unsupported DMControl training-resume state.")
        if state.get("task") != self.task_name or state.get("observation_type") != "state":
            raise ValueError("DMControl task/observation changed across resume.")
        task_random = self._env.task.random
        if not isinstance(task_random, np.random.RandomState):
            raise TypeError("DMControl task.random is not numpy.random.RandomState.")
        probe = np.random.RandomState()
        probe.set_state(copy.deepcopy(state["task_random_state"]))


__all__ = ["DMControlEnv"]
