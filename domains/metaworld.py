"""Gymnasium adapter for the legacy Meta-World revision used by TD-MPC2."""

import json
from contextlib import contextmanager
from importlib import metadata

import gymnasium as gym
import numpy as np
from gymnasium import spaces


METAWORLD_REVISION = "04be337a12305e393c0caf0cbf5ec7755c7c8feb"
PROTOCOL = "metaworld-tdmpc2-04be337-state-final-success-v1"


def _load_metaworld_environments():
    hint = "See environments/metaworld/README.md for the pinned TD-MPC2 runtime."
    try:
        direct_url = metadata.distribution("metaworld").read_text("direct_url.json")
        revision = json.loads(direct_url or "{}").get("vcs_info", {}).get("commit_id")
        if revision != METAWORLD_REVISION:
            raise ImportError(
                f"Expected Meta-World commit {METAWORLD_REVISION}, found "
                f"{revision or 'an unverified installation'}. {hint}"
            )
        for package, expected in (("gym", "0.21.0"), ("mujoco-py", "2.1.2.14")):
            installed = metadata.version(package)
            if installed != expected:
                raise ImportError(f"Expected {package}=={expected}, found {installed}. {hint}")
    except metadata.PackageNotFoundError as exc:
        raise ImportError(f"Missing {exc.name}. {hint}") from exc

    from mujoco_py import functions
    if functions.mj_version() != 210:
        raise ImportError(f"MetaWorld-v0 requires MuJoCo 2.1.0. {hint}")
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    return ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


def _task_name(task):
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must name a Meta-World v2 task")
    task = task.strip().lower().replace("_", "-")
    if task.startswith("mw-"):
        task = task[3:]
    if task.endswith("-goal-observable"):
        task = task[:-len("-goal-observable")]
    if task.endswith("-v2"):
        task = task[:-3]
    return task


class MetaWorldEnv(gym.Env):
    """One goal-observable v2 task, with fresh object/goal positions each reset.

    Observations have 39 coordinates; actions are Cartesian gripper deltas and
    gripper effort. Two raw rewards are summed for each of 100 decisions.
    Success is the final raw step's info['success'], never an episode maximum.
    Backend reset randomness is isolated from the learner's global NumPy RNG.
    Like other global-RNG legacy adapters, instances must be stepped serially
    within a process (use separate processes for parallel environments).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 40}
    evaluation_success_key = "success"
    observation_type = "state"
    action_repeat = 2
    episode_length = 100
    protocol = PROTOCOL
    backend_revision = METAWORLD_REVISION

    def __init__(self, task, *, obs="state", render_mode=None, seed=None,
                 protocol=PROTOCOL):
        super().__init__()
        if protocol != PROTOCOL:
            raise ValueError(f"Unsupported Meta-World protocol: {protocol!r}")
        if obs != "state":
            raise ValueError("MetaWorld-v0 currently supports only obs='state'")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.task_name = _task_name(task)
        self.render_mode = render_mode
        self._elapsed_steps = 0
        self._needs_reset = True
        self._env = None
        self._random = np.random.RandomState(seed)
        environments = _load_metaworld_environments()
        key = self.task_name + "-v2-goal-observable"
        if key not in environments:
            raise ValueError(f"Unknown Meta-World v2 task: {task!r}")
        self._task_class = environments[key]
        try:
            self._open_backend(seed)
            observation_space = self._env.observation_space
            if observation_space.shape != (39,):
                raise ValueError("Expected the pinned v2 39-dimensional state space")
            raw_actions = self._env.action_space
            if raw_actions.shape != (4,) or not (
                np.all(raw_actions.low == -1) and np.all(raw_actions.high == 1)
            ):
                raise ValueError("Expected four normalized Meta-World actions")
            self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
            self.observation_space = spaces.Box(
                observation_space.low.astype(np.float32),
                observation_space.high.astype(np.float32),
                dtype=np.float32,
            )
        except Exception:
            self.close()
            raise

    def _open_backend(self, seed):
        with self._random_context():
            self._env = self._task_class(seed=seed)
        self._env._freeze_rand_vec = False
        self._env.model.cam_pos[2] = [0.75, 0.075, 0.7]

    @contextmanager
    def _random_context(self):
        # The legacy reset samples the process-global NumPy RNG.
        previous = np.random.get_state()
        np.random.set_state(self._random.get_state())
        try:
            yield
        finally:
            self._random.set_state(np.random.get_state())
            np.random.set_state(previous)

    @staticmethod
    def _observation(observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (39,) or not np.isfinite(observation).all():
            raise ValueError("Meta-World returned an invalid state observation")
        return observation.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._env is None:
            raise RuntimeError("Cannot reset a closed Meta-World environment")
        if options:
            raise ValueError("MetaWorld-v0 does not support reset options")
        self._needs_reset = True
        if seed is not None:
            self._random = np.random.RandomState(seed)
            self.action_space.seed(seed)
            # Legacy observations include a frame from the preceding episode,
            # even on reset. Start at upstream's seeded constructor state for
            # an explicit Gymnasium seed; ordinary resets carry history forward.
            self.close()
            self._open_backend(seed)
        with self._random_context():
            observation = self._env.reset()
            observation = self._observation(observation)
            # Preserve upstream's reset/no-op ordering: return the observation
            # from BEFORE this uncounted settling step, not its successor.
            settled_observation, _, _, _ = self._env.step(np.zeros(self.action_space.shape))
            if settled_observation is None:
                raise RuntimeError(
                    f"MuJoCo 2.1.0 produced no stable state while resetting {self.task_name} "
                    f"on Meta-World {METAWORLD_REVISION}. Check the simulator warning."
                )
        self._elapsed_steps = 0
        self._needs_reset = False
        return observation, {}

    def step(self, action):
        if self._needs_reset or self._env is None:
            raise RuntimeError("Call reset() before stepping MetaWorld-v0")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("Meta-World requires a finite action with shape (4,)")
        if np.any(np.abs(action) > 1.0):
            raise ValueError("Meta-World actions must be within [-1, 1]")
        reward = 0.0
        with self._random_context():
            for _ in range(self.action_repeat):
                observation, raw_reward, _, info = self._env.step(
                    action.copy()
                )
                reward += float(raw_reward)
        if not np.isfinite(reward):
            raise ValueError("Meta-World returned a nonfinite reward")
        self._elapsed_steps += 1
        truncated = self._elapsed_steps >= self.episode_length
        self._needs_reset = truncated
        return self._observation(observation), reward, False, truncated, dict(info)

    def render(self):
        if self.render_mode != "rgb_array":
            raise RuntimeError("Construct MetaWorld-v0 with render_mode='rgb_array'")
        if self._env is None:
            raise RuntimeError("Cannot render a closed Meta-World environment")
        return self._env.render(
            offscreen=True, resolution=(384, 384), camera_name="corner2"
        ).copy()

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
        self._needs_reset = True
