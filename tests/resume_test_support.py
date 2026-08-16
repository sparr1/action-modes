"""Shared exact-resume fixtures; this module is support, not a test suite."""

import copy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.TDMPC2 import TDMPC2Baseline
from tests.test_tdmpc2_correctness import tiny_params
from utils.resume_runtime import register_test_resume_environment
from utils.resume_training import TrainingResumeSession


class BoundaryEnv(gym.Env):
    metadata = {}

    def __init__(self, *, on_first_done=None):
        self.observation_space = gym.spaces.Box(
            -10.0, 10.0, shape=(3,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(1,), dtype=np.float32
        )
        self.spec = SimpleNamespace(
            id="BoundaryResume-v0", max_episode_steps=2
        )
        self.on_first_done = on_first_done
        self.trace = []
        self._episode_step = 0
        self._done_count = 0

    def reset(self, *, seed=None, options=None):
        del options
        super().reset(seed=seed)
        self._episode_step = 0
        return self.np_random.normal(size=3).astype(np.float32), {}

    def step(self, action):
        self._episode_step += 1
        action = np.asarray(action, dtype=np.float32).copy()
        reward = float(self._episode_step + 0.25 * action[0])
        truncated = self._episode_step == 2
        observation = self.np_random.normal(size=3).astype(np.float32)
        self.trace.append((action, reward, observation.copy(), truncated))
        if truncated:
            self._done_count += 1
            if self._done_count == 1 and self.on_first_done is not None:
                self.on_first_done()
        return observation, reward, False, truncated, {}

    def training_resume_state(self):
        return {
            "schema_version": 1,
            "rng": copy.deepcopy(self.np_random.bit_generator.state),
        }

    def validate_training_resume_state(self, state):
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise ValueError("invalid BoundaryEnv resume state")
        probe = copy.deepcopy(self.np_random)
        probe.bit_generator.state = copy.deepcopy(state["rng"])

    def load_training_resume_state(self, state):
        self.validate_training_resume_state(state)
        self.np_random.bit_generator.state = copy.deepcopy(state["rng"])


register_test_resume_environment(BoundaryEnv, episode_steps=2)


class _RemoteRun:
    def __init__(self):
        self.history = []
        self.finish_count = 0

    def log(self, payload, step):
        assert step == len(self.history)
        self.history.append({"_step": int(step), **dict(payload)})

    def scan_history(self, *, keys=None, page_size):
        del page_size
        if keys is None:
            return [dict(row) for row in self.history]
        return [{key: row[key] for key in keys} for row in self.history]

    def finish(self):
        self.finish_count += 1


class _FakeWandb:
    __version__ = "0.17.4"

    def __init__(self):
        self.runs = {}

    def init(
        self,
        *,
        id=None,
        resume=None,
        project=None,
        entity=None,
        name=None,
        config=None,
        mode=None,
        dir=None,
        tags=None,
        group=None,
    ):
        del project, entity, name, config, mode, dir, tags, group
        if resume == "never":
            assert id not in self.runs
            self.runs[id] = _RemoteRun()
        else:
            assert resume == "must" and id in self.runs
        return self.runs[id]

    def define_metric(self, *_args, **_kwargs):
        return None

    def Api(self, *, timeout=None):
        assert timeout == 20
        module = self

        class API:
            def run(self, path):
                return module.runs[path.rsplit("/", 1)[-1]]

        return API()


def _model(
    env,
    *,
    algorithm=TDMPC2Baseline,
    inner_scope=None,
    total_steps=4,
    **param_overrides,
):
    params = tiny_params(
        episode_length=2,
        total_steps=total_steps,
        seed_steps=1,
        pretrain_steps=1,
        utd=1,
        train_unroll_horizon=1,
        outer_planning_horizon=1,
        batch_size=1,
        buffer_size=16,
        wandb=True,
        wandb_mode="online",
        wandb_step_every=2,
        mpc=False,
        dropout=0.0,
    )
    params.update(param_overrides)
    if inner_scope is not None:
        for key in (
            "inner_adaptation",
            "inner_iterations",
            "inner_rollouts",
            "inner_horizon",
            "inner_updates_per_iteration",
            "inner_tau",
        ):
            params.pop(key, None)
        params.update(
            inner_operator="sac",
            inner_model_step_budget=1,
            inner_rounds=1,
            inner_rollout_horizon=1,
            inner_critic_updates_per_action=1,
            inner_actor_updates_per_action=1,
            inner_temperature_updates_per_action=1,
            inner_batch_size=1,
            inner_replay_capacity=4,
            inner_actor_adaptation="clone",
            inner_critic_adaptation="clone",
            inner_temperature_mode="auto",
            inner_temperature_initialization="fixed",
            inner_actor_scope=inner_scope,
            inner_critic_scope=inner_scope,
            inner_temperature_scope=inner_scope,
            inner_replay_scope=inner_scope,
            inner_actor_optimizer_scope=inner_scope,
            inner_critic_optimizer_scope=inner_scope,
            inner_temperature_optimizer_scope=inner_scope,
        )
    model = algorithm(
        algorithm.__name__,
        env,
        params,
        {
            "seed": 7,
            "device": "cpu",
            "env": "BoundaryResume-v0",
            "total_steps": total_steps,
        },
        {},
    )
    return model.enable_training_resume(total_timesteps=total_steps)


def _session(path, *, mode, segment, generation=None, total_steps=4):
    return TrainingResumeSession.open(
        path,
        mode=mode,
        scientific_identity={"fingerprint": "resume-tests"},
        total_steps=total_steps,
        checkpoint_minutes=10_000,
        drain_after_seconds=None,
        resume_generation=generation,
        segment_id=segment,
    )


def _assert_tree_equal_nan(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(
            actual, expected, rtol=0, atol=0, equal_nan=True
        )
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_equal_nan(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_tree_equal_nan(actual_item, expected_item)
    else:
        assert actual == expected


def _replay_state(buffer, *, max_rows=65_536):
    """Materialize the production sharded replay protocol for test comparison."""
    return {
        "metadata": buffer.training_state_metadata(),
        "shards": list(buffer.iter_training_state_shards(max_rows=max_rows)),
    }
