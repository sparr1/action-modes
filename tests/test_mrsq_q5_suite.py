"""Task routing and real environment/GPU checks for the remaining Q5 suite."""
from copy import deepcopy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from slurm.mrsq_q5_suite import ROOT, SUITE, TASKS, manifest_for, validate_recipe
from slurm.dog_stand_mrsq import REDUCTIONS
from tests.test_ambi_root_local_sac import _build_cfg


@pytest.mark.parametrize('task', TASKS)
def test_task_only_recipe(task):
    _, manifest, run = validate_recipe(task)
    cfg = _build_cfg(**run['alg_params'])
    assert manifest['env_params']['task'] == task
    assert run['seed'] == 55 and run['total_steps'] == 1_000_000
    assert cfg.num_q == cfg.q_pair_size == 5
    assert all(getattr(cfg, key) == value for key, value in REDUCTIONS.items())
    assert run['checkpoint_every'] == 25_000 and run['save_strat'] == 'all'
    assert cfg.critic_geometry_diagnostics and cfg.critic_geometry_every == 1000


@pytest.mark.parametrize('task', TASKS)
def test_worker_subprocess_routes_to_its_own_task(task):
    program = ('import json; from slurm import inner_entropy_oscar as x; '
               'print(json.dumps([str(x.MANIFEST),str(x.CAMPAIGN_ROOT)]))')
    output = subprocess.check_output([sys.executable, '-c', program], cwd=ROOT, text=True,
        env=dict(os.environ, AMBI_ENTROPY_VARIANT='mrsq_q5_suite', AMBI_MRSQ_TASK=task))
    assert json.loads(output) == [str(manifest_for(task)), str(SUITE / task)]


@pytest.mark.parametrize('task', ('dog-stand', 'dog-walk', '../dog-run'))
def test_existing_and_unknown_tasks_are_excluded(task):
    with pytest.raises(ValueError):
        validate_recipe(task)


@pytest.mark.skipif(os.environ.get('AMBI_RUN_REAL_DMCONTROL_TESTS') != '1', reason='Opt-in real DMControl check')
@pytest.mark.parametrize('task', TASKS)
def test_real_task_action_repeat(task):
    import gymnasium as gym
    from dm_control import suite
    from dm_control.suite.wrappers import action_scale
    import domains  # noqa: F401
    family, behavior = task.split('-')
    obs_dim, action_dim = {'dog': (223, 38), 'humanoid': (67, 21)}[family]
    env = gym.make('DMControl-v0', task=task, obs='state', render_mode=None)
    raw = action_scale.Wrapper(suite.load(family, behavior, task_kwargs={'random': 55},
                                       visualize_reward=False), minimum=-1., maximum=1.)
    def flatten(values):
        return np.concatenate([np.asarray(v).reshape(-1) for v in values.values()]).astype(np.float32)
    try:
        obs, _ = env.reset(seed=55)
        np.testing.assert_array_equal(obs, flatten(raw.reset().observation))
        assert obs.shape == (obs_dim,) and env.action_space.shape == (action_dim,)
        assert env.unwrapped.action_repeat == 2 and env.spec.max_episode_steps == 500
        assert env.unwrapped._effective_control_timestep == pytest.approx(2 * raw.control_timestep())
        rng = np.random.default_rng(55)
        for _ in range(5):
            action = rng.uniform(-1., 1., size=(action_dim,)).astype(env.action_space.dtype)
            reward = 0.
            for _ in range(2):
                step = raw.step(action)
                reward += float(step.reward)
            actual_obs, actual_reward, terminated, truncated, _ = env.step(action)
            np.testing.assert_array_equal(actual_obs, flatten(step.observation))
            assert actual_reward == pytest.approx(reward, rel=0, abs=1e-12)
            assert not terminated and not truncated
    finally:
        env.close()
        raw.physics.free()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires scheduled CUDA GPU')
@pytest.mark.parametrize('family', ('dog', 'humanoid'))
def test_cuda_q5_family_parity(family, monkeypatch):
    from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
    from tests import test_inner_entropy_graph as checks
    original = checks.make_agent
    obs_dim, action_dim = {'dog': (223, 38), 'humanoid': (67, 21)}[family]
    def make_agent(entropy, scope, update, *, device, compiled):
        cfg = deepcopy(original(entropy, scope, update).cfg)
        cfg.action_dim = action_dim
        cfg.action_dims = [action_dim]
        cfg.obs_shape = {'state': (obs_dim,)}
        cfg.obs_shapes = [{'state': (obs_dim,)}]
        cfg.ent_coef = cfg.inner_temperature = .0021
        cfg.target_entropy = cfg.inner_target_entropy = -float(action_dim)
        cfg.num_q = cfg.q_pair_size = 5
        for key, value in REDUCTIONS.items():
            setattr(cfg, key, value)
        cfg.device = device
        cfg.compile = cfg.compile_strict = compiled
        agent = AMBITDMPC2Agent(cfg)
        agent.model.eval()
        agent.actor_loss_scale.fill_(4.)
        return agent
    monkeypatch.setattr(checks, 'make_agent', make_agent)
    checks.test_cuda_capture_matches_uncaptured_normalized_solve('squashed', 'action', 'per_update')
