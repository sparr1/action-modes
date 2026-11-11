"""Dog task replication, TD-MPC2 action-repeat parity, and GPU graph parity."""
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_ambi_root_local_sac import _build_cfg

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('task', ('run', 'walk', 'trot'))
def test_task_only_recipe(task):
    folder = ROOT / 'configs/dmcontrol'
    source = json.loads((folder/'algs/ambi_inner_entropy_posttanh_outer_adapt.json').read_text())
    target = json.loads((folder/f'algs/ambi_posttanh_outer_adapt_dog_{task}.json').read_text())
    params = target['alg_params'].copy()
    for value in (source, target):
        value['alg_params'] = {k:v for k,v in value['alg_params'].items() if not k.startswith('wandb')}
    assert source == target
    source_manifest = json.loads((folder/'experiments/ambi_inner_entropy.json').read_text())
    manifest = json.loads((folder/f'experiments/ambi_posttanh_outer_adapt_dog_{task}.json').read_text())
    source_manifest['env_params']['task'] = 'dog-'+task
    for key in ('overrides_alg','env_params','trials','logs','save_trials','checkpoint_every','save_strat','log_info','log_type'):
        assert source_manifest[key] == manifest[key]
    assert manifest['configs'] == ['ambi_posttanh_outer_adapt_dog_'+task]
    cfg = _build_cfg(**params)
    assert cfg.inner_execution_action == 'policy_sample'
    assert cfg.ent_coef == cfg.inner_temperature == .0021
    assert cfg.inner_actor_loss_scale_scope == 'action' and cfg.inner_actor_loss_scale_update == 'per_update'
    assert cfg.critic_geometry_diagnostics and cfg.critic_geometry_every == 1000
    assert cfg.compile and cfg.compile_strict and cfg.inner_cuda_graph


@pytest.mark.skipif(os.environ.get('AMBI_RUN_REAL_DMCONTROL_TESTS') != '1', reason='Opt-in real DMControl check')
@pytest.mark.parametrize('task', ('run', 'walk', 'trot'))
def test_real_dog_matches_tdmpc2_two_control_steps(task):
    import gymnasium as gym
    from dm_control import suite
    from dm_control.suite.wrappers import action_scale
    import domains  # noqa: F401
    env = gym.make('DMControl-v0',task='dog-'+task,obs='state',render_mode=None)
    raw = action_scale.Wrapper(suite.load('dog',task,task_kwargs={'random':55},visualize_reward=False),minimum=-1.,maximum=1.)
    def flatten(values):
        return np.concatenate([np.asarray(v).reshape(-1) for v in values.values()]).astype(np.float32)
    try:
        observation,_ = env.reset(seed=55)
        np.testing.assert_array_equal(observation,flatten(raw.reset().observation))
        assert observation.shape == (223,) and env.action_space.shape == (38,)
        assert env.unwrapped.action_repeat == 2 and env.spec.max_episode_steps == 500
        assert raw.control_timestep() == pytest.approx(.015)
        assert env.unwrapped._effective_control_timestep == pytest.approx(.03)
        generator = np.random.default_rng(55)
        for _ in range(5):
            action = generator.uniform(-1.,1.,size=(38,)).astype(env.action_space.dtype)
            reward = 0.
            for repeat in range(2):
                step=raw.step(action); reward += float(step.reward)
            observed,actual,terminated,truncated,_ = env.step(action)
            np.testing.assert_array_equal(observed,flatten(step.observation))
            assert actual == pytest.approx(reward,rel=0,abs=1e-12)
            assert not terminated and not truncated
    finally:
        env.close();raw.physics.free()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires scheduled CUDA GPU')
def test_cuda_dog_capture_matches_uncaptured(monkeypatch):
    from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
    from tests import test_inner_entropy_graph as graph_tests
    original = graph_tests.make_agent
    def dog_agent(entropy,scope,update,*,device,compiled):
        cfg=deepcopy(original(entropy,scope,update).cfg)
        cfg.action_dim=38;cfg.action_dims=[38]
        cfg.obs_shape={'state':(223,)};cfg.obs_shapes=[{'state':(223,)}]
        cfg.ent_coef=.0021;cfg.inner_temperature=.0021
        cfg.target_entropy=-38.;cfg.inner_target_entropy=-38.
        cfg.device=device;cfg.compile=compiled;cfg.compile_strict=compiled
        agent=AMBITDMPC2Agent(cfg);agent.model.eval();agent.actor_loss_scale.fill_(4.)
        return agent
    monkeypatch.setattr(graph_tests,'make_agent',dog_agent)
    graph_tests.test_cuda_capture_matches_uncaptured_normalized_solve('squashed','action','per_update')
