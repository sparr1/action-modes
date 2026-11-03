"""Guard the requested recipes and GPU parity of fixed/fresh/carried alpha."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.whole_inner_graph import (
    WholeInnerGraph, _copy_temperature, _initialize_run_temperature,
    validate_training_graph,
)
from tests.test_ambi_root_local_sac import _build_cfg
from tests.test_inner_entropy_graph import (
    assert_tree_close, assert_graph_resume_continuation, inner_snapshot, outer_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('ambi_prior_entropy_nod_carry_q', 'ambi_prior_entropy_auto_action', 'ambi_prior_entropy_auto_run')
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires scheduled CUDA GPU.')


def params(index):
    return json.loads((ROOT / f'configs/dmcontrol/algs/{NAMES[index]}.json').read_text())['alg_params']


def make_agent(index, *, device='cpu', compiled=False):
    p = params(index)
    p.update(model_size=None, enc_dim=16, mlp_dim=16, latent_dim=8,
             num_enc_layers=2, simnorm_dim=4, batch_size=4, buffer_size=32,
             inner_rounds=1, inner_rollouts_per_round=4, inner_batch_size=4,
             inner_replay_capacity=12, inner_critic_updates_per_round=2,
             inner_actor_updates_per_round=2, inner_cuda_graph=False,
             critic_geometry_diagnostics=False, wandb=False, wandb_mode='disabled',
             compile=compiled, compile_strict=compiled)
    cfg = _build_cfg(**p)
    # Retain production action/observation shapes and all critic RNG branches.
    cfg.action_dim, cfg.action_dims = 21, [21]
    cfg.obs_shape, cfg.device = {'state': (67,)}, device
    cfg.obs_shapes = [cfg.obs_shape]
    agent = AMBITDMPC2Agent(cfg)
    agent.model.eval()
    return agent


@pytest.mark.parametrize('index', range(3))
def test_production_contract(index):
    p = params(index)
    cfg = _build_cfg(**p)
    engine = SimpleNamespace(cfg=cfg, device=torch.device('cuda'), _active_trace=None, _retrace_enabled=False)
    validate_training_graph(engine)
    assert cfg.inner_critic_updates_per_action == 128
    assert cfg.inner_actor_updates_per_action == 32
    assert cfg.inner_temperature_updates_per_action == (0 if index == 0 else 32)
    assert p['inner_cuda_graph'] and p['compile'] and p['compile_strict']
    assert p['critic_ensemble_backend'] == 'vmap'
    assert p['critic_geometry_diagnostics'] and p['critic_geometry_every'] == 1000
    assert p['outer_critic_target'] == p['inner_sac_critic_target'] == 'reward_only'
    assert p['log_std_mapping'] == p['inner_log_std_mapping'] == 'tdmpc2_tanh'
    assert p['outer_actor_entropy_mode'] == p['inner_actor_entropy_mode'] == 'squashed'
    if index:
        assert p['ent_coef'] == 'auto_1.0' and p['target_entropy'] == p['inner_target_entropy'] == -10.5
        assert p['sac_actor_loss_scale_mode'] == 'none'


def test_no_d_only_changes_coefficients_and_publication():
    original = json.loads((ROOT / 'configs/dmcontrol/algs/ambi_inner_entropy_posttanh_carry_adapt.json').read_text())
    changed = json.loads((ROOT / f'configs/dmcontrol/algs/{NAMES[0]}.json').read_text())
    for config in (original, changed):
        config['alg_params'] = {k:v for k,v in config['alg_params'].items() if not k.startswith('wandb')}
    original['alg_params'].update(ent_coef=.0001, inner_temperature=.0001)
    assert original == changed


def test_run_carry_only_changes_temperature_lifetimes():
    fresh, carry = params(1), params(2)
    for p in (fresh, carry):
        for k in list(p):
            if k.startswith('wandb'):
                p.pop(k)
    fresh.update(inner_temperature_scope='run', inner_temperature_optimizer_scope='run')
    assert fresh == carry


def test_temperature_bridge_copies_moments_without_aliasing_or_warmup_leak():
    agent = make_agent(2)
    source = agent.inner_engine.state
    _initialize_run_temperature(agent.inner_engine, source)
    destination = SimpleNamespace(log_alpha=None, temperature_optim=None, temperature_lifetime_steps=0)
    _initialize_run_temperature(agent.inner_engine, destination)
    for state in (source, destination):
        state.temperature_optim.zero_grad()
        (state.log_alpha * 3).backward()
        state.temperature_optim.step()
        state.temperature_lifetime_steps = 1
    pointer = destination.log_alpha.data_ptr()
    _copy_temperature(source, destination)
    assert destination.log_alpha.data_ptr() == pointer != source.log_alpha.data_ptr()
    assert_tree_close(source.temperature_optim.state_dict(), destination.temperature_optim.state_dict())
    agent.reset()
    assert source.log_alpha is agent.inner_engine.state.log_alpha
    assert source.temperature_lifetime_steps == 1
    fresh = SimpleNamespace(log_alpha=None, temperature_optim=None, temperature_lifetime_steps=0)
    _initialize_run_temperature(agent.inner_engine, fresh)
    _copy_temperature(fresh, destination)
    assert destination.temperature_lifetime_steps == 0
    assert all(torch.count_nonzero(v) == 0 for v in destination.temperature_optim.state[destination.log_alpha].values())


@CUDA
@pytest.mark.parametrize('index', range(3))
def test_cuda_recipe_parity(index):
    native, graph = [make_agent(index, device='cuda', compiled=True) for _ in range(2)]
    graph.load(deepcopy(native.checkpoint_state()))
    root = torch.zeros(1, native.cfg.latent_dim, device='cuda')
    control = WholeInnerGraph(native.inner_engine, root, capture=False)
    captured = WholeInnerGraph(graph.inner_engine, root)
    scale = native.actor_loss_scale.clone()
    for i in range(3):
        cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        before = [outer_snapshot(a) for a in (native, graph)]
        kwargs = {'initial_actor_loss_scale': scale} if index == 0 else {}
        expected = control(root + .1*i, **kwargs)
        actual = captured(root + .1*i, **kwargs)
        assert_tree_close(actual, expected, rtol=5e-5, atol=5e-7)
        assert_tree_close(inner_snapshot(graph.inner_engine), inner_snapshot(native.inner_engine), rtol=5e-5, atol=5e-7)
        for agent, snapshot in zip((native, graph), before):
            assert_tree_close(outer_snapshot(agent), snapshot)
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), gpu_rng, rtol=0, atol=0)
        torch.testing.assert_close(control.rng_state, captured.rng_state, rtol=0, atol=0)
        if index == 0:
            scale = expected[1]['inner_actor_loss_scale_final'].reshape_as(scale).clone()
        else:
            initial_step = 2*i if index == 2 else 0
            assert actual[1]['inner_temperature_optimizer_step_initial'] == initial_step
            assert actual[1]['inner_temperature_optimizer_step_final'] == initial_step + 2
        for agent in (native, graph):
            with torch.no_grad():
                next(agent.model._pi.parameters()).add_(.0001)
                if agent.log_ent_coef is not None:
                    agent.log_ent_coef.add_(.03)
    # A separate live adapter checks native diagnostics and exact recapture,
    # including inherited-alpha refresh versus persistent run alpha/Adam.
    training = make_agent(index, device='cuda', compiled=True)
    training.cfg.inner_cuda_graph = True
    training.act(torch.zeros(67), collect_diagnostics=False)
    assert_graph_resume_continuation(training)
