"""Full-ensemble aggregation preserves the sampled Dog Stand training contract."""
from copy import deepcopy
import pytest
import torch
from slurm.dog_stand_mrsq import REDUCTIONS, sized_plan, validate_recipe
from tests.test_ambi_root_local_sac import _build_cfg


@pytest.mark.parametrize('index,n', ((0, 5), (1, 10)))
def test_recipe_and_reductions(index, n):
    _, manifest, run = validate_recipe(index)
    cfg = _build_cfg(**run['alg_params'])
    assert cfg.num_q == cfg.q_pair_size == n
    assert all(getattr(cfg, key) == value for key, value in REDUCTIONS.items())
    assert run['seed'] == 55 and manifest['trials'] == 1
    assert run['total_steps'] == 1_000_000 and manifest['env_params']['task'] == 'dog-stand'
    assert run['checkpoint_every'] == manifest['checkpoint_every'] == 25_000
    assert run['save_strat'] == manifest['save_strat'] == 'all'
    assert cfg.inner_execution_action == cfg.inner_behavior_action == 'policy_sample'
    assert cfg.critic_geometry_diagnostics and cfg.critic_geometry_every == 1000
    assert cfg.compile and cfg.compile_strict and cfg.inner_cuda_graph


@pytest.mark.parametrize('n', (5, 10))
def test_full_ensemble_does_not_sample_pairs(n, monkeypatch):
    from RL.tdmpc2_core.common.q_representation import QRepresentation
    cfg = _build_cfg(**validate_recipe(0 if n == 5 else 1)[2]['alg_params'])
    backend = QRepresentation.from_config(cfg)
    def fail(*args, **kwargs):
        pytest.fail('Full-ensemble aggregation must not sample a critic subset')
    monkeypatch.setattr(backend, '_pair_indices', fail)
    values = torch.arange(float(n)).reshape(n, 1, 1).requires_grad_()
    torch.testing.assert_close(backend.reduce(values, 'min_all'), values.min(0).values)
    mean = backend.reduce(values, 'mean_all')
    torch.testing.assert_close(mean, values.mean(0))
    mean.sum().backward()
    torch.testing.assert_close(values.grad, torch.full_like(values, 1 / n))
    assert backend.sample_pair_indices('cpu') is None


def test_sizing_accounts_for_full_run_overhead():
    rows = [dict(index=0, physical_cpus=n, measured={'seconds_per_decision': s},
                 checkpoint_bytes=100, peak_host_memory_bytes=4*2**30) for n,s in ((1,.2),(2,.198))]
    plan = sized_plan(rows, 200000, .18)
    assert plan['cpus'] == 1 and plan['wall_hours'] == 72 and plan['memory_gib'] == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires scheduled CUDA GPU')
@pytest.mark.parametrize('n', (5, 10))
def test_cuda_full_ensemble_dog_parity(n, monkeypatch):
    from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
    from tests import test_inner_entropy_graph as checks
    original = checks.make_agent
    def dog_agent(entropy, scope, update, *, device, compiled):
        cfg = deepcopy(original(entropy, scope, update).cfg)
        cfg.action_dim = 38; cfg.action_dims = [38]
        cfg.obs_shape = {'state': (223,)}; cfg.obs_shapes = [{'state': (223,)}]
        cfg.ent_coef = cfg.inner_temperature = .0021
        cfg.target_entropy = cfg.inner_target_entropy = -38.
        cfg.num_q = cfg.q_pair_size = n
        for key, value in REDUCTIONS.items():
            setattr(cfg, key, value)
        cfg.device = device; cfg.compile = cfg.compile_strict = compiled
        agent = AMBITDMPC2Agent(cfg)
        agent.model.eval(); agent.actor_loss_scale.fill_(4.)
        return agent
    monkeypatch.setattr(checks, 'make_agent', dog_agent)
    checks.test_cuda_capture_matches_uncaptured_normalized_solve('squashed', 'action', 'per_update')
