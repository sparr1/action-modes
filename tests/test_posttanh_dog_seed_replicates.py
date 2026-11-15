"""Seed replications preserve the resolved Dog training protocol."""
import pytest
from slurm.posttanh_dog_seed_replicates import validate_recipe
from tests.test_ambi_root_local_sac import _build_cfg


@pytest.mark.parametrize('task', ('walk', 'trot', 'run'))
@pytest.mark.parametrize('index', (0, 1))
def test_only_seed_and_publication_labels_change(task, index):
    _, manifest, run = validate_recipe(task, index)
    cfg = _build_cfg(**run['alg_params'])
    assert run['seed'] == 56 + index and run['total_steps'] == 1_000_000
    assert run['checkpoint_every'] == manifest['checkpoint_every'] == 25_000
    assert run['save_strat'] == manifest['save_strat'] == 'all'
    assert manifest['trials'] == 1
    assert cfg.inner_execution_action == cfg.inner_behavior_action == 'policy_sample'
    assert cfg.ent_coef == cfg.inner_temperature == .0021
    assert cfg.critic_geometry_diagnostics and cfg.critic_geometry_every == 1000
    assert cfg.critic_geometry_batch_size == 64
    assert cfg.critic_geometry_obs_noise_std == .01 and cfg.critic_geometry_action_noise_std == .02
    assert cfg.compile and cfg.compile_strict and cfg.inner_cuda_graph


@pytest.mark.parametrize('task,index', [('walk', 2), ('stand', 0)])
def test_invalid_cell_is_rejected(task, index):
    with pytest.raises(AssertionError):
        validate_recipe(task, index)
