"""Dog Walk changes only the task of the validated five-critic recipe."""
import json
import os
import subprocess
import sys
import pytest
from slurm.dog_walk_mrsq import ROOT, CAMPAIGN, MANIFEST, validate_recipe
from slurm.dog_stand_mrsq import REDUCTIONS
from tests.test_ambi_root_local_sac import _build_cfg


def test_dog_walk_recipe_preserves_full_ensemble_training():
    _, manifest, run = validate_recipe()
    cfg = _build_cfg(**run['alg_params'])
    assert run['seed'] == 55 and manifest['trials'] == 1
    assert manifest['env_params']['task'] == 'dog-walk'
    assert cfg.num_q == cfg.q_pair_size == 5
    assert all(getattr(cfg, key) == value for key, value in REDUCTIONS.items())
    assert run['total_steps'] == 1_000_000
    assert run['checkpoint_every'] == manifest['checkpoint_every'] == 25_000
    assert run['save_strat'] == manifest['save_strat'] == 'all'
    assert cfg.inner_execution_action == cfg.inner_behavior_action == 'policy_sample'
    assert cfg.critic_geometry_diagnostics and cfg.critic_geometry_every == 1000
    assert cfg.compile and cfg.compile_strict and cfg.inner_cuda_graph


def test_subprocess_worker_uses_dog_walk_manifest_and_campaign():
    program = ('import json; from slurm import inner_entropy_oscar as x; '
               'print(json.dumps([str(x.MANIFEST),str(x.CAMPAIGN_ROOT)]))')
    output = subprocess.check_output([sys.executable, '-c', program], cwd=ROOT, text=True,
                                     env=dict(os.environ, AMBI_ENTROPY_VARIANT='dog_walk_mrsq'))
    assert json.loads(output) == [str(MANIFEST), str(CAMPAIGN)]


@pytest.mark.parametrize('index', (-1, 1))
def test_only_requested_seed_arm_exists(index):
    with pytest.raises(AssertionError):
        validate_recipe(index)


@pytest.mark.skipif(os.environ.get('AMBI_RUN_REAL_DMCONTROL_TESTS') != '1', reason='Opt-in real DMControl check')
def test_real_dog_walk_two_control_steps():
    from tests.test_posttanh_dog import test_real_dog_matches_tdmpc2_two_control_steps
    test_real_dog_matches_tdmpc2_two_control_steps('walk')
