"""Dog Stand preserves the existing sampled learner and three distinct seeds."""
import os
import pytest
from slurm.posttanh_dog_stand import validate_recipe
from tests.test_ambi_root_local_sac import _build_cfg


@pytest.mark.parametrize('index', (0, 1, 2))
def test_dog_stand_recipe(index):
    _, manifest, run = validate_recipe(index)
    cfg = _build_cfg(**run['alg_params'])
    assert run['seed'] == 55 + index and manifest['env_params']['task'] == 'dog-stand'
    assert run['total_steps'] == 1_000_000 and manifest['trials'] == 1
    assert run['checkpoint_every'] == manifest['checkpoint_every'] == 25_000
    assert run['save_strat'] == manifest['save_strat'] == 'all'
    assert cfg.inner_execution_action == cfg.inner_behavior_action == 'policy_sample'
    assert cfg.ent_coef == cfg.inner_temperature == .0021
    assert cfg.critic_geometry_diagnostics and cfg.critic_geometry_every == 1000
    assert cfg.critic_geometry_batch_size == 64
    assert cfg.critic_geometry_obs_noise_std == .01 and cfg.critic_geometry_action_noise_std == .02
    assert cfg.compile and cfg.compile_strict and cfg.inner_cuda_graph


@pytest.mark.parametrize('index', (-1, 3))
def test_invalid_seed_index_rejected(index):
    with pytest.raises(AssertionError):
        validate_recipe(index)


@pytest.mark.skipif(os.environ.get('AMBI_RUN_REAL_DMCONTROL_TESTS') != '1', reason='Opt-in real DMControl check')
def test_real_dog_stand_two_control_steps():
    from tests.test_posttanh_dog import test_real_dog_matches_tdmpc2_two_control_steps
    test_real_dog_matches_tdmpc2_two_control_steps('stand')


def test_matching_memory_does_not_mutate_held_job(monkeypatch):
    from slurm import posttanh_dog_stand as launcher
    fields = {'JobState': 'PENDING', 'Reason': 'JobHeldUser', 'MinMemoryNode': '8G'}
    monkeypatch.setattr(launcher.base, 'slurm_fields', lambda job: fields)
    def unexpected(*args, **kwargs):
        pytest.fail('An allocation already matching the gate should not be changed')
    monkeypatch.setattr(launcher.subprocess, 'run', unexpected)
    launcher.set_pending_memory(123, 8)


def test_memory_resize_uses_integer_megabytes(monkeypatch):
    from slurm import posttanh_dog_stand as launcher
    fields = {'JobState': 'PENDING', 'Reason': 'JobHeldUser', 'MinMemoryNode': '8G'}
    monkeypatch.setattr(launcher.base, 'slurm_fields', lambda job: dict(fields))
    commands = []
    def slurm_update(command, *, check):
        # Oscar rejects a G suffix for this scontrol parameter, unlike sbatch --mem.
        assert command == ['scontrol', 'update', 'JobId=123', 'MinMemoryNode=12288']
        assert check
        commands.append(command)
        fields['MinMemoryNode'] = '12G'
    monkeypatch.setattr(launcher.subprocess, 'run', slurm_update)
    launcher.set_pending_memory(123, 12)
    assert len(commands) == 1


def test_memory_resize_verifies_scheduler_readback(monkeypatch):
    from slurm import posttanh_dog_stand as launcher
    fields = {'JobState': 'PENDING', 'Reason': 'JobHeldUser', 'MinMemoryNode': '8G'}
    monkeypatch.setattr(launcher.base, 'slurm_fields', lambda job: dict(fields))
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *args, **kwargs: None)
    with pytest.raises(AssertionError):
        launcher.set_pending_memory(123, 12)


def test_memory_resize_rejects_running_job(monkeypatch):
    from slurm import posttanh_dog_stand as launcher
    monkeypatch.setattr(launcher.base, 'slurm_fields', lambda job: {'JobState': 'RUNNING'})
    with pytest.raises(AssertionError):
        launcher.set_pending_memory(123, 8)
