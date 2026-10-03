"""Freeze horizon campaign provenance, independent axes, reuse and GPU checks."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from slurm import ambi_aux_horizon_mppi_campaign as campaign


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def backbone_fixture(tmp_path, horizon):
    baseline = campaign.read(campaign.BASELINE)
    baseline['alg_params']['train_unroll_horizon'] = horizon
    baseline['resolved_runtime'] = {'observation': {'shape': [67], 'mode': 'state', 'action_dim': 21, 'episode_length': 500}}
    path = tmp_path / f'h{horizon}' / 'model_575000'
    path.parent.mkdir(parents=True)
    path.write_bytes(f'unused-metadata-only-weights-h{horizon}'.encode())
    metadata = {'schema_version': 1,
        'checkpoint': {'kind': 'periodic', 'step': 575000, 'episode': 1150, 'best_score': None, 'best_window': 100},
        'trial_run_params': baseline,
        'experiment_params': {'env_params': {'task': 'humanoid-walk', 'obs': 'state', 'render_mode': None}}}
    write(Path(str(path) + '.metadata.json'), metadata)
    source = f'entity/ambi/trainh{horizon}'
    launch = path.parent / 'launch.json'
    write(launch, {'binding': {'source_commit': 'a' * 40}, 'wandb': {'path': source}, 'run_params': baseline})
    return {'train_horizon': horizon, 'source_run': source, 'training_launch': str(launch),
            'checkpoints': [{'step': 575000, 'path': str(path), 'sha256': campaign.digest(path),
                             'metadata_sha256': campaign.digest(str(path) + '.metadata.json')}]}


@pytest.mark.parametrize('horizon,budget', [(1, 4096), (3, 12336), (5, 20576), (7, 28816)])
def test_mppi_matrix_changes_only_planning_horizon_and_derived_budget(horizon, budget):
    matrix = campaign.matrix_for('entity/project/backbone', horizon)
    expected = campaign.read(campaign.ROOT / 'configs/research/ambi_aux_return_mppi.json')
    expected['description'] = matrix['description']
    expected['source_run'] = 'entity/project/backbone'
    expected['shared_alg_params']['inner_rollout_horizon'] = horizon
    expected['shared_alg_params']['inner_model_step_budget'] = budget
    assert matrix == expected
    assert 'train_unroll_horizon' not in matrix['shared_alg_params']
    assert 'inner_mppi_num_samples' not in matrix['shared_alg_params']
    assert matrix['shared_alg_params']['inner_critic_source'] == 'sac'


@pytest.mark.parametrize('horizon', campaign.HORIZONS)
def test_training_metadata_matches_exact_backbone_and_launch(tmp_path, horizon):
    backbone = backbone_fixture(tmp_path, horizon)
    context = campaign.validate_checkpoint(backbone, backbone['checkpoints'][0])
    assert context.trial_run_params['alg_params']['train_unroll_horizon'] == horizon


@pytest.mark.parametrize('failure', ['horizon', 'entropy', 'actor', 'detached', 'weights', 'sidecar', 'source', 'step'])
def test_wrong_or_changed_checkpoint_is_rejected(tmp_path, failure):
    backbone = backbone_fixture(tmp_path, 1)
    row = backbone['checkpoints'][0]
    path = Path(row['path'])
    sidecar = Path(str(path) + '.metadata.json')
    metadata = campaign.read(sidecar)
    if failure == 'horizon': backbone['train_horizon'] = 3
    elif failure == 'entropy': metadata['trial_run_params']['alg_params']['target_entropy'] = -21
    elif failure == 'actor': metadata['trial_run_params']['alg_params']['log_std_mapping'] = 'tdmpc2_tanh'
    elif failure == 'detached': metadata['trial_run_params']['alg_params']['aux_return_detach_representation'] = True
    elif failure == 'weights': path.write_bytes(b'replaced')
    elif failure == 'sidecar': row['metadata_sha256'] = '0' * 64
    elif failure == 'source': backbone['source_run'] = 'entity/ambi/another'
    elif failure == 'step': metadata['checkpoint']['step'] = 550000
    if failure in {'entropy', 'actor', 'detached', 'step'}:
        write(sidecar, metadata)
        row['metadata_sha256'] = campaign.digest(sidecar)
    with pytest.raises(ValueError):
        campaign.validate_checkpoint(backbone, row)


def test_prepare_derives_hashes_and_creates_independent_explicit_curves(tmp_path, monkeypatch):
    from utils import ambi_benchmark
    commit = subprocess.check_output(['git', '-C', str(campaign.ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    monkeypatch.setattr(campaign, 'source_commit', lambda: commit)
    monkeypatch.setattr(ambi_benchmark, 'code_identity', lambda: {'commit': commit, 'dirty': False})
    backbones = [backbone_fixture(tmp_path, horizon) for horizon in campaign.HORIZONS]
    for backbone in backbones:
        for row in backbone['checkpoints']:
            del row['sha256'], row['metadata_sha256']
    inventory = tmp_path / 'inventory.json'
    write(inventory, {'schema_version': 1, 'backbones': backbones})
    prepared = campaign.prepare(inventory, tmp_path / 'campaign', tmp_path / 'registry', 'test attempt')
    assert len(prepared['curves']) == 36
    assert len(prepared['prior_tasks']) == 4 and len(prepared['mppi_tasks']) == 16
    assert len({curve['run_id'] for curve in prepared['curves']}) == 36
    assert {(t['train_horizon'], t['planning_horizon']) for t in prepared['mppi_tasks']} == {
        (train, plan) for train in campaign.HORIZONS for plan in campaign.HORIZONS}
    for backbone in prepared['backbones']:
        assert backbone['training_source_commit'] == 'a' * 40
        assert len(backbone['core_recipe_sha256']) == 64
        assert len(backbone['checkpoints'][0]['sha256']) == 64
        assert set(backbone['prior_bundles']) == {'575000'}
    with pytest.raises(FileExistsError):
        campaign.prepare(inventory, tmp_path / 'campaign', tmp_path / 'registry', 'test attempt')


def test_input_refuses_mismatched_checkpoint_grids(tmp_path):
    backbones = [backbone_fixture(tmp_path, horizon) for horizon in campaign.HORIZONS]
    backbones[-1]['checkpoints'] = []
    with pytest.raises(ValueError, match='empty'):
        campaign.validate_input({'schema_version': 1, 'backbones': backbones})


@pytest.mark.parametrize('training_horizon', campaign.HORIZONS)
def test_real_h1_mppi_preserves_each_checkpoint_training_horizon(tmp_path, training_horizon):
    import evaluate_ambi_checkpoint as evaluator
    from tests.test_ambi_root_local_sac import _tiny_model, _tiny_params
    options = dict(aux_return_mode='sac', inner_operator='none', train_unroll_horizon=training_horizon,
                   inner_rounds=0, inner_rollouts_per_round=0, inner_updates_per_round=0)
    model = _tiny_model(**options)
    checkpoint = tmp_path / 'model_575000'
    model.agent.save(checkpoint)
    model.env.close()
    metadata = {'schema_version': 1,
        'checkpoint': {'kind': 'periodic', 'step': 575000, 'episode': 1150, 'best_score': None, 'best_window': 100},
        'trial_run_params': {'alg': 'AMBITDMPC2/AMBITDMPC2', 'env': 'Pendulum-v1',
            'seed': 55, 'device': 'cpu', 'total_steps': 2000000, 'alg_params': _tiny_params(**options)},
        'experiment_params': {'env_params': {'max_episode_steps': 3}}}
    write(Path(str(checkpoint) + '.metadata.json'), metadata)
    matrix = tmp_path / 'matrix.json'
    write(matrix, campaign.matrix_for('entity/project/test', 1))
    payload = evaluator.evaluate_matrix(matrix, checkpoint, seeds=[101, 102], max_steps=3,
                                        device='cpu', bundle_dir=tmp_path / 'bundle')
    campaign.validate_result(payload, {'train_horizon': training_horizon, 'planning_horizon': 1},
                             smoke=True, device='cpu')
    assert len(payload['results']) == 2
    assert all(result['model_metrics']['inner_model_steps']['mean'] == 4096 for result in payload['results'])
