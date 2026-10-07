"""Spectral grid pairing, source pinning and evidence-bound production gates."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluate_ambi_transfer_campaign import listed_cells
from slurm import ambi_spectral_transfer_campaign as launcher
from slurm import ambi_transfer_discovery_campaign as common
from utils.transfer_campaign import load_campaign, resolved_cell

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / 'configs/research/ambi_spectral_transfer_575k.json'


def test_versioned_default_is_compact_and_contains_fair_matrix_controls():
    raw = common.read(MATRIX)
    assert raw == launcher.generate_matrix()
    matrix = load_campaign(MATRIX)
    launcher.validate_matrix(matrix)
    assert len(matrix['arms']) == 16
    assert len(listed_cells(matrix)) == 192
    assert matrix['horizons'] == [1, 2, 3] and matrix['rounds'] == [1, 2, 4, 6]
    assert matrix['spectral_grid']['methods'] == ['svd']
    assert matrix['spectral_grid']['ranks'] == [32]
    assert matrix['spectral_grid']['strengths'] == [1.0]
    assert matrix['seeds'] == [101, 102, 103] and matrix['max_steps'] == 500
    assert {arm['parameter_scope'] for arm in matrix['arms'].values()} == {'matrices'}
    assert not any('reference' in key for key in matrix)
    for name, arm in matrix['arms'].items():
        if not name.startswith('spectral_') or name.endswith('_norm'):
            continue
        matched = matrix['arms'][name + '_norm']
        for component in ('actor', 'critic'):
            if component + '_spectral' in arm:
                selected = arm[component + '_spectral']
                assert matched[component + '_spectral'] == {**selected, 'norm_matched': True}
    base = dict(algorithm_config=dict(alg_params=dict(inner_replay_capacity=3072)))
    resolved = resolved_cell(base, matrix, horizon=3, rounds=6, arm='spectral_svd_r32_s1_joint')
    params = resolved['algorithm_config']['alg_params']
    assert params['inner_replay_capacity'] == 3072
    assert params['inner_critic_updates_per_round'] == 16
    assert params['inner_actor_updates_per_round'] == 4


def test_generator_exposes_all_filters_and_independent_rank_strength_j_choices():
    matrix = launcher.generate_matrix(methods=['svd', 'activation', 'gradient'], ranks=[1, 16],
        strengths=[.25, 1.], rounds=[1, 4], components=['actor', 'joint'])
    launcher.validate_matrix(matrix)
    assert len(matrix['arms']) == 1 + 3 * 2 + 3 * 2 * 2 * 2 * 2
    assert len(listed_cells(matrix)) == len(matrix['arms']) * 3 * 2
    assert len({cell['name'] for cell in listed_cells(matrix)}) == len(listed_cells(matrix))
    assert {cell['J'] for cell in listed_cells(matrix)} == {1, 4}
    assert not any(name.endswith('_critic') for name in matrix['arms'])


def test_generated_sorted_json_roundtrip_is_portable_and_never_launches(tmp_path, monkeypatch):
    output = tmp_path / 'generated.json'
    monkeypatch.setattr(launcher.sys, 'argv', ['generate', 'generate', '--output', str(output),
        '--methods', 'svd', 'activation', 'gradient', '--ranks', '3', '--components', 'critic'])
    monkeypatch.setattr(common, 'source', lambda: pytest.fail('Generation must not prepare a run.'))
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *args, **kwargs: pytest.fail('Generation must not submit.'))
    launcher.main()
    matrix = load_campaign(output)
    launcher.validate_matrix(matrix)
    assert 3 in matrix['spectral_diagnostics']['ranks']
    assert Path(matrix['base_matrix_path']) == ROOT / 'configs/research/ambi_critic_transfer_575k.json'
    assert list(common.read(output)['arms']) == sorted(matrix['arms'])
    with pytest.raises(FileExistsError):
        launcher.main()


@pytest.mark.parametrize('kwargs', [dict(methods=[]), dict(methods=['svd', 'svd']), dict(methods=['random']),
    dict(ranks=[0]), dict(ranks=[True]), dict(ranks=[1.5]), dict(strengths=[float('nan')]),
    dict(strengths=[1.1]), dict(strengths=[True]), dict(rounds=[0]), dict(rounds=[2, 2]),
    dict(components=[]), dict(components=['all'])])
def test_generator_rejects_invalid_or_duplicate_axes(kwargs):
    with pytest.raises(ValueError):
        launcher.generate_matrix(**kwargs)


@pytest.mark.parametrize('change', ['checkpoint', 'seeds', 'horizon', 'budget', 'missing_norm',
    'all_parameter_scope', 'missing_fresh', 'historical_reference', 'diagnostics_off', 'spectral_off',
    'probe_missing', 'first_handoff_missing', 'candidate_rank_missing'])
def test_manifest_cannot_drop_controls_or_drift_protocol(change):
    matrix = launcher.generate_matrix()
    if change == 'checkpoint':
        matrix['checkpoint_contract']['sha256'] = 'wrong'
    elif change == 'seeds':
        matrix['seeds'] = [1, 2, 3]
    elif change == 'horizon':
        matrix['horizons'] = [1, 3]
    elif change == 'budget':
        matrix['actor_updates'] = 8
    elif change == 'missing_norm':
        matrix['arms'].pop('spectral_svd_r32_s1_joint_norm')
    elif change == 'all_parameter_scope':
        matrix['arms']['matrix_bernoulli05_actor']['parameter_scope'] = 'all'
    elif change == 'missing_fresh':
        matrix['arms'].pop('fresh')
    elif change == 'historical_reference':
        matrix['historical_reference'] = '/old/results'
    elif change == 'diagnostics_off':
        matrix['diagnostics'] = {}
    elif change == 'spectral_off':
        matrix['spectral_diagnostics'] = {}
    elif change == 'probe_missing':
        matrix.pop('spectral_probe')
    elif change == 'first_handoff_missing':
        matrix['spectral_diagnostics']['decisions'] = [0]
    else:
        matrix['spectral_diagnostics']['ranks'] = [1, 2]
    with pytest.raises(ValueError):
        launcher.validate_matrix(matrix)


def _mock_prepare(tmp_path, monkeypatch, *, alter_cells=None, bad_hash=None, dirty=False, expected='tested'):
    checkpoint = tmp_path / 'checkpoint.pt'
    real_digest = common.digest
    monkeypatch.setattr(common, 'digest', lambda path: ('wrong' if bad_hash == 'checkpoint' else launcher.CHECKPOINT_SHA)
        if Path(path) == checkpoint else ('wrong' if bad_hash == 'metadata' else launcher.METADATA_SHA)
        if str(path) == str(checkpoint) + '.metadata.json' else real_digest(path))
    def source():
        if dirty:
            raise ValueError('Campaign requires a clean tested checkout.')
        return dict(source_commit='tested', source_tree='tree', source_dir=str(ROOT))
    monkeypatch.setattr(common, 'source', source)
    cells = listed_cells(load_campaign(MATRIX))
    if alter_cells:
        alter_cells(cells)
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda *args, **kwargs: json.dumps(cells))
    args = SimpleNamespace(root=tmp_path / 'prepared', checkpoint=checkpoint, matrix=MATRIX, expected_source=expected)
    launcher.prepare(args)
    return common.read(args.root / 'campaign.json')


def test_prepare_pins_arms_inputs_clean_source_and_all_maximum_budget_smokes(tmp_path, monkeypatch):
    campaign = _mock_prepare(tmp_path, monkeypatch)
    matrix = common.read(MATRIX)
    assert campaign['family'] == 'spectral_transfer'
    assert campaign['source_commit'] == 'tested' and campaign['source_tree'] == 'tree'
    assert campaign['matrix_sha256'] == common.digest(MATRIX)
    assert campaign['base_matrix_sha256'] == common.digest(Path(campaign['base_matrix_path']))
    assert campaign['metadata_path'] == campaign['checkpoint'] + '.metadata.json'
    for key in ('arms', 'diagnostics', 'spectral_diagnostics', 'spectral_probe'):
        assert campaign[key] == matrix[key]
    assert campaign['historical_reference'] is None and campaign['historical_references'] == []
    smoke = [campaign['cells'][index] for index in campaign['smoke_indices']]
    assert len(smoke) == 17
    maximum = [cell for cell in smoke if cell['H'] == 3 and cell['J'] == 6]
    assert {cell['arm'] for cell in maximum} == set(matrix['arms'])
    assert any(cell['arm'] == 'fresh' and cell['H'] == 1 and cell['J'] == 1 for cell in smoke)
    assert campaign['smoke_seeds'] == [101, 102] and campaign['smoke_steps'] == 3


@pytest.mark.parametrize('kwargs', [dict(bad_hash='checkpoint'), dict(bad_hash='metadata'),
    dict(dirty=True), dict(expected='untested'),
    dict(alter_cells=lambda cells: cells.pop()),
    dict(alter_cells=lambda cells: cells[0].update(index=99)),
    dict(alter_cells=lambda cells: cells[0].update(name=cells[1]['name'])),
    dict(alter_cells=lambda cells: cells[0].update(H=2)),
])
def test_prepare_rejects_unpinned_inputs_before_creating_output(tmp_path, monkeypatch, kwargs):
    with pytest.raises(ValueError):
        _mock_prepare(tmp_path, monkeypatch, **kwargs)
    assert not (tmp_path / 'prepared').exists()


@pytest.mark.parametrize('corruption', ['status', 'smoke', 'cell', 'result_sha256', 'manifest_sha256',
    'source_commit', 'campaign_sha256', 'validation'])
def test_production_requires_verified_smoke_receipts_and_bound_results(tmp_path, monkeypatch, corruption):
    root = tmp_path / 'campaign'
    root.mkdir()
    cell = dict(index=0, name='h3_j6_fresh', H=3, J=6, arm='fresh')
    source = dict(source_commit='tested', source_tree='tree', source_dir=str(ROOT))
    campaign = dict(**source, family='spectral_transfer', cells=[cell], smoke_indices=[0])
    for path_key, hash_key in (('checkpoint', 'checkpoint_sha256'), ('metadata_path', 'metadata_sha256'),
                               ('base_matrix_path', 'base_matrix_sha256')):
        path = root / path_key
        path.write_text(path_key)
        campaign[path_key], campaign[hash_key] = str(path), common.digest(path)
    common.write(root / 'campaign.json', campaign)
    smoke_dir = root / 'smoke' / cell['name']
    smoke_dir.mkdir(parents=True)
    for filename in ('results.json', 'manifest.json'):
        common.write(smoke_dir / filename, {})
    receipt = dict(status='complete', smoke=True, cell=cell, source_commit='tested',
        campaign_sha256=common.digest(root / 'campaign.json'),
        result_sha256=common.digest(smoke_dir / 'results.json'),
        manifest_sha256=common.digest(smoke_dir / 'manifest.json'))
    if corruption != 'validation':
        receipt[corruption] = 'corrupt'
    common.write(smoke_dir / 'worker-completion.json', receipt)
    monkeypatch.setattr(common, 'source', lambda: source)
    def invalid_result(*args, **kwargs):
        assert kwargs['smoke'] is True
        raise ValueError('Diagnostic isolation verification failed.')
    monkeypatch.setattr(common, 'validate_result', invalid_result)
    monkeypatch.setattr(common.subprocess, 'run', lambda *args, **kwargs: pytest.fail('Must reject before evaluation.'))
    with pytest.raises(ValueError):
        common.worker(SimpleNamespace(root=root, index=0, smoke=False))
    assert not (root / 'settings').exists()


def test_oscar_wrapper_requires_pinned_source_and_real_cuda_without_fixing_array_concurrency():
    script = (ROOT / 'slurm/run_ambi_spectral_transfer_oscar.sbatch').read_text()
    assert 'EXPECTED_ACTION_MODES_SHA' in script and 'git status --porcelain' in script
    assert 'args=(prepare' in script and 'slurm/ambi_spectral_transfer_campaign.py "${args[@]}"' in script
    assert 'slurm/ambi_transfer_discovery_campaign.py "${args[@]}"' in script
    assert 'torch.cuda.is_available()' in script and 'probe.item() == 1024.0' in script
    assert 'Expected L40S timing hardware' in script and '--array' not in script
    assert 'SLURM_RESTART_COUNT' in script and 'WANDB_MODE=disabled' in script


@pytest.mark.parametrize('changed', ['checkpoint', 'metadata_path', 'base_matrix_path'])
def test_worker_rechecks_every_pinned_scientific_input_before_compute(tmp_path, monkeypatch, changed):
    root = tmp_path / 'campaign'
    root.mkdir()
    source = dict(source_commit='tested', source_tree='tree', source_dir=str(ROOT))
    campaign = dict(**source, family='spectral_transfer')
    for path_key, hash_key in (('checkpoint', 'checkpoint_sha256'), ('metadata_path', 'metadata_sha256'),
                               ('base_matrix_path', 'base_matrix_sha256')):
        path = root / path_key
        path.write_text(path_key)
        campaign[path_key], campaign[hash_key] = str(path), common.digest(path)
    common.write(root / 'campaign.json', campaign)
    Path(campaign[changed]).write_text('altered since preparation')
    monkeypatch.setattr(common, 'source', lambda: source)
    with pytest.raises(ValueError, match='pinned input changed'):
        common.worker(SimpleNamespace(root=root, index=0, smoke=True))
