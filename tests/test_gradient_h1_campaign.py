"""H1 gradient transfer grids retain paired budgets and evidence-bound smoke gates."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluate_ambi_transfer_campaign import listed_cells
from slurm import ambi_spectral_transfer_campaign as launcher
from slurm import ambi_transfer_discovery_campaign as common
from utils.transfer_campaign import load_campaign

ROOT = Path(__file__).resolve().parents[1]


def gradient_grid(**kwargs):
    options = dict(methods=['gradient', 'gradient_gate', 'gradient_projection'], ranks=[1, 4],
        strengths=[.5, 1.], horizons=[1], rounds=[1, 2, 4, 6], components=['actor', 'critic', 'joint'])
    return launcher.generate_matrix(**(options | kwargs))



def test_versioned_gradient_h1_config_has_reviewed_scope_and_publication_identity():
    path = ROOT / 'configs/research/ambi_gradient_transfer_h1_575k.json'
    assert common.read(path) == launcher.generate_gradient_h1()
    matrix = load_campaign(path)
    launcher.validate_matrix(matrix)
    assert matrix['spectral_grid'] == dict(methods=['gradient', 'gradient_gate', 'gradient_projection'],
        ranks=[1, 4], strengths=[.5, 1.], components=['actor', 'critic', 'joint'])
    assert matrix['horizons'] == [1] and matrix['rounds'] == [1, 2, 4, 6]
    assert len(listed_cells(matrix)) == 232
    assert matrix['publication']['gradient_alignment_view'] is True
    assert matrix['publication']['slug_prefix'] == 'gradienth1575'
    assert matrix['publication']['group_prefix'] == 'gradient-h1-transfer-575k'
    assert matrix['publication']['title'] == '575K H1 gradient transfer · SVD, gating and projection'
    assert 'reuse' not in matrix


def test_h1_gradient_methods_have_exact_unique_cell_grid_and_paired_controls():
    matrix = gradient_grid()
    launcher.validate_matrix(matrix)
    cells = listed_cells(matrix)
    assert matrix['horizons'] == [1] and matrix['rounds'] == [1, 2, 4, 6]
    assert len(matrix['arms']) == 58
    assert len(cells) == len({cell['name'] for cell in cells}) == 232
    assert {cell['H'] for cell in cells} == {1}
    assert matrix['seeds'] == [101, 102, 103] and matrix['max_steps'] == 500
    assert matrix['critic_updates'] == 16 and matrix['actor_updates'] == 4
    assert matrix['rollouts'] == 128 and matrix['batch_size'] == 256
    expected_methods = {'gradient': 24, 'gradient_gate': 12, 'gradient_projection': 12}
    for method, expected in expected_methods.items():
        arms = [arm for arm in matrix['arms'].values()
                if next((value['method'] for key, value in arm.items() if key.endswith('_spectral')), None) == method]
        assert len(arms) == expected
    for name, arm in matrix['arms'].items():
        if not name.startswith('spectral_') or name.endswith('_norm'):
            continue
        for key, spec in arm.items():
            if key.endswith('_spectral'):
                assert matrix['arms'][name + '_norm'][key] == dict(spec, norm_matched=True)
                assert (spec['rank'] in (1, 4)) if spec['method'] == 'gradient' else spec['rank'] is None
    gate = matrix['arms']['spectral_gradient_gate_s0p5_actor_norm']
    projection = matrix['arms']['spectral_gradient_projection_s0p5_actor_norm']
    assert 'per-layer' in gate['description'] and 'global component' in projection['description']
    assert not any('reuse' in key or 'reference' in key for key in matrix)


@pytest.mark.parametrize('method', ['gradient_gate', 'gradient_projection'])
def test_nonrank_gradient_methods_never_duplicate_over_supplied_ranks(method):
    settings = dict(methods=[method], horizons=[1], strengths=[.5, 1.], components=['critic'])
    matrix = launcher.generate_matrix(ranks=[1, 4, 32], **settings)
    assert matrix == launcher.generate_matrix(ranks=[], **settings)
    launcher.validate_matrix(matrix)
    assert matrix['spectral_grid']['ranks'] == []
    assert len(matrix['arms']) == 8
    assert all(arm['critic_spectral']['rank'] is None for name, arm in matrix['arms'].items()
               if name.startswith('spectral_'))


@pytest.mark.parametrize('horizons', [[], [0], [4], [1, 1], [1, 4], [True], [1.], ['1']])
def test_horizon_axis_rejects_invalid_values_in_generation_and_validation(horizons):
    with pytest.raises(ValueError, match='horizons'):
        launcher.generate_matrix(horizons=horizons)
    matrix = launcher.generate_matrix()
    matrix['horizons'] = horizons
    with pytest.raises(ValueError, match='horizons'):
        launcher.validate_matrix(matrix)


@pytest.mark.parametrize('horizons', [[1], [2], [3], [1, 3], [3, 2, 1]])
def test_horizon_axis_accepts_explicit_unique_supported_subsets(horizons):
    matrix = launcher.generate_matrix(horizons=horizons)
    launcher.validate_matrix(matrix)
    assert matrix['horizons'] == horizons
    assert {cell['H'] for cell in listed_cells(matrix)} == set(horizons)


def test_h1_cannot_weaken_original_rank_extension_reuse_contract():
    matrix = launcher.generate_rank_extension()
    matrix['horizons'] = [1]
    with pytest.raises(ValueError, match='restricted'):
        launcher.validate_matrix(matrix)


def test_h1_generator_cli_writes_config_without_preparation_or_submission(tmp_path, monkeypatch):
    output = tmp_path / 'gradient-h1.json'
    monkeypatch.setattr(launcher.sys, 'argv', ['generate', 'generate', '--output', str(output),
        '--methods', 'gradient', 'gradient_gate', 'gradient_projection', '--ranks', '1', '4',
        '--strengths', '.5', '1', '--horizons', '1', '--rounds', '1', '2', '4', '6'])
    monkeypatch.setattr(common, 'source', lambda: pytest.fail('Generation must not prepare.'))
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k: pytest.fail('Generation must not submit.'))
    launcher.main()
    matrix = load_campaign(output)
    launcher.validate_matrix(matrix)
    assert matrix['horizons'] == [1]
    assert len(listed_cells(matrix)) == 232


def test_h1_preparation_smokes_every_arm_at_max_j_and_minimum_fresh(tmp_path, monkeypatch):
    matrix_path = tmp_path / 'matrix.json'
    common.write(matrix_path, gradient_grid())
    checkpoint = tmp_path / 'checkpoint.pt'
    digest = common.digest
    monkeypatch.setattr(common, 'digest', lambda path: launcher.CHECKPOINT_SHA
        if Path(path) == checkpoint else launcher.METADATA_SHA
        if str(path) == str(checkpoint) + '.metadata.json' else digest(path))
    monkeypatch.setattr(common, 'source', lambda: dict(source_commit='tested', source_tree='tree', source_dir=str(ROOT)))
    cells = listed_cells(load_campaign(matrix_path))
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda *a, **k: json.dumps(cells))
    args = SimpleNamespace(root=tmp_path / 'prepared', checkpoint=checkpoint,
        matrix=matrix_path, expected_source='tested')
    launcher.prepare(args)
    campaign = common.read(args.root / 'campaign.json')
    assert campaign['H'] == [1] and campaign['J'] == [1, 2, 4, 6]
    assert campaign['cells'] == cells
    assert campaign['historical_references'] == []
    assert campaign['smoke_seeds'] == [101, 102] and campaign['smoke_steps'] == 3
    smoke = [campaign['cells'][index] for index in campaign['smoke_indices']]
    assert len(smoke) == 59
    assert all(cell['H'] == 1 for cell in smoke)
    assert {cell['arm'] for cell in smoke if cell['J'] == 6} == set(campaign['arms'])
    assert [(cell['J'], cell['arm']) for cell in smoke if cell['J'] != 6] == [(1, 'fresh')]
