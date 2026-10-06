"""The J8 extension schedules only new cells and pins lower-budget references."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluate_ambi_transfer_campaign import listed_cells
from slurm import ambi_transfer_discovery_campaign as launcher
from utils.transfer_campaign import load_campaign, resolved_cell


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT/'configs/research/ambi_bernoulli_j8_575k.json'
PREVIOUS = ROOT/'configs/research/ambi_bernoulli_transfer_575k.json'


def prepare(tmp_path, monkeypatch, matrix_path=MATRIX):
    matrix = launcher.read(matrix_path)
    cells = [dict(index=index, name=f'h{h}_j{j}_{arm}', H=h, J=j, arm=arm)
        for index, (j, h, arm) in enumerate((j, h, arm) for j in reversed(matrix['rounds'])
            for h in matrix['horizons'] for arm in matrix['arms'])]
    checkpoint = tmp_path/'checkpoint.pt'
    real_digest = launcher.digest
    monkeypatch.setattr(launcher, 'digest', lambda path: launcher.CHECKPOINT_SHA if Path(path) == checkpoint
        else launcher.METADATA_SHA if str(path) == str(checkpoint)+'.metadata.json' else real_digest(path))
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda *args, **kwargs: json.dumps(cells))
    monkeypatch.setattr(launcher, 'source', lambda: dict(source_commit='tested', source_tree='tree', source_dir=str(ROOT)))
    reference_calls = []
    def reference(root, horizons, rounds, **kwargs):
        reference_calls.append((root, horizons, rounds, kwargs))
        return dict(root=str(root), kind=kwargs.get('kind', 'fresh'))
    monkeypatch.setattr(launcher, 'reference_manifest', reference)
    destination = tmp_path/'new'
    launcher.prepare(SimpleNamespace(root=destination, matrix=matrix_path, checkpoint=checkpoint,
        reference_root=tmp_path/'fresh', bernoulli_reference_root=tmp_path/'p50'))
    return launcher.read(destination/'campaign.json'), reference_calls


def test_j8_grid_keeps_p50_mechanisms_diagnostics_pairing_and_replay_budget():
    matrix, old = load_campaign(MATRIX), load_campaign(PREVIOUS)
    assert matrix['rounds'] == [8] and launcher.bernoulli_probabilities(matrix) == [.5]
    for key in ('checkpoint_contract', 'base_preset', 'horizons', 'arms', 'seeds',
                'controller_seed', 'max_steps', 'critic_updates', 'actor_updates',
                'rollouts', 'batch_size', 'diagnostics'):
        assert matrix[key] == old[key]
    cells = listed_cells(matrix)
    assert len(cells) == 9 and len(cells)*len(matrix['seeds']) == 27
    assert {cell['J'] for cell in cells} == {8}
    assert not {cell['name'] for cell in cells}.intersection(cell['name'] for cell in listed_cells(old))
    base = dict(selector='base', algorithm_config=dict(alg_params=dict(inner_replay_capacity=3072)))
    params = resolved_cell(base, matrix, horizon=3, rounds=8,
        arm='bernoulli_a05_c05')['algorithm_config']['alg_params']
    assert params['inner_replay_capacity'] == 3072
    assert params['inner_rollout_horizon'] * params['inner_rounds'] * params['inner_rollouts_per_round'] == 3072


def test_j8_prepare_pins_three_maximum_budget_smokes_and_both_reference_grids(tmp_path, monkeypatch):
    campaign, calls = prepare(tmp_path, monkeypatch)
    assert campaign['J'] == [8] and len(campaign['cells']) == 9
    assert campaign['smoke_indices'] == [6, 7, 8]
    smoke = [campaign['cells'][index] for index in campaign['smoke_indices']]
    assert {cell['H'] for cell in smoke} == {3}
    assert {cell['J'] for cell in smoke} == {8}
    assert {cell['arm'] for cell in smoke} == set(launcher.read(MATRIX)['arms'])
    assert campaign['smoke_seeds'] == [101, 102] and campaign['smoke_steps'] == 3
    assert campaign['diagnostics']['enabled']
    assert calls[0][1:3] == ([1, 2, 3], [1, 2, 4, 6, 8])
    assert calls[1][1:3] == ([1, 2, 3], [1, 2, 4, 6])
    assert calls[1][3]['kind'] == 'bernoulli'
    assert campaign['reference_rounds'] == dict(fresh=[1, 2, 4, 6, 8], bernoulli=[1, 2, 4, 6])
    assert [reference['kind'] for reference in campaign['historical_references']] == ['fresh', 'bernoulli']
    assert campaign['matrix_sha256'] == launcher.digest(MATRIX)


@pytest.mark.parametrize('change', ['missing_reference', 'different_reference', 'extra_j', 'wrong_probability', 'old_override'])
def test_j8_exceptions_do_not_loosen_other_campaign_protocols(tmp_path, monkeypatch, change):
    matrix = launcher.read(MATRIX)
    if change == 'missing_reference':
        matrix.pop('reference_rounds')
    elif change == 'different_reference':
        matrix['reference_rounds']['fresh'] = [8]
    elif change == 'extra_j':
        matrix['rounds'] = [6, 8]
    elif change == 'wrong_probability':
        matrix = launcher.read(ROOT/'configs/research/ambi_bernoulli_probability_575k.json')
        matrix['rounds'] = [8]
    else:
        matrix['rounds'] = [1, 2, 4, 6]
    changed = tmp_path/'matrix.json'
    changed.write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match='[Rr]eference|H/J'):
        prepare(tmp_path, monkeypatch, changed)
