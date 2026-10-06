"""J10 schedules nine new settings and requires all three audited reference sets."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluate_ambi_transfer_campaign import listed_cells
from slurm import ambi_transfer_discovery_campaign as launcher
from utils.transfer_campaign import load_campaign, resolved_cell


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT/'configs/research/ambi_bernoulli_j10_575k.json'
J8_MATRIX = ROOT/'configs/research/ambi_bernoulli_j8_575k.json'


def prepare(tmp_path, monkeypatch, matrix_path=MATRIX, missing=None):
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
        return dict(root=str(root), kind=kwargs.get('kind', 'fresh'), rounds=rounds)
    monkeypatch.setattr(launcher, 'reference_manifest', reference)
    args = SimpleNamespace(root=tmp_path/'new', matrix=matrix_path, checkpoint=checkpoint,
        reference_root=tmp_path/'fresh', bernoulli_reference_root=tmp_path/'p50',
        bernoulli_j8_reference_root=tmp_path/'p50-j8')
    if missing:
        setattr(args, missing, None)
    launcher.prepare(args)
    return launcher.read(args.root/'campaign.json'), reference_calls


def test_j10_only_grid_preserves_pairing_diagnostics_and_retains_full_replay():
    matrix, old = load_campaign(MATRIX), load_campaign(J8_MATRIX)
    assert matrix['rounds'] == [10] and launcher.bernoulli_probabilities(matrix) == [.5]
    for key in ('checkpoint_contract', 'base_preset', 'horizons', 'arms', 'seeds',
                'controller_seed', 'max_steps', 'critic_updates', 'actor_updates',
                'rollouts', 'batch_size', 'diagnostics'):
        assert matrix[key] == old[key]
    cells = listed_cells(matrix)
    assert len(cells) == 9 and len(cells)*len(matrix['seeds']) == 27
    assert {cell['J'] for cell in cells} == {10}
    assert not {cell['name'] for cell in cells}.intersection(cell['name'] for cell in listed_cells(old))
    base = dict(selector='base', algorithm_config=dict(alg_params=dict(inner_replay_capacity=3072)))
    for h in (1, 2, 3):
        params = resolved_cell(base, matrix, horizon=h, rounds=10,
            arm='bernoulli_a05_c05')['algorithm_config']['alg_params']
        assert params['inner_replay_capacity'] == max(3072, h*10*128)
    assert params['inner_replay_capacity'] == 3840


def test_j10_preparation_pins_three_max_budget_smokes_and_three_reference_sets(tmp_path, monkeypatch):
    campaign, calls = prepare(tmp_path, monkeypatch)
    assert campaign['J'] == [10] and len(campaign['cells']) == 9
    assert campaign['smoke_indices'] == [6, 7, 8]
    smoke = [campaign['cells'][index] for index in campaign['smoke_indices']]
    assert {cell['H'] for cell in smoke} == {3} and {cell['J'] for cell in smoke} == {10}
    assert {cell['arm'] for cell in smoke} == set(launcher.read(MATRIX)['arms'])
    assert campaign['smoke_seeds'] == [101, 102] and campaign['smoke_steps'] == 3
    assert campaign['diagnostics']['enabled']
    assert [call[2] for call in calls] == [[1, 2, 4, 6, 8, 10], [1, 2, 4, 6], [8]]
    assert [call[3].get('kind', 'fresh') for call in calls] == ['fresh', 'bernoulli', 'bernoulli']
    assert all(call[1] == [1, 2, 3] for call in calls)
    assert campaign['reference_rounds'] == dict(fresh=[1, 2, 4, 6, 8, 10], bernoulli=[1, 2, 4, 6], bernoulli_j8=[8])
    assert len(campaign['historical_references']) == 3
    assert campaign['matrix_sha256'] == launcher.digest(MATRIX)


@pytest.mark.parametrize('missing', ['reference_root', 'bernoulli_reference_root', 'bernoulli_j8_reference_root'])
def test_j10_cannot_silently_omit_a_required_reference_root(tmp_path, monkeypatch, missing):
    with pytest.raises(ValueError, match='reference roots'):
        prepare(tmp_path, monkeypatch, missing=missing)
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('previous', ['ambi_bernoulli_j8_575k.json',
    'ambi_bernoulli_transfer_575k.json', 'ambi_bernoulli_probability_575k.json'])
def test_other_campaigns_reject_the_additional_j8_reference_argument(tmp_path, monkeypatch, previous):
    with pytest.raises(ValueError, match='only authorized for the J10 extension'):
        prepare(tmp_path, monkeypatch, ROOT/'configs/research'/previous)


@pytest.mark.parametrize('change', ['extra_j', 'wrong_reference', 'wrong_probability'])
def test_j10_scope_exception_does_not_allow_other_unrequested_settings(tmp_path, monkeypatch, change):
    matrix = launcher.read(MATRIX)
    if change == 'extra_j':
        matrix['rounds'] = [8, 10]
    elif change == 'wrong_reference':
        matrix['reference_rounds']['bernoulli_j8'] = [6, 8]
    else:
        matrix = launcher.read(ROOT/'configs/research/ambi_bernoulli_probability_575k.json')
        matrix['rounds'] = [10]
    changed = tmp_path/'matrix.json'
    changed.write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match='reference|H/J'):
        prepare(tmp_path, monkeypatch, changed)
