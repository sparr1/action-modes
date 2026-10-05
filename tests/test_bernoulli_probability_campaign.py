"""Probability screens preserve the paired protocol and audited reuse boundaries."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluate_ambi_transfer_campaign import listed_cells
from slurm import ambi_transfer_discovery_campaign as launcher
from tests.test_transfer_discovery_publication import fixture, _save
from utils.transfer_campaign import load_campaign


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / 'configs/research/ambi_bernoulli_probability_575k.json'
PREVIOUS_MATRIX = ROOT / 'configs/research/ambi_bernoulli_transfer_575k.json'


def prepared(tmp_path, monkeypatch):
    matrix = load_campaign(MATRIX)
    cells = listed_cells(matrix)
    checkpoint = tmp_path / 'checkpoint.pt'
    real_digest = launcher.digest
    monkeypatch.setattr(launcher, 'digest', lambda p: launcher.CHECKPOINT_SHA if Path(p) == checkpoint
        else launcher.METADATA_SHA if str(p) == str(checkpoint)+'.metadata.json' else real_digest(p))
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda args, **kwargs: json.dumps(cells))
    identity = dict(source_commit='tested', source_tree='tree', source_dir=str(ROOT))
    monkeypatch.setattr(launcher, 'source', lambda: identity)
    root = tmp_path / 'new'
    launcher.prepare(SimpleNamespace(root=root, matrix=MATRIX, checkpoint=checkpoint,
        reference_root=None, bernoulli_reference_root=None))
    return root, launcher.read(root / 'campaign.json')


def test_probability_matrix_adds_only_72_requested_cells_and_eight_smokes(tmp_path, monkeypatch):
    matrix = load_campaign(MATRIX)
    previous = load_campaign(PREVIOUS_MATRIX)
    assert launcher.bernoulli_probabilities(matrix) == [.25, .75]
    assert not set(matrix['arms']).intersection(previous['arms'])
    for key in ('checkpoint_contract', 'base_preset', 'horizons', 'rounds', 'seeds',
                'controller_seed', 'max_steps', 'critic_updates', 'actor_updates',
                'rollouts', 'batch_size', 'diagnostics'):
        assert matrix[key] == previous[key]
    root, saved = prepared(tmp_path, monkeypatch)
    assert len(saved['cells']) == 72
    assert len(saved['cells']) * len(saved['seeds']) == 216
    assert saved['matrix_sha256'] == launcher.digest(MATRIX)
    smoke = [saved['cells'][index] for index in saved['smoke_indices']]
    assert len(smoke) == 8
    assert {cell['arm'] for cell in smoke} == set(matrix['arms'])
    assert {cell['H'] for cell in smoke} == {1, 2, 3}
    assert {cell['J'] for cell in smoke} == {2, 6}
    assert saved['smoke_seeds'] == [101, 102] and saved['smoke_steps'] == 3
    assert saved['diagnostics']['enabled']
    assert saved['historical_references'] == []


@pytest.mark.parametrize('change', ['probability', 'mixture', 'missing_arm', 'extra_mechanism'])
def test_unauthorized_probability_or_mechanism_changes_are_rejected(change):
    matrix = launcher.read(MATRIX)
    if change == 'probability':
        matrix['probabilities'] = [.1, .9]
    elif change == 'mixture':
        matrix['arms']['bernoulli_a025_c025']['critic_bernoulli_p'] = .75
    elif change == 'missing_arm':
        matrix['arms'].pop('bernoulli_a075_c0')
    else:
        matrix['arms']['bernoulli_a075_c0']['replay_fraction'] = .25
    with pytest.raises(ValueError, match='authorized'):
        launcher.bernoulli_probabilities(matrix)


def test_worker_passes_exact_probability_matrix_and_requires_all_smokes(tmp_path, monkeypatch):
    root, campaign = prepared(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda *args, **kwargs: 'NVIDIA L40S')
    def execute(command, **kwargs):
        calls.append(command)
        directory = Path(command[command.index('--output-dir') + 1])
        _save(directory / 'results.json', {})
        _save(directory / 'manifest.json', {})
    monkeypatch.setattr(launcher.subprocess, 'run', execute)
    monkeypatch.setattr(launcher, 'validate_result', lambda *args, **kwargs: ({'episodes': [1, 2, 3]}, {}))
    args = SimpleNamespace(root=root, index=0, smoke=False)
    with pytest.raises(FileNotFoundError):
        launcher.worker(args)
    assert calls == []
    for index in campaign['smoke_indices']:
        _save(root / 'smoke' / campaign['cells'][index]['name'] / 'worker-completion.json',
              dict(source_commit=campaign['source_commit'], campaign_sha256=launcher.digest(root/'campaign.json')))
    launcher.worker(args)
    command = calls[0]
    assert command[command.index('--campaign') + 1] == str(MATRIX)
    assert command[command.index('--arm') + 1] == campaign['cells'][0]['arm']
    assert command[command.index('--max-steps') + 1] == '500'
    assert '--smoke' not in command


def historical_bernoulli(tmp_path):
    historical, old_cell, _, template_manifest, template_result = fixture(tmp_path)
    historical.update(checkpoint_sha256=launcher.CHECKPOINT_SHA,
        campaign_kind='bernoulli-weight-transfer-v1', matrix_path=str(PREVIOUS_MATRIX),
        matrix_sha256=launcher.digest(PREVIOUS_MATRIX), diagnostics={}, cells=[])
    for index, arm in enumerate(launcher.read(PREVIOUS_MATRIX)['arms']):
        historical['cells'].append(dict(old_cell, index=index, arm=arm, name=f'h1_j1_{arm}'))
    _save(tmp_path/'campaign.json', historical)
    for cell in historical['cells']:
        directory = tmp_path/'settings'/cell['name']
        manifest, result = deepcopy(template_manifest), deepcopy(template_result)
        for value in (manifest, result):
            value.update(cell_id=cell['name'], arm=cell['arm'], checkpoint_sha256=launcher.CHECKPOINT_SHA,
                campaign_sha256=historical['matrix_sha256'])
        _save(directory/'manifest.json', manifest)
        _save(directory/'results.json', result)
        _save(directory/'worker-completion.json', dict(status='complete', cell=cell, smoke=False,
            source_commit=historical['source_commit'], campaign_sha256=launcher.digest(tmp_path/'campaign.json'),
            result_sha256=launcher.digest(directory/'results.json'),
            manifest_sha256=launcher.digest(directory/'manifest.json')))
    return historical


def test_historical_probability_reference_pins_every_result_and_original_source(tmp_path):
    historical = historical_bernoulli(tmp_path)
    reference = launcher.reference_manifest(tmp_path, [1], [1], kind='bernoulli', expected_matrix=launcher.read(MATRIX))
    assert reference['kind'] == 'bernoulli' and reference['probability'] == .5
    assert reference['source_commit'] == historical['source_commit']
    assert len(reference['records']) == 3
    for record in reference['records']:
        assert set(record['hashes']) == {'results.json', 'manifest.json', 'worker-completion.json'}
    # Reusing a subset, result with modified evidence, or mismatched diagnostic protocol is forbidden.
    with pytest.raises(ValueError, match='incomplete'):
        launcher.reference_manifest(tmp_path, [1, 2], [1], kind='bernoulli', expected_matrix=launcher.read(MATRIX))
    changed = launcher.read(MATRIX)
    changed['diagnostics']['state_count'] = 64
    with pytest.raises(ValueError, match='protocol differs'):
        launcher.reference_manifest(tmp_path, [1], [1], kind='bernoulli', expected_matrix=changed)
    directory = tmp_path/'settings'/historical['cells'][0]['name']
    result = launcher.read(directory/'results.json')
    result['episodes'][0]['return'] += 1
    _save(directory/'results.json', result)
    with pytest.raises(ValueError, match='receipt binding'):
        launcher.reference_manifest(tmp_path, [1], [1], kind='bernoulli', expected_matrix=launcher.read(MATRIX))


def test_sbatch_forwards_both_read_only_reference_roots():
    script = (ROOT/'slurm/run_ambi_transfer_discovery_oscar.sbatch').read_text()
    assert 'args+=(--reference-root "$REFERENCE_CAMPAIGN_ROOT")' in script
    assert 'args+=(--bernoulli-reference-root "$BERNOULLI_REFERENCE_CAMPAIGN_ROOT")' in script
