"""Audited reuse keeps original evidence separate from newly scheduled ranks."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from evaluate_ambi_transfer_campaign import listed_cells
from slurm import ambi_spectral_transfer_campaign as launcher
from slurm import ambi_transfer_discovery_campaign as common
from utils.transfer_campaign import load_campaign, resolved_cell

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / 'configs/research/ambi_spectral_transfer_r1_r4_575k.json'


def test_versioned_extension_has_exact_new_and_reused_panels():
    matrix = load_campaign(MATRIX)
    raw = common.read(MATRIX)
    assert raw == launcher.generate_rank_extension()
    launcher.validate_matrix(matrix)
    assert len(listed_cells(matrix)) == 336
    assert len(matrix['arms']) == 28
    assert matrix['reuse']['new_ranks'] == [1, 4]
    inherited = set(launcher.generate_matrix()['arms'])
    assert sum(c['arm'] not in inherited for c in listed_cells(matrix)) == 144
    assert sum(c['arm'] in inherited for c in listed_cells(matrix)) == 192
    assert common.digest(ROOT / 'configs/research/ambi_spectral_transfer_575k.json') == matrix['reuse']['matrix_sha256']


@pytest.mark.parametrize('field,value', [('new_ranks', [1]), ('source_commit', 'other'),
    ('campaign_sha256', 'other'), ('matrix_sha256', 'other'), ('kind', 'generic-history')])
def test_reuse_contract_is_explicitly_narrow(field, value):
    matrix = launcher.generate_rank_extension()
    matrix['reuse'][field] = value
    with pytest.raises(ValueError, match='restricted'):
        launcher.validate_matrix(matrix)


def test_scientific_fingerprint_ignores_only_reviewed_reporting_changes(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], text=True).strip()
    git('init', '-q')
    paths = ['evaluate_ambi_transfer_campaign.py', 'RL/model.py', 'domains/env.py',
             'utils/world.py', 'environments/dmcontrol/uv.lock', 'configs/research/base.json',
             'slurm/ambi_transfer_discovery_publish.py', 'tests/test_new.py', 'README.md']
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original\n')
    def commit():
        git('add', '--all')
        git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')
        return git('rev-parse', 'HEAD')
    old = commit()
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    expected = launcher.scientific_source(old)
    for name in ('slurm/ambi_transfer_discovery_publish.py', 'tests/test_new.py', 'README.md'):
        (tmp_path / name).write_text('reporting change\n')
    assert launcher.scientific_source(commit()) == expected
    for name in paths[:6]:
        (tmp_path / name).write_text('scientific change\n')
        assert launcher.scientific_source(commit()) != expected
        (tmp_path / name).write_text('original\n')
        commit()


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    """Small files exercise all bindings; validate_result has its own real tests."""
    old_root = tmp_path / 'reference'
    old_root.mkdir()
    old_matrix_path = old_root / 'matrix.json'
    common.write(old_matrix_path, launcher.generate_matrix())
    old_matrix = load_campaign(old_matrix_path)
    base_path = Path(old_matrix['base_matrix_path'])
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'checkpoint fixture')
    metadata = Path(str(checkpoint) + '.metadata.json')
    metadata.write_text('{}')
    real_digest = common.digest
    def digest(path):
        path = Path(path)
        if path == checkpoint:
            return launcher.CHECKPOINT_SHA
        if path == metadata:
            return launcher.METADATA_SHA
        return real_digest(path)
    monkeypatch.setattr(common, 'digest', digest)
    old = dict(schema_version=1, protocol=old_matrix['protocol'],
        family='spectral_transfer', campaign_kind='spectral-transfer-v1',
        source_commit=launcher.RANK_EXTENSION_REUSE['source_commit'], source_tree='oldtree', source_dir='oldsource',
        matrix_path=str(old_matrix_path), matrix_sha256=digest(old_matrix_path),
        checkpoint=str(checkpoint), checkpoint_sha256=launcher.CHECKPOINT_SHA,
        metadata_path=str(metadata), metadata_sha256=launcher.METADATA_SHA,
        base_matrix_path=str(base_path), base_matrix_sha256=digest(base_path), checkpoint_step=575000,
        H=[1, 2, 3], J=[1, 2, 4, 6], seeds=[101, 102, 103], controller_seed=55, max_steps=500,
        gpu_hardware='L40S', arms=old_matrix['arms'], cells=listed_cells(old_matrix),
        historical_references=[], diagnostics=old_matrix['diagnostics'],
        spectral_diagnostics=old_matrix['spectral_diagnostics'], spectral_probe=old_matrix['spectral_probe'])
    common.write(old_root / 'campaign.json', old)
    contract = {**launcher.RANK_EXTENSION_REUSE, 'campaign_sha256': digest(old_root / 'campaign.json'),
                'matrix_sha256': digest(old_matrix_path)}
    monkeypatch.setattr(launcher, 'RANK_EXTENSION_REUSE', contract)
    matrix = launcher.generate_rank_extension()
    matrix_path = tmp_path / 'new-matrix.json'
    common.write(matrix_path, matrix)
    new = {**deepcopy(old), 'source_commit': 'new', 'source_tree': 'newtree', 'source_dir': str(ROOT),
        'matrix_path': str(matrix_path), 'matrix_sha256': digest(matrix_path), 'arms': matrix['arms'],
        'reuse': matrix['reuse'], 'runtime': launcher.evaluation_runtime()}
    source_files = dict(files={'evaluate_ambi_transfer_campaign.py': 'same'}, sha256='same')
    monkeypatch.setattr('evaluate_ambi_transfer_campaign.source_identity', lambda: {**source_files, 'git_head': 'new'})
    monkeypatch.setattr(launcher, 'scientific_source', lambda commit: dict(files={'runtime': 'same'}, sha256='same'))
    base = dict(algorithm_config=dict(alg_params=dict(inner_replay_capacity=3072)))
    monkeypatch.setattr('utils.checkpoint_context.load_checkpoint_context', lambda *a, **k: object())
    monkeypatch.setattr('utils.ambi_research.resolve_preset', lambda *a, **k: deepcopy(base))
    monkeypatch.setattr(common, 'validate_result', lambda directory, campaign, cell, **kwargs:
        (common.read(directory / 'results.json'), common.read(directory / 'manifest.json')))
    for cell in old['cells']:
        directory = old_root / 'settings' / cell['name']
        directory.mkdir(parents=True)
        result = dict(episodes=[dict(seed=seed, length=500, return_value=float(seed)) for seed in old['seeds']])
        manifest = dict(resolved=resolved_cell(base, matrix, horizon=cell['H'], rounds=cell['J'], arm=cell['arm']),
            diagnostics=old['diagnostics'], device='cuda', compile=True, runtime=new['runtime'],
            source={**source_files, 'git_head': old['source_commit']})
        common.write(directory / 'results.json', result)
        common.write(directory / 'manifest.json', manifest)
        common.write(directory / 'worker-completion.json', dict(status='complete', smoke=False, cell=cell,
            source_commit=old['source_commit'], campaign_sha256=contract['campaign_sha256'],
            result_sha256=digest(directory / 'results.json'), manifest_sha256=digest(directory / 'manifest.json'),
            hardware='NVIDIA L40S, fixture-uuid, fixture-driver'))
    return SimpleNamespace(root=old_root, old=old, campaign=new, matrix_path=matrix_path,
        checkpoint=checkpoint, first=old_root / 'settings' / old['cells'][0]['name'])


def test_reference_binding_preserves_all_original_evidence(evidence):
    before = {path: path.read_bytes() for path in evidence.root.rglob('*.json')}
    reference = launcher.spectral_reference_manifest(evidence.root, evidence.campaign)
    assert len(reference['records']) == 192
    verified = launcher.verify_spectral_reference(reference, evidence.campaign)
    assert [cell for cell, _, _ in verified] == evidence.old['cells']
    assert all(item[2] is reference for item in verified)
    assert before == {path: path.read_bytes() for path in evidence.root.rglob('*.json')}


@pytest.mark.parametrize('field', ['checkpoint_sha256', 'metadata_sha256', 'base_matrix_sha256',
    'controller_seed', 'max_steps', 'H', 'J', 'seeds', 'diagnostics', 'spectral_diagnostics', 'spectral_probe'])
def test_reference_rejects_protocol_drift(evidence, field):
    evidence.campaign[field] = 'different'
    with pytest.raises(ValueError, match='protocol differs'):
        launcher.spectral_reference_manifest(evidence.root, evidence.campaign)


def test_reference_rejects_scientific_source_change(evidence, monkeypatch):
    monkeypatch.setattr(launcher, 'scientific_source', lambda commit: dict(files={}, sha256=commit))
    with pytest.raises(ValueError, match='implementation differs'):
        launcher.spectral_reference_manifest(evidence.root, evidence.campaign)


@pytest.mark.parametrize('field,value', [('status', 'failed'), ('smoke', True),
    ('source_commit', 'other'), ('campaign_sha256', 'other'), ('result_sha256', 'other'),
    ('manifest_sha256', 'other'), ('cell', {}), ('hardware', 'NVIDIA A100')])
def test_reference_rejects_unbound_completion(evidence, field, value):
    path = evidence.first / 'worker-completion.json'
    receipt = common.read(path)
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='receipt'):
        launcher.spectral_reference_manifest(evidence.root, evidence.campaign)


@pytest.mark.parametrize('field,value', [('resolved', {}), ('device', 'cpu'), ('compile', False),
    ('source', dict(files={}, sha256='bad')), ('diagnostics', {}), ('runtime', {})])
def test_reference_rejects_changed_scientific_manifest(evidence, field, value):
    path = evidence.first / 'manifest.json'
    manifest = common.read(path)
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='resolved learner'):
        launcher.spectral_reference_manifest(evidence.root, evidence.campaign)


@pytest.mark.parametrize('filename', ['results.json', 'manifest.json', 'worker-completion.json'])
def test_publication_rejects_evidence_modified_after_preparation(evidence, filename):
    reference = launcher.spectral_reference_manifest(evidence.root, evidence.campaign)
    path = evidence.first / filename
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='evidence changed'):
        launcher.verify_spectral_reference(reference, evidence.campaign)


def test_prepare_schedules_only_new_cells_and_smokes(evidence, monkeypatch, tmp_path):
    monkeypatch.setattr(common, 'source', lambda: {key: evidence.campaign[key]
        for key in ('source_commit', 'source_tree', 'source_dir')})
    cells = listed_cells(load_campaign(evidence.matrix_path))
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda *a, **k: json.dumps(cells))
    args = SimpleNamespace(root=tmp_path / 'prepared', checkpoint=evidence.checkpoint,
        matrix=evidence.matrix_path, expected_source='new', spectral_reference_root=evidence.root)
    launcher.prepare(args)
    campaign = common.read(args.root / 'campaign.json')
    assert len(campaign['cells']) == 144
    assert campaign['production_indices'] == list(range(144))
    assert campaign['comparison_cell_count'] == 336 and campaign['reused_cell_count'] == 192
    assert len(campaign['smoke_indices']) == 12
    assert all(campaign['cells'][index]['H'] == 3 and campaign['cells'][index]['J'] == 6
               for index in campaign['smoke_indices'])
    old_names = {c['name'] for c in evidence.old['cells']}
    assert not old_names.intersection(c['name'] for c in campaign['cells'])


def test_extension_refuses_missing_reference_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'digest', lambda path:
        launcher.METADATA_SHA if str(path).endswith('.metadata.json') else launcher.CHECKPOINT_SHA)
    monkeypatch.setattr(common, 'source', lambda: dict(source_commit='new', source_tree='tree', source_dir=str(ROOT)))
    args = SimpleNamespace(root=tmp_path / 'prepared', checkpoint=tmp_path / 'model.pt',
        matrix=MATRIX, expected_source='new', spectral_reference_root=None)
    with pytest.raises(ValueError, match='requires --spectral-reference-root'):
        launcher.prepare(args)
    assert not args.root.exists()


def test_worker_refuses_accidental_reference_rerun(evidence, tmp_path, monkeypatch):
    campaign = deepcopy(evidence.campaign)
    campaign.update(cells=[dict(evidence.old['cells'][0], index=0)], production_indices=[0],
        historical_references=[], smoke_indices=[0])
    root = tmp_path / 'worker'
    root.mkdir()
    common.write(root / 'campaign.json', campaign)
    monkeypatch.setattr(common, 'source', lambda: {key: campaign[key]
        for key in ('source_commit', 'source_tree', 'source_dir')})
    with pytest.raises(ValueError, match='Refusing to rerun'):
        common.worker(SimpleNamespace(root=root, index=0, smoke=False))
    assert not (root / 'settings').exists()
