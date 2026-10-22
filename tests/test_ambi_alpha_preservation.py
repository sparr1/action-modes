"""Preservation uses the positive sampled MPPI500k control and independent chains."""
from copy import deepcopy
import gzip
import json

import pytest

from slurm import ambi_alpha_preservation as driver


def cell(stream=4):
    return {**driver.read(driver.PROFILE)['checkpoints'][driver.CHECKPOINT_ID], **driver.intervention(stream),
            'inventory_path': '/inventory', 'science_id': 'preservation-stream'}


def test_matrix_and_primary_endpoint_are_explicit():
    driver.validate_matrix(driver.read(driver.MATRIX))
    assert driver.PRIMARY['episode_index'] == 2
    assert driver.PRIMARY['outcomes'] == ['paired_gain_vs_prior', 'paired_delta_vs_reset']
    assert driver.read(driver.MATRIX)['temperature_chain']['checkpoints'] == ['mppi_aux_500k']
    assert driver.intervention(0)['outer_alpha'] == driver.read(driver.PROFILE)['checkpoints']['mppi_aux_500k']['outer_alpha']


@pytest.mark.parametrize('key,value', [('inner_actor_scope', 'run'), ('inner_outer_policy_kl_coef', .01),
    ('inner_temperature_updates_per_actor', 100), ('inner_temperature_lr', .03), ('inner_execution_action', 'mean')])
def test_extra_interventions_are_rejected(key, value):
    matrix = driver.read(driver.MATRIX); matrix['shared_alg_params'][key] = value
    with pytest.raises(ValueError, match='Only temperature'):
        driver.validate_matrix(matrix)


def test_worker_uses_preservation_matrix_sampled_chain_and_no_reference_job(tmp_path):
    command = driver.evaluator_command(cell(), tmp_path, 8)
    assert command[command.index('--matrix') + 1] == str(driver.MATRIX)
    assert command[command.index('--preset') + 1] == 'alpha_preservation/run'
    assert command[command.index('--seeds') + 1:command.index('--controller-seed')] == ['105', '101']
    assert '--temperature-chain' in command and '--reference-bundle' not in command


def artifact(tmp_path):
    from test_ambi_alpha_carry_chain import artifact as chain_artifact
    original = chain_artifact(tmp_path, stream=4, steps=8)
    selected = cell()
    payload = driver.read(tmp_path / 'results.json'); manifest = driver.read(tmp_path / 'bundle/manifest.json')
    payload['checkpoint_sha256'] = manifest['checkpoint']['sha256'] = selected['checkpoint_sha256']
    payload['results'][0]['selector'] = manifest['runs'][0]['selector'] = selected['selector']
    manifest['runs'][0]['result'] = deepcopy(payload['results'][0])
    for relative in manifest['runs'][0]['trace_files']:
        path = tmp_path / 'bundle' / relative
        with gzip.open(path, 'rt') as source: rows = [json.loads(line) for line in source]
        for event in rows:
            if event['phase'] == 'decision':
                for key in ('decision/inner_alpha_initial', 'decision/inner_alpha_final'):
                    event['metrics'][key] *= selected['outer_alpha'] / original['outer_alpha']
        with gzip.open(path, 'wt') as target: target.writelines(json.dumps(row) + '\n' for row in rows)
    (tmp_path / 'results.json').write_text(json.dumps(payload))
    (tmp_path / 'bundle/manifest.json').write_text(json.dumps(manifest))
    return selected


def test_existing_chain_validator_uses_the_500k_alpha_and_cross_episode_continuity(tmp_path):
    selected = artifact(tmp_path)
    validation, summary = driver.validate_result(tmp_path, selected, 8)
    assert validation['lifecycle_validated'] and validation['chain_boundaries_validated'] == 1
    assert summary['rows'][0]['alpha_initial'] == pytest.approx(selected['outer_alpha'])
    assert summary['rows'][8]['optimizer_step_initial'] == 256
    assert summary['rows'][8]['alpha_initial'] == pytest.approx(summary['rows'][7]['alpha_final'])
    assert summary['checkpoint_id'] == 'mppi_aux_500k'
    summary['rows'][8]['alpha_initial'] = selected['outer_alpha']
    with pytest.raises(ValueError, match='Alpha value discontinuity'):
        driver.validate_decisions(summary, selected, selected['seeds'][:2], 8)


def references_fixture(tmp_path, monkeypatch):
    pin = driver.read(driver.PROFILE)['checkpoints'][driver.CHECKPOINT_ID]
    old = dict(source_commit='b' * 40, matrix_sha256=driver.sha256(driver.REPO / driver.references.MATRIX),
               profile_sha256=driver.sha256(driver.PROFILE), cells=[])
    calls = []
    monkeypatch.setattr(driver.references, 'validate_campaign', lambda value: calls.append('campaign'))
    monkeypatch.setattr(driver, 'trace_hashes', lambda path: {'trace': 'pinned'})
    inventory = tmp_path / 'inventory.json'; inventory.write_text('{}')
    for mode in ('prior', 'entropy'):
        output = tmp_path / mode; (output / 'bundle').mkdir(parents=True)
        selected = {**pin, **driver.references.intervention(mode), 'output_dir': str(output),
                    'inventory_path': str(inventory), 'inventory_sha256': driver.sha256(inventory)}
        old['cells'].append(selected)
        cfg = {'inner_' + scope + '_scope': 'action' for scope in (
            'actor', 'critic', 'replay', 'temperature', 'actor_optimizer', 'critic_optimizer', 'temperature_optimizer')}
        cfg['inner_outer_policy_kl_coef'] = 0.
        (output / 'results.json').write_text(json.dumps({'results': [{'resolved_config': cfg, 'episodes': []}]}))
        (output / 'bundle/manifest.json').write_text('{}')
        receipt = dict(validated=True, smoke=False, source_commit=old['source_commit'],
            **driver.references.intervention(mode), validation={'passed': True},
            trace_sha256={'trace': 'pinned'}, results_sha256=driver.sha256(output / 'results.json'),
            manifest_sha256=driver.sha256(output / 'bundle/manifest.json'))
        (output / 'receipt.json').write_text(json.dumps(receipt))
    def validate(output, selected): calls.append(selected['mode']); return {'passed': True}
    monkeypatch.setattr(driver.references, 'validate_episode_result', validate)
    monkeypatch.setattr(driver.original, 'compact_summary_from_bundle', lambda *args, **kwargs: dict(
        rows=[dict(seed=seed, decision_index=d) for seed in driver.SEEDS for d in range(500)],
        episodes=[], provenance={}))
    return old, calls


def test_existing_controls_are_revalidated_and_pinned_without_rerunning(tmp_path, monkeypatch):
    old, calls = references_fixture(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs): raise AssertionError('No new evaluator job for controls')
    monkeypatch.setattr(driver.subprocess, 'run', forbidden)
    refs = driver.reference_snapshots(tmp_path / 'new', old)
    assert calls == ['campaign', 'prior', 'entropy']
    assert set(refs['reference_summaries']) == {'prior', 'reset'}
    assert len(refs['reference_files']) == 7
    driver.chain.verify_files(refs['reference_files'])
    for item in refs['reference_summaries'].values():
        assert driver.sha256(item['path']) == item['sha256']


@pytest.mark.parametrize('change', ['result', 'receipt', 'scope', 'validation'])
def test_control_integrity_or_validation_failure_blocks_preparation(tmp_path, monkeypatch, change):
    old, _ = references_fixture(tmp_path, monkeypatch)
    output = tmp_path / 'entropy'
    receipt = driver.read(output / 'receipt.json')
    if change == 'result': (output / 'results.json').write_text('{}')
    elif change == 'receipt':
        receipt['smoke'] = True; (output / 'receipt.json').write_text(json.dumps(receipt))
    elif change == 'scope':
        payload = driver.read(output / 'results.json'); payload['results'][0]['resolved_config']['inner_temperature_scope'] = 'episode'
        (output / 'results.json').write_text(json.dumps(payload)); receipt['results_sha256'] = driver.sha256(output / 'results.json')
        (output / 'receipt.json').write_text(json.dumps(receipt))
    else:
        monkeypatch.setattr(driver.references, 'validate_episode_result', lambda *args: {'passed': False})
    with pytest.raises(ValueError): driver.reference_snapshots(tmp_path / 'new', old)


def test_prepare_refuses_overwriting_campaign(tmp_path):
    (tmp_path / 'campaign.json').write_text('{}')
    with pytest.raises(FileExistsError): driver.prepare(tmp_path)
