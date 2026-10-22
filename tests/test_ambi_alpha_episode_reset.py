"""Episode-boundary reset is a matched one-axis control of the prior chains."""
from copy import deepcopy
import gzip
import json

import pytest

from slurm import ambi_alpha_episode_reset as driver


def cell(stream=4):
    return {**driver.read(driver.PROFILE)['checkpoints'][driver.CHECKPOINT_ID],
            **driver.intervention(stream), 'inventory_path': '/inventory', 'science_id': 'episode-reset-stream'}


def artifact(tmp_path):
    from test_ambi_alpha_preservation import artifact as carry_artifact
    carry_artifact(tmp_path)
    selected = cell()
    payload, manifest = driver.read(tmp_path / 'results.json'), driver.read(tmp_path / 'bundle/manifest.json')
    result, run = payload['results'][0], manifest['runs'][0]
    result['selector'] = run['selector'] = selected['selector']
    for key in ('temperature_chain', 'temperature_chain_seeds', 'temperature_chain_kind'):
        result.pop(key, None)
    manifest['protocol'].pop('temperature_chain')
    for config in (result['resolved_config'], run['resolved_config']):
        config.update(driver.OVERRIDES)
    for episode in result['episodes'] + run['episodes']:
        episode.pop('temperature_chain_episode_index')
    run['result'] = deepcopy(result)
    for relative in run['trace_files']:
        path = tmp_path / 'bundle' / relative
        with gzip.open(path, 'rt') as source:
            rows = [json.loads(line) for line in source]
        for event in rows:
            if event['phase'] != 'decision':
                continue
            decision = event['decision_index']
            alpha = selected['outer_alpha'] * .999 ** decision
            event['metrics'].update({'decision/inner_alpha_initial': alpha,
                'decision/inner_alpha_final': alpha * .999,
                'decision/inner_temperature_optimizer_step_initial': decision * 32,
                'decision/inner_temperature_optimizer_step_final': (decision + 1) * 32,
                'decision/inner_temperature_lifetime_steps_initial': decision * 32,
                'decision/inner_temperature_lifetime_steps_final': (decision + 1) * 32,
                'decision/inner_temperature_episode_reset': int(decision == 0),
                'decision/inner_temperature_chain_reset': 0})
        with gzip.open(path, 'wt') as target:
            target.writelines(json.dumps(row) + '\n' for row in rows)
    (tmp_path / 'results.json').write_text(json.dumps(payload))
    (tmp_path / 'bundle/manifest.json').write_text(json.dumps(manifest))
    return selected


def test_matrix_changes_only_episode_boundary_temperature_reset():
    matrix = driver.read(driver.MATRIX)
    driver.validate_matrix(matrix)
    old = driver.read(driver.preservation.MATRIX)['shared_alg_params']
    assert {key for key, value in matrix['shared_alg_params'].items() if old[key] != value} == {
        'inner_temperature_scope', 'inner_temperature_optimizer_scope'}
    assert driver.PRIMARY['episode_index'] == 2
    assert 'paired_delta_vs_chain_carry' in driver.PRIMARY['outcomes']
    for episode in range(3):
        assert sorted(seeds[episode] for seeds in driver.STREAMS) == driver.SEEDS


@pytest.mark.parametrize('key,value', [('inner_temperature_scope', 'action'),
    ('inner_temperature_optimizer_scope', 'run'), ('inner_actor_scope', 'episode'),
    ('inner_temperature_lr', .03), ('inner_outer_policy_kl_coef', .01)])
def test_extra_intervention_is_rejected(key, value):
    matrix = driver.read(driver.MATRIX); matrix['shared_alg_params'][key] = value
    with pytest.raises(ValueError, match='Only temperature'):
        driver.validate_matrix(matrix)


def test_evaluator_runs_one_process_without_chain_flag_and_preserves_seed_order(tmp_path):
    full, smoke = driver.evaluator_command(cell(), tmp_path), driver.evaluator_command(cell(), tmp_path, 8)
    assert full[full.index('--seeds') + 1:full.index('--controller-seed')] == ['105', '101', '102']
    assert smoke[smoke.index('--seeds') + 1:smoke.index('--controller-seed')] == ['105', '101']
    for command in (full, smoke):
        assert not {'--temperature-chain', '--reference-bundle', '--eval-run-dir', '--wandb'}.intersection(command)
        assert command[command.index('--matrix') + 1] == str(driver.MATRIX)
        assert command[command.index('--preset') + 1] == 'alpha_episode_reset/episode'
    with pytest.raises(ValueError): driver.evaluator_command(cell(), tmp_path, True)


def test_full_validation_requires_checkpoint_alpha_and_fresh_adam_each_episode(tmp_path):
    selected = artifact(tmp_path)
    validation, summary = driver.validate_result(tmp_path, selected, 8)
    assert validation['passed'] and validation['episode_boundaries_validated'] == 1
    assert validation['chain_boundaries_validated'] == 0
    assert summary['rows'][0]['seed'] == 105 and summary['rows'][8]['seed'] == 101
    assert summary['rows'][8]['global_decision_index'] == 8
    for start in (0, 8):
        assert summary['rows'][start]['alpha_initial'] == selected['outer_alpha']
        assert summary['rows'][start]['optimizer_step_initial'] == 0
        assert summary['rows'][start]['lifetime_steps_initial'] == 0
        assert summary['rows'][start]['episode_reset'] == 1
        assert summary['rows'][start + 1]['alpha_initial'] == pytest.approx(summary['rows'][start]['alpha_final'])
        assert summary['rows'][start + 1]['optimizer_step_initial'] == 32
    assert summary['rows'][8]['alpha_initial'] != summary['rows'][7]['alpha_final']


@pytest.mark.parametrize('index,field,value', [(8, 'optimizer_step_initial', 256),
    (8, 'lifetime_steps_initial', 256), (8, 'alpha_initial', .01), (8, 'episode_reset', 0),
    (9, 'optimizer_step_initial', 0), (9, 'lifetime_steps_initial', 0),
    (9, 'alpha_initial', .023066775873303413), (0, 'chain_reset', 1),
    (0, 'entropy', float('nan')), (8, 'stream_index', 0), (8, 'temperature_updates', 3200)])
def test_wrong_reset_or_within_episode_continuity_is_rejected(tmp_path, index, field, value):
    selected = artifact(tmp_path)
    _, summary = driver.validate_result(tmp_path, selected, 8)
    summary['rows'][index][field] = value
    with pytest.raises(ValueError): driver.validate_decisions(summary, selected, selected['seeds'][:2], 8)


@pytest.mark.parametrize('change', ['missing', 'reordered', 'reward'])
def test_incomplete_reordered_or_wrong_reward_summary_rejected(tmp_path, change):
    selected = artifact(tmp_path)
    _, summary = driver.validate_result(tmp_path, selected, 8)
    if change == 'missing': summary['rows'].pop()
    elif change == 'reordered': summary['rows'][0], summary['rows'][1] = summary['rows'][1], summary['rows'][0]
    else: summary['episodes'][1]['return'] += 1
    with pytest.raises(ValueError): driver.validate_decisions(summary, selected, selected['seeds'][:2], 8)


def test_missing_trace_diagnostic_cannot_publish(tmp_path):
    from test_ambi_early_h3_j8_validation import alter_trace
    selected = artifact(tmp_path)
    manifest = driver.read(tmp_path / 'bundle/manifest.json')
    def corrupt(rows):
        next(r for r in rows if r['phase'] == 'decision')['metrics'].pop('decision/inner_temperature_episode_reset')
    alter_trace(tmp_path / 'bundle', manifest, corrupt)
    with pytest.raises(ValueError, match='Missing decision diagnostic'):
        driver.validate_result(tmp_path, selected, 8)


@pytest.mark.parametrize('changed', ['chain-summary.json', 'results.json', 'trace', 'receipt'])
def test_receipt_rehashes_artifacts_and_rejects_old_protocol(tmp_path, changed):
    selected = artifact(tmp_path)
    validation, summary = driver.validate_result(tmp_path, selected, 8)
    driver.write_new(tmp_path / 'chain-summary.json', summary)
    campaign = dict(source_commit='a' * 40)
    receipt = dict(validated=True, smoke=True, source_commit=campaign['source_commit'], protocol=driver.PROTOCOL,
        science_id=selected['science_id'], cell_id=selected['cell_id'], checkpoint_sha256=selected['checkpoint_sha256'],
        validation=validation, results_sha256=driver.sha256(tmp_path / 'results.json'),
        manifest_sha256=driver.sha256(tmp_path / 'bundle/manifest.json'),
        chain_summary_sha256=driver.sha256(tmp_path / 'chain-summary.json'), trace_sha256=driver.trace_hashes(tmp_path / 'bundle'))
    driver.write_new(tmp_path / 'receipt.json', receipt)
    assert driver.load_stream(campaign, selected, tmp_path, 8)[0] == summary
    if changed == 'receipt':
        receipt['protocol'] = driver.preservation.PROTOCOL
        (tmp_path / 'receipt.json').write_text(json.dumps(receipt))
    elif changed == 'trace':
        path = tmp_path / 'bundle' / next(iter(receipt['trace_sha256']))
        path.write_bytes(path.read_bytes() + b'changed')
    else: (tmp_path / changed).write_text('{}')
    with pytest.raises(ValueError): driver.load_stream(campaign, selected, tmp_path, 8)


def reference_fixture(tmp_path, monkeypatch):
    from test_ambi_alpha_preservation import references_fixture
    original, _ = references_fixture(tmp_path, monkeypatch)
    original_path = tmp_path / 'campaign.json'; original_path.write_text(json.dumps(original))
    refs = driver.preservation.reference_snapshots(tmp_path / 'old', original)
    snapshots = {kind: driver.read(item['path']) for kind, item in refs['reference_summaries'].items()}
    old = dict(source_commit='a'*40, profile_sha256=driver.sha256(driver.PROFILE),
        reference_campaign=str(original_path), reference_campaign_sha256=driver.sha256(original_path), cells=[])
    streams, receipts, calls = [], [], []
    for index in range(5):
        output = tmp_path / f'old-stream{index}'; (output / 'bundle').mkdir(parents=True)
        selected = {**cell(index), 'output_dir': str(output), 'science_id': f'old-science-{index}'}
        old['cells'].append(selected)
        summary = dict(stream_index=index, science_id=selected['science_id'])
        streams.append(summary)
        for path in ('results.json', 'bundle/manifest.json', 'chain-summary.json'):
            (output / path).write_text(json.dumps(summary))
        (output / 'bundle/trace.jsonl.gz').write_bytes(b'fixed-trace')
        receipt = dict(validation={'passed': True}, results_sha256=driver.sha256(output/'results.json'),
            manifest_sha256=driver.sha256(output/'bundle/manifest.json'),
            chain_summary_sha256=driver.sha256(output/'chain-summary.json'),
            trace_sha256={'trace.jsonl.gz':driver.sha256(output/'bundle/trace.jsonl.gz')})
        receipts.append(receipt); (output/'receipt.json').write_text(json.dumps(receipt))
    snapshot = dict(complete=True, streams=streams, reference_summaries=snapshots)
    monkeypatch.setattr(driver.preservation, 'validate_campaign', lambda old: None)
    monkeypatch.setattr(driver.preservation, 'publication_snapshot', lambda old: {'checkpoints': {driver.CHECKPOINT_ID:snapshot}})
    monkeypatch.setattr(driver.preservation, 'load_stream', lambda old, selected: (streams[selected['stream_index']], receipts[selected['stream_index']]))
    def validate(output, selected):
        index = selected['stream_index']; calls.append(index)
        return receipts[index]['validation'], streams[index]
    monkeypatch.setattr(driver.preservation, 'validate_result', validate)
    return old, snapshot, calls


def test_complete_reference_chains_are_independently_validated_and_pinned(tmp_path, monkeypatch):
    old, _, calls = reference_fixture(tmp_path, monkeypatch)
    refs = driver.reference_snapshots(tmp_path/'new', old)
    assert calls == list(range(5))
    assert [entry['stream_index'] for entry in refs['reference_streams']] == list(range(5))
    assert set(refs['reference_summaries']) == {'prior','reset'}
    driver.chain.verify_files(refs['reference_files'])
    (tmp_path/'old-stream2/bundle/trace.jsonl.gz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Pinned reference changed'):
        driver.chain.verify_files(refs['reference_files'])


@pytest.mark.parametrize('change', ['incomplete','reordered','validation'])
def test_reference_protocol_failures_block_preparation(tmp_path, monkeypatch, change):
    old, snapshot, _ = reference_fixture(tmp_path, monkeypatch)
    if change == 'incomplete': snapshot['complete'] = False
    elif change == 'reordered': old['cells'][1]['seeds'] = driver.STREAMS[0]
    else: monkeypatch.setattr(driver.preservation, 'validate_result', lambda *args: ({'passed':False}, {}))
    with pytest.raises(ValueError): driver.reference_snapshots(tmp_path/'new', old)


def test_prepare_refuses_existing_campaign(tmp_path):
    (tmp_path/'campaign.json').write_text('{}')
    with pytest.raises(FileExistsError): driver.prepare(tmp_path)
