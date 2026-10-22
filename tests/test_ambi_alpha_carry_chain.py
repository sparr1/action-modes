"""Ordered chains carry alpha/Adam across boundaries while preserving paired seeds."""
from copy import deepcopy
import gzip
import json

import pytest

from slurm import ambi_alpha_carry_chain as driver


def cell(stream=0, checkpoint='sac_aux_100k'):
    return {**driver.read(driver.PROFILE)['checkpoints'][checkpoint], **driver.intervention(checkpoint, stream),
            'inventory_path': '/inventory', 'science_id': 'chain-science'}


def test_rotations_balance_every_seed_at_every_episode_position():
    driver.validate_matrix(driver.read(driver.MATRIX))
    assert len(driver.STREAMS) == 5
    assert all(len(set(seeds)) == 3 for seeds in driver.STREAMS)
    for episode in range(3):
        assert sorted(seeds[episode] for seeds in driver.STREAMS) == driver.SEEDS


@pytest.mark.parametrize('source_name', ['ambi_full_h3_j8_fresh', 'ambi_full_h3_mppi_aux_return'])
def test_resolved_configuration_changes_only_alpha_lifetime(source_name):
    from test_ambi_research_presets import _build_cfg
    base = driver.read(driver.REPO / 'configs/dmcontrol/algs' / (source_name + '.json'))
    base['alg_params'] = {key: value for key, value in
        {**base['alg_params'], **driver.read(driver.MATRIX)['shared_alg_params']}.items() if value is not None}
    cfg = _build_cfg({'algorithm_config': base})
    assert cfg.inner_temperature_scope == cfg.inner_temperature_optimizer_scope == 'run'
    assert cfg.inner_actor_scope == cfg.inner_critic_scope == cfg.inner_replay_scope == 'action'
    assert cfg.inner_actor_optimizer_scope == cfg.inner_critic_optimizer_scope == 'action'
    assert (cfg.inner_actor_updates_per_action, cfg.inner_critic_updates_per_action,
            cfg.inner_temperature_updates_per_action) == (32, 128, 32)
    assert cfg.inner_temperature_lr == .0003 and cfg.inner_outer_policy_kl_coef == 0


@pytest.mark.parametrize('key,value', [('inner_temperature_updates_per_actor', 100),
    ('inner_actor_scope', 'run'), ('inner_sac_return_estimator', 'retrace'), ('inner_temperature_lr', .03)])
def test_additional_interventions_rejected(key, value):
    matrix = driver.read(driver.MATRIX)
    matrix['shared_alg_params'][key] = value
    with pytest.raises(ValueError, match='Only temperature'):
        driver.validate_matrix(matrix)


def test_commands_preserve_order_and_cannot_use_independent_episode_registry(tmp_path):
    selected = cell(4)
    full, smoke = driver.evaluator_command(selected, tmp_path), driver.evaluator_command(selected, tmp_path, 8)
    assert full[full.index('--seeds') + 1:full.index('--controller-seed')] == ['105', '101', '102']
    assert smoke[smoke.index('--seeds') + 1:smoke.index('--controller-seed')] == ['105', '101']
    for command in (full, smoke):
        assert '--temperature-chain' in command
        assert not {'--reference-bundle', '--eval-run-dir', '--wandb'}.intersection(command)
    with pytest.raises(ValueError): driver.evaluator_command(selected, tmp_path, True)


def artifact(tmp_path, stream=4, steps=8):
    from test_ambi_early_h3_j8_validation import artifact as baseline_artifact, summary
    selected = cell(stream)
    seeds = selected['seeds'][:2]
    _, bundle, payload, manifest = baseline_artifact(tmp_path, seeds=seeds, steps=steps, mode='sample',
                                                     config_overrides=driver.OVERRIDES)
    result, run = payload['results'][0], manifest['runs'][0]
    result.update(selector=selected['selector'], temperature_chain=True, temperature_chain_seeds=seeds,
                  temperature_chain_kind=driver.PROTOCOL)
    run['selector'] = selected['selector']
    payload['checkpoint_sha256'] = manifest['checkpoint']['sha256'] = selected['checkpoint_sha256']
    manifest['protocol']['temperature_chain'] = dict(kind=driver.PROTOCOL, ordered_environment_seeds=seeds)
    result['return'] = summary([steps * .1] * 2)
    for episode_index, (episode, saved) in enumerate(zip(result['episodes'], run['episodes'])):
        episode['return'] = saved['return'] = steps * .1
        episode['temperature_chain_episode_index'] = saved['temperature_chain_episode_index'] = episode_index
    run['result'] = deepcopy(result)
    for episode_index, relative in enumerate(run['trace_files']):
        path = bundle / relative
        with gzip.open(path, 'rt') as source:
            rows = [json.loads(line) for line in source]
        for event in rows:
            if event['phase'] == 'decision':
                global_step = episode_index * steps + event['decision_index']
                alpha = selected['outer_alpha'] * .999 ** global_step
                event['metrics'].update({'decision/inner_alpha_initial': alpha, 'decision/inner_alpha_final': alpha * .999,
                    'decision/inner_actor_entropy': 2., 'decision/inner_target_entropy': -10.5,
                    'decision/inner_temperature_optimizer_step_initial': global_step * 32,
                    'decision/inner_temperature_optimizer_step_final': (global_step + 1) * 32,
                    'decision/inner_temperature_lifetime_steps_initial': global_step * 32,
                    'decision/inner_temperature_lifetime_steps_final': (global_step + 1) * 32,
                    'decision/inner_temperature_episode_reset': 0,
                    'decision/inner_temperature_chain_reset': int(global_step == 0)})
        with gzip.open(path, 'wt') as dest:
            dest.writelines(json.dumps(row) + '\n' for row in rows)
    (tmp_path / 'results.json').write_text(json.dumps(payload))
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    return selected


def test_full_trace_validation_preserves_unsorted_seed_order_and_boundary_state(tmp_path):
    selected = artifact(tmp_path)
    validation, summary = driver.validate_result(tmp_path, selected, 8)
    assert validation['passed'] and validation['chain_boundaries_validated'] == 1
    assert summary['rows'][0]['seed'] == 105 and summary['rows'][8]['seed'] == 101
    assert summary['rows'][8]['optimizer_step_initial'] == 256
    assert summary['rows'][8]['alpha_initial'] == pytest.approx(summary['rows'][7]['alpha_final'])
    assert summary['rows'][8]['cumulative_reward'] == .1
    assert summary['rows'][8]['chain_reset'] == summary['rows'][8]['episode_reset'] == 0
    assert summary['episodes'][1]['episode_index'] == 1


@pytest.mark.parametrize('field,value', [('optimizer_step_initial', 0), ('lifetime_steps_initial', 0),
    ('alpha_initial', 1.), ('episode_reset', 1), ('chain_reset', 1), ('global_decision_index', 0),
    ('entropy', float('nan')), ('stream_index', 0), ('temperature_updates', 3200)])
def test_boundary_reset_or_wrong_identity_is_rejected(tmp_path, field, value):
    selected = artifact(tmp_path)
    _, summary = driver.validate_result(tmp_path, selected, 8)
    summary['rows'][8][field] = value
    with pytest.raises(ValueError): driver.validate_decisions(summary, selected, selected['seeds'][:2], 8)


def test_independent_episode_metadata_cannot_masquerade_as_chain(tmp_path):
    selected = artifact(tmp_path)
    payload = driver.read(tmp_path / 'results.json')
    payload['results'][0]['temperature_chain'] = False
    (tmp_path / 'results.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='chain execution'):
        driver.validate_result(tmp_path, selected, 8)


def test_incomplete_trace_cannot_produce_validated_chain(tmp_path):
    from test_ambi_early_h3_j8_validation import alter_trace
    selected = artifact(tmp_path)
    manifest = driver.read(tmp_path / 'bundle/manifest.json')
    alter_trace(tmp_path / 'bundle', manifest, lambda rows: rows.pop())
    with pytest.raises(ValueError, match='Incomplete decision'):
        driver.validate_result(tmp_path, selected, 8)


def test_prepare_refuses_to_overwrite(tmp_path):
    (tmp_path / 'campaign.json').write_text('{}')
    with pytest.raises(FileExistsError): driver.prepare(tmp_path)


@pytest.mark.parametrize('changed', ['chain-summary.json', 'results.json', 'trace', 'receipt'])
def test_reloading_a_stream_rehashes_every_artifact(tmp_path, changed):
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
        receipt['smoke'] = False
        (tmp_path / 'receipt.json').write_text(json.dumps(receipt))
    elif changed == 'trace':
        path = tmp_path / 'bundle' / next(iter(receipt['trace_sha256']))
        path.write_bytes(path.read_bytes() + b'changed')
    else:
        (tmp_path / changed).write_text('{}')
    with pytest.raises(ValueError): driver.load_stream(campaign, selected, tmp_path, 8)
