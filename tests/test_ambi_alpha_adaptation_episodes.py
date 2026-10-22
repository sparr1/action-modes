"""Temperature-only workload, persistence and immutable reference contracts."""
from copy import deepcopy
import gzip
import json

import pytest

from slurm import ambi_alpha_adaptation_episodes as driver


def cell(mode='carry', checkpoint='sac_aux_100k'):
    return {**driver.read(driver.PROFILE)['checkpoints'][checkpoint],
            **driver.intervention(checkpoint, mode), 'inventory_path': '/inventory', 'reference_bundle': '/prior'}


def test_matrix_changes_only_temperature_work_or_lifetime():
    driver.validate_matrix(driver.read(driver.MATRIX))
    assert driver.intervention('sac_aux_100k', 'repeat100')['expected_temperature_steps'] == 3200
    assert driver.intervention('mppi_aux_400k', 'carry')['expected_temperature_steps'] == 32
    assert driver.overrides(cell('carry'))['inner_temperature_scope'] == 'episode'
    assert driver.overrides(cell('repeat100'))['inner_temperature_updates_per_actor'] == 100


@pytest.mark.parametrize('source_name', ['ambi_full_h3_j8_fresh', 'ambi_full_h3_mppi_aux_return'])
@pytest.mark.parametrize('mode', ['reset', 'repeat100', 'carry'])
def test_resolved_backbone_configs_keep_actor_critic_work_and_temperature_intervention(source_name, mode):
    from test_ambi_research_presets import _build_cfg
    base = driver.read(driver.REPO / 'configs/dmcontrol/algs' / (source_name + '.json'))
    matrix = driver.read(driver.MATRIX)
    params = {**base['alg_params'], **matrix['shared_alg_params'], **driver.OVERRIDES[mode]}
    base['alg_params'] = {key: value for key, value in params.items() if value is not None}
    cfg = _build_cfg({'algorithm_config': base})
    assert cfg.inner_actor_updates_per_action == 32
    assert cfg.inner_critic_updates_per_action == 128
    assert cfg.inner_temperature_updates_per_actor == (100 if mode == 'repeat100' else 1)
    assert cfg.inner_temperature_updates_per_action == (3200 if mode == 'repeat100' else 32)
    assert cfg.inner_temperature_scope == cfg.inner_temperature_optimizer_scope == ('episode' if mode == 'carry' else 'action')
    assert cfg.inner_actor_scope == cfg.inner_critic_scope == cfg.inner_replay_scope == 'action'
    assert cfg.inner_outer_policy_kl_coef == 0 and cfg.inner_sac_return_estimator == 'one_step'


@pytest.mark.parametrize('key,value', [('inner_actor_lr', .003), ('inner_outer_policy_kl_coef', .01),
    ('inner_actor_scope', 'episode'), ('inner_sac_return_estimator', 'retrace'), ('inner_temperature_lr', .03)])
def test_second_interventions_are_rejected(key, value):
    matrix = driver.read(driver.MATRIX)
    matrix['shared_alg_params'][key] = value
    with pytest.raises(ValueError, match='Shared configuration'):
        driver.validate_matrix(matrix)


def test_full_run_reuses_prior_and_two_episode_smoke_exercises_reset(tmp_path):
    full = driver.evaluator_command(cell(), tmp_path)
    smoke = driver.evaluator_command(cell(), tmp_path, 8)
    assert full[full.index('--reference-bundle') + 1] == '/prior'
    assert '--reference-bundle' not in smoke
    assert smoke[smoke.index('--seeds') + 1:smoke.index('--controller-seed')] == ['101', '102']
    assert full[full.index('--seeds') + 1:full.index('--controller-seed')] == list(map(str, driver.SEEDS))
    for length in (True, 0, 500):
        with pytest.raises(ValueError):
            driver.evaluator_command(cell(), tmp_path, length)


def artifact(tmp_path, mode):
    from test_ambi_early_h3_j8_validation import artifact as baseline_artifact, summary
    selected = cell(mode)
    _, bundle, payload, manifest = baseline_artifact(tmp_path, seeds=(101, 102), steps=8, mode='sample',
                                                     config_overrides=driver.overrides(selected))
    result, run = payload['results'][0], manifest['runs'][0]
    result['selector'] = run['selector'] = selected['selector']
    payload['checkpoint_sha256'] = manifest['checkpoint']['sha256'] = selected['checkpoint_sha256']
    count = selected['expected_temperature_steps']
    result['model_metrics']['inner_temperature_optimizer_steps'] = summary([count] * 16)
    result['return'] = summary([.8, .8])
    for episode, saved in zip(result['episodes'], run['episodes']):
        episode['return'] = saved['return'] = .8
        episode['model_metrics']['inner_temperature_optimizer_steps'] = count
        saved['model_metrics']['inner_temperature_optimizer_steps'] = count
    run['result'] = deepcopy(result)
    for relative in run['trace_files']:
        path = bundle / relative
        with gzip.open(path, 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        for event in rows:
            event['temperature_updates'] *= selected['temperature_repeats']
            if event['phase'] == 'update':
                event['metrics']['temperature_updates'] = selected['temperature_repeats'] if event['updated_temperature'] else 0
            if event['phase'] == 'decision':
                decision = event['decision_index']
                step = decision * count if mode == 'carry' else 0
                alpha = selected['outer_alpha'] * .99 ** decision if mode == 'carry' else selected['outer_alpha']
                event['metrics'].update({'decision/inner_temperature_optimizer_steps': count,
                    'decision/inner_alpha_initial': alpha, 'decision/inner_alpha_final': alpha * .99,
                    'decision/inner_actor_entropy': 2., 'decision/inner_target_entropy': -10.5,
                    'decision/' + driver.LIFECYCLE[0]: step, 'decision/' + driver.LIFECYCLE[1]: step + count,
                    'decision/' + driver.LIFECYCLE[2]: step, 'decision/' + driver.LIFECYCLE[3]: step + count,
                    'decision/' + driver.LIFECYCLE[4]: int(decision == 0)})
        with gzip.open(path, 'wt') as stream:
            stream.writelines(json.dumps(row) + '\n' for row in rows)
    (tmp_path / 'results.json').write_text(json.dumps(payload))
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    return selected, manifest


@pytest.mark.parametrize('mode', ['reset', 'repeat100', 'carry'])
def test_complete_traces_validate_repeats_carry_and_episode_reset(tmp_path, mode):
    selected, _ = artifact(tmp_path, mode)
    validation = driver.validate_result(tmp_path, selected, 8)
    assert validation['passed'] and validation['lifecycle_validated']
    assert validation['decisions'] == 16
    compact = driver.read(tmp_path / 'decision-summary.json')
    assert len(compact['rows']) == 16
    assert compact['rows'][8]['episode_reset'] == 1
    assert compact['rows'][8]['optimizer_step_initial'] == 0


@pytest.mark.parametrize('key,value', [('optimizer_step_initial', 0), ('alpha_initial', 1.),
    ('lifetime_steps_initial', 0), ('episode_reset', 1), ('temperature_updates', 3200)])
def test_carry_diagnostics_reject_partial_or_broken_transfer(tmp_path, key, value):
    selected, _ = artifact(tmp_path, 'carry')
    compact = driver.compact_summary_from_bundle(selected, tmp_path / 'bundle', True)
    compact['rows'][1][key] = value
    with pytest.raises(ValueError):
        driver.validate_decisions(compact, selected, [101, 102], 8)


def test_episode_boundary_cannot_leak_temperature_or_optimizer_state(tmp_path):
    selected, _ = artifact(tmp_path, 'carry')
    compact = driver.compact_summary_from_bundle(selected, tmp_path / 'bundle', True)
    compact['rows'][8]['optimizer_step_initial'] = 256
    with pytest.raises(ValueError, match='Adam state'):
        driver.validate_decisions(compact, selected, [101, 102], 8)


def test_repeated_temperature_trace_counts_are_actual_adam_steps(tmp_path):
    from test_ambi_early_h3_j8_validation import alter_trace
    selected, manifest = artifact(tmp_path, 'repeat100')
    def corrupt(rows):
        next(row for row in rows if row['phase'] == 'update' and row['updated_temperature'])['metrics']['temperature_updates'] = 1
    alter_trace(tmp_path / 'bundle', manifest, corrupt)
    with pytest.raises(ValueError, match='repeat count'):
        driver.validate_result(tmp_path, selected, 8)


def test_baseline_extraction_retains_decision_curves_and_hashes(tmp_path):
    selected, _ = artifact(tmp_path, 'reset')
    selected.update(baseline_bundle=str(tmp_path / 'bundle'), baseline_decision_summary=str(tmp_path / 'references/decision-summary.json'),
                    baseline_manifest_sha256='abc', baseline_results_sha256='def')
    compact = driver.compact_summary_from_bundle(selected, selected['baseline_bundle'])
    assert compact['rows'][-1]['cumulative_reward'] == pytest.approx(.8)
    driver.write_new(selected['baseline_decision_summary'], compact)
    driver.write_new(tmp_path / 'references/baseline-receipt.json', dict(
        baseline_manifest_sha256='abc', baseline_results_sha256='def',
        decision_summary_sha256=driver.sha256(selected['baseline_decision_summary'])))
    assert driver.load_baseline_decision_summary(selected) == compact
    compact['rows'][0]['alpha_initial'] = 100.
    (tmp_path / 'references/decision-summary.json').write_text(json.dumps(compact))
    with pytest.raises(ValueError, match='summary changed'):
        driver.load_baseline_decision_summary(selected)


def test_prepare_never_overwrites_existing_campaign(tmp_path):
    (tmp_path / 'campaign.json').write_text('{"keep": true}')
    with pytest.raises(FileExistsError):
        driver.prepare(tmp_path)


def test_launcher_uses_explicit_resources_offline_clean_source():
    text = (driver.REPO / 'slurm/run_ambi_alpha_adaptation_episodes_oscar.sbatch').read_text()
    assert 'WANDB_MODE=disabled' in text and 'EXPECTED_ACTION_MODES_SHA' in text
    assert '#SBATCH' not in text and 'SLURM_CPUS_PER_TASK' in text
