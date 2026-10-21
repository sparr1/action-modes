"""Guard paired Retrace objectives, work, traces and immutable publication inputs."""
from copy import deepcopy
import json

import pytest

from slurm import ambi_retrace_kl_mppi as driver


def cell(mode='retrace_kl', step=400000):
    pin = next(r for r in driver.read(driver.PROFILE)['checkpoints'] if r['step'] == step)
    return {**pin, **driver.intervention(mode, step), 'inventory_path': '/inventory.json'}


def test_four_resolved_cells_keep_actor_recipe_and_use_complete_retrace_trajectories():
    from test_ambi_research_presets import _build_cfg
    matrix = driver.read(driver.MATRIX)
    driver.validate_matrix(matrix)
    base = driver.read(driver.REPO / 'configs/dmcontrol/algs/ambi_full_h3_mppi_aux_return.json')
    for step in driver.ALPHAS:
        for mode in driver.MODES:
            selected = cell(mode, step)
            variant = matrix['comparisons']['retrace_kl_mppi']['variants'][selected['cell_id']]
            config = deepcopy(base)
            params = {**base['alg_params'], **matrix['shared_alg_params'], **variant['alg_params']}
            config['alg_params'] = {k: v for k, v in params.items() if v is not None}
            cfg = _build_cfg({'algorithm_config': config})
            assert cfg.inner_sac_return_estimator == 'retrace'
            assert cfg.inner_retrace_lambda == 1 and cfg.inner_retrace_batch_trajectories == 86
            assert cfg.inner_batch_size == 256
            assert cfg.inner_critic_updates_per_action == 128
            assert cfg.inner_actor_updates_per_action == 32
            assert cfg.inner_temperature_updates_per_action == selected['expected_temperature_steps']
            assert cfg.inner_outer_policy_kl_coef == selected['outer_alpha']
            assert cfg.inner_critic_source == cfg.inner_horizon_critic_source == 'aux_return'
            assert cfg.inner_sac_critic_target == 'reward_only'
            assert cfg.sac_actor_loss_scale_mode == 'none'


@pytest.mark.parametrize('key,value', [('inner_retrace_lambda', .5), ('inner_retrace_batch_trajectories', 85),
    ('inner_sac_return_estimator', 'one_step'), ('inner_actor_lr', .003), ('inner_critic_updates_per_round', 32)])
def test_recipe_mutations_are_rejected(key, value):
    matrix = driver.read(driver.MATRIX)
    matrix['shared_alg_params'][key] = value
    with pytest.raises(ValueError, match='Shared configuration'):
        driver.validate_matrix(matrix)


def test_same_checkpoint_and_reference_hashes_are_reused():
    profile = driver.read(driver.PROFILE)
    previous = driver.read(driver.REPO / 'configs/research/ambi_mppi_checkpoint_comparison_profile.json')
    for pin, old in zip(profile['checkpoints'], previous['checkpoints']):
        assert all(pin[key] == value for key, value in old.items())
        assert pin['outer_alpha'] == driver.ALPHAS[pin['step']]
        assert pin['checkpoint_version'] == 3
    assert profile['existing_curves_profile'].endswith('/20261020-15ce8500/curves-profile.json')


def test_actual_prior_reference_is_included_in_preflight_and_full_run_but_not_smoke(tmp_path):
    selected = cell()
    for spec_only in (False, True):
        command = driver.evaluator_command(selected, tmp_path, specification_only=spec_only)
        assert command[command.index('--reference-bundle') + 1] == selected['reference_bundle']
        assert command[command.index('--seeds') + 1:command.index('--controller-seed')] == list(map(str, driver.SEEDS))
        assert command[command.index('--max-steps') + 1] == '500'
    for steps in (8, 32):
        command = driver.evaluator_command(selected, tmp_path, steps)
        assert '--reference-bundle' not in command
        assert command[command.index('--max-steps') + 1] == str(steps)
        assert command[command.index('--seeds') + 1:command.index('--controller-seed')] == ['101']
    for steps in (True, 0, 4, 500):
        with pytest.raises(ValueError):
            driver.evaluator_command(selected, tmp_path, steps)


def artifact(tmp_path, mode):
    from test_ambi_early_h3_j8_validation import artifact as full_trace_artifact, summary
    selected = cell(mode)
    _, bundle, payload, manifest = full_trace_artifact(tmp_path, seeds=(101,), steps=8, mode='sample',
        temperature_steps=selected['expected_temperature_steps'], config_overrides=driver.overrides(selected))
    payload['checkpoint_sha256'] = manifest['checkpoint']['sha256'] = selected['checkpoint_sha256']
    result, run = payload['results'][0], manifest['runs'][0]
    result['selector'] = run['selector'] = selected['selector']
    alpha = selected['outer_alpha'] if selected['entropy_enabled'] else 0.
    constants = dict(inner_actor_entropy=-6., inner_alpha_initial=alpha, inner_alpha_final=alpha,
        inner_alpha_delta=0., inner_target_entropy=-10.5, inner_final_outer_policy_kl=12., inner_outer_policy_kl=10.,
        inner_retrace_trajectory_draws=128 * 86, inner_retrace_critic_rows=128 * 86 * 3,
        inner_retrace_trace_coefficient_mean=.6, inner_retrace_effective_trace_length=1.5,
        inner_retrace_correction_abs_mean=.2)
    result['model_metrics'].update({k: summary([v] * 8) for k, v in constants.items()})
    for ep, saved in zip(result['episodes'], run['episodes']):
        ep['model_metrics'].update(constants)
        saved['model_metrics'].update(constants)
    run['result'] = deepcopy(result)
    (tmp_path / 'results.json').write_text(json.dumps(payload))
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    return selected, payload, manifest


@pytest.mark.parametrize('mode', driver.MODES)
def test_full_smoke_trace_validation_checks_both_temperature_arms_and_retrace_metrics(tmp_path, mode):
    selected, _, _ = artifact(tmp_path, mode)
    validation = driver.validate_result(tmp_path, selected, 8)
    assert validation['passed'] and validation['decisions'] == 8
    assert validation['critic_batch_trajectories'] == 86
    assert validation['critic_rows_per_update'] == 258
    assert validation['retrace_diagnostics']['inner_retrace_effective_trace_length'] == 1.5
    assert len(driver.trace_hashes(tmp_path / 'bundle')) == 1


@pytest.mark.parametrize('change', ['missing_metric', 'invalid_coefficient', 'invalid_length', 'negative_correction',
                                    'wrong_work', 'wrong_alpha', 'wrong_selector', 'wrong_checkpoint'])
def test_runtime_validation_rejects_incomplete_or_wrong_retrace_results(tmp_path, change):
    selected, payload, manifest = artifact(tmp_path, 'retrace_both')
    result = payload['results'][0]
    if change == 'missing_metric': result['model_metrics'].pop(driver.RETRACE_DIAGNOSTICS[0])
    elif change == 'invalid_coefficient': result['model_metrics'][driver.RETRACE_DIAGNOSTICS[0]]['max'] = 1.1
    elif change == 'invalid_length': result['model_metrics'][driver.RETRACE_DIAGNOSTICS[1]]['max'] = 2.1
    elif change == 'negative_correction': result['model_metrics'][driver.RETRACE_DIAGNOSTICS[2]]['min'] = -.1
    elif change == 'wrong_work': result['model_metrics']['inner_retrace_critic_rows']['mean'] = 32768
    elif change == 'wrong_alpha': result['model_metrics']['inner_alpha_initial']['mean'] = .1
    elif change == 'wrong_selector': result['selector'] = 'wrong'
    elif change == 'wrong_checkpoint': payload['checkpoint_sha256'] = 'wrong'
    manifest['runs'][0]['result'] = deepcopy(result)
    (tmp_path / 'results.json').write_text(json.dumps(payload))
    (tmp_path / 'bundle/manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        driver.validate_result(tmp_path, selected, 8)


def test_normalized_publication_requires_exact_paired_gains_and_all_work(tmp_path):
    selected = cell()
    prior_path = tmp_path / 'prior.json'
    priors = [dict(seed=s, solver_seed=s * 10, return_=float(s)) for s in driver.SEEDS]
    for ep in priors:
        ep['return'] = ep.pop('return_')
    prior_path.write_text(json.dumps({'results': [{'episodes': priors}]}))
    selected['reference_results'] = str(prior_path)
    episodes = [{**ep, 'return': ep['return'] + 2, 'paired_return_delta': 2, 'length': 500, 'capped': False} for ep in priors]
    metrics = {'eval/frozen_state_unchanged': True, 'work/environment_decisions': 2500,
        'work/actor_updates': 80000, 'work/critic_updates': 320000, 'work/temperature_updates': 0,
        'work/retrace_trajectory_draws': 128 * 86 * 2500, 'work/retrace_critic_rows': 128 * 86 * 3 * 2500}
    for key in (*driver.RETRACE_DIAGNOSTICS, 'inner_actor_entropy', 'inner_actor_entropy_max',
                'inner_final_outer_policy_kl', 'inner_alpha_initial', 'inner_alpha_final', 'inner_alpha_delta'):
        metrics['diagnostics/' + key + '_mean'] = 0.
    record = dict(checkpoint={'step': selected['step'], 'sha256': selected['checkpoint_sha256']},
                  episodes=episodes, metrics=metrics)
    driver.validate_record(selected, record)
    wrong = deepcopy(record); wrong['episodes'][0]['paired_return_delta'] = 1.
    with pytest.raises(ValueError, match='Paired return'):
        driver.validate_record(selected, wrong)
    wrong = deepcopy(record); wrong['metrics']['work/retrace_critic_rows'] = 0
    with pytest.raises(ValueError, match='work mismatch'):
        driver.validate_record(selected, wrong)


def test_launcher_is_offline_and_has_distinct_cpu_gate_runs():
    text = (driver.REPO / 'slurm/run_ambi_retrace_kl_mppi_oscar.sbatch').read_text()
    assert 'WANDB_MODE=disabled' in text and 'EXPECTED_ACTION_MODES_SHA' in text
    assert '--index 0 --smoke-steps 32 --cpu-threads 1' in text
    assert '--index 0 --smoke-steps 32 --cpu-threads 2' in text
    assert '--index 3 --smoke-steps 8 --cpu-threads 1' in text
    assert '#SBATCH' not in text  # Live resource sizing belongs to the submission.
