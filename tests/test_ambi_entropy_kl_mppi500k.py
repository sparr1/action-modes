"""Guard checkpoint identity, paired references and the four objective settings."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import pytest

from slurm import ambi_entropy_kl_mppi500k as driver


def campaign(root):
    profile = driver.read(driver.REPO / driver.PROFILE)
    row = profile['checkpoints'][driver.CHECKPOINT_ID]
    prior = root / 'production' / 'prior'
    payload = dict(schema_version=1, campaign_id=driver.CAMPAIGN_ID, seeds=driver.SEEDS,
                   controller_seed=55, max_steps=500, execution_action='sample', source_commit='a'*40,
                   matrix_sha256=driver.sha256(driver.REPO / driver.MATRIX),
                   profile_sha256=driver.sha256(driver.REPO / driver.PROFILE), cells=[])
    inventory = root / 'inventory.json'
    inventory.write_text('{}')
    for i, mode in enumerate(driver.MODES):
        payload['cells'].append({**row, **driver.intervention(mode), 'index': i,
            'output_dir': str(root / 'production' / mode), 'run_id': 'run'+str(i),
            'run_dir': str(root / 'registries' / mode), 'reference_output_dir': str(prior),
            'reference_bundle': str(prior / 'bundle'), 'inventory_path': str(inventory),
            'inventory_sha256': driver.sha256(inventory)})
    return payload


def test_all_four_matrices_materialize_with_correct_actual_update_counts():
    from test_ambi_research_presets import _build_cfg
    from utils.ambi_benchmark import frozen_execution_action
    matrix = driver.read(driver.REPO / driver.MATRIX)
    driver.validate_matrix(matrix)
    base = driver.read(driver.REPO / 'configs/dmcontrol/algs/ambi_full_h3_mppi_aux_return.json')
    for mode in driver.MODES:
        run = deepcopy(base)
        params = {**base['alg_params'], **matrix['shared_alg_params'], **driver.overrides(mode)}
        params = {k: v for k, v in params.items() if v is not None}
        run['alg_params'] = params
        cfg = _build_cfg({'algorithm_config': run})
        assert frozen_execution_action({'algorithm_config': run,
                                        'evaluation': {'execution_action': 'sample'}}) == 'sample'
        assert cfg.inner_temperature_updates_per_action == driver.intervention(mode)['expected_temperature_steps']
        assert cfg.inner_actor_updates_per_action == (0 if mode == 'prior' else 32)
        assert cfg.inner_critic_updates_per_action == (0 if mode == 'prior' else 128)
        assert cfg.sac_actor_loss_scale_mode == 'none'
        assert cfg.inner_temperature_mode == ('auto' if mode in ('entropy', 'both') else 'inherit_outer')
        assert cfg.inner_entropy_enabled == (mode in ('entropy', 'both'))
        assert cfg.inner_outer_policy_kl_coef == driver.intervention(mode)['beta']
        assert cfg.inner_horizon_critic_source == 'aux_return'


@pytest.mark.parametrize('field,value', [('inner_actor_lr', .003), ('inner_critic_updates_per_round', 48),
                                         ('inner_entropy_enabled', False)])
def test_shared_science_cannot_change(field, value):
    matrix = driver.read(driver.REPO / driver.MATRIX)
    matrix['shared_alg_params'][field] = value
    with pytest.raises(ValueError, match='Shared configuration'):
        driver.validate_matrix(matrix)


def test_campaign_rejects_wrong_checkpoint_mode_order_and_aliasing(tmp_path):
    value = campaign(tmp_path)
    driver.validate_campaign(value)
    for field, wrong in [('step', 400000), ('outer_alpha', .013), ('mode', 'both'),
                         ('beta', 1.), ('expected_temperature_steps', 32), ('checkpoint_sha256', 'wrong')]:
        changed = deepcopy(value)
        changed['cells'][0][field] = wrong
        with pytest.raises(ValueError):
            driver.validate_campaign(changed)
    value['cells'][1]['run_dir'] = value['cells'][0]['run_dir']
    with pytest.raises(ValueError, match='Duplicate cell ownership'):
        driver.validate_campaign(value)


def test_prepare_and_smoke_commands_never_need_an_unfinished_prior(tmp_path):
    value = campaign(tmp_path)
    for cell in value['cells']:
        spec = driver.evaluator_command(cell, tmp_path, specification_only=True)
        smoke = driver.evaluator_command(cell, tmp_path, smoke_steps=32)
        full = driver.evaluator_command(cell, tmp_path)
        assert '--reference-bundle' not in spec and '--reference-bundle' not in smoke
        assert ('--reference-bundle' in full) == (cell['mode'] != 'prior')
        assert smoke[smoke.index('--seeds')+1:smoke.index('--controller-seed')] == ['101']
        assert full[full.index('--max-steps')+1] == '500'
        assert '--eval-run-dir' not in full and '--wandb' not in full
    with pytest.raises(ValueError):
        driver.evaluator_command(value['cells'][0], tmp_path, smoke_steps=True)


def test_actual_saved_alpha_is_checked(tmp_path):
    import torch
    path = tmp_path / 'checkpoint.pt'
    logalpha = torch.tensor(driver.OUTER_ALPHA).log()
    torch.save({'log_ent_coef': logalpha, 'checkpoint_version': 3,
                'entropy_spec': {'target_entropy': -10.5}}, path)
    cell = dict(checkpoint=str(path), outer_alpha=logalpha.exp().item(), checkpoint_version=3)
    assert driver.verify_outer_alpha(cell) == cell['outer_alpha']
    cell['outer_alpha'] = .013
    with pytest.raises(ValueError, match='Actual checkpoint alpha'):
        driver.verify_outer_alpha(cell)


@pytest.mark.parametrize('mode', ['entropy', 'kl', 'both'])
def test_adapted_result_checks_actual_alpha_and_temperature(tmp_path, monkeypatch, mode):
    from slurm import ambi_early_h3_j8_validation as common
    cell = {**driver.intervention(mode), 'checkpoint_sha256': 'pinned'}
    kwargs = {}
    def validate(*a, **k):
        kwargs.update(k)
        return {'selector': cell['selector'], 'execution_action': 'sample'}
    monkeypatch.setattr(common, 'validate_result', validate)
    values = {'inner_actor_entropy': 1., 'inner_alpha_initial': driver.OUTER_ALPHA if cell['entropy_enabled'] else 0.,
              'inner_alpha_final': .02 if cell['entropy_enabled'] else 0., 'inner_alpha_delta': 0.,
              'inner_final_outer_policy_kl': .1, 'inner_outer_policy_kl': .1, 'inner_target_entropy': -10.5}
    if mode == 'entropy':
        values.pop('inner_outer_policy_kl')  # Kernel omits a disabled regularizer statistic.
    metrics = {k: dict(count=32, min=v, max=v, mean=v) for k, v in values.items()}
    path = tmp_path / 'results.json'
    path.write_text(json.dumps({'checkpoint_sha256': 'pinned', 'results': [{'model_metrics': metrics}]}))
    driver.validate_episode_result(tmp_path, cell, 32)
    assert kwargs['expected_temperature_steps'] == cell['expected_temperature_steps']
    assert kwargs['config_overrides']['sac_actor_loss_scale_mode'] == 'none'
    metrics['inner_alpha_initial']['max'] += .01
    path.write_text(json.dumps({'checkpoint_sha256': 'pinned', 'results': [{'model_metrics': metrics}]}))
    with pytest.raises(ValueError, match='inner_alpha_initial'):
        driver.validate_episode_result(tmp_path, cell, 32)


def prior_artifact(tmp_path, *, steps=2):
    from test_ambi_early_h3_j8_validation import artifact, save, summary
    cell = {**driver.read(driver.REPO / driver.PROFILE)['checkpoints'][driver.CHECKPOINT_ID],
            **driver.intervention('prior')}
    result_path, bundle, payload, manifest = artifact(tmp_path, seeds=(101,), steps=steps, mode='sample')
    result_path.rename(tmp_path / 'results.json')
    result, run = payload['results'][0], manifest['runs'][0]
    payload['checkpoint_sha256'] = manifest['checkpoint']['sha256'] = cell['checkpoint_sha256']
    result['selector'] = run['selector'] = cell['selector']
    for cfg in (result['resolved_config'], run['resolved_config']):
        cfg.update(driver.overrides('prior'), inner_outer_policy_kl_coef=0., sac_actor_loss_scale_mode='none')
    constants = {k: 0. for k in ('inner_active','inner_model_steps','inner_critic_optimizer_steps',
                 'inner_actor_optimizer_steps','inner_temperature_optimizer_steps','inner_compile_fallback')}
    constants.update(inner_alpha_initial=driver.OUTER_ALPHA, inner_alpha_final=driver.OUTER_ALPHA, inner_alpha_delta=0.)
    result['model_metrics'] = {k: summary([v]*steps) for k,v in constants.items()}
    result['episodes'][0]['model_metrics'] = dict(constants)
    result['episodes'][0]['togo_probe_model_steps'] = 192
    run['episodes'][0].update(result['episodes'][0])
    run['togo_probe_rows'] = [r for r in run['togo_probe_rows'] if r['round_index'] == 0]
    path = bundle / run['trace_files'][0]
    with gzip.open(path, 'rt') as f:
        events = [json.loads(line) for line in f]
    events = [e for e in events if e['phase'] in ('initial','decision') or e['phase']=='probe' and e['round_index']==0]
    indices = {}
    for event in events:
        key=event['decision_index'];event['event_index']=indices.get(key,0);indices[key]=event['event_index']+1
        for k in ('round_index','critic_updates','actor_updates','temperature_updates'):event[k]=0
        if event['phase']=='decision':event['metrics'].update({'decision/'+k:v for k,v in constants.items()})
    with gzip.open(path,'wt') as f:f.writelines(json.dumps(e)+'\n' for e in events)
    run['result'] = deepcopy(result)
    save(tmp_path/'results.json',payload);save(bundle/'manifest.json',manifest)
    return cell, payload, manifest


def test_prior_has_every_sampled_decision_but_no_updates(tmp_path):
    cell, payload, manifest = prior_artifact(tmp_path)
    validated = driver.validate_prior_result(tmp_path, cell, [101], 2)
    assert validated['decisions'] == 2 and validated['inner_updates'] == 0
    assert validated['probe_boundaries'] == 1
    from test_ambi_early_h3_j8_validation import alter_trace
    alter_trace(tmp_path/'bundle',manifest,lambda rows: rows[-1].update(actor_updates=1))
    with pytest.raises(ValueError, match='trace counters'):
        driver.validate_prior_result(tmp_path,cell,[101],2)


def test_production_refuses_missing_prior_before_gpu(tmp_path, monkeypatch):
    value = campaign(tmp_path)
    path=tmp_path/'campaign.json';path.write_text(json.dumps(value))
    monkeypatch.setattr(driver,'source',lambda:value['source_commit'])
    monkeypatch.setattr(driver,'verify_checkpoint',lambda *a:None)
    monkeypatch.setattr(driver,'verify_outer_alpha',lambda *a:None)
    monkeypatch.setattr(driver.subprocess,'run',lambda *a,**k:pytest.fail('GPU must not start'))
    with pytest.raises(FileNotFoundError):driver.run(path,1)
    assert not Path(value['cells'][1]['output_dir']).exists()


def test_launcher_requires_explicit_resources_and_source():
    script=(driver.REPO/'slurm/run_ambi_entropy_kl_mppi500k_oscar.sbatch').read_text()
    for expected in ('SLURM_CPUS_PER_TASK:?', 'EXPECTED_ACTION_MODES_SHA:?', 'WANDB_MODE=disabled',
                     'ambi_entropy_kl_mppi500k.py run'):
        assert expected in script
    assert '#SBATCH --time' not in script
