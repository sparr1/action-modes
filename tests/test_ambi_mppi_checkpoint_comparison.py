"""Scientific pins and reference-reuse guards for the MPPI checkpoint campaign."""
from copy import deepcopy
import json
import pytest

from slurm import ambi_mppi_checkpoint_comparison as driver


def test_resolved_mppi_recipe_matches_training_planner_and_has_no_updates():
    from test_ambi_research_presets import _build_cfg
    from utils.ambi_benchmark import reference_protocol_for
    from utils.ambi_research import load_preset_matrix
    matrix = load_preset_matrix(driver.MATRIX)
    driver.validate_matrix(matrix)
    base = driver.read(driver.REPO/'configs/dmcontrol/algs/ambi_full_h3_mppi_aux_return.json')
    base['alg_params'].update(matrix['shared_alg_params'])
    resolved = dict(algorithm_config=base, evaluation=matrix['evaluation'],
                    environment={'id':'DMControl-v0', 'params':{'task':'humanoid-walk','obs':'state'}})
    cfg = _build_cfg(resolved)
    assert cfg.inner_mppi_num_samples == 512
    assert cfg.inner_mppi_iterations == 8
    assert cfg.inner_actor_updates_per_action == cfg.inner_critic_updates_per_action == 0
    assert cfg.inner_temperature_updates_per_action == 0
    assert cfg.aux_return_mode == 'sac'
    assert cfg.inner_horizon_critic_source == 'aux_return'
    protocol = reference_protocol_for(resolved,55,500)
    assert protocol['action_rule'] == 'mppi_weighted_elite'
    assert protocol['prior_reference'] == 'ambi_sampled_prior-v1'


@pytest.mark.parametrize('key,value',[('inner_rollout_horizon',2),('inner_mppi_iterations',6),
    ('inner_horizon_critic_source','sac'),('inner_mppi_warm_start_scope','action'),
    ('inner_mppi_eval_action','mean'),('wandb',True)])
def test_scientific_settings_cannot_change(key,value):
    matrix=driver.read(driver.MATRIX)
    matrix['shared_alg_params'][key]=value
    with pytest.raises(ValueError,match='MPPI settings'):
        driver.validate_matrix(matrix)


def test_only_authorized_checkpoint_hashes_and_saved_sampled_priors():
    rows=driver.read(driver.PROFILE)['checkpoints']
    assert [r['step'] for r in rows] == [400000,500000]
    assert len({r['checkpoint_sha256'] for r in rows}) == 2
    assert all(r['source_run']=='rwgao_b-brown-university/ambi/mppi7530927x0' for r in rows)
    old=driver.read(driver.REPO/'configs/research/ambi_entropy_kl_mppi500k_profile.json')
    assert rows[1]['checkpoint_sha256']==old['checkpoints']['mppi_aux_500k']['checkpoint_sha256']
    assert 'baselines-8401ce40/cell-07/bundle' in rows[0]['reference_bundle']
    assert 'production/prior/bundle' in rows[1]['reference_bundle']


def test_smoke_is_separate_and_full_episodes_reuse_reference(tmp_path):
    cell=dict(driver.read(driver.PROFILE)['checkpoints'][0],inventory_path='/inventory.json')
    full=driver.evaluator_command(cell,tmp_path)
    assert full[full.index('--reference-bundle')+1]==cell['reference_bundle']
    assert full[full.index('--seeds')+1:full.index('--controller-seed')]==list(map(str,driver.SEEDS))
    assert full[full.index('--max-steps')+1]=='500'
    smoke=driver.evaluator_command(cell,tmp_path,32)
    assert '--reference-bundle' not in smoke
    assert smoke[smoke.index('--seeds')+1:smoke.index('--controller-seed')]==['101']
    assert '--reference-bundle' not in driver.evaluator_command(cell,tmp_path,specification_only=True)
    with pytest.raises(ValueError): driver.evaluator_command(cell,tmp_path,True)


def test_reference_requires_exact_checkpoint_sampled_prior_and_complete_seeds(tmp_path):
    bundle=tmp_path/'bundle';bundle.mkdir()
    (bundle/'manifest.json').write_text('{}')
    path=tmp_path/'results.json'
    payload=dict(checkpoint_sha256='checkpoint',results=[dict(execution_action='sample',
        outer_state_unchanged=True,resolved_config={'inner_operator':'none'},episodes=[
            dict(seed=s,length=500,truncated_by_evaluator=False) for s in driver.SEEDS])])
    def pins(value):
        path.write_text(json.dumps(value))
        return dict(reference_bundle=str(bundle),reference_manifest_sha256=driver.sha256(bundle/'manifest.json'),
                    reference_results=str(path),reference_results_sha256=driver.sha256(path),checkpoint_sha256='checkpoint')
    driver.reference_pins(pins(payload))
    for field,value in [('execution_action','mean'),('outer_state_unchanged',False),('episodes',[])]:
        changed=deepcopy(payload);changed['results'][0][field]=value
        with pytest.raises(ValueError): driver.reference_pins(pins(changed))
    changed=deepcopy(payload);changed['checkpoint_sha256']='different'
    with pytest.raises(ValueError,match='checkpoint'): driver.reference_pins(pins(changed))
    cell=pins(payload);path.write_text('{}')
    with pytest.raises(ValueError,match='changed'): driver.reference_pins(cell)


def test_mppi_trace_guard_accepts_empty_manifest_and_rejects_sac_trace_files(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'runs': [{'trace_files': []}]}))
    assert driver.trace_hashes(tmp_path) == {}
    for runs in ([{'trace_files': ['trace.jsonl.gz']}], [], [{}, {}], [{}]):
        path.write_text(json.dumps({'runs': runs}))
        with pytest.raises(ValueError, match='without SAC optimization traces'):
            driver.trace_hashes(tmp_path)
