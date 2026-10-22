"""Ten fresh sampled episodes retain paired RNG and exact reset/carry lifetimes."""
from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest

from slurm import ambi_alpha_long_episodes as driver


def cell(mode='episode_reset', stream=4):
    return {**driver.read(driver.PROFILE)['checkpoints'][driver.CHECKPOINT_ID],
            **driver.intervention(mode, stream), 'inventory_path': '/inventory',
            'science_id': mode + '-stream' + str(stream)}


def artifact(tmp_path, mode='episode_reset', episodes=2, steps=8):
    from test_ambi_early_h3_j8_validation import artifact as base_artifact, summary
    selected=cell(mode)
    seeds=selected['seeds'][:episodes]
    _,bundle,payload,manifest=base_artifact(tmp_path,seeds=seeds,steps=steps,mode='sample',
                                         config_overrides=driver.config_overrides(selected))
    result,run=payload['results'][0],manifest['runs'][0]
    result['selector']=run['selector']=selected['selector']
    payload['checkpoint_sha256']=manifest['checkpoint']['sha256']=selected['checkpoint_sha256']
    result['return']=summary([steps*.1]*episodes)
    if mode=='run_carry':
        result.update(temperature_chain=True,temperature_chain_seeds=seeds,temperature_chain_kind=driver.chain.PROTOCOL)
        manifest['protocol']['temperature_chain']=dict(kind=driver.chain.PROTOCOL,ordered_environment_seeds=seeds)
    for index,(episode,saved) in enumerate(zip(result['episodes'],run['episodes'])):
        episode['return']=saved['return']=steps*.1
        if mode=='run_carry':
            episode['temperature_chain_episode_index']=saved['temperature_chain_episode_index']=index
    run['result']=deepcopy(result)
    for episode,relative in enumerate(run['trace_files']):
        path=bundle/relative
        with gzip.open(path,'rt') as source: rows=[json.loads(line) for line in source]
        for event in rows:
            if event['phase']!='decision': continue
            local=event['decision_index']; global_index=episode*steps+local
            lifetime=global_index if mode=='run_carry' else local
            alpha=selected['outer_alpha']*.999**lifetime
            event['metrics'].update({'decision/inner_alpha_initial':alpha,'decision/inner_alpha_final':alpha*.999,
                'decision/inner_actor_entropy':2.,'decision/inner_target_entropy':-10.5,
                'decision/inner_temperature_optimizer_step_initial':lifetime*32,
                'decision/inner_temperature_optimizer_step_final':(lifetime+1)*32,
                'decision/inner_temperature_lifetime_steps_initial':lifetime*32,
                'decision/inner_temperature_lifetime_steps_final':(lifetime+1)*32,
                'decision/inner_temperature_episode_reset':int(local==0 and mode=='episode_reset'),
                'decision/inner_temperature_chain_reset':int(global_index==0 and mode=='run_carry')})
        with gzip.open(path,'wt') as target: target.writelines(json.dumps(row)+'\n' for row in rows)
    (tmp_path/'results.json').write_text(json.dumps(payload)); (bundle/'manifest.json').write_text(json.dumps(manifest))
    return selected


def test_layout_preserves_old_prefix_and_ten_unique_seeds_per_history():
    driver.validate_matrix(driver.read(driver.MATRIX))
    assert len(driver.STREAMS)==5 and driver.PRIMARY['episode_index']==9
    for index,seeds in enumerate(driver.STREAMS):
        assert len(seeds)==len(set(seeds))==10
        assert seeds[:3]==driver.chain.STREAMS[index]
        assert set(seeds)==set(driver.SEEDS)
    for episode in range(10):
        assert sorted(seeds[episode] for seeds in driver.STREAMS)==list(range(101 if episode<5 else 106,106 if episode<5 else 111))


@pytest.mark.parametrize('key,value',[('inner_behavior_action','mean'),('inner_temperature_lr',.03),
    ('inner_outer_policy_kl_coef',.01),('inner_actor_scope','run'),('inner_temperature_scope','action')])
def test_unrequested_interventions_are_rejected(key,value):
    matrix=driver.read(driver.MATRIX);matrix['shared_alg_params'][key]=value
    with pytest.raises(ValueError,match='Inner learning'): driver.validate_matrix(matrix)


@pytest.mark.parametrize('mode',driver.MODES)
def test_actual_config_keeps_sampled_learning_and_update_dose(mode):
    from test_ambi_research_presets import _build_cfg
    matrix=driver.read(driver.MATRIX)
    base=driver.read(driver.REPO/'configs/dmcontrol/algs/ambi_full_h3_mppi_aux_return.json')
    params={**base['alg_params'],**matrix['shared_alg_params'],
            **matrix['comparisons']['alpha_long_episodes']['variants'][mode]['alg_params']}
    base['alg_params']={key:value for key,value in params.items() if value is not None}
    cfg=_build_cfg({'algorithm_config':base})
    assert cfg.inner_temperature_scope==cfg.inner_temperature_optimizer_scope==('episode' if mode=='episode_reset' else 'run')
    assert cfg.inner_behavior_action==cfg.inner_execution_action=='policy_sample'
    assert cfg.inner_behavior_std_scale==cfg.inner_execution_std_scale==1
    assert (cfg.inner_actor_updates_per_action,cfg.inner_critic_updates_per_action,cfg.inner_temperature_updates_per_action)==(32,128,32)


@pytest.mark.parametrize('mode',driver.MODES)
def test_production_uses_all_ten_ordered_seeds_and_two_episode_smoke(tmp_path,mode):
    selected=cell(mode)
    full,smoke=driver.evaluator_command(selected,tmp_path),driver.evaluator_command(selected,tmp_path,8)
    assert full[full.index('--seeds')+1:full.index('--controller-seed')]==list(map(str,selected['seeds']))
    assert smoke[smoke.index('--seeds')+1:smoke.index('--controller-seed')]==['105','101']
    assert full[full.index('--max-steps')+1]=='500'
    for command in (full,smoke):
        assert ('--temperature-chain' in command)==(mode=='run_carry')
        assert command[command.index('--matrix')+1]==str(driver.MATRIX)
        assert command[command.index('--preset')+1]=='alpha_long_episodes/'+mode
        assert not {'--reference-bundle','--eval-run-dir','--wandb'}.intersection(command)


@pytest.mark.parametrize('mode',driver.MODES)
def test_two_episode_smoke_requires_correct_execution_and_lifecycle(tmp_path,mode):
    selected=artifact(tmp_path,mode)
    validation,summary=driver.validate_result(tmp_path,selected,8)
    assert validation['passed'] and validation['execution_action']=='sample'
    assert summary['long_protocol']==driver.PROTOCOL and summary['arm']==mode
    boundary=summary['rows'][8]
    assert boundary['optimizer_step_initial']==(0 if mode=='episode_reset' else 256)
    assert boundary['alpha_initial']==pytest.approx(selected['outer_alpha'] if mode=='episode_reset' else summary['rows'][7]['alpha_final'])


@pytest.mark.parametrize('mode',driver.MODES)
def test_ten_episode_trace_and_last_boundary_supported_without_seed_aliasing(tmp_path,mode):
    from slurm.ambi_early_h3_j8_validation import validate_result
    selected=artifact(tmp_path,mode,episodes=10,steps=2)
    result=validate_result(tmp_path/'results.json',tmp_path/'bundle',selected['seeds'],2,
                           config_overrides=driver.config_overrides(selected))
    assert result['episodes']==10 and result['decisions']==20
    summary=driver.compact_summary(selected,tmp_path/'bundle',selected['seeds'],2)
    driver.validate_decisions(summary,selected,selected['seeds'],2)
    assert [episode['seed'] for episode in summary['episodes']]==selected['seeds']
    assert summary['rows'][18]['episode_index']==9
    assert summary['rows'][18]['optimizer_step_initial']==(0 if mode=='episode_reset' else 18*32)
    assert summary['rows'][19]['optimizer_step_final']==(64 if mode=='episode_reset' else 20*32)


@pytest.mark.parametrize('mode',driver.MODES)
@pytest.mark.parametrize('field,value',[('alpha_initial',.01),('optimizer_step_initial',64),
    ('lifetime_steps_initial',64),('entropy',float('nan')),('global_decision_index',0)])
def test_state_or_identity_discontinuities_are_rejected(tmp_path,mode,field,value):
    selected=artifact(tmp_path,mode);_,summary=driver.validate_result(tmp_path,selected,8)
    summary['rows'][8][field]=value
    with pytest.raises(ValueError):driver.validate_decisions(summary,selected,selected['seeds'][:2],8)


@pytest.mark.parametrize('mode',driver.MODES)
def test_missing_last_episode_or_reordered_rows_rejected(tmp_path,mode):
    selected=artifact(tmp_path,mode,episodes=10,steps=2)
    summary=driver.compact_summary(selected,tmp_path/'bundle',selected['seeds'],2)
    missing=deepcopy(summary);missing['rows']=missing['rows'][:-2]
    with pytest.raises(ValueError):driver.validate_decisions(missing,selected,selected['seeds'],2)
    summary['rows'][18],summary['rows'][19]=summary['rows'][19],summary['rows'][18]
    with pytest.raises(ValueError):driver.validate_decisions(summary,selected,selected['seeds'],2)


@pytest.mark.parametrize('mode',driver.MODES)
def test_mean_execution_cannot_masquerade_as_sampled(tmp_path,mode):
    selected=artifact(tmp_path,mode)
    payload=driver.read(tmp_path/'results.json');manifest=driver.read(tmp_path/'bundle/manifest.json')
    payload['deterministic_execution']=True;payload['results'][0]['execution_action']='mean'
    manifest['protocol']['action_rule']='tanh_mean';manifest['runs'][0]['result']=deepcopy(payload['results'][0])
    (tmp_path/'results.json').write_text(json.dumps(payload));(tmp_path/'bundle/manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='real-action execution'):driver.validate_result(tmp_path,selected,8)


@pytest.mark.parametrize('mode',driver.MODES)
def test_sparse_optional_metrics_are_not_imputed(tmp_path,mode):
    from test_ambi_early_h3_j8_validation import alter_trace
    selected=artifact(tmp_path,mode);manifest=driver.read(tmp_path/'bundle/manifest.json')
    def add(rows):next(row for row in rows if row['phase']=='decision')['metrics']['decision/inner_final_outer_policy_kl']=1.25
    alter_trace(tmp_path/'bundle',manifest,add)
    _,summary=driver.validate_result(tmp_path,selected,8)
    assert summary['rows'][0]['root_prior_kl']==1.25 and 'root_prior_kl' not in summary['rows'][1]
    assert summary['optional_diagnostic_counts']['root_prior_kl']==1


@pytest.mark.parametrize('mode',driver.MODES)
@pytest.mark.parametrize('changed',['chain-summary.json','results.json','trace','receipt'])
def test_reload_rehashes_artifacts_and_enforces_sampled_arm(tmp_path,mode,changed):
    selected=artifact(tmp_path,mode);validation,summary=driver.validate_result(tmp_path,selected,8)
    driver.write_new(tmp_path/'chain-summary.json',summary);campaign=dict(source_commit='a'*40)
    receipt=dict(validated=True,smoke=True,source_commit=campaign['source_commit'],protocol=driver.PROTOCOL,
        science_id=selected['science_id'],cell_id=selected['cell_id'],checkpoint_sha256=selected['checkpoint_sha256'],
        execution_action='sample',arm=mode,validation=validation,results_sha256=driver.sha256(tmp_path/'results.json'),
        manifest_sha256=driver.sha256(tmp_path/'bundle/manifest.json'),chain_summary_sha256=driver.sha256(tmp_path/'chain-summary.json'),
        trace_sha256=driver.trace_hashes(tmp_path/'bundle'))
    driver.write_new(tmp_path/'receipt.json',receipt)
    assert driver.load_stream(campaign,selected,tmp_path,8)[0]==summary
    if changed=='receipt':receipt['execution_action']='mean';(tmp_path/'receipt.json').write_text(json.dumps(receipt))
    elif changed=='trace':
        path=tmp_path/'bundle'/next(iter(receipt['trace_sha256']));path.write_bytes(path.read_bytes()+b'changed')
    else:(tmp_path/changed).write_text('{}')
    with pytest.raises(ValueError):driver.load_stream(campaign,selected,tmp_path,8)


def reference_fixture(tmp_path,monkeypatch):
    pin=driver.read(driver.PROFILE)['checkpoints'][driver.CHECKPOINT_ID]
    inventory=tmp_path/'inventory.json';inventory.write_text('{}')
    old=dict(source_commit='b'*40,profile_sha256=driver.sha256(driver.PROFILE),cells=[])
    calls=[]
    for index in range(5):
        output=tmp_path/f'old{index}';(output/'bundle').mkdir(parents=True)
        selected={**pin,'stream_index':index,'seeds':driver.STREAMS[index][:3],'output_dir':str(output),
                  'inventory_path':str(inventory),'inventory_sha256':driver.sha256(inventory)}
        old['cells'].append(selected)
        for filename in ('results.json','bundle/manifest.json','chain-summary.json'):(output/filename).write_text('{}')
        receipt=dict(validation={'passed':True},elapsed_seconds=3000+index,results_sha256=driver.sha256(output/'results.json'),
            manifest_sha256=driver.sha256(output/'bundle/manifest.json'),chain_summary_sha256=driver.sha256(output/'chain-summary.json'),trace_sha256={})
        (output/'receipt.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(driver.carry,'validate_campaign',lambda value:None)
    monkeypatch.setattr(driver.carry,'load_stream',lambda old,selected:({},driver.read(Path(selected['output_dir'])/'receipt.json')))
    def validate(output,selected):calls.append(selected['stream_index']);return {'passed':True},{}
    monkeypatch.setattr(driver.carry,'validate_result',validate)
    return old,calls


def test_old_results_are_pinned_for_runtime_only_not_fake_long_controls(tmp_path,monkeypatch):
    old,calls=reference_fixture(tmp_path,monkeypatch)
    refs=driver.reference_snapshots(tmp_path/'new',old)
    assert calls==list(range(5)) and 'reference_streams' not in refs and 'reference_summaries' not in refs
    assert len(refs['observed_full_history_receipts'])==5
    assert 'no performance controls' in refs['historical_use']
    driver.chain.verify_files(refs['reference_files'])
    first=refs['observed_full_history_receipts'][0];Path(first['path']).write_text('{}')
    with pytest.raises(ValueError):driver.chain.verify_files(refs['reference_files'])


@pytest.mark.parametrize('change',['prefix','checkpoint','validation','timing'])
def test_historical_provenance_or_timing_failure_blocks_preparation(tmp_path,monkeypatch,change):
    old,_=reference_fixture(tmp_path,monkeypatch)
    if change=='prefix':old['cells'][0]['seeds']=[101,102,106]
    elif change=='checkpoint':old['cells'][0]['checkpoint_sha256']='wrong'
    elif change=='validation':monkeypatch.setattr(driver.carry,'validate_result',lambda *args:({'passed':False},{}))
    else:
        path=Path(old['cells'][0]['output_dir'])/'receipt.json';receipt=driver.read(path)
        receipt['elapsed_seconds']=float('nan');path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):driver.reference_snapshots(tmp_path/'new',old)


def test_prepare_refuses_existing_campaign(tmp_path):
    (tmp_path/'campaign.json').write_text('{}')
    with pytest.raises(FileExistsError):driver.prepare(tmp_path)
