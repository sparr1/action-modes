"""Publication keeps full paired coverage, frequency work and missing-data semantics."""
from copy import deepcopy
import sys
from types import SimpleNamespace

import pytest

from slurm import ambi_closed_loop_sac_scale_publish as publication
from utils.ambi_benchmark import solver_seed


def episodes(offset):
    return [dict(seed=s,solver_seed=solver_seed(55,'episode',s),length=500,truncated_by_evaluator=False,
        **{'return':float(offset+s-100)},control_seconds=50.,togo_probe_seconds=5.) for s in publication.SEEDS]


def fixture_campaign():
    prior=dict(episodes=episodes(350),checkpoint_step=575000,checkpoint_sha256='575k',
        performance_run_id='prior',manifest_sha256='prior-manifest',bundle='prior-bundle')
    cells=[]
    for h in (1,2,3):
        for p in (1,5):
            for j in publication.ROUNDS:
                name=f'h{h}_p{p}_j{j}'
                params=dict(inner_rollouts_per_round=1024,inner_batch_size=4096,inner_updates_per_round=20,
                    inner_actor_update_interval=p,inner_critic_target_update_interval=2,inner_rollout_horizon=h,
                    inner_rounds=j,inner_critic_source='aux_return',inner_horizon_critic_source='aux_return',
                    inner_sac_critic_target='reward_only',inner_terminal_entropy='none',inner_temperature_mode='auto',
                    inner_entropy_enabled=True,inner_execution_action='mean',inner_replay_capacity=1024*h*j)
                cells.append(dict(name=name,H=h,J=j,N=1024,B=4096,G=20,P=p,T=2,C=20,A=20//p,target_updates=10,
                    checkpoint_step=575000,training_decisions=575000,checkpoint_sha256='575k',reused=False,
                    initial_alpha=.0046,params=params,requested_alg_params=deepcopy(params),selector='sweep/'+name,
                    actual_selector='sweep/'+name,prior_reference=deepcopy(prior),
                    performance_run_id='performance-'+name,training_run_id='training-'+name,
                    directory='directory-'+name,bundle='bundle-'+name,run_dir='run-'+name))
    return dict(cells=cells,group='test-sac-scale',label='575k SAC scale',source_run='backbone',
        source_commit='tested',inventory='inventory.json',overview_run_id='overview',publisher_workers=3)


def completed(campaign):
    return {c['name']:[dict(e,**{'return':e['return']+c['J']},paired_return_delta=c['J'])
            for e in c['prior_reference']['episodes']] for c in campaign['cells']}


def test_pending_sweep_never_invents_outcomes_or_timings():
    aggregate=publication.aggregate_results(fixture_campaign(),{})
    assert aggregate['evaluated']==aggregate['reused']==0
    assert len(aggregate['points'])==36 and aggregate['episodes']==[]
    for point in aggregate['points']:
        assert point['prior_return_mean']==353
        assert all(point[k] is None for k in ('return_mean','return_std','paired_gain_mean','paired_gain_ci95_low',
            'paired_gain_ci95_high','paired_episodes','control_seconds','control_ms_per_decision','probe_seconds'))
    assert publication.numeric_rows(aggregate)==[]
    charts=publication.chart_payloads(aggregate)
    assert len(charts)==2
    assert charts['comparison/return_mean_vs_J']['keys']==['Frozen prior (reused)']
    assert charts['comparison/paired_gain_mean_vs_J']['ys']==[[0.]*6]


def test_complete_sweep_pairs_by_seeds_and_separates_schedules_and_latency():
    campaign=fixture_campaign()
    aggregate=publication.aggregate_results(campaign,{k:list(reversed(v)) for k,v in completed(campaign).items()})
    assert aggregate['evaluated']==36 and len(aggregate['episodes'])==180
    assert aggregate['bootstrap_resamples']==2000 and aggregate['bootstrap_seed']==20260912
    for point in aggregate['points']:
        assert point['paired_gain_mean']==point['paired_gain_ci95_low']==point['paired_gain_ci95_high']==point['J']
        assert point['return_episodes']==point['paired_episodes']==5
        assert point['control_seconds']==250 and point['control_ms_per_decision']==100
        assert point['probe_seconds']==25 and point['probe_ms_per_decision']==10
    charts=publication.chart_payloads(aggregate)
    assert len(charts['comparison/return_mean_vs_J']['keys'])==7
    assert len(charts['comparison/control_ms_per_decision_vs_J']['keys'])==6
    rows=publication.numeric_rows(aggregate)
    assert len(rows)==36 and len({name for name,_ in rows})==36
    assert all('axis/inner_rounds' in row for _,row in rows)


@pytest.mark.parametrize('damage',['duplicate','shared_id','shared_run_dir','wrong_prior','wrong_checkpoint',
    'reused','wrong_cadence','wrong_batch','phased_budget','eviction','reset','bad_alpha','different_alpha'])
def test_scope_rejects_ownership_or_scientific_drift(damage):
    campaign=fixture_campaign(); cell=campaign['cells'][0]
    if damage=='duplicate': campaign['cells'][-1]=deepcopy(cell)
    elif damage=='shared_id': campaign['cells'][1]['training_run_id']=cell['performance_run_id']
    elif damage=='shared_run_dir': campaign['cells'][1]['run_dir']=cell['run_dir']
    elif damage=='wrong_prior': cell['prior_reference']['checkpoint_step']=650000
    elif damage=='wrong_checkpoint': cell['checkpoint_step']=650000
    elif damage=='reused': cell['reused']=True
    elif damage=='wrong_cadence': cell['params']['inner_actor_update_interval']=3
    elif damage=='wrong_batch': cell['B']=2048
    elif damage=='phased_budget': cell['params']['inner_critic_updates_per_round']=20
    elif damage=='eviction': cell['params']['inner_replay_capacity']=1
    elif damage=='reset': cell['params']['inner_replay_reset_each_round']=True
    elif damage=='bad_alpha': cell['initial_alpha']=float('nan')
    else: cell['initial_alpha']=.001
    with pytest.raises(ValueError): publication.aggregate_results(campaign,{})


@pytest.mark.parametrize('damage',['missing','duplicate','solver','short','nonfinite','gain','unknown','negative_time'])
def test_incomplete_or_invalid_result_cannot_enter_measured_curves(damage):
    campaign=fixture_campaign(); values=completed(campaign); first=values[campaign['cells'][0]['name']]
    if damage=='missing': first.pop()
    elif damage=='duplicate': first[0]=deepcopy(first[1])
    elif damage=='solver': first[0]['solver_seed']+=1
    elif damage=='short': first[0]['length']=499
    elif damage=='nonfinite': first[0]['return']=float('nan')
    elif damage=='gain': first[0]['paired_return_delta']+=1
    elif damage=='unknown': values['unknown']=episodes(0)
    else: first[0]['control_seconds']=-1
    with pytest.raises(ValueError): publication.aggregate_results(campaign,values)


def test_missing_latency_is_unavailable_not_zero_and_measured_elapsed_is_separate():
    campaign=fixture_campaign(); cell=campaign['cells'][0]; values={cell['name']:completed(campaign)[cell['name']]}
    for episode in values[cell['name']]: episode.pop('control_seconds'); episode.pop('togo_probe_seconds')
    aggregate=publication.aggregate_results(campaign,values,{cell['name']:{'evaluation_elapsed_seconds':500}})
    point=next(p for p in aggregate['points'] if p['setting']==cell['name'])
    assert point['evaluation_elapsed_seconds']==500 and point['control_ms_per_decision'] is None
    assert 'comparison/control_ms_per_decision_vs_J' not in publication.chart_payloads(aggregate)


def test_status_preserves_publication_failures_and_pending_values(monkeypatch):
    campaign=fixture_campaign(); cell=campaign['cells'][0]; values={cell['name']:completed(campaign)[cell['name']]}
    monkeypatch.setattr(publication,'publication_complete',lambda c:c==campaign['cells'][1])
    rows=publication.status_rows(campaign,publication.aggregate_results(campaign,values),values,{2:None},{3:'failed'})
    assert [r['status'] for r in rows[:5]]==['evaluated_awaiting_publication','published','publishing','publication_failed','queued_or_running']
    assert rows[4]['return_mean'] is None
    assert all(r['performance_url'] and r['training_url'] for r in rows)


def test_real_disabled_wandb_accepts_pending_and_completed_payloads(tmp_path):
    import wandb
    campaign=fixture_campaign(); run=wandb.init(mode='disabled',dir=str(tmp_path))
    try:
        for values in ({},completed(campaign)):
            aggregate=publication.aggregate_results(campaign,values)
            statuses=[dict(p,status='queued_or_running') for p in aggregate['points']]
            payload=publication.overview_log(wandb,aggregate,statuses)
            assert len(payload['comparison/points'].data)==36
            assert len(payload['campaign/settings'].data)==36
            run.log(payload)
    finally: run.finish()


def test_frequency_training_publisher_records_actual_work_and_full_traces(tmp_path,monkeypatch):
    from utils import ambi_benchmark,ambi_diagnostic_series,eval_series,eval_series_data
    campaign=fixture_campaign(); cell=next(c for c in campaign['cells'] if c['H']==3 and c['P']==5 and c['J']==10)
    cell.update(directory=str(tmp_path),bundle=str(tmp_path/'bundle')); bundle=tmp_path/'bundle'; bundle.mkdir()
    manifest=dict(code={'commit':'tested'},runs=[dict(resolved_config=cell['params'],trace_files=[])])
    publication.write(tmp_path/'worker-completion.json',dict(manifest_sha256='manifest',trace_sha256={}))
    monkeypatch.setattr(publication,'validate_completed',lambda *a,**k:manifest)
    record=dict(identity={'planner':'new'},metrics={'eval/paired_episodes':5})
    monkeypatch.setattr(eval_series_data,'load_records',lambda *a,**k:[record])
    monkeypatch.setattr(eval_series,'load_run',lambda *a:dict(identity=record['identity']))
    monkeypatch.setattr(ambi_benchmark,'stage_completed_bundle',lambda *a,**k:{cell['actual_selector']:{'status':'queued'}})
    monkeypatch.setattr(publication,'publish_performance',lambda *a:dict(run_id=cell['performance_run_id'],published=1))
    calls=[]
    def summary(*a,**k):
        calls.append(k); return dict(update_curves=[],per_seed_decisions=[],decision_curves=[])
    monkeypatch.setattr(publication,'training_summary',summary)
    monkeypatch.setattr(ambi_diagnostic_series,'record_from_model_bundle',lambda *a,**k:dict(status='complete',rows=[{}]*27500))
    monkeypatch.setattr(ambi_diagnostic_series,'write_diagnostic_bundle',lambda *a:None)
    monkeypatch.setattr(ambi_diagnostic_series,'diagnostic_history',lambda *a:[])
    run=SimpleNamespace(summary={},define_metric=lambda *a,**k:None,log=lambda *a:None,log_artifact=lambda *a:None,finish=lambda **k:None)
    configs=[]; artifacts=[]
    def init(**kwargs): configs.append(kwargs); return run
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=init,Artifact=lambda *a,**k:SimpleNamespace(add_file=lambda *a,**k:artifacts.append((a,k)))))
    publication._publish_cell(campaign,cell)
    assert calls==[{'expected_steps':500}]
    assert configs[0]['config']['N']==1024 and configs[0]['config']['B']==4096 and configs[0]['config']['P']==5
    assert run.summary['training/critic_updates']==500000
    assert run.summary['training/actor_updates']==100000
    assert run.summary['training/target_updates']==250000
    assert run.summary['diagnostic/paired_rows']==27500
    assert publication.read(tmp_path/'training-publication.json')['status']=='complete'
    assert publication.read(tmp_path/'publication-completion.json')['training_run_id']==cell['training_run_id']
    assert any(k.get('name')=='model-series/report.html' for _,k in artifacts)
    with pytest.raises(RuntimeError,match='already started'): publication._publish_cell(campaign,cell)


def test_actual_campaign_matrix_is_accepted_after_preparation_metadata():
    from slurm.ambi_closed_loop_sac_scale import cells
    campaign=fixture_campaign(); source=cells()
    fixture_by_axis={(c['H'],c['J'],c['P']):c for c in campaign['cells']}
    campaign['cells']=[dict(fixture_by_axis[c['H'],c['J'],c['P']],**c) for c in source]
    assert len(publication.validate_scope(campaign))==36


def test_watcher_publishes_every_new_setting_with_bounded_subprocesses(tmp_path,monkeypatch):
    import threading
    campaign=fixture_campaign(); values=completed(campaign)
    for cell in campaign['cells']:
        cell['directory']=str(tmp_path/cell['name']); directory=tmp_path/cell['name']; directory.mkdir()
        publication.write(directory/'worker-completion.json',{'status':'complete'})
    publication.write(tmp_path/'campaign.json',campaign)
    published=set(); running=set(); maximum=[0]; lock=threading.Lock(); configs=[]
    monkeypatch.setattr(publication,'verify_references',lambda *a:None)
    monkeypatch.setattr(publication,'load_completed',lambda c,cell:(values[cell['name']],{'source':'verified'},{'evaluation_elapsed_seconds':5.}))
    monkeypatch.setattr(publication,'publication_complete',lambda cell:cell['name'] in published)
    def dispatch(command,**kwargs):
        index=int(command[-1])
        with lock: running.add(index); maximum[0]=max(maximum[0],len(running))
        with lock: published.add(campaign['cells'][index]['name']); running.remove(index)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(publication.subprocess,'run',dispatch)
    monkeypatch.setattr(publication.time,'sleep',lambda *a:None)
    monkeypatch.setattr(publication,'overview_log',lambda w,a,s:{'evaluated':a['evaluated']})
    run=SimpleNamespace(summary={},define_metric=lambda *a,**k:None,log=lambda *a:None,finish=lambda **k:None)
    def init(**kwargs): configs.append(kwargs); return run
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=init))
    publication.watch(SimpleNamespace(root=tmp_path))
    assert maximum[0]<=3 and len(published)==36
    assert run.summary['status']=='complete'
    assert configs[0]['config']['total_settings']==36
    assert publication.read(tmp_path/'campaign-completion.json')['published']==36
    assert publication.read(tmp_path/'comparison-results.json')['evaluated']==36
    with pytest.raises(RuntimeError,match='already started'): publication.watch(SimpleNamespace(root=tmp_path))
