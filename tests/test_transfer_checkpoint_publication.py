"""The curve adapter preserves pairing, provenance and absent measurements."""
from copy import deepcopy
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import pytest

from utils import transfer_checkpoint_publication as adapter
from utils.ambi_benchmark import solver_seed
from utils.eval_series import SeriesError,stage_record
from slurm import ambi_transfer_checkpoint_publish as publish


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def episode(seed,value):
    return dict(seed=seed,solver_seed=solver_seed(55,'episode',seed),
        episode_solver_seed=solver_seed(55,'episode',seed),length=500,steps=500,
        **{'return':value},truncated=True,terminated=False,truncated_by_evaluator=False,smoke=False)


@pytest.fixture
def campaign(tmp_path):
    candidate=dict(setting_id='h1_j4_actor1_critic0p5',H=1,J=4,arm='actor1_critic0p5',
        label='H1 J4 · actor 100%, critic 50%',arm_definition={'actor_rho':1.,'critic_rho':.5})
    data=dict(schema_version=1,protocol=adapter.PROTOCOL,source_run='entity/ambi/aux6428346x0',
        source_commit='a'*40,scientific_source=dict(git_head='a'*40,files={'science.py':'d'*64},sha256='f'*64),
        historical_compatibility={'approved_original_commit':'b'*40,'protected_dependencies_unchanged':True},
        seeds=[101,102,103,104,105],controller_seed=55,max_steps=500,candidates=[candidate],checkpoints=[],cells=[])
    for index,step in enumerate((25000,50000,75000)):
        checkpoint=dict(step=step,checkpoint_sha256=str(index+1)*64)
        bundle=tmp_path/'priors'/str(step)
        eps=[episode(seed,value) for seed,value in zip([101,102,103,104,105],[10,20,30,40,50])]
        manifest=dict(status='complete',code=dict(commit='b'*40,dirty=False),
            checkpoint=dict(source_run=data['source_run'],sha256=checkpoint['checkpoint_sha256'],metadata={'checkpoint':{'step':step}}),
            protocol=dict(action_rule='tanh_mean',controller_seed=55,max_steps=500,seed_scheme='sha256-v1',observation='state',
                environment={'id':'DMControl-v0','params':{'task':'humanoid-walk','obs':'state'}}),
            runs=[dict(selector='prior',resolved_config={'inner_operator':'none','inner_actor_source':'sac'},
                episodes=eps,trace_files=['traces.jsonl'],result=dict(episodes=eps,action_rule='tanh_mean',
                    environment_seeds=[101,102,103,104,105],outer_state_unchanged=True,outer_updates_before=2,outer_updates_after=2))])
        save(bundle/'manifest.json',manifest);(bundle/'traces.jsonl').write_text('{}\n')
        save(bundle/'checkpoint.metadata.json',manifest['checkpoint']['metadata'])
        checkpoint.update(metadata=str(bundle/'checkpoint.metadata.json'),metadata_sha256=adapter.digest(bundle/'checkpoint.metadata.json'),
            prior_manifest_code=deepcopy(manifest['code']))
        checkpoint['prior_pin']=dict(bundle=str(bundle),manifest_sha256=adapter.digest(bundle/'manifest.json'),
            checkpoint_step=step,checkpoint_sha256=checkpoint['checkpoint_sha256'],
            selector='prior',trace_sha256={'traces.jsonl':adapter.digest(bundle/'traces.jsonl')})
        data['checkpoints'].append(checkpoint)
        data['cells'].append(dict(index=index,checkpoint_index=index,step=step,name=f"step-{step}/{candidate['setting_id']}",
            result_dir=str(tmp_path/'results'/str(step)),**candidate))
    return data


def prior(campaign,index=0):
    checkpoint=campaign['checkpoints'][index]
    return adapter.normalize_prior(campaign,checkpoint,checkpoint['prior_pin'])


def transfer(campaign,index=0):
    cell=campaign['cells'][index];cp=campaign['checkpoints'][index];directory=Path(cell['result_dir'])
    directory.mkdir(parents=True,exist_ok=True)
    eps=[]
    for seed,reward,seconds in zip([101,102,103,104,105],[.04,.03,.10,.12,.14],[.1,.2,.3,.4,.5]):
        rows=[dict(seed=seed,decision=i,reward=reward,control_seconds=seconds if i>=10 else 100.) for i in range(500)]
        (directory/f'decisions-seed-{seed}.jsonl').write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
        eps.append(dict(episode(seed,sum(row['reward'] for row in rows)),control_seconds=sum(row['control_seconds'] for row in rows)))
    manifest=dict(checkpoint_sha256=cp['checkpoint_sha256'],checkpoint_step=cp['step'],horizon=cell['H'],rounds=cell['J'],arm=cell['arm'],
        seeds=campaign['seeds'],controller_seed=55,max_steps=500,semantics={'action':'adapted_actor_mean'},
        source=dict(git_head='b'*40,sha256='f'*64),campaign_sha256='c'*64,resolved={'inner_rounds':4})
    result=dict(complete=True,frozen_outer_verified=True,smoke=False,episodes=eps)
    receipt=dict(historical_reuse=True,source=manifest['source'])
    save(directory/'manifest.json',manifest);save(directory/'results.json',result);save(directory/'receipt.json',receipt)
    return adapter.normalize_transfer(campaign,cell,receipt,result,manifest,prior(campaign,index),receipt_path=directory/'receipt.json')


def test_prior_selects_explicit_seeds_and_preserves_original_identity(campaign):
    record=prior(campaign)
    assert [row['seed'] for row in record['episodes']]==[101,102,103,104,105]
    assert record['metrics']['eval/return_mean']==30
    assert record['metrics']['eval/return_sample_std']==pytest.approx(statistics.stdev([10,20,30,40,50]))
    assert record['metrics']['eval/paired_gain_mean']==0
    assert record['metrics']['runtime/late_control_seconds_per_decision'] is None
    assert record['provenance']['original_environment_seeds']==[101,102,103,104,105]
    assert record['provenance']['original_code']['commit']=='b'*40
    assert record['identity']['science']['comparison_source']['git_head']=='a'*40


@pytest.mark.parametrize('mutation',[
    lambda m:m['checkpoint'].update(sha256='0'*64),
    lambda m:m['checkpoint'].update(source_run='other'),
    lambda m:m['runs'][0]['resolved_config'].update(inner_operator='sac'),
    lambda m:m['runs'][0]['resolved_config'].update(inner_actor_source='other'),
    lambda m:m['runs'][0]['result'].update(outer_updates_after=3),
    lambda m:m['runs'][0]['episodes'][0].update(length=499),
    lambda m:m['runs'][0]['episodes'][0].update(solver_seed=0),
    lambda m:m['runs'][0]['episodes'][0].update(truncated_by_evaluator=True),
    lambda m:m['protocol']['environment']['params'].update(task='other'),
    lambda m:m['protocol'].update(action_rule='sample'),
    lambda m:m['code'].update(dirty=True),
    lambda m:m['code'].update(commit='c'*40),
    lambda m:m['checkpoint']['metadata'].update(unexpected='other recipe'),
])
def test_prior_rejects_protocol_mismatches(campaign,mutation):
    pin=campaign['checkpoints'][0]['prior_pin'];path=Path(pin['bundle'])/'manifest.json'
    manifest=adapter.read(path);mutation(manifest);save(path,manifest);pin['manifest_sha256']=adapter.digest(path)
    with pytest.raises(SeriesError):prior(campaign)


def test_prior_pin_and_trace_are_immutable(campaign):
    pin=campaign['checkpoints'][0]['prior_pin'];path=Path(pin['bundle'])/'traces.jsonl'
    path.write_text('modified')
    with pytest.raises(SeriesError,match='trace hash'):prior(campaign)
    path.unlink()
    with pytest.raises(SeriesError,match='missing'):prior(campaign)


def test_transfer_pairs_before_aggregating_and_excludes_first_ten_timings(campaign):
    record=transfer(campaign)
    assert record['metrics']['eval/return_mean']==pytest.approx((20+15+50+60+70)/5)
    assert record['metrics']['eval/paired_gain_mean']==pytest.approx((10-5+20+20+20)/5)
    assert record['metrics']['eval/paired_gain_sample_std']==pytest.approx(statistics.stdev([10,-5,20,20,20]))
    assert record['metrics']['runtime/late_control_seconds_per_decision']==pytest.approx(.3)
    assert record['metrics']['runtime/late_control_sample_std']==pytest.approx(statistics.stdev([.1,.2,.3,.4,.5]))
    assert record['metrics']['runtime/control_seconds_per_decision']>2
    assert record['metrics']['work/environment_decisions']==2500
    assert record['metrics']['runtime/control_seconds_per_decision']==pytest.approx((1000+490*.3)/500)
    assert record['metrics']['runtime/control_sample_std']==pytest.approx(statistics.stdev([.1,.2,.3,.4,.5])*.98)
    assert record['metrics']['eval/return_min']==pytest.approx(15)
    assert record['provenance']['original_source']['git_head']=='b'*40
    assert record['provenance']['worker_receipt']['historical_reuse'] is True
    assert record['provenance']['timing_decision_range']==[10,499]


@pytest.mark.parametrize('defect',['missing_decision','wrong_seed','sum','negative_time'])
def test_transfer_rejects_invalid_decision_traces(campaign,defect):
    transfer(campaign);cell=campaign['cells'][0];directory=Path(cell['result_dir'])
    path=directory/'decisions-seed-101.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
    if defect=='missing_decision':rows.pop()
    if defect=='wrong_seed':rows[0]['seed']=103
    if defect=='sum':rows[0]['reward']=1
    if defect=='negative_time':rows[0]['control_seconds']=-1
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    with pytest.raises(SeriesError):
        adapter.normalize_transfer(campaign,cell,adapter.read(directory/'receipt.json'),adapter.read(directory/'results.json'),
            adapter.read(directory/'manifest.json'),prior(campaign),receipt_path=directory/'receipt.json')


def test_pending_points_remain_null_and_separate_line_segments(campaign):
    record=transfer(campaign)
    points,progress=adapter.table_rows(campaign,adapter.settings(campaign)[1],{25000:record,75000:record},
        {50000:dict(state='running',seed=102,decision=3,completed_episodes=1)})
    assert [row['state'] for row in points]==['complete','running','complete']
    assert points[1]['return_mean'] is None and points[1]['gain_mean'] is None and points[1]['late_seconds'] is None
    assert points[0]['segment']!=points[2]['segment']
    assert progress[1]['completed_episodes']==1 and progress[1]['decision']==3
    assert points[0]['return_upper']==pytest.approx(record['metrics']['eval/return_mean']+record['metrics']['eval/return_sample_std'])


def prepare(campaign,tmp_path):
    root=tmp_path/'campaign';output=tmp_path/'publication';save(root/'campaign.json',campaign)
    args=SimpleNamespace(root=root,publication_root=output,entity='entity',project='ambi-inner-bench',owner='owner',attempt_label='test')
    return args,publish.prepare(args)


def test_prepare_is_idempotent_and_only_stages_completed_priors(campaign,tmp_path):
    args,state=prepare(campaign,tmp_path);again=publish.prepare(args)
    assert state==again
    assert len(state['runs'])==2 and state['prepared'] is True
    assert len(publish.records_by_step(state['runs']['prior']['run_dir']))==3
    key=campaign['candidates'][0]['setting_id']
    assert publish.records_by_step(state['runs'][key]['run_dir'])=={}
    args.attempt_label='new'
    with pytest.raises(SeriesError,match='binding differs'):publish.prepare(args)


def test_append_keeps_immutable_completed_record(campaign,tmp_path):
    args,state=prepare(campaign,tmp_path);record=transfer(campaign);key=campaign['candidates'][0]['setting_id']
    run=state['runs'][key]['run_dir']
    assert stage_record(run,record)['status']=='staged'
    assert stage_record(run,record)['status']=='already_staged'
    mutated=deepcopy(record);mutated['metrics']['eval/return_mean']+=1
    with pytest.raises(SeriesError,match='different accepted'):stage_record(run,mutated)


def test_collection_surfaces_bad_receipt_and_never_accepts_partial_episode(campaign,tmp_path,monkeypatch):
    args,state=prepare(campaign,tmp_path)
    from slurm import ambi_transfer_curve_campaign as worker
    marker=args.root/'receipts'/'0.json';save(marker,{'bad':True})
    def reject(*args,**kwargs):raise ValueError('Corrupt complete receipt')
    monkeypatch.setattr(worker,'validate_receipt',reject)
    directory=Path(campaign['cells'][1]['result_dir'])
    save(directory/'progress.json',dict(status='running',seed=102,decision=9,completed_episodes=1))
    result=publish.collect(args.root,campaign,state,jobs_active=True)
    key=campaign['candidates'][0]['setting_id'];curve=result['curves'][key]
    assert result['completed']==0 and len(result['failures'])==1
    assert [row['state'] for row in curve['progress']]==['validation_failed','running','pending']
    assert all(row['return_mean'] is None for row in curve['points'])


def test_summary_payload_does_not_create_history_rows(campaign):
    points,progress=adapter.table_rows(campaign,adapter.settings(campaign)[1],{}, {})
    sdk=SimpleNamespace(Table=lambda **kwargs:kwargs)
    result=publish.summary_payload(sdk,dict(points=points,progress=progress,completed=0,total=3))
    assert result[publish.TABLE_KEY]['columns']==adapter.POINT_COLUMNS
    assert len(result[publish.PROGRESS_KEY]['data'])==3
    assert result['transfer_curves/status']=='pending'


def test_five_seed_protocol_and_explicit_colors(campaign):
    assert [row['color'] for row in adapter.settings(campaign)]==['#000000',adapter.COLORS[0]]
    campaign['seeds'].pop()
    with pytest.raises(SeriesError,match='fixed five-seed'):adapter.identity(campaign,adapter.settings(campaign)[0])


def fresh_control(campaign, record):
    fresh=dict(setting_id='h1_j4_fresh',H=1,J=4,arm='fresh',role='fresh',
        label='Fresh SAC · H1 J4',arm_definition={'actor_rho':0.,'critic_rho':0.})
    campaign['candidates'].append(fresh)
    candidate=campaign['candidates'][0]
    candidate.update(role='transfer',fresh_setting_id=fresh['setting_id'])
    reference=deepcopy(record)
    reference['identity']=adapter.identity(campaign,fresh)
    # Deliberately reverse seed order and use differences whose variance is
    # unlike either marginal variance; subtraction must happen before moments.
    differences=[-10,20,5,-4,30]
    for row,delta in zip(reference['episodes'],differences):row['return']-=delta
    reference['episodes'].reverse()
    return fresh,reference,differences


def test_fresh_gain_waits_for_control_then_pairs_by_seed_without_rewriting_records(campaign):
    record=transfer(campaign);fresh,reference,differences=fresh_control(campaign,record)
    setting=adapter.settings(campaign)[1];before=deepcopy(record)
    points,_=adapter.table_rows(campaign,setting,{25000:record},{})
    assert points[0]['return_mean'] is not None
    assert points[0]['fresh_gain_mean'] is None and points[0]['fresh_comparison_state']=='pending'
    points,_=adapter.table_rows(campaign,setting,{25000:record},{},fresh_records={25000:reference})
    assert points[0]['fresh_gain_mean']==pytest.approx(statistics.mean(differences))
    assert points[0]['fresh_gain_std']==pytest.approx(statistics.stdev(differences))
    assert points[0]['fresh_comparison_state']=='complete' and points[0]['fresh_setting']==fresh['setting_id']
    assert record==before
    self_points,_=adapter.table_rows(campaign,fresh,{25000:reference},{},fresh_records={25000:reference})
    assert self_points[0]['fresh_gain_mean']==self_points[0]['fresh_gain_std']==0


@pytest.mark.parametrize('defect',['checkpoint','budget','seed','identity'])
def test_matched_fresh_cannot_mix_checkpoint_budget_seed_or_setting(campaign,defect):
    record=transfer(campaign);fresh,reference,_=fresh_control(campaign,record)
    if defect=='checkpoint':reference['checkpoint']['sha256']='wrong'
    if defect=='budget':reference['identity']['planner']['settings']['inner_rounds']=2
    if defect=='seed':reference['episodes'].pop()
    if defect=='identity':reference['identity']['planner']['setting_id']='different-control'
    with pytest.raises(SeriesError):
        adapter.table_rows(campaign,adapter.settings(campaign)[1],{25000:record},{},fresh_records={25000:reference})


def test_fresh_gain_has_its_own_gaps_when_returns_are_already_complete(campaign):
    record=transfer(campaign);fresh,reference,_=fresh_control(campaign,record)
    records={25000:record,50000:record,75000:record}
    points,_=adapter.table_rows(campaign,adapter.settings(campaign)[1],records,{},
        fresh_records={25000:reference,75000:reference})
    assert points[0]['segment']==points[2]['segment']
    assert points[0]['fresh_segment']!=points[2]['fresh_segment']
    assert points[1]['fresh_gain_mean'] is None


def test_overview_logs_registered_tables_separately_and_resumes_one_persisted_identity(campaign,tmp_path,monkeypatch):
    args,state=prepare(campaign,tmp_path)
    snapshot=publish.collect(args.root,campaign,state,jobs_active=True)
    calls=[];logged=[];finished=[]
    def init(**kwargs):
        persisted=adapter.read(args.publication_root/'publication.json')
        assert persisted['overview']['run_id']==kwargs['id']
        assert kwargs['id'] not in [row['run_id'] for row in state['runs'].values()]
        calls.append(kwargs)
        return SimpleNamespace(log=lambda value:logged.append(value),summary={},finish=lambda **kw:finished.append(kw))
    def forbidden(*args,**kwargs):raise AssertionError('Overview cannot own scientific publication.')
    monkeypatch.setattr(publish,'Publisher',forbidden)
    sdk=SimpleNamespace(init=init,Table=lambda **kwargs:kwargs)
    before={key:adapter.read(Path(entry['run_dir'])/'publication.json') for key,entry in state['runs'].items()}
    publish.publish_overview(campaign,state,snapshot,args.publication_root,sdk)
    assert len(calls)==1 and calls[0]['job_type']=='transfer-checkpoint-overview'
    assert calls[0]['config']['transfer_curve_overview']==state['publication_id']
    assert calls[0]['config']['scientific_run_ids']=={key:value['run_id'] for key,value in state['runs'].items()}
    assert len(logged[0][publish.TABLE_KEY]['data'])==len(logged[0][publish.PROGRESS_KEY]['data'])==6
    assert logged[0]['campaign/completed']==0 and logged[0]['campaign/total']==3
    publish.publish_overview(campaign,state,snapshot,args.publication_root,sdk)
    assert len(calls)==1
    state['layout']=dict(status='verified',url='https://example.test/view')
    publish.publish_overview(campaign,state,snapshot,args.publication_root,sdk)
    assert len(calls)==2 and calls[0]['id']==calls[1]['id'] and all(not entry for entry in finished)
    assert before=={key:adapter.read(Path(entry['run_dir'])/'publication.json') for key,entry in state['runs'].items()}


def test_overview_uncertain_failure_reuses_allocated_run_without_marking_snapshot_published(campaign,tmp_path):
    args,state=prepare(campaign,tmp_path)
    snapshot=publish.collect(args.root,campaign,state,jobs_active=True)
    calls=[]
    def init(**kwargs):
        calls.append(kwargs['id'])
        def fail(value):raise RuntimeError('uncertain logging response')
        return SimpleNamespace(log=fail,summary={},finish=lambda **kw:None)
    sdk=SimpleNamespace(init=init,Table=lambda **kwargs:kwargs)
    for _ in range(2):
        with pytest.raises(RuntimeError,match='uncertain logging'):
            publish.publish_overview(campaign,state,snapshot,args.publication_root,sdk)
        state=adapter.read(args.publication_root/'publication.json')
        assert 'snapshot_sha256' not in state['overview']
    assert calls[0]==calls[1]
