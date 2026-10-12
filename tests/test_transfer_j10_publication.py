"""Single-cell J10 publication must preserve paired science and visible results."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import statistics

import pytest

from slurm import publish_ambi_transfer_j10 as pub
from utils import wandb_transfer_j10_layout as layout
from utils import transfer_checkpoint_publication as adapter
from utils.ambi_benchmark import solver_seed
from utils.eval_series import SeriesError, _record_fingerprint
from utils.transfer_campaign_diagnostics import FAMILIES


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def candidate(j):
    return dict(setting_id=f'h1_j{j}_bernoulli_a0_c05', J=j, H=1, arm='bernoulli_a0_c05',
        label=f'J{j} critic50%', role='transfer', arm_definition=dict(critic_bernoulli_p=.5, parameter_scope='all'))


def campaign_at(root):
    c = dict(protocol=adapter.PROTOCOL, selection_id='h1-j10-800k-bernoulli-v1',
        source_run='entity/ambi/test', seeds=list(range(101,106)), controller_seed=55, max_steps=500,
        gpu_hardware='L40S', scientific_source={'sha256':'a'*64, 'files':{'science.py':'b'*64}},
        diagnostics=dict(enabled=True, decisions=[0,1,25,100,250,499], stationary_decisions=[25,250]),
        checkpoints=[dict(step=800000, checkpoint_sha256='c'*64, metadata_sha256='d'*64, prior_pin={'pin':'fixed'})],
        candidates=[candidate(10)])
    c['cells'] = [dict(candidate(10), index=0, checkpoint_index=0, step=800000,
        result_dir=str(root/'results'), name='step-800000/'+pub.SETTING)]
    return c


def episode(seed, value):
    metrics={f'{stage}_{key}':.1 for stage in ('prior','initial','final') for key in
        ('critic_rmse','policy_kl','action_gradient_gain','actor_feature_effective_rank_fraction')}
    metrics['final_critic_selection_regret']=.2
    return dict(seed=seed, solver_seed=solver_seed(55,'episode',seed), length=500, steps=500,
        **{'return':value}, truncated=True, terminated=False, truncated_by_evaluator=False, smoke=False,
        control_seconds_per_decision=.8, late_control_seconds_per_decision=.7, diagnostic_seconds=1.,
        diagnostics=dict(complete=True, completed_decisions=[0,1,25,100,250,499], samples=6,
            completed_stationary_decisions=[25,250], families=list(FAMILIES), summary=metrics))


def record(campaign, setting, j):
    values=[100,300,200,500,400] if j=='prior' else [100+j,300+j*2,200+j*3,500+j*4,400+j*5]
    eps=[episode(seed,val) for seed,val in zip(campaign['seeds'], values)]
    params=dict(compile=True,inner_critic_updates_per_round=16,inner_actor_updates_per_round=4,
        inner_rollouts_per_round=128,inner_batch_size=256,inner_rollout_horizon=1)
    if j=='prior':
        for e in eps:
            e.pop('control_seconds_per_decision');e.pop('late_control_seconds_per_decision')
    return adapter._record(campaign, setting, campaign['checkpoints'][0], eps, {'source.json':'/tmp/unused-source.json'},
        dict(worker_receipt=dict(hardware='NVIDIA L40S',frozen_outer_verified=True),
             resolved_config={'algorithm_config':{'alg_params':params}}), '/tmp/unused-source.json',
        {'eval/episodes':5,'eval/frozen_state_unchanged':True,'eval/return_mean':statistics.mean(values),
         'eval/return_sample_std':statistics.stdev(values)})


def registry(root, campaign, setting, j):
    ident=adapter.identity(campaign,setting)
    run_id=f'run{j}';directory=root/'registry'/str(j)/run_id
    data=record(campaign,setting,j)
    save(directory/'run.json',dict(schema_version=1,run_dir=str(directory.resolve()), run_id=run_id,
        identity=ident,identity_sha256=pub.fingerprint(ident),entity=pub.ENTITY,project=pub.PROJECT))
    save(directory/'records'/(data['record_id']+'.json'),data)
    save(directory/'publication.json',dict(records={data['record_id']:dict(checkpoint_step=800000,status='published',
        artifact_sha256={'source.json':'e'*64},record_sha256=_record_fingerprint(data,{'source.json':'e'*64}))}))
    return dict(run_dir=str(directory),run_id=run_id)


@pytest.fixture
def sources(tmp_path):
    campaign=campaign_at(tmp_path/'new');roots=[]
    for j in [4,6,8]:
        root=tmp_path/str(j)/'publication';old=deepcopy(campaign);old['candidates']=[candidate(j)]
        old_path=root.parent/'campaign'/'campaign.json';save(old_path,old)
        runs={candidate(j)['setting_id']:registry(root,old,candidate(j),j)}
        if j==4:runs['prior']=registry(root,old,adapter.settings(old)[0],'prior')
        save(root/'publication.json',dict(prepared=True,entity=pub.ENTITY,project=pub.PROJECT,
            campaign_root=str(old_path.parent),campaign_sha256=adapter.digest(old_path),runs=runs))
        roots.append(root)
    return campaign,roots


def test_reference_import_matches_complete_immutable_record_protocol(sources):
    c,roots=sources;records,pins=pub.load_references(c,roots)
    assert set(records)=={4,6,8,'prior'} and len(pins)==18
    assert all(adapter.digest(p)==sha for p,sha in pins.items())
    assert records[8]['episodes'][0]['seed']==101


@pytest.mark.parametrize('mutation',[
    lambda c:c['seeds'].reverse(), lambda c:c['checkpoints'][0].update(checkpoint_sha256='f'*64),
    lambda c:c['scientific_source'].update(sha256='f'*64),lambda c:c.update(gpu_hardware='A5000'),
    lambda c:c['candidates'][0].update(arm_definition={'critic_bernoulli_p':.75,'parameter_scope':'all'}),
])
def test_reference_import_rejects_incompatible_science(sources,mutation):
    c,roots=sources;mutation(c)
    with pytest.raises(SeriesError):pub.load_references(c,roots)


def test_reference_duplicate_and_tampered_record_rejected(sources):
    c,roots=sources
    with pytest.raises(SeriesError,match='distinct'):pub.load_references(c,[roots[0]]*3)
    directory=next((roots[0]/'registry'/'4').iterdir())
    path=next((directory/'records').iterdir());value=json.loads(path.read_text());value['episodes'][0]['return']+=1;save(path,value)
    with pytest.raises(SeriesError,match='fingerprint'):pub.load_references(c,roots)


def test_unpublished_or_duplicate_checkpoint_cannot_be_reused(sources):
    c,roots=sources;path=next((roots[0]/'registry'/'4').iterdir())/'publication.json'
    journal=json.loads(path.read_text());entry=next(iter(journal['records'].values()));entry['status']='staged';save(path,journal)
    with pytest.raises(SeriesError,match='not published'):pub.load_references(c,roots)
    entry['status']='published';journal['records']['duplicate']=deepcopy(entry);save(path,journal)
    with pytest.raises(SeriesError,match='duplicate/missing'):pub.load_references(c,roots)


@pytest.mark.parametrize('kind',['coverage','metric','timing','wrong_parameter_name'])
def test_missing_metrics_or_dose_rejected_even_with_finite_returns(sources,kind):
    c,roots=sources;records,_=pub.load_references(c,roots);r=records[8]
    if kind=='coverage':r['episodes'][0]['diagnostics']['completed_decisions'].pop()
    if kind=='metric':r['episodes'][0]['diagnostics']['summary'].pop('final_critic_selection_regret')
    if kind=='timing':r['episodes'][0].pop('late_control_seconds_per_decision')
    if kind=='wrong_parameter_name':
        params=r['provenance']['resolved_config']['algorithm_config']['alg_params'];params['inner_rollouts']=params.pop('inner_rollouts_per_round')
    with pytest.raises((SeriesError,RuntimeError)):pub._validate_episodes(r,c)


def test_partial_progress_never_becomes_a_final_return(sources,tmp_path):
    c,roots=sources;refs,_=pub.load_references(c,roots);root=tmp_path/'new'
    save(Path(c['cells'][0]['result_dir'])/'progress.json',dict(cell_id=pub.SETTING,smoke=False,total_episodes=5,
        completed_episodes=4,status='running',seed=105,decision=100))
    s=pub.collect(root,c,refs)
    assert s['completed']==4 and not s['validated_complete']
    assert all(row['J']!=10 for row in s['summary']) and len(s['episodes'])==20
    assert s['progress'][0]['decision']==100


def test_paired_gain_is_seed_aligned_and_sample_sd_not_sem(sources):
    c,roots=sources;refs,_=pub.load_references(c,roots);r=record(c,candidate(10),10);r['episodes'].reverse();refs[10]=r
    s=pub.summarize(refs,dict(J=10,state='complete',completed_episodes=5))
    row=next(x for x in s['pairs'] if x['J']==10 and x['baseline']=='J8')
    assert row['mean']==6 and row['sd']==pytest.approx(statistics.stdev([2,4,6,8,10]))
    assert row['win_rate']==1 and row['upper']==row['mean']+row['sd']
    assert s['validated_complete'] and len(s['episodes'])==25
    prior=next(x for x in s['episodes'] if x['J'] is None)
    assert prior['controller_time_s'] is None


def test_reference_pins_prevent_rebinding_run(sources,tmp_path):
    c,roots=sources;_,pins=pub.load_references(c,roots);out=tmp_path/'out';out.mkdir()
    state=pub.publication_state(tmp_path/'new',out,c,pins)
    assert pub.publication_state(tmp_path/'new',out,c,pins)['run_id']==state['run_id']
    bad=dict(pins);bad[next(iter(bad))]='z'*64
    with pytest.raises(SeriesError,match='identity changed'):pub.publication_state(tmp_path/'new',out,c,bad)


class SummarySubDict:
    def __init__(self,**data):self.data=data
    def get(self,key,default=None):return self.data.get(key,default)


def test_sdk_summary_subdict_acknowledgement(sources):
    c,roots=sources;refs,_=pub.load_references(c,roots);s=pub.summarize(refs,dict(state='pending',completed_episodes=0))
    summary={'transfer_j10/snapshot_sha256':s['snapshot_sha256']}
    for key,(name,columns) in pub.TABLES.items():summary[key]=SummarySubDict(nrows=len(s[name]),ncols=len(columns))
    api=SimpleNamespace(run=lambda path:SimpleNamespace(summary=summary))
    assert pub.acknowledged(api,{'run_id':'abc'},s)
    summary[pub.SUMMARY_KEY].data['nrows']-=1
    assert not pub.acknowledged(api,{'run_id':'abc'},s)


def test_locked_wandb_actual_serialization_has_no_truncation(sources):
    import wandb
    c,roots=sources;refs,_=pub.load_references(c,roots);refs[10]=record(c,candidate(10),10)
    snapshot=pub.summarize(refs,dict(state='complete',completed_episodes=5))
    payload=pub.table_payload(wandb,snapshot)
    for key,(name,columns) in pub.TABLES.items():
        data=payload[key]._to_table_json(warn=False)
        assert len(data['data'])==len(snapshot[name]) and data['columns']==columns
    bad=deepcopy(snapshot);bad['diagnostics']=[snapshot['diagnostics'][0]]*10001
    with pytest.raises(SeriesError,match='truncate'):pub.table_payload(wandb,bad)


class Service:
    def __init__(self,fail=False):
        spec=dict(section=dict(runSets=[],panelBankConfig=dict(sections=[],panelPlacementOverrides={})))
        self.views=[dict(id='personal',name=layout.DEFAULT_VIEW_NAME,type='project-view',spec=json.dumps(spec))]
        self.chart=None;self.writes=0;self.fail=fail
    def execute_graphql(self,query,variables):
        if 'CreateTransferJ10Chart' in query:
            self.chart=dict(type='vega2',spec=variables['spec']);return {'createCustomChart':{'chart':self.chart}}
        if 'TransferJ10Chart' in query:return {'customChart':deepcopy(self.chart)}
        if 'mutation ' in query:
            self.writes+=1
            if not self.fail:self.views.append(dict(id='new',name=variables['name'],type='project-view',spec=variables['spec']))
            return {}
        return {'project':{'allViews':{'edges':[{'node':deepcopy(v)} for v in self.views]}}}


def install(tmp_path,service):
    return layout.ensure_saved_view(SimpleNamespace(_service_api=service),entity='entity',project=pub.PROJECT,
        publication_id='abc123',receipt_dir=tmp_path)


def test_visible_view_is_idempotent_and_preserves_existing_views(tmp_path):
    service=Service();before=deepcopy(service.views)
    first=install(tmp_path,service);second=install(tmp_path,service)
    assert first['status']=='verified' and second['changed'] is False and service.writes==1
    assert service.views[:1]==before and '?nw=transferj10abc123' in first['url']
    chart=layout.chart_definition()
    assert chart['layer'][0]['encoding']['x']['axis']['values']==[4,6,8,10]
    assert chart['encoding']['color']['scale']['domain']==list(layout.METHODS)
    assert any(p['config'].get('mediaKeys')==[pub.PROGRESS_KEY] for section in layout.sections('entity') for p in section['panels'])


def test_user_edited_view_is_preserved_and_install_failure_surfaces(tmp_path):
    service=Service();install(tmp_path,service)
    edited=json.loads(service.views[-1]['spec']);edited['section']['panelBankConfig']['sections'][0]['name']='user name'
    service.views[-1]['spec']=json.dumps(edited)
    with pytest.raises(layout.J10LayoutConflict):install(tmp_path,service)
    assert service.writes==1
    with pytest.raises(layout.ResultsLayoutError):install(tmp_path/'failure',Service(fail=True))
    assert json.loads((tmp_path/'failure'/'results-layout-receipt.json').read_text())['status']=='failed'


def test_ended_worker_does_not_leave_stale_running_dashboard(sources,tmp_path):
    c,roots=sources;refs,_=pub.load_references(c,roots)
    save(Path(c['cells'][0]['result_dir'])/'progress.json',dict(cell_id=pub.SETTING,smoke=False,total_episodes=5,
        completed_episodes=4,status='running',seed=105,decision=100))
    snapshot=pub.collect(tmp_path/'new',c,refs,active=False)
    assert snapshot['state']=='incomplete' and snapshot['progress'][0]['error']
    assert snapshot['completed']==4 and snapshot['progress'][0]['decision']==100


def test_completed_receipt_is_normalized_from_full_traces(sources,tmp_path,monkeypatch):
    from slurm import ambi_transfer_curve_campaign as worker
    c,roots=sources;refs,_=pub.load_references(c,roots);cell=c['cells'][0];directory=Path(cell['result_dir'])
    source=record(c,candidate(10),10);eps=source['episodes']
    for e in eps:
        rows=[dict(seed=e['seed'],decision=i,reward=e['return']/500,control_seconds=1.) for i in range(500)]
        path=directory/f"decisions-seed-{e['seed']}.jsonl";path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('\n'.join(json.dumps(row) for row in rows))
        e['control_seconds']=500.
    result=dict(complete=True,frozen_outer_verified=True,smoke=False,episodes=eps)
    manifest=dict(checkpoint_sha256='c'*64,checkpoint_step=800000,horizon=1,rounds=10,arm='bernoulli_a0_c05',
        seeds=c['seeds'],controller_seed=55,max_steps=500,semantics={'action':'adapted_actor_mean'},source={},
        campaign_sha256='a'*64,resolved=source['provenance']['resolved_config'])
    receipt=source['provenance']['worker_receipt'];save(tmp_path/'new'/'receipts'/'0.json',receipt)
    monkeypatch.setattr(worker,'validate_receipt',lambda *a:receipt)
    monkeypatch.setattr(worker,'validate_result',lambda *a:(result,manifest))
    snapshot=pub.collect(tmp_path/'new',c,refs)
    assert snapshot['state']=='complete' and snapshot['validated_complete'] and snapshot['completed']==5
    assert len(snapshot['episodes'])==25
    sd=next(row for row in snapshot['summary'] if row['J']==10 and row['metric']=='episode_return_sd')
    assert sd['mean']>0 and all(sd[k] is None for k in ('sd','min','max','lower','upper'))
    path=directory/'decisions-seed-101.jsonl';path.write_text(path.read_text().splitlines()[0])
    with pytest.raises(SeriesError,match='coverage'):pub.collect(tmp_path/'new',c,refs)


def test_jobless_once_does_not_query_scheduler(sources,tmp_path,monkeypatch):
    c,roots=sources;root=tmp_path/'new';save(root/'campaign.json',c)
    monkeypatch.setattr(pub,'gpu_jobs_active',lambda *a:pytest.fail('jobless --once queried Slurm'))
    monkeypatch.setattr(pub,'publish_once',lambda *a:dict(url='verified-view'))
    args=SimpleNamespace(root=root,publication_root=tmp_path/'pub',reference_publication_root=roots,
        validate_only=False,gpu_job_id=[],worker_grace_seconds=0,once=True,poll_seconds=1)
    pub.watch(args)
    status=json.loads((tmp_path/'pub'/'publisher-status.json').read_text())
    assert status['status']=='incomplete' and not status['validated_complete']
