"""Scope, canonical identities and post-run guards for the three-panel campaign."""
from copy import deepcopy
from functools import lru_cache
import json
from types import SimpleNamespace

import pytest

from slurm import ambi_closed_loop_alpha0_lambda1_h4 as campaign
from utils.ambi_benchmark import solver_seed


@lru_cache(None)
def resolved(h, j, index=None):
    from tests.test_ambi_root_local_sac import _build_cfg
    params = (campaign.historical_cell(h,j)['params'] if index is None
              else campaign.cells()[index]['params'])
    return vars(_build_cfg(**params,aux_return_mode='sac',log_std_mapping='direct_clamp',
                           target_entropy=-10.5,sac_actor_loss_scale_mode='none'))


def episodes(steps=500, seeds=campaign.SEEDS):
    return [dict(seed=s,solver_seed=solver_seed(55,'episode',s),length=steps,
                 truncated_by_evaluator=steps!=500,togo_round_summaries=[],
                 **{'return':float(s)}) for s in seeds]


def protocol(mode='mean', steps=500):
    return dict(action_rule=campaign.action_rule(mode),max_steps=steps,controller_seed=55,
                seed_scheme='sha256-v1',environment={'id':'test'})


def test_exact_62_cell_scope_and_ten_maximum_budget_smokes():
    panel=campaign.cells()
    assert len(panel)==62 and len({c['selector'] for c in panel})==62
    assert [(c['experiment_arm'],c['H'],c['estimator'],c['J']) for c in panel]==campaign.identities()
    assert [sum(c['experiment_arm']==arm for c in panel) for arm in ('alpha_zero','lambda_one_mean','h4_mean')]==[36,18,8]
    assert [c['J'] for c in panel[:10]]==[10]*9+[14]
    assert len({(c['experiment_arm'],c['H'],c['estimator']) for c in panel[:10]})==10
    assert all(c['H']<=3 or c['experiment_arm']=='h4_mean' for c in panel)


@pytest.mark.parametrize('index',range(62))
def test_every_cell_resolves_exact_budgets_and_real_canonical_planner(index):
    from utils.eval_series_data import planner_identity
    cell=campaign.cells()[index];cfg=resolved(min(cell['H'],3),cell['J'],index)
    reference=resolved(min(cell['H'],3),cell['J'])
    campaign.matching_config(cfg,reference,cell)
    planner=planner_identity(cfg,{},'AMBITDMPC2',campaign.action_rule(cell['execution_mode']))
    campaign.matching_planner(planner,reference,cell)
    assert cfg['inner_critic_source']==cfg['inner_horizon_critic_source']=='aux_return'
    assert cfg['inner_sac_critic_target']=='reward_only' and cfg['inner_terminal_entropy']=='none'
    assert cfg['inner_actor_source']==cfg['inner_horizon_actor_source']=='sac'
    assert cfg['inner_critic_updates_per_action']==16*cell['J']
    assert cfg['inner_actor_updates_per_action']==4*cell['J']
    assert cfg['inner_temperature_updates_per_action']==(0 if cell['alpha_mode']=='zero' else 4*cell['J'])
    assert cfg['inner_model_step_budget']==128*cell['H']*cell['J']<=cfg['inner_replay_capacity']
    assert cfg['inner_retrace_batch_trajectories']==({1:256,2:128,3:86}[cell['H']] if cell['estimator']=='retrace' else None)
    assert cfg['inner_retrace_lambda']==(cell['retrace_lambda'] if cell['estimator']=='retrace' else 1.)
    if cell['alpha_mode']=='zero':
        assert not cfg['inner_entropy_enabled'] and cfg['inner_temperature_mode']=='inherit_outer'
        assert cell['initial_alpha']==0 and cfg['inner_primary_temperature_updates_per_round']==0
        assert 'inner_temperature_updates_per_action' not in planner['settings']
    else:
        assert cfg['inner_entropy_enabled'] and cfg['inner_temperature_mode']=='auto'
    changed=deepcopy(planner);changed['settings']['inner_critic_source']='sac'
    with pytest.raises(AssertionError):campaign.matching_planner(changed,reference,cell)
    with pytest.raises(AssertionError):campaign.matching_config({**cfg,'inner_actor_lr':.01},reference,cell)


@pytest.mark.parametrize('key,value',[
    ('inner_entropy_enabled',True),('inner_temperature_mode','auto'),
    ('inner_sac_critic_target','entropy_augmented'),('inner_critic_source','sac'),
    ('inner_horizon_critic_source','sac'),('inner_eval_execution_action','mean'),
    ('inner_replay_capacity',3072),('inner_actor_updates_per_round',5),
])
def test_matrix_rejects_undeclared_changes(tmp_path,key,value):
    matrix=campaign.read(campaign.MATRIX)
    name=matrix['evaluation']['default_presets'][0].split('/')[1]
    matrix['comparisons']['sweep']['variants'][name]['alg_params'][key]=value
    path=tmp_path/'matrix.json';path.write_text(json.dumps(matrix))
    with pytest.raises(AssertionError):campaign.cells(path)


def test_h4_capacity_full_retention_and_insufficient_capacity_rejected():
    from tests.test_ambi_root_local_sac import _build_cfg
    h4=[c for c in campaign.cells() if c['H']==4]
    assert {c['J']:c['params']['inner_replay_capacity'] for c in h4}=={
        1:3072,2:3072,4:3072,6:3072,8:4096,10:5120,12:6144,14:7168}
    params={**h4[0]['params'],'inner_replay_capacity':5376,'aux_return_mode':'sac'}
    with pytest.raises(ValueError,match='capacity'):_build_cfg(**params)


def test_prepare_creates_real_canonical_identities_with_exact_references(tmp_path,monkeypatch):
    import evaluate_ambi_checkpoint
    from utils import eval_series
    from utils.eval_series_data import planner_identity
    panel=campaign.cells();refs={}
    keys={(h,j,e,mode) for h in (1,2,3) for j in campaign.ROUNDS for e,mode in
          [('one_step','mean'),('one_step','policy_sample'),('retrace','policy_sample')]}
    keys|={(3,j,'one_step','mean') for j in (12,14)}
    for h,j,e,mode in sorted(keys):
        cfg={**resolved(h,j),**campaign.estimator_settings(e,h),campaign.EXECUTION_KEY:mode}
        pin=dict(H=h,J=j,estimator=e,execution=mode,performance_run_id=f'{h}-{j}-{e}-{mode}')
        refs[(h,j,e,mode)]={**pin,'resolved_config':cfg,'episodes':episodes(),'protocol':protocol(mode),
            'identity':dict(backbone=campaign.SOURCE_RUN,protocol=protocol(mode),
                           planner=planner_identity(cfg,{},'AMBITDMPC2',campaign.action_rule(mode)))}
    references=tmp_path/'refs.json';references.write_text(json.dumps({'references':list(refs.values()),'prior_reference':{}}))
    monkeypatch.setattr(campaign,'source_commit',lambda:'tested')
    monkeypatch.setattr(campaign,'digest',lambda p:campaign.CHECKPOINT_SHA)
    monkeypatch.setattr(campaign,'load_reference',lambda pin,inventory:deepcopy(refs[campaign.reference_key(pin)]))
    monkeypatch.setattr(campaign,'load_prior',lambda pin,inventory:{'episodes':episodes()})
    def evaluate(*args,**kwargs):
        specs=kwargs['eval_series_spec_dir'];specs.mkdir()
        assert kwargs['reference_bundle'] is None and kwargs['seeds']==campaign.SEEDS
        for index,cell in enumerate(panel):
            cfg=resolved(min(cell['H'],3),cell['J'],index)
            identity=dict(backbone=campaign.SOURCE_RUN,protocol=protocol(cell['execution_mode']),
                planner=planner_identity(cfg,{},'AMBITDMPC2',campaign.action_rule(cell['execution_mode'])))
            campaign.write(specs/(cell['selector'].replace('/','__')+'.json'),{'identity':identity})
    monkeypatch.setattr(evaluate_ambi_checkpoint,'evaluate_matrix',evaluate)
    created=[]
    def create(*args):
        created.append(args[2]);return dict(run_id=f'run{len(created)}',run_dir=str(tmp_path/f'run{len(created)}'))
    monkeypatch.setattr(eval_series,'create_run',create)
    state=campaign.prepare(SimpleNamespace(root=tmp_path/'new',matrix=campaign.MATRIX,references=references,
        checkpoint=tmp_path/'checkpoint',inventory=tmp_path/'inventory',registry=tmp_path/'registry',group='test',label='Test'))
    assert len(created)==len(set(created))==62 and len(state['references'])==56
    assert state['smoke_indices']==list(range(10)) and state['publisher_workers']==3
    assert state['h4_overview_run_id']!=state['overview_run_id'] and state['h4_group']=='test-h4-mean'
    for cell in state['cells']:
        ref=cell['paired_reference'];h,j,e,mode=campaign.reference_key(ref)
        assert h==min(cell['H'],3) and j==cell['J']
        assert (e,mode)==((cell['estimator'],'policy_sample') if cell['alpha_mode']=='zero' else ('one_step','mean'))


@pytest.mark.parametrize('index,smoke',[(0,True),(1,False),(6,False),(9,True)])
def test_completion_checks_alpha_execution_and_full_frozen_configuration(tmp_path,monkeypatch,index,smoke):
    cell=deepcopy(campaign.cells()[index]);cfg=resolved(min(cell['H'],3),cell['J'],index)
    steps,seeds=(3,[101]) if smoke else (500,campaign.SEEDS)
    reference=dict(resolved_config=resolved(min(cell['H'],3),cell['J']),
                   protocol=protocol(),runtime={'python':'locked'},episodes=episodes())
    cell.update(validation_reference=reference,paired_reference=reference)
    metric=lambda v:dict(mean=v,min=v,max=v,std=0.)
    sampled=cell['execution_mode']=='policy_sample'
    result=dict(selector=cell['selector'],action_rule=campaign.action_rule(cell['execution_mode']),
        deterministic_execution=not sampled,resolved_device='cuda:0',model_metrics={
            'inner_alpha_initial':metric(cell['initial_alpha']),
            'inner_alpha':metric(cell['initial_alpha']),
            'inner_alpha_final':metric(cell['initial_alpha']),
            'inner_alpha_delta':metric(0),
            'inner_eval_execution_sampled':metric(int(sampled)),
            'inner_eval_execution_mean_action_l2':metric(float(sampled)),
            'inner_sac_compile_fallback':metric(0)})
    manifest=dict(reference=None,code=dict(commit='tested',dirty=False,runtime=reference['runtime']),
        checkpoint=dict(source_run=campaign.SOURCE_RUN),protocol=protocol(cell['execution_mode'],steps),
        runs=[dict(selector=cell['selector'],resolved_config=cfg,result=result,
                   config={'alg_params':cell['params']},episodes=episodes(steps,seeds))])
    def validate(bundle,passed,**kwargs):
        assert kwargs['seeds']==seeds and kwargs['steps']==steps and not kwargs['paired']
        return manifest
    monkeypatch.setattr(campaign,'validate',validate)
    monkeypatch.setattr(campaign,'verify_reference',lambda reference:None)
    monkeypatch.setattr(campaign,'validate_probe_rows',lambda *args,**kwargs:None)
    campaign.validate_completed(tmp_path,cell,{'source_commit':'tested'},smoke=smoke)
    result['model_metrics']['inner_alpha_initial']=metric(.1)
    with pytest.raises(AssertionError):campaign.validate_completed(tmp_path,cell,{'source_commit':'tested'},smoke=smoke)
    result['model_metrics']['inner_alpha_initial']=metric(cell['initial_alpha'])
    if cell['alpha_mode']=='zero':
        result['model_metrics']['inner_alpha_final']=metric(.1)
        with pytest.raises(AssertionError):campaign.validate_completed(tmp_path,cell,{'source_commit':'tested'},smoke=smoke)
        result['model_metrics']['inner_alpha_final']=metric(0)
    result['model_metrics']['inner_sac_compile_fallback']=metric(1)
    with pytest.raises(AssertionError):campaign.validate_completed(tmp_path,cell,{'source_commit':'tested'},smoke=smoke)


@pytest.mark.parametrize('index,smoke',[(0,True),(6,True),(9,True),(61,False)])
def test_worker_uses_short_smoke_or_all_five_full_episodes(tmp_path,monkeypatch,index,smoke):
    import torch
    import evaluate_ambi_checkpoint
    from utils import ambi_seed_shards
    panel=campaign.cells()
    for cell in panel:cell['directory']=str(tmp_path/cell['name'])
    campaign.write(tmp_path/'campaign.json',dict(source_commit='tested',cells=panel,smoke_indices=list(range(10)),
        matrix='matrix',checkpoint='checkpoint',inventory='inventory'))
    monkeypatch.setattr(campaign,'source_commit',lambda:'tested')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'Test GPU')
    def evaluate(*args,**kwargs):
        assert kwargs['seeds']==([101] if smoke else campaign.SEEDS)
        assert kwargs['max_steps']==(3 if smoke else 500) and kwargs['reference_bundle'] is None
        kwargs['bundle_dir'].mkdir();(kwargs['bundle_dir']/'manifest.json').write_text('{}')
    monkeypatch.setattr(evaluate_ambi_checkpoint,'evaluate_matrix',evaluate)
    monkeypatch.setattr(campaign,'validate_completed',lambda *args,**kwargs:{'runs':[{'trace_files':[]}]})
    monkeypatch.setattr(campaign,'training_summary',lambda *args,**kwargs:{'trace_rows_checked':3})
    monkeypatch.setattr(ambi_seed_shards,'seal_episode_bundle',lambda bundle:None)
    receipt=campaign.worker(SimpleNamespace(root=tmp_path,index=index,smoke=smoke))
    assert receipt['status']=='complete' and receipt['smoke']==smoke and receipt['H']==panel[index]['H']


def test_versioned_reference_inventory_contains_exact_56_sources_and_prior():
    pins=campaign.read(campaign.REFERENCES)
    assert len(pins['references'])==len({campaign.reference_key(p) for p in pins['references']})==56
    assert all(p['critic_kind']=='return_only' and p['alpha_mode']=='adaptive' for p in pins['references'])
    assert pins['prior_reference']['bundle'] and len(pins['prior_reference']['manifest_sha256'])==64
