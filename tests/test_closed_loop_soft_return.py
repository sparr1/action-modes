"""Mixed soft-inner/return-terminal scope, resolved controls and provenance."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from slurm import ambi_closed_loop_soft_return as campaign
from utils.ambi_benchmark import solver_seed


def episodes():
    return [dict(seed=s, solver_seed=solver_seed(55,'episode',s), length=500,
                 truncated_by_evaluator=False, **{'return':float(s)}) for s in campaign.SEEDS]


@pytest.mark.parametrize('execution,count', [('policy_sample',36),('both',72)])
def test_selected_grid_has_exact_scope_and_boundary_smokes(execution,count):
    panel = campaign.cells(campaign.BOTH_MATRIX if execution=='both' else campaign.MATRIX,execution=execution)
    assert len(panel)==count
    assert [(c['H'],c['estimator'],c['J'],c['execution_mode']) for c in panel] == campaign.identities(execution)
    assert all(c['J']==10 for c in panel[:6*len(campaign.execution_modes(execution))])
    assert len({c['selector'] for c in panel})==count
    for cell in panel:
        expected = campaign.historical_cell(cell['H'],cell['J'])['requested_alg_params']
        expected.update({campaign.EXECUTION_KEY:cell['execution_mode'],
                         **campaign.estimator_settings(cell['estimator'],cell['H']),
                         'inner_critic_source':'sac','inner_sac_critic_target':'entropy_augmented'})
        assert cell['requested_alg_params']==expected
        assert cell['critic_kind']=='soft_return'
        assert expected['inner_horizon_critic_source']=='aux_return' and expected['inner_terminal_entropy']=='none'


@pytest.mark.parametrize('index',range(72))
def test_every_cell_resolves_mixed_routes_without_changing_update_or_replay_budget(index):
    from tests.test_ambi_root_local_sac import _build_cfg
    cell=campaign.cells(campaign.BOTH_MATRIX,execution='both')[index]
    cfg=_build_cfg(**cell['params'],aux_return_mode='sac',log_std_mapping='direct_clamp',
                   target_entropy=-10.5,sac_actor_loss_scale_mode='none')
    assert cfg.inner_critic_source=='sac' and cfg.inner_horizon_critic_source=='aux_return'
    assert cfg.inner_sac_critic_target=='entropy_augmented' and cfg.inner_terminal_entropy=='none'
    assert cfg.inner_actor_source==cfg.inner_horizon_actor_source=='sac'
    assert cfg.inner_eval_execution_action==cell['execution_mode']
    assert cfg.inner_model_step_budget==128*cell['H']*cell['J']
    assert cfg.inner_critic_updates_per_action==16*cell['J']
    assert cfg.inner_actor_updates_per_action==cfg.inner_temperature_updates_per_action==4*cell['J']
    assert cfg.inner_replay_capacity==(3840 if cell['J']==10 else 3072)
    assert cfg.inner_model_step_budget<=cfg.inner_replay_capacity
    assert cfg.inner_retrace_batch_trajectories==({1:256,2:128,3:86}[cell['H']] if cell['estimator']=='retrace' else None)


@pytest.mark.parametrize('key,value',[('inner_critic_source','aux_return'),('inner_horizon_critic_source','sac'),
    ('inner_sac_critic_target','reward_only'),('inner_terminal_entropy','outer'),
    ('inner_retrace_lambda',1.),('inner_replay_capacity',3072),('inner_eval_execution_action','mean')])
def test_matrix_rejects_unrequested_routing_and_control_changes(tmp_path,key,value):
    matrix=campaign.read(campaign.MATRIX)
    name=matrix['evaluation']['default_presets'][1].split('/')[1]
    matrix['comparisons']['sweep']['variants'][name]['alg_params'][key]=value
    path=tmp_path/'changed.json';path.write_text(json.dumps(matrix))
    with pytest.raises(AssertionError):campaign.cells(path)


def test_both_execution_requires_explicit_grid_selection():
    with pytest.raises(AssertionError):campaign.cells(campaign.BOTH_MATRIX)
    with pytest.raises(AssertionError):campaign.cells(campaign.MATRIX,execution='both')


@pytest.mark.parametrize('mode',['mean','policy_sample'])
@pytest.mark.parametrize('estimator',['one_step','retrace'])
def test_historical_comparison_allows_only_declared_changes(mode,estimator):
    cell=next(c for c in campaign.cells(campaign.BOTH_MATRIX,execution='both')
              if c['H']==3 and c['estimator']==estimator and c['execution_mode']==mode)
    mean=campaign.historical_cell(3,10)['params']
    actual={**mean,**campaign.estimator_settings(estimator,3),campaign.EXECUTION_KEY:mode,
            'inner_critic_source':'sac','inner_sac_critic_target':'entropy_augmented'}
    campaign.matching_config(actual,mean,cell)
    for key,value in [('inner_actor_lr',.01),('inner_horizon_critic_source','sac'),('inner_terminal_entropy','outer')]:
        with pytest.raises(AssertionError):campaign.matching_config({**actual,key:value},mean,cell)
    with pytest.raises(AssertionError):
        campaign.matching_config(actual,{**mean,'inner_critic_source':'sac'},cell)


@pytest.mark.parametrize('execution',['policy_sample','both'])
def test_prepare_creates_only_requested_new_identities_and_exact_reward_references(tmp_path,monkeypatch,execution):
    import evaluate_ambi_checkpoint
    from utils import eval_series
    from utils.eval_series_data import planner_identity
    panel=campaign.cells(campaign.BOTH_MATRIX if execution=='both' else campaign.MATRIX,execution=execution)
    pins=[];refs={}
    for h in (1,2,3):
        for j in (1,2,4,6,8,10):
            for estimator,mode in [('one_step','mean'),('one_step','policy_sample'),('retrace','policy_sample')]:
                pin=dict(H=h,J=j,estimator=estimator,execution=mode,performance_run_id=f'{h}-{j}-{estimator}-{mode}')
                pins.append(pin)
                cfg={**campaign.historical_cell(h,j)['params'],**campaign.estimator_settings(estimator,h),
                     campaign.EXECUTION_KEY:mode}
                protocol=dict(action_rule=campaign.action_rule(mode),max_steps=500,controller_seed=55,
                              seed_scheme='sha256-v1',environment={'id':'test'})
                refs[campaign.reference_key(pin)]={**pin,'resolved_config':cfg,'episodes':episodes(),
                    'protocol':protocol,'identity':dict(backbone=campaign.SOURCE_RUN,protocol=protocol,
                      planner=planner_identity(cfg,{},'AMBITDMPC2',protocol['action_rule']))}
    references=tmp_path/'refs.json';references.write_text(json.dumps({'references':pins}))
    monkeypatch.setattr(campaign,'source_commit',lambda:'tested')
    monkeypatch.setattr(campaign,'digest',lambda p:campaign.CHECKPOINT_SHA)
    monkeypatch.setattr(campaign,'load_reference',lambda pin,inventory:deepcopy(refs[campaign.reference_key(pin)]))
    def evaluate(*args,**kwargs):
        specs=kwargs['eval_series_spec_dir'];specs.mkdir()
        assert kwargs['reference_bundle'] is None and kwargs['seeds']==campaign.SEEDS
        for cell in panel:
            identity=deepcopy(refs[(cell['H'],cell['J'],'one_step','mean')]['identity'])
            rule=campaign.action_rule(cell['execution_mode'])
            identity['protocol']['action_rule']=rule
            cfg={**refs[(cell['H'],cell['J'],'one_step','mean')]['resolved_config'],
                campaign.EXECUTION_KEY:cell['execution_mode'],
                **campaign.estimator_settings(cell['estimator'],cell['H']),
                'inner_critic_source':'sac','inner_sac_critic_target':'entropy_augmented'}
            identity['planner']=planner_identity(cfg,{},'AMBITDMPC2',rule)
            assert 'inner_critic_source' not in identity['planner']['settings']
            campaign.write(specs/(cell['selector'].replace('/','__')+'.json'),{'identity':identity})
    monkeypatch.setattr(evaluate_ambi_checkpoint,'evaluate_matrix',evaluate)
    created=[]
    def create(*args):
        created.append(args[2]);return dict(run_id=f'run{len(created)}',run_dir=str(tmp_path/f'run{len(created)}'))
    monkeypatch.setattr(eval_series,'create_run',create)
    state=campaign.prepare(SimpleNamespace(root=tmp_path/'new',matrix=None,references=references,execution=execution,
        checkpoint=tmp_path/'checkpoint',inventory=tmp_path/'inventory',registry=tmp_path/'registry',group='test',label='Test'))
    count=36 if execution=='policy_sample' else 72
    assert len(created)==len(set(created))==count and len(state['references'])==54
    assert state['smoke_indices']==list(range(6 if execution=='policy_sample' else 12))
    assert state['prior_reference'] is None
    for cell in state['cells']:
        missing=cell['estimator']=='retrace' and cell['execution_mode']=='mean'
        assert (cell['reward_reference'] is None)==missing
        if not missing:
            assert campaign.reference_key(cell['reward_reference'])==(cell['H'],cell['J'],cell['estimator'],cell['execution_mode'])


def test_real_planner_identity_omits_default_sac_source_but_keeps_return_terminal():
    from utils.eval_series_data import planner_identity
    cell=campaign.cells()[0]
    mean=campaign.historical_cell(cell['H'],cell['J'])['params']
    before=planner_identity(mean,{},'AMBITDMPC2','tanh_mean')
    after=planner_identity(cell['params'],{},'AMBITDMPC2',campaign.ACTION_RULE)
    assert before['settings']['inner_critic_source']=='aux_return'
    assert 'inner_critic_source' not in after['settings']
    assert after['settings']['inner_horizon_critic_source']=='aux_return'
    campaign.matching_planner(after,before,cell)
    changed=deepcopy(after)
    changed['settings']['inner_critic_source']='aux_return'
    with pytest.raises(AssertionError):campaign.matching_planner(changed,before,cell)
    changed=deepcopy(after)
    changed['settings'].pop('inner_horizon_critic_source')
    with pytest.raises(AssertionError):campaign.matching_planner(changed,before,cell)
