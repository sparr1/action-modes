"""Resolved J6/J8 compute identities and full transition retention."""
from copy import deepcopy

import pytest
import torch

from slurm.ambi_aux_hj_sweep import MATRIX, CHECKPOINT_SHA, read, cells
from slurm.ambi_aux_round_budget import match_round_identity, historical_baseline, ORIGINAL_SOURCE
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_component_model
from utils.eval_series_data import planner_identity

PATH = MATRIX.with_name('ambi_aux_j68_625k.json')


def identity(cell, matrix_path=PATH):
    matrix = read(matrix_path)
    params = {k:v for k,v in {**matrix['shared_alg_params'],**cell['params'],
              'aux_return_mode':'sac','log_std_mapping':'direct_clamp'}.items() if v is not None}
    cfg = _build_cfg(**params)
    return dict(identity=dict(backbone={'id':'b'},protocol={'id':'p'},science={'id':'s'},
        planner=planner_identity(vars(cfg),{},'AMBITDMPC2/AMBITDMPC2','tanh_mean')),
        checkpoint=dict(step=625000,sha256=CHECKPOINT_SHA))


@pytest.mark.parametrize('cell',cells(PATH),ids=lambda c:c['name'])
def test_round_grid_preserves_nonbudget_settings(cell):
    matrix = read(PATH);old = read(MATRIX)
    assert matrix['shared_alg_params'] == {**old['shared_alg_params'],'inner_replay_capacity':3072}
    arm = cell['name'].split('_h')[0]
    expected = old['comparisons']['sweep']['variants'][f"{arm}_h{cell['H']}_j4"]['alg_params']
    assert cell['params'] == {**expected,'inner_rollout_horizon':cell['H'],'inner_rounds':cell['J'],
                             'inner_replay_capacity':3072,'inner_component_update_order':'critic_first'}
    baseline = next(c for c in cells(MATRIX) if c['H']==cell['H'] and c['J']==4 and c['name'].startswith(arm))
    match_round_identity(identity(cell),identity(baseline,MATRIX),cell['J'])
    assert 128*cell['H']*cell['J'] <= 3072


def test_exact_grid_and_original_execution():
    panel = cells(PATH)
    assert len(panel)==12
    assert {(c['name'].split('_h')[0],c['H'],c['J']) for c in panel} == {
        (arm,h,j) for arm in ('soft_soft','soft_return') for h in (1,2,3) for j in (6,8)}
    assert [c['J'] for c in panel]==[8]*6+[6]*6
    assert not read(PATH).get('execution')
    assert read(PATH)['evaluation'] == {**read(MATRIX)['evaluation'],
        'default_presets':['sweep/'+c['name'] for c in panel]}


def test_compute_identity_rejects_unintended_changes():
    panel = cells(PATH)
    old = identity(next(c for c in cells(MATRIX) if c['name']=='soft_soft_h3_j4'), MATRIX)
    new = identity(next(c for c in panel if c['name']=='soft_soft_h3_j8_jscale'))
    match_round_identity(new,old,8)
    for key,value in [('inner_replay_capacity',2048),('inner_actor_updates_per_round',8),
                      ('inner_model_step_budget',2048),('inner_component_update_order','interleaved'),
                      ('inner_replay_reset_each_round',True),('inner_critic_target_tau',.1)]:
        wrong = deepcopy(new);wrong['identity']['planner']['settings'][key]=value
        with pytest.raises(AssertionError):match_round_identity(wrong,old,8)


def test_historical_pairing_preserves_episode_seeds():
    from utils.eval_series_data import scientific_identity
    from slurm.ambi_aux_hj_sweep import polyak_comparison
    old = identity(next(c for c in cells(MATRIX) if c['name']=='soft_soft_h3_j4'), MATRIX)
    new = identity(next(c for c in cells(PATH) if c['name']=='soft_soft_h3_j8_jscale'))
    old['identity']['science'] = scientific_identity('AMBITDMPC2/AMBITDMPC2',None,ORIGINAL_SOURCE)
    old.update(record_id='original',metrics={'eval/frozen_state_unchanged':True},
        episodes=[dict(seed=s,solver_seed=55,length=500,truncated_by_evaluator=False,
                       **{'return':float(s)}) for s in range(101,106)])
    new.pop('checkpoint')  # Preparation passes a specification, not a completed record.
    baseline = historical_baseline(new,old,8)
    episodes = [{**e,'return':e['return']+3} for e in reversed(old['episodes'])]
    assert polyak_comparison(episodes,baseline)['metrics']['comparison/j4_gain_mean']==3
    episodes[0]['solver_seed']=56
    with pytest.raises(AssertionError):polyak_comparison(episodes,baseline)
    old['identity']['science']={'unknown_source':True}
    with pytest.raises(AssertionError):historical_baseline(new,old,8)


def test_h3_j8_retains_every_transition_through_last_update(monkeypatch):
    holder = _tiny_component_model(inner_rounds=8,inner_rollouts_per_round=128,
        inner_rollout_horizon=3,inner_replay_capacity=3072,inner_batch_size=256,
        inner_critic_updates_per_round=32,inner_actor_updates_per_round=4,
        inner_finite_horizon=True,inner_component_update_order='critic_first')
    try:
        engine=holder.agent.inner_engine;sample=engine._sample_batch;last_round=[]
        def check_sample(indices=None):
            batch=sample(indices);replay=engine.state.replay
            assert replay.size==replay.next_sample_id
            if replay.size==3072:
                torch.testing.assert_close(replay.sample_id[:3072].sort().values,
                                           torch.arange(3072,dtype=torch.long))
                last_round.append(batch['sample_ids'].clone())
            return batch
        monkeypatch.setattr(engine,'_sample_batch',check_sample)
        engine.reset_for_evaluation(3818519826)
        holder.agent.act(torch.tensor([.7,.3,-.2]),t0=True,eval_mode=True)
        assert len(last_round)==36
        assert any(bool((ids<384).any()) for ids in last_round)
        assert holder.agent.last_inner_metrics['inner_critic_optimizer_steps']==256
        assert holder.agent.last_inner_metrics['inner_actor_optimizer_steps']==32
    finally:
        holder.close()
