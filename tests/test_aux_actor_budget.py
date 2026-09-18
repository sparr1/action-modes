"""Actor-budget scientific pairing and deterministic-control acceptance."""
import copy
import json
from pathlib import Path

import pytest

from slurm.ambi_aux_actor_budget import actor_budget_baseline, without_timing
from slurm.ambi_aux_hj_sweep import MATRIX, cells, actor_updates, polyak_comparison, CHECKPOINT_SHA


def test_actor_grid_preserves_every_nonbudget_setting():
    old=json.loads(MATRIX.read_text())
    new=json.loads(MATRIX.with_name('ambi_aux_actor_budget_625k.json').read_text())
    panel=cells(MATRIX.with_name('ambi_aux_actor_budget_625k.json'))
    assert len(panel)==12
    assert {(c['name'].split('_h')[0],c['H'],c['J'],actor_updates(c)) for c in panel}=={
        (arm,h,1,a) for arm in ('soft_soft','soft_return') for h in (2,3) for a in (4,8,16)}
    for c in panel:
        name=c['name'].rsplit('_a',1)[0]
        before={**old['shared_alg_params'],**old['comparisons']['sweep']['variants'][name]['alg_params']}
        after={**new['shared_alg_params'],**c['params']}
        before['inner_replay_reset_each_round']=False
        before['inner_actor_updates_per_round']=actor_updates(c)
        assert before==after
    assert new['execution']['mode']=='deterministic-v1'


def test_pairing_rejects_critic_change_and_mismatched_episode_seeds():
    record=dict(identity=dict(backbone={'id':'b'},protocol={'id':'p'},science={'id':'s'},
        planner=dict(settings=dict(inner_rounds=1,inner_actor_updates_per_round=4,
            inner_actor_updates_per_action=4,inner_temperature_updates_per_action=4,
            inner_critic_updates_per_round=32))),checkpoint=dict(sha256=CHECKPOINT_SHA,step=625000),
        metrics={'eval/frozen_state_unchanged':True},record_id='baseline',
        episodes=[dict(seed=s,solver_seed=55,length=500,truncated_by_evaluator=False,**{'return':float(s)})
                  for s in range(101,106)])
    candidate=copy.deepcopy(record)
    for k in ('inner_actor_updates_per_round','inner_actor_updates_per_action','inner_temperature_updates_per_action'):
        candidate['identity']['planner']['settings'][k]=16
    baseline=actor_budget_baseline(candidate,record,16)
    episodes=[{**e,'return':e['return']+3} for e in reversed(record['episodes'])]
    assert polyak_comparison(episodes,baseline)['metrics']['comparison/a4_gain_mean']==3
    candidate['identity']['planner']['settings']['inner_critic_updates_per_round']=64
    with pytest.raises(AssertionError): actor_budget_baseline(candidate,record,16)
    episodes[0]['solver_seed']=56
    with pytest.raises(AssertionError): polyak_comparison(episodes,baseline)


def test_control_comparison_ignores_only_timing():
    a={'metrics':{'actor_loss':1.,'inner_update_seconds':2.},'episode':{'return':3.}}
    b=copy.deepcopy(a);b['metrics']['inner_update_seconds']=9.
    assert without_timing(a)==without_timing(b)
    b['metrics']['actor_loss']+=1e-7
    assert without_timing(a)!=without_timing(b)

