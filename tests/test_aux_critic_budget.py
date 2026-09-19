"""Reduced critic work preserves H/J, objectives, replay, and historical pairing."""
from copy import deepcopy

import pytest

from slurm.ambi_aux_hj_sweep import MATRIX, CHECKPOINT_SHA, cells, read, polyak_comparison
from slurm.ambi_aux_critic_budget import match_critic_identity, historical_critic_baseline, ORIGINAL_SOURCE, EXTENSION_SOURCE
from tests.test_aux_round_budget import identity

PATH = MATRIX.with_name('ambi_aux_soft_critic_budget_625k.json')


def original(cell):
    path = MATRIX.with_name('ambi_aux_j68_625k.json') if cell['J'] in (6,8) else MATRIX
    return next(c for c in cells(path) if c['name'].startswith('soft_soft_')
                and c['H']==cell['H'] and c['J']==cell['J']), path


def test_exact_soft_soft_grid_and_original_execution():
    panel=cells(PATH)
    assert len(panel)==18 and len({c['name'] for c in panel})==18
    assert {(c['H'],c['J'],c['params']['inner_critic_updates_per_round']) for c in panel} == {
        (h,j,c) for h in (1,2,3) for j in (2,4,8) for c in (8,16)}
    assert not read(PATH).get('execution') and read(PATH)['critic_budget_sweep']
    assert all(c['name'].startswith('soft_soft_') for c in panel)


@pytest.mark.parametrize('cell',cells(PATH),ids=lambda c:c['name'])
def test_only_critic_budget_changes(cell):
    old,path=original(cell)
    before={**read(path)['shared_alg_params'],**old['params']}
    after={**read(PATH)['shared_alg_params'],**cell['params']}
    before.update(inner_replay_capacity=3072,inner_component_update_order='critic_first',
                  inner_critic_updates_per_round=cell['params']['inner_critic_updates_per_round'])
    assert before==after
    match_critic_identity(identity(cell,PATH),identity(old,path),cell['params']['inner_critic_updates_per_round'])
    assert 128*cell['H']*cell['J'] <= after['inner_replay_capacity']


def test_rejects_unintended_settings():
    cell=cells(PATH)[0];old,path=original(cell)
    candidate,baseline=identity(cell,PATH),identity(old,path)
    for key,value in [('inner_critic_target_tau',.0199),('inner_actor_lr',1e-4),
                      ('inner_critic_lr',1e-4),('inner_rounds',4),('inner_rollout_horizon',2),
                      ('inner_actor_updates_per_round',8),('inner_replay_reset_each_round',True),
                      ('inner_critic_updates_per_action',256),('inner_terminal_entropy','none')]:
        wrong=deepcopy(candidate);wrong['identity']['planner']['settings'][key]=value
        with pytest.raises(AssertionError):match_critic_identity(wrong,baseline,16)


@pytest.mark.parametrize('j',[2,4,8])
def test_historical_source_and_paired_c32_gain(j):
    from utils.eval_series_data import scientific_identity
    cell=next(c for c in cells(PATH) if c['H']==3 and c['J']==j)
    old,path=original(cell);candidate=identity(cell,PATH);baseline=identity(old,path)
    candidate.pop('checkpoint')  # Preparation supplies a specification.
    source=EXTENSION_SOURCE if j==8 else ORIGINAL_SOURCE
    baseline['identity']['science']=scientific_identity('AMBITDMPC2/AMBITDMPC2',None,source)
    baseline.update(record_id='c32',metrics={'eval/frozen_state_unchanged':True},
        episodes=[dict(seed=s,solver_seed=55,length=500,truncated_by_evaluator=False,
                       **{'return':float(s)}) for s in range(101,106)])
    reference=historical_critic_baseline(candidate,baseline,16)
    assert reference['kind']=='critic_budget' and reference['source_commit']==source
    new=[{**e,'return':e['return']-4} for e in reversed(baseline['episodes'])]
    comparison=polyak_comparison(new,reference)
    assert comparison['metrics']['comparison/c32_gain_mean']==-4
    assert comparison['metrics']['comparison/c32_gain_ci95_low']==-4
    new[0]['solver_seed']=56
    with pytest.raises(AssertionError):polyak_comparison(new,reference)
    baseline['identity']['science']={'unknown':True}
    with pytest.raises(AssertionError):historical_critic_baseline(candidate,baseline,16)
