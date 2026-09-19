"""Conditioning comparison rejects unrelated changes and preserves seed pairing."""
from copy import deepcopy

import pytest

from slurm.ambi_aux_hj_sweep import cells, polyak_comparison
from slurm.ambi_aux_horizon_campaign import match_identity, baseline_result, BASELINE_SOURCE
from slurm.ambi_horizon_diagnostics_smoke import MATRIX
from tests.test_aux_round_budget import identity


@pytest.mark.parametrize('cell', [c for c in cells(MATRIX) if c['name'].endswith('one_hot')], ids=lambda c:c['name'])
def test_exact_horizon_identity_and_paired_gain(cell):
    candidate=identity(cell,MATRIX)
    baseline=deepcopy(candidate)
    settings=baseline['identity']['planner']['settings']
    assert settings.pop('inner_horizon_conditioning')=='one_hot'
    assert settings.pop('horizon_conditioning_horizon')==cell['H']
    if cell['J']!=1:
        from utils.eval_series_data import scientific_identity
        baseline['identity']['science']=scientific_identity('AMBITDMPC2/AMBITDMPC2',None,BASELINE_SOURCE)
    baseline.update(record_id='control',metrics={'eval/frozen_state_unchanged':True},
        episodes=[dict(seed=s,solver_seed=55,length=500,truncated_by_evaluator=False,
                       **{'return':float(s)}) for s in range(101,106)])
    result=baseline_result(candidate,baseline,historical=cell['J']!=1)
    new=[{**e,'return':e['return']+3} for e in reversed(baseline['episodes'])]
    comparison=polyak_comparison(new,result)
    assert comparison['metrics']['comparison/unconditioned_gain_mean']==3
    assert comparison['metrics']['comparison/unconditioned_gain_ci95_low']==3
    new[0]['solver_seed']=56
    with pytest.raises(AssertionError):polyak_comparison(new,result)
    for key,value in [('inner_critic_updates_per_round',32),('inner_actor_lr',1e-4),
                      ('inner_rollout_horizon',1),('inner_rounds',16),('inner_terminal_entropy','none')]:
        wrong=deepcopy(candidate);wrong['identity']['planner']['settings'][key]=value
        with pytest.raises(AssertionError):baseline_result(wrong,baseline,historical=cell['J']!=1)
    baseline['identity']['science']={'unknown':True}
    with pytest.raises(AssertionError):baseline_result(candidate,baseline,historical=cell['J']!=1)
