"""Hold-H campaign cadence, artifact reuse, and truthful compute publication."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import ambi_actor_transfer_campaign as campaign
from slurm import ambi_actor_transfer_publish as publisher
from tests.test_actor_transfer_campaign import synthetic_trace, write_trace

MATRIX=campaign.ROOT/'configs/research/ambi_actor_transfer_hold_h_575k.json'


def test_hold_h_grid_differs_only_in_cadence_and_actor_reset():
    held=campaign.cells(MATRIX);original=campaign.cells()
    assert len(held)==36
    for cell,prior in zip(held,original):
        params=deepcopy(cell['requested_alg_params'])
        assert params.pop('inner_solve_interval')==cell['H']==cell['solve_interval']
        assert params==prior['requested_alg_params']
        assert cell['solves_per_episode']=={1:500,2:250,3:167}[cell['H']]
        assert cell['held_decisions_per_episode']==500-cell['solves_per_episode']
        assert params['inner_first_action_rounds'] is None
    for cold,warm in zip(held[::2],held[1::2]):
        c,w=deepcopy(cold['requested_alg_params']),deepcopy(warm['requested_alg_params'])
        assert c.pop('inner_actor_scope')=='action' and w.pop('inner_actor_scope')=='episode'
        assert c==w


def held_trace(cell,steps):
    interval=cell['solve_interval']
    legacy={k:v for k,v in cell.items() if k!='solve_interval'}
    source=synthetic_trace(legacy,steps=campaign.solve_count(cell,steps))
    events=[]
    for d in range(steps):
        solved=d%interval==0
        flags=dict(inner_solve_performed=float(solved),inner_policy_held=float(not solved),
                   inner_solve_index=d//interval,inner_action_age=d%interval,
                   inner_solve_interval=interval,inner_episode_decision_index=d)
        if solved:
            rows=deepcopy([r for r in source if r['decision_index']==d//interval])
            for r in rows:
                r['decision_index']=d
                if r['phase']=='decision':r['metrics'].update({'decision/'+k:v for k,v in flags.items()})
            events.extend(rows)
        else:
            metrics=dict(inner_rounds=0,inner_first_action_rounds_applied=0,inner_actor_transferred=0,
                inner_critic_optimizer_steps=0,inner_actor_optimizer_steps=0,inner_temperature_optimizer_steps=0,
                inner_model_steps=0,inner_compile_fallback=0,**flags)
            events.append(dict(episode_id='seed-101',decision_index=d,phase='decision',
                               metrics={'decision/'+k:v for k,v in metrics.items()}))
    return events


@pytest.mark.parametrize('h,solves,held',[(1,500,0),(2,250,250),(3,167,333)])
def test_actual_full_episode_trace_counts_and_partial_h3_block(tmp_path,h,solves,held):
    cell=next(c for c in campaign.cells(MATRIX) if c['H']==h and c['J']==1 and c['transfer_mode']=='actor_warm')
    rows=held_trace(cell,500);write_trace(tmp_path,rows)
    summary=campaign.summarize_trace(tmp_path,cell,seeds=[101],steps=500)
    assert summary['decisions']==500 and summary['solve_decisions']==solves and summary['held_decisions']==held
    assert summary['total_rounds']==solves
    if h==3:
        decisions=[r for r in rows if r['phase']=='decision']
        assert decisions[-2]['metrics']['decision/inner_solve_performed']==1
        assert decisions[-1]['metrics']['decision/inner_action_age']==1


@pytest.mark.parametrize('failure',['held_probe','held_updates','bad_age','missing_decision','incorrect_lifetime','stale_togo'])
def test_hold_trace_rejects_silent_extra_work_or_wrong_persistence(tmp_path,failure):
    cell=next(c for c in campaign.cells(MATRIX) if c['H']==3 and c['J']==2 and c['transfer_mode']=='actor_warm')
    rows=held_trace(cell,7)
    held=next(r for r in rows if r['decision_index']==1)
    if failure=='held_probe':rows.append(dict(held,phase='probe',round_index=0))
    elif failure=='held_updates':held['metrics']['decision/inner_actor_optimizer_steps']=4
    elif failure=='bad_age':held['metrics']['decision/inner_action_age']=0
    elif failure=='missing_decision':rows.remove(held)
    elif failure=='stale_togo':held['metrics']['decision/inner_togo_return_mean']=2.
    else:next(r for r in rows if r['phase']=='initial' and r['decision_index']==3)['metrics']['inner_actor_lifetime_updates_initial']=24
    write_trace(tmp_path,rows)
    with pytest.raises(AssertionError):campaign.summarize_trace(tmp_path,cell,seeds=[101],steps=7)


def compatible_h1_pair():
    new=deepcopy(campaign.cells(MATRIX)[0]);old=deepcopy(campaign.cells()[0])
    for cell,science in ((new,{'source_sha256':'candidate'}),(old,{'source_sha256':'baseline'})):
        cell.update(expected_config=deepcopy(cell['requested_alg_params']),metadata_sha256='metadata',
                    checkpoint_sha256=campaign.CHECKPOINT_SHA,initial_alpha=.01,
                    identity=dict(science=science,backbone='source',protocol={'seeds':campaign.SEEDS},
                                  planner={'type':'sac','settings':{'inner_rounds':1}}))
    return new,old


def test_h1_config_equivalence_keeps_scientific_revision_for_separate_audit():
    new,old=compatible_h1_pair();campaign.check_h1_equivalence(new,old)
    assert new['identity']['science']!=old['identity']['science']
    for field,key,value in [('expected_config','inner_actor_lr',.01),('requested_alg_params','inner_solve_interval',2)]:
        changed=deepcopy(old);changed[field][key]=value
        with pytest.raises(AssertionError):campaign.check_h1_equivalence(new,changed)
    changed=deepcopy(new);changed['H']=changed['solve_interval']=2
    with pytest.raises(AssertionError):campaign.check_h1_equivalence(changed,old)


def test_h1_audit_pins_both_revisions_rng_and_complete_grid(tmp_path):
    baseline={'source_sha256':'old'};candidate={'source_sha256':'new'}
    proof=dict(schema_version=1,kind='actor-transfer-h1-cadence-compatibility',status='verified',
        baseline_science=baseline,candidate_science=candidate,solve_interval=1,rounds=list(campaign.ROUNDS),
        modes=list(campaign.MODES),episodes_per_case=2,decisions_per_episode=3,
        compared=['actions','actor','critic','optimizer','scalar_state','rng'],differences=[])
    path=tmp_path/'audit.json';path.write_text(json.dumps(proof))
    campaign.validate_h1_compatibility(path,baseline,candidate)
    for key,value in [('candidate_science',baseline),('differences',['rng']),('rounds',[1]),('compared',['actions'])]:
        path.write_text(json.dumps({**proof,key:value}))
        with pytest.raises(AssertionError):campaign.validate_h1_compatibility(path,baseline,candidate)


def test_hold_h_overview_uses_solves_and_amortized_real_decision_time():
    panel=campaign.cells(MATRIX)
    for index,cell in enumerate(panel):cell['performance_run_id']=f'run-{index}'
    record=dict(matrix=str(MATRIX),study_protocol=campaign.HOLD_PROTOCOL,first_action_rounds=None,
                checkpoint_step=campaign.CHECKPOINT_STEP,checkpoint_sha256=campaign.CHECKPOINT_SHA,
                cells=panel,overview_run_id='overview')
    aggregate=publisher.aggregate_results(record,{})
    h3=next(p for p in aggregate['points'] if p['H']==3)
    assert h3['solves_per_episode']==167 and h3['held_decisions_per_episode']==333
    h3.update(return_mean=100.,control_seconds_per_decision=.1)
    fake=SimpleNamespace(Table=lambda **kw:kw,plot=SimpleNamespace(line_series=lambda **kw:kw))
    payload=publisher.overview_payload(fake,aggregate)
    assert payload['comparison/h3_return_vs_rounds']['xname']=='J rounds / solve'
    assert 'all 500' in payload['comparison/h3_return_vs_compute']['xname']
    assert 'solve_interval' in payload['comparison/returns_and_compute']['columns']
