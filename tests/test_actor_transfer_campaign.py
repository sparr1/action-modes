"""Scientific workload and publication contracts for variable-budget actor transfer."""
from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest

from slurm import ambi_actor_transfer_campaign as campaign
from slurm import ambi_actor_transfer_publish as publication


def test_complete_horizon_round_mode_panel_and_first_solve_capacity():
    panel = campaign.cells()
    assert len(panel) == 36
    assert {(c['H'], c['J'], c['transfer_mode']) for c in panel} == {
        (h, j, mode) for h in (1, 2, 3) for j in (1, 2, 4, 6, 8, 10) for mode in ('cold', 'actor_warm')}
    for cold, warm in zip(panel[::2], panel[1::2]):
        assert not cold['reused'] and not warm['reused']
        a, b = deepcopy(cold['requested_alg_params']), deepcopy(warm['requested_alg_params'])
        assert a.pop('inner_actor_scope') == 'action' and b.pop('inner_actor_scope') == 'episode'
        assert a == b
        assert a['inner_replay_capacity'] >= 128*cold['H']*10
        assert all(a[f'inner_{name}_scope'] == 'action' for name in campaign.OTHER_SCOPES)
        assert a['inner_first_action_rounds'] == 10


def synthetic_trace(cell, steps=3):
    events = []
    for d in range(steps):
        common = dict(episode_id='seed-101', decision_index=d, metrics={})
        events.append(dict(common, phase='initial', replay_size=0, metrics={
            'inner_rounds':campaign.effective_rounds(cell,d),
            'alpha':campaign.INITIAL_ALPHA,
            'inner_actor_lifetime_updates_initial':4*(10+(d-1)*cell['J']) if cell['transfer_mode']=='actor_warm' and d>0 else 0,
            'inner_actor_transferred':float(cell['transfer_mode']=='actor_warm' and d>0),
            **{name+'_optimizer_steps_initial':0 for name in ('actor','critic','temperature')}}))
        events.append(dict(common, phase='transfer_probe', stage='initial', round_index=0,
                           metrics={'actor_std_mean': 1.+d}))
        events.append(dict(common, phase='probe', stage='initial', round_index=0, metrics={'togo': float(d)}))
        budget = campaign.effective_rounds(cell, d)
        for r in range(1, budget+1):
            events.append(dict(common, phase='collection', round_index=r, replay_size=r*128*cell['H']))
            for c in range((r-1)*16+1, r*16+1):
                events.append(dict(common, phase='update', round_index=r, updated_critic=True,
                    updated_actor=False, updated_temperature=False, critic_updates=c, actor_updates=(r-1)*4))
            if r == 1:
                events.append(dict(common, phase='transfer_probe', stage='before_first_actor_block',
                                   round_index=r, metrics={'actor_std_mean': 2.+d}))
                events.append(dict(common, phase='probe', stage='before_first_actor_block', round_index=r, metrics={'togo':float(r+d)}))
            for a in range((r-1)*4+1, r*4+1):
                events.append(dict(common, phase='update', round_index=r, updated_critic=False,
                    updated_actor=True, updated_temperature=True, critic_updates=r*16, actor_updates=a))
            if r == 1:
                events.append(dict(common, phase='transfer_probe', stage='after_first_actor_block',
                                   round_index=r, metrics={'actor_std_mean': 3.+d}))
                events.append(dict(common, phase='probe', stage='after_first_actor_block', round_index=r, metrics={'togo':float(r+d)}))
            events.append(dict(common, phase='transfer_probe', stage='post_round', round_index=r, metrics={'actor_std_mean':4.+d}))
            events.append(dict(common, phase='probe', stage='post_round', round_index=r, metrics={'togo': float(r+d)}))
        metrics = dict(inner_rounds=budget, inner_first_action_rounds_applied=float(d==0), inner_actor_transferred=float(cell['transfer_mode']=='actor_warm' and d>0),
            inner_critic_optimizer_steps=16*budget, inner_actor_optimizer_steps=4*budget,
            inner_temperature_optimizer_steps=4*budget, inner_model_steps=128*cell['H']*budget, inner_compile_fallback=0)
        events.append(dict(common, phase='decision', metrics={'decision/'+k:v for k,v in metrics.items()}))
    return events


def write_trace(tmp_path, events):
    (tmp_path/'manifest.json').write_text(json.dumps({'runs':[{'trace_files':['trace.jsonl.gz']}]}))
    with gzip.open(tmp_path/'trace.jsonl.gz', 'wt') as handle:
        for event in events: handle.write(json.dumps(event)+'\n')


def test_trace_validation_uses_actual_first_and_subsequent_budgets(tmp_path):
    cell = campaign.cells()[1]  # H1/J1 actor warm; first J10 then J1 twice.
    write_trace(tmp_path, synthetic_trace(cell))
    result = campaign.summarize_trace(tmp_path, cell, seeds=[101], steps=3)
    assert result['total_rounds'] == 12 and result['decisions'] == 3
    initial = [r for r in result['stage_rows'] if r['stage']=='initial' and r['phase']=='transfer_probe']
    assert {r['decision_group']:r['mean'] for r in initial} == {'first':1., 'steady':2.5}


@pytest.mark.parametrize('corruption', ['short_first', 'carry_cold', 'replay', 'update_order', 'nonfinite'])
def test_trace_validation_rejects_silent_protocol_changes(tmp_path, corruption):
    cell = campaign.cells()[0]
    events = synthetic_trace(cell)
    if corruption == 'short_first':
        events = [e for e in events if not (e['decision_index']==0 and e.get('round_index',0)>1)]
    elif corruption == 'carry_cold':
        next(e for e in events if e['phase']=='decision')['metrics']['decision/inner_actor_transferred'] = 1.
    elif corruption == 'replay':
        next(e for e in events if e['phase']=='initial')['replay_size'] = 128
    elif corruption == 'update_order':
        next(e for e in events if e['phase']=='update')['actor_updates'] = 1
    else:
        next(e for e in events if e['phase']=='transfer_probe')['metrics']['actor_std_mean'] = float('nan')
    write_trace(tmp_path, events)
    with pytest.raises(AssertionError): campaign.summarize_trace(tmp_path, cell, seeds=[101], steps=3)


def fake_campaign():
    panel = campaign.cells()
    for i,c in enumerate(panel): c['performance_run_id'] = f'run-{i}'
    return dict(matrix=str(campaign.MATRIX), study_protocol=campaign.PROTOCOL,
        checkpoint_step=campaign.CHECKPOINT_STEP, checkpoint_sha256=campaign.CHECKPOINT_SHA,
        first_action_rounds=10, cells=panel, overview_run_id='overview')


def result(shift):
    return dict(episodes=[dict(seed=s, solver_seed=s+1000, length=500, truncated_by_evaluator=False,
                **{'return':float(s+shift)}) for s in campaign.SEEDS],
                metrics={'runtime/control_seconds':5.,'runtime/control_seconds_per_decision':.002,
                         'runtime/steady_latency_p95_seconds':.003}, diagnostics={'stage_rows':[]})


def test_pairing_only_complete_cells_and_preserves_runtime_names():
    source = fake_campaign()
    complete = {'cold_h1_j1_c16':result(0),'actor_warm_h1_j1_c16':result(4)}
    aggregate = publication.aggregate_results(source, complete)
    assert aggregate['completed']==2 and len(aggregate['points'])==36
    pair, = aggregate['paired_comparisons']
    assert pair['warm_minus_cold_mean']==pair['ci95_low']==pair['ci95_high']==4.
    assert pair['paired_episodes']==5
    assert aggregate['points'][0]['runtime/steady_latency_p95_seconds']==.003
    assert all(p['return_mean'] is None for p in aggregate['points'][2:])
    complete['actor_warm_h1_j1_c16']['episodes'][0]['solver_seed'] += 1
    with pytest.raises(ValueError): publication.aggregate_results(source, complete)


def test_publication_rejects_duplicate_ids_or_scientific_scope():
    source = fake_campaign()
    source['cells'][1]['performance_run_id'] = source['cells'][0]['performance_run_id']
    with pytest.raises(AssertionError): publication.validate_scope(source)
    source = fake_campaign(); source['cells'][0]['params']['inner_first_action_rounds'] = 1
    with pytest.raises(AssertionError): publication.validate_scope(source)


def test_overview_splits_complete_diagnostics_without_table_truncation():
    from types import SimpleNamespace
    class Table:
        def __init__(self, *, columns, data):
            self.columns, self.data = columns, data
            assert len(data) <= 10000
    fake = SimpleNamespace(Table=Table, plot=SimpleNamespace(line_series=lambda **kwargs:kwargs))
    source = fake_campaign()
    aggregate = publication.aggregate_results(source, {})
    for cell in source['cells']:
        for period, rounds in [('first',10), ('steady',cell['J'])]:
            for stage, r in [('initial',0),('before_first_actor_block',1),('after_first_actor_block',1),
                             *[('post_round',r) for r in range(1,rounds+1)]]:
                for phase in ('probe','transfer_probe'):
                    for metric in range(80):
                        aggregate['diagnostics'].append(dict(setting=cell['name'],H=cell['H'],J=cell['J'],
                            transfer_mode=cell['transfer_mode'],phase=phase,stage=stage,round_index=r,
                            decision_group=period,metric=f'metric_{metric}',mean=0.,std=0.,episodes=5))
    payload = publication.overview_payload(fake, aggregate)
    tables = [v for k,v in payload.items() if k.startswith('diagnostics/')]
    assert len(tables)==24
    assert sum(len(t.data) for t in tables)==len(aggregate['diagnostics'])
