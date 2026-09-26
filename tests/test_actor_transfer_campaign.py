"""Scientific workload and publication contracts for uniform-budget actor transfer."""
from copy import deepcopy
import gzip
import json
from pathlib import Path

import pytest

from slurm import ambi_actor_transfer_campaign as campaign
from slurm import ambi_actor_transfer_publish as publication


def test_complete_horizon_round_mode_panel_and_uniform_solve_capacity():
    panel = campaign.cells()
    assert len(panel) == 36
    assert {(c['H'], c['J'], c['transfer_mode']) for c in panel} == {
        (h, j, mode) for h in (1, 2, 3) for j in (1, 2, 4, 6, 8, 10) for mode in ('cold', 'actor_warm')}
    for cold, warm in zip(panel[::2], panel[1::2]):
        assert not cold['reused'] and not warm['reused']
        a, b = deepcopy(cold['requested_alg_params']), deepcopy(warm['requested_alg_params'])
        assert a.pop('inner_actor_scope') == 'action' and b.pop('inner_actor_scope') == 'episode'
        assert a == b
        assert a['inner_replay_capacity'] == max(3072,128*cold['H']*cold['J'])
        assert all(a[f'inner_{name}_scope'] == 'action' for name in campaign.OTHER_SCOPES)
        assert a['inner_first_action_rounds'] is None
        assert all(campaign.effective_rounds(cold,d)==cold['J'] for d in (0,1,499))


def synthetic_trace(cell, steps=3):
    events = []
    for d in range(steps):
        common = dict(episode_id='seed-101', decision_index=d, metrics={})
        events.append(dict(common, phase='initial', replay_size=0, metrics={
            'inner_rounds':campaign.effective_rounds(cell,d),
            'alpha':campaign.INITIAL_ALPHA,
            'inner_actor_lifetime_updates_initial':4*d*cell['J'] if cell['transfer_mode']=='actor_warm' and d>0 else 0,
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
        metrics = dict(inner_rounds=budget, inner_first_action_rounds_applied=float(bool(cell.get('reused')) and d==0), inner_actor_transferred=float(cell['transfer_mode']=='actor_warm' and d>0),
            inner_critic_optimizer_steps=16*budget, inner_actor_optimizer_steps=4*budget,
            inner_temperature_optimizer_steps=4*budget, inner_model_steps=128*cell['H']*budget, inner_compile_fallback=0)
        events.append(dict(common, phase='decision', metrics={'decision/'+k:v for k,v in metrics.items()}))
    return events


def write_trace(tmp_path, events):
    (tmp_path/'manifest.json').write_text(json.dumps({'runs':[{'trace_files':['trace.jsonl.gz']}]}))
    with gzip.open(tmp_path/'trace.jsonl.gz', 'wt') as handle:
        for event in events: handle.write(json.dumps(event)+'\n')


def test_trace_validation_uses_selected_budget_at_first_and_subsequent_decisions(tmp_path):
    cell = campaign.cells()[1]  # H1/J1 actor warm; J1 at all three decisions.
    write_trace(tmp_path, synthetic_trace(cell))
    result = campaign.summarize_trace(tmp_path, cell, seeds=[101], steps=3)
    assert result['total_rounds'] == 3 and result['decisions'] == 3
    initial = [r for r in result['stage_rows'] if r['stage']=='initial' and r['phase']=='transfer_probe']
    assert {r['decision_group']:r['mean'] for r in initial} == {'first':1., 'steady':2.5}


@pytest.mark.parametrize('corruption', ['missing_first_collection', 'first_override', 'carry_cold', 'replay', 'update_order', 'nonfinite'])
def test_trace_validation_rejects_silent_protocol_changes(tmp_path, corruption):
    cell = campaign.cells()[0]
    events = synthetic_trace(cell)
    if corruption == 'missing_first_collection':
        events = [e for e in events if not (e['decision_index']==0 and e['phase']=='collection')]
    elif corruption == 'first_override':
        next(e for e in events if e['phase']=='decision')['metrics']['decision/inner_first_action_rounds_applied'] = 1.
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
        first_action_rounds=None, cells=panel, overview_run_id='overview')


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
        for period, rounds in [('first',cell['J']), ('steady',cell['J'])]:
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


def j10_equivalent_pair():
    cell = deepcopy(next(c for c in campaign.cells() if c['H']==3 and c['J']==10 and c['transfer_mode']=='actor_warm'))
    cell.update(expected_config=deepcopy(cell['requested_alg_params']), checkpoint_sha256=campaign.CHECKPOINT_SHA,
        metadata_sha256='metadata', initial_alpha=campaign.INITIAL_ALPHA,
        identity={'backbone':'source', 'protocol':{'seeds':campaign.SEEDS},
                  'science':{'files':{'controller.py':'same-implementation'}},
                  'planner':{'type':'sac','settings':{'inner_rounds':10,'inner_actor_scope':'episode'}}})
    source = deepcopy(cell)
    source['requested_alg_params']['inner_first_action_rounds'] = 10
    source['expected_config']['inner_first_action_rounds'] = 10
    source['identity']['planner']['settings']['inner_first_action_rounds'] = 10
    source['identity']['planner']['semantics'] = dict(evaluation_protocol='actor-transfer-v1', first_action_rounds=10)
    return cell, source


def test_j10_reuse_allows_only_redundant_first_budget_and_keeps_original_identity():
    cell, source = j10_equivalent_pair()
    before = deepcopy(source)
    campaign.check_j10_equivalence(cell,source)
    assert source==before and cell['identity']!=source['identity']


@pytest.mark.parametrize('change', ['lower_j','horizon','critic_lr','replay_capacity','checkpoint','implementation','episode_protocol'])
def test_j10_reuse_rejects_other_scientific_differences(change):
    cell,source = j10_equivalent_pair()
    if change=='lower_j':cell['J']=source['J']=8
    elif change=='horizon':source['H']=2
    elif change=='critic_lr':source['expected_config']['inner_critic_lr']=.0004
    elif change=='replay_capacity':source['requested_alg_params']['inner_replay_capacity']=3072
    elif change=='checkpoint':source['checkpoint_sha256']='other-checkpoint'
    elif change=='implementation':source['identity']['science']['files']['controller.py']='changed-implementation'
    else:source['identity']['protocol']['seeds']=[102,103,104,105,106]
    with pytest.raises(AssertionError):campaign.check_j10_equivalence(cell,source)


def reuse_audit_fixture(tmp_path,monkeypatch):
    import utils.eval_series as registry
    import utils.eval_series_data as records
    cell,source = j10_equivalent_pair()
    bundle=tmp_path/'bundle';bundle.mkdir()
    cell.update(reused=True,directory=str(tmp_path),bundle=str(bundle),run_dir='registry',performance_run_id='original-run')
    manifest={'runs':[{'trace_files':['trace.gz']}]}
    (bundle/'manifest.json').write_text(json.dumps(manifest))
    (bundle/'trace.gz').write_bytes(b'original-full-trace')
    summary={'trace_rows_checked':2500}
    (tmp_path/'transfer-diagnostics.json').write_text(json.dumps(summary))
    receipt=dict(status='complete',smoke=False,study_protocol='actor-transfer-v1',J=10,first_action_rounds=10,
        cell=cell['name'],transfer_mode=cell['transfer_mode'],H=3,gpu='NVIDIA L40S',checkpoint_sha256=campaign.CHECKPOINT_SHA,
        checkpoint_step=campaign.CHECKPOINT_STEP,manifest_sha256=campaign.digest(bundle/'manifest.json'),
        trace_sha256={'trace.gz':campaign.digest(bundle/'trace.gz')},trace_rows_checked=2500,
        diagnostics_sha256=campaign.digest(tmp_path/'transfer-diagnostics.json'))
    (tmp_path/'worker-completion.json').write_text(json.dumps(receipt))
    publication=dict(status='complete',cell=cell['name'],performance={'run_id':'original-run'})
    (tmp_path/'publication-completion.json').write_text(json.dumps(publication))
    monkeypatch.setattr(campaign,'_reuse_source',lambda c,p:({},source))
    monkeypatch.setattr(campaign,'validate_completed',lambda b,c,p:manifest)
    calls=[]
    def summarize(b,c,*,seeds,steps):
        calls.append((seeds,steps,c['reused']))
        return summary
    monkeypatch.setattr(campaign,'summarize_trace',summarize)
    monkeypatch.setattr(records,'load_records',lambda *a,**kw:[{'identity':cell['identity'],'provenance':{'missing_artifact_files':[]}}])
    monkeypatch.setattr(registry,'load_run',lambda p:{'identity':cell['identity']})
    return cell,{'inventory':'inventory'},receipt,calls


def test_reuse_audit_requires_full_trace_panel_and_original_publication(tmp_path,monkeypatch):
    cell,current,receipt,calls=reuse_audit_fixture(tmp_path,monkeypatch)
    campaign.validate_reused_j10_source(current,cell)
    assert calls==[(campaign.SEEDS,500,True)]
    publication=json.loads((tmp_path/'publication-completion.json').read_text())
    publication['performance']['run_id']='new-incorrect-identity'
    (tmp_path/'publication-completion.json').write_text(json.dumps(publication))
    with pytest.raises(AssertionError):campaign.validate_reused_j10_source(current,cell)


@pytest.mark.parametrize('failure',['incomplete','smoke','hardware','trace_hash','missing_publication'])
def test_reuse_audit_rejects_incomplete_or_unverified_sources(tmp_path,monkeypatch,failure):
    cell,current,receipt,calls=reuse_audit_fixture(tmp_path,monkeypatch)
    if failure=='incomplete':receipt['status']='incomplete'
    elif failure=='smoke':receipt['smoke']=True
    elif failure=='hardware':receipt['gpu']='NVIDIA A100'
    elif failure=='trace_hash':(Path(cell['bundle'])/'trace.gz').write_bytes(b'modified')
    else:(tmp_path/'publication-completion.json').unlink()
    (tmp_path/'worker-completion.json').write_text(json.dumps(receipt))
    with pytest.raises((AssertionError,FileNotFoundError)):campaign.validate_reused_j10_source(current,cell)


def test_legacy_j10_trace_preserves_first_override_flag_without_relabeling(tmp_path):
    cell=next(c for c in campaign.cells() if c['H']==1 and c['J']==10 and c['transfer_mode']=='actor_warm')
    cell['reused']=True
    events=synthetic_trace(cell)
    write_trace(tmp_path,events)
    result=campaign.summarize_trace(tmp_path,cell,seeds=[101],steps=3)
    assert result['total_rounds']==30
    assert events[-1]['metrics']['decision/inner_first_action_rounds_applied']==0.
    cell['reused']=False
    with pytest.raises(AssertionError):campaign.summarize_trace(tmp_path,cell,seeds=[101],steps=3)
