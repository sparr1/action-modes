"""Seed completeness, paired uncertainty, timing and publication integrity."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from slurm import publish_ambi_critic_bypass as pub
from utils import wandb_critic_bypass_layout as layout


def campaign(seeds=(101, 102), smoke=True):
    return dict(protocol=pub.PROTOCOL, campaign_id='a' * 64, arms=list(layout.ARMS),
        seeds=list(seeds), controller_seed=55, max_steps=4 if smoke else 500, smoke=smoke,
        task_list=[dict(task_id=f'{a}-seed-{s}', arm=a, env_seed=s, controller_seed=55)
                   for a in layout.ARMS for s in seeds])


def record(arm='actor_mean', seed=101, cfg=None):
    cfg = campaign() if cfg is None else cfg
    length = cfg['max_steps']
    values = [float(i + 1) for i in range(length)]
    selection = [v / 10 for v in values]
    choices = {name: dict(learned_q=2., selection_model=dict(mean=3.), validation_model=dict(mean=4.),
                          validation_gain_vs_actor=dict(mean=1.), validation_gain_vs_prior=dict(mean=2.))
               for name in ('actor_mean', 'learned_q', 'model_score', 'prior_mean')}
    return dict(schema_version=1, protocol=pub.PROTOCOL, campaign_id=cfg['campaign_id'],
        task_id=f'{arm}-seed-{seed}', arm=arm, env_seed=seed, controller_seed=55, complete=True,
        episode_return=float(seed * 10 + layout.ARMS.index(arm)), episode_length=length,
        terminated=False, truncated=True, controller_times_s=values, selection_times_s=selection,
        controller_time_mean_s=float(np.mean(values)), controller_time_p95_s=float(np.percentile(values, 95)),
        selection_time_mean_s=float(np.mean(selection)), selection_time_p95_s=float(np.percentile(selection, 95)),
        checks=dict(outer_state_unchanged=True, full_episode=True), diagnostics=[dict(decision=0, choices=choices)])


def write_result(root, result):
    path = root / 'tasks' / result['task_id'] / 'result.json'
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(result))


def fill(root, cfg):
    for task in cfg['task_list']:
        write_result(root, record(task['arm'], task['env_seed'], cfg))


def test_complete_seed_coverage_gates_summaries_and_pairs(tmp_path):
    cfg = campaign(seeds=range(101, 121), smoke=False)
    for arm in layout.ARMS:
        for seed in range(101, 120):
            write_result(tmp_path, record(arm, seed, cfg))
    partial = pub.collect(tmp_path, cfg)
    assert partial['complete'] == 95 and partial['pending'] == 5
    assert not partial['summary'] and not partial['pairs']
    write_result(tmp_path, record('actor_mean', 120, cfg))
    one_arm = pub.collect(tmp_path, cfg)
    assert {r['arm'] for r in one_arm['summary']} == {'actor_mean'}
    assert not one_arm['pairs']
    for arm in layout.ARMS[1:]: write_result(tmp_path, record(arm, 120, cfg))
    final = pub.collect(tmp_path, cfg, workers_active=False)
    assert final['state'] == 'complete' and final['complete'] == final['total'] == 100
    assert len(final['summary']) == 30 and len(final['pairs']) == 20
    assert all(r['n'] == 20 for r in final['summary'] + final['pairs'])


def test_pairing_uses_seed_identity_and_bootstraps_differences(tmp_path):
    cfg = campaign(seeds=(103, 101, 102))
    fill(tmp_path, cfg)
    snap = pub.collect(tmp_path, cfg)
    pair = next(r for r in snap['pairs'] if r['arm'] == 'model_score' and r['baseline'] == 'actor_mean')
    # Across-seed returns vary widely; exactly paired gains do not.
    assert pair['mean'] == pair['ci_low'] == pair['ci_high'] == 2.
    assert pair['sd'] == 0 and pair['win_rate'] == 1 and pair['tie_rate'] == 0
    assert pub.collect(tmp_path, cfg)['snapshot_sha256'] == snap['snapshot_sha256']


def test_pooled_p95_is_not_mean_of_episode_percentiles(tmp_path):
    cfg = campaign()
    fill(tmp_path, cfg)
    second = record(seed=102)
    second['controller_times_s'] = [10., 10., 10., 10.]
    second['controller_time_mean_s'] = second['controller_time_p95_s'] = 10.
    write_result(tmp_path, second)
    snap = pub.collect(tmp_path, cfg)
    p95 = next(r for r in snap['summary'] if r['arm'] == 'actor_mean' and r['metric'] == 'controller_time_p95_s')
    assert p95['mean'] == 10 and p95['ci_low'] is None


@pytest.mark.parametrize('mutation,match', [
    (lambda r: r.update(env_seed=102), 'identity'),
    (lambda r: r.update(campaign_id='b' * 64), 'identity'),
    (lambda r: r.update(complete=False), 'complete'),
    (lambda r: r['checks'].update(outer_state_unchanged=False), 'checks'),
    (lambda r: r.update(episode_return=float('nan')), 'finite'),
    (lambda r: r.update(controller_time_mean_s=100.), 'disagrees'),
    (lambda r: r['controller_times_s'].pop(), 'coverage'),
    (lambda r: r['diagnostics'].append(deepcopy(r['diagnostics'][0])), 'Duplicate'),
])
def test_invalid_results_rejected(mutation, match):
    data = record(); mutation(data)
    with pytest.raises(pub.BypassDataError, match=match):
        pub.normalize_result(data, campaign()['task_list'][0], campaign())


def test_duplicate_unexpected_and_changed_results_rejected(tmp_path):
    cfg = campaign(); result = record(); write_result(tmp_path, result)
    before = pub.collect(tmp_path, cfg)
    result['episode_return'] += 1; write_result(tmp_path, result)
    with pytest.raises(pub.BypassDataError, match='changed'):
        pub.verify_immutable_results(before, pub.collect(tmp_path, cfg))
    path = tmp_path / 'tasks/duplicate/result.json'; path.parent.mkdir(); path.write_text(json.dumps(result))
    with pytest.raises(pub.BypassDataError, match='Unexpected'):
        pub.collect(tmp_path, cfg)
    cfg['task_list'][1] = cfg['task_list'][0]
    with pytest.raises(pub.BypassDataError, match='Duplicate'):
        pub.validate_campaign(cfg)


def test_requested_diagnostics_must_not_silently_disappear():
    cfg = campaign(); cfg['diagnostic_decisions'] = [0, 3]
    data = record()
    with pytest.raises(pub.BypassDataError, match='requested same-state'):
        pub.normalize_result(data, cfg['task_list'][0], cfg)
    data['diagnostics'].append({**deepcopy(data['diagnostics'][0]), 'decision': 3})
    assert len(pub.normalize_result(data, cfg['task_list'][0], cfg)[2]) == 40


def test_missing_failed_episodes_explicit_never_zero(tmp_path):
    cfg = campaign(); snap = pub.collect(tmp_path, cfg)
    assert len(snap['progress']) == snap['pending'] == 10 and not snap['episodes']
    path = tmp_path / 'tasks/actor_mean-seed-101/progress.json'
    path.parent.mkdir(parents=True); path.write_text(json.dumps(dict(status='failed', error='bad check')))
    snap = pub.collect(tmp_path, cfg, workers_active=False)
    assert snap['failed'] == 1 and snap['pending'] == 9 and snap['state'] == 'failed'
    write_result(tmp_path, record())
    with pytest.raises(pub.BypassDataError, match='conflicts'):
        pub.collect(tmp_path, cfg)


def test_historical_prior_only_contributes_return_and_explicit_timing_coverage(tmp_path):
    cfg = campaign(); fill(tmp_path, cfg)
    reference = record('prior')
    reference.update(reused_reference=True, runtime=dict(timing_comparable=False), reference_source='/verified/bundle', reference_sha256='f' * 64,
                     controller_times_s=[], selection_times_s=[], **{k: None for k in pub.TIMING})
    write_result(tmp_path, reference)
    snap = pub.collect(tmp_path, cfg)
    prior = [r for r in snap['summary'] if r['arm'] == 'prior']
    assert next(r for r in prior if r['metric'] == 'episode_return')['n'] == 2
    assert all(r['n'] == 1 and 'historical timing excluded' in r['estimator'] for r in prior if 'time_' in r['metric'])
    reference['controller_time_mean_s'] = 0
    with pytest.raises(pub.BypassDataError, match='unavailable'):
        pub.normalize_result(reference, next(t for t in cfg['task_list'] if t['task_id'] == reference['task_id']), cfg)


def test_snapshot_report_identity_and_acknowledgement(tmp_path):
    root, out = tmp_path / 'science', tmp_path / 'publication'; out.mkdir()
    cfg = campaign(); fill(root, cfg); snap = pub.collect(root, cfg)
    pub.write_report(out, snap)
    assert json.loads((out / 'snapshot.json').read_text()) == snap
    assert '10/10 complete' in (out / 'report.html').read_text()
    state = pub.publication_state(root, out, cfg, 'entity', 'project')
    assert pub.publication_state(root, out, cfg, 'entity', 'project') == state
    changed = deepcopy(cfg); changed['checkpoint'] = 'changed'
    with pytest.raises(pub.BypassDataError, match='identity changed'):
        pub.publication_state(root, out, changed, 'entity', 'project')
    summary = {'critic_bypass/snapshot_sha256': snap['snapshot_sha256']}
    summary.update({key: dict(nrows=len(snap[name]), ncols=len(columns)) for key, (name, columns) in pub.TABLES.items()})
    api = SimpleNamespace(run=lambda _: SimpleNamespace(summary=summary))
    assert pub.acknowledged(api, state, snap)
    summary[pub.PAIRS_KEY]['nrows'] += 1
    assert not pub.acknowledged(api, state, snap)


def test_acknowledgement_accepts_sdk_nested_summary_mapping(tmp_path):
    class SummarySubDictLike:
        """SDK nested summary wrappers expose get without being a dict."""
        def __init__(self, values): self.values = values
        def get(self, key, default=None): return self.values.get(key, default)

    snapshot = pub.collect(tmp_path, campaign())
    values = {'critic_bypass/snapshot_sha256': snapshot['snapshot_sha256']}
    values.update({key: SummarySubDictLike(dict(nrows=len(snapshot[name]), ncols=len(columns)))
                   for key, (name, columns) in pub.TABLES.items()})
    summary = SummarySubDictLike(values)
    api = SimpleNamespace(run=lambda _: SimpleNamespace(summary=summary))
    state = dict(entity='entity', project='project', run_id='run')
    assert pub.acknowledged(api, state, snapshot)
    values[pub.PROGRESS_KEY].values['nrows'] += 1
    assert not pub.acknowledged(api, state, snapshot)
    values[pub.PROGRESS_KEY] = None
    assert not pub.acknowledged(api, state, snapshot)


def test_lost_ack_reconciles_without_duplicate_upload(tmp_path, monkeypatch):
    cfg = campaign(); snap = pub.collect(tmp_path, cfg)
    state = dict(entity='e', project='p', run_id='same', campaign_sha256='abc')
    remote, calls = {}, dict(read=0, init=0, log=0)
    def get_run(_):
        calls['read'] += 1
        return SimpleNamespace(summary={} if calls['read'] <= 2 else remote)
    def log(payload): calls['log'] += 1; remote.update(payload)
    def init(**kwargs):
        calls['init'] += 1
        assert kwargs['id'] == 'same' and kwargs['resume'] == 'allow'
        return SimpleNamespace(log=log, finish=lambda: None)
    wb = SimpleNamespace(Api=lambda **_: SimpleNamespace(run=get_run), init=init,
                         Table=lambda columns, data: dict(nrows=len(data), ncols=len(columns)))
    monkeypatch.setattr(pub, 'ensure_saved_view', lambda *args, **kwargs: dict(status='verified', url='saved-view'))
    with pytest.raises(pub.SnapshotUncertain): pub.publish_once(wb, state, cfg, snap, tmp_path, acknowledgement_seconds=0)
    assert pub.publish_once(wb, state, cfg, snap, tmp_path)['status'] == 'verified'
    assert calls['init'] == calls['log'] == 1


def template():
    return dict(section=dict(runSets=[{'keep': 'existing filters'}], panelBankConfig=dict(sections=[{'__id__': 'untouched'}], panelPlacementOverrides={})))


class Service:
    def __init__(self, lost=False, apply=True):
        self.views = [dict(id='personal', name=layout.DEFAULT_VIEW_NAME, type='project-view', spec=json.dumps(template()), displayName='Personal'),
                      dict(id='other', name='nw-existing-transfer-v', type='project-view', spec=json.dumps(template()), displayName='Transfer')]
        self.chart, self.chart_writes, self.view_writes, self.lost, self.apply = None, 0, 0, lost, apply
    def execute_graphql(self, query, variables):
        if 'CreateCriticBypassChart' in query:
            self.chart_writes += 1
            self.chart = dict(id=variables['name'], type=variables['type'], spec=variables['spec'])
            return {'createCustomChart': {'chart': deepcopy(self.chart)}}
        if 'CriticBypassChart' in query: return {'customChart': deepcopy(self.chart)}
        if 'mutation ' in query:
            self.view_writes += 1
            if self.apply:
                self.views.append(dict(id='new', name=variables['name'], type=variables['type'], spec=variables['spec'], displayName=variables['displayName']))
            if self.lost: raise TimeoutError('response lost')
            return {'upsertView': {'view': {'id': 'new'}}}
        return {'project': {'allViews': {'edges': [{'node': deepcopy(v)} for v in self.views]}}}


def install(service, tmp_path):
    return layout.ensure_saved_view(SimpleNamespace(_service_api=service), entity='entity', project='ambi-inner-bench', publication_id='abc123', receipt_dir=tmp_path)


def test_distinct_visible_colors_idempotent_layout_and_existing_views_preserved(tmp_path):
    service = Service(lost=True); before = deepcopy(service.views)
    result = install(service, tmp_path)
    assert result['url'].endswith('?nw=criticbypassabc123')
    assert result['uncertain_response_reconciled'] and not result['browser_verified']
    assert service.views[:2] == before
    assert not install(service, tmp_path)['changed']
    assert service.chart_writes == service.view_writes == 1
    panels = [p for s in layout.sections('entity') for p in s['panels']]
    assert len([p for p in panels if p['viewType'] == 'Vega2']) == 8
    assert len(set(layout.COLORS)) == len(layout.ARMS) == 5
    assert 'pending' in json.dumps(panels).lower() and 'fail' in json.dumps(panels).lower()
    assert "=== 'complete'" in layout.chart_definition()['transform'][0]['filter']


def test_user_edited_layout_not_overwritten_and_failure_receipt_is_explicit(tmp_path):
    service = Service(); install(service, tmp_path)
    spec = json.loads(service.views[-1]['spec']); spec['section']['runSets'][0]['name'] = 'User edit'
    service.views[-1]['spec'] = json.dumps(spec); before = deepcopy(service.views)
    with pytest.raises(layout.BypassLayoutConflict, match='preserving user edits'): install(service, tmp_path)
    assert service.views == before and service.view_writes == 1
    assert json.loads((tmp_path / 'results-layout-receipt.json').read_text())['status'] == 'failed'
    service = Service(lost=True, apply=False)
    with pytest.raises(layout.ResultsLayoutError, match='readback'): install(service, tmp_path)


def test_publish_layout_conflict_is_not_retried(tmp_path):
    calls = []
    def conflict(): calls.append(1); raise layout.BypassLayoutConflict('owned view edited')
    with pytest.raises(layout.BypassLayoutConflict): pub.publish_with_retry(conflict, tmp_path, delays=(0, 0))
    assert len(calls) == 1
