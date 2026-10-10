"""Compact audit publication must preserve identity and surface unavailable data."""
from copy import deepcopy
import errno
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slurm import publish_ambi_action_audit as pub
from utils import wandb_action_audit_layout as layout


def campaign():
    return dict(histories=['prior'], seeds=[101], decisions=[25, 100],
                checkpoint={'step': 800000, 'sha256': 'abc'}, protocol='action-audit-test')


def record(decision=25):
    return dict(identity=pub.identity('prior', 101, decision), checks={'outer_frozen': True, 'restore': {'passed': True}},
        selection_scores={'actions': [], 'critics': [dict(name='fresh', horizon=1, rmse=.4, actions=128)]},
        heldout_scores={'baseline_label': 'prior_mean', 'actions': [dict(label=label, model={f'h{h}': {'mean': 10. * h + i} for h in (1, 3)}) for i, label in enumerate(layout.CANDIDATES)],
                        'critics': [dict(name='fresh', horizon=1, rmse=.2, description='not a scalar')]},
        real_scores={'baseline_label': 'prior_mean', 'actions': [dict(label=label, horizon=h, complete=True, model_mean=100. * h + i,
            real_prefix_value_mean=9. * h + i * 2, real_tail_mean=8. * h + i * 3,
            model_gain_vs_baseline_mean=i, model_gain_vs_baseline_se=.01,
            real_prefix_value_gain_vs_baseline_mean=i * 2, real_prefix_value_gain_vs_baseline_se=.02,
            real_tail_gain_vs_baseline_mean=i * 3, real_tail_gain_vs_baseline_se=.03,
            model_prefix_bias_mean=1., terminal_bias_mean=2., model_prefix_bias_se=.1, terminal_bias_se=.2)
            for h in (1, 3) for i, label in enumerate(layout.CANDIDATES)]})


def write_root(root, data):
    path = root / 'tasks/prior-seed-101' / f'root-{data["identity"]["decision"]}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_normalization_keeps_model_banks_separate_and_uses_exact_paired_gains_and_se():
    values, critics, checks, state = pub.normalize_root(record(), pub.identity('prior', 101, 25))
    assert state == 'complete' and len(values) == 20
    assert values[1]['heldout_model_value'] == 11 and values[1]['heldout_model_gain'] == 1
    assert values[1]['model_value'] == 101 and values[1]['model_gain'] == 1
    assert values[1]['prefix_gain'] == 2 and values[1]['tail_gain'] == 3
    assert values[1]['model_prefix_bias_se'] == .1
    assert values[1]['model_gain_se'] == .01 and values[1]['prefix_gain_se'] == .02 and values[1]['tail_gain_se'] == .03
    assert len(critics) == 2
    assert [(r['bank_scope'], r['rmse']) for r in critics] == [('selection', .4), ('heldout', .2)]
    assert critics[0]['actions'] == 128 and critics[1]['actions'] is None
    assert all('description' not in r for r in critics)
    assert all(c['state'] == 'pass' for c in checks)


@pytest.mark.parametrize('mutation,match', [
    (lambda r: r['identity'].update(seed=102), 'identity'),
    (lambda r: r.update(checks={}), 'checks'),
    (lambda r: r['real_scores']['actions'].pop(), 'all ten'),
    (lambda r: r['real_scores']['actions'].append(deepcopy(r['real_scores']['actions'][0])), 'Duplicate'),
    (lambda r: r['heldout_scores']['actions'][1]['model']['h1'].update(mean=float('nan')), 'finite'),
])
def test_bad_scientific_rows_fail_fast(mutation, match):
    data = record(); mutation(data)
    with pytest.raises(pub.AuditDataError, match=match):
        pub.normalize_root(data, pub.identity('prior', 101, 25))


def test_pending_failed_and_complete_roots_are_explicit_and_missing_is_never_zero(tmp_path):
    snap = pub.collect(tmp_path, campaign())
    assert (snap['complete'], snap['pending'], snap['failed']) == (0, 2, 0)
    assert not snap['measurements']
    write_root(tmp_path, record())
    snap = pub.collect(tmp_path, campaign())
    assert (snap['complete'], snap['pending']) == (1, 1)
    bad = record(100); bad['checks']['restore']['passed'] = False
    write_root(tmp_path, bad)
    snap = pub.collect(tmp_path, campaign(), workers_active=False)
    assert (snap['complete'], snap['failed'], snap['state']) == (1, 1, 'failed')
    assert all(r['state'] == 'failed' for r in snap['measurements'] if r['decision'] == 100)
    assert snap['snapshot_sha256'] == pub.collect(tmp_path, campaign(), workers_active=False)['snapshot_sha256']


def test_snapshot_all_complete_and_local_report_retains_exact_rows(tmp_path):
    root, out = tmp_path / 'science', tmp_path / 'publication'; out.mkdir()
    for d in (25, 100): write_root(root, record(d))
    snap = pub.collect(root, campaign(), workers_active=False)
    assert snap['state'] == 'complete' and snap['complete'] == snap['total'] == 2
    pub.write_report(out, snap)
    assert json.loads((out / 'snapshot.json').read_text()) == snap
    assert '2/2 complete' in (out / 'report.html').read_text()
    assert 'broad_best_h1' in (out / 'report.html').read_text()


def test_publication_resume_keeps_id_and_rejects_changed_source(tmp_path):
    root, out = tmp_path / 'science', tmp_path / 'publication'
    first = pub.publication_state(root, out, campaign(), 'entity', 'project')
    assert pub.publication_state(root, out, campaign(), 'entity', 'project') == first
    changed = campaign(); changed['seeds'].append(102)
    with pytest.raises(pub.AuditDataError, match='identity changed'):
        pub.publication_state(root, out, changed, 'entity', 'project')


@pytest.mark.parametrize('number', [errno.ESTALE, errno.EAGAIN, errno.ETIMEDOUT])
def test_filesystem_retry_reopens_file_without_retrying_parser_errors(tmp_path, monkeypatch, number):
    p = tmp_path / 'record.json'; p.write_text('{"ok":true}')
    original, calls = Path.read_text, []
    def flaky(path, *args, **kwargs):
        calls.append(path)
        if len(calls) < 3: raise OSError(number, 'transient')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', flaky); monkeypatch.setattr(pub.time, 'sleep', lambda _: None)
    assert pub.read(p) == {'ok': True} and len(calls) == 3
    p.write_text('{broken'); calls.clear()
    monkeypatch.setattr(Path, 'read_text', original)
    with pytest.raises(json.JSONDecodeError): pub.read(p)


def test_duplicate_json_and_nonfinite_values_fail(tmp_path):
    p = tmp_path / 'r.json'
    for data in ('{"x":1,"x":2}', '{"x":NaN}'):
        p.write_text(data)
        with pytest.raises(pub.AuditDataError): pub.read(p)


def test_transient_ack_retries_boundedly_but_identity_and_layout_conflicts_do_not(tmp_path, monkeypatch):
    monkeypatch.setattr(pub.time, 'sleep', lambda _: None)
    calls = []
    def transient():
        calls.append(1)
        if len(calls) < 3: raise pub.SnapshotUncertain('visibility delay')
        return 'same snapshot'
    assert pub.publish_with_retry(transient, tmp_path, delays=(0, 0)) == 'same snapshot'
    assert len(calls) == 3
    for cls in (pub.AuditDataError, layout.AuditLayoutConflict):
        calls.clear()
        def conflict():
            calls.append(1); raise cls('permanent')
        with pytest.raises(cls): pub.publish_with_retry(conflict, tmp_path, delays=(0, 0))
        assert len(calls) == 1


def test_acknowledgement_requires_exact_snapshot_and_table_sizes():
    snap = pub.collect(Path('/does-not-exist'), campaign())
    summary = {'action_audit/snapshot_sha256': snap['snapshot_sha256']}
    for key, (name, columns) in pub.TABLES.items():
        summary[key] = dict(nrows=len(snap[name]), ncols=len(columns))
    api = SimpleNamespace(run=lambda _: SimpleNamespace(summary=summary))
    state = dict(entity='e', project='p', run_id='same')
    assert pub.acknowledged(api, state, snap)
    summary[pub.PROGRESS_KEY]['nrows'] = 0
    assert not pub.acknowledged(api, state, snap)


def test_lost_snapshot_ack_is_reconciled_without_second_upload(tmp_path, monkeypatch):
    snapshot = pub.collect(Path('/does-not-exist'), campaign())
    state = dict(entity='e', project='p', run_id='same', campaign_sha256='abc')
    remote, calls = {}, dict(read=0, init=0, log=0)
    def get_run(path):
        calls['read'] += 1
        return SimpleNamespace(summary={} if calls['read'] <= 2 else remote)
    def log(payload):
        calls['log'] += 1
        remote.update(payload)
    def init(**kwargs):
        calls['init'] += 1
        assert kwargs['id'] == 'same' and kwargs['resume'] == 'allow'
        return SimpleNamespace(log=log, finish=lambda: None)
    wb = SimpleNamespace(Api=lambda **_: SimpleNamespace(run=get_run), init=init,
                         Table=lambda columns, data: dict(nrows=len(data), ncols=len(columns)))
    monkeypatch.setattr(pub, 'ensure_saved_view', lambda *args, **kwargs: dict(status='verified', url='saved-view'))
    with pytest.raises(pub.SnapshotUncertain):
        pub.publish_once(wb, state, campaign(), snapshot, tmp_path, acknowledgement_seconds=0)
    assert pub.publish_once(wb, state, campaign(), snapshot, tmp_path)['status'] == 'verified'
    assert calls['init'] == calls['log'] == 1


def test_failed_worker_progress_and_incomplete_real_branch_are_explicit(tmp_path):
    data = record(); data['real_scores']['actions'][0]['complete'] = False
    write_root(tmp_path, data)
    path = tmp_path / 'tasks/prior-seed-101/progress.json'
    path.write_text(json.dumps(dict(status='failed', error='worker failure')))
    snap = pub.collect(tmp_path, campaign())
    assert snap['failed'] == 2
    assert all(r['state'] == 'failed' for r in snap['measurements'])
    assert any(r['state'] == 'fail' and r['check'].startswith('real/') for r in snap['checks'])


def test_completed_job_ids_are_not_queried_directly(monkeypatch):
    commands = []
    def query(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout='123_4\n456\n')
    monkeypatch.setattr(pub.subprocess, 'run', query)
    assert pub.gpu_jobs_active(['123'])
    assert not pub.gpu_jobs_active(['789'])
    assert all('-j' not in c for c in commands)


def template():
    return dict(section=dict(runSets=[{'keep': 'existing filters'}], panelBankConfig=dict(sections=[{'__id__': 'untouched'}], panelPlacementOverrides={})))


class Service:
    def __init__(self, lost=False, apply=True):
        self.views = [dict(id='personal', name=layout.DEFAULT_VIEW_NAME, type='project-view', spec=json.dumps(template()), displayName='Personal'),
                      dict(id='other', name='nw-existing-transfer-v', type='project-view', spec=json.dumps(template()), displayName='Transfer')]
        self.chart, self.chart_writes, self.view_writes, self.lost, self.apply = None, 0, 0, lost, apply
    def execute_graphql(self, query, variables):
        if 'CreateActionAuditChart' in query:
            self.chart_writes += 1
            self.chart = dict(id=variables['name'], type=variables['type'], spec=variables['spec'])
            return {'createCustomChart': {'chart': deepcopy(self.chart)}}
        if 'ActionAuditChart' in query: return {'customChart': deepcopy(self.chart)}
        if 'mutation ' in query:
            self.view_writes += 1
            if self.apply:
                self.views.append(dict(id='new', name=variables['name'], type=variables['type'], spec=variables['spec'], displayName=variables['displayName']))
            if self.lost: raise TimeoutError('response lost')
            return {'upsertView': {'view': {'id': 'new'}}}
        return {'project': {'allViews': {'edges': [{'node': deepcopy(v)} for v in self.views]}}}


def install(service, tmp_path):
    return layout.ensure_saved_view(SimpleNamespace(_service_api=service), entity='entity', project='ambi-inner-bench', publication_id='abc123', receipt_dir=tmp_path)


def test_separate_visible_view_preserves_all_existing_views_and_recovers_lost_response(tmp_path):
    service = Service(lost=True); before = deepcopy(service.views)
    result = install(service, tmp_path)
    assert result['url'].endswith('?nw=actionauditabc123')
    assert result['uncertain_response_reconciled'] and not result['browser_verified']
    assert service.views[:2] == before
    assert not install(service, tmp_path)['changed']
    assert service.chart_writes == service.view_writes == 1
    panels = [p for s in layout.sections('entity') for p in s['panels']]
    assert len([p for p in panels if p['viewType'] == 'Vega2']) == 6
    assert len(set(layout.COLORS)) == len(layout.CANDIDATES)
    assert 'pending' in json.dumps(panels).lower() and 'fail' in json.dumps(panels).lower()
    definition = layout.chart_definition()
    assert "=== 'complete'" in definition['transform'][0]['filter']
    assert 'isFinite' in definition['transform'][1]['filter']


def test_modified_audit_view_and_chart_are_never_overwritten(tmp_path):
    service = Service(); install(service, tmp_path)
    spec = json.loads(service.views[-1]['spec']); spec['section']['runSets'][0]['name'] = 'User edit'
    service.views[-1]['spec'] = json.dumps(spec); before = deepcopy(service.views)
    with pytest.raises(layout.AuditLayoutConflict, match='preserving user edits'): install(service, tmp_path)
    assert service.views == before and service.view_writes == 1
    service.chart['spec'] = '{}'
    with pytest.raises(layout.AuditLayoutConflict, match='chart could not be verified'): install(service, tmp_path)
    assert service.chart_writes == 1


def test_missing_view_readback_reports_failure_without_success_receipt(tmp_path):
    service = Service(lost=True, apply=False)
    with pytest.raises(layout.ResultsLayoutError, match='readback'): install(service, tmp_path)
    assert json.loads((tmp_path / 'results-layout-receipt.json').read_text())['status'] == 'failed'
