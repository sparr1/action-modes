#!/usr/bin/env python3
"""Publish replaceable compact action-audit snapshots; retain raw branches locally."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import fcntl
import getpass
import hashlib
import html
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.wandb_results_layout import _hash, _write
from utils.wandb_action_audit_layout import (
    AuditLayoutConflict, CANDIDATES, CHECKS_KEY, CRITICS_KEY, PROGRESS_KEY, TABLE_KEY,
    ensure_saved_view,
)

ENTITY = 'rwgao_b-brown-university'
PROJECT = 'ambi-inner-bench'
MEASUREMENT_COLUMNS = ('history', 'seed', 'decision', 'candidate', 'horizon', 'state',
    'model_value', 'prefix_value', 'tail_value', 'model_gain', 'prefix_gain', 'tail_gain',
    'model_gain_se', 'prefix_gain_se', 'tail_gain_se', 'heldout_model_value', 'heldout_model_gain',
    'model_prefix_bias', 'terminal_bias', 'model_prefix_bias_se', 'terminal_bias_se')
PROGRESS_COLUMNS = ('history', 'seed', 'decision', 'state', 'check_state', 'message')
CHECK_COLUMNS = ('history', 'seed', 'decision', 'check', 'state', 'value')
CRITIC_COLUMNS = ('history', 'seed', 'decision', 'bank_scope', 'critic', 'horizon', 'state',
    'actions', 'samples', 'bias', 'rmse', 'centered_rmse', 'relative_bias', 'relative_rmse',
    'spearman', 'pearson', 'top_action_label', 'reference_top_action_label',
    'top_action_regret', 'top_action_regret_se', 'selected_action_gain', 'selected_action_gain_se')
TABLES = {TABLE_KEY: ('measurements', MEASUREMENT_COLUMNS), PROGRESS_KEY: ('progress', PROGRESS_COLUMNS),
          CHECKS_KEY: ('checks', CHECK_COLUMNS), CRITICS_KEY: ('critics', CRITIC_COLUMNS)}


class AuditDataError(ValueError):
    """Invalid scientific data or a changed prepared identity; do not retry it."""


class SnapshotUncertain(RuntimeError):
    """A replaceable overview snapshot has not appeared on the remote server yet."""


def read(path):
    for attempt, delay in enumerate((.25, 1., 4., None)):
        try:
            text = Path(path).read_text()
            def pairs(items):
                result = {}
                for k, v in items:
                    if k in result:
                        raise AuditDataError(f'Duplicate JSON key {k!r}: {path}')
                    result[k] = v
                return result
            return json.loads(text, object_pairs_hook=pairs,
                              parse_constant=lambda value: (_ for _ in ()).throw(AuditDataError('Nonfinite JSON value: ' + value)))
        except OSError as exc:
            if exc.errno not in (errno.ESTALE, errno.EAGAIN, errno.ETIMEDOUT) or attempt == 3:
                raise
            time.sleep(delay)


def exists(path):
    for attempt, delay in enumerate((.25, 1., 4., None)):
        try:
            return Path(path).exists()
        except OSError as exc:
            if exc.errno not in (errno.ESTALE, errno.EAGAIN, errno.ETIMEDOUT) or attempt == 3:
                raise
            time.sleep(delay)


def finite(value, field):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AuditDataError('Expected a finite numeric ' + field)
    return float(value)


def identity(history, seed, decision):
    return dict(history=history, seed=int(seed), decision=int(decision))


def check_rows(checks, ident, prefix=''):
    rows = []
    for key, value in sorted(checks.items()):
        name = prefix + key
        if isinstance(value, dict) and not {'passed', 'status', 'ok'}.intersection(value):
            rows.extend(check_rows(value, ident, name + '/'))
            continue
        flag = value
        if isinstance(value, dict):
            flag = value.get('passed', value.get('ok', value.get('status')))
        state = ('pass' if flag is True or flag in ('pass', 'passed', 'complete', 'verified') else
                 'fail' if flag is False or flag in ('fail', 'failed', 'error') else
                 'pending' if flag is None or flag == 'pending' else 'observed')
        rows.append({**ident, 'check': name, 'state': state, 'value': json.dumps(value, sort_keys=True)})
    return rows


def normalize_root(record, ident):
    if record.get('identity') != ident:
        raise AuditDataError('Root identity does not match its declared campaign location.')
    checks = record.get('checks')
    if not isinstance(checks, dict) or not checks:
        raise AuditDataError('A completed root must include explicit execution checks.')
    check_values = check_rows(checks, ident)
    state = 'failed' if any(c['state'] == 'fail' for c in check_values) else 'complete'
    if any(c['state'] == 'pending' for c in check_values):
        state = 'pending'
    real = record.get('real_scores', {}).get('actions')
    heldout = record.get('heldout_scores', {}).get('actions')
    if not isinstance(real, list) or not isinstance(heldout, list):
        raise AuditDataError('Root requires real_scores.actions and heldout_scores.actions.')
    if any(record[name].get('baseline_label') != 'prior_mean' for name in ('real_scores', 'heldout_scores')):
        raise AuditDataError('Published gains require the declared same-root prior_mean baseline.')
    observed = set()
    for row in real:
        key = (row['label'], row['horizon'])
        if key in observed:
            raise AuditDataError('Duplicate candidate/horizon in real scores.')
        observed.add(key)
        if row['label'] not in CANDIDATES or row['horizon'] not in (1, 3):
            raise AuditDataError('Unknown compact real-audit candidate or horizon.')
        if row.get('complete') is not True:
            state = 'failed'
            check_values.append({**ident, 'check': f'real/{row["label"]}/h{row["horizon"]}',
                                 'state': 'fail', 'value': 'Real scorer did not complete all requested branches.'})
    if observed != {(label, h) for label in CANDIDATES for h in (1, 3)}:
        raise AuditDataError('A completed root must contain all ten declared candidates at both horizons.')
    priors = {r['horizon']: r for r in real if r['label'] == 'prior_mean'}
    if set(priors) != {1, 3}:
        raise AuditDataError('Both H1 and H3 paired prior references are required.')
    heldout_by_label = {r['label']: r for r in heldout}
    if len(heldout_by_label) != len(heldout):
        raise AuditDataError('Duplicate heldout candidate label.')
    measurements = []
    for row in sorted(real, key=lambda x: (x['horizon'], CANDIDATES.index(x['label']))):
        label, h = row['label'], row['horizon']
        prior = priors[h]
        try:
            heldout_model = finite(heldout_by_label[label]['model'][f'h{h}']['mean'], 'heldout model value')
            heldout_prior = finite(heldout_by_label['prior_mean']['model'][f'h{h}']['mean'], 'heldout prior model value')
        except KeyError as exc:
            raise AuditDataError('Missing paired heldout model score.') from exc
        prefix = finite(row['real_prefix_value_mean'], 'real prefix value')
        tail = finite(row['real_tail_mean'], 'real tail value')
        # Paired fields use the real audit's common noise bank. The independent
        # 32-draw heldout model bank stays separate; never mix marginal SEs.
        model = finite(row['model_mean'], 'paired model value')
        measurements.append({**ident, 'candidate': label, 'horizon': h, 'state': state,
            'model_value': model, 'prefix_value': prefix, 'tail_value': tail,
            'model_gain': finite(row['model_gain_vs_baseline_mean'], 'paired model gain'),
            'prefix_gain': finite(row['real_prefix_value_gain_vs_baseline_mean'], 'paired prefix gain'),
            'tail_gain': finite(row['real_tail_gain_vs_baseline_mean'], 'paired tail gain'),
            'model_gain_se': finite(row['model_gain_vs_baseline_se'], 'paired model gain SE'),
            'prefix_gain_se': finite(row['real_prefix_value_gain_vs_baseline_se'], 'paired prefix gain SE'),
            'tail_gain_se': finite(row['real_tail_gain_vs_baseline_se'], 'paired tail gain SE'),
            'heldout_model_value': heldout_model, 'heldout_model_gain': heldout_model - heldout_prior,
            'model_prefix_bias': finite(row['model_prefix_bias_mean'], 'model-prefix bias'),
            'terminal_bias': finite(row['terminal_bias_mean'], 'terminal bias'),
            'model_prefix_bias_se': finite(row.get('model_prefix_bias_se'), 'model-prefix paired SE'),
            'terminal_bias_se': finite(row.get('terminal_bias_se'), 'terminal paired SE')})
    critics = []
    for scope in ('selection', 'heldout'):
        for critic in record.get(scope + '_scores', {}).get('critics', []):
            row = {**ident, 'bank_scope': scope, 'critic': critic['name'], 'horizon': critic['horizon'], 'state': state}
            for metric in CRITIC_COLUMNS[7:]:
                value = critic.get(metric)
                row[metric] = value if metric.endswith('_label') else finite(value, metric)
            critics.append(row)
    return measurements, critics, check_values, state


def collect(root, campaign, *, workers_active=True):
    progress, measurements, critics, checks, fingerprints = [], [], [], [], []
    for history in campaign['histories']:
        if not re.fullmatch(r'[A-Za-z0-9_]+', history):
            raise AuditDataError('Invalid history directory component.')
        for seed in campaign['seeds']:
            task = root / 'tasks' / f'{history}-seed-{seed}'
            failure = next((read(task / name) for name in ('failed.json', 'failure.json', 'error.json') if exists(task / name)), None)
            task_progress = read(task / 'progress.json') if exists(task / 'progress.json') else {}
            task_manifest = read(task / 'manifest.json') if exists(task / 'manifest.json') else {}
            if task_progress.get('status') == 'failed':
                failure = task_progress
            for decision in campaign['decisions']:
                ident = identity(history, seed, decision)
                path = task / f'root-{decision}.json'
                if exists(path):
                    record = read(path)
                    values, cs, check_values, state = normalize_root(record, ident)
                    measurements.extend(values); critics.extend(cs); checks.extend(check_values)
                    fingerprints.append([history, seed, decision, _hash(record)])
                    progress.append({**ident, 'state': state,
                        'check_state': 'fail' if state == 'failed' else 'pending' if state == 'pending' else 'pass',
                        'message': 'Integrity checks failed; measurements excluded from charts.' if state == 'failed' else ''})
                else:
                    unreachable = decision in task_manifest.get('unreached_decisions', [])
                    state = 'failed' if failure is not None or unreachable else 'pending'
                    progress.append({**ident, 'state': state, 'check_state': 'pending',
                        'message': json.dumps(failure, sort_keys=True)[:500] if failure is not None else
                                   'Source trajectory ended before this requested root.' if unreachable else
                                   'Waiting for the requested root.' if workers_active else 'Workers stopped before this root was available.'})
                    checks.append({**ident, 'check': 'root_available', 'state': 'pending', 'value': None})
    counts = {k: sum(r['state'] == k for r in progress) for k in ('complete', 'failed', 'pending')}
    total = len(progress)
    state = 'complete' if counts['complete'] == total else 'failed' if not workers_active else 'running'
    result = dict(schema_version=1, state=state, total=total, **counts, progress=progress,
                  measurements=measurements, critics=critics, checks=checks, root_fingerprints=fingerprints)
    result['snapshot_sha256'] = _hash(result)
    return result


def write_report(out, snapshot):
    _write(out / 'snapshot.json', snapshot)
    def table(title, columns, values):
        heads = ''.join('<th>' + html.escape(k) + '</th>' for k in columns)
        body = ''.join('<tr>' + ''.join('<td>' + html.escape('' if r.get(k) is None else str(r[k])) + '</td>' for k in columns) + '</tr>' for r in values)
        return '<h2>' + html.escape(title) + '</h2><div><table><thead><tr>' + heads + '</tr></thead><tbody>' + body + '</tbody></table></div>'
    document = '<!doctype html><meta charset="utf-8"><title>800k action audit</title><style>body{font:14px system-ui;margin:2rem}div{overflow:auto}td,th{padding:.4rem;border-bottom:1px solid #ddd;text-align:left}th{background:#eee}</style>'
    document += f'<h1>800k matched-action audit</h1><p>{snapshot["complete"]}/{snapshot["total"]} complete; {snapshot["failed"]} failed; {snapshot["pending"]} pending.</p>'
    document += '<p>Local retained report. Model and finite real-tail values are distinct; integrity pass is not a reward-improvement claim. Missing values remain blank.</p>'
    for key, (name, columns) in TABLES.items():
        document += table(name.title(), columns, snapshot[name])
    temporary = out / 'report.html.tmp'
    temporary.write_text(document)
    temporary.replace(out / 'report.html')


def publication_state(root, out, campaign, entity, project):
    path = out / 'publication.json'
    expected = dict(schema_version=1, campaign_root=str(root), campaign_sha256=_hash(campaign), entity=entity, project=project)
    if exists(path):
        state = read(path)
        if any(state.get(k) != v for k, v in expected.items()):
            raise AuditDataError('Prepared publication identity changed; refusing to reuse the run.')
        return state
    state = {**expected, 'run_id': uuid.uuid4().hex}
    _write(path, state)
    return state


@contextmanager
def publisher_lock(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'publisher.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def gpu_jobs_active(ids):
    if not ids:
        return True
    # Completed arrays can disappear from squeue; querying those IDs directly
    # yields an error rather than an empty queue on Oscar.
    requested = {str(job).split('_', 1)[0] for job in ids}
    output = subprocess.run(['squeue', '--noheader', '--user', getpass.getuser(), '-o', '%i'],
                            check=True, text=True, capture_output=True, timeout=30)
    live = {line.strip().split('_', 1)[0] for line in output.stdout.splitlines() if line.strip()}
    return bool(requested & live)


def acknowledged(api, state, snapshot):
    remote = api.run(f'{state["entity"]}/{state["project"]}/{state["run_id"]}')
    summary = remote.summary
    if summary.get('action_audit/snapshot_sha256') != snapshot['snapshot_sha256']:
        return False
    for key, (name, columns) in TABLES.items():
        value = summary.get(key)
        if value is None or value.get('nrows') != len(snapshot[name]) or value.get('ncols') != len(columns):
            return False
    return True


def publish_once(wandb, state, campaign, snapshot, out, *, acknowledgement_seconds=30):
    api = wandb.Api(timeout=30)
    try:
        already = acknowledged(api, state, snapshot)
    except Exception:
        already = False
    if not already:
        run = wandb.init(entity=state['entity'], project=state['project'], id=state['run_id'], resume='allow', mode='online',
            name='800k matched-action audit', group='action-audit-800k-' + state['run_id'][:8], job_type='action-audit-overview',
            config={'action_audit_publication': state['run_id'], 'campaign_sha256': state['campaign_sha256'],
                    'checkpoint': campaign['checkpoint'], 'protocol': campaign.get('protocol'),
                    'publication_semantics': 'replaceable compact snapshot; raw scientific records remain on Oscar'})
        try:
            payload = {key: wandb.Table(columns=list(columns), data=[[r.get(c) for c in columns] for r in snapshot[name]])
                       for key, (name, columns) in TABLES.items()}
            payload.update({'action_audit/snapshot_sha256': snapshot['snapshot_sha256'],
                            'action_audit/complete': snapshot['complete'], 'action_audit/total': snapshot['total'],
                            'action_audit/failed': snapshot['failed'], 'action_audit/pending': snapshot['pending'],
                            'action_audit/status': snapshot['state']})
            run.log(payload)
        finally:
            run.finish()
        deadline = time.monotonic() + acknowledgement_seconds
        while True:
            try:
                if acknowledged(wandb.Api(timeout=30), state, snapshot):
                    break
            except Exception:
                pass
            if time.monotonic() >= deadline:
                raise SnapshotUncertain('Compact snapshot is not yet visible; safe to replace under the same run ID.')
            time.sleep(min(5., max(0., deadline - time.monotonic())))
    layout = ensure_saved_view(wandb.Api(timeout=30), entity=state['entity'], project=state['project'],
                               publication_id=state['run_id'], receipt_dir=out / 'results-layout')
    _write(out / 'published.json', dict(snapshot_sha256=snapshot['snapshot_sha256'], run_id=state['run_id'],
                                       completed=snapshot['complete'], total=snapshot['total'], layout=layout))
    return layout


def publish_with_retry(operation, out, *, delays=(5., 15., 30., 60., 60.)):
    for attempt in range(len(delays) + 1):
        try:
            result = operation()
            _write(out / 'publisher-retry.json', dict(status='healthy', attempts=attempt + 1))
            return result
        except (AuditDataError, AuditLayoutConflict, KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            status = getattr(exc, 'status_code', None) or getattr(getattr(exc, 'response', None), 'status_code', None)
            terminal = status in (400, 401, 403, 404) or attempt == len(delays)
            _write(out / 'publisher-retry.json', dict(status='failed' if terminal else 'retrying', attempt=attempt + 1,
                error_type=type(exc).__name__, error=str(exc), delay_seconds=None if terminal else delays[attempt]))
            if terminal:
                raise
            time.sleep(delays[attempt])


def watch(args):
    import wandb
    root, out = args.root.resolve(), args.publication_root.resolve()
    if root == out or out.is_relative_to(root) or root.is_relative_to(out):
        raise AuditDataError('Publication state must be outside the scientific campaign directory.')
    with publisher_lock(out):
        campaign = read(root / 'campaign.json')
        state = publication_state(root, out, campaign, args.entity, args.project)
        previous = None
        inactive_since = None
        while True:
            try:
                active = gpu_jobs_active(args.gpu_job_id)
                inactive_since = None if active else inactive_since or time.monotonic()
                # Allow completed workers' final atomic records to become visible.
                still_waiting = active or time.monotonic() - inactive_since < args.worker_grace_seconds
                snapshot = collect(root, campaign, workers_active=still_waiting)
                write_report(out, snapshot)
                if snapshot['snapshot_sha256'] != previous:
                    layout = publish_with_retry(lambda: publish_once(wandb, state, campaign, snapshot, out), out)
                    previous = snapshot['snapshot_sha256']
                    print(f'{snapshot["complete"]}/{snapshot["total"]} complete; {snapshot["failed"]} failed; {layout["url"]}', flush=True)
                _write(out / 'publisher-status.json', dict(status=snapshot['state'], completed=snapshot['complete'],
                    total=snapshot['total'], failed=snapshot['failed'], pending=snapshot['pending'], run_id=state['run_id']))
                if snapshot['state'] == 'complete' or args.once:
                    return
                if not still_waiting:
                    raise AuditDataError('Workers stopped before all requested roots passed execution checks; see progress table.')
                time.sleep(args.poll_seconds)
            except BaseException as exc:
                try:
                    _write(out / 'publisher-failure.json', dict(error_type=type(exc).__name__, error=str(exc), run_id=state['run_id']))
                except Exception:
                    pass
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--publication-root', type=Path, required=True)
    parser.add_argument('--entity', default=ENTITY)
    parser.add_argument('--project', default=PROJECT)
    parser.add_argument('--gpu-job-id', action='append', default=[])
    parser.add_argument('--poll-seconds', type=float, default=60.)
    parser.add_argument('--worker-grace-seconds', type=float, default=120.)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or args.worker_grace_seconds < 0:
        parser.error('Poll interval must be positive and grace interval nonnegative.')
    if not args.gpu_job_id and not args.once:
        parser.error('Watch mode requires at least one --gpu-job-id.')
    watch(args)


if __name__ == '__main__':
    main()
