#!/usr/bin/env python3
"""Publish verified full-episode critic-bypass results without large artifacts."""
from __future__ import annotations

import argparse
import html
import math
from pathlib import Path
import re
import sys
import time
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm.publish_ambi_action_audit import (
    AuditDataError as BypassDataError, SnapshotUncertain, exists, read,
    gpu_jobs_active, publisher_lock,
)
from utils.wandb_results_layout import _hash, _write
from utils.wandb_critic_bypass_layout import (
    ARMS, LABELS, BypassLayoutConflict, DIAGNOSTICS_KEY, EPISODES_KEY,
    PAIRS_KEY, PROGRESS_KEY, SUMMARY_KEY, ensure_saved_view,
)

ENTITY, PROJECT = 'rwgao_b-brown-university', 'ambi-inner-bench'
PROTOCOL = 'closed-loop-critic-bypass-v1'
BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES = 20260912, 2000
TIMING = ('controller_time_mean_s', 'controller_time_p95_s',
          'selection_time_mean_s', 'selection_time_p95_s')
EPISODE_COLUMNS = ('task_id', 'arm', 'label', 'env_seed', 'controller_seed', 'state',
                   'episode_return', 'episode_length', *TIMING, 'reference_reused')
PROGRESS_COLUMNS = ('task_id', 'arm', 'label', 'env_seed', 'controller_seed',
                    'state', 'decisions', 'check_state', 'message')
SUMMARY_COLUMNS = ('arm', 'label', 'baseline', 'metric', 'state', 'n', 'mean',
                   'ci_low', 'ci_high', 'sd', 'win_rate', 'tie_rate', 'estimator')
DIAGNOSTIC_COLUMNS = ('arm', 'label', 'env_seed', 'controller_seed', 'decision',
                      'candidate', 'metric', 'value')
TABLES = {SUMMARY_KEY: ('summary', SUMMARY_COLUMNS), PAIRS_KEY: ('pairs', SUMMARY_COLUMNS),
          EPISODES_KEY: ('episodes', EPISODE_COLUMNS), PROGRESS_KEY: ('progress', PROGRESS_COLUMNS),
          DIAGNOSTICS_KEY: ('diagnostics', DIAGNOSTIC_COLUMNS)}


def finite(value, field, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BypassDataError('Expected finite numeric ' + field)
    if nonnegative and value < 0:
        raise BypassDataError('Expected nonnegative ' + field)
    return float(value)


def validate_campaign(campaign):
    if campaign.get('protocol') != PROTOCOL or not re.fullmatch(r'[0-9a-f]{64}', str(campaign.get('campaign_id', ''))):
        raise BypassDataError('Invalid campaign protocol or campaign_id.')
    arms, seeds = campaign.get('arms'), campaign.get('seeds')
    if not isinstance(arms, list) or len(arms) != len(set(arms)) or set(arms) != set(ARMS):
        raise BypassDataError('Campaign requires exactly the five declared arms.')
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int for s in seeds) or len(seeds) != len(set(seeds)):
        raise BypassDataError('Campaign seeds must be unique integers.')
    if type(campaign.get('controller_seed')) is not int or campaign['controller_seed'] != 55:
        raise BypassDataError('This protocol requires controller seed 55.')
    if type(campaign.get('max_steps')) is not int or not 0 < campaign['max_steps'] <= 500:
        raise BypassDataError('Invalid episode decision limit.')
    if not campaign.get('smoke', False) and (set(seeds) != set(range(101, 121)) or campaign['max_steps'] != 500):
        raise BypassDataError('Production requires all 20 seeds 101–120 and 500-decision episodes.')
    tasks = campaign.get('task_list')
    expected = {f'{a}-seed-{s}': dict(task_id=f'{a}-seed-{s}', arm=a, env_seed=s,
                                    controller_seed=campaign['controller_seed']) for a in arms for s in seeds}
    if not isinstance(tasks, list) or len(tasks) != len(expected):
        raise BypassDataError('Campaign task list must cover the exact arm/seed grid.')
    observed = {}
    for task in tasks:
        ident = {k: task.get(k) for k in ('task_id', 'arm', 'env_seed', 'controller_seed')}
        key = ident['task_id']
        if key in observed or key not in expected or ident != expected[key]:
            raise BypassDataError('Duplicate or conflicting campaign task identity.')
        observed[key] = ident
    return [observed[f'{a}-seed-{s}'] for a in ARMS for s in sorted(seeds)]


def normalize_result(record, task, campaign):
    if record.get('schema_version') != 1 or record.get('protocol') != PROTOCOL:
        raise BypassDataError('Unsupported task result schema or protocol.')
    if record.get('campaign_id') != campaign['campaign_id'] or any(record.get(k) != v for k, v in task.items()):
        raise BypassDataError('Result identity conflicts with the campaign/task location.')
    if record.get('complete') is not True:
        raise BypassDataError('An atomic result.json must be complete.')
    checks = record.get('checks')
    if not isinstance(checks, dict) or checks.get('outer_state_unchanged') is not True or any(v is not True for v in checks.values()):
        raise BypassDataError('Completed result has missing or failed integrity checks.')
    length = record.get('episode_length')
    if type(length) is not int or not 0 < length <= campaign['max_steps']:
        raise BypassDataError('Invalid episode length.')
    if length < campaign['max_steps'] and record.get('terminated') is not True and record.get('truncated') is not True:
        raise BypassDataError('A short episode must have an explicit environment ending.')
    episode = {**task, 'label': LABELS[ARMS.index(task['arm'])], 'state': 'complete',
               'episode_return': finite(record.get('episode_return'), 'episode_return'),
               'episode_length': length, 'reference_reused': record.get('reused_reference') is True}
    times = {}
    reused = episode['reference_reused']
    if reused and (task['arm'] != 'prior' or record.get('runtime', {}).get('timing_comparable') is not False or
                   not re.fullmatch(r'[0-9a-f]{64}', str(record.get('reference_sha256', ''))) or
                   not isinstance(record.get('reference_source'), str) or not record['reference_source']):
        raise BypassDataError('Reused prior requires verified reference provenance and unavailable comparable timing.')
    for stem in ('controller', 'selection'):
        values = record.get(stem + '_times_s')
        if reused:
            if values != [] or any(record.get(f'{stem}_time_{statistic}_s') is not None for statistic in ('mean', 'p95')):
                raise BypassDataError('Historical reference timing must remain unavailable, not zero.')
            times[stem] = []
            episode.update({f'{stem}_time_{statistic}_s': None for statistic in ('mean', 'p95')})
            continue
        if not isinstance(values, list) or len(values) != length:
            raise BypassDataError('Exact per-decision ' + stem + ' timing coverage is required.')
        values = [finite(v, stem + '_times_s', nonnegative=True) for v in values]
        times[stem] = values
        for statistic, computed in (('mean', float(np.mean(values))), ('p95', float(np.percentile(values, 95)))):
            field = f'{stem}_time_{statistic}_s'
            reported = finite(record.get(field), field, nonnegative=True)
            if not math.isclose(reported, computed, rel_tol=1e-7, abs_tol=1e-10):
                raise BypassDataError('Reported timing disagrees with per-decision timing: ' + field)
            episode[field] = computed
    if any(s > c + 1e-8 for c, s in zip(times['controller'], times['selection'])):
        raise BypassDataError('Selection overhead exceeds total controller time.')
    diagnostics = []
    raw_diagnostics = record.get('diagnostics', [])
    if not isinstance(raw_diagnostics, list):
        raise BypassDataError('Diagnostics must be a list of same-state records.')
    if task['arm'] in ('actor_mean', 'learned_q', 'model_score') and 'diagnostic_decisions' in campaign:
        requested = {d for d in campaign['diagnostic_decisions'] if d < length}
        observed = [r.get('decision') for r in raw_diagnostics if isinstance(r, dict)]
        if len(observed) != len(requested) or set(observed) != requested:
            raise BypassDataError('Missing or duplicate requested same-state diagnostic decisions.')
    # Only the compact scalar diagnostic schema crosses the W&B boundary.
    # Raw actions, score arrays and checkpoints stay in scientific storage.
    for row in raw_diagnostics:
        if not isinstance(row, dict):
            raise BypassDataError('Diagnostic row must be an object.')
        decision = row.get('decision')
        if type(decision) is not int or not 0 <= decision < length:
            raise BypassDataError('Diagnostic decision lies outside its episode.')
        choices = row.get('choices')
        if not isinstance(choices, dict) or set(choices) != {'actor_mean', 'learned_q', 'model_score', 'prior_mean'}:
            raise BypassDataError('Same-state diagnostics require all four candidate choices.')
        for candidate, values in choices.items():
            if not isinstance(values, dict):
                raise BypassDataError('Unknown same-state diagnostic candidate.')
            metrics = dict(heldout_model_value=values['validation_model']['mean'],
                heldout_model_gain_vs_actor=values['validation_gain_vs_actor']['mean'],
                heldout_model_gain_vs_prior=values['validation_gain_vs_prior']['mean'],
                selection_model_value=values['selection_model']['mean'], learned_q=values['learned_q'])
            for metric in ('heldout_model_value', 'heldout_model_gain_vs_actor', 'heldout_model_gain_vs_prior',
                           'selection_model_value', 'learned_q', 'action_distance_vs_actor'):
                if metric in metrics:
                    diagnostics.append(dict(arm=task['arm'], label=episode['label'], env_seed=task['env_seed'],
                        controller_seed=task['controller_seed'], decision=decision, candidate=candidate,
                        metric=metric, value=finite(metrics[metric], 'diagnostic ' + metric)))
    if len({(r['decision'], r['candidate'], r['metric']) for r in diagnostics}) != len(diagnostics):
        raise BypassDataError('Duplicate same-state diagnostic row.')
    return episode, times, diagnostics


def estimate(values):
    """Equal environment-seed weighting; fixed draws also preserve paired CIs."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    result = dict(n=n, mean=float(values.mean()), sd=float(values.std(ddof=1)) if n > 1 else None,
                  ci_low=None, ci_high=None)
    if n > 1:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        draws = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
        lo, hi = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
        result.update(ci_low=float(lo), ci_high=float(hi))
    return result


def summarize(episodes, timings, campaign):
    by_arm = {arm: {r['env_seed']: r for r in episodes if r['arm'] == arm} for arm in ARMS}
    complete = {arm: rows for arm, rows in by_arm.items() if set(rows) == set(campaign['seeds'])}
    summary, pairs = [], []
    seeds = sorted(campaign['seeds'])
    for arm, rows in complete.items():
        ident = dict(arm=arm, label=LABELS[ARMS.index(arm)], baseline='none', state='complete')
        for metric in ('episode_return', 'controller_time_mean_s', 'selection_time_mean_s'):
            values = [rows[s][metric] for s in seeds if rows[s][metric] is not None]
            if values:
                summary.append({**ident, 'metric': metric, **estimate(values),
                                'estimator': 'equal environment-seed mean; percentile bootstrap; historical timing excluded' if len(values) != len(seeds) else 'equal environment-seed mean; percentile bootstrap'})
        for stem in ('controller', 'selection'):
            pooled = [v for s in seeds for v in timings[f'{arm}-seed-{s}'][stem]]
            if pooled:
                timed_seeds = sum(bool(timings[f'{arm}-seed-{s}'][stem]) for s in seeds)
                summary.append({**ident, 'metric': stem + '_time_p95_s', 'n': timed_seeds,
                                'mean': float(np.percentile(pooled, 95)), 'ci_low': None, 'ci_high': None,
                                'sd': None, 'estimator': 'pooled decision p95; no interval; historical timing excluded' if timed_seeds != len(seeds) else 'pooled decision p95; no interval'})
        summary.append({**ident, 'metric': 'episode_return_sd', 'n': len(seeds),
                        'mean': float(np.std([rows[s]['episode_return'] for s in seeds], ddof=1)) if len(seeds) > 1 else None,
                        'ci_low': None, 'ci_high': None, 'sd': None, 'estimator': 'environment-seed sample standard deviation'})
        for baseline, base in complete.items():
            if baseline == arm:
                continue
            deltas = [rows[s]['episode_return'] - base[s]['episode_return'] for s in seeds]
            pairs.append({**ident, 'baseline': baseline, 'metric': 'episode_return', **estimate(deltas),
                          'win_rate': float(np.mean(np.asarray(deltas) > 0)),
                          'tie_rate': float(np.mean(np.asarray(deltas) == 0)),
                          'estimator': 'paired environment-seed difference; percentile bootstrap'})
    return summary, pairs


def collect(root, campaign, *, workers_active=True):
    tasks = validate_campaign(campaign)
    expected = {task['task_id'] for task in tasks}
    for path in (root / 'tasks').glob('*/result.json'):
        if path.parent.name not in expected:
            raise BypassDataError('Unexpected or duplicated result directory: ' + path.parent.name)
    episodes, progress, diagnostics, fingerprints, timings = [], [], [], [], {}
    for task in tasks:
        directory = root / 'tasks' / task['task_id']
        ident = {**task, 'label': LABELS[ARMS.index(task['arm'])]}
        path = directory / 'result.json'
        failed = next((read(directory / name) for name in ('failure.json', 'failed.json', 'error.json') if exists(directory / name)), None)
        task_progress = read(directory / 'progress.json') if exists(directory / 'progress.json') else {}
        if task_progress.get('status') == 'failed':
            failed = task_progress
        if exists(path):
            if failed is not None:
                raise BypassDataError('A completed result conflicts with a failure marker: ' + task['task_id'])
            record = read(path)
            episode, values, ds = normalize_result(record, task, campaign)
            episodes.append(episode); timings[task['task_id']] = values; diagnostics.extend(ds)
            fingerprints.append(dict(task_id=task['task_id'], sha256=_hash(record)))
            progress.append({**ident, 'state': 'complete', 'decisions': episode['episode_length'], 'check_state': 'pass', 'message': ''})
        else:
            state = 'failed' if failed is not None else 'pending'
            progress.append({**ident, 'state': state, 'decisions': task_progress.get('decisions', task_progress.get('decision', 0)),
                'check_state': 'fail' if failed is not None else 'pending',
                'message': str(failed)[:500] if failed is not None else 'Waiting for complete episode and integrity checks.' if workers_active else 'Workers stopped before a complete result was available.'})
    summary, pairs = summarize(episodes, timings, campaign)
    counts = {k: sum(r['state'] == k for r in progress) for k in ('complete', 'pending', 'failed')}
    state = 'complete' if counts['complete'] == len(tasks) else 'running' if workers_active else 'failed'
    result = dict(schema_version=1, protocol=PROTOCOL, campaign_id=campaign['campaign_id'], state=state,
        total=len(tasks), **counts, episodes=episodes, progress=progress, summary=summary, pairs=pairs,
        diagnostics=diagnostics, result_fingerprints=fingerprints,
        uncertainty=dict(unit='environment_seed', confidence=.95, bootstrap_resamples=BOOTSTRAP_RESAMPLES,
                         bootstrap_seed=BOOTSTRAP_SEED, final_summary_requires_all_requested_seeds=True))
    result['snapshot_sha256'] = _hash(result)
    return result


def verify_immutable_results(previous, current):
    before = {r['task_id']: r['sha256'] for r in previous.get('result_fingerprints', [])}
    after = {r['task_id']: r['sha256'] for r in current['result_fingerprints']}
    if any(after.get(task) != sha for task, sha in before.items()):
        raise BypassDataError('A previously published scientific result changed or disappeared.')


def write_report(out, snapshot):
    if exists(out / 'snapshot.json'):
        verify_immutable_results(read(out / 'snapshot.json'), snapshot)
    _write(out / 'snapshot.json', snapshot)
    document = '<!doctype html><meta charset="utf-8"><title>800k critic bypass</title><style>body{font:14px system-ui;margin:2rem}div{overflow:auto}td,th{padding:.4rem;border-bottom:1px solid #ddd;text-align:left}th{background:#eee}</style>'
    document += f'<h1>800k full-episode critic bypass</h1><p>{snapshot["complete"]}/{snapshot["total"]} complete; {snapshot["failed"]} failed; {snapshot["pending"]} pending.</p>'
    document += '<p>Final arm summaries and paired intervals require all requested seeds. Return is undiscounted environment reward; model diagnostics are separate. Timing excludes diagnostic work. Raw records remain on Oscar.</p>'
    for _, (name, columns) in TABLES.items():
        document += '<h2>' + html.escape(name.title()) + '</h2><div><table><thead><tr>' + ''.join('<th>' + html.escape(c) + '</th>' for c in columns) + '</tr></thead><tbody>'
        document += ''.join('<tr>' + ''.join('<td>' + html.escape('' if row.get(c) is None else str(row[c])) + '</td>' for c in columns) + '</tr>' for row in snapshot[name])
        document += '</tbody></table></div>'
    temporary = out / 'report.html.tmp'; temporary.write_text(document); temporary.replace(out / 'report.html')


def publication_state(root, out, campaign, entity, project):
    expected = dict(schema_version=1, campaign_root=str(root), campaign_sha256=_hash(campaign), entity=entity, project=project)
    path = out / 'publication.json'
    if exists(path):
        state = read(path)
        if any(state.get(k) != v for k, v in expected.items()):
            raise BypassDataError('Prepared publication identity changed; refusing to reuse the run.')
        return state
    state = {**expected, 'run_id': uuid.uuid4().hex}; _write(path, state)
    return state


def acknowledged(api, state, snapshot):
    summary = api.run(f'{state["entity"]}/{state["project"]}/{state["run_id"]}').summary
    if summary.get('critic_bypass/snapshot_sha256') != snapshot['snapshot_sha256']:
        return False
    for key, (name, columns) in TABLES.items():
        # The public SDK wraps nested summary values in SummarySubDict rather
        # than dict. Use its mapping interface, as for the outer summary.
        value = summary.get(key)
        if value is None or value.get('nrows') != len(snapshot[name]) or value.get('ncols') != len(columns):
            return False
    return True


def publish_once(wandb, state, campaign, snapshot, out, *, acknowledgement_seconds=30):
    try:
        already = acknowledged(wandb.Api(timeout=30), state, snapshot)
    except Exception:
        already = False
    if not already:
        run = wandb.init(entity=state['entity'], project=state['project'], id=state['run_id'], resume='allow', mode='online',
            name='800k critic bypass | H1 J6 | 20 seeds', group='critic-bypass-800k-' + state['run_id'][:8], job_type='critic-bypass-overview',
            config={'critic_bypass_publication': state['run_id'], 'campaign_sha256': state['campaign_sha256'],
                    'campaign_id': campaign['campaign_id'], 'checkpoint': campaign.get('checkpoint'), 'protocol': PROTOCOL,
                    'publication_semantics': 'replaceable compact snapshot; final summaries require complete seed coverage'})
        try:
            payload = {key: wandb.Table(columns=list(columns), data=[[r.get(c) for c in columns] for r in snapshot[name]])
                       for key, (name, columns) in TABLES.items()}
            payload.update({'critic_bypass/snapshot_sha256': snapshot['snapshot_sha256'],
                            **{'critic_bypass/' + k: snapshot[k] for k in ('complete', 'total', 'failed', 'pending')},
                            'critic_bypass/status': snapshot['state']})
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
                raise SnapshotUncertain('Compact snapshot not yet visible; safe to replace under the same run ID.')
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
        except (BypassDataError, BypassLayoutConflict, KeyboardInterrupt, SystemExit):
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
        raise BypassDataError('Publication state must be outside the scientific campaign directory.')
    with publisher_lock(out):
        campaign = read(root / 'campaign.json'); validate_campaign(campaign)
        if campaign.get('smoke'):
            raise BypassDataError('Smoke results must not be published as the production comparison.')
        state = publication_state(root, out, campaign, args.entity, args.project)
        previous, inactive_since = None, None
        while True:
            try:
                active = gpu_jobs_active(args.gpu_job_id)
                inactive_since = None if active else inactive_since or time.monotonic()
                still_waiting = active or time.monotonic() - inactive_since < args.worker_grace_seconds
                snapshot = collect(root, campaign, workers_active=still_waiting)
                write_report(out, snapshot)
                if snapshot['snapshot_sha256'] != previous:
                    result = publish_with_retry(lambda: publish_once(wandb, state, campaign, snapshot, out), out)
                    previous = snapshot['snapshot_sha256']
                    print(f'{snapshot["complete"]}/{snapshot["total"]} episodes complete; {result["url"]}', flush=True)
                _write(out / 'publisher-status.json', dict(status=snapshot['state'], completed=snapshot['complete'],
                    total=snapshot['total'], failed=snapshot['failed'], pending=snapshot['pending'], run_id=state['run_id']))
                if snapshot['state'] == 'complete' or args.once:
                    return
                if not still_waiting:
                    raise BypassDataError('Workers stopped before all requested episodes passed integrity checks.')
                time.sleep(args.poll_seconds)
            except BaseException as exc:
                _write(out / 'publisher-failure.json', dict(error_type=type(exc).__name__, error=str(exc), run_id=state['run_id']))
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
