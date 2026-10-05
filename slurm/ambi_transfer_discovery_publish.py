"""Publish complete discovery panels while keeping partial episodes as progress."""
from __future__ import annotations

import argparse
import errno
import json
import math
from pathlib import Path
import statistics
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_transfer_discovery_campaign import (
    digest as _digest, read as _read, validate_result as _validate_result,
)
from slurm.ambi_transfer_sweep_publish import publisher_lock
from slurm.ambi_closed_loop_publish import gpu_jobs_active
from utils.ambi_benchmark import atomic_json

ENTITY = 'rwgao_b-brown-university'
PROJECT = 'ambi-inner-bench'
PUBLICATION_STATE_VERSION = 2
ESTALE_RETRY_DELAYS = (1, 2, 4, 8)
COLUMNS = {
    'settings': ['name','H','J','arm','state','completed_episodes','current_seed','decision','error'],
    'results': ['name','H','J','arm','state','return_mean','return_std','paired_vs_fresh_mean',
                'paired_vs_fresh_std','controller_seconds_per_decision','episodes'],
    'episodes': ['name','H','J','arm','seed','solver_seed','return','length','control_seconds'],
}


def _retry_estale(operation, *args, **kwargs):
    """Retry only transient stale file handles, at most five attempts total."""
    for attempt in range(len(ESTALE_RETRY_DELAYS) + 1):
        try:
            return operation(*args, **kwargs)
        except OSError as error:
            if error.errno != errno.ESTALE or attempt == len(ESTALE_RETRY_DELAYS):
                raise
            delay = ESTALE_RETRY_DELAYS[attempt]
            print(f'Stale file handle; retrying local publication I/O in {delay}s '
                  f'(attempt {attempt + 2}/5).', file=sys.stderr, flush=True)
            time.sleep(delay)


def read(path):
    return _retry_estale(_read, path)


def digest(path):
    return _retry_estale(_digest, path)


def write(path, value):
    return _retry_estale(atomic_json, Path(path), value, overwrite=True)


def publication_state(root, out, campaign, *, entity, project):
    """Resume only the same destination and immutable evaluation campaign.

    The caller holds the publication-root lock. Publisher code may evolve while
    evaluation stays pinned: source_commit identifies the evaluation campaign,
    not the reporting checkout. Old state without a destination is ambiguous
    and requires a new publication root rather than migration in place.
    """
    expected = dict(schema_version=PUBLICATION_STATE_VERSION, entity=entity, project=project,
        campaign_sha256=digest(root/'campaign.json'), source_commit=campaign['source_commit'],
        campaign_root=str(root))
    state_path = out/'publication.json'
    if _retry_estale(state_path.exists):
        state = read(state_path)
        if state.get('schema_version') != PUBLICATION_STATE_VERSION or not all(
                field in state for field in ('entity', 'project')):
            raise ValueError('Legacy publication state has no verified destination binding; '
                             'use a new publication root.')
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError('Publication destination or campaign binding differs; '
                             'use a new publication root.')
        if not isinstance(state.get('run_id'), str) or not state['run_id']:
            raise ValueError('Publication state has no valid run identity.')
        return state
    state = dict(expected, run_id=uuid.uuid4().hex)
    write(state_path, state)
    return state


def episode_time(episode):
    return episode['control_seconds']


def _snapshot_once(root, campaign, active, completed):
    settings, results, episodes = [], [], []
    for cell in campaign['cells']:
        name = cell['name']; directory = root/'settings'/name
        progress = _read(directory/'progress.json') if (directory/'progress.json').exists() else {}
        failure = _read(directory/'worker-failure.json') if (directory/'worker-failure.json').exists() else {}
        if name not in completed and (directory/'worker-completion.json').exists():
            receipt = _read(directory/'worker-completion.json')
            if (receipt['campaign_sha256'] != _digest(root/'campaign.json')
                    or receipt['result_sha256'] != _digest(directory/'results.json')
                    or receipt['manifest_sha256'] != _digest(directory/'manifest.json')
                    or receipt['source_commit'] != campaign['source_commit']
                    or receipt['cell'] != cell or receipt['smoke'] is not False
                    or receipt['status'] != 'complete'):
                raise ValueError('Completed cell receipt binding differs: ' + name)
            result, _ = _validate_result(directory, campaign, cell)
            completed[name] = result['episodes']
        state = 'complete' if name in completed else 'failed' if failure else 'running' if directory.exists() and active else 'pending' if active else 'incomplete'
        data = completed.get(name)
        settings.append(dict(cell, state=state, completed_episodes=len(data) if data else progress.get('completed_episodes', 0),
            current_seed=progress.get('seed'), decision=progress.get('decision'), error=failure.get('error')))
        row = dict(cell, state=state, return_mean=None, return_std=None, paired_vs_fresh_mean=None,
                   paired_vs_fresh_std=None, controller_seconds_per_decision=None, episodes=0)
        if data:
            returns = [e['return'] for e in data]
            row.update(return_mean=statistics.mean(returns), return_std=statistics.stdev(returns), episodes=len(data),
                controller_seconds_per_decision=sum(episode_time(e) for e in data)/sum(e['length'] for e in data))
            for episode in data:
                episodes.append(dict(cell, **episode))
        results.append(row)
    fresh = {(c['H'], c['J']): completed[c['name']] for c in campaign['cells']
             if c['arm'] == 'rho_a0_c0' and c['name'] in completed}
    for row in results:
        data = completed.get(row['name']); reference = fresh.get((row['H'], row['J']))
        if data and reference:
            lookup = {(e['seed'],e['solver_seed']): e['return'] for e in reference}
            if {(e['seed'],e['solver_seed']) for e in data} != set(lookup):
                raise ValueError('Solver/environment seed pairs differ.')
            differences = [e['return']-lookup[(e['seed'],e['solver_seed'])] for e in data]
            row.update(paired_vs_fresh_mean=statistics.mean(differences), paired_vs_fresh_std=statistics.stdev(differences))
    return dict(settings=settings, results=results, episodes=episodes, completed=len(completed), total=len(settings),
                failed=sum(s['state']=='failed' for s in settings))


def snapshot(root, campaign, active, completed):
    # One retry boundary covers stat, receipt and result reads without nesting
    # per-file retry budgets. Already verified immutable results may stay cached.
    return _retry_estale(_snapshot_once, root, campaign, active, completed)


def payload(wandb, value, campaign):
    result = {'discovery/'+key: wandb.Table(columns=columns, data=[[row.get(k) for k in columns] for row in value[key]])
              for key, columns in COLUMNS.items()}
    arms = list(dict.fromkeys(c['arm'] for c in campaign['cells']))
    for h in campaign['H']:
        for axis, field in [('j','J'), ('compute','controller_seconds_per_decision')]:
            series = [sorted([r for r in value['results'] if r['H']==h and r['arm']==arm and r['return_mean'] is not None],
                             key=lambda r:r[field]) for arm in arms]
            result[f'discovery/h{h}_return_vs_{axis}'] = wandb.plot.line_series(
                xs=[[r[field] for r in group] for group in series],
                ys=[[r['return_mean'] for r in group] for group in series], keys=arms,
                title=f'H{h}: complete three-seed return versus {axis}',
                xname='J rounds per solve' if axis=='j' else 'Controller seconds per decision')
    result.update({'campaign/completed':value['completed'],'campaign/total':value['total'],'campaign/failed':value['failed']})
    return result


def _record_failure(run, out, error):
    """Keep a secondary reporting outage from replacing the original failure."""
    reports = (
        ('failure receipt', lambda: write(out/'publisher-failure.json',
            dict(error_type=type(error).__name__, error=str(error)))),
        ('run summary', lambda: run.summary.update(
            {'status':'publisher_failed', 'publisher/error':str(error)})),
        ('run finish', lambda: run.finish(exit_code=1)),
    )
    for label, report in reports:
        try:
            report()
        except Exception as reporting_error:
            print(f'Could not update {label}: {reporting_error}. '
                  f'Original publisher failure: {type(error).__name__}: {error}',
                  file=sys.stderr, flush=True)


def watch(args):
    import wandb
    from utils.wandb_transfer_discovery_layout import ensure_discovery_saved_view
    root, out = args.root.resolve(), args.publication_root.resolve()
    if out.is_relative_to(root):
        raise ValueError('Publication state must be separate from evaluation outputs.')
    campaign = read(root/'campaign.json')
    with publisher_lock(out):
        state = publication_state(root, out, campaign, entity=args.entity, project=args.project)
        run = wandb.init(entity=state['entity'], project=state['project'], id=state['run_id'], resume='allow', mode='online',
            name='575K transfer discovery | 14 mechanisms | H1/2/3 J1/2/4/6/8/10',
            group='transfer-discovery-575k-'+state['run_id'][:8], job_type='transfer-discovery-overview',
            tags=['575k','transfer-discovery','full-episode','development-screen'],
            config={**{k:v for k,v in campaign.items() if k!='cells'},
                    'publication_id':state['run_id']})
        completed = {}; previous = None; terminal_since = None
        try:
            initial = snapshot(root, campaign, True, completed)
            run.log(payload(wandb, initial, campaign))
            layout = ensure_discovery_saved_view(wandb.Api(timeout=30), entity=state['entity'], project=state['project'],
                receipt_dir=out/'results-layout', run_id=run.id)
            run.summary.update({'results_layout/status':layout['status'],'results_layout/url':layout['url'],
                                'status':'running','completed_settings':len(completed),'total_settings':len(campaign['cells'])})
            print('LIVE OVERVIEW '+layout['url'], flush=True)
            while True:
                active = gpu_jobs_active(args.gpu_job_id)
                value = snapshot(root, campaign, active, completed)
                stamp = json.dumps(value, sort_keys=True)
                if stamp != previous:
                    write(out/'snapshot.json', value)
                    run.log(payload(wandb, value, campaign))
                    run.summary.update({'status':'running' if active else 'publishing',
                        'completed_settings':value['completed'],'total_settings':value['total'],'failed_settings':value['failed']})
                    previous = stamp
                    print(f"Complete {value['completed']}/{value['total']}; failed {value['failed']}", flush=True)
                if value['completed'] == value['total']:
                    break
                if not active:
                    terminal_since = terminal_since or time.time()
                    if time.time()-terminal_since >= 120:
                        break
                else:
                    terminal_since = None
                if args.once:
                    break
                time.sleep(args.poll_seconds)
            complete = value['completed']==value['total']
            status = 'complete' if complete else 'running' if active else 'incomplete'
            write(out/'publisher-status.json', dict(status=status, overview_url=layout['url'], completed=value['completed']))
            run.summary['status'] = status
            if complete:
                artifact = wandb.Artifact('transfer-discovery-'+state['run_id'], type='transfer-discovery-results',
                    metadata={'checkpoint_step':575000,'source_commit':campaign['source_commit']})
                artifact.add_file(str(root/'campaign.json'))
                artifact.add_file(str(out/'snapshot.json'))
                for cell in campaign['cells']:
                    for filename in ('manifest.json','results.json','worker-completion.json'):
                        artifact.add_file(str(root/'settings'/cell['name']/filename), name=cell['name']+'/'+filename)
                run.log_artifact(artifact).wait()
            run.finish(exit_code=0 if complete or active else 1)
        except BaseException as error:
            _record_failure(run, out, error)
            raise


def destination_name(value):
    if not value or value != value.strip() or '/' in value or '\\' in value or any(char.isspace() for char in value):
        raise argparse.ArgumentTypeError('W&B entity/project must be a nonempty name without whitespace or slashes.')
    return value


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--publication-root', type=Path, required=True)
    parser.add_argument('--entity', type=destination_name, default=ENTITY)
    parser.add_argument('--project', type=destination_name, default=PROJECT)
    parser.add_argument('--gpu-job-id', action='append', required=True)
    parser.add_argument('--poll-seconds', type=float, default=60)
    parser.add_argument('--once', action='store_true')
    return parser


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.poll_seconds <= 0:
        argument_parser.error('poll-seconds must be positive')
    watch(args)


if __name__ == '__main__':
    main()
