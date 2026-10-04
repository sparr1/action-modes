"""Publish complete discovery panels while keeping partial episodes as progress."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_transfer_discovery_campaign import digest, read, validate_result
from slurm.ambi_transfer_sweep_publish import publisher_lock
from slurm.ambi_closed_loop_publish import gpu_jobs_active
from utils.ambi_benchmark import atomic_json

ENTITY = 'rwgao_b-brown-university'
PROJECT = 'ambi'
COLUMNS = {
    'settings': ['name','H','J','arm','state','completed_episodes','current_seed','decision','error'],
    'results': ['name','H','J','arm','state','return_mean','return_std','paired_vs_fresh_mean',
                'paired_vs_fresh_std','controller_seconds_per_decision','episodes'],
    'episodes': ['name','H','J','arm','seed','solver_seed','return','length','control_seconds'],
}


def write(path, value):
    atomic_json(Path(path), value, overwrite=True)


def episode_time(episode):
    return episode['control_seconds']


def snapshot(root, campaign, active, completed):
    settings, results, episodes = [], [], []
    for cell in campaign['cells']:
        name = cell['name']; directory = root/'settings'/name
        progress = read(directory/'progress.json') if (directory/'progress.json').exists() else {}
        failure = read(directory/'worker-failure.json') if (directory/'worker-failure.json').exists() else {}
        if name not in completed and (directory/'worker-completion.json').exists():
            receipt = read(directory/'worker-completion.json')
            if (receipt['campaign_sha256'] != digest(root/'campaign.json')
                    or receipt['result_sha256'] != digest(directory/'results.json')
                    or receipt['manifest_sha256'] != digest(directory/'manifest.json')
                    or receipt['source_commit'] != campaign['source_commit']
                    or receipt['cell'] != cell or receipt['smoke'] is not False
                    or receipt['status'] != 'complete'):
                raise ValueError('Completed cell receipt binding differs: ' + name)
            result, _ = validate_result(directory, campaign, cell)
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


def watch(args):
    import wandb
    from utils.wandb_transfer_discovery_layout import ensure_discovery_results_layout
    root, out = args.root.resolve(), args.publication_root.resolve()
    if out.is_relative_to(root):
        raise ValueError('Publication state must be separate from evaluation outputs.')
    campaign = read(root/'campaign.json')
    with publisher_lock(out):
        state_path = out/'publication.json'
        if state_path.exists():
            state = read(state_path)
            if state['campaign_sha256'] != digest(root/'campaign.json'):
                raise ValueError('Publisher campaign changed.')
        else:
            state = dict(campaign_sha256=digest(root/'campaign.json'), run_id=uuid.uuid4().hex,
                         source_commit=campaign['source_commit'], campaign_root=str(root))
            write(state_path, state)
        run = wandb.init(entity=ENTITY, project=PROJECT, id=state['run_id'], resume='allow', mode='online',
            name='575K transfer discovery | 14 mechanisms | H1/2/3 J1/2/4/6/8/10',
            group='transfer-discovery-575k-'+state['run_id'][:8], job_type='transfer-discovery-overview',
            tags=['575k','transfer-discovery','full-episode','development-screen'],
            config={k:v for k,v in campaign.items() if k!='cells'})
        completed = {}; previous = None; terminal_since = None
        try:
            initial = snapshot(root, campaign, True, completed)
            run.log(payload(wandb, initial, campaign))
            layout = ensure_discovery_results_layout(wandb.Api(timeout=30), entity=ENTITY, project=PROJECT,
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
            write(out/'publisher-failure.json', dict(error_type=type(error).__name__, error=str(error)))
            run.summary.update({'status':'publisher_failed','publisher/error':str(error)})
            run.finish(exit_code=1)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--publication-root', type=Path, required=True)
    parser.add_argument('--gpu-job-id', action='append', required=True)
    parser.add_argument('--poll-seconds', type=float, default=60)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error('poll-seconds must be positive')
    watch(args)


if __name__ == '__main__':
    main()
