"""Publish closed-loop critic comparisons as visible returns and paired J curves.

The overview contains only completed, validated episodes. It is separate from
the authoritative per-planner evaluation runs and their full training traces.
An evaluated point may be visible before artifact publication finishes; its
status explicitly distinguishes those two events.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slurm.ambi_aux_hj_sweep import (ENTITY, PROJECT, SEEDS, actor_updates,
                                    critic_updates, digest, publish_cell, read,
                                    validate, write)

ARMS = {'soft': 'Soft critic', 'return_only': 'Return-only critic'}


def moments(values):
    values = list(values)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError('Expected finite, nonempty episode measurements')
    return {'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.,
            'episodes': len(values)}


def indexed_episodes(episodes, seeds=SEEDS):
    """Pair by actual environment and solver seeds, never by list position."""
    result = {(e['seed'], e['solver_seed']): e for e in episodes}
    if (len(result) != len(episodes) or len(result) != len(seeds)
            or sorted(e['seed'] for e in episodes) != sorted(seeds)):
        raise ValueError('Missing, repeated, or unexpected paired episode seed')
    for episode in episodes:
        if (episode['length'] != 500 or episode.get('truncated_by_evaluator') is not False
                or not math.isfinite(episode['return'])):
            raise ValueError('Overview accepts only finite, full 500-decision episodes')
    return result


def aggregate_results(campaign, prior_episodes, completed):
    """Pure comparison payload used by both native history and plot tables."""
    prior = indexed_episodes(prior_episodes)
    cells = {c['name']: c for c in campaign['cells']}
    points, rows, direct, by_pair = [], [], [], {}
    for name, episodes in completed.items():
        cell = cells[name]
        arm, j = cell['critic_kind'], cell['J']
        if arm not in ARMS or (arm, j) in by_pair:
            raise ValueError('Expected one completed result per critic and J')
        indexed = indexed_episodes(episodes)
        if indexed.keys() != prior.keys():
            raise ValueError('Prior and evaluation solver seeds differ')
        gains = []
        for key in sorted(prior):
            ep, baseline = indexed[key], prior[key]['return']
            gain = ep['return'] - baseline
            if not math.isclose(ep['paired_return_delta'], gain, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError('Recorded prior gain does not match paired episode returns')
            gains.append(gain)
            rows.append({'setting': name, 'critic': arm, 'J': j, 'seed': key[0],
                         'solver_seed': key[1], 'return': ep['return'],
                         'prior_return': baseline, 'paired_gain': gain})
        points.append({'setting': name, 'critic': arm, 'J': j,
                       'return': moments(e['return'] for e in indexed.values()),
                       'paired_gain': moments(gains)})
        by_pair[arm, j] = indexed
    for j in sorted({c['J'] for c in cells.values()}):
        if all((arm, j) in by_pair for arm in ARMS):
            soft, ret = by_pair['soft', j], by_pair['return_only', j]
            paired = [{'J': j, 'seed': key[0], 'solver_seed': key[1],
                       'soft_return': soft[key]['return'], 'return_only_return': ret[key]['return'],
                       'return_minus_soft': ret[key]['return'] - soft[key]['return']}
                      for key in sorted(prior)]
            direct.append({'J': j, 'difference': moments(e['return_minus_soft'] for e in paired),
                           'episodes': paired})
    return {'prior': moments(e['return'] for e in prior.values()),
            'prior_episodes': [prior[key] for key in sorted(prior)],
            'points': sorted(points, key=lambda p: (p['critic'], p['J'])),
            'episodes': sorted(rows, key=lambda p: (p['critic'], p['J'], p['seed'])),
            'direct': direct}


def chart_payloads(campaign, aggregate):
    """Keep the critic arms separate, omit unevaluated J, and label baseline."""
    result = {}
    for metric, title, yname in (
            ('return', 'Closed-loop episode return vs inner rounds', 'Episode return'),
            ('paired_gain', 'Paired improvement over frozen prior', 'Return minus paired prior')):
        xs, ys, keys = [], [], []
        for arm, label in ARMS.items():
            points = [p for p in aggregate['points'] if p['critic'] == arm]
            if points:
                xs.append([p['J'] for p in points])
                ys.append([p[metric]['mean'] for p in points])
                keys.append(label)
        # The same observed prior applies to every J; this is an explicit
        # horizontal reference, never a fabricated inner-solve measurement.
        js = sorted({c['J'] for c in campaign['cells']})
        xs.append(js)
        ys.append([aggregate['prior']['mean'] if metric == 'return' else 0.] * len(js))
        keys.append('Frozen prior (reused)' if metric == 'return' else 'No improvement')
        result['comparison/' + metric + '_vs_J'] = dict(xs=xs, ys=ys, keys=keys, title=title, xname='Inner rounds J', yname=yname)
    if aggregate['direct']:
        result['comparison/return_minus_soft_vs_J'] = dict(
            xs=[[p['J'] for p in aggregate['direct']]],
            ys=[[p['difference']['mean'] for p in aggregate['direct']]],
            keys=['Return-only minus soft (paired)'],
            title='Critic effect: return-only minus soft', xname='Inner rounds J', yname='Paired return difference')
    return result


def numeric_rows(aggregate):
    """One native numeric point per setting, with disjoint critic namespaces."""
    rows = []
    for p in aggregate['points']:
        prefix = 'closed_loop/' + p['critic']
        rows.append((p['setting'], {'axis/inner_rounds': p['J'],
                     **{f'{prefix}/{metric}_{stat}': value
                        for metric in ('return', 'paired_gain')
                        for stat, value in p[metric].items()},
                     'closed_loop/prior/return_mean': aggregate['prior']['mean'],
                     'closed_loop/prior/return_std': aggregate['prior']['std']}))
    for p in aggregate['direct']:
        rows.append(('paired_J' + str(p['J']), {'axis/inner_rounds': p['J'],
                     **{'closed_loop/return_minus_soft/' + stat: value
                        for stat, value in p['difference'].items()}}))
    return rows


def overview_log(wandb, campaign, aggregate, statuses):
    """Rich plots plus exact per-seed values; no history-row index as science x."""
    result = {}
    for key, payload in chart_payloads(campaign, aggregate).items():
        # SDK 0.17.4 line_series has a single x label and no y-label argument.
        options = {k: v for k, v in payload.items() if k != 'yname'}
        result[key] = wandb.plot.line_series(**options)
    def table(rows, columns):
        return wandb.Table(columns=columns, data=[[r.get(k) for k in columns] for r in rows])
    result['comparison/paired_episodes'] = table(aggregate['episodes'],
        ['setting', 'critic', 'J', 'seed', 'solver_seed', 'return', 'prior_return', 'paired_gain'])
    result['comparison/prior_episodes'] = table(aggregate['prior_episodes'], ['seed', 'solver_seed', 'return', 'length'])
    result['comparison/critic_difference_episodes'] = table(
        [ep for point in aggregate['direct'] for ep in point['episodes']],
        ['J', 'seed', 'solver_seed', 'soft_return', 'return_only_return', 'return_minus_soft'])
    result['campaign/settings'] = table(statuses,
        ['setting', 'critic', 'J', 'status', 'return_mean', 'return_std', 'paired_gain_mean', 'paired_gain_std',
         'performance_url', 'training_url', 'failure'])
    result['campaign/evaluated'] = len(aggregate['points'])
    result['campaign/published'] = sum(r['status'] == 'published' for r in statuses)
    result['campaign/failed_publications'] = sum(r['status'] == 'publication_failed' for r in statuses)
    return result


def load_completed(campaign, cell):
    directory, bundle = Path(cell['directory']), Path(cell['bundle'])
    receipt = read(directory / 'worker-completion.json')
    if receipt['status'] != 'complete' or receipt['cell'] != cell['name']:
        raise ValueError('Worker completion identity mismatch')
    if digest(bundle / 'manifest.json') != receipt['manifest_sha256']:
        raise ValueError('Completed manifest changed after validation')
    manifest = validate(bundle, cell, checkpoint_step=campaign['checkpoint_step'],
                        checkpoint_sha=campaign['checkpoint_sha256'])
    return manifest['runs'][0]['episodes']


def watch(args):
    import wandb
    campaign = read(args.root / 'campaign.json')
    prior_manifest = Path(campaign['reference']) / 'manifest.json'
    if digest(prior_manifest) != campaign['prior_manifest_sha256']:
        raise ValueError('Verified prior manifest changed')
    prior_runs = [r for r in read(prior_manifest)['runs'] if r.get('episodes')
                  and r.get('config', {}).get('alg_params', {}).get('inner_operator') == 'none']
    prior_run, = prior_runs
    aggregate_results(campaign, prior_run['episodes'], {})
    marker = args.root / 'watcher-started.json'
    if marker.exists():
        raise RuntimeError('Overview already started; inspect its W&B history before recovery')
    write(marker, {'pid': os.getpid(), 'started': time.time()})
    publishers = int(campaign.get('publisher_workers', 2))
    if not 1 <= publishers <= 4:
        raise ValueError('Expected 1–4 CPU publication subprocesses')
    urls = {c['name']: {kind + '_url': f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{c[kind + "_run_id"]}'
                        for kind in ('performance', 'training')} for c in campaign['cells']}
    config = {key: campaign.get(key) for key in ('checkpoint_step', 'checkpoint_sha256', 'source_run',
              'source_commit', 'initial_alpha', 'target_entropy', 'prior_manifest_sha256', 'prior_source_science')}
    config.update(campaign_group=campaign['group'], protocol='closed-loop-refinement-v1', protocol_variant='SAC auxiliary-critic comparison; adaptive actor entropy',
                  J=sorted({c['J'] for c in campaign['cells']}), H=3, N=128, B=256, C=16, A=4,
                  environment_seeds=SEEDS, controller_seed=55, max_decisions=500,
                  execution='Fresh adaptation at every real decision, then deterministic actor mean action',
                  prior_reference=str(campaign['reference']),
                  critic_comparison={'soft': 'SAC soft-Q initialization, entropy-augmented fitting, soft-Q terminal with entropy correction',
                                     'return_only': 'Auxiliary return-Q initialization, reward-only fitting, return-Q terminal'},
                  uncertainty='Sample standard deviation across five paired episodes; exploratory screen',
                  result_links=urls)
    run = wandb.init(entity=ENTITY, project=PROJECT, id=campaign['overview_run_id'], resume='never',
                     name=campaign.get('label', 'Closed-loop critic comparison at 575k'),
                     group=campaign['group'], job_type='closed-loop-comparison',
                     tags=['closed-loop', 'soft-vs-return-only', 'target10p5-shared'], config=config, mode='online')
    run.define_metric('axis/inner_rounds')
    run.define_metric('closed_loop/*', step_metric='axis/inner_rounds')
    run.summary.update(dict(status='running', result_type='Full-episode closed-loop environment returns',
                            evaluated=0, published=0, total_settings=len(campaign['cells'])))
    attempted, completed, futures, failures, logged = set(), {}, {}, {}, set()
    previous, terminal_since = None, None

    def launch(index):
        cell = campaign['cells'][index]
        with (Path(cell['directory']) / 'publisher.log').open('w') as log:
            return subprocess.run([sys.executable, __file__, 'publish', '--root', str(args.root), '--index', str(index)],
                                  stdout=log, stderr=subprocess.STDOUT).returncode

    def status_rows(aggregate, terminal=False):
        points = {p['setting']: p for p in aggregate['points']}
        rows = []
        for index, cell in enumerate(campaign['cells']):
            directory, name = Path(cell['directory']), cell['name']
            publication = directory / 'publication-completion.json'
            if publication.exists():
                receipt = read(publication)
                if receipt['status'] != 'complete' or receipt['cell'] != name:
                    raise ValueError('Publication completion identity mismatch')
                state = 'published'
            elif index in failures:
                state = 'publication_failed'
            elif index in futures:
                state = 'publishing'
            elif name in completed:
                state = 'evaluated_awaiting_publication'
            else:
                state = 'evaluation_incomplete' if terminal else 'queued_or_running'
            point = points.get(name, {})
            rows.append(dict(setting=name, critic=cell['critic_kind'], J=cell['J'], status=state,
                             return_mean=point.get('return', {}).get('mean'),
                             return_std=point.get('return', {}).get('std'),
                             paired_gain_mean=point.get('paired_gain', {}).get('mean'),
                             paired_gain_std=point.get('paired_gain', {}).get('std'),
                             failure=failures.get(index), **urls[name]))
        return rows

    def update(terminal=False):
        nonlocal previous
        aggregate = aggregate_results(campaign, prior_run['episodes'], completed)
        statuses = status_rows(aggregate, terminal)
        stamp = [(r['setting'], r['status']) for r in statuses]
        if stamp != previous:
            for identity, row in numeric_rows(aggregate):
                if identity not in logged:
                    run.log(row)
                    logged.add(identity)
            run.log(overview_log(wandb, campaign, aggregate, statuses))
            run.summary.update(dict(evaluated=len(completed), published=sum(r['status'] == 'published' for r in statuses),
                                    prior_return_mean=aggregate['prior']['mean'], prior_return_std=aggregate['prior']['std']))
            write(args.root / 'comparison-results.json', aggregate)
            write(args.root / 'progress.json', dict(rows=statuses, failures=failures, evaluated=len(completed)))
            print(f'Evaluated {len(completed)}/{len(statuses)}; published {sum(r["status"] == "published" for r in statuses)}', flush=True)
            previous = stamp
        return statuses

    try:
        update()  # Baseline, plot panels, complete settings table exist at launch.
        with ThreadPoolExecutor(max_workers=publishers) as pool:
            while True:
                for index, future in list(futures.items()):
                    if future.done():
                        rc = future.result()
                        if rc:
                            failures[index] = f'Publisher exited {rc}; inspect {campaign["cells"][index]["directory"]}/publisher.log'
                        del futures[index]
                for index, cell in enumerate(campaign['cells']):
                    directory = Path(cell['directory'])
                    if (directory / 'worker-completion.json').exists() and cell['name'] not in completed:
                        completed[cell['name']] = load_completed(campaign, cell)
                    if (len(futures) < publishers and index not in attempted and cell['name'] in completed
                            and not (directory / 'publication-completion.json').exists()):
                        attempted.add(index)
                        futures[index] = pool.submit(launch, index)
                statuses = update()
                if all(r['status'] == 'published' for r in statuses):
                    break
                submission = args.root / 'submission.json'
                if submission.exists():
                    ids = [str(j) for j in read(submission)['gpu_job_ids']]
                    active = subprocess.check_output(['squeue', '--noheader', '--jobs', ','.join(ids), '-o', '%i'], text=True).strip()
                    if not active and not futures:
                        terminal_since = terminal_since or time.time()
                        if time.time() - terminal_since > 90:
                            break
                    else:
                        terminal_since = None
                time.sleep(15)
        statuses = update(terminal=True)
        complete = all(r['status'] == 'published' for r in statuses)
        status = 'complete' if complete else 'incomplete'
        run.summary.update(dict(status=status, failed_publications=len(failures)))
        write(args.root / 'campaign-completion.json', dict(status=status, rows=statuses, failures=failures))
        if not complete:
            raise RuntimeError('Closed-loop campaign incomplete; inspect visible status table and publisher logs')
        run.finish()
    except BaseException as exc:
        run.summary.update(dict(status='failed', failure=str(exc)))
        run.finish(exit_code=1)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['watch', 'publish'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    (watch if args.mode == 'watch' else publish_cell)(args)


if __name__ == '__main__':
    main()
