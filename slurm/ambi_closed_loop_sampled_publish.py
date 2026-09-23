"""Visible sampled-versus-mean full-episode comparisons with complete diagnostics."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_aux_hj_sweep import (ENTITY, PROJECT, SEEDS, actor_updates, critic_updates,
                                    digest, publish_performance, read, training_summary, write)
from slurm.ambi_closed_loop_publish import gpu_jobs_active, indexed_episodes, moments
from slurm.ambi_closed_loop_sampled import (ACTION_RULE, CHECKPOINT_SHA, validate_completed,
                                           verify_mean_reference, verify_receipt)

def paired_comparison(sampled_episodes, mean_episodes):
    """Estimate the execution change at paired environment/controller seeds."""
    import numpy as np
    sampled, mean = indexed_episodes(sampled_episodes), indexed_episodes(mean_episodes)
    if sampled.keys() != mean.keys():
        raise ValueError('Sampled and mean execution solver seeds differ')
    rows = [dict(seed=k[0], solver_seed=k[1], mean_return=mean[k]['return'],
                 sampled_return=sampled[k]['return'], sample_minus_mean=sampled[k]['return']-mean[k]['return'])
            for k in sorted(mean)]
    delta = np.asarray([r['sample_minus_mean'] for r in rows])
    draws = np.random.default_rng(20260912).integers(0, 5, size=(2000, 5))
    low, high = np.percentile(delta[draws].mean(axis=1), [2.5, 97.5])
    return dict(rows=rows, bootstrap_seed=20260912, bootstrap_resamples=2000,
                metrics={'comparison/sample_minus_mean_mean': float(delta.mean()),
                         'comparison/sample_minus_mean_std': float(delta.std(ddof=1)),
                         'comparison/sample_minus_mean_ci95_low': float(low),
                         'comparison/sample_minus_mean_ci95_high': float(high),
                         'comparison/sample_minus_mean_paired_episodes': len(delta)})


def execution_proof(manifest):
    metrics = manifest['runs'][0]['result']['model_metrics']
    return {f'execution/{label}_{stat}': metrics[key][stat]
            for key, label in [('inner_eval_execution_sampled', 'sample_flag'),
                               ('inner_eval_execution_mean_action_l2', 'mean_action_l2')]
            for stat in ('mean', 'min', 'max')}


def aggregate_results(campaign, completed):
    points, mean_points, rows, means = [], [], [], []
    names, rounds = set(), set()
    for cell in sorted(campaign['cells'], key=lambda c: c['J']):
        name, j = cell['name'], cell['J']
        if name in names or j in rounds:
            raise ValueError('Expected exactly one return-only execution comparison per J')
        names.add(name); rounds.add(j)
        mean = cell['mean_reference']['episodes']
        indexed_episodes(mean)
        mean_points.append(dict(J=j, setting=cell['mean_name'], execution='mean',
                                return_stats=moments(e['return'] for e in mean)))
        means.extend(dict(J=j, setting=cell['mean_name'], **{k:e[k] for k in ('seed', 'solver_seed', 'return', 'length')})
                     for e in mean)
        if name in completed:
            comparison = paired_comparison(completed[name], mean)
            points.append(dict(J=j, setting=name, execution='policy_sample',
                               return_stats=moments(e['return'] for e in completed[name]),
                               difference=moments(r['sample_minus_mean'] for r in comparison['rows']),
                               comparison_metrics=comparison['metrics']))
            rows.extend(dict(J=j, setting=name, **r) for r in comparison['rows'])
    if set(completed) - names:
        raise ValueError('Unexpected completed sampled setting')
    return dict(points=points, mean_points=mean_points, episodes=rows, means=means,
                comparison='Sampled minus historical mean execution; no prior-relative interpretation.',
                uncertainty='Five matched environment/controller seeds; exploratory bootstrap intervals.')


def numeric_rows(aggregate):
    rows = []
    for point in aggregate['mean_points']:
        rows.append((f'mean_J{point["J"]}', {'axis/inner_rounds': point['J'],
                    **{f'sampled_execution/mean/return_{key}': value for key, value in point['return_stats'].items()}}))
    for point in aggregate['points']:
        rows.append((f'sampled_J{point["J"]}', {'axis/inner_rounds': point['J'],
                    **{f'sampled_execution/sample/return_{key}': value for key, value in point['return_stats'].items()},
                    **{f'sampled_execution/sample_minus_mean/{key}': value for key, value in point['difference'].items()},
                    **{f'sampled_execution/sample_minus_mean/{key.removeprefix("comparison/sample_minus_mean_")}': value
                       for key, value in point['comparison_metrics'].items()}}))
    return rows


def chart_payloads(aggregate):
    mean, sampled = aggregate['mean_points'], aggregate['points']
    result = {'comparison/return_vs_J': dict(
        xs=[[p['J'] for p in mean], [p['J'] for p in sampled]],
        ys=[[p['return_stats']['mean'] for p in mean], [p['return_stats']['mean'] for p in sampled]],
        keys=['Historical mean execution', 'Sampled execution'],
        title='H3 return-only: full-episode return by execution mode', xname='Inner rounds J')}
    if sampled:
        result['comparison/sample_minus_mean_vs_J'] = dict(
            xs=[p['J'] for p in sampled], ys=[[p['difference']['mean'] for p in sampled]],
            keys=['Sampled minus historical mean'], title='Paired execution improvement (same J and seeds)',
            xname='Inner rounds J')
    return result


def overview_log(wandb, aggregate, statuses):
    result = {key: wandb.plot.line_series(**payload) for key, payload in chart_payloads(aggregate).items()}
    def table(rows, columns):
        return wandb.Table(columns=columns, data=[[r.get(k) for k in columns] for r in rows])
    result['comparison/paired_episodes'] = table(aggregate['episodes'],
        ['setting', 'J', 'seed', 'solver_seed', 'mean_return', 'sampled_return', 'sample_minus_mean'])
    result['comparison/historical_mean_episodes'] = table(aggregate['means'],
        ['setting', 'J', 'seed', 'solver_seed', 'return', 'length'])
    result['campaign/settings'] = table(statuses,
        ['setting', 'J', 'execution', 'status', 'return_mean', 'return_std', 'sample_minus_mean',
         'sample_minus_mean_ci95_low', 'sample_minus_mean_ci95_high',
         'performance_url', 'training_url', 'historical_mean_url', 'failure'])
    result['campaign/evaluated'] = len(aggregate['points'])
    result['campaign/published'] = sum(r['status'] == 'published' for r in statuses)
    result['campaign/failed_publications'] = sum(r['status'] == 'publication_failed' for r in statuses)
    return result


def load_completed(campaign, cell):
    directory, bundle = Path(cell['directory']), Path(cell['bundle'])
    receipt = read(directory / 'worker-completion.json')
    assert receipt['cell'] == cell['name'] and receipt['execution'] == 'policy_sample'
    verify_receipt(bundle, receipt)
    manifest = validate_completed(bundle, cell, campaign)
    return manifest['runs'][0]['episodes'], execution_proof(manifest)


def publish_cell(args):
    from utils.ambi_benchmark import stage_completed_bundle
    from utils.eval_series import load_run
    from utils.eval_series_data import load_records
    from utils.ambi_diagnostic_series import record_from_model_bundle, write_diagnostic_bundle, diagnostic_history
    import wandb
    campaign = read(args.root/'campaign.json'); cell = campaign['cells'][args.index]
    directory, bundle = Path(cell['directory']), Path(cell['bundle'])
    if publication_complete(cell): return
    receipt = read(directory/'worker-completion.json')
    verify_receipt(bundle, receipt)
    assert receipt['cell'] == cell['name'] and receipt['execution'] == 'policy_sample'
    checkpoint_step = campaign.get('checkpoint_step', 625000)
    manifest = validate_completed(bundle, cell, campaign)
    record, = load_records(bundle,inventory_path=campaign['inventory'])
    assert record['identity'] == load_run(cell['run_dir'])['identity']
    assert record['metrics'].get('eval/paired_episodes', 0) == 0
    assert all('paired_return_delta' not in episode for episode in record['episodes'])
    comparison = paired_comparison(record['episodes'], cell['mean_reference']['episodes'])
    staged = stage_completed_bundle(bundle,{cell['actual_selector']:cell['run_dir']},inventory_path=campaign['inventory'])
    assert staged[cell['actual_selector']]['status'] == 'queued'
    performance = publish_performance(cell['run_dir'])
    summary = training_summary(bundle,cell)
    write(directory/'training-summary.json',summary)
    comparison_file = 'sample-minus-mean-comparison.json'
    if comparison:
        write(directory/comparison_file, comparison)
    write(directory/'historical-mean-reference.json', cell['mean_reference'])
    diagnostic = record_from_model_bundle(bundle,cell['actual_selector'],campaign['group']+'-'+cell['name'],
                                         bootstrap_resamples=2000,bootstrap_seed=20260912)
    assert diagnostic['status'] == 'complete' and len(diagnostic['rows']) == 2500*(cell['J']+1)
    write_diagnostic_bundle(directory/'model-series',diagnostic)
    journal = directory/'training-publication.json'
    if journal.exists(): raise RuntimeError('Training publication uncertain; inspect remote run before retry')
    write(journal,dict(status='uncertain',run_id=cell['training_run_id']))
    run = wandb.init(entity=ENTITY,project=PROJECT,id=cell['training_run_id'],resume='never',
                     name='Sampled execution | Inner training | '+cell['name']+f' | {checkpoint_step//1000}k',group=campaign['group'],
                     job_type='inner-training-diagnostics',tags=['closed-loop','H-J-sweep',cell['name'].rsplit('_h',1)[0]],
                     config=dict(H=cell['H'],J=cell['J'],N=128,B=256,C=critic_updates(cell),A=actor_updates(cell),checkpoint_step=checkpoint_step,
                                 checkpoint_sha256=campaign.get('checkpoint_sha256', CHECKPOINT_SHA),
                                 campaign_group=campaign['group'],
                                 source_run=campaign.get('source_run'),
                                 critic_kind=cell.get('critic_kind'),
                                 overview_url=(f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{campaign["overview_run_id"]}'
                                               if campaign.get('overview_run_id') else None),
                                 execution_mode='policy_sample', action_rule=ACTION_RULE,
                                 comparison='sampled minus historical adapted mean, matched J and seeds',
                                 mean_reference=cell['mean_reference']['performance_run_id'],
                                 setting=cell['name'],resolved_config=manifest['runs'][0]['resolved_config'],
                                 source_code=manifest['code'],reused=cell['reused'],
                                 inner_critic_target_tau=manifest['runs'][0]['resolved_config']['inner_critic_target_tau'],
                                 inner_replay_capacity=manifest['runs'][0]['resolved_config']['inner_replay_capacity'],
                                 inner_replay_reset_each_round=manifest['runs'][0]['resolved_config'].get('inner_replay_reset_each_round',False),
                                 baseline_performance_run_id=cell['mean_reference']['performance_run_id'],
                                 performance_run_id=cell['performance_run_id'],
                                 aggregation='Update curves average all decision roots; decision curves weight five seeds equally.',
                                 probe_objective='Reward plus terminal Q; excludes explicit entropy.'),mode='online')
    try:
        for axis in ('critic_update','actor_update','decision'):
            run.define_metric('axis/'+axis)
            prefix = {'critic_update':'critic','actor_update':'actor','decision':'episode'}[axis]
            run.define_metric(prefix+'/*',step_metric='axis/'+axis)
        run.define_metric('seed/*',step_metric='axis/decision')
        run.define_metric('diagnostic/actor_updates')
        run.define_metric('diagnostic/*',step_metric='diagnostic/actor_updates')
        for row in summary['update_curves']:
            prefix = 'critic' if row['axis']=='critic_update' else 'actor'
            run.log({'axis/'+row['axis']:row['index'],
                     **{f'{prefix}/{k}/{s}':v for k,stats in row['metrics'].items() for s,v in stats.items()}})
        per_seed = defaultdict(dict)
        for row in summary['per_seed_decisions']:
            per_seed[row['decision']].update({f"seed/{row['episode_id']}/{k}":v for k,v in row['metrics'].items()})
        for row in summary['decision_curves']:
            run.log({'axis/decision':row['decision'],**per_seed[row['decision']],
                     **{f'episode/{k}/{s}':v for k,stats in row['metrics'].items() for s,v in stats.items()}})
        for row in diagnostic_history(diagnostic): run.log(row)
        if comparison: run.log(comparison['metrics'])
        artifact = wandb.Artifact('inner-training-'+cell['training_run_id'],type='inner-training-traces',
                                  metadata=dict(manifest_sha256=receipt['manifest_sha256'],reused=cell['reused']))
        for name in ['manifest.json',*manifest['runs'][0]['trace_files']]: artifact.add_file(str(bundle/name),name='bundle/'+name)
        artifact.add_file(str(directory/'training-summary.json'),name='training-summary.json')
        if (bundle/'execution.json').exists():
            artifact.add_file(str(bundle/'execution.json'),name='execution.json')
        if comparison: artifact.add_file(str(directory/comparison_file),name=comparison_file)
        artifact.add_file(str(directory/'historical-mean-reference.json'),name='historical-mean-reference.json')
        artifact.add_file(str(Path(cell['mean_reference']['bundle'])/'manifest.json'),name='historical-mean-manifest.json')
        for name in ('manifest.json','paired-rows.jsonl.gz','report.html'):
            artifact.add_file(str(directory/'model-series'/name),name='model-series/'+name)
        run.log_artifact(artifact)
        run.summary.update({**record['metrics'],**(comparison['metrics'] if comparison else {}),
                            **execution_proof(manifest),
                            'status':'complete','reused':cell['reused'],
                            'diagnostic/paired_rows':len(diagnostic['rows']),
                            'training/decisions':2500,'training/critic_updates':2500*critic_updates(cell)*cell['J'],
                            'training/actor_updates':2500*actor_updates(cell)*cell['J'],
                            'performance_url':f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{cell["performance_run_id"]}'})
        run.finish()
    except BaseException:
        run.finish(exit_code=1); raise
    write(journal,dict(status='complete',run_id=cell['training_run_id']))
    write(directory/'publication-completion.json',dict(status='complete',cell=cell['name'],reused=cell['reused'],
          performance=performance,training_run_id=cell['training_run_id'],
          metrics={**record['metrics'], **comparison['metrics'], **execution_proof(manifest)}))


def publication_complete(cell):
    path = Path(cell['directory']) / 'publication-completion.json'
    if not path.exists():
        return False
    receipt = read(path)
    training = read(path.with_name('training-publication.json'))
    if (receipt.get('status') != 'complete' or receipt.get('cell') != cell['name']
            or receipt.get('training_run_id') != cell['training_run_id']
            or training.get('status') != 'complete' or training.get('run_id') != cell['training_run_id']
            or receipt.get('performance', {}).get('run_id') != cell['performance_run_id']
            or receipt.get('performance', {}).get('published') != 1):
        raise ValueError('Sampled publication completion identity mismatch')
    return True


def watch(args):
    """One CPU overview owner; every complete point is logged once by identity."""
    import wandb
    campaign = read(args.root / 'campaign.json')
    for cell in campaign['cells']:
        verify_mean_reference(cell['mean_reference'], traces=True)
    aggregate_results(campaign, {})
    marker = args.root / 'watcher-started.json'
    if marker.exists():
        raise RuntimeError('Overview already started; inspect W&B history and publication journals before recovery')
    write(marker, dict(pid=os.getpid(), started=time.time()))
    publishers = int(campaign.get('publisher_workers', 2))
    assert 1 <= publishers <= 3
    urls = {c['name']: {**{kind + '_url': f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{c[kind + "_run_id"]}'
                            for kind in ('performance', 'training')},
                       'historical_mean_url': f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{c["mean_reference"]["performance_run_id"]}'}
            for c in campaign['cells']}
    config = {key: campaign[key] for key in ('checkpoint_step', 'checkpoint_sha256', 'source_run',
              'source_commit', 'initial_alpha', 'target_entropy', 'mean_source_commit')}
    config.update(campaign_group=campaign['group'], protocol='closed-loop-refinement-sampled-execution-v1',
                  protocol_variant='Final adapted squashed Gaussian sample; frozen evaluation API retained',
                  J=[1, 2, 4], H=3, C=16, A=4, N=128, B=256, critic_kind='return_only',
                  execution_mode='policy_sample', action_rule=ACTION_RULE, execution_std_scale=1.0,
                  inner_replay_capacity=3072, inner_replay_scope='action', inner_replay_reset_each_round=False,
                  environment_seeds=SEEDS, controller_seed=55, max_decisions=500,
                  execution='Fresh frozen-prior adaptation at every real decision, then tanh(mu + std * epsilon)',
                  mean_comparison=campaign['comparison'], prior_reference=None,
                  uncertainty='Five matched environment/controller seeds; 2,000 seed bootstrap resamples; exploratory 95% interval.',
                  historical_mean_manifest_sha256={str(c['J']): c['mean_reference']['manifest_sha256'] for c in campaign['cells']},
                  result_links=urls)
    run = wandb.init(entity=ENTITY, project=PROJECT, id=campaign['overview_run_id'], resume='never',
                     name=campaign['label'], group=campaign['group'], job_type='sampled-execution-comparison',
                     tags=['closed-loop', 'sampled-vs-mean', 'return-only', 'H3'], config=config, mode='online')
    run.define_metric('axis/inner_rounds')
    run.define_metric('sampled_execution/*', step_metric='axis/inner_rounds')
    run.define_metric('execution/*', step_metric='axis/inner_rounds')
    run.summary.update(dict(status='running', evaluated=0, published=0, total_settings=3,
                            result_type='Full-episode sampled versus historical mean execution returns'))
    attempted, completed, proofs, futures, failures, logged = set(), {}, {}, {}, {}, set()
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
            name = cell['name']
            if publication_complete(cell):
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
            metrics = point.get('comparison_metrics', {})
            rows.append(dict(setting=name, J=cell['J'], execution='policy_sample', status=state,
                             return_mean=point.get('return_stats', {}).get('mean'),
                             return_std=point.get('return_stats', {}).get('std'),
                             sample_minus_mean=metrics.get('comparison/sample_minus_mean_mean'),
                             sample_minus_mean_ci95_low=metrics.get('comparison/sample_minus_mean_ci95_low'),
                             sample_minus_mean_ci95_high=metrics.get('comparison/sample_minus_mean_ci95_high'),
                             failure=failures.get(index), **urls[name]))
        return rows

    def update(terminal=False):
        nonlocal previous
        aggregate = aggregate_results(campaign, completed)
        statuses = status_rows(aggregate, terminal)
        # A publication may already exist when a new overview owner starts.
        # Loading its validated measurements must still refresh the plots even
        # when its display status stays "published" across both updates.
        stamp = ([(r['setting'], r['status']) for r in statuses], sorted(completed))
        if stamp != previous:
            for identity, row in numeric_rows(aggregate):
                if identity not in logged:
                    if identity.startswith('sampled_'):
                        cell, = [c for c in campaign['cells'] if c['J'] == row['axis/inner_rounds']]
                        row.update(proofs[cell['name']])
                    run.log(row)
                    logged.add(identity)
            run.log(overview_log(wandb, aggregate, statuses))
            run.summary.update(dict(evaluated=len(completed), published=sum(r['status'] == 'published' for r in statuses),
                                    execution_proof_by_setting=proofs))
            write(args.root / 'comparison-results.json', aggregate)
            write(args.root / 'progress.json', dict(rows=statuses, failures=failures, evaluated=len(completed)))
            print(f'Evaluated {len(completed)}/3; published {sum(r["status"] == "published" for r in statuses)}', flush=True)
            previous = stamp
        return statuses

    try:
        update()
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
                        completed[cell['name']], proofs[cell['name']] = load_completed(campaign, cell)
                    if (len(futures) < publishers and index not in attempted and cell['name'] in completed
                            and not publication_complete(cell)):
                        attempted.add(index)
                        futures[index] = pool.submit(launch, index)
                statuses = update()
                if all(r['status'] == 'published' for r in statuses):
                    break
                submission = args.root / 'submission.json'
                if submission.exists() and not futures:
                    if not gpu_jobs_active(read(submission)['gpu_job_ids']):
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
            raise RuntimeError('Sampled campaign incomplete; inspect status table and publication logs')
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
    {'watch': watch, 'publish': publish_cell}[args.mode](args)


if __name__ == '__main__':
    main()
