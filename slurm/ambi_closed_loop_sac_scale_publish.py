"""Publish the paired 575k SAC scale sweep without reusing legacy schedule guards."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_aux_hj_sweep import ENTITY, PROJECT, SEEDS, digest, publish_performance, read, write
from slurm.ambi_closed_loop_publish import gpu_jobs_active, indexed_episodes, moments
from slurm.ambi_closed_loop_reward_retrace_publish import paired_comparison
from slurm.ambi_closed_loop_sampled import verify_receipt
from slurm.ambi_closed_loop_checkpoint_sweep_publish import publication_complete

ROUNDS = [1, 2, 4, 6, 8, 10]
DIMENSIONS = ['H', 'J', 'N', 'B', 'G', 'P', 'T', 'C', 'A', 'target_updates']
POINT_COLUMNS = ['setting', *DIMENSIONS, 'training_decisions', 'checkpoint_sha256',
    'performance_run_id', 'training_run_id', 'initial_alpha', 'return_mean', 'return_std',
    'return_episodes', 'prior_return_mean', 'prior_return_std', 'paired_gain_mean',
    'paired_gain_std', 'paired_gain_ci95_low', 'paired_gain_ci95_high', 'paired_episodes',
    'control_seconds', 'control_ms_per_decision', 'probe_seconds', 'probe_ms_per_decision',
    'evaluation_elapsed_seconds', 'warmup_including_compile_seconds', 'serialization_seconds']
EPISODE_COLUMNS = ['setting', *DIMENSIONS, 'seed', 'solver_seed', 'return', 'prior_return', 'paired_gain',
                   'control_seconds', 'probe_seconds']
STATUS_COLUMNS = [*POINT_COLUMNS, 'status', 'performance_url', 'training_url', 'failure']


def validate_completed(*args, **kwargs):
    from slurm.ambi_closed_loop_sac_scale import validate_completed as validate
    return validate(*args, **kwargs)


def training_summary(*args, **kwargs):
    from slurm.ambi_closed_loop_sac_scale import training_summary as summarize
    return summarize(*args, **kwargs)


def _url(run_id):
    return f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}'


def validate_scope(campaign):
    cells = campaign['cells']
    expected = {(h, j, 1024, 4096, 20, p, 2) for h in (1, 2, 3) for j in ROUNDS for p in (1, 5)}
    observed = [(c['H'], c['J'], c['N'], c['B'], c['G'], c['P'], c['T']) for c in cells]
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError('Expected H1/2/3, J1/2/4/6/8/10, N1024/B4096, G20/P1-or-5/T2')
    ids = [campaign['overview_run_id']] + [c[k] for c in cells for k in ('performance_run_id', 'training_run_id')]
    if len(ids) != len(set(ids)) or len({c['run_dir'] for c in cells}) != len(cells):
        raise ValueError('Each setting must own distinct performance, training and overview identities')
    if len({c['name'] for c in cells}) != len(cells):
        raise ValueError('Duplicate setting names')
    if len({c['checkpoint_sha256'] for c in cells}) != 1 or len({c['initial_alpha'] for c in cells}) != 1:
        raise ValueError('All settings must inherit one pinned checkpoint and temperature')
    prior = None
    for cell in cells:
        if (cell['checkpoint_step'], cell['training_decisions'], cell['reused']) != (575000, 575000, False):
            raise ValueError('Only new 575k refinement evaluations belong in this sweep')
        if (cell['C'], cell['A'], cell['target_updates']) != (cell['G'], cell['G']//cell['P'], cell['G']//cell['T']):
            raise ValueError('Advertised optimizer work differs from the frequency schedule')
        if not math.isfinite(cell['initial_alpha']) or cell['initial_alpha'] <= 0:
            raise ValueError('Expected the saved positive adaptive temperature')
        p = cell['params']
        expected_params = dict(inner_rollouts_per_round=cell['N'], inner_batch_size=cell['B'],
            inner_updates_per_round=cell['G'], inner_actor_update_interval=cell['P'],
            inner_critic_target_update_interval=cell['T'], inner_rollout_horizon=cell['H'], inner_rounds=cell['J'],
            inner_critic_source='aux_return', inner_horizon_critic_source='aux_return',
            inner_sac_critic_target='reward_only', inner_terminal_entropy='none',
            inner_temperature_mode='auto', inner_entropy_enabled=True, inner_execution_action='mean')
        if any(p.get(k) != v for k, v in expected_params.items()):
            raise ValueError('Planner settings differ from the requested return/return scale sweep')
        if p.get('inner_critic_updates_per_round') is not None or p.get('inner_actor_updates_per_round') is not None:
            raise ValueError('Frequency scheduling cannot carry phased component budgets')
        if p['inner_replay_capacity'] < cell['N']*cell['H']*cell['J'] or p.get('inner_replay_reset_each_round', False):
            raise ValueError('Replay must retain every imagined transition in the solve')
        reference = cell['prior_reference']
        if (reference['checkpoint_step'], reference['checkpoint_sha256']) != (575000, cell['checkpoint_sha256']):
            raise ValueError('Prior belongs to a different checkpoint')
        current = indexed_episodes(reference['episodes'])
        if prior is not None and prior != current:
            raise ValueError('Every setting must use the same paired prior')
        prior = current
    return cells


def verify_references(campaign):
    from slurm.ambi_closed_loop_checkpoint_sweep import load_prior
    first = campaign['cells'][0]['prior_reference']
    actual = load_prior(first, campaign['inventory'])
    if indexed_episodes(actual['episodes']) != indexed_episodes(first['episodes']):
        raise ValueError('Prior changed after preparation')
    for cell in campaign['cells']:
        if any(cell['prior_reference'][k] != first[k] for k in ('bundle', 'manifest_sha256', 'checkpoint_sha256')):
            raise ValueError('Prior provenance differs across settings')


def aggregate_results(campaign, completed, timings=None):
    """Only complete paired panels become measured curves; pending values remain null."""
    cells = validate_scope(campaign)
    if set(completed) - {c['name'] for c in cells}:
        raise ValueError('Unknown completed setting')
    points, rows = [], []
    for cell in sorted(cells, key=lambda c: (c['H'], c['B'], c['P'], c['J'])):
        reference = cell['prior_reference']; prior = moments(e['return'] for e in reference['episodes'])
        point = dict.fromkeys(POINT_COLUMNS)
        point.update({k: cell[k] for k in DIMENSIONS})
        point.update(setting=cell['name'], training_decisions=575000, checkpoint_sha256=cell['checkpoint_sha256'],
            performance_run_id=cell['performance_run_id'], training_run_id=cell['training_run_id'],
            initial_alpha=cell['initial_alpha'], prior_return_mean=prior['mean'], prior_return_std=prior['std'])
        episodes = completed.get(cell['name'])
        if episodes is not None:
            stats = moments(e['return'] for e in episodes)
            comparison = paired_comparison(episodes, reference['episodes'])
            point.update(return_mean=stats['mean'], return_std=stats['std'], return_episodes=stats['episodes'])
            for suffix in ('mean', 'std', 'ci95_low', 'ci95_high'):
                point['paired_gain_'+suffix] = comparison['metrics']['comparison/sample_minus_mean_'+suffix]
            point['paired_episodes'] = comparison['metrics']['comparison/sample_minus_mean_paired_episodes']
            indexed = indexed_episodes(episodes)
            for row in comparison['rows']:
                episode = indexed[row['seed'], row['solver_seed']]
                if not math.isclose(episode.get('paired_return_delta', row['sample_minus_mean']), row['sample_minus_mean'], abs_tol=1e-9):
                    raise ValueError('Stored paired gain differs from the frozen prior')
                rows.append(dict(setting=cell['name'], **{k:cell[k] for k in DIMENSIONS},
                    seed=row['seed'], solver_seed=row['solver_seed'], **{'return':row['sampled_return']},
                    prior_return=row['mean_return'], paired_gain=row['sample_minus_mean'],
                    control_seconds=episode.get('control_seconds'), probe_seconds=episode.get('togo_probe_seconds')))
            for source, total, average in [('control_seconds','control_seconds','control_ms_per_decision'),
                                           ('togo_probe_seconds','probe_seconds','probe_ms_per_decision')]:
                if all(source in e for e in episodes):
                    values = [e[source] for e in episodes]
                    if not all(math.isfinite(v) and v >= 0 for v in values):
                        raise ValueError('Invalid episode runtime')
                    point[total] = sum(values); point[average] = 1000*sum(values)/sum(e['length'] for e in episodes)
            for key, value in (timings or {}).get(cell['name'], {}).items():
                if key in POINT_COLUMNS:
                    if not math.isfinite(value) or value < 0: raise ValueError('Invalid measured timing')
                    point[key] = value
        points.append(point)
    return dict(points=points, episodes=rows, evaluated=len(completed), new_evaluated=len(completed), reused=0,
        bootstrap_seed=20260912, bootstrap_resamples=2000,
        uncertainty='Five paired environment seeds; 2,000 environment-seed bootstrap resamples; exploratory 95% intervals.',
        timing_semantics='Control excludes independent probes and warmup. Elapsed is per-worker evaluation time, not campaign wall time.')


def series_key(point):
    return f"h{point['H']}_n{point['N']}_b{point['B']}_g{point['G']}_p{point['P']}_t{point['T']}"


def numeric_rows(aggregate):
    rows = []
    for point in aggregate['points']:
        if point['return_mean'] is None: continue
        prefix = 'sac_scale/'+series_key(point)
        rows.append((point['setting'], {'axis/inner_rounds':point['J'], **{prefix+'/'+key:point[key]
            for key in ('return_mean','paired_gain_mean','paired_gain_ci95_low','paired_gain_ci95_high',
                        'control_ms_per_decision','probe_ms_per_decision') if point[key] is not None}}))
    return rows


def chart_payloads(aggregate):
    groups = defaultdict(list)
    for point in aggregate['points']:
        if point['return_mean'] is not None: groups[series_key(point)].append(point)
    charts = {}
    for metric, title in [('return_mean','Full-episode return'), ('paired_gain_mean','Paired gain over frozen prior'),
                          ('control_ms_per_decision','Decision latency, excluding probes')]:
        xs, ys, keys = [], [], []
        for key, points in sorted(groups.items()):
            valid = [p for p in points if p[metric] is not None]
            if not valid: continue
            xs.append([p['J'] for p in valid]); ys.append([p[metric] for p in valid]); keys.append(key)
        if metric != 'control_ms_per_decision':
            xs.append(ROUNDS); ys.append([aggregate['points'][0]['prior_return_mean'] if metric == 'return_mean' else 0.]*len(ROUNDS))
            keys.append('Frozen prior (reused)' if metric == 'return_mean' else 'No improvement')
        if xs:
            charts['comparison/'+metric+'_vs_J'] = dict(xs=xs, ys=ys, keys=keys, title=title, xname='Inner rounds J')
    return charts


def overview_log(wandb, aggregate, statuses):
    payload = {key:wandb.plot.line_series(**value) for key,value in chart_payloads(aggregate).items()}
    for key, rows, columns in [('comparison/points',aggregate['points'],POINT_COLUMNS),
                              ('comparison/paired_episodes',aggregate['episodes'],EPISODE_COLUMNS),
                              ('campaign/settings',statuses,STATUS_COLUMNS)]:
        payload[key] = wandb.Table(columns=columns, data=[[r.get(c) for c in columns] for r in rows])
    payload.update({'campaign/evaluated':aggregate['evaluated'],
                    'campaign/published':sum(r['status']=='published' for r in statuses)})
    return payload


def load_completed(campaign, cell):
    receipt = read(Path(cell['directory'])/'worker-completion.json')
    verify_receipt(cell['bundle'], receipt)
    if receipt['cell'] != cell['name'] or receipt['status'] != 'complete':
        raise ValueError('Worker identity differs')
    manifest = validate_completed(Path(cell['bundle']), cell, campaign)
    run, = manifest['runs']
    proof = dict(source_code=manifest['code'], resolved_config=run['resolved_config'],
        checkpoint_sha256=cell['checkpoint_sha256'], initial_alpha=cell['initial_alpha'],
        model_metrics=run['result']['model_metrics'])
    timings = {'evaluation_elapsed_seconds':manifest['elapsed_seconds']}
    for key in ('warmup_including_compile_seconds','serialization_seconds'):
        if key in run: timings[key] = run[key]
    return run['episodes'], proof, timings


def publish_cell(args):
    campaign = read(args.root/'campaign.json'); validate_scope(campaign)
    cell = campaign['cells'][args.index]; directory = Path(cell['directory'])
    with (directory/'.sac-scale-publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        load_completed(campaign, cell)
        if publication_complete(cell): return
        _publish_cell(campaign, cell)
        if not publication_complete(cell): raise RuntimeError('Performance or training publication incomplete')


def _publish_cell(campaign, cell):
    """Independent performance/trace ownership, with durable uncertain-publication guards."""
    from utils.ambi_benchmark import stage_completed_bundle
    from utils.eval_series import load_run
    from utils.eval_series_data import load_records
    from utils.ambi_diagnostic_series import record_from_model_bundle, write_diagnostic_bundle, diagnostic_history
    import wandb
    directory, bundle = Path(cell['directory']), Path(cell['bundle'])
    receipt = read(directory/'worker-completion.json')
    manifest = validate_completed(bundle, cell, campaign)
    record, = load_records(bundle, inventory_path=campaign['inventory'])
    if record['identity'] != load_run(cell['run_dir'])['identity'] or record['metrics']['eval/paired_episodes'] != 5:
        raise ValueError('Performance identity or pairing differs')
    journal = directory/'training-publication.json'
    if journal.exists():
        raise RuntimeError('Training publication already started; inspect its journal and remote run before recovery')
    staged = stage_completed_bundle(bundle,{cell['actual_selector']:cell['run_dir']},inventory_path=campaign['inventory'])
    if staged[cell['actual_selector']]['status'] != 'queued': raise ValueError('Unexpected staged record status')
    performance = publish_performance(cell['run_dir'])
    summary = training_summary(bundle,cell,expected_steps=500); write(directory/'training-summary.json',summary)
    diagnostic = record_from_model_bundle(bundle,cell['actual_selector'],campaign['group']+'-'+cell['name'],
                                         bootstrap_resamples=2000,bootstrap_seed=20260912)
    if diagnostic['status'] != 'complete' or len(diagnostic['rows']) != 2500*(cell['J']+1):
        raise ValueError('Incomplete diagnostic coverage')
    write_diagnostic_bundle(directory/'model-series',diagnostic)
    write(journal,dict(status='uncertain',run_id=cell['training_run_id'],manifest_sha256=receipt['manifest_sha256']))
    run = wandb.init(entity=ENTITY,project=PROJECT,id=cell['training_run_id'],resume='never',
        name='Inner training | '+cell['name']+' | 575k',group=campaign['group'],job_type='inner-training-diagnostics',
        tags=['closed-loop','sac-scale','frequency-schedule','return-return'],mode='online',
        config=dict(**{k:cell[k] for k in DIMENSIONS},checkpoint_step=575000,checkpoint_sha256=cell['checkpoint_sha256'],
            setting=cell['name'],campaign_group=campaign['group'],source_run=campaign['source_run'],
            source_code=manifest['code'],resolved_config=manifest['runs'][0]['resolved_config'],
            initial_alpha=cell['initial_alpha'],overview_url=_url(campaign['overview_run_id']),
            performance_url=_url(cell['performance_run_id']),reused=False,
            aggregation='Update curves average decision roots; per-decision curves weight five seeds equally.',
            probe_objective='Reward plus frozen terminal return Q; no explicit entropy bonus.'))
    try:
        for axis,prefix in [('critic_update','critic'),('actor_update','actor'),('decision','episode')]:
            run.define_metric('axis/'+axis); run.define_metric(prefix+'/*',step_metric='axis/'+axis)
        run.define_metric('seed/*',step_metric='axis/decision')
        run.define_metric('diagnostic/actor_updates'); run.define_metric('diagnostic/*',step_metric='diagnostic/actor_updates')
        for row in summary['update_curves']:
            prefix = 'critic' if row['axis']=='critic_update' else 'actor'
            run.log({'axis/'+row['axis']:row['index'], **{f'{prefix}/{k}/{s}':v for k,stats in row['metrics'].items() for s,v in stats.items()}})
        per_seed = defaultdict(dict)
        for row in summary['per_seed_decisions']:
            per_seed[row['decision']].update({f"seed/{row['episode_id']}/{k}":v for k,v in row['metrics'].items()})
        for row in summary['decision_curves']:
            run.log({'axis/decision':row['decision'],**per_seed[row['decision']],
                **{f'episode/{k}/{s}':v for k,stats in row['metrics'].items() for s,v in stats.items()}})
        for row in diagnostic_history(diagnostic): run.log(row)
        artifact = wandb.Artifact('inner-training-'+cell['training_run_id'],type='inner-training-traces',
            metadata=dict(manifest_sha256=receipt['manifest_sha256'],reused=False))
        for name in ['manifest.json',*manifest['runs'][0]['trace_files']]: artifact.add_file(str(bundle/name),name='bundle/'+name)
        artifact.add_file(str(directory/'training-summary.json'),name='training-summary.json')
        for name in ('manifest.json','paired-rows.jsonl.gz','report.html'):
            artifact.add_file(str(directory/'model-series'/name),name='model-series/'+name)
        if (bundle/'execution.json').exists(): artifact.add_file(str(bundle/'execution.json'),name='execution.json')
        run.log_artifact(artifact)
        run.summary.update({**record['metrics'],'status':'complete','reused':False,
            'diagnostic/paired_rows':len(diagnostic['rows']),'training/decisions':2500,
            'training/critic_updates':2500*cell['G']*cell['J'],
            'training/actor_updates':2500*(cell['G']*cell['J']//cell['P']),
            'training/target_updates':2500*(cell['G']*cell['J']//cell['T']),
            'performance_url':_url(cell['performance_run_id'])})
        run.finish()
    except BaseException:
        run.finish(exit_code=1); raise
    write(journal,dict(status='complete',run_id=cell['training_run_id'],manifest_sha256=receipt['manifest_sha256']))
    write(directory/'publication-completion.json',dict(status='complete',cell=cell['name'],reused=False,
        performance=performance,training_run_id=cell['training_run_id'],metrics=record['metrics']))


def status_rows(campaign, aggregate, completed, futures, failures, terminal=False):
    points = {p['setting']:p for p in aggregate['points']}; rows = []
    for index,cell in enumerate(campaign['cells']):
        if publication_complete(cell): status='published'
        elif index in failures: status='publication_failed'
        elif index in futures: status='publishing'
        elif cell['name'] in completed: status='evaluated_awaiting_publication'
        else: status='evaluation_incomplete' if terminal else 'queued_or_running'
        rows.append(dict(points[cell['name']],status=status,failure=failures.get(index),
            performance_url=_url(cell['performance_run_id']),training_url=_url(cell['training_run_id'])))
    return rows


def watch(args):
    """One overview owner; three bounded subprocesses publish completed settings."""
    import wandb
    campaign = read(args.root/'campaign.json'); cells = validate_scope(campaign); verify_references(campaign)
    marker = args.root/'watcher-started.json'
    if marker.exists(): raise RuntimeError('Watcher already started; inspect journals and remote history before recovery')
    publishers = int(campaign.get('publisher_workers',3))
    if not 1 <= publishers <= 3: raise ValueError('Expected one to three publisher subprocesses')
    with marker.open('x') as stream:
        json.dump(dict(pid=os.getpid(),started=time.time(),source_commit=campaign['source_commit']),stream)
    run = wandb.init(entity=ENTITY,project=PROJECT,id=campaign['overview_run_id'],resume='never',
        name=campaign['label'],group=campaign['group'],job_type='sac-scale-comparison',mode='online',
        tags=['closed-loop','sac-scale','return-return','575k'],config=dict(
            protocol='closed-loop-sac-scale-v1',source_run=campaign['source_run'],source_commit=campaign['source_commit'],
            checkpoint_step=575000,checkpoint_sha256=cells[0]['checkpoint_sha256'],
            environment_seeds=SEEDS,controller_seed=55,max_decisions=500,togo_return_rollouts=32,
            target_entropy=-10.5,execution='mean',critic_strategy='return/return',temperature='inherited adaptive',
            total_settings=len(cells),new_settings=len(cells),reused_settings=0,
            uncertainty='Five paired seeds; 2000 paired bootstrap resamples; exploratory 95% intervals.',
            settings=[dict(setting=c['name'],**{k:c[k] for k in DIMENSIONS},
                performance_url=_url(c['performance_run_id']),training_url=_url(c['training_run_id'])) for c in cells]))
    run.define_metric('axis/inner_rounds'); run.define_metric('sac_scale/*',step_metric='axis/inner_rounds')
    attempted,completed,proofs,timings,futures,failures,logged = set(),{},{},{},{},{},set()
    previous,terminal_since = None,None
    def launch(index):
        with (Path(cells[index]['directory'])/'publisher.log').open('w') as log:
            return subprocess.run([sys.executable,__file__,'publish','--root',str(args.root),'--index',str(index)],
                                  stdout=log,stderr=subprocess.STDOUT).returncode
    def update(terminal=False):
        nonlocal previous
        aggregate=aggregate_results(campaign,completed,timings); statuses=status_rows(campaign,aggregate,completed,futures,failures,terminal)
        stamp=([(r['setting'],r['status']) for r in statuses],sorted(completed))
        if stamp != previous:
            for identity,row in numeric_rows(aggregate):
                if identity not in logged: run.log(row); logged.add(identity)
            run.log(overview_log(wandb,aggregate,statuses))
            published=sum(r['status']=='published' for r in statuses)
            run.summary.update(dict(status='running',evaluated=len(completed),published=published,total_settings=len(cells),protocol_proof_by_setting=proofs))
            write(args.root/'comparison-results.json',aggregate)
            write(args.root/'progress.json',dict(rows=statuses,failures=failures,evaluated=len(completed),published=published))
            print(f'Evaluated {len(completed)}/{len(cells)}; published {published}/{len(cells)}',flush=True); previous=stamp
        return statuses
    try:
        update()
        with ThreadPoolExecutor(max_workers=publishers) as pool:
            while True:
                for index,future in list(futures.items()):
                    if future.done():
                        rc=future.result()
                        if rc: failures[index]=f"Publisher exited {rc}; inspect {cells[index]['directory']}/publisher.log"
                        del futures[index]
                for index,cell in enumerate(cells):
                    if (Path(cell['directory'])/'worker-completion.json').exists() and cell['name'] not in completed:
                        completed[cell['name']],proofs[cell['name']],timings[cell['name']]=load_completed(campaign,cell)
                    if len(futures)<publishers and index not in attempted and cell['name'] in completed and not publication_complete(cell):
                        attempted.add(index); futures[index]=pool.submit(launch,index)
                statuses=update()
                if all(r['status']=='published' for r in statuses): break
                submission=args.root/'submission.json'
                if submission.exists() and not futures:
                    if not gpu_jobs_active(read(submission)['gpu_job_ids']):
                        terminal_since=terminal_since or time.time()
                        if time.time()-terminal_since>90: break
                    else: terminal_since=None
                time.sleep(15)
        statuses=update(terminal=True)
        complete=len(completed)==len(cells) and all(r['status']=='published' for r in statuses)
        write(args.root/'campaign-completion.json',dict(status='complete' if complete else 'incomplete',rows=statuses,
            failures=failures,evaluated=len(completed),published=sum(r['status']=='published' for r in statuses)))
        run.summary.update(dict(status='complete' if complete else 'incomplete',failed_publications=len(failures)))
        if not complete: raise RuntimeError('SAC scale sweep incomplete; inspect progress and publisher logs')
        run.finish()
    except BaseException as exc:
        write(args.root/'campaign-completion.json',dict(status='failed',failure=str(exc),failures=failures))
        run.summary.update(dict(status='failed',failure=str(exc))); run.finish(exit_code=1); raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['watch','publish']); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--index',type=int); args=parser.parse_args()
    {'watch':watch,'publish':publish_cell}[args.mode](args)


if __name__=='__main__': main()
