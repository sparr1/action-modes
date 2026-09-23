"""Publish mixed soft-inner/return-bootstrap outcomes and exact matched contrasts."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_aux_hj_sweep import (ENTITY, PROJECT, SEEDS, actor_updates, critic_updates,
                                    publish_performance, read, training_summary, write)
from slurm.ambi_closed_loop_publish import gpu_jobs_active, indexed_episodes, moments
from slurm.ambi_closed_loop_reward_retrace_publish import (
    execution_proof, paired_comparison, publication_complete,
)
from slurm.ambi_closed_loop_sampled import verify_receipt
from slurm.ambi_closed_loop_soft_return import action_rule, validate_completed, verify_reference

CONTRASTS = {
    'soft_return_minus_reward': ('soft_return', 'reward'),
    'retrace_minus_one_step': ('retrace', 'one_step'),
    'sampled_minus_mean': ('sampled', 'mean'),
}
POINT_AXES = ('setting', 'H', 'J', 'estimator', 'retrace_lambda', 'execution',
              'critic_kind', 'historical', 'performance_run_id')


def paired_delta(left, right, kind):
    """Reuse the established seed-paired bootstrap, with explicit contrast names."""
    left_name, right_name = CONTRASTS[kind]
    comparison = paired_comparison(left, right)
    rows = [dict(seed=r['seed'], solver_seed=r['solver_seed'],
                 **{left_name + '_return': r['sampled_return'],
                    right_name + '_return': r['mean_return'], kind: r['sample_minus_mean']})
            for r in comparison['rows']]
    return dict(rows=rows, difference=moments(r[kind] for r in rows),
                metrics={k.replace('sample_minus_mean', kind): v
                         for k, v in comparison['metrics'].items()},
                bootstrap_seed=comparison['bootstrap_seed'],
                bootstrap_resamples=comparison['bootstrap_resamples'])


def _key(value, *, cell=False):
    execution = value['execution_mode'] if cell else value['execution']
    if execution not in ('mean', 'policy_sample') or value['estimator'] not in ('one_step', 'retrace'):
        raise ValueError('Unexpected execution mode or estimator')
    return value['H'], value['J'], value['estimator'], execution


def _point(value, episodes, *, historical):
    h, j, estimator, execution = _key(value, cell=not historical)
    indexed_episodes(episodes)
    return dict(H=h, J=j, estimator=estimator,
                retrace_lambda=value.get('retrace_lambda', .9 if estimator == 'retrace' else None),
                execution=execution, critic_kind='return_only' if historical else 'soft_return',
                setting=(f'historical_reward_h{h}_j{j}_{estimator}_{execution}' if historical else value['name']),
                historical=historical, performance_run_id=value['performance_run_id'],
                baseline_performance_run_id=None,
                return_stats=moments(e['return'] for e in episodes),
                **{kind + suffix: None for kind in CONTRASTS for suffix in ('_difference', '_metrics')})


def aggregate_results(campaign, completed):
    """Keep absent baselines missing; compare only the explicitly matched axes."""
    points, episode_rows = [], []
    comparisons = {kind: [] for kind in CONTRASTS}
    references, cells, new_by_key = {}, {}, {}

    def add_point(point, episodes):
        points.append(point)
        episode_rows.extend(dict(**{k: point[k] for k in POINT_AXES},
                                 **{k: e[k] for k in ('seed', 'solver_seed', 'return', 'length')})
                            for e in episodes)

    for reference in campaign['references']:
        key = _key(reference)
        if key in references or reference.get('critic_kind', 'return_only') != 'return_only':
            raise ValueError('Duplicate or non-reward historical reference')
        references[key] = reference
        add_point(_point(reference, reference['episodes'], historical=True), reference['episodes'])
    cell_keys = set()
    for cell in campaign['cells']:
        key = _key(cell, cell=True)
        if cell['name'] in cells or key in cell_keys or cell['critic_kind'] != 'soft_return':
            raise ValueError('Duplicate or non-mixed campaign setting')
        cells[cell['name']] = cell
        cell_keys.add(key)
        reference = cell.get('reward_reference')
        if (reference is None) != (key not in references):
            raise ValueError('Reward comparison reference availability does not match historical results')
        if reference is not None:
            if (_key(reference) != key or key not in references
                    or reference['performance_run_id'] != references[key]['performance_run_id']
                    or indexed_episodes(reference['episodes']) != indexed_episodes(references[key]['episodes'])):
                raise ValueError('Reward comparison reference does not match the complete setting')
    if set(completed) - cells.keys():
        raise ValueError('Unexpected completed setting')
    for name, episodes in completed.items():
        cell = cells[name]
        point = _point(cell, episodes, historical=False)
        add_point(point, episodes)
        new_by_key[_key(cell, cell=True)] = point, episodes

    def add_comparison(point, episodes, baseline_episodes, kind, baseline_id):
        comparison = paired_delta(episodes, baseline_episodes, kind)
        point[kind + '_difference'] = comparison['difference']
        point[kind + '_metrics'] = comparison['metrics']
        comparisons[kind].extend(dict(**{k: point[k] for k in POINT_AXES},
                                      baseline_performance_run_id=baseline_id, **row)
                                 for row in comparison['rows'])

    for (h, j, estimator, execution), (point, episodes) in new_by_key.items():
        reference = cells[point['setting']].get('reward_reference')
        if reference is not None:
            point['baseline_performance_run_id'] = reference['performance_run_id']
            add_comparison(point, episodes, reference['episodes'], 'soft_return_minus_reward',
                           reference['performance_run_id'])
        baseline = new_by_key.get((h, j, 'one_step', execution))
        if estimator == 'retrace' and baseline is not None:
            add_comparison(point, episodes, baseline[1], 'retrace_minus_one_step',
                           baseline[0]['performance_run_id'])
        baseline = new_by_key.get((h, j, estimator, 'mean'))
        if execution == 'policy_sample' and baseline is not None:
            add_comparison(point, episodes, baseline[1], 'sampled_minus_mean',
                           baseline[0]['performance_run_id'])
    points.sort(key=lambda p: (p['H'], p['critic_kind'], p['estimator'], p['execution'], p['J']))
    episode_rows.sort(key=lambda r: (r['setting'], r['seed']))
    for rows in comparisons.values():
        rows.sort(key=lambda r: (r['setting'], r['seed']))
    return dict(points=points, episodes=episode_rows, comparisons=comparisons,
                new_evaluated=len(completed),
                comparison='Mixed soft-inner/return-bootstrap minus reward-inner at identical H/J/estimator/execution; '
                           'estimator and execution contrasts use only new mixed-critic cells.',
                uncertainty='Five matched environment/controller seeds; exploratory paired bootstrap intervals.')


def expected_pairs(campaign):
    keys = {_key(c, cell=True) for c in campaign['cells']}
    return dict(soft_return_minus_reward=sum(c.get('reward_reference') is not None for c in campaign['cells']),
                retrace_minus_one_step=sum(e == 'retrace' and (h, j, 'one_step', mode) in keys
                                          for h, j, e, mode in keys),
                sampled_minus_mean=sum(mode == 'policy_sample' and (h, j, e, 'mean') in keys
                                       for h, j, e, mode in keys))


def pair_counts(aggregate):
    return {kind: len({r['setting'] for r in aggregate['comparisons'][kind]}) for kind in CONTRASTS}


def numeric_rows(aggregate):
    rows = []
    for point in aggregate['points']:
        prefix = f"soft_return/{point['critic_kind']}/h{point['H']}/{point['estimator']}/{point['execution']}"
        rows.append((point['setting'], {'axis/inner_rounds': point['J'],
                     **{f'{prefix}/return_{k}': v for k, v in point['return_stats'].items()}}))
        for kind in CONTRASTS:
            stats, metrics = point[kind + '_difference'], point[kind + '_metrics']
            if stats is not None:
                rows.append((point['setting'] + '/' + kind, {'axis/inner_rounds': point['J'],
                             **{f'{prefix}/{kind}/{k}': v for k, v in stats.items()},
                             **{f'{prefix}/{kind}/{k.removeprefix("comparison/" + kind + "_")}': v
                                for k, v in metrics.items()}}))
    return rows


def chart_payloads(aggregate):
    result = {}
    for kind, field, title in [('return', 'return_stats', 'Full-episode return'),
                              *[(k, k + '_difference', k.replace('_', ' ')) for k in CONTRASTS]]:
        groups = defaultdict(list)
        for point in aggregate['points']:
            if point[field] is not None:
                groups[(point['H'], point['critic_kind'], point['estimator'], point['execution'])].append(point)
        groups = sorted(groups.items())
        if groups:
            result['comparison/' + kind + '_vs_J'] = dict(
                xs=[[p['J'] for p in values] for _, values in groups],
                ys=[[p[field]['mean'] for p in values] for _, values in groups],
                keys=[f'H{h} | {critic} | {estimator} | {mode}' for (h, critic, estimator, mode), _ in groups],
                title=title, xname='Inner rounds J')
    return result


def overview_log(wandb, aggregate, statuses):
    result = {key: wandb.plot.line_series(**payload) for key, payload in chart_payloads(aggregate).items()}
    def table(rows, columns):
        return wandb.Table(columns=columns, data=[[r.get(k) for k in columns] for r in rows])
    scalar_points = []
    for point in aggregate['points']:
        scalar = {**point, **{'return_' + k: v for k, v in point['return_stats'].items()}}
        for kind in CONTRASTS:
            metrics = point[kind + '_metrics'] or {}
            scalar.update({kind + '_' + stat: metrics.get('comparison/' + kind + '_' + stat)
                           for stat in ('mean', 'std', 'ci95_low', 'ci95_high', 'paired_episodes')})
        scalar_points.append(scalar)
    result['comparison/points'] = table(scalar_points, [*POINT_AXES, 'baseline_performance_run_id',
        'return_mean', 'return_std', 'return_episodes',
        *[kind + '_' + stat for kind in CONTRASTS for stat in ('mean', 'std', 'ci95_low', 'ci95_high', 'paired_episodes')]])
    result['comparison/episodes'] = table(aggregate['episodes'], [*POINT_AXES, 'seed', 'solver_seed', 'return', 'length'])
    for kind, (left, right) in CONTRASTS.items():
        result['comparison/' + kind + '_episodes'] = table(aggregate['comparisons'][kind],
            [*POINT_AXES, 'baseline_performance_run_id', 'seed', 'solver_seed', left + '_return', right + '_return', kind])
    result['campaign/settings'] = table(statuses,
        ['setting', 'H', 'J', 'estimator', 'retrace_lambda', 'execution', 'critic_kind', 'status',
         'return_mean', 'return_std', *CONTRASTS, 'performance_url', 'training_url',
         'reward_reference_url', 'historical_mean_url', 'failure'])
    result['campaign/evaluated'] = aggregate['new_evaluated']
    result['campaign/published'] = sum(r['status'] == 'published' for r in statuses)
    return result


def load_completed(campaign, cell):
    bundle = Path(cell['bundle'])
    receipt = read(Path(cell['directory']) / 'worker-completion.json')
    verify_receipt(bundle, receipt)
    assert receipt['cell'] == cell['name'] and receipt['execution'] == cell['execution_mode']
    assert receipt['estimator'] == cell['estimator']
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
    assert receipt['cell'] == cell['name'] and receipt['execution'] == cell['execution_mode']
    checkpoint_step = campaign['checkpoint_step']
    manifest = validate_completed(bundle, cell, campaign)
    record, = load_records(bundle,inventory_path=campaign['inventory'])
    assert record['identity'] == load_run(cell['run_dir'])['identity']
    assert record['metrics'].get('eval/paired_episodes', 0) == 0
    assert all('paired_return_delta' not in episode for episode in record['episodes'])
    reward_reference = cell.get('reward_reference')
    if reward_reference is not None:
        verify_reference(reward_reference)
        assert _key(reward_reference) == _key(cell, cell=True)
    comparison = (paired_delta(record['episodes'], reward_reference['episodes'], 'soft_return_minus_reward')
                  if reward_reference is not None else None)
    staged = stage_completed_bundle(bundle,{cell['actual_selector']:cell['run_dir']},inventory_path=campaign['inventory'])
    assert staged[cell['actual_selector']]['status'] == 'queued'
    performance = publish_performance(cell['run_dir'])
    summary = training_summary(bundle,cell)
    write(directory/'training-summary.json',summary)
    comparison_file = 'soft-return-minus-reward-comparison.json'
    if comparison:
        write(directory/comparison_file, comparison)
    write(directory/'validation-mean-reference.json', cell['mean_reference'])
    if reward_reference is not None:
        write(directory/'reward-reference.json', reward_reference)
    diagnostic = record_from_model_bundle(bundle,cell['actual_selector'],campaign['group']+'-'+cell['name'],
                                         bootstrap_resamples=2000,bootstrap_seed=20260912)
    assert diagnostic['status'] == 'complete' and len(diagnostic['rows']) == 2500*(cell['J']+1)
    write_diagnostic_bundle(directory/'model-series',diagnostic)
    journal = directory/'training-publication.json'
    if journal.exists(): raise RuntimeError('Training publication uncertain; inspect remote run before retry')
    write(journal,dict(status='uncertain',run_id=cell['training_run_id']))
    run = wandb.init(entity=ENTITY,project=PROJECT,id=cell['training_run_id'],resume='never',
                     name='Soft inner, return bootstrap | Inner training | '+cell['name']+f' | {checkpoint_step//1000}k',group=campaign['group'],
                     job_type='inner-training-diagnostics',tags=['closed-loop','H-J-sweep',cell['name'].rsplit('_h',1)[0]],
                     config=dict(H=cell['H'],J=cell['J'],estimator=cell['estimator'],retrace_lambda=cell['retrace_lambda'],
                                 inner_retrace_batch_trajectories=cell['params'].get('inner_retrace_batch_trajectories'),N=128,B=256,C=critic_updates(cell),A=actor_updates(cell),checkpoint_step=checkpoint_step,
                                 checkpoint_sha256=campaign['checkpoint_sha256'],
                                 campaign_group=campaign['group'],
                                 source_run=campaign.get('source_run'),
                                 critic_kind=cell.get('critic_kind'),
                                 overview_url=(f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{campaign["overview_run_id"]}'
                                               if campaign.get('overview_run_id') else None),
                                 execution_mode=cell['execution_mode'], action_rule=action_rule(cell['execution_mode']),
                                 comparison='Mixed critic minus exactly matched reward control when available; new estimator/execution pairs are in the overview',
                                 validation_mean_reference=cell['mean_reference']['performance_run_id'],
                                 setting=cell['name'],resolved_config=manifest['runs'][0]['resolved_config'],
                                 source_code=manifest['code'],reused=cell['reused'],
                                 inner_critic_target_tau=manifest['runs'][0]['resolved_config']['inner_critic_target_tau'],
                                 inner_replay_capacity=manifest['runs'][0]['resolved_config']['inner_replay_capacity'],
                                 inner_replay_reset_each_round=manifest['runs'][0]['resolved_config'].get('inner_replay_reset_each_round',False),
                                 baseline_performance_run_id=(reward_reference['performance_run_id'] if reward_reference else None),
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
        artifact.add_file(str(directory/'validation-mean-reference.json'),name='validation-mean-reference.json')
        if reward_reference is not None:
            artifact.add_file(str(directory/'reward-reference.json'),name='reward-reference.json')
            artifact.add_file(str(Path(reward_reference['bundle'])/'manifest.json'),name='reward-reference-manifest.json')
        artifact.add_file(str(Path(cell['mean_reference']['bundle'])/'manifest.json'),name='validation-mean-manifest.json')
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
          metrics={**record['metrics'], **(comparison['metrics'] if comparison else {}), **execution_proof(manifest)}))



def watch(args):
    """One CPU overview owner; every complete point is logged once by identity."""
    import wandb
    campaign = read(args.root / 'campaign.json')
    for reference in campaign['references']:
        verify_reference(reference, traces=True)
    aggregate_results(campaign, {})
    marker = args.root / 'watcher-started.json'
    if marker.exists():
        raise RuntimeError('Overview already started; inspect W&B history and publication journals before recovery')
    write(marker, dict(pid=os.getpid(), started=time.time()))
    publishers = int(campaign.get('publisher_workers', 2))
    assert 1 <= publishers <= 3
    urls = {c['name']: {
        **{kind + '_url': f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{c[kind + "_run_id"]}'
           for kind in ('performance', 'training')},
        'historical_mean_url': f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{c["mean_reference"]["performance_run_id"]}',
        'reward_reference_url': (f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{c["reward_reference"]["performance_run_id"]}'
                                 if c.get('reward_reference') else None),
    } for c in campaign['cells']}
    config = {key: campaign[key] for key in ('checkpoint_step', 'checkpoint_sha256', 'source_run',
              'source_commit', 'initial_alpha', 'target_entropy')}
    config.update(campaign_group=campaign['group'], protocol='closed-loop-refinement-soft-return-v1',
                  J=campaign['J'], H=campaign['H'], estimator=campaign['estimator'],
                  execution_modes=campaign['execution_modes'], critic_kind='soft_return',
                  retrace_lambda=.9, C=16, A=4, N=128, B=256, execution_std_scale=1.0,
                  inner_replay_capacity_by_J={str(j): 3840 if j == 10 else 3072 for j in campaign['J']},
                  inner_retrace_batch_trajectories_by_H={'1': 256, '2': 128, '3': 86},
                  inner_replay_scope='action', inner_replay_reset_each_round=False,
                  environment_seeds=SEEDS, controller_seed=55, max_decisions=500,
                  comparison='Mixed soft-inner/return-bootstrap versus exact matched reward controls; '
                             'Retrace and execution contrasts use only new mixed-critic measurements.',
                  prior_reference=None, expected_pairs=expected_pairs(campaign),
                  uncertainty='Five matched seeds; 2,000 paired bootstrap resamples; exploratory 95% interval.',
                  result_links=urls)
    run = wandb.init(entity=ENTITY, project=PROJECT, id=campaign['overview_run_id'], resume='never',
                     name=campaign['label'], group=campaign['group'], job_type='soft-return-comparison',
                     tags=['closed-loop', 'soft-return', 'retrace'], config=config, mode='online')
    run.define_metric('axis/inner_rounds')
    run.define_metric('soft_return/*', step_metric='axis/inner_rounds')
    run.define_metric('execution/*', step_metric='axis/inner_rounds')
    run.summary.update(dict(status='running', evaluated=0, published=0, total_settings=len(campaign['cells']),
                            result_type='Full-episode soft-inner/return-bootstrap comparison'))
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
            rows.append(dict(setting=name, H=cell['H'], J=cell['J'], estimator=cell['estimator'],
                             retrace_lambda=cell['retrace_lambda'], execution=cell['execution_mode'],
                             critic_kind=cell['critic_kind'], status=state,
                             return_mean=point.get('return_stats', {}).get('mean'),
                             return_std=point.get('return_stats', {}).get('std'),
                             **{kind: (point.get(kind + '_difference') or {}).get('mean') for kind in CONTRASTS},
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
                    if identity in proofs:
                        row.update(proofs[identity])
                    run.log(row)
                    logged.add(identity)
            run.log(overview_log(wandb, aggregate, statuses))
            run.summary.update(dict(evaluated=len(completed), published=sum(r['status'] == 'published' for r in statuses),
                                    execution_proof_by_setting=proofs, pair_counts=pair_counts(aggregate)))
            write(args.root / 'comparison-results.json', aggregate)
            write(args.root / 'progress.json', dict(rows=statuses, failures=failures, evaluated=len(completed)))
            print('Evaluated %d/%d; published %d' % (len(completed), len(campaign['cells']),
                  sum(r['status'] == 'published' for r in statuses)), flush=True)
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
        aggregate = aggregate_results(campaign, completed)
        observed_pairs, required_pairs = pair_counts(aggregate), expected_pairs(campaign)
        complete = (all(r['status'] == 'published' for r in statuses)
                    and len(completed) == len(campaign['cells']) and observed_pairs == required_pairs)
        status = 'complete' if complete else 'incomplete'
        run.summary.update(dict(status=status, failed_publications=len(failures), pair_counts=observed_pairs, expected_pairs=required_pairs))
        write(args.root / 'campaign-completion.json', dict(status=status, rows=statuses, failures=failures, pair_counts=observed_pairs, expected_pairs=required_pairs))
        if not complete:
            raise RuntimeError('Mixed-critic campaign incomplete; inspect status table and publication logs')
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
