"""Publish alpha-zero, lambda-one, and H4 evaluations against pinned controls."""
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
                                    digest, publish_performance, read, training_summary, write)
from slurm.ambi_closed_loop_publish import gpu_jobs_active, indexed_episodes, moments
from slurm.ambi_closed_loop_reward_retrace_publish import execution_proof, paired_comparison, publication_complete
from slurm.ambi_closed_loop_sampled import verify_receipt
from slurm.ambi_closed_loop_soft_return import action_rule

CONTRASTS = {
    'alpha_zero_minus_alpha_on': ('alpha_zero', 'alpha_on', 'Alpha zero minus adaptive alpha, matched estimator and sampled execution'),
    'lambda_one_mean_minus_one_step_mean': ('lambda_one_mean', 'one_step_mean', 'Retrace lambda 1 minus one-step, both mean execution'),
    'lambda_one_mean_minus_lambda09_sampled': ('lambda_one_mean', 'lambda09_sampled', 'Combined lambda and execution change: lambda 1 mean minus lambda 0.9 sampled'),
    'h4_minus_h3_mean': ('h4_mean', 'h3_mean', 'H4 minus H3, adaptive alpha and one-step mean execution'),
}
ARM_CONTRAST = {'alpha_zero': 'alpha_zero_minus_alpha_on',
                'lambda_one_mean': 'lambda_one_mean_minus_one_step_mean',
                'h4_mean': 'h4_minus_h3_mean'}
POINT_AXES = ('setting', 'experiment_arm', 'H', 'J', 'estimator', 'retrace_lambda', 'execution',
              'alpha_mode', 'critic_kind', 'historical', 'performance_run_id', 'baseline_performance_run_id')


def paired_delta(left, right, kind):
    """Paired bootstrap over complete environment/controller-seed pairs."""
    left_name, right_name, _ = CONTRASTS[kind]
    comparison = paired_comparison(left, right)
    rows = [dict(seed=r['seed'], solver_seed=r['solver_seed'],
                 **{left_name + '_return': r['sampled_return'],
                    right_name + '_return': r['mean_return'], kind: r['sample_minus_mean']})
            for r in comparison['rows']]
    return dict(rows=rows, difference=moments(r[kind] for r in rows),
                metrics={k.replace('sample_minus_mean', kind): v for k, v in comparison['metrics'].items()},
                bootstrap_seed=comparison['bootstrap_seed'], bootstrap_resamples=comparison['bootstrap_resamples'])


def reference_id(reference):
    return reference['performance_run_id']


def comparison_references(cell, references):
    result = [(ARM_CONTRAST[cell['experiment_arm']], cell['paired_reference'])]
    if cell['experiment_arm'] == 'lambda_one_mean':
        secondary, = [r for r in references if (r['H'], r['J'], r['estimator'], r['execution']) ==
                       (cell['H'], cell['J'], 'retrace', 'policy_sample')]
        result.append(('lambda_one_mean_minus_lambda09_sampled', secondary))
    return result


def _validate_pair(cell, reference, kind):
    expected = dict(H=cell['H'], J=cell['J'], estimator=cell['estimator'],
                    execution=cell['execution_mode'])
    if kind == 'lambda_one_mean_minus_one_step_mean':
        expected.update(estimator='one_step', execution='mean')
    elif kind == 'lambda_one_mean_minus_lambda09_sampled':
        expected.update(estimator='retrace', execution='policy_sample')
    elif kind == 'h4_minus_h3_mean':
        expected.update(H=3, estimator='one_step', execution='mean')
    if any(reference.get(k) != v for k, v in expected.items()):
        raise ValueError('Reference axes do not match ' + kind)
    if reference.get('alpha_mode', 'adaptive') != 'adaptive':
        raise ValueError('Expected an adaptive-alpha historical control')
    if reference.get('critic_kind', 'return_only') != 'return_only':
        raise ValueError('Expected a reward-only historical control')
    if reference['estimator'] == 'retrace' and reference.get('retrace_lambda', .9) != .9:
        raise ValueError('Expected historical Retrace lambda 0.9')


def _point(value, episodes, *, historical):
    indexed_episodes(episodes)
    return dict(setting='historical_' + reference_id(value) if historical else value['name'],
                experiment_arm='historical' if historical else value['experiment_arm'], H=value['H'], J=value['J'],
                estimator=value['estimator'],
                retrace_lambda=value.get('retrace_lambda', .9 if value['estimator'] == 'retrace' else None),
                execution=value['execution'] if historical else value['execution_mode'],
                alpha_mode='adaptive' if historical else ('zero' if value['experiment_arm'] == 'alpha_zero' else 'adaptive'),
                critic_kind='return_only', historical=historical, performance_run_id=value['performance_run_id'],
                baseline_performance_run_id=None if historical else reference_id(value['paired_reference']),
                return_stats=moments(e['return'] for e in episodes),
                **{kind + suffix: None for kind in CONTRASTS for suffix in ('_difference', '_metrics')})


def aggregate_results(campaign, completed):
    points, episode_rows, comparisons = [], [], {kind: [] for kind in CONTRASTS}
    references, cells, scientific_keys = {}, {}, set()
    panel_settings = {arm: set() for arm in ARM_CONTRAST}

    def add_point(point, episodes):
        points.append(point)
        episode_rows.extend(dict(**{k: point[k] for k in POINT_AXES},
                                 **{k: e[k] for k in ('seed', 'solver_seed', 'return', 'length')})
                            for e in episodes)

    for reference in campaign['references']:
        rid = reference_id(reference)
        if rid in references:
            raise ValueError('Duplicate historical reference identity')
        references[rid] = reference
        add_point(_point(reference, reference['episodes'], historical=True), reference['episodes'])
    for cell in campaign['cells']:
        key = (cell['experiment_arm'], cell['H'], cell['J'], cell['estimator'], cell['execution_mode'])
        if cell['name'] in cells or key in scientific_keys or cell['experiment_arm'] not in ARM_CONTRAST:
            raise ValueError('Duplicate or unknown campaign setting')
        cells[cell['name']] = cell
        scientific_keys.add(key)
        panel_settings[cell['experiment_arm']].add(cell['name'])
        for kind, reference in comparison_references(cell, campaign['references']):
            _validate_pair(cell, reference, kind)
            rid = reference_id(reference)
            if (rid not in references or indexed_episodes(reference['episodes']) !=
                    indexed_episodes(references[rid]['episodes'])):
                raise ValueError('Comparison reference is not the pinned historical result')
            for field in ('H', 'J', 'estimator', 'execution', 'manifest_sha256'):
                if reference.get(field) != references[rid].get(field):
                    raise ValueError('Comparison reference metadata differs from its pin')
            panel_settings[cell['experiment_arm']].add('historical_' + rid)
    if set(completed) - cells.keys():
        raise ValueError('Unexpected completed setting')
    for name, episodes in completed.items():
        cell = cells[name]
        point = _point(cell, episodes, historical=False)
        add_point(point, episodes)
        for kind, reference in comparison_references(cell, campaign['references']):
            comparison = paired_delta(episodes, reference['episodes'], kind)
            point[kind + '_difference'] = comparison['difference']
            point[kind + '_metrics'] = comparison['metrics']
            comparisons[kind].extend({**{k: point[k] for k in POINT_AXES},
                                     'baseline_performance_run_id': reference_id(reference), **row}
                                     for row in comparison['rows'])
    points.sort(key=lambda p: (p['experiment_arm'], p['H'], p['estimator'], p['execution'], p['J']))
    episode_rows.sort(key=lambda r: (r['setting'], r['seed']))
    for rows in comparisons.values():
        rows.sort(key=lambda r: (r['setting'], r['seed']))
    return dict(points=points, episodes=episode_rows, comparisons=comparisons,
                panel_settings={arm: sorted(names) for arm, names in panel_settings.items()},
                new_evaluated=len(completed),
                comparison={kind: details[2] for kind, details in CONTRASTS.items()},
                uncertainty='Five matched seeds; 2,000 paired bootstrap resamples; exploratory 95% intervals.')


def expected_pairs(campaign):
    counts = dict.fromkeys(CONTRASTS, 0)
    for cell in campaign['cells']:
        for kind, _ in comparison_references(cell, campaign['references']):
            counts[kind] += 1
    return counts


def pair_counts(aggregate):
    return {kind: len({r['setting'] for r in aggregate['comparisons'][kind]}) for kind in CONTRASTS}


def numeric_rows(aggregate):
    rows = []
    for point in aggregate['points']:
        trace = 'one_step' if point['estimator'] == 'one_step' else 'retrace_lambda_' + str(point['retrace_lambda'])
        prefix = f"alpha0_lambda1_h4/{point['alpha_mode']}/h{point['H']}/{trace}/{point['execution']}"
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


def _legend(point):
    estimator = 'one-step' if point['estimator'] == 'one_step' else f"Retrace λ{point['retrace_lambda']:g}"
    execution = 'sampled' if point['execution'] == 'policy_sample' else 'mean'
    return f"H{point['H']} | alpha {point['alpha_mode']} | {estimator} | {execution}"


def chart_payloads(aggregate):
    result = {}
    plots = [(f'panel_{letter}/return_vs_J', 'return_stats', title, set(aggregate['panel_settings'][arm]))
             for letter, arm, title in [('a', 'alpha_zero', 'A: Alpha zero versus adaptive alpha, sampled execution'),
                                       ('b', 'lambda_one_mean', 'B: Retrace lambda 1 mean and historical controls'),
                                       ('c', 'h4_mean', 'C: H4 versus H3, one-step mean execution')]]
    plots += [('comparison/' + kind + '_vs_J', kind + '_difference', details[2], None)
              for kind, details in CONTRASTS.items()]
    for key, field, title, settings in plots:
        groups = defaultdict(list)
        for point in aggregate['points']:
            if point[field] is not None and (settings is None or point['setting'] in settings):
                groups[_legend(point)].append(point)
        if groups:
            series = [(label, sorted(values, key=lambda p: p['J'])) for label, values in sorted(groups.items())]
            if any(len({p['J'] for p in values}) != len(values) for _, values in series):
                raise ValueError('Plot would combine duplicate scientific points')
            result[key] = dict(xs=[[p['J'] for p in values] for _, values in series],
                               ys=[[p[field]['mean'] for p in values] for _, values in series],
                               keys=[label for label, _ in series], title=title, xname='Inner rounds J')
    return result


def overview_log(wandb, aggregate, statuses):
    result = {key: wandb.plot.line_series(**payload) for key, payload in chart_payloads(aggregate).items()}
    def table(rows, columns):
        return wandb.Table(columns=columns, data=[[r.get(k) for k in columns] for r in rows])
    points = []
    for point in aggregate['points']:
        scalar = {**point, **{'return_' + k: v for k, v in point['return_stats'].items()}}
        for kind in CONTRASTS:
            metrics = point[kind + '_metrics'] or {}
            scalar.update({kind + '_' + stat: metrics.get('comparison/' + kind + '_' + stat)
                           for stat in ('mean', 'std', 'ci95_low', 'ci95_high', 'paired_episodes')})
        scalar.update({'in_panel_' + letter: point['setting'] in aggregate['panel_settings'][arm]
                       for letter, arm in [('a', 'alpha_zero'), ('b', 'lambda_one_mean'), ('c', 'h4_mean')]})
        points.append(scalar)
    result['comparison/points'] = table(points, [*POINT_AXES, 'in_panel_a', 'in_panel_b', 'in_panel_c', 'return_mean', 'return_std', 'return_episodes',
        *[kind + '_' + stat for kind in CONTRASTS for stat in ('mean', 'std', 'ci95_low', 'ci95_high', 'paired_episodes')]])
    result['comparison/episodes'] = table(aggregate['episodes'], [*POINT_AXES, 'seed', 'solver_seed', 'return', 'length'])
    for kind, (left, right, _) in CONTRASTS.items():
        result['comparison/' + kind + '_episodes'] = table(aggregate['comparisons'][kind],
            [*POINT_AXES, 'seed', 'solver_seed', left + '_return', right + '_return', kind])
    result['campaign/settings'] = table(statuses,
        [*POINT_AXES[:-2], 'status', 'return_mean', 'return_std', *CONTRASTS,
         'performance_url', 'training_url', 'paired_reference_url', 'secondary_reference_url', 'failure'])
    result['campaign/evaluated'] = aggregate['new_evaluated']
    result['campaign/published'] = sum(r['status'] == 'published' for r in statuses)
    return result


def verify_reference(*args, **kwargs):
    from slurm.ambi_closed_loop_alpha0_lambda1_h4 import verify_reference as verify
    return verify(*args, **kwargs)


def validate_completed(*args, **kwargs):
    from slurm.ambi_closed_loop_alpha0_lambda1_h4 import validate_completed as validate
    return validate(*args, **kwargs)


def protocol_proof(manifest):
    metrics = manifest['runs'][0]['result']['model_metrics']
    result = execution_proof(manifest)
    for key in ('inner_alpha_initial', 'inner_alpha_final', 'inner_temperature_optimizer_steps'):
        for stat in ('mean', 'min', 'max'):
            result[f'protocol/{key}/{stat}'] = metrics[key][stat]
    return result


def load_completed(campaign, cell):
    bundle = Path(cell['bundle'])
    receipt = read(Path(cell['directory']) / 'worker-completion.json')
    verify_receipt(bundle, receipt)
    assert receipt['cell'] == cell['name'] and receipt['execution'] == cell['execution_mode']
    assert receipt['estimator'] == cell['estimator']
    manifest = validate_completed(bundle, cell, campaign)
    return manifest['runs'][0]['episodes'], protocol_proof(manifest)


def publish_cell(args):
    from utils.ambi_benchmark import stage_completed_bundle
    from utils.eval_series import load_run
    from utils.eval_series_data import load_records
    from utils.ambi_diagnostic_series import record_from_model_bundle, write_diagnostic_bundle, diagnostic_history
    import wandb
    campaign = read(args.root / 'campaign.json'); cell = campaign['cells'][args.index]
    directory, bundle = Path(cell['directory']), Path(cell['bundle'])
    if publication_complete(cell):
        return
    receipt = read(directory / 'worker-completion.json')
    verify_receipt(bundle, receipt)
    assert receipt['cell'] == cell['name'] and receipt['execution'] == cell['execution_mode']
    manifest = validate_completed(bundle, cell, campaign)
    record, = load_records(bundle, inventory_path=campaign['inventory'])
    assert record['identity'] == load_run(cell['run_dir'])['identity']
    assert record['metrics'].get('eval/paired_episodes', 0) == 0
    assert all('paired_return_delta' not in episode for episode in record['episodes'])
    comparisons, paired_refs = {}, {}
    for kind, reference in comparison_references(cell, campaign['references']):
        verify_reference(reference)
        _validate_pair(cell, reference, kind)
        comparisons[kind] = paired_delta(record['episodes'], reference['episodes'], kind)
        paired_refs[kind] = reference
    comparison_metrics = {k: v for comparison in comparisons.values() for k, v in comparison['metrics'].items()}
    staged = stage_completed_bundle(bundle, {cell['actual_selector']: cell['run_dir']}, inventory_path=campaign['inventory'])
    assert staged[cell['actual_selector']]['status'] == 'queued'
    performance = publish_performance(cell['run_dir'])
    summary = training_summary(bundle, cell)
    write(directory / 'training-summary.json', summary)
    write(directory / 'paired-comparisons.json', comparisons)
    write(directory / 'paired-references.json', paired_refs)
    diagnostic = record_from_model_bundle(bundle, cell['actual_selector'], campaign['group'] + '-' + cell['name'],
                                         bootstrap_resamples=2000, bootstrap_seed=20260912)
    assert diagnostic['status'] == 'complete' and len(diagnostic['rows']) == 2500 * (cell['J'] + 1)
    write_diagnostic_bundle(directory / 'model-series', diagnostic)
    journal = directory / 'training-publication.json'
    if journal.exists():
        raise RuntimeError('Training publication uncertain; inspect remote run before retry')
    write(journal, dict(status='uncertain', run_id=cell['training_run_id']))
    cfg = manifest['runs'][0]['resolved_config']
    run = wandb.init(entity=ENTITY, project=PROJECT, id=cell['training_run_id'], resume='never',
        name='Inner training | ' + cell['name'] + ' | 575k', group=campaign['group'],
        job_type='inner-training-diagnostics', tags=['closed-loop', cell['experiment_arm']],
        config=dict(H=cell['H'], J=cell['J'], estimator=cell['estimator'], retrace_lambda=cell['retrace_lambda'],
                    experiment_arm=cell['experiment_arm'], alpha_mode=cell['alpha_mode'],
                    inner_entropy_enabled=cfg['inner_entropy_enabled'], inner_temperature_mode=cfg['inner_temperature_mode'],
                    inner_retrace_batch_trajectories=cfg.get('inner_retrace_batch_trajectories'),
                    N=128, B=256, C=critic_updates(cell), A=actor_updates(cell),
                    checkpoint_step=campaign['checkpoint_step'], checkpoint_sha256=campaign['checkpoint_sha256'],
                    campaign_group=campaign['group'], source_run=campaign['source_run'], critic_kind='return_only',
                    overview_url=f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{campaign["overview_run_id"]}',
                    execution_mode=cell['execution_mode'], action_rule=action_rule(cell['execution_mode']),
                    comparison={kind: CONTRASTS[kind][2] for kind in comparisons},
                    setting=cell['name'], resolved_config=cfg, source_code=manifest['code'], reused=False,
                    inner_critic_target_tau=cfg['inner_critic_target_tau'], inner_replay_capacity=cfg['inner_replay_capacity'],
                    inner_replay_scope=cfg['inner_replay_scope'], inner_replay_reset_each_round=False,
                    baseline_performance_run_ids={kind: reference_id(ref) for kind, ref in paired_refs.items()},
                    performance_run_id=cell['performance_run_id'],
                    aggregation='Update curves average all decision roots; decision curves weight five seeds equally.',
                    probe_objective='Reward plus terminal return Q; excludes explicit entropy.'), mode='online')
    try:
        for axis in ('critic_update', 'actor_update', 'decision'):
            run.define_metric('axis/' + axis)
            prefix = {'critic_update': 'critic', 'actor_update': 'actor', 'decision': 'episode'}[axis]
            run.define_metric(prefix + '/*', step_metric='axis/' + axis)
        run.define_metric('seed/*', step_metric='axis/decision')
        run.define_metric('diagnostic/actor_updates')
        run.define_metric('diagnostic/*', step_metric='diagnostic/actor_updates')
        for row in summary['update_curves']:
            prefix = 'critic' if row['axis'] == 'critic_update' else 'actor'
            run.log({'axis/' + row['axis']: row['index'],
                     **{f'{prefix}/{k}/{s}': v for k, stats in row['metrics'].items() for s, v in stats.items()}})
        per_seed = defaultdict(dict)
        for row in summary['per_seed_decisions']:
            per_seed[row['decision']].update({f"seed/{row['episode_id']}/{k}": v for k, v in row['metrics'].items()})
        for row in summary['decision_curves']:
            run.log({'axis/decision': row['decision'], **per_seed[row['decision']],
                     **{f'episode/{k}/{s}': v for k, stats in row['metrics'].items() for s, v in stats.items()}})
        for row in diagnostic_history(diagnostic):
            run.log(row)
        run.log(comparison_metrics)
        artifact = wandb.Artifact('inner-training-' + cell['training_run_id'], type='inner-training-traces',
                                  metadata=dict(manifest_sha256=receipt['manifest_sha256'], reused=False))
        for name in ['manifest.json', *manifest['runs'][0]['trace_files']]:
            artifact.add_file(str(bundle / name), name=name)
        for name in ('training-summary.json', 'paired-comparisons.json', 'paired-references.json'):
            artifact.add_file(str(directory / name), name=name)
        if (bundle / 'execution.json').exists():
            artifact.add_file(str(bundle / 'execution.json'), name='execution.json')
        for kind, ref in paired_refs.items():
            artifact.add_file(str(Path(ref['bundle']) / 'manifest.json'), name=kind + '-reference-manifest.json')
        for name in ('manifest.json', 'paired-rows.jsonl.gz', 'report.html'):
            artifact.add_file(str(directory / 'model-series' / name), name='model-series/' + name)
        run.log_artifact(artifact)
        run.summary.update({**record['metrics'], **comparison_metrics, **protocol_proof(manifest),
            'status': 'complete', 'reused': False, 'diagnostic/paired_rows': len(diagnostic['rows']),
            'training/decisions': 2500, 'training/critic_updates': 2500 * critic_updates(cell) * cell['J'],
            'training/actor_updates': 2500 * actor_updates(cell) * cell['J'],
            'training/temperature_updates': 2500 * actor_updates(cell) * cell['J'] if cfg['inner_entropy_enabled'] else 0,
            'performance_url': f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{cell["performance_run_id"]}'})
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise
    write(journal, dict(status='complete', run_id=cell['training_run_id']))
    write(directory / 'publication-completion.json', dict(status='complete', cell=cell['name'], reused=False,
          performance=performance, training_run_id=cell['training_run_id'],
          metrics={**record['metrics'], **comparison_metrics, **protocol_proof(manifest)}))


def _url(run_id):
    return f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}'


def watch(args):
    """One overview owner and bounded publication subprocesses; no evaluation."""
    import wandb
    campaign = read(args.root / 'campaign.json')
    for reference in campaign['references']:
        verify_reference(reference, traces=True)
    aggregate_results(campaign, {})
    verify_prior(campaign)
    marker = args.root / 'watcher-started.json'
    if marker.exists():
        raise RuntimeError('Overview already started; inspect W&B history and publication journals before recovery')
    write(marker, dict(pid=os.getpid(), started=time.time()))
    publishers = int(campaign.get('publisher_workers', 3))
    assert 1 <= publishers <= 3
    urls = {}
    for cell in campaign['cells']:
        comparisons = comparison_references(cell, campaign['references'])
        urls[cell['name']] = dict(performance_url=_url(cell['performance_run_id']),
            training_url=_url(cell['training_run_id']), paired_reference_url=_url(reference_id(comparisons[0][1])),
            secondary_reference_url=_url(reference_id(comparisons[1][1])) if len(comparisons) > 1 else None)
    config = {key: campaign[key] for key in ('checkpoint_step', 'checkpoint_sha256', 'source_run', 'source_commit')}
    config.update(campaign_group=campaign['group'], protocol='closed-loop-alpha0-lambda1-h4-v1',
        arms=list(ARM_CONTRAST), total_settings=len(campaign['cells']), H=[1, 2, 3, 4], J=[1, 2, 4, 6, 8, 10, 12, 14],
        estimator=['one_step', 'retrace'], C=16, A=4, N=128, B=256, execution_std_scale=1.0,
        settings=[dict(setting=c['name'], arm=c['experiment_arm'], H=c['H'], J=c['J'],
                       estimator=c['estimator'], retrace_lambda=c['retrace_lambda'],
                       execution=c['execution_mode'], alpha_mode=c['alpha_mode'],
                       inner_replay_capacity=c['params']['inner_replay_capacity']) for c in campaign['cells']],
        inner_replay_scope='action', inner_replay_reset_each_round=False, environment_seeds=SEEDS,
        controller_seed=55, max_decisions=500, comparisons={k: v[2] for k, v in CONTRASTS.items()},
        expected_pairs=expected_pairs(campaign),
        uncertainty='Five matched seeds; 2,000 paired bootstrap resamples; exploratory 95% interval.',
        h4_overview_url=_url(campaign['h4_overview_run_id']), result_links=urls)
    run = wandb.init(entity=ENTITY, project=PROJECT, id=campaign['overview_run_id'], resume='never',
        name=campaign['label'], group=campaign['group'], job_type='alpha0-lambda1-h4-comparison',
        tags=['closed-loop', 'alpha-zero', 'retrace-lambda-one', 'h4'], config=config, mode='online')
    run.define_metric('axis/inner_rounds')
    for prefix in ('alpha0_lambda1_h4', 'execution', 'protocol'):
        run.define_metric(prefix + '/*', step_metric='axis/inner_rounds')
    run.summary.update(dict(status='running', evaluated=0, published=0, total_settings=len(campaign['cells'])))
    attempted, completed, proofs, futures, failures, logged = set(), {}, {}, {}, {}, set()
    previous, terminal_since, companion = None, None, None

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
            if publication_complete(cell): state = 'published'
            elif index in failures: state = 'publication_failed'
            elif index in futures: state = 'publishing'
            elif name in completed: state = 'evaluated_awaiting_publication'
            else: state = 'evaluation_incomplete' if terminal else 'queued_or_running'
            point = points.get(name, {})
            rows.append(dict(setting=name, experiment_arm=cell['experiment_arm'], H=cell['H'], J=cell['J'],
                estimator=cell['estimator'], retrace_lambda=cell['retrace_lambda'], execution=cell['execution_mode'],
                alpha_mode=cell['alpha_mode'], critic_kind='return_only', historical=False,
                status=state, return_mean=point.get('return_stats', {}).get('mean'),
                return_std=point.get('return_stats', {}).get('std'),
                **{kind: (point.get(kind + '_difference') or {}).get('mean') for kind in CONTRASTS},
                failure=failures.get(index), **urls[name]))
        return rows

    def update(terminal=False):
        nonlocal previous
        aggregate = aggregate_results(campaign, completed)
        statuses = status_rows(aggregate, terminal)
        stamp = ([(r['setting'], r['status']) for r in statuses], sorted(completed))
        if stamp != previous:
            for identity, row in numeric_rows(aggregate):
                if identity not in logged:
                    row.update(proofs.get(identity, {}))
                    run.log(row)
                    logged.add(identity)
            run.log(overview_log(wandb, aggregate, statuses))
            run.summary.update(dict(evaluated=len(completed), published=sum(r['status'] == 'published' for r in statuses),
                                    protocol_proof_by_setting=proofs, pair_counts=pair_counts(aggregate)))
            write(args.root / 'comparison-results.json', aggregate)
            write(args.root / 'progress.json', dict(rows=statuses, failures=failures, evaluated=len(completed)))
            print('Evaluated %d/%d; published %d' % (len(completed), len(campaign['cells']),
                  sum(r['status'] == 'published' for r in statuses)), flush=True)
            previous = stamp
        return statuses

    try:
        update()
        # A separate process owns the compatibility overview's W&B SDK run.
        # Its only inputs are validated local results; it consumes no GPU work.
        with (args.root / 'h4-overview.log').open('w') as companion_log:
            companion = subprocess.Popen([sys.executable, __file__, 'h4-overview', '--root', str(args.root)],
                                         stdout=companion_log, stderr=subprocess.STDOUT)
            with ThreadPoolExecutor(max_workers=publishers) as pool:
                while True:
                    if companion.poll() not in (None, 0):
                        raise RuntimeError('H4 overview publisher failed; inspect h4-overview.log')
                    for index, future in list(futures.items()):
                        if future.done():
                            rc = future.result()
                            if rc:
                                failures[index] = f'Publisher exited {rc}; inspect {campaign["cells"][index]["directory"]}/publisher.log'
                            del futures[index]
                    for index, cell in enumerate(campaign['cells']):
                        if (Path(cell['directory']) / 'worker-completion.json').exists() and cell['name'] not in completed:
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
            observed, required = pair_counts(aggregate_results(campaign, completed)), expected_pairs(campaign)
            complete = (all(r['status'] == 'published' for r in statuses) and len(completed) == len(campaign['cells'])
                        and observed == required)
            status = 'complete' if complete else 'incomplete'
            write(args.root / 'campaign-completion.json', dict(status=status, rows=statuses, failures=failures,
                                                              pair_counts=observed, expected_pairs=required))
            if companion.wait(timeout=120) != 0:
                raise RuntimeError('H4 overview publisher failed; inspect h4-overview.log')
            companion_receipt = read(args.root / 'h4-overview-completion.json')
            if (companion_receipt.get('status') != 'complete' or companion_receipt.get('points') != 8
                    or companion_receipt.get('run_id') != campaign['h4_overview_run_id']):
                raise RuntimeError('H4 overview completion receipt does not match the eight published settings')
            run.summary.update(dict(status=status, failed_publications=len(failures), pair_counts=observed, expected_pairs=required))
            if not complete:
                raise RuntimeError('Campaign incomplete; inspect status table and publication logs')
            run.finish()
    except BaseException as exc:
        write(args.root / 'campaign-completion.json', dict(status='failed', failure=str(exc), failures=failures))
        run.summary.update(dict(status='failed', failure=str(exc)))
        run.finish(exit_code=1)
        if companion is not None and companion.poll() is None:
            try:
                companion.wait(timeout=45)
            except subprocess.TimeoutExpired:
                companion.terminate()
        raise


def verify_prior(campaign):
    """Reload the pinned historical prior; it has no standalone performance run."""
    from slurm.ambi_closed_loop_alpha0_lambda1_h4 import load_prior
    prior = load_prior(campaign['prior_reference'], campaign['inventory'])
    if indexed_episodes(prior['episodes']) != indexed_episodes(campaign['prior_reference']['episodes']):
        raise ValueError('Frozen prior episodes differ from the verified campaign reference')
    return prior


def h4_results(campaign, aggregate):
    prior = indexed_episodes(campaign['prior_reference']['episodes'])
    episodes = defaultdict(list)
    for row in aggregate['episodes']:
        if row['experiment_arm'] == 'h4_mean':
            episodes[row['setting']].append({**row, 'truncated_by_evaluator': False})
    points, paired = [], []
    for point in aggregate['points']:
        if point['experiment_arm'] != 'h4_mean':
            continue
        if (point['H'], point['estimator'], point['execution'], point['alpha_mode']) != (4, 'one_step', 'mean', 'adaptive'):
            raise ValueError('H4 compatibility overview accepts only its mean one-step arm')
        comparison = paired_comparison(episodes[point['setting']], list(prior.values()))
        delta = [row['sample_minus_mean'] for row in comparison['rows']]
        points.append(dict(setting=point['setting'], J=point['J'], critic='return_only',
                           performance_run_id=point['performance_run_id'],
                           return_stats=point['return_stats'], paired_gain=moments(delta),
                           paired_metrics={key.replace('sample_minus_mean', 'paired_gain'): value
                                           for key, value in comparison['metrics'].items()}))
        paired.extend(dict(setting=point['setting'], J=point['J'], seed=row['seed'], solver_seed=row['solver_seed'],
                           return_only_return=row['sampled_return'], prior_return=row['mean_return'],
                           paired_gain=row['sample_minus_mean']) for row in comparison['rows'])
    points.sort(key=lambda point: point['J'])
    if len({point['J'] for point in points}) != len(points):
        raise ValueError('Duplicate H4 measurement')
    return dict(points=points, paired_rows=paired, prior=moments(e['return'] for e in prior.values()),
                prior_manifest_sha256=campaign['prior_reference']['manifest_sha256'],
                comparison='H4 minus frozen prior; distinct from main Panel C H4 minus H3.')


def h4_chart_payloads(result, rounds):
    points = result['points']
    xs = [point['J'] for point in points]
    return {
        'comparison/return_vs_J': dict(
            xs=[xs, rounds], ys=[[p['return_stats']['mean'] for p in points], [result['prior']['mean']] * len(rounds)],
            keys=['Return-only critic', 'Frozen prior (reused)'], title='H4 full-episode return', xname='Inner rounds J'),
        'comparison/paired_gain_vs_J': dict(
            xs=[xs, rounds], ys=[[p['paired_gain']['mean'] for p in points], [0.] * len(rounds)],
            keys=['Return-only critic', 'No improvement'], title='H4 paired improvement over frozen prior', xname='Inner rounds J'),
    }


def h4_overview(args):
    """Compatibility chart owner; consumes validated H4 rows without rerunning them."""
    import wandb
    campaign = read(args.root / 'campaign.json')
    prior = verify_prior(campaign)
    marker = args.root / 'h4-watcher-started.json'
    if marker.exists():
        raise RuntimeError('H4 overview already started; inspect before recovery')
    write(marker, dict(pid=os.getpid(), started=time.time()))
    cells = [c for c in campaign['cells'] if c['experiment_arm'] == 'h4_mean']
    rounds = sorted(c['J'] for c in cells)
    assert rounds == [1, 2, 4, 6, 8, 10, 12, 14]
    run = wandb.init(entity=ENTITY, project=PROJECT, id=campaign['h4_overview_run_id'], resume='never',
        name=campaign['h4_label'], group=campaign['h4_group'], job_type='closed-loop-critic-comparison',
        tags=['closed-loop', 'H4', 'return-only', 'mean'], config=dict(
            campaign_group=campaign['h4_group'], parent_campaign_group=campaign['group'], H=4, J=rounds,
            C=16, A=4, N=128, B=256, checkpoint_step=campaign['checkpoint_step'],
            checkpoint_sha256=campaign['checkpoint_sha256'], source_commit=campaign['source_commit'],
            source_run=campaign['source_run'], execution_mode='mean', action_rule='tanh_mean',
            alpha_mode='adaptive', estimator='one_step', critic_kind='return_only',
            prior_reference=dict(bundle=prior['bundle'], manifest_sha256=prior['manifest_sha256'],
                                 source_commit=prior['source_commit'], performance_run_id=None,
                                 published_reference_series=prior.get('published_reference_series')),
            parent_overview_url=_url(campaign['overview_run_id']), environment_seeds=SEEDS, controller_seed=55,
            inner_replay_capacity_by_J={str(c['J']): c['params']['inner_replay_capacity'] for c in cells},
            inner_replay_scope='action', inner_replay_reset_each_round=False,
            comparison='H4 paired improvement uses the verified frozen prior, not the H3 horizon comparator.'), mode='online')
    run.define_metric('axis/inner_rounds')
    run.define_metric('h4/*', step_metric='axis/inner_rounds')
    run.summary.update(dict(status='running', evaluated=0, published=0, total_settings=8))
    previous, logged = None, set()
    try:
        while True:
            if (args.root / 'comparison-results.json').exists() and (args.root / 'progress.json').exists():
                result = h4_results(campaign, read(args.root / 'comparison-results.json'))
                statuses = [r for r in read(args.root / 'progress.json')['rows'] if r['experiment_arm'] == 'h4_mean']
                stamp = ([(p['setting'], p['J']) for p in result['points']], [(r['setting'], r['status']) for r in statuses])
                if stamp != previous:
                    for point in result['points']:
                        if point['setting'] not in logged:
                            run.log({'axis/inner_rounds': point['J'],
                                     **{'h4/return_' + key: value for key, value in point['return_stats'].items()},
                                     **{'h4/paired_gain_' + key: value for key, value in point['paired_gain'].items()},
                                     **{'h4/' + key.removeprefix('comparison/'): value for key, value in point['paired_metrics'].items()}})
                            logged.add(point['setting'])
                    payload = {key: wandb.plot.line_series(**value) for key, value in h4_chart_payloads(result, rounds).items()}
                    columns = ['setting', 'J', 'seed', 'solver_seed', 'return_only_return', 'prior_return', 'paired_gain']
                    payload['comparison/paired_episodes'] = wandb.Table(columns=columns, data=[
                        [row[key] for key in columns] for row in result['paired_rows']])
                    columns = ['setting', 'H', 'J', 'status', 'performance_url', 'training_url']
                    payload['campaign/settings'] = wandb.Table(columns=columns, data=[
                        [row[key] for key in columns] for row in statuses])
                    run.log(payload)
                    run.summary.update(dict(evaluated=len(result['points']), published=sum(r['status'] == 'published' for r in statuses)))
                    write(args.root / 'h4-comparison-results.json', result)
                    previous = stamp
                complete = len(result['points']) == len(statuses) == 8 and all(r['status'] == 'published' for r in statuses)
                if complete:
                    run.summary.update(dict(status='complete', failure=None))
                    run.finish()
                    write(args.root / 'h4-overview-completion.json', dict(status='complete', run_id=campaign['h4_overview_run_id'], points=8))
                    return
            if (args.root / 'campaign-completion.json').exists():
                raise RuntimeError('Main campaign ended before all eight H4 points were published')
            time.sleep(15)
    except BaseException as exc:
        run.summary.update(dict(status='failed', failure=str(exc)))
        run.finish(exit_code=1)
        write(args.root / 'h4-overview-completion.json', dict(status='failed', failure=str(exc), run_id=campaign['h4_overview_run_id']))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['watch', 'publish', 'h4-overview'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    {'watch': watch, 'publish': publish_cell, 'h4-overview': h4_overview}[args.mode](args)


if __name__ == '__main__':
    main()
