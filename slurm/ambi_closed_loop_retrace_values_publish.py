"""Publish fresh Retrace value-sampling comparisons and their inner diagnostics."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_aux_hj_sweep import (ENTITY, PROJECT, SEEDS, Moments, actor_updates,
                                    critic_updates, publish_performance, read, write)
from slurm.ambi_closed_loop_publish import gpu_jobs_active, indexed_episodes, moments
from slurm.ambi_closed_loop_retrace_values import ACTION_RULE, CHECKPOINT_SHA, validate_completed
from slurm.ambi_closed_loop_reward_retrace_publish import execution_proof
from slurm.ambi_closed_loop_sampled import verify_receipt

BOOTSTRAP_SEED = 20260924
BOOTSTRAP_RESAMPLES = 2000
COORDINATES = ('H', 'J', 'Ki', 'Kb', 'retrace_lambda')
SAMPLE_PAIRS = ((1, 1), (4, 1), (1, 4), (4, 4))


def setting_key(cell):
    return tuple(cell[k] for k in COORDINATES)


def curve_key(point):
    return point['H'], point['Ki'], point['Kb'], point['retrace_lambda']


def curve_label(key):
    h, ki, kb, lam = key
    return f'H{h} | Ki{ki}/Kb{kb} | lambda{lam:g}'


def campaign_cells(campaign):
    cells = {cell['name']: cell for cell in campaign['cells']}
    keys = [setting_key(cell) for cell in campaign['cells']]
    if len(cells) != len(keys) or len(set(keys)) != len(keys):
        raise ValueError('Duplicate campaign setting identity')
    for cell in cells.values():
        if (cell['retrace_lambda'] != 1.0 or campaign['retrace_lambda'] != 1.0
                or cell['H'] not in (2, 3) or (cell['Ki'], cell['Kb']) not in SAMPLE_PAIRS
                or cell.get('reused', False)):
            raise ValueError('Campaign requires fresh lambda-one Retrace settings')
        for field, value in (('inner_retrace_value_samples', cell['Ki']),
                             ('inner_retrace_boundary_value_samples', cell['Kb']),
                             ('inner_retrace_lambda', 1.0)):
            if cell['params'].get(field) != value:
                raise ValueError('Campaign sample counts or lambda disagree with resolved request')
    return cells


def interval(values):
    import numpy as np
    values = np.asarray(list(values), dtype=float)
    result = moments(values.tolist())
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(values),
                                                        size=(BOOTSTRAP_RESAMPLES, len(values)))
    low, high = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
    return {**result, 'ci95_low': float(low), 'ci95_high': float(high)}


def paired_comparison(episodes, baseline):
    current, reference = indexed_episodes(episodes), indexed_episodes(baseline)
    if current.keys() != reference.keys():
        raise ValueError('Paired environment/controller solver seeds differ')
    rows = [dict(seed=key[0], solver_seed=key[1], baseline_return=reference[key]['return'],
                 return_value=current[key]['return'],
                 paired_gain=current[key]['return'] - reference[key]['return'])
            for key in sorted(current)]
    return dict(rows=rows, stats=interval(row['paired_gain'] for row in rows))


def aggregate_results(campaign, completed, measurements=None):
    """Pair only complete fresh cells with matching H/J/lambda and actual seeds."""
    cells = campaign_cells(campaign)
    if set(completed) - cells.keys():
        raise ValueError('Unexpected completed setting')
    measurements = measurements or {}
    lookup, points, rows, paired = {}, [], [], []
    for name, episodes in completed.items():
        indexed_episodes(episodes)
        lookup[setting_key(cells[name])] = episodes
    for name, episodes in completed.items():
        cell = cells[name]
        point = {key: cell[key] for key in COORDINATES}
        point.update(setting=name, performance_run_id=cell['performance_run_id'],
                     return_stats=interval(e['return'] for e in episodes),
                     **measurements.get(name, {}))
        baseline_key = (cell['H'], cell['J'], 1, 1, cell['retrace_lambda'])
        baseline = lookup.get(baseline_key)
        if baseline is not None:
            comparison = paired_comparison(episodes, baseline)
            point['gain_stats'] = comparison['stats']
            paired.extend(dict(setting=name, **{key: cell[key] for key in COORDINATES}, **row)
                          for row in comparison['rows'])
        points.append(point)
        rows.extend(dict(setting=name, **{key: cell[key] for key in COORDINATES},
                         performance_run_id=cell['performance_run_id'],
                         **{key: ep[key] for key in ('seed', 'solver_seed', 'return', 'length')})
                    for ep in episodes)
    points.sort(key=setting_key)
    return dict(points=points, episodes=rows, paired_comparisons=paired, evaluated=len(completed),
                total_settings=len(cells), total_episodes=5 * len(cells),
                comparison='Each arm minus freshly evaluated Ki1/Kb1 at matching H/J/lambda and environment/controller seeds.',
                uncertainty='Five paired seeds; exploratory percentile bootstrap 95% intervals.',
                bootstrap_seed=BOOTSTRAP_SEED, bootstrap_resamples=BOOTSTRAP_RESAMPLES)


def numeric_rows(aggregate):
    result = []
    for point in aggregate['points']:
        h, _, ki, kb, lam = setting_key(point)
        prefix = f'reward/lambda{lam:g}/h{h}/ki{ki}/kb{kb}'
        result.append((point['setting'], {'axis/inner_rounds': point['J'],
            **{f'{prefix}/return_{key}': value for key, value in point['return_stats'].items()},
            **{f'{prefix}/{key}': point[key] for key in ('control_seconds_per_decision', 'control_seconds')
               if key in point}}))
        if 'gain_stats' in point:
            result.append((point['setting'] + '/paired_gain', {'axis/inner_rounds': point['J'],
                **{f'{prefix}/paired_gain_{key}': value for key, value in point['gain_stats'].items()}}))
    return result


def chart_payloads(aggregate):
    result = {}
    for name, field, statistic, title in (
        ('return_vs_J', 'return_stats', 'mean', 'Full-episode return'),
        ('paired_gain_vs_J', 'gain_stats', 'mean', 'Paired gain versus fresh Ki1/Kb1'),
        ('control_seconds_per_decision_vs_J', 'control_seconds_per_decision', None, 'Control seconds per real decision'),
        ('control_seconds_vs_J', 'control_seconds', None, 'Total control seconds for five episodes'),
    ):
        groups = defaultdict(list)
        for point in aggregate['points']:
            if field in point:
                groups[curve_key(point)].append(point)
        if not groups:
            continue
        items = [(key, sorted(points, key=lambda p: p['J'])) for key, points in sorted(groups.items())]
        result['comparison/' + name] = dict(xs=[[p['J'] for p in points] for _, points in items],
            ys=[[p[field][statistic] if statistic else p[field] for p in points] for _, points in items],
            keys=[curve_label(key) for key, _ in items], title=title, xname='Inner rounds J')
    return result


def diagnostic_chart_payloads(campaign, curves):
    """Compare four arms at the same final J, never merge different solve budgets."""
    cells = campaign_cells(campaign)
    if set(curves) - cells.keys():
        raise ValueError('Unexpected diagnostic setting')
    groups = defaultdict(list)
    for name, payload in curves.items():
        cell = cells[name]
        for axis, rows, metrics in (
            ('critic_updates', payload.get('critic', []),
             ('critic_loss', 'td_error_abs_mean', 'retrace_effective_trace_length', 'retrace_trace_coefficient_mean')),
            ('actor_updates', payload.get('actor', []), ('actor_loss', 'actor_entropy', 'alpha_used')),
            ('round', payload.get('probe', []), ('togo_return_mean', 'togo_return_gain_vs_initial')),
        ):
            for metric in metrics:
                selected = [row for row in rows if metric in row['metrics']]
                if selected:
                    groups[(cell['H'], cell['J'], cell['retrace_lambda'], axis, metric)].append(
                        (cell['Ki'], cell['Kb'], selected))
    result = {}
    for (h, j, lam, axis, metric), arms in sorted(groups.items()):
        arms.sort(key=lambda arm: (arm[0], arm[1]))
        result[f'inner/lambda{lam:g}/h{h}/j{j}/{metric}'] = dict(
            xs=[[row['index'] for row in rows] for _, _, rows in arms],
            ys=[[row['metrics'][metric]['mean'] for row in rows] for _, _, rows in arms],
            keys=[f'Ki{ki}/Kb{kb}' for ki, kb, _ in arms],
            title=f'H{h}, J{j}, lambda{lam:g}: {metric}', xname=axis.replace('_', ' '))
    return result


def confidence_html(aggregate, field, title):
    """Self-contained SVG bands; the locked evaluation runtime needs no plotting package."""
    from html import escape
    points = [point for point in aggregate['points'] if field in point]
    values = [point[field][key] for point in points for key in ('ci95_low', 'ci95_high', 'mean')]
    if field == 'gain_stats':
        values.append(0.)
    low, high = min(values, default=0.), max(values, default=1.)
    pad = max((high - low) * .08, .1)
    low, high = low - pad, high + pad
    y = lambda value: 345 - (value - low) / (high - low) * 265
    colors = ('#2563eb', '#ea580c', '#16a34a', '#9333ea')
    fragments = ['<html><head><meta charset="utf-8"><style>body{margin:0;background:white;font:14px sans-serif;color:#172033}svg{width:100%;height:auto}text{font-family:sans-serif;fill:#172033}</style></head><body>',
                 '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 430" role="img">',
                 f'<title>{escape(title)} with 95% bootstrap intervals</title>',
                 f'<text x="600" y="27" text-anchor="middle" font-size="20">{escape(title)}</text>']
    for h, left in ((2, 75), (3, 675)):
        x = lambda j: left + (j - 1) / 9 * 450
        fragments.append(f'<text x="{left + 225}" y="56" text-anchor="middle" font-size="17">H={h}, lambda=1</text>')
        for tick in range(5):
            value = low + (high - low) * tick / 4
            py = y(value)
            fragments.append(f'<line x1="{left}" x2="{left + 450}" y1="{py}" y2="{py}" stroke="#e2e8f0"/>')
            fragments.append(f'<text x="{left - 8}" y="{py + 4}" text-anchor="end" font-size="11">{value:.3g}</text>')
        for j in (1, 2, 4, 6, 8, 10):
            fragments.append(f'<text x="{x(j)}" y="366" text-anchor="middle">{j}</text>')
        fragments.append(f'<text x="{left + 225}" y="390" text-anchor="middle">Inner rounds J</text>')
        if field == 'gain_stats':
            fragments.append(f'<line x1="{left}" x2="{left + 450}" y1="{y(0)}" y2="{y(0)}" stroke="#64748b" stroke-dasharray="4 3"/>')
        for (ki, kb), color in zip(SAMPLE_PAIRS, colors):
            arm = sorted((p for p in points if p['H'] == h and p['Ki'] == ki and p['Kb'] == kb), key=lambda p: p['J'])
            if not arm:
                continue
            upper = [(x(p['J']), y(p[field]['ci95_high'])) for p in arm]
            lower = [(x(p['J']), y(p[field]['ci95_low'])) for p in reversed(arm)]
            polygon = ' '.join(f'{px:.2f},{py:.2f}' for px, py in upper + lower)
            line = ' '.join(f"{x(p['J']):.2f},{y(p[field]['mean']):.2f}" for p in arm)
            fragments.append(f'<polygon points="{polygon}" fill="{color}" opacity=".15"/>')
            fragments.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2"/>')
            for p in arm:
                px, stats = x(p['J']), p[field]
                fragments.append(f'<line x1="{px}" x2="{px}" y1="{y(stats["ci95_low"])}" y2="{y(stats["ci95_high"])}" stroke="{color}" opacity=".65"/>')
                fragments.append(f'<circle cx="{px}" cy="{y(stats["mean"])}" r="4" fill="{color}"><title>J{p["J"]}: {stats["mean"]:.3f}; 95% [{stats["ci95_low"]:.3f}, {stats["ci95_high"]:.3f}]</title></circle>')
    for index, ((ki, kb), color) in enumerate(zip(SAMPLE_PAIRS, colors)):
        left = 215 + index * 205
        fragments.append(f'<line x1="{left}" x2="{left + 24}" y1="410" y2="410" stroke="{color}" stroke-width="3"/>')
        fragments.append(f'<text x="{left + 30}" y="415">Ki{ki}/Kb{kb}</text>')
    fragments.append('</svg><p style="text-align:center;margin:4px 12px 12px">Five matched seeds; exploratory 95% percentile bootstrap intervals. Missing measurements remain absent.</p></body></html>')
    return ''.join(fragments)

def overview_log(wandb, aggregate, statuses, *, campaign=None, curves=None):
    charts = chart_payloads(aggregate)
    if campaign is not None:
        charts.update(diagnostic_chart_payloads(campaign, curves or {}))
    output = {key: wandb.plot.line_series(**payload) for key, payload in charts.items()}
    if aggregate['points']:
        for field, title in (('return_stats', 'Full-episode return'), ('gain_stats', 'Paired improvement')):
            if any(field in point for point in aggregate['points']):
                output['comparison/' + field + '_95ci'] = wandb.Html(confidence_html(aggregate, field, title), inject=False)
    def table(rows, columns):
        return wandb.Table(columns=columns, data=[[row.get(key) for key in columns] for row in rows])
    output['comparison/episodes'] = table(aggregate['episodes'],
        ['setting', *COORDINATES, 'seed', 'solver_seed', 'return', 'length', 'performance_run_id'])
    output['comparison/paired_episodes'] = table(aggregate['paired_comparisons'],
        ['setting', *COORDINATES, 'seed', 'solver_seed', 'baseline_return', 'return_value', 'paired_gain'])
    output['comparison/points'] = table([
        {**point, **{'return_' + key: value for key, value in point['return_stats'].items()},
         **{'gain_' + key: value for key, value in point.get('gain_stats', {}).items()}}
        for point in aggregate['points']],
        ['setting', *COORDINATES, 'return_mean', 'return_std', 'return_ci95_low', 'return_ci95_high',
         'gain_mean', 'gain_std', 'gain_ci95_low', 'gain_ci95_high', 'control_seconds_per_decision', 'performance_run_id'])
    output['campaign/settings'] = table(statuses,
        ['setting', *COORDINATES, 'status', 'pair_status', 'return_mean', 'paired_gain',
         'performance_url', 'training_url', 'failure'])
    output.update({'campaign/evaluated': aggregate['evaluated'],
                   'campaign/published': sum(row['status'] == 'published' for row in statuses),
                   'campaign/total_settings': aggregate['total_settings'],
                   'campaign/total_episodes': aggregate['total_episodes']})
    return output


def load_completed(campaign, cell):
    receipt = read(Path(cell['directory']) / 'worker-completion.json')
    if receipt['cell'] != cell['name'] or receipt['execution'] != 'policy_sample':
        raise ValueError('Worker completion identity mismatch')
    verify_receipt(Path(cell['bundle']), receipt)
    manifest = validate_completed(Path(cell['bundle']), cell, campaign)
    episodes = manifest['runs'][0]['episodes']
    indexed_episodes(episodes)
    measurement = execution_proof(manifest)
    if all(isinstance(ep.get('control_seconds'), (int, float)) and math.isfinite(ep['control_seconds'])
           for ep in episodes):
        measurement['control_seconds'] = sum(ep['control_seconds'] for ep in episodes)
        measurement['control_seconds_per_decision'] = measurement['control_seconds'] / sum(ep['length'] for ep in episodes)
    return episodes, measurement


def training_summary(bundle, cell, *, expected_steps=500):
    """Stream all rows, assigning Retrace diagnostics to their critic update."""
    manifest = read(Path(bundle)/'manifest.json')
    run, = manifest['runs']
    curves = defaultdict(lambda: defaultdict(Moments))
    decisions = defaultdict(lambda: defaultdict(Moments))
    counts = defaultdict(lambda: defaultdict(int))
    for name in run['trace_files']:
        with gzip.open(Path(bundle)/name,'rt') as f:
            for line in f:
                e = json.loads(line)
                ep, decision = e['episode_id'], e['decision_index']
                key = (ep, decision)
                assert 0 <= decision < expected_steps
                assert not e.get('nonfinite'), e.get('nonfinite')
                phase = e['phase']
                if phase == 'initial':
                    assert e['replay_size'] == 0
                    counts[key]['initial'] += 1
                if phase == 'collection':
                    retained = 1 if cell['params'].get('inner_replay_reset_each_round',False) else e['round_index']
                    assert e['replay_size'] == retained*128*cell['H']
                    counts[key]['collection'] += 1
                if phase == 'update':
                    for component in ('critic','actor','temperature'):
                        if e.get('updated_'+component):
                            counts[key][component] += 1
                    if cell['params'].get('inner_component_update_order') == 'interleaved':
                        # Validate the chronology, not just final optimizer totals.
                        c, a = counts[key]['critic'], counts[key]['actor']
                        interval = critic_updates(cell) // actor_updates(cell)
                        assert bool(e.get('updated_critic')) != bool(e.get('updated_actor'))
                        assert e['critic_updates'] == c and e['actor_updates'] == a
                        assert c == a*interval if e.get('updated_actor') else a*interval < c <= (a+1)*interval
                        assert bool(e.get('updated_temperature')) == bool(e.get('updated_actor'))
                    elif cell['params'].get('inner_component_update_order') == 'critic_first':
                        c, a, r = counts[key]['critic'], counts[key]['actor'], e['round_index']
                        assert bool(e.get('updated_critic')) != bool(e.get('updated_actor'))
                        assert e['critic_updates'] == c and e['actor_updates'] == a
                        if e.get('updated_critic'):
                            assert a == (r-1)*actor_updates(cell) and (r-1)*critic_updates(cell) < c <= r*critic_updates(cell)
                        else:
                            assert c == r*critic_updates(cell) and (r-1)*actor_updates(cell) < a <= r*actor_updates(cell)
                        assert bool(e.get('updated_temperature')) == bool(e.get('updated_actor') and cell['params']['inner_entropy_enabled'])
                    for metric,value in e['metrics'].items():
                        assert value is not None
                        if metric.startswith(('critic_', 'q_', 'td_error', 'retrace_')):
                            axis, index = 'critic_update', e['critic_updates']
                        elif metric.startswith(('temperature_', 'alpha_')):
                            if not e.get('updated_actor'): continue
                            axis, index = 'actor_update', e['actor_updates']
                        else:
                            axis, index = 'actor_update', e['actor_updates']
                        curves[(axis,index)][metric].add(value)
                        decisions[(ep,decision)][metric].add(value)
                elif phase == 'decision':
                    for metric,value in e['metrics'].items():
                        assert value is not None
                        decisions[(ep,decision)][metric].add(value)
                    counts[key]['decision'] += 1
    expected_n = len(run['episodes'])*expected_steps
    assert len(counts) == expected_n
    for count in counts.values():
        assert dict(count) == dict(initial=1, collection=cell['J'], critic=critic_updates(cell)*cell['J'],
                                  actor=actor_updates(cell)*cell['J'],
                                  **({'temperature':actor_updates(cell)*cell['J']} if cell['params']['inner_entropy_enabled'] else {}), decision=1), count
    required = {'critic_loss','critic_grad_norm','td_error_abs_mean','q_target_mean','actor_loss',
                'actor_grad_norm','actor_entropy','alpha_used'}
    assert required <= {k for d in curves.values() for k in d}, required
    packed = []
    for (axis,index),metrics in sorted(curves.items()):
        packed.append(dict(axis=axis,index=index,metrics={k:v.summary() for k,v in metrics.items()}))
    # Each seed/decision is retained; cross-seed summaries weight seeds equally.
    by_decision = defaultdict(lambda: defaultdict(Moments))
    per_seed = []
    for (ep,decision),metrics in sorted(decisions.items()):
        means = {k:v.total/v.n for k,v in metrics.items()}
        per_seed.append(dict(episode_id=ep,decision=decision,metrics=means))
        for k,v in means.items(): by_decision[decision][k].add(v)
    return dict(update_curves=packed, per_seed_decisions=per_seed,
                decision_curves=[dict(decision=i,metrics={k:v.summary() for k,v in ms.items()})
                                 for i,ms in sorted(by_decision.items())],
                trace_rows_checked=sum(sum(v.values()) for v in counts.values()),
                metric_catalog=manifest['metric_catalog'])


def publication_complete(cell):
    path = Path(cell['directory']) / 'publication-completion.json'
    if not path.exists():
        return False
    receipt = read(path)
    training = read(path.with_name('training-publication.json'))
    if (receipt.get('status') != 'complete' or receipt.get('cell') != cell['name']
            or receipt.get('setting_key') != list(setting_key(cell))
            or receipt.get('training_run_id') != cell['training_run_id']
            or training.get('status') != 'complete' or training.get('run_id') != cell['training_run_id']
            or receipt.get('performance', {}).get('run_id') != cell['performance_run_id']
            or receipt.get('performance', {}).get('published') != 1):
        raise ValueError('Value-sampling publication completion identity mismatch')
    return True


def publish_cell(args):
    from utils.ambi_benchmark import stage_completed_bundle
    from utils.eval_series import load_run
    from utils.eval_series_data import load_records
    from utils.ambi_diagnostic_series import record_from_model_bundle, write_diagnostic_bundle, diagnostic_history
    import wandb
    campaign = read(args.root / 'campaign.json')
    campaign_cells(campaign)
    cell = campaign['cells'][args.index]
    directory, bundle = Path(cell['directory']), Path(cell['bundle'])
    if publication_complete(cell):
        return
    _, measurement = load_completed(campaign, cell)
    receipt = read(directory / 'worker-completion.json')
    manifest = read(bundle / 'manifest.json')
    record, = load_records(bundle, inventory_path=campaign['inventory'])
    assert record['identity'] == load_run(cell['run_dir'])['identity']
    assert record['metrics'].get('eval/paired_episodes', 0) == 0
    assert all('paired_return_delta' not in ep for ep in record['episodes'])
    staged = stage_completed_bundle(bundle, {cell['actual_selector']: cell['run_dir']},
                                    inventory_path=campaign['inventory'])
    assert staged[cell['actual_selector']]['status'] == 'queued'
    performance = publish_performance(cell['run_dir'])
    summary = training_summary(bundle, cell)
    write(directory / 'training-summary.json', summary)
    diagnostic = record_from_model_bundle(bundle, cell['actual_selector'], campaign['group'] + '-' + cell['name'],
                                         bootstrap_resamples=BOOTSTRAP_RESAMPLES, bootstrap_seed=BOOTSTRAP_SEED)
    assert diagnostic['status'] == 'complete' and len(diagnostic['rows']) == 2500 * (cell['J'] + 1)
    write_diagnostic_bundle(directory / 'model-series', diagnostic)
    curves = {kind: [dict(index=row['index'], metrics=row['metrics']) for row in summary['update_curves']
                     if row['axis'] == kind + '_update'] for kind in ('critic', 'actor')}
    curves['probe'] = [dict(index=row['round_index'], metrics=row['metrics']) for row in diagnostic['summaries']]
    write(directory / 'overview-curves.json', curves)
    journal = directory / 'training-publication.json'
    if journal.exists():
        raise RuntimeError('Training publication uncertain; inspect remote run before retry')
    write(journal, dict(status='uncertain', run_id=cell['training_run_id']))
    config = {key: cell[key] for key in COORDINATES}
    config.update(inner_retrace_value_samples=cell['Ki'], inner_retrace_boundary_value_samples=cell['Kb'],
                  estimator='retrace', execution_mode='policy_sample', action_rule=ACTION_RULE,
                  N=128, B=256, C=critic_updates(cell), A=actor_updates(cell),
                  checkpoint_step=campaign['checkpoint_step'], checkpoint_sha256=campaign['checkpoint_sha256'],
                  source_run=campaign['source_run'], campaign_group=campaign['group'], setting=cell['name'],
                  resolved_config=manifest['runs'][0]['resolved_config'], source_code=manifest['code'],
                  overview_url=f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{campaign["overview_run_id"]}',
                  performance_url=f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{cell["performance_run_id"]}',
                  comparison='Fresh matched Ki1/Kb1 comparisons appear in the campaign overview.',
                  aggregation='Update curves average all decision roots; decision and probe curves weight five seeds equally.',
                  probe_objective='Reward plus frozen terminal Q; excludes explicit entropy.')
    run = wandb.init(entity=ENTITY, project=PROJECT, id=cell['training_run_id'], resume='never',
                     name=f'Retrace lambda1 | H{cell["H"]} J{cell["J"]} Ki{cell["Ki"]}/Kb{cell["Kb"]} | Inner training | 575k',
                     group=campaign['group'], job_type='inner-training-diagnostics',
                     tags=['closed-loop', 'retrace-value-samples', 'return-only'], config=config, mode='online')
    try:
        for axis, prefix in (('critic_update', 'critic'), ('actor_update', 'actor'), ('decision', 'episode')):
            run.define_metric('axis/' + axis)
            run.define_metric(prefix + '/*', step_metric='axis/' + axis)
        run.define_metric('seed/*', step_metric='axis/decision')
        run.define_metric('diagnostic/actor_updates')
        run.define_metric('diagnostic/*', step_metric='diagnostic/actor_updates')
        for row in summary['update_curves']:
            prefix = 'critic' if row['axis'] == 'critic_update' else 'actor'
            run.log({'axis/' + row['axis']: row['index'],
                     **{f'{prefix}/{key}/{stat}': value for key, stats in row['metrics'].items()
                        for stat, value in stats.items()}})
        per_seed = defaultdict(dict)
        for row in summary['per_seed_decisions']:
            per_seed[row['decision']].update({f'seed/{row["episode_id"]}/{key}': value
                                             for key, value in row['metrics'].items()})
        for row in summary['decision_curves']:
            run.log({'axis/decision': row['decision'], **per_seed[row['decision']],
                     **{f'episode/{key}/{stat}': value for key, stats in row['metrics'].items()
                        for stat, value in stats.items()}})
        for row in diagnostic_history(diagnostic):
            run.log(row)
        artifact = wandb.Artifact('inner-training-' + cell['training_run_id'], type='inner-training-traces',
                                  metadata=dict(manifest_sha256=receipt['manifest_sha256'],
                                                setting_key=list(setting_key(cell))))
        for name in ['manifest.json', *manifest['runs'][0]['trace_files']]:
            artifact.add_file(str(bundle / name), name='bundle/' + name)
        if (bundle / 'execution.json').exists():
            artifact.add_file(str(bundle / 'execution.json'), name='bundle/execution.json')
        for name in ('training-summary.json', 'overview-curves.json'):
            artifact.add_file(str(directory / name), name=name)
        for name in ('manifest.json', 'paired-rows.jsonl.gz', 'report.html'):
            artifact.add_file(str(directory / 'model-series' / name), name='model-series/' + name)
        run.log_artifact(artifact)
        run.summary.update({**record['metrics'], **measurement, 'status': 'complete',
                            'diagnostic/paired_rows': len(diagnostic['rows']),
                            'training/decisions': 2500,
                            'training/critic_updates': 2500 * critic_updates(cell) * cell['J'],
                            'training/actor_updates': 2500 * actor_updates(cell) * cell['J'],
                            'performance_url': config['performance_url'], 'overview_url': config['overview_url']})
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise
    write(journal, dict(status='complete', run_id=cell['training_run_id']))
    write(directory / 'publication-completion.json', dict(status='complete', cell=cell['name'],
          setting_key=list(setting_key(cell)), performance=performance, training_run_id=cell['training_run_id'],
          metrics={**record['metrics'], **measurement}))


def watch(args):
    """One CPU overview owner; complete measurements and delayed pairs log once."""
    import wandb
    campaign = read(args.root / 'campaign.json')
    campaign_cells(campaign)
    marker = args.root / 'watcher-started.json'
    if marker.exists():
        raise RuntimeError('Overview already started; inspect W&B and publication journals before recovery')
    with marker.open('x') as handle:
        json.dump(dict(pid=os.getpid(), started=time.time()), handle)
    publishers = int(campaign.get('publisher_workers', 2))
    if not 1 <= publishers <= 3:
        raise ValueError('Publisher concurrency must be between one and three')
    urls = {cell['name']: {kind + '_url': f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{cell[kind + "_run_id"]}'
                          for kind in ('performance', 'training')} for cell in campaign['cells']}
    config = {key: campaign[key] for key in ('checkpoint_step', 'checkpoint_sha256', 'source_run',
              'source_commit', 'initial_alpha', 'target_entropy', 'H', 'J', 'sample_pairs', 'retrace_lambda')}
    config.update(campaign_group=campaign['group'], protocol='closed-loop-refinement-sampled-execution-v1',
                  estimator='retrace', C=16, A=4, N=128, B=256, critic_kind='return_only',
                  execution_mode='policy_sample', action_rule=ACTION_RULE, environment_seeds=SEEDS,
                  controller_seed=55, max_decisions=500, total_settings=len(campaign['cells']),
                  total_episodes=5 * len(campaign['cells']), result_links=urls,
                  comparison='Fresh same-H/J/lambda Ki1/Kb1; no historical results reused.',
                  uncertainty='Five paired seeds; 2,000 bootstrap resamples; exploratory 95% intervals.',
                  inner_temperature_mode='auto', inner_temperature_initialization='inherit_outer')
    run = wandb.init(entity=ENTITY, project=PROJECT, id=campaign['overview_run_id'], resume='never',
                     name=campaign['label'], group=campaign['group'], job_type='retrace-value-sampling-comparison',
                     tags=['closed-loop', 'return-only', 'retrace', 'value-samples'], config=config, mode='online')
    run.define_metric('axis/inner_rounds')
    run.define_metric('reward/*', step_metric='axis/inner_rounds')
    run.summary.update(dict(status='running', evaluated=0, published=0,
                            total_settings=len(campaign['cells']), total_episodes=5 * len(campaign['cells'])))
    attempted, completed, measurements, curves, futures = set(), {}, {}, {}, {}
    failures, invalid, logged = {}, {}, set()
    previous, terminal_since = None, None

    def launch(index):
        cell = campaign['cells'][index]
        with (Path(cell['directory']) / 'publisher.log').open('w') as log:
            return subprocess.run([sys.executable, __file__, 'publish', '--root', str(args.root), '--index', str(index)],
                                  stdout=log, stderr=subprocess.STDOUT).returncode

    def update(terminal=False):
        nonlocal previous
        aggregate = aggregate_results(campaign, completed, measurements)
        points = {point['setting']: point for point in aggregate['points']}
        statuses = []
        for index, cell in enumerate(campaign['cells']):
            name, point = cell['name'], points.get(cell['name'], {})
            if publication_complete(cell):
                status = 'published'
            elif index in failures:
                status = 'publication_failed'
            elif index in invalid:
                status = 'evaluation_failed'
            elif index in futures:
                status = 'publishing'
            elif name in completed:
                status = 'evaluated_awaiting_publication'
            else:
                status = 'evaluation_incomplete' if terminal else 'queued_or_running'
            statuses.append(dict(setting=name, **{key: cell[key] for key in COORDINATES}, status=status,
                pair_status='complete' if 'gain_stats' in point else 'waiting_for_both_cells',
                return_mean=point.get('return_stats', {}).get('mean'),
                paired_gain=point.get('gain_stats', {}).get('mean'),
                failure=failures.get(index, invalid.get(index)), **urls[name]))
        stamp = ([(row['setting'], row['status']) for row in statuses], sorted(completed), sorted(curves))
        if stamp != previous:
            for identity, row in numeric_rows(aggregate):
                if identity not in logged:
                    run.log(row)
                    logged.add(identity)
            run.log(overview_log(wandb, aggregate, statuses, campaign=campaign, curves=curves))
            published = sum(row['status'] == 'published' for row in statuses)
            run.summary.update(dict(evaluated=len(completed), published=published,
                                    paired_settings=sum('gain_stats' in point for point in aggregate['points']),
                                    diagnostic_settings=len(curves)))
            write(args.root / 'comparison-results.json', aggregate)
            write(args.root / 'progress.json', dict(rows=statuses, failures={**invalid, **failures},
                                                  evaluated=len(completed), published=published))
            print(f'Evaluated {len(completed)}/{len(campaign["cells"])}; published {published}', flush=True)
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
                    directory, name = Path(cell['directory']), cell['name']
                    if index not in invalid and name not in completed:
                        if (directory / 'worker-completion.json').exists():
                            try:
                                completed[name], measurements[name] = load_completed(campaign, cell)
                            except Exception as exc:
                                invalid[index] = f'Worker result validation failed: {exc}'
                        elif (directory / 'worker-failure.json').exists():
                            invalid[index] = json.dumps(read(directory / 'worker-failure.json'), sort_keys=True)
                    if name in completed and name not in curves and (directory / 'overview-curves.json').exists():
                        curves[name] = read(directory / 'overview-curves.json')
                    if (len(futures) < publishers and index not in attempted and name in completed
                            and not publication_complete(cell)):
                        attempted.add(index)
                        futures[index] = pool.submit(launch, index)
                statuses = update()
                if all(row['status'] == 'published' for row in statuses):
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
        aggregate = aggregate_results(campaign, completed, measurements)
        complete = (all(row['status'] == 'published' for row in statuses)
                    and all('gain_stats' in point for point in aggregate['points'])
                    and len(aggregate['points']) == len(campaign['cells']))
        status = 'complete' if complete else 'incomplete'
        run.summary.update(dict(status=status, failed_publications=len(failures), failed_evaluations=len(invalid)))
        write(args.root / 'campaign-completion.json', dict(status=status, rows=statuses, failures={**invalid, **failures}))
        if not complete:
            raise RuntimeError('Retrace value-sampling campaign incomplete; inspect status table and publication logs')
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
    if args.mode == 'publish' and args.index is None:
        parser.error('--index is required for publish')
    {'watch': watch, 'publish': publish_cell}[args.mode](args)


if __name__ == '__main__':
    main()
