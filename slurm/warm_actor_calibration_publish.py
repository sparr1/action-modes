#!/usr/bin/env python3
"""Read immutable branching-evaluation shards and publish a compact comparison.

The ``report`` command is entirely local. ``publish`` and ``watch`` are explicit
W&B mutations owned by one CPU process; simulation workers never call W&B.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import html
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.wandb_results_layout import (DEFAULT_VIEW_NAME, _write, ensure_results_layout)

LAYOUT_VERSION = 'warm-actor-calibration-v1'
IMAGE_KEYS = ('calibration/gains', 'calibration/errors',
              'calibration/action_selection', 'calibration/replanning')
TABLE_KEYS = ('calibration/progress', 'calibration/measurements')
DEFAULT_GLOBS = ('prefix/*/seed-*/decision-*.json', 'replan/*/seed-*/decision-*.json')
IDENTITY = ('source_cell', 'H', 'J', 'seed', 'decision', 'branch_kind',
            'actor_family', 'round', 'action_mode', 'replicate')
GROUP = ('source_cell', 'H', 'J', 'branch_kind', 'actor_family', 'round', 'action_mode')
RETURN_KEYS = ('predicted_model_return', 'real_bootstrapped_return', 'real_mc_return')
OPTIONAL_DIAGNOSTICS = ('critic_head_sd', 'critic_preference_gap')
TERMINAL_REDUCTIONS = ('expected_mean_pair', 'expected_min_pair', 'min_all')
EXPLANATION = (
    'Identical saved simulator roots; paired actor/prior branches. Prefix returns '
    'are discounted reward-only values (gamma 0.99): H candidate actions then a '
    'sampled frozen-prior tail. Replanning returns are remaining-episode raw '
    'reward and must not be compared numerically with prefix values. Means '
    'average replicates within root, roots within seed, then seeds equally. '
    'The measurements table contains exploratory 95% seed-cluster bootstrap '
    'intervals; native chart lines show means only. Five seeds '
    'do not provide a confirmatory uncertainty estimate. Pending results are '
    'missing, never zero. Partial results can have fewer roots/seeds; consult '
    'the measurements table. Cold candidates are fresh solves at warm-visited '
    'states, not independent cold-controller episodes. Round k inside J10 '
    'retains the J10 source history and is not a J=k campaign.'
)


def read(path):
    return json.loads(Path(path).read_text())


def key(row, fields=IDENTITY):
    return tuple(row[name] for name in fields)


def normalize_record(record, shard):
    row = {name: shard[name] for name in IDENTITY if name in shard}
    row.update(record)
    absent = [name for name in IDENTITY if name not in row]
    if absent:
        raise ValueError(f'Missing record identity fields: {absent}')
    if row['branch_kind'] not in {'prefix', 'replan'}:
        raise ValueError('Unknown branch kind')
    if row['actor_family'] not in {'warm', 'cold', 'prior'}:
        raise ValueError('Unknown actor family')
    if row['action_mode'] not in {'sample', 'mean'}:
        raise ValueError('Unknown action mode')
    for name in ('H', 'J', 'seed', 'decision', 'round', 'replicate'):
        if isinstance(row[name], bool) or not isinstance(row[name], int):
            raise ValueError(f'{name} must be an integer')
    if row['round'] < 0 or row['round'] > row['J']:
        raise ValueError('Actor stage lies outside its source solve budget')
    required = RETURN_KEYS if row['branch_kind'] == 'prefix' else ('real_mc_return',)
    for name in required + OPTIONAL_DIAGNOSTICS:
        if name not in row:
            if name in required:
                raise ValueError(f'Missing {name}')
            continue
        if row[name] is None and name in OPTIONAL_DIAGNOSTICS:
            continue
        if not isinstance(row[name], (int, float)) or not math.isfinite(row[name]):
            raise ValueError(f'{name} must be finite')
    if row['branch_kind'] == 'prefix':
        a, b, c = (row[name] for name in RETURN_KEYS)
        row.update(modeled_prefix_error=a-b, terminal_value_error=b-c,
                   total_value_error=a-c)
        for reduction in TERMINAL_REDUCTIONS:
            model_key, real_key = (f'predicted_model_return_{reduction}',
                                   f'real_bootstrapped_return_{reduction}')
            if model_key in row and real_key in row:
                model, real = row[model_key], row[real_key]
                if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (model, real)):
                    raise ValueError('Terminal reduction returns must be finite')
                row.update({f'modeled_prefix_error_{reduction}': model-real,
                            f'terminal_value_error_{reduction}': real-c,
                            f'total_value_error_{reduction}': model-c,
                            f'terminal_value_abs_error_{reduction}': abs(real-c)})
    for name in ('source_cell', 'H', 'J', 'seed', 'decision'):
        if name in shard and row[name] != shard[name]:
            raise ValueError(f'Shard and record disagree on {name}')
    return row


def validate_shard_scope(rows, campaign):
    """Do not declare a root complete with missing candidate modes or replicates."""
    first = rows[0]
    kind = first['branch_kind']
    tasks = campaign.get('prefixes' if kind == 'prefix' else 'replans')
    if tasks is None:  # Standalone local reports may provide only explicit counts.
        return
    fields = ('source_cell', 'H', 'J', 'seed', 'decision')
    matches = [task for task in tasks if key(task, fields) == key(first, fields)]
    if len(matches) != 1 or any(key(row, fields) != key(first, fields) for row in rows):
        raise ValueError('Completed shard is not one unique declared simulator root')
    if kind == 'prefix':
        stages = [stage for stage in (0, 1, 2, 4, 6, 8, 10) if stage <= first['J']]
        candidates = [(family, stage) for family in ('warm', 'cold') for stage in stages] + [('prior', 0)]
        expected = {(family, stage, mode, replicate) for family, stage in candidates
                    for mode in ('sample', 'mean') for replicate in range(campaign['rollouts'])}
    else:
        expected = {(family, stage, 'mean', 0) for family, stage in
                    (('prior', 0), ('warm', 0), ('warm', 4), ('warm', first['J']))}
    actual = {key(row, ('actor_family', 'round', 'action_mode', 'replicate')) for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError('Completed root has missing, duplicated, or unexpected branch coverage')


def load_shards(root, campaign):
    """Only completed shards count; reject duplicate observations and corruption."""
    paths = sorted({path for pattern in campaign.get('result_globs', DEFAULT_GLOBS)
                    for path in Path(root).glob(pattern)})
    records, fingerprints, seen = [], [], set()
    completed = defaultdict(int)
    for path in paths:
        payload = path.read_bytes()
        shard = json.loads(payload)
        if shard.get('status') != 'complete':
            continue
        rows = [normalize_record(row, shard) for row in shard.get('records', [])]
        if not rows:
            raise ValueError(f'Completed shard contains no records: {path}')
        kinds = {row['branch_kind'] for row in rows}
        if len(kinds) != 1:
            raise ValueError(f'Shard mixes prefix and replanning returns: {path}')
        validate_shard_scope(rows, campaign)
        for row in rows:
            identity = key(row)
            if identity in seen:
                raise ValueError(f'Duplicate branch identity: {identity}')
            seen.add(identity)
        records.extend(rows)
        completed[next(iter(kinds))] += 1
        fingerprints.append(dict(path=str(path.relative_to(root)),
                                 sha256=hashlib.sha256(payload).hexdigest(), records=len(rows)))
    if campaign.get('expected_capture_shards'):
        for task in campaign.get('captures', []):
            path = Path(task['capture_directory'])/'manifest.json'
            if path.exists():
                payload = path.read_bytes()
                if json.loads(payload).get('status') == 'complete':
                    completed['capture'] += 1
                    fingerprints.append(dict(path=str(path.relative_to(root)),
                        sha256=hashlib.sha256(payload).hexdigest(), records=0))
    progress = [dict(branch_kind=kind, complete=completed[kind],
                     expected=campaign.get(f'expected_{kind}_shards', 0))
                for kind in (('capture', 'prefix', 'replan') if campaign.get('expected_capture_shards') else ('prefix', 'replan'))]
    for status in progress:
        if status['complete'] > status['expected']:
            raise ValueError('Completed shards exceed the declared campaign scope')
        status['status'] = ('complete' if status['complete'] == status['expected']
                            else 'running' if status['complete'] else 'pending')
    return records, progress, fingerprints


def derive_paired_metrics(records):
    references = {}
    modes = {}
    for row in records:
        root_key = key(row, ('source_cell', 'H', 'J', 'seed', 'decision',
                            'branch_kind', 'action_mode', 'replicate'))
        if row['actor_family'] == 'prior' or (row['actor_family'] == 'warm' and row['round'] == 0):
            label = 'prior' if row['actor_family'] == 'prior' else 'inherited'
            ref_key = root_key + (label,)
            if ref_key in references:
                raise ValueError(f'Duplicate paired {label} reference')
            references[ref_key] = row
        modes[key(row, tuple(name for name in IDENTITY if name != 'action_mode')) +
              (row['action_mode'],)] = row
    result = []
    for original in records:
        row = dict(original)
        root_key = key(row, ('source_cell', 'H', 'J', 'seed', 'decision',
                            'branch_kind', 'action_mode', 'replicate'))
        for reference in ('inherited', 'prior'):
            baseline = references.get(root_key + (reference,))
            if baseline is None:
                continue
            for metric in RETURN_KEYS:
                if metric in row and metric in baseline:
                    row[f'{metric}_gain_vs_{reference}'] = row[metric] - baseline[metric]
        sample = modes.get(key(row, tuple(name for name in IDENTITY if name != 'action_mode')) + ('sample',))
        if row['action_mode'] == 'mean' and sample is not None:
            for metric in RETURN_KEYS:
                if metric in row and metric in sample:
                    row[f'{metric}_mean_minus_sample'] = row[metric] - sample[metric]
        result.append(row)
    return result


def seed_balanced(values, *, bootstrap_samples=2000):
    """Cluster at seed: replicate -> root -> seed -> equal-weight seed mean."""
    roots = defaultdict(list)
    for seed, decision, value in values:
        roots[(seed, decision)].append(float(value))
    seeds = defaultdict(list)
    for (seed, _decision), points in roots.items():
        seeds[seed].append(statistics.fmean(points))
    seed_means = {seed: statistics.fmean(points) for seed, points in sorted(seeds.items())}
    observed = list(seed_means.values())
    result = dict(mean=statistics.fmean(observed), ci95_low=None, ci95_high=None,
                  n_seeds=len(seeds), n_roots=len(roots), n_branches=len(values),
                  seed_means=seed_means)
    if len(observed) > 1:
        rng = random.Random(260926)
        draws = sorted(statistics.fmean(rng.choices(observed, k=len(observed)))
                       for _ in range(bootstrap_samples))
        result.update(ci95_low=draws[int(.025 * (bootstrap_samples-1))],
                      ci95_high=draws[int(.975 * (bootstrap_samples-1))])
    return result


def aggregate(records):
    grouped = defaultdict(list)
    metrics = set(RETURN_KEYS + OPTIONAL_DIAGNOSTICS +
                  ('modeled_prefix_error', 'terminal_value_error', 'total_value_error'))
    metrics.update(f'{metric}_gain_vs_{ref}' for metric in RETURN_KEYS for ref in ('inherited', 'prior'))
    metrics.update(f'{metric}_mean_minus_sample' for metric in RETURN_KEYS)
    metrics.update(f'{metric}_{reduction}' for reduction in TERMINAL_REDUCTIONS
                   for metric in ('predicted_model_return', 'real_bootstrapped_return',
                                  'modeled_prefix_error', 'terminal_value_error',
                                  'total_value_error', 'terminal_value_abs_error'))
    for row in derive_paired_metrics(records):
        for metric in metrics:
            if metric in row and row[metric] is not None:
                grouped[key(row, GROUP) + (metric,)].append((row['seed'], row['decision'], row[metric]))
    result = []
    for identity, points in sorted(grouped.items()):
        row = dict(zip(GROUP + ('metric',), identity))
        row.update(seed_balanced(points))
        result.append(row)
    return result


def aggregate_periods(records):
    """Temporal summaries retain the same per-root pairing and seed clustering."""
    windows = {'early': lambda decision: decision < 100,
               'middle': lambda decision: 100 <= decision < 300,
               'late': lambda decision: decision >= 300}
    return {name: aggregate([row for row in records if includes(row['decision'])])
            for name, includes in windows.items()}


def _plot_series(axis, measurements, *, metric, family='warm', mode='sample', label=None,
                 color=None, linestyle='-', kind='prefix'):
    points = sorted((r for r in measurements if r['metric'] == metric and
                     r['actor_family'] == family and r['action_mode'] == mode and
                     r['branch_kind'] == kind), key=lambda r: r['round'])
    if not points:
        return False
    line, = axis.plot([r['round'] for r in points], [r['mean'] for r in points],
                      marker='o', markersize=3, label=label, color=color, linestyle=linestyle)
    bounded = [r for r in points if r['ci95_low'] is not None]
    if bounded:
        axis.fill_between([r['round'] for r in bounded], [r['ci95_low'] for r in bounded],
                          [r['ci95_high'] for r in bounded], alpha=.12, color=line.get_color())
    return True


def render_figures(report, output):
    try:
        import matplotlib
    except ModuleNotFoundError:
        # The locked compute runtime deliberately excludes plotting packages.
        # Native W&B plots below remain available without matplotlib or Pillow.
        return {}
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    cells = report['cells']
    titles = ('Predicted and measured improvement over the inherited actor',
              'Model-prefix and terminal-value error',
              'Mean-action minus sampled-action return',
              'Stopping this solve: improvement in remaining episode return')
    files = {}
    for panel, title in zip(IMAGE_KEYS, titles):
        figure, axes = plt.subplots(1, max(1, len(cells)), figsize=(max(5, 4*len(cells)), 3.8), squeeze=False)
        for axis, cell in zip(axes.flat, cells):
            rows = [r for r in report['measurements'] if r['source_cell'] == cell['source_cell']]
            plotted = False
            if panel.endswith('/gains'):
                for metric, family, label, color, style in (
                    ('real_mc_return_gain_vs_inherited', 'warm', 'Warm: real', '#176f9c', '-'),
                    ('predicted_model_return_gain_vs_inherited', 'warm', 'Warm: predicted', '#cd6d15', '-'),
                    ('real_mc_return_gain_vs_inherited', 'cold', 'Cold: real', '#64748b', '--')):
                    plotted |= _plot_series(axis, rows, metric=metric, family=family, label=label, color=color, linestyle=style)
                axis.set_ylabel('Discounted gain; sampled prefix')
            elif panel.endswith('/errors'):
                for metric, label, color in (
                    ('modeled_prefix_error', 'Model prefix: A − B', '#b75a1a'),
                    ('terminal_value_error', 'Terminal: B − C', '#6f51ab')):
                    plotted |= _plot_series(axis, rows, metric=metric, label=label, color=color)
                axis.set_ylabel('Predicted − measured value')
            elif panel.endswith('/action_selection'):
                for family, label, color in (('warm', 'Warm', '#176f9c'), ('cold', 'Cold', '#64748b')):
                    plotted |= _plot_series(axis, rows, metric='real_mc_return_mean_minus_sample',
                        family=family, mode='mean', label=label, color=color)
                axis.set_ylabel('Discounted real return difference')
            else:
                plotted |= _plot_series(axis, rows, metric='real_mc_return_gain_vs_inherited',
                    mode='mean', label='Warm; original future J', color='#176f9c', kind='replan')
                prior = [r for r in rows if r['branch_kind'] == 'replan' and r['actor_family'] == 'prior'
                         and r['metric'] == 'real_mc_return_gain_vs_inherited']
                if prior:
                    axis.axhline(prior[0]['mean'], color='#64748b', linestyle=':', label='Prior first action')
                    plotted = True
                axis.set_ylabel('Remaining episode raw reward gain')
            axis.set_title(f"H={cell['H']}, source J={cell['J']}")
            axis.set_xlabel('Completed rounds at captured solve')
            axis.axhline(0, linewidth=.6, color='#a1a1aa')
            axis.grid(alpha=.15)
            if plotted:
                axis.legend(fontsize=7, frameon=False)
            else:
                axis.text(.5, .5, 'Pending results', transform=axis.transAxes,
                          ha='center', color='#64748b')
            axis.spines[['top', 'right']].set_visible(False)
        figure.suptitle(title, fontsize=12)
        figure.tight_layout()
        stem = panel.split('/')[-1]
        figure.savefig(output / f'{stem}.png', dpi=150)
        figure.savefig(output / f'{stem}.pdf')
        plt.close(figure)
        files[panel] = output / f'{stem}.png'
    return files


def build_report(root, *, render=True):
    root = Path(root)
    campaign = read(root / 'campaign.json')
    records, progress, fingerprints = load_shards(root, campaign)
    cells = campaign.get('cells', [])
    cells = [dict(source_cell=c.get('source_cell', c.get('name')), H=c['H'], J=c['J']) for c in cells]
    if not cells:
        cells = [dict(source_cell=f'warm_h{h}_j{j}', H=h, J=j)
                 for h, j in ((3, 6), (3, 8), (3, 10), (1, 8))]
    status = 'complete' if all(row['status'] == 'complete' for row in progress) else 'pending' if not records else 'running'
    result = dict(schema_version=1, protocol=campaign.get('protocol', LAYOUT_VERSION),
                  status=status, progress=progress, cells=cells,
                  record_count=len(records), source_shards=fingerprints,
                  explanation=EXPLANATION, measurements=aggregate(records),
                  period_measurements=aggregate_periods(records),
                  period_definitions={'early': 'decision < 100',
                                      'middle': '100 <= decision < 300',
                                      'late': 'decision >= 300'})
    result['data_sha256'] = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()
    output = root / 'report'
    output.mkdir(parents=True, exist_ok=True)
    _write(output / 'summary.json', result)
    files = render_figures(result, output) if render else {}
    progress_html = ''.join(f"<li>{r['branch_kind']}: {r['complete']}/{r['expected']} roots ({r['status']})</li>" for r in progress)
    figure_html = ''.join(f'<figure><img src="{p.name}" alt="{html.escape(k)}"></figure>' for k, p in files.items())
    if not files:
        figure_html = '<p>Native charts are published to W&amp;B. Optional local PNG/PDF rendering requires matplotlib.</p>'
    page = ('<!doctype html><html><head><meta charset="utf-8"><title>Warm actor calibration</title>'
            '<style>body{font:16px system-ui;margin:32px auto;max-width:1500px;padding:0 20px;color:#17233b}'
            'p{max-width:1000px;line-height:1.5}figure{margin:25px 0}img{width:100%;height:auto}</style></head><body>'
            f'<h1>Warm actor calibration — {status}</h1><p>{html.escape(EXPLANATION)}</p>'
            f'<ul>{progress_html}</ul>{figure_html}<p><a href="summary.json">Exact measurements and provenance</a></p></body></html>')
    (output / 'index.html').write_text(page)
    return result, files


def calibration_sections():
    prefix = 'ambi-' + LAYOUT_VERSION
    def panel(name, kind, config, width=12, height=7):
        return dict(__id__=prefix+'-'+name, layout=dict(x=0, y=0, w=width, h=height),
                    viewType=kind, config=config, isAuto=False)
    overview = [panel('explanation', 'Markdown Panel', {'value': '### Matched-state actor calibration\n\n'+EXPLANATION}, 24, 5),
                panel('progress', 'Media Browser', {'chartTitle': 'Pending and completed simulator roots',
                      'mediaKeys': ['calibration/progress']}, 24)]
    charts = [panel(name.split('/')[-1], 'Vega2', {
        'transform': {'name': 'tableWithLeafColNames'},
        'userQuery': {'queryFields': [{
            'name': 'runSets', 'args': [{'name': 'runSets', 'value': '${runSets}'}, {'name': 'limit', 'value': 500.0}],
            'fields': [{'name': 'summaryTable', 'args': [{'name': 'tableKey', 'value': name+'_table'}], 'fields': []},
                       {'name': 'id', 'value': []}, {'name': 'name', 'value': []}]}]},
        'panelDefId': 'wandb/lineseries/v0',
        'fieldSettings': {'step': 'step', 'lineKey': 'lineKey', 'lineVal': 'lineVal'},
        'stringSettings': {'title': title, 'xname': 'Completed rounds at captured solve'}}, 24, 8)
        for name, title in zip(IMAGE_KEYS, ('Predicted and real improvement', 'Error decomposition',
                                           'Mean versus sampled action', 'Replanning continuation'))]
    tables = [panel('measurements', 'Media Browser', {'chartTitle': 'Paired seed-balanced measurements and exploratory intervals',
               'mediaKeys': ['calibration/measurements']}, 24)]
    return [dict(__id__=prefix+'-'+name, name=title, isOpen=True, type='flow',
                 flowConfig=dict(snapToColumns=True, columnsPerPage=1, rowsPerPage=len(panels),
                                 gutterWidth=16, boxWidth=1200, boxHeight=360),
                 sorted=0, pinned=True, isPanelsAuto=False, panels=panels)
            for name, title, panels in (('overview', 'Calibration | progress', overview),
                                       ('results', 'Calibration | four diagnostic summaries', charts),
                                       ('tables', 'Calibration | exact measurements', tables))]


def install_layout(wandb, run, campaign, root):
    try:
        receipt = ensure_results_layout(wandb.Api(timeout=30),
            entity=campaign['wandb_entity'], project=campaign['wandb_project'],
            receipt_dir=Path(root)/'results-layout', layout_version=LAYOUT_VERSION,
            sections=calibration_sections(), chart_keys=IMAGE_KEYS, table_keys=TABLE_KEYS,
            view_name=campaign.get('wandb_view_name', DEFAULT_VIEW_NAME), run_id=campaign['overview_run_id'])
        run.summary.update({'results_layout/status': receipt['status'],
                            'results_layout/schema_verified': True,
                            'results_layout/url': receipt['url']})
        return receipt
    except Exception as exc:
        failure = dict(status='failed', error_type=type(exc).__name__,
                       message='Result layout verification failed; inspect layout receipts. Simulation is unaffected.')
        _write(Path(root)/'results-layout-failure.json', failure)
        run.summary.update({'results_layout/status': 'failed', 'results_layout/schema_verified': False,
                            'results_layout/error_type': type(exc).__name__})
        print(failure['message'], flush=True)
        return failure


def payload(wandb, report, files):
    progress_cols = ['branch_kind', 'complete', 'expected', 'status']
    columns = list(GROUP) + ['metric', 'mean', 'ci95_low', 'ci95_high', 'n_seeds', 'n_roots', 'n_branches']
    result = {
        'calibration/progress': wandb.Table(columns=progress_cols,
            data=[[r[name] for name in progress_cols] for r in report['progress']]),
        'calibration/measurements': wandb.Table(columns=columns,
            data=[[r[name] for name in columns] for r in report['measurements']]),
        'calibration/complete_roots': sum(r['complete'] for r in report['progress']),
        'calibration/expected_roots': sum(r['expected'] for r in report['progress']),
        'calibration/branch_records': report['record_count'],
    }
    panels = {
        'calibration/gains': ('Discounted gain over inherited actor; sampled prefixes', (
            ('real_mc_return_gain_vs_inherited', 'warm', 'sample', 'prefix', 'warm real'),
            ('predicted_model_return_gain_vs_inherited', 'warm', 'sample', 'prefix', 'warm predicted'),
            ('real_mc_return_gain_vs_inherited', 'cold', 'sample', 'prefix', 'cold real'))),
        'calibration/errors': ('Warm sampled prefixes: predicted minus measured value', (
            ('modeled_prefix_error', 'warm', 'sample', 'prefix', 'model prefix'),
            ('terminal_value_error', 'warm', 'sample', 'prefix', 'terminal Q'))),
        'calibration/action_selection': ('Mean minus sampled action: discounted real return', (
            ('real_mc_return_mean_minus_sample', 'warm', 'mean', 'prefix', 'warm'),
            ('real_mc_return_mean_minus_sample', 'cold', 'mean', 'prefix', 'cold'))),
        'calibration/replanning': ('Remaining episode raw reward gain over inherited actor', (
            ('real_mc_return_gain_vs_inherited', 'warm', 'mean', 'replan', 'warm; original future J'),)),
    }
    for chart_key, (title, series) in panels.items():
        xs, ys, labels = [], [], []
        for cell in report['cells']:
            for metric, family, mode, kind, label in series:
                points = sorted((r for r in report['measurements'] if r['source_cell'] == cell['source_cell']
                    and r['metric'] == metric and r['actor_family'] == family and r['action_mode'] == mode
                    and r['branch_kind'] == kind), key=lambda r: r['round'])
                if points:
                    xs.append([r['round'] for r in points])
                    ys.append([r['mean'] for r in points])
                    labels.append(f"H{cell['H']}/J{cell['J']} {label}")
        result[chart_key] = wandb.plot.line_series(xs=xs or [[]], ys=ys or [[]],
            keys=labels or ['Pending measurements'], title=title,
            xname='Completed rounds; source J retained in history')
    return result


def publish(args):
    import wandb
    root = Path(args.root)
    campaign = read(root/'campaign.json')
    for name in ('overview_run_id', 'wandb_entity', 'wandb_project'):
        if not campaign.get(name):
            raise ValueError(f'Campaign must pin {name} before publication')
    run = wandb.init(entity=campaign['wandb_entity'], project=campaign['wandb_project'],
        id=campaign['overview_run_id'], name=campaign.get('name', 'Warm actor matched-state calibration'),
        group=campaign.get('group'), resume='allow', job_type='calibration-overview',
        config={k: v for k, v in campaign.items() if k not in {'jobs', 'job_ids'}})
    layout = install_layout(wandb, run, campaign, root)
    previous = None
    try:
        while True:
            report, files = build_report(root)
            if report['data_sha256'] != previous:
                run.log(payload(wandb, report, files))
                run.summary.update({'status': report['status'], 'data_sha256': report['data_sha256'],
                                    'branch_records': report['record_count']})
                receipt = dict(status=report['status'], run_id=run.id, data_sha256=report['data_sha256'],
                               progress=report['progress'], results_layout=layout)
                _write(root/'publication-progress.json', receipt)
                print(json.dumps({k: receipt[k] for k in ('status', 'run_id', 'progress')}), flush=True)
                previous = report['data_sha256']
            if args.command != 'watch' or report['status'] == 'complete':
                break
            submission = root/'submission.json'
            if submission.exists():
                from slurm.ambi_closed_loop_publish import gpu_jobs_active
                job_ids = read(submission).get('gpu_job_ids')
                if job_ids and not gpu_jobs_active(job_ids):
                    failure = dict(status='evaluation_incomplete', progress=report['progress'],
                        message='All recorded compute jobs ended before the declared root inventory completed.')
                    run.summary.update(failure)
                    _write(root/'publication-incomplete.json', failure)
                    raise RuntimeError(failure['message'])
            time.sleep(args.poll_seconds)
        if report['status'] == 'complete':
            _write(root/'publication-completion.json', receipt)
    except Exception:
        run.summary.update({'status': 'publication_failed'})
        raise
    finally:
        run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('report', 'publish', 'watch'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=30.)
    parser.add_argument('--no-render', action='store_true', help='Local report only: skip matplotlib figures')
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error('--poll-seconds must be between 1 and 60')
    if args.command == 'report':
        report, _files = build_report(args.root, render=not args.no_render)
        print(json.dumps({'status': report['status'], 'progress': report['progress']}))
    else:
        publish(args)


if __name__ == '__main__':
    main()
