"""CPU-owned publication of actor-transfer curves, diagnostics, and paired comparisons."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_actor_transfer_campaign import (
    CHECKPOINT_SHA, CHECKPOINT_STEP, ENTITY, FIRST_ROUNDS, HORIZONS, MODES, PROJECT,
    PROTOCOL, ROUNDS, SEEDS, cells, digest, read, summarize_trace, validate_completed, write,
)
from slurm.ambi_aux_hj_sweep import publish_performance
from slurm.ambi_closed_loop_checkpoint_sweep import load_prior
from slurm.ambi_closed_loop_publish import gpu_jobs_active, indexed_episodes, moments
from slurm.ambi_closed_loop_reward_retrace_publish import paired_comparison


def url(run_id):
    return f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}'


def validate_scope(campaign):
    expected = cells(campaign['matrix'])
    assert campaign['study_protocol'] == PROTOCOL
    assert campaign['checkpoint_step'] == CHECKPOINT_STEP and campaign['checkpoint_sha256'] == CHECKPOINT_SHA
    assert campaign['first_action_rounds'] == FIRST_ROUNDS
    assert len(campaign['cells']) == len(expected)
    for actual, wanted in zip(campaign['cells'], expected):
        for key, value in wanted.items(): assert actual[key] == value, key
    ids = [c['performance_run_id'] for c in campaign['cells']] + [campaign['overview_run_id']]
    assert len(set(ids)) == len(ids)
    return campaign['cells']


def load_completed(campaign, cell):
    from utils.eval_series_data import load_records
    directory, bundle = Path(cell['directory']), Path(cell['bundle'])
    receipt = read(directory / 'worker-completion.json')
    assert receipt['cell'] == cell['name'] and receipt['status'] == 'complete' and not receipt['smoke']
    assert digest(bundle / 'manifest.json') == receipt['manifest_sha256']
    assert all(digest(bundle / n) == sha for n, sha in receipt['trace_sha256'].items())
    assert digest(directory / 'transfer-diagnostics.json') == receipt['diagnostics_sha256']
    manifest = validate_completed(bundle, cell, campaign)
    record, = load_records(bundle, inventory_path=campaign['inventory'])
    assert record['identity'] == cell['identity']
    assert not record['provenance']['missing_artifact_files']
    return dict(episodes=manifest['runs'][0]['episodes'], metrics=record['metrics'],
                diagnostics=read(directory / 'transfer-diagnostics.json'))


def publish_cell(args):
    from utils.ambi_benchmark import stage_completed_bundle
    from utils.eval_series import load_run
    campaign = read(args.root / 'campaign.json'); validate_scope(campaign)
    cell = campaign['cells'][args.index]; directory = Path(cell['directory'])
    completed = load_completed(campaign, cell)
    assert load_run(cell['run_dir'])['identity'] == cell['identity']
    receipt_path = directory / 'publication-completion.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        assert receipt['status'] == 'complete' and receipt['performance']['run_id'] == cell['performance_run_id']
        return receipt
    staged = stage_completed_bundle(cell['bundle'], {cell['selector']: cell['run_dir']},
                                     inventory_path=campaign['inventory'])
    assert staged[cell['selector']]['status'] == 'queued'
    performance = publish_performance(cell['run_dir'])
    receipt = dict(status='complete', cell=cell['name'], performance=performance,
                   metrics=completed['metrics'])
    write(receipt_path, receipt)
    return receipt


def aggregate_results(campaign, completed):
    """Pair only complete seed panels; missing cells are null and remain visible."""
    panel = validate_scope(campaign)
    assert not set(completed) - {cell['name'] for cell in panel}
    points, pairs, diagnostic_rows = [], [], []
    for cell in panel:
        point = dict(setting=cell['name'], H=cell['H'], J=cell['J'], transfer_mode=cell['transfer_mode'],
                     first_action_rounds=FIRST_ROUNDS, performance_url=url(cell['performance_run_id']),
                     return_mean=None, return_std=None, control_seconds=None, control_seconds_per_decision=None,
                     paired_gain_mean=None)
        if cell['name'] in completed:
            item = completed[cell['name']]; metrics = item['metrics']
            returns = moments(ep['return'] for ep in item['episodes'])
            point.update(return_mean=returns['mean'], return_std=returns['std'],
                         control_seconds=metrics.get('runtime/control_seconds'),
                         control_seconds_per_decision=metrics.get('runtime/control_seconds_per_decision'),
                         paired_gain_mean=metrics.get('eval/paired_gain_mean'))
            # Preserve new first/steady latency and diagnostic costs without renaming units.
            point.update({key: value for key, value in metrics.items() if key.startswith('runtime/')})
            for row in item['diagnostics']['stage_rows']:
                diagnostic_rows.append(dict(setting=cell['name'], H=cell['H'], J=cell['J'],
                                            transfer_mode=cell['transfer_mode'], **row))
        points.append(point)
    for h in HORIZONS:
        for j in ROUNDS:
            cold, warm = (f'{mode}_h{h}_j{j}_c16' for mode in MODES)
            if cold in completed and warm in completed:
                comparison = paired_comparison(completed[warm]['episodes'], completed[cold]['episodes'])
                metrics = comparison['metrics']
                pairs.append(dict(H=h, J=j,
                    warm_minus_cold_mean=metrics['comparison/sample_minus_mean_mean'],
                    warm_minus_cold_std=metrics['comparison/sample_minus_mean_std'],
                    ci95_low=metrics['comparison/sample_minus_mean_ci95_low'],
                    ci95_high=metrics['comparison/sample_minus_mean_ci95_high'],
                    paired_episodes=metrics['comparison/sample_minus_mean_paired_episodes'],
                    episodes=[dict(seed=r['seed'], solver_seed=r['solver_seed'], cold_return=r['mean_return'],
                        warm_return=r['sampled_return'], warm_minus_cold=r['sample_minus_mean']) for r in comparison['rows']]))
    return dict(points=points, paired_comparisons=pairs, diagnostics=diagnostic_rows,
        completed=len(completed), total=len(panel), bootstrap_seed=20260912, bootstrap_resamples=2000,
        uncertainty='Five paired environment seeds on one trained backbone; exploratory unadjusted 95% episode-bootstrap intervals.')


def overview_payload(wandb, aggregate):
    point_columns = ['setting', 'H', 'J', 'transfer_mode', 'first_action_rounds', 'return_mean', 'return_std',
                     'control_seconds', 'control_seconds_per_decision', 'paired_gain_mean', 'performance_url']
    point_columns += sorted({k for p in aggregate['points'] for k in p if k.startswith('runtime/')})
    pair_columns = ['H', 'J', 'warm_minus_cold_mean', 'warm_minus_cold_std', 'ci95_low', 'ci95_high', 'paired_episodes']
    diag_columns = ['setting', 'H', 'J', 'transfer_mode', 'phase', 'stage', 'round_index', 'decision_group',
                    'metric', 'mean', 'std', 'episodes']
    payload = {'campaign/completed': aggregate['completed'], 'campaign/total': aggregate['total'],
        'comparison/returns_and_compute': wandb.Table(columns=point_columns,
            data=[[row.get(k) for k in point_columns] for row in aggregate['points']]),
        'comparison/warm_minus_cold': wandb.Table(columns=pair_columns,
            data=[[row.get(k) for k in pair_columns] for row in aggregate['paired_comparisons']])}
    # Separate tables keep all stages/metrics visible without a single oversized table.
    for phase in ('transfer_probe', 'probe'):
        for h in HORIZONS:
            for mode in MODES:
                for period in ('first', 'steady'):
                    rows = [r for r in aggregate['diagnostics'] if r['phase'] == phase and r['H'] == h
                            and r['transfer_mode'] == mode and r['decision_group'] == period]
                    assert len(rows) <= 10000, 'Split diagnostic tables before exceeding the W&B row limit'
                    payload[f'diagnostics/{phase}_h{h}_{mode}_{period}'] = wandb.Table(columns=diag_columns,
                        data=[[row.get(k) for k in diag_columns] for row in rows])
    for h in HORIZONS:
        series = [[p for p in aggregate['points'] if p['H'] == h and p['transfer_mode'] == mode
                   and p['return_mean'] is not None] for mode in MODES]
        if any(series):
            payload[f'comparison/h{h}_return_vs_rounds'] = wandb.plot.line_series(
                xs=[[p['J'] for p in rows] for rows in series], ys=[[p['return_mean'] for p in rows] for rows in series],
                keys=list(MODES), title=f'H{h}: episode return versus subsequent solve rounds (first J10)', xname='Subsequent J')
            timed = [[p for p in rows if p['control_seconds_per_decision'] is not None] for rows in series]
            payload[f'comparison/h{h}_return_vs_compute'] = wandb.plot.line_series(
                xs=[[p['control_seconds_per_decision'] for p in rows] for rows in timed],
                ys=[[p['return_mean'] for p in rows] for rows in timed], keys=list(MODES),
                title=f'H{h}: episode return versus controller time including first solve', xname='Controller seconds / decision')
    return payload



def install_results_layout(wandb, run, campaign, root):
    """A presentation failure must not interrupt publication of completed science."""
    from utils.wandb_results_layout import DEFAULT_VIEW_NAME, ensure_actor_transfer_results_layout
    try:
        receipt = ensure_actor_transfer_results_layout(wandb.Api(timeout=30), entity=ENTITY,
            project=PROJECT, receipt_dir=Path(root) / 'results-layout',
            view_name=os.environ.get('WANDB_RESULTS_VIEW_NAME', DEFAULT_VIEW_NAME),
            run_id=campaign['overview_run_id'])
        run.summary.update({'results_layout/status': receipt['status'],
            'results_layout/url': receipt['url'], 'results_layout/workspace_url': receipt['workspace_url'],
            'results_layout/schema_verified': True})
        write(Path(root) / 'results-layout' / 'campaign-results-layout.json', receipt)
        print('Results layout schema verified (browser check required): ' + receipt['url'], flush=True)
        return receipt
    except Exception as exc:
        receipt = dict(status='failed', error_type=type(exc).__name__, schema_verified=False,
            message='Results panels could not be verified. Evaluation and metric publication continue; inspect results-layout receipts.')
        write(Path(root) / 'results-layout-failure.json', receipt)
        try:
            run.summary.update({'results_layout/status': 'failed', 'results_layout/schema_verified': False,
                                'results_layout/error_type': type(exc).__name__})
        except Exception:
            pass  # Local receipt and the explicit watcher message remain available.
        print('RESULTS LAYOUT FAILED: ' + receipt['message'] + ' (' + type(exc).__name__ + ')', flush=True)
        return receipt

def watch(args):
    import wandb
    campaign = read(args.root / 'campaign.json'); panel = validate_scope(campaign)
    prior = load_prior(campaign['prior_reference'], campaign['inventory'])
    assert indexed_episodes(prior['episodes']) == indexed_episodes(campaign['prior_reference']['episodes'])
    marker = args.root / 'watcher-started.json'
    if marker.exists(): raise RuntimeError('Watcher already started; inspect remote history and journals before recovery')
    write(marker, dict(pid=os.getpid(), started=time.time()))
    run = wandb.init(entity=ENTITY, project=PROJECT, id=campaign['overview_run_id'], resume='never',
        name=campaign['label'], group=campaign['group'], job_type='actor-transfer-comparison',
        tags=['actor-transfer', '575k', 'mean', 'return-only'], mode='online',
        config=dict(study_protocol=PROTOCOL, source_run=campaign['source_run'], source_commit=campaign['source_commit'],
            checkpoint_step=CHECKPOINT_STEP, checkpoint_sha256=CHECKPOINT_SHA,
            campaign_group=campaign['group'],
            H=list(HORIZONS), J=list(ROUNDS), modes=list(MODES), first_action_rounds=FIRST_ROUNDS,
            C=16, A=4, N=128, B=256, seeds=SEEDS, controller_seed=55, max_steps=500,
            prior_reference=campaign['prior_reference'], timing_note=campaign['timing_note']))
    layout_receipt = install_results_layout(wandb, run, campaign, args.root)
    attempted, completed, futures, failures = set(), {}, {}, {}
    previous, terminal_since, previous_completed = None, None, -1
    def launch(index):
        with (Path(panel[index]['directory']) / 'publisher.log').open('w') as log:
            return subprocess.run([sys.executable, __file__, 'publish', '--root', str(args.root), '--index', str(index)],
                                  stdout=log, stderr=subprocess.STDOUT).returncode
    try:
        with ThreadPoolExecutor(max_workers=campaign['publisher_workers']) as pool:
            while True:
                for index, future in list(futures.items()):
                    if future.done():
                        code = future.result()
                        if code: failures[index] = code
                        del futures[index]
                for index, cell in enumerate(panel):
                    directory = Path(cell['directory'])
                    if cell['name'] not in completed and (directory / 'worker-completion.json').exists():
                        completed[cell['name']] = load_completed(campaign, cell)
                    if (cell['name'] in completed and index not in attempted and len(futures) < campaign['publisher_workers']
                            and not (directory / 'publication-completion.json').exists()):
                        attempted.add(index); futures[index] = pool.submit(launch, index)
                states = [dict(setting=c['name'], state=('published' if (Path(c['directory']) / 'publication-completion.json').exists()
                    else 'publication_failed' if i in failures else 'publishing' if i in futures
                    else 'evaluated' if c['name'] in completed else 'queued_or_running')) for i, c in enumerate(panel)]
                stamp = [r['state'] for r in states]
                if stamp != previous:
                    aggregate = aggregate_results(campaign, completed)
                    write(args.root / 'comparison.json', aggregate)
                    write(args.root / 'progress.json', dict(settings=states, failures=failures))
                    if len(completed) != previous_completed:
                        run.log(overview_payload(wandb, aggregate))
                        previous_completed = len(completed)
                    run.log({'campaign/settings': wandb.Table(columns=['setting', 'state'],
                        data=[[r['setting'], r['state']] for r in states])})
                    previous = stamp
                    print(f'Evaluated {len(completed)}/{len(panel)}; published {stamp.count("published")}; failures {len(failures)}', flush=True)
                if all(s == 'published' for s in stamp): break
                submission = args.root / 'submission.json'
                if submission.exists() and not gpu_jobs_active(read(submission)['gpu_job_ids']) and not futures:
                    terminal_since = terminal_since or time.time()
                    if time.time() - terminal_since > 60: break
                else: terminal_since = None
                time.sleep(15)
        complete = len(completed) == len(panel) and not failures and all(
            (Path(cell['directory']) / 'publication-completion.json').exists() for cell in panel)
        artifact = wandb.Artifact('actor-transfer-comparison-' + campaign['overview_run_id'], type='actor-transfer-diagnostics')
        for filename in ('campaign.json', 'comparison.json', 'progress.json'):
            artifact.add_file(str(args.root / filename), name=filename)
        for cell in panel:
            path = Path(cell['directory']) / 'transfer-diagnostics.json'
            if path.exists(): artifact.add_file(str(path), name=cell['name'] + '/transfer-diagnostics.json')
        run.log_artifact(artifact)
        run.summary.update(dict(status='complete' if complete else 'incomplete', completed_settings=len(completed), total_settings=len(panel)))
        run.finish(exit_code=0 if complete else 1)
        write(args.root / 'publication-summary.json', dict(status='complete' if complete else 'incomplete',
            overview_url=url(campaign['overview_run_id']), results_layout=layout_receipt,
            completed=len(completed), total=len(panel), failures=failures))
        if not complete: raise RuntimeError('Campaign incomplete; inspect GPU and publication receipts')
    except BaseException:
        run.finish(exit_code=1); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('watch', 'publish'):
        p = sub.add_parser(command); p.add_argument('--root', type=Path, required=True)
        if command == 'publish': p.add_argument('--index', type=int, required=True)
    args = parser.parse_args(); {'watch': watch, 'publish': publish_cell}[args.command](args)


if __name__ == '__main__': main()
