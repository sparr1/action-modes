"""Publish full-episode fixed-horizon J comparisons at the pinned 650k checkpoint.

Each new J owns a performance and training run. The historical H3/J10 and
frozen prior/MPPI references retain their original identities and are never
republished; H1 and H2 evaluate all eight round budgets.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_aux_hj_sweep import ENTITY, PROJECT, SEEDS, read, write
from slurm.ambi_closed_loop_publish import campaign_horizon, gpu_jobs_active, indexed_episodes, moments
from slurm.ambi_closed_loop_reward_retrace_publish import paired_comparison
from slurm.ambi_closed_loop_sampled import verify_receipt
from slurm.ambi_closed_loop_checkpoint_sweep_publish import publication_complete, protocol_proof

ROUNDS = [1, 2, 4, 6, 8, 10, 12, 14]
MPPI_FIELDS = [f'mppi_{kind}_{metric}' for kind in ('soft', 'return') for metric in
               ('return_mean', 'return_std', 'paired_gain_mean', 'paired_gain_ci95_low', 'paired_gain_ci95_high')]
POINT_COLUMNS = ['setting', 'H', 'J', 'critic_kind', 'critic_scheme', 'training_decisions', 'checkpoint_sha256', 'reused',
    'performance_run_id', 'training_run_id', 'prior_performance_run_id', 'initial_alpha',
    'return_mean', 'return_std', 'return_episodes', 'prior_return_mean', 'prior_return_std',
    'paired_gain_mean', 'paired_gain_std', 'paired_gain_ci95_low', 'paired_gain_ci95_high',
    'paired_episodes', *MPPI_FIELDS]
EPISODE_COLUMNS = ['setting', 'H', 'J', 'critic_kind', 'critic_scheme', 'reused', 'seed', 'solver_seed', 'return', 'prior_return', 'paired_gain']
STATUS_COLUMNS = [*POINT_COLUMNS, 'status', 'performance_url', 'training_url', 'failure']
REFERENCE_COLUMNS = ['kind', 'label', 'training_decisions', 'checkpoint_sha256', 'return_mean', 'return_std',
    'paired_gain_mean', 'paired_gain_ci95_low', 'paired_gain_ci95_high', 'performance_url', 'compute_budget_note']


def _url(run_id):
    return f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}' if run_id else None


def campaign_critic(campaign):
    kinds = {cell['critic_kind'] for cell in campaign['cells']}
    if len(kinds) != 1 or not kinds <= {'return_only', 'soft'}:
        raise ValueError('Expected one shared return-only or soft critic scheme')
    kind, = kinds
    if campaign.get('critic_kind', kind) != kind:
        raise ValueError('Declared critic kind differs from the cells')
    return kind


def critic_scheme(kind):
    return 'soft_soft' if kind == 'soft' else 'return_return'


def validate_scope(campaign):
    cells = campaign['cells']
    horizon = campaign_horizon(campaign)
    kind = campaign_critic(campaign)
    if kind == 'soft' and horizon != 1:
        raise ValueError('The new soft/soft campaign selects H1')
    if horizon not in {1, 2, 3} or campaign.get('H', horizon) != horizon:
        raise ValueError('Expected a single declared horizon H1, H2, or H3')
    if sorted(c['J'] for c in cells) != ROUNDS or len({c['name'] for c in cells}) != 8:
        raise ValueError('Expected exactly J1/2/4/6/8/10/12/14')
    if [c['J'] for c in cells if c['reused']] != ([10] if horizon == 3 and kind == 'return_only' else []):
        raise ValueError('Only the pinned return-only H3/J10 evaluation may be reused')
    ids = [campaign['overview_run_id']] + [c[k] for c in cells for k in ('performance_run_id', 'training_run_id')]
    if len(set(ids)) != len(ids) or len({c['run_dir'] for c in cells}) != 8:
        raise ValueError('Each planner, training, and overview identity must be distinct')
    if len({c['checkpoint_sha256'] for c in cells}) != 1 or len({c['initial_alpha'] for c in cells}) != 1:
        raise ValueError('All J settings must inherit the same checkpoint')
    prior = None
    for cell in cells:
        if ((cell['H'], cell['checkpoint_step'], cell['training_decisions'], cell['estimator'],
                cell['execution_mode'], cell['alpha_mode'], cell['critic_kind'])
                != (horizon, 650000, 650000, 'one_step', 'mean', 'adaptive', kind)):
            raise ValueError('Unexpected checkpoint or J-sweep method')
        params = cell['params']
        expected = ('sac', 'sac', 'entropy_augmented', 'outer') if kind == 'soft' else ('aux_return', 'aux_return', 'reward_only', 'none')
        if tuple(params[key] for key in ('inner_critic_source', 'inner_horizon_critic_source',
                'inner_sac_critic_target', 'inner_terminal_entropy')) != expected:
            raise ValueError('Critic sources, target, and terminal entropy must match the declared scheme')
        if params['inner_actor_source'] != 'sac' or params['inner_horizon_actor_source'] != 'sac':
            raise ValueError('Both actor sources must retain the SAC prior')
        if (params['inner_critic_updates_per_round'], params['inner_actor_updates_per_round'], params['inner_replay_capacity']) != (16, 4, max(3072, 384*cell['J'])):
            raise ValueError('J sweep requires C16/A4 and replay retaining all imagined transitions')
        if not math.isfinite(cell['initial_alpha']) or cell['initial_alpha'] <= 0:
            raise ValueError('Expected the checkpoint saved positive temperature')
        for reference in [cell['prior_reference'], *([cell['reuse_reference']] if cell['reused'] else [])]:
            if (reference['checkpoint_step'], reference['checkpoint_sha256']) != (650000, cell['checkpoint_sha256']):
                raise ValueError('Reference belongs to a different checkpoint')
        current = indexed_episodes(cell['prior_reference']['episodes'])
        if prior is not None and current != prior:
            raise ValueError('All J settings require the same frozen-prior reference')
        prior = current
    for reference in campaign.get('mppi_references', {}).values():
        if (reference['checkpoint_step'], reference['checkpoint_sha256']) != (650000, cells[0]['checkpoint_sha256']):
            raise ValueError('MPPI reference belongs to a different checkpoint')
        if indexed_episodes(reference['episodes']).keys() != prior.keys():
            raise ValueError('MPPI reference seeds differ from the frozen prior')
    return cells


def verify_references(campaign):
    from slurm.ambi_closed_loop_j_sweep import load_prior, load_reused
    prior = campaign['cells'][0]['prior_reference']
    if indexed_episodes(load_prior(prior, campaign['inventory'])['episodes']) != indexed_episodes(prior['episodes']):
        raise ValueError('Prior changed after preparation')
    for cell in campaign['cells']:
        if cell['reused'] and indexed_episodes(load_reused(cell['reuse_reference'], campaign['inventory'])['episodes']) != indexed_episodes(cell['reuse_reference']['episodes']):
            raise ValueError('Historical H3/J10 result changed after preparation')
    # Enrichment reopens hash-pinned manifests and validates their source identity.
    from slurm.ambi_closed_loop_j_sweep import load_mppi_references
    references = campaign.get('mppi_references', {})
    if references:
        actual = load_mppi_references(next(iter(references.values())), campaign['inventory'], prior)
        for kind, reference in references.items():
            if indexed_episodes(actual[kind]['episodes']) != indexed_episodes(reference['episodes']):
                raise ValueError('MPPI changed after preparation')


def comparison_stats(episodes, prior):
    return {key.removeprefix('comparison/sample_minus_mean_'): value
            for key, value in paired_comparison(episodes, prior)['metrics'].items()}


def aggregate_results(campaign, completed):
    """Missing measurements remain null; references describe fixed historical runs."""
    cells = validate_scope(campaign)
    if set(completed) - {c['name'] for c in cells if not c['reused']}:
        raise ValueError('Unexpected or republished historical result')
    kind = campaign_critic(campaign)
    scheme = critic_scheme(kind)
    prior = cells[0]['prior_reference']
    prior_stats = moments(e['return'] for e in prior['episodes'])
    references = [dict(kind='prior', label='Frozen prior (reused)', training_decisions=650000,
        checkpoint_sha256=cells[0]['checkpoint_sha256'], return_mean=prior_stats['mean'], return_std=prior_stats['std'],
        paired_gain_mean=0., paired_gain_ci95_low=0., paired_gain_ci95_high=0.,
        performance_url=_url(prior.get('performance_run_id')), compute_budget_note='No planning.')]
    reference_fields = {}
    for mppi_kind, reference in campaign.get('mppi_references', {}).items():
        if mppi_kind not in {'soft', 'return_only'}:
            raise ValueError('Unexpected MPPI reference kind')
        stats = moments(e['return'] for e in reference['episodes'])
        paired = comparison_stats(reference['episodes'], prior['episodes'])
        row = dict(kind=mppi_kind, label=f'MPPI {"soft" if mppi_kind == "soft" else "return-only"} critic (fixed historical budget)',
            training_decisions=650000, checkpoint_sha256=reference['checkpoint_sha256'],
            return_mean=stats['mean'], return_std=stats['std'],
            **{'paired_gain_' + key: paired[key] for key in ('mean', 'ci95_low', 'ci95_high')},
            performance_url=_url(reference.get('performance_run_id')),
            compute_budget_note='Fixed historical MPPI compute; J does not apply. Budgets differ from refinement.')
        references.append(row)
        prefix = 'mppi_' + ('return' if mppi_kind == 'return_only' else mppi_kind) + '_'
        reference_fields.update({prefix + key: row[key] for key in ('return_mean', 'return_std',
            'paired_gain_mean', 'paired_gain_ci95_low', 'paired_gain_ci95_high')})
    points, rows = [], []
    for cell in sorted(cells, key=lambda c: c['J']):
        point = dict.fromkeys(POINT_COLUMNS)
        point.update(setting=cell['name'], H=cell['H'], J=cell['J'], critic_kind=kind, critic_scheme=scheme, training_decisions=650000,
            checkpoint_sha256=cell['checkpoint_sha256'], reused=cell['reused'],
            performance_run_id=cell['performance_run_id'], training_run_id=cell['training_run_id'],
            prior_performance_run_id=prior.get('performance_run_id'), initial_alpha=cell['initial_alpha'],
            prior_return_mean=prior_stats['mean'], prior_return_std=prior_stats['std'], **reference_fields)
        episodes = cell['reuse_reference']['episodes'] if cell['reused'] else completed.get(cell['name'])
        if episodes is not None:
            stats, paired = moments(e['return'] for e in episodes), comparison_stats(episodes, prior['episodes'])
            point.update(return_mean=stats['mean'], return_std=stats['std'], return_episodes=stats['episodes'],
                **{'paired_gain_' + key: paired[key] for key in ('mean', 'std', 'ci95_low', 'ci95_high')},
                paired_episodes=paired['paired_episodes'])
            indexed = indexed_episodes(episodes)
            for key, baseline in sorted(indexed_episodes(prior['episodes']).items()):
                episode = indexed[key]
                gain = episode['return'] - baseline['return']
                if 'paired_return_delta' in episode and not math.isclose(episode['paired_return_delta'], gain, rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError('Stored gain differs from the matched prior')
                rows.append(dict(setting=cell['name'], H=cell['H'], J=cell['J'], critic_kind=kind, critic_scheme=scheme, reused=cell['reused'], seed=key[0], solver_seed=key[1],
                    **{'return': episode['return']}, prior_return=baseline['return'], paired_gain=gain))
        points.append(point)
    reused = sum(c['reused'] for c in cells)
    return dict(H=cells[0]['H'], critic_kind=kind, critic_scheme=scheme, points=points, episodes=rows, references=references, evaluated=len(completed) + reused,
        new_evaluated=len(completed), reused=reused, bootstrap_seed=20260912, bootstrap_resamples=2000,
        comparison='Refined full-episode return minus matched frozen-prior return at checkpoint 650k.',
        uncertainty='Five paired environment seeds; 2,000 paired bootstrap resamples; exploratory 95% intervals.')


def numeric_rows(aggregate):
    rows = []
    for point in aggregate['points']:
        axis = {'axis/inner_rounds': point['J']}
        rows.append((point['setting'] + '/references', {**axis, 'j_sweep/prior_return_mean': point['prior_return_mean'],
            **{'j_sweep/' + key: point[key] for key in MPPI_FIELDS if point[key] is not None}}))
        if point['return_mean'] is not None:
            rows.append((point['setting'], {**axis, **{'j_sweep/' + key: point[key] for key in
                ('return_mean', 'return_std', 'return_episodes', 'paired_gain_mean', 'paired_gain_std',
                 'paired_gain_ci95_low', 'paired_gain_ci95_high', 'paired_episodes')}}))
    return rows


def chart_payloads(aggregate):
    observed = [p for p in aggregate['points'] if p['return_mean'] is not None]
    label = f"H{aggregate['H']} " + ('soft/soft refinement' if aggregate['critic_kind'] == 'soft' else 'refinement')
    payloads = {}
    for metric, title in [('return', 'Full-episode return at checkpoint 650k'),
                           ('paired_gain', 'Paired improvement over the frozen prior at 650k')]:
        references = aggregate['references']
        payloads['comparison/' + metric + '_vs_J'] = dict(
            xs=([[p['J'] for p in observed]] if observed else []) + [ROUNDS] * len(references),
            ys=([[p[metric + '_mean'] for p in observed]] if observed else []) + [[r[metric + '_mean']] * len(ROUNDS) for r in references],
            keys=([label] if observed else []) + [r['label'] for r in references],
            title=title, xname='Inner rounds J')
    return payloads


def overview_log(wandb, aggregate, statuses):
    payload = {key: wandb.plot.line_series(**value) for key, value in chart_payloads(aggregate).items()}
    for key, rows, columns in [('comparison/points', aggregate['points'], POINT_COLUMNS),
            ('comparison/paired_episodes', aggregate['episodes'], EPISODE_COLUMNS),
            ('comparison/references', aggregate['references'], REFERENCE_COLUMNS), ('campaign/settings', statuses, STATUS_COLUMNS)]:
        payload[key] = wandb.Table(columns=columns, data=[[row.get(column) for column in columns] for row in rows])
    payload.update({'campaign/evaluated': aggregate['evaluated'], 'campaign/new_evaluated': aggregate['new_evaluated'],
        'campaign/new_published': sum(row['status'] == 'published' for row in statuses), 'campaign/reused': aggregate['reused']})
    return payload


def load_completed(campaign, cell):
    from slurm.ambi_closed_loop_j_sweep import validate_completed
    if cell['reused']:
        raise ValueError('Historical J10 comes only from its verified reference')
    receipt = read(Path(cell['directory']) / 'worker-completion.json')
    verify_receipt(Path(cell['bundle']), receipt)
    if receipt['cell'] != cell['name'] or receipt['status'] != 'complete':
        raise ValueError('Worker completion identity differs')
    manifest = validate_completed(Path(cell['bundle']), cell, campaign)
    return manifest['runs'][0]['episodes'], protocol_proof(manifest, cell)


def publication_campaign(campaign, cell):
    if cell['reused']:
        raise ValueError('Never republish the historical J10 performance or diagnostic run')
    return dict(cells=[deepcopy(cell)], checkpoint_step=650000, checkpoint_sha256=cell['checkpoint_sha256'],
        inventory=campaign['inventory'], group=campaign['group'], source_run=campaign['source_run'],
        source_commit=campaign['source_commit'], overview_run_id=campaign['overview_run_id'])


def publish_cell(args):
    from slurm.ambi_aux_hj_sweep import publish_cell as publish_full_traces
    campaign = read(args.root / 'campaign.json')
    validate_scope(campaign)
    cell = campaign['cells'][args.index]
    adapter = publication_campaign(campaign, cell)
    load_completed(campaign, cell)
    if publication_complete(cell):
        return
    root = Path(cell['directory']) / 'publication-adapter'
    root.mkdir(exist_ok=True)
    path = root / 'campaign.json'
    if path.exists() and read(path) != adapter:
        raise ValueError('Per-J publication adapter changed')
    write(path, adapter)
    publish_full_traces(SimpleNamespace(root=root, index=0))
    if not publication_complete(cell):
        raise RuntimeError('Per-J performance and trace publication did not complete')


def status_rows(campaign, aggregate, completed, futures, failures, terminal=False):
    points = {p['setting']: p for p in aggregate['points']}
    rows = []
    for index, cell in enumerate(campaign['cells']):
        if cell['reused']: state = 'reused'
        elif publication_complete(cell): state = 'published'
        elif index in failures: state = 'publication_failed'
        elif index in futures: state = 'publishing'
        elif cell['name'] in completed: state = 'evaluated_awaiting_publication'
        else: state = 'evaluation_incomplete' if terminal else 'queued_or_running'
        rows.append(dict(points[cell['name']], status=state, failure=failures.get(index),
            performance_url=_url(cell['performance_run_id']), training_url=_url(cell['training_run_id'])))
    return rows


def watch(args):
    """One overview owner and at most three independent full-trace publishers."""
    import wandb
    campaign = read(args.root / 'campaign.json')
    cells = validate_scope(campaign)
    horizon, total = cells[0]['H'], len(cells)
    kind = campaign_critic(campaign)
    reused = sum(c['reused'] for c in cells)
    new = total - reused
    verify_references(campaign)
    proofs = {c['name']: protocol_proof(read(Path(c['bundle']) / 'manifest.json'), c) for c in cells if c['reused']}
    marker = args.root / 'watcher-started.json'
    if marker.exists():
        raise RuntimeError('Overview already started; inspect journals and remote history before recovery')
    write(marker, dict(pid=os.getpid(), started=time.time()))
    publishers = int(campaign.get('publisher_workers', 3))
    if not 1 <= publishers <= 3:
        raise ValueError('Expected one to three bounded publishers')
    config = dict(protocol='closed-loop-h3-j-sweep-v1' if horizon == 3 else 'closed-loop-hj-sweep-v1', campaign_group=campaign['group'], source_run=campaign['source_run'],
        source_commit=campaign['source_commit'], checkpoint_step=650000, checkpoint_sha256=cells[0]['checkpoint_sha256'],
        H=horizon, J=ROUNDS, C=16, A=4, N=128, B=256, execution_mode='mean', action_rule='tanh_mean',
        estimator='one_step', critic_kind=kind, critic_scheme=critic_scheme(kind),
        inner_critic_source=cells[0]['params']['inner_critic_source'],
        inner_horizon_critic_source=cells[0]['params']['inner_horizon_critic_source'],
        inner_sac_critic_target=cells[0]['params']['inner_sac_critic_target'],
        inner_terminal_entropy=cells[0]['params']['inner_terminal_entropy'],
        performance_objective='Undiscounted raw environment reward; no entropy bonus or terminal bootstrap.', alpha_mode='adaptive', inner_temperature_mode='auto',
        inner_temperature_initialization='inherit_outer', initial_alpha=cells[0]['initial_alpha'], target_entropy=-10.5,
        togo_return_rollouts=32, inner_replay_capacity_by_J={str(c['J']): c['params']['inner_replay_capacity'] for c in cells},
        inner_replay_scope='action', inner_replay_reset_each_round=False,
        environment_seeds=SEEDS, controller_seed=55, max_decisions=500, total_settings=total, new_settings=new, reused_settings=reused,
        settings=[dict(setting=c['name'], H=c['H'], J=c['J'], critic_kind=kind, critic_scheme=critic_scheme(kind), reused=c['reused'], replay_capacity=c['params']['inner_replay_capacity'],
            performance_url=_url(c['performance_run_id']), training_url=_url(c['training_run_id'])) for c in cells],
        references=aggregate_results(campaign, {})['references'],
        uncertainty='Five paired environment seeds; 2,000 paired bootstrap resamples; exploratory 95% intervals.')
    run = wandb.init(entity=ENTITY, project=PROJECT, id=campaign['overview_run_id'], resume='never', name=campaign['label'],
        group=campaign['group'], job_type='closed-loop-J-comparison', tags=['closed-loop', 'J-sweep', f'H{horizon}', '650k', 'soft-soft' if kind == 'soft' else 'return-only', 'mean'],
        config=config, mode='online')
    run.define_metric('axis/inner_rounds')
    run.define_metric('j_sweep/*', step_metric='axis/inner_rounds')
    attempted, completed, futures, failures, logged = set(), {}, {}, {}, set()
    previous, terminal_since = None, None

    def launch(index):
        with (Path(cells[index]['directory']) / 'publisher.log').open('w') as log:
            return subprocess.run([sys.executable, __file__, 'publish', '--root', str(args.root), '--index', str(index)],
                                  stdout=log, stderr=subprocess.STDOUT).returncode

    def update(terminal=False):
        nonlocal previous
        aggregate = aggregate_results(campaign, completed)
        statuses = status_rows(campaign, aggregate, completed, futures, failures, terminal)
        stamp = ([(row['setting'], row['status']) for row in statuses], sorted(completed))
        if stamp != previous:
            for identity, row in numeric_rows(aggregate):
                if identity not in logged:
                    run.log(row); logged.add(identity)
            run.log(overview_log(wandb, aggregate, statuses))
            published = sum(row['status'] == 'published' for row in statuses)
            progress = dict(rows=statuses, failures=failures, evaluated=aggregate['evaluated'],
                            new_evaluated=len(completed), published=published, reused=reused)
            run.summary.update(dict(status='running', evaluated=aggregate['evaluated'], new_evaluated=len(completed),
                published=published, reused=reused, total_settings=total, protocol_proof_by_setting=proofs))
            write(args.root / 'comparison-results.json', aggregate)
            write(args.root / 'progress.json', progress)
            print(f"H{horizon}: evaluated {aggregate['evaluated']}/{total} ({reused} reused); newly published {published}/{new}", flush=True)
            previous = stamp
        return statuses

    try:
        update()
        with ThreadPoolExecutor(max_workers=publishers) as pool:
            while True:
                for index, future in list(futures.items()):
                    if future.done():
                        if future.result(): failures[index] = f"Publisher failed; inspect {cells[index]['directory']}/publisher.log"
                        del futures[index]
                for index, cell in enumerate(cells):
                    if cell['reused']: continue
                    if (Path(cell['directory']) / 'worker-completion.json').exists() and cell['name'] not in completed:
                        completed[cell['name']], proofs[cell['name']] = load_completed(campaign, cell)
                    if len(futures) < publishers and index not in attempted and cell['name'] in completed and not publication_complete(cell):
                        attempted.add(index); futures[index] = pool.submit(launch, index)
                statuses = update()
                if all(row['status'] in {'published', 'reused'} for row in statuses): break
                submission = args.root / 'submission.json'
                if submission.exists() and not futures:
                    if not gpu_jobs_active(read(submission)['gpu_job_ids']):
                        terminal_since = terminal_since or time.time()
                        if time.time() - terminal_since > 90: break
                    else: terminal_since = None
                time.sleep(15)
        statuses = update(terminal=True)
        complete = len(completed) == new and all(row['status'] in {'published', 'reused'} for row in statuses)
        status = 'complete' if complete else 'incomplete'
        write(args.root / 'campaign-completion.json', dict(status=status, rows=statuses, failures=failures,
            evaluated=len(completed) + reused, new_evaluated=len(completed), reused=reused,
            published=sum(row['status'] == 'published' for row in statuses)))
        run.summary.update(dict(status=status, failed_publications=len(failures)))
        if not complete: raise RuntimeError('J sweep incomplete; inspect status table and publisher logs')
        run.finish()
    except BaseException as exc:
        write(args.root / 'campaign-completion.json', dict(status='failed', failure=str(exc), failures=failures))
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
