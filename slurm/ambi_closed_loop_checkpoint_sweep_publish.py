"""Publish the reduced H3/J10 checkpoint curve and full per-checkpoint traces.

New checkpoints share one authoritative performance curve and own independent
diagnostic runs. Performance publication is serialized while diagnostic uploads
can run concurrently. The overview includes the immutable historical 575k result
without republishing it into the new attempt.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_aux_hj_sweep import ENTITY, PROJECT, SEEDS, publish_performance, read, write
from slurm.ambi_closed_loop_publish import gpu_jobs_active, indexed_episodes, moments
from slurm.ambi_closed_loop_reward_retrace_publish import paired_comparison
from slurm.ambi_closed_loop_sampled import verify_receipt

STEPS = [100000, 200000, 300000, 400000, *range(500000, 1000001, 25000)]
POINT_COLUMNS = [
    'setting', 'training_decisions', 'checkpoint_sha256', 'reused', 'performance_run_id',
    'training_run_id', 'prior_performance_run_id', 'initial_alpha', 'return_mean', 'return_std',
    'return_episodes', 'prior_return_mean', 'prior_return_std', 'paired_gain_mean', 'paired_gain_std',
    'paired_gain_ci95_low', 'paired_gain_ci95_high', 'paired_episodes',
]
EPISODE_COLUMNS = ['setting', 'training_decisions', 'reused', 'seed', 'solver_seed',
                   'return', 'prior_return', 'paired_gain']
STATUS_COLUMNS = [*POINT_COLUMNS, 'status', 'performance_url', 'training_url', 'failure']


def validate_completed(*args, **kwargs):
    from slurm.ambi_closed_loop_checkpoint_sweep import validate_completed as validate
    return validate(*args, **kwargs)


def verify_references(campaign):
    """Reopen immutable sources before using their precomputed episode rows."""
    from slurm.ambi_closed_loop_checkpoint_sweep import load_prior, load_reused
    for cell in campaign['cells']:
        prior = load_prior(cell['prior_reference'], campaign['inventory'])
        if indexed_episodes(prior['episodes']) != indexed_episodes(cell['prior_reference']['episodes']):
            raise ValueError('Prior changed after campaign preparation')
        if cell['reused']:
            reused = load_reused(cell['reuse_reference'], campaign['inventory'])
            if indexed_episodes(reused['episodes']) != indexed_episodes(cell['reuse_reference']['episodes']):
                raise ValueError('Historical H3/J10 result changed after preparation')


def validate_scope(campaign):
    cells = campaign['cells']
    if sorted(c['checkpoint_step'] for c in cells) != STEPS or len({c['name'] for c in cells}) != 25:
        raise ValueError('Expected the 25 explicitly selected checkpoint settings')
    if [c['checkpoint_step'] for c in cells if c['reused']] != [575000]:
        raise ValueError('Only the pinned 575k evaluation may be reused')
    new = [c for c in cells if not c['reused']]
    historical, = [c for c in cells if c['reused']]
    if len({c['performance_run_id'] for c in new}) != 1 or len({c['run_dir'] for c in new}) != 1:
        raise ValueError('New checkpoints must share one authoritative performance curve')
    ids = [c['training_run_id'] for c in cells] + [new[0]['performance_run_id'], historical['performance_run_id'], campaign['overview_run_id']]
    if len(set(ids)) != len(ids):
        raise ValueError('Training, historical, new-performance, and overview identities must remain distinct')
    for cell in cells:
        if ((cell['H'], cell['J'], cell['estimator'], cell['execution_mode'], cell['alpha_mode'], cell['critic_kind'])
                != (3, 10, 'one_step', 'mean', 'adaptive', 'return_only')):
            raise ValueError('Unexpected checkpoint-sweep method')
        if cell['training_decisions'] != cell['checkpoint_step']:
            raise ValueError('Checkpoint axis differs from its source')
        if not math.isfinite(cell['initial_alpha']) or cell['initial_alpha'] <= 0:
            raise ValueError('Each checkpoint must supply its own saved positive temperature')
        for reference in [cell['prior_reference'], *([cell['reuse_reference']] if cell['reused'] else [])]:
            if (reference['checkpoint_step'], reference['checkpoint_sha256']) != (cell['checkpoint_step'], cell['checkpoint_sha256']):
                raise ValueError('Reference belongs to a different checkpoint')
        indexed_episodes(cell['prior_reference']['episodes'])
    return cells


def aggregate_results(campaign, completed):
    """All requested points remain visible; absent results are null, never zero."""
    cells = validate_scope(campaign)
    if set(completed) - {c['name'] for c in cells if not c['reused']}:
        raise ValueError('Unexpected or republished historical result')
    points, rows = [], []
    for cell in sorted(cells, key=lambda c: c['checkpoint_step']):
        prior = cell['prior_reference']
        prior_stats = moments(e['return'] for e in prior['episodes'])
        point = dict.fromkeys(POINT_COLUMNS)
        point.update(setting=cell['name'], training_decisions=cell['checkpoint_step'],
            checkpoint_sha256=cell['checkpoint_sha256'], reused=cell['reused'],
            performance_run_id=cell['performance_run_id'], training_run_id=cell['training_run_id'],
            prior_performance_run_id=prior.get('performance_run_id'), initial_alpha=cell['initial_alpha'],
            prior_return_mean=prior_stats['mean'], prior_return_std=prior_stats['std'])
        episodes = cell['reuse_reference']['episodes'] if cell['reused'] else completed.get(cell['name'])
        if episodes is not None:
            stats = moments(e['return'] for e in episodes)
            comparison = paired_comparison(episodes, prior['episodes'])
            point.update(return_mean=stats['mean'], return_std=stats['std'], return_episodes=stats['episodes'])
            for suffix in ('mean', 'std', 'ci95_low', 'ci95_high'):
                point['paired_gain_' + suffix] = comparison['metrics']['comparison/sample_minus_mean_' + suffix]
            point['paired_episodes'] = comparison['metrics']['comparison/sample_minus_mean_paired_episodes']
            indexed = indexed_episodes(episodes)
            for row in comparison['rows']:
                episode = indexed[row['seed'], row['solver_seed']]
                if ('paired_return_delta' in episode and not math.isclose(episode['paired_return_delta'],
                        row['sample_minus_mean'], rel_tol=1e-10, abs_tol=1e-10)):
                    raise ValueError('Stored paired gain differs from the checkpoint-matched prior')
                rows.append(dict(setting=cell['name'], training_decisions=cell['checkpoint_step'], reused=cell['reused'],
                    seed=row['seed'], solver_seed=row['solver_seed'], **{'return': row['sampled_return']},
                    prior_return=row['mean_return'], paired_gain=row['sample_minus_mean']))
        points.append(point)
    return dict(points=points, episodes=rows, evaluated=len(completed) + 1, new_evaluated=len(completed), reused=1,
        comparison='Refined return minus the frozen-prior return at the same checkpoint and matched seeds.',
        uncertainty='Five paired environment seeds; 2,000 paired bootstrap resamples; exploratory 95% intervals.',
        bootstrap_seed=20260912, bootstrap_resamples=2000)


def numeric_rows(aggregate):
    rows = []
    for point in aggregate['points']:
        axis = {'checkpoint/training_decisions': point['training_decisions']}
        rows.append((point['setting'] + '/prior', {**axis,
            'checkpoint_sweep/prior_return_mean': point['prior_return_mean'],
            'checkpoint_sweep/prior_return_std': point['prior_return_std'],
            'checkpoint_sweep/initial_alpha': point['initial_alpha']}))
        if point['return_mean'] is not None:
            rows.append((point['setting'], {**axis, **{'checkpoint_sweep/' + key: point[key]
                for key in ('return_mean', 'return_std', 'return_episodes', 'paired_gain_mean', 'paired_gain_std',
                            'paired_gain_ci95_low', 'paired_gain_ci95_high', 'paired_episodes')}}))
    return rows


def chart_payloads(aggregate):
    observed = [p for p in aggregate['points'] if p['return_mean'] is not None]
    checkpoints = [p['training_decisions'] for p in aggregate['points']]
    measured = [p['training_decisions'] for p in observed]
    return {
        'comparison/return_vs_checkpoint': dict(xs=[measured, checkpoints],
            ys=[[p['return_mean'] for p in observed], [p['prior_return_mean'] for p in aggregate['points']]],
            keys=['H3/J10 refinement', 'Frozen prior (reused)'],
            title='Full-episode return across training checkpoints', xname='Training decisions'),
        'comparison/paired_gain_vs_checkpoint': dict(xs=[measured, checkpoints],
            ys=[[p['paired_gain_mean'] for p in observed], [0.] * len(checkpoints)],
            keys=['Refinement minus matched prior', 'No improvement'],
            title='Paired improvement over the checkpoint-matched prior', xname='Training decisions'),
    }


def overview_log(wandb, aggregate, statuses):
    payload = {key: wandb.plot.line_series(**value) for key, value in chart_payloads(aggregate).items()}
    for key, rows, columns in [('comparison/points', aggregate['points'], POINT_COLUMNS),
                               ('comparison/paired_episodes', aggregate['episodes'], EPISODE_COLUMNS),
                               ('campaign/settings', statuses, STATUS_COLUMNS)]:
        payload[key] = wandb.Table(columns=columns, data=[[row.get(column) for column in columns] for row in rows])
    payload.update({'campaign/evaluated': aggregate['evaluated'], 'campaign/new_evaluated': aggregate['new_evaluated'],
                    'campaign/new_published': sum(row['status'] == 'published' for row in statuses), 'campaign/reused': 1})
    return payload


def protocol_proof(manifest, cell):
    result = manifest['runs'][0]['result']
    metrics = result['model_metrics']
    proof = {'checkpoint_step': cell['checkpoint_step'], 'checkpoint_sha256': cell['checkpoint_sha256'],
        'source_code': manifest['code'], 'reused': cell['reused'], 'initial_alpha': cell['initial_alpha'],
        'checkpoint_state_proof': cell.get('checkpoint_state_proof'),
        'resolved_config': manifest['runs'][0]['resolved_config']}
    for key in ('inner_alpha_initial', 'inner_alpha_final',
                'inner_critic_optimizer_steps', 'inner_actor_optimizer_steps', 'inner_temperature_optimizer_steps',
                'inner_model_steps', 'inner_buffer_size', 'inner_compile_fallback'):
        proof[key] = metrics[key]
    if 'inner_eval_execution_sampled' in metrics:
        proof['inner_eval_execution_sampled'] = metrics['inner_eval_execution_sampled']
    elif cell['reused']:
        # The historical mean-action evaluator predates execution sampling and
        # did not record this metric. Preserve its explicit protocol evidence.
        if (manifest['protocol']['action_rule'] != 'tanh_mean'
                or manifest['runs'][0]['resolved_config']['inner_execution_action'] != 'mean'):
            raise ValueError('Historical mean execution requires explicit protocol evidence')
        proof['execution'] = dict(protocol_action_rule=manifest['protocol']['action_rule'],
                                  configured_inner_execution_action='mean',
                                  evidence='Validated historical protocol and planner; newer result flags were not recorded.')
        proof['unavailable_historical_metrics'] = ['inner_eval_execution_sampled']
    else:
        raise ValueError('New evaluations require the execution sampling metric')
    return proof


def load_completed(campaign, cell):
    if cell['reused']:
        raise ValueError('Historical data is loaded only from its verified reference')
    receipt = read(Path(cell['directory']) / 'worker-completion.json')
    verify_receipt(Path(cell['bundle']), receipt)
    if receipt['cell'] != cell['name'] or receipt['status'] != 'complete':
        raise ValueError('Worker completion identity differs')
    manifest = validate_completed(Path(cell['bundle']), cell, campaign)
    return manifest['runs'][0]['episodes'], protocol_proof(manifest, cell)


def publication_campaign(campaign, cell):
    """The existing full-trace publisher receives only this checkpoint's state."""
    if cell['reused']:
        raise ValueError('Never republish the historical 575k performance or diagnostic run')
    return dict(cells=[deepcopy(cell)], checkpoint_step=cell['checkpoint_step'],
        checkpoint_sha256=cell['checkpoint_sha256'], inventory=campaign['inventory'],
        group=campaign['group'], source_run=campaign['source_run'], source_commit=campaign['source_commit'],
        overview_run_id=campaign['overview_run_id'])


def serialized_performance(run_dir):
    """Block only competing curve publication; training uploads remain parallel."""
    with (Path(run_dir) / '.checkpoint-sweep-publisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return publish_performance(run_dir)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def publication_complete(cell):
    """A cumulative curve counter cannot prove this checkpoint was published."""
    path = Path(cell['directory']) / 'publication-completion.json'
    if not path.exists():
        return False
    receipt = read(path)
    training = read(path.with_name('training-publication.json'))
    entries = [entry for entry in read(Path(cell['run_dir']) / 'publication.json')['records'].values()
               if entry['checkpoint_step'] == cell['checkpoint_step'] and not entry.get('record_kind')]
    if (receipt.get('status') != 'complete' or receipt.get('cell') != cell['name']
            or receipt.get('training_run_id') != cell['training_run_id']
            or training.get('status') != 'complete' or training.get('run_id') != cell['training_run_id']
            or receipt.get('performance', {}).get('run_id') != cell['performance_run_id']
            or len(entries) != 1 or entries[0]['status'] != 'published'
            or entries[0]['checkpoint_sha256'] != cell['checkpoint_sha256']):
        raise ValueError('Checkpoint performance or training publication identity mismatch')
    return True


def publish_cell(args):
    from slurm.ambi_aux_hj_sweep import publish_cell as publish_full_traces
    campaign = read(args.root / 'campaign.json')
    validate_scope(campaign)
    cell = campaign['cells'][args.index]
    adapter = publication_campaign(campaign, cell)
    load_completed(campaign, cell)  # Strong source, checkpoint, config, and runtime checks first.
    if publication_complete(cell):
        return
    root = Path(cell['directory']) / 'publication-adapter'
    root.mkdir(exist_ok=True)
    path = root / 'campaign.json'
    if path.exists() and read(path) != adapter:
        raise ValueError('Per-checkpoint publication adapter changed')
    write(path, adapter)
    publish_full_traces(SimpleNamespace(root=root, index=0), performance_publisher=serialized_performance)
    if not publication_complete(cell):
        raise RuntimeError('Per-checkpoint performance and trace publication did not complete')


def _url(run_id):
    return f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}'


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
    """One CPU overview owner, with up to three independent upload subprocesses."""
    import wandb
    campaign = read(args.root / 'campaign.json')
    cells = validate_scope(campaign)
    verify_references(campaign)
    proofs = {cell['name']: protocol_proof(read(Path(cell['bundle']) / 'manifest.json'), cell)
              for cell in cells if cell['reused']}
    marker = args.root / 'watcher-started.json'
    if marker.exists():
        raise RuntimeError('Overview already started; inspect its journals and remote history before recovery')
    write(marker, dict(pid=os.getpid(), started=time.time()))
    publishers = int(campaign.get('publisher_workers', 3))
    if not 1 <= publishers <= 3:
        raise ValueError('Expected one to three bounded checkpoint publishers')
    config = dict(protocol='closed-loop-checkpoint-sweep-v1', campaign_group=campaign['group'],
        source_run=campaign['source_run'], source_commit=campaign['source_commit'],
        checkpoint_steps=STEPS, H=3, J=10, C=16, A=4, N=128, B=256,
        execution_mode='mean', action_rule='tanh_mean', estimator='one_step', critic_kind='return_only',
        alpha_mode='adaptive', inner_temperature_mode='auto', inner_temperature_initialization='inherit_outer',
        target_entropy=-10.5, togo_return_rollouts=32,
        inner_replay_capacity=3840, inner_replay_scope='action', inner_replay_reset_each_round=False,
        environment_seeds=SEEDS, controller_seed=55, max_decisions=500,
        total_settings=25, new_settings=24, reused_settings=1,
        prior_reference='Existing exact-checkpoint frozen-prior mean episodes; never reevaluated.',
        temperature='Initialize independently from each checkpoint saved SAC alpha; adapt within each solve.',
        settings=[dict(setting=c['name'], training_decisions=c['checkpoint_step'],
            checkpoint_sha256=c['checkpoint_sha256'], initial_alpha=c['initial_alpha'], reused=c['reused'],
            checkpoint_state_proof=c.get('checkpoint_state_proof'),
            performance_url=_url(c['performance_run_id']), training_url=_url(c['training_run_id'])) for c in cells],
        uncertainty='Five paired environment seeds; 2,000 paired bootstrap resamples; exploratory 95% intervals.')
    run = wandb.init(entity=ENTITY, project=PROJECT, id=campaign['overview_run_id'], resume='never',
        name=campaign['label'], group=campaign['group'], job_type='closed-loop-checkpoint-comparison',
        tags=['closed-loop', 'checkpoint-sweep', 'H3', 'J10', 'return-only', 'mean'], config=config, mode='online')
    run.define_metric('checkpoint/training_decisions')
    run.define_metric('checkpoint_sweep/*', step_metric='checkpoint/training_decisions')
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
                    run.log(row)
                    logged.add(identity)
            run.log(overview_log(wandb, aggregate, statuses))
            published = sum(row['status'] == 'published' for row in statuses)
            run.summary.update(dict(status='running', evaluated=aggregate['evaluated'], new_evaluated=len(completed),
                published=published, reused=1, total_settings=25, protocol_proof_by_setting=proofs))
            write(args.root / 'comparison-results.json', aggregate)
            write(args.root / 'progress.json', dict(rows=statuses, failures=failures, evaluated=aggregate['evaluated'],
                                                   new_evaluated=len(completed), published=published, reused=1))
            print(f"Evaluated {aggregate['evaluated']}/25 (one reused); newly published {published}/24", flush=True)
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
                            failures[index] = f"Publisher exited {rc}; inspect {cells[index]['directory']}/publisher.log"
                        del futures[index]
                for index, cell in enumerate(cells):
                    if cell['reused']:
                        continue
                    if (Path(cell['directory']) / 'worker-completion.json').exists() and cell['name'] not in completed:
                        completed[cell['name']], proofs[cell['name']] = load_completed(campaign, cell)
                    if (len(futures) < publishers and index not in attempted and cell['name'] in completed
                            and not publication_complete(cell)):
                        attempted.add(index)
                        futures[index] = pool.submit(launch, index)
                statuses = update()
                if all(row['status'] in {'published', 'reused'} for row in statuses):
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
        complete = len(completed) == 24 and all(row['status'] in {'published', 'reused'} for row in statuses)
        status = 'complete' if complete else 'incomplete'
        write(args.root / 'campaign-completion.json', dict(status=status, rows=statuses, failures=failures,
            evaluated=len(completed) + 1, new_evaluated=len(completed), reused=1,
            published=sum(row['status'] == 'published' for row in statuses)))
        run.summary.update(dict(status=status, failed_publications=len(failures)))
        if not complete:
            raise RuntimeError('Checkpoint campaign incomplete; inspect status table and publisher logs')
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
