"""Publish critic-only LoRA H/J/rank sweeps against immutable paired dense controls."""
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
from slurm.ambi_closed_loop_publish import gpu_jobs_active, indexed_episodes, moments
from slurm.ambi_closed_loop_j_sweep_publish import comparison_stats
from slurm.ambi_closed_loop_sampled import verify_receipt
from slurm.ambi_closed_loop_checkpoint_sweep_publish import publication_complete, protocol_proof

HORIZONS = [1, 2, 3]
ROUNDS = [1, 2, 4, 6, 8, 10]
RANKS = [16, 96]
UNCERTAINTY = 'Five paired environment seeds; 2,000 paired bootstrap resamples; exploratory 95% intervals, unadjusted for multiple comparisons.'
STATS = ('mean', 'std', 'ci95_low', 'ci95_high')
POINT_COLUMNS = ['setting', 'H', 'J', 'critic_adaptation', 'lora_rank', 'lora_layers', 'lora_scale', 'lora_weight_decay', 'measurement', 'reused',
    'training_decisions', 'checkpoint_sha256', 'initial_alpha', 'performance_run_id', 'training_run_id',
    'uniform_performance_run_id', 'uniform_training_run_id', 'prior_performance_run_id',
    'return_mean', 'return_std', 'return_episodes', 'uniform_return_mean', 'uniform_return_std',
    'prior_return_mean', 'prior_return_std',
    *['lora_minus_dense_' + stat for stat in STATS], *['paired_gain_' + stat for stat in STATS],
    'paired_episodes']
EPISODE_COLUMNS = ['setting', 'H', 'J', 'lora_rank', 'reused', 'measurement', 'seed', 'solver_seed',
    'return', 'uniform_return', 'prior_return', 'lora_minus_dense', 'paired_gain']
STATUS_COLUMNS = [*POINT_COLUMNS, 'status', 'performance_url', 'training_url',
    'uniform_performance_url', 'uniform_training_url', 'failure']
REFERENCE_COLUMNS = ['kind', 'label', 'H', 'J', 'training_decisions', 'checkpoint_sha256',
    'return_mean', 'return_std', 'performance_run_id', 'training_run_id', 'performance_url', 'training_url']


def _url(run_id):
    return f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}' if run_id else None


def validate_scope(campaign):
    cells = campaign['cells']
    expected_grid = [(rank, h, j) for rank in RANKS for h in HORIZONS for j in ROUNDS]
    if (sorted((c['lora_rank'], c['H'], c['J']) for c in cells) != expected_grid
            or len({c['name'] for c in cells}) != len(expected_grid)):
        raise ValueError('Expected exactly rank16/96 by H1/2/3 by J1/2/4/6/8/10')
    if (campaign.get('H', HORIZONS) != HORIZONS or campaign.get('J', ROUNDS) != ROUNDS
            or campaign.get('ranks', RANKS) != RANKS):
        raise ValueError('Declared rank/H/J grid differs from the cells')
    ids = [campaign['overview_run_id']] + [c[k] for c in cells for k in ('performance_run_id', 'training_run_id')]
    if len(set(ids)) != len(ids) or len({c['run_dir'] for c in cells}) != len(cells):
        raise ValueError('Each LoRA cell, diagnostic, and overview must have distinct ownership')
    reference_ids = {reference[key] for cell in cells for reference in
        (cell['prior_reference'], cell['uniform_reference']) for key in ('performance_run_id', 'training_run_id')
        if reference.get(key)}
    if reference_ids & set(ids):
        raise ValueError('New publications cannot overwrite any historical reference identity')
    if len({c['checkpoint_sha256'] for c in cells}) != 1 or len({c['initial_alpha'] for c in cells}) != 1:
        raise ValueError('All cells must use the same checkpoint and saved temperature')
    prior, controls = None, {}
    for cell in cells:
        if (cell['checkpoint_step'], cell['training_decisions'], cell['estimator'], cell['execution_mode'],
                cell['alpha_mode'], cell['critic_kind']) != (575000, 575000, 'one_step', 'mean', 'adaptive', 'return_only'):
            raise ValueError('Unexpected checkpoint or LoRA method')
        if cell['reused']:
            raise ValueError('Every LoRA cell requires a new measurement, including J1')
        if not math.isfinite(cell['initial_alpha']) or cell['initial_alpha'] <= 0:
            raise ValueError('Expected the checkpoint saved positive temperature')
        if (cell['lora_layers'], cell['lora_scale'], cell['lora_weight_decay'], cell['replay_strategy']) != (
                'input_hidden', 1., .0002, 'uniform'):
            raise ValueError('Unexpected LoRA placement, scale, decay, or replay metadata')
        params = cell['params']
        expected = dict(inner_replay_strategy='uniform', inner_actor_adaptation='clone',
            inner_critic_adaptation='lora_rl', inner_critic_lora_rank=cell['lora_rank'],
            inner_critic_lora_layers='input_hidden', inner_critic_lora_scale=1., inner_critic_lora_weight_decay=.0002,
            inner_critic_updates_per_round=16, inner_actor_updates_per_round=4,
            inner_actor_source='sac', inner_horizon_actor_source='sac',
            inner_critic_source='aux_return', inner_horizon_critic_source='aux_return',
            inner_sac_critic_target='reward_only', inner_terminal_entropy='none')
        if any(params.get(k) != v for k, v in expected.items()):
            raise ValueError('Critic-only LoRA, replay, update budgets, or return/return semantics differ')
        if params['inner_replay_capacity'] < 128 * cell['H'] * cell['J']:
            raise ValueError('Replay must retain every collected round')
        for reference in (cell['prior_reference'], cell['uniform_reference']):
            if (reference['checkpoint_step'], reference['checkpoint_sha256']) != (575000, cell['checkpoint_sha256']):
                raise ValueError('Reference belongs to a different checkpoint')
            indexed_episodes(reference['episodes'])
        uniform = cell['uniform_reference']
        key = (cell['H'], cell['J'])
        if (uniform['H'], uniform['J']) != key:
            raise ValueError('Dense uniform reference belongs to a different H/J setting')
        current = cell['prior_reference']
        if prior is not None and current != prior:
            raise ValueError('All settings require the same frozen-prior reference')
        if indexed_episodes(uniform['episodes']).keys() != indexed_episodes(current['episodes']).keys():
            raise ValueError('Dense/prior reference solver seeds differ')
        if key in controls and controls[key] != uniform:
            raise ValueError('Both ranks require the same exact dense reference for each H/J setting')
        prior, controls[key] = current, uniform
    return cells


def verify_references(campaign):
    from slurm.ambi_closed_loop_critic_lora import load_prior, load_uniform
    prior = campaign['cells'][0]['prior_reference']
    if indexed_episodes(load_prior(prior, campaign['inventory'])['episodes']) != indexed_episodes(prior['episodes']):
        raise ValueError('Prior changed after preparation')
    verified = set()
    for cell in campaign['cells']:
        key = (cell['H'], cell['J'])
        if key in verified:
            continue
        reference = cell['uniform_reference']
        if indexed_episodes(load_uniform(reference, campaign['inventory'])['episodes']) != indexed_episodes(reference['episodes']):
            raise ValueError('Dense reference changed after preparation')
        verified.add(key)


def _reference_row(reference, *, kind, label, horizon=None, rounds=None):
    stats = moments(e['return'] for e in reference['episodes'])
    return dict(kind=kind, label=label, H=horizon, J=rounds, training_decisions=575000,
        checkpoint_sha256=reference['checkpoint_sha256'], return_mean=stats['mean'], return_std=stats['std'],
        performance_run_id=reference.get('performance_run_id'), training_run_id=reference.get('training_run_id'),
        performance_url=_url(reference.get('performance_run_id')), training_url=_url(reference.get('training_run_id')))


def aggregate_results(campaign, completed):
    """Missing LoRA measurements remain null; all dense/prior controls are reused."""
    cells = validate_scope(campaign)
    if set(completed) - {c['name'] for c in cells}:
        raise ValueError('Unexpected result identity')
    prior = cells[0]['prior_reference']
    prior_stats = moments(e['return'] for e in prior['episodes'])
    references = [_reference_row(prior, kind='prior', label='Frozen prior (reused)')]
    points, rows, referenced = [], [], set()
    for cell in sorted(cells, key=lambda c: (c['lora_rank'], c['H'], c['J'])):
        uniform = cell['uniform_reference']
        uniform_stats = moments(e['return'] for e in uniform['episodes'])
        key = (cell['H'], cell['J'])
        if key not in referenced:
            references.append(_reference_row(uniform, kind='dense_uniform',
                label=f"H{cell['H']}/J{cell['J']} dense uniform (reused)", horizon=cell['H'], rounds=cell['J']))
            referenced.add(key)
        point = dict.fromkeys(POINT_COLUMNS)
        point.update(setting=cell['name'], H=cell['H'], J=cell['J'], critic_adaptation='lora_rl',
            lora_rank=cell['lora_rank'], lora_layers=cell['lora_layers'], lora_scale=cell['lora_scale'],
            lora_weight_decay=cell['lora_weight_decay'], measurement='new critic-only LoRA evaluation', reused=False,
            training_decisions=575000, checkpoint_sha256=cell['checkpoint_sha256'], initial_alpha=cell['initial_alpha'],
            performance_run_id=cell['performance_run_id'], training_run_id=cell['training_run_id'],
            uniform_performance_run_id=uniform['performance_run_id'], uniform_training_run_id=uniform['training_run_id'],
            prior_performance_run_id=prior.get('performance_run_id'),
            uniform_return_mean=uniform_stats['mean'], uniform_return_std=uniform_stats['std'],
            prior_return_mean=prior_stats['mean'], prior_return_std=prior_stats['std'])
        episodes = completed.get(cell['name'])
        if episodes is not None:
            stats = moments(e['return'] for e in episodes)
            paired = comparison_stats(episodes, prior['episodes'])
            direct = comparison_stats(episodes, uniform['episodes'])
            point.update(return_mean=stats['mean'], return_std=stats['std'], return_episodes=stats['episodes'],
                **{'paired_gain_' + key: paired[key] for key in STATS},
                **{'lora_minus_dense_' + key: direct[key] for key in STATS}, paired_episodes=direct['paired_episodes'])
            indexed, baseline = indexed_episodes(episodes), indexed_episodes(uniform['episodes'])
            for key, prior_episode in sorted(indexed_episodes(prior['episodes']).items()):
                episode = indexed[key]
                gain = episode['return'] - prior_episode['return']
                if 'paired_return_delta' in episode and not math.isclose(episode['paired_return_delta'], gain, rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError('Stored gain differs from the matched prior')
                rows.append(dict(setting=cell['name'], H=cell['H'], J=cell['J'], lora_rank=cell['lora_rank'], reused=False,
                    measurement=point['measurement'], seed=key[0], solver_seed=key[1], **{'return': episode['return']},
                    uniform_return=baseline[key]['return'], prior_return=prior_episode['return'], paired_gain=gain,
                    lora_minus_dense=episode['return'] - baseline[key]['return']))
        points.append(point)
    return dict(points=points, episodes=rows, references=references, evaluated=len(completed),
        new_evaluated=len(completed), reused=0, reused_dense_controls=18, bootstrap_seed=20260912, bootstrap_resamples=2000,
        comparison='Critic-only LoRA minus exact-H/J dense critic with uniform replay at matched environment/controller seeds and checkpoint 575k.',
        reuse_note='The frozen prior and 18 dense controls are reused across both ranks; all 36 LoRA settings are new measurements.',
        uncertainty=UNCERTAINTY)


def numeric_rows(aggregate):
    rows, referenced = [], set()
    for point in aggregate['points']:
        axis = {'axis/inner_rounds': point['J']}
        key = (point['H'], point['J'])
        if key not in referenced:
            prefix = f"critic_lora_sweep/H{point['H']}/"
            rows.append((f"H{point['H']}/J{point['J']}/references", {**axis,
                prefix + 'dense_return_mean': point['uniform_return_mean'], prefix + 'prior_return_mean': point['prior_return_mean']}))
            referenced.add(key)
        if point['return_mean'] is not None:
            prefix = f"critic_lora_sweep/r{point['lora_rank']}/H{point['H']}/"
            keys = ['return_mean', 'return_std', 'return_episodes', 'paired_episodes',
                *['paired_gain_' + s for s in STATS], *['lora_minus_dense_' + s for s in STATS]]
            rows.append((point['setting'], {**axis, **{prefix + key: point[key] for key in keys}}))
    return rows


def chart_payloads(aggregate):
    returns = dict(xs=[], ys=[], keys=[], title='575k return/return: critic-only LoRA versus dense critic', xname='Inner rounds J')
    gains = dict(xs=[ROUNDS], ys=[[0.] * len(ROUNDS)], keys=['No change versus dense critic'],
        title='Paired LoRA minus dense return (95% bootstrap bounds)', xname='Inner rounds J')
    charts = {'comparison/return_vs_J': returns, 'comparison/lora_minus_dense_vs_J': gains}
    for horizon in HORIZONS:
        points = [p for p in aggregate['points'] if p['H'] == horizon]
        dense = [p for p in points if p['lora_rank'] == RANKS[0]]
        returns['xs'].append([p['J'] for p in dense]); returns['ys'].append([p['uniform_return_mean'] for p in dense])
        returns['keys'].append(f'H{horizon} dense critic (reused)')
        # Per-horizon panels keep the rank comparison readable beside the combined overview.
        detail = dict(xs=[ROUNDS], ys=[[p['uniform_return_mean'] for p in dense]], keys=['Dense critic (reused)'],
            title=f'H{horizon}: critic-only LoRA rank comparison', xname='Inner rounds J')
        for rank in RANKS:
            observed = [p for p in points if p['lora_rank'] == rank and p['return_mean'] is not None]
            xs, ys = [p['J'] for p in observed], [p['return_mean'] for p in observed]
            returns['xs'].append(xs); returns['ys'].append(ys); returns['keys'].append(f'H{horizon} LoRA r{rank}')
            detail['xs'].append(xs); detail['ys'].append(ys); detail['keys'].append(f'LoRA r{rank}')
            for stat, label in [('mean', 'mean'), ('ci95_low', '95% lower'), ('ci95_high', '95% upper')]:
                gains['xs'].append(xs); gains['ys'].append([p['lora_minus_dense_' + stat] for p in observed])
                gains['keys'].append(f'H{horizon} r{rank} {label}')
        charts[f'comparison/H{horizon}_return_vs_J'] = detail
    returns['xs'].append(ROUNDS); returns['ys'].append([aggregate['references'][0]['return_mean']] * len(ROUNDS))
    returns['keys'].append('Frozen prior (reused)')
    return charts


def overview_log(wandb, aggregate, statuses):
    payload = {key: wandb.plot.line_series(**value) for key, value in chart_payloads(aggregate).items()}
    for key, rows, columns in [('comparison/points', aggregate['points'], POINT_COLUMNS),
            ('comparison/paired_episodes', aggregate['episodes'], EPISODE_COLUMNS),
            ('comparison/references', aggregate['references'], REFERENCE_COLUMNS), ('campaign/settings', statuses, STATUS_COLUMNS)]:
        payload[key] = wandb.Table(columns=columns, data=[[row.get(column) for column in columns] for row in rows])
    payload.update({'campaign/evaluated': aggregate['evaluated'], 'campaign/new_evaluated': aggregate['new_evaluated'],
        'campaign/new_published': sum(row['status'] == 'published' for row in statuses), 'campaign/reused': 0, 'campaign/reused_dense_controls': 18})
    return payload


def load_completed(campaign, cell):
    from slurm.ambi_closed_loop_critic_lora import validate_completed
    if cell['reused']:
        raise ValueError('Every LoRA setting requires a new worker result')
    receipt = read(Path(cell['directory']) / 'worker-completion.json')
    verify_receipt(Path(cell['bundle']), receipt)
    if receipt['cell'] != cell['name'] or receipt['status'] != 'complete':
        raise ValueError('Worker completion identity differs')
    manifest = validate_completed(Path(cell['bundle']), cell, campaign)
    proof = protocol_proof(manifest, cell)
    proof['critic_adaptation'] = {key: manifest['runs'][0]['resolved_config'][key] for key in
        ('inner_actor_adaptation', 'inner_critic_adaptation', 'inner_critic_lora_rank',
         'inner_critic_lora_layers', 'inner_critic_lora_scale', 'inner_critic_lora_weight_decay', 'inner_replay_strategy')}
    return manifest['runs'][0]['episodes'], proof


def publication_campaign(campaign, cell):
    if cell['reused']:
        raise ValueError('Never publish a reused control as a LoRA result')
    return dict(cells=[deepcopy(cell)], checkpoint_step=575000, checkpoint_sha256=cell['checkpoint_sha256'],
        inventory=campaign['inventory'], group=campaign['group'], source_run=campaign['source_run'],
        source_commit=campaign['source_commit'], overview_run_id=campaign['overview_run_id'])


def publish_cell(args):
    from slurm.ambi_aux_hj_sweep import publish_cell as publish_full_traces
    campaign = read(args.root / 'campaign.json')
    validate_scope(campaign)
    cell = campaign['cells'][args.index]
    adapter = publication_campaign(campaign, cell)
    load_completed(campaign, cell)
    if publication_complete(cell): return
    root = Path(cell['directory']) / 'publication-adapter'
    root.mkdir(exist_ok=True)
    path = root / 'campaign.json'
    if path.exists() and read(path) != adapter:
        raise ValueError('Per-setting publication adapter changed')
    write(path, adapter)
    publish_full_traces(SimpleNamespace(root=root, index=0))
    if not publication_complete(cell):
        raise RuntimeError('LoRA performance and trace publication did not complete')


def status_rows(campaign, aggregate, completed, futures, failures, terminal=False):
    points = {p['setting']: p for p in aggregate['points']}
    rows = []
    for index, cell in enumerate(campaign['cells']):
        if publication_complete(cell): state = 'published'
        elif index in failures: state = 'publication_failed'
        elif index in futures: state = 'publishing'
        elif cell['name'] in completed: state = 'evaluated_awaiting_publication'
        else: state = 'evaluation_incomplete' if terminal else 'queued_or_running'
        uniform = cell['uniform_reference']
        rows.append(dict(points[cell['name']], status=state, failure=failures.get(index),
            performance_url=_url(cell['performance_run_id']), training_url=_url(cell['training_run_id']),
            uniform_performance_url=_url(uniform['performance_run_id']), uniform_training_url=_url(uniform['training_run_id'])))
    return rows


def watch(args):
    """One overview owner and bounded independent full-trace publishers."""
    import wandb
    campaign = read(args.root / 'campaign.json')
    cells = validate_scope(campaign)
    verify_references(campaign)
    proofs = {}
    marker = args.root / 'watcher-started.json'
    if marker.exists():
        raise RuntimeError('Overview already started; inspect journals and remote history before recovery')
    publishers = int(campaign.get('publisher_workers', 3))
    if not 1 <= publishers <= 3:
        raise ValueError('Expected one to three bounded publishers')
    write(marker, dict(pid=os.getpid(), started=time.time()))
    config = dict(protocol='closed-loop-critic-lora-hj-sweep-v1', campaign_group=campaign['group'], source_run=campaign['source_run'],
        source_commit=campaign['source_commit'], checkpoint_step=575000, checkpoint_sha256=cells[0]['checkpoint_sha256'],
        H=HORIZONS, J=ROUNDS, ranks=RANKS, C=16, A=4, N=128, B=256, execution_mode='mean', action_rule='tanh_mean',
        estimator='one_step', critic_kind='return_only', critic_scheme='return_return',
        inner_replay_strategy='uniform', inner_actor_adaptation='clone', inner_critic_adaptation='lora_rl',
        inner_critic_lora_layers='input_hidden', inner_critic_lora_scale=1., inner_critic_lora_weight_decay=.0002,
        adaptation='Fresh critic input/hidden LoRA factors at each real decision; dense actor, trainable critic heads, biases and normalization.',
        performance_objective='Undiscounted raw environment reward; no entropy bonus or terminal bootstrap.',
        alpha_mode='adaptive', inner_temperature_mode='auto', inner_temperature_initialization='inherit_outer',
        initial_alpha=cells[0]['initial_alpha'], target_entropy=-10.5, aux_return_detach_representation=False,
        inner_replay_scope='action', inner_replay_reset_each_round=False, togo_return_rollouts=32,
        environment_seeds=SEEDS, controller_seed=55, max_decisions=500, total_settings=36, new_settings=36, reused_settings=0, reused_dense_controls=18,
        settings=[dict(setting=c['name'], H=c['H'], J=c['J'], lora_rank=c['lora_rank'], reused=c['reused'], replay_capacity=c['params']['inner_replay_capacity'],
            performance_url=_url(c['performance_run_id']), training_url=_url(c['training_run_id']),
            uniform_performance_url=_url(c['uniform_reference']['performance_run_id']),
            uniform_training_url=_url(c['uniform_reference']['training_run_id'])) for c in cells],
        references=aggregate_results(campaign, {})['references'], uncertainty=UNCERTAINTY)
    run = wandb.init(entity=ENTITY, project=PROJECT, id=campaign['overview_run_id'], resume='never', name=campaign['label'],
        group=campaign['group'], job_type='closed-loop-critic-LoRA-comparison', tags=['closed-loop', 'critic-lora', 'r16', 'r96', '575k', 'return-only', 'mean'],
        config=config, mode='online')
    run.define_metric('axis/inner_rounds')
    run.define_metric('critic_lora_sweep/*', step_metric='axis/inner_rounds')
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
            progress = dict(rows=statuses, failures=failures, evaluated=aggregate['evaluated'], new_evaluated=len(completed), published=published, reused=0, reused_dense_controls=18)
            run.summary.update(dict(status='running', evaluated=aggregate['evaluated'], new_evaluated=len(completed),
                published=published, reused=0, reused_dense_controls=18, total_settings=36, protocol_proof_by_setting=proofs))
            write(args.root / 'comparison-results.json', aggregate)
            write(args.root / 'progress.json', progress)
            print(f"Critic LoRA: evaluated {aggregate['evaluated']}/36; published {published}/36; 18 dense controls reused", flush=True)
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
                    if (Path(cell['directory']) / 'worker-completion.json').exists() and cell['name'] not in completed:
                        completed[cell['name']], proofs[cell['name']] = load_completed(campaign, cell)
                    if len(futures) < publishers and index not in attempted and cell['name'] in completed and not publication_complete(cell):
                        attempted.add(index); futures[index] = pool.submit(launch, index)
                statuses = update()
                if all(row['status'] == 'published' for row in statuses): break
                submission = args.root / 'submission.json'
                if submission.exists() and not futures:
                    if not gpu_jobs_active(read(submission)['gpu_job_ids']):
                        terminal_since = terminal_since or time.time()
                        if time.time() - terminal_since > 90: break
                    else: terminal_since = None
                time.sleep(15)
        statuses = update(terminal=True)
        complete = len(completed) == 36 and all(row['status'] == 'published' for row in statuses)
        status = 'complete' if complete else 'incomplete'
        write(args.root / 'campaign-completion.json', dict(status=status, rows=statuses, failures=failures,
            evaluated=len(completed), new_evaluated=len(completed), reused=0, reused_dense_controls=18,
            published=sum(row['status'] == 'published' for row in statuses)))
        run.summary.update(dict(status=status, failed_publications=len(failures)))
        if not complete: raise RuntimeError('Critic LoRA sweep incomplete; inspect status table and publisher logs')
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
