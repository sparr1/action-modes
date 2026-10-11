#!/usr/bin/env python3
"""Verified C32/C64 critic-dose publication with immutable C16 reference reuse."""
from __future__ import annotations
import argparse
from copy import deepcopy
import html
import math
from pathlib import Path
import re
import sys
import time
import uuid
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slurm import publish_ambi_critic_bypass as previous
from slurm.publish_ambi_action_audit import (
    AuditDataError as DoseDataError, SnapshotUncertain, exists, read,
    gpu_jobs_active, publisher_lock,
)
from utils.wandb_results_layout import _hash, _write
from utils.wandb_critic_dose_layout import (
    ARMS, METHODS, DoseLayoutConflict, DIAGNOSTICS_KEY, EPISODES_KEY,
    PAIRS_KEY, PROGRESS_KEY, SUMMARY_KEY, ensure_saved_view,
)

ENTITY, PROJECT = previous.ENTITY, previous.PROJECT
PROTOCOL = 'closed-loop-critic-dose-v1'
REFERENCE_SNAPSHOT_SHA256 = '0db43b26c71a7b946118679aea3a3d0eddbcd9ff68dde608104a1f12596e12e4'
REFERENCE_CAMPAIGN_ID = '8e726a66d2e61924c611d8207795e33801edffcdd08d657d80311c6cd021eb70'
NEW_ARMS = ('actor_mean_c32', 'learned_q_c32', 'actor_mean_c64', 'learned_q_c64')
REF_ARMS = dict(actor_mean='actor_mean_c16', learned_q='learned_q_c16', model_score='model_score_c16', prior='prior', mppi_h3='mppi_h3')
EXTRA_COLUMNS = ('method', 'critic_updates', 'source')
EPISODE_COLUMNS = (*previous.EPISODE_COLUMNS, *EXTRA_COLUMNS)
PROGRESS_COLUMNS = (*previous.PROGRESS_COLUMNS, *EXTRA_COLUMNS)
SUMMARY_COLUMNS = (*previous.SUMMARY_COLUMNS, *EXTRA_COLUMNS)
DIAGNOSTIC_METRICS = ('heldout_model_value', 'heldout_model_gain_vs_actor', 'heldout_model_gain_vs_prior',
                      'selection_model_value', 'learned_q', 'action_distance_vs_actor',
                      'bank_spearman_model', 'bank_model_regret_of_qpick', 'heldout_q_gain_vs_actor')
DIAGNOSTIC_COLUMNS = ('arm', 'label', 'env_seed', 'controller_seed', 'decision', 'candidate',
                      *EXTRA_COLUMNS, *DIAGNOSTIC_METRICS)
TABLES = {SUMMARY_KEY: ('summary', SUMMARY_COLUMNS), PAIRS_KEY: ('pairs', SUMMARY_COLUMNS),
          EPISODES_KEY: ('episodes', EPISODE_COLUMNS), PROGRESS_KEY: ('progress', PROGRESS_COLUMNS),
          DIAGNOSTICS_KEY: ('diagnostic_table', DIAGNOSTIC_COLUMNS)}
finite, estimate = previous.finite, previous.estimate


def arm_info(arm):
    if arm not in ARMS:
        raise DoseDataError('Unknown critic-dose arm.')
    if arm in ('prior', 'mppi_h3'):
        method, dose = (METHODS[3] if arm == 'prior' else METHODS[4]), None
    else:
        selector, dose = arm.rsplit('_c', 1)
        method = METHODS[('actor_mean', 'learned_q', 'model_score').index(selector)]
        dose = int(dose)
    return dict(arm=arm, label=method if dose is None else f'{method} C{dose}', method=method,
                critic_updates=dose, source='new' if arm in NEW_ARMS else 'reference')


def validate_campaign(campaign):
    if campaign.get('protocol') != PROTOCOL or not re.fullmatch(r'[0-9a-f]{64}', str(campaign.get('campaign_id', ''))):
        raise DoseDataError('Invalid critic-dose campaign identity.')
    if campaign.get('arms') != list(NEW_ARMS) or campaign.get('controller_seed') != 55:
        raise DoseDataError('Expected the four authorized C32/C64 arms with controller seed 55.')
    seeds = campaign.get('seeds')
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise DoseDataError('Unique integer environment seeds are required.')
    if not campaign.get('smoke') and (seeds != list(range(101, 121)) or campaign.get('max_steps') != 500):
        raise DoseDataError('Production requires all 20 seeds and full 500-decision episodes.')
    if campaign.get('reference_snapshot_sha256') != REFERENCE_SNAPSHOT_SHA256 or campaign.get('reference_campaign_id') != REFERENCE_CAMPAIGN_ID:
        raise DoseDataError('Campaign must pin the verified C16 reference identity.')
    expected = [dict(task_id=f'{arm}-seed-{seed}', arm=arm, env_seed=seed, controller_seed=55,
                     selector=arm.rsplit('_c', 1)[0], critic_updates_per_round=int(arm.rsplit('_c', 1)[1]))
                for arm in NEW_ARMS for seed in seeds]
    actual = campaign.get('task_list')
    expected_by_id = {task['task_id']: task for task in expected}
    if (not isinstance(actual, list) or len(actual) != len(expected) or
            len({task.get('task_id') for task in actual}) != len(expected) or
            any(task != expected_by_id.get(task.get('task_id')) for task in actual)):
        raise DoseDataError('Duplicate, conflicting or missing task in critic-dose campaign grid.')
    if (campaign.get('horizon'), campaign.get('rounds'), campaign.get('mc_rollouts'), campaign.get('validation_rollouts')) != (1, 6, 32, 32):
        raise DoseDataError('Critic dose must preserve H1/J6 and 32 selection/validation draws.')
    if (campaign.get('actor_updates_per_round'), campaign.get('rollouts_per_round'), campaign.get('batch_size')) != (4, 128, 256):
        raise DoseDataError('Critic dose must preserve A4/N128/B256.')
    decisions = campaign.get('diagnostic_decisions')
    if not isinstance(decisions, list) or not decisions or len(set(decisions)) != len(decisions) or any(type(d) is not int or not 0 <= d < campaign['max_steps'] for d in decisions):
        raise DoseDataError('Invalid required diagnostic decisions.')
    return actual


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='stable'); ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.
        start = end
    return ranks


def ranking_rows(record, arm):
    """Same selection-bank statistics for old/new doses; held-out gains separate."""
    rows = []
    for item in record.get('diagnostics', []):
        if item.get('bank_count') != 770 or item.get('replay_count') != 768 or item.get('mc_rollouts') != 32:
            raise DoseDataError('Diagnostic bank or draw count differs from H1/J6 protocol.')
        if item.get('selection_seed') == item.get('validation_seed') or any(type(item.get(k)) is not int for k in ('selection_seed', 'validation_seed')):
            raise DoseDataError('Selection and validation require distinct integer RNG seeds.')
        banks = []
        for field in ('selection_q_scores', 'selection_model_means'):
            values = item.get(field)
            if not isinstance(values, list) or len(values) != 770:
                raise DoseDataError('Missing complete candidate-bank diagnostic scores.')
            banks.append(np.asarray([finite(v, field) for v in values]))
        q, model = banks
        choices = item.get('choices', {})
        if set(choices) != {'actor_mean', 'learned_q', 'model_score', 'prior_mean'}:
            raise DoseDataError('All four counterfactual diagnostic choices are required.')
        expected_indices = dict(actor_mean=0, prior_mean=1, learned_q=int(q.argmax()), model_score=int(model.argmax()))
        for name, choice in choices.items():
            if choice.get('index') != expected_indices[name]:
                raise DoseDataError('Diagnostic chosen index disagrees with its declared selector.')
            action = choice.get('normalized_action')
            if not isinstance(action, list) or not action or any(abs(finite(v, 'normalized action')) > 1.000001 for v in action):
                raise DoseDataError('Invalid diagnostic normalized action.')
            for metric in ('selection_model', 'validation_model', 'validation_gain_vs_prior', 'validation_gain_vs_actor'):
                statistics = choice.get(metric, {})
                draws = statistics.get('draws')
                if not isinstance(draws, list) or len(draws) != 32:
                    raise DoseDataError('Every diagnostic estimate requires its exact 32 draws.')
                draws = [finite(v, 'diagnostic draw') for v in draws]
                if not math.isclose(finite(statistics.get('mean'), 'diagnostic mean'), float(np.mean(draws)), rel_tol=1e-6, abs_tol=1e-6):
                    raise DoseDataError('Diagnostic mean differs from its draws.')
                finite(statistics.get('se'), 'diagnostic SE', nonnegative=True)
            index = expected_indices[name]
            if not math.isclose(finite(choice.get('learned_q'), 'choice Q'), float(q[index]), rel_tol=1e-6, abs_tol=1e-6):
                raise DoseDataError('Chosen Q score disagrees with its candidate bank.')
            if not math.isclose(choice['selection_model']['mean'], float(model[index]), rel_tol=1e-6, abs_tol=1e-6):
                raise DoseDataError('Chosen model score disagrees with its candidate bank.')
        for choice in choices.values():
            for suffix, baseline in (('prior', 'prior_mean'), ('actor', 'actor_mean')):
                expected = np.asarray(choice['validation_model']['draws']) - np.asarray(choices[baseline]['validation_model']['draws'])
                if not np.allclose(choice[f'validation_gain_vs_{suffix}']['draws'], expected, rtol=1e-5, atol=1e-5):
                    raise DoseDataError('Held-out gain draws must pair the declared candidate and baseline.')
        qr, mr = average_ranks(q), average_ranks(model)
        rho = float(np.corrcoef(qr, mr)[0, 1]) if np.std(qr) > 0 and np.std(mr) > 0 else None
        values = dict(bank_spearman_model=rho,
                      bank_model_regret_of_qpick=float(model.max() - model[expected_indices['learned_q']]),
                      heldout_q_gain_vs_actor=choices['learned_q']['validation_gain_vs_actor']['mean'])
        for metric, value in values.items():
            rows.append({**arm_info(arm), 'env_seed': record['env_seed'], 'controller_seed': 55,
                         'decision': item['decision'], 'candidate': 'bank', 'metric': metric,
                         'value': value})
    return rows


def compact_diagnostics(rows):
    """Keep every scalar in <=4,200 rows; SDK media tables truncate at 10,000."""
    grouped = {}
    for row in rows:
        metric = row['metric']
        if metric not in DIAGNOSTIC_METRICS:
            raise DoseDataError('Unknown diagnostic scalar cannot be silently omitted.')
        key = tuple(row[k] for k in ('arm', 'env_seed', 'controller_seed', 'decision', 'candidate'))
        output = grouped.setdefault(key, {k: v for k, v in row.items() if k not in ('metric', 'value')})
        if metric in output:
            raise DoseDataError('Duplicate diagnostic scalar in compact publication.')
        output[metric] = row['value']
    result = list(grouped.values())
    if len(result) > 10000:
        raise DoseDataError('Compact diagnostic table exceeds the SDK media row limit.')
    return result


def normalize_result(record, task, campaign):
    if record.get('protocol') != PROTOCOL or any(record.get(k) != v for k, v in task.items()):
        raise DoseDataError('Critic-dose result identity conflicts with task.')
    required_checks = {'outer_state_unchanged', 'full_episode', 'finite_returns', 'action_bounds',
                       'complete_candidate_banks', 'independent_selection_validation',
                       'critic_update_dose_verified', 'diagnostic_coverage'}
    if not required_checks <= record.get('checks', {}).keys() or any(record['checks'][key] is not True for key in required_checks):
        raise DoseDataError('Required critic-dose scientific integrity checks missing or failed.')
    expected_work = dict(actor_updates=24, critic_updates=6 * task['critic_updates_per_round'], model_steps=768)
    if record.get('expected_work_per_decision') != expected_work or record.get('realized_work_totals') != {k: v * record['episode_length'] for k, v in expected_work.items()}:
        raise DoseDataError('Realized critic/actor/model work differs from authorized dose.')
    runtime = record.get('runtime', {})
    if runtime.get('execution') != 'eager' or 'A5000' not in str(runtime.get('gpu', '')):
        raise DoseDataError('Comparable timings require eager A5000 execution.')
    # Reuse the established value/timing/coverage validator through an explicit
    # local projection. The original identity remains unchanged and fingerprinted.
    adapted_record = deepcopy(record)
    adapted_record.update(protocol=previous.PROTOCOL, arm=task['selector'], task_id=f'{task["selector"]}-seed-{task["env_seed"]}')
    adapted_task = {k: adapted_record[k] for k in ('task_id', 'arm', 'env_seed', 'controller_seed')}
    episode, timings, diagnostics = previous.normalize_result(adapted_record, adapted_task, campaign)
    info = arm_info(task['arm'])
    episode.update(**info, task_id=task['task_id'])
    for row in diagnostics: row.update(info)
    diagnostics.extend(ranking_rows(record, task['arm']))
    return episode, timings, diagnostics


def load_references(snapshot_path, root):
    snapshot = read(snapshot_path)
    if snapshot.get('snapshot_sha256') != REFERENCE_SNAPSHOT_SHA256 or _hash({k: v for k, v in snapshot.items() if k != 'snapshot_sha256'}) != REFERENCE_SNAPSHOT_SHA256:
        raise DoseDataError('Reference snapshot content/hash is not the pinned completed campaign.')
    if snapshot.get('campaign_id') != REFERENCE_CAMPAIGN_ID or snapshot.get('complete') != 100 or snapshot.get('state') != 'complete':
        raise DoseDataError('Reference campaign must be the pinned complete 100-episode result.')
    old_campaign = read(root / 'campaign.json')
    verified = previous.collect(root, old_campaign, workers_active=False)
    if verified != snapshot:
        raise DoseDataError('Raw reference results do not exactly reproduce the pinned snapshot.')
    episodes, summaries, diagnostics = [], [], []
    for row in snapshot['episodes']:
        info = arm_info(REF_ARMS[row['arm']])
        episodes.append({**row, **info, 'task_id': f'{info["arm"]}-seed-{row["env_seed"]}'})
        raw = read(root / 'tasks' / row['task_id'] / 'result.json')
        if not raw.get('reused_reference') and (raw.get('runtime', {}).get('execution') != 'eager' or 'A5000' not in str(raw.get('runtime', {}).get('gpu'))):
            raise DoseDataError('Reference timing must be eager A5000 or explicitly unavailable.')
        diagnostics.extend(ranking_rows(raw, info['arm']))
    for row in snapshot['summary']:
        summaries.append({**row, **arm_info(REF_ARMS[row['arm']])})
    for row in snapshot['diagnostics']:
        diagnostics.append({**row, **arm_info(REF_ARMS[row['arm']])})
    return dict(snapshot_sha256=REFERENCE_SNAPSHOT_SHA256, episodes=episodes, summary=summaries,
                diagnostics=diagnostics, result_fingerprints=[{**r, 'task_id': 'reference/' + r['task_id']} for r in snapshot['result_fingerprints']])


def summarize(episodes, timings, diagnostics, campaign, reference):
    summary = deepcopy(reference['summary'])
    grouped = {arm: {r['env_seed']: r for r in episodes if r['arm'] == arm} for arm in ARMS}
    complete = {arm: rows for arm, rows in grouped.items() if set(rows) == set(campaign['seeds'])}
    seeds = sorted(campaign['seeds'])
    for arm in NEW_ARMS:
        if arm not in complete: continue
        rows, ident = complete[arm], {**arm_info(arm), 'state': 'complete', 'baseline': 'none'}
        for metric in ('episode_return', 'controller_time_mean_s', 'selection_time_mean_s'):
            summary.append({**ident, 'metric': metric, **estimate([rows[s][metric] for s in seeds]), 'estimator': 'equal environment-seed mean; percentile bootstrap'})
        for stem in ('controller', 'selection'):
            pooled = [v for s in seeds for v in timings[f'{arm}-seed-{s}'][stem]]
            summary.append({**ident, 'metric': stem + '_time_p95_s', 'n': len(seeds), 'mean': float(np.percentile(pooled, 95)),
                            'ci_low': None, 'ci_high': None, 'sd': None, 'estimator': 'pooled decision p95; no interval'})
        summary.append({**ident, 'metric': 'episode_return_sd', 'n': len(seeds), 'mean': float(np.std([rows[s]['episode_return'] for s in seeds], ddof=1)) if len(seeds) > 1 else None,
                        'ci_low': None, 'ci_high': None, 'sd': None, 'estimator': 'environment-seed sample standard deviation'})
    # Root-level diagnostics are not independent samples: average within seed.
    for arm in complete:
        for metric in ('bank_spearman_model', 'bank_model_regret_of_qpick', 'heldout_q_gain_vs_actor'):
            raw_by_seed = {s: [r['value'] for r in diagnostics if r['arm'] == arm and r['env_seed'] == s and r['metric'] == metric] for s in seeds}
            if not any(raw_by_seed.values()): continue
            if not all(raw_by_seed.values()):
                raise DoseDataError('Some completed seeds lack finite diagnostic estimates.')
            by_seed = {s: [v for v in raw_by_seed[s] if v is not None] for s in seeds}
            values = [float(np.mean(by_seed[s])) for s in seeds if by_seed[s]]
            if metric != 'bank_spearman_model' and len(values) != len(seeds):
                raise DoseDataError('Missing finite diagnostic estimates.')
            statistic = estimate(values) if values else dict(n=0, mean=None, ci_low=None, ci_high=None, sd=None)
            summary.append({**arm_info(arm), 'state': 'complete', 'baseline': 'none', 'metric': metric,
                **statistic, 'estimator': 'mean defined roots within environment seed; defined seeds only; percentile bootstrap' if metric == 'bank_spearman_model' else 'mean roots within environment seed; percentile bootstrap'})
            if metric == 'bank_spearman_model':
                summary.append({**arm_info(arm), 'state': 'complete', 'baseline': 'none', 'metric': 'bank_spearman_defined_root_fraction',
                    **estimate([len(by_seed[s]) / len(raw_by_seed[s]) for s in seeds]),
                    'estimator': 'fraction of roots with nonconstant Q and model scores; all environment seeds'})
    pairs = []
    for arm, rows in complete.items():
        targets = {'prior': 'prior'} if arm != 'prior' else {}
        selector = arm.rsplit('_c', 1)[0]
        if selector in ('actor_mean', 'learned_q'):
            targets['matching_c16'] = selector + '_c16'
        for baseline, other in targets.items():
            if other not in complete: continue
            differences = [rows[s]['episode_return'] - complete[other][s]['episode_return'] for s in seeds]
            pairs.append({**arm_info(arm), 'baseline': baseline, 'state': 'complete', 'metric': 'episode_return', **estimate(differences),
                'win_rate': float(np.mean(np.asarray(differences) > 0)), 'tie_rate': float(np.mean(np.asarray(differences) == 0)),
                'estimator': 'paired environment-seed difference; percentile bootstrap'})
    return summary, pairs


def collect(root, campaign, reference, *, workers_active=True):
    tasks = validate_campaign(campaign)
    expected = {t['task_id'] for t in tasks}
    for path in (root / 'tasks').glob('*/result.json'):
        if path.parent.name not in expected:
            raise DoseDataError('Unexpected or duplicate critic-dose result directory.')
    episodes, diagnostics = deepcopy(reference['episodes']), deepcopy(reference['diagnostics'])
    fingerprints, progress, timings = deepcopy(reference['result_fingerprints']), [], {}
    for task in tasks:
        directory = root / 'tasks' / task['task_id']; path = directory / 'result.json'
        failure = next((read(directory / name) for name in ('failure.json', 'failed.json', 'error.json') if exists(directory / name)), None)
        current = read(directory / 'progress.json') if exists(directory / 'progress.json') else {}
        if current.get('status') == 'failed': failure = current
        identity = {**task, **arm_info(task['arm'])}
        if exists(path):
            if failure is not None: raise DoseDataError('Completed result conflicts with failure marker.')
            record = read(path)
            episode, timing, ds = normalize_result(record, task, campaign)
            episodes.append(episode); diagnostics.extend(ds); timings[task['task_id']] = timing
            fingerprints.append(dict(task_id=task['task_id'], sha256=_hash(record)))
            progress.append({**identity, 'state': 'complete', 'decisions': episode['episode_length'], 'check_state': 'pass', 'message': ''})
        else:
            progress.append({**identity, 'state': 'failed' if failure is not None else 'pending',
                'decisions': current.get('decision', 0), 'check_state': 'fail' if failure is not None else 'pending',
                'message': str(failure)[:500] if failure is not None else 'Waiting for complete verified episode.' if workers_active else 'Workers stopped before result was available.'})
    summary, pairs = summarize(episodes, timings, diagnostics, campaign, reference)
    counts = {k: sum(r['state'] == k for r in progress) for k in ('complete', 'pending', 'failed')}
    snapshot = dict(schema_version=1, protocol=PROTOCOL, campaign_id=campaign['campaign_id'],
        reference_snapshot_sha256=reference['snapshot_sha256'], reference_episodes=len(reference['episodes']),
        total=len(tasks), **counts, state='complete' if counts['complete'] == len(tasks) else 'running' if workers_active else 'failed',
        episodes=episodes, progress=progress, summary=summary, pairs=pairs, diagnostics=diagnostics,
        diagnostic_table=compact_diagnostics(diagnostics), result_fingerprints=fingerprints)
    snapshot['snapshot_sha256'] = _hash(snapshot)
    return snapshot


def verify_immutable_results(previous, current):
    before = {r['task_id']: r['sha256'] for r in previous.get('result_fingerprints', [])}
    after = {r['task_id']: r['sha256'] for r in current['result_fingerprints']}
    if any(after.get(task) != sha for task, sha in before.items()):
        raise DoseDataError('A previously published scientific result changed or disappeared.')


def write_report(out, snapshot):
    if exists(out / 'snapshot.json'):
        verify_immutable_results(read(out / 'snapshot.json'), snapshot)
    _write(out / 'snapshot.json', snapshot)
    document = '<!doctype html><meta charset="utf-8"><title>800k critic update dose</title><style>body{font:14px system-ui;margin:2rem}div{overflow:auto}td,th{padding:.4rem;border-bottom:1px solid #ddd;text-align:left}th{background:#eee}</style>'
    document += f'<h1>800k full-episode critic update dose</h1><p>{snapshot["complete"]}/{snapshot["total"]} complete; {snapshot["failed"]} failed; {snapshot["pending"]} pending.</p>'
    document += '<p>Final arm summaries and paired intervals require all requested seeds. Return is undiscounted environment reward; model diagnostics are separate. Timing excludes diagnostic work. Raw records remain on Oscar.</p>'
    for _, (name, columns) in TABLES.items():
        document += '<h2>' + html.escape(name.title()) + '</h2><div><table><thead><tr>' + ''.join('<th>' + html.escape(c) + '</th>' for c in columns) + '</tr></thead><tbody>'
        document += ''.join('<tr>' + ''.join('<td>' + html.escape('' if row.get(c) is None else str(row[c])) + '</td>' for c in columns) + '</tr>' for row in snapshot[name])
        document += '</tbody></table></div>'
    temporary = out / 'report.html.tmp'; temporary.write_text(document); temporary.replace(out / 'report.html')


def publication_state(root, out, campaign, entity, project):
    expected = dict(schema_version=1, campaign_root=str(root), campaign_sha256=_hash(campaign), entity=entity, project=project)
    path = out / 'publication.json'
    if exists(path):
        state = read(path)
        if any(state.get(k) != v for k, v in expected.items()):
            raise DoseDataError('Prepared publication identity changed; refusing to reuse the run.')
        return state
    state = {**expected, 'run_id': uuid.uuid4().hex}; _write(path, state)
    return state


def acknowledged(api, state, snapshot):
    summary = api.run(f'{state["entity"]}/{state["project"]}/{state["run_id"]}').summary
    if summary.get('critic_dose/snapshot_sha256') != snapshot['snapshot_sha256']:
        return False
    for key, (name, columns) in TABLES.items():
        # The public SDK wraps nested summary values in SummarySubDict rather
        # than dict. Use its mapping interface, as for the outer summary.
        value = summary.get(key)
        if value is None or value.get('nrows') != len(snapshot[name]) or value.get('ncols') != len(columns):
            return False
    return True


def publish_once(wandb, state, campaign, snapshot, out, *, acknowledgement_seconds=30):
    try:
        already = acknowledged(wandb.Api(timeout=30), state, snapshot)
    except Exception:
        already = False
    if not already:
        run = wandb.init(entity=state['entity'], project=state['project'], id=state['run_id'], resume='allow', mode='online',
            name='800k critic updates | C16 C32 C64 | H1 J6 | 20 seeds', group='critic-dose-800k-' + state['run_id'][:8], job_type='critic-dose-overview',
            config={'critic_dose_publication': state['run_id'], 'campaign_sha256': state['campaign_sha256'],
                    'campaign_id': campaign['campaign_id'], 'reference_snapshot_sha256': REFERENCE_SNAPSHOT_SHA256, 'checkpoint': campaign.get('checkpoint'), 'protocol': PROTOCOL,
                    'publication_semantics': 'replaceable compact snapshot; final summaries require complete seed coverage'})
        try:
            payload = {}
            for key, (name, columns) in TABLES.items():
                table = wandb.Table(columns=list(columns), data=[[r.get(c) for c in columns] for r in snapshot[name]])
                # SDK summary metadata can claim all rows even when media JSON
                # silently truncates. Verify the actual serialization pre-upload.
                serialized = table._to_table_json(warn=False)
                if len(serialized['data']) != len(snapshot[name]) or serialized['columns'] != list(columns):
                    raise DoseDataError('SDK compact-table serialization truncated or changed data: ' + key)
                payload[key] = table
            payload.update({'critic_dose/snapshot_sha256': snapshot['snapshot_sha256'],
                            **{'critic_dose/' + k: snapshot[k] for k in ('complete', 'total', 'failed', 'pending')},
                            'critic_dose/status': snapshot['state']})
            run.log(payload)
        finally:
            run.finish()
        deadline = time.monotonic() + acknowledgement_seconds
        while True:
            try:
                if acknowledged(wandb.Api(timeout=30), state, snapshot):
                    break
            except Exception:
                pass
            if time.monotonic() >= deadline:
                raise SnapshotUncertain('Compact snapshot not yet visible; safe to replace under the same run ID.')
            time.sleep(min(5., max(0., deadline - time.monotonic())))
    layout = ensure_saved_view(wandb.Api(timeout=30), entity=state['entity'], project=state['project'],
                               publication_id=state['run_id'], receipt_dir=out / 'results-layout')
    _write(out / 'published.json', dict(snapshot_sha256=snapshot['snapshot_sha256'], run_id=state['run_id'],
                                      completed=snapshot['complete'], total=snapshot['total'], layout=layout))
    return layout


def publish_with_retry(operation, out, *, delays=(5., 15., 30., 60., 60.)):
    for attempt in range(len(delays) + 1):
        try:
            result = operation()
            _write(out / 'publisher-retry.json', dict(status='healthy', attempts=attempt + 1))
            return result
        except (DoseDataError, DoseLayoutConflict, KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            status = getattr(exc, 'status_code', None) or getattr(getattr(exc, 'response', None), 'status_code', None)
            terminal = status in (400, 401, 403, 404) or attempt == len(delays)
            _write(out / 'publisher-retry.json', dict(status='failed' if terminal else 'retrying', attempt=attempt + 1,
                error_type=type(exc).__name__, error=str(exc), delay_seconds=None if terminal else delays[attempt]))
            if terminal:
                raise
            time.sleep(delays[attempt])


def watch(args):
    import wandb
    root, out = args.root.resolve(), args.publication_root.resolve()
    if root == out or out.is_relative_to(root) or root.is_relative_to(out):
        raise DoseDataError('Publication state must be outside the scientific campaign directory.')
    with publisher_lock(out):
        campaign = read(root / 'campaign.json'); validate_campaign(campaign)
        if campaign.get('smoke'):
            raise DoseDataError('Smoke results must not be published as the production comparison.')
        reference = load_references(args.reference_snapshot.resolve(), args.reference_root.resolve())
        state = publication_state(root, out, campaign, args.entity, args.project)
        previous, inactive_since = None, None
        while True:
            try:
                active = gpu_jobs_active(args.gpu_job_id)
                inactive_since = None if active else inactive_since or time.monotonic()
                still_waiting = active or time.monotonic() - inactive_since < args.worker_grace_seconds
                snapshot = collect(root, campaign, reference, workers_active=still_waiting)
                write_report(out, snapshot)
                if snapshot['snapshot_sha256'] != previous:
                    result = publish_with_retry(lambda: publish_once(wandb, state, campaign, snapshot, out), out)
                    previous = snapshot['snapshot_sha256']
                    print(f'{snapshot["complete"]}/{snapshot["total"]} episodes complete; {result["url"]}', flush=True)
                _write(out / 'publisher-status.json', dict(status=snapshot['state'], completed=snapshot['complete'],
                    total=snapshot['total'], failed=snapshot['failed'], pending=snapshot['pending'], run_id=state['run_id']))
                if snapshot['state'] == 'complete' or args.once:
                    return
                if not still_waiting:
                    raise DoseDataError('Workers stopped before all requested episodes passed integrity checks.')
                time.sleep(args.poll_seconds)
            except BaseException as exc:
                _write(out / 'publisher-failure.json', dict(error_type=type(exc).__name__, error=str(exc), run_id=state['run_id']))
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--publication-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--reference-snapshot', type=Path, required=True)
    parser.add_argument('--entity', default=ENTITY)
    parser.add_argument('--project', default=PROJECT)
    parser.add_argument('--gpu-job-id', action='append', default=[])
    parser.add_argument('--poll-seconds', type=float, default=60.)
    parser.add_argument('--worker-grace-seconds', type=float, default=120.)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or args.worker_grace_seconds < 0:
        parser.error('Poll interval must be positive and grace interval nonnegative.')
    if not args.gpu_job_id and not args.once:
        parser.error('Watch mode requires at least one --gpu-job-id.')
    watch(args)


if __name__ == '__main__':
    main()
