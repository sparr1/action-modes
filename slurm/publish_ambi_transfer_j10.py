#!/usr/bin/env python3
"""Publish one 800k J10 cell against immutable J4/J6/J8 Bernoulli references.

Only compact scalar tables are uploaded. The old publications and raw artifacts
are read-only; a separate owned view is installed and verified on every resume.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_transfer_checkpoint_publish import read, write, fingerprint
from slurm.ambi_transfer_sweep_publish import publisher_lock
from slurm.ambi_closed_loop_publish import gpu_jobs_active
from utils.eval_series import SeriesError, load_run, validate_record, _record_fingerprint
from utils.transfer_checkpoint_publication import (
    PROTOCOL, digest, require, identity, settings, _episodes, normalize_transfer,
)
from utils.wandb_transfer_j10_layout import (
    SUMMARY_KEY, PAIRS_KEY, EPISODES_KEY, PROGRESS_KEY, DIAGNOSTICS_KEY,
    J10LayoutConflict, ensure_saved_view,
)

ENTITY = 'rwgao_b-brown-university'
PROJECT = 'ambi-inner-bench'
SETTING = 'h1_j10_bernoulli_a0_c05'
SUMMARY_COLUMNS = ['J', 'method', 'label', 'metric', 'baseline', 'state', 'n',
                   'mean', 'sd', 'lower', 'upper', 'min', 'max', 'win_rate']
EPISODE_COLUMNS = ['J', 'method', 'seed', 'return', 'length', 'gain_prior', 'gain_J8',
                   'controller_time_s', 'steady_controller_time_s']
PROGRESS_COLUMNS = ['J', 'state', 'completed_episodes', 'total_episodes', 'seed', 'decision', 'error']
DIAGNOSTIC_COLUMNS = ['J', 'seed', 'metric', 'value']
TABLES = {SUMMARY_KEY: ('summary', SUMMARY_COLUMNS), PAIRS_KEY: ('pairs', SUMMARY_COLUMNS),
          EPISODES_KEY: ('episodes', EPISODE_COLUMNS), PROGRESS_KEY: ('progress', PROGRESS_COLUMNS),
          DIAGNOSTICS_KEY: ('diagnostics', DIAGNOSTIC_COLUMNS)}


def validate_campaign(campaign):
    require(campaign.get('protocol') == PROTOCOL and campaign.get('selection_id') == 'h1-j10-800k-bernoulli-v1',
            'Expected the explicit single-checkpoint J10 campaign.')
    require(campaign.get('seeds') == [101, 102, 103, 104, 105]
            and campaign.get('controller_seed') == 55 and campaign.get('max_steps') == 500
            and campaign.get('gpu_hardware') == 'L40S', 'J10 episode/timing protocol differs.')
    require(len(campaign.get('checkpoints', [])) == 1 and campaign['checkpoints'][0]['step'] == 800000,
            'J10 publication requires exactly checkpoint 800k.')
    require(len(campaign.get('candidates', [])) == len(campaign.get('cells', [])) == 1,
            'J10 publication requires exactly one cell.')
    setting, cell = campaign['candidates'][0], campaign['cells'][0]
    expected = dict(setting_id=SETTING, H=1, J=10, arm='bernoulli_a0_c05')
    require(all(setting.get(k) == v and cell.get(k) == v for k, v in expected.items())
            and cell.get('step') == 800000 and cell.get('checkpoint_index') == cell.get('index') == 0,
            'Wrong J10 transfer setting.')
    definition = setting.get('arm_definition', {})
    require(definition.get('critic_bernoulli_p') == .5 and definition.get('parameter_scope') == 'all'
            and definition.get('actor_bernoulli_p', 0) == 0 and cell.get('arm_definition') == definition,
            'Expected critic-only Bernoulli 50% copying of all parameters.')


def _validate_episodes(record, campaign, *, prior=False):
    require(len(record['episodes']) == 5 and _episodes(record['episodes'], campaign) == record['episodes'],
            'Reference must contain exactly the ordered five full episodes.')
    require(record['metrics'].get('eval/episodes') == 5
            and record['metrics'].get('eval/frozen_state_unchanged') is True, 'Missing frozen-state/episode proof.')
    for suffix, value in (('mean', statistics.mean(e['return'] for e in record['episodes'])),
                          ('sample_std', statistics.stdev(e['return'] for e in record['episodes']))):
        require(math.isclose(record['metrics']['eval/return_' + suffix], value, rel_tol=1e-10, abs_tol=1e-8),
                'Published return summary differs from episodes.')
    if prior:
        return
    from utils.transfer_campaign_diagnostics import verify_episode_diagnostics
    for episode in record['episodes']:
        verify_episode_diagnostics(episode, campaign['diagnostics'], smoke=False)
        values = episode['diagnostics'].get('summary', {})
        require(values and all(type(v) in (int, float) and math.isfinite(v) for v in values.values()),
                'Reference diagnostic scalar summaries missing or nonfinite.')
        require({'final_critic_rmse', 'final_critic_selection_regret'} <= values.keys(),
                'Required critic diagnostics missing.')
        for key in ('control_seconds_per_decision', 'late_control_seconds_per_decision'):
            require(type(episode.get(key)) in (int, float) and math.isfinite(episode[key]) and episode[key] > 0,
                    'Reference controller timing missing or invalid.')
    receipt = record['provenance'].get('worker_receipt', {})
    require('L40S' in receipt.get('hardware', '') and receipt.get('frozen_outer_verified') is True,
            'Reference timing hardware/frozen-state proof differs.')
    params = record['provenance']['resolved_config']['algorithm_config']['alg_params']
    expected = {'compile': True, 'inner_critic_updates_per_round': 16, 'inner_actor_updates_per_round': 4,
                'inner_rollouts_per_round': 128, 'inner_batch_size': 256, 'inner_rollout_horizon': 1}
    require(all(params.get(k) == v for k, v in expected.items()), 'Reference C/A/N/B or compile protocol differs.')


def _accepted_record(entry, previous, setting, campaign, pins, *, prior=False):
    directory = Path(entry['run_dir'])
    registry = load_run(directory)
    require(registry['run_id'] == entry['run_id'] and registry['entity'] == ENTITY
            and registry['project'] == PROJECT and registry['identity'] == identity(previous, setting),
            'Reference registry identity differs.')
    journal = read(directory / 'publication.json')
    selected = [(rid, row) for rid, row in journal['records'].items() if row['checkpoint_step'] == 800000]
    require(len(selected) == 1, 'Expected exactly one accepted 800k record; duplicate/missing checkpoint.')
    rid, accepted = selected[0]
    require(accepted.get('status') == 'published', 'Reference record is not published.')
    path = directory / 'records' / (rid + '.json')
    record = validate_record(read(path), registry['identity'])
    checkpoint = campaign['checkpoints'][0]
    require(record['record_id'] == rid and record['checkpoint'] == dict(step=800000, sha256=checkpoint['checkpoint_sha256'])
            and _record_fingerprint(record, accepted['artifact_sha256']) == accepted['record_sha256'],
            'Reference content/checkpoint differs from its immutable accepted fingerprint.')
    _validate_episodes(record, campaign, prior=prior)
    for file in (directory / 'run.json', directory / 'publication.json', path):
        pins[str(file.resolve())] = digest(file)
    return record


def load_references(campaign, publication_roots):
    """Import accepted records, never re-log or rewrite their scientific runs."""
    validate_campaign(campaign)
    require(len(publication_roots) == 3 and len({str(Path(p).resolve()) for p in publication_roots}) == 3,
            'Provide three distinct original J4, J6 and J8 publication roots.')
    records, pins = {}, {}
    for root in map(Path, publication_roots):
        path = root / 'publication.json'; state = read(path)
        old_path = Path(state['campaign_root']) / 'campaign.json'; previous = read(old_path)
        require(state.get('prepared') is True and state.get('entity') == ENTITY and state.get('project') == PROJECT
                and state['campaign_sha256'] == digest(old_path), 'Reference publication binding changed.')
        require(all(campaign.get(k) == previous.get(k) for k in
                    ('protocol', 'source_run', 'seeds', 'controller_seed', 'max_steps', 'scientific_source', 'diagnostics', 'gpu_hardware')),
                'Reference scientific implementation or episode protocol differs.')
        checkpoint = [cp for cp in previous['checkpoints'] if cp['step'] == 800000]
        require(len(checkpoint) == 1 and all(checkpoint[0].get(k) == campaign['checkpoints'][0].get(k)
                for k in ('checkpoint_sha256', 'metadata_sha256', 'prior_pin')), 'Reference checkpoint or prior pin differs.')
        matches = [s for s in previous['candidates'] if s.get('H') == 1 and s.get('J') in (4, 6, 8)
                   and s.get('arm') == 'bernoulli_a0_c05']
        require(len(matches) == 1, 'Reference root must identify exactly one requested Bernoulli method.')
        setting = matches[0]; j = setting['J']
        require(j not in records and setting['arm_definition'] == campaign['candidates'][0]['arm_definition'],
                'Duplicate reference J or changed transfer definition.')
        records[j] = _accepted_record(state['runs'][setting['setting_id']], previous, setting, campaign, pins)
        # The reused prior registry can retain the original campaign identity.
        if j == 4:
            records['prior'] = _accepted_record(state['runs']['prior'], previous, settings(previous)[0], campaign, pins, prior=True)
        for file in (path, old_path):
            pins[str(file.resolve())] = digest(file)
    require(set(records) == {4, 6, 8, 'prior'}, 'Reference J4/J6/J8/prior coverage differs.')
    return records, pins


def summary_row(j, metric, values, *, baseline='none', prior=False, interval=True):
    mean, sd = statistics.mean(values), statistics.stdev(values)
    method = 'Frozen prior' if prior else 'Bernoulli 50% critic'
    return dict(J=j, method=method, label='Frozen prior' if prior else f'Bernoulli critic 50% · J{j}',
        metric=metric, baseline=baseline, state='complete', n=len(values), mean=mean, sd=sd,
        lower=mean-sd if interval else None, upper=mean+sd if interval else None,
        min=min(values), max=max(values), win_rate=sum(v > 0 for v in values)/len(values) if baseline != 'none' else None)


def summarize(records, progress):
    summary, pairs, episodes, diagnostics = [], [], [], []
    prior_returns = {e['seed']: e['return'] for e in records['prior']['episodes']}
    j8_returns = {e['seed']: e['return'] for e in records[8]['episodes']}
    for key in ('prior', 4, 6, 8, 10):
        if key not in records:
            continue
        record = records[key]; j = None if key == 'prior' else key
        values = record['episodes']; is_prior = j is None
        summary.append(summary_row(j, 'episode_return', [e['return'] for e in values], prior=is_prior))
        variance = summary_row(j, 'episode_return_sd', [e['return'] for e in values], prior=is_prior, interval=False)
        variance.update(mean=variance['sd'], sd=None, min=None, max=None); summary.append(variance)
        for baseline, lookup in (('prior', prior_returns), ('J8', j8_returns)):
            if not is_prior:
                pairs.append(summary_row(j, 'episode_return', [e['return'] - lookup[e['seed']] for e in values], baseline=baseline))
        if not is_prior:
            for metric, field in (('controller_time_s', 'control_seconds_per_decision'),
                                  ('steady_controller_time_s', 'late_control_seconds_per_decision')):
                summary.append(summary_row(j, metric, [e[field] for e in values]))
            metric_sets = [set(e['diagnostics']['summary']) for e in values]
            require(all(m == metric_sets[0] for m in metric_sets), 'Diagnostic summary metric coverage differs across seeds.')
            for metric in sorted(metric_sets[0]):
                summary.append(summary_row(j, metric, [e['diagnostics']['summary'][metric] for e in values]))
                diagnostics.extend(dict(J=j, seed=e['seed'], metric=metric, value=e['diagnostics']['summary'][metric]) for e in values)
        episodes.extend(dict(J=j, method='Frozen prior' if is_prior else 'Bernoulli 50% critic', seed=e['seed'],
            **{'return': e['return']}, length=e['length'], gain_prior=e['return']-prior_returns[e['seed']],
            gain_J8=e['return']-j8_returns[e['seed']], controller_time_s=e.get('control_seconds_per_decision'),
            steady_controller_time_s=e.get('late_control_seconds_per_decision')) for e in values)
    complete = 5 if 10 in records else progress['completed_episodes']
    result = dict(summary=summary, pairs=pairs, episodes=episodes, diagnostics=diagnostics, progress=[progress],
                  completed=complete, total=5, state=progress['state'], validated_complete=10 in records)
    result['snapshot_sha256'] = fingerprint(result)
    return result


def collect(root, campaign, references, *, active=True):
    from slurm.ambi_transfer_curve_campaign import validate_receipt, validate_result, receipt_path
    records = dict(references); cell = campaign['cells'][0]; directory = Path(cell['result_dir'])
    marker = receipt_path(root, cell)
    progress = dict(J=10, state='pending' if active else 'incomplete', completed_episodes=0,
                    total_episodes=5, seed=None, decision=None, error=None)
    if marker.exists():
        receipt = validate_receipt(root, campaign, cell)
        result, manifest = validate_result(directory, campaign, cell)
        record = normalize_transfer(campaign, cell, receipt, result, manifest, references['prior'], receipt_path=marker)
        _validate_episodes(record, campaign)
        records[10] = record
        progress.update(state='complete', completed_episodes=5)
    else:
        if (directory / 'progress.json').exists():
            status = read(directory / 'progress.json')
            require(status.get('cell_id') == SETTING and status.get('smoke') is False
                    and status.get('total_episodes') == 5, 'Progress belongs to a different cell or smoke run.')
            count = status.get('completed_episodes', 0)
            require(type(count) is int and 0 <= count <= 5, 'Invalid partial episode progress.')
            progress.update(completed_episodes=count, seed=status.get('seed'), decision=status.get('decision'),
                state='awaiting_receipt' if status.get('status') == 'complete' else status.get('status', progress['state']),
                error=status.get('error'))
        failure = Path(root) / 'failures' / '0.json'
        if failure.exists():
            progress.update(state='failed', error=json.dumps(read(failure), sort_keys=True))
        elif not active:
            progress.update(state='incomplete', error=progress['error'] or 'GPU work ended without a validated five-seed receipt.')
    return summarize(records, progress)


def publication_state(root, out, campaign, pins):
    expected = dict(schema_version=1, campaign_root=str(root), campaign_sha256=fingerprint(campaign),
                    entity=ENTITY, project=PROJECT, reference_files=pins)
    path = out / 'publication.json'
    if path.exists():
        state = read(path)
        require(all(state.get(k) == v for k, v in expected.items()), 'Publication or reference identity changed.')
        return state
    state = dict(expected, run_id=uuid.uuid4().hex); write(path, state)
    return state


def acknowledged(api, state, snapshot):
    summary = api.run(f'{ENTITY}/{PROJECT}/{state["run_id"]}').summary
    if summary.get('transfer_j10/snapshot_sha256') != snapshot['snapshot_sha256']:
        return False
    for key, (name, columns) in TABLES.items():
        value = summary.get(key)
        # SDK SummarySubDict provides .get but is not a dict.
        if value is None or value.get('nrows') != len(snapshot[name]) or value.get('ncols') != len(columns):
            return False
    return True


def table_payload(wandb, snapshot):
    result = {}
    for key, (name, columns) in TABLES.items():
        table = wandb.Table(columns=columns, data=[[row.get(c) for c in columns] for row in snapshot[name]])
        serialized = table._to_table_json(warn=False)
        require(serialized['columns'] == columns and len(serialized['data']) == len(snapshot[name]),
                'W&B media table would truncate or change columns; publication stopped.')
        result[key] = table
    return result


def publish_once(wandb, state, campaign, snapshot, out, *, acknowledgement_seconds=30):
    try:
        already = acknowledged(wandb.Api(timeout=30), state, snapshot)
    except Exception:
        already = False
    if not already:
        run = wandb.init(entity=ENTITY, project=PROJECT, id=state['run_id'], resume='allow', mode='online',
            name='800k Bernoulli critic 50% | J4,6,8,10', group='transfer-j10-800k', job_type='transfer-j10-overview',
            config=dict(transfer_j10_publication=state['run_id'], campaign_sha256=state['campaign_sha256'],
                        protocol=PROTOCOL, checkpoint=campaign['checkpoints'][0]['checkpoint_sha256'],
                        reference_files_sha256=fingerprint(state['reference_files'])))
        try:
            payload = table_payload(wandb, snapshot)
            payload.update({'transfer_j10/snapshot_sha256': snapshot['snapshot_sha256'],
                'transfer_j10/completed_episodes': snapshot['completed'], 'transfer_j10/total_episodes': 5,
                'transfer_j10/validated_complete': snapshot['validated_complete'], 'transfer_j10/status': snapshot['state']})
            run.log(payload)
        finally:
            run.finish()
        deadline = time.monotonic() + acknowledgement_seconds
        while not acknowledged(wandb.Api(timeout=30), state, snapshot):
            if time.monotonic() >= deadline:
                raise RuntimeError('Snapshot acknowledgement not visible; retry safely under the same run ID.')
            time.sleep(min(5., max(0., deadline-time.monotonic())))
    layout = ensure_saved_view(wandb.Api(timeout=30), entity=ENTITY, project=PROJECT,
                               publication_id=state['run_id'], receipt_dir=out / 'results-layout')
    write(out / 'published.json', dict(snapshot_sha256=snapshot['snapshot_sha256'], run_id=state['run_id'], layout=layout))
    return layout


def publish_with_retry(operation, out):
    for attempt, delay in enumerate((5, 15, 30, 60, 60, None)):
        try:
            return operation()
        except (SeriesError, J10LayoutConflict):
            raise
        except Exception as exc:
            write(out / 'publisher-retry.json', dict(attempt=attempt+1, error=f'{type(exc).__name__}: {exc}', delay_seconds=delay))
            if delay is None:
                raise
            time.sleep(delay)


def watch(args):
    root, out = args.root.resolve(), args.publication_root.resolve()
    require(root != out and not root.is_relative_to(out) and not out.is_relative_to(root),
            'Publication root must be separate from scientific campaign output.')
    with publisher_lock(out):
        campaign = read(root / 'campaign.json')
        references, pins = load_references(campaign, args.reference_publication_root)
        if args.validate_only:
            print(json.dumps(dict(status='validated', reference_J=[4, 6, 8], prior=True, reference_files=len(pins))))
            return
        import wandb
        state = publication_state(root, out, campaign, pins)
        previous, inactive_since = None, None
        while True:
            try:
                require(all(digest(path) == sha for path, sha in pins.items()), 'Pinned reference changed during publication.')
                require(fingerprint(read(root / 'campaign.json')) == state['campaign_sha256'], 'Campaign changed during publication.')
                active = gpu_jobs_active(args.gpu_job_id) if args.gpu_job_id else False
                inactive_since = None if active else inactive_since or time.monotonic()
                waiting = active or time.monotonic()-inactive_since < args.worker_grace_seconds
                snapshot = collect(root, campaign, references, active=waiting)
                if (out / 'snapshot.json').exists():
                    before = read(out / 'snapshot.json')
                    require(not before['validated_complete'] or before['snapshot_sha256'] == snapshot['snapshot_sha256'],
                            'Previously finalized result changed or disappeared.')
                write(out / 'snapshot.json', snapshot)
                if snapshot['snapshot_sha256'] != previous:
                    layout = publish_with_retry(lambda: publish_once(wandb, state, campaign, snapshot, out), out)
                    previous = snapshot['snapshot_sha256']
                    print(json.dumps(dict(completed=snapshot['completed'], total=5, state=snapshot['state'], url=layout['url'])), flush=True)
                write(out / 'publisher-status.json', dict(status=snapshot['state'], completed=snapshot['completed'],
                    total=5, validated_complete=snapshot['validated_complete'], run_id=state['run_id']))
                if snapshot['validated_complete'] or args.once:
                    return
                require(waiting, 'GPU work stopped before the complete five-seed result passed validation.')
                time.sleep(args.poll_seconds)
            except BaseException as exc:
                write(out / 'publisher-failure.json', dict(error_type=type(exc).__name__, error=str(exc), run_id=state['run_id']))
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--publication-root', type=Path, required=True)
    parser.add_argument('--reference-publication-root', type=Path, action='append', required=True)
    parser.add_argument('--gpu-job-id', action='append', default=[])
    parser.add_argument('--poll-seconds', type=float, default=60.)
    parser.add_argument('--worker-grace-seconds', type=float, default=180.)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(argv)
    if not 0 < args.poll_seconds <= 60 or args.worker_grace_seconds < 0:
        parser.error('Poll must be in (0,60]; grace must be nonnegative.')
    if not args.gpu_job_id and not (args.once or args.validate_only):
        parser.error('Watch requires --gpu-job-id.')
    watch(args)


if __name__ == '__main__':
    main()
