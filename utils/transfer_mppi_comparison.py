"""Validate existing MPPI results for a presentation-only checkpoint overlay.

No evaluation registry or scientific history is modified. The two original
MPPI runs remain the owners of their immutable results and diagnostic files.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import uuid

from utils.eval_series import _atomic_json, _digest, _record_fingerprint, validate_record
from utils.transfer_checkpoint_publication import (
    POINT_COLUMNS, PROGRESS_COLUMNS, _episodes, require, moments,
)

STYLES = [
    dict(setting_id='mppi_h3_return_q', label='MPPI H3 · Return Q', color='#cc79a7', role='comparison'),
    dict(setting_id='mppi_h3_soft_q', label='MPPI H3 · Soft Q', color='#56b4e9', role='comparison'),
]
SOURCES = {
    'return_q': ('mppi-return-q', '24f3f6be21b74961beeb7e846b880613', 'aux_return'),
    'soft_q': ('mppi-soft-q', '88e1989e3f31474c987798c1bd5a0370', 'sac'),
}


def read(path):
    return json.loads(Path(path).read_text())


def verified_records(directory, campaign, *, run_id, terminal=None):
    """Verify exported records against their original publication fingerprints."""
    directory = Path(directory)
    registry, journal = read(directory/'run.json'), read(directory/'publication.json')
    ident = registry['identity']
    require(registry['run_id'] == run_id and registry['identity_sha256'] == _digest(ident),
            'Source run identity changed.')
    require(ident['backbone'] == campaign['source_run'], 'MPPI comparison backbone differs.')
    protocol, planner = ident['protocol'], ident['planner']
    require(protocol['environment_seeds'] == campaign['seeds'] and
            protocol['controller_seed'] == campaign['controller_seed'] and
            protocol['max_steps'] == campaign['max_steps'] == 500 and protocol['mode'] == 'episodes',
            'Comparison episode protocol differs.')
    if terminal is None:
        require(planner['type'] == 'prior' and protocol['action_rule'] == 'tanh_mean',
                'Comparison reference must be the frozen prior mean.')
    else:
        cfg = planner['settings']
        required = dict(inner_operator='mppi', inner_rollout_horizon=3,
                        inner_horizon_critic_source=terminal, inner_mppi_iterations=8,
                        inner_mppi_num_samples=512, inner_mppi_num_elites=64,
                        inner_mppi_num_pi_trajs=24, inner_mppi_warm_start_scope='episode',
                        inner_mppi_temperature=0.5, inner_mppi_min_std=0.05, inner_mppi_max_std=2.0)
        require(planner['type'] == 'mppi' and protocol['action_rule'] == 'mppi_proposal_mean'
                and all(cfg.get(key, 'sac' if key == 'inner_horizon_critic_source' else None) == value
                        for key, value in required.items()),
                'MPPI H3 route or planner budget differs.')
        env = protocol['environment']
        require(env['id'] == 'DMControl-v0' and env['params']['task'] == 'humanoid-walk'
                and env['params']['obs'] == 'state', 'MPPI environment differs.')
    expected = {cp['step']: cp['checkpoint_sha256'] for cp in campaign['checkpoints']}
    records, pins = {}, {}
    for rid, entry in journal['records'].items():
        record = validate_record(read(directory/'records'/(rid+'.json')), ident)
        if 'record_kind' in record:
            continue  # Paired supplements are not additional evaluated episodes.
        require(entry['status'] == 'published' and rid == record['record_id'] and
                _record_fingerprint(record, entry['artifact_sha256']) == entry['record_sha256'],
                'Source result is unpublished or differs from its accepted fingerprint.')
        cp = record['checkpoint']; step = cp['step']
        require(step not in records and expected.get(step) == cp['sha256'] and
                entry['checkpoint_step'] == step and entry['checkpoint_sha256'] == cp['sha256'],
                'Duplicate or mismatched comparison checkpoint.')
        require(len(record['episodes']) == len(campaign['seeds']), 'Comparison must use exactly five episodes.')
        episodes = _episodes(record['episodes'], campaign)
        metrics = record['metrics']; mean, sd = moments([e['return'] for e in episodes])
        require(metrics['eval/frozen_state_unchanged'] is True and metrics['eval/episodes'] == len(episodes),
                'Comparison frozen-state or episode proof missing.')
        require(math.isclose(mean, metrics['eval/return_mean'], rel_tol=1e-10, abs_tol=1e-8)
                and math.isclose(sd, metrics['eval/return_sample_std'], rel_tol=1e-10, abs_tol=1e-8),
                'Stored MPPI statistics differ from its episodes.')
        if terminal is not None:
            require(record['provenance']['resolved_config'].get('inner_horizon_critic_source', 'sac') == terminal,
                    'Resolved terminal critic differs from the selected MPPI route.')
            require(all(metrics.get('work/'+key) == 0 for key in
                        ('actor_updates', 'critic_updates', 'temperature_updates')),
                    'MPPI source contains learning updates.')
        records[step] = episodes
        pins[str(step)] = dict(record_id=rid, record_sha256=entry['record_sha256'],
                               checkpoint_sha256=cp['sha256'])
    require(set(records) == set(expected), 'MPPI comparison checkpoint coverage is incomplete.')
    return records, dict(run_id=run_id, identity_sha256=registry['identity_sha256'], records=pins)


def build_overlay(campaign, source_root):
    root = Path(source_root)
    prior_id = read(root/'base-actor/run.json')['run_id']
    priors, prior_pin = verified_records(root/'base-actor', campaign, run_id=prior_id)
    points, progress, sources = [], [], {}
    for style, (key, (directory, run_id, terminal)) in zip(STYLES, SOURCES.items()):
        records, sources[key] = verified_records(root/directory, campaign, run_id=run_id, terminal=terminal)
        for step, episodes in sorted(records.items()):
            lookup = {e['seed']: e['return'] for e in priors[step]}
            row = {name: None for name in POINT_COLUMNS}
            row.update(step=step, setting=style['setting_id'], label=style['label'], state='complete',
                       segment=0, fresh_segment=0, episodes=len(episodes),
                       fresh_comparison_state='not_applicable', return_min=min(e['return'] for e in episodes))
            for prefix, values in [('return', [e['return'] for e in episodes]),
                                   ('gain', [e['return']-lookup[e['seed']] for e in episodes])]:
                mean, sd = moments(values)
                row.update({prefix+'_mean':mean, prefix+'_std':sd,
                            prefix+'_lower':mean-sd, prefix+'_upper':mean+sd})
            points.append(row)
            progress.append(dict(step=step, setting=style['setting_id'], label=style['label'],
                                 state='complete', completed_episodes=len(episodes), seed=None,
                                 decision=None, error=None))
    return dict(format_version=1, curves=deepcopy(STYLES), points=points, progress=progress,
                sources=sources, prior=prior_pin,
                timing_note='Omitted: historical MPPI GPU models vary; matching steady timings unavailable.',
                semantics='Same frozen SAC proposal/terminal actor; only terminal soft-Q versus return-Q route differs.')


def publish_overlay(data, campaign_sha256, state, receipt_path, wandb):
    """Persist one overlay ID before init; retries cannot create extra runs."""
    path = Path(receipt_path)
    content_hash = _digest(data)
    expected = dict(format_version=1, publication_id=state['publication_id'],
                    campaign_sha256=campaign_sha256, content_sha256=content_hash,
                    curves=data['curves'], source_runs={k:v['run_id'] for k,v in data['sources'].items()})
    if path.exists():
        receipt = read(path)
        require(all(receipt.get(k) == v for k,v in expected.items()), 'Existing overlay has different source/content pins.')
        if receipt.get('status') == 'published':
            return receipt
    else:
        receipt = dict(expected, run_id=uuid.uuid4().hex, status='allocated')
        _atomic_json(path, receipt)
    run = wandb.init(entity=state['entity'], project=state['project'], id=receipt['run_id'],
        resume='allow', mode='online', name='MPPI H3 · checkpoint comparison references',
        job_type='transfer-checkpoint-comparison', group='transfer-backbone-'+state['publication_id'][:8],
        config=dict(transfer_curve_overview=state['publication_id'], publication_only=True,
                    comparison_content_sha256=content_hash, comparison_sources=data['sources'],
                    comparison_prior=data['prior'], source_campaign_sha256=campaign_sha256,
                    timing_note=data['timing_note'], comparison_semantics=data['semantics']),
        dir=str(path.parent), reinit=True)
    try:
        run.log({'transfer_curves/points':wandb.Table(columns=POINT_COLUMNS,
                     data=[[row[name] for name in POINT_COLUMNS] for row in data['points']]),
                 'transfer_curves/progress':wandb.Table(columns=PROGRESS_COLUMNS,
                     data=[[row[name] for name in PROGRESS_COLUMNS] for row in data['progress']])})
        run.summary.update(dict(publication_only=True, comparison_completed_points=len(data['points']),
                                comparison_content_sha256=content_hash))
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise
    receipt['status'] = 'published'
    _atomic_json(path, receipt)
    return receipt
