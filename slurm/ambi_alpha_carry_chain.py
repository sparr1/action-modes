"""Independent frozen-checkpoint streams carrying only temperature across episodes.

Each worker owns one ordered three-episode chain. Reused references and new
streams have distinct scientific identities; publication never pools dependent
episodes as independent seeds. GPU work remains offline.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from slurm import ambi_alpha_adaptation_episodes as previous
from slurm.ambi_early_h3_j8_campaign import read, write_new, sha256, verify_checkpoint
from slurm.ambi_entropy_kl_episodes import source, trace_hashes, OUTER_ALPHAS
from slurm.ambi_mppi_checkpoint_comparison import require

MATRIX = REPO / 'configs/research/ambi_alpha_carry_chain.json'
PROFILE = previous.PROFILE
BASE = Path('/oscar/scratch/rgao48/ambi/alpha-adaptation/20261021-9ad45fe9')
PROTOCOL = 'run-temperature-chain-v1'
CAMPAIGN_ID = 'alpha-carry-chain-20261022'
CHECKPOINTS = ('sac_aux_100k', 'mppi_aux_400k')
SEEDS = [101, 102, 103, 104, 105]
STREAMS = [[SEEDS[(stream + episode) % 5] for episode in range(3)] for stream in range(5)]
OVERRIDES = dict(inner_temperature_scope='run', inner_temperature_optimizer_scope='run',
    inner_temperature_updates_per_actor=1, inner_sac_return_estimator='one_step',
    inner_actor_scope='action', inner_critic_scope='action', inner_replay_scope='action',
    inner_actor_optimizer_scope='action', inner_critic_optimizer_scope='action',
    inner_actor_lr=.0003, inner_critic_lr=.0003, inner_temperature_lr=.0003,
    inner_entropy_enabled=True, inner_outer_policy_kl_coef=0., inner_outer_action_l2_coef=0.,
    inner_actor_initialization='prior', inner_critic_initialization='prior',
    inner_temperature_initialization='inherit_outer', inner_target_entropy='inherit_outer')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def intervention(name, stream_index):
    require(name in CHECKPOINTS and type(stream_index) is int and stream_index in range(5), 'Unknown chain cell')
    return dict(checkpoint_id=name, stream_index=stream_index, seeds=STREAMS[stream_index],
                cell_id=f'{name}__stream{stream_index}', selector='alpha_carry_chain/run',
                mode='run_carry', outer_alpha=OUTER_ALPHAS[name], expected_temperature_steps=32)


def validate_matrix(matrix):
    old = read(previous.MATRIX)
    require(matrix['base_alg_config'] == 'checkpoint', 'Expected checkpoint-derived networks')
    require(matrix['shared_alg_params'] == {**old['shared_alg_params'],
        'inner_temperature_scope': 'run', 'inner_temperature_optimizer_scope': 'run'},
        'Only temperature lifetime may change')
    require(matrix['evaluation'] == {**old['evaluation'], 'default_presets': ['alpha_carry_chain/run']},
            'Evaluation protocol changed')
    require(matrix['temperature_chain'] == dict(protocol=PROTOCOL, streams=5, episodes_per_stream=3,
        ordered_seeds=STREAMS, independent_unit='stream', checkpoints=list(CHECKPOINTS)), 'Chain layout changed')
    require(set(matrix['comparisons']) == {'alpha_carry_chain'}, 'Unexpected chain comparison')
    group = matrix['comparisons']['alpha_carry_chain']
    require(group['reference'] == 'run' and set(group['variants']) == {'run'}, 'Unexpected chain arm')
    require(group['variants']['run']['alg_params'] == {}
            and group['variants']['run']['evaluation'] == {'execution_action': 'sample'}, 'Unexpected intervention')


def validate_campaign(campaign):
    validate_matrix(read(MATRIX))
    require(all(campaign.get(k) == v for k, v in dict(schema_version=1, campaign_id=CAMPAIGN_ID,
        protocol=PROTOCOL, streams=STREAMS, controller_seed=55, max_steps=500,
        execution_action='sample', independent_unit='stream').items()), 'Campaign protocol changed')
    require(campaign['matrix_sha256'] == sha256(MATRIX) and campaign['profile_sha256'] == sha256(PROFILE),
            'Pinned configuration changed')
    root = Path(campaign['root'])
    require(root.is_absolute() and set(campaign['checkpoints']) == set(CHECKPOINTS), 'Invalid campaign root/checkpoints')
    require(len(campaign['cells']) == 10, 'Exactly ten independent streams are required')
    for index, cell in enumerate(campaign['cells']):
        name, stream_index = CHECKPOINTS[index // 5], index % 5
        spec = {**read(PROFILE)['checkpoints'][name], **intervention(name, stream_index)}
        require(cell['index'] == index and all(cell.get(k) == v for k, v in spec.items()), 'Chain ordering or checkpoint changed')
        require(cell['output_dir'] == str(root / 'production' / cell['cell_id']), 'Stream output ownership changed')
        require(cell['science_id'] == canonical_hash(dict(protocol=PROTOCOL, source=campaign['source_commit'],
            matrix=campaign['matrix_sha256'], checkpoint=cell['checkpoint_sha256'], stream=stream_index,
            seeds=cell['seeds'], controller_seed=55)), 'Stream scientific identity changed')
    for key in ('cell_id', 'output_dir', 'science_id'):
        require(len({c[key] for c in campaign['cells']}) == 10, 'Stream ownership collision: ' + key)


def verify_files(files):
    for path, expected in files.items():
        require(sha256(path) == expected, 'Pinned reference changed: ' + path)


def prepare_references(root, name, old_campaign):
    cell, = [c for c in old_campaign['cells'] if c['checkpoint_id'] == name and c['mode'] == 'carry']
    verify_checkpoint(cell)
    pins = previous.reference_pins(cell)
    require(pins == {k: cell[k] for k in pins}, 'Reference pins changed')
    carry_output = Path(cell['output_dir'])
    carry_receipt = read(carry_output / 'receipt.json')
    require(carry_receipt['validated'] is True and not carry_receipt['smoke']
            and carry_receipt['source_commit'] == old_campaign['source_commit']
            and carry_receipt['cell_id'] == cell['cell_id'] and carry_receipt['validation']['lifecycle_validated'] is True,
            'Within-episode carry reference is incomplete or mismatched')
    files = {str(carry_output / key): carry_receipt[hash_key] for key, hash_key in (
        ('results.json', 'results_sha256'), ('bundle/manifest.json', 'manifest_sha256'),
        ('decision-summary.json', 'decision_summary_sha256'))}
    verify_files(files)
    require(trace_hashes(carry_output / 'bundle') == carry_receipt['trace_sha256'], 'Carry reference trace changed')
    carry = read(carry_output / 'decision-summary.json')
    previous.validate_decisions(carry, cell, SEEDS, 500)
    payload = read(carry_output / 'results.json')['results'][0]
    require(payload['outer_state_unchanged'] is True and not payload['nonfinite_model_metrics']
            and not payload['nonfinite_trace_metrics'] and payload['execution_action'] == 'sample'
            and payload['outer_updates_before'] == payload['outer_updates_after']
            and [e['seed'] for e in payload['episodes']] == SEEDS
            and all(e['length'] == 500 and not e['truncated_by_evaluator'] for e in payload['episodes']),
            'Within-episode carry reference lacks complete frozen episodes')
    reset = previous.load_baseline_decision_summary(cell)
    require([(r['seed'], r['decision_index']) for r in reset['rows']]
            == [(s, d) for s in SEEDS for d in range(500)], 'Reset reference incomplete')
    prior_payload = read(Path(cell['reference_bundle']).parent / 'results.json')['results'][0]
    prior = dict(schema_version=1, checkpoint_id=name, mode='prior', rows=[],
                 episodes=prior_payload['episodes'], provenance=dict(bundle=cell['reference_bundle']))
    for kind in ('reference', 'baseline'):
        bundle = Path(cell[kind + '_bundle'])
        files[str(bundle / 'manifest.json')] = cell[kind + '_manifest_sha256']
        files[str(bundle.parent / 'results.json')] = cell[kind + '_results_sha256']
    for path in (carry_output / 'receipt.json', Path(cell['baseline_decision_summary']),
                 Path(cell['baseline_decision_summary']).with_name('baseline-receipt.json'), Path(cell['inventory_path'])):
        files[str(path)] = sha256(path)
    summaries = {}
    for kind, data in [('prior', prior), ('reset', reset), ('episode_carry', carry)]:
        path = root / 'references' / name / f'{kind}-summary.json'
        write_new(path, data)
        summaries[kind] = dict(path=str(path), sha256=sha256(path))
    return dict(checkpoint_id=name, checkpoint_sha256=cell['checkpoint_sha256'],
                reference_files=files, reference_summaries=summaries,
                inventory_path=cell['inventory_path'], inventory_sha256=sha256(cell['inventory_path']))


def prepare(root, reference_campaign=BASE / 'campaign.json'):
    root = Path(root).resolve()
    if (root / 'campaign.json').exists():
        raise FileExistsError('A chain campaign already exists here')
    validate_matrix(read(MATRIX))
    old = read(reference_campaign)
    previous.validate_campaign(old)
    campaign = dict(schema_version=1, campaign_id=CAMPAIGN_ID, root=str(root), protocol=PROTOCOL,
        source_commit=source(), matrix_sha256=sha256(MATRIX), profile_sha256=sha256(PROFILE),
        reference_campaign=str(Path(reference_campaign).resolve()), reference_campaign_sha256=sha256(reference_campaign),
        streams=STREAMS, controller_seed=55, max_steps=500, execution_action='sample', independent_unit='stream',
        checkpoints={}, cells=[])
    for name in CHECKPOINTS:
        refs = prepare_references(root, name, old)
        refs['science_id'] = canonical_hash(dict(protocol=PROTOCOL, source=campaign['source_commit'],
            matrix=campaign['matrix_sha256'], checkpoint=refs['checkpoint_sha256'], streams=STREAMS, controller_seed=55))
        campaign['checkpoints'][name] = refs
        for stream_index in range(5):
            cell = {**read(PROFILE)['checkpoints'][name], **intervention(name, stream_index),
                'index': len(campaign['cells']), 'inventory_path': refs['inventory_path'],
                'inventory_sha256': refs['inventory_sha256']}
            cell['output_dir'] = str(root / 'production' / cell['cell_id'])
            cell['science_id'] = canonical_hash(dict(protocol=PROTOCOL, source=campaign['source_commit'],
                matrix=campaign['matrix_sha256'], checkpoint=cell['checkpoint_sha256'], stream=stream_index,
                seeds=cell['seeds'], controller_seed=55))
            campaign['cells'].append(cell)
    validate_campaign(campaign)
    require(source() == campaign['source_commit'], 'Source changed during preparation')
    write_new(root / 'campaign.json', campaign)
    return campaign


def evaluator_command(cell, output, smoke_steps=None):
    require(smoke_steps is None or type(smoke_steps) is int and smoke_steps in (4, 8), 'Invalid smoke length')
    output = Path(output)
    return [sys.executable, str(REPO / 'evaluate_ambi_checkpoint.py'), '--matrix', str(MATRIX),
        '--checkpoint', cell['checkpoint'], '--metadata', cell['metadata'], '--preset', cell['selector'],
        '--device', 'cuda', '--seeds', *map(str, cell['seeds'][:2] if smoke_steps else cell['seeds']),
        '--controller-seed', '55', '--max-steps', str(smoke_steps or 500), '--temperature-chain',
        '--output', str(output / 'results.json'), '--bundle-dir', str(output / 'bundle'),
        '--checkpoint-inventory', cell['inventory_path']]


def compact_summary(cell, bundle, seeds, steps):
    summary = previous.compact_summary_from_bundle(cell, bundle, require_lifecycle=True)
    chain_resets = {}
    manifest = read(Path(bundle) / 'manifest.json')
    for relative in manifest['runs'][0]['trace_files']:
        path = (Path(bundle) / relative).resolve(strict=True)
        require(path.is_relative_to(Path(bundle).resolve()), 'Trace escapes bundle')
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                event = json.loads(line)
                if event['phase'] == 'decision':
                    key = int(event['episode_id'].removeprefix('seed-')), event['decision_index']
                    require(key not in chain_resets, 'Duplicate chain decision')
                    chain_resets[key] = event['metrics'].get('decision/inner_temperature_chain_reset')
    ordering = {seed: i for i, seed in enumerate(seeds)}
    summary['rows'].sort(key=lambda row: (ordering[row['seed']], row['decision_index']))
    for row in summary['rows']:
        episode = ordering[row['seed']]
        row.update(stream_index=cell['stream_index'], episode_index=episode,
                   global_decision_index=episode * steps + row['decision_index'],
                   chain_reset=chain_resets[row['seed'], row['decision_index']], controller_seed=55)
    for episode_index, episode in enumerate(summary['episodes']):
        episode.update(stream_index=cell['stream_index'], episode_index=episode_index, controller_seed=55)
    summary.update(protocol=PROTOCOL, cell_id=cell['cell_id'], science_id=cell['science_id'],
                   stream_index=cell['stream_index'], seeds=seeds, steps_per_episode=steps)
    return summary


def validate_decisions(summary, cell, seeds, steps):
    require(summary.get('protocol') == PROTOCOL and summary.get('science_id') == cell['science_id']
            and summary.get('stream_index') == cell['stream_index'], 'Wrong chain summary identity')
    expected = [(e, seed, decision, e * steps + decision) for e, seed in enumerate(seeds) for decision in range(steps)]
    require([(r['episode_index'], r['seed'], r['decision_index'], r['global_decision_index'])
             for r in summary['rows']] == expected, 'Incomplete, duplicate or reordered chain decisions')
    alpha = cell['outer_alpha']
    for row in summary['rows']:
        for key, value in row.items():
            if type(value) in (int, float):
                require(math.isfinite(value), 'Non-finite chain diagnostic: ' + key)
        step = row['global_decision_index'] * 32
        require(row['stream_index'] == cell['stream_index'] and row['controller_seed'] == 55, 'Stream RNG identity changed')
        require(row['temperature_updates'] == 32 and row['target_entropy'] == -10.5, 'Temperature workload/target changed')
        require(row['optimizer_step_initial'] == row['lifetime_steps_initial'] == step
                and row['optimizer_step_final'] == row['lifetime_steps_final'] == step + 32,
                'Alpha Adam/lifetime reset or discontinuity across chain')
        require(row['episode_reset'] == 0 and row['chain_reset'] == int(step == 0), 'Incorrect chain reset boundary')
        require(row['alpha_initial'] > 0 and row['alpha_final'] > 0
                and math.isclose(row['alpha_initial'], alpha, rel_tol=1e-6, abs_tol=1e-10), 'Alpha value discontinuity')
        require(row['control_seconds'] >= 0, 'Negative control time')
        alpha = row['alpha_final']
    require([e['seed'] for e in summary['episodes']] == seeds, 'Chain episode ordering changed')
    for index, episode in enumerate(summary['episodes']):
        require(episode['episode_index'] == episode['temperature_chain_episode_index'] == index
                and episode['stream_index'] == cell['stream_index'] and episode['length'] == steps,
                'Chain episode identity/coverage changed')
        rows = summary['rows'][index * steps:(index + 1) * steps]
        require(math.isclose(sum(r['reward'] for r in rows), episode['return'], rel_tol=1e-8, abs_tol=1e-7)
                and math.isclose(rows[-1]['cumulative_reward'], episode['return'], rel_tol=1e-8, abs_tol=1e-7),
                'Chain trace rewards disagree with episode return')


def validate_result(output, cell, smoke_steps=None):
    from slurm.ambi_early_h3_j8_validation import validate_result as full_validation
    output = Path(output)
    seeds, steps = (cell['seeds'][:2], smoke_steps) if smoke_steps else (cell['seeds'], 500)
    validation = full_validation(output / 'results.json', output / 'bundle', expected_seeds=seeds,
                                 max_steps=steps, expected_temperature_steps=32, config_overrides=OVERRIDES)
    payload, manifest = read(output / 'results.json'), read(output / 'bundle/manifest.json')
    result, = payload['results']
    require(payload['checkpoint_sha256'] == cell['checkpoint_sha256'] and result['selector'] == cell['selector']
            and result['execution_action'] == 'sample' and result.get('temperature_chain') is True
            and result.get('temperature_chain_seeds') == seeds
            and result.get('temperature_chain_kind') == PROTOCOL, 'Wrong checkpoint or chain execution protocol')
    require(manifest['protocol'].get('temperature_chain') == dict(kind=PROTOCOL, ordered_environment_seeds=seeds),
            'Missing explicit chain bundle protocol')
    summary = compact_summary(cell, output / 'bundle', seeds, steps)
    validate_decisions(summary, cell, seeds, steps)
    validation.update(lifecycle_validated=True, chain_boundaries_validated=len(seeds)-1,
                      temperature_steps_per_decision=32, protocol=PROTOCOL)
    return validation, summary


def run(campaign_path, index, smoke_steps=None, smoke_root=None):
    campaign_path = Path(campaign_path).resolve()
    campaign = read(campaign_path)
    validate_campaign(campaign)
    require(type(index) is int and index in range(10), 'Invalid stream index')
    require(source() == campaign['source_commit'], 'Wrong or dirty source checkout')
    cell = campaign['cells'][index]
    verify_checkpoint(cell)
    require(sha256(cell['inventory_path']) == cell['inventory_sha256'], 'Inventory changed')
    require(smoke_steps is not None or smoke_root is None, 'Alternate root is for smoke only')
    if smoke_root is not None:
        smoke_root = Path(smoke_root).resolve()
        require(smoke_root.is_relative_to(Path(campaign['root'])), 'Smoke path escapes campaign')
    output = (Path(smoke_root or Path(campaign['root']) / 'smoke') / f"{cell['cell_id']}-steps{smoke_steps}"
              if smoke_steps else Path(cell['output_dir']))
    output.mkdir(parents=True, exist_ok=False)
    command = evaluator_command(cell, output, smoke_steps)
    write_new(output / 'started.json', dict(command=command, job_id=os.environ.get('SLURM_JOB_ID'),
                                          cell_id=cell['cell_id'], source_commit=campaign['source_commit']))
    started, before = time.perf_counter(), resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        subprocess.run(command, cwd=REPO, check=True)
        compute, after = time.perf_counter() - started, resource.getrusage(resource.RUSAGE_CHILDREN)
        validation, summary = validate_result(output, cell, smoke_steps)
        write_new(output / 'chain-summary.json', summary)
        require(source() == campaign['source_commit'], 'Source changed during evaluation')
        cpu = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
        receipt = dict(validated=True, smoke=smoke_steps is not None, source_commit=campaign['source_commit'],
            protocol=PROTOCOL, cell_id=cell['cell_id'], science_id=cell['science_id'],
            checkpoint_id=cell['checkpoint_id'], checkpoint_sha256=cell['checkpoint_sha256'],
            stream_index=cell['stream_index'], validation=validation,
            elapsed_seconds=time.perf_counter()-started, child_elapsed_seconds=compute,
            cpu_threads=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')), child_cpu_seconds=cpu,
            child_cpu_cores_mean=cpu/compute, child_maxrss_kib=after.ru_maxrss,
            results_sha256=sha256(output / 'results.json'), manifest_sha256=sha256(output / 'bundle/manifest.json'),
            chain_summary_sha256=sha256(output / 'chain-summary.json'), trace_sha256=trace_hashes(output / 'bundle'))
        write_new(output / 'receipt.json', receipt)
        return receipt
    except BaseException as exc:
        write_new(output / 'failure.json', dict(exception_type=type(exc).__name__, message=str(exc)))
        raise


def load_stream(campaign, cell, output=None, smoke_steps=None):
    output = Path(output or cell['output_dir'])
    receipt = read(output / 'receipt.json')
    require(receipt.get('validated') is True and receipt.get('smoke') is (smoke_steps is not None)
            and receipt.get('source_commit') == campaign['source_commit'] and receipt.get('protocol') == PROTOCOL
            and receipt.get('science_id') == cell['science_id'] and receipt.get('cell_id') == cell['cell_id']
            and receipt.get('checkpoint_sha256') == cell['checkpoint_sha256']
            and receipt['validation'].get('passed') is True and receipt['validation'].get('lifecycle_validated') is True,
            'Unvalidated or mismatched stream receipt')
    for path, key in [('results.json', 'results_sha256'), ('bundle/manifest.json', 'manifest_sha256'),
                      ('chain-summary.json', 'chain_summary_sha256')]:
        require(sha256(output / path) == receipt[key], 'Stream artifact changed: ' + path)
    require(trace_hashes(output / 'bundle') == receipt['trace_sha256'], 'Stream trace changed')
    summary = read(output / 'chain-summary.json')
    validate_decisions(summary, cell, cell['seeds'][:2] if smoke_steps else cell['seeds'], smoke_steps or 500)
    return summary, receipt


def publication_snapshot(campaign):
    campaign = read(campaign) if isinstance(campaign, (str, Path)) else campaign
    validate_campaign(campaign)
    require(sha256(campaign['reference_campaign']) == campaign['reference_campaign_sha256'], 'Reference campaign changed')
    checkpoints = {}
    for name, reference in campaign['checkpoints'].items():
        verify_files(reference['reference_files'])
        summaries = {}
        for kind, item in reference['reference_summaries'].items():
            require(sha256(item['path']) == item['sha256'], 'Copied reference summary changed')
            summaries[kind] = read(item['path'])
        entry = dict(reference_summaries=summaries, streams=[], status=[], expected_streams=5,
                     science_id=reference['science_id'])
        for cell in (c for c in campaign['cells'] if c['checkpoint_id'] == name):
            output = Path(cell['output_dir'])
            status = dict(cell_id=cell['cell_id'], index=cell['index'], stream_index=cell['stream_index'])
            if (output / 'receipt.json').exists():
                try:
                    summary, receipt = load_stream(campaign, cell)
                    entry['streams'].append(summary)
                    status.update(state='complete', receipt_sha256=sha256(output / 'receipt.json'))
                except (ValueError, KeyError, FileNotFoundError) as exc:
                    status.update(state='validation_failed', message=str(exc))
            elif (output / 'failure.json').exists():
                status.update(state='failed', failure=read(output / 'failure.json'))
            else:
                status['state'] = 'running' if (output / 'started.json').exists() else 'pending'
            entry['status'].append(status)
        entry['complete'] = len(entry['streams']) == 5
        checkpoints[name] = entry
    return dict(campaign=campaign, checkpoints=checkpoints,
                complete=all(row['complete'] for row in checkpoints.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reference-campaign', type=Path, default=BASE / 'campaign.json')
    p = sub.add_parser('run'); p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--index', type=int, choices=range(10), required=True)
    p.add_argument('--smoke-steps', type=int, choices=(4, 8)); p.add_argument('--smoke-root', type=Path)
    args = parser.parse_args()
    result = (prepare(args.root, args.reference_campaign) if args.command == 'prepare'
              else run(args.campaign, args.index, args.smoke_steps, args.smoke_root))
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
