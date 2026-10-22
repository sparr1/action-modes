"""Matched three-episode control resetting temperature and Adam between episodes.

The checkpoint, seed streams, sampled execution and inner solve remain fixed.
Only cross-episode temperature persistence changes from the preserved campaign.
"""
from __future__ import annotations

import argparse
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
from slurm import ambi_alpha_carry_chain as chain
from slurm import ambi_alpha_adaptation_episodes as original
from slurm import ambi_entropy_kl_mppi500k as references
from slurm import ambi_alpha_preservation as preservation

read, write_new, sha256 = chain.read, chain.write_new, chain.sha256
source, require, verify_checkpoint = chain.source, chain.require, chain.verify_checkpoint
trace_hashes, canonical_hash = chain.trace_hashes, chain.canonical_hash
MATRIX = REPO / 'configs/research/ambi_alpha_episode_reset.json'
PROFILE = REPO / references.PROFILE
BASE = Path('/oscar/scratch/rgao48/ambi/alpha-preservation/20261022-b315ca33')
PROTOCOL, STREAMS, SEEDS = 'episode-temperature-reset-v1', chain.STREAMS, chain.SEEDS
CAMPAIGN_ID, CHECKPOINT_ID = 'alpha-episode-reset-20261022', 'mppi_aux_500k'
PRIMARY = dict(episode_index=2, independent_unit='stream',
               outcomes=['paired_gain_vs_prior', 'paired_delta_vs_reset', 'paired_delta_vs_chain_carry'],
               interpretation='positive delta_vs_chain_carry means episode reset improves on cross-episode carry')
OVERRIDES = {**chain.OVERRIDES, 'inner_temperature_scope': 'episode',
             'inner_temperature_optimizer_scope': 'episode'}


def intervention(stream_index):
    require(type(stream_index) is int and stream_index in range(5), 'Unknown episode-reset stream')
    pin = read(PROFILE)['checkpoints'][CHECKPOINT_ID]
    return dict(checkpoint_id=CHECKPOINT_ID, stream_index=stream_index, seeds=STREAMS[stream_index],
        cell_id=f'{CHECKPOINT_ID}__stream{stream_index}', selector='alpha_episode_reset/episode',
        mode='carry', outer_alpha=pin['outer_alpha'], expected_temperature_steps=32)


def validate_matrix(matrix):
    baseline = read(preservation.MATRIX)
    require(matrix['base_alg_config'] == 'checkpoint', 'Expected checkpoint-derived networks')
    require(matrix['shared_alg_params'] == {**baseline['shared_alg_params'],
        'inner_temperature_scope': 'episode', 'inner_temperature_optimizer_scope': 'episode'},
        'Only temperature lifetime may change from cross-episode carry')
    require(matrix['evaluation'] == {**baseline['evaluation'], 'default_presets': ['alpha_episode_reset/episode']},
            'Episode-reset evaluation protocol changed')
    require(matrix['temperature_chain'] == dict(protocol=PROTOCOL, streams=5, episodes_per_stream=3,
        ordered_seeds=STREAMS, independent_unit='stream', checkpoints=[CHECKPOINT_ID]), 'Episode-reset chain layout changed')
    group = matrix['comparisons'].get('alpha_episode_reset', {})
    require(set(matrix['comparisons']) == {'alpha_episode_reset'} and group.get('reference') == 'episode'
            and set(group.get('variants', {})) == {'episode'}
            and group['variants']['episode']['alg_params'] == {}
            and group['variants']['episode']['evaluation'] == {'execution_action': 'sample'}, 'Unexpected episode-reset arm')


def validate_campaign(campaign):
    validate_matrix(read(MATRIX))
    require(all(campaign.get(k) == v for k, v in dict(schema_version=1, campaign_id=CAMPAIGN_ID,
        protocol=PROTOCOL, streams=STREAMS, controller_seed=55, max_steps=500, execution_action='sample',
        independent_unit='stream', primary_analysis=PRIMARY).items()), 'Episode-reset campaign protocol changed')
    require(campaign['matrix_sha256'] == sha256(MATRIX) and campaign['profile_sha256'] == sha256(PROFILE),
            'Episode-reset configuration changed')
    root = Path(campaign['root'])
    require(root.is_absolute() and set(campaign['checkpoints']) == {CHECKPOINT_ID}
            and len(campaign['cells']) == 5, 'Episode-reset needs exactly five MPPI500k streams')
    require(set(campaign['checkpoints'][CHECKPOINT_ID]['reference_summaries']) == {'prior', 'reset'},
            'Only the existing prior and fresh-alpha controls are allowed')
    refs = campaign['checkpoints'][CHECKPOINT_ID]
    require([item['stream_index'] for item in refs['reference_streams']] == list(range(5))
            and len({item['science_id'] for item in refs['reference_streams']}) == 5,
            'Exactly five paired cross-episode reference streams are required')
    for index, cell in enumerate(campaign['cells']):
        spec = {**read(PROFILE)['checkpoints'][CHECKPOINT_ID], **intervention(index)}
        require(cell['index'] == index and all(cell.get(k) == v for k, v in spec.items()), 'Episode-reset stream/checkpoint changed')
        require(cell['output_dir'] == str(root / 'production' / cell['cell_id']), 'Episode-reset output ownership changed')
        require(cell['science_id'] == canonical_hash(dict(protocol=PROTOCOL, source=campaign['source_commit'],
            matrix=campaign['matrix_sha256'], checkpoint=cell['checkpoint_sha256'], stream=index,
            seeds=cell['seeds'], controller_seed=55)), 'Episode-reset scientific identity changed')
    for key in ('cell_id', 'output_dir', 'science_id'):
        require(len({c[key] for c in campaign['cells']}) == 5, 'Episode-reset stream collision: ' + key)


def reference_snapshots(root, old):
    """Validate complete paired chains and controls; copy compact immutable pins."""
    preservation.validate_campaign(old)
    require(old['profile_sha256'] == sha256(PROFILE), 'Historical checkpoint profile changed')
    snapshot = preservation.publication_snapshot(old)['checkpoints'][CHECKPOINT_ID]
    require(snapshot['complete'] and len(snapshot['streams']) == 5,
            'All five cross-episode reference streams must be complete')
    # Reuse the existing full control validator, including frozen state and dose.
    refs = preservation.reference_snapshots(root, read(old['reference_campaign']))
    require(all(read(refs['reference_summaries'][kind]['path']) == value
                for kind, value in snapshot['reference_summaries'].items()),
            'Historical prior/fresh controls changed')
    refs['reference_files'][old['reference_campaign']] = old['reference_campaign_sha256']
    refs['reference_streams'] = []
    for index, cell in enumerate(old['cells']):
        require(cell['stream_index'] == index and cell['seeds'] == STREAMS[index],
                'Historical cross-episode stream order changed')
        output = Path(cell['output_dir'])
        summary, receipt = preservation.load_stream(old, cell)
        validation, independently_rebuilt = preservation.validate_result(output, cell)
        require(validation == receipt['validation'] and independently_rebuilt == summary,
                'Historical cross-episode stream failed independent validation')
        for filename, key in [('results.json', 'results_sha256'),
                              ('bundle/manifest.json', 'manifest_sha256'),
                              ('chain-summary.json', 'chain_summary_sha256')]:
            refs['reference_files'][str(output / filename)] = receipt[key]
        refs['reference_files'][str(output / 'receipt.json')] = sha256(output / 'receipt.json')
        for relative, digest in receipt['trace_sha256'].items():
            refs['reference_files'][str(output / 'bundle' / relative)] = digest
        path = root / 'references' / CHECKPOINT_ID / f'chain-stream{index}-summary.json'
        write_new(path, summary)
        refs['reference_streams'].append(dict(path=str(path), sha256=sha256(path),
            stream_index=index, science_id=cell['science_id'], source_commit=old['source_commit']))
    return refs


def prepare(root, reference_campaign=BASE / 'campaign.json'):
    root = Path(root).resolve()
    if (root / 'campaign.json').exists():
        raise FileExistsError('An episode-reset campaign already exists here')
    validate_matrix(read(MATRIX))
    commit = source()
    pin = read(PROFILE)['checkpoints'][CHECKPOINT_ID]
    verify_checkpoint(pin)
    references.verify_outer_alpha(pin)
    old = read(reference_campaign)
    refs = reference_snapshots(root, old)
    campaign = dict(schema_version=1, campaign_id=CAMPAIGN_ID, root=str(root), protocol=PROTOCOL,
        source_commit=commit, matrix_sha256=sha256(MATRIX), profile_sha256=sha256(PROFILE),
        reference_campaign=str(Path(reference_campaign).resolve()), reference_campaign_sha256=sha256(reference_campaign),
        streams=STREAMS, controller_seed=55, max_steps=500, execution_action='sample', independent_unit='stream',
        primary_analysis=PRIMARY, checkpoints={CHECKPOINT_ID: refs}, cells=[])
    refs['science_id'] = canonical_hash(dict(protocol=PROTOCOL, source=commit, matrix=campaign['matrix_sha256'],
        checkpoint=pin['checkpoint_sha256'], streams=STREAMS, controller_seed=55))
    for index in range(5):
        cell = {**pin, **intervention(index), 'index': index, 'inventory_path': refs['inventory_path'],
                'inventory_sha256': refs['inventory_sha256']}
        cell['output_dir'] = str(root / 'production' / cell['cell_id'])
        cell['science_id'] = canonical_hash(dict(protocol=PROTOCOL, source=commit, matrix=campaign['matrix_sha256'],
            checkpoint=pin['checkpoint_sha256'], stream=index, seeds=cell['seeds'], controller_seed=55))
        campaign['cells'].append(cell)
    validate_campaign(campaign)
    require(source() == commit, 'Source changed during preparation')
    write_new(root / 'campaign.json', campaign)
    return campaign


def evaluator_command(cell, output, smoke_steps=None):
    command = chain.evaluator_command(cell, output, smoke_steps)
    command[command.index('--matrix') + 1] = str(MATRIX)
    command.remove('--temperature-chain')
    return command


def compact_summary(cell, bundle, seeds, steps):
    # Shared extraction preserves trace diagnostics; only protocol identity differs.
    summary = chain.compact_summary(cell, bundle, seeds, steps)
    summary['protocol'] = PROTOCOL
    return summary


def validate_decisions(summary, cell, seeds, steps):
    require(summary.get('protocol') == PROTOCOL and summary.get('science_id') == cell['science_id']
            and summary.get('cell_id') == cell['cell_id']
            and summary.get('checkpoint_id') == CHECKPOINT_ID
            and summary.get('stream_index') == cell['stream_index']
            and summary.get('seeds') == seeds and summary.get('steps_per_episode') == steps,
            'Wrong episode-reset summary identity')
    expected = [(e, seed, d, e * steps + d) for e, seed in enumerate(seeds) for d in range(steps)]
    require([(r.get('episode_index'), r.get('seed'), r.get('decision_index'), r.get('global_decision_index'))
             for r in summary['rows']] == expected, 'Incomplete, duplicate or reordered episode-reset decisions')
    for row in summary['rows']:
        for key, value in row.items():
            if type(value) in (int, float):
                require(math.isfinite(value), 'Non-finite episode-reset diagnostic: ' + key)
        require(row['stream_index'] == cell['stream_index'] and row['controller_seed'] == 55,
                'Stream RNG identity changed')
        require(row.get('chain_reset') == 0, 'Unexpected cross-episode temperature state')
        require(row['control_seconds'] >= 0, 'Negative control time')
    # mode=carry means decision-local continuity, checkpoint/Adam reset at each seed.
    original.validate_decisions(summary, cell, seeds, steps)
    require([episode['seed'] for episode in summary['episodes']] == seeds, 'Episode ordering changed')
    for index, episode in enumerate(summary['episodes']):
        require(episode['episode_index'] == index and episode['stream_index'] == cell['stream_index']
                and episode['length'] == steps and 'temperature_chain_episode_index' not in episode,
                'Episode-reset identity or coverage changed')
        rows = summary['rows'][index * steps:(index + 1) * steps]
        require(math.isclose(sum(r['reward'] for r in rows), episode['return'], rel_tol=1e-8, abs_tol=1e-7),
                'Episode trace rewards disagree with return')


def validate_result(output, cell, smoke_steps=None):
    from slurm.ambi_early_h3_j8_validation import validate_result as full_validation
    output = Path(output)
    seeds, steps = (cell['seeds'][:2], smoke_steps) if smoke_steps else (cell['seeds'], 500)
    validation = full_validation(output / 'results.json', output / 'bundle', expected_seeds=seeds,
        max_steps=steps, expected_temperature_steps=32, config_overrides=OVERRIDES)
    payload, manifest = read(output / 'results.json'), read(output / 'bundle/manifest.json')
    result, = payload['results']
    require(payload['checkpoint_sha256'] == cell['checkpoint_sha256'] and result['selector'] == cell['selector']
            and result['execution_action'] == 'sample' and not result.get('temperature_chain')
            and not manifest['protocol'].get('temperature_chain'),
            'Wrong checkpoint or episode-reset execution protocol')
    summary = compact_summary(cell, output / 'bundle', seeds, steps)
    validate_decisions(summary, cell, seeds, steps)
    validation.update(lifecycle_validated=True, episode_boundaries_validated=len(seeds)-1,
        chain_boundaries_validated=0, temperature_steps_per_decision=32, protocol=PROTOCOL)
    return validation, summary


def load_stream(campaign, cell, output=None, smoke_steps=None):
    output = Path(output or cell['output_dir'])
    receipt = read(output / 'receipt.json')
    require(receipt.get('validated') is True and receipt.get('smoke') is (smoke_steps is not None)
            and receipt.get('source_commit') == campaign['source_commit'] and receipt.get('protocol') == PROTOCOL
            and receipt.get('science_id') == cell['science_id'] and receipt.get('cell_id') == cell['cell_id']
            and receipt.get('checkpoint_sha256') == cell['checkpoint_sha256']
            and receipt['validation'].get('passed') is True and receipt['validation'].get('lifecycle_validated') is True,
            'Unvalidated or mismatched episode-reset receipt')
    for path, key in [('results.json', 'results_sha256'), ('bundle/manifest.json', 'manifest_sha256'),
                      ('chain-summary.json', 'chain_summary_sha256')]:
        require(sha256(output / path) == receipt[key], 'Episode-reset artifact changed: ' + path)
    require(trace_hashes(output / 'bundle') == receipt['trace_sha256'], 'Episode-reset trace changed')
    summary = read(output / 'chain-summary.json')
    validate_decisions(summary, cell, cell['seeds'][:2] if smoke_steps else cell['seeds'], smoke_steps or 500)
    return summary, receipt


def run(campaign_path, index, smoke_steps=None, smoke_root=None):
    campaign_path = Path(campaign_path).resolve()
    campaign = read(campaign_path)
    validate_campaign(campaign)
    require(type(index) is int and index in range(5), 'Invalid episode-reset stream index')
    require(source() == campaign['source_commit'], 'Wrong or dirty source checkout')
    cell = campaign['cells'][index]
    verify_checkpoint(cell)
    require(sha256(cell['inventory_path']) == cell['inventory_sha256'], 'Checkpoint inventory changed')
    require(smoke_steps is not None or smoke_root is None, 'Alternate root is for smoke only')
    if smoke_root is not None:
        smoke_root = Path(smoke_root).resolve()
        require(smoke_root.is_relative_to(Path(campaign['root'])), 'Smoke root escapes campaign')
    output = (Path(smoke_root or Path(campaign['root']) / 'smoke') / f"{cell['cell_id']}-steps{smoke_steps}"
              if smoke_steps else Path(cell['output_dir']))
    command = evaluator_command(cell, output, smoke_steps)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'started.json', dict(command=command, job_id=os.environ.get('SLURM_JOB_ID'),
        cell_id=cell['cell_id'], source_commit=campaign['source_commit']))
    start, before = time.perf_counter(), resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        subprocess.run(command, cwd=REPO, check=True)
        compute, after = time.perf_counter()-start, resource.getrusage(resource.RUSAGE_CHILDREN)
        validation, summary = validate_result(output, cell, smoke_steps)
        write_new(output / 'chain-summary.json', summary)
        require(source() == campaign['source_commit'], 'Source changed during evaluation')
        cpu = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
        receipt = dict(validated=True, smoke=smoke_steps is not None, source_commit=campaign['source_commit'],
            protocol=PROTOCOL, cell_id=cell['cell_id'], science_id=cell['science_id'], checkpoint_id=CHECKPOINT_ID,
            checkpoint_sha256=cell['checkpoint_sha256'], stream_index=index, validation=validation,
            elapsed_seconds=time.perf_counter()-start, child_elapsed_seconds=compute,
            cpu_threads=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')), child_cpu_seconds=cpu,
            child_cpu_cores_mean=cpu/compute, child_maxrss_kib=after.ru_maxrss,
            results_sha256=sha256(output / 'results.json'), manifest_sha256=sha256(output / 'bundle/manifest.json'),
            chain_summary_sha256=sha256(output / 'chain-summary.json'), trace_sha256=trace_hashes(output / 'bundle'))
        write_new(output / 'receipt.json', receipt)
        return receipt
    except BaseException as exc:
        write_new(output / 'failure.json', dict(exception_type=type(exc).__name__, message=str(exc)))
        raise


def publication_snapshot(campaign):
    campaign = read(campaign) if isinstance(campaign, (str, Path)) else campaign
    validate_campaign(campaign)
    require(sha256(campaign['reference_campaign']) == campaign['reference_campaign_sha256'], 'Historical campaign changed')
    reference = campaign['checkpoints'][CHECKPOINT_ID]
    chain.verify_files(reference['reference_files'])
    summaries = {}
    for kind, item in reference['reference_summaries'].items():
        require(sha256(item['path']) == item['sha256'], 'Episode-reset reference summary changed')
        summaries[kind] = read(item['path'])
    reference_streams = []
    for item in reference['reference_streams']:
        require(sha256(item['path']) == item['sha256'], 'Cross-episode reference summary changed')
        summary = read(item['path'])
        require(summary['stream_index'] == item['stream_index'] and summary['science_id'] == item['science_id'],
                'Cross-episode reference identity changed')
        reference_streams.append(summary)
    entry = dict(reference_summaries=summaries, reference_streams=reference_streams, streams=[], status=[],
                 expected_streams=5, science_id=reference['science_id'])
    for cell in campaign['cells']:
        output = Path(cell['output_dir'])
        status = dict(cell_id=cell['cell_id'], index=cell['index'], stream_index=cell['stream_index'])
        if (output / 'receipt.json').exists():
            try:
                summary, _ = load_stream(campaign, cell)
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
    return dict(campaign=campaign, checkpoints={CHECKPOINT_ID: entry}, complete=entry['complete'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reference-campaign', type=Path, default=BASE / 'campaign.json')
    p = sub.add_parser('run'); p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--index', type=int, choices=range(5), required=True)
    p.add_argument('--smoke-steps', type=int, choices=(4, 8)); p.add_argument('--smoke-root', type=Path)
    args = parser.parse_args()
    result = (prepare(args.root, args.reference_campaign) if args.command == 'prepare'
              else run(args.campaign, args.index, args.smoke_steps, args.smoke_root))
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
