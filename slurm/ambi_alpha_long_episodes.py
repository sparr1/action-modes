"""Ten sampled episodes comparing episode-reset and persistent temperature.

Both arms are new, matched full histories. Older short histories supply only
checkpoint provenance and measured runtime, never invented long-run controls.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
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
from slurm import ambi_alpha_episode_reset as reset
from slurm import ambi_alpha_preservation as carry

read, write_new, sha256 = chain.read, chain.write_new, chain.sha256
source, require, verify_checkpoint = chain.source, chain.require, chain.verify_checkpoint
trace_hashes, canonical_hash = chain.trace_hashes, chain.canonical_hash
MATRIX = REPO / 'configs/research/ambi_alpha_long_episodes.json'
PROFILE, CHECKPOINT_ID = reset.PROFILE, reset.CHECKPOINT_ID
BASE = Path('/oscar/scratch/rgao48/ambi/alpha-preservation/20261022-b315ca33')
PROTOCOL, CAMPAIGN_ID = 'alpha-long-episodes-v1', 'alpha-long-episodes-20261022'
SEEDS = list(range(101, 111))
STREAMS = [[101 + (i + d) % 5 for d in range(5)] + [106 + (i + d) % 5 for d in range(5)]
           for i in range(5)]
MODES = ('episode_reset', 'run_carry')
OPTIONAL_DIAGNOSTICS = dict(root_prior_kl='inner_final_outer_policy_kl',
    actor_pre_tanh_abs_mean='inner_actor_pre_tanh_abs_mean', actor_pre_tanh_abs_max='inner_actor_pre_tanh_abs_max',
    actor_pre_tanh_abs_ge_7p6_fraction='inner_actor_pre_tanh_abs_ge_7p6_fraction',
    actor_action_exact_saturation_fraction='inner_actor_action_exact_saturation_fraction',
    actor_q_mean_all_minus_min_all='inner_actor_q_mean_all_minus_min_all',
    critic_td_error_abs_mean='inner_td_error_abs_mean')
PRIMARY = dict(episode_index=9, independent_unit='stream',
    outcomes=['carry_minus_reset_episode_10', 'carry_minus_reset_trajectory', 'late_five_minus_first_five_advantage'],
    late_contrast='mean(carry-reset, episodes 6..10)-mean(carry-reset, episodes 1..5)',
    interpretation='negative late contrast means the relative carry advantage decreases in later episodes')


def intervention(mode, stream_index):
    require(mode in MODES and type(stream_index) is int and stream_index in range(5), 'Unknown execution cell')
    pin = read(PROFILE)['checkpoints'][CHECKPOINT_ID]
    return dict(checkpoint_id=CHECKPOINT_ID, stream_index=stream_index, seeds=STREAMS[stream_index],
        cell_id=f'{CHECKPOINT_ID}__{mode}__stream{stream_index}', selector='alpha_long_episodes/' + mode,
        mode=mode, arm=mode, execution_action='sample', outer_alpha=pin['outer_alpha'], expected_temperature_steps=32,
        temperature_scope='episode' if mode == 'episode_reset' else 'run')


def lifecycle_cell(cell):
    """Explicit adapter for existing lifecycle validators, independent of execution."""
    require(cell['mode'] in MODES, 'Unknown execution lifecycle')
    return {**cell, 'mode': 'carry' if cell['mode'] == 'episode_reset' else 'run_carry'}


def config_overrides(cell):
    return reset.OVERRIDES if cell['mode'] == 'episode_reset' else chain.OVERRIDES


def validate_matrix(matrix):
    baseline = read(reset.MATRIX)
    require(matrix['base_alg_config'] == 'checkpoint', 'Expected checkpoint-derived networks')
    require(matrix['shared_alg_params'] == baseline['shared_alg_params'],
            'Inner learning and sampled behavior must match the shorter controls')
    require(matrix['evaluation'] == {**baseline['evaluation'], 'seeds': SEEDS,
        'default_presets': ['alpha_long_episodes/' + mode for mode in MODES]}, 'Long sampled protocol changed')
    require(matrix['temperature_chain'] == dict(protocol=PROTOCOL, streams=5, episodes_per_stream=10,
        ordered_seeds=STREAMS, independent_unit='stream', checkpoints=[CHECKPOINT_ID], modes=list(MODES)),
        'Ten-episode layout changed')
    group = matrix['comparisons'].get('alpha_long_episodes', {})
    require(set(matrix['comparisons']) == {'alpha_long_episodes'} and group.get('reference') == 'episode_reset'
            and set(group.get('variants', {})) == set(MODES), 'Unexpected long-history arm')
    for mode in MODES:
        expected = {} if mode == 'episode_reset' else {'inner_temperature_scope': 'run',
                                                      'inner_temperature_optimizer_scope': 'run'}
        require(group['variants'][mode]['alg_params'] == expected
                and group['variants'][mode]['evaluation'] == {'execution_action': 'sample'},
                'Only temperature lifetime may differ between arms')


def science_id(campaign, cell):
    return canonical_hash(dict(protocol=PROTOCOL, source=campaign['source_commit'], matrix=campaign['matrix_sha256'],
        checkpoint=cell['checkpoint_sha256'], mode=cell['mode'], stream=cell['stream_index'],
        seeds=cell['seeds'], controller_seed=55))


def validate_campaign(campaign):
    validate_matrix(read(MATRIX))
    require(all(campaign.get(key) == value for key, value in dict(schema_version=1, campaign_id=CAMPAIGN_ID,
        protocol=PROTOCOL, streams=STREAMS, controller_seed=55, max_steps=500, execution_action='sample',
        independent_unit='stream', primary_analysis=PRIMARY, modes=list(MODES), episodes_per_stream=10).items()),
        'Long-history campaign protocol changed')
    require(campaign['matrix_sha256'] == sha256(MATRIX) and campaign['profile_sha256'] == sha256(PROFILE),
            'Long-history configuration changed')
    root = Path(campaign['root'])
    require(root.is_absolute() and set(campaign['checkpoints']) == {CHECKPOINT_ID} and len(campaign['cells']) == 10,
            'Long-history campaign requires ten MPPI500k histories')
    refs = campaign['checkpoints'][CHECKPOINT_ID]
    runtime = refs['observed_full_history_receipts']
    require(len(runtime) == 5 and len({item['path'] for item in runtime}) == 5
            and all(refs['reference_files'].get(item['path']) == item['sha256'] for item in runtime),
            'Five verified historical timing receipts are required')
    for index, cell in enumerate(campaign['cells']):
        spec = {**read(PROFILE)['checkpoints'][CHECKPOINT_ID], **intervention(MODES[index // 5], index % 5)}
        require(cell['index'] == index and all(cell.get(key) == value for key, value in spec.items()),
                'Long-history arm ordering or checkpoint changed')
        require(cell['output_dir'] == str(root / 'production' / cell['cell_id'])
                and cell['science_id'] == science_id(campaign, cell), 'Long-history scientific identity changed')
    for key in ('cell_id', 'output_dir', 'science_id'):
        require(len({cell[key] for cell in campaign['cells']}) == 10, 'Long-history stream collision: ' + key)



def with_optional_diagnostics(summary, bundle):
    """Copy already-recorded finite metrics; do not run policy probes or impute."""
    summary, bundle = deepcopy(summary), Path(bundle).resolve()
    rows = {(row['seed'], row['decision_index']): row for row in summary['rows']}
    counts = {key: 0 for key in OPTIONAL_DIAGNOSTICS}
    for relative in read(bundle / 'manifest.json')['runs'][0]['trace_files']:
        path = (bundle / relative).resolve(strict=True)
        require(path.is_relative_to(bundle), 'Optional diagnostic trace escapes bundle')
        with gzip.open(path, 'rt') as source:
            for line in source:
                event = json.loads(line)
                if event['phase'] != 'decision':
                    continue
                key = int(event['episode_id'].removeprefix('seed-')), event['decision_index']
                require(key in rows, 'Optional diagnostic lacks a validated decision')
                for name, metric in OPTIONAL_DIAGNOSTICS.items():
                    value = event['metrics'].get('decision/' + metric)
                    if value is None:
                        continue
                    require(type(value) in (int, float) and math.isfinite(value), 'Non-finite optional diagnostic: ' + name)
                    rows[key][name] = value
                    counts[name] += 1
    summary['optional_diagnostic_counts'] = counts
    return summary

def reference_snapshots(root, old):
    """Pin old short histories only for source validation and runtime sizing."""
    carry.validate_campaign(old)
    require(old['profile_sha256'] == sha256(PROFILE), 'Historical checkpoint profile changed')
    pin = read(PROFILE)['checkpoints'][CHECKPOINT_ID]
    inventory = old['cells'][0]['inventory_path']
    require(sha256(inventory) == old['cells'][0]['inventory_sha256'], 'Historical checkpoint inventory changed')
    refs = dict(checkpoint_id=CHECKPOINT_ID, checkpoint_sha256=pin['checkpoint_sha256'],
        inventory_path=inventory, inventory_sha256=sha256(inventory),
        reference_files={inventory: sha256(inventory)}, observed_full_history_receipts=[],
        historical_source_commit=old['source_commit'], historical_episodes_per_stream=3,
        historical_use='checkpoint provenance and measured runtime only; no performance controls')
    for index, cell in enumerate(old['cells']):
        require(cell['stream_index'] == index and cell['seeds'] == STREAMS[index][:3]
                and all(cell.get(key) == value for key, value in pin.items()),
                'Historical stream prefix or checkpoint changed')
        output = Path(cell['output_dir'])
        summary, receipt = carry.load_stream(old, cell)
        validation, rebuilt = carry.validate_result(output, cell)
        require(validation == receipt['validation'] and rebuilt == summary,
                'Historical timing source failed independent full validation')
        elapsed = receipt['elapsed_seconds']
        require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0,
                'Historical timing must be positive and finite')
        for filename, key in [('results.json', 'results_sha256'), ('bundle/manifest.json', 'manifest_sha256'),
                              ('chain-summary.json', 'chain_summary_sha256')]:
            refs['reference_files'][str(output / filename)] = receipt[key]
        path = str(output / 'receipt.json')
        refs['reference_files'][path] = sha256(path)
        refs['observed_full_history_receipts'].append(dict(path=path, sha256=sha256(path)))
        for relative, digest in receipt['trace_sha256'].items():
            refs['reference_files'][str(output / 'bundle' / relative)] = digest
    return refs


def prepare(root, reference_campaign=BASE / 'campaign.json'):
    root = Path(root).resolve()
    if (root / 'campaign.json').exists():
        raise FileExistsError('An long-history campaign already exists here')
    validate_matrix(read(MATRIX))
    commit = source()
    pin = read(PROFILE)['checkpoints'][CHECKPOINT_ID]
    verify_checkpoint(pin)
    reset.references.verify_outer_alpha(pin)
    refs = reference_snapshots(root, read(reference_campaign))
    campaign = dict(schema_version=1, campaign_id=CAMPAIGN_ID, root=str(root), protocol=PROTOCOL,
        source_commit=commit, matrix_sha256=sha256(MATRIX), profile_sha256=sha256(PROFILE),
        reference_campaign=str(Path(reference_campaign).resolve()), reference_campaign_sha256=sha256(reference_campaign),
        streams=STREAMS, controller_seed=55, max_steps=500, execution_action='sample', independent_unit='stream',
        primary_analysis=PRIMARY, modes=list(MODES), episodes_per_stream=10,
        checkpoints={CHECKPOINT_ID: refs}, cells=[])
    refs['science_id'] = canonical_hash(dict(protocol=PROTOCOL, source=commit, matrix=campaign['matrix_sha256'],
        checkpoint=pin['checkpoint_sha256'], streams=STREAMS, modes=MODES, controller_seed=55))
    for mode in MODES:
        for stream_index in range(5):
            cell = {**pin, **intervention(mode, stream_index), 'index': len(campaign['cells']),
                    'inventory_path': refs['inventory_path'], 'inventory_sha256': refs['inventory_sha256']}
            cell['output_dir'] = str(root / 'production' / cell['cell_id'])
            cell['science_id'] = science_id(campaign, cell)
            campaign['cells'].append(cell)
    validate_campaign(campaign)
    require(source() == commit, 'Source changed during execution preparation')
    write_new(root / 'campaign.json', campaign)
    return campaign


def evaluator_command(cell, output, smoke_steps=None):
    command = chain.evaluator_command(cell, output, smoke_steps)
    command[command.index('--matrix') + 1] = str(MATRIX)
    if cell['mode'] == 'episode_reset':
        command.remove('--temperature-chain')
    else:
        require(cell['mode'] == 'run_carry', 'Unknown execution arm')
    return command


def compact_summary(cell, bundle, seeds, steps):
    owner = reset if cell['mode'] == 'episode_reset' else chain
    summary = owner.compact_summary(lifecycle_cell(cell), bundle, seeds, steps)
    summary.update(long_protocol=PROTOCOL, arm=cell['mode'], execution_action='sample')
    return with_optional_diagnostics(summary, bundle)


def validate_decisions(summary, cell, seeds, steps):
    require(summary.get('long_protocol') == PROTOCOL and summary.get('arm') == cell['mode']
            and summary.get('execution_action') == 'sample' and summary.get('cell_id') == cell['cell_id']
            and summary.get('checkpoint_id') == CHECKPOINT_ID and summary.get('seeds') == seeds
            and summary.get('steps_per_episode') == steps, 'Wrong long-history summary identity')
    owner = reset if cell['mode'] == 'episode_reset' else chain
    owner.validate_decisions(summary, lifecycle_cell(cell), seeds, steps)


def validate_result(output, cell, smoke_steps=None):
    from slurm.ambi_early_h3_j8_validation import validate_result as full_validation
    output = Path(output)
    seeds, steps = (cell['seeds'][:2], smoke_steps) if smoke_steps else (cell['seeds'], 500)
    validation = full_validation(output / 'results.json', output / 'bundle', expected_seeds=seeds,
        max_steps=steps, expected_temperature_steps=32, config_overrides=config_overrides(cell))
    payload, manifest = read(output / 'results.json'), read(output / 'bundle/manifest.json')
    result, = payload['results']
    require(payload['checkpoint_sha256'] == cell['checkpoint_sha256'] and result['selector'] == cell['selector']
            and result['execution_action'] == 'sample', 'Wrong checkpoint or real-action execution mode')
    if cell['mode'] == 'run_carry':
        require(result.get('temperature_chain') is True and result.get('temperature_chain_seeds') == seeds
                and result.get('temperature_chain_kind') == chain.PROTOCOL
                and manifest['protocol'].get('temperature_chain') == dict(kind=chain.PROTOCOL, ordered_environment_seeds=seeds),
                'Missing cross-episode carry execution protocol')
    else:
        require(not result.get('temperature_chain') and not manifest['protocol'].get('temperature_chain'),
                'Episode-reset execution cannot retain cross-episode state')
    summary = compact_summary(cell, output / 'bundle', seeds, steps)
    validate_decisions(summary, cell, seeds, steps)
    validation.update(lifecycle_validated=True, temperature_steps_per_decision=32, protocol=PROTOCOL,
        execution_action='sample', arm=cell['mode'],
        episode_boundaries_validated=(len(seeds)-1 if cell['mode'] == 'episode_reset' else 0),
        chain_boundaries_validated=(len(seeds)-1 if cell['mode'] == 'run_carry' else 0))
    return validation, summary


def load_stream(campaign, cell, output=None, smoke_steps=None):
    output = Path(output or cell['output_dir'])
    receipt = read(output / 'receipt.json')
    require(receipt.get('validated') is True and receipt.get('smoke') is (smoke_steps is not None)
            and receipt.get('source_commit') == campaign['source_commit'] and receipt.get('protocol') == PROTOCOL
            and receipt.get('science_id') == cell['science_id'] and receipt.get('cell_id') == cell['cell_id']
            and receipt.get('checkpoint_sha256') == cell['checkpoint_sha256']
            and receipt.get('arm') == cell['mode'] and receipt.get('execution_action') == 'sample'
            and receipt['validation'].get('passed') is True and receipt['validation'].get('lifecycle_validated') is True,
            'Unvalidated or mismatched execution receipt')
    for path, key in [('results.json', 'results_sha256'), ('bundle/manifest.json', 'manifest_sha256'),
                      ('chain-summary.json', 'chain_summary_sha256')]:
        require(sha256(output / path) == receipt[key], 'Long-history artifact changed: ' + path)
    require(trace_hashes(output / 'bundle') == receipt['trace_sha256'], 'Long-history trace changed')
    summary = read(output / 'chain-summary.json')
    validate_decisions(summary, cell, cell['seeds'][:2] if smoke_steps else cell['seeds'], smoke_steps or 500)
    return summary, receipt


def run(campaign_path, index, smoke_steps=None, smoke_root=None):
    campaign_path = Path(campaign_path).resolve()
    campaign = read(campaign_path)
    validate_campaign(campaign)
    require(type(index) is int and index in range(10), 'Invalid execution stream index')
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
            checkpoint_sha256=cell['checkpoint_sha256'], stream_index=cell['stream_index'], arm=cell['mode'],
            execution_action='sample', validation=validation, elapsed_seconds=time.perf_counter()-start,
            child_elapsed_seconds=compute, cpu_threads=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')),
            child_cpu_seconds=cpu, child_cpu_cores_mean=cpu/compute, child_maxrss_kib=after.ru_maxrss,
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
    entry = dict(streams={mode: [] for mode in MODES}, status=[], expected_streams=5,
        expected_new_streams=10, episodes_per_stream=10, science_id=reference['science_id'],
        historical_provenance=dict(source_commit=reference['historical_source_commit'],
            use=reference['historical_use'], timing_receipts=reference['observed_full_history_receipts']))
    for cell in campaign['cells']:
        output = Path(cell['output_dir'])
        status = dict(cell_id=cell['cell_id'], index=cell['index'], stream_index=cell['stream_index'],
                      arm=cell['mode'], mode=cell['mode'])
        if (output / 'receipt.json').exists():
            try:
                summary, _ = load_stream(campaign, cell)
                entry['streams'][cell['mode']].append(summary)
                status.update(state='complete', receipt_sha256=sha256(output / 'receipt.json'))
            except (ValueError, KeyError, FileNotFoundError) as exc:
                status.update(state='validation_failed', message=str(exc))
        elif (output / 'failure.json').exists():
            status.update(state='failed', failure=read(output / 'failure.json'))
        else:
            status['state'] = 'running' if (output / 'started.json').exists() else 'pending'
        entry['status'].append(status)
    entry['complete'] = all(len(values) == 5 for values in entry['streams'].values())
    return dict(campaign=campaign, checkpoints={CHECKPOINT_ID: entry}, complete=entry['complete'])


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
