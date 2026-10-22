"""Check that temperature carry preserves useful sampled H3/J8 improvement.

Five independent MPPI500k chains, each three complete episodes. The historical
sampled prior and fresh-alpha SAC controls are revalidated and reused. Only
temperature and its Adam state persist; no new reference evaluations are run.
"""
from __future__ import annotations

import argparse
import json
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

read, write_new, sha256 = chain.read, chain.write_new, chain.sha256
source, require, verify_checkpoint = chain.source, chain.require, chain.verify_checkpoint
trace_hashes, canonical_hash = chain.trace_hashes, chain.canonical_hash
validate_result, compact_summary, validate_decisions, load_stream = (
    chain.validate_result, chain.compact_summary, chain.validate_decisions, chain.load_stream)
MATRIX = REPO / 'configs/research/ambi_alpha_preservation.json'
PROFILE = REPO / references.PROFILE
BASE = Path('/oscar/scratch/rgao48/ambi/entropy-kl-mppi500k/20261020-v2/episodes')
PROTOCOL, STREAMS, SEEDS = chain.PROTOCOL, chain.STREAMS, chain.SEEDS
CAMPAIGN_ID, CHECKPOINT_ID = 'alpha-preservation-20261022', 'mppi_aux_500k'
PRIMARY = dict(episode_index=2, independent_unit='stream',
               outcomes=['paired_gain_vs_prior', 'paired_delta_vs_reset'],
               interpretation='positive delta_vs_reset means carry improves on fresh alpha')


def intervention(stream_index):
    require(type(stream_index) is int and stream_index in range(5), 'Unknown preservation stream')
    pin = read(PROFILE)['checkpoints'][CHECKPOINT_ID]
    return dict(checkpoint_id=CHECKPOINT_ID, stream_index=stream_index, seeds=STREAMS[stream_index],
        cell_id=f'{CHECKPOINT_ID}__stream{stream_index}', selector='alpha_preservation/run',
        mode='run_carry', outer_alpha=pin['outer_alpha'], expected_temperature_steps=32)


def validate_matrix(matrix):
    baseline = read(REPO / references.MATRIX)
    require(matrix['base_alg_config'] == 'checkpoint', 'Expected checkpoint-derived networks')
    require(matrix['shared_alg_params'] == {**baseline['shared_alg_params'],
        'inner_temperature_scope': 'run', 'inner_temperature_optimizer_scope': 'run',
        'inner_temperature_updates_per_actor': 1, 'inner_sac_return_estimator': 'one_step'},
        'Only temperature lifetime may change from the sampled fresh-alpha baseline')
    require(matrix['evaluation'] == {**baseline['evaluation'], 'default_presets': ['alpha_preservation/run']},
            'Preservation evaluation protocol changed')
    require(matrix['temperature_chain'] == dict(protocol=PROTOCOL, streams=5, episodes_per_stream=3,
        ordered_seeds=STREAMS, independent_unit='stream', checkpoints=[CHECKPOINT_ID]), 'Preservation chain layout changed')
    group = matrix['comparisons'].get('alpha_preservation', {})
    require(set(matrix['comparisons']) == {'alpha_preservation'} and group.get('reference') == 'run'
            and set(group.get('variants', {})) == {'run'}
            and group['variants']['run']['alg_params'] == {}
            and group['variants']['run']['evaluation'] == {'execution_action': 'sample'}, 'Unexpected preservation arm')


def validate_campaign(campaign):
    validate_matrix(read(MATRIX))
    require(all(campaign.get(k) == v for k, v in dict(schema_version=1, campaign_id=CAMPAIGN_ID,
        protocol=PROTOCOL, streams=STREAMS, controller_seed=55, max_steps=500, execution_action='sample',
        independent_unit='stream', primary_analysis=PRIMARY).items()), 'Preservation campaign protocol changed')
    require(campaign['matrix_sha256'] == sha256(MATRIX) and campaign['profile_sha256'] == sha256(PROFILE),
            'Preservation configuration changed')
    root = Path(campaign['root'])
    require(root.is_absolute() and set(campaign['checkpoints']) == {CHECKPOINT_ID}
            and len(campaign['cells']) == 5, 'Preservation needs exactly five MPPI500k streams')
    require(set(campaign['checkpoints'][CHECKPOINT_ID]['reference_summaries']) == {'prior', 'reset'},
            'Only the existing prior and fresh-alpha controls are allowed')
    for index, cell in enumerate(campaign['cells']):
        spec = {**read(PROFILE)['checkpoints'][CHECKPOINT_ID], **intervention(index)}
        require(cell['index'] == index and all(cell.get(k) == v for k, v in spec.items()), 'Preservation stream/checkpoint changed')
        require(cell['output_dir'] == str(root / 'production' / cell['cell_id']), 'Preservation output ownership changed')
        require(cell['science_id'] == canonical_hash(dict(protocol=PROTOCOL, source=campaign['source_commit'],
            matrix=campaign['matrix_sha256'], checkpoint=cell['checkpoint_sha256'], stream=index,
            seeds=cell['seeds'], controller_seed=55)), 'Preservation scientific identity changed')
    for key in ('cell_id', 'output_dir', 'science_id'):
        require(len({c[key] for c in campaign['cells']}) == 5, 'Preservation stream collision: ' + key)


def reference_snapshots(root, old):
    """Revalidate complete original results, then pin copies without changing them."""
    references.validate_campaign(old)
    require(old['matrix_sha256'] == sha256(REPO / references.MATRIX)
            and old['profile_sha256'] == sha256(PROFILE), 'Historical control configuration changed')
    pin = read(PROFILE)['checkpoints'][CHECKPOINT_ID]
    files, summaries, validations = {}, {}, {}
    for kind, mode in [('prior', 'prior'), ('reset', 'entropy')]:
        cell, = [c for c in old['cells'] if c['mode'] == mode]
        require(all(cell.get(k) == v for k, v in pin.items()), 'Control checkpoint differs from preservation checkpoint')
        output = Path(cell['output_dir'])
        receipt = read(output / 'receipt.json')
        expected = dict(validated=True, smoke=False, source_commit=old['source_commit'], **references.intervention(mode))
        require(all(receipt.get(k) == v for k, v in expected.items()), 'Control receipt is not validated complete production')
        for filename, hash_key in [('results.json', 'results_sha256'), ('bundle/manifest.json', 'manifest_sha256')]:
            path = output / filename
            require(sha256(path) == receipt[hash_key], 'Historical control artifact changed')
            files[str(path)] = receipt[hash_key]
        require(trace_hashes(output / 'bundle') == receipt['trace_sha256'], 'Historical control trace changed')
        validation = references.validate_episode_result(output, cell)
        require(validation == receipt['validation'], 'Historical control failed independent revalidation')
        payload = read(output / 'results.json')['results'][0]
        cfg = payload['resolved_config']
        for scope in ('actor', 'critic', 'replay', 'temperature', 'actor_optimizer', 'critic_optimizer', 'temperature_optimizer'):
            require(cfg['inner_' + scope + '_scope'] == 'action', 'Historical control contains persistent state')
        require(cfg.get('inner_temperature_updates_per_actor', 1) == 1
                and cfg.get('inner_sac_return_estimator', 'one_step') == 'one_step'
                and cfg['inner_outer_policy_kl_coef'] == 0., 'Historical control has an extra intervention')
        if kind == 'reset':
            summary = original.compact_summary_from_bundle({**cell, 'mode': 'reset'}, output / 'bundle')
            require([(r['seed'], r['decision_index']) for r in summary['rows']]
                    == [(seed, decision) for seed in SEEDS for decision in range(500)], 'Incomplete reset decision curves')
        else:
            summary = dict(schema_version=1, checkpoint_id=CHECKPOINT_ID, mode='prior', rows=[],
                           episodes=payload['episodes'], provenance=dict(bundle=str(output / 'bundle'),
                               manifest_sha256=receipt['manifest_sha256'], checkpoint_sha256=pin['checkpoint_sha256']))
        summary['provenance'].update(source_run=pin['source_run'], source_commit=old['source_commit'],
            results_sha256=receipt['results_sha256'], receipt_sha256=sha256(output / 'receipt.json'))
        path = root / 'references' / CHECKPOINT_ID / (kind + '-summary.json')
        write_new(path, summary)
        summaries[kind] = dict(path=str(path), sha256=sha256(path))
        files[str(output / 'receipt.json')] = sha256(output / 'receipt.json')
        validations[kind] = validation
    inventory = old['cells'][0]['inventory_path']
    require(sha256(inventory) == old['cells'][0]['inventory_sha256'], 'Historical checkpoint inventory changed')
    files[inventory] = sha256(inventory)
    return dict(checkpoint_id=CHECKPOINT_ID, checkpoint_sha256=pin['checkpoint_sha256'],
                reference_files=files, reference_summaries=summaries, reference_validation=validations,
                inventory_path=inventory, inventory_sha256=files[inventory])


def prepare(root, reference_campaign=BASE / 'campaign.json'):
    root = Path(root).resolve()
    if (root / 'campaign.json').exists():
        raise FileExistsError('A preservation campaign already exists here')
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
    return command


def run(campaign_path, index, smoke_steps=None, smoke_root=None):
    campaign_path = Path(campaign_path).resolve()
    campaign = read(campaign_path)
    validate_campaign(campaign)
    require(type(index) is int and index in range(5), 'Invalid preservation stream index')
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
        require(sha256(item['path']) == item['sha256'], 'Preservation reference summary changed')
        summaries[kind] = read(item['path'])
    entry = dict(reference_summaries=summaries, streams=[], status=[], expected_streams=5, science_id=reference['science_id'])
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
