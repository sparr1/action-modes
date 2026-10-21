"""Matched Retrace KL-only and entropy-plus-KL frozen MPPI-backbone evaluations.

GPU workers stay offline and retain full traces on Oscar. A separate CPU
publisher validates immutable receipts and appends two persistent method curves.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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
from slurm.ambi_early_h3_j8_campaign import read, write_new, sha256, verify_checkpoint
from slurm.ambi_entropy_kl_mppi500k import source, verify_outer_alpha, _constant
from slurm.ambi_entropy_kl_episodes import trace_hashes
from slurm.ambi_mppi_checkpoint_comparison import reference_pins, require

MATRIX = REPO / 'configs/research/ambi_retrace_kl_mppi.json'
PROFILE = REPO / 'configs/research/ambi_retrace_kl_mppi_profile.json'
BASELINE_MATRIX = REPO / 'configs/research/ambi_early_h3_j8.json'
CAMPAIGN_ID = 'retrace-kl-mppi-20261021'
MODES = ('retrace_kl', 'retrace_both')
SEEDS = [101, 102, 103, 104, 105]
ALPHAS = {400000: .013081684708595276, 500000: .023066775873303413}
RETRACE = dict(inner_sac_return_estimator='retrace', inner_retrace_lambda=1.,
               inner_retrace_batch_trajectories=86)
RETRACE_DIAGNOSTICS = ('inner_retrace_trace_coefficient_mean',
    'inner_retrace_effective_trace_length', 'inner_retrace_correction_abs_mean')


def intervention(mode, step):
    require(mode in MODES and step in ALPHAS, 'Unknown Retrace comparison cell')
    entropy = mode == 'retrace_both'
    return dict(mode=mode, step=step, selector=f'retrace_kl_mppi/{mode}_{step}',
        cell_id=f'{mode}_{step}', outer_alpha=ALPHAS[step], beta=ALPHAS[step],
        entropy_enabled=entropy, expected_temperature_steps=32 if entropy else 0)


def overrides(cell):
    return dict(RETRACE, inner_entropy_enabled=cell['entropy_enabled'],
        inner_outer_policy_kl_coef=cell['beta'],
        inner_temperature_mode='auto' if cell['entropy_enabled'] else 'inherit_outer',
        inner_temperature_lr=.0003, sac_actor_loss_scale_mode='none')


def validate_matrix(matrix):
    base = read(BASELINE_MATRIX)
    require(matrix['base_alg_config'] == 'checkpoint'
            and matrix['shared_alg_params'] == {**base['shared_alg_params'], **RETRACE},
            'Shared configuration must preserve the H3/J8 baseline plus exact Retrace settings')
    cells = [intervention(mode, step) for step in ALPHAS for mode in MODES]
    expected_eval = {**base['evaluation'], 'execution_action': 'sample',
                     'default_presets': [c['selector'] for c in cells]}
    require(matrix['evaluation'] == expected_eval, 'Evaluation seeds, execution or probes changed')
    require(set(matrix['comparisons']) == {'retrace_kl_mppi'}, 'Unexpected comparison group')
    group = matrix['comparisons']['retrace_kl_mppi']
    require(group['reference'] == cells[0]['cell_id']
            and set(group['variants']) == {c['cell_id'] for c in cells}, 'Expected exactly four Retrace cells')
    for cell in cells:
        variant = group['variants'][cell['cell_id']]
        require(variant['alg_params'] == dict(inner_outer_policy_kl_coef=cell['beta'],
                    inner_entropy_enabled=cell['entropy_enabled'])
                and variant['evaluation'] == {'execution_action': 'sample'}, 'Wrong objective intervention')


def validate_campaign(campaign):
    profile = read(PROFILE)
    require(campaign['campaign_id'] == CAMPAIGN_ID and campaign['seeds'] == SEEDS
            and campaign['controller_seed'] == 55 and campaign['max_steps'] == 500
            and campaign['execution_action'] == 'sample', 'Campaign protocol changed')
    require(campaign['matrix_sha256'] == sha256(MATRIX) and campaign['profile_sha256'] == sha256(PROFILE),
            'Pinned configuration changed')
    require([r['step'] for r in profile['checkpoints']] == list(ALPHAS), 'Unexpected checkpoint profile')
    expected = [{**pin, **intervention(mode, pin['step'])}
                for pin in profile['checkpoints'] for mode in MODES]
    require(len(campaign['cells']) == len(expected), 'Exactly four cells are required')
    for index, (cell, wanted) in enumerate(zip(campaign['cells'], expected)):
        require(cell['index'] == index and all(cell.get(k) == v for k, v in wanted.items()),
                'Checkpoint, objective or cell order changed')
    for key in ('cell_id', 'run_id', 'run_dir', 'output_dir'):
        require(len({c[key] for c in campaign['cells']}) == 4, 'Cell ownership collision: ' + key)
    require(campaign['existing_curves_profile'] == profile['existing_curves_profile'],
            'Wrong predecessor curve profile')
    require(sha256(campaign['curves_profile']) == campaign['curves_profile_sha256'], 'Extended curve profile changed')


def evaluator_command(cell, output, smoke_steps=None, *, specification_only=False):
    require(smoke_steps is None or (type(smoke_steps) is int and smoke_steps in (8, 32)), 'Invalid smoke length')
    output = Path(output)
    command = [sys.executable, str(REPO / 'evaluate_ambi_checkpoint.py'), '--matrix', str(MATRIX),
        '--checkpoint', cell['checkpoint'], '--metadata', cell['metadata'], '--preset', cell['selector'],
        '--device', 'cuda', '--seeds', *map(str, [101] if smoke_steps else SEEDS),
        '--controller-seed', '55', '--max-steps', str(smoke_steps or 500),
        '--output', str(output / 'results.json'), '--bundle-dir', str(output / 'bundle'),
        '--checkpoint-inventory', cell['inventory_path']]
    # Preflight MUST resolve the same actual prior reference as production.
    if smoke_steps is None:
        command += ['--reference-bundle', cell['reference_bundle']]
    return command


def prepare(root):
    from utils import eval_series as series
    from slurm import ambi_entropy_kl_curves as curves
    root = Path(root).resolve()
    require(not (root / 'campaign.json').exists(), 'Campaign already exists; inspect before retrying')
    validate_matrix(read(MATRIX))
    pin_profile = read(PROFILE)
    predecessor_path = pin_profile['existing_curves_profile']
    predecessor = read(predecessor_path)
    require(set(predecessor['identities']) == {'prior', 'entropy', 'kl', 'both', 'mppi'},
            'Expected the existing five-controller profile')
    # This is read-only for an already-prepared five-curve catalog.
    previous_catalog = series._read(Path(predecessor['root']) / 'catalog.json')
    require({r['mode'] for r in previous_catalog['curves']} == set(predecessor['identities']),
            'Existing catalog/profile methods disagree')
    for row in previous_catalog['curves']:
        series.validate_identity(series.load_run(row['run_dir']), predecessor['identities'][row['mode']])
    profile = deepcopy(predecessor)
    campaign = dict(schema_version=1, campaign_id=CAMPAIGN_ID, source_commit=source(),
        matrix_sha256=sha256(MATRIX), profile_sha256=sha256(PROFILE), seeds=SEEDS,
        controller_seed=55, max_steps=500, execution_action='sample', cells=[],
        existing_curves_profile=predecessor_path, existing_curves_profile_sha256=sha256(predecessor_path))
    for pin in pin_profile['checkpoints']:
        verify_checkpoint(pin)
        verify_outer_alpha(pin)
        reference_pins(pin)
        inventory = root / 'inputs' / f"{pin['step']}.json"
        write_new(inventory, dict(source_run=pin['source_run'], checkpoints=[dict(pin,
            path=pin['checkpoint'], sha256=pin['checkpoint_sha256'])]))
        for mode in MODES:
            cell = dict(pin, **intervention(mode, pin['step']))
            cell.update(index=len(campaign['cells']), inventory_path=str(inventory),
                inventory_sha256=sha256(inventory), output_dir=str(root / 'production' / cell['cell_id']))
            specdir = root / 'specs' / cell['cell_id']
            command = evaluator_command(cell, root / 'specification-only' / cell['cell_id'], specification_only=True)
            subprocess.run(command + ['--eval-series-spec-dir', str(specdir)], cwd=REPO, check=True)
            spec = read(specdir / (cell['selector'].replace('/', '__') + '.json'))
            registry = series.create_run(root / 'publication' / 'runs', spec, CAMPAIGN_ID,
                'ambi-inner-bench', 'rwgao_b-brown-university', 'oscar-rgao48')
            cell.update(run_id=registry['run_id'], run_dir=registry['run_dir'])
            identity = curves.canonical_identity(registry['identity'], mode, cell['outer_alpha'])
            if mode in profile['identities']:
                require(profile['identities'][mode] == identity, 'Retrace checkpoint recipes differ')
            profile['identities'][mode] = identity
            campaign['cells'].append(cell)
    campaign['curves_profile'] = str(root / 'curves-profile.json')
    write_new(campaign['curves_profile'], profile)
    campaign['curves_profile_sha256'] = sha256(campaign['curves_profile'])
    validate_campaign(campaign)
    require(source() == campaign['source_commit'], 'Source changed during preparation')
    write_new(root / 'campaign.json', campaign)
    catalog = curves.register_campaign(profile, campaign)
    require(catalog['curves'][:5] == previous_catalog['curves'], 'Existing curve identities changed')
    return dict(campaign=str(root / 'campaign.json'), catalog=catalog)


def validate_result(output, cell, smoke_steps=None):
    from slurm.ambi_early_h3_j8_validation import validate_result as validate_full_trace
    output = Path(output)
    seeds, steps = ([101], smoke_steps) if smoke_steps is not None else (SEEDS, 500)
    validation = validate_full_trace(output / 'results.json', output / 'bundle',
        expected_seeds=seeds, max_steps=steps, expected_temperature_steps=cell['expected_temperature_steps'],
        config_overrides=overrides(cell))
    payload = read(output / 'results.json')
    result, = payload['results']
    require(payload['checkpoint_sha256'] == cell['checkpoint_sha256']
            and result['selector'] == cell['selector'] and result['execution_action'] == 'sample',
            'Checkpoint, selector or execution changed')
    diagnostics = (*RETRACE_DIAGNOSTICS, 'inner_actor_entropy', 'inner_alpha_initial', 'inner_alpha_final',
                   'inner_alpha_delta', 'inner_final_outer_policy_kl', 'inner_outer_policy_kl')
    for key in diagnostics:
        stat = result['model_metrics'].get(key, {})
        require(stat.get('count') == len(seeds) * steps and all(type(stat.get(k)) in (int, float)
                and math.isfinite(stat[k]) for k in ('mean', 'min', 'max')), 'Missing finite diagnostic: ' + key)
    metrics = result['model_metrics']
    require(0 <= metrics[RETRACE_DIAGNOSTICS[0]]['min'] <= metrics[RETRACE_DIAGNOSTICS[0]]['max'] <= 1.000001
            and 1 <= metrics[RETRACE_DIAGNOSTICS[1]]['min'] <= metrics[RETRACE_DIAGNOSTICS[1]]['max'] <= 2.000001
            and metrics[RETRACE_DIAGNOSTICS[2]]['min'] >= 0, 'Invalid Retrace trace diagnostics')
    constants = dict(inner_alpha_initial=cell['outer_alpha'] if cell['entropy_enabled'] else 0.,
        inner_target_entropy=-10.5, inner_retrace_trajectory_draws=128 * 86,
        inner_retrace_critic_rows=128 * 86 * 3)
    if not cell['entropy_enabled']:
        constants.update(inner_alpha_final=0., inner_alpha_delta=0.)
    for key, value in constants.items():
        _constant(metrics, key, value, len(seeds) * steps)
        for ep in result['episodes']:
            _constant(ep['model_metrics'], key, value)
    if smoke_steps is None:
        from utils.eval_series_data import load_records
        record, = load_records(output / 'bundle' / 'manifest.json', source_run=cell['source_run'],
                               inventory_path=cell['inventory_path'])
        validate_record(cell, record)
    validation.update(retrace_lambda=1., critic_batch_trajectories=86,
        critic_rows_per_update=258, retrace_diagnostics={key: metrics[key]['mean'] for key in RETRACE_DIAGNOSTICS})
    return validation


def validate_record(cell, record):
    require(record['checkpoint'] == {'step': cell['step'], 'sha256': cell['checkpoint_sha256']},
            'Normalized checkpoint changed')
    require(record['metrics'].get('eval/frozen_state_unchanged') is True
            and record['metrics'].get('work/environment_decisions') == 2500, 'Incomplete frozen episodes')
    prior = {ep['seed']: ep for ep in read(cell['reference_results'])['results'][0]['episodes']}
    require([ep['seed'] for ep in record['episodes']] == SEEDS, 'Wrong episode seed coverage')
    for ep in record['episodes']:
        base = prior[ep['seed']]
        require(ep['length'] == 500 and not ep.get('capped') and ep['solver_seed'] == base['solver_seed']
                and math.isclose(ep['paired_return_delta'], ep['return'] - base['return'], abs_tol=1e-8),
                'Paired return gain or full-episode reference mismatch')
    expected = dict(actor_updates=80000, critic_updates=320000,
        temperature_updates=cell['expected_temperature_steps'] * 2500,
        retrace_trajectory_draws=128 * 86 * 2500, retrace_critic_rows=128 * 86 * 3 * 2500)
    require(all(record['metrics'].get('work/' + key) == value for key, value in expected.items()),
            'Published optimizer or Retrace work mismatch')
    for key in (*RETRACE_DIAGNOSTICS, 'inner_actor_entropy', 'inner_actor_entropy_max',
                'inner_final_outer_policy_kl', 'inner_alpha_initial', 'inner_alpha_final', 'inner_alpha_delta'):
        value = record['metrics'].get('diagnostics/' + key + '_mean')
        require(type(value) in (int, float) and math.isfinite(value), 'Missing publication diagnostic: ' + key)


def run(campaign_path, index, smoke_steps=None, cpu_threads=None):
    campaign_path = Path(campaign_path).resolve()
    campaign = read(campaign_path)
    validate_campaign(campaign)
    require(type(index) is int and index in range(4), 'Invalid cell index')
    require(source() == campaign['source_commit'], 'Wrong or dirty source checkout')
    cell = campaign['cells'][index]
    verify_checkpoint(cell)
    verify_outer_alpha(cell)
    pins = reference_pins(cell)
    require(sha256(cell['inventory_path']) == cell['inventory_sha256'], 'Inventory changed')
    if cpu_threads is not None:
        available = sorted(os.sched_getaffinity(0))
        require(type(cpu_threads) is int and cpu_threads in (1, 2) and cpu_threads <= len(available),
                'Invalid CPU allocation')
        os.sched_setaffinity(0, available[:cpu_threads])
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = str(cpu_threads)
    output = (campaign_path.parent / 'smoke' / f"{cell['cell_id']}-steps{smoke_steps}-cpu{cpu_threads or 1}"
              if smoke_steps else Path(cell['output_dir']))
    command = evaluator_command(cell, output, smoke_steps)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'started.json', dict(command=command, job_id=os.environ.get('SLURM_JOB_ID')))
    started, before = time.perf_counter(), resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        subprocess.run(command, cwd=REPO, check=True)
        compute_seconds = time.perf_counter() - started
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        validation = validate_result(output, cell, smoke_steps)
        require(reference_pins(cell) == pins and source() == campaign['source_commit'], 'Source/reference changed')
        user_cpu, system_cpu = after.ru_utime - before.ru_utime, after.ru_stime - before.ru_stime
        receipt = dict(validated=True, smoke=bool(smoke_steps), source_commit=campaign['source_commit'],
            checkpoint_sha256=cell['checkpoint_sha256'], cell_id=cell['cell_id'], validation=validation,
            elapsed_seconds=time.perf_counter()-started, child_elapsed_seconds=compute_seconds,
            cpu_threads=cpu_threads, child_user_cpu_seconds=user_cpu, child_system_cpu_seconds=system_cpu,
            child_cpu_cores_mean=(user_cpu + system_cpu) / compute_seconds, child_maxrss_kib=after.ru_maxrss,
            results_sha256=sha256(output / 'results.json'), manifest_sha256=sha256(output / 'bundle/manifest.json'),
            trace_sha256=trace_hashes(output / 'bundle'), **pins)
        write_new(output / 'receipt.json', receipt)
        return receipt
    except BaseException as exc:
        write_new(output / 'failure.json', dict(exception_type=type(exc).__name__, message=str(exc)))
        raise


def publish(campaign_path, require_complete=False):
    from utils import eval_series as series
    from utils.eval_series_data import load_records
    from slurm import ambi_entropy_kl_curves as curves
    campaign = read(campaign_path)
    validate_campaign(campaign)
    os.environ.setdefault('WANDB__REQUIRE_CORE', 'true')
    profile = read(campaign['curves_profile'])
    catalog = curves.register_campaign(profile, campaign)
    for cell in campaign['cells']:
        output, receipt_path = Path(cell['output_dir']), Path(cell['output_dir']) / 'receipt.json'
        if not receipt_path.exists():
            require(not (output / 'failure.json').exists(), 'Retrace worker failed; inspect saved failure')
            require(not require_complete, 'Retrace result is still pending')
            continue
        receipt = read(receipt_path)
        require(receipt['validated'] and not receipt['smoke'] and receipt['source_commit'] == campaign['source_commit']
                and receipt['checkpoint_sha256'] == cell['checkpoint_sha256'] and receipt['cell_id'] == cell['cell_id'],
                'Unvalidated Retrace receipt')
        require(receipt['results_sha256'] == sha256(output / 'results.json')
                and receipt['manifest_sha256'] == sha256(output / 'bundle/manifest.json')
                and receipt['trace_sha256'] == trace_hashes(output / 'bundle'), 'Validated result changed')
        require(all(receipt[key] == value for key, value in reference_pins(cell).items()), 'Prior reference changed')
        record, = load_records(output / 'bundle/manifest.json', source_run=cell['source_run'],
                               inventory_path=cell['inventory_path'])
        validate_record(cell, record)
        traces = receipt['trace_sha256']
        trace_index = Path(cell['run_dir']) / f"remote-traces-{cell['step']}.json"
        series._atomic_json(trace_index, dict(policy='full traces retained on Oscar',
            files=[dict(name=name, path=record['artifact_files'][name], sha256=digest,
                        size_bytes=Path(record['artifact_files'][name]).stat().st_size)
                   for name, digest in traces.items()]))
        record['artifact_files'] = {k: v for k, v in record['artifact_files'].items() if k not in traces}
        record['artifact_files'].update({'validation-receipt.json': str(receipt_path), 'remote-traces.json': str(trace_index)})
        record['provenance']['artifact_policy'] = 'compact summaries; complete traces retained and hashed on Oscar'
        require(sum(Path(p).stat().st_size for p in record['artifact_files'].values()) <= 16 * 1024 * 1024,
                'Retrace summary exceeds upload budget')
        series.stage_record(cell['run_dir'], record)
        with series.Publisher(cell['run_dir'], owner='oscar-rgao48') as publisher:
            publisher.run.name = f"Retrace {cell['mode']} | H3 J8 | {cell['step']//1000}k | beta=outer alpha"
            publisher.publish_pending()
    return curves.publish(profile, catalog, require_complete=require_complete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--root', type=Path, required=True)
    p = sub.add_parser('run'); p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--index', type=int, choices=range(4), required=True)
    p.add_argument('--smoke-steps', type=int, choices=(8, 32)); p.add_argument('--cpu-threads', type=int, choices=(1, 2))
    p = sub.add_parser('publish'); p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare': result = prepare(args.root)
    elif args.command == 'run': result = run(args.campaign, args.index, args.smoke_steps, args.cpu_threads)
    else: result = publish(args.campaign, args.require_complete)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
