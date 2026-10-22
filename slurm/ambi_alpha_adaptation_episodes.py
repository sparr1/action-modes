"""Pinned sampled closed-loop temperature adaptation sanity check.

Only repeated temperature updates and episode temperature/Adam persistence are
new production cells. Historical baseline/prior are hash-pinned and verified.
GPU workers retain all traces offline; publication is a separate CPU task.
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
from slurm.ambi_early_h3_j8_campaign import read, write_new, sha256, verify_checkpoint
from slurm.ambi_entropy_kl_episodes import trace_hashes, source, OUTER_ALPHAS
from slurm.ambi_mppi_checkpoint_comparison import require

MATRIX = REPO / 'configs/research/ambi_alpha_adaptation_episodes.json'
PROFILE = REPO / 'configs/research/ambi_entropy_probe.json'
BASELINE_MATRIX = REPO / 'configs/research/ambi_early_h3_j8.json'
REFERENCE_ROOT = Path('/oscar/scratch/rgao48/ambi/early-h3-j8-eval/20261018-a0d73c82')
CAMPAIGN_ID = 'alpha-adaptation-episodes-20261021'
CELLS = {'sac_aux_100k': '09', 'mppi_aux_400k': '07'}
MODES = ('repeat100', 'carry')
SEEDS = [101, 102, 103, 104, 105]
OVERRIDES = {
    'reset': {},
    'repeat100': {'inner_temperature_updates_per_actor': 100},
    'carry': {'inner_temperature_scope': 'episode', 'inner_temperature_optimizer_scope': 'episode'},
}
LIFECYCLE = ('inner_temperature_optimizer_step_initial', 'inner_temperature_optimizer_step_final',
             'inner_temperature_lifetime_steps_initial', 'inner_temperature_lifetime_steps_final',
             'inner_temperature_episode_reset')


def intervention(name, mode):
    require(name in CELLS and mode in OVERRIDES, 'Unknown temperature comparison cell')
    return dict(checkpoint_id=name, cell_id=f'{name}__{mode}', mode=mode,
                selector=f'alpha_adaptation/{mode}', outer_alpha=OUTER_ALPHAS[name],
                temperature_lr=.0003, temperature_repeats=100 if mode == 'repeat100' else 1,
                expected_temperature_steps=3200 if mode == 'repeat100' else 32)


def overrides(cell):
    return {'inner_temperature_updates_per_actor': 1, 'inner_sac_return_estimator': 'one_step',
            **OVERRIDES[cell['mode']]}


def validate_matrix(matrix):
    baseline = read(BASELINE_MATRIX)
    require(matrix['base_alg_config'] == 'checkpoint', 'Expected checkpoint-derived networks')
    require(matrix['shared_alg_params'] == {**baseline['shared_alg_params'],
            'inner_temperature_updates_per_actor': 1, 'inner_sac_return_estimator': 'one_step'},
            'Shared configuration changed beyond explicit default settings')
    require(matrix['evaluation'] == {**baseline['evaluation'], 'execution_action': 'sample',
            'default_presets': [f'alpha_adaptation/{mode}' for mode in MODES]}, 'Evaluation protocol changed')
    require(set(matrix['comparisons']) == {'alpha_adaptation'}, 'Unexpected comparison group')
    group = matrix['comparisons']['alpha_adaptation']
    require(group['reference'] == 'reset' and set(group['variants']) == set(OVERRIDES), 'Unexpected arms')
    for mode, variant in group['variants'].items():
        require(variant['alg_params'] == OVERRIDES[mode] and variant['evaluation'] == {'execution_action': 'sample'},
                'Temperature-only intervention changed')


def reference_pins(cell):
    pins = {}
    for kind in ('reference', 'baseline'):
        root = Path(cell[kind + '_bundle'])
        manifest, payload = read(root / 'manifest.json'), read(root.parent / 'results.json')
        require(manifest['status'] == 'complete' and len(manifest['runs']) == len(payload['results']) == 1,
                'Incomplete reference bundle')
        run, result = manifest['runs'][0], payload['results'][0]
        require(manifest['checkpoint']['sha256'] == payload['checkpoint_sha256'] == cell['checkpoint_sha256'],
                'Reference checkpoint differs')
        require(manifest['protocol']['action_rule'] == 'tanh_sample'
                and manifest['protocol']['max_steps'] == 500 and result['execution_action'] == 'sample'
                and result['controller_seed'] == 55 and result['seed_scheme'] == 'sha256-v1'
                and result['outer_state_unchanged'] is True
                and result['outer_updates_before'] == result['outer_updates_after'], 'Wrong reference protocol')
        require([e['seed'] for e in result['episodes']] == SEEDS
                and all(e['length'] == 500 and not e['truncated_by_evaluator'] for e in result['episodes']),
                'Reference episode coverage changed')
        require(run['result']['episodes'] == result['episodes'], 'Reference result/bundle episode mismatch')
        if kind == 'baseline':
            settings = result['resolved_config']
            for key, value in read(BASELINE_MATRIX)['shared_alg_params'].items():
                if value is not None:
                    require(settings.get(key) == value, 'Historical baseline setting mismatch: ' + key)
            require(settings.get('inner_temperature_updates_per_actor', 1) == 1
                    and settings.get('inner_sac_return_estimator', 'one_step') == 'one_step',
                    'Historical baseline temperature or estimator changed')
            require(result['selector'] == 'early_h3_j8/sample' and bool(run['trace_files']),
                    'Expected complete historical baseline traces')
        else:
            require(result['resolved_config']['inner_operator'] == 'none', 'Prior is not a no-inner reference')
        pins[kind + '_manifest_sha256'] = sha256(root / 'manifest.json')
        pins[kind + '_results_sha256'] = sha256(root.parent / 'results.json')
    return pins


def validate_campaign(campaign):
    expected = dict(schema_version=1, campaign_id=CAMPAIGN_ID, seeds=SEEDS,
                    controller_seed=55, max_steps=500, execution_action='sample')
    require(all(campaign.get(k) == v for k, v in expected.items()), 'Campaign protocol changed')
    require(campaign['matrix_sha256'] == sha256(MATRIX) and campaign['profile_sha256'] == sha256(PROFILE),
            'Pinned configuration changed')
    wanted = [{**read(PROFILE)['checkpoints'][name], **intervention(name, mode)} for name in CELLS for mode in MODES]
    require(len(campaign['cells']) == len(wanted), 'Exactly four production cells are required')
    for index, (cell, spec) in enumerate(zip(campaign['cells'], wanted)):
        require(cell['index'] == index and all(cell.get(k) == v for k, v in spec.items()),
                'Cell checkpoint, mode or order changed')
    for key in ('cell_id', 'run_id', 'run_dir', 'output_dir'):
        require(len({c[key] for c in campaign['cells']}) == 4, 'Cell ownership collision: ' + key)


def evaluator_command(cell, output, smoke_steps=None):
    require(smoke_steps is None or (type(smoke_steps) is int and smoke_steps in (8, 32)), 'Invalid smoke length')
    output = Path(output)
    command = [sys.executable, str(REPO / 'evaluate_ambi_checkpoint.py'), '--matrix', str(MATRIX),
        '--checkpoint', cell['checkpoint'], '--metadata', cell['metadata'], '--preset', cell['selector'],
        '--device', 'cuda', '--seeds', *map(str, [101, 102] if smoke_steps else SEEDS),
        '--controller-seed', '55', '--max-steps', str(smoke_steps or 500),
        '--output', str(output / 'results.json'), '--bundle-dir', str(output / 'bundle'),
        '--checkpoint-inventory', cell['inventory_path']]
    if smoke_steps is None:
        command += ['--reference-bundle', cell['reference_bundle']]
    return command


def prepare(root):
    from utils import eval_series as series
    root = Path(root).resolve()
    if (root / 'campaign.json').exists():
        raise FileExistsError('An episode campaign already exists here')
    validate_matrix(read(MATRIX))
    campaign = dict(schema_version=1, campaign_id=CAMPAIGN_ID, source_commit=source(),
        matrix_sha256=sha256(MATRIX), profile_sha256=sha256(PROFILE), seeds=SEEDS,
        controller_seed=55, max_steps=500, execution_action='sample', cells=[])
    for name, reference_cell in CELLS.items():
        pin = read(PROFILE)['checkpoints'][name]
        verify_checkpoint(pin)
        for mode in MODES:
            cell = {**pin, **intervention(name, mode), 'index': len(campaign['cells']),
                'reference_bundle': str(REFERENCE_ROOT / 'baselines-8401ce40' / f'cell-{reference_cell}' / 'bundle'),
                'baseline_bundle': str(REFERENCE_ROOT / 'production' / f'cell-{reference_cell}' / 'bundle'),
                'inventory_path': str(REFERENCE_ROOT / 'inputs' / f"{pin['backbone_id']}.json"),
                'baseline_decision_summary': str(root / 'references' / name / 'decision-summary.json')}
            cell.update(reference_pins(cell))
            cell['inventory_sha256'] = sha256(cell['inventory_path'])
            cell['output_dir'] = str(root / 'production' / cell['cell_id'])
            specdir = root / 'specs' / cell['cell_id']
            subprocess.run(evaluator_command(cell, root / 'specification-only' / cell['cell_id'])
                           + ['--eval-series-spec-dir', str(specdir)], cwd=REPO, check=True)
            spec = read(specdir / (cell['selector'].replace('/', '__') + '.json'))
            registry = series.create_run(root / 'publication' / 'runs', spec, CAMPAIGN_ID,
                                        'ambi-inner-bench', 'rwgao_b-brown-university', 'oscar-rgao48')
            cell.update(run_id=registry['run_id'], run_dir=registry['run_dir'])
            campaign['cells'].append(cell)
    validate_campaign(campaign)
    require(source() == campaign['source_commit'], 'Source changed during preparation')
    write_new(root / 'campaign.json', campaign)
    return campaign


def compact_summary_from_bundle(cell, bundle, require_lifecycle=False):
    bundle = Path(bundle).resolve()
    manifest = read(bundle / 'manifest.json')
    run, = manifest['runs']
    rows = []
    mapping = dict(alpha_initial='inner_alpha_initial', alpha_final='inner_alpha_final',
        entropy='inner_actor_entropy', target_entropy='inner_target_entropy',
        temperature_updates='inner_temperature_optimizer_steps',
        optimizer_step_initial=LIFECYCLE[0], optimizer_step_final=LIFECYCLE[1],
        lifetime_steps_initial=LIFECYCLE[2], lifetime_steps_final=LIFECYCLE[3], episode_reset=LIFECYCLE[4],
        reward='reward', control_seconds='control_seconds')
    for relative in run['trace_files']:
        path = (bundle / relative).resolve(strict=True)
        require(path.is_relative_to(bundle), 'Trace path escapes bundle')
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                event = json.loads(line)
                if event['phase'] != 'decision':
                    continue
                row = dict(seed=int(event['episode_id'].removeprefix('seed-')),
                           decision_index=event['decision_index'])
                for name, metric in mapping.items():
                    value = event['metrics'].get('decision/' + metric)
                    if value is not None:
                        require(type(value) in (int, float) and math.isfinite(value), 'Non-finite decision diagnostic')
                    require(value is not None or (not require_lifecycle and metric in LIFECYCLE),
                            'Missing decision diagnostic: ' + metric)
                    row[name] = value
                rows.append(row)
    rows.sort(key=lambda r: (r['seed'], r['decision_index']))
    rewards = {}
    for row in rows:
        rewards[row['seed']] = rewards.get(row['seed'], 0.) + row['reward']
        row['cumulative_reward'] = rewards[row['seed']]
    return dict(schema_version=1, checkpoint_id=cell['checkpoint_id'], mode=cell['mode'],
                rows=rows, episodes=deepcopy(run['result']['episodes']),
                provenance=dict(bundle=str(bundle), manifest_sha256=sha256(bundle / 'manifest.json'),
                                checkpoint_sha256=manifest['checkpoint']['sha256']))


def validate_decisions(summary, cell, seeds, steps):
    expected = [(seed, decision) for seed in seeds for decision in range(steps)]
    require([(r['seed'], r['decision_index']) for r in summary['rows']] == expected,
            'Missing or duplicate decision diagnostics')
    previous = {}
    for row in summary['rows']:
        seed, decision = row['seed'], row['decision_index']
        count = cell['expected_temperature_steps']
        initial_step = decision * count if cell['mode'] == 'carry' else 0
        require(row['temperature_updates'] == count and row['target_entropy'] == -10.5,
                'Temperature workload or target mismatch')
        require(row['optimizer_step_initial'] == initial_step and row['optimizer_step_final'] == initial_step + count,
                'Adam state continuity/reset mismatch')
        require(row['lifetime_steps_initial'] == initial_step and row['lifetime_steps_final'] == initial_step + count,
                'Temperature lifetime continuity/reset mismatch')
        require(row['episode_reset'] == int(decision == 0), 'Temperature episode reset mismatch')
        expected_alpha = previous[seed] if cell['mode'] == 'carry' and decision else cell['outer_alpha']
        require(math.isclose(row['alpha_initial'], expected_alpha, rel_tol=1e-6, abs_tol=1e-10),
                'Temperature value continuity/reset mismatch')
        require(row['alpha_initial'] > 0 and row['alpha_final'] > 0, 'Non-positive temperature')
        previous[seed] = row['alpha_final']
    for episode in summary['episodes']:
        last = next(r for r in reversed(summary['rows']) if r['seed'] == episode['seed'])
        require(math.isclose(last['cumulative_reward'], episode['return'], rel_tol=1e-8, abs_tol=1e-7),
                'Trace rewards disagree with episode return')


def validate_result(output, cell, smoke_steps=None):
    from slurm.ambi_early_h3_j8_validation import validate_result as validate_full
    output = Path(output)
    seeds, steps = ([101, 102], smoke_steps) if smoke_steps else (SEEDS, 500)
    validation = validate_full(output / 'results.json', output / 'bundle', expected_seeds=seeds,
        max_steps=steps, expected_temperature_steps=cell['expected_temperature_steps'], config_overrides=overrides(cell))
    payload = read(output / 'results.json')
    result, = payload['results']
    require(payload['checkpoint_sha256'] == cell['checkpoint_sha256']
            and result['selector'] == cell['selector'] and result['execution_action'] == 'sample',
            'Checkpoint, selector or execution mismatch')
    summary = compact_summary_from_bundle(cell, output / 'bundle', require_lifecycle=True)
    validate_decisions(summary, cell, seeds, steps)
    write_new(output / 'decision-summary.json', summary)
    validation.update(lifecycle_validated=True, temperature_steps_per_decision=cell['expected_temperature_steps'])
    return validation


def load_baseline_decision_summary(cell):
    path = Path(cell['baseline_decision_summary'])
    receipt = read(path.with_name('baseline-receipt.json'))
    require(receipt['decision_summary_sha256'] == sha256(path), 'Baseline decision summary changed')
    require(receipt['baseline_manifest_sha256'] == cell['baseline_manifest_sha256']
            and receipt['baseline_results_sha256'] == cell['baseline_results_sha256'], 'Wrong baseline provenance')
    return read(path)


def extract_baseline(cell):
    path = Path(cell['baseline_decision_summary'])
    if path.exists():
        return load_baseline_decision_summary(cell)
    summary = compact_summary_from_bundle({**cell, 'mode': 'reset'}, cell['baseline_bundle'])
    require([(r['seed'], r['decision_index']) for r in summary['rows']]
            == [(s, d) for s in SEEDS for d in range(500)], 'Incomplete historical baseline decisions')
    write_new(path, summary)
    write_new(path.with_name('baseline-receipt.json'), dict(
        baseline_manifest_sha256=cell['baseline_manifest_sha256'], baseline_results_sha256=cell['baseline_results_sha256'],
        trace_sha256=trace_hashes(cell['baseline_bundle']), decision_summary_sha256=sha256(path)))
    return summary


def run(campaign_path, index, smoke_steps=None, cpu_threads=None, baseline=False):
    campaign_path = Path(campaign_path).resolve()
    campaign = read(campaign_path)
    validate_campaign(campaign)
    require(type(index) is int and index in range(4), 'Invalid cell index')
    require(not baseline or smoke_steps is not None, 'Baseline is reused; only a fresh smoke is allowed')
    require(source() == campaign['source_commit'], 'Wrong or dirty source checkout')
    cell = dict(campaign['cells'][index])
    if baseline:
        cell.update(intervention(cell['checkpoint_id'], 'reset'))
    verify_checkpoint(cell)
    pins = reference_pins(cell)
    require(all(cell.get(k) == v for k, v in pins.items()), 'Pinned references changed')
    require(sha256(cell['inventory_path']) == cell['inventory_sha256'], 'Inventory changed')
    if cpu_threads is not None:
        available = sorted(os.sched_getaffinity(0))
        require(type(cpu_threads) is int and cpu_threads in (1, 2) and cpu_threads <= len(available), 'Invalid CPU allocation')
        os.sched_setaffinity(0, available[:cpu_threads])
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = str(cpu_threads)
    output = (campaign_path.parent / 'smoke' / f"{cell['cell_id']}-steps{smoke_steps}-cpu{cpu_threads or 1}"
              if smoke_steps else Path(cell['output_dir']))
    output.mkdir(parents=True, exist_ok=False)
    command = evaluator_command(cell, output, smoke_steps)
    write_new(output / 'started.json', dict(command=command, job_id=os.environ.get('SLURM_JOB_ID')))
    start, before = time.perf_counter(), resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        subprocess.run(command, cwd=REPO, check=True)
        compute = time.perf_counter() - start
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        validation = validate_result(output, cell, smoke_steps)
        if baseline:
            historical = extract_baseline(cell)
            old = {(r['seed'], r['decision_index']): r for r in historical['rows']}
            new = read(output / 'decision-summary.json')
            errors = {key: max(abs(r[key] - old[r['seed'], r['decision_index']][key]) for r in new['rows'])
                      for key in ('reward', 'alpha_initial', 'alpha_final', 'entropy')}
            validation['baseline_prefix_max_absolute_differences'] = errors
            validation['baseline_prefix_matches'] = all(math.isclose(r[key], old[r['seed'], r['decision_index']][key],
                rel_tol=2e-4, abs_tol=2e-5) for r in new['rows'] for key in errors)
            require(validation['baseline_prefix_matches'], 'Fresh default smoke differs from historical baseline prefix')
        require(reference_pins(cell) == pins and source() == campaign['source_commit'], 'Source or reference changed during evaluation')
        cpu = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
        receipt = dict(validated=True, smoke=bool(smoke_steps), baseline=baseline,
            source_commit=campaign['source_commit'], checkpoint_id=cell['checkpoint_id'], checkpoint_sha256=cell['checkpoint_sha256'],
            cell_id=cell['cell_id'], mode=cell['mode'], validation=validation, elapsed_seconds=time.perf_counter()-start,
            child_elapsed_seconds=compute, cpu_threads=cpu_threads, child_cpu_seconds=cpu,
            child_cpu_cores_mean=cpu/compute, child_maxrss_kib=after.ru_maxrss,
            results_sha256=sha256(output / 'results.json'), manifest_sha256=sha256(output / 'bundle/manifest.json'),
            decision_summary_sha256=sha256(output / 'decision-summary.json'), trace_sha256=trace_hashes(output / 'bundle'), **pins)
        write_new(output / 'receipt.json', receipt)
        return receipt
    except BaseException as exc:
        write_new(output / 'failure.json', dict(exception_type=type(exc).__name__, message=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--root', type=Path, required=True)
    p = sub.add_parser('run'); p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--index', type=int, choices=range(4), required=True)
    p.add_argument('--smoke-steps', type=int, choices=(8, 32)); p.add_argument('--cpu-threads', type=int, choices=(1, 2))
    p.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    result = prepare(args.root) if args.command == 'prepare' else run(args.campaign, args.index, args.smoke_steps, args.cpu_threads, args.baseline)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
