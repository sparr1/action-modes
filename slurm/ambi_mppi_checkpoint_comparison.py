"""Frozen MPPI comparator at 400k/500k with retained sampled-prior references.

GPU workers never contact W&B. A CPU publisher validates complete bundles,
publishes compact source records, and appends the persistent comparison curve.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
import resource
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from slurm.ambi_early_h3_j8_campaign import read, write_new, sha256, verify_checkpoint
from slurm.ambi_entropy_kl_mppi500k import source

MATRIX = REPO / 'configs/research/ambi_mppi_checkpoint_comparison.json'
PROFILE = REPO / 'configs/research/ambi_mppi_checkpoint_comparison_profile.json'
CAMPAIGN = 'mppi-checkpoint-comparison-20261020'
SEEDS = [101, 102, 103, 104, 105]
SELECTOR = 'mppi_checkpoint/mppi'
SETTINGS = dict(inner_operator='mppi', inner_rollout_horizon=3,
    inner_mppi_iterations=8, inner_model_step_budget=12336,
    inner_mppi_num_elites=64, inner_mppi_num_pi_trajs=24,
    inner_mppi_temperature=.5, inner_mppi_min_std=.05, inner_mppi_max_std=2.,
    inner_mppi_warm_start_scope='episode', inner_mppi_eval_action='weighted_elite',
    inner_actor_source='sac', inner_critic_source='aux_return',
    inner_horizon_actor_source='sac', inner_horizon_critic_source='aux_return',
    mppi_terminal_q_reduction='mean_pair', inner_diagnostic_rollouts=0,
    inner_diagnostics_every=1000, wandb=False, compile=True, compile_strict=True,
    inner_outer_policy_kl_coef=0., inner_actor_writeback_coef=0.,
    inner_critic_writeback_coef=0., inner_explorer_mode='none')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def trace_hashes(bundle):
    """MPPI retains episode metrics and must not manufacture SAC trace files."""
    runs = read(Path(bundle) / 'manifest.json')['runs']
    require(len(runs) == 1 and runs[0].get('trace_files') == [],
            'MPPI requires one result without SAC optimization traces')
    return {}


def validate_matrix(matrix):
    overrides = {k:v for k,v in SETTINGS.items() if k != 'mppi_terminal_q_reduction'}
    require(matrix['base_alg_config'] == 'checkpoint' and matrix['shared_alg_params'] == overrides,
            'MPPI settings differ from the pinned comparison')
    require(matrix['evaluation'] == dict(seeds=SEEDS, controller_seed=55, max_steps=500,
        execution_action='mean', prior_reference='ambi_sampled_prior-v1', togo_return_rollouts=0,
        default_presets=[SELECTOR]), 'Wrong evaluation/reference protocol')
    group = matrix['comparisons']
    require(set(group) == {'mppi_checkpoint'} and set(group['mppi_checkpoint']['variants']) == {'mppi'}
            and group['mppi_checkpoint']['variants']['mppi']['alg_params'] == {},
            'Only the MPPI comparator is authorized')


def reference_pins(cell):
    paths = dict(reference_manifest_sha256=Path(cell['reference_bundle']) / 'manifest.json',
                 reference_results_sha256=Path(cell['reference_results']))
    hashes = {key: sha256(path) for key, path in paths.items()}
    require(all(cell[key] == value for key, value in hashes.items()), 'Saved sampled prior changed')
    payload = read(cell['reference_results'])
    require(payload['checkpoint_sha256'] == cell['checkpoint_sha256'], 'Prior checkpoint mismatch')
    prior, = payload['results']
    require(prior['execution_action'] == 'sample' and prior['outer_state_unchanged']
            and prior['resolved_config']['inner_operator'] == 'none', 'Reference is not the frozen sampled prior')
    require([ep['seed'] for ep in prior['episodes']] == SEEDS
            and all(ep['length'] == 500 and not ep.get('truncated_by_evaluator') for ep in prior['episodes']),
            'Prior lacks five complete matched episodes')
    return hashes


def evaluator_command(cell, output, smoke_steps=None, specification_only=False):
    require(smoke_steps is None or (type(smoke_steps) is int and smoke_steps in (8, 32, 128)), 'Invalid smoke length')
    output = Path(output)
    args = [sys.executable, str(REPO / 'evaluate_ambi_checkpoint.py'), '--matrix', str(MATRIX),
        '--checkpoint', cell['checkpoint'], '--metadata', cell['metadata'], '--preset', SELECTOR,
        '--device', 'cuda', '--seeds', *map(str, [101] if smoke_steps else SEEDS),
        '--controller-seed', '55', '--max-steps', str(smoke_steps or 500),
        '--output', str(output / 'results.json'), '--bundle-dir', str(output / 'bundle'),
        '--checkpoint-inventory', cell['inventory_path']]
    if smoke_steps is None:
        args += ['--reference-bundle', cell['reference_bundle']]
    return args


def validate_campaign(campaign):
    require(campaign['campaign_id'] == CAMPAIGN and campaign['seeds'] == SEEDS
            and campaign['controller_seed'] == 55 and campaign['max_steps'] == 500,
            'Campaign protocol changed')
    require(campaign['matrix_sha256'] == sha256(MATRIX) and campaign['profile_sha256'] == sha256(PROFILE),
            'Pinned configuration changed')
    pins = read(PROFILE)['checkpoints']
    require(len(campaign['cells']) == 2, 'Exactly two checkpoints are required')
    for index, (cell, pin) in enumerate(zip(campaign['cells'], pins)):
        require(all(cell.get(k) == v for k, v in pin.items()) and cell['index'] == index
                and cell['mode'] == 'mppi' and cell['selector'] == SELECTOR, 'Checkpoint or planner changed')
    require(len({c['output_dir'] for c in campaign['cells']}) == 2, 'Output ownership collision')


def prepare(root):
    from utils import eval_series as series
    from slurm import ambi_entropy_kl_curves as curves
    root = Path(root).resolve()
    require(not (root / 'campaign.json').exists(), 'Campaign already exists; inspect before retrying')
    validate_matrix(read(MATRIX))
    campaign = dict(campaign_id=CAMPAIGN, source_commit=source(), seeds=SEEDS, controller_seed=55,
        max_steps=500, matrix_sha256=sha256(MATRIX), profile_sha256=sha256(PROFILE), cells=[])
    profile = deepcopy(read(curves.PROFILE))
    for index, pin in enumerate(read(PROFILE)['checkpoints']):
        verify_checkpoint(pin)
        reference_pins(pin)
        cell = dict(pin, index=index, mode='mppi', selector=SELECTOR, cell_id=f"mppi_{pin['step']}",
            output_dir=str(root / 'production' / str(pin['step'])))
        inventory = root / 'inputs' / f"{pin['step']}.json"
        write_new(inventory, dict(source_run=pin['source_run'], checkpoints=[dict(pin,
            path=pin['checkpoint'], sha256=pin['checkpoint_sha256'])]))
        cell.update(inventory_path=str(inventory), inventory_sha256=sha256(inventory))
        specdir = root / 'specs' / cell['cell_id']
        command = evaluator_command(cell, root / 'specification-only' / cell['cell_id'], specification_only=True)
        subprocess.run(command + ['--eval-series-spec-dir', str(specdir)], cwd=REPO, check=True)
        spec = read(specdir / (SELECTOR.replace('/', '__') + '.json'))
        registry = series.create_run(root / 'publication' / 'runs', spec, CAMPAIGN,
            'ambi-inner-bench', 'rwgao_b-brown-university', 'oscar-rgao48')
        cell.update(run_id=registry['run_id'], run_dir=registry['run_dir'])
        identity = registry['identity']
        if 'mppi' in profile['identities']:
            require(profile['identities']['mppi'] == identity, 'MPPI checkpoints differ in evaluation recipe')
        profile['identities']['mppi'] = identity
        campaign['cells'].append(cell)
    campaign['curves_profile'] = str(root / 'curves-profile.json')
    write_new(campaign['curves_profile'], profile)
    campaign['curves_profile_sha256'] = sha256(campaign['curves_profile'])
    validate_campaign(campaign)
    write_new(root / 'campaign.json', campaign)
    catalog = curves.register_campaign(profile, campaign)
    return dict(campaign=str(root / 'campaign.json'), catalog=catalog)


def validate_result(output, cell, smoke_steps=None):
    from utils.ambi_benchmark import solver_seed
    output = Path(output)
    payload, manifest = read(output / 'results.json'), read(output / 'bundle' / 'manifest.json')
    require(payload['checkpoint_sha256'] == cell['checkpoint_sha256'] and payload['frozen_outer_learning'],
            'Wrong checkpoint or unfrozen evaluation')
    result, = payload['results']
    saved, = manifest['runs']
    require(manifest['status'] == saved['status'] == 'complete' and result['selector'] == SELECTOR,
            'Incomplete MPPI result')
    require(result['outer_state_unchanged'] and result['outer_updates_before'] == result['outer_updates_after'],
            'Outer model changed')
    require(str(result['resolved_device']).startswith('cuda') and not result['nonfinite_model_metrics']
            and not result.get('nonfinite_trace_metrics'), 'Nonfinite result or wrong device')
    cfg = result['resolved_config']
    require(all(cfg.get(k) == v for k, v in SETTINGS.items()) and cfg['inner_mppi_num_samples'] == 512,
            'MPPI runtime settings changed')
    require(manifest['protocol']['action_rule'] == 'mppi_weighted_elite'
            and manifest['protocol']['prior_reference'] == 'ambi_sampled_prior-v1', 'Wrong executed action protocol')
    seeds, steps = ([101], smoke_steps) if smoke_steps else (SEEDS, 500)
    episodes = result['episodes']
    require([ep['seed'] for ep in episodes] == seeds, 'Wrong episode seeds')
    for ep in episodes:
        require(ep['length'] == steps and not ep['terminated'] and ep['truncated']
                and ep['truncated_by_evaluator'] == bool(smoke_steps)
                and ep['solver_seed'] == solver_seed(55, 'episode', ep['seed'])
                and math.isfinite(ep['return']) and not ep['nonfinite_model_metrics'], 'Invalid episode')
    expected = dict(inner_mppi_iterations=8, inner_model_steps=12336,
        inner_critic_optimizer_steps=0, inner_actor_optimizer_steps=0,
        inner_temperature_optimizer_steps=0, inner_compile_fallback=0)
    for key, value in expected.items():
        stat = result['model_metrics'][key]
        require(stat['count'] == len(seeds) * steps and all(math.isclose(stat[k], value, abs_tol=1e-9)
                for k in ('min', 'max', 'mean')), 'Wrong MPPI work: ' + key)
    for ep in episodes:
        for key, value in expected.items():
            require(math.isclose(ep['model_metrics'][key], value, abs_tol=1e-9),
                    'Incorrect episode MPPI work: ' + key)
    require(not trace_hashes(output / 'bundle'), 'MPPI should retain episode metrics without SAC traces')
    if not smoke_steps:
        from utils.eval_series_data import load_records
        record, = load_records(output / 'bundle' / 'manifest.json', source_run=cell['source_run'],
                               inventory_path=cell['inventory_path'])
        prior = {ep['seed']: ep for ep in read(cell['reference_results'])['results'][0]['episodes']}
        for ep in record['episodes']:
            require(math.isclose(ep['paired_return_delta'], ep['return'] - prior[ep['seed']]['return'],
                                 abs_tol=1e-8), 'Paired gain mismatch')
    return dict(validated=True, episodes=len(seeds), decisions=len(seeds)*steps,
        mean_return=statistics.mean(ep['return'] for ep in episodes),
        sample_std_return=statistics.stdev(ep['return'] for ep in episodes) if len(episodes)>1 else None,
        control_seconds_per_decision=sum(ep['control_seconds'] for ep in episodes)/(len(seeds)*steps),
        outer_state_unchanged=True, optimizer_steps=0, model_steps_per_decision=12336)


def run(campaign_path, index, smoke_steps=None, cpu_threads=None):
    campaign_path = Path(campaign_path).resolve()
    campaign = read(campaign_path)
    validate_campaign(campaign)
    require(source() == campaign['source_commit'], 'Wrong or dirty source checkout')
    cell = campaign['cells'][index]
    verify_checkpoint(cell)
    pins = reference_pins(cell)
    require(sha256(cell['inventory_path']) == cell['inventory_sha256'], 'Inventory changed')
    if cpu_threads is not None:
        available = sorted(os.sched_getaffinity(0))
        require(cpu_threads in (1,2) and cpu_threads <= len(available), 'Invalid CPU allocation')
        os.sched_setaffinity(0, available[:cpu_threads])
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[name] = str(cpu_threads)
    output = (campaign_path.parent / 'smoke' / f"{cell['step']}-cpu{cpu_threads or 1}"
              if smoke_steps else Path(cell['output_dir']))
    output.mkdir(parents=True, exist_ok=False)
    command = evaluator_command(cell, output, smoke_steps)
    write_new(output / 'started.json', dict(command=command, job_id=os.environ.get('SLURM_JOB_ID')))
    started = time.perf_counter()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        subprocess.run(command, cwd=REPO, check=True)
        validation = validate_result(output, cell, smoke_steps)
        require(reference_pins(cell) == pins and source() == campaign['source_commit'], 'Source/reference changed')
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        receipt = dict(validated=True, smoke=bool(smoke_steps), source_commit=campaign['source_commit'],
            checkpoint_sha256=cell['checkpoint_sha256'], cell_id=cell['cell_id'], validation=validation,
            elapsed_seconds=time.perf_counter()-started,
            child_cpu_seconds=usage_after.ru_utime+usage_after.ru_stime-usage_before.ru_utime-usage_before.ru_stime,
            cpu_threads=cpu_threads or int(os.environ.get('SLURM_CPUS_PER_TASK', '1')),
            results_sha256=sha256(output/'results.json'),
            manifest_sha256=sha256(output/'bundle/manifest.json'), trace_sha256=trace_hashes(output/'bundle'), **pins)
        write_new(output/'receipt.json', receipt)
        return receipt
    except BaseException as exc:
        write_new(output/'failure.json', dict(exception_type=type(exc).__name__, message=str(exc)))
        raise


def publish(campaign_path, require_complete=False):
    from utils import eval_series as series
    from utils.eval_series_data import load_records
    from slurm import ambi_entropy_kl_curves as curves
    campaign = read(campaign_path)
    validate_campaign(campaign)
    require(sha256(campaign['curves_profile']) == campaign['curves_profile_sha256'], 'Curve profile changed')
    os.environ.setdefault('WANDB__REQUIRE_CORE', 'true')
    profile = read(campaign['curves_profile'])
    catalog = curves.register_campaign(profile, campaign)
    for cell in campaign['cells']:
        output = Path(cell['output_dir'])
        receipt_path = output/'receipt.json'
        if not receipt_path.exists():
            require(not (output/'failure.json').exists(), 'MPPI worker failed; inspect its saved failure')
            require(not require_complete, 'MPPI result is still pending')
            continue
        receipt = read(receipt_path)
        require(receipt['validated'] and not receipt['smoke'] and receipt['source_commit']==campaign['source_commit']
                and receipt['checkpoint_sha256']==cell['checkpoint_sha256'], 'Unvalidated MPPI receipt')
        require(receipt['manifest_sha256']==sha256(output/'bundle/manifest.json')
                and receipt['results_sha256']==sha256(output/'results.json')
                and receipt['trace_sha256']==trace_hashes(output/'bundle'), 'MPPI result changed after validation')
        reference_pins(cell)
        record, = load_records(output/'bundle/manifest.json', source_run=cell['source_run'],
                               inventory_path=cell['inventory_path'])
        traces = set(receipt['trace_sha256'])
        record['artifact_files'] = {k:v for k,v in record['artifact_files'].items() if k not in traces}
        record['artifact_files']['validation-receipt.json'] = str(receipt_path)
        record['provenance']['artifact_policy'] = 'episode summaries; no SAC optimization traces for MPPI'
        require(sum(Path(p).stat().st_size for p in record['artifact_files'].values()) <= 16*1024*1024,
                'MPPI summary exceeds upload budget')
        series.stage_record(cell['run_dir'], record)
        with series.Publisher(cell['run_dir'], owner='oscar-rgao48') as publisher:
            publisher.run.name = f"MPPI reference | H3 K8 | {cell['step']//1000}k | aux-return"
            publisher.publish_pending()
    return curves.publish(profile, catalog, require_complete=require_complete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--root', type=Path, required=True)
    p = sub.add_parser('run'); p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--index', type=int, choices=(0,1), required=True)
    p.add_argument('--smoke-steps', type=int, choices=(8,32,128)); p.add_argument('--cpu-threads', type=int, choices=(1,2))
    p = sub.add_parser('publish'); p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare': result = prepare(args.root)
    elif args.command == 'run': result = run(args.campaign, args.index, args.smoke_steps, args.cpu_threads)
    else: result = publish(args.campaign, args.require_complete)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
