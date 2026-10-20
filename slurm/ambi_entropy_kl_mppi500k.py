"""Pinned MPPI500k sampled prior and three full-episode refinement methods.

GPU workers validate immutable local results; a separate CPU publisher owns W&B.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from slurm.ambi_early_h3_j8_campaign import read, write_new, sha256, verify_checkpoint
from slurm.ambi_entropy_kl_episodes import trace_hashes

MATRIX = 'configs/research/ambi_entropy_kl_mppi500k.json'
PROFILE = 'configs/research/ambi_entropy_kl_mppi500k_profile.json'
BASELINE_MATRIX = 'configs/research/ambi_early_h3_j8.json'
CAMPAIGN_ID = 'entropy-kl-mppi500k-20261020'
CHECKPOINT_ID = 'mppi_aux_500k'
MODES = ('prior', 'entropy', 'kl', 'both')
SEEDS = [101, 102, 103, 104, 105]
OUTER_ALPHA = 0.023066775873303413
TEMPERATURE_LR = 0.0003


def source():
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise RuntimeError('Use a clean committed source checkout')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def intervention(mode):
    if mode not in MODES:
        raise ValueError('Unknown refinement mode')
    return {'checkpoint_id': CHECKPOINT_ID, 'cell_id': CHECKPOINT_ID + '__' + mode,
            'mode': mode, 'selector': 'entropy_kl_mppi500k/' + mode,
            'outer_alpha': OUTER_ALPHA, 'beta': OUTER_ALPHA if mode in ('kl', 'both') else 0.,
            'entropy_enabled': mode in ('entropy', 'both'),
            'expected_temperature_steps': 32 if mode in ('entropy', 'both') else 0,
            'temperature_lr': TEMPERATURE_LR}


def overrides(mode):
    cell = intervention(mode)
    if mode == 'prior':
        return {'inner_operator': 'none', 'inner_rounds': 0, 'inner_rollouts_per_round': 0,
                'inner_model_step_budget': 0, 'inner_temperature_mode': 'inherit_outer',
                'inner_entropy_enabled': False, 'inner_finite_horizon': False,
                'inner_critic_updates_per_round': None, 'inner_actor_updates_per_round': None}
    result = {'inner_outer_policy_kl_coef': cell['beta']}
    if mode == 'kl':
        result['inner_entropy_enabled'] = False
    return result


def validate_matrix(matrix, baseline=None):
    baseline = read(REPO / BASELINE_MATRIX) if baseline is None else baseline
    if matrix.get('base_alg_config') != 'checkpoint' or matrix['shared_alg_params'] != baseline['shared_alg_params']:
        raise ValueError('Shared configuration must equal the existing H3/J8 baseline')
    for key in ('seeds', 'controller_seed', 'max_steps', 'togo_return_rollouts', 'togo_return_every'):
        if matrix['evaluation'][key] != baseline['evaluation'][key]:
            raise ValueError('Changed evaluation setting: ' + key)
    group = matrix['comparisons'].get('entropy_kl_mppi500k', {})
    if (set(matrix['comparisons']) != {'entropy_kl_mppi500k'} or group.get('reference') != 'prior'
            or set(group.get('variants', {})) != set(MODES)):
        raise ValueError('Expected only prior, entropy, KL and both')
    for mode, variant in group['variants'].items():
        if variant['alg_params'] != overrides(mode) or variant.get('evaluation') != {'execution_action': 'sample'}:
            raise ValueError('Unexpected intervention settings: ' + mode)
    if (matrix['evaluation'].get('execution_action') != 'sample'
            or matrix['evaluation']['default_presets'] != [intervention(mode)['selector'] for mode in MODES]):
        raise ValueError('Expected four sampled selectors in fixed order')


def validate_campaign(campaign, profile=None):
    profile = read(REPO / PROFILE) if profile is None else profile
    row = profile['checkpoints'][CHECKPOINT_ID]
    if (set(profile['checkpoints']) != {CHECKPOINT_ID} or row['step'] != 500000
            or row['outer_alpha'] != OUTER_ALPHA or row['backbone_id'] != 'mppi_aux'
            or row['entropy_spec']['target_entropy'] != -10.5):
        raise ValueError('Wrong pinned checkpoint profile')
    top = {'schema_version': 1, 'campaign_id': CAMPAIGN_ID, 'seeds': SEEDS,
           'controller_seed': 55, 'max_steps': 500, 'execution_action': 'sample'}
    if any(campaign.get(k) != v for k, v in top.items()) or len(campaign['cells']) != 4:
        raise ValueError('Changed four-cell campaign protocol')
    for index, (mode, cell) in enumerate(zip(MODES, campaign['cells'])):
        expected = {**row, **intervention(mode), 'index': index}
        if any(cell.get(k) != v for k, v in expected.items()):
            raise ValueError('Cell checkpoint, intervention or order changed')
    for key in ('cell_id', 'run_id', 'run_dir', 'output_dir'):
        values = [cell[key] for cell in campaign['cells']]
        if len(set(values)) != 4:
            raise ValueError('Duplicate cell ownership: ' + key)
    prior = campaign['cells'][0]
    for cell in campaign['cells']:
        if (cell['reference_output_dir'] != prior['output_dir']
                or cell['reference_bundle'] != str(Path(prior['output_dir']) / 'bundle')):
            raise ValueError('Reference must be this campaign sampled prior')


def verify_outer_alpha(cell):
    """Read the already SHA-pinned portable state, not a sidecar default alpha."""
    import torch
    state = torch.load(cell['checkpoint'], map_location='cpu', weights_only=False)
    alpha = state['log_ent_coef'].exp().clamp_min(1e-8).item()
    if (alpha != cell['outer_alpha'] or state['checkpoint_version'] != cell['checkpoint_version']
            or state['entropy_spec']['target_entropy'] != -10.5):
        raise ValueError('Actual checkpoint alpha/version/target differs from pin')
    return alpha


def prepare(root):
    from utils import eval_series as series
    root = Path(root).resolve()
    if (root / 'campaign.json').exists():
        raise FileExistsError('An episode campaign already exists here')
    profile = read(REPO / PROFILE)
    validate_matrix(read(REPO / MATRIX))
    row = dict(profile['checkpoints'][CHECKPOINT_ID])
    verify_checkpoint(row)
    verify_outer_alpha(row)
    campaign = {'schema_version': 1, 'campaign_id': CAMPAIGN_ID, 'source_commit': source(),
                'matrix_sha256': sha256(REPO / MATRIX), 'profile_sha256': sha256(REPO / PROFILE),
                'seeds': SEEDS, 'controller_seed': 55, 'max_steps': 500,
                'execution_action': 'sample', 'cells': []}
    inventory = root / 'inputs' / 'mppi_aux.json'
    write_new(inventory, {'source_run': row['source_run'], 'checkpoints': [
        {**row, 'path': row['checkpoint'], 'sha256': row['checkpoint_sha256']}]})
    prior_output = root / 'production' / 'prior'
    for mode in MODES:
        cell = {**row, **intervention(mode), 'index': len(campaign['cells']),
                'inventory_path': str(inventory), 'inventory_sha256': sha256(inventory),
                'reference_output_dir': str(prior_output), 'reference_bundle': str(prior_output / 'bundle'),
                'output_dir': str(root / 'production' / mode)}
        spec_dir = root / 'specs' / cell['cell_id']
        command = evaluator_command(cell, root / 'specification-only' / mode, specification_only=True)
        subprocess.run(command + ['--eval-series-spec-dir', str(spec_dir)], cwd=REPO, check=True)
        spec = read(spec_dir / (cell['selector'].replace('/', '__') + '.json'))
        registry = series.create_run(root / 'publication' / 'runs', spec, CAMPAIGN_ID,
                                     'ambi-inner-bench', 'rwgao_b-brown-university', 'oscar-rgao48')
        cell.update(run_dir=registry['run_dir'], run_id=registry['run_id'])
        campaign['cells'].append(cell)
    validate_campaign(campaign, profile)
    if source() != campaign['source_commit']:
        raise RuntimeError('Source changed during preparation')
    write_new(root / 'campaign.json', campaign)
    return campaign


def evaluator_command(cell, output, smoke_steps=None, *, specification_only=False):
    if smoke_steps is not None and (type(smoke_steps) is not int or smoke_steps not in (8, 32)):
        raise ValueError('Smoke must contain 8 or 32 full decisions')
    output = Path(output)
    command = [sys.executable, str(REPO / 'evaluate_ambi_checkpoint.py'),
               '--matrix', str(REPO / MATRIX), '--checkpoint', cell['checkpoint'],
               '--metadata', cell['metadata'], '--preset', cell['selector'], '--device', 'cuda',
               '--seeds', *map(str, [101] if smoke_steps is not None else SEEDS),
               '--controller-seed', '55', '--max-steps', str(smoke_steps or 500),
               '--output', str(output / 'results.json'), '--bundle-dir', str(output / 'bundle'),
               '--checkpoint-inventory', cell['inventory_path']]
    if not specification_only and smoke_steps is None and cell['mode'] != 'prior':
        command += ['--reference-bundle', cell['reference_bundle']]
    return command


def _constant(metrics, key, value, count=None):
    from slurm.ambi_early_h3_j8_validation import _require
    values = [metrics.get(key)] if count is None else [metrics.get(key, {}).get(k) for k in ('mean', 'min', 'max')]
    _require(all(type(v) in (int, float) and math.isclose(v, value, rel_tol=1e-6, abs_tol=1e-9)
                 for v in values), 'Unexpected metric ' + key)
    if count is not None:
        _require(metrics[key].get('count') == count, 'Incomplete metric ' + key)


def validate_prior_result(output, cell, seeds, steps):
    """Check every prior decision, zero update counts and initial-only probe traces."""
    from slurm.ambi_early_h3_j8_validation import _read, _require, _same, _finite, _summary, CONFIG, PROBE
    from utils.ambi_benchmark import solver_seed
    root = (output / 'bundle').resolve()
    payload, manifest = _read(output / 'results.json'), _read(root / 'manifest.json')
    _same(payload, {'schema_version': 1, 'frozen_outer_learning': True, 'deterministic_execution': False,
                    'checkpoint_sha256': cell['checkpoint_sha256']}, 'Prior payload')
    _require(len(payload['results']) == len(manifest['runs']) == 1, 'Expected exactly one prior controller')
    result, run = payload['results'][0], manifest['runs'][0]
    _same(manifest, {'status': 'complete'}, 'Prior bundle')
    _same(manifest['checkpoint'], {'sha256': cell['checkpoint_sha256']}, 'Prior checkpoint')
    _same(run, {'status': 'complete', 'selector': cell['selector']}, 'Prior run')
    _same(result, {'selector': cell['selector'], 'environment_seeds': seeds, 'execution_action': 'sample',
                  'controller_seed': 55, 'seed_scheme': 'sha256-v1', 'outer_state_unchanged': True}, 'Prior result')
    _require(result['outer_updates_before'] == result['outer_updates_after'], 'Prior changed outer update count')
    _require(str(result['resolved_device']).startswith('cuda'), 'Prior did not use CUDA')
    _require(not result['nonfinite_model_metrics'] and not result['nonfinite_trace_metrics']
             and not run.get('nonfinite_trace_metrics', {}), 'Nonfinite prior result')
    config = {**CONFIG, **overrides('prior'), 'inner_outer_policy_kl_coef': 0.,
              'sac_actor_loss_scale_mode': 'none'}
    for cfg in (result['resolved_config'], run['resolved_config']):
        _same(cfg, config, 'Prior resolved config')
    _same(manifest['protocol'], {'action_rule': 'tanh_sample', 'max_steps': steps,
                                'controller_seed': 55, 'seed_scheme': 'sha256-v1'}, 'Prior protocol')
    for value in (result, run):
        _same(value['togo_return_probe'], PROBE, 'Prior probe settings')
    for key in ('episodes', 'outer_updates_before', 'outer_updates_after', 'outer_state_unchanged', 'execution_action'):
        _require(run['result'][key] == result[key], 'Prior saved result mismatch: ' + key)
    episodes = result['episodes']
    _require([e['seed'] for e in episodes] == seeds and len(run['episodes']) == len(seeds), 'Prior episode coverage')
    constants = {key: 0. for key in ('inner_active', 'inner_model_steps', 'inner_critic_optimizer_steps',
                 'inner_actor_optimizer_steps', 'inner_temperature_optimizer_steps', 'inner_compile_fallback')}
    constants.update(inner_alpha_initial=cell['outer_alpha'], inner_alpha_final=cell['outer_alpha'], inner_alpha_delta=0.)
    control = probe_time = 0.
    for episode, saved in zip(episodes, run['episodes']):
        _same(episode, {'length': steps, 'terminated': False, 'truncated': True,
                       'truncated_by_evaluator': steps < 500,
                       'solver_seed': solver_seed(55, 'episode', episode['seed'])}, 'Prior episode')
        _require(not episode['nonfinite_model_metrics'], 'Nonfinite prior episode')
        for key, value in constants.items():
            _constant(episode['model_metrics'], key, value)
        for key in ('seed', 'return', 'length', 'control_seconds', 'solver_seed'):
            _require(saved[key] == episode[key], 'Prior episode/bundle mismatch')
        _finite(episode['return'], 'Nonfinite prior return')
        control += _finite(episode['control_seconds'], 'Invalid prior timing', nonnegative=True)
        probe_time += _finite(episode['togo_probe_seconds'], 'Invalid prior probe timing', nonnegative=True)
        _require(episode['togo_probe_model_steps'] == len(range(0, steps, 100)) * 192, 'Prior probe workload')
    for key, value in constants.items():
        _constant(result['model_metrics'], key, value, len(seeds) * steps)
    _summary(result['return'], [e['return'] for e in episodes], 'Prior return')
    _summary(result['episode_length'], [steps] * len(seeds), 'Prior length')
    expected = {(f'seed-{seed}', d) for seed in seeds for d in range(steps)}
    expected_probes = {(f'seed-{seed}', d) for seed in seeds for d in range(0, steps, 100)}
    recorded = {}
    for row in run['togo_probe_rows']:
        key = (row['episode_id'], row['decision_index'])
        _require(key in expected_probes and key not in recorded, 'Prior probe coverage/duplication')
        _same(row, {'round_index': 0, 'actor_updates': 0, 'critic_updates': 0}, 'Prior probe counters')
        recorded[key] = row['metrics']
    _require(set(recorded) == expected_probes, 'Incomplete prior probe records')
    states, seen_probes, trace_rows, compressed = {}, set(), 0, 0
    hashes = trace_hashes(root)
    for relative in hashes:
        path = root / relative
        compressed += path.stat().st_size
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                event = json.loads(line)
                trace_rows += 1
                key = (event['episode_id'], event['decision_index'])
                _require(key in expected and event['run_id'] == run['id'], 'Unexpected prior trace identity')
                state = states.setdefault(key, {'last': -1, 'done': False})
                _require(not state['done'] and event['event_index'] == state['last'] + 1, 'Prior trace ordering')
                _require(state['last'] != -1 or event['phase'] == 'initial', 'Prior initial trace missing')
                state['last'] = event['event_index']
                _same(event, {'round_index': 0, 'critic_updates': 0, 'actor_updates': 0,
                              'temperature_updates': 0}, 'Prior trace counters')
                _require(not event.get('nonfinite') and event['phase'] in ('initial', 'probe', 'decision'), 'Prior trace update/nonfinite')
                for value in event['metrics'].values():
                    _finite(value, 'Nonfinite prior trace metric')
                if event['phase'] == 'probe':
                    _require(key in expected_probes and key not in seen_probes, 'Unexpected prior probe trace')
                    _require(event['metrics'] == recorded[key], 'Prior trace/probe disagreement')
                    _same(event['metrics'], {'probe_model_steps': 192}, 'Prior probe work')
                    seen_probes.add(key)
                if event['phase'] == 'decision':
                    for metric, value in constants.items():
                        _constant(event['metrics'], 'decision/' + metric, value)
                    state['done'] = True
    _require(set(states) == expected and all(s['done'] for s in states.values()), 'Incomplete prior decision traces')
    _require(seen_probes == expected_probes, 'Incomplete prior probe traces')
    returns = [e['return'] for e in episodes]
    return {'passed': True, 'selector': cell['selector'], 'execution_action': 'sample', 'seeds': seeds,
            'episodes': len(seeds), 'decisions': len(seeds) * steps, 'mean_return': statistics.mean(returns),
            'sample_std_return': statistics.stdev(returns) if len(returns) > 1 else None,
            'control_seconds': control, 'control_seconds_per_decision': control / (len(seeds) * steps),
            'probe_seconds': probe_time, 'probe_boundaries': len(seen_probes), 'trace_rows': trace_rows,
            'trace_compressed_bytes': compressed, 'outer_state_unchanged': True, 'inner_updates': 0}


def validate_episode_result(output, cell, smoke_steps=None):
    from slurm.ambi_early_h3_j8_validation import validate_result
    output = Path(output)
    seeds, steps = ([101], smoke_steps) if smoke_steps is not None else (SEEDS, 500)
    if cell['mode'] == 'prior':
        return validate_prior_result(output, cell, seeds, steps)
    validation = validate_result(output / 'results.json', output / 'bundle', expected_seeds=seeds,
        max_steps=steps, expected_temperature_steps=cell['expected_temperature_steps'], config_overrides={
            'inner_entropy_enabled': cell['entropy_enabled'], 'inner_outer_policy_kl_coef': cell['beta'],
            'inner_temperature_mode': 'auto' if cell['entropy_enabled'] else 'inherit_outer',
            'inner_temperature_lr': TEMPERATURE_LR, 'sac_actor_loss_scale_mode': 'none'})
    if validation['selector'] != cell['selector'] or validation['execution_action'] != 'sample':
        raise ValueError('Wrong selector or execution mode')
    payload = read(output / 'results.json')
    if payload['checkpoint_sha256'] != cell['checkpoint_sha256']:
        raise ValueError('Wrong evaluated checkpoint')
    result = payload['results'][0]
    diagnostic_metrics = ['inner_actor_entropy', 'inner_alpha_initial', 'inner_alpha_final',
                          'inner_alpha_delta', 'inner_final_outer_policy_kl']
    if cell['beta'] > 0:
        diagnostic_metrics.append('inner_outer_policy_kl')
    for metric in diagnostic_metrics:
        aggregate = result['model_metrics'].get(metric, {})
        if (aggregate.get('count') != len(seeds) * steps
                or any(type(aggregate.get(k)) not in (int, float) or not math.isfinite(aggregate[k])
                       for k in ('mean', 'min', 'max'))):
            raise ValueError('Incomplete objective diagnostic: ' + metric)
    alpha = cell['outer_alpha'] if cell['entropy_enabled'] else 0.
    _constant(result['model_metrics'], 'inner_alpha_initial', alpha, len(seeds) * steps)
    if not cell['entropy_enabled']:
        _constant(result['model_metrics'], 'inner_alpha_final', 0., len(seeds) * steps)
    _constant(result['model_metrics'], 'inner_target_entropy', -10.5, len(seeds) * steps)
    return validation


def validated_prior_reference(campaign):
    """Return immutable receipt pins only after revalidating the full prior bundle."""
    cell = campaign['cells'][0]
    output = Path(cell['output_dir'])
    receipt = read(output / 'receipt.json')
    expected = {**intervention('prior'), 'validated': True, 'smoke': False,
                'source_commit': campaign['source_commit']}
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('Prior receipt is not this validated production baseline')
    for key, path in (('results_sha256', output / 'results.json'),
                      ('manifest_sha256', output / 'bundle' / 'manifest.json')):
        if receipt[key] != sha256(path):
            raise ValueError('Prior result/manifest changed')
    if receipt['trace_sha256'] != trace_hashes(output / 'bundle'):
        raise ValueError('Prior trace changed')
    validation = validate_episode_result(output, cell)
    if validation != receipt['validation']:
        raise ValueError('Prior validation receipt disagrees with saved result')
    return {'prior_receipt_sha256': sha256(output / 'receipt.json'),
            'prior_manifest_sha256': receipt['manifest_sha256'],
            'prior_results_sha256': receipt['results_sha256'],
            'prior_trace_sha256': receipt['trace_sha256']}


def run(campaign_path, index, smoke_steps=None):
    campaign_path = Path(campaign_path).resolve()
    campaign = read(campaign_path)
    if (source() != campaign['source_commit'] or sha256(REPO / MATRIX) != campaign['matrix_sha256']
            or sha256(REPO / PROFILE) != campaign['profile_sha256']):
        raise RuntimeError('Campaign source or configuration changed')
    validate_matrix(read(REPO / MATRIX))
    validate_campaign(campaign)
    if type(index) is not int or index not in range(4):
        raise ValueError('Cell index must be in [0, 3]')
    cell = campaign['cells'][index]
    verify_checkpoint(cell)
    verify_outer_alpha(cell)
    if sha256(cell['inventory_path']) != cell['inventory_sha256']:
        raise RuntimeError('Checkpoint inventory changed')
    reference = validated_prior_reference(campaign) if smoke_steps is None and cell['mode'] != 'prior' else {}
    output = (campaign_path.parent / 'smoke' / cell['cell_id'] if smoke_steps is not None else Path(cell['output_dir']))
    command = evaluator_command(cell, output, smoke_steps)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / 'started.json', {'command': command, 'job_id': os.environ.get('SLURM_JOB_ID')})
    started = time.perf_counter()
    try:
        subprocess.run(command, cwd=REPO, check=True)
        validation = validate_episode_result(output, cell, smoke_steps)
        if source() != campaign['source_commit']:
            raise RuntimeError('Source changed during evaluation')
        if reference and validated_prior_reference(campaign) != reference:
            raise RuntimeError('Prior reference changed during evaluation')
        receipt = {'validated': True, 'smoke': smoke_steps is not None, **intervention(cell['mode']),
                   'source_commit': campaign['source_commit'], 'validation': validation,
                   'elapsed_seconds': time.perf_counter() - started,
                   'results_sha256': sha256(output / 'results.json'),
                   'manifest_sha256': sha256(output / 'bundle' / 'manifest.json'),
                   'trace_sha256': trace_hashes(output / 'bundle'), **reference}
        write_new(output / 'receipt.json', receipt)
        return receipt
    except BaseException as exc:
        write_new(output / 'failure.json', {'complete': False, 'cell_id': cell['cell_id'],
                  'exception_type': type(exc).__name__, 'message': str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--root', type=Path, required=True)
    p = sub.add_parser('run')
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--index', type=int, choices=range(4), required=True)
    p.add_argument('--smoke-steps', type=int, choices=(8, 32))
    args = parser.parse_args()
    result = prepare(args.root) if args.command == 'prepare' else run(args.campaign, args.index, args.smoke_steps)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
