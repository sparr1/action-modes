"""Prepare and evaluate the paired 575k soft/return critic comparison.

Each worker owns all five episodes for one planner. The prior is a hash-pinned
completed reference; neither preparation nor workers regenerate it. Publication
is owned separately by the CPU campaign watcher.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slurm.ambi_aux_hj_sweep import (
    ENTITY, PROJECT, SEEDS, digest, read, training_summary, validate, write,
)

MATRIX = ROOT / 'configs/research/ambi_closed_loop_critics_575k.json'
SOURCE_RUN = 'rwgao_b-brown-university/ambi/aux6428346x0'
CHECKPOINT_STEP = 575000
CHECKPOINT_SHA = '0c6955db7cb8555a67d7863344b70be68f4b3250814d131e647ee6f9ef01a042'
PRIOR_SOURCE_COMMIT = '0830be2d694b7eea21436b305eb416fcf049b001'
INITIAL_ALPHA = .004603903274983168
GROUP = 'closed-loop-critics-575k-20260922'


def source_commit():
    status = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True)
    if status.strip():
        raise RuntimeError('Prepare and evaluate only from a clean source checkout.')
    return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def cells(matrix_path=MATRIX):
    """Keep the generic H/J cell shape, with every concrete requested override."""
    matrix = read(matrix_path)
    assert matrix['source_run'] == SOURCE_RUN
    assert matrix['evaluation']['seeds'] == SEEDS
    assert matrix['evaluation']['controller_seed'] == 55
    assert matrix['evaluation']['max_steps'] == 500
    assert matrix['evaluation']['togo_return_rollouts'] == 32
    result = []
    for selector in matrix['evaluation']['default_presets']:
        comparison, name = selector.split('/')
        variant = matrix['comparisons'][comparison]['variants'][name]
        requested = {**matrix['shared_alg_params'], **variant['alg_params']}
        params = {key: value for key, value in requested.items() if value is not None}
        kind = 'soft' if name.startswith('soft_soft_') else 'return_only'
        assert name.startswith(('soft_soft_', 'return_return_alpha_'))
        horizon = params['inner_rollout_horizon']
        assert horizon in (1, 2, 3)
        assert f'_h{horizon}_' in name
        assert params['inner_rounds'] in (1, 2, 4, 6, 8)
        assert params['inner_critic_updates_per_round'] == 16
        assert params['inner_actor_updates_per_round'] == 4
        assert params['inner_rollouts_per_round'] == 128 and params['inner_batch_size'] == 256
        assert params['inner_replay_capacity'] == 3072
        assert params['inner_entropy_enabled'] and params['inner_temperature_mode'] == 'auto'
        assert params['inner_temperature_initialization'] == params['inner_target_entropy'] == 'inherit_outer'
        expected = ('sac', 'entropy_augmented', 'outer') if kind == 'soft' else ('aux_return', 'reward_only', 'none')
        assert (params['inner_critic_source'], params['inner_sac_critic_target'], params['inner_terminal_entropy']) == expected
        assert params['inner_horizon_critic_source'] == expected[0]
        result.append(dict(name=name, selector=selector, params=params, requested_alg_params=requested,
                           H=horizon, J=params['inner_rounds'], critic_kind=kind))
    selected = [(cell['J'], cell['critic_kind']) for cell in result]
    original = [(j, arm) for j in (4, 2, 1) for arm in ('soft', 'return_only')]
    extensions = [[(j, arm) for arm in ('soft', 'return_only')] for j in (6, 8)]
    assert selected in [original, *extensions], 'Expected the original screen or both J6/J8 critic arms.'
    assert len({cell['H'] for cell in result}) == 1, 'A campaign must use one common horizon.'
    return result


def campaign_horizon(campaign):
    """Accept historical H3 campaigns without an explicit top-level H."""
    horizons = {cell['H'] for cell in campaign['cells']}
    assert len(horizons) == 1, 'A campaign must use one common horizon.'
    horizon, = horizons
    assert horizon in (1, 2, 3) and campaign.get('H', horizon) == horizon
    assert all(cell['params']['inner_rollout_horizon'] == horizon for cell in campaign['cells'])
    return horizon


def check_prior(manifest, record):
    """Pin the historical reference without relabeling its scientific source."""
    from utils.ambi_benchmark import solver_seed
    assert manifest['status'] == 'complete'
    assert manifest['checkpoint']['sha256'] == CHECKPOINT_SHA
    assert manifest['checkpoint']['metadata']['checkpoint']['step'] == CHECKPOINT_STEP
    assert manifest['checkpoint']['source_run'] == SOURCE_RUN
    assert manifest['code']['commit'] == PRIOR_SOURCE_COMMIT and manifest['code']['dirty'] is False
    assert record['checkpoint']['sha256'] == CHECKPOINT_SHA and record['checkpoint']['step'] == CHECKPOINT_STEP
    assert record['identity']['backbone'] == SOURCE_RUN
    assert record['identity']['planner'] == {'type': 'prior', 'action_rule': 'tanh_mean'}
    assert record['metrics']['eval/frozen_state_unchanged']
    assert [e['seed'] for e in record['episodes']] == SEEDS
    assert all(e['length'] == 500 and not e['truncated_by_evaluator'] and
               math.isfinite(e['return']) and e['solver_seed'] == solver_seed(55, 'episode', e['seed'])
               for e in record['episodes'])
    run, = manifest['runs']
    cfg = run['resolved_config']
    assert cfg['target_entropy'] == -10.5 and cfg['aux_return_detach_representation'] is False
    assert cfg['aux_return_mode'] == 'sac' and cfg['inner_operator'] == 'none'
    assert cfg['log_std_mapping'] == 'direct_clamp'
    assert cfg['sac_actor_loss_scale_mode'] == cfg['aux_return_sac_actor_loss_scale_mode'] == 'none'
    alpha = run['result']['model_metrics']['inner_alpha_initial']
    assert all(math.isclose(alpha[stat], INITIAL_ALPHA, rel_tol=1e-6) for stat in ('mean', 'min', 'max'))
    protocol = manifest['protocol']
    assert protocol['action_rule'] == 'tanh_mean' and protocol['controller_seed'] == 55
    assert protocol['seed_scheme'] == 'sha256-v1' and protocol['max_steps'] == 500
    assert protocol['environment']['id'] == 'DMControl-v0'
    assert protocol['environment']['params']['task'] == 'humanoid-walk'
    assert protocol['observation'] == 'state'


def prepare(args):
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.eval_series import create_run
    from utils.eval_series_data import load_records
    commit = source_commit()
    panel = cells(args.matrix)
    assert digest(args.checkpoint) == CHECKPOINT_SHA
    prior_manifest = read(args.reference / 'manifest.json')
    prior, = load_records(args.reference, inventory_path=args.inventory)
    check_prior(prior_manifest, prior)
    args.root.mkdir(parents=True, exist_ok=False)
    evaluate_matrix(args.matrix, args.checkpoint, seeds=SEEDS, controller_seed=55, max_steps=500,
                    bundle_dir=args.root / 'unused', checkpoint_inventory=args.inventory,
                    reference_bundle=args.reference, eval_series_spec_dir=args.root / 'specs')
    for cell in panel:
        directory = args.root / cell['name']
        directory.mkdir()
        spec = read(args.root / 'specs' / (cell['selector'].replace('/', '__') + '.json'))
        assert spec['identity']['backbone'] == prior['identity']['backbone']
        assert spec['identity']['protocol'] == prior['identity']['protocol']
        registry = create_run(args.registry, spec, args.group + '-' + cell['name'],
                              PROJECT, ENTITY, 'oscar-rgao48')
        cell.update(directory=str(directory), bundle=str(directory / 'bundle'),
                    actual_selector=cell['selector'], reused=False,
                    run_dir=registry['run_dir'], performance_run_id=registry['run_id'],
                    training_run_id=uuid.uuid4().hex)
    campaign = dict(schema_version=1, group=args.group, label=args.label,
                    matrix=str(args.matrix.resolve()), checkpoint=str(args.checkpoint.resolve()),
                    checkpoint_step=CHECKPOINT_STEP, checkpoint_sha256=CHECKPOINT_SHA,
                    source_run=SOURCE_RUN, inventory=str(args.inventory.resolve()),
                    reference=str(args.reference.resolve()),
                    prior_manifest_sha256=digest(args.reference / 'manifest.json'),
                    prior_source_science=prior['identity']['science'],
                    prior_source_commit=PRIOR_SOURCE_COMMIT,
                    prior_compatibility_note='Audited prior inference and evaluation protocol are unchanged; '
                        'the historical prior retains its original scientific identity.',
                    source_commit=commit, source_dir=str(ROOT), initial_alpha=INITIAL_ALPHA,
                    target_entropy=-10.5, H=panel[0]['H'], overview_run_id=uuid.uuid4().hex, cells=panel,
                    publisher_workers=2)
    write(args.root / 'campaign.json', campaign)
    print(json.dumps(dict(root=str(args.root), conditions=len(panel), reused=0,
                          overview_run_id=campaign['overview_run_id'])), flush=True)
    return campaign


def validate_completed(bundle, cell, campaign, *, smoke=False):
    from utils.ambi_benchmark import episode_protocol, solver_seed
    seeds, steps = ([101], 3) if smoke else (SEEDS, 500)
    manifest = validate(bundle, cell, seeds=seeds, steps=steps, paired=not smoke,
                        checkpoint_sha=campaign['checkpoint_sha256'],
                        checkpoint_step=campaign['checkpoint_step'])
    run, = manifest['runs']
    cfg, result = run['resolved_config'], run['result']
    assert cfg['inner_rollout_horizon'] == cell['H'] == campaign_horizon(campaign)
    assert manifest['code']['commit'] == campaign['source_commit'] and manifest['code']['dirty'] is False
    assert manifest['checkpoint']['source_run'] == campaign['source_run']
    assert run['selector'] == cell['selector'] and result['selector'] == cell['selector']
    assert cfg['compile'] and cfg['compile_strict'] and result['resolved_device'].startswith('cuda')
    assert cfg['target_entropy'] == campaign['target_entropy'] == -10.5
    assert cfg['aux_return_detach_representation'] is False
    assert cfg['sac_actor_loss_scale_mode'] == cfg['aux_return_sac_actor_loss_scale_mode'] == 'none'
    for key, expected in cell['requested_alg_params'].items():
        actual = run['config']['alg_params']
        assert key not in actual if expected is None else actual.get(key) == expected, key
    for key, stats in result['model_metrics'].items():
        assert all(math.isfinite(value) for value in stats.values()), key
        if key.endswith('_fallback'):
            assert stats['min'] == stats['mean'] == stats['max'] == 0, key
    for stat in ('mean', 'min', 'max'):
        assert math.isclose(result['model_metrics']['inner_alpha_initial'][stat],
                            campaign['initial_alpha'], rel_tol=1e-6)
    assert all(math.isfinite(value) for value in result['return'].values())
    reference_path = Path(campaign['reference']) / 'manifest.json'
    assert digest(reference_path) == campaign['prior_manifest_sha256']
    prior = read(reference_path)
    assert prior['checkpoint']['sha256'] == manifest['checkpoint']['sha256']
    assert prior['checkpoint']['source_run'] == manifest['checkpoint']['source_run']
    assert manifest['code']['runtime'] == prior['code']['runtime']
    expected_protocol = {**prior['protocol'], 'max_steps': steps}
    assert episode_protocol(manifest['protocol']) == episode_protocol(expected_protocol)
    prior_episodes = {e['seed']: e for e in prior['runs'][0]['episodes']}
    if not smoke:
        assert manifest['reference']['manifest_sha256'] == campaign['prior_manifest_sha256']
    for episode in run['episodes']:
        assert episode['solver_seed'] == solver_seed(55, 'episode', episode['seed'])
        assert episode['solver_seed'] == prior_episodes[episode['seed']]['solver_seed']
        assert math.isfinite(episode['return'])
        if not smoke:
            expected = episode['return'] - prior_episodes[episode['seed']]['return']
            assert math.isclose(episode['paired_return_delta'], expected, abs_tol=1e-9)
        else:
            assert 'paired_return_delta' not in episode
        summaries = episode['togo_round_summaries']
        assert len(summaries) == cell['J'] + 1
        for row in summaries:
            assert all(stats['count'] == steps for stats in row['metrics'].values())
    validate_probe_rows(run, cell, seeds=seeds, steps=steps)
    return manifest


def validate_probe_rows(run, cell, *, seeds, steps):
    """Probe work follows the selected horizon, including the initial prior probe."""
    expected_rows = {(f'seed-{seed}', decision, r) for seed in seeds
                     for decision in range(steps) for r in range(cell['J'] + 1)}
    rows = run['togo_probe_rows']
    assert len(rows) == len(expected_rows)
    assert {(r['episode_id'], r['decision_index'], r['round_index']) for r in rows} == expected_rows
    for row in rows:
        r, metrics = row['round_index'], row['metrics']
        assert row['critic_updates'] == 16*r and row['actor_updates'] == 4*r
        assert all(isinstance(value, (int, float)) and math.isfinite(value) for value in metrics.values())
        factor = 2 if r == 0 else 1
        assert metrics['probe_model_steps'] == 32*cell['H']*factor
        assert metrics['probe_q_evaluations'] == 32*factor


def worker(args):
    import torch
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.ambi_seed_shards import seal_episode_bundle
    campaign = read(args.root / 'campaign.json')
    assert source_commit() == campaign['source_commit']
    assert campaign['checkpoint_step'] == CHECKPOINT_STEP and campaign['checkpoint_sha256'] == CHECKPOINT_SHA
    assert campaign['source_run'] == SOURCE_RUN
    horizon = campaign_horizon(campaign)
    if not 0 <= args.index < len(campaign['cells']):
        raise ValueError('Worker index is outside the prepared campaign.')
    cell = campaign['cells'][args.index]
    if args.smoke:
        assert cell['J'] == max(c['J'] for c in campaign['cells']), (
            'Smoke must exercise the largest round budget of each critic arm.')
    assert torch.cuda.is_available()
    directory = args.root / 'smoke' / cell['name'] if args.smoke else Path(cell['directory'])
    directory.mkdir(parents=True, exist_ok=not args.smoke)
    bundle = directory / 'bundle'
    evaluate_matrix(campaign['matrix'], campaign['checkpoint'], selectors=[cell['selector']],
                    seeds=[101] if args.smoke else SEEDS, controller_seed=55,
                    max_steps=3 if args.smoke else 500, device='cuda', bundle_dir=bundle,
                    checkpoint_inventory=campaign['inventory'],
                    reference_bundle=None if args.smoke else campaign['reference'])
    manifest = validate_completed(bundle, cell, campaign, smoke=args.smoke)
    summary = training_summary(bundle, cell, expected_steps=3 if args.smoke else 500)
    seal_episode_bundle(bundle)
    receipt = dict(status='complete', cell=cell['name'], selector=cell['selector'],
                   bundle=str(bundle), reused=False, smoke=args.smoke,
                   checkpoint_step=CHECKPOINT_STEP, checkpoint_sha256=CHECKPOINT_SHA, H=horizon,
                   manifest_sha256=digest(bundle / 'manifest.json'),
                   trace_sha256={name: digest(bundle / name) for name in manifest['runs'][0]['trace_files']},
                   trace_rows_checked=summary['trace_rows_checked'], gpu=torch.cuda.get_device_name(0))
    write(directory / 'validation.json', receipt)
    write(directory / 'worker-completion.json', receipt)
    print('COMPLETE ' + cell['name'], flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('root', 'checkpoint', 'inventory', 'reference', 'registry'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--matrix', type=Path, default=MATRIX)
    prep.add_argument('--group', default=GROUP)
    prep.add_argument('--label', default='575k target -10.5/shared | closed-loop soft vs return Q')
    run = sub.add_parser('worker')
    run.add_argument('--root', type=Path, required=True)
    run.add_argument('--index', type=int, required=True)
    run.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        worker(args)


if __name__ == '__main__':
    main()
