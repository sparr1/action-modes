"""Fresh lambda-one Retrace comparisons with independent value action averages."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slurm.ambi_aux_hj_sweep import ENTITY, PROJECT, SEEDS, digest, read, training_summary, validate, write
from slurm.ambi_closed_loop_critics import (
    CHECKPOINT_SHA, CHECKPOINT_STEP, INITIAL_ALPHA, SOURCE_RUN, source_commit, validate_probe_rows,
)
from slurm.ambi_closed_loop_reward_retrace import historical_cell
from slurm.ambi_closed_loop_sampled import ACTION_RULE, EXECUTION_KEY

MATRIX = ROOT / 'configs/research/ambi_closed_loop_retrace_values_575k.json'
GROUP = 'closed-loop-retrace-values-l1-575k-20260924'
SAMPLE_PAIRS = ((1, 1), (4, 1), (1, 4), (4, 4))
SMOKE_INDICES = (3, 7)


def identities():
    """Longest solves first; each H/J's fresh baseline precedes its candidates."""
    return [(h, j, ki, kb) for j in (10, 8, 6, 4, 2, 1) for h in (2, 3)
            for ki, kb in SAMPLE_PAIRS]


def cell_key(cell):
    return cell['H'], cell['J'], cell['Ki'], cell['Kb']


def cell_name(horizon, rounds, samples, boundary_samples):
    return f'reward_reward_h{horizon}_j{rounds}_c16_retrace_l1_ki{samples}_kb{boundary_samples}_sampled'


def cells(matrix_path=MATRIX):
    matrix = read(matrix_path)
    expected_selectors = ['sweep/' + cell_name(*key) for key in identities()]
    assert matrix['source_run'] == SOURCE_RUN and matrix['base_alg_config'] == 'checkpoint'
    assert matrix['evaluation'] == dict(controller_seed=55, seeds=SEEDS, max_steps=500,
        togo_return_rollouts=32, default_presets=expected_selectors)
    result = []
    for key, selector in zip(identities(), expected_selectors):
        h, j, ki, kb = key
        name = selector.split('/')[1]
        requested = {**matrix['shared_alg_params'],
                     **matrix['comparisons']['sweep']['variants'][name]['alg_params']}
        # Compare configuration recipes only; this imports no historical results.
        expected = {**historical_cell(h, j)['requested_alg_params'],
            EXECUTION_KEY: 'policy_sample', 'inner_sac_return_estimator': 'retrace',
            'inner_retrace_lambda': 1.0, 'inner_retrace_batch_trajectories': math.ceil(256 / h),
            'inner_retrace_value_samples': ki, 'inner_retrace_boundary_value_samples': kb}
        assert requested == expected, name
        result.append(dict(name=name, selector=selector, actual_selector=selector,
            requested_alg_params=requested, params={k: v for k, v in requested.items() if v is not None},
            H=h, J=j, Ki=ki, Kb=kb, value_samples=ki, boundary_value_samples=kb,
            critic_kind='return_only', estimator='retrace', retrace_lambda=1.0,
            execution_mode='policy_sample'))
    assert len(result) == len({cell_key(cell) for cell in result}) == 48
    return result


def prepare(args):
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.eval_series import create_run
    commit, panel = source_commit(), cells(args.matrix)
    assert digest(args.checkpoint) == CHECKPOINT_SHA
    args.root.mkdir(parents=True, exist_ok=False)
    evaluate_matrix(args.matrix, args.checkpoint, seeds=SEEDS, controller_seed=55, max_steps=500,
                    bundle_dir=args.root / 'unused', checkpoint_inventory=args.inventory,
                    reference_bundle=None, eval_series_spec_dir=args.root / 'specs')
    seen = set()
    for cell in panel:
        directory = args.root / cell['name']
        directory.mkdir()
        spec = read(args.root / 'specs' / (cell['selector'].replace('/', '__') + '.json'))
        identity = spec['identity']
        assert identity['backbone'] == SOURCE_RUN
        assert identity['protocol']['action_rule'] == identity['planner']['action_rule'] == ACTION_RULE
        settings = identity['planner']['settings']
        for name, expected in dict(inner_sac_return_estimator='retrace', inner_retrace_lambda=1.0,
            inner_rollout_horizon=cell['H'], inner_rounds=cell['J']).items():
            assert settings[name] == expected, name
        for name, expected in [('inner_retrace_value_samples', cell['Ki']),
                               ('inner_retrace_boundary_value_samples', cell['Kb'])]:
            assert settings.get(name, 1) == expected, name
        serialized = json.dumps(identity, sort_keys=True)
        assert serialized not in seen, 'Every sample-count setting needs an independent publication identity.'
        seen.add(serialized)
        registry = create_run(args.registry, spec, args.group + '-' + cell['name'], PROJECT, ENTITY, 'oscar-rgao48')
        cell.update(directory=str(directory), bundle=str(directory / 'bundle'), reused=False,
                    run_dir=registry['run_dir'], performance_run_id=registry['run_id'],
                    training_run_id=uuid.uuid4().hex)
    campaign = dict(schema_version=1, group=args.group, label=args.label,
        matrix=str(args.matrix.resolve()), matrix_sha256=digest(args.matrix),
        checkpoint=str(args.checkpoint.resolve()), checkpoint_step=CHECKPOINT_STEP,
        checkpoint_sha256=CHECKPOINT_SHA, source_run=SOURCE_RUN,
        inventory=str(args.inventory.resolve()), source_commit=commit, source_dir=str(ROOT),
        initial_alpha=INITIAL_ALPHA, target_entropy=-10.5, cells=panel,
        references=[], prior_reference=None, H=[2, 3], J=[1, 2, 4, 6, 8, 10],
        sample_pairs=[list(pair) for pair in SAMPLE_PAIRS], estimator=['retrace'], retrace_lambda=1.0,
        execution_mode='policy_sample', publisher_workers=2, overview_run_id=uuid.uuid4().hex,
        smoke_indices=list(SMOKE_INDICES))
    write(args.root / 'campaign.json', campaign)
    print(f'Prepared {len(panel)} fresh settings; overview {campaign["overview_run_id"]}', flush=True)
    return campaign


def validate_completed(bundle, cell, campaign, *, smoke=False):
    from utils.ambi_benchmark import solver_seed
    seeds, steps = ([101], 3) if smoke else (SEEDS, 500)
    assert campaign['retrace_lambda'] == cell['retrace_lambda'] == 1.0
    assert cell_key(cell) in identities()
    manifest = validate(bundle, cell, seeds=seeds, steps=steps, paired=False,
                        checkpoint_sha=CHECKPOINT_SHA, checkpoint_step=CHECKPOINT_STEP)
    assert not manifest.get('reference')
    assert manifest['code']['commit'] == campaign['source_commit'] and manifest['code']['dirty'] is False
    assert manifest['checkpoint']['source_run'] == SOURCE_RUN
    protocol = manifest['protocol']
    assert protocol['action_rule'] == ACTION_RULE and protocol['controller_seed'] == 55
    assert protocol['seed_scheme'] == 'sha256-v1' and protocol['max_steps'] == steps
    assert protocol['environment']['id'] == 'DMControl-v0'
    assert protocol['environment']['params']['task'] == 'humanoid-walk'
    assert protocol['observation'] == 'state'
    run, = manifest['runs']
    cfg, result = run['resolved_config'], run['result']
    assert run['selector'] == result['selector'] == cell['selector']
    assert result['action_rule'] == ACTION_RULE and result['deterministic_execution'] is False
    assert cfg['compile'] and cfg['compile_strict'] and result['resolved_device'].startswith('cuda')
    assert cfg['target_entropy'] == campaign['target_entropy'] == -10.5
    assert cfg['aux_return_mode'] == 'sac' and cfg['aux_return_detach_representation'] is False
    assert cfg['sac_actor_loss_scale_mode'] == cfg['aux_return_sac_actor_loss_scale_mode'] == 'none'
    assert cfg['log_std_mapping'] == 'direct_clamp'
    assert cfg['q_representation'] == 'distributional' and cfg['num_q'] == 5
    for key, expected in cell['requested_alg_params'].items():
        actual = run['config']['alg_params']
        assert key not in actual if expected is None else actual.get(key) == expected, key
    for key, stats in result['model_metrics'].items():
        assert all(math.isfinite(value) for value in stats.values()), key
        if key.endswith('_fallback'):
            assert stats['min'] == stats['mean'] == stats['max'] == 0, key
    trajectories = math.ceil(256 / cell['H'])
    expected_metrics = dict(inner_retrace_value_samples=cell['Ki'],
        inner_retrace_boundary_value_samples=cell['Kb'],
        inner_retrace_trajectory_draws=16 * cell['J'] * trajectories,
        inner_retrace_critic_rows=16 * cell['J'] * trajectories * cell['H'],
        inner_retrace_replay_trajectories=128 * cell['J'],
        inner_retrace_requested_capacity=cell['params']['inner_replay_capacity'],
        inner_retrace_effective_capacity=cell['params']['inner_replay_capacity'] // cell['H'] * cell['H'],
        inner_eval_execution_sampled=1)
    for stat in ('mean', 'min', 'max'):
        assert math.isclose(result['model_metrics']['inner_alpha_initial'][stat], INITIAL_ALPHA, rel_tol=1e-6)
        for metric, expected in expected_metrics.items():
            assert result['model_metrics'][metric][stat] == expected, (metric, stat)
        distance = result['model_metrics']['inner_eval_execution_mean_action_l2'][stat]
        assert distance >= 0 if stat == 'min' else distance > 0
    for episode in run['episodes']:
        assert episode['solver_seed'] == solver_seed(55, 'episode', episode['seed'])
        assert 'paired_return_delta' not in episode and math.isfinite(episode['return'])
        if not smoke:
            assert not episode['truncated_by_evaluator']
        for row in episode['togo_round_summaries']:
            assert all(stats['count'] == steps for stats in row['metrics'].values())
    validate_probe_rows(run, cell, seeds=seeds, steps=steps)
    return manifest


def worker(args):
    import torch
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.ambi_seed_shards import seal_episode_bundle
    campaign = read(args.root / 'campaign.json')
    assert source_commit() == campaign['source_commit']
    assert digest(campaign['matrix']) == campaign['matrix_sha256']
    assert campaign['checkpoint_sha256'] == CHECKPOINT_SHA and campaign['checkpoint_step'] == CHECKPOINT_STEP
    assert campaign['source_run'] == SOURCE_RUN and campaign['retrace_lambda'] == 1.0
    assert 0 <= args.index < len(campaign['cells'])
    cell = campaign['cells'][args.index]
    assert cell_key(cell) == identities()[args.index]
    if args.smoke:
        assert args.index in campaign['smoke_indices'] and cell['J'] == 10
    assert torch.cuda.is_available()
    directory = args.root / 'smoke' / cell['name'] if args.smoke else Path(cell['directory'])
    directory.mkdir(parents=True, exist_ok=not args.smoke)
    bundle = directory / 'bundle'
    try:
        evaluate_matrix(campaign['matrix'], campaign['checkpoint'], selectors=[cell['selector']],
                        seeds=[101] if args.smoke else SEEDS, controller_seed=55,
                        max_steps=3 if args.smoke else 500, device='cuda', bundle_dir=bundle,
                        checkpoint_inventory=campaign['inventory'], reference_bundle=None)
        manifest = validate_completed(bundle, cell, campaign, smoke=args.smoke)
        summary = training_summary(bundle, cell, expected_steps=3 if args.smoke else 500)
        seal_episode_bundle(bundle)
        receipt = dict(status='complete', cell=cell['name'], selector=cell['selector'], bundle=str(bundle),
            reused=False, smoke=args.smoke, execution='policy_sample', estimator='retrace',
            retrace_lambda=1.0, Ki=cell['Ki'], Kb=cell['Kb'],
            checkpoint_step=CHECKPOINT_STEP, checkpoint_sha256=CHECKPOINT_SHA, H=cell['H'], J=cell['J'],
            manifest_sha256=digest(bundle / 'manifest.json'),
            trace_sha256={name: digest(bundle / name) for name in manifest['runs'][0]['trace_files']},
            trace_rows_checked=summary['trace_rows_checked'], gpu=torch.cuda.get_device_name(0))
        write(directory / 'validation.json', receipt)
        write(directory / 'worker-completion.json', receipt)
    except Exception as error:
        write(directory / 'worker-failure.json', dict(status='failed', cell=cell['name'],
              smoke=args.smoke, error_type=type(error).__name__, error=str(error)))
        raise
    print('COMPLETE ' + cell['name'], flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('root', 'checkpoint', 'inventory', 'registry'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--matrix', type=Path, default=MATRIX)
    prep.add_argument('--group', default=GROUP)
    prep.add_argument('--label', default='Retrace lambda1 | H2/H3 J1–10 | inner/boundary value actions 1/4 | 575k')
    run = sub.add_parser('worker')
    run.add_argument('--root', type=Path, required=True)
    run.add_argument('--index', type=int, required=True)
    run.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    {'prepare': prepare, 'worker': worker}[args.command](args)


if __name__ == '__main__':
    main()
