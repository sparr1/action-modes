"""Probe inherited reward-critic/model gradients at shared simulator roots.

Writes local, immutable JSON bundles. Does not learn, submit jobs or publish.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from evaluate_ambi_calibration import _bank_protocol, collect_root_bank, load_root_bank
from evaluate_ambi_checkpoint import (
    _close_resources, _file_sha256, _initialize_frozen_model, _make_env, _outer_state_digest,
)
from utils.ambi_benchmark import atomic_json, canonical_hash, read_json, solver_seed
from utils.ambi_gradient_probe import (
    PriorRewardObjective, frozen_parameters, probe_root, summarize_roots, summarize_calibration,
)
from utils.ambi_real_calibration import SimulatorSnapshot
from utils.checkpoint_context import load_checkpoint_context


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='Fresh diagnostic bundle directory.')
    parser.add_argument('--root-bank', type=Path, help='Reuse the exact states AND anchor actions of another backbone.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--decisions', type=int, nargs='+', default=None)
    parser.add_argument('--seed', type=int, default=55, help='Private diagnostic noise seed; paired across backbones.')
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--tail-steps', type=int, default=1000)
    parser.add_argument('--rollouts', type=int, default=4)
    parser.add_argument('--gradient-rollouts', type=int, default=32)
    parser.add_argument('--epsilons', type=float, nargs='+', default=[.03, .1],
                        help='L2 step lengths in pre-tanh action coordinates.')
    parser.add_argument('--random-directions', type=int, default=1)
    parser.add_argument('--q-reduction', choices=['mean_all', 'mean_pair', 'min_all', 'min_pair'], default='mean_all')
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--plan-only', action='store_true', help='Validate metadata and report work; no model/simulator run.')
    return parser


def _validate_options(args):
    for key in ('horizon', 'tail_steps', 'rollouts', 'gradient_rollouts', 'torch_threads'):
        if getattr(args, key) < 1:
            raise ValueError(f'{key} must be positive.')
    if args.random_directions < 0:
        raise ValueError('random_directions must be nonnegative.')
    if not 0 <= args.seed < 2**32:
        raise ValueError('seed must lie in [0, 2**32).')
    if (not args.epsilons or len(set(args.epsilons)) != len(args.epsilons)
            or any(not np.isfinite(x) or x <= 0 for x in args.epsilons)):
        raise ValueError('epsilons must be distinct finite positive numbers.')
    if args.root_bank is not None and (args.seeds is not None or args.decisions is not None):
        raise ValueError('A reused root bank defines seeds and decisions; do not override its coverage.')
    if args.root_bank is None:
        args.seeds = [101, 102] if args.seeds is None else args.seeds
        args.decisions = [0, 100] if args.decisions is None else args.decisions
        for key, limit in [('seeds', 2**32), ('decisions', 500)]:
            values = getattr(args, key)
            if not values or len(set(values)) != len(values) or any(not 0 <= x < limit for x in values):
                raise ValueError(f'{key} must be unique integers in [0, {limit}).')
        args.decisions = sorted(args.decisions)


def _resolved(context):
    config = copy.deepcopy(context.trial_run_params)
    if config['alg'] != 'AMBITDMPC2/AMBITDMPC2':
        raise ValueError('This probe requires an AMBITDMPC2 checkpoint.')
    params = config.setdefault('alg_params', {})
    if params.get('obs', 'state') != 'state' or params.get('episodic', False):
        raise ValueError('This probe requires continuing state observations.')
    if params.get('critic_value_mode', 'single') != 'single':
        raise ValueError('This probe currently requires a single-value critic.')
    auxiliary = params.get('aux_return_mode', 'off')
    if not (auxiliary == 'sac' or (auxiliary == 'off' and params.get('outer_critic_target') == 'reward_only')):
        raise ValueError('Need an auxiliary SAC-return critic or reward-only primary critic; soft Q is unsupported.')
    # No inner act()/predict() call is made. Retain saved learning settings for provenance.
    params.update(compile=False, compile_strict=False, wandb=False)
    environment = dict(id=config['env'], params=copy.deepcopy(context.experiment_params.get('env_params', {})))
    if environment['id'] != 'DMControl-v0' or environment['params'].get('task') != 'humanoid-walk':
        raise ValueError('Simulator restoration is reviewed only for DMControl humanoid-walk state.')
    if environment['params'].get('obs', 'state') != 'state':
        raise ValueError('Simulator restoration requires state observations.')
    return dict(algorithm_config=config, environment=environment, selector='inherited_gradient_probe')


def load_shared_bank(path, resolved):
    """Intentionally permit different checkpoint hashes, never different environments."""
    raw = read_json(path)
    bank = load_root_bank(path, raw['checkpoint_sha256'], raw['protocol'])
    expected = _bank_protocol(resolved, [], 0, dict(max_steps=1, decisions=[0]), '')
    for key in ('environment', 'env_wrappers', 'env_wrapper', 'action_rule'):
        if bank['protocol'].get(key) != expected[key]:
            raise ValueError(f'Common-root bank has incompatible {key}.')
    for root in bank['roots']:
        anchor = np.asarray(root.get('anchor_pretanh_action', []), dtype=np.float32)
        if anchor.ndim != 1 or not len(anchor) or not np.isfinite(anchor).all():
            raise ValueError('Root bank lacks finite anchor actions; create it with probe_ambi_gradients.py.')
    return bank


def _science_identity():
    root = Path(__file__).resolve().parent
    paths = [Path(__file__), root / 'utils/ambi_gradient_probe.py', root / 'utils/ambi_real_calibration.py',
             root / 'utils/ambi_benchmark.py', root / 'utils/checkpoint_context.py',
             root / 'evaluate_ambi_calibration.py', root / 'evaluate_ambi_checkpoint.py',
             root / 'domains/dmcontrol.py', root / 'RL/AMBITDMPC2.py',
             *sorted((root / 'RL/tdmpc2_core').rglob('*.py'))]
    return canonical_hash({str(path.relative_to(root)): _file_sha256(path) for path in paths})


def run_probe(args):
    _validate_options(args)
    checkpoint = args.checkpoint.expanduser().resolve()
    context = load_checkpoint_context(checkpoint, args.metadata)
    resolved = _resolved(context)
    bank = load_shared_bank(args.root_bank, resolved) if args.root_bank is not None else None
    count = len(bank['roots']) if bank else len(args.seeds) * len(args.decisions)
    directions = (2 if args.horizon == 1 else 3) + args.random_directions
    candidates = 1 + 2 * len(args.epsilons) * directions
    plan = dict(scope='inherited_prior_action_gradient', roots=count,
                candidate_actions_per_root_upper_bound=candidates,
                branch_simulator_decisions_upper_bound=count * candidates * args.rollouts * (args.horizon + args.tail_steps),
                source_collection_decisions=0 if bank else len(args.seeds) * (max(args.decisions) + 1),
                horizon=args.horizon, tail_steps=args.tail_steps, rollouts=args.rollouts,
                gradient_rollouts=args.gradient_rollouts, epsilons=args.epsilons,
                q_reduction=args.q_reduction, seed=args.seed,
                continuation='sampled_frozen_primary_prior', entropy_bonus=False,
                coordinates='action_gradients; unit_pre_tanh_directions',
                gradient_noise='independent_of_paired_evaluation_noise')
    if args.plan_only:
        return plan
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError('Choose a fresh output directory; existing results are immutable.')
    torch.set_num_threads(args.torch_threads)
    science, checkpoint_hash = _science_identity(), _file_sha256(checkpoint)
    model, env, branch_envs, error = None, None, [], None
    started, results = time.perf_counter(), []
    output.mkdir(parents=True)
    atomic_json(output / 'plan.json', plan)
    atomic_json(output / 'checkpoint.metadata.json', context.metadata)
    try:
        env = _make_env(resolved)
        model, run_config = _initialize_frozen_model(resolved, env, checkpoint, args.seed, device=args.device)
        frozen_digest = _outer_state_digest(model)
        collection = dict(root_collection_seconds=0., root_collection_simulator_decisions=0)
        with frozen_parameters(model.agent.model):
            if bank is None:
                protocol = _bank_protocol(resolved, args.seeds, args.seed,
                    dict(max_steps=max(args.decisions) + 1, decisions=args.decisions), science)
                bank, collection = collect_root_bank(env, model, checkpoint_sha256=checkpoint_hash, protocol=protocol)
                anchor_objective = PriorRewardObjective(model, reduction=args.q_reduction, seed=args.seed)
                for root in bank['roots']:
                    root['anchor_pretanh_action'] = anchor_objective.anchor(root['observation']).tolist()
                bank['anchor_policy'] = 'source_checkpoint_primary_prior_tanh_mean'
                bank['id'] = canonical_hash({key: value for key, value in bank.items() if key != 'id'})
            atomic_json(output / 'root-bank.json', bank)
            for _ in range(args.rollouts):
                branch = _make_env(resolved)
                branch_envs.append(branch)
                branch.reset(seed=bank['roots'][0]['seed'])
            for index, root in enumerate(bank['roots']):
                root_started = time.perf_counter()
                objective = PriorRewardObjective(model, reduction=args.q_reduction,
                    seed=solver_seed(args.seed, 'gradient-critic-pair', root['root_id']))
                anchor = np.asarray(root['anchor_pretanh_action'], np.float32)
                if anchor.shape != (model.cfg.action_dim,):
                    raise ValueError('Bank anchor action dimension differs from the checkpoint.')
                noises, noise_seeds = {}, {}
                for label, steps, rollouts in [('gradient', args.horizon + 1, args.gradient_rollouts),
                                               ('evaluation', args.horizon + args.tail_steps, args.rollouts)]:
                    seed = solver_seed(args.seed, 'action-gradient-' + label, root['root_id'])
                    noises[label] = np.random.default_rng(seed).standard_normal(
                        (steps, rollouts, model.cfg.action_dim)).astype(np.float32)
                    noise_seeds[label] = seed
                measured = probe_root(objective, branch_envs, SimulatorSnapshot.from_dict(root['snapshot']),
                    root['observation'], anchor, horizon=args.horizon,
                    gradient_noise=noises['gradient'], evaluation_noise=noises['evaluation'],
                    epsilons=args.epsilons, random_directions=args.random_directions,
                    direction_seed=solver_seed(args.seed, 'action-gradient-directions', root['root_id']),
                    original_remaining_steps=500 - root['decision_index'])
                record = dict(root_id=root['root_id'], episode_id=root['episode_id'],
                    decision_index=root['decision_index'], snapshot_sha256=root['snapshot']['sha256'],
                    noise_seeds=noise_seeds, critic_source=objective.source,
                    pair_indices=None if objective.pair_indices is None else objective.pair_indices.cpu().tolist(),
                    seconds=time.perf_counter() - root_started, **measured)
                atomic_json(output / 'roots' / f'{index:04d}.json', record)
                results.append(record)
                print(f"Completed root {index + 1}/{count}: {root['root_id']}", flush=True)
        if _outer_state_digest(model) != frozen_digest:
            raise RuntimeError('Frozen outer state changed during the gradient probe.')
        result = dict(schema_version=1, kind='ambi_inherited_gradient_probe', complete=True,
            checkpoint=dict(sha256=checkpoint_hash, step=context.metadata['checkpoint']['step'],
                            metadata_sha256=_file_sha256(context.source)),
            source_sha256=science, root_bank_id=bank['id'], root_bank_checkpoint_sha256=bank['checkpoint_sha256'],
            protocol=plan, resolved_run_config=run_config, outer_state_unchanged=True,
            timing=dict(total_seconds=time.perf_counter() - started, **collection),
            branch_simulator_decisions=sum(row['work']['simulator_decisions'] for row in results),
            root_records=[dict(path=f'roots/{i:04d}.json', sha256=canonical_hash(row)) for i, row in enumerate(results)],
            summaries=summarize_roots(results), calibration_summaries=summarize_calibration(results))
        result['sha256'] = canonical_hash(result)
        atomic_json(output / 'result.json', result)
        return result
    except BaseException as exc:
        error = exc
        atomic_json(output / 'failure.json', dict(complete=False, error=str(exc), completed_roots=len(results)))
        raise
    finally:
        _close_resources(*branch_envs, model, env, primary_error=error)


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = run_probe(args)
    print(json.dumps(result if args.plan_only else dict(
        complete=result['complete'], output=str(args.output.resolve()),
        branch_simulator_decisions=result['branch_simulator_decisions']), indent=2))


if __name__ == '__main__':
    main()
