"""Frozen reward/reward sampled execution, one-step versus trajectory Retrace."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slurm.ambi_aux_hj_sweep import ENTITY, PROJECT, SEEDS, digest, read, training_summary, validate, write
from slurm.ambi_closed_loop_critics import (
    CHECKPOINT_SHA, CHECKPOINT_STEP, INITIAL_ALPHA, SOURCE_RUN,
    cells as original_cells, source_commit, validate_probe_rows,
)
from slurm.ambi_closed_loop_publish import indexed_episodes
from slurm.ambi_closed_loop_sampled import ACTION_RULE, EXECUTION_KEY, verify_receipt

MATRIX = ROOT / 'configs/research/ambi_closed_loop_reward_retrace_575k.json'
REFERENCES = ROOT / 'configs/research/ambi_closed_loop_reward_retrace_refs_575k.json'
GROUP = 'closed-loop-reward-j10-retrace-575k-20260923'
ESTIMATOR_KEY = 'inner_sac_return_estimator'
RETRACE_DEFAULTS = {ESTIMATOR_KEY: 'one_step', 'inner_retrace_lambda': 1.0,
                    'inner_retrace_batch_trajectories': None}


def identities():
    return ([(h, e, 10) for h in (1, 2, 3) for e in ('one_step', 'retrace')]
            + [(h, e, j) for j in (8, 6, 4, 2, 1) for h in (1, 2, 3)
               for e in ('one_step', 'retrace') if e == 'retrace' or j >= 6])


def historical_cell(horizon, rounds):
    suffix = '' if horizon == 3 else f'_h{horizon}'
    if rounds >= 6:
        suffix = f'_h{horizon}_j{rounds}'
    matrix = ROOT / f'configs/research/ambi_closed_loop_critics{suffix}_575k.json'
    return next(c for c in original_cells(matrix) if c['critic_kind'] == 'return_only' and c['J'] == rounds)


def estimator_settings(estimator, horizon):
    assert estimator in ('one_step', 'retrace')
    return {ESTIMATOR_KEY: estimator, 'inner_retrace_lambda': .9 if estimator == 'retrace' else 1.0,
            'inner_retrace_batch_trajectories': math.ceil(256 / horizon) if estimator == 'retrace' else None}


def cells(matrix_path=MATRIX):
    matrix = read(matrix_path)
    assert matrix['source_run'] == SOURCE_RUN
    expected_evaluation = dict(controller_seed=55, seeds=SEEDS, max_steps=500, togo_return_rollouts=32,
        default_presets=[f'sweep/reward_reward_h{h}_j{j}_c16_{e}_sampled' for h, e, j in identities()])
    assert matrix['evaluation'] == expected_evaluation
    result = []
    for (h, estimator, j), selector in zip(identities(), expected_evaluation['default_presets']):
        name = selector.split('/')[1]
        requested = {**matrix['shared_alg_params'],
                     **matrix['comparisons']['sweep']['variants'][name]['alg_params']}
        original = historical_cell(h, j)
        expected = {**original['requested_alg_params'], EXECUTION_KEY: 'policy_sample',
                    **estimator_settings(estimator, h)}
        assert requested == expected, name
        result.append(dict(name=name, selector=selector, actual_selector=selector,
            requested_alg_params=requested, params={k: v for k, v in requested.items() if v is not None},
            H=h, J=j, critic_kind='return_only', estimator=estimator,
            retrace_lambda=.9 if estimator == 'retrace' else None, execution_mode='policy_sample',
            mean_name=original['name']))
    assert len(result) == 27
    return result


def normalized_config(config):
    return {**RETRACE_DEFAULTS, EXECUTION_KEY: 'mean', **deepcopy(config)}


def matching_config(actual, reference, cell):
    """Only execution and the declared return estimator may change."""
    before, after = normalized_config(reference), normalized_config(actual)
    assert before[ESTIMATOR_KEY] == 'one_step'
    assert before['inner_retrace_lambda'] == 1.0 and before['inner_retrace_batch_trajectories'] is None
    before.update({EXECUTION_KEY: 'policy_sample', **estimator_settings(cell['estimator'], cell['H'])})
    assert after == before, {k: (before.get(k), after.get(k)) for k in before.keys() | after.keys()
                             if before.get(k) != after.get(k)}


def matching_protocol(actual, reference, *, steps=500):
    from utils.ambi_benchmark import episode_protocol
    assert reference['action_rule'] in ('tanh_mean', ACTION_RULE)
    assert actual['action_rule'] == ACTION_RULE
    assert episode_protocol(actual) == episode_protocol({**reference, 'action_rule': ACTION_RULE, 'max_steps': steps})


def reference_key(reference):
    return reference['H'], reference['J'], reference['execution']


def load_reference(pin, inventory):
    from utils.eval_series_data import load_records
    bundle = Path(pin['bundle'])
    receipt = read(bundle.parent / 'worker-completion.json')
    assert receipt['manifest_sha256'] == pin['manifest_sha256']
    manifest = verify_receipt(bundle, receipt)
    assert manifest['code']['commit'] == pin['source_commit'] and manifest['code']['dirty'] is False
    assert manifest['checkpoint']['sha256'] == CHECKPOINT_SHA
    assert manifest['checkpoint']['source_run'] == SOURCE_RUN
    assert pin.get('estimator', 'one_step') == 'one_step'
    h, j, execution = reference_key(pin)
    assert h in (1, 2, 3) and j in (1, 2, 4, 6, 8, 10)
    assert execution in ('mean', 'policy_sample')
    original = historical_cell(h, j)
    run, = manifest['runs']
    expected_selector = original['selector'] + ('_sampled' if execution == 'policy_sample' else '')
    assert run['selector'] == expected_selector
    assert manifest['protocol']['action_rule'] == ('tanh_mean' if execution == 'mean' else ACTION_RULE)
    assert normalized_config(run['resolved_config'])[ESTIMATOR_KEY] == 'one_step'
    if execution == 'policy_sample':
        original['params'][EXECUTION_KEY] = 'policy_sample'
    original['actual_selector'] = expected_selector
    original['selector'] = expected_selector
    validate(bundle, original, paired=execution == 'mean', checkpoint_sha=CHECKPOINT_SHA, checkpoint_step=CHECKPOINT_STEP)
    indexed_episodes(run['episodes'])
    record, = load_records(bundle, inventory_path=inventory)
    assert record['identity']['backbone'] == SOURCE_RUN and record['metrics']['eval/frozen_state_unchanged']
    publication = read(bundle.parent / 'publication-completion.json')
    assert publication['status'] == 'complete' and publication['performance']['published'] == 1
    assert publication['performance']['run_id'] == pin['performance_run_id']
    return {**pin, 'bundle': str(bundle.resolve()), 'trace_sha256': receipt['trace_sha256'],
            'identity': record['identity'], 'record_id': record['record_id'], 'episodes': run['episodes'],
            'resolved_config': run['resolved_config'], 'protocol': manifest['protocol'],
            'runtime': manifest['code']['runtime']}


def verify_reference(reference, *, traces=False):
    bundle = Path(reference['bundle'])
    assert digest(bundle / 'manifest.json') == reference['manifest_sha256']
    manifest = read(bundle / 'manifest.json')
    assert manifest['code']['commit'] == reference['source_commit'] and manifest['code']['dirty'] is False
    assert manifest['checkpoint']['sha256'] == CHECKPOINT_SHA
    run, = manifest['runs']
    assert run['episodes'] == reference['episodes'] and run['resolved_config'] == reference['resolved_config']
    assert manifest['protocol'] == reference['protocol']
    if traces:
        assert set(reference['trace_sha256']) == set(run['trace_files'])
        assert all(digest(bundle / name) == sha for name, sha in reference['trace_sha256'].items())
    return manifest


def prepare(args):
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.eval_series import create_run
    commit, panel = source_commit(), cells(args.matrix)
    assert digest(args.checkpoint) == CHECKPOINT_SHA
    pins = read(args.references)
    pins = pins['references'] if isinstance(pins, dict) else pins
    refs = {reference_key(p): load_reference(p, args.inventory) for p in pins}
    expected = {(h, j, 'mean') for h in (1, 2, 3) for j in (1, 2, 4, 6, 8, 10)}
    expected |= {(h, j, 'policy_sample') for h in (1, 2, 3) for j in (1, 2, 4)}
    assert len(refs) == len(pins) and set(refs) == expected
    for (h, j, execution), reference in refs.items():
        if execution == 'policy_sample':
            mean = refs[(h, j, 'mean')]
            matching_config(reference['resolved_config'], mean['resolved_config'],
                            {'H': h, 'estimator': 'one_step'})
            matching_protocol(reference['protocol'], mean['protocol'])
            assert indexed_episodes(reference['episodes']).keys() == indexed_episodes(mean['episodes']).keys()
    for cell in panel:
        cell['mean_reference'] = refs[(cell['H'], cell['J'], 'mean')]
        if cell['estimator'] == 'retrace':
            cell['one_step_reference'] = refs.get((cell['H'], cell['J'], 'policy_sample'))
            cell['one_step_cell'] = (f'reward_reward_h{cell["H"]}_j{cell["J"]}_c16_one_step_sampled'
                                     if cell['J'] >= 6 else None)
    args.root.mkdir(parents=True, exist_ok=False)
    evaluate_matrix(args.matrix, args.checkpoint, seeds=SEEDS, controller_seed=55, max_steps=500,
                    bundle_dir=args.root / 'unused', checkpoint_inventory=args.inventory,
                    reference_bundle=None, eval_series_spec_dir=args.root / 'specs')
    for cell in panel:
        directory = args.root / cell['name']; directory.mkdir()
        spec = read(args.root / 'specs' / (cell['selector'].replace('/', '__') + '.json'))
        reference = cell['mean_reference']['identity']
        assert spec['identity']['backbone'] == reference['backbone'] == SOURCE_RUN
        matching_protocol(spec['identity']['protocol'], reference['protocol'])
        before, after = deepcopy(reference['planner']), spec['identity']['planner']
        before['action_rule'] = ACTION_RULE
        before['settings'] = normalized_config(before['settings'])
        before['settings'].update({EXECUTION_KEY: 'policy_sample', **estimator_settings(cell['estimator'], cell['H'])})
        after = {**after, 'settings': normalized_config(after['settings'])}
        assert before == after
        registry = create_run(args.registry, spec, args.group + '-' + cell['name'], PROJECT, ENTITY, 'oscar-rgao48')
        cell.update(directory=str(directory), bundle=str(directory / 'bundle'), reused=False,
                    run_dir=registry['run_dir'], performance_run_id=registry['run_id'], training_run_id=uuid.uuid4().hex)
    campaign = dict(schema_version=1, group=args.group, label=args.label,
        matrix=str(args.matrix.resolve()), checkpoint=str(args.checkpoint.resolve()),
        checkpoint_step=CHECKPOINT_STEP, checkpoint_sha256=CHECKPOINT_SHA, source_run=SOURCE_RUN,
        inventory=str(args.inventory.resolve()), source_commit=commit, source_dir=str(ROOT),
        initial_alpha=INITIAL_ALPHA, target_entropy=-10.5, cells=panel, references=list(refs.values()),
        H=[1, 2, 3], J=[1, 2, 4, 6, 8, 10], estimator=['one_step', 'retrace'], retrace_lambda=.9,
        execution_mode='policy_sample', prior_reference=None, publisher_workers=2,
        overview_run_id=uuid.uuid4().hex, smoke_indices=list(range(6)))
    write(args.root / 'campaign.json', campaign)
    print(f'Prepared {len(panel)} settings; overview {campaign["overview_run_id"]}', flush=True)
    return campaign


def validate_completed(bundle, cell, campaign, *, smoke=False):
    from utils.ambi_benchmark import solver_seed
    seeds, steps = ([101], 3) if smoke else (SEEDS, 500)
    manifest = validate(bundle, cell, seeds=seeds, steps=steps, paired=False,
                        checkpoint_sha=CHECKPOINT_SHA, checkpoint_step=CHECKPOINT_STEP)
    assert not manifest.get('reference')
    assert manifest['code']['commit'] == campaign['source_commit'] and manifest['code']['dirty'] is False
    assert manifest['checkpoint']['source_run'] == SOURCE_RUN
    reference = cell['mean_reference']; verify_reference(reference)
    matching_protocol(manifest['protocol'], reference['protocol'], steps=steps)
    assert manifest['code']['runtime'] == reference['runtime']
    run, = manifest['runs']; cfg, result = run['resolved_config'], run['result']
    matching_config(cfg, reference['resolved_config'], cell)
    assert cfg['inner_rollout_horizon'] == cell['H'] and cfg['inner_rounds'] == cell['J']
    assert run['selector'] == result['selector'] == cell['selector']
    assert result['action_rule'] == ACTION_RULE and result['deterministic_execution'] is False
    assert cfg['compile'] and cfg['compile_strict'] and result['resolved_device'].startswith('cuda')
    for key, expected in cell['requested_alg_params'].items():
        actual = run['config']['alg_params']
        assert key not in actual if expected is None else actual.get(key) == expected, key
    for key, stats in result['model_metrics'].items():
        assert all(math.isfinite(value) for value in stats.values()), key
        if key.endswith('_fallback'):
            assert stats['min'] == stats['mean'] == stats['max'] == 0, key
    for stat in ('mean', 'min', 'max'):
        assert math.isclose(result['model_metrics']['inner_alpha_initial'][stat], INITIAL_ALPHA, rel_tol=1e-6)
        assert result['model_metrics']['inner_eval_execution_sampled'][stat] == 1
        distance = result['model_metrics']['inner_eval_execution_mean_action_l2'][stat]
        assert distance >= 0 if stat == 'min' else distance > 0
    mean = indexed_episodes(reference['episodes'])
    for episode in run['episodes']:
        assert (episode['seed'], episode['solver_seed']) in mean
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
    assert 0 <= args.index < len(campaign['cells'])
    cell = campaign['cells'][args.index]
    assert (cell['H'], cell['estimator'], cell['J']) == identities()[args.index]
    if args.smoke:
        assert args.index in campaign['smoke_indices'] and cell['J'] == 10
    assert torch.cuda.is_available()
    directory = args.root / 'smoke' / cell['name'] if args.smoke else Path(cell['directory'])
    directory.mkdir(parents=True, exist_ok=not args.smoke)
    bundle = directory / 'bundle'
    evaluate_matrix(campaign['matrix'], campaign['checkpoint'], selectors=[cell['selector']],
                    seeds=[101] if args.smoke else SEEDS, controller_seed=55,
                    max_steps=3 if args.smoke else 500, device='cuda', bundle_dir=bundle,
                    checkpoint_inventory=campaign['inventory'], reference_bundle=None)
    manifest = validate_completed(bundle, cell, campaign, smoke=args.smoke)
    summary = training_summary(bundle, cell, expected_steps=3 if args.smoke else 500)
    seal_episode_bundle(bundle)
    receipt = dict(status='complete', cell=cell['name'], selector=cell['selector'], bundle=str(bundle),
        reused=False, smoke=args.smoke, execution='policy_sample', estimator=cell['estimator'],
        checkpoint_step=CHECKPOINT_STEP, checkpoint_sha256=CHECKPOINT_SHA, H=cell['H'], J=cell['J'],
        manifest_sha256=digest(bundle / 'manifest.json'),
        trace_sha256={name: digest(bundle / name) for name in manifest['runs'][0]['trace_files']},
        trace_rows_checked=summary['trace_rows_checked'], gpu=torch.cuda.get_device_name(0))
    write(directory / 'validation.json', receipt); write(directory / 'worker-completion.json', receipt)
    print('COMPLETE ' + cell['name'], flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('root', 'checkpoint', 'inventory', 'registry'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--references', type=Path, default=REFERENCES)
    prep.add_argument('--matrix', type=Path, default=MATRIX)
    prep.add_argument('--group', default=GROUP)
    prep.add_argument('--label', default='Reward/reward sampled execution | one-step vs Retrace | H1/H2/H3 J1–10 | 575k')
    run = sub.add_parser('worker')
    run.add_argument('--root', type=Path, required=True); run.add_argument('--index', type=int, required=True)
    run.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); {'prepare': prepare, 'worker': worker}[args.command](args)


if __name__ == '__main__':
    main()
