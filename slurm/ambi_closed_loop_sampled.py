"""Frozen return-critic evaluation with sampled final-action execution.

Historical mean-execution bundles remain immutable. They are explicit
cross-action comparators, never mislabeled as a sampled-policy prior reference.
"""
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
    campaign_horizon, cells as mean_cells, source_commit, validate_probe_rows,
)
from slurm.ambi_closed_loop_publish import indexed_episodes

MATRIX = ROOT / 'configs/research/ambi_closed_loop_sampled_h3_575k.json'
GROUP = 'closed-loop-sampled-h3-575k-20260923'
MEAN_SOURCE_COMMIT = '95dbae6769e976970e94a3987b070b31dbbd5b37'
ACTION_RULE = 'squashed_gaussian_sample'
EXECUTION_KEY = 'inner_eval_execution_action'
MEAN_RUN_IDS = {1: '29e3eb9b9601408892f19c7a8f122f6b',
                2: 'd901130bd24d449b9bd522f6c08075f7',
                4: '19a3a76204164edfb6d2a479eac32f3d'}
MEAN_MANIFEST_SHA = {1: 'e763d7434ab1e0deec681c60ceaa9a1d59124a32713154c9231b5388f87a2b68',
                     2: '3e910b892d4b5f0ab0db43b21c406b71b8d31cdf302c1ec83892903e872f4b21',
                     4: 'f6c45d7176a446286b31416cb62ef47c8d59a80db5c131cd814d82d976eded3c'}
# Keep the original constants as H3 aliases for existing callers and campaigns.
MEAN_SOURCE_COMMITS = {1: '4f98943012ff3019fe0d556eef9654c481486fc6',
                       2: '589ed0c0ba469245085af4db0ae740c692786695',
                       3: MEAN_SOURCE_COMMIT}
MEAN_RUN_IDS_BY_H = {
    1: {1: 'c0b322af35be413d926413edecdc4577', 2: '7a2d91b77e3e40ecae0eeda84d105436',
        4: '317a78ebb0a6448392bbc1403168ae45'},
    2: {1: 'dbfeac96d48d4c35af37ddc10217e3eb', 2: '70cf2d82db1242a89318bf579e7f3793',
        4: 'afbc2cba4f034c009942d6df813f60ce'},
    3: MEAN_RUN_IDS,
}
MEAN_MANIFEST_SHA_BY_H = {
    1: {1: '3ba92866e9986e04c0604685e4362aeae5b02c132194f12a28ce2676f590e709',
        2: 'ad94caa3290b5f4c700a267dc1993ad2daa4c10b887d9a310d32d2422bc32f3e',
        4: '9ac47a0c305a580046025c61e2904516c60a48d1d375fadfcd2ae6a5a1e62bab'},
    2: {1: '81aff60813f50651a674771bd5919523350c9eecdba74ced94d7c0007659d44f',
        2: '2871399343f57ccd8d753baf6ad9647d1b8158f382c7e2f0d5e8d19c00d23741',
        4: 'f10b5d5dbf1f67502131f6c9737e687b6b0d578e0c66522e47e8bdcc261450f2'},
    3: MEAN_MANIFEST_SHA,
}


def mean_matrix(horizon):
    assert horizon in (1, 2, 3), 'Expected rollout horizon H1, H2 or H3'
    name = 'ambi_closed_loop_critics_575k.json' if horizon == 3 else f'ambi_closed_loop_critics_h{horizon}_575k.json'
    return ROOT / 'configs/research' / name


def cells(matrix_path=MATRIX):
    matrix = read(matrix_path)
    horizon = matrix['shared_alg_params']['inner_rollout_horizon']
    old = read(mean_matrix(horizon))
    assert matrix['source_run'] == SOURCE_RUN
    expected_evaluation = {**old['evaluation'], 'default_presets': [
        f'sweep/return_return_alpha_h{horizon}_j{j}_c16_sampled' for j in (4, 2, 1)]}
    assert matrix['evaluation'] == expected_evaluation
    assert matrix['shared_alg_params'] == {**old['shared_alg_params'], EXECUTION_KEY: 'policy_sample'}
    historical = {c['J']: c for c in mean_cells(mean_matrix(horizon)) if c['critic_kind'] == 'return_only'}
    result = []
    for j, selector in zip((4, 2, 1), matrix['evaluation']['default_presets']):
        name = selector.split('/')[1]
        variant = matrix['comparisons']['sweep']['variants'][name]
        expected = old['comparisons']['sweep']['variants'][name.removesuffix('_sampled')]['alg_params']
        assert variant['alg_params'] == expected
        cell = deepcopy(historical[j])
        cell.update(name=name, selector=selector, execution_mode='policy_sample',
                    mean_name=name.removesuffix('_sampled'))
        cell['params'][EXECUTION_KEY] = 'policy_sample'
        cell['requested_alg_params'][EXECUTION_KEY] = 'policy_sample'
        result.append(cell)
    return result


def matching_protocol(sampled, mean, *, steps=500):
    """Allow exactly the named action-rule change in the environment protocol."""
    from utils.ambi_benchmark import episode_protocol
    assert mean['action_rule'] == 'tanh_mean'
    assert sampled['action_rule'] == ACTION_RULE
    assert episode_protocol(sampled) == episode_protocol({**mean, 'action_rule': ACTION_RULE, 'max_steps': steps})


def matching_config(sampled, mean):
    """All inherited and resolved scientific settings must otherwise agree."""
    assert sampled.get(EXECUTION_KEY) == 'policy_sample'
    assert mean.get(EXECUTION_KEY, 'mean') == 'mean'
    before, after = deepcopy(mean), deepcopy(sampled)
    before.pop(EXECUTION_KEY, None)
    after.pop(EXECUTION_KEY)
    assert after == before, {key: (before.get(key), after.get(key))
                             for key in before.keys() | after.keys() if before.get(key) != after.get(key)}


def verify_receipt(bundle, receipt):
    bundle = Path(bundle)
    assert receipt['status'] == 'complete'
    assert digest(bundle / 'manifest.json') == receipt['manifest_sha256']
    manifest = read(bundle / 'manifest.json')
    run, = manifest['runs']
    assert set(receipt['trace_sha256']) == set(run['trace_files'])
    assert all(digest(bundle / name) == sha for name, sha in receipt['trace_sha256'].items())
    return manifest


def historical_reference(root, cell, inventory):
    from utils.eval_series_data import load_records
    directory = Path(root) / cell['mean_name']
    bundle = directory / 'bundle'
    receipt = read(directory / 'worker-completion.json')
    horizon = cell['H']
    mean_commit = MEAN_SOURCE_COMMITS[horizon]
    assert receipt['manifest_sha256'] == MEAN_MANIFEST_SHA_BY_H[horizon][cell['J']]
    manifest = verify_receipt(bundle, receipt)
    original = next(c for c in mean_cells(mean_matrix(horizon)) if c['name'] == cell['mean_name'])
    validate(bundle, original, checkpoint_sha=CHECKPOINT_SHA, checkpoint_step=CHECKPOINT_STEP)
    assert manifest['code']['commit'] == mean_commit and manifest['code']['dirty'] is False
    assert manifest['checkpoint']['source_run'] == SOURCE_RUN
    assert manifest['protocol']['action_rule'] == 'tanh_mean'
    run, = manifest['runs']
    assert run['selector'] == original['selector']
    indexed_episodes(run['episodes'])
    record, = load_records(bundle, inventory_path=inventory)
    assert record['identity']['backbone'] == SOURCE_RUN
    assert record['identity']['protocol']['action_rule'] == 'tanh_mean'
    assert record['metrics']['eval/frozen_state_unchanged']
    publication = read(directory / 'publication-completion.json')
    assert publication['status'] == 'complete'
    assert publication['performance']['run_id'] == MEAN_RUN_IDS_BY_H[horizon][cell['J']]
    assert publication['performance']['published'] == 1
    return dict(bundle=str(bundle.resolve()), manifest_sha256=receipt['manifest_sha256'],
                trace_sha256=receipt['trace_sha256'], source_commit=mean_commit, H=horizon, J=cell['J'],
                science=record['identity']['science'], identity=record['identity'],
                record_id=record['record_id'], performance_run_id=MEAN_RUN_IDS_BY_H[horizon][cell['J']],
                episodes=run['episodes'], resolved_config=run['resolved_config'],
                protocol=manifest['protocol'], runtime=manifest['code']['runtime'])


def verify_mean_reference(reference, *, traces=False):
    path = Path(reference['bundle']) / 'manifest.json'
    assert digest(path) == reference['manifest_sha256'], 'Historical mean manifest changed'
    manifest = read(path)
    horizon = reference['resolved_config']['inner_rollout_horizon']
    rounds = reference['resolved_config']['inner_rounds']
    assert reference.get('H', horizon) == horizon and reference.get('J', rounds) == rounds
    assert manifest['code']['commit'] == reference['source_commit'] == MEAN_SOURCE_COMMITS[horizon]
    assert reference['manifest_sha256'] == MEAN_MANIFEST_SHA_BY_H[horizon][rounds]
    assert manifest['checkpoint']['sha256'] == CHECKPOINT_SHA
    run, = manifest['runs']
    assert run['episodes'] == reference['episodes']
    assert run['resolved_config'] == reference['resolved_config']
    assert manifest['protocol'] == reference['protocol']
    if traces:
        assert set(reference['trace_sha256']) == set(run['trace_files'])
        assert all(digest(path.parent / name) == sha for name, sha in reference['trace_sha256'].items())
    return manifest


def prepare(args):
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.eval_series import create_run
    commit = source_commit()
    panel = cells(args.matrix)
    horizon = campaign_horizon({'cells': panel})
    group = args.group or f'closed-loop-sampled-h{horizon}-575k-20260923'
    label = args.label or f'H{horizon} return-only | sampled vs mean execution | J1/J2/J4 | 575k'
    assert digest(args.checkpoint) == CHECKPOINT_SHA
    for cell in panel:
        cell['mean_reference'] = historical_reference(args.mean_reference_root, cell, args.inventory)
    args.root.mkdir(parents=True, exist_ok=False)
    evaluate_matrix(args.matrix, args.checkpoint, seeds=SEEDS, controller_seed=55, max_steps=500,
                    bundle_dir=args.root / 'unused', checkpoint_inventory=args.inventory,
                    reference_bundle=None, eval_series_spec_dir=args.root / 'specs')
    for cell in panel:
        directory = args.root / cell['name']
        directory.mkdir()
        spec = read(args.root / 'specs' / (cell['selector'].replace('/', '__') + '.json'))
        before, after = cell['mean_reference']['identity'], spec['identity']
        assert before['backbone'] == after['backbone'] == SOURCE_RUN
        matching_protocol(after['protocol'], before['protocol'])
        expected = deepcopy(before['planner'])
        expected['action_rule'] = ACTION_RULE
        expected['settings'][EXECUTION_KEY] = 'policy_sample'
        assert after['planner'] == expected
        registry = create_run(args.registry, spec, group + '-' + cell['name'], PROJECT, ENTITY, 'oscar-rgao48')
        cell.update(directory=str(directory), bundle=str(directory / 'bundle'),
                    actual_selector=cell['selector'], reused=False, run_dir=registry['run_dir'],
                    performance_run_id=registry['run_id'], training_run_id=uuid.uuid4().hex)
    campaign = dict(schema_version=1, group=group, label=label, H=horizon,
                    matrix=str(args.matrix.resolve()), checkpoint=str(args.checkpoint.resolve()),
                    checkpoint_step=CHECKPOINT_STEP, checkpoint_sha256=CHECKPOINT_SHA,
                    source_run=SOURCE_RUN, inventory=str(args.inventory.resolve()),
                    source_commit=commit, source_dir=str(ROOT), initial_alpha=INITIAL_ALPHA,
                    target_entropy=-10.5, overview_run_id=uuid.uuid4().hex, cells=panel, publisher_workers=2,
                    execution_mode='policy_sample', mean_source_commit=MEAN_SOURCE_COMMITS[horizon],
                    mean_reference_root=str(args.mean_reference_root.resolve()),
                    comparison='Sampled minus historical mean execution of the same adapted return-only actor recipe.',
                    prior_reference=None)
    write(args.root / 'campaign.json', campaign)
    print(f'Prepared {len(panel)} sampled settings; overview {campaign["overview_run_id"]}', flush=True)
    return campaign


def validate_completed(bundle, cell, campaign, *, smoke=False):
    from utils.ambi_benchmark import solver_seed
    seeds, steps = ([101], 3) if smoke else (SEEDS, 500)
    manifest = validate(bundle, cell, seeds=seeds, steps=steps, paired=False,
                        checkpoint_sha=CHECKPOINT_SHA, checkpoint_step=CHECKPOINT_STEP)
    assert not manifest.get('reference'), 'Sampled bundles must not attach a mean-action prior reference'
    assert manifest['code']['commit'] == campaign['source_commit'] and manifest['code']['dirty'] is False
    assert manifest['checkpoint']['source_run'] == SOURCE_RUN
    reference = cell['mean_reference']
    verify_mean_reference(reference)
    matching_protocol(manifest['protocol'], reference['protocol'], steps=steps)
    assert manifest['code']['runtime'] == reference['runtime']
    run, = manifest['runs']
    cfg, result = run['resolved_config'], run['result']
    assert cfg['inner_rollout_horizon'] == cell['H'] == campaign_horizon(campaign)
    matching_config(cfg, reference['resolved_config'])
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
        key = (episode['seed'], episode['solver_seed'])
        assert key in mean and episode['solver_seed'] == solver_seed(55, 'episode', episode['seed'])
        assert 'paired_return_delta' not in episode
        assert math.isfinite(episode['return'])
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
    horizon = campaign_horizon(campaign)
    assert cell['H'] == horizon and cell['J'] in (1, 2, 4)
    if args.smoke:
        assert cell['J'] == 4, 'Smoke must exercise the largest round budget'
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
    receipt = dict(status='complete', cell=cell['name'], selector=cell['selector'],
                   bundle=str(bundle), reused=False, smoke=args.smoke, execution='policy_sample',
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
    for name in ('root', 'checkpoint', 'inventory', 'mean-reference-root', 'registry'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--matrix', type=Path, default=MATRIX)
    prep.add_argument('--group', help='Default: horizon-specific sampled campaign group')
    prep.add_argument('--label', help='Default: horizon-specific sampled versus mean comparison label')
    run = sub.add_parser('worker')
    run.add_argument('--root', type=Path, required=True)
    run.add_argument('--index', type=int, required=True)
    run.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    {'prepare': prepare, 'worker': worker}[args.command](args)


if __name__ == '__main__':
    main()
