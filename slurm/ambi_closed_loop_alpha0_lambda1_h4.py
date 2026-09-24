"""Three frozen reward-critic panels: zero alpha, lambda-one mean, and H4 mean."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slurm.ambi_aux_hj_sweep import ENTITY, PROJECT, SEEDS, digest, read, training_summary, validate, write
from slurm.ambi_closed_loop_critics import (
    CHECKPOINT_SHA, CHECKPOINT_STEP, INITIAL_ALPHA, SOURCE_RUN, check_prior,
    source_commit, validate_probe_rows,
)
from slurm.ambi_closed_loop_publish import indexed_episodes
from slurm.ambi_closed_loop_sampled import ACTION_RULE, EXECUTION_KEY, verify_receipt
from slurm.ambi_closed_loop_reward_retrace import (
    ESTIMATOR_KEY, estimator_settings, historical_cell, normalized_config, verify_reference,
)
from slurm.ambi_closed_loop_soft_return import action_rule, matching_protocol

MATRIX = ROOT / 'configs/research/ambi_closed_loop_alpha0_lambda1_h4_575k.json'
REFERENCES = ROOT / 'configs/research/ambi_closed_loop_alpha0_lambda1_h4_refs_575k.json'
GROUP = 'closed-loop-alpha0-lambda1-h4-575k-20260923'
ROUNDS = (1, 2, 4, 6, 8, 10)


def identities():
    """First ten cells are the maximum-budget smoke for each distinct path."""
    alpha = [('alpha_zero', h, e, j) for j in reversed(ROUNDS) for h in (1, 2, 3)
             for e in ('one_step', 'retrace')]
    lam = [('lambda_one_mean', h, 'retrace', j) for j in reversed(ROUNDS) for h in (1, 2, 3)]
    h4 = [('h4_mean', 4, 'one_step', j) for j in (14, 12, 10, 8, 6, 4, 2, 1)]
    return alpha[:6] + lam[:3] + h4[:1] + alpha[6:] + lam[3:] + h4[1:]


def requested_params(arm, h, estimator, j):
    original = historical_cell(min(h, 3), j)['requested_alg_params']
    params = {**original, EXECUTION_KEY: 'policy_sample' if arm == 'alpha_zero' else 'mean',
              **estimator_settings(estimator, h), 'inner_rollout_horizon': h,
              'inner_replay_capacity': max(original['inner_replay_capacity'], 128 * h * j)}
    if arm == 'alpha_zero':
        params.update(inner_entropy_enabled=False, inner_temperature_mode='inherit_outer')
    elif arm == 'lambda_one_mean':
        params['inner_retrace_lambda'] = 1.0
    return params


def cell_name(arm, h, estimator, j):
    return f'{arm}_h{h}_j{j}_c16_{estimator}'


def cells(matrix_path=MATRIX):
    from utils.ambi_research import load_preset_matrix
    matrix = load_preset_matrix(matrix_path)
    assert matrix['source_run'] == SOURCE_RUN
    selectors = ['sweep/' + cell_name(*identity) for identity in identities()]
    assert matrix['evaluation'] == dict(controller_seed=55, seeds=SEEDS, max_steps=500,
                                       togo_return_rollouts=32, default_presets=selectors)
    comparison = matrix['comparisons']['sweep']
    assert comparison['reference'] == 'prior'
    assert comparison['variants']['prior']['alg_params']['inner_operator'] == 'none'
    assert set(comparison['variants']) == {'prior', *[s.split('/')[1] for s in selectors]}
    result = []
    for (arm, h, estimator, j), selector in zip(identities(), selectors):
        name = selector.split('/')[1]
        requested = {**matrix['shared_alg_params'],
                     **matrix['comparisons']['sweep']['variants'][name]['alg_params']}
        assert requested == requested_params(arm, h, estimator, j), name
        result.append(dict(name=name, selector=selector, actual_selector=selector,
            requested_alg_params=requested, params={k:v for k,v in requested.items() if v is not None},
            experiment_arm=arm, alpha_mode='zero' if arm == 'alpha_zero' else 'adaptive',
            H=h, J=j, critic_kind='return_only', estimator=estimator,
            retrace_lambda=requested['inner_retrace_lambda'] if estimator == 'retrace' else None,
            execution_mode=requested[EXECUTION_KEY], initial_alpha=0.0 if arm == 'alpha_zero' else INITIAL_ALPHA,
            comparison={'alpha_zero':'alpha_zero_minus_alpha_on',
                        'lambda_one_mean':'lambda_one_mean_minus_one_step_mean',
                        'h4_mean':'h4_minus_h3_mean'}[arm]))
    assert len(result) == 62
    return result


def expected_config(reference, cell):
    """Declared overrides plus the resolver's explicitly checked derived budgets.

    The reference is always adaptive one-step mean at the same J and H<=3.
    Tests resolve every cell through AMBITDMPC2 and check this whole mapping.
    """
    before = normalized_config(reference)
    assert before['inner_critic_source'] == before['inner_horizon_critic_source'] == 'aux_return'
    assert before['inner_sac_critic_target'] == 'reward_only' and before['inner_terminal_entropy'] == 'none'
    assert before['inner_entropy_enabled'] and before['inner_temperature_mode'] == 'auto'
    assert before[ESTIMATOR_KEY] == 'one_step' and before[EXECUTION_KEY] == 'mean'
    assert before['inner_rounds'] == cell['J'] and before['inner_rollout_horizon'] == min(cell['H'], 3)
    before.update({EXECUTION_KEY:cell['execution_mode'], **estimator_settings(cell['estimator'],cell['H'])})
    if cell['experiment_arm'] == 'alpha_zero':
        before.update(inner_entropy_enabled=False, inner_temperature_mode='inherit_outer',
                      inner_temperature_updates_per_action=0,
                      inner_primary_temperature_updates_per_round=0,
                      inner_primary_temperature_updates_per_action=0,
                      inner_total_optimizer_steps_per_action=20*cell['J'],
                      inner_primary_optimizer_steps_per_action=20*cell['J'])
    elif cell['experiment_arm'] == 'lambda_one_mean':
        before['inner_retrace_lambda'] = 1.0
    elif cell['experiment_arm'] == 'h4_mean':
        capacity = cell['params']['inner_replay_capacity']
        before.update(inner_rollout_horizon=4, inner_horizon=4,
                      inner_model_step_budget=512*cell['J'],
                      inner_horizon_ratio=4/before['train_unroll_horizon'],
                      inner_nominal_transitions_per_round=512, inner_nominal_critic_utd=16/512,
                      inner_replay_capacity=capacity, inner_buffer_size=capacity)
    else:
        raise AssertionError(cell['experiment_arm'])
    return before


def matching_config(actual, reference, cell):
    before, after = expected_config(reference, cell), normalized_config(actual)
    assert before == after, {k:(before.get(k),after.get(k)) for k in before.keys() | after.keys()
                             if before.get(k) != after.get(k)}


def matching_planner(actual, reference, cell):
    """Use the real canonicalizer: default/inactive fields must be omitted identically."""
    from utils.eval_series_data import planner_identity
    expected = planner_identity(expected_config(reference,cell), {}, 'AMBITDMPC2/AMBITDMPC2',
                                action_rule(cell['execution_mode']))
    assert actual == expected, (expected, actual)


def reference_key(reference):
    return reference['H'], reference['J'], reference['estimator'], reference['execution']


def load_reference(pin, inventory):
    from utils.eval_series_data import load_records
    # Reuse the completed J1--10 loader, with a narrow extension for H3/J12,14.
    if pin['J'] <= 10:
        from slurm.ambi_closed_loop_soft_return import load_reference as load_existing
        return load_existing(pin, inventory)
    assert reference_key(pin) in ((3,12,'one_step','mean'), (3,14,'one_step','mean'))
    bundle = Path(pin['bundle']); receipt = read(bundle.parent/'worker-completion.json')
    assert receipt['manifest_sha256'] == pin['manifest_sha256']
    manifest = verify_receipt(bundle,receipt)
    assert manifest['code']['commit'] == pin['source_commit'] and manifest['code']['dirty'] is False
    assert manifest['checkpoint']['source_run'] == SOURCE_RUN
    original = historical_cell(pin['H'],pin['J']); original['actual_selector'] = original['selector']
    validate(bundle,original,checkpoint_sha=CHECKPOINT_SHA,checkpoint_step=CHECKPOINT_STEP)
    run, = manifest['runs']; cfg = normalized_config(run['resolved_config'])
    assert cfg[ESTIMATOR_KEY] == 'one_step' and cfg[EXECUTION_KEY] == 'mean'
    assert cfg['inner_entropy_enabled'] and cfg['inner_temperature_mode'] == 'auto'
    assert manifest['protocol']['action_rule'] == 'tanh_mean'
    indexed_episodes(run['episodes'])
    record, = load_records(bundle,inventory_path=inventory)
    assert record['identity']['backbone'] == SOURCE_RUN and record['metrics']['eval/frozen_state_unchanged']
    publication = read(bundle.parent/'publication-completion.json')
    assert publication['status'] == 'complete' and publication['performance']['published'] == 1
    assert publication['performance']['run_id'] == pin['performance_run_id']
    return {**pin,'bundle':str(bundle.resolve()),'trace_sha256':receipt['trace_sha256'],
            'identity':record['identity'],'record_id':record['record_id'],'episodes':run['episodes'],
            'resolved_config':run['resolved_config'],'protocol':manifest['protocol'],
            'runtime':manifest['code']['runtime']}


def load_prior(pin, inventory):
    from utils.eval_series_data import load_records
    bundle = Path(pin['bundle']); manifest = read(bundle/'manifest.json')
    assert digest(bundle/'manifest.json') == pin['manifest_sha256']
    record, = load_records(bundle,inventory_path=inventory)
    check_prior(manifest,record)
    assert manifest['code']['commit'] == pin['source_commit']
    return {**pin, 'bundle':str(bundle.resolve()), 'episodes':record['episodes'],
            'identity':record['identity'], 'record_id':record['record_id'],
            'protocol':manifest['protocol'],
            'trace_sha256':{name:digest(bundle/name) for name in manifest['runs'][0]['trace_files']}}


def prepare_specs(args, panel):
    """Keep sampled and mean protocols in distinct metadata-only invocations."""
    from evaluate_ambi_checkpoint import evaluate_matrix
    specs = {}
    for mode in ('policy_sample','mean'):
        selectors = [cell['selector'] for cell in panel if cell['execution_mode'] == mode]
        result = evaluate_matrix(args.matrix,args.checkpoint,selectors=selectors,
            seeds=SEEDS,controller_seed=55,max_steps=500,bundle_dir=args.root/'unused',
            checkpoint_inventory=args.inventory,reference_bundle=None,
            eval_series_spec_dir=args.root/'specs'/mode)
        assert result['mode'] == 'evaluation_series_specifications' and set(result['specs']) == set(selectors)
        specs.update(result['specs'])
    assert set(specs) == {cell['selector'] for cell in panel}
    return specs


def prepare(args):
    from utils.eval_series import create_run
    from slurm.ambi_closed_loop_reward_retrace import matching_config as match_sampled_reference
    commit, panel = source_commit(), cells(args.matrix)
    assert digest(args.checkpoint) == CHECKPOINT_SHA
    pins = read(args.references)
    refs = {reference_key(p):load_reference(p,args.inventory) for p in pins['references']}
    expected = {(h,j,e,mode) for h in (1,2,3) for j in ROUNDS
                for e,mode in [('one_step','mean'),('one_step','policy_sample'),('retrace','policy_sample')]}
    expected |= {(3,j,'one_step','mean') for j in (12,14)}
    assert len(refs) == len(pins['references']) and set(refs) == expected
    for (h,j,estimator,execution), reference in refs.items():
        if execution == 'policy_sample':
            mean = refs[(h,j,'one_step','mean')]
            match_sampled_reference(reference['resolved_config'],mean['resolved_config'],
                                    {'H':h,'estimator':estimator})
            matching_protocol(reference['protocol'],mean['protocol'])
            assert indexed_episodes(reference['episodes']).keys() == indexed_episodes(mean['episodes']).keys()
    prior = load_prior(pins['prior_reference'],args.inventory)
    for cell in panel:
        cell['validation_reference'] = refs[(min(cell['H'],3),cell['J'],'one_step','mean')]
        key = ((cell['H'],cell['J'],cell['estimator'],'policy_sample')
               if cell['experiment_arm'] == 'alpha_zero'
               else (min(cell['H'],3),cell['J'],'one_step','mean'))
        cell['paired_reference'] = refs[key]
        assert indexed_episodes(cell['paired_reference']['episodes']).keys() == indexed_episodes(prior['episodes']).keys()
    args.root.mkdir(parents=True,exist_ok=False)
    specs = prepare_specs(args,panel)
    for cell in panel:
        directory = args.root/cell['name']; directory.mkdir()
        spec = read(specs[cell['selector']])
        reference = cell['validation_reference']
        assert spec['identity']['backbone'] == reference['identity']['backbone'] == SOURCE_RUN
        matching_protocol(spec['identity']['protocol'],reference['identity']['protocol'],execution=cell['execution_mode'])
        matching_planner(spec['identity']['planner'],reference['resolved_config'],cell)
        registry = create_run(args.registry,spec,args.group+'-'+cell['name'],PROJECT,ENTITY,'oscar-rgao48')
        cell.update(directory=str(directory),bundle=str(directory/'bundle'),reused=False,
                    run_dir=registry['run_dir'],performance_run_id=registry['run_id'],training_run_id=uuid.uuid4().hex)
    campaign = dict(schema_version=1,group=args.group,label=args.label,
        matrix=str(args.matrix.resolve()),checkpoint=str(args.checkpoint.resolve()),
        checkpoint_step=CHECKPOINT_STEP,checkpoint_sha256=CHECKPOINT_SHA,source_run=SOURCE_RUN,
        inventory=str(args.inventory.resolve()),source_commit=commit,source_dir=str(ROOT),
        initial_alpha=INITIAL_ALPHA,target_entropy=-10.5,cells=panel,references=list(refs.values()),
        H=[1,2,3,4],J=[1,2,4,6,8,10,12,14],estimator=['one_step','retrace'],
        retrace_lambda=[.9,1.0],critic_kind='return_only',execution_modes=['policy_sample','mean'],
        experiment_arms=['alpha_zero','lambda_one_mean','h4_mean'],
        prior_reference=prior,publisher_workers=3,overview_run_id=uuid.uuid4().hex,
        h4_overview_run_id=uuid.uuid4().hex,h4_group=args.group+'-h4-mean',
        h4_label='Return-only H4 | mean | 575k',smoke_indices=list(range(10)))
    write(args.root/'campaign.json',campaign)
    print(f'Prepared {len(panel)} settings; overview {campaign["overview_run_id"]}',flush=True)
    return campaign


def validate_completed(bundle, cell, campaign, *, smoke=False):
    from utils.ambi_benchmark import solver_seed
    seeds, steps = ([101],3) if smoke else (SEEDS,500)
    manifest = validate(bundle,cell,seeds=seeds,steps=steps,paired=False,
                        checkpoint_sha=CHECKPOINT_SHA,checkpoint_step=CHECKPOINT_STEP)
    assert not manifest.get('reference')
    assert manifest['code']['commit'] == campaign['source_commit'] and manifest['code']['dirty'] is False
    assert manifest['checkpoint']['source_run'] == SOURCE_RUN
    reference = cell['validation_reference']; verify_reference(reference)
    verify_reference(cell['paired_reference'])
    matching_protocol(manifest['protocol'],reference['protocol'],execution=cell['execution_mode'],steps=steps)
    assert manifest['code']['runtime'] == reference['runtime']
    run, = manifest['runs']; cfg, result = run['resolved_config'],run['result']
    matching_config(cfg,reference['resolved_config'],cell)
    assert run['selector'] == result['selector'] == cell['selector']
    assert result['action_rule'] == action_rule(cell['execution_mode'])
    assert result['deterministic_execution'] is (cell['execution_mode'] == 'mean')
    assert cfg['compile'] and cfg['compile_strict'] and result['resolved_device'].startswith('cuda')
    for key, expected in cell['requested_alg_params'].items():
        actual = run['config']['alg_params']
        assert key not in actual if expected is None else actual.get(key) == expected, key
    for key, stats in result['model_metrics'].items():
        assert all(math.isfinite(value) for value in stats.values()), key
        if key.endswith('_fallback'):
            assert stats['min'] == stats['mean'] == stats['max'] == 0, key
    for stat in ('mean','min','max'):
        assert math.isclose(result['model_metrics']['inner_alpha_initial'][stat],cell['initial_alpha'],rel_tol=1e-6)
        if cell['alpha_mode'] == 'zero':
            for key in ('inner_alpha','inner_alpha_final','inner_alpha_delta'):
                assert result['model_metrics'][key][stat] == 0, key
        sampled = cell['execution_mode'] == 'policy_sample'
        assert result['model_metrics']['inner_eval_execution_sampled'][stat] == int(sampled)
        distance = result['model_metrics']['inner_eval_execution_mean_action_l2'][stat]
        assert (distance >= 0 if stat == 'min' else distance > 0) if sampled else distance == 0
    baseline = indexed_episodes(cell['paired_reference']['episodes'])
    for episode in run['episodes']:
        assert (episode['seed'],episode['solver_seed']) in baseline
        assert episode['solver_seed'] == solver_seed(55,'episode',episode['seed'])
        assert 'paired_return_delta' not in episode and math.isfinite(episode['return'])
        if not smoke:
            assert not episode['truncated_by_evaluator']
        for row in episode['togo_round_summaries']:
            assert all(stats['count'] == steps for stats in row['metrics'].values())
    validate_probe_rows(run,cell,seeds=seeds,steps=steps)
    return manifest


def worker(args):
    import torch
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.ambi_seed_shards import seal_episode_bundle
    campaign = read(args.root/'campaign.json')
    assert source_commit() == campaign['source_commit']
    assert 0 <= args.index < len(campaign['cells'])
    cell = campaign['cells'][args.index]
    assert (cell['experiment_arm'],cell['H'],cell['estimator'],cell['J']) == identities()[args.index]
    if args.smoke:
        assert args.index in campaign['smoke_indices']
        assert cell['J'] == (14 if cell['experiment_arm'] == 'h4_mean' else 10)
    assert torch.cuda.is_available()
    directory = args.root/'smoke'/cell['name'] if args.smoke else Path(cell['directory'])
    directory.mkdir(parents=True,exist_ok=not args.smoke)
    bundle = directory/'bundle'
    evaluate_matrix(campaign['matrix'],campaign['checkpoint'],selectors=[cell['selector']],
                    seeds=[101] if args.smoke else SEEDS,controller_seed=55,
                    max_steps=3 if args.smoke else 500,device='cuda',bundle_dir=bundle,
                    checkpoint_inventory=campaign['inventory'],reference_bundle=None)
    manifest = validate_completed(bundle,cell,campaign,smoke=args.smoke)
    summary = training_summary(bundle,cell,expected_steps=3 if args.smoke else 500)
    seal_episode_bundle(bundle)
    receipt = dict(status='complete',cell=cell['name'],selector=cell['selector'],bundle=str(bundle),
        reused=False,smoke=args.smoke,execution=cell['execution_mode'],estimator=cell['estimator'],
        experiment_arm=cell['experiment_arm'],alpha_mode=cell['alpha_mode'],
        checkpoint_step=CHECKPOINT_STEP,checkpoint_sha256=CHECKPOINT_SHA,H=cell['H'],J=cell['J'],
        manifest_sha256=digest(bundle/'manifest.json'),
        trace_sha256={name:digest(bundle/name) for name in manifest['runs'][0]['trace_files']},
        trace_rows_checked=summary['trace_rows_checked'],gpu=torch.cuda.get_device_name(0))
    write(directory/'validation.json',receipt); write(directory/'worker-completion.json',receipt)
    print('COMPLETE '+cell['name'],flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    prep = sub.add_parser('prepare')
    for name in ('root','checkpoint','inventory','registry'):
        prep.add_argument('--'+name,type=Path,required=True)
    prep.add_argument('--references',type=Path,default=REFERENCES)
    prep.add_argument('--matrix',type=Path,default=MATRIX)
    prep.add_argument('--group',default=GROUP)
    prep.add_argument('--label',default='Return critics | alpha zero, lambda one, H4 | 575k')
    run = sub.add_parser('worker')
    run.add_argument('--root',type=Path,required=True); run.add_argument('--index',type=int,required=True)
    run.add_argument('--smoke',action='store_true')
    args = parser.parse_args(); {'prepare':prepare,'worker':worker}[args.command](args)


if __name__ == '__main__':
    main()
