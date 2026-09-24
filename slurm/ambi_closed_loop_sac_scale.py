"""575k return/return rollout and batch scaling with critic-clock SAC updates."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import gzip
import itertools
import json
import math
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slurm.ambi_aux_hj_sweep import ENTITY, PROJECT, SEEDS, Moments, digest, read, write
from slurm.ambi_closed_loop_checkpoint_sweep import (
    checkpoint_state_proof, load_prior, matching_config, resolve_config, verify_reference,
)
from slurm.ambi_closed_loop_critics import CHECKPOINT_SHA, CHECKPOINT_STEP, INITIAL_ALPHA, SOURCE_RUN, source_commit
from slurm.ambi_closed_loop_publish import indexed_episodes
from slurm.ambi_closed_loop_reward_retrace import historical_cell

MATRIX = ROOT / 'configs/research/ambi_closed_loop_sac_scale_575k.json'
REFERENCES = ROOT / 'configs/research/ambi_closed_loop_h3_j10_checkpoint_refs.json'
GROUP = 'closed-loop-sac-scale-575k-20260924'
ROUNDS = (1, 2, 4, 6, 8, 10)
SMOKE_STEPS = 8


def requested_params(h, j, n=1024, b=4096, g=20, p=5, t=2):
    assert type(h) is int and h in (1, 2, 3)
    assert type(j) is int and 1 <= j <= 10
    assert all(type(v) is int and v > 0 for v in (n, b, g, p, t))
    return {**historical_cell(1, 1)['requested_alg_params'],
        'inner_rounds': j, 'inner_rollout_horizon': h, 'inner_rollouts_per_round': n,
        'inner_batch_size': b, 'inner_replay_capacity': max(3072, j*n*h),
        'inner_critic_updates_per_round': None, 'inner_actor_updates_per_round': None,
        'inner_updates_per_round': g, 'inner_actor_update_interval': p,
        'inner_critic_target_update_interval': t,
        'inner_eval_execution_action': 'mean', 'inner_sac_return_estimator': 'one_step',
        'inner_retrace_lambda': 1.0, 'inner_retrace_batch_trajectories': None}


def cell_name(h, j, n, b, g, p, t):
    return f'return_return_h{h}_j{j}_n{n}_b{b}_g{g}_p{p}_t{t}'


def make_matrix(*, horizons=(1, 2, 3), rounds=ROUNDS, rollouts=(1024,), batches=(4096,),
                actor_intervals=(5, 1), critic_steps=20, target_interval=2):
    """Build a concrete, versionable grid; requested changes need no worker edits."""
    axes = dict(horizons=list(horizons), rounds=list(rounds), rollouts=list(rollouts),
                batches=list(batches), actor_intervals=list(actor_intervals),
                critic_steps=critic_steps, target_interval=target_interval)
    assert all(values and len(set(values)) == len(values) for values in
               (horizons, rounds, rollouts, batches, actor_intervals))
    shared = requested_params(1, 1)
    variants = {'prior': {'description': 'Existing frozen policy-mean reference; never rerun.',
        'alg_params': dict(inner_operator='none', inner_rounds=0, inner_rollouts_per_round=0,
            inner_updates_per_round=0, inner_actor_update_interval=None,
            inner_critic_updates_per_round=None, inner_actor_updates_per_round=None,
            inner_finite_horizon=False, inner_critic_source='sac', inner_horizon_critic_source='sac',
            inner_terminal_entropy='none', inner_entropy_enabled=False,
            inner_temperature_mode='inherit_outer')}}
    selectors = []
    for h, j, n, b, p in itertools.product(horizons, rounds, rollouts, batches, actor_intervals):
        name = cell_name(h, j, n, b, critic_steps, p, target_interval)
        selectors.append('sweep/'+name)
        params = requested_params(h, j, n, b, critic_steps, p, target_interval)
        variants[name] = dict(alg_params={k:v for k,v in params.items() if v != shared[k]})
    return dict(schema_version=1, description='575k return/return SAC scaling: fresh full-episode mean execution.',
        base_alg_config='checkpoint', source_run=SOURCE_RUN, checkpoint_steps=[CHECKPOINT_STEP],
        sac_scale_grid=axes, shared_alg_params=shared,
        evaluation=dict(controller_seed=55, seeds=SEEDS, max_steps=500,
                        togo_return_rollouts=32, default_presets=selectors),
        comparisons=dict(sweep=dict(reference='prior', variants=variants)))


def cells(matrix_path=MATRIX):
    from utils.ambi_research import load_preset_matrix
    matrix = load_preset_matrix(matrix_path)
    axes = matrix['sac_scale_grid']
    assert matrix == make_matrix(**axes), 'Matrix differs from its declared scientific grid.'
    result = []
    for selector in matrix['evaluation']['default_presets']:
        name = selector.split('/')[1]
        params = {**matrix['shared_alg_params'], **matrix['comparisons']['sweep']['variants'][name]['alg_params']}
        h,j,n,b,g,p,t = (params[k] for k in ('inner_rollout_horizon','inner_rounds',
            'inner_rollouts_per_round','inner_batch_size','inner_updates_per_round',
            'inner_actor_update_interval','inner_critic_target_update_interval'))
        result.append(dict(name=name, selector=selector, actual_selector=selector,
            checkpoint_step=CHECKPOINT_STEP, training_decisions=CHECKPOINT_STEP,
            requested_alg_params=deepcopy(params), params={k:v for k,v in params.items() if v is not None},
            H=h,J=j,N=n,B=b,G=g,P=p,T=t,C=g,A=g//p,target_updates=g//t,
            critic_kind='return_only', estimator='one_step', execution_mode='mean',
            alpha_mode='adaptive', reused=False))
    return result


def prepare(args):
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.eval_series import create_run
    from utils.eval_series_data import planner_identity
    panel = cells(args.matrix); commit = source_commit(); pins = read(args.references)
    inventory = read(args.inventory)
    assert inventory['source_run'] == pins['source_run'] == SOURCE_RUN
    row, = [r for r in inventory['checkpoints'] if r['step'] == CHECKPOINT_STEP]
    pin, = [r for r in pins['prior_references'] if r['checkpoint_step'] == CHECKPOINT_STEP]
    prior = load_prior(pin, args.inventory)
    assert row['sha256'] == prior['checkpoint_sha256'] == CHECKPOINT_SHA
    assert row['metadata_sha256'] == prior['metadata_sha256']
    assert digest(row['path']) == row['sha256'] and digest(row['metadata_path']) == row['metadata_sha256']
    assert Path(row['metadata_path']) == Path(row['path']+'.metadata.json')
    assert read(row['metadata_path']) == read(Path(prior['bundle'])/'manifest.json')['checkpoint']['metadata']
    proof = checkpoint_state_proof(row['path'], prior)
    assert math.isclose(proof['initial_alpha'], INITIAL_ALPHA, rel_tol=1e-6)
    args.root.mkdir(parents=True, exist_ok=False)
    for cell in panel:
        directory = args.root/cell['name']; directory.mkdir()
        cell.update(checkpoint=row['path'], checkpoint_sha256=row['sha256'],
            metadata_sha256=row['metadata_sha256'], prior_reference=deepcopy(prior),
            initial_alpha=proof['initial_alpha'], checkpoint_state_proof=deepcopy(proof),
            expected_config=resolve_config(args.matrix, row['path'], cell['selector']))
        result = evaluate_matrix(args.matrix, row['path'], selectors=[cell['selector']],
            seeds=SEEDS, controller_seed=55, max_steps=500, bundle_dir=directory/'unused',
            checkpoint_inventory=args.inventory, reference_bundle=prior['bundle'],
            eval_series_spec_dir=directory/'specs')
        assert result['mode'] == 'evaluation_series_specifications' and set(result['specs']) == {cell['selector']}
        spec = read(result['specs'][cell['selector']])
        assert spec['identity']['backbone'] == prior['identity']['backbone'] == SOURCE_RUN
        assert spec['identity']['protocol'] == prior['identity']['protocol']
        assert spec['identity']['planner'] == planner_identity(cell['expected_config'], {}, 'AMBITDMPC2/AMBITDMPC2', 'tanh_mean')
        registry = create_run(args.registry, spec, args.group+'-'+cell['name'], PROJECT, ENTITY, 'oscar-rgao48')
        cell.update(directory=str(directory), bundle=str(directory/'bundle'), identity=spec['identity'],
            run_dir=registry['run_dir'], performance_run_id=registry['run_id'], training_run_id=uuid.uuid4().hex)
    # Maximum-work H1/H3 cover both target-boundary paths and each policy cadence.
    smoke = [i for i,c in enumerate(panel) if c['H'] in (1, 3)
             and c['J'] == max(x['J'] for x in panel) and c['N'] == max(x['N'] for x in panel)
             and c['B'] == max(x['B'] for x in panel)]
    campaign = dict(schema_version=1, group=args.group, label=args.label, matrix=str(args.matrix.resolve()),
        matrix_sha256=digest(args.matrix),
        inventory=str(args.inventory.resolve()), source_commit=commit, source_dir=str(ROOT), source_run=SOURCE_RUN,
        checkpoint_step=CHECKPOINT_STEP, checkpoint_sha256=CHECKPOINT_SHA,
        checkpoint_steps=[CHECKPOINT_STEP]*len(panel), target_entropy=-10.5, cells=panel,
        production_indices=list(range(len(panel))), smoke_indices=smoke, smoke_steps=SMOKE_STEPS,
        publisher_workers=3, overview_run_id=uuid.uuid4().hex,
        sac_scale_grid=read(args.matrix)['sac_scale_grid'],
        prior_compatibility_note='Verified prior mean episodes retain their historical identity and are reused without evaluation.',
        schedule='Each critic slot shares its minibatch with a due actor/temperature update; target cadence is independent.')
    write(args.root/'campaign.json', campaign)
    print(f'Prepared {len(panel)} new SAC-scale evaluations, {len(smoke)} smoke cells; overview {campaign["overview_run_id"]}', flush=True)
    return campaign


def validate_probe_rows(run, cell, *, seeds, steps):
    expected = {(f'seed-{seed}', d, r) for seed in seeds for d in range(steps) for r in range(cell['J']+1)}
    rows = run['togo_probe_rows']
    assert len(rows) == len(expected)
    assert {(r['episode_id'],r['decision_index'],r['round_index']) for r in rows} == expected
    for row in rows:
        r,metrics = row['round_index'],row['metrics']
        assert row['critic_updates'] == cell['G']*r and row['actor_updates'] == cell['G']*r//cell['P']
        assert all(isinstance(v,(int,float)) and math.isfinite(v) for v in metrics.values())
        factor = 2 if r == 0 else 1
        assert metrics['probe_model_steps'] == 32*cell['H']*factor
        assert metrics['probe_q_evaluations'] == 32*factor


def validate_completed(bundle, cell, campaign, *, smoke=False):
    from utils.ambi_benchmark import episode_protocol, solver_seed
    seeds,steps = ([101], campaign.get('smoke_steps', SMOKE_STEPS)) if smoke else (SEEDS,500)
    manifest = read(Path(bundle)/'manifest.json')
    assert manifest['status'] == 'complete' and manifest['code']['dirty'] is False
    assert manifest['code']['commit'] == campaign['source_commit']
    assert manifest['checkpoint']['source_run'] == SOURCE_RUN
    assert manifest['checkpoint']['sha256'] == cell['checkpoint_sha256'] == CHECKPOINT_SHA
    assert manifest['checkpoint']['metadata']['checkpoint']['step'] == CHECKPOINT_STEP
    prior = cell['prior_reference']; verify_reference(prior)
    assert manifest['code']['runtime'] == prior['runtime']
    assert episode_protocol(manifest['protocol']) == episode_protocol({**prior['protocol'], 'max_steps':steps})
    run, = manifest['runs']; cfg,result = run['resolved_config'],run['result']
    assert run['status'] == 'complete'
    matching_config(cfg, cell['expected_config'])
    assert run['selector'] == result['selector'] == cell['selector']
    assert result['action_rule'] == 'tanh_mean' and result['deterministic_execution']
    assert cfg['compile'] and cfg['compile_strict'] and result['resolved_device'].startswith('cuda')
    for key,value in cell['requested_alg_params'].items():
        actual = run['config']['alg_params']
        assert key not in actual if value is None else actual.get(key) == value, key
    for component in ('actor','critic','temperature','replay','actor_optimizer','critic_optimizer','temperature_optimizer'):
        assert cfg[f'inner_{component}_scope'] == 'action'
    assert cfg['inner_critic_source'] == cfg['inner_horizon_critic_source'] == 'aux_return'
    assert cfg['inner_sac_critic_target'] == 'reward_only' and cfg['inner_terminal_entropy'] == 'none'
    assert cfg['inner_entropy_enabled'] and cfg['inner_temperature_mode'] == 'auto'
    assert result['outer_state_unchanged'] and result['outer_updates_before'] == result['outer_updates_after']
    assert not result['nonfinite_model_metrics'] and not result['nonfinite_trace_metrics']
    assert result['environment_seeds'] == seeds and result['controller_seed'] == 55
    for key,stats in result['model_metrics'].items():
        assert all(math.isfinite(v) for v in stats.values()), key
        if key.endswith('_fallback'):
            assert stats['min'] == stats['mean'] == stats['max'] == 0, key
    j,h,n,g,p,t = (cell[k] for k in ('J','H','N','G','P','T'))
    expected = dict(inner_model_steps=n*h*j, inner_buffer_size=n*h*j,
        inner_replay_draws=g*j*cell['B'],
        inner_critic_optimizer_steps=g*j, inner_actor_optimizer_steps=g*j//p,
        inner_temperature_optimizer_steps=g*j//p, inner_critic_target_updates=g*j//t,
        inner_actor_target_updates=0, inner_target_updates=g*j//t,
        inner_eval_execution_sampled=0, inner_eval_execution_mean_action_l2=0)
    for key,value in expected.items():
        for stat in ('mean','min','max'):
            assert result['model_metrics'][key][stat] == value, (key,stat)
    for stat in ('mean','min','max'):
        assert math.isclose(result['model_metrics']['inner_alpha_initial'][stat],cell['initial_alpha'],rel_tol=1e-6)
    probe = run['togo_return_probe']
    assert probe['rollouts'] == 32 and probe['horizon'] == h
    assert not probe['entropy_bonus'] and probe['cadence'] == 'initial_and_after_each_round'
    assert [e['seed'] for e in run['episodes']] == seeds and len(run['trace_files']) == len(seeds)
    baseline = indexed_episodes(prior['episodes'])
    assert not manifest.get('reference') if smoke else manifest['reference']['manifest_sha256'] == prior['manifest_sha256']
    for episode in run['episodes']:
        key = (episode['seed'],episode['solver_seed']); assert key in baseline
        assert episode['solver_seed'] == solver_seed(55,'episode',episode['seed'])
        assert episode['length'] == steps and math.isfinite(episode['return'])
        if smoke:
            assert 'paired_return_delta' not in episode
        else:
            assert not episode['truncated_by_evaluator']
            assert math.isclose(episode['paired_return_delta'],episode['return']-baseline[key]['return'],abs_tol=1e-9)
        assert [(r['round_index'],r['critic_updates'],r['actor_updates']) for r in episode['togo_round_summaries']] == [
            (r,r*g,r*g//p) for r in range(j+1)]
        for row in episode['togo_round_summaries']:
            assert all(stats['count'] == steps for stats in row['metrics'].values())
    validate_probe_rows(run,cell,seeds=seeds,steps=steps)
    return manifest


def training_summary(bundle, cell, *, expected_steps=500):
    """Stream traces, verify the critic clock, and retain update/decision diagnostics."""
    manifest = read(Path(bundle)/'manifest.json'); run, = manifest['runs']
    curves = defaultdict(lambda: defaultdict(Moments)); decisions = defaultdict(lambda: defaultdict(Moments))
    counts = defaultdict(lambda: defaultdict(int)); row_count = 0
    expected_ids = {f'seed-{e["seed"]}' for e in run['episodes']}
    g,p,t,n,h,j = (cell[k] for k in ('G','P','T','N','H','J'))
    for name in run['trace_files']:
        with gzip.open(Path(bundle)/name,'rt') as stream:
            for line in stream:
                e = json.loads(line); row_count += 1
                ep,decision = e['episode_id'],e['decision_index']; key = (ep,decision); count = counts[key]
                assert ep in expected_ids and 0 <= decision < expected_steps
                assert not e.get('nonfinite'), e.get('nonfinite')
                for value in e['metrics'].values():
                    assert isinstance(value,(int,float)) and math.isfinite(value)
                phase = e['phase']
                if phase == 'initial':
                    assert not count and e['replay_size'] == 0
                    assert e['critic_updates'] == e['actor_updates'] == e['temperature_updates'] == 0
                    count['initial'] += 1
                elif phase == 'collection':
                    r = e['round_index']; assert count['initial'] == 1
                    assert r == count['collection']+1 and count['critic'] == (r-1)*g
                    assert e['replay_size'] == r*n*h
                    count['collection'] += 1
                elif phase == 'update':
                    r = e['round_index']; assert count['collection'] == r and count['decision'] == 0
                    c = count['critic']+1
                    assert (r-1)*g < c <= r*g
                    assert e['updated_critic'] and bool(e['updated_actor']) == (c%p == 0)
                    assert bool(e['updated_temperature']) == (c%p == 0)
                    assert e['critic_updates'] == c and e['actor_updates'] == e['temperature_updates'] == c//p
                    assert e['replay_size'] == r*n*h
                    count['critic'] = c
                    count['actor'] += int(e['updated_actor']); count['temperature'] += int(e['updated_temperature'])
                    for metric,value in e['metrics'].items():
                        if metric.startswith(('critic_','q_','td_error','inner_outer_replay')):
                            axis,index = 'critic_update',c
                        else:
                            if not e['updated_actor']: continue
                            axis,index = 'actor_update',c//p
                        curves[(axis,index)][metric].add(value); decisions[key][metric].add(value)
                elif phase == 'decision':
                    assert count['critic'] == j*g and count['actor'] == count['temperature'] == j*g//p
                    assert count['collection'] == j and count['decision'] == 0
                    assert e['metrics']['decision/inner_critic_target_updates'] == j*g//t
                    assert e['metrics']['decision/inner_target_updates'] == j*g//t
                    assert e['metrics']['decision/inner_actor_target_updates'] == 0
                    assert e['metrics']['decision/inner_replay_draws'] == j*g*cell['B']
                    for metric,value in e['metrics'].items(): decisions[key][metric].add(value)
                    count['decision'] += 1
    assert set(counts) == {(ep,d) for ep in expected_ids for d in range(expected_steps)}
    for count in counts.values():
        assert dict(count) == dict(initial=1,collection=j,critic=j*g,actor=j*g//p,temperature=j*g//p,decision=1)
    required = {'critic_loss','critic_grad_norm','td_error_abs_mean','q_target_mean','actor_loss',
                'actor_grad_norm','actor_entropy','alpha_used'}
    assert required <= {k for metrics in curves.values() for k in metrics}
    packed = [dict(axis=a,index=i,metrics={k:v.summary() for k,v in metrics.items()})
              for (a,i),metrics in sorted(curves.items())]
    by_decision = defaultdict(lambda: defaultdict(Moments)); per_seed = []
    for (ep,d),metrics in sorted(decisions.items()):
        means = {k:v.total/v.n for k,v in metrics.items()}
        per_seed.append(dict(episode_id=ep,decision=d,metrics=means))
        for k,v in means.items(): by_decision[d][k].add(v)
    return dict(update_curves=packed,per_seed_decisions=per_seed,
        decision_curves=[dict(decision=d,metrics={k:v.summary() for k,v in ms.items()}) for d,ms in sorted(by_decision.items())],
        trace_rows_checked=row_count,metric_catalog=manifest['metric_catalog'])


def timing_summary(manifest, summary, *, worker_seconds):
    run, = manifest['runs']; episodes = run['episodes']; decisions = sum(e['length'] for e in episodes)
    control = [r['metrics']['decision/control_seconds'] for r in summary['per_seed_decisions']]
    return dict(worker_seconds=worker_seconds, initialization_seconds=run.get('initialization_seconds',0),
        warmup_including_compile_seconds=run.get('warmup_including_compile_seconds',0),
        serialization_seconds=run.get('serialization_seconds',0), decisions=decisions,
        control_seconds=sum(e['control_seconds'] for e in episodes),
        probe_seconds=sum(e.get('togo_probe_seconds',0) for e in episodes),
        control_seconds_per_decision=sum(control)/len(control),
        first_decision_control_seconds=control[0],
        subsequent_control_seconds_per_decision=sum(control[1:])/len(control[1:]) if len(control)>1 else None)


def worker(args):
    started = time.perf_counter()
    import torch
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.ambi_seed_shards import seal_episode_bundle
    campaign = read(args.root/'campaign.json'); assert source_commit() == campaign['source_commit']
    assert campaign['checkpoint_step'] == CHECKPOINT_STEP and campaign['checkpoint_sha256'] == CHECKPOINT_SHA
    assert args.index in (campaign['smoke_indices'] if args.smoke else campaign['production_indices'])
    cell = campaign['cells'][args.index]
    assert not cell['reused'] and torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    assert digest(campaign['matrix']) == campaign['matrix_sha256']
    assert digest(cell['checkpoint']) == cell['checkpoint_sha256']
    assert digest(cell['checkpoint']+'.metadata.json') == cell['metadata_sha256']
    verify_reference(cell['prior_reference'],traces=True)
    directory = args.root/'smoke'/cell['name'] if args.smoke else Path(cell['directory'])
    directory.mkdir(parents=True,exist_ok=not args.smoke); bundle = directory/'bundle'
    steps = campaign.get('smoke_steps',SMOKE_STEPS) if args.smoke else 500
    evaluate_matrix(campaign['matrix'],cell['checkpoint'],selectors=[cell['selector']],
        seeds=[101] if args.smoke else SEEDS,controller_seed=55,max_steps=steps,
        device='cuda',bundle_dir=bundle,checkpoint_inventory=campaign['inventory'],
        reference_bundle=None if args.smoke else cell['prior_reference']['bundle'])
    manifest = validate_completed(bundle,cell,campaign,smoke=args.smoke)
    summary = training_summary(bundle,cell,expected_steps=steps)
    timing = timing_summary(manifest,summary,worker_seconds=time.perf_counter()-started)
    timing.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    seal_episode_bundle(bundle)
    receipt = dict(status='complete',cell=cell['name'],selector=cell['selector'],bundle=str(bundle),
        reused=False,smoke=args.smoke,execution='mean',estimator='one_step',alpha_mode='adaptive',
        **{key:cell[key] for key in ('H','J','N','B','G','P','T')},
        checkpoint_step=CHECKPOINT_STEP,checkpoint_sha256=CHECKPOINT_SHA,
        manifest_sha256=digest(bundle/'manifest.json'),
        trace_sha256={name:digest(bundle/name) for name in manifest['runs'][0]['trace_files']},
        trace_rows_checked=summary['trace_rows_checked'],gpu=torch.cuda.get_device_name(0),timing=timing,
        schedule_validation='Every critic/actor/temperature trace step and every decision target total; target ordering covered by scheduler unit tests.')
    write(directory/'validation.json',receipt); write(directory/'worker-completion.json',receipt)
    print('COMPLETE '+cell['name']+' '+json.dumps(timing),flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    prep = sub.add_parser('prepare')
    for name in ('root','inventory','registry'): prep.add_argument('--'+name,type=Path,required=True)
    prep.add_argument('--matrix',type=Path,default=MATRIX); prep.add_argument('--references',type=Path,default=REFERENCES)
    prep.add_argument('--group',default=GROUP); prep.add_argument('--label',default='575k return/return SAC scaling')
    run = sub.add_parser('worker'); run.add_argument('--root',type=Path,required=True)
    run.add_argument('--index',type=int,required=True); run.add_argument('--smoke',action='store_true')
    args = parser.parse_args(); {'prepare':prepare,'worker':worker}[args.command](args)


if __name__ == '__main__': main()
