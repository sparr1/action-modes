"""Paired frozen-checkpoint episodes and fixed-state actor-gradient diagnostics."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from evaluate_ambi_checkpoint import (_make_env, _initialize_frozen_model, _outer_state_digest,
                                      _close_resources, _file_sha256)
from utils.ambi_benchmark import atomic_json, canonical_hash, solver_seed
from utils.checkpoint_context import load_checkpoint_context
from utils.collapse_diagnostics import (FrozenPriorController, PairedInnerController,
                                       gradient_probe, episode)


def resolved(checkpoint, device, *, compiled=False):
    context = load_checkpoint_context(checkpoint)
    config = copy.deepcopy(context.trial_run_params)
    params = config['alg_params']
    if (config['alg'] != 'AMBITDMPC2/AMBITDMPC2' or params.get('aux_return_mode','off') != 'off'
            or params.get('critic_value_mode','single') != 'single'
            or params.get('inner_actor_entropy_mode') != 'squashed'
            or params.get('inner_actor_loss_scale_scope','action') != 'action'
            or params.get('inner_outer_replay_fraction',0) != 0):
        raise ValueError('Campaign requires single-component, squashed-entropy, fresh inner SAC.')
    params.update(wandb=False, wandb_mode='disabled', compile=compiled, compile_strict=compiled,
                  inner_cuda_graph=False)
    config.update(device=device, seed=55)
    environment = dict(id=config['env'], params=copy.deepcopy(context.experiment_params.get('env_params',{})))
    if environment['params'].get('task') != 'humanoid-walk' or params.get('obs','state') != 'state':
        raise ValueError('This protocol requires Humanoid Walk state observations.')
    return dict(algorithm_config=config, environment=environment, selector='collapse_diagnostics'), context


def initialize(checkpoint, device, *, compiled=False):
    config, context = resolved(checkpoint, device, compiled=compiled)
    env = _make_env(config)
    try:
        model, _ = _initialize_frozen_model(config, env, checkpoint, 55, device=device)
    except BaseException:
        env.close()
        raise
    return model, env, config, context


def identity(checkpoint, context):
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=_file_sha256(checkpoint),
        metadata_sha256=_file_sha256(context.source), training_step=context.metadata['checkpoint']['step'],
        evaluator_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        evaluator_dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip()))


def load_bank(path):
    data = json.loads(Path(path).read_text())
    content = {k:v for k,v in data.items() if k != 'id'}
    if data.get('kind') != 'collapse_upright_bank_v1' or data.get('id') != canonical_hash(content):
        raise ValueError('Invalid or changed fixed-state bank.')
    if len(data['roots']) != data['requested_roots']:
        raise ValueError('Incomplete bank.')
    for root in data['roots']:
        if root['head_height'] < 1.2 or root['torso_upright'] < .8:
            raise ValueError('Non-upright bank state.')
        if not np.isfinite(root['observation']).all():
            raise ValueError('Invalid bank observation.')
    return data


def make_bank(args):
    model, env, config, context = initialize(args.checkpoint,args.device)
    try:
        before = _outer_state_digest(model)
        controller = FrozenPriorController(model,'mean')
        candidates, episodes = [], []
        for seed in range(201,241):
            obs,_ = env.reset(seed=seed)
            total = 0.
            for step in range(500):
                physics = env.unwrapped._env.physics
                height, upright = float(physics.head_height()), float(physics.torso_upright())
                if step >= 25 and step % 25 == 0 and height >= 1.2 and upright >= .8:
                    candidates.append(dict(root_id=f'{seed}-{step}',seed=seed,decision=step,
                        observation=np.asarray(obs,dtype=np.float32).tolist(), head_height=height,torso_upright=upright))
                obs,reward,done,truncated,_ = env.step(controller.action(obs,t0=step==0))
                total += float(reward)
                if done or truncated: break
            episodes.append(dict(seed=seed,return_value=total,steps=step+1))
        atomic_json(args.output/'collection.json',dict(eligible_candidates=len(candidates),
            episodes=episodes,source=identity(args.checkpoint,context)))
        if len(candidates)<args.roots:
            raise ValueError(f'Only {len(candidates)} eligible upright states; need {args.roots}.')
        indices = np.linspace(0,len(candidates)-1,args.roots,dtype=int)
        data = dict(kind='collapse_upright_bank_v1',requested_roots=args.roots,
            source=identity(args.checkpoint,context), environment=config['environment'],
            selection='Evenly spaced over all eligible states from fixed prior-mean seeds 201..240, steps 25..475.',
            roots=[candidates[i] for i in indices], eligible_candidates=len(candidates),collection_episodes=episodes)
        if before != _outer_state_digest(model): raise AssertionError('Bank collection mutated outer state.')
        data['id'] = canonical_hash(data)
        atomic_json(args.output/'bank.json',data)
        return dict(complete=True,bank_id=data['id'],roots=len(data['roots']))
    finally:
        _close_resources(model,env)


def gradients(args):
    bank = load_bank(args.bank)
    model,env,config,context = initialize(args.checkpoint,args.device)
    rows,started = [],time.perf_counter()
    try:
        if bank['environment'] != config['environment']: raise ValueError('Bank environment mismatch.')
        before = _outer_state_digest(model)
        engine = model.agent.inner_engine
        for root in bank['roots'][:args.roots]:
            obs = np.asarray(root['observation'],dtype=np.float32)
            for repeat in range(args.repeats):
                seed = solver_seed(1729,'fixed-gradient',root['root_id'],repeat)
                pair_rng = np.random.default_rng(seed)
                pair = pair_rng.choice(model.cfg.num_q,2,replace=False).tolist()
                base = dict(root_id=root['root_id'],repeat=repeat)
                initial = gradient_probe(model,obs,samples=args.samples,seed=seed,pair=pair)
                rows.append(dict(**base,phase='inherited',**initial))
                solve_seed = solver_seed(55,'fixed-solve',root['root_id'],repeat)
                engine.reset_for_evaluation(solve_seed,reuse_action_pool=True)
                model.predict(obs,deterministic=True,episode_start=True,collect_diagnostics=False)
                state = engine.state
                metrics = model.agent.last_inner_metrics
                scale = metrics['inner_actor_loss_scale_final']
                final = gradient_probe(model,obs,policy=state.actor,critic=state.critic,scale=scale,
                    alpha=engine.alpha,samples=args.samples,seed=seed,pair=pair)
                rows.append(dict(**base,phase='adapted',solve_seed=solve_seed,**final))
            atomic_json(args.output/'progress.json',dict(completed_roots=len(rows)//(2*args.repeats),
                requested_roots=min(args.roots,len(bank['roots'])),elapsed_seconds=time.perf_counter()-started))
        after = _outer_state_digest(model)
        if before != after: raise AssertionError('Gradient diagnostics mutated outer state.')
        result = dict(complete=True,kind='collapse_gradients_v1',source=identity(args.checkpoint,context),
            bank_id=bank['id'],rows=rows,seconds=time.perf_counter()-started,
            outer_state_unchanged=True,outer_digest=before,
            protocol=dict(roots=min(args.roots,len(bank['roots'])),repeats=args.repeats,samples=args.samples,
                gradient_sign='ascent',coordinates='actor_parameters',
                scope='inherited and final locally adapted actor/critic on each identical upright root',
                action_sampling='fresh fixed-input marginal noise, paired across phases/checkpoints',
                dropout='off for diagnostic forwards; saved dropout setting retained during inner learning',
                limitation='Fixed-state gradients, not gradients on the original training minibatches; no historical inner state is recovered.'),
            resolved_config=config)
        atomic_json(args.output/'result.json',result)
        return dict(complete=True,rows=len(rows),seconds=result['seconds'])
    finally:
        _close_resources(model,env)


def episodes(args):
    compiled = args.device.startswith('cuda') and not args.eager
    model,env,config,context = initialize(args.checkpoint,args.device,compiled=compiled)
    started,rows = time.perf_counter(),[]
    try:
        before = _outer_state_digest(model)
        arms = args.arms or ['prior_mean','prior_sample','inner_mean','inner_sample']
        if args.arms is None and model.cfg.actor_distribution == 'gsde': arms.insert(2,'prior_iid')
        inner = None
        for arm in arms:
            mode = arm.split('_',1)[1]
            if arm.startswith('prior_'):
                controller = FrozenPriorController(model,mode)
            else:
                if not args.device.startswith('cuda'): raise ValueError('Full inner episodes require CUDA.')
                if inner is None:
                    obs,_ = env.reset(seed=args.seeds[0])
                    inner = PairedInnerController(model,obs,capture=not args.eager)
                controller = inner
            for seed in args.seeds:
                noise_seed = solver_seed(55,'episode',seed)
                if arm.startswith('inner_'): controller.reset(noise_seed,mode)
                else: controller.reset(noise_seed)
                row = episode(env,controller,env_seed=seed,noise_seed=noise_seed,max_steps=args.steps)
                rows.append(dict(arm=arm,**row))
                atomic_json(args.output/'progress.json',dict(completed_episodes=len(rows),
                    total_episodes=len(arms)*len(args.seeds),rows=rows,elapsed_seconds=time.perf_counter()-started))
        if before != _outer_state_digest(model): raise AssertionError('Episode evaluation mutated outer state.')
        result = dict(complete=True,kind='collapse_episodes_v1',source=identity(args.checkpoint,context),
            rows=rows,outer_state_unchanged=True,outer_digest=before,resolved_config=config,
            seconds=time.perf_counter()-started,graph_setup_seconds=inner.setup_seconds if inner else 0.,
            protocol=dict(seeds=args.seeds,max_steps=args.steps,arms=arms,outer_learning=False,
                inner_learning='saved recipe, fresh actor and critic at each real decision',
                inner_rng='paired private graph stream; mean discards the sampled execution output',
                frozen_gsde='prior_sample holds matrices at saved interval; prior_iid redraws each decision',
                no_drop_label='Checkpoint labels describe training windows, not the outcomes of these evaluations.'))
        atomic_json(args.output/'result.json',result)
        return dict(complete=True,episodes=len(rows),seconds=result['seconds'])
    finally:
        _close_resources(model,env)


def parity(args):
    """GPU graph/eager arithmetic and paired mean/sample policy-state validation."""
    model,env,config,context = initialize(args.checkpoint,args.device,compiled=True)
    try:
        obs,_ = env.reset(seed=101)
        before = _outer_state_digest(model)
        graph = PairedInnerController(model,obs,capture=True)
        native = PairedInnerController(model,obs,capture=False)
        comparisons=[]
        for mode in ['mean','sample']:
            graph.reset(991,mode);native.reset(991,mode)
            for step in range(3):
                a,b=graph.action(obs,t0=step==0),native.action(obs,t0=step==0)
                np.testing.assert_allclose(a,b,rtol=2e-4,atol=3e-5)
                for component in ['actor','critic']:
                    left=getattr(graph.runner.engine.state,component).state_dict()
                    right=getattr(native.runner.engine.state,component).state_dict()
                    for key in left: torch.testing.assert_close(left[key],right[key],rtol=2e-4,atol=3e-5)
                comparisons.append(dict(mode=mode,step=step,max_action_difference=float(np.max(np.abs(a-b)))))
        graph.reset(991,'mean');graph.action(obs,t0=True)
        mean_actor=copy.deepcopy(graph.runner.engine.state.actor.state_dict())
        graph.reset(991,'sample');graph.action(obs,t0=True)
        for key,value in graph.runner.engine.state.actor.state_dict().items():
            torch.testing.assert_close(mean_actor[key],value,rtol=0,atol=0)
        # Use actual full inner solves + environment steps for CPU sizing.
        # Keep torch threads at one: the graph is GPU work, with serial physics.
        available=sorted(os.sched_getaffinity(0))
        profiles=[]
        try:
            for cpus in [1,2,1,2]:
                if len(available)<cpus: continue
                os.sched_setaffinity(0,available[:cpus])
                graph.reset(992,'sample')
                row=episode(env,graph,env_seed=101,noise_seed=992,max_steps=args.profile_steps)
                profiles.append(dict(cpus=cpus,**row))
        finally:
            os.sched_setaffinity(0,available)
        if before!=_outer_state_digest(model):raise AssertionError('Parity mutated outer state.')
        import resource
        result=dict(complete=True,source=identity(args.checkpoint,context),comparisons=comparisons,
            paired_actor_exactly_equal=True,outer_state_unchanged=True,graph_setup_seconds=graph.setup_seconds,
            cpu_profiles=profiles,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(args.output/'result.json',result)
        return result
    finally:
        _close_resources(model,env)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['bank','gradients','episodes','parity'])
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--bank',type=Path)
    parser.add_argument('--roots',type=int,default=32)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--samples',type=int,default=32)
    parser.add_argument('--seeds',type=int,nargs='+',default=[101,102,103,104,105])
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--arms',nargs='+',choices=['prior_mean','prior_sample','prior_iid','inner_mean','inner_sample'])
    parser.add_argument('--torch-threads',type=int,default=1)
    parser.add_argument('--profile-steps',type=int,default=30)
    parser.add_argument('--eager',action='store_true')
    args=parser.parse_args()
    for name in ['roots','repeats','samples','steps','torch_threads','profile_steps']:
        if getattr(args,name)<1:parser.error(name+' must be positive')
    if args.steps>500:parser.error('At most 500 decisions per episode')
    if len(set(args.seeds))!=len(args.seeds):parser.error('Duplicate evaluation seeds')
    if args.output.exists():raise FileExistsError('Choose a fresh output directory.')
    args.output.mkdir(parents=True)
    os.environ['WANDB_MODE']='disabled'
    torch.set_num_threads(args.torch_threads)
    torch.set_float32_matmul_precision('high')
    try:
        result=globals()[{'bank':'make_bank'}.get(args.mode,args.mode)](args)
        print(json.dumps(result,indent=2))
    except BaseException as exc:
        atomic_json(args.output/'failure.json',dict(complete=False,error=str(exc),type=type(exc).__name__))
        raise


if __name__=='__main__': main()
