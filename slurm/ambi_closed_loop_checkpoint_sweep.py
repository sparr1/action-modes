"""Reduced checkpoint curve for H3/J10 closed-loop return-critic refinement."""
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
from slurm.ambi_closed_loop_critics import SOURCE_RUN, source_commit, validate_probe_rows
from slurm.ambi_closed_loop_publish import indexed_episodes
from slurm.ambi_closed_loop_reward_retrace import historical_cell, normalized_config
from slurm.ambi_closed_loop_sampled import verify_receipt

MATRIX = ROOT/'configs/research/ambi_closed_loop_h3_j10_checkpoints.json'
REFERENCES = ROOT/'configs/research/ambi_closed_loop_h3_j10_checkpoint_refs.json'
GROUP = 'closed-loop-h3-j10-checkpoints-20260924'
STEPS = (100000, 200000, 300000, 400000, *range(500000,1000001,25000))
SELECTOR = 'sweep/return_return_alpha_h3_j10_c16'


def requested_params():
    return {**historical_cell(3,10)['requested_alg_params'],
            'inner_eval_execution_action':'mean', 'inner_sac_return_estimator':'one_step',
            'inner_retrace_lambda':1.0, 'inner_retrace_batch_trajectories':None}


def cells(matrix_path=MATRIX):
    from utils.ambi_research import load_preset_matrix
    matrix = load_preset_matrix(matrix_path)
    assert matrix['source_run'] == SOURCE_RUN and matrix['base_alg_config'] == 'checkpoint'
    assert matrix['checkpoint_steps'] == list(STEPS)
    assert matrix['evaluation'] == dict(controller_seed=55,seeds=SEEDS,max_steps=500,
                                       togo_return_rollouts=32,default_presets=[SELECTOR])
    comparison = matrix['comparisons']['sweep']
    assert comparison['reference'] == 'prior'
    assert set(comparison['variants']) == {'prior',SELECTOR.split('/')[1]}
    assert comparison['variants']['prior']['alg_params']['inner_operator'] == 'none'
    params = {**matrix['shared_alg_params'],**comparison['variants'][SELECTOR.split('/')[1]]['alg_params']}
    assert params == requested_params()
    return [dict(name=f'step_{step}',checkpoint_step=step,training_decisions=step,
                 selector=SELECTOR,actual_selector=SELECTOR,params={k:v for k,v in params.items() if v is not None},
                 requested_alg_params=deepcopy(params),H=3,J=10,critic_kind='return_only',
                 estimator='one_step',execution_mode='mean',alpha_mode='adaptive',reused=step==575000)
            for step in STEPS]


def check_episodes(episodes):
    from utils.ambi_benchmark import solver_seed
    indexed_episodes(episodes)
    assert [e['seed'] for e in episodes] == SEEDS
    assert all(e['length'] == 500 and not e['truncated_by_evaluator'] and math.isfinite(e['return'])
               and e['solver_seed'] == solver_seed(55,'episode',e['seed']) for e in episodes)


def verify_reference(reference, *, traces=False):
    bundle = Path(reference['bundle']); manifest = read(bundle/'manifest.json')
    assert digest(bundle/'manifest.json') == reference['manifest_sha256']
    assert manifest['status'] == 'complete' and manifest['code']['dirty'] is False
    assert manifest['code']['commit'] == reference['source_commit']
    assert manifest['checkpoint']['source_run'] == SOURCE_RUN
    assert manifest['checkpoint']['sha256'] == reference['checkpoint_sha256']
    assert manifest['checkpoint']['metadata']['checkpoint']['step'] == reference['checkpoint_step']
    run, = manifest['runs']
    if 'episodes' in reference:
        assert run['episodes'] == reference['episodes']
        assert run['resolved_config'] == reference['resolved_config']
        assert manifest['protocol'] == reference['protocol']
    if traces:
        assert set(reference['trace_sha256']) == set(run['trace_files'])
        assert all(digest(bundle/name) == sha for name,sha in reference['trace_sha256'].items())
    return manifest


def enrich_reference(pin, inventory):
    from utils.eval_series_data import load_records
    manifest = verify_reference(pin); bundle = Path(pin['bundle']); run, = manifest['runs']
    record, = load_records(bundle,inventory_path=inventory)
    assert record['identity']['backbone'] == SOURCE_RUN and record['metrics']['eval/frozen_state_unchanged']
    check_episodes(run['episodes'])
    assert not record['provenance']['missing_artifact_files']
    return {**pin,'episodes':run['episodes'],'identity':record['identity'],'record_id':record['record_id'],
            'resolved_config':run['resolved_config'],'protocol':manifest['protocol'],'runtime':manifest['code']['runtime'],
            'trace_sha256':{name:digest(bundle/name) for name in run['trace_files']}}


def load_prior(pin, inventory):
    reference = enrich_reference(pin,inventory); manifest = verify_reference(reference)
    run, = manifest['runs']; cfg = run['resolved_config']; result = run['result']
    assert reference['identity']['planner'] == {'type':'prior','action_rule':'tanh_mean'}
    assert cfg['inner_operator'] == 'none' and cfg['target_entropy'] == -10.5
    assert cfg['aux_return_mode'] == 'sac' and cfg['aux_return_detach_representation'] is False
    assert cfg['log_std_mapping'] == 'direct_clamp'
    assert cfg['sac_actor_loss_scale_mode'] == cfg['aux_return_sac_actor_loss_scale_mode'] == 'none'
    for key in ('inner_alpha_initial','inner_alpha_final','inner_alpha'):
        assert all(math.isclose(result['model_metrics'][key][stat],pin['initial_alpha'],rel_tol=1e-7)
                   for stat in ('mean','min','max'))
    protocol = reference['protocol']
    assert protocol['action_rule'] == 'tanh_mean' and protocol['controller_seed'] == 55
    assert protocol['max_steps'] == 500 and protocol['seed_scheme'] == 'sha256-v1'
    assert protocol['environment']['id'] == 'DMControl-v0'
    assert protocol['environment']['params']['task'] == 'humanoid-walk' and protocol['observation'] == 'state'
    assert result['outer_state_unchanged'] and result['outer_updates_before'] == result['outer_updates_after']
    return reference


def load_reused(pin, inventory):
    assert pin['checkpoint_step'] == 575000 and pin['performance_run_id'] == '6ed12e4bb02f4895bc6d0835294016e0'
    reference = enrich_reference(pin,inventory); bundle = Path(pin['bundle'])
    receipt = read(bundle.parent/'worker-completion.json')
    assert receipt['manifest_sha256'] == pin['manifest_sha256']
    verify_receipt(bundle,receipt)
    original = historical_cell(3,10); original['actual_selector'] = original['selector']
    validate(bundle,original,checkpoint_sha=pin['checkpoint_sha256'],checkpoint_step=575000)
    publication = read(bundle.parent/'publication-completion.json')
    assert publication['status'] == 'complete' and publication['performance']['published'] == 1
    assert publication['performance']['run_id'] == pin['performance_run_id']
    assert publication['training_run_id'] == pin['training_run_id']
    manifest = verify_reference(reference)
    alpha = manifest['runs'][0]['result']['model_metrics']['inner_alpha_initial']
    assert alpha['min'] == alpha['mean'] == alpha['max'] and alpha['mean'] > 0
    reference.update(initial_alpha=alpha['mean'],reference_manifest_sha256=manifest['reference']['manifest_sha256'])
    return reference


def checkpoint_state_proof(checkpoint, prior):
    """Inspect the trusted, hash-pinned saved temperature; never copy 575k state."""
    import torch
    state = torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert isinstance(state,dict) and 'model' in state and 'log_ent_coef' in state and 'aux_return_state' in state
    saved = state['log_ent_coef']
    assert saved.numel() == 1 and bool(torch.isfinite(saved).all())
    alpha = float(saved.exp().clamp_min(1e-8).item())
    assert math.isclose(alpha,prior['initial_alpha'],rel_tol=1e-6)
    # This backbone has no actor Q normalization; all saved state is still loaded by the evaluator.
    assert prior['resolved_config']['sac_actor_loss_scale_mode'] == 'none'
    assert prior['resolved_config']['aux_return_sac_actor_loss_scale_mode'] == 'none'
    return dict(initial_alpha=alpha,alpha_source='checkpoint.log_ent_coef.exp',
                checkpoint_sha256=prior['checkpoint_sha256'],actor_q_scale_mode='none',
                aux_return_actor_q_scale_mode='none')


def resolve_config(matrix_path, checkpoint, selector=SELECTOR):
    from utils.checkpoint_context import load_checkpoint_context
    from utils.ambi_research import load_preset_matrix, resolve_preset
    from utils.eval_series_data import resolved_checkpoint_config
    context = load_checkpoint_context(checkpoint)
    resolved = resolve_preset(matrix_path,selector,matrix=load_preset_matrix(matrix_path),checkpoint_context=context)
    return resolved_checkpoint_config({'metadata':context.metadata},resolved)


def matching_config(actual, expected):
    """CPU metadata preflight versus CUDA execution, with no scientific differences."""
    a,b = normalized_config(actual),normalized_config(expected)
    a.pop('device',None); b.pop('device',None)
    assert a == b, {k:(b.get(k),a.get(k)) for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}


def prepare(args):
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.eval_series import create_run, validate_identity
    from utils.eval_series_data import planner_identity
    panel = cells(args.matrix); commit = source_commit(); pins = read(args.references)
    inventory = read(args.inventory)
    assert inventory['source_run'] == pins['source_run'] == SOURCE_RUN
    assert pins['checkpoint_steps'] == list(STEPS)
    rows = {row['step']:row for row in inventory['checkpoints']}
    assert len(rows) == len(inventory['checkpoints']) and set(STEPS) <= rows.keys()
    assert [p['checkpoint_step'] for p in pins['prior_references']] == list(STEPS)
    priors = {p['checkpoint_step']:load_prior(p,args.inventory) for p in pins['prior_references']}
    reused = load_reused(pins['reused_evaluation'],args.inventory)
    for cell in panel:
        step = cell['checkpoint_step']; row = rows[step]; prior = priors[step]
        assert row['sha256'] == prior['checkpoint_sha256'] and row['metadata_sha256'] == prior['metadata_sha256']
        assert digest(row['path']) == row['sha256'] and digest(row['metadata_path']) == row['metadata_sha256']
        assert Path(row['metadata_path']) == Path(row['path']+'.metadata.json')
        assert read(row['metadata_path']) == read(Path(prior['bundle'])/'manifest.json')['checkpoint']['metadata']
        proof = checkpoint_state_proof(row['path'],prior)
        cell.update(checkpoint=row['path'],checkpoint_sha256=row['sha256'],metadata_sha256=row['metadata_sha256'],
                    prior_reference=prior,initial_alpha=proof['initial_alpha'],checkpoint_state_proof=proof,
                    expected_config=resolve_config(args.matrix,row['path']))
    args.root.mkdir(parents=True,exist_ok=False)
    registry = None
    for cell in panel:
        directory = args.root/cell['name']; directory.mkdir()
        result = evaluate_matrix(args.matrix,cell['checkpoint'],selectors=[SELECTOR],seeds=SEEDS,
            controller_seed=55,max_steps=500,bundle_dir=directory/'unused',checkpoint_inventory=args.inventory,
            reference_bundle=cell['prior_reference']['bundle'],eval_series_spec_dir=directory/'specs')
        assert result['mode'] == 'evaluation_series_specifications' and set(result['specs']) == {SELECTOR}
        spec = read(result['specs'][SELECTOR]); prior = cell['prior_reference']
        assert spec['identity']['backbone'] == prior['identity']['backbone'] == SOURCE_RUN
        assert spec['identity']['protocol'] == prior['identity']['protocol']
        assert spec['identity']['planner'] == planner_identity(cell['expected_config'],{},'AMBITDMPC2/AMBITDMPC2','tanh_mean')
        cell.update(directory=str(directory),identity=spec['identity'])
        if cell['reused']:
            matching_config(reused['resolved_config'],cell['expected_config'])
            assert reused['identity']['planner'] == spec['identity']['planner']
            assert reused['identity']['protocol'] == spec['identity']['protocol']
            assert reused['checkpoint_sha256'] == cell['checkpoint_sha256']
            assert reused['reference_manifest_sha256'] == prior['manifest_sha256']
            assert math.isclose(reused['initial_alpha'],cell['initial_alpha'],rel_tol=1e-6)
            baseline = indexed_episodes(prior['episodes'])
            for episode in reused['episodes']:
                key = (episode['seed'],episode['solver_seed'])
                assert math.isclose(episode['paired_return_delta'],episode['return']-baseline[key]['return'],abs_tol=1e-9)
            cell.update(bundle=reused['bundle'],reuse_reference=reused,
                performance_run_id=reused['performance_run_id'],training_run_id=reused['training_run_id'],
                run_dir=str(args.registry/reused['performance_run_id']))
        else:
            if registry is None:
                registry = create_run(args.registry,spec,args.group,PROJECT,ENTITY,'oscar-rgao48')
            validate_identity(registry,spec['identity'])
            cell.update(bundle=str(directory/'bundle'),run_dir=registry['run_dir'],
                performance_run_id=registry['run_id'],training_run_id=uuid.uuid4().hex)
    campaign = dict(schema_version=1,group=args.group,label=args.label,matrix=str(args.matrix.resolve()),
        inventory=str(args.inventory.resolve()),source_commit=commit,source_dir=str(ROOT),source_run=SOURCE_RUN,
        target_entropy=-10.5,H=3,J=10,cells=panel,checkpoint_steps=list(STEPS),
        production_indices=[i for i,c in enumerate(panel) if not c['reused']],smoke_indices=[0,len(panel)-1],
        publisher_workers=3,overview_run_id=uuid.uuid4().hex,
        performance_run_id=registry['run_id'],performance_run_dir=registry['run_dir'],
        prior_compatibility_note='Prior mean inference and full-episode pairing are unchanged; historical references retain original scientific identities.',
        reused_compatibility_note='Exact H3/J10 planner and full-episode protocol reused at 575k; historical source and run IDs retained.')
    write(args.root/'campaign.json',campaign)
    print(f'Prepared 25 checkpoints, 24 new evaluations; overview {campaign["overview_run_id"]}',flush=True)
    return campaign


def validate_completed(bundle, cell, campaign, *, smoke=False):
    from utils.ambi_benchmark import episode_protocol, solver_seed
    seeds,steps = ([101],3) if smoke else (SEEDS,500)
    assert not cell['reused']
    manifest = validate(bundle,cell,seeds=seeds,steps=steps,paired=not smoke,
                        checkpoint_sha=cell['checkpoint_sha256'],checkpoint_step=cell['checkpoint_step'])
    assert manifest['code']['commit'] == campaign['source_commit'] and manifest['code']['dirty'] is False
    assert manifest['checkpoint']['source_run'] == SOURCE_RUN
    prior = cell['prior_reference']; verify_reference(prior)
    assert manifest['code']['runtime'] == prior['runtime']
    assert episode_protocol(manifest['protocol']) == episode_protocol({**prior['protocol'],'max_steps':steps})
    run, = manifest['runs']; cfg,result = run['resolved_config'],run['result']
    matching_config(cfg,cell['expected_config'])
    assert run['selector'] == result['selector'] == cell['selector']
    assert result['action_rule'] == 'tanh_mean' and result['deterministic_execution']
    assert cfg['compile'] and cfg['compile_strict'] and result['resolved_device'].startswith('cuda')
    for key,value in cell['requested_alg_params'].items():
        actual = run['config']['alg_params']
        assert key not in actual if value is None else actual.get(key) == value, key
    for key,stats in result['model_metrics'].items():
        assert all(math.isfinite(v) for v in stats.values()),key
        if key.endswith('_fallback'):
            assert stats['min'] == stats['mean'] == stats['max'] == 0,key
    for stat in ('mean','min','max'):
        assert math.isclose(result['model_metrics']['inner_alpha_initial'][stat],cell['initial_alpha'],rel_tol=1e-6)
        assert result['model_metrics']['inner_eval_execution_sampled'][stat] == 0
        assert result['model_metrics']['inner_eval_execution_mean_action_l2'][stat] == 0
    baseline = indexed_episodes(prior['episodes'])
    assert not manifest.get('reference') if smoke else manifest['reference']['manifest_sha256'] == prior['manifest_sha256']
    for episode in run['episodes']:
        key = (episode['seed'],episode['solver_seed']); assert key in baseline
        assert episode['solver_seed'] == solver_seed(55,'episode',episode['seed'])
        if smoke:
            assert 'paired_return_delta' not in episode
        else:
            assert not episode['truncated_by_evaluator']
            assert math.isclose(episode['paired_return_delta'],episode['return']-baseline[key]['return'],abs_tol=1e-9)
        for row in episode['togo_round_summaries']:
            assert all(stats['count'] == steps for stats in row['metrics'].values())
    validate_probe_rows(run,cell,seeds=seeds,steps=steps)
    return manifest


def worker(args):
    import torch
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.ambi_seed_shards import seal_episode_bundle
    campaign = read(args.root/'campaign.json'); assert source_commit() == campaign['source_commit']
    assert args.index in (campaign['smoke_indices'] if args.smoke else campaign['production_indices'])
    cell = campaign['cells'][args.index]
    assert cell['checkpoint_step'] == campaign['checkpoint_steps'][args.index]
    assert not cell['reused'] and torch.cuda.is_available()
    assert digest(cell['checkpoint']) == cell['checkpoint_sha256']
    assert digest(cell['checkpoint']+'.metadata.json') == cell['metadata_sha256']
    verify_reference(cell['prior_reference'],traces=True)
    directory = args.root/'smoke'/cell['name'] if args.smoke else Path(cell['directory'])
    directory.mkdir(parents=True,exist_ok=not args.smoke); bundle = directory/'bundle'
    evaluate_matrix(campaign['matrix'],cell['checkpoint'],selectors=[cell['selector']],
        seeds=[101] if args.smoke else SEEDS,controller_seed=55,max_steps=3 if args.smoke else 500,
        device='cuda',bundle_dir=bundle,checkpoint_inventory=campaign['inventory'],
        reference_bundle=None if args.smoke else cell['prior_reference']['bundle'])
    manifest = validate_completed(bundle,cell,campaign,smoke=args.smoke)
    summary = training_summary(bundle,cell,expected_steps=3 if args.smoke else 500)
    seal_episode_bundle(bundle)
    receipt = dict(status='complete',cell=cell['name'],selector=cell['selector'],bundle=str(bundle),
        reused=False,smoke=args.smoke,execution='mean',estimator='one_step',alpha_mode='adaptive',H=cell['H'],J=cell['J'],
        checkpoint_step=cell['checkpoint_step'],checkpoint_sha256=cell['checkpoint_sha256'],
        manifest_sha256=digest(bundle/'manifest.json'),
        trace_sha256={name:digest(bundle/name) for name in manifest['runs'][0]['trace_files']},
        trace_rows_checked=summary['trace_rows_checked'],gpu=torch.cuda.get_device_name(0))
    write(directory/'validation.json',receipt); write(directory/'worker-completion.json',receipt)
    print('COMPLETE '+cell['name'],flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    prep = sub.add_parser('prepare')
    for name in ('root','inventory','registry'):prep.add_argument('--'+name,type=Path,required=True)
    prep.add_argument('--references',type=Path,default=REFERENCES); prep.add_argument('--matrix',type=Path,default=MATRIX)
    prep.add_argument('--group',default=GROUP); prep.add_argument('--label',default='H3/J10 return critics | checkpoint curve')
    run = sub.add_parser('worker'); run.add_argument('--root',type=Path,required=True)
    run.add_argument('--index',type=int,required=True); run.add_argument('--smoke',action='store_true')
    args = parser.parse_args(); {'prepare':prepare,'worker':worker}[args.command](args)


if __name__ == '__main__':main()
