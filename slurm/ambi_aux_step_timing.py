"""Matched-budget stepwise H3/C16/A4 evaluation with immutable round controls."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import uuid

BASELINE_SOURCES = {32:'1f748ab3201df1b1e5c4d416faafc093949df692',
                    128:'fe87ae07b2a7ed751cd6865a3f60b2eae88e6abb'}


def match_identity(candidate, baseline, n, *, baseline_science=None):
    for key in ('backbone','protocol'):
        assert candidate['identity'][key]==baseline['identity'][key],key
    assert baseline['identity']['science']==(baseline_science or candidate['identity']['science'])
    planner=deepcopy(baseline['identity']['planner']);s=planner['settings']
    assert n in BASELINE_SOURCES and s['inner_rollouts_per_round']==n
    for key,value in dict(inner_rollout_horizon=3,inner_rounds=8,inner_batch_size=256,
                          inner_critic_updates_per_round=16,inner_actor_updates_per_round=4,
                          inner_model_step_budget=24*n,inner_replay_capacity=3072,
                          inner_critic_target_tau=.01,inner_actor_lr=3e-4,inner_critic_lr=3e-4,
                          inner_sac_critic_target='entropy_augmented',inner_terminal_entropy='outer',
                          inner_entropy_enabled=True,inner_temperature_mode='auto').items():
        assert s[key]==value,key
    assert s.get('inner_update_timing','round')=='round'
    assert s.get('inner_component_update_order','critic_first')=='critic_first'
    assert s.get('inner_horizon_conditioning','none')=='none'
    assert s.get('inner_steps_per_update') is None
    assert not s.get('inner_replay_reset_each_round',False)
    assert s.get('inner_critic_source','sac')==s.get('inner_horizon_critic_source','sac')=='sac'
    s['inner_update_timing']='step'
    assert candidate['identity']['planner']==planner


def baseline_result(candidate, baseline, cell):
    from slurm.ambi_aux_hj_sweep import CHECKPOINT_SHA, SEEDS, rollouts
    from utils.eval_series_data import scientific_identity
    n=rollouts(cell);source=BASELINE_SOURCES[n]
    match_identity(candidate,baseline,n,baseline_science=scientific_identity('AMBITDMPC2/AMBITDMPC2',None,source))
    assert baseline['checkpoint']['sha256']==CHECKPOINT_SHA and baseline['checkpoint']['step']==625000
    assert baseline['metrics']['eval/frozen_state_unchanged']
    assert sorted(e['seed'] for e in baseline['episodes'])==SEEDS
    assert all(e['length']==500 and not e['truncated_by_evaluator'] for e in baseline['episodes'])
    return dict(kind='step_timing',record_id=baseline['record_id'],source_commit=source,
                comparison_note='Only collection/update timing changes. H3 step doses C5/A1,C5/A1,C6/A2; '
                                'same per-round C16/A4, checkpoint, seeds, H/J/N/B, full replay, objectives, alpha and tau. '
                                'Historical mixed-GPU references; exploratory five-seed comparison.',
                source_comparison=dict(baseline=baseline['identity']['science'],candidate=candidate['identity']['science']),
                episodes=[{k:e[k] for k in ('seed','solver_seed','return')} for e in baseline['episodes']])


def prepare_campaign(args):
    from evaluate_ambi_checkpoint import evaluate_matrix
    from utils.eval_series import create_run
    from utils.eval_series_data import load_records
    from slurm.ambi_aux_hj_sweep import CHECKPOINT_SHA,SEEDS,PROJECT,ENTITY,cells,digest,read,write,rollouts
    from slurm.ambi_aux_horizon_campaign import checked_control
    from slurm.ambi_aux_rollout_batch import reference_cell
    root=args.root;root.mkdir(parents=True,exist_ok=False)
    assert digest(args.checkpoint)==CHECKPOINT_SHA
    previous=read(args.baseline_campaign/'campaign.json')
    assert previous['source_commit']==BASELINE_SOURCES[32]
    assert read(args.baseline_campaign/'campaign-completion.json')['status']=='complete'
    controls={n:next(c for c in previous['cells'] if c['name']==name) for n,name in
              [(32,'soft_soft_h3_j8_c16_n32_b256'),(128,'soft_soft_h3_j8_c16_n128_b256_reused')]}
    references={n:checked_control(c,args.inventory,traces=True) for n,c in controls.items()}
    prior,=load_records(args.reference,inventory_path=args.inventory)
    assert prior['checkpoint']['sha256']==CHECKPOINT_SHA and prior['checkpoint']['step']==625000
    assert prior['identity']['planner']=={'type':'prior','action_rule':'tanh_mean'}
    assert prior['metrics']['eval/frozen_state_unchanged'] and sorted(e['seed'] for e in prior['episodes'])==SEEDS
    assert all(e['length']==500 and not e['truncated_by_evaluator'] for e in prior['episodes'])
    evaluate_matrix(args.matrix,args.checkpoint,seeds=SEEDS,controller_seed=55,max_steps=500,
                    bundle_dir=root/'unused',checkpoint_inventory=args.inventory,reference_bundle=args.reference,
                    eval_series_spec_dir=root/'specs')
    panel=[]
    for cell in cells(args.matrix):
        directory=root/cell['name'];directory.mkdir()
        spec=read(root/'specs'/f"{cell['selector'].replace('/','__')}.json")
        assert spec['identity']['backbone']==prior['identity']['backbone']
        assert spec['identity']['protocol']==prior['identity']['protocol']
        n=rollouts(cell);record,receipt=references[n];control=controls[n]
        baseline=baseline_result(spec,record,cell)
        baseline.update(bundle=control['bundle'],manifest_sha256=receipt['manifest_sha256'],performance_run_id=control['performance_run_id'])
        registry=create_run(args.registry,spec,args.group+'-'+cell['name'],PROJECT,ENTITY,'oscar-rgao48')
        cell.update(directory=str(directory),bundle=str(directory/'bundle'),actual_selector=cell['selector'],
                    reused=False,run_dir=registry['run_dir'],performance_run_id=registry['run_id'],training_run_id=uuid.uuid4().hex,baseline=baseline)
        panel.append(cell)
    assert len(panel)==2
    for n in (32,128):
        directory=root/f'reused_round_n{n}';directory.mkdir();record,receipt=references[n]
        panel.append(reference_cell(controls[n],directory,None,record,receipt,kind='step_timing'))
    campaign=dict(schema_version=1,group=args.group,label=args.label,matrix=str(args.matrix.resolve()),
                  checkpoint=str(args.checkpoint),checkpoint_sha256=CHECKPOINT_SHA,inventory=str(args.inventory),
                  reference=str(args.reference),prior_manifest_sha256=digest(args.reference/'manifest.json'),
                  prior_source_science=prior['identity']['science'],
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  overview_run_id=uuid.uuid4().hex,cells=panel,step_timing_sweep=True,publisher_workers=2,
                  baseline_campaign=str(args.baseline_campaign),step_timing_controls=controls,
                  production_indices=[0,1],new_episodes=10,reused_episodes=10)
    write(root/'campaign.json',campaign)
    print(json.dumps(dict(root=str(root),conditions=4,new_settings=2,reused=2,new_episodes=10,
                          overview_run_id=campaign['overview_run_id'])),flush=True)


def publication_baseline(campaign, cell, record):
    from slurm.ambi_aux_hj_sweep import rollouts
    from slurm.ambi_aux_horizon_campaign import checked_control
    control=campaign['step_timing_controls'][str(rollouts(cell))]
    baseline,receipt=checked_control(control,campaign['inventory'],traces=True)
    result=baseline_result(record,baseline,cell)
    assert result['record_id']==cell['baseline']['record_id']
    assert receipt['manifest_sha256']==cell['baseline']['manifest_sha256']
    result.update(bundle=control['bundle'],manifest_sha256=receipt['manifest_sha256'],performance_run_id=control['performance_run_id'])
    return result
