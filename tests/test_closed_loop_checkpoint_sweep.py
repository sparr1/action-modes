"""Checkpoint selection, metadata identity and per-checkpoint saved-state guards."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from slurm import ambi_closed_loop_checkpoint_sweep as campaign
from utils.ambi_benchmark import solver_seed


def episodes():
    return [dict(seed=s,solver_seed=solver_seed(55,'episode',s),length=500,
                 truncated_by_evaluator=False,**{'return':float(s)}) for s in campaign.SEEDS]


def test_exact_selection_and_requested_recipe():
    panel=campaign.cells()
    assert len(panel)==25 and [c['checkpoint_step'] for c in panel]==list(campaign.STEPS)
    assert [i for i,c in enumerate(panel) if c['reused']]==[7]
    assert all(c['H']==3 and c['J']==10 and c['execution_mode']=='mean' for c in panel)
    assert all(c['params']['inner_critic_source']==c['params']['inner_horizon_critic_source']=='aux_return' for c in panel)
    assert all(c['params']['inner_entropy_enabled'] and c['params']['inner_temperature_mode']=='auto' for c in panel)
    assert all(c['params']['inner_replay_capacity']==3840 for c in panel)
    assert 'inner_temperature' not in panel[0]['params']
    pins=campaign.read(campaign.REFERENCES)
    assert [p['checkpoint_step'] for p in pins['prior_references']]==list(campaign.STEPS)
    assert len({p['initial_alpha'] for p in pins['prior_references']})>20
    assert pins['reused_evaluation']['performance_run_id']=='6ed12e4bb02f4895bc6d0835294016e0'


@pytest.mark.parametrize('key,value',[('inner_rollout_horizon',4),('inner_temperature',.0046),
    ('inner_sac_return_estimator','retrace'),('inner_eval_execution_action','policy_sample'),
    ('inner_replay_capacity',3072)])
def test_recipe_mutations_are_rejected(tmp_path,key,value):
    matrix=campaign.read(campaign.MATRIX); matrix['shared_alg_params'][key]=value
    matrix['comparisons']['sweep']['variants'][campaign.SELECTOR.split('/')[1]]['alg_params'][key]=value
    path=tmp_path/'matrix.json';path.write_text(json.dumps(matrix))
    with pytest.raises(AssertionError):campaign.cells(path)


def test_full_prepare_uses_real_metadata_evaluator_specs_and_one_shared_registry(tmp_path,monkeypatch):
    """Exercise complete prepare; no learner or environment may be constructed."""
    import torch
    import evaluate_ambi_checkpoint as evaluator
    from utils import ambi_benchmark
    from utils.eval_series import load_run
    from utils.eval_series_data import identity_for_ambi_checkpoint,resolved_checkpoint_config
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=campaign.ROOT,text=True).strip()
    monkeypatch.setattr(ambi_benchmark,'code_identity',lambda:dict(commit=commit,dirty=False))
    monkeypatch.setattr(campaign,'source_commit',lambda:commit)
    monkeypatch.setattr(evaluator,'_make_env',lambda *a,**k:pytest.fail('preparation constructed environment'))
    monkeypatch.setattr(evaluator,'evaluate_preset',lambda *a,**k:pytest.fail('preparation evaluated episode'))
    inventory=tmp_path/'inventory.json';rows=[];priors={};metadata_by_step={}
    params=dict(aux_return_mode='sac',aux_return_detach_representation=False,target_entropy=-10.5,
        log_std_mapping='direct_clamp',sac_actor_loss_scale_mode='none',train_unroll_horizon=3)
    trial=dict(alg='AMBITDMPC2/AMBITDMPC2',env='DMControl-v0',seed=55,total_steps=1000000,
        alg_params=params,resolved_runtime={'observation':dict(mode='state',shape=[67],action_dim=21,episode_length=500)})
    for index,step in enumerate(campaign.STEPS):
        checkpoint=tmp_path/f'checkpoint-{step}.pt'
        alpha=torch.tensor(-7.+index/30.)
        torch.save(dict(model={},aux_return_state={},log_ent_coef=alpha),checkpoint)
        metadata=dict(schema_version=1,trial_run_params=trial,
            experiment_params={'env_params':{'task':'humanoid-walk','obs':'state'}},
            checkpoint=dict(kind='periodic',step=step,episode=step//500,best_score=None,best_window=1))
        sidecar=Path(str(checkpoint)+'.metadata.json');sidecar.write_text(json.dumps(metadata))
        row=dict(step=step,path=str(checkpoint),metadata_path=str(sidecar),sha256=campaign.digest(checkpoint),
                 metadata_sha256=campaign.digest(sidecar));rows.append(row);metadata_by_step[step]=metadata
    inventory.write_text(json.dumps(dict(source_run=campaign.SOURCE_RUN,checkpoints=rows)))
    for row in rows:
        step=row['step'];metadata=metadata_by_step[step]
        cp=dict(path=row['path'],metadata=metadata,sha256=row['sha256'],source_run=campaign.SOURCE_RUN)
        prior_resolved=dict(algorithm_config={**trial,'alg_params':{**params,'inner_operator':'none'}},
                            environment=dict(id='DMControl-v0',params=metadata['experiment_params']['env_params']))
        protocol=ambi_benchmark.protocol_for(prior_resolved,55,500)
        identity=identity_for_ambi_checkpoint(cp,prior_resolved,protocol,campaign.SEEDS,
            dict(commit=commit,dirty=False),path=row['path'],inventory_path=inventory)
        assert identity['planner']=={'type':'prior','action_rule':'tanh_mean'}
        assert 'environment_seeds' not in protocol and identity['protocol']['environment_seeds']==campaign.SEEDS
        directory=tmp_path/f'prior-{step}';directory.mkdir()
        manifest=dict(schema_version=1,status='complete',checkpoint=cp,protocol=protocol,
            runs=[dict(status='complete',config={'alg_params':{'inner_operator':'none'}},episodes=episodes())])
        campaign.write(directory/'manifest.json',manifest)
        prior=dict(checkpoint_step=step,checkpoint_sha256=row['sha256'],metadata_sha256=row['metadata_sha256'],
            bundle=str(directory),manifest_sha256=campaign.digest(directory/'manifest.json'),source_commit=commit,
            initial_alpha=float(torch.load(row['path'],weights_only=False)['log_ent_coef'].exp().item()),
            episodes=episodes(),identity=identity,protocol=protocol,
            resolved_config=resolved_checkpoint_config(cp,prior_resolved))
        priors[step]=prior
    reuse=deepcopy(priors[575000]);cfg=campaign.resolve_config(campaign.MATRIX,rows[7]['path'])
    from utils.eval_series_data import planner_identity
    reuse['reference_manifest_sha256']=priors[575000]['manifest_sha256']
    for episode in reuse['episodes']:episode['paired_return_delta']=0.
    reuse.update(performance_run_id='6ed12e4bb02f4895bc6d0835294016e0',training_run_id='original-training',
        resolved_config=cfg,identity={**reuse['identity'],'planner':planner_identity(cfg,{},'AMBITDMPC2/AMBITDMPC2','tanh_mean')})
    references=tmp_path/'references.json';campaign.write(references,dict(source_run=campaign.SOURCE_RUN,
        checkpoint_steps=list(campaign.STEPS),prior_references=list(priors.values()),reused_evaluation=reuse))
    monkeypatch.setattr(campaign,'load_prior',lambda pin,inv:deepcopy(priors[pin['checkpoint_step']]))
    monkeypatch.setattr(campaign,'load_reused',lambda pin,inv:deepcopy(reuse))
    args=SimpleNamespace(root=tmp_path/'campaign',matrix=campaign.MATRIX,inventory=inventory,
        references=references,registry=tmp_path/'registry',group='regression',label='Regression')
    result=campaign.prepare(args)
    assert len(result['cells'])==25 and len(result['production_indices'])==24
    assert result['smoke_indices']==[0,24] and 7 not in result['production_indices']
    assert len(list(args.registry.iterdir()))==1
    new=[c for c in result['cells'] if not c['reused']]
    assert len({c['performance_run_id'] for c in new})==1
    assert len({c['training_run_id'] for c in new})==24
    assert len({c['initial_alpha'] for c in result['cells']})==25
    assert result['cells'][7]['performance_run_id']==reuse['performance_run_id']
    for cell in result['cells']:
        spec=campaign.read(Path(cell['directory'])/'specs'/(campaign.SELECTOR.replace('/','__')+'.json'))
        assert spec['selector']==campaign.SELECTOR
        assert spec['identity']['planner']['type']=='sac'
        assert spec['identity']['planner']==cell['identity']['planner']
        assert cell['checkpoint_state_proof']['checkpoint_sha256']==cell['checkpoint_sha256']
        if not cell['reused']:assert load_run(cell['run_dir'])['identity']==spec['identity']
        assert not (Path(cell['directory'])/'unused').exists()


def test_saved_alpha_is_checkpoint_specific_and_cannot_use_575k_value(tmp_path):
    import torch
    checkpoint=tmp_path/'checkpoint.pt'
    torch.save(dict(model={},aux_return_state={},log_ent_coef=torch.tensor(-7.)),checkpoint)
    prior=dict(initial_alpha=float(torch.tensor(-7.).exp()),checkpoint_sha256='saved',
               resolved_config=dict(sac_actor_loss_scale_mode='none',aux_return_sac_actor_loss_scale_mode='none'))
    assert campaign.checkpoint_state_proof(checkpoint,prior)['initial_alpha']==prior['initial_alpha']
    prior['initial_alpha']=.004603903274983168
    with pytest.raises(AssertionError):campaign.checkpoint_state_proof(checkpoint,prior)


def test_resolved_matching_allows_only_device_and_inactive_legacy_defaults():
    expected=dict(device='cpu',inner_sac_return_estimator='one_step',inner_retrace_lambda=1.,
                  inner_retrace_batch_trajectories=None,inner_eval_execution_action='mean',discount=.99)
    campaign.matching_config({'device':'cuda','discount':.99},expected)
    with pytest.raises(AssertionError):campaign.matching_config({'device':'cuda','discount':.98},expected)


def test_worker_rejects_reused_checkpoint_before_evaluation(tmp_path,monkeypatch):
    campaign.write(tmp_path/'campaign.json',dict(source_commit='source',smoke_indices=[0,24],
        production_indices=[i for i in range(25) if i!=7],cells=campaign.cells()))
    monkeypatch.setattr(campaign,'source_commit',lambda:'source')
    with pytest.raises(AssertionError):campaign.worker(SimpleNamespace(root=tmp_path,index=7,smoke=False))


def test_launcher_syntax_and_read_only_gpu_publication_boundary():
    path=campaign.ROOT/'slurm/run_ambi_closed_loop_checkpoint_sweep_oscar.sbatch'
    subprocess.run(['bash','-n',str(path)],check=True)
    text=path.read_text()
    assert 'CHECKPOINT_PATH' not in text
    assert 'CHECKPOINT_INVENTORY' in text and 'EXPECTED_ACTION_MODES_SHA' in text
    assert 'ambi_closed_loop_checkpoint_sweep_publish.py' in text


def test_completion_validates_checkpoint_specific_alpha_work_and_pairing(tmp_path,monkeypatch):
    from tests.test_ambi_root_local_sac import _build_cfg
    from utils.ambi_benchmark import episode_protocol
    cell=campaign.cells()[0]
    cfg=vars(_build_cfg(**cell['params'],aux_return_mode='sac',log_std_mapping='direct_clamp',
        target_entropy=-10.5,sac_actor_loss_scale_mode='none'))
    cfg=json.loads(json.dumps(cfg,default=str))
    prior=dict(bundle='historical-prior',episodes=episodes(),runtime={'python':'locked'},
        manifest_sha256='prior-hash',protocol=dict(action_rule='tanh_mean',controller_seed=55,max_steps=500,
            seed_scheme='sha256-v1',environment={'id':'DMControl-v0','params':{'task':'humanoid-walk'}}))
    cell.update(checkpoint_sha256='selected-checkpoint',initial_alpha=.0017,expected_config=cfg,prior_reference=prior)
    metrics={key:dict(mean=value,min=value,max=value) for key,value in dict(
        inner_model_steps=3840,inner_buffer_size=3840,inner_critic_optimizer_steps=160,
        inner_actor_optimizer_steps=40,inner_temperature_optimizer_steps=40,inner_compile_fallback=0,
        inner_alpha_initial=.0017,inner_eval_execution_sampled=0,inner_eval_execution_mean_action_l2=0).items()}
    completed=deepcopy(episodes())
    for episode in completed:
        episode['return']+=10;episode['paired_return_delta']=10
        episode['togo_round_summaries']=[dict(round_index=r,critic_updates=r*16,actor_updates=r*4,
            metrics={'score':dict(count=500,mean=1.)}) for r in range(11)]
    result=dict(selector=campaign.SELECTOR,action_rule='tanh_mean',deterministic_execution=True,
        resolved_device='cuda:0',outer_state_unchanged=True,outer_updates_before=10,outer_updates_after=10,
        nonfinite_model_metrics=[],nonfinite_trace_metrics=[],environment_seeds=campaign.SEEDS,
        controller_seed=55,model_metrics=metrics)
    run=dict(selector=campaign.SELECTOR,status='complete',config={'alg_params':cell['params']},
        resolved_config=cfg,result=result,episodes=completed,trace_files=[str(s) for s in campaign.SEEDS],
        togo_return_probe=dict(rollouts=32,horizon=3,entropy_bonus=False,cadence='initial_and_after_each_round'))
    manifest=dict(status='complete',checkpoint=dict(sha256='selected-checkpoint',source_run=campaign.SOURCE_RUN,
        metadata={'checkpoint':{'step':100000}}),code=dict(commit='current',dirty=False,runtime=prior['runtime']),
        protocol=prior['protocol'],reference={'manifest_sha256':'prior-hash'},runs=[run])
    bundle=tmp_path/'bundle';bundle.mkdir()
    monkeypatch.setattr(campaign,'verify_reference',lambda *a,**k:None)
    monkeypatch.setattr(campaign,'validate_probe_rows',lambda *a,**k:None)
    campaign.write(bundle/'manifest.json',manifest)
    campaign.validate_completed(bundle,cell,{'source_commit':'current'})
    for mutate in (
        lambda m:m['runs'][0]['result']['model_metrics']['inner_alpha_initial'].update(mean=.004603903274983168),
        lambda m:m['checkpoint'].update(sha256='wrong-checkpoint'),
        lambda m:m['runs'][0]['episodes'][0].update(paired_return_delta=9.),
        lambda m:m['runs'][0]['result']['model_metrics']['inner_critic_optimizer_steps'].update(min=159),
        lambda m:m['runs'][0]['resolved_config'].update(inner_temperature_mode='inherit_outer'),
        lambda m:m['code'].update(commit='other-source'),
    ):
        changed=deepcopy(manifest);mutate(changed);campaign.write(bundle/'manifest.json',changed)
        with pytest.raises(AssertionError):campaign.validate_completed(bundle,cell,{'source_commit':'current'})
