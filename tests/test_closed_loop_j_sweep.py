"""Round-budget scope, original-result reuse and real metadata/registry preparation."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from slurm import ambi_closed_loop_j_sweep as campaign
from utils.ambi_benchmark import solver_seed


def episodes():
    return [dict(seed=s,solver_seed=solver_seed(55,'episode',s),length=500,
                 truncated_by_evaluator=False,**{'return':float(s)}) for s in campaign.SEEDS]


def test_scope_preserves_historical_575k_recipes_at_new_checkpoint():
    panel = campaign.cells()
    assert [c['J'] for c in panel] == [1,2,4,6,8,10,12,14]
    assert [c['params']['inner_replay_capacity'] for c in panel] == [3072]*5+[3840,4608,5376]
    assert [i for i,c in enumerate(panel) if c['reused']] == [5]
    assert all(c['H']==3 and c['checkpoint_step']==650000 and c['execution_mode']=='mean' for c in panel)
    assert all(c['params']['inner_critic_source']==c['params']['inner_horizon_critic_source']=='aux_return' for c in panel)
    assert all(c['params']['inner_entropy_enabled'] and c['params']['inner_temperature_mode']=='auto' for c in panel)
    assert all('inner_temperature' not in c['params'] for c in panel)
    pins = campaign.read(campaign.REFERENCES)
    assert pins['prior_reference']['initial_alpha'] == campaign.INITIAL_ALPHA
    assert pins['reused_evaluation']['performance_run_id'] == '4aa700149cec4161886b916c4b135c07'
    assert pins['reused_evaluation']['publication_entry']['checkpoint_step'] == 650000


@pytest.mark.parametrize('key,value',[('inner_rollout_horizon',4),('inner_temperature',.0046),
    ('inner_sac_return_estimator','retrace'),('inner_eval_execution_action','policy_sample'),
    ('inner_replay_capacity',3840),('inner_critic_updates_per_round',32)])
def test_recipe_mutations_rejected(tmp_path,key,value):
    matrix = campaign.read(campaign.MATRIX)
    matrix['comparisons']['sweep']['variants']['return_return_alpha_h3_j1_c16']['alg_params'][key] = value
    path = tmp_path/'matrix.json'; path.write_text(json.dumps(matrix))
    with pytest.raises(AssertionError): campaign.cells(path)


def test_full_prepare_real_metadata_and_seven_independent_registries(tmp_path,monkeypatch):
    """Use actual evaluate_matrix specification generation; never build a learner."""
    import torch
    import evaluate_ambi_checkpoint as evaluator
    from utils import ambi_benchmark
    from utils.eval_series import load_run
    from utils.eval_series_data import identity_for_ambi_checkpoint, resolved_checkpoint_config, planner_identity
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=campaign.ROOT,text=True).strip()
    monkeypatch.setattr(ambi_benchmark,'code_identity',lambda:dict(commit=commit,dirty=False))
    monkeypatch.setattr(campaign,'source_commit',lambda:commit)
    monkeypatch.setattr(evaluator,'_make_env',lambda *a,**k:pytest.fail('preparation constructed environment'))
    monkeypatch.setattr(evaluator,'evaluate_preset',lambda *a,**k:pytest.fail('preparation evaluated episode'))
    checkpoint = tmp_path/'checkpoint.pt'
    torch.save(dict(model={},aux_return_state={},log_ent_coef=torch.tensor(campaign.INITIAL_ALPHA).log()),checkpoint)
    params = dict(aux_return_mode='sac',aux_return_detach_representation=False,target_entropy=-10.5,
        log_std_mapping='direct_clamp',sac_actor_loss_scale_mode='none',train_unroll_horizon=3)
    trial = dict(alg='AMBITDMPC2/AMBITDMPC2',env='DMControl-v0',seed=55,total_steps=1000000,
        alg_params=params,resolved_runtime={'observation':dict(mode='state',shape=[67],action_dim=21,episode_length=500)})
    metadata = dict(schema_version=1,trial_run_params=trial,
        experiment_params={'env_params':{'task':'humanoid-walk','obs':'state'}},
        checkpoint=dict(kind='periodic',step=650000,episode=1300,best_score=None,best_window=1))
    sidecar = Path(str(checkpoint)+'.metadata.json'); sidecar.write_text(json.dumps(metadata))
    row = dict(step=650000,path=str(checkpoint),metadata_path=str(sidecar),sha256=campaign.digest(checkpoint),
               metadata_sha256=campaign.digest(sidecar))
    monkeypatch.setattr(campaign,'CHECKPOINT_SHA',row['sha256'])
    inventory = tmp_path/'inventory.json'
    campaign.write(inventory,dict(source_run=campaign.SOURCE_RUN,checkpoints=[row]))
    cp = dict(path=row['path'],metadata=metadata,sha256=row['sha256'],source_run=campaign.SOURCE_RUN)
    prior_resolved = dict(algorithm_config={**trial,'alg_params':{**params,'inner_operator':'none'}},
                         environment=dict(id='DMControl-v0',params=metadata['experiment_params']['env_params']))
    protocol = ambi_benchmark.protocol_for(prior_resolved,55,500)
    identity = identity_for_ambi_checkpoint(cp,prior_resolved,protocol,campaign.SEEDS,
        dict(commit=commit,dirty=False),path=row['path'],inventory_path=inventory)
    assert identity['planner'] == {'type':'prior','action_rule':'tanh_mean'}
    assert identity['protocol']['environment_seeds'] == campaign.SEEDS
    directory = tmp_path/'prior'; directory.mkdir()
    manifest = dict(schema_version=1,status='complete',checkpoint=cp,protocol=protocol,
        runs=[dict(status='complete',config={'alg_params':{'inner_operator':'none'}},episodes=episodes())])
    campaign.write(directory/'manifest.json',manifest)
    prior = dict(checkpoint_step=650000,checkpoint_sha256=row['sha256'],metadata_sha256=row['metadata_sha256'],
        bundle=str(directory),manifest_sha256=campaign.digest(directory/'manifest.json'),source_commit=commit,
        initial_alpha=campaign.INITIAL_ALPHA,episodes=episodes(),identity=identity,protocol=protocol,
        resolved_config=resolved_checkpoint_config(cp,prior_resolved))
    reuse = deepcopy(prior)
    cfg = campaign.resolve_config(campaign.MATRIX,checkpoint,campaign.cells()[5]['selector'])
    reuse['reference_manifest_sha256'] = prior['manifest_sha256']
    for episode in reuse['episodes']: episode['paired_return_delta'] = 0.
    reuse.update(performance_run_id='4aa700149cec4161886b916c4b135c07',training_run_id='b4bfda4e5a2d497bb5fe7e6d74fadafc',
        run_dir='/original/shared/curve',resolved_config=cfg,
        identity={**reuse['identity'],'planner':planner_identity(cfg,{},'AMBITDMPC2/AMBITDMPC2','tanh_mean')})
    references = tmp_path/'references.json'
    campaign.write(references,dict(source_run=campaign.SOURCE_RUN,checkpoint_step=650000,
        prior_reference=prior,reused_evaluation=reuse))
    monkeypatch.setattr(campaign,'load_prior',lambda pin,inv:deepcopy(prior))
    monkeypatch.setattr(campaign,'load_reused',lambda pin,inv:deepcopy(reuse))
    args = SimpleNamespace(root=tmp_path/'campaign',matrix=campaign.MATRIX,inventory=inventory,
        references=references,registry=tmp_path/'registry',group='regression',label='Regression')
    result = campaign.prepare(args)
    assert result['production_indices'] == [0,1,2,3,4,6,7] and result['smoke_indices'] == [7]
    assert result['checkpoint_steps'] == [650000]*8 and len(list(args.registry.iterdir())) == 7
    new = [c for c in result['cells'] if not c['reused']]
    assert len({c['performance_run_id'] for c in new}) == len({c['training_run_id'] for c in new}) == 7
    assert result['cells'][5]['performance_run_id'] == reuse['performance_run_id']
    assert result['cells'][5]['run_dir'] == reuse['run_dir']
    for cell in result['cells']:
        spec = campaign.read(Path(cell['directory'])/'specs'/(cell['selector'].replace('/','__')+'.json'))
        assert spec['identity']['planner']['type'] == 'sac' and spec['identity']['planner'] == cell['identity']['planner']
        assert cell['expected_config']['inner_rounds'] == cell['J']
        assert cell['checkpoint_state_proof']['checkpoint_sha256'] == row['sha256']
        if not cell['reused']: assert load_run(cell['run_dir'])['identity'] == spec['identity']
        assert not (Path(cell['directory'])/'unused').exists()


def test_reused_publication_validates_specific_record_not_cumulative_total(tmp_path,monkeypatch):
    from utils import eval_series
    bundle = tmp_path/'original'/'bundle'; bundle.mkdir(parents=True)
    run_dir = tmp_path/'registry'; (run_dir/'records').mkdir(parents=True)
    pin = dict(bundle=str(bundle),run_dir=str(run_dir),performance_run_id='original-performance',
               training_run_id='original-training',manifest_sha256='manifest',record_id='record')
    reference = dict(identity={'preserved':'identity'},record_id='record',episodes=episodes())
    campaign.write(bundle.parent/'publication-completion.json',dict(status='complete',
        performance=dict(published=7,accepted=7,run_id=pin['performance_run_id']),training_run_id=pin['training_run_id']))
    entry = dict(status='published',checkpoint_step=650000,checkpoint_sha256=campaign.CHECKPOINT_SHA,
                 artifact_sha256={'manifest.json':'manifest'},record_sha256='unchanged')
    campaign.write(run_dir/'publication.json',{'records':{'record':entry,'another':{'status':'published'}}})
    campaign.write(run_dir/'records'/'record.json',dict(reference,
        checkpoint=dict(step=650000,sha256=campaign.CHECKPOINT_SHA)))
    monkeypatch.setattr(eval_series,'load_run',lambda _:dict(run_id=pin['performance_run_id'],identity=reference['identity']))
    assert campaign.verify_publication(pin,reference) == entry
    for field,value in [('checkpoint_step',625000),('checkpoint_sha256','wrong'),('status','queued')]:
        changed = {**entry,field:value}; campaign.write(run_dir/'publication.json',{'records':{'record':changed}})
        with pytest.raises(AssertionError): campaign.verify_publication(pin,reference)


def test_launcher_and_shared_worker_reject_reused_budget(tmp_path,monkeypatch):
    from slurm import ambi_closed_loop_checkpoint_sweep as shared
    path = campaign.ROOT/'slurm/run_ambi_closed_loop_j_sweep_oscar.sbatch'
    subprocess.run(['bash','-n',str(path)],check=True)
    assert 'ambi_closed_loop_j_sweep_publish.py' in path.read_text()
    campaign.write(tmp_path/'campaign.json',dict(source_commit='source',checkpoint_steps=[650000]*8,
        smoke_indices=[7],production_indices=[0,1,2,3,4,6,7],cells=campaign.cells()))
    monkeypatch.setattr(shared,'source_commit',lambda:'source')
    with pytest.raises(AssertionError): campaign.worker(SimpleNamespace(root=tmp_path,index=5,smoke=False))


def test_mppi_two_run_bundle_uses_real_record_normalization(tmp_path):
    """Both historical arms normalize independently without a selector argument."""
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=campaign.ROOT,text=True).strip()
    protocol = dict(action_rule='mppi_proposal_mean',controller_seed=55,max_steps=500,
        seed_scheme='sha256-v1',environment=dict(id='DMControl-v0',params={'task':'humanoid-walk','obs':'state'}),
        observation='state',env_wrapper=None,env_wrappers=[])
    completed = [dict(e,truncated=True,terminated=False) for e in episodes()]
    runs = []
    for selector,source in [('bootstrap/soft_q','sac'),('bootstrap/return_q','aux_return')]:
        runs.append(dict(selector=selector,status='complete',config={'alg':'AMBITDMPC2/AMBITDMPC2'},
            resolved_config=dict(inner_operator='mppi',inner_horizon_critic_source=source),
            episodes=completed,trace_files=[],result=dict(outer_state_unchanged=True,
                outer_updates_before=50,outer_updates_after=50,action_rule='mppi_proposal_mean',
                environment_seeds=campaign.SEEDS)))
    manifest = dict(schema_version=1,status='complete',protocol=protocol,runs=runs,
        checkpoint=dict(sha256=campaign.CHECKPOINT_SHA,source_run=campaign.SOURCE_RUN,
            metadata=dict(checkpoint={'step':650000},trial_run_params={'alg':'AMBITDMPC2/AMBITDMPC2'})),
        code=dict(commit=commit,dirty=False,runtime={'locked':True}))
    bundle = tmp_path/'mppi'; bundle.mkdir(); campaign.write(bundle/'manifest.json',manifest)
    inventory = tmp_path/'inventory.json'
    campaign.write(inventory,dict(source_run=campaign.SOURCE_RUN,
        checkpoints=[dict(step=650000,sha256=campaign.CHECKPOINT_SHA,path='/pinned/checkpoint')]))
    pin = dict(bundle=str(bundle),manifest_sha256=campaign.digest(bundle/'manifest.json'),
        checkpoint_step=650000,checkpoint_sha256=campaign.CHECKPOINT_SHA,source_commit=commit)
    prior = dict(protocol={**protocol,'action_rule':'tanh_mean'},runtime=manifest['code']['runtime'])
    result = campaign.load_mppi_references(pin,inventory,prior)
    assert set(result) == {'soft','return_only'}
    assert result['soft']['identity']['planner']['type'] == 'mppi'
    assert result['soft']['record_id'] != result['return_only']['record_id']
    assert result['soft']['selector'] == 'bootstrap/soft_q'
    assert result['return_only']['selector'] == 'bootstrap/return_q'
    for mutate in (
        lambda m:m['runs'][1]['resolved_config'].update(inner_horizon_critic_source='sac'),
        lambda m:m['runs'][0]['result'].update(outer_state_unchanged=False),
        lambda m:m['protocol'].update(controller_seed=56),
    ):
        changed = deepcopy(manifest); mutate(changed); campaign.write(bundle/'manifest.json',changed)
        changed_pin = {**pin,'manifest_sha256':campaign.digest(bundle/'manifest.json')}
        with pytest.raises((AssertionError,ValueError)):
            campaign.load_mppi_references(changed_pin,inventory,prior)
