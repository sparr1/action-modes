"""Scientific grid, metadata preparation, frequency traces and timing receipts."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from slurm import ambi_closed_loop_sac_scale as campaign
from utils.ambi_benchmark import solver_seed


def test_exact_article_grid_and_replay_retention():
    panel = campaign.cells()
    assert len(panel) == 36
    assert {(c['H'],c['J'],c['P']) for c in panel} == {
        (h,j,p) for h in (1,2,3) for j in (1,2,4,6,8,10) for p in (1,5)}
    assert all(c['checkpoint_step']==575000 and c['N']==1024 and c['B']==4096
               and c['G']==20 and c['T']==2 and not c['reused'] for c in panel)
    for cell in panel:
        p = cell['params']
        assert p['inner_critic_source'] == p['inner_horizon_critic_source'] == 'aux_return'
        assert p['inner_sac_critic_target']=='reward_only' and p['inner_sac_return_estimator']=='one_step'
        assert p['inner_replay_capacity'] == max(3072,cell['N']*cell['H']*cell['J'])
        assert p['inner_entropy_enabled'] and p['inner_temperature_mode']=='auto'
        assert 'inner_critic_updates_per_round' not in p and 'inner_actor_updates_per_round' not in p
        assert p['inner_temperature_initialization']=='inherit_outer' and 'inner_temperature' not in p


@pytest.mark.parametrize('key,value', [
    ('inner_critic_updates_per_round',20), ('inner_actor_updates_per_round',4),
    ('inner_replay_capacity',3072), ('inner_temperature',0.0046),
    ('inner_sac_return_estimator','retrace'), ('inner_eval_execution_action','policy_sample'),
    ('inner_critic_target_tau',0.1), ('inner_critic_target_update_interval',1),
])
def test_undeclared_recipe_mutations_rejected(tmp_path,key,value):
    matrix = campaign.read(campaign.MATRIX)
    selector = matrix['evaluation']['default_presets'][-1].split('/')[1]
    matrix['comparisons']['sweep']['variants'][selector]['alg_params'][key] = value
    path=tmp_path/'matrix.json'; campaign.write(path,matrix)
    with pytest.raises(AssertionError): campaign.cells(path)


def test_explicit_grid_extension_needs_no_worker_change(tmp_path):
    matrix = campaign.make_matrix(horizons=(1,3),rounds=tuple(range(1,11)),
        rollouts=(128,1024),batches=(2048,4096),actor_intervals=(5,1))
    path=tmp_path/'matrix.json'; campaign.write(path,matrix)
    assert len(campaign.cells(path)) == 160


def episodes():
    return [dict(seed=s,solver_seed=solver_seed(55,'episode',s),length=500,
        truncated_by_evaluator=False,**{'return':float(s)}) for s in campaign.SEEDS]


def test_prepare_real_metadata_and_independent_registries(tmp_path,monkeypatch):
    """Metadata preflight resolves every production cell without constructing an env."""
    import torch
    import evaluate_ambi_checkpoint as evaluator
    from utils import ambi_benchmark
    from utils.eval_series import load_run
    from utils.eval_series_data import identity_for_ambi_checkpoint,resolved_checkpoint_config
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=campaign.ROOT,text=True).strip()
    monkeypatch.setattr(ambi_benchmark,'code_identity',lambda:dict(commit=commit,dirty=False))
    monkeypatch.setattr(campaign,'source_commit',lambda:commit)
    monkeypatch.setattr(evaluator,'_make_env',lambda *a,**k:pytest.fail('prepare constructed environment'))
    monkeypatch.setattr(evaluator,'evaluate_preset',lambda *a,**k:pytest.fail('prepare ran episode'))
    checkpoint=tmp_path/'checkpoint.pt'
    torch.save(dict(model={},aux_return_state={},log_ent_coef=torch.tensor(campaign.INITIAL_ALPHA).log()),checkpoint)
    params=dict(aux_return_mode='sac',aux_return_detach_representation=False,target_entropy=-10.5,
        log_std_mapping='direct_clamp',sac_actor_loss_scale_mode='none',train_unroll_horizon=3)
    trial=dict(alg='AMBITDMPC2/AMBITDMPC2',env='DMControl-v0',seed=55,total_steps=1000000,
        alg_params=params,resolved_runtime={'observation':dict(mode='state',shape=[67],action_dim=21,episode_length=500)})
    metadata=dict(schema_version=1,trial_run_params=trial,
        experiment_params={'env_params':{'task':'humanoid-walk','obs':'state'}},
        checkpoint=dict(kind='periodic',step=575000,episode=1150,best_score=None,best_window=1))
    sidecar=Path(str(checkpoint)+'.metadata.json'); campaign.write(sidecar,metadata)
    row=dict(step=575000,path=str(checkpoint),metadata_path=str(sidecar),sha256=campaign.digest(checkpoint),
             metadata_sha256=campaign.digest(sidecar))
    monkeypatch.setattr(campaign,'CHECKPOINT_SHA',row['sha256'])
    inventory=tmp_path/'inventory.json'; campaign.write(inventory,dict(source_run=campaign.SOURCE_RUN,checkpoints=[row]))
    cp=dict(path=row['path'],metadata=metadata,sha256=row['sha256'],source_run=campaign.SOURCE_RUN)
    prior_resolved=dict(algorithm_config={**trial,'alg_params':{**params,'inner_operator':'none'}},
                        environment=dict(id='DMControl-v0',params=metadata['experiment_params']['env_params']))
    protocol=ambi_benchmark.protocol_for(prior_resolved,55,500)
    identity=identity_for_ambi_checkpoint(cp,prior_resolved,protocol,campaign.SEEDS,
        dict(commit=commit,dirty=False),path=row['path'],inventory_path=inventory)
    directory=tmp_path/'prior'; directory.mkdir()
    manifest=dict(schema_version=1,status='complete',checkpoint=cp,protocol=protocol,
        runs=[dict(status='complete',config={'alg_params':{'inner_operator':'none'}},episodes=episodes())])
    campaign.write(directory/'manifest.json',manifest)
    prior=dict(checkpoint_step=575000,checkpoint_sha256=row['sha256'],metadata_sha256=row['metadata_sha256'],
        bundle=str(directory),manifest_sha256=campaign.digest(directory/'manifest.json'),source_commit=commit,
        initial_alpha=campaign.INITIAL_ALPHA,episodes=episodes(),identity=identity,protocol=protocol,
        resolved_config=resolved_checkpoint_config(cp,prior_resolved))
    references=tmp_path/'references.json'
    campaign.write(references,dict(source_run=campaign.SOURCE_RUN,prior_references=[prior]))
    monkeypatch.setattr(campaign,'load_prior',lambda pin,inv:deepcopy(prior))
    args=SimpleNamespace(root=tmp_path/'campaign',matrix=campaign.MATRIX,inventory=inventory,
        references=references,registry=tmp_path/'registry',group='scale-test',label='scale')
    result=campaign.prepare(args)
    assert len(result['cells'])==36 and result['production_indices']==list(range(36))
    assert result['smoke_indices']==[10,11,34,35] and result['smoke_steps']==8
    assert result['matrix_sha256']==campaign.digest(campaign.MATRIX)
    assert len({c['performance_run_id'] for c in result['cells']})==36
    assert len({c['training_run_id'] for c in result['cells']})==36
    for cell in result['cells']:
        cfg=cell['expected_config']; total=cell['G']*cell['J']
        assert cfg['inner_actor_updates_per_action']==cfg['inner_temperature_updates_per_action']==total//cell['P']
        assert cfg['inner_critic_updates_per_action']==total
        assert cfg['inner_critic_updates_per_round'] is None and cfg['inner_actor_updates_per_round'] is None
        assert cfg['inner_critic_target_update_interval']==2
        assert load_run(cell['run_dir'])['identity']==cell['identity']
        assert cell['identity']['planner']['settings']['inner_actor_update_interval']==cell['P']
        assert cell['identity']['planner']['settings']['inner_batch_size']==4096
        assert not (Path(cell['directory'])/'unused').exists()


def trace_fixture(tmp_path):
    """Non-divisible rounds exercise a clock that must carry across collection."""
    cell=dict(H=2,J=3,N=1024,B=4096,G=3,P=2,T=2)
    rows=[]
    def event(phase,r,c,a,metrics=None,**extra):
        rows.append(dict(episode_id='seed-101',decision_index=0,phase=phase,round_index=r,
            critic_updates=c,actor_updates=a,temperature_updates=a,replay_size=r*2048,
            metrics=metrics or {},**extra))
    event('initial',0,0,0)
    for r in range(1,4):
        event('collection',r,(r-1)*3,(r-1)*3//2)
        for c in range((r-1)*3+1,r*3+1):
            metrics=dict(critic_loss=1.,critic_grad_norm=1.,td_error_abs_mean=1.,q_target_mean=1.,alpha_used=.1)
            if c%2==0: metrics.update(actor_loss=1.,actor_grad_norm=1.,actor_entropy=1.)
            event('update',r,c,c//2,metrics,updated_critic=True,updated_actor=c%2==0,updated_temperature=c%2==0)
    event('decision',3,9,4,{'decision/inner_critic_target_updates':4.,'decision/inner_target_updates':4.,
        'decision/inner_actor_target_updates':0.,'decision/control_seconds':1.,'decision/inner_replay_draws':9*4096})
    manifest=dict(runs=[dict(episodes=[dict(seed=101,length=1)],trace_files=['trace.jsonl.gz'])],metric_catalog={})
    campaign.write(tmp_path/'manifest.json',manifest)
    return cell,rows


def save_trace(path,rows):
    with gzip.open(path/'trace.jsonl.gz','wt') as stream:
        for row in rows: stream.write(json.dumps(row)+'\n')


def test_trace_clock_crosses_rounds_and_records_target_totals(tmp_path):
    cell,rows=trace_fixture(tmp_path); save_trace(tmp_path,rows)
    summary=campaign.training_summary(tmp_path,cell,expected_steps=1)
    assert summary['trace_rows_checked']==14
    critic=[r for r in summary['update_curves'] if r['axis']=='critic_update']
    actor=[r for r in summary['update_curves'] if r['axis']=='actor_update']
    assert [r['index'] for r in critic]==list(range(1,10))
    assert [r['index'] for r in actor]==[1,2,3,4]


@pytest.mark.parametrize('mutation',[
    lambda rows:rows[2].update(updated_actor=True),
    lambda rows:rows[3].update(updated_temperature=False),
    lambda rows:rows[6].update(actor_updates=1),
    lambda rows:rows[1].update(replay_size=128),
    lambda rows:rows[0].update(critic_updates=9),
    lambda rows:rows[-1]['metrics'].update({'decision/inner_critic_target_updates':9}),
    lambda rows:rows[-1]['metrics'].update({'decision/inner_replay_draws':4096}),
    lambda rows:rows[4]['metrics'].update(critic_loss=float('nan')),
    lambda rows:rows.append(deepcopy(rows[-1])),
])
def test_trace_corruption_rejected(tmp_path,mutation):
    cell,rows=trace_fixture(tmp_path); mutation(rows); save_trace(tmp_path,rows)
    with pytest.raises(AssertionError): campaign.training_summary(tmp_path,cell,expected_steps=1)


def test_probe_work_scales_with_horizon_and_critic_clock():
    cell=dict(H=3,J=3,G=3,P=2)
    rows=[dict(episode_id='seed-101',decision_index=0,round_index=r,critic_updates=3*r,actor_updates=3*r//2,
        metrics=dict(probe_model_steps=96*(2 if r==0 else 1),probe_q_evaluations=32*(2 if r==0 else 1)))
        for r in range(4)]
    run=dict(togo_probe_rows=rows)
    campaign.validate_probe_rows(run,cell,seeds=[101],steps=1)
    rows[-1]['actor_updates']=3
    with pytest.raises(AssertionError): campaign.validate_probe_rows(run,cell,seeds=[101],steps=1)


def test_timing_separates_compile_and_control():
    manifest=dict(runs=[dict(initialization_seconds=2,warmup_including_compile_seconds=10,
        serialization_seconds=.5,episodes=[dict(length=3,control_seconds=6,togo_probe_seconds=.6)])])
    summary=dict(per_seed_decisions=[dict(metrics={'decision/control_seconds':x}) for x in (3,1,2)])
    timing=campaign.timing_summary(manifest,summary,worker_seconds=19.1)
    assert timing['warmup_including_compile_seconds']==10 and timing['control_seconds_per_decision']==2
    assert timing['first_decision_control_seconds']==3 and timing['subsequent_control_seconds_per_decision']==1.5
    assert timing['probe_seconds']==.6
