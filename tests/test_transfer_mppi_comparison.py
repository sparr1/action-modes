from copy import deepcopy
import json
import statistics
from types import SimpleNamespace

import pytest

from utils import transfer_mppi_comparison as comparison
from utils.ambi_benchmark import solver_seed
from utils.eval_series import SeriesError, _digest, _record_fingerprint


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def exported(tmp_path):
    campaign = dict(source_run='entity/ambi/backbone', seeds=[101,102,103,104,105],
                    controller_seed=55, max_steps=500,
                    checkpoints=[dict(step=25000, checkpoint_sha256='a'*64)])
    prior_returns = [10,20,30,40,50]
    returns = [11,24,29,42,55]
    sources = [('base-actor','prior-run',None)] + list(comparison.SOURCES.values())
    for directory, run_id, terminal in sources:
        cfg = dict(inner_operator='mppi', inner_rollout_horizon=3, inner_mppi_iterations=8,
                   inner_mppi_num_samples=512, inner_mppi_num_elites=64, inner_mppi_num_pi_trajs=24,
                   inner_mppi_warm_start_scope='episode', inner_mppi_temperature=0.5,
                   inner_mppi_min_std=0.05, inner_mppi_max_std=2.)
        if terminal == 'aux_return':
            cfg['inner_horizon_critic_source'] = terminal
        ident = dict(backbone=campaign['source_run'],
            planner=dict(type='mppi' if terminal else 'prior',settings=cfg),
            protocol=dict(environment_seeds=campaign['seeds'],controller_seed=55,max_steps=500,
                mode='episodes',action_rule='mppi_proposal_mean' if terminal else 'tanh_mean',
                environment=dict(id='DMControl-v0',params=dict(task='humanoid-walk',obs='state'))),
            science={'source_sha256':'b'*64})
        values = returns if terminal else prior_returns
        episodes = [dict(seed=seed,solver_seed=solver_seed(55,'episode',seed),length=500,
                         truncated=True,terminated=False,truncated_by_evaluator=False,
                         **{'return':value}) for seed,value in zip(campaign['seeds'],values)]
        episodes.reverse()  # Pair by seed, not position.
        metrics={'eval/frozen_state_unchanged':True,'eval/episodes':5,
                 'eval/return_mean':statistics.mean(values),'eval/return_sample_std':statistics.stdev(values),
                 **{'work/'+key:0 for key in ('actor_updates','critic_updates','temperature_updates')}}
        record=dict(identity=ident,checkpoint={'step':25000,'sha256':'a'*64},episodes=episodes,
                    metrics=metrics,artifact_files={'results.json':'/original/results.json'},
                    record_id='record',source_result_path='/original/results.json',label=directory,
                    provenance={'resolved_config':{'inner_horizon_critic_source':terminal or 'sac'}})
        hashes={'results.json':'c'*64}
        journal={'records':{'record':dict(status='published',artifact_sha256=hashes,
                    record_sha256=_record_fingerprint(record,hashes),checkpoint_step=25000,checkpoint_sha256='a'*64)}}
        save(tmp_path/directory/'run.json',dict(run_id=run_id,identity=ident,identity_sha256=_digest(ident)))
        save(tmp_path/directory/'publication.json',journal)
        save(tmp_path/directory/'records/record.json',record)
    return campaign,tmp_path


def test_seed_pairing_and_noncomparable_fields(exported):
    campaign,root=exported
    data=comparison.build_overlay(campaign,root)
    assert len(data['points']) == 2
    for row in data['points']:
        assert row['gain_mean'] == statistics.mean([1,4,-1,2,5])
        assert row['gain_std'] == statistics.stdev([1,4,-1,2,5])
        assert row['return_min'] == 11
        assert row['fresh_comparison_state'] == 'not_applicable'
        assert all(row[key] is None for key in ('fresh_gain_mean','fresh_gain_std',
                                              'control_seconds','control_std','late_seconds','late_std'))


@pytest.mark.parametrize('change', ['backbone','checkpoint','missing','unpublished','tampered'])
def test_rejects_incompatible_or_unverified_data(exported,change):
    campaign,root=exported
    directory=root/'mppi-return-q'
    if change=='backbone':campaign['source_run']='another/backbone'
    elif change=='checkpoint':campaign['checkpoints'][0]['checkpoint_sha256']='d'*64
    elif change in ('missing','unpublished'):
        journal=comparison.read(directory/'publication.json')
        if change=='missing':journal['records']={}
        else:journal['records']['record']['status']='queued'
        save(directory/'publication.json',journal)
    else:
        record=comparison.read(directory/'records/record.json');record['episodes'][0]['return']+=1
        save(directory/'records/record.json',record)
    with pytest.raises(SeriesError):comparison.build_overlay(campaign,root)


def test_overlay_retry_reuses_id_and_published_retry_is_noop(exported,tmp_path):
    campaign,root=exported;data=comparison.build_overlay(campaign,root)
    state=dict(entity='entity',project='project',publication_id='campaign')
    path=tmp_path/'overlay.json';calls=[];logs=[]
    def initialize(**kwargs):
        calls.append(kwargs)
        def finish(**kwargs):
            if len(calls)==1 and not kwargs:raise RuntimeError('uncertain acknowledgement')
        return SimpleNamespace(log=logs.append,summary={},finish=finish)
    wandb=SimpleNamespace(init=initialize,Table=lambda **kwargs:kwargs)
    with pytest.raises(RuntimeError):comparison.publish_overlay(data,'c'*64,state,path,wandb)
    assert comparison.read(path)['status']=='allocated'
    first_id=comparison.read(path)['run_id']
    receipt=comparison.publish_overlay(data,'c'*64,state,path,wandb)
    assert receipt['status']=='published' and all(call['id']==first_id for call in calls)
    assert calls[0]['config']['transfer_curve_overview']==state['publication_id']
    assert set(logs[-1])=={'transfer_curves/points','transfer_curves/progress'}
    assert comparison.publish_overlay(data,'c'*64,state,path,wandb)==receipt and len(calls)==2
    changed=deepcopy(data);changed['points'][0]['return_mean']+=1
    with pytest.raises(SeriesError,match='pins'):
        comparison.publish_overlay(changed,'c'*64,state,path,wandb)
