"""Exact weighted reductions, including missing groups and unequal seed coverage."""

import gzip
import json

import pytest

from utils.horizon_training_summary import HorizonTrainingSummary, seed_means


def event(seed, decision, count, qsum, *, index=1, h=1):
    prefix = f'critic_horizon_{h}_'
    return dict(episode_id=seed, decision_index=decision, updated_critic=True,
                critic_updates=index, metrics={prefix+'sample_count':count,
                    prefix+'predicted_q_sum':qsum, prefix+'target_q_sum':qsum*2,
                    prefix+'td_error_abs_sum':abs(qsum)})


def test_weight_samples_within_seed_then_weight_seeds_equally():
    stats = HorizonTrainingSummary()
    # Seed a: one sample at 10 and three at 2 -> mean 4, not 6.
    # Seed b: two at 20 -> equal-seed mean (4+20)/2 = 12, not 56/6.
    for row in (event('a',0,1,10), event('a',1,3,6), event('b',0,2,40),
                event('a',0,0,0,h=2), event('b',0,0,0,h=2)):
        stats.add(row)
    curves = stats.update_curves()[('critic_update',1)]
    assert curves['critic_horizon_1_predicted_q_mean'] == dict(mean=12, min=4, max=20, count=2, sample_count=6)
    assert curves['critic_horizon_1_target_q_mean']['mean'] == 24
    assert curves['critic_horizon_2_sample_count']['mean'] == 0
    assert 'critic_horizon_2_predicted_q_mean' not in curves
    assert stats.decision_curves()[0]['critic_horizon_1_predicted_q_mean']['mean'] == 15
    assert seed_means(stats.decisions[('a',1)])['critic_horizon_1_sample_count'] == 3


def test_decision_means_pool_updates_before_cross_seed_average():
    stats=HorizonTrainingSummary()
    for row in (event('a',0,1,10),event('a',0,3,6,index=2),event('b',0,2,40),
                event('b',0,0,0,index=2)):
        stats.add(row)
    assert stats.decision_curves()[0]['critic_horizon_1_predicted_q_mean']['mean'] == 12
    assert stats.update_curves()[('critic_update',2)]['critic_horizon_1_predicted_q_mean']['count'] == 1


@pytest.mark.parametrize('change', ['missing', 'negative', 'empty_sum', 'wrong_axis'])
def test_corrupt_group_rejected(change):
    row=event('a',0,0,0)
    if change=='missing': row['metrics'].pop('critic_horizon_1_target_q_sum')
    if change=='negative': row['metrics']['critic_horizon_1_sample_count']=-1
    if change=='empty_sum': row['metrics']['critic_horizon_1_predicted_q_sum']=1
    if change=='wrong_axis': row['updated_critic']=False
    with pytest.raises(ValueError): HorizonTrainingSummary().add(row)


def test_existing_training_publisher_receives_derived_means_and_coverage(tmp_path):
    from slurm.ambi_aux_hj_sweep import training_summary
    cell={'H':1,'J':1,'params':{'inner_critic_updates_per_round':2,
                              'inner_actor_updates_per_round':1,'inner_entropy_enabled':True}}
    rows=[]
    for seed, pairs in [('a',[(1,10),(3,6)]),('b',[(2,40),(0,0)])]:
        base=dict(episode_id=seed,decision_index=0,round_index=1,replay_size=0,metrics={})
        rows.append(dict(base,phase='initial'))
        rows.append(dict(base,phase='collection',replay_size=128))
        for index,(count,qsum) in enumerate(pairs,1):
            row=event(seed,0,count,qsum,index=index)
            row.update(phase='update',actor_updates=0)
            row['metrics'].update(critic_loss=1.,critic_grad_norm=1.,td_error_abs_mean=1.,q_target_mean=1.)
            rows.append(row)
        rows.append(dict(base,phase='update',updated_actor=True,updated_temperature=True,actor_updates=1,
                         metrics=dict(actor_loss=2.,actor_grad_norm=2.,actor_entropy=2.,alpha_used=.2)))
        rows.append(dict(base,phase='decision',metrics={'decision/reward':7.}))
    manifest={'runs':[{'trace_files':['trace.jsonl.gz'],'episodes':[{},{}]}],'metric_catalog':{}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with gzip.open(tmp_path/'trace.jsonl.gz','wt') as output:
        output.write('\n'.join(json.dumps(row) for row in rows)+'\n')
    summary=training_summary(tmp_path,cell,expected_steps=1)
    assert summary['decision_curves'][0]['metrics']['critic_horizon_1_predicted_q_mean']['mean']==12
    assert summary['decision_curves'][0]['metrics']['critic_horizon_1_predicted_q_mean']['sample_count']==6
    assert summary['decision_curves'][0]['metrics']['decision/reward']['mean']==7
    assert summary['update_curves'][0]['metrics']['actor_loss']['mean']==2
