"""Interleaving preserves work, minibatches and frozen state, and changes order."""
from copy import deepcopy
import json

import pytest
import torch

from tests.test_ambi_root_local_sac import _tiny_component_model, _build_cfg
from tests.test_ambi_inner_decoupling import _assert_tree_equal
from slurm.ambi_aux_hj_sweep import MATRIX, cells


@pytest.mark.parametrize('actors', [4, 8, 16, 32])
@pytest.mark.parametrize('sampling', ['with_replacement', 'without_replacement'])
def test_interleaved_order_minibatches_and_target_cadence(monkeypatch, actors, sampling):
    logs = {}
    for order in ('critic_first', 'interleaved'):
        holder = _tiny_component_model(
            inner_component_update_order=order, inner_rounds=2,
            inner_rollouts_per_round=8, inner_rollout_horizon=3,
            inner_replay_capacity=48, inner_batch_size=16,
            inner_critic_updates_per_round=32, inner_actor_updates_per_round=actors,
            inner_replay_sampling=sampling, inner_critic_target_update_interval=1,
            inner_sac_critic_target='entropy_augmented', inner_finite_horizon=True,
        )
        try:
            with torch.no_grad():
                generator = torch.Generator().manual_seed(811)
                for head in holder.agent.model._Qs:
                    head[-1].weight.copy_(torch.randn(head[-1].weight.shape, generator=generator)*.1)
            outer = deepcopy(holder.agent.model.state_dict())
            engine = holder.agent.inner_engine
            events, batches, alpha_seen, actor_seen = [], {'c': [], 'a': []}, [], []
            critic, policy = engine._sac_critic_step, engine._sac_policy_step
            def critic_step(batch, alpha, **kwargs):
                events.append('c'); batches['c'].append(batch['sample_ids'].clone())
                alpha_seen.append(float(alpha))
                actor_seen.append(torch.cat([p.detach().flatten() for p in engine.state.actor.parameters()]).clone())
                return critic(batch, alpha, **kwargs)
            def policy_step(batch, **kwargs):
                events.append('a'); batches['a'].append(batch['sample_ids'].clone())
                return policy(batch, **kwargs)
            monkeypatch.setattr(engine, '_sac_critic_step', critic_step)
            monkeypatch.setattr(engine, '_sac_policy_step', policy_step)
            engine.reset_for_evaluation(3818519826)
            holder.agent.act(torch.tensor([.7, .3, -.2]), t0=True, eval_mode=True)
            _assert_tree_equal(outer, holder.agent.model.state_dict())
            expected = (['c']*32+['a']*actors if order=='critic_first'
                        else (['c']*(32//actors)+['a'])*actors)
            assert events == expected*2
            metrics = holder.agent.last_inner_metrics
            assert metrics['inner_critic_optimizer_steps']==64
            assert metrics['inner_actor_optimizer_steps']==metrics['inner_temperature_optimizer_steps']==2*actors
            assert metrics['inner_critic_target_updates']==64
            assert metrics['inner_model_steps']==metrics['inner_buffer_size']==48
            assert all(torch.equal(actor_seen[0], x) for x in actor_seen[:32//actors])
            if order=='interleaved':
                assert not torch.equal(actor_seen[0], actor_seen[32//actors])
                assert alpha_seen[0]!=alpha_seen[32//actors]
            else:
                assert all(torch.equal(actor_seen[0], x) for x in actor_seen[:32])
                assert len(set(alpha_seen[:32]))==1
            logs[order] = dict(batches=batches, rng=deepcopy(engine.rng.training_state_dict()))
        finally:
            holder.close()
    # Sample IDs and RNG advancement are equal even with multiple collection rounds.
    _assert_tree_equal(logs['critic_first'], logs['interleaved'])


@pytest.mark.parametrize('overrides', [
    {'inner_critic_updates_per_round': 0}, {'inner_actor_updates_per_round': 0},
    {'inner_actor_updates_per_round': 64}, {'inner_actor_updates_per_round': 3},
    {'inner_critic_updates_per_round': 'auto'},
    {'inner_operator': 'td3'}, {'inner_actor_adaptation': 'frozen'},
    {'inner_critic_adaptation': 'frozen'}, {'inner_outer_replay_fraction': .5},
    {'inner_update_timing': 'step', 'inner_steps_per_update': 1},
    {'inner_updates_per_round': 4}, {'inner_component_update_order': 'invalid'},
])
def test_unsupported_interleaving_rejected(overrides):
    with pytest.raises(ValueError):
        _build_cfg(**{'inner_component_update_order':'interleaved',
                         'inner_critic_updates_per_round':32, 'inner_actor_updates_per_round':4,
                         **overrides})


def test_update_order_identity_and_matched_grid():
    from utils.eval_series_data import planner_identity
    base = {'inner_operator': 'sac', 'inner_rounds': 1}
    def identity(params):
        return planner_identity(params, {}, 'AMBITDMPC2/AMBITDMPC2', 'tanh_mean')
    assert identity(base)==identity({**base,'inner_component_update_order':'critic_first'})
    assert identity(base)!=identity({**base,'inner_component_update_order':'interleaved'})
    old = json.loads(MATRIX.with_name('ambi_aux_actor_budget_625k.json').read_text())
    path = MATRIX.with_name('ambi_aux_interleaved_625k.json')
    new = json.loads(path.read_text())
    assert new['shared_alg_params']==old['shared_alg_params']
    assert len(cells(path))==12
    for cell in cells(path):
        before = old['comparisons']['sweep']['variants'][cell['name'].removesuffix('_interleaved')]['alg_params']
        assert cell['params']=={**before,'inner_component_update_order':'interleaved'}


def test_phased_pairing_allows_only_update_order():
    from slurm.ambi_aux_interleaved import phased_baseline
    from slurm.ambi_aux_hj_sweep import CHECKPOINT_SHA, polyak_comparison
    old = dict(identity=dict(backbone={'id':'b'}, protocol={'id':'p'}, science={'id':'old'},
                             planner=dict(settings=dict(inner_rounds=1, inner_critic_updates_per_round=32,
                                                        inner_actor_updates_per_round=8))),
               checkpoint=dict(step=625000,sha256=CHECKPOINT_SHA),record_id='r',
               metrics={'eval/frozen_state_unchanged':True},
               episodes=[dict(seed=s,solver_seed=55,length=500,truncated_by_evaluator=False,
                              **{'return':float(s)}) for s in range(101,106)])
    new = deepcopy(old)
    new['identity']['science'] = {'id':'new'}
    new['identity']['planner']['settings']['inner_component_update_order'] = 'interleaved'
    baseline = phased_baseline(new,old,baseline_science={'id':'old'})
    episodes = [{**e,'return':e['return']+3} for e in reversed(old['episodes'])]
    assert polyak_comparison(episodes,baseline)['metrics']['comparison/phased_gain_mean']==3
    for group,key,value in [('science','id','wrong'),('protocol','id','wrong')]:
        wrong = deepcopy(old); wrong['identity'][group][key]=value
        with pytest.raises(AssertionError): phased_baseline(new,wrong,baseline_science={'id':'old'})
    wrong = deepcopy(new); wrong['identity']['planner']['settings']['inner_actor_updates_per_round']=16
    with pytest.raises(AssertionError): phased_baseline(wrong,old,baseline_science={'id':'old'})
    episodes[0]['solver_seed']=56
    with pytest.raises(AssertionError): polyak_comparison(episodes,baseline)
