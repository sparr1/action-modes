"""Round-sized replay must behave exactly like clearing before collection."""
from copy import deepcopy
import json

import pytest
import torch

from slurm.ambi_aux_hj_sweep import MATRIX, CHECKPOINT_SHA, cells, round_replay_baseline, polyak_comparison
from tests.test_ambi_root_local_sac import _tiny_component_model, _build_cfg
from tests.test_ambi_inner_decoupling import _assert_tree_equal


def test_round_replay_grid_changes_only_reset():
    original=json.loads(MATRIX.read_text())
    matrix=MATRIX.with_name('ambi_aux_round_replay_sweep_625k.json')
    updated=json.loads(matrix.read_text())
    assert original['shared_alg_params']==updated['shared_alg_params']
    panel=cells(matrix)
    assert len(panel)==18
    assert {(c['name'].split('_h')[0],c['H'],c['J']) for c in panel} == {
        (arm,h,j) for arm in ('soft_soft','soft_return') for h in (1,2,3) for j in (1,2,4)}
    for c in panel:
        before=original['comparisons']['sweep']['variants'][c['name'].removesuffix('_roundreplay')]['alg_params']
        assert c['params']=={**before,'inner_replay_reset_each_round':True}
    assert updated['shared_alg_params']['inner_critic_target_tau']==.01


def _parameters(holder):
    state=holder.agent.inner_engine._action_pool
    return deepcopy({
        'actor':state.actor.state_dict(),'critic':state.critic.state_dict(),
        'target':state.critic_target.state_dict(),'alpha':state.log_alpha.detach(),
        'actor_optim':state.actor_optim.state_dict(),'critic_optim':state.critic_optim.state_dict(),
        'alpha_optim':state.temperature_optim.state_dict(),
        'rng':holder.agent.inner_engine.rng.training_state_dict(),
    })


@pytest.mark.parametrize('horizon',[1,2,3])
@pytest.mark.parametrize('rounds',[1,2,4])
def test_all_updates_use_current_round_and_match_explicit_clear(monkeypatch,horizon,rounds):
    capacity=128*horizon
    options=dict(inner_rollouts_per_round=128,inner_rollout_horizon=horizon,inner_rounds=rounds,
                 inner_critic_updates_per_round=32,inner_actor_updates_per_round=4,
                 inner_batch_size=256,inner_finite_horizon=True,inner_update_timing='round',
                 inner_critic_target_tau=.01,inner_temperature_lr=3e-4,dropout=.01)
    bounded=_tiny_component_model(inner_replay_capacity=2048,inner_replay_reset_each_round=True,**options)
    cleared=_tiny_component_model(inner_replay_capacity=2048,**options)
    try:
        for holder in (bounded,cleared):
            with torch.no_grad():
                gen=torch.Generator().manual_seed(811)
                for head in holder.agent.model._Qs:
                    head[-1].weight.copy_(torch.randn(head[-1].weight.shape,generator=gen)*.1)
        engine=bounded.agent.inner_engine
        original_collect=engine._collect_round
        def empty_then_collect(root):
            assert engine.state.replay.size == 0
            return original_collect(root)
        monkeypatch.setattr(engine,'_collect_round',empty_then_collect)
        sample=engine._sample_batch
        draws=[]
        def checked_sample(indices=None):
            batch=sample(indices)
            replay=engine.state.replay
            # Monotone IDs are assigned at collection time, not sampled anew.
            end=replay.next_sample_id
            assert replay.size==capacity and end%capacity==0
            assert bool(((batch['sample_ids']>=end-capacity)&(batch['sample_ids']<end)).all())
            draws.append(end//capacity)
            return batch
        monkeypatch.setattr(engine,'_sample_batch',checked_sample)
        reference=cleared.agent.inner_engine
        collect=reference._collect_round
        def clear_then_collect(root):
            reference.state.replay.clear(preserve_sample_ids=True)
            return collect(root)
        monkeypatch.setattr(reference,'_collect_round',clear_then_collect)
        obs=torch.tensor([.7,.3,-.2])
        actions=[]
        for holder in (bounded,cleared):
            holder.agent.inner_engine.reset_for_evaluation(3818519826)
            actions.append(holder.agent.act(obs,t0=True,eval_mode=True))
        torch.testing.assert_close(actions[0],actions[1],rtol=0,atol=0)
        _assert_tree_equal(_parameters(bounded),_parameters(cleared))
        assert draws==[r for r in range(1,rounds+1) for _ in range(36)]
        metrics=bounded.agent.last_inner_metrics
        assert metrics['inner_model_steps']==capacity*rounds
        assert metrics['inner_buffer_size']==capacity
        assert metrics['inner_critic_optimizer_steps']==32*rounds
        assert metrics['inner_actor_optimizer_steps']==4*rounds
    finally:
        bounded.close();cleared.close()


def test_round_replay_baseline_rejects_other_setting_changes():
    settings=dict(inner_replay_capacity=2048,inner_rollouts_per_round=128,inner_rollout_horizon=3,
                  inner_rounds=4,inner_critic_target_tau=.01)
    record=dict(identity=dict(backbone={'id':'b'},protocol={'id':'p'},science={'id':'s'},
                              planner=dict(settings=settings)),
                checkpoint=dict(step=625000,sha256=CHECKPOINT_SHA),record_id='r',
                metrics={'eval/frozen_state_unchanged':True},
                episodes=[dict(seed=s,solver_seed=s+1,length=500,truncated_by_evaluator=False,
                               **{'return':float(s)}) for s in range(101,106)])
    spec=deepcopy(record)
    spec['identity']['planner']['settings']['inner_replay_reset_each_round']=True
    baseline=round_replay_baseline(spec,record)
    episodes=[{**e,'return':e['return']+3} for e in record['episodes']]
    comparison=polyak_comparison(episodes,baseline)
    assert comparison['metrics']['comparison/all_round_replay_gain_mean']==3
    assert comparison['metrics']['comparison/all_round_replay_gain_ci95_low']==3
    assert not any('tau001' in k for k in comparison['metrics'])
    spec['identity']['planner']['settings']['inner_critic_target_tau']=.1
    with pytest.raises(AssertionError):round_replay_baseline(spec,record)


@pytest.mark.parametrize('overrides',[
    {'inner_replay_reset_each_round':'true'},
    {'inner_replay_reset_each_round':1},
    {'inner_replay_scope':'run'},
    {'inner_update_timing':'step','inner_steps_per_update':1},
    {'inner_outer_replay_fraction':.5},
    {'inner_operator':'td3'},
])
def test_round_reset_rejects_ambiguous_semantics(overrides):
    with pytest.raises(ValueError,match='inner_replay_reset_each_round'):
        _build_cfg(**{'inner_replay_reset_each_round':True,**overrides})


def test_reset_flag_identity_and_capacity():
    from utils.eval_series_data import planner_identity
    cfg=_build_cfg(inner_rounds=4,inner_rollouts_per_round=8,inner_rollout_horizon=2,
                   inner_replay_reset_each_round=True,inner_replay_capacity=16)
    assert cfg.inner_replay_capacity==16
    base={'inner_operator':'sac','inner_rounds':2}
    old=planner_identity(base,{},'AMBITDMPC2/AMBITDMPC2','tanh_mean')
    explicit=planner_identity({**base,'inner_replay_reset_each_round':False},{},'AMBITDMPC2/AMBITDMPC2','tanh_mean')
    changed=planner_identity({**base,'inner_replay_reset_each_round':True},{},'AMBITDMPC2/AMBITDMPC2','tanh_mean')
    assert old==explicit and old!=changed
    with pytest.raises(ValueError,match='inner_replay_capacity'):
        _build_cfg(inner_rounds=4,inner_rollouts_per_round=8,inner_rollout_horizon=2,
                   inner_replay_reset_each_round=True,inner_replay_capacity=15)


def test_clear_keeps_transition_ids_only_when_requested():
    from RL.tdmpc2_core.common.latent_buffer import LatentReplayBuffer
    replay=LatentReplayBuffer(8,2,1,'cpu')
    def append(n):
        replay.add_batch(torch.ones(n,2),torch.ones(n,1),torch.ones(n,1),torch.ones(n,2),torch.zeros(n,1))
    append(5)
    replay.clear(preserve_sample_ids=True)
    assert replay.size==0 and replay.next_sample_id==5
    append(2)
    assert replay.sample_id.tolist()[:2]==[5,6]
    replay.clear()
    assert replay.size==0 and replay.next_sample_id==0
