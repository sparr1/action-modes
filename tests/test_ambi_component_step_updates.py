"""Asymmetric step cadence preserves budgets, causal ordering and H1 behavior."""
from copy import deepcopy
import pytest
import torch
from RL.tdmpc2_core.inner_trace import InnerActionTrace
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_component_model
from tests.test_ambi_inner_step_updates import _snapshot
from tests.test_ambi_latency_contract import _assert_tree_equal


def model(**overrides):
    options=dict(inner_update_timing='step',inner_rollout_horizon=3,train_unroll_horizon=3,
                 inner_rounds=2,inner_rollouts_per_round=4,inner_batch_size=8,
                 inner_replay_capacity=96,inner_critic_updates_per_round=16,
                 inner_actor_updates_per_round=4,inner_finite_horizon=True,
                 inner_component_update_order='critic_first',dropout=.1)
    options.update(overrides)
    return _tiny_component_model(**options)


@pytest.mark.parametrize('h,c,a',[(1,16,4),(2,16,4),(3,16,4),(7,3,2),(3,0,4),(3,16,0)])
def test_exact_depth_doses_replay_and_target_counts(h,c,a,monkeypatch):
    holder=model(inner_rollout_horizon=h,train_unroll_horizon=h,
                 inner_critic_updates_per_round=c,inner_actor_updates_per_round=a)
    engine=holder.agent.inner_engine;calls=[];run=engine._run_component_update_counts
    def record(**kw):
        calls.append((engine.state.replay.size,kw['critic_count'],kw['actor_count']))
        return run(**kw)
    monkeypatch.setattr(engine,'_run_component_update_counts',record)
    try:
        for _ in range(2):
            calls.clear();trace=InnerActionTrace(probes=True,probe_rollouts=2)
            holder.agent.act(torch.tensor([.2,.5,-.1]),t0=True,eval_mode=True,trace=trace)
            expected=[(4*(r*h+d),c*d//h-c*(d-1)//h,a*d//h-a*(d-1)//h)
                      for r in range(2) for d in range(1,h+1)]
            assert calls==expected
            metrics=holder.agent.last_inner_metrics
            assert metrics['inner_model_steps']==metrics['inner_buffer_size']==8*h
            assert metrics['inner_critic_optimizer_steps']==metrics['inner_critic_target_updates']==2*c
            assert metrics['inner_actor_optimizer_steps']==metrics['inner_temperature_optimizer_steps']==2*a
            assert metrics['inner_replay_draws']==2*(c+a)*8
            assert metrics['inner_requested_update_slots']==2*(c+a)
            replay=engine._action_pool.replay
            assert replay.size==replay.next_sample_id==8*h
            torch.testing.assert_close(replay.sample_id[:8*h],torch.arange(8*h))
            assert replay.horizon_end[:8*h,0].tolist()==([0.]*(4*(h-1))+[1.]*4)*2
            assert [e['metrics']['collection_rollout_step'] for e in trace.events if e['phase']=='collection']==list(range(1,h+1))*2
    finally:holder.close()


def test_updated_actor_is_used_at_next_depth_without_restarting_branches(monkeypatch):
    holder=model();engine=holder.agent.inner_engine;calls=[];sample=engine._policy_action
    def record(z,actor,**kw):
        out=sample(z,actor,**kw)
        if z.shape[0]==4:
            calls.append((engine.state.actor_steps,z.clone(),[p.detach().clone() for p in actor.parameters()]))
        return out
    monkeypatch.setattr(engine,'_policy_action',record)
    try:
        holder.agent.act(torch.tensor([.2,.5,-.1]),t0=True,eval_mode=True)
        assert [c[0] for c in calls]==[0,1,2,4,5,6]
        assert any(not torch.equal(x,y) for x,y in zip(calls[0][2],calls[1][2]))
        replay=engine._action_pool.replay
        for start in (0,4,12,16):
            torch.testing.assert_close(replay.z[start+4:start+8],replay.next_z[start:start+4],rtol=0,atol=0)
        torch.testing.assert_close(calls[0][1],calls[3][1],rtol=0,atol=0)
    finally:holder.close()


@pytest.mark.parametrize('n',[32,128])
@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA validation runs on Oscar'))])
def test_h1_matches_round_actions_parameters_optimizers_replay_rng_and_work(n,device):
    opts=dict(device=device,compile=device=='cuda',compile_strict=True,inner_rollout_horizon=1,inner_rounds=2,inner_rollouts_per_round=n,
              inner_batch_size=256,inner_replay_capacity=3072)
    round_model=model(inner_update_timing='round',**opts);step_model=model(**opts)
    try:
        for _ in range(2):
            a=round_model.agent.act(torch.tensor([.2,.5,-.1]),t0=True,eval_mode=True)
            b=step_model.agent.act(torch.tensor([.2,.5,-.1]),t0=True,eval_mode=True)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            _assert_tree_equal(_snapshot(round_model.agent),_snapshot(step_model.agent))
            for key in ('inner_model_steps','inner_replay_draws','inner_critic_optimizer_steps','inner_actor_optimizer_steps','inner_critic_target_updates'):
                assert round_model.agent.last_inner_metrics[key]==step_model.agent.last_inner_metrics[key]
    finally:round_model.close();step_model.close()


def test_trace_does_not_change_training_and_outer_weights_stay_frozen():
    ordinary=model();traced=model();before=deepcopy(ordinary.agent.model.state_dict())
    try:
        a=ordinary.agent.act(torch.tensor([.2,.5,-.1]),t0=True,eval_mode=True)
        b=traced.agent.act(torch.tensor([.2,.5,-.1]),t0=True,eval_mode=True,trace=InnerActionTrace(probes=True,probe_rollouts=2))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        _assert_tree_equal(_snapshot(ordinary.agent),_snapshot(traced.agent))
        _assert_tree_equal(before,ordinary.agent.model.state_dict())
    finally:ordinary.close();traced.close()


def test_without_replacement_checks_first_eligible_depth():
    opts=dict(inner_update_timing='step',inner_rollout_horizon=3,inner_rounds=2,
              inner_rollouts_per_round=4,inner_replay_capacity=24,inner_batch_size=8,
              inner_replay_sampling='without_replacement',inner_critic_updates_per_round=16,
              inner_actor_updates_per_round=4)
    with pytest.raises(ValueError,match='before the first update'):_build_cfg(**opts)
    cfg=_build_cfg(**{**opts,'inner_critic_updates_per_round':1,'inner_actor_updates_per_round':1})
    assert cfg.inner_expected_update_slots==4


def test_component_step_rejects_conflicting_interval_or_interleaved_order():
    opts=dict(inner_update_timing='step',inner_critic_updates_per_round=16,inner_actor_updates_per_round=4)
    with pytest.raises(ValueError,match='joint SAC'):_build_cfg(**opts,inner_steps_per_update=4)
    with pytest.raises(ValueError,match='critic_first'):_build_cfg(**opts,inner_component_update_order='interleaved')


def test_early_termination_only_runs_doses_for_collected_depths(monkeypatch):
    holder=model(episodic=True)
    monkeypatch.setattr(holder.agent.model,'termination',lambda z:z.new_ones((len(z),1)))
    try:
        holder.agent.act(torch.tensor([.2,.5,-.1]),t0=True,eval_mode=True)
        m=holder.agent.last_inner_metrics
        assert m['inner_model_steps']==8 and m['inner_critic_optimizer_steps']==10
        assert m['inner_actor_optimizer_steps']==m['inner_temperature_optimizer_steps']==2
    finally:holder.close()


def test_component_step_compiled_graphs_reused(monkeypatch):
    torch._dynamo.reset();graphs=[];compile_=torch.compile
    def backend(graph,inputs):graphs.append(graph);return graph.forward
    monkeypatch.setattr(torch,'compile',lambda fn,**kw:compile_(fn,backend=backend,**kw))
    holder=model(compile=True,compile_strict=True)
    try:
        holder.agent.act(torch.tensor([.2,.5,-.1]),t0=True,eval_mode=True)
        count=len(graphs);assert count>0
        holder.agent.act(torch.tensor([.2,.5,-.1]),t0=True,eval_mode=True)
        assert len(graphs)==count and holder.agent.last_inner_metrics['inner_compile_fallback']==0
    finally:holder.close();torch._dynamo.reset()
