"""Whole-round ERE eligibility, update clocks, diagnostics and compatibility."""

from copy import deepcopy
import math

import pytest
import torch

from RL.tdmpc2_core.common.ere import round_windows
from RL.tdmpc2_core.common.latent_buffer import LatentReplayBuffer
from RL.tdmpc2_core.inner_trace import InnerActionTrace, metric_catalog
from tests.test_ambi_root_local_sac import _build_cfg, _tiny_model, _tiny_component_model
from tests.test_ambi_inner_step_updates import _assert_tree_equal
from utils.eval_series_data import planner_identity
from utils.resume_identity import scientific_trial_parameters


def _snapshot(agent):
    pool = agent.inner_engine._action_pool
    result = {}
    for name in ('actor', 'critic', 'critic_target', 'actor_optim',
                 'critic_optim', 'temperature_optim', 'replay'):
        value = getattr(pool, name)
        result[name] = None if value is None else deepcopy(value.state_dict())
    result['log_alpha'] = deepcopy(pool.log_alpha)
    result['outer'] = deepcopy(agent.model.state_dict())
    result['rng'] = deepcopy(agent.inner_engine.rng.training_state_dict())
    return result


@pytest.fixture
def models():
    opened = []
    def make(*, component=True, **kwargs):
        options = dict(inner_replay_strategy="ere", inner_diagnostic_rollouts=0)
        if component:
            options.update(inner_critic_updates_per_round=4, inner_actor_updates_per_round=2)
        options.update(kwargs)
        model = (_tiny_component_model if component else _tiny_model)(**options)
        opened.append(model)
        return model
    yield make
    for model in opened:
        model.env.close()


@pytest.mark.parametrize("rounds", [1, 2, 3, 8])
@pytest.mark.parametrize("updates", [0, 1, 2, 4, 8, 16])
@pytest.mark.parametrize("fraction,minimum", [(1.,1),(.25,1),(.018,1),(.25,3),(.001,12)])
def test_formula_bounds_and_endpoints(rounds, updates, fraction, minimum):
    values = round_windows(rounds, updates, fraction, minimum)
    assert len(values) == updates
    assert all(1 <= w <= rounds for w in values)
    assert values == sorted(values, reverse=True)
    if updates:
        assert values[0] == rounds
    if updates > 1:
        assert values[-1] == min(rounds, max(minimum, math.ceil(rounds*fraction)))
    assert round_windows(8, 4, .25, 1) == [8, 6, 4, 2]
    assert round_windows(8, 16, .25, 1)[-1] == 2


@pytest.mark.parametrize("key,value", [
    ("inner_ere_actor", 0), ("inner_ere_actor", "false"),
    ("inner_replay_strategy", "recent"), ("inner_replay_strategy", None),
    ("inner_ere_final_fraction", 0), ("inner_ere_final_fraction", -1),
    ("inner_ere_final_fraction", 1.1), ("inner_ere_final_fraction", float("nan")),
    ("inner_ere_final_fraction", float("inf")), ("inner_ere_final_fraction", True),
    ("inner_ere_min_rounds", 0), ("inner_ere_min_rounds", 1.5),
    ("inner_ere_min_rounds", True), ("inner_ere_min_rounds", "2"),
])
def test_invalid_options(key,value):
    with pytest.raises(ValueError, match=key):
        _build_cfg(**{key:value})


@pytest.mark.parametrize("options", [
    {"inner_update_timing":"step", "inner_steps_per_update":2},
    {"inner_replay_scope":"episode"},
    {"inner_actor_scope":"episode"},
    {"inner_critic_scope":"episode"},
    {"inner_temperature_scope":"episode"},
    {"inner_operator":"td3"},
    {"inner_explorer_mode":"shared_mixture"},
    {"inner_outer_replay_fraction":.5},
])
def test_unsupported_ere_combinations(options):
    with pytest.raises(ValueError, match="ere"):
        _build_cfg(inner_replay_strategy="ere", **options)


def _append(replay, start, count):
    values = torch.arange(start,start+count,dtype=torch.float32).reshape(-1,1)
    replay.add_batch(values,values,values,values+1,torch.zeros_like(values))


@pytest.mark.parametrize("replacement",[False,True])
@pytest.mark.parametrize("sizes",[(3,),(3,4),(8,3)])
def test_recent_indices_handle_ring_wrap_and_preserve_full_window_rng(replacement,sizes):
    replay = LatentReplayBuffer(5,1,1,"cpu")
    start=0
    for count in sizes:
        _append(replay,start,count); start+=count
    generator=torch.Generator().manual_seed(19)
    indices=replay.draw_recent_indices(2,2,replacement=replacement,generator=generator)
    batch=replay.sample(2,indices=indices)
    assert set(batch['sample_ids'].tolist()) <= {start-2,start-1}
    if not replacement: assert len(set(batch['sample_ids'].tolist())) == 2
    a,b=[torch.Generator().manual_seed(32) for _ in range(2)]
    indices=replay.draw_recent_indices(2,replay.size,replacement=replacement,generator=a)
    expected=replay.sample(2,replacement=replacement,generator=b)['indices']
    torch.testing.assert_close(indices,expected,rtol=0,atol=0)
    torch.testing.assert_close(a.get_state(),b.get_state(),rtol=0,atol=0)
    with pytest.raises(ValueError,match="eligible"):
        replay.draw_recent_indices(3,2,replacement=False,generator=a)


@pytest.mark.parametrize("order",["critic_first","interleaved"])
@pytest.mark.parametrize("sampling",["with_replacement","without_replacement"])
def test_component_windows_depth_eligibility_and_budgets(models,monkeypatch,order,sampling):
    model=models(inner_component_update_order=order, inner_horizon_diagnostics=True,
                 inner_finite_horizon=True, inner_replay_sampling=sampling)
    engine=model.agent.inner_engine
    original=engine._sample_batch
    draws=[]
    def sample(indices=None):
        batch=original(indices)
        draws.append((list(engine.state.replay_rounds),batch['sample_ids'].clone()))
        return batch
    monkeypatch.setattr(engine,'_sample_batch',sample)
    for _ in range(2):
        draws.clear()
        trace=InnerActionTrace()
        model.agent.act(torch.zeros(3),trace=trace,collect_diagnostics=False)
        events=[e for e in trace.events if e['phase']=='update']
        assert len(events)==len(draws)==18
        for r in range(1,4):
            for component,K in [('critic',4),('actor',2)]:
                selected=[e for e in events if e['round_index']==r and e['updated_'+component]]
                assert [e['metrics'][component+'_replay_window_rounds'] for e in selected] == round_windows(r,K,.25,1)
        for event,(rounds,ids) in zip(events,draws):
            component='critic' if event['updated_critic'] else 'actor'
            window=int(event['metrics'][component+'_replay_window_rounds'])
            assert ids.min()>=rounds[-window][1] and ids.max()<rounds[-1][2]
            # Whole depth-1/depth-2 populations are eligible, even at W=1.
            assert rounds[-1][2]-rounds[-window][1] == 4*window
            ages=(rounds[-1][0]-1-ids//4).float()
            assert event['metrics'][component+'_replay_round_age_mean']==float(ages.mean())
            assert event['metrics'][component+'_replay_newest_round_fraction']==float((ages==0).float().mean())
            assert event['metrics'][component+'_replay_batch_unique_fraction']==ids.unique().numel()/ids.numel()
            assert event['updated_temperature']==event['updated_actor']
        metrics=model.agent.last_inner_metrics
        assert metrics['inner_model_steps']==12
        assert metrics['inner_critic_optimizer_steps']==12
        assert metrics['inner_actor_optimizer_steps']==metrics['inner_temperature_optimizer_steps']==6
        assert metrics['inner_critic_target_updates']==12
        assert metrics['inner_replay_draws']==72
        for component,steps in [('critic',12),('actor',6)]:
            assert sum(metrics[f'inner_{component}_replay_round_{r}_sample_count'] for r in range(1,4))==4*steps
        assert engine.state.replay_rounds==[] and engine.state.replay_round_counts=={}


@pytest.mark.parametrize("component",[False,True])
@pytest.mark.parametrize("replacement",["with_replacement","without_replacement"])
def test_uniform_equivalence_and_action_lifecycle(models,component,replacement):
    baseline=models(component=component,inner_replay_strategy='uniform',inner_replay_sampling=replacement)
    endpoint=models(component=component,inner_ere_final_fraction=1.,inner_replay_sampling=replacement)
    ignored=models(component=component,inner_replay_strategy='uniform',inner_ere_final_fraction=.1,
                   inner_ere_min_rounds=3,inner_replay_sampling=replacement)
    global_rng=torch.random.get_rng_state().clone()
    for _ in range(2):
        expected=baseline.agent.act(torch.zeros(3),collect_diagnostics=False)
        for model in (endpoint,ignored):
            actual=model.agent.act(torch.zeros(3),collect_diagnostics=False)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            _assert_tree_equal(_snapshot(model.agent),_snapshot(baseline.agent))
        torch.testing.assert_close(torch.random.get_rng_state(),global_rng,rtol=0,atol=0)


@pytest.mark.parametrize('ere_actor', [True, False])
def test_diagnostics_do_not_change_batches_or_rng(models, ere_actor):
    plain=models(inner_ere_actor=ere_actor)
    traced=models(inner_ere_actor=ere_actor)
    diagnosed=models(inner_ere_actor=ere_actor)
    for _ in range(2):
        expected=plain.agent.act(torch.zeros(3),collect_diagnostics=False)
        for model,diagnostics,trace in [(traced,False,InnerActionTrace()),(diagnosed,True,None)]:
            actual=model.agent.act(torch.zeros(3),collect_diagnostics=diagnostics,trace=trace)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            left,right=_snapshot(model.agent),_snapshot(plain.agent)
            # Existing model diagnostics own an independent RNG stream; ERE
            # sampling must leave every training/acting stream unchanged.
            for snapshot in (left,right):
                for key in ('streams','phase_streams'):
                    snapshot['rng'][key].pop('diagnostics')
            _assert_tree_equal(left,right)


def test_joint_slots_share_windows_and_minibatches(models):
    model=models(component=False,inner_updates_per_round=4)
    trace=InnerActionTrace()
    model.agent.act(torch.zeros(3),trace=trace,collect_diagnostics=False)
    events=[e for e in trace.events if e['phase']=='update']
    assert len(events)==12
    for event in events:
        assert event['updated_actor'] and event['updated_critic'] and event['updated_temperature']
        for suffix in ('window_rounds','window_transitions','round_age_mean','batch_unique_fraction'):
            assert event['metrics']['critic_replay_'+suffix]==event['metrics']['actor_replay_'+suffix]
    assert model.agent.last_inner_metrics['inner_replay_draws']==48


def test_round_reset_is_uniform_and_keeps_generation_counts(models):
    baseline=models(inner_replay_strategy='uniform',inner_replay_reset_each_round=True,inner_replay_capacity=4)
    ere=models(inner_replay_reset_each_round=True,inner_replay_capacity=4)
    for _ in range(2):
        expected=baseline.agent.act(torch.zeros(3),collect_diagnostics=False)
        trace=InnerActionTrace()
        actual=ere.agent.act(torch.zeros(3),collect_diagnostics=False,trace=trace)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        _assert_tree_equal(_snapshot(ere.agent),_snapshot(baseline.agent))
        for event in trace.events:
            if event['phase']=='update':
                component='critic' if event['updated_critic'] else 'actor'
                assert event['metrics'][component+'_replay_window_rounds']==1
                assert event['metrics'][component+'_replay_round_age_mean']==0
        metrics=ere.agent.last_inner_metrics
        assert [metrics[f'inner_critic_replay_round_{r}_sample_count'] for r in range(1,4)]==[16]*3


def test_episodic_round_boundaries_use_realized_counts(models,monkeypatch):
    model=models(episodic=True,inner_rollout_horizon=3,inner_replay_capacity=18,train_unroll_horizon=3)
    engine=model.agent.inner_engine
    calls=iter([True,False,True,False,False,True])
    def termination(z):
        return z.new_full((z.shape[0],1),float(next(calls)))
    monkeypatch.setattr(model.agent.model,'termination',termination)
    trace=InnerActionTrace()
    model.agent.act(torch.zeros(3),trace=trace,collect_diagnostics=False)
    collection=[e for e in trace.events if e['phase']=='collection']
    assert [e['metrics']['collection_transitions'] for e in collection]==[2,4,6]
    for r,size in enumerate([2,4,6],1):
        events=[e for e in trace.events if e['phase']=='update' and e['round_index']==r and e['updated_critic']]
        assert events[-1]['metrics']['critic_replay_window_transitions']==size
    assert engine._action_pool.replay.size==12


def test_small_eligible_pool_rejects_without_replacement_before_rng_or_updates(models):
    model=models(inner_replay_sampling='without_replacement')
    engine=model.agent.inner_engine
    with engine.rng.fork('initialization'): engine._prepare_workspace(t0=True)
    replay=engine.state.replay
    # Sizes differ: sufficient total replay, but newest round cannot fill B=4.
    for start,count in [(0,4),(4,2)]:
        z=torch.zeros(count,model.cfg.latent_dim)
        replay.add_batch(z,torch.zeros(count,1),torch.zeros(count,1),z,torch.zeros(count,1))
    engine.state.replay_rounds=[(1,0,4),(2,4,6)]
    rng=engine.rng.generator('replay').get_state().clone()
    with pytest.raises(ValueError,match='eligible'):
        engine._run_update_counts(critic_count=4,actor_count=0,temperature_count=0)
    torch.testing.assert_close(rng,engine.rng.generator('replay').get_state(),rtol=0,atol=0)
    assert engine.state.critic_steps==0


def test_identity_defaults_and_active_parameters():
    base={'alg':'AMBITDMPC2/AMBITDMPC2','alg_params':{}}
    uniform={**base,'alg_params':dict(inner_replay_strategy='uniform',inner_ere_final_fraction=.1,inner_ere_min_rounds=3)}
    assert scientific_trial_parameters(base)==scientific_trial_parameters(uniform)
    active={**base,'alg_params':dict(inner_replay_strategy='ere')}
    explicit={**base,'alg_params':dict(inner_replay_strategy='ere',inner_ere_final_fraction=.25,inner_ere_min_rounds=1)}
    assert scientific_trial_parameters(active)==scientific_trial_parameters(explicit)
    assert scientific_trial_parameters(active)!=scientific_trial_parameters(base)
    old=vars(_build_cfg()); historical=deepcopy(old)
    for key in ('inner_replay_strategy','inner_ere_final_fraction','inner_ere_min_rounds'): historical.pop(key)
    def identity(cfg): return planner_identity(cfg,{},base['alg'],'tanh_mean')
    assert identity(old)==identity(historical)
    cfg=vars(_build_cfg(inner_replay_strategy='ere'))
    assert identity(cfg)!=identity(old)
    for key,value in [('inner_ere_final_fraction',.5),('inner_ere_min_rounds',2)]:
        assert identity({**cfg,key:value})!=identity(cfg)


def test_trace_catalog_and_training_logging(models):
    model=models()
    model.agent.act(torch.zeros(3),collect_diagnostics=True)
    model._record_action_metrics(planned=True,action_seconds=0.)
    name='inner_critic_replay_round_1_sample_count'
    catalog=metric_catalog([name,'critic_replay_window_fraction','actor_replay_round_age_mean'])
    assert catalog[name]['sampling_phase']=='post_decision'
    assert catalog['critic_replay_window_fraction']['preferred_axis']=='critic_updates'
    assert catalog['actor_replay_round_age_mean']['preferred_axis']=='actor_updates'
    summary=model._wandb_train_window.snapshot()
    assert 'train/'+name in summary
    assert 'train/inner_critic_replay_window_fraction' in summary
    assert 'train/inner_actor_replay_round_age_mean' in summary


def test_rounding_boundary_and_all_depths_accessible():
    assert round_windows(8,2,math.nextafter(.25,1),1)==[8,3]
    assert round_windows(8,2,math.nextafter(.25,0),1)==[8,2]
    replay=LatentReplayBuffer(12,1,1,'cpu')
    _append(replay,0,12)
    # Enumerate the last two complete N=2,H=2 rounds without replacement.
    ids=replay.draw_recent_indices(8,8,replacement=False,generator=torch.Generator().manual_seed(12))
    batch=replay.sample(8,indices=ids)
    assert sorted(batch['sample_ids'].tolist())==list(range(4,12))
    depths=batch['sample_ids'].remainder(4)//2
    assert depths.bincount().tolist()==[4,4]


@pytest.mark.parametrize('options', [dict(inner_ere_final_fraction=1.),dict(inner_ere_min_rounds=3)])
def test_full_phase_preserves_matrix_rng_call_shape(models,monkeypatch,options):
    model=models(**options)
    engine=model.agent.inner_engine
    with engine.rng.fork('initialization'): engine._prepare_workspace(t0=True)
    replay=engine.state.replay
    z=torch.zeros(12,model.cfg.latent_dim)
    replay.add_batch(z,torch.zeros(12,1),torch.zeros(12,1),z,torch.zeros(12,1))
    engine.state.replay_rounds=[(1,0,4),(2,4,8),(3,8,12)]
    calls=[]
    original=torch.randint
    def draw(*args,**kwargs):
        calls.append(args)
        return original(*args,**kwargs)
    monkeypatch.setattr(torch,'randint',draw)
    engine._draw_update_indices(4)
    assert calls==[(12,(4,4))]


@pytest.mark.parametrize('rounds', [1,3])
@pytest.mark.parametrize('critic,actor', [(0,0),(0,1),(1,0),(1,1)])
def test_zero_single_component_slots_are_uniform(models,rounds,critic,actor):
    options=dict(inner_rounds=rounds,inner_critic_updates_per_round=critic,inner_actor_updates_per_round=actor)
    ere=models(**options)
    uniform=models(inner_replay_strategy='uniform',**options)
    actual=ere.agent.act(torch.zeros(3),collect_diagnostics=False,trace=InnerActionTrace())
    expected=uniform.agent.act(torch.zeros(3),collect_diagnostics=False)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    _assert_tree_equal(_snapshot(ere.agent),_snapshot(uniform.agent))
    assert ere.agent.last_inner_metrics['inner_replay_draws']==4*rounds*(critic+actor)


@pytest.mark.parametrize('updates', [0,1])
def test_zero_single_joint_slots_are_uniform(models,updates):
    ere=models(component=False,inner_updates_per_round=updates)
    uniform=models(component=False,inner_updates_per_round=updates,inner_replay_strategy='uniform')
    actual=ere.agent.act(torch.zeros(3),collect_diagnostics=False)
    expected=uniform.agent.act(torch.zeros(3),collect_diagnostics=False)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    _assert_tree_equal(_snapshot(ere.agent),_snapshot(uniform.agent))
    assert ere.agent.last_inner_metrics['inner_replay_draws']==12*updates


def test_exact_resume_keeps_ere_semantics_without_round_tensors(models):
    source,restored,uniform=models(),models(),models(inner_replay_strategy='uniform')
    source.agent.act(torch.zeros(3),collect_diagnostics=True)
    source.agent.prepare_training_resume_boundary()
    saved=deepcopy(source.agent.training_state_dict())
    spec=saved['outer']['critic_target_spec']['inner_solve']
    assert {key:spec[key] for key in ('replay_strategy','ere_final_fraction','ere_min_rounds')}==dict(
        replay_strategy='ere',ere_final_fraction=.25,ere_min_rounds=1)
    def check_no_round_tensors(value):
        if isinstance(value,dict):
            assert not {'replay_rounds','replay_round_counts'} & value.keys()
            for child in value.values(): check_no_round_tensors(child)
    check_no_round_tensors(saved)
    restored.agent.load_training_state_dict(saved)
    expected=source.agent.act(torch.zeros(3),collect_diagnostics=False)
    actual=restored.agent.act(torch.zeros(3),collect_diagnostics=False)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    _assert_tree_equal(_snapshot(restored.agent),_snapshot(source.agent))
    assert 'replay_strategy' not in uniform.agent._critic_target_spec().get('inner_solve',{})
    for target in (uniform,models(inner_ere_final_fraction=.5),models(inner_ere_min_rounds=2)):
        target.agent.prepare_training_resume_boundary()
        before=deepcopy(target.agent.model.state_dict())
        with pytest.raises(ValueError,match='critic-target specification'):
            target.agent.load_training_state_dict(saved)
        _assert_tree_equal(target.agent.model.state_dict(),before)


@pytest.mark.parametrize('order', ['critic_first','interleaved'])
@pytest.mark.parametrize('ere_actor', [True, False])
def test_strict_graphs_reused_as_windows_narrow(models,monkeypatch,order,ere_actor):
    torch._dynamo.reset()
    graphs=[]
    real_compile=torch.compile
    def backend(graph,inputs):
        graphs.append(graph)
        return graph.forward
    def compile_counted(function,**kwargs):
        assert kwargs['fullgraph']
        return real_compile(function,backend=backend,**kwargs)
    monkeypatch.setattr(torch,'compile',compile_counted)
    model=models(compile=True,compile_strict=True,inner_finite_horizon=True,
                 inner_ere_actor=ere_actor,
                 inner_component_update_order=order)
    try:
        model.agent.act(torch.zeros(3),collect_diagnostics=False,trace=InnerActionTrace())
        count=len(graphs)
        assert count>0
        for _ in range(2):
            model.agent.act(torch.zeros(3),collect_diagnostics=False,trace=InnerActionTrace())
            assert len(graphs)==count
        assert model.agent.last_inner_metrics['inner_compile_fallback']==0
    finally:
        torch._dynamo.reset()


def test_inductor_ere_solve_matches_eager(models):
    torch._dynamo.reset()
    options=dict(inner_finite_horizon=True,inner_component_update_order='interleaved')
    eager=models(**options)
    compiled=models(compile=True,compile_strict=True,**options)
    try:
        for _ in range(2):
            expected=eager.agent.act(torch.zeros(3),collect_diagnostics=False)
            actual=compiled.agent.act(torch.zeros(3),collect_diagnostics=False)
            torch.testing.assert_close(actual,expected,rtol=1e-4,atol=3e-6)
            for name in ('actor','critic','critic_target'):
                a=getattr(compiled.agent.inner_engine._action_pool,name)
                b=getattr(eager.agent.inner_engine._action_pool,name)
                for left,right in zip(a.parameters(),b.parameters()):
                    torch.testing.assert_close(left,right,rtol=1e-4,atol=3e-6)
            _assert_tree_equal(compiled.agent.inner_engine.rng.training_state_dict(),
                               eager.agent.inner_engine.rng.training_state_dict())
        assert compiled.agent.last_inner_metrics['inner_compile_fallback']==0
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize('order', ['critic_first', 'interleaved'])
@pytest.mark.parametrize('sampling', ['with_replacement', 'without_replacement'])
def test_critic_only_ere_actor_full_windows_and_budgets(models, monkeypatch, order, sampling):
    model = models(inner_ere_actor=False, inner_component_update_order=order,
                   inner_replay_sampling=sampling)
    engine = model.agent.inner_engine
    calls = []
    original = torch.randint
    def randint(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, 'randint', randint)
    for _ in range(2):
        trace = InnerActionTrace()
        model.agent.act(torch.zeros(3), collect_diagnostics=False, trace=trace)
        for r in range(1, 4):
            events = [e for e in trace.events if e['phase'] == 'update' and e['round_index'] == r]
            for component, expected in [('critic', round_windows(r,4,.25,1)), ('actor', [r]*2)]:
                selected = [e for e in events if e['updated_'+component]]
                assert [e['metrics'][component+'_replay_window_rounds'] for e in selected] == expected
                if component == 'actor':
                    assert all(e['updated_temperature'] for e in selected)
                    assert all(e['metrics']['actor_replay_window_fraction'] == 1 for e in selected)
        metrics = model.agent.last_inner_metrics
        assert metrics['inner_critic_optimizer_steps'] == 12
        assert metrics['inner_actor_optimizer_steps'] == metrics['inner_temperature_optimizer_steps'] == 6
        assert metrics['inner_replay_draws'] == 72
        assert sum(metrics[f'inner_actor_replay_round_{r}_sample_count'] for r in range(1,4)) == 24
    if sampling == 'with_replacement':
        assert (12, (2,4)) in calls  # Historical full-phase matrix RNG draw.


def test_critic_only_ere_identity_joint_rejection_and_uniform_endpoint(models):
    with pytest.raises(ValueError, match='component'):
        _build_cfg(inner_replay_strategy='ere', inner_ere_actor=False)
    base = dict(alg='AMBITDMPC2/AMBITDMPC2', alg_params=dict(inner_replay_strategy='ere'))
    both = deepcopy(base); both['alg_params']['inner_ere_actor'] = True
    critic = deepcopy(base); critic['alg_params']['inner_ere_actor'] = False
    assert scientific_trial_parameters(base) == scientific_trial_parameters(both)
    assert scientific_trial_parameters(base) != scientific_trial_parameters(critic)
    uniform = models(inner_replay_strategy='uniform', inner_ere_actor=False)
    endpoint = models(inner_ere_final_fraction=1., inner_ere_actor=False)
    torch.testing.assert_close(uniform.agent.act(torch.zeros(3),collect_diagnostics=False),
                               endpoint.agent.act(torch.zeros(3),collect_diagnostics=False),rtol=0,atol=0)
    _assert_tree_equal(_snapshot(uniform.agent), _snapshot(endpoint.agent))
    active = models(inner_ere_actor=False)
    assert active.agent._critic_target_spec()['inner_solve']['ere_actor'] is False
