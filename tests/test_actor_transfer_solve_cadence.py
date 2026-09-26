"""Frozen feedback execution between solves, resets, and interval-one parity."""

import ast
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
from types import MethodType

import pytest
import torch

from RL.tdmpc2_core import inner_improvement
from tests.test_aux_actor_transfer import _params, _trace
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _build_cfg, _model_from_params


def cadence_params(interval=3, scope="episode", **overrides):
    return _params(inner_actor_scope=scope, inner_solve_interval=interval,
                   inner_rollout_horizon=interval, inner_first_action_rounds=None,
                   inner_rounds=2, inner_replay_capacity=None, **overrides)


def learner_snapshot(engine):
    snapshot = {}
    for name in ("actor", "critic", "critic_target", "actor_optim", "critic_optim", "temperature_optim"):
        component = getattr(engine.state, name) or getattr(engine._action_pool, name)
        snapshot[name] = _clone_tree(component.state_dict())
    snapshot['alpha'] = _clone_tree(engine._action_pool.log_alpha)
    replay = engine._action_pool.replay
    snapshot['replay'] = _clone_tree(replay.state_dict())
    return snapshot


@pytest.mark.parametrize("scope", ["action", "episode"])
@pytest.mark.parametrize("interval", [2, 3])
def test_holds_execute_fixed_actor_at_fresh_observations_without_learning(scope, interval):
    model = _model_from_params(cadence_params(interval, scope))
    try:
        engine = model.agent.inner_engine
        prepare = engine._prepare_workspace
        previous_actor = None
        solve_count = 0

        def checked_prepare(*, t0):
            nonlocal solve_count
            prepare(t0=t0)
            state = engine.state
            expected = engine._actor_base.state_dict() if scope == 'action' or t0 else previous_actor
            _assert_tree_equal(state.actor.state_dict(), expected)
            _assert_tree_equal(state.critic.state_dict(), engine._critic_base.state_dict())
            _assert_tree_equal(state.critic_target.state_dict(), engine._critic_base.state_dict())
            assert state.replay.size == 0
            assert state.actor_lifetime_steps == (solve_count * 4 if scope == 'episode' else 0)
            assert state.critic_lifetime_steps == state.temperature_lifetime_steps == 0
            for optimizer in (state.actor_optim, state.critic_optim, state.temperature_optim):
                for values in optimizer.state.values():
                    for name in ('step', 'exp_avg', 'exp_avg_sq'):
                        assert torch.count_nonzero(values[name]) == 0
            torch.testing.assert_close(engine.alpha, engine._initial_inner_alpha())
            solve_count += 1

        engine._prepare_workspace = checked_prepare
        outer = _clone_tree(model.agent.model.state_dict())
        global_rng = torch.random.get_rng_state().clone()
        previous_action = None
        changed_held_action = False
        for decision in range(2 * interval + 2):
            observation = torch.tensor([1. + .11 * decision, .2 - .13 * decision, -.1 + .17 * decision])
            hold = decision % interval != 0
            if hold:
                actor = engine._held_actor
                before = learner_snapshot(engine)
                rng_before = engine.rng.training_state_dict()
                with torch.no_grad():
                    root = model.agent.model.encode(observation.unsqueeze(0))
                    expected_action = model.agent.model.policy_stats(root, policy=actor,
                        log_std_mapping=model.cfg.inner_log_std_mapping,
                        log_std_min=model.cfg.inner_log_std_min,
                        log_std_max=model.cfg.inner_log_std_max)['mean'][0]
            trace = _trace(interval)
            actual = model.agent.act(observation, t0=decision == 0, eval_mode=True, trace=trace)
            metrics = model.agent.last_inner_metrics
            assert metrics['inner_solve_interval'] == interval
            assert metrics['inner_episode_decision_index'] == decision
            assert metrics['inner_solve_index'] == decision // interval
            assert metrics['inner_action_age'] == decision % interval
            assert metrics['inner_solve_performed'] == (not hold)
            assert metrics['inner_policy_held'] == hold
            assert metrics['inner_first_action_rounds_applied'] == 0
            assert metrics['inner_compile_fallback'] == 0
            if hold:
                torch.testing.assert_close(actual, expected_action.cpu(), rtol=0, atol=0)
                changed_held_action |= not torch.equal(actual, previous_action)
                assert engine._held_actor is actor
                _assert_tree_equal(learner_snapshot(engine), before)
                rng_after = engine.rng.training_state_dict()
                _assert_tree_equal(rng_before, rng_after)
                assert trace.events == []
                assert model.agent.last_inner_rollout_lengths == []
                for key in ('inner_rounds', 'inner_model_steps', 'inner_model_steps_budget',
                            'inner_critic_optimizer_steps', 'inner_actor_optimizer_steps',
                            'inner_temperature_optimizer_steps', 'inner_q_evaluations',
                            'inner_replay_draws', 'inner_buffer_size'):
                    assert metrics[key] == 0
                assert metrics['inner_policy_evaluations'] == 1
                assert 'inner_alpha' not in metrics
                assert not any('loss' in key or 'grad_norm' in key for key in metrics)
            else:
                assert metrics['inner_rounds'] == 2
                assert metrics['inner_model_steps'] == 4 * interval
                assert metrics['inner_actor_optimizer_steps'] == 4
                assert metrics['inner_actor_transferred'] == (scope == 'episode' and decision > 0)
                assert trace.events[0]['phase'] == 'initial'
                previous_actor = _clone_tree(engine._held_actor.state_dict())
            previous_action = actual
            _assert_tree_equal(model.agent.model.state_dict(), outer)
            torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
        assert solve_count == 3 and changed_held_action
    finally:
        model.close()


@pytest.mark.parametrize("scope", ["action", "episode"])
@pytest.mark.parametrize("reset", ['t0', 'episode', 'evaluation', 'clear'])
def test_episode_and_evaluation_boundaries_invalidate_held_policy(scope, reset):
    model = _model_from_params(cadence_params(3, scope))
    try:
        engine = model.agent.inner_engine
        model.agent.act(torch.ones(3), t0=True, eval_mode=True)
        model.agent.act(torch.zeros(3), eval_mode=True)
        assert engine._held_actor is not None and engine._episode_decision_index == 2
        if reset == 'episode': engine.reset_episode()
        elif reset == 'evaluation': engine.reset_for_evaluation(177, reuse_action_pool=True)
        elif reset == 'clear': engine.clear_all()
        if reset != 't0':
            assert engine._held_actor is None and engine._episode_decision_index == 0
        trace = _trace(3)
        model.agent.act(torch.ones(3), t0=reset == 't0', eval_mode=True, trace=trace)
        metrics = model.agent.last_inner_metrics
        assert metrics['inner_solve_performed'] == 1
        assert metrics['inner_solve_index'] == metrics['inner_action_age'] == 0
        assert metrics['inner_actor_transferred'] == 0
        initial = next(e for e in trace.events if e['phase'] == 'transfer_probe')
        assert initial['metrics']['transfer_mean_action_delta_l2'] == 0
    finally:
        model.close()


def test_held_cadence_is_rejected_for_training_before_inner_state_or_rng_changes():
    model = _model_from_params(cadence_params())
    try:
        engine = model.agent.inner_engine
        before = engine.rng.training_state_dict()
        with pytest.raises(ValueError, match='only for frozen evaluation'):
            model.agent.act(torch.ones(3), t0=True, eval_mode=False)
        assert engine.action_index == 0 and engine._held_actor is None
        _assert_tree_equal(engine.rng.training_state_dict(), before)
    finally:
        model.close()


@pytest.mark.parametrize('value', [True, 0, -1, 1.5, '3'])
def test_solve_interval_requires_positive_integer(value):
    params = cadence_params(); params['inner_solve_interval'] = value
    with pytest.raises(ValueError, match='inner_solve_interval'):
        _build_cfg(**params)


@pytest.mark.parametrize('overrides', [
    {'inner_first_action_rounds':3}, {'inner_eval_execution_action':'policy_sample'},
    {'inner_actor_scope':'run'}, {'inner_critic_scope':'episode'},
    {'inner_actor_adaptation':'frozen'}, {'inner_component_update_order':'actor_first'},
])
def test_unsupported_cadence_configurations_fail_before_execution(overrides):
    params = cadence_params(); params.update(overrides)
    with pytest.raises(ValueError):
        _build_cfg(**params)


@lru_cache(maxsize=1)
def legacy_act_method():
    """Pin reuse proof to the actual uniform-J implementation before cadence."""
    root = Path(__file__).resolve().parents[1]
    # An absolute executable and close_fds=False allow posix_spawn on macOS,
    # avoiding a fork after PyTorch has started OpenMP worker threads.
    source = subprocess.check_output([shutil.which('git'),'-C',str(root),'show',
        '9d3d82b:RL/tdmpc2_core/inner_improvement.py'], close_fds=False, text=True)
    engine_class = next(node for node in ast.parse(source).body
                        if isinstance(node,ast.ClassDef) and node.name=='InnerImprovementEngine')
    method = next(node for node in engine_class.body if isinstance(node,ast.FunctionDef) and node.name=='_act')
    module = ast.Module(body=[method],type_ignores=[])
    namespace = dict(vars(inner_improvement))
    exec(compile(ast.fix_missing_locations(module),'pinned-9d3d82b-inner-act','exec'),namespace)
    return namespace['_act']


@pytest.mark.parametrize('scope', ['action','episode'])
@pytest.mark.parametrize('rounds', [1,10])
def test_interval_one_matches_pinned_legacy_actions_learning_and_all_rng(scope, rounds):
    legacy_method = legacy_act_method()
    params = cadence_params(1,scope); params['inner_rounds'] = rounds
    # Match the study's five-head distributional route and exercise pair
    # sampling/dropout RNG, beyond the twin-scalar cadence tests above.
    params.update(q_representation='distributional',num_q=5,dropout=.01)
    current = _model_from_params(params)
    legacy = _model_from_params(params)
    try:
        legacy.agent.model.load_state_dict(current.agent.model.state_dict())
        legacy.agent.inner_engine._act = MethodType(legacy_method,legacy.agent.inner_engine)
        for episode in range(2):
            for model in (current,legacy):
                model.agent.inner_engine.reset_for_evaluation(173+episode,reuse_action_pool=True)
            for decision in range(3):
                observation = torch.tensor([1.,.2*decision,-.1])
                actions = [model.agent.act(observation,t0=decision==0,eval_mode=True,trace=_trace(1))
                           for model in (current,legacy)]
                torch.testing.assert_close(actions[0],actions[1],rtol=0,atol=0)
                left,right = current.agent.inner_engine,legacy.agent.inner_engine
                _assert_tree_equal(left.rng.training_state_dict(),right.rng.training_state_dict())
                _assert_tree_equal(learner_snapshot(left),learner_snapshot(right))
                _assert_tree_equal(current.agent.model.state_dict(),legacy.agent.model.state_dict())
                assert left._held_actor is None
                for key,value in legacy.agent.last_inner_metrics.items():
                    if not key.endswith('_seconds'):
                        assert current.agent.last_inner_metrics[key] == value, key
    finally:
        current.close(); legacy.close()


@pytest.mark.parametrize('scope',['action','episode'])
def test_compiled_dense_solve_and_eager_hold_match_eager_controller(monkeypatch,scope):
    compile_original = torch.compile
    monkeypatch.setattr(torch,'compile',lambda function,**kwargs:compile_original(function,backend='eager',**kwargs))
    eager = _model_from_params(cadence_params(3,scope))
    compiled = _model_from_params(cadence_params(3,scope,compile=True,compile_strict=True))
    try:
        compiled.agent.model.load_state_dict(eager.agent.model.state_dict())
        for episode in range(2):
            for model in (eager,compiled):
                model.agent.inner_engine.reset_for_evaluation(179+episode,reuse_action_pool=True)
            for decision in range(5):
                actions = [model.agent.act(torch.tensor([1.,.2*decision,-.1]),t0=decision==0,
                    eval_mode=True,trace=_trace(3)) for model in (eager,compiled)]
                torch.testing.assert_close(actions[0],actions[1],rtol=0,atol=0)
                assert compiled.agent.last_inner_metrics['inner_compile_fallback'] == 0
                _assert_tree_equal(eager.agent.inner_engine.rng.training_state_dict(),
                                   compiled.agent.inner_engine.rng.training_state_dict())
    finally:
        eager.close(); compiled.close()
