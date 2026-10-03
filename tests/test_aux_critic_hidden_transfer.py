"""Trainable critic bodies survive solves; each solve gets a fresh output head."""

import math

import pytest
import torch

from tests.test_actor_transfer_solve_cadence import learner_snapshot
from tests.test_aux_actor_transfer import _trace
from tests.test_aux_critic_transfer import (
    assert_optimizer_reset, component_ids, critic_params,
)
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _build_cfg, _model_from_params


def hidden_params(critic="return", **overrides):
    return critic_params(critic, inner_critic_transfer_head="random", **overrides)


def body_state(critic):
    # Include normalization parameters and buffers, excluding only final Linear.
    return [{key: value.detach().clone() for key, value in q[:-1].state_dict().items()}
            for q in critic]


def head_state(critic):
    return [_clone_tree(q[-1].state_dict()) for q in critic]


@pytest.mark.parametrize("critic", ["soft", "return"])
@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_body_carries_exactly_and_every_head_is_fresh_before_target_copy(critic, representation):
    model = _model_from_params(hidden_params(
        critic, q_representation=representation,
        num_q=5 if representation == "distributional" else 2,
    ))
    try:
        engine = model.agent.inner_engine
        original_prepare = engine._prepare_workspace
        previous_body = previous_heads = None
        identities, parameter_ids, initial_heads = [], [], []
        episode_solves = 0

        def checked_prepare(*, t0):
            nonlocal episode_solves
            original_prepare(t0=t0)
            state = engine.state
            if t0:
                episode_solves = 0
            expected = body_state(engine._critic_base) if t0 else previous_body
            _assert_tree_equal(body_state(state.critic), expected)
            _assert_tree_equal(state.critic_target.state_dict(), state.critic.state_dict())
            _assert_tree_equal(state.actor.state_dict(), engine._actor_base.state_dict())
            for member, base in zip(state.critic, engine._critic_base):
                head = member[-1]
                bound = math.sqrt(6. / (head.in_features + head.out_features))
                assert torch.count_nonzero(head.weight) > 0
                assert head.weight.abs().max() <= bound
                assert torch.count_nonzero(head.bias) == 0
                assert not torch.equal(head.weight, base[-1].weight)
            if previous_heads is not None:
                assert all(not torch.equal(q[-1].weight, old["weight"])
                           for q, old in zip(state.critic, previous_heads))
            assert all(p.requires_grad for p in state.critic.parameters())
            assert all(not p.requires_grad for p in state.critic_target.parameters())
            assert state.replay.size == 0
            assert state.critic_lifetime_steps == episode_solves * 2
            assert state.critic_steps == state.actor_steps == state.temperature_steps == 0
            assert_optimizer_reset(state.actor_optim, state.actor.parameters())
            assert_optimizer_reset(state.critic_optim, state.critic.parameters())
            assert_optimizer_reset(state.temperature_optim, [state.log_alpha])
            torch.testing.assert_close(engine.alpha, engine._initial_inner_alpha(), rtol=0, atol=0)
            identities.append(component_ids(state))
            parameter_ids.append(tuple(id(p) for p in state.critic.parameters()))
            initial_heads.append(head_state(state.critic))
            episode_solves += 1

        engine._prepare_workspace = checked_prepare
        outer = _clone_tree(model.agent.checkpoint_state())
        global_rng = torch.random.get_rng_state().clone()
        for t0 in (True, False, False, True):
            trace = _trace(2)
            model.agent.act(torch.tensor([1., .2, -.1]), t0=t0, eval_mode=True, trace=trace)
            metrics = model.agent.last_inner_metrics
            assert metrics["inner_critic_head_reinitialized"] == 1
            assert metrics["inner_critic_target_reinitialized"] == 1
            assert metrics["inner_critic_transferred"] == (not t0)
            assert metrics["inner_actor_transferred"] == 0
            assert trace.events[0]["metrics"]["inner_critic_head_reinitialized"] == 1
            assert metrics["inner_critic_optimizer_steps"] == 2
            assert metrics["inner_actor_optimizer_steps"] == 2
            previous_body = body_state(engine.state.critic)
            previous_heads = head_state(engine.state.critic)
            # The retained representation is trained, not a frozen feature extractor.
            assert any(not torch.equal(a[key], b[key]) for a, b in zip(
                previous_body, body_state(engine._critic_base)) for key in a)
            with torch.no_grad():
                for p in engine.state.critic_target.parameters():
                    p.add_(42.)
            _assert_tree_equal(model.agent.checkpoint_state(), outer)
            torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
        assert len(initial_heads) == 4
        assert all(ids == identities[0] for ids in identities)
        assert all(ids == parameter_ids[0] for ids in parameter_ids)
    finally:
        model.close()


@pytest.mark.parametrize("critic", ["soft", "return"])
@pytest.mark.parametrize("interval", [1, 2, 3])
def test_held_decisions_do_not_reset_heads_learn_or_advance_rng(critic, interval):
    model = _model_from_params(hidden_params(
        critic, inner_rollout_horizon=interval, inner_solve_interval=interval,
    ))
    try:
        engine = model.agent.inner_engine
        # H2/H3 finish in an incomplete third block.
        for decision in range(2 * interval + 1):
            solved = decision % interval == 0
            if not solved:
                before = learner_snapshot(engine)
                rng_before = _clone_tree(engine.rng.training_state_dict())
            trace = _trace(interval)
            model.agent.act(torch.tensor([1., .1 * decision, -.2]),
                            t0=decision == 0, eval_mode=True, trace=trace)
            metrics = model.agent.last_inner_metrics
            assert metrics["inner_critic_head_reinitialized"] == solved
            assert metrics["inner_critic_target_reinitialized"] == solved
            assert metrics["inner_critic_transferred"] == (solved and decision > 0)
            assert metrics["inner_solve_index"] == decision // interval
            assert metrics["inner_action_age"] == decision % interval
            assert engine.state.critic_lifetime_steps == (decision // interval + 1) * 2
            if not solved:
                assert trace.events == []
                _assert_tree_equal(learner_snapshot(engine), before)
                _assert_tree_equal(engine.rng.training_state_dict(), rng_before)
    finally:
        model.close()


@pytest.mark.parametrize("critic", ["soft", "return"])
@pytest.mark.parametrize("boundary", ["episode", "evaluation", "clear", "load"])
def test_boundaries_restore_checkpoint_body_and_reset_head(critic, boundary):
    model = _model_from_params(hidden_params(critic, inner_solve_interval=3, inner_rollout_horizon=3))
    try:
        engine = model.agent.inner_engine
        checkpoint = _clone_tree(model.agent.checkpoint_state())
        model.agent.act(torch.ones(3), t0=True, eval_mode=True)
        model.agent.act(torch.zeros(3), eval_mode=True)
        if boundary == "episode":
            engine.reset_episode()
        elif boundary == "evaluation":
            engine.reset_for_evaluation(177, reuse_action_pool=True)
        elif boundary == "clear":
            engine.clear_all()
        else:
            model.agent.load(checkpoint)
        assert engine.state.critic is engine._held_actor is None
        original_prepare = engine._prepare_workspace

        def checked_prepare(*, t0):
            original_prepare(t0=t0)
            _assert_tree_equal(body_state(engine.state.critic), body_state(engine._critic_base))
            _assert_tree_equal(engine.state.critic_target.state_dict(), engine.state.critic.state_dict())
            assert engine.state.critic_lifetime_steps == 0
            assert all(torch.count_nonzero(q[-1].bias) == 0 for q in engine.state.critic)

        engine._prepare_workspace = checked_prepare
        model.agent.act(torch.ones(3), eval_mode=True)
        assert model.agent.last_inner_metrics["inner_critic_transferred"] == 0
        assert model.agent.last_inner_metrics["inner_critic_head_reinitialized"] == 1
    finally:
        model.close()


@pytest.mark.parametrize("critic", ["soft", "return"])
def test_head_draws_reproduce_with_allocation_reuse_and_probes(critic):
    params = hidden_params(critic, inner_rollout_horizon=3, inner_solve_interval=3,
                           q_representation="distributional", num_q=5, dropout=.01)
    plain, observed = _model_from_params(params), _model_from_params(params)
    try:
        observed.agent.load(_clone_tree(plain.agent.checkpoint_state()))
        snapshots = []
        global_rng = torch.random.get_rng_state().clone()
        for _ in range(2):
            for model in (plain, observed):
                model.agent.inner_engine.reset_for_evaluation(187, reuse_action_pool=True)
            episode = []
            for decision in range(4):
                obs = torch.tensor([1., -.1 * decision, .3])
                a = plain.agent.act(obs, t0=decision == 0, eval_mode=True)
                b = observed.agent.act(obs, t0=decision == 0, eval_mode=True, trace=_trace(3))
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                left, right = plain.agent.inner_engine, observed.agent.inner_engine
                _assert_tree_equal(learner_snapshot(left), learner_snapshot(right))
                _assert_tree_equal(left.rng.training_state_dict(), right.rng.training_state_dict())
                episode.append((a.clone(), learner_snapshot(left), _clone_tree(left.rng.training_state_dict())))
            snapshots.append(episode)
        _assert_tree_equal(snapshots[0], snapshots[1])
        torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    finally:
        plain.close()
        observed.close()


@pytest.mark.parametrize("critic", ["soft", "return"])
@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_compiled_hidden_transfer_matches_eager_across_solves_and_reused_episodes(monkeypatch, critic, backend):
    original_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs:
                        original_compile(function, backend=backend, **kwargs))
    params = hidden_params(critic, inner_rollout_horizon=3, inner_solve_interval=3)
    eager = _model_from_params(params)
    compiled = _model_from_params(dict(params, compile=True, compile_strict=True))
    try:
        compiled.agent.load(_clone_tree(eager.agent.checkpoint_state()))
        for _ in range(2):
            for model in (eager, compiled):
                model.agent.inner_engine.reset_for_evaluation(191, reuse_action_pool=True)
            for decision in range(4):
                actions = [model.agent.act(torch.tensor([1., .2 * decision, -.1]),
                    t0=decision == 0, eval_mode=True, trace=_trace(3)) for model in (eager, compiled)]
                torch.testing.assert_close(actions[0], actions[1], atol=1e-5, rtol=1e-4)
                assert compiled.agent.last_inner_metrics["inner_compile_fallback"] == 0
                for a, b in zip(eager.agent.inner_engine.state.critic.parameters(),
                                compiled.agent.inner_engine.state.critic.parameters()):
                    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
                _assert_tree_equal(eager.agent.inner_engine.rng.training_state_dict(),
                                   compiled.agent.inner_engine.rng.training_state_dict())
    finally:
        eager.close()
        compiled.close()


@pytest.mark.parametrize("overrides", [
    {"aux_return_mode": "off"}, {"inner_critic_scope": "action"},
    {"inner_actor_scope": "episode"}, {"inner_critic_scope": "run"},
    {"inner_actor_optimizer_scope": "episode"}, {"inner_critic_optimizer_scope": "episode"},
    {"inner_temperature_scope": "episode"}, {"inner_replay_scope": "episode"},
    {"inner_critic_target_initialization": "outer_target"},
    {"inner_critic_adaptation": "frozen"}, {"inner_rebase_persistent": True},
    {"inner_critic_writeback_coef": .1}, {"inner_actor_writeback_coef": .1},
])
def test_random_transfer_head_rejects_unsupported_lifecycles(overrides):
    with pytest.raises(ValueError):
        _build_cfg(**hidden_params(**overrides))


@pytest.mark.parametrize("value", ["prior", "fresh", "", True, None])
def test_invalid_head_modes_rejected(value):
    params = critic_params(inner_critic_transfer_head=value)
    with pytest.raises(ValueError):
        _build_cfg(**params)
