"""Weight-only diagnostic branches preserve the ordinary action-local learner."""

from copy import deepcopy

import pytest
import numpy as np
import torch

from RL.tdmpc2_core.inner_trace import InnerActionTrace
from tests.test_aux_critic_transfer import critic_params, assert_optimizer_reset
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _model_from_params
from utils.transfer_diagnostics import solve_fork


def params(objective="return", horizon=3, **overrides):
    return critic_params(objective, inner_critic_scope="action", inner_rounds=2,
                         inner_rollout_horizon=horizon, **overrides)


def changed_state(module, amount):
    result = _clone_tree(module.state_dict())
    for value in result.values():
        if torch.is_floating_point(value):
            value.add_(amount)
    return result


def test_full_trace_probes_preserve_solve_weights_optimizers_and_rng():
    model = _model_from_params(params())
    try:
        engine = model.agent.inner_engine
        observation = np.array([1., .2, -.1], dtype=np.float32)
        rng = deepcopy(engine.rng.training_state_dict())
        _, _, donor = solve_fork(model, observation, rng, capture_rounds=())
        global_rng = torch.random.get_rng_state().clone()
        records = []
        for enabled in (False, True):
            action, trace, final = solve_fork(model, observation, rng, donor=donor,
                branch="joint", full_trace_probes=enabled, probe_seed=192, probe_rollouts=4)
            records.append(dict(action=action, trace=trace, final=final,
                rng=deepcopy(engine.rng.training_state_dict()),
                optimizers={name: _clone_tree(getattr(engine._action_pool, name).state_dict())
                    for name in ("actor_optim", "critic_optim", "temperature_optim")}))
        plain, observed = records
        np.testing.assert_array_equal(plain["action"], observed["action"])
        for component in ("actor", "critic", "critic_target"):
            _assert_tree_equal(plain["final"].state_dict(component), observed["final"].state_dict(component))
        _assert_tree_equal(plain["rng"], observed["rng"])
        _assert_tree_equal(plain["optimizers"], observed["optimizers"])
        torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
        assert not any(event["phase"] == "transfer_probe" for event in plain["trace"].events)
        assert any("togo_return_mean" in event["metrics"] for event in observed["trace"].events)
    finally:
        model.close()


@pytest.mark.parametrize("objective", ["return", "soft"])
@pytest.mark.parametrize("horizon", [1, 3, 5])
def test_network_snapshots_are_observational_and_have_exact_boundaries(objective, horizon):
    plain = _model_from_params(params(objective, horizon))
    observed = _model_from_params(params(objective, horizon))
    try:
        observed.agent.model.load_state_dict(plain.agent.model.state_dict())
        global_rng = torch.random.get_rng_state().clone()
        outer = _clone_tree(observed.agent.model.state_dict())
        trace = InnerActionTrace(capture_learners=True)
        x = torch.tensor([1., .2, -.1])
        expected = plain.agent.act(x, t0=True, eval_mode=True)
        actual = observed.agent.act(x, t0=True, eval_mode=True, trace=trace)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        left, right = plain.agent.inner_engine, observed.agent.inner_engine
        _assert_tree_equal(left.rng.training_state_dict(), right.rng.training_state_dict())
        for component in ("actor", "critic", "critic_target"):
            a, b = getattr(left._action_pool, component), getattr(right._action_pool, component)
            _assert_tree_equal(a.state_dict(), b.state_dict())
            assert [m.training for m in a.modules()] == [m.training for m in b.modules()]
        for component in ("actor_optim", "critic_optim", "temperature_optim"):
            _assert_tree_equal(getattr(left._action_pool, component).state_dict(),
                               getattr(right._action_pool, component).state_dict())
        snapshots = trace.learner_snapshots
        assert [(s.stage, s.round_index, s.critic_updates, s.actor_updates) for s in snapshots] == [
            ("initial", 0, 0, 0), ("after_first_critic_block", 1, 2, 0),
            ("after_first_actor_block", 1, 2, 2), ("post_round", 1, 2, 2),
            ("post_round", 2, 4, 4), ("pre_reset", 2, 4, 4),
        ]
        for component in ("actor", "critic", "critic_target"):
            _assert_tree_equal(snapshots[-1].state_dict(component),
                               getattr(right._action_pool, component).state_dict())
        assert snapshots[0].alpha > 0
        assert snapshots[0].policy_bounds == trace._policy_bounds(right.cfg)
        exported = snapshots[-1].make_module("actor")
        with torch.no_grad():
            next(exported.parameters()).add_(1.)
        _assert_tree_equal(snapshots[-1].state_dict("actor"), right._action_pool.actor.state_dict())
        _assert_tree_equal(observed.agent.model.state_dict(), outer)
        torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
    finally:
        plain.close()
        observed.close()


@pytest.mark.parametrize("carry_actor,carry_critic,target_mode", [
    (False, False, None), (True, False, None), (False, True, None),
    (True, True, None), (True, True, "online"), (True, True, "donor"),
])
def test_independent_donors_leave_replay_adam_temperature_and_prior_fresh(
    carry_actor, carry_critic, target_mode,
):
    model = _model_from_params(params())
    try:
        engine = model.agent.inner_engine
        actor = changed_state(engine._actor_base, .03)
        critic = changed_state(engine._critic_base, .05)
        target = changed_state(engine._critic_base, .08)
        outer = _clone_tree(model.agent.checkpoint_state())
        prepare = engine._apply_diagnostic_initialization
        checked = []

        def inspect_initialization():
            prepare()
            state = engine.state
            assert state.replay.size == 0
            assert state.actor_steps == state.critic_steps == state.temperature_steps == 0
            assert_optimizer_reset(state.actor_optim, state.actor.parameters())
            assert_optimizer_reset(state.critic_optim, state.critic.parameters())
            assert_optimizer_reset(state.temperature_optim, [state.log_alpha])
            torch.testing.assert_close(engine.alpha, engine._initial_inner_alpha(), rtol=0, atol=0)
            checked.append(True)

        engine._apply_diagnostic_initialization = inspect_initialization
        trace = InnerActionTrace(capture_learners=True)
        selected_target = target if target_mode == "donor" else target_mode
        with engine.diagnostic_initialization(
            actor=actor if carry_actor else None, critic=critic if carry_critic else None,
            target=selected_target,
        ):
            # The caller cannot mutate an already installed donor by accident.
            actor_before = deepcopy(actor)
            if carry_actor:
                for value in actor.values():
                    value.zero_()
            model.agent.act(torch.ones(3), t0=True, eval_mode=True, trace=trace)
        initial = trace.learner_snapshots[0]
        _assert_tree_equal(initial.state_dict("actor"),
                           actor_before if carry_actor else engine._actor_base.state_dict())
        _assert_tree_equal(initial.state_dict("critic"),
                           critic if carry_critic else engine._critic_base.state_dict())
        expected_target = (target if target_mode == "donor" else critic
                           if target_mode == "online" else engine._critic_base.state_dict())
        _assert_tree_equal(initial.state_dict("critic_target"), expected_target)
        fresh = InnerActionTrace(capture_learners=True)
        model.agent.act(torch.ones(3), t0=False, eval_mode=True, trace=fresh)
        for component, base in (("actor", engine._actor_base), ("critic", engine._critic_base),
                                ("critic_target", engine._critic_base)):
            _assert_tree_equal(fresh.learner_snapshots[0].state_dict(component), base.state_dict())
        assert len(checked) == 2
        assert engine._diagnostic_initialization is engine._diagnostic_collection_actor is None
        _assert_tree_equal(model.agent.checkpoint_state(), outer)
    finally:
        model.close()


@pytest.mark.parametrize("objective", ["return", "soft"])
def test_fixed_collection_actor_pairs_data_without_replacing_update_actor(objective):
    plain = _model_from_params(params(objective))
    shifted = _model_from_params(params(objective))
    try:
        shifted.agent.model.load_state_dict(plain.agent.model.state_dict())
        traces = []
        for model, carry in ((plain, False), (shifted, True)):
            engine = model.agent.inner_engine
            donor = changed_state(engine._actor_base, .1) if carry else None
            trace = InnerActionTrace(capture_learners=True)
            with engine.diagnostic_initialization(actor=donor, collection_actor="prior"):
                model.agent.act(torch.ones(3), t0=True, eval_mode=True, trace=trace)
            traces.append(trace)
            if carry:
                _assert_tree_equal(trace.learner_snapshots[0].state_dict("actor"), donor)
        hashes = [[e["replay_sha256"] for e in t.events if e["phase"] == "collection"]
                  for t in traces]
        assert len(hashes[0]) == 2
        assert hashes[0] == hashes[1]
        assert any(not torch.equal(value, traces[1].learner_snapshots[-1].state_dict("actor")[key])
                   for key, value in traces[0].learner_snapshots[-1].state_dict("actor").items())
        _assert_tree_equal(plain.agent.inner_engine.rng.training_state_dict(),
                           shifted.agent.inner_engine.rng.training_state_dict())
    finally:
        plain.close()
        shifted.close()


def test_invalid_donor_is_atomic_and_context_cleans_up_on_failure():
    model = _model_from_params(params())
    try:
        engine = model.agent.inner_engine
        actor = changed_state(engine._actor_base, .03)
        invalid = changed_state(engine._critic_base, .04)
        invalid.pop(next(iter(invalid)))
        trace = InnerActionTrace(capture_learners=True)
        with pytest.raises(ValueError, match="incompatible"):
            with engine.diagnostic_initialization(actor=actor, critic=invalid):
                model.agent.act(torch.ones(3), t0=True, eval_mode=True, trace=trace)
        _assert_tree_equal(engine.state.actor.state_dict(), engine._actor_base.state_dict())
        assert trace.events == trace.learner_snapshots == []
        assert engine._diagnostic_initialization is engine._diagnostic_collection_actor is None
        with engine.diagnostic_initialization():
            with pytest.raises(ValueError, match="frozen evaluation"):
                model.agent.act(torch.ones(3), t0=True, eval_mode=False)
        model.agent.act(torch.ones(3), t0=True, eval_mode=True)
    finally:
        model.close()


def test_snapshot_exports_actual_scale_and_sparse_round_selection_keeps_boundaries():
    model = _model_from_params(params(aux_return_sac_actor_loss_scale_mode="tdmpc2_percentile_range",
                                     sac_actor_loss_scale_mode="none", inner_temperature_mode="fixed"))
    try:
        engine = model.agent.inner_engine
        model.agent.aux_return.actor_loss_scale.fill_(7.)
        trace = InnerActionTrace(capture_learners=True, learner_rounds=[1])
        model.agent.act(torch.ones(3), t0=True, eval_mode=True, trace=trace)
        assert [s.stage for s in trace.learner_snapshots] == [
            "initial", "after_first_critic_block", "after_first_actor_block", "post_round", "pre_reset",
        ]
        assert all(s.actor_loss_scale == 7. for s in trace.learner_snapshots)
        assert trace.learner_snapshots[-1].round_index == 2
        assert model.agent.aux_return.actor_loss_scale.item() == 7.
        initial = trace.events[0]
        assert initial["replay_size"] == 0
        assert all(initial["metrics"][f"{component}_optimizer_steps_initial"] == 0
                   for component in ("actor", "critic", "temperature"))
    finally:
        model.close()
