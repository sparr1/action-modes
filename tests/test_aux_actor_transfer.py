"""Actor-only persistence, fresh critic learning, and observation-only probes."""

import pytest
import torch

from RL.tdmpc2_core.inner_trace import InnerActionTrace
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _build_cfg, _model_from_params, _tiny_params


def _params(**overrides):
    params = _tiny_params(
        aux_return_mode="sac", inner_actor_source="sac",
        inner_critic_source="aux_return", inner_horizon_actor_source="sac",
        inner_horizon_critic_source="aux_return", inner_sac_critic_target="reward_only",
        inner_actor_scope="episode", inner_rebase_persistent=False,
        inner_rounds=1, inner_first_action_rounds=3,
        inner_critic_updates_per_round=2, inner_actor_updates_per_round=2,
        inner_q_actor_reduction="mean_pair", inner_critic_target_tau=.01,
        inner_replay_capacity=18,
    )
    params.pop("inner_updates_per_round")
    params.update(overrides)
    return params


def _trace(horizon):
    return InnerActionTrace(transfer_probes=True, probes=True, probe_mode="outer_tail",
                            probe_horizon=horizon, probe_rollouts=2, probe_seed=173)


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_actor_is_only_transferred_state_and_first_dose_is_episode_local(horizon):
    model = _model_from_params(_params(inner_rollout_horizon=horizon))
    try:
        engine = model.agent.inner_engine
        original_prepare = engine._prepare_workspace
        inherited = None
        actor_identity = None
        initial_states = []

        def checked_prepare(*, t0):
            original_prepare(t0=t0)
            state = engine.state
            _assert_tree_equal(state.critic.state_dict(), engine._critic_base.state_dict())
            _assert_tree_equal(state.critic_target.state_dict(), engine._critic_base.state_dict())
            assert state.replay.size == 0
            assert state.critic_lifetime_steps == state.temperature_lifetime_steps == 0
            for optimizer in (state.actor_optim, state.critic_optim, state.temperature_optim):
                for values in optimizer.state.values():
                    for name in ("step", "exp_avg", "exp_avg_sq"):
                        assert torch.count_nonzero(values[name]) == 0
            torch.testing.assert_close(engine.alpha, engine._initial_inner_alpha())
            if t0:
                _assert_tree_equal(state.actor.state_dict(), engine._actor_base.state_dict())
                assert state.actor_lifetime_steps == 0
            else:
                _assert_tree_equal(state.actor.state_dict(), inherited)
                assert id(state.actor) == actor_identity
                assert state.actor_lifetime_steps == 6
            initial_states.append(t0)

        engine._prepare_workspace = checked_prepare
        outer = _clone_tree(model.agent.model.state_dict())
        rng = torch.random.get_rng_state().clone()
        for index, t0 in enumerate((True, False, True)):
            trace = _trace(horizon)
            model.agent.act(torch.tensor([1., .2, -.1]), t0=t0, eval_mode=True, trace=trace)
            metrics = model.agent.last_inner_metrics
            dose = 3 if t0 else 1
            assert metrics["inner_rounds"] == dose
            assert metrics["inner_model_steps_budget"] == dose * 2 * horizon
            assert metrics["inner_critic_optimizer_steps"] == dose * 2
            assert metrics["inner_actor_optimizer_steps"] == dose * 2
            assert metrics["inner_actor_transferred"] == (not t0)
            assert metrics["inner_first_action_rounds_applied"] == t0
            initial = trace.events[0]
            assert initial["metrics"]["inner_actor_transferred"] == (not t0)
            for component in ("actor", "critic", "temperature"):
                assert initial["metrics"][f"{component}_optimizer_steps_initial"] == 0
            assert initial["replay_size"] == 0
            root_events = [event for event in trace.events if event["phase"] == "transfer_probe"]
            assert [event["stage"] for event in root_events] == [
                "initial", "before_first_actor_block", "after_first_actor_block",
                *(["post_round"] * dose),
            ]
            assert root_events[1]["critic_updates"] == 2
            assert root_events[1]["actor_updates"] == 0
            assert root_events[2]["actor_updates"] == 2
            assert root_events[0]["metrics"]["probe_q_evaluations"] == 6
            assert "transfer_root_q_inner_actor_head_0" in root_events[0]["metrics"]
            assert all(event["metrics"]["probe_seconds"] >= 0 for event in root_events)
            if t0:
                assert root_events[0]["metrics"]["transfer_mean_action_delta_l2"] == 0
            inherited = _clone_tree(engine.state.actor.state_dict())
            actor_identity = id(engine.state.actor)
            assert engine.state.critic is engine.state.critic_target is None
            assert engine.state.actor_optim is engine.state.critic_optim is None
            assert engine.state.replay is engine.state.log_alpha is None
        assert initial_states == [True, False, True]
        _assert_tree_equal(model.agent.model.state_dict(), outer)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    finally:
        model.close()


@pytest.mark.parametrize("scope", ["action", "episode"])
def test_probes_preserve_actions_learning_state_and_all_rng_streams(scope):
    plain = _model_from_params(_params(inner_actor_scope=scope))
    traced = _model_from_params(_params(inner_actor_scope=scope))
    try:
        traced.agent.model.load_state_dict(plain.agent.model.state_dict())
        for t0 in (True, False, False):
            x = torch.tensor([1., -.1, .3])
            action = plain.agent.act(x, t0=t0, eval_mode=True)
            trace = _trace(2)
            observed = traced.agent.act(x, t0=t0, eval_mode=True, trace=trace)
            torch.testing.assert_close(observed, action, rtol=0, atol=0)
            left, right = plain.agent.inner_engine, traced.agent.inner_engine
            _assert_tree_equal(left.rng.training_state_dict(), right.rng.training_state_dict())
            for component in ("actor", "critic", "critic_target"):
                a = getattr(left.state, component) or getattr(left._action_pool, component)
                b = getattr(right.state, component) or getattr(right._action_pool, component)
                _assert_tree_equal(a.state_dict(), b.state_dict())
            for component in ("actor_optim", "critic_optim", "temperature_optim"):
                a, b = getattr(left._action_pool, component), getattr(right._action_pool, component)
                _assert_tree_equal(a.state_dict(), b.state_dict())
            _assert_tree_equal(left._action_pool.log_alpha, right._action_pool.log_alpha)
    finally:
        plain.close()
        traced.close()


def test_evaluation_episode_reset_reuses_allocations_but_not_actor_values():
    model = _model_from_params(_params())
    try:
        engine = model.agent.inner_engine
        actions, actor_ids, optimizer_ids = [], [], []
        for _ in range(3):
            engine.reset_for_evaluation(177, reuse_action_pool=True)
            trace = _trace(2)
            actions.append(model.agent.act(torch.ones(3), t0=True, eval_mode=True, trace=trace))
            actor_ids.append(id(engine.state.actor))
            optimizer_ids.append(id(engine._action_pool.actor_optim))
            initial = next(event for event in trace.events if event["phase"] == "transfer_probe")
            assert initial["metrics"]["transfer_mean_action_delta_l2"] == 0
            model.agent.act(torch.zeros(3), t0=False, eval_mode=True)
        assert len(set(actor_ids)) == len(set(optimizer_ids)) == 1
        for action in actions[1:]:
            torch.testing.assert_close(action, actions[0], rtol=0, atol=0)
    finally:
        model.close()


@pytest.mark.parametrize("overrides,match", [
    ({"inner_critic_scope": "episode"}, "scope"),
    ({"inner_actor_optimizer_scope": "episode"}, "scope"),
    ({"inner_temperature_scope": "episode"}, "scope"),
    ({"inner_replay_scope": "episode"}, "scope"),
    ({"inner_rebase_persistent": True}, "actor-only"),
    ({"inner_first_action_rounds": True}, "integer"),
    ({"inner_first_action_rounds": 0}, "positive"),
    ({"inner_first_action_rounds": -1}, "integer"),
    ({"inner_rounds": 0}, "positive"),
    ({"inner_replay_capacity": 8}, "capacity"),
])
def test_transfer_and_first_dose_validation(overrides, match):
    with pytest.raises(ValueError, match=match):
        _build_cfg(**_params(**overrides))


def test_first_dose_replay_capacity_defaults_to_larger_solve():
    cfg = _build_cfg(**_params(inner_replay_capacity=None))
    assert cfg.inner_replay_capacity == 12


def test_transfer_probe_decodes_every_distributional_head_and_exact_pair_minimum():
    model = _model_from_params(_params(q_representation="distributional", num_q=5))
    try:
        engine = model.agent.inner_engine
        with torch.no_grad():
            for index, head in enumerate(engine._horizon_critic):
                head[-1].weight.zero_()
                head[-1].bias.zero_()
                head[-1].bias[-1] = index + 1
        trace = _trace(2)
        model.agent.act(torch.zeros(3), t0=True, eval_mode=True, trace=trace)
        event = next(event for event in trace.events if event["phase"] == "transfer_probe")
        metrics = event["metrics"]
        values = torch.tensor([metrics[f"transfer_root_q_frozen_actor_head_{i}"] for i in range(5)])
        expected_pair_min = torch.stack([torch.minimum(values[i], values[j])
                                         for i in range(5) for j in range(i + 1, 5)]).mean()
        assert metrics["transfer_root_q_frozen_actor_mean_all"] == pytest.approx(values.mean().item())
        assert metrics["transfer_root_q_frozen_actor_min_all"] == pytest.approx(values.min().item())
        assert metrics["transfer_root_q_frozen_actor_expected_min_pair"] == pytest.approx(expected_pair_min.item())
    finally:
        model.close()


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_compilation_matches_transfer_across_decisions_and_episode_resets(monkeypatch, backend):
    original_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs:
                        original_compile(function, backend=backend, **kwargs))
    eager = _model_from_params(_params())
    compiled = _model_from_params(_params(compile=True, compile_strict=True))
    try:
        compiled.agent.model.load_state_dict(eager.agent.model.state_dict())
        for episode in range(2):
            for model in (eager, compiled):
                model.agent.inner_engine.reset_for_evaluation(170 + episode, reuse_action_pool=True)
            for t0 in (True, False):
                actions = [model.agent.act(torch.ones(3), t0=t0, eval_mode=True,
                                           trace=_trace(2)) for model in (eager, compiled)]
                torch.testing.assert_close(actions[0], actions[1], atol=1e-5, rtol=1e-4)
                assert compiled.agent.last_inner_metrics["inner_compile_fallback"] == 0
                for a, b in zip(eager.agent.inner_engine.state.actor.parameters(),
                                compiled.agent.inner_engine.state.actor.parameters()):
                    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
    finally:
        eager.close()
        compiled.close()
