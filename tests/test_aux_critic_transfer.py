"""Critic-only warm starts retain values, not stale targets or optimizer state."""

import pytest
import torch

from tests.test_actor_transfer_solve_cadence import learner_snapshot
from tests.test_aux_actor_transfer import _params as actor_params, _trace
from tests.test_ambi_inner_decoupling import _assert_tree_equal, _clone_tree
from tests.test_ambi_root_local_sac import _build_cfg, _model_from_params


def critic_params(critic="return", **overrides):
    source = "aux_return" if critic == "return" else "sac"
    settings = dict(
        inner_actor_scope="action", inner_critic_scope="episode",
        inner_critic_source=source, inner_horizon_critic_source=source,
        inner_sac_critic_target="reward_only" if critic == "return" else "entropy_augmented",
        inner_finite_horizon=True,
        inner_terminal_entropy="none" if critic == "return" else "outer",
        inner_critic_target_initialization="online", inner_rebase_persistent=False,
        inner_first_action_rounds=None, inner_rounds=1, inner_replay_capacity=None,
    )
    settings.update(overrides)
    return actor_params(**settings)


def assert_optimizer_reset(optimizer, parameters):
    actual = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    expected = list(parameters)
    assert len(actual) == len(expected)
    assert all(left is right for left, right in zip(actual, expected))
    for values in optimizer.state.values():
        for name in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.count_nonzero(values[name]) == 0


def component_ids(state):
    return tuple(id(getattr(state, name)) for name in (
        "actor", "critic", "critic_target", "actor_optim", "critic_optim",
        "temperature_optim", "log_alpha", "replay",
    ))


@pytest.mark.parametrize("critic", ["soft", "return"])
@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_only_online_critic_survives_each_solve_and_target_starts_from_it(critic, horizon):
    model = _model_from_params(critic_params(critic, inner_rollout_horizon=horizon))
    try:
        engine = model.agent.inner_engine
        prepare = engine._prepare_workspace
        previous_critic = None
        prepare_count = 0
        identities = []

        def checked_prepare(*, t0):
            nonlocal prepare_count
            prepare(t0=t0)
            state = engine.state
            expected = engine._critic_base.state_dict() if t0 else previous_critic
            _assert_tree_equal(state.critic.state_dict(), expected)
            _assert_tree_equal(state.critic_target.state_dict(), expected)
            _assert_tree_equal(state.actor.state_dict(), engine._actor_base.state_dict())
            assert state.critic_lifetime_steps == (0 if t0 else 2 * prepare_count)
            assert state.actor_lifetime_steps == state.temperature_lifetime_steps == 0
            assert state.replay.size == 0
            torch.testing.assert_close(engine.alpha, engine._initial_inner_alpha(), rtol=0, atol=0)
            assert_optimizer_reset(state.actor_optim, state.actor.parameters())
            assert_optimizer_reset(state.critic_optim, state.critic.parameters())
            assert_optimizer_reset(state.temperature_optim, [state.log_alpha])
            assert all(parameter.requires_grad for parameter in state.critic.parameters())
            assert all(not parameter.requires_grad for parameter in state.critic_target.parameters())
            identities.append(component_ids(state))
            prepare_count += 1

        engine._prepare_workspace = checked_prepare
        expected_base = model.agent.model._aux_return_Qs if critic == "return" else model.agent.model._Qs
        assert engine._critic_base is engine._horizon_critic is expected_base
        outer = _clone_tree(model.agent.checkpoint_state())
        global_rng = torch.random.get_rng_state().clone()
        for index, t0 in enumerate((True, False, False, True)):
            trace = _trace(horizon)
            model.agent.act(torch.tensor([1., .2, -.1]), t0=t0, eval_mode=True, trace=trace)
            metrics = model.agent.last_inner_metrics
            assert metrics["inner_rounds"] == 1
            assert metrics["inner_model_steps_budget"] == 2 * horizon
            assert metrics["inner_critic_optimizer_steps"] == metrics["inner_actor_optimizer_steps"] == 2
            assert metrics["inner_actor_transferred"] == 0
            assert metrics["inner_critic_transferred"] == (not t0)
            assert metrics["inner_critic_target_reinitialized"] == 1
            assert metrics["inner_critic_updates_initial"] == (0 if t0 else 2 * index)
            initial = trace.events[0]
            assert initial["phase"] == "initial" and initial["replay_size"] == 0
            for key in ("inner_critic_transferred", "inner_critic_target_reinitialized", "inner_critic_updates_initial"):
                assert initial["metrics"][key] == metrics[key]
            for name in ("actor", "critic", "temperature"):
                assert initial["metrics"][f"{name}_optimizer_steps_initial"] == 0
            assert engine.state.critic_lifetime_steps == (2 if t0 else 2 * (index + 1))
            previous_critic = _clone_tree(engine.state.critic.state_dict())
            assert any(not torch.equal(value, engine._critic_base.state_dict()[key])
                       for key, value in previous_critic.items())
            assert engine.state.actor is engine.state.actor_optim is None
            assert engine.state.critic_optim is engine.state.replay is engine.state.log_alpha is None
            # Make stale-target reuse observable even if an update happened to
            # leave the target and online critic numerically close.
            with torch.no_grad():
                for parameter in engine.state.critic_target.parameters():
                    parameter.add_(3.)
            _assert_tree_equal(model.agent.checkpoint_state(), outer)
            torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
        assert prepare_count == 4
        assert all(identity == identities[0] for identity in identities[1:])
    finally:
        model.close()


@pytest.mark.parametrize("critic", ["soft", "return"])
@pytest.mark.parametrize("interval", [1, 2, 3])
def test_critic_transfer_cadence_preserves_held_learning_and_rng(critic, interval):
    model = _model_from_params(critic_params(critic, inner_rollout_horizon=interval,
                                           inner_solve_interval=interval))
    try:
        engine = model.agent.inner_engine
        previous_critic = None
        prepare = engine._prepare_workspace
        solves = 0

        def checked_prepare(*, t0):
            nonlocal solves
            prepare(t0=t0)
            expected = engine._critic_base.state_dict() if t0 else previous_critic
            _assert_tree_equal(engine.state.critic.state_dict(), expected)
            _assert_tree_equal(engine.state.critic_target.state_dict(), expected)
            _assert_tree_equal(engine.state.actor.state_dict(), engine._actor_base.state_dict())
            assert engine.state.critic_lifetime_steps == solves * 2
            solves += 1

        engine._prepare_workspace = checked_prepare
        outer = _clone_tree(model.agent.checkpoint_state())
        global_rng = torch.random.get_rng_state().clone()
        # H=2/3 end on an incomplete third execution block.
        for decision in range(2 * interval + 1):
            held = decision % interval != 0
            observation = torch.tensor([1. + .13 * decision, -.2 * decision, .4])
            if held:
                learning_before = learner_snapshot(engine)
                counters_before = (engine.state.critic_lifetime_steps, engine.state.critic_steps)
                rng_before = _clone_tree(engine.rng.training_state_dict())
                actor = engine._held_actor
                with torch.no_grad():
                    root = model.agent.model.encode(observation.unsqueeze(0))
                    expected_action = model.agent.model.policy_stats(
                        root, policy=actor, log_std_mapping=model.cfg.inner_log_std_mapping,
                        log_std_min=model.cfg.inner_log_std_min,
                        log_std_max=model.cfg.inner_log_std_max,
                    )["mean"][0]
            trace = _trace(interval)
            action = model.agent.act(observation, t0=decision == 0, eval_mode=True, trace=trace)
            metrics = model.agent.last_inner_metrics
            assert metrics["inner_solve_performed"] == (not held)
            assert metrics["inner_policy_held"] == held
            assert metrics["inner_solve_index"] == decision // interval
            assert metrics["inner_action_age"] == decision % interval
            assert metrics["inner_episode_decision_index"] == decision
            assert metrics["inner_actor_transferred"] == 0
            assert metrics["inner_critic_transferred"] == (not held and decision > 0)
            assert metrics["inner_critic_target_reinitialized"] == (not held)
            if held:
                torch.testing.assert_close(action, expected_action.cpu(), rtol=0, atol=0)
                assert engine._held_actor is actor and trace.events == []
                _assert_tree_equal(learner_snapshot(engine), learning_before)
                _assert_tree_equal(engine.rng.training_state_dict(), rng_before)
                assert (engine.state.critic_lifetime_steps, engine.state.critic_steps) == counters_before
                for key in ("inner_rounds", "inner_model_steps", "inner_replay_draws",
                            "inner_critic_optimizer_steps", "inner_actor_optimizer_steps",
                            "inner_temperature_optimizer_steps"):
                    assert metrics[key] == 0
            else:
                assert metrics["inner_critic_updates_initial"] == 2 * (solves - 1)
                assert metrics["inner_critic_optimizer_steps"] == metrics["inner_actor_optimizer_steps"] == 2
                previous_critic = _clone_tree(engine.state.critic.state_dict())
            _assert_tree_equal(model.agent.checkpoint_state(), outer)
            torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
        assert solves == 3
    finally:
        model.close()


@pytest.mark.parametrize("critic", ["soft", "return"])
@pytest.mark.parametrize("boundary", ["t0", "episode", "evaluation", "clear", "load"])
def test_episode_and_checkpoint_boundaries_discard_carried_critic(critic, boundary):
    model = _model_from_params(critic_params(critic, inner_solve_interval=3, inner_rollout_horizon=3))
    try:
        engine = model.agent.inner_engine
        checkpoint = _clone_tree(model.agent.checkpoint_state())
        model.agent.act(torch.ones(3), t0=True, eval_mode=True)
        model.agent.act(torch.zeros(3), eval_mode=True)
        assert engine.state.critic is not None and engine._held_actor is not None
        if boundary == "episode":
            engine.reset_episode()
        elif boundary == "evaluation":
            engine.reset_for_evaluation(177, reuse_action_pool=True)
        elif boundary == "clear":
            engine.clear_all()
        elif boundary == "load":
            model.agent.load(checkpoint)
        if boundary != "t0":
            assert engine.state.critic is engine._held_actor is None
            assert engine._episode_decision_index == 0
        prepare = engine._prepare_workspace

        def checked_prepare(*, t0):
            prepare(t0=t0)
            _assert_tree_equal(engine.state.critic.state_dict(), engine._critic_base.state_dict())
            _assert_tree_equal(engine.state.critic_target.state_dict(), engine._critic_base.state_dict())
            assert engine.state.critic_lifetime_steps == 0

        engine._prepare_workspace = checked_prepare
        model.agent.act(torch.ones(3), t0=boundary == "t0", eval_mode=True, trace=_trace(3))
        metrics = model.agent.last_inner_metrics
        assert metrics["inner_solve_performed"] == metrics["inner_critic_target_reinitialized"] == 1
        assert metrics["inner_solve_index"] == metrics["inner_action_age"] == 0
        assert metrics["inner_critic_transferred"] == metrics["inner_critic_updates_initial"] == 0
    finally:
        model.close()


@pytest.mark.parametrize("critic", ["soft", "return"])
def test_evaluation_reuses_all_allocations_without_carrying_values_or_rng(critic):
    model = _model_from_params(critic_params(critic))
    try:
        engine = model.agent.inner_engine
        prepare = engine._prepare_workspace
        ids = []

        def checked_prepare(*, t0):
            prepare(t0=t0)
            ids.append(component_ids(engine.state))
            assert_optimizer_reset(engine.state.critic_optim, engine.state.critic.parameters())

        engine._prepare_workspace = checked_prepare
        first_action, first_learning, first_rng = None, None, None
        for _ in range(3):
            engine.reset_for_evaluation(177, reuse_action_pool=True)
            action = model.agent.act(torch.ones(3), t0=True, eval_mode=True)
            snapshot = learner_snapshot(engine)
            rng = _clone_tree(engine.rng.training_state_dict())
            if first_action is None:
                first_action, first_learning, first_rng = action, snapshot, rng
            else:
                torch.testing.assert_close(action, first_action, rtol=0, atol=0)
                _assert_tree_equal(snapshot, first_learning)
                _assert_tree_equal(rng, first_rng)
            model.agent.act(torch.zeros(3), eval_mode=True)
        assert all(identity == ids[0] for identity in ids[1:])
    finally:
        model.close()


@pytest.mark.parametrize("critic", ["soft", "return"])
@pytest.mark.parametrize("updates", [2, 3])
def test_target_update_clock_restarts_per_solve_while_critic_lifetime_accumulates(critic, updates):
    model = _model_from_params(critic_params(critic,
        inner_critic_updates_per_round=updates, inner_critic_target_update_interval=3))
    try:
        engine = model.agent.inner_engine
        for solve in range(3):
            starting_critic = _clone_tree(
                (engine._critic_base if solve == 0 else engine.state.critic).state_dict())
            model.agent.act(torch.tensor([1., -.1, .3]), t0=solve == 0, eval_mode=True)
            metrics = model.agent.last_inner_metrics
            assert metrics["inner_target_updates"] == updates // 3
            assert metrics["inner_critic_updates_initial"] == solve * updates
            assert engine.state.critic_lifetime_steps == (solve + 1) * updates
            if updates == 2:
                # Even the second/third solve must not use lifetime step 3/6
                # to advance a freshly copied target before its own step 3.
                _assert_tree_equal(engine.state.critic_target.state_dict(), starting_critic)
    finally:
        model.close()


@pytest.mark.parametrize("critic", ["soft", "return"])
@pytest.mark.parametrize("interval", [1, 3])
def test_critic_transfer_probes_do_not_change_controller_state_or_rng(critic, interval):
    params = critic_params(critic, inner_rollout_horizon=interval, inner_solve_interval=interval,
                           q_representation="distributional", num_q=5, dropout=.01)
    plain, observed = _model_from_params(params), _model_from_params(params)
    try:
        observed.agent.load(_clone_tree(plain.agent.checkpoint_state()))
        for model in (plain, observed):
            model.agent.inner_engine.reset_for_evaluation(187, reuse_action_pool=True)
        global_rng = torch.random.get_rng_state().clone()
        for decision in range(interval + 1):
            observation = torch.tensor([1., -.1 * decision, .3])
            plain_action = plain.agent.act(observation, t0=decision == 0, eval_mode=True)
            trace = _trace(interval)
            observed_action = observed.agent.act(observation, t0=decision == 0, eval_mode=True, trace=trace)
            torch.testing.assert_close(observed_action, plain_action, rtol=0, atol=0)
            _assert_tree_equal(learner_snapshot(plain.agent.inner_engine), learner_snapshot(observed.agent.inner_engine))
            _assert_tree_equal(plain.agent.inner_engine.rng.training_state_dict(), observed.agent.inner_engine.rng.training_state_dict())
            _assert_tree_equal(plain.agent.checkpoint_state(), observed.agent.checkpoint_state())
            torch.testing.assert_close(torch.random.get_rng_state(), global_rng, rtol=0, atol=0)
            if decision % interval == 0:
                probes = [event for event in trace.events if event["phase"] == "transfer_probe"]
                assert [event["stage"] for event in probes] == [
                    "initial", "before_first_actor_block", "after_first_actor_block", "post_round",
                ]
                assert probes[0]["metrics"]["transfer_mean_action_delta_l2"] == 0
            else:
                assert trace.events == []
    finally:
        plain.close()
        observed.close()


@pytest.mark.parametrize("critic", ["soft", "return"])
@pytest.mark.parametrize("interval", [1, 3])
@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_compiled_critic_transfer_matches_eager_across_solves_and_episodes(monkeypatch, critic, interval, backend):
    compile_original = torch.compile
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs:
                        compile_original(function, backend=backend, **kwargs))
    params = critic_params(critic, inner_rollout_horizon=interval, inner_solve_interval=interval)
    eager = _model_from_params(params)
    compiled = _model_from_params(dict(params, compile=True, compile_strict=True))
    try:
        compiled.agent.load(_clone_tree(eager.agent.checkpoint_state()))
        for episode in range(2):
            for model in (eager, compiled):
                model.agent.inner_engine.reset_for_evaluation(191 + episode, reuse_action_pool=True)
            for decision in range(interval + 1):
                actions = [model.agent.act(torch.tensor([1., .2 * decision, -.1]),
                    t0=decision == 0, eval_mode=True, trace=_trace(interval)) for model in (eager, compiled)]
                torch.testing.assert_close(actions[0], actions[1], atol=1e-5, rtol=1e-4)
                assert compiled.agent.last_inner_metrics["inner_compile_fallback"] == 0
                left, right = eager.agent.inner_engine, compiled.agent.inner_engine
                for component in ("critic", "critic_target"):
                    for a, b in zip(getattr(left.state, component).parameters(),
                                    getattr(right.state, component).parameters()):
                        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
                assert left.state.critic_lifetime_steps == right.state.critic_lifetime_steps
                _assert_tree_equal(left.rng.training_state_dict(), right.rng.training_state_dict())
    finally:
        eager.close()
        compiled.close()


@pytest.mark.parametrize("overrides", [
    {"inner_actor_scope": "episode"}, {"inner_actor_scope": "run"},
    {"inner_critic_target_initialization": "outer_target"},
    {"inner_critic_scope": "run"}, {"inner_rebase_persistent": True},
    {"inner_actor_adaptation": "frozen"}, {"inner_critic_adaptation": "frozen"},
    {"inner_critic_adaptation": "lora_rl", "inner_critic_lora_rank": 2},
    {"inner_actor_optimizer_scope": "episode"}, {"inner_critic_optimizer_scope": "episode"},
    {"inner_temperature_optimizer_scope": "episode"}, {"inner_temperature_scope": "episode"},
    {"inner_replay_scope": "episode"}, {"inner_actor_writeback_coef": .1},
    {"inner_critic_writeback_coef": .1},
])
def test_critic_only_transfer_rejects_unsupported_persistence_and_writeback(overrides):
    with pytest.raises(ValueError):
        _build_cfg(**critic_params(**overrides))


def test_generic_persistent_critic_keeps_existing_target_semantics():
    model = _model_from_params(critic_params("soft", aux_return_mode="off"))
    try:
        engine = model.agent.inner_engine
        with engine.rng.fork("initialization"):
            engine._prepare_workspace(t0=True)
        with torch.no_grad():
            for parameter in engine.state.critic_target.parameters():
                parameter.add_(3.)
        previous_target = _clone_tree(engine.state.critic_target.state_dict())
        with engine.rng.fork("initialization"):
            engine._prepare_workspace(t0=False)
        _assert_tree_equal(engine.state.critic_target.state_dict(), previous_target)
    finally:
        model.close()
