"""State, RNG, and ownership invariants for random-explorer inner SAC."""

from copy import deepcopy
from collections.abc import Mapping

import pytest
import torch

from tests.test_ambi_root_local_sac import _tiny_component_model, _tiny_model


def _explorer_model(mode, **overrides):
    params = {
        "inner_rounds": 1,
        "inner_rollouts_per_round": 4,
        "inner_rollout_horizon": 2,
        "inner_batch_size": 8,
        "inner_replay_capacity": 8,
        "inner_critic_updates_per_round": 1,
        "inner_actor_updates_per_round": 1,
        "inner_explorer_mode": mode,
        "inner_prior_rollout_weight": 0.5,
        "inner_sac_critic_target": "entropy_augmented",
        "inner_execution_policy_source": "primary",
    }
    params.update(overrides)
    return _tiny_component_model(**params)


def _flat_parameters(module):
    return torch.cat(
        [parameter.detach().cpu().reshape(-1) for parameter in module.parameters()]
    )


def _module_parameter_ids(module):
    return {id(parameter) for parameter in module.parameters() if parameter.requires_grad}


def _optimizer_parameter_ids(optimizer):
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def _optimizer_steps(optimizer):
    return {
        int(torch.as_tensor(parameter_state["step"]).item())
        for parameter_state in optimizer.state.values()
    }


def _assert_tree_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, Mapping):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_tree_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _assert_finite_metrics(metrics, names):
    for name in names:
        value = torch.as_tensor(metrics[name])
        assert torch.isfinite(value).all(), name


def test_frozen_explorer_keeps_exact_source_rows_and_never_changes(monkeypatch):
    model = _explorer_model("frozen_random")
    engine = model.agent.inner_engine
    before = {}
    original_collect = engine._collect_explorer_round

    def capture_before_updates(root_z):
        before["explorer"] = _flat_parameters(engine.state.explorer_actor)
        return original_collect(root_z)

    monkeypatch.setattr(engine, "_collect_explorer_round", capture_before_updates)
    try:
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        pool = engine._action_pool
        metrics = model.agent.last_inner_metrics
        sources = pool.replay.source[: pool.replay.size].reshape(-1).long()

        assert pool.replay.store_source is True
        assert torch.bincount(sources, minlength=2).tolist() == [4, 4]
        assert metrics["inner_primary_rollouts"] == 2
        assert metrics["inner_explorer_rollouts"] == 2
        assert metrics["inner_primary_transitions"] == 4
        assert metrics["inner_explorer_transitions"] == 4
        assert metrics["inner_primary_replay_fraction"] == pytest.approx(0.5)
        assert metrics["inner_explorer_replay_fraction"] == pytest.approx(0.5)
        torch.testing.assert_close(
            _flat_parameters(pool.explorer_actor),
            before["explorer"],
            rtol=0,
            atol=0,
        )
        assert not any(
            parameter.requires_grad for parameter in pool.explorer_actor.parameters()
        )
        assert pool.explorer_actor_params == []
        assert pool.explorer_actor_optim is None
        assert metrics["inner_explorer_actor_optimizer_steps"] == 0
        assert metrics["inner_explorer_critic_optimizer_steps"] == 0
        assert metrics["inner_explorer_temperature_optimizer_steps"] == 0
    finally:
        model.env.close()


def test_explorer_initialization_is_seeded_private_fresh_and_allocation_reused():
    first = _explorer_model(
        "frozen_random",
        inner_rollout_horizon=1,
        inner_replay_capacity=4,
        inner_batch_size=4,
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
        inner_execution_policy_source="mixture_sample",
    )
    second = _explorer_model(
        "frozen_random",
        inner_rollout_horizon=1,
        inner_replay_capacity=4,
        inner_batch_size=4,
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
        inner_execution_policy_source="mixture_sample",
    )
    try:
        global_state = torch.random.get_rng_state().clone()
        first_action_1 = first.agent.act(
            torch.zeros(3), eval_mode=True, collect_diagnostics=False
        )
        torch.testing.assert_close(
            torch.random.get_rng_state(), global_state, rtol=0, atol=0
        )
        second_action_1 = second.agent.act(
            torch.zeros(3), eval_mode=True, collect_diagnostics=False
        )
        torch.testing.assert_close(
            torch.random.get_rng_state(), global_state, rtol=0, atol=0
        )

        first_pool = first.agent.inner_engine._action_pool
        second_pool = second.agent.inner_engine._action_pool
        first_params_1 = _flat_parameters(first_pool.explorer_actor)
        second_params_1 = _flat_parameters(second_pool.explorer_actor)
        first_actor_identity = id(first_pool.explorer_actor)
        second_actor_identity = id(second_pool.explorer_actor)
        torch.testing.assert_close(first_params_1, second_params_1, rtol=0, atol=0)
        torch.testing.assert_close(first_action_1, second_action_1, rtol=0, atol=0)
        assert (
            first.agent.last_inner_metrics["inner_selector_primary_wins"]
            == second.agent.last_inner_metrics["inner_selector_primary_wins"]
        )

        first_action_2 = first.agent.act(
            torch.zeros(3), eval_mode=True, collect_diagnostics=False
        )
        second_action_2 = second.agent.act(
            torch.zeros(3), eval_mode=True, collect_diagnostics=False
        )
        torch.testing.assert_close(
            torch.random.get_rng_state(), global_state, rtol=0, atol=0
        )
        first_params_2 = _flat_parameters(first_pool.explorer_actor)
        second_params_2 = _flat_parameters(second_pool.explorer_actor)

        assert id(first_pool.explorer_actor) == first_actor_identity
        assert id(second_pool.explorer_actor) == second_actor_identity
        assert not torch.equal(first_params_1, first_params_2)
        torch.testing.assert_close(first_params_2, second_params_2, rtol=0, atol=0)
        torch.testing.assert_close(first_action_2, second_action_2, rtol=0, atol=0)
        assert (
            first.agent.last_inner_metrics["inner_selector_primary_wins"]
            == second.agent.last_inner_metrics["inner_selector_primary_wins"]
        )
    finally:
        first.env.close()
        second.env.close()


def test_trainable_explorer_reuses_but_resets_its_adam_state_each_root(monkeypatch):
    model = _explorer_model(
        "shared_mixture",
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=1,
    )
    engine = model.agent.inner_engine
    original_prepare = engine._prepare_explorer_workspace
    prepared_states = []

    def capture_reset_state():
        original_prepare()
        optimizer = engine.state.explorer_actor_optim
        prepared_states.append(
            {
                "steps": _optimizer_steps(optimizer),
                "moments_zero": all(
                    torch.count_nonzero(value) == 0
                    for parameter_state in optimizer.state.values()
                    for key, value in parameter_state.items()
                    if key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
                ),
            }
        )

    monkeypatch.setattr(engine, "_prepare_explorer_workspace", capture_reset_state)
    try:
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        pool = engine._action_pool
        optimizer_identity = id(pool.explorer_actor_optim)
        actor_identity = id(pool.explorer_actor)
        moment_storage = {
            (id(parameter), key): value.data_ptr()
            for parameter, parameter_state in pool.explorer_actor_optim.state.items()
            for key, value in parameter_state.items()
            if key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
        }
        assert _optimizer_steps(pool.explorer_actor_optim) == {1}

        model.agent.act(torch.zeros(3), collect_diagnostics=False)

        assert id(pool.explorer_actor_optim) == optimizer_identity
        assert id(pool.explorer_actor) == actor_identity
        assert _optimizer_steps(pool.explorer_actor_optim) == {1}
        assert prepared_states == [
            {"steps": set(), "moments_zero": True},
            {"steps": {0}, "moments_zero": True},
        ]
        assert {
            (id(parameter), key): value.data_ptr()
            for parameter, parameter_state in pool.explorer_actor_optim.state.items()
            for key, value in parameter_state.items()
            if key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
        } == moment_storage
    finally:
        model.env.close()


@pytest.mark.parametrize("estimator", ["stratified", "weighted"])
def test_shared_mixture_updates_both_actors_from_one_joint_step(
    estimator, monkeypatch
):
    model = _explorer_model(
        "shared_mixture", inner_mixture_target_estimator=estimator
    )
    engine = model.agent.inner_engine
    before = {}
    original_collect = engine._collect_explorer_round

    def capture_before_updates(root_z):
        before["primary"] = _flat_parameters(engine.state.actor)
        before["explorer"] = _flat_parameters(engine.state.explorer_actor)
        return original_collect(root_z)

    monkeypatch.setattr(engine, "_collect_explorer_round", capture_before_updates)
    outer_actor = _flat_parameters(model.agent.model._pi).clone()
    outer_critic = _flat_parameters(model.agent.model._Qs).clone()
    try:
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        pool = engine._action_pool
        metrics = model.agent.last_inner_metrics

        assert not torch.equal(before["primary"], _flat_parameters(pool.actor))
        assert not torch.equal(before["explorer"], _flat_parameters(pool.explorer_actor))
        assert _optimizer_steps(pool.actor_optim) == {1}
        assert _optimizer_steps(pool.explorer_actor_optim) == {1}
        assert _optimizer_steps(pool.critic_optim) == {1}
        assert pool.explorer_critic is None
        assert pool.explorer_critic_optim is None
        assert pool.explorer_log_alpha is None
        assert metrics["inner_actor_optimizer_steps"] == 1
        assert metrics["inner_explorer_actor_optimizer_steps"] == 1
        assert metrics["inner_critic_optimizer_steps"] == 1
        assert metrics["inner_explorer_critic_optimizer_steps"] == 0
        _assert_finite_metrics(
            metrics,
            (
                "inner_actor_loss",
                "inner_explorer_actor_grad_norm",
                "inner_mixture_log_prob",
                "inner_critic_loss",
                "inner_primary_td_error_abs_mean",
                "inner_explorer_td_error_abs_mean",
            ),
        )
        torch.testing.assert_close(
            _flat_parameters(model.agent.model._pi), outer_actor, rtol=0, atol=0
        )
        torch.testing.assert_close(
            _flat_parameters(model.agent.model._Qs), outer_critic, rtol=0, atol=0
        )
    finally:
        model.env.close()


def test_separate_critics_have_disjoint_ownership_and_one_pooled_batch(monkeypatch):
    model = _explorer_model(
        "separate_critics",
        inner_replay_sampling="without_replacement",
    )
    engine = model.agent.inner_engine
    paired_sources = []
    original_step = engine._separate_critics_step

    def capture_paired_batch(batch, **kwargs):
        paired_sources.append(batch["source"].detach().cpu().reshape(-1))
        return original_step(batch, **kwargs)

    monkeypatch.setattr(engine, "_separate_critics_step", capture_paired_batch)
    try:
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        pool = engine._action_pool
        metrics = model.agent.last_inner_metrics

        assert len(paired_sources) == 1
        assert torch.bincount(paired_sources[0].long(), minlength=2).tolist() == [4, 4]
        assert pool.replay.store_source is True
        assert pool.actor is not pool.explorer_actor
        assert pool.critic is not pool.explorer_critic
        assert pool.critic_target is not pool.explorer_critic_target
        assert pool.actor_optim is not pool.explorer_actor_optim
        assert pool.critic_optim is not pool.explorer_critic_optim
        assert pool.temperature_optim is not pool.explorer_temperature_optim
        assert pool.log_alpha is not pool.explorer_log_alpha
        assert pool.log_alpha.data_ptr() != pool.explorer_log_alpha.data_ptr()
        assert _optimizer_parameter_ids(pool.actor_optim) == _module_parameter_ids(
            pool.actor
        )
        assert _optimizer_parameter_ids(
            pool.explorer_actor_optim
        ) == _module_parameter_ids(pool.explorer_actor)
        assert _optimizer_parameter_ids(pool.critic_optim) == _module_parameter_ids(
            pool.critic
        )
        assert _optimizer_parameter_ids(
            pool.explorer_critic_optim
        ) == _module_parameter_ids(pool.explorer_critic)
        assert _optimizer_parameter_ids(pool.temperature_optim) == {id(pool.log_alpha)}
        assert _optimizer_parameter_ids(pool.explorer_temperature_optim) == {
            id(pool.explorer_log_alpha)
        }
        assert not any(
            parameter.requires_grad for parameter in pool.critic_target.parameters()
        )
        assert not any(
            parameter.requires_grad
            for parameter in pool.explorer_critic_target.parameters()
        )
        for optimizer in (
            pool.actor_optim,
            pool.explorer_actor_optim,
            pool.critic_optim,
            pool.explorer_critic_optim,
            pool.temperature_optim,
            pool.explorer_temperature_optim,
        ):
            assert _optimizer_steps(optimizer) == {1}
        assert metrics["inner_critic_optimizer_steps"] == 1
        assert metrics["inner_explorer_critic_optimizer_steps"] == 1
        assert metrics["inner_actor_optimizer_steps"] == 1
        assert metrics["inner_explorer_actor_optimizer_steps"] == 1
        assert metrics["inner_temperature_optimizer_steps"] == 1
        assert metrics["inner_explorer_temperature_optimizer_steps"] == 1
        assert metrics["inner_critic_target_updates"] == 1
        assert metrics["inner_explorer_critic_target_updates"] == 1
    finally:
        model.env.close()


@pytest.mark.parametrize(
    "source",
    ["primary", "explorer", "mixture_sample", "outer_q_gate", "outer_soft_handoff"],
)
def test_execution_selectors_smoke_and_report_exact_controller_cost(source):
    model = _explorer_model(
        "frozen_random",
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
        inner_execution_policy_source=source,
        inner_execution_handoff_samples=2,
    )
    try:
        action = model.agent.act(
            torch.zeros(3), eval_mode=True, collect_diagnostics=False
        )
        metrics = model.agent.last_inner_metrics

        assert torch.isfinite(action).all()
        assert (
            metrics["inner_selector_primary_wins"]
            + metrics["inner_selector_explorer_wins"]
            == 1
        )
        if source == "primary":
            assert metrics["inner_selector_primary_wins"] == 1
        elif source == "explorer":
            assert metrics["inner_selector_explorer_wins"] == 1
        expected_selector_steps = 8 if source == "outer_soft_handoff" else 0
        assert metrics["inner_selector_model_steps"] == expected_selector_steps
        assert metrics["inner_optimization_model_steps"] == 8
        assert metrics["inner_total_model_steps"] == 8 + expected_selector_steps
        _assert_finite_metrics(
            metrics,
            (
                "inner_selector_score_margin",
                "inner_selector_score_variance",
                "inner_primary_explorer_action_l2",
            ),
        )
    finally:
        model.env.close()


def test_outer_q_gate_tie_breaks_to_primary(monkeypatch):
    model = _explorer_model(
        "frozen_random",
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
        inner_execution_policy_source="outer_q_gate",
    )

    q_calls = []
    policy_stats_calls = []
    original_policy_stats = model.agent.model.policy_stats

    def counted_policy_stats(*args, **kwargs):
        policy_stats_calls.append(kwargs["policy"])
        return original_policy_stats(*args, **kwargs)

    def tied_outer_q(z, action, **kwargs):
        q_calls.append((action.detach().clone(), dict(kwargs)))
        return z.new_zeros(z.shape[0], 1)

    monkeypatch.setattr(model.agent.model, "policy_stats", counted_policy_stats)
    monkeypatch.setattr(model.agent.model, "Q", tied_outer_q)
    try:
        model.agent.act(torch.zeros(3), eval_mode=True, collect_diagnostics=False)
        metrics = model.agent.last_inner_metrics
        assert metrics["inner_selector_primary_wins"] == 1
        assert metrics["inner_selector_explorer_wins"] == 0
        assert metrics["inner_selector_score_margin"] == 0
        assert metrics["inner_fixed_q_counterfactual_primary_wins"] == 1
        assert metrics["inner_fixed_q_counterfactual_explorer_wins"] == 0
        assert metrics["inner_fixed_q_counterfactual_explorer_rate"] == 0
        assert metrics["inner_fixed_q_counterfactual_execution_agreement"] == 1
        assert metrics["inner_fixed_q_counterfactual_primary_q"] == 0
        assert metrics["inner_fixed_q_counterfactual_explorer_q"] == 0
        assert metrics["inner_fixed_q_counterfactual_margin"] == 0
        assert len(policy_stats_calls) == 2
        assert all(policy is not None for policy in policy_stats_calls)
        assert policy_stats_calls[0] is not policy_stats_calls[1]
        assert len(q_calls) == 2
        assert all(
            kwargs["target"] is True and kwargs["reduction"] == "min_all"
            for _, kwargs in q_calls
        )
    finally:
        model.env.close()


def test_cross_explorer_mode_resume_rejection_is_transactional():
    source = _explorer_model(
        "separate_critics",
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
    )
    destination = _explorer_model(
        "shared_mixture",
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
    )
    try:
        source.agent.act(torch.zeros(3), collect_diagnostics=False)
        destination.agent.act(torch.zeros(3), collect_diagnostics=False)
        source.agent.inner_engine.prepare_training_resume_boundary()
        destination.agent.inner_engine.prepare_training_resume_boundary()
        incompatible = deepcopy(source.agent.inner_engine.training_state_dict())
        pristine = deepcopy(destination.agent.inner_engine.training_state_dict())

        assert incompatible["version"] == 2
        assert incompatible["explorer_mode"] == "separate_critics"
        with pytest.raises(ValueError, match="mode is incompatible"):
            destination.agent.inner_engine.load_training_state_dict(incompatible)
        _assert_tree_equal(
            destination.agent.inner_engine.training_state_dict(), pristine
        )
    finally:
        source.env.close()
        destination.env.close()


@pytest.mark.parametrize(
    ("mode", "explicit_doses", "expected_explorer_steps"),
    [
        ("frozen_random", {}, (0, 0, 0)),
        ("shared_mixture", {}, (4, 0, 0)),
        ("separate_critics", {}, (4, 4, 4)),
        (
            "separate_critics",
            {
                "inner_explorer_actor_updates_per_round": 1,
                "inner_explorer_critic_updates_per_round": 2,
                "inner_explorer_temperature_updates_per_round": 3,
            },
            (1, 2, 3),
        ),
    ],
)
def test_episodic_auto_uses_realized_primary_and_inherited_explorer_doses(
    mode, explicit_doses, expected_explorer_steps
):
    model = _tiny_model(
        episodic=True,
        train_unroll_horizon=3,
        outer_planning_horizon=3,
        inner_rounds=1,
        inner_rollouts_per_round=4,
        inner_rollout_horizon=3,
        inner_updates_per_round="auto",
        inner_batch_size=4,
        inner_replay_capacity=12,
        inner_explorer_mode=mode,
        inner_prior_rollout_weight=0.5,
        inner_sac_critic_target="entropy_augmented",
        **explicit_doses,
    )

    def terminate_immediately(z, task=None, unnormalized=False):
        del task, unnormalized
        return torch.ones(z.shape[0], 1, device=z.device)

    try:
        model.agent.model.termination = terminate_immediately
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        metrics = model.agent.last_inner_metrics
        pool = model.agent.inner_engine._action_pool
        sources = pool.replay.source[: pool.replay.size].reshape(-1).long()

        assert metrics["inner_model_steps"] == 4
        assert metrics["inner_primary_transitions"] == 2
        assert metrics["inner_explorer_transitions"] == 2
        assert torch.bincount(sources, minlength=2).tolist() == [2, 2]
        assert metrics["inner_critic_optimizer_steps"] == 4
        assert metrics["inner_actor_optimizer_steps"] == 4
        assert metrics["inner_temperature_optimizer_steps"] == 4
        assert metrics["inner_explorer_actor_optimizer_steps"] == (
            expected_explorer_steps[0]
        )
        assert metrics["inner_explorer_critic_optimizer_steps"] == (
            expected_explorer_steps[1]
        )
        assert metrics["inner_explorer_temperature_optimizer_steps"] == (
            expected_explorer_steps[2]
        )
        # Canonical G scheduling shares each sampled batch across that slot's
        # critic-first and actor/temperature work.
        assert metrics["inner_update_slots"] == 4
        assert metrics["inner_requested_update_slots"] == 4
        assert metrics["inner_primary_termination_rate"] == pytest.approx(1.0)
        assert metrics["inner_explorer_termination_rate"] == pytest.approx(1.0)
        _assert_finite_metrics(
            metrics,
            (
                "inner_primary_reward_mean",
                "inner_explorer_reward_mean",
                "inner_primary_termination_rate",
                "inner_explorer_termination_rate",
            ),
        )
    finally:
        model.env.close()
