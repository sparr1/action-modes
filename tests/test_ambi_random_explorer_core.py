"""Focused integration checks for action-local two-policy inner SAC."""

import pytest
import torch

from tests.test_ambi_root_local_sac import _tiny_component_model, _tiny_model


def _explorer_model(mode, **overrides):
    params = {
        "inner_rounds": 1,
        "inner_rollouts_per_round": 4,
        "inner_rollout_horizon": 2,
        "inner_batch_size": 4,
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


def _counterfactual_action(metrics, action_dim):
    return torch.stack(
        [
            torch.as_tensor(
                metrics[f"inner_fixed_q_counterfactual_action_{index}"]
            )
            for index in range(action_dim)
        ]
    )


@pytest.mark.parametrize(
    ("mode", "r_actor_steps", "r_critic_steps"),
    [
        ("frozen_random", 0, 0),
        ("shared_mixture", 1, 0),
        ("separate_critics", 1, 1),
    ],
)
def test_two_policy_modes_keep_budget_and_own_optimizer_steps(
    mode, r_actor_steps, r_critic_steps
):
    model = _explorer_model(mode)
    try:
        global_rng = torch.random.get_rng_state().clone()
        action = model.agent.act(torch.zeros(3), collect_diagnostics=False)
        torch.testing.assert_close(torch.random.get_rng_state(), global_rng)
        assert torch.isfinite(action).all()

        metrics = model.agent.last_inner_metrics
        assert metrics["inner_primary_rollouts"] == 2
        assert metrics["inner_explorer_rollouts"] == 2
        assert metrics["inner_primary_transitions"] == 4
        assert metrics["inner_explorer_transitions"] == 4
        assert metrics["inner_primary_replay_fraction"] == pytest.approx(0.5)
        assert metrics["inner_explorer_replay_fraction"] == pytest.approx(0.5)
        assert metrics["inner_actor_optimizer_steps"] == 1
        assert metrics["inner_critic_optimizer_steps"] == 1
        assert metrics["inner_explorer_actor_optimizer_steps"] == r_actor_steps
        assert metrics["inner_explorer_critic_optimizer_steps"] == r_critic_steps
        assert (
            metrics["inner_primary_replay_samples"]
            + metrics["inner_explorer_replay_samples"]
            == metrics["inner_replay_draws"]
        )
        assert (
            metrics["inner_primary_replay_sample_fraction"]
            + metrics["inner_explorer_replay_sample_fraction"]
            == pytest.approx(1.0)
        )
        assert (
            metrics["inner_primary_td_error_abs_count"]
            + metrics["inner_explorer_td_error_abs_count"]
            == 4
        )
        assert metrics["inner_fixed_q_counterfactual_policy_evaluations"] == 2
        assert metrics["inner_fixed_q_counterfactual_q_evaluations"] == 2
        assert (
            metrics["inner_fixed_q_counterfactual_primary_wins"]
            + metrics["inner_fixed_q_counterfactual_explorer_wins"]
            == 1
        )
        assert metrics["inner_fixed_q_counterfactual_explorer_rate"] == metrics[
            "inner_fixed_q_counterfactual_explorer_wins"
        ]
        counterfactual_action = _counterfactual_action(
            metrics, int(model.agent.cfg.action_dim)
        )
        assert torch.isfinite(counterfactual_action).all()

        pool = model.agent.inner_engine._action_pool
        assert pool.explorer_actor is not None
        if mode == "frozen_random":
            assert not any(parameter.requires_grad for parameter in pool.explorer_actor.parameters())
            assert pool.explorer_actor_optim is None
        else:
            assert pool.explorer_actor_optim is not None
        if mode == "separate_critics":
            assert pool.explorer_critic is not None
            assert pool.explorer_critic_target is not None
            assert pool.explorer_critic_optim is not None
    finally:
        model.env.close()


@pytest.mark.parametrize("estimator", ["stratified", "weighted"])
def test_shared_mixture_estimators_run_with_exact_source_independent_batch(estimator):
    model = _explorer_model(
        "shared_mixture", inner_mixture_target_estimator=estimator
    )
    try:
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        metrics = model.agent.last_inner_metrics
        assert metrics["inner_critic_optimizer_steps"] == 1
        assert metrics["inner_actor_optimizer_steps"] == 1
        assert metrics["inner_explorer_actor_optimizer_steps"] == 1
        assert torch.isfinite(torch.as_tensor(metrics["inner_mixture_log_prob"]))
        assert "inner_primary_td_error_abs_mean" in metrics
        assert "inner_explorer_td_error_abs_mean" in metrics
    finally:
        model.env.close()


@pytest.mark.parametrize(
    "source",
    ["primary", "explorer", "mixture_sample", "outer_q_gate", "outer_soft_handoff"],
)
def test_all_two_policy_execution_selectors_are_seeded_and_bookkept(source):
    model = _explorer_model(
        "frozen_random",
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
        inner_execution_policy_source=source,
        inner_execution_handoff_samples=2,
    )
    try:
        action = model.agent.act(
            torch.zeros(3), collect_diagnostics=False, eval_mode=True
        )
        metrics = model.agent.last_inner_metrics
        assert torch.isfinite(action).all()
        assert (
            metrics["inner_selector_primary_wins"]
            + metrics["inner_selector_explorer_wins"]
            == 1
        )
        assert (
            metrics["inner_fixed_q_counterfactual_primary_wins"]
            + metrics["inner_fixed_q_counterfactual_explorer_wins"]
            == 1
        )
        assert metrics["inner_fixed_q_counterfactual_explorer_rate"] == metrics[
            "inner_fixed_q_counterfactual_explorer_wins"
        ]
        assert metrics["inner_fixed_q_counterfactual_policy_evaluations"] == 2
        assert metrics["inner_fixed_q_counterfactual_q_evaluations"] == 2
        assert metrics["inner_fixed_q_counterfactual_execution_agreement"] in {
            0,
            1,
        }
        assert metrics["inner_fixed_q_counterfactual_action_l2_to_executed"] >= 0
        _counterfactual_action(metrics, int(model.agent.cfg.action_dim))
        for name in (
            "inner_fixed_q_counterfactual_primary_q",
            "inner_fixed_q_counterfactual_explorer_q",
            "inner_fixed_q_counterfactual_margin",
        ):
            assert torch.isfinite(torch.as_tensor(metrics[name]))
        expected_steps = 8 if source == "outer_soft_handoff" else 0
        assert metrics["inner_selector_model_steps"] == expected_steps
        assert metrics["inner_total_model_steps"] == 8 + expected_steps
    finally:
        model.env.close()


def test_fixed_q_counterfactual_metrics_are_absent_without_explorer():
    model = _tiny_component_model(
        inner_rounds=1,
        inner_rollouts_per_round=4,
        inner_rollout_horizon=2,
        inner_batch_size=4,
        inner_replay_capacity=8,
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
    )
    try:
        model.agent.act(torch.zeros(3), collect_diagnostics=False, eval_mode=True)
        assert not any(
            key.startswith("inner_fixed_q_counterfactual_")
            for key in model.agent.last_inner_metrics
        )
    finally:
        model.env.close()


def test_active_explorer_checkpoint_records_mode_and_rejects_cross_mode_load():
    source = _explorer_model(
        "separate_critics",
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
    )
    restored = _explorer_model(
        "separate_critics",
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
    )
    incompatible = _tiny_component_model(
        inner_rounds=1,
        inner_rollouts_per_round=4,
        inner_rollout_horizon=2,
        inner_batch_size=4,
        inner_replay_capacity=8,
        inner_critic_updates_per_round=0,
        inner_actor_updates_per_round=0,
    )
    try:
        source.agent.act(torch.zeros(3), collect_diagnostics=False)
        source.agent.inner_engine.prepare_training_resume_boundary()
        state = source.agent.inner_engine.training_state_dict()
        assert state["version"] == 2
        assert state["explorer_mode"] == "separate_critics"
        restored.agent.inner_engine.load_training_state_dict(state)
        with pytest.raises(ValueError, match="cannot be loaded"):
            incompatible.agent.inner_engine.load_training_state_dict(state)
    finally:
        source.env.close()
        restored.env.close()
        incompatible.env.close()


@pytest.mark.parametrize("mode", ["frozen_random", "shared_mixture", "separate_critics"])
def test_canonical_auto_uses_realized_episodic_transitions_for_inherited_doses(mode):
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
    )

    def terminate_immediately(z, task=None, unnormalized=False):
        del task, unnormalized
        return torch.ones(z.shape[0], 1, device=z.device)

    try:
        model.agent.model.termination = terminate_immediately
        model.agent.act(torch.zeros(3), collect_diagnostics=False)
        metrics = model.agent.last_inner_metrics
        assert metrics["inner_model_steps"] == 4
        assert metrics["inner_critic_optimizer_steps"] == 4
        assert metrics["inner_actor_optimizer_steps"] == 4
        assert metrics["inner_update_slots"] == 4
        assert metrics["inner_requested_update_slots"] == 4
        assert metrics["inner_replay_draws"] == 16
        assert (
            metrics["inner_primary_replay_samples"]
            + metrics["inner_explorer_replay_samples"]
            == 16
        )
        expected_r = 0 if mode == "frozen_random" else 4
        assert metrics["inner_explorer_actor_optimizer_steps"] == expected_r
        expected_r_q = 4 if mode == "separate_critics" else 0
        assert metrics["inner_explorer_critic_optimizer_steps"] == expected_r_q
    finally:
        model.env.close()
