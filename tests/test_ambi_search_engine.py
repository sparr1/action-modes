import math
import os
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core.inner_improvement import InnerImprovementEngine
from utils.wandb_utils import WandbAccumulator


def _base_params(**overrides):
    """Small canonical schedule that still exercises one complete root solve."""
    params = {
        "device": "cpu",
        "model_size": None,
        "enc_dim": 16,
        "mlp_dim": 16,
        "latent_dim": 8,
        "num_enc_layers": 2,
        "simnorm_dim": 4,
        "num_bins": 11,
        "vmin": -5,
        "vmax": 5,
        "batch_size": 2,
        "train_unroll_horizon": 3,
        "outer_planning_horizon": 2,
        "buffer_size": 32,
        "seed_steps": 2,
        "pretrain_steps": 1,
        "utd": 1,
        "compile": False,
        "episodic": False,
        "discount": 0.99,
        "wandb": False,
        "dropout": 0.0,
        "q_representation": "scalar",
        "num_q": 2,
        "q_num_bins": 11,
        "q_vmin": -5,
        "q_vmax": 5,
        "inner_operator": "sac",
        "inner_rounds": 1,
        "inner_rollouts_per_round": 2,
        "inner_rollout_horizon": 3,
        "inner_critic_updates_per_round": 1,
        "inner_actor_updates_per_round": 1,
        "inner_batch_size": 2,
        "inner_replay_capacity": 6,
        "inner_actor_adaptation": "clone",
        "inner_critic_adaptation": "clone",
        "inner_temperature_mode": "inherit_outer",
        "inner_critic_target_tau": 1.0,
        "inner_critic_target_update_interval": 1,
        "inner_diagnostic_rollouts": 0,
        "inner_behavior_action": "policy_sample",
        "inner_behavior_std_scale": 1.0,
        "inner_behavior_noise_std": 0.0,
        "inner_mppi_num_elites": 2,
        "inner_mppi_num_pi_trajs": 0,
    }
    params.update(overrides)
    return params


def _legacy_params(**overrides):
    """Historical total-budget spelling used by the seeded regression."""
    params = _base_params()
    for key in (
        "inner_rollouts_per_round",
        "inner_critic_updates_per_round",
        "inner_actor_updates_per_round",
    ):
        params.pop(key)
    params.update(
        inner_model_step_budget=6,
        inner_critic_updates_per_action=1,
        inner_actor_updates_per_action=1,
        inner_temperature_updates_per_action=0,
    )
    params.update(overrides)
    return params


def _model(params):
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    return AMBITDMPC2(
        "AMBITDMPC2",
        env,
        params,
        {"seed": 73, "device": "cpu", "env": "test", "total_steps": 10},
        {},
    )


def _clone_state(module):
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def _assert_state_unchanged(module, before):
    assert module.state_dict().keys() == before.keys()
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def _assert_finite_tree(values):
    for name, value in values.items():
        if torch.is_tensor(value):
            assert bool(torch.isfinite(value).all().item()), name
        elif isinstance(value, (int, float)):
            assert math.isfinite(float(value)), name


def _metrics_logging_harness(metrics):
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.agent = SimpleNamespace(
        last_inner_rollout_lengths=[], last_inner_metrics=metrics
    )
    algorithm._wandb_train_window = WandbAccumulator()
    algorithm._wandb_inner_seconds = 0.0
    algorithm._wandb_inner_actions = 0
    algorithm._wandb_inner_steps = 0
    algorithm._wandb_outer_policy_seconds = 0.0
    algorithm._wandb_outer_policy_actions = 0
    algorithm._inner_steps_total = 0
    algorithm._inner_updates_total = 0
    algorithm._outer_policy_episode_selected = False
    return algorithm


def _finite_q_params(layout, estimator, **overrides):
    params = _base_params(
        inner_q_objective="finite_horizon",
        inner_critic_horizon_mode=layout,
        inner_return_estimator=estimator,
        inner_search_replay_retention="action",
        inner_offpolicy_mode="none",
        inner_search_bootstrap_critic="target",
        inner_target_update_event="optimizer_step",
        inner_depth_update_order="mixed",
    )
    if estimator == "n_step":
        params.update(
            inner_return_steps=2,
            inner_search_replay_retention="round",
        )
    elif estimator == "lambda_return":
        params.update(
            inner_return_lambda=0.5,
            inner_search_replay_retention="round",
        )
    elif estimator == "full_suffix":
        params.update(
            inner_search_replay_retention="round",
            inner_search_bootstrap_critic="none",
            inner_target_update_event="none",
        )
    elif estimator == "retrace":
        params.update(
            inner_return_lambda=0.5,
            inner_search_replay_retention="action",
            inner_offpolicy_mode="per_decision_is",
        )
    params.update(overrides)
    return params


Q_RECIPES = [
    (layout, estimator)
    for layout in ("shared", "depth_conditioned", "stage_heads")
    for estimator in ("td0", "n_step", "lambda_return", "full_suffix", "retrace")
]


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize(("layout", "estimator"), Q_RECIPES)
def test_each_finite_q_architecture_estimator_runs_one_action(
    layout, estimator, representation
):
    model = _model(
        _finite_q_params(
            layout,
            estimator,
            q_representation=representation,
            num_q=2 if representation == "scalar" else 5,
        )
    )
    outer_before = _clone_state(model.agent.model)

    action = model.agent.act(torch.zeros(3), t0=True, eval_mode=False)
    metrics = model.agent.last_inner_metrics

    assert action.shape == (model.cfg.action_dim,)
    assert bool(torch.isfinite(action).all().item())
    assert metrics["inner_model_steps"] == 6
    assert metrics["inner_critic_optimizer_steps"] == 1
    assert metrics["inner_actor_optimizer_steps"] == 1
    if model.cfg.inner_search_replay_retention == "round":
        assert metrics["inner_buffer_size"] == 0
        assert metrics["inner_buffer_peak_size"] == 6
    else:
        assert metrics["inner_buffer_size"] == 6
        assert metrics["inner_buffer_peak_size"] == 6
    sampled_depth_counts = []
    for depth in (1, 2, 3):
        sampled_depth_counts.append(metrics[f"inner_depth_{depth}_sample_count"])
        assert f"inner_depth_{depth}_q_mean" in metrics
        assert f"inner_depth_{depth}_target_mean" in metrics
        assert f"inner_depth_{depth}_critic_loss" in metrics
    assert sum(float(count) for count in sampled_depth_counts) == 2
    _assert_finite_tree(metrics)
    _assert_state_unchanged(model.agent.model, outer_before)

    target = model.agent.inner_engine._action_pool.critic_target
    if estimator == "full_suffix":
        assert target is None
        assert metrics["inner_critic_target_updates"] == 0
    else:
        assert target is not None
        assert metrics["inner_critic_target_updates"] == 1


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize("layout", ["shared", "depth_conditioned", "stage_heads"])
@pytest.mark.parametrize("retention", ["round", "action"])
def test_each_vtrace_layout_and_retention_runs_one_action(
    layout, retention, representation
):
    params = _base_params(
        inner_operator="vtrace",
        inner_q_objective="finite_horizon",
        inner_critic_horizon_mode=layout,
        inner_return_estimator="td0",
        inner_return_lambda=0.5,
        inner_search_replay_retention=retention,
        inner_offpolicy_mode="per_decision_is",
        inner_search_bootstrap_critic="target",
        inner_target_update_event="optimizer_step",
        inner_depth_update_order="mixed",
        inner_vtrace_distill_updates=1,
        inner_vtrace_distill_action_samples=1,
        outer_critic_target="reward_only",
        inner_sac_critic_target="reward_only",
        q_representation=representation,
        num_q=2 if representation == "scalar" else 5,
        inner_rounds=2,
        inner_replay_capacity=12,
    )
    model = _model(params)
    outer_before = _clone_state(model.agent.model)

    action = model.agent.act(torch.zeros(3), t0=True, eval_mode=False)
    metrics = model.agent.last_inner_metrics

    assert action.shape == (model.cfg.action_dim,)
    assert bool(torch.isfinite(action).all().item())
    assert metrics["inner_model_steps"] == 12
    assert metrics["inner_critic_optimizer_steps"] == 2
    assert metrics["inner_actor_optimizer_steps"] == 2
    assert metrics["inner_vtrace_distill_optimizer_steps"] == 1
    assert metrics["inner_requested_update_slots"] == 4
    assert metrics["inner_update_slots"] == 4
    assert metrics["inner_critic_target_updates"] == 2
    assert metrics["inner_vtrace_ratio_mean"] > 0
    assert metrics["inner_vtrace_ess"] > 0
    assert "inner_leaf_contribution" in metrics
    assert "inner_actor_policy_gradient_loss" in metrics
    assert "inner_actor_entropy_loss" in metrics
    assert "inner_vtrace_actor_advantage" in metrics
    assert "inner_vtrace_pg_ratio_clipped_fraction" in metrics
    assert model.agent.inner_engine._action_pool.critic_target is not None
    assert model.agent.inner_engine._action_pool.replay.size == (
        0 if retention == "round" else 4
    )
    assert metrics["inner_buffer_size"] == (0 if retention == "round" else 12)
    assert metrics["inner_buffer_peak_size"] == (6 if retention == "round" else 12)
    _assert_finite_tree(metrics)
    _assert_state_unchanged(model.agent.model, outer_before)


@pytest.mark.parametrize(
    "offpolicy_mode", ["uncorrected", "per_decision_is", "resimulate"]
)
def test_action_retained_n_step_replay_correction_recipes(offpolicy_mode):
    model = _model(
        _finite_q_params(
            "depth_conditioned",
            "n_step",
            inner_rounds=2,
            inner_replay_capacity=12,
            inner_batch_size=6,
            inner_replay_sampling="without_replacement",
            inner_search_replay_retention="action",
            inner_offpolicy_mode=offpolicy_mode,
        )
    )
    outer_before = _clone_state(model.agent.model)

    model.agent.act(torch.zeros(3), t0=True, eval_mode=False)
    metrics = model.agent.last_inner_metrics

    assert metrics["inner_model_steps"] == 12
    assert metrics["inner_critic_optimizer_steps"] == 2
    assert model.agent.inner_engine._action_pool.replay.size == 4
    assert metrics["inner_buffer_size"] == 12
    assert metrics["inner_buffer_peak_size"] == 12
    if offpolicy_mode == "per_decision_is":
        assert metrics["inner_ratio_mean"] > 0
        assert metrics["inner_ess"] > 0
        assert metrics["inner_pdis_weight_ess"] > 0
        assert metrics["inner_pdis_weight_normalized_ess"] > 0
    if offpolicy_mode == "resimulate":
        assert metrics["inner_target_model_steps"] > 0
        assert metrics["inner_optimization_model_steps"] > metrics["inner_model_steps"]
    else:
        assert metrics["inner_target_model_steps"] == 0
    _assert_finite_tree(metrics)
    _assert_state_unchanged(model.agent.model, outer_before)


@pytest.mark.parametrize("leaf_source", ["outer_target", "outer_online"])
def test_finite_leaf_source_routes_outer_q_without_mutation(leaf_source, monkeypatch):
    model = _model(
        _finite_q_params(
            "shared",
            "full_suffix",
            inner_leaf_q_source=leaf_source,
        )
    )
    outer_before = _clone_state(model.agent.model)
    observed_target_flags = []
    original_q = model.agent.model.Q

    def tracked_q(*args, **kwargs):
        observed_target_flags.append(bool(kwargs.get("target", False)))
        return original_q(*args, **kwargs)

    monkeypatch.setattr(model.agent.model, "Q", tracked_q)
    model.agent.act(
        torch.zeros(3),
        t0=True,
        eval_mode=False,
        collect_diagnostics=False,
    )

    assert observed_target_flags
    assert set(observed_target_flags) == {leaf_source == "outer_target"}
    _assert_state_unchanged(model.agent.model, outer_before)


@pytest.mark.parametrize("component", ["actor", "critic"])
def test_finite_shared_q_supports_actor_and_critic_lora(component):
    overrides = {f"inner_{component}_adaptation": "lora"}
    model = _model(_finite_q_params("shared", "td0", **overrides))
    outer_before = _clone_state(model.agent.model)

    action = model.agent.act(torch.zeros(3), t0=True, eval_mode=False)

    assert bool(torch.isfinite(action).all().item())
    assert model.agent.last_inner_metrics[
        f"inner_{component}_optimizer_steps"
    ] == 1
    _assert_finite_tree(model.agent.last_inner_metrics)
    _assert_state_unchanged(model.agent.model, outer_before)


@pytest.mark.parametrize(
    ("bootstrap", "event", "expected_target", "expected_updates"),
    [
        ("target", "optimizer_step", True, 1),
        ("target", "round_end", True, 1),
        ("frozen_target", "none", True, 0),
        ("online", "none", False, 0),
    ],
)
def test_finite_q_target_inventory_and_update_events(
    bootstrap, event, expected_target, expected_updates
):
    model = _model(
        _finite_q_params(
            "shared",
            "td0",
            inner_search_bootstrap_critic=bootstrap,
            inner_target_update_event=event,
        )
    )

    model.agent.act(torch.zeros(3), t0=True, eval_mode=False)

    state = model.agent.inner_engine.state
    target = model.agent.inner_engine._action_pool.critic_target
    assert (target is not None) is expected_target
    assert state.critic_target_steps == expected_updates
    assert model.agent.last_inner_metrics["inner_critic_target_updates"] == expected_updates
    if bootstrap == "frozen_target":
        q_input = torch.randn(4, model.cfg.latent_dim + model.cfg.action_dim)
        expected = model.agent.model._Qs(q_input)
        for depth in (1, 2, 3):
            torch.testing.assert_close(target(q_input, depth), expected)


@pytest.mark.parametrize("event", ["optimizer_step", "round_end"])
def test_finite_target_ema_interpolation_is_exact(event, monkeypatch):
    tau = 0.25
    model = _model(
        _finite_q_params(
            "depth_conditioned",
            "td0",
            inner_target_update_event=event,
            inner_critic_target_tau=tau,
        )
    )
    engine = model.agent.inner_engine
    captured = {}
    original_prepare = engine._prepare_search_workspace

    def tracked_prepare(*, t0):
        result = original_prepare(t0=t0)
        captured["target_parameters"] = [
            parameter.detach().clone()
            for parameter in engine.state.critic_target.parameters()
        ]
        return result

    monkeypatch.setattr(engine, "_prepare_search_workspace", tracked_prepare)
    model.agent.act(torch.zeros(3), t0=True, eval_mode=False)

    online = engine._action_pool.critic
    target = engine._action_pool.critic_target
    assert target is not None
    assert model.agent.last_inner_metrics["inner_critic_target_updates"] == 1
    for initial, source, actual in zip(
        captured["target_parameters"], online.parameters(), target.parameters()
    ):
        torch.testing.assert_close(
            actual,
            initial.lerp(source.detach(), tau),
            rtol=0,
            atol=2e-8,
        )


def test_finite_optimizer_step_ema_obeys_interval_and_retains_update_snapshot(
    monkeypatch,
):
    """Interval two updates after step two, not again after the third step."""
    tau = 0.25
    model = _model(
        _finite_q_params(
            "depth_conditioned",
            "td0",
            inner_critic_updates_per_round=3,
            inner_target_update_event="optimizer_step",
            inner_critic_target_tau=tau,
            inner_critic_target_update_interval=2,
        )
    )
    engine = model.agent.inner_engine
    captured = {}
    original_prepare = engine._prepare_search_workspace

    def tracked_prepare(*, t0):
        result = original_prepare(t0=t0)
        target = engine.state.critic_target
        original_update = target.update_from

        def tracked_update(source, *, tau, remaining_horizon=None):
            captured.setdefault(
                "initial",
                [parameter.detach().clone() for parameter in target.parameters()],
            )
            result = original_update(
                source, tau=tau, remaining_horizon=remaining_horizon
            )
            captured.setdefault("calls", []).append(
                {
                    "tau": tau,
                    "remaining_horizon": remaining_horizon,
                    "source": [
                        parameter.detach().clone() for parameter in source.parameters()
                    ],
                    "target": [
                        parameter.detach().clone() for parameter in target.parameters()
                    ],
                }
            )
            return result

        monkeypatch.setattr(target, "update_from", tracked_update)
        return result

    monkeypatch.setattr(engine, "_prepare_search_workspace", tracked_prepare)
    model.agent.act(torch.zeros(3), t0=True, eval_mode=False)

    calls = captured["calls"]
    assert len(calls) == 1
    assert calls[0]["tau"] == tau
    assert calls[0]["remaining_horizon"] is None
    assert engine.state.critic_steps == 3
    assert engine.state.critic_target_steps == 1
    assert model.agent.last_inner_metrics["inner_critic_target_updates"] == 1

    for initial, source_at_step_two, target_at_step_two, final_target in zip(
        captured["initial"],
        calls[0]["source"],
        calls[0]["target"],
        engine._action_pool.critic_target.parameters(),
    ):
        torch.testing.assert_close(
            target_at_step_two,
            initial.lerp(source_at_step_two, tau),
            rtol=0,
            atol=2e-8,
        )
        torch.testing.assert_close(
            final_target, target_at_step_two, rtol=0, atol=0
        )


def test_vtrace_distillation_hard_resets_target_and_adaptation_optimizer(monkeypatch):
    model = _model(
        _base_params(
            inner_operator="vtrace",
            inner_q_objective="finite_horizon",
            inner_critic_horizon_mode="stage_heads",
            inner_return_lambda=0.5,
            inner_search_replay_retention="round",
            inner_offpolicy_mode="per_decision_is",
            inner_search_bootstrap_critic="target",
            inner_target_update_event="optimizer_step",
            inner_vtrace_distill_updates=2,
            inner_vtrace_distill_action_samples=2,
            outer_critic_target="reward_only",
            inner_sac_critic_target="reward_only",
        )
    )
    engine = model.agent.inner_engine
    observed = {"calls": 0}
    original_distill = engine._distill_search_value

    def tracked_distill():
        metrics = original_distill()
        observed["calls"] += 1
        assert engine.state.critic_target is not None
        for online, target in zip(
            engine.state.critic.parameters(),
            engine.state.critic_target.parameters(),
        ):
            torch.testing.assert_close(target, online, rtol=0, atol=0)
        assert engine.state.critic_steps == 0
        assert engine.state.critic_lifetime_steps == 0
        assert engine.state.critic_target_steps == 0
        assert engine.state.target_steps == 0
        assert not engine.state.critic_optim.state
        return metrics

    monkeypatch.setattr(engine, "_distill_search_value", tracked_distill)
    model.agent.act(torch.zeros(3), t0=True, eval_mode=False)

    assert observed["calls"] == 1
    assert model.agent.last_inner_metrics[
        "inner_vtrace_distill_optimizer_steps"
    ] == 2
    assert "inner_vtrace_distill_error" in model.agent.last_inner_metrics


def test_finite_q_root_diagnostics_route_online_critic_at_full_depth(monkeypatch):
    model = _model(_finite_q_params("stage_heads", "td0"))
    engine = model.agent.inner_engine
    observed_root_depths = []
    original = engine._search_q_predictions

    def tracked(z, action, remaining_horizon, **kwargs):
        if z.shape[0] == 1:
            observed_root_depths.append(remaining_horizon.detach().clone())
        return original(z, action, remaining_horizon, **kwargs)

    monkeypatch.setattr(engine, "_search_q_predictions", tracked)
    model.agent.act(torch.zeros(3), t0=True, eval_mode=True)
    metrics = model.agent.last_inner_metrics

    root_depths = [
        depth
        for depth in observed_root_depths
        if bool((depth == 3).all().item())
    ]
    assert len(root_depths) == 2
    for depth in root_depths:
        torch.testing.assert_close(depth, torch.tensor([3]))
    assert metrics["inner_search_root_remaining_horizon"] == 3
    assert "inner_search_root_q_outer_action" in metrics
    assert "inner_search_root_q_improved_action" in metrics
    assert "inner_search_root_q_action_gain" in metrics
    _assert_finite_tree(metrics)


def test_finite_root_diagnostics_are_omitted_when_collection_is_disabled():
    model = _model(_finite_q_params("stage_heads", "td0"))

    model.agent.act(
        torch.zeros(3),
        t0=True,
        eval_mode=True,
        collect_diagnostics=False,
    )

    assert not any(
        key.startswith("inner_search_root_")
        for key in model.agent.last_inner_metrics
    )


def test_depth_statistics_preserve_exact_sparse_population_moments():
    history = [
        {
            "depth_1_sample_count": 2.0,
            "depth_1_q_sum": 4.0,
            "depth_1_q_squared_sum": 10.0,
            "depth_1_q_count": 2.0,
            "depth_1_q_minimum": 1.0,
            "depth_1_q_maximum": 3.0,
            "depth_1_target_sum": 6.0,
            "depth_1_target_squared_sum": 20.0,
            "depth_1_target_count": 2.0,
            "depth_1_target_minimum": 2.0,
            "depth_1_target_maximum": 4.0,
            "depth_1_critic_loss_sum": 5.0,
            "depth_1_critic_loss_squared_sum": 17.0,
            "depth_1_critic_loss_count": 2.0,
            "depth_1_critic_loss_minimum": 1.0,
            "depth_1_critic_loss_maximum": 4.0,
            "depth_2_sample_count": 0.0,
            "depth_2_q_sum": 0.0,
            "depth_2_q_squared_sum": 0.0,
            "depth_2_q_count": 0.0,
        },
        {
            "depth_1_sample_count": 1.0,
            "depth_1_q_sum": 8.0,
            "depth_1_q_squared_sum": 64.0,
            "depth_1_q_count": 1.0,
            "depth_1_q_minimum": 8.0,
            "depth_1_q_maximum": 8.0,
            "depth_1_target_sum": 10.0,
            "depth_1_target_squared_sum": 100.0,
            "depth_1_target_count": 1.0,
            "depth_1_target_minimum": 10.0,
            "depth_1_target_maximum": 10.0,
            "depth_1_critic_loss_sum": 9.0,
            "depth_1_critic_loss_squared_sum": 81.0,
            "depth_1_critic_loss_count": 1.0,
            "depth_1_critic_loss_minimum": 9.0,
            "depth_1_critic_loss_maximum": 9.0,
            "depth_2_sample_count": 0.0,
            "depth_2_q_sum": 0.0,
            "depth_2_q_squared_sum": 0.0,
            "depth_2_q_count": 0.0,
        },
    ]

    metrics = InnerImprovementEngine._average_update_metrics(history)

    assert metrics["inner_depth_1_sample_count"] == pytest.approx(3.0)
    assert metrics["inner_depth_1_q_count"] == pytest.approx(3.0)
    assert metrics["inner_depth_1_q_mean"] == pytest.approx(4.0)
    assert metrics["inner_depth_1_q_std"] == pytest.approx(math.sqrt(26.0 / 3.0))
    assert metrics["inner_depth_1_q_min"] == pytest.approx(1.0)
    assert metrics["inner_depth_1_q_max"] == pytest.approx(8.0)
    assert metrics["inner_depth_1_target_mean"] == pytest.approx(16.0 / 3.0)
    assert metrics["inner_depth_1_target_min"] == pytest.approx(2.0)
    assert metrics["inner_depth_1_target_max"] == pytest.approx(10.0)
    assert metrics["inner_depth_1_critic_loss_mean"] == pytest.approx(14.0 / 3.0)
    assert metrics["inner_depth_1_critic_loss_min"] == pytest.approx(1.0)
    assert metrics["inner_depth_1_critic_loss_max"] == pytest.approx(9.0)
    assert metrics["inner_depth_2_sample_count"] == pytest.approx(0.0)
    assert metrics["inner_depth_2_q_count"] == pytest.approx(0.0)
    assert "inner_depth_2_q_min" not in metrics
    assert "inner_depth_2_q_max" not in metrics


def test_finite_search_metrics_reach_wandb_with_exact_sparse_weights():
    first = {
        "inner_active": 1.0,
        "inner_rollouts": 0.0,
        "inner_steps": 0.0,
        "inner_updates": 0.0,
        "inner_critic_optimizer_steps": 2.0,
        "inner_actor_optimizer_steps": 1.0,
        "inner_target_model_steps": 3.0,
        "inner_vtrace_distill_optimizer_steps": 4.0,
        "inner_depth_1_sample_count": 2.0,
        "inner_depth_1_q_count": 2.0,
        "inner_depth_1_q_mean": 2.0,
        "inner_depth_1_q_std": 1.0,
        "inner_depth_1_q_min": 1.0,
        "inner_depth_1_q_max": 3.0,
        "inner_depth_1_target_count": 2.0,
        "inner_depth_1_target_mean": 4.0,
        "inner_depth_1_target_std": 1.0,
        "inner_depth_1_target_min": 3.0,
        "inner_depth_1_target_max": 5.0,
        "inner_depth_1_critic_loss_count": 2.0,
        "inner_depth_1_critic_loss_mean": 6.0,
        "inner_depth_1_critic_loss_std": 1.0,
        "inner_depth_1_critic_loss_min": 5.0,
        "inner_depth_1_critic_loss_max": 7.0,
        "inner_depth_2_sample_count": 0.0,
        "inner_depth_2_q_count": 0.0,
        "inner_depth_2_q_mean": 0.0,
        "inner_buffer_peak_size": 10.0,
        "inner_bootstrap_contribution": 0.5,
        "inner_leaf_contribution": 0.25,
        "inner_effective_return_length": 2.0,
        "inner_ratio_mean": 1.0,
        "inner_ratio_max": 1.5,
        "inner_ratio_clipped_fraction": 0.25,
        "inner_ess": 1.5,
        "inner_normalized_ess": 0.75,
        "inner_pdis_weight_mean": 0.8,
        "inner_pdis_weight_max": 1.2,
        "inner_pdis_weight_ess": 1.8,
        "inner_pdis_weight_normalized_ess": 0.9,
        "inner_vtrace_ratio_mean": 0.9,
        "inner_vtrace_ratio_max": 1.1,
        "inner_vtrace_ratio_clipped_fraction": 0.1,
        "inner_vtrace_ess": 1.9,
        "inner_vtrace_normalized_ess": 0.95,
        "inner_actor_policy_gradient_loss": 1.25,
        "inner_actor_entropy_loss": -0.5,
        "inner_vtrace_actor_advantage": 0.75,
        "inner_vtrace_pg_ratio_clipped_fraction": 0.2,
        "inner_vtrace_distill_error": 0.125,
        "inner_vtrace_distill_error_initial": 0.5,
        "inner_vtrace_distill_updates": 4.0,
        "inner_search_root_q_outer_action": 3.0,
        "inner_search_root_q_improved_action": 4.0,
        "inner_search_root_q_action_gain": 1.0,
        "inner_search_root_q_abs_mean": 3.5,
        "inner_search_root_v_h": 3.75,
        "inner_search_root_v_h_abs_mean": 3.75,
        "inner_search_root_remaining_horizon": 3.0,
    }
    algorithm = _metrics_logging_harness(first)
    algorithm._record_action_metrics(planned=True, action_seconds=0.0)
    algorithm.agent.last_inner_metrics = {
        "inner_active": 1.0,
        "inner_rollouts": 0.0,
        "inner_steps": 0.0,
        "inner_updates": 0.0,
        "inner_critic_optimizer_steps": 1.0,
        "inner_actor_optimizer_steps": 2.0,
        "inner_target_model_steps": 2.0,
        "inner_vtrace_distill_optimizer_steps": 0.0,
        "inner_depth_1_sample_count": 1.0,
        "inner_depth_1_q_count": 1.0,
        "inner_depth_1_q_mean": 8.0,
        "inner_depth_1_q_std": 0.0,
        "inner_depth_1_q_min": 8.0,
        "inner_depth_1_q_max": 8.0,
        "inner_depth_2_sample_count": 2.0,
        "inner_depth_2_q_count": 2.0,
        "inner_depth_2_q_mean": 6.0,
        "inner_depth_2_q_std": 2.0,
        "inner_depth_2_q_min": 4.0,
        "inner_depth_2_q_max": 8.0,
        "inner_buffer_peak_size": 14.0,
    }
    algorithm._record_action_metrics(planned=True, action_seconds=0.0)

    payload = algorithm._wandb_train_window.pop()

    assert payload["train/inner_target_model_steps"] == pytest.approx(5.0)
    assert payload["train/inner_vtrace_distill_optimizer_steps"] == pytest.approx(4.0)
    assert payload["train/inner_depth_1_sample_count"] == pytest.approx(3.0)
    assert payload["train/inner_depth_1_q_count"] == pytest.approx(3.0)
    assert payload["train/inner_depth_1_q_mean"] == pytest.approx(4.0)
    assert payload["train/inner_depth_1_q_std"] == pytest.approx(
        math.sqrt(26.0 / 3.0)
    )
    assert payload["train/inner_depth_1_q_min"] == pytest.approx(1.0)
    assert payload["train/inner_depth_1_q_max"] == pytest.approx(8.0)
    assert payload["train/inner_depth_2_sample_count"] == pytest.approx(2.0)
    assert payload["train/inner_depth_2_q_count"] == pytest.approx(2.0)
    assert payload["train/inner_depth_2_q_mean"] == pytest.approx(6.0)
    assert payload["train/inner_buffer_peak_size"] == pytest.approx(12.0)
    assert payload["train/inner_search_root_q_action_gain"] == pytest.approx(1.0)
    assert payload["train/inner_search_root_q_action_gain_count"] == pytest.approx(1.0)

    expected_finite_diagnostics = {
        "inner_bootstrap_contribution",
        "inner_leaf_contribution",
        "inner_effective_return_length",
        "inner_ratio_mean",
        "inner_ratio_max",
        "inner_ratio_clipped_fraction",
        "inner_ess",
        "inner_normalized_ess",
        "inner_pdis_weight_mean",
        "inner_pdis_weight_max",
        "inner_pdis_weight_ess",
        "inner_pdis_weight_normalized_ess",
        "inner_vtrace_ratio_mean",
        "inner_vtrace_ratio_max",
        "inner_vtrace_ratio_clipped_fraction",
        "inner_vtrace_ess",
        "inner_vtrace_normalized_ess",
        "inner_actor_policy_gradient_loss",
        "inner_actor_entropy_loss",
        "inner_vtrace_actor_advantage",
        "inner_vtrace_pg_ratio_clipped_fraction",
        "inner_vtrace_distill_error",
        "inner_vtrace_distill_error_initial",
        "inner_vtrace_distill_updates",
        "inner_search_root_q_outer_action",
        "inner_search_root_q_improved_action",
        "inner_search_root_q_abs_mean",
        "inner_search_root_v_h",
        "inner_search_root_v_h_abs_mean",
        "inner_search_root_remaining_horizon",
    }
    assert {
        key.removeprefix("train/")
        for key in payload
        if key.startswith("train/")
    }.issuperset(expected_finite_diagnostics)


def test_vtrace_root_diagnostics_route_value_at_full_depth(monkeypatch):
    model = _model(
        _base_params(
            inner_operator="vtrace",
            inner_q_objective="finite_horizon",
            inner_critic_horizon_mode="stage_heads",
            inner_return_lambda=0.5,
            inner_search_replay_retention="round",
            inner_offpolicy_mode="per_decision_is",
            inner_search_bootstrap_critic="target",
            inner_target_update_event="optimizer_step",
            inner_depth_update_order="mixed",
            inner_vtrace_distill_updates=1,
            inner_vtrace_distill_action_samples=1,
            outer_critic_target="reward_only",
            inner_sac_critic_target="reward_only",
        )
    )
    engine = model.agent.inner_engine
    observed_root_depths = []
    original_diagnostics = engine._diagnostics

    def tracked_diagnostics(root_z, improved_policy):
        original_forward = engine.state.critic.forward

        def tracked_forward(z, remaining_horizon):
            if z.shape[0] == 1:
                observed_root_depths.append(remaining_horizon.detach().clone())
            return original_forward(z, remaining_horizon)

        monkeypatch.setattr(engine.state.critic, "forward", tracked_forward)
        return original_diagnostics(root_z, improved_policy)

    monkeypatch.setattr(engine, "_diagnostics", tracked_diagnostics)
    model.agent.act(torch.zeros(3), t0=True, eval_mode=True)
    metrics = model.agent.last_inner_metrics

    assert len(observed_root_depths) == 1
    torch.testing.assert_close(observed_root_depths[0], torch.tensor([3]))
    assert metrics["inner_search_root_remaining_horizon"] == 3
    assert "inner_search_root_v_h" in metrics
    assert "inner_search_root_v_h_abs_mean" in metrics
    _assert_finite_tree(metrics)


@pytest.mark.parametrize("layout", ["depth_conditioned", "stage_heads"])
def test_hard_depth_stage_propagation_runs_leaf_to_root(layout):
    model = _model(
        _finite_q_params(
            layout,
            "td0",
            inner_search_bootstrap_critic="target",
            inner_target_update_event="depth_stage",
            inner_depth_update_order="backward",
            inner_critic_target_tau=1.0,
            inner_critic_target_update_interval=1,
        )
    )

    model.agent.act(torch.zeros(3), t0=True, eval_mode=False)
    metrics = model.agent.last_inner_metrics

    assert metrics["inner_critic_optimizer_steps"] == 3
    assert metrics["inner_critic_target_updates"] == 3
    for depth in (1, 2, 3):
        assert metrics[f"inner_depth_{depth}_sample_count"] > 0
    _assert_finite_tree(metrics)


@pytest.mark.parametrize(
    ("bootstrap", "event", "expected_target_updates"),
    [
        ("online", "none", 0),
        ("frozen_target", "none", 0),
        ("target", "optimizer_step", 3),
        ("target", "round_end", 1),
    ],
)
def test_backward_q_sweep_is_independent_of_target_strategy(
    bootstrap, event, expected_target_updates, monkeypatch
):
    model = _model(
        _finite_q_params(
            "shared",
            "td0",
            inner_search_bootstrap_critic=bootstrap,
            inner_target_update_event=event,
            inner_depth_update_order="backward",
        )
    )
    engine = model.agent.inner_engine
    observed_depths = []
    original_sample = engine._sample_search_anchors

    def tracked_sample(*, remaining_horizon=None):
        observed_depths.append(remaining_horizon)
        return original_sample(remaining_horizon=remaining_horizon)

    monkeypatch.setattr(engine, "_sample_search_anchors", tracked_sample)
    try:
        model.agent.act(torch.zeros(3), t0=True, eval_mode=False)
        metrics = model.agent.last_inner_metrics

        # Three critic stages precede the one mixed-depth actor draw.
        assert observed_depths == [1, 2, 3, None]
        assert metrics["inner_critic_optimizer_steps"] == 3
        assert metrics["inner_actor_optimizer_steps"] == 1
        assert metrics["inner_critic_target_updates"] == expected_target_updates
        assert metrics["inner_requested_update_slots"] == 4
        assert metrics["inner_update_slots"] == 4
        assert model.cfg.inner_critic_updates_per_action == 3
        assert model.cfg.inner_expected_update_slots == 4
        assert model.cfg.inner_nominal_critic_utd == pytest.approx(3 / 6)
        for depth in (1, 2, 3):
            assert metrics[f"inner_depth_{depth}_sample_count"] == 2
        _assert_finite_tree(metrics)
    finally:
        model.env.close()


@pytest.mark.parametrize(
    "estimator", ["td0", "n_step", "lambda_return", "full_suffix", "retrace"]
)
def test_backward_q_sweep_supports_every_finite_return_estimator(estimator):
    model = _model(
        _finite_q_params(
            "stage_heads",
            estimator,
            inner_depth_update_order="backward",
        )
    )
    try:
        model.agent.act(torch.zeros(3), t0=True, eval_mode=False)
        metrics = model.agent.last_inner_metrics

        assert metrics["inner_critic_optimizer_steps"] == 3
        assert metrics["inner_actor_optimizer_steps"] == 1
        assert metrics["inner_critic_target_updates"] == (
            0 if estimator == "full_suffix" else 3
        )
        assert metrics["inner_requested_update_slots"] == 4
        assert sum(
            metrics[f"inner_depth_{depth}_sample_count"]
            for depth in (1, 2, 3)
        ) == 6
        _assert_finite_tree(metrics)
    finally:
        model.env.close()


def _assert_nested_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            _assert_nested_equal(left_value, right_value)
    else:
        assert left == right


@pytest.mark.parametrize(
    ("operator", "expected_regions"),
    [
        (
            "sac",
            {
                "_search_rollout_kernel",
                "_search_outer_leaf_kernel",
                "_search_critic_loss_kernel",
                "_search_actor_loss_kernel",
            },
        ),
        (
            "vtrace",
            {
                "_search_rollout_kernel",
                "_search_outer_leaf_kernel",
                "_search_value_critic_loss_kernel",
                "_search_vtrace_actor_loss_kernel",
            },
        ),
    ],
)
def test_finite_search_identity_compile_matches_eager_and_invokes_regions(
    operator, expected_regions, monkeypatch
):
    if operator == "sac":
        params = _finite_q_params("depth_conditioned", "td0")
    else:
        params = _base_params(
            inner_operator="vtrace",
            inner_q_objective="finite_horizon",
            inner_critic_horizon_mode="stage_heads",
            inner_return_lambda=0.5,
            inner_search_replay_retention="round",
            inner_offpolicy_mode="per_decision_is",
            inner_search_bootstrap_critic="target",
            inner_target_update_event="optimizer_step",
            inner_depth_update_order="mixed",
            inner_vtrace_distill_updates=1,
            inner_vtrace_distill_action_samples=1,
            outer_critic_target="reward_only",
            inner_sac_critic_target="reward_only",
        )

    eager = _model(dict(params, compile=False))
    compile_requests = []
    compiled_invocations = []

    def fake_compile(function, **kwargs):
        name = getattr(function, "__name__", type(function).__name__)
        compile_requests.append((name, kwargs))
        # Compiler construction is operational work and must not perturb the
        # scientific global RNG stream.
        torch.rand(())

        def identity_compiled(*args, **call_kwargs):
            compiled_invocations.append(name)
            return function(*args, **call_kwargs)

        return identity_compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    compiled = _model(dict(params, compile=True, compile_strict=True))
    observation = torch.tensor([0.125, -0.25, 0.5])
    try:
        global_rng_before = torch.random.get_rng_state().clone()
        eager_action = eager.agent.act(
            observation, t0=True, eval_mode=False, collect_diagnostics=False
        )
        torch.testing.assert_close(
            torch.random.get_rng_state(), global_rng_before, rtol=0, atol=0
        )
        compiled_action = compiled.agent.act(
            observation, t0=True, eval_mode=False, collect_diagnostics=False
        )
        torch.testing.assert_close(
            torch.random.get_rng_state(), global_rng_before, rtol=0, atol=0
        )

        torch.testing.assert_close(eager_action, compiled_action, rtol=0, atol=0)
        eager_metrics = eager.agent.last_inner_metrics
        compiled_metrics = compiled.agent.last_inner_metrics
        assert eager_metrics.keys() == compiled_metrics.keys()
        for key in eager_metrics:
            if key.endswith("_seconds"):
                continue
            _assert_nested_equal(eager_metrics[key], compiled_metrics[key])
        _assert_nested_equal(
            eager.agent.inner_engine.rng.training_state_dict(),
            compiled.agent.inner_engine.rng.training_state_dict(),
        )

        requested_names = {name for name, _ in compile_requests}
        assert expected_regions <= requested_names
        assert expected_regions <= set(compiled_invocations)
        for name, kwargs in compile_requests:
            if name in expected_regions:
                assert kwargs == {"fullgraph": True, "dynamic": False}
    finally:
        eager.env.close()
        compiled.env.close()


@pytest.mark.parametrize(
    ("kind", "overrides"),
    [
        ("q", {"inner_critic_horizon_mode": "depth_conditioned"}),
        (
            "q_lora",
            {
                "inner_critic_horizon_mode": "shared",
                "inner_actor_adaptation": "lora",
                "inner_critic_adaptation": "lora",
            },
        ),
        (
            "vtrace",
            {
                "inner_operator": "vtrace",
                "inner_q_objective": "finite_horizon",
                "inner_critic_horizon_mode": "stage_heads",
                "inner_return_estimator": "td0",
                "inner_return_lambda": 0.5,
                "inner_search_replay_retention": "round",
                "inner_offpolicy_mode": "per_decision_is",
                "inner_search_bootstrap_critic": "target",
                "inner_target_update_event": "optimizer_step",
                "inner_depth_update_order": "mixed",
                "inner_vtrace_distill_updates": 1,
                "inner_vtrace_distill_action_samples": 1,
                "outer_critic_target": "reward_only",
                "inner_sac_critic_target": "reward_only",
            },
        ),
    ],
)
def test_search_reuses_root_allocations_but_resets_scientific_state(
    kind, overrides, monkeypatch
):
    """Action locality must not require replacing Dynamo-guarded objects."""
    params = _finite_q_params("shared", "td0", **overrides)
    model = _model(params)
    engine = model.agent.inner_engine
    q_input = torch.randn(4, model.cfg.latent_dim + model.cfg.action_dim)
    observations = []
    original_prepare = engine._prepare_search_workspace

    def optimizer_is_reset(optimizer):
        if optimizer is None:
            return True
        return all(
            not torch.is_tensor(value) or bool((value == 0).all().item())
            for parameter_state in optimizer.state.values()
            for value in parameter_state.values()
        )

    def tracked_prepare(*, t0):
        result = original_prepare(t0=t0)
        state = engine.state
        record = {
            "ids": tuple(
                id(value)
                for value in (
                    state,
                    state.actor,
                    state.critic,
                    state.critic_target,
                    state.actor_optim,
                    state.critic_optim,
                    state.replay,
                )
            ),
            "critic_state": _clone_state(state.critic),
        }
        assert state.replay.size == 0
        assert state.replay.next_trajectory_id == 0
        assert state.replay.trajectory_id_offset == 0
        assert state.critic_steps == state.actor_steps == 0
        assert state.target_model_steps == state.value_distill_steps == 0
        assert optimizer_is_reset(state.actor_optim)
        assert optimizer_is_reset(state.critic_optim)
        if kind.startswith("q"):
            expected = model.agent.model._Qs(q_input)
            for depth in (1, 2, 3):
                torch.testing.assert_close(state.critic(q_input, depth), expected)
                torch.testing.assert_close(
                    state.critic_target(q_input, depth), expected
                )
        else:
            for online, target in zip(
                state.critic.parameters(), state.critic_target.parameters()
            ):
                torch.testing.assert_close(online, target, rtol=0, atol=0)
        observations.append(record)
        return result

    monkeypatch.setattr(engine, "_prepare_search_workspace", tracked_prepare)
    try:
        model.agent.act(
            torch.zeros(3), t0=True, eval_mode=False, collect_diagnostics=False
        )
        first_final = _clone_state(engine._action_pool.critic)
        with torch.no_grad():
            next(model.agent.model._pi.parameters()).add_(0.125)
            next(model.agent.model._Qs.parameters()).add_(0.25)
        model.agent.act(
            torch.ones(3), t0=False, eval_mode=False, collect_diagnostics=False
        )
        pooled_ids = tuple(
            id(value)
            for value in (
                engine._action_pool.actor,
                engine._action_pool.critic,
                engine._action_pool.critic_target,
                engine._action_pool.actor_optim,
                engine._action_pool.critic_optim,
                engine._action_pool.replay,
            )
        )
        workspace_id = id(engine.state)
        engine.reset_for_evaluation(991)
        assert id(engine.state) == workspace_id
        assert tuple(
            id(value)
            for value in (
                engine._action_pool.actor,
                engine._action_pool.critic,
                engine._action_pool.critic_target,
                engine._action_pool.actor_optim,
                engine._action_pool.critic_optim,
                engine._action_pool.replay,
            )
        ) == pooled_ids
        model.agent.act(
            torch.full((3,), 0.25),
            t0=True,
            eval_mode=True,
            collect_diagnostics=False,
        )

        assert len(observations) == 3
        assert observations[0]["ids"] == observations[1]["ids"]
        assert observations[1]["ids"] == observations[2]["ids"]
        if kind == "vtrace":
            # A new root receives a genuinely fresh random V initialization,
            # even though its live parameter objects are reused.
            assert any(
                not torch.equal(value, first_final[name])
                for name, value in observations[1]["critic_state"].items()
            )
    finally:
        model.env.close()


def test_successor_bootstrap_shapes_do_not_depend_on_depth_composition(
    monkeypatch,
):
    """Mixed V0/Vh rows must not grow dynamic=False compile caches."""
    model = _model(
        _finite_q_params(
            "depth_conditioned", "td0", inner_leaf_value_samples=2
        )
    )
    engine = model.agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_search_workspace(t0=True)

    leaf_row_counts = []
    inner_row_counts = []
    leaf_region = engine._compile_regions["search_leaf"]
    original_inner_value = engine._search_inner_value

    def tracked_leaf(flat_z, policy_noise, pair_indices):
        leaf_row_counts.append(flat_z.shape[0])
        return leaf_region(flat_z, policy_noise, pair_indices)

    def tracked_inner_value(
        z,
        remaining_horizon,
        *,
        generator,
        pair_indices=None,
        active_mask=None,
    ):
        inner_row_counts.append(z.shape[0])
        return original_inner_value(
            z,
            remaining_horizon,
            generator=generator,
            pair_indices=pair_indices,
            active_mask=active_mask,
        )

    engine._compile_regions["search_leaf"] = tracked_leaf
    monkeypatch.setattr(engine, "_search_inner_value", tracked_inner_value)
    next_z = torch.randn(4, model.cfg.latent_dim)
    compositions = (
        torch.tensor([0, 0, 0, 0]),
        torch.tensor([1, 2, 3, 1]),
        torch.tensor([0, 2, 0, 1]),
    )
    try:
        for depths in compositions:
            with engine.rng.fork("bootstrap") as generator:
                pair = model.agent.model.q_backend.sample_pair_indices(
                    model.agent.device, generator=generator
                )
                value = engine._search_successor_value(
                    next_z,
                    depths,
                    generator=generator,
                    pair_indices=pair,
                )
            assert value.shape == (4, 1)
            assert torch.isfinite(value).all()

        # Two outer-action samples are flattened for every one of the four
        # padded rows, and the inner critic likewise always sees four rows,
        # even for all-leaf or all-inner compositions.
        assert leaf_row_counts == [8, 8, 8]
        assert inner_row_counts == [4, 4, 4]
    finally:
        model.env.close()


@pytest.mark.parametrize("kind", ["q", "q_lora", "vtrace"])
@pytest.mark.skipif(
    os.environ.get("AMBI_RUN_REAL_COMPILE_TESTS") != "1",
    reason="real torch.compile regression is opt-in",
)
def test_strict_dynamo_reuses_search_graphs_across_three_actions(kind, monkeypatch):
    """A real fullgraph Dynamo backend must compile only on the first action."""
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    if kind == "q":
        params = _finite_q_params("depth_conditioned", "td0")
    elif kind == "q_lora":
        params = _finite_q_params(
            "shared",
            "td0",
            inner_actor_adaptation="lora",
            inner_critic_adaptation="lora",
        )
    else:
        params = _base_params(
            inner_operator="vtrace",
            inner_q_objective="finite_horizon",
            inner_critic_horizon_mode="stage_heads",
            inner_return_estimator="td0",
            inner_return_lambda=0.5,
            inner_search_replay_retention="round",
            inner_offpolicy_mode="per_decision_is",
            inner_search_bootstrap_critic="target",
            inner_target_update_event="optimizer_step",
            inner_depth_update_order="mixed",
            inner_vtrace_distill_updates=1,
            inner_vtrace_distill_action_samples=1,
            outer_critic_target="reward_only",
            inner_sac_critic_target="reward_only",
        )

    real_compile = torch.compile
    compiled_graphs = []

    def counted_compile(function, **kwargs):
        name = getattr(function, "__name__", type(function).__name__)
        owner = getattr(function, "__self__", None)
        if owner is not None:
            name = f"{name}:{id(owner)}"

        def backend(graph_module, example_inputs):
            del example_inputs
            compiled_graphs.append((name, graph_module))
            return graph_module.forward

        return real_compile(function, backend=backend, **kwargs)

    torch._dynamo.reset()
    monkeypatch.setattr(torch, "compile", counted_compile)
    model = _model(dict(params, compile=True, compile_strict=True))
    try:
        graph_counts = []
        for index in range(3):
            if index == 2:
                # Evaluation resets must reset scientific state and RNG, but
                # must not invalidate the action-local compiled graph cache.
                model.agent.inner_engine.reset_for_evaluation(991)
            action = model.agent.act(
                torch.full((3,), 0.1 * index),
                t0=index in {0, 2},
                eval_mode=False,
                collect_diagnostics=False,
            )
            assert torch.isfinite(action).all()
            graph_counts.append(len(compiled_graphs))
        assert graph_counts[0] > 0
        identities = [
            f"outer_online:{id(model.agent.model._Qs)}",
            f"outer_target:{id(model.agent.model._target_Qs)}",
            f"inner_online:{id(model.agent.inner_engine._action_pool.critic.ensemble) if kind != 'vtrace' else -1}",
            f"inner_target:{id(model.agent.inner_engine._action_pool.critic_target.ensemble) if kind != 'vtrace' else -1}",
        ]
        assert graph_counts[1:] == graph_counts[:1] * 2, "\n".join(
            [name for name, _ in compiled_graphs] + identities
        )
    finally:
        model.env.close()
        torch._dynamo.reset()


def test_finite_q_actor_step_does_not_accumulate_critic_gradients(monkeypatch):
    """The compile-safe Q forward must retain dQ/da but isolate Q parameters."""
    model = _model(_finite_q_params("depth_conditioned", "td0"))
    engine = model.agent.inner_engine
    observed = []
    original = engine._search_q_policy_step

    def tracked_policy_step(batch, *, update_actor, update_temperature):
        for parameter in engine.state.critic.parameters():
            parameter.grad = None
        result = original(
            batch,
            update_actor=update_actor,
            update_temperature=update_temperature,
        )
        observed.append(
            [parameter.grad for parameter in engine.state.critic.parameters()]
        )
        return result

    monkeypatch.setattr(engine, "_search_q_policy_step", tracked_policy_step)
    try:
        model.agent.act(torch.zeros(3), t0=True, eval_mode=False)
        assert observed
        assert all(gradient is None for call in observed for gradient in call)
        assert model.agent.last_inner_metrics["inner_actor_grad_norm"] > 0
    finally:
        model.env.close()


def test_uncorrected_noisy_suffix_records_the_executed_action_coordinate(
    monkeypatch,
):
    """Soft suffix entropy must be evaluated at the perturbed linked action."""
    model = _model(
        _finite_q_params(
            "shared",
            "n_step",
            inner_search_replay_retention="action",
            inner_offpolicy_mode="uncorrected",
            inner_behavior_action="mean_plus_gaussian",
            inner_behavior_noise_std=100.0,
        )
    )
    engine = model.agent.inner_engine
    captured = {}
    original = engine._run_search_round_updates

    def tracked_updates(**kwargs):
        replay = engine.state.replay
        valid = replay.valid[: replay.size]
        action = replay.action[: replay.size][valid.expand_as(replay.action[: replay.size])]
        action = action.reshape(-1, model.cfg.action_dim)
        pre_tanh = replay.pre_tanh_action[: replay.size][
            valid.expand_as(replay.pre_tanh_action[: replay.size])
        ].reshape(-1, model.cfg.action_dim)
        z = replay.z[: replay.size][
            valid.expand_as(replay.z[: replay.size])
        ].reshape(-1, model.cfg.latent_dim)
        stored_log_prob = replay.behavior_log_prob[: replay.size][valid].reshape(-1, 1)
        captured.update(
            action=action.detach().clone(),
            pre_tanh=pre_tanh.detach().clone(),
            stored_log_prob=stored_log_prob.detach().clone(),
            recomputed_log_prob=engine._search_behavior_log_prob(
                z, pre_tanh
            ).detach().clone(),
        )
        return original(**kwargs)

    monkeypatch.setattr(engine, "_run_search_round_updates", tracked_updates)
    try:
        model.agent.act(torch.zeros(3), t0=True, eval_mode=False)
        assert bool((captured["action"].abs() == 1).any().item())
        assert bool(torch.isfinite(captured["pre_tanh"]).all().item())
        torch.testing.assert_close(
            captured["pre_tanh"].tanh(),
            captured["action"].clamp(-1.0 + 1e-6, 1.0 - 1e-6),
        )
        torch.testing.assert_close(
            captured["stored_log_prob"], captured["recomputed_log_prob"]
        )
    finally:
        model.env.close()


def test_no_inner_control_executes_without_search_or_outer_mutation():
    params = _base_params()
    for key in (
        "inner_rounds",
        "inner_rollouts_per_round",
        "inner_critic_updates_per_round",
        "inner_actor_updates_per_round",
    ):
        params.pop(key)
    params.update(inner_operator="none", inner_q_objective="legacy_continuing")
    model = _model(params)
    outer_before = _clone_state(model.agent.model)

    action = model.agent.act(torch.zeros(3), t0=True, eval_mode=False)

    assert bool(torch.isfinite(action).all().item())
    assert model.agent.last_inner_metrics["inner_active"] == 0
    assert model.agent.last_inner_metrics["inner_model_steps"] == 0
    assert model.agent.last_inner_metrics["inner_critic_optimizer_steps"] == 0
    assert model.agent.last_inner_metrics["inner_actor_optimizer_steps"] == 0
    _assert_state_unchanged(model.agent.model, outer_before)


def test_explicit_legacy_objective_is_seeded_bitwise_default():
    implicit = _model(_legacy_params())
    explicit = _model(_legacy_params(inner_q_objective="legacy_continuing"))
    observation = torch.tensor([0.25, -0.5, 0.75])
    global_rng_before = torch.random.get_rng_state().clone()

    implicit_action = implicit.agent.act(observation, t0=True, eval_mode=False)
    global_rng_after_implicit = torch.random.get_rng_state().clone()
    explicit_action = explicit.agent.act(observation, t0=True, eval_mode=False)
    global_rng_after_explicit = torch.random.get_rng_state().clone()

    torch.testing.assert_close(implicit_action, explicit_action, rtol=0, atol=0)
    torch.testing.assert_close(global_rng_after_implicit, global_rng_before, rtol=0, atol=0)
    torch.testing.assert_close(global_rng_after_explicit, global_rng_before, rtol=0, atol=0)
    _assert_nested_equal(
        implicit.agent.inner_engine.rng.training_state_dict(),
        explicit.agent.inner_engine.rng.training_state_dict(),
    )
    assert implicit.agent.last_inner_metrics.keys() == (
        explicit.agent.last_inner_metrics.keys()
    )
    assert not any(
        key.startswith("inner_search_root_")
        for key in implicit.agent.last_inner_metrics
    )
    for key in implicit.agent.last_inner_metrics:
        if key.endswith("_seconds"):
            continue
        left = implicit.agent.last_inner_metrics[key]
        right = explicit.agent.last_inner_metrics[key]
        _assert_nested_equal(left, right)
