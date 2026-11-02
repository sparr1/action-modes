"""Training diagnostics report physical semantics without changing learning."""

import math

import pytest
import torch

from RL.tdmpc2_core.common.policy_diagnostics import (
    entropy_coefficient_diagnostics,
    policy_sample_diagnostics,
)
from tests.test_ambi_actor_loss_scaling import _scaled_model, _stub_actor_batch
from tests.test_ambi_root_local_sac import _tiny_model
from tests.test_split_value_outer import stub_actor
from RL.tdmpc2_core.inner_trace import metric_catalog


def test_joint_entropy_and_saturation_reuse_sample_without_rng_or_gradients():
    action = torch.tensor([[1., -.99, 0.], [-1., .98, .5]], requires_grad=True)
    mean = torch.tensor([[.999, -1., 0.], [.5, -.5, 0.]], requires_grad=True)
    log_std = torch.tensor([[0., 1., -1.], [.5, .5, .5]], requires_grad=True)
    log_prob = torch.tensor([[2.], [4.]], requires_grad=True)
    rng_before = torch.random.get_rng_state().clone()
    metrics = policy_sample_diagnostics(action, {
        "mean": mean, "log_std": log_std, "log_prob": log_prob,
    })
    assert metrics["actor_pre_tanh_entropy"].item() == pytest.approx(
        3 * .5 * math.log(2 * math.pi * math.e) + .75
    )
    assert metrics["actor_post_tanh_entropy"].item() == -3.
    assert metrics["actor_action_abs_ge_0p99_fraction"].item() == .5
    assert metrics["actor_mean_action_abs_ge_0p99_fraction"].item() == pytest.approx(2 / 6)
    assert metrics["actor_mean_action_exact_saturation_fraction"].item() == pytest.approx(1 / 6)
    assert all(value.ndim == 0 and not value.requires_grad for value in metrics.values())
    assert all(value.grad is None for value in (action, mean, log_std, log_prob))
    assert torch.equal(rng_before, torch.random.get_rng_state())


@pytest.mark.parametrize("mode,multiplier", [("squashed", 1), ("tdmpc2_scaled", 21)])
@pytest.mark.parametrize("alpha_value", [0., 1e-4])
def test_coefficients_convert_to_return_units_and_snapshot(mode, multiplier, alpha_value):
    alpha = torch.tensor(alpha_value, requires_grad=True)
    scale = torch.tensor([7.], requires_grad=True)
    metrics = entropy_coefficient_diagnostics(alpha, scale, mode, 21)
    with torch.no_grad():
        alpha.fill_(2.)
        scale.fill_(100.)
    assert metrics["actor_entropy_coefficient_normalized"].item() == pytest.approx(alpha_value)
    assert metrics["actor_entropy_coefficient_return_units"].item() == pytest.approx(alpha_value * 7)
    assert metrics["actor_entropy_coefficient_joint_units"].item() == pytest.approx(alpha_value * 7 * multiplier)
    assert all(not value.requires_grad for value in metrics.values())


def test_outer_diagnostics_report_just_updated_scale_and_snapshot(monkeypatch):
    holder = _scaled_model()
    try:
        agent = holder.agent
        zs, _, _ = _stub_actor_batch(monkeypatch, agent)
        metrics = agent._update_actor(zs)
        assert metrics["actor_post_tanh_entropy"].item() == .25
        assert "actor_pre_tanh_entropy" not in metrics  # Lightweight policy stub.
        assert metrics["actor_entropy_coefficient_return_units"].item() == pytest.approx(13.5)
        assert metrics["actor_q_range_instant"].item() == pytest.approx(27.)
        agent._actor_loss_scale_value.fill_(99.)
        assert metrics["actor_loss_scale"].item() == pytest.approx(27.)
        assert metrics["actor_q_range_ema"].item() == pytest.approx(27.)
    finally:
        holder.close()


def test_python_coefficient_stays_on_scale_device_and_dtype():
    scale = torch.empty((), device="meta", dtype=torch.float64)
    metrics = entropy_coefficient_diagnostics(1e-4, scale, "tdmpc2_scaled", 21)
    assert all(value.device.type == "meta" for value in metrics.values())
    assert all(value.dtype == torch.float64 for value in metrics.values())


def test_split_value_coefficient_uses_entry_scale_not_next_scale(monkeypatch):
    holder = _tiny_model(
        critic_value_mode="return_entropy", q_representation="distributional",
        ent_coef=.5, sac_actor_loss_scale_mode="tdmpc2_percentile_range",
        sac_actor_loss_scale_tau=1., batch_size=4,
    )
    try:
        agent = holder.agent
        agent._actor_loss_scale_value.fill_(4.)
        zs, _ = stub_actor(agent, monkeypatch)
        metrics = agent._update_actor(zs)
        assert metrics["actor_loss_scale_next"].item() == pytest.approx(27.)
        assert metrics["actor_entropy_coefficient_return_units"].item() == pytest.approx(2.)
    finally:
        holder.close()


def test_inner_diagnostic_units_are_explicit():
    catalog = metric_catalog()
    for prefix in ("actor_", "explorer_actor_"):
        assert catalog[prefix + "pre_tanh_entropy"]["unit"] == "nats"
        assert catalog[prefix + "post_tanh_entropy"]["unit"] == "nats"
        assert catalog[prefix + "entropy_coefficient_joint_units"]["unit"] == "value_per_nat"
        assert catalog[prefix + "q_percentile_range_over_scale"]["unit"] == "ratio"


def test_inner_policy_metrics_use_optimizer_counts_and_scale_gauges_use_decisions():
    holder = _tiny_model()
    try:
        holder.agent.last_inner_metrics = {
            "inner_actor_optimizer_steps": 3,
            "inner_explorer_actor_optimizer_steps": 5,
            "inner_actor_post_tanh_entropy": -4.,
            "inner_actor_entropy_coefficient_return_units": .2,
            "inner_actor_q_percentile_range_over_scale": 2.,
            "inner_explorer_actor_pre_tanh_entropy": 1.,
            "inner_explorer_actor_mean_action_abs_ge_0p99_fraction": .3,
            "inner_actor_loss_scale_final_over_outer": 1.5,
        }
        holder._record_action_metrics(planned=True, action_seconds=0.)
        metrics = holder._wandb_train_window.snapshot()
        for name, count in (
            ("inner_actor_post_tanh_entropy", 3),
            ("inner_actor_entropy_coefficient_return_units", 3),
            ("inner_actor_q_percentile_range_over_scale", 3),
            ("inner_explorer_actor_pre_tanh_entropy", 5),
            ("inner_explorer_actor_mean_action_abs_ge_0p99_fraction", 5),
            ("inner_actor_loss_scale_final_over_outer", 1),
        ):
            assert metrics[f"train/{name}"] == pytest.approx(holder.agent.last_inner_metrics[name])
            assert metrics[f"train/{name}_count"] == count
    finally:
        holder.close()
