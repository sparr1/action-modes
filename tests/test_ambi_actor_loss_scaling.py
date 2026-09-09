import pytest
import torch

import RL.tdmpc2_core.ambi_agent as ambi_agent_module
from RL.tdmpc2_core.common import math as td_math
from RL.tdmpc2_core.common.scale import linear_percentiles, percentile_range
from tests.test_ambi_inner_decoupling import (
    _assert_tree_equal,
    _clone_tree,
    _model,
)


_MODE = "tdmpc2_percentile_range"


def _scaled_model(**overrides):
    params = {
        "sac_actor_loss_scale_mode": _MODE,
        "sac_actor_loss_scale_tau": 1.0,
        "ent_coef": 0.5,
        "outer_critic_target": "reward_only",
        "inner_sac_critic_target": "reward_only",
    }
    params.update(overrides)
    return _model(**params)


def _stub_actor_batch(monkeypatch, agent, *, log_prob=-0.25):
    steps = int(agent.cfg.train_unroll_horizon) + 1
    batch = int(agent.cfg.batch_size)
    assert batch == 4
    depth_zero = torch.tensor(
        [0.0, 10.0, 20.0, 30.0], device=agent.device
    ).reshape(1, batch, 1)
    q_values = depth_zero.repeat(steps, 1, 1)
    q_values[1:] *= 10.0

    def policy(z):
        anchor = next(agent.model._pi.parameters()).reshape(-1)[0]
        action = torch.zeros(
            *z.shape[:-1], agent.cfg.action_dim, device=z.device, dtype=z.dtype
        ) + anchor * 0.0
        sampled_log_prob = torch.full(
            (*z.shape[:-1], 1), log_prob, device=z.device, dtype=z.dtype
        ) + anchor * 0.0
        return action, {
            "log_prob": sampled_log_prob,
            "entropy": -sampled_log_prob,
        }

    def critic(z, action, **kwargs):
        value = q_values.to(device=z.device, dtype=z.dtype) + action[..., :1] * 0.0
        if kwargs.get("reduction") == "all":
            return value.unsqueeze(0).expand(
                int(agent.cfg.num_q), *value.shape
            )
        return value

    monkeypatch.setattr(agent.model, "pi", policy)
    monkeypatch.setattr(agent.model, "Q", critic)
    zs = torch.zeros(
        steps,
        batch,
        agent.cfg.latent_dim,
        device=agent.device,
    )
    return zs, q_values, log_prob


def test_shared_percentile_math_matches_tdmpc2_linear_interpolation_and_floor():
    values = torch.tensor([[0.0], [10.0], [20.0], [30.0], [40.0]])
    percentiles = torch.tensor([5.0, 95.0])

    torch.testing.assert_close(
        linear_percentiles(values, percentiles),
        torch.tensor([[2.0], [38.0]]),
    )
    torch.testing.assert_close(
        percentile_range(values, percentiles), torch.tensor([36.0])
    )
    torch.testing.assert_close(
        percentile_range(torch.ones(4, 1), percentiles), torch.ones(1)
    )


def test_actor_loss_scale_ema_is_detached_and_never_drops_below_one():
    model = _scaled_model(sac_actor_loss_scale_tau=0.25)
    agent = model.agent
    q_values = torch.tensor(
        [[0.0], [10.0], [20.0], [30.0]], requires_grad=True
    )

    agent._update_actor_loss_scale(q_values)

    # P95-P5 is 27, so one EMA step from 1 at tau=.25 is 7.5.
    torch.testing.assert_close(agent.actor_loss_scale, torch.tensor([7.5]))
    assert not agent.actor_loss_scale.requires_grad
    assert agent.actor_loss_scale.grad_fn is None
    assert q_values.grad is None

    agent._update_actor_loss_scale(torch.ones_like(q_values))
    # The instantaneous estimate is floored at one before the EMA update.
    torch.testing.assert_close(agent.actor_loss_scale, torch.tensor([5.875]))
    assert agent.actor_loss_scale.item() >= 1.0


@pytest.mark.parametrize("tau", [0.01, 0.25, 1.0])
def test_actor_scale_matches_independent_tdmpc2_estimator_over_updates(tau):
    agent = _scaled_model(sac_actor_loss_scale_tau=tau).agent
    reference_scale = torch.ones(1)
    # Independent equivalent of upstream RunningScale at commit 8bbc14e:
    # linear P95-P5, floor before lerp, and initialize the EMA to one.
    # torch.quantile avoids using AMBI's percentile implementation as its oracle.
    batches = (
        [30.0, -20.0, 7.0, 2.0, 100.0, -1.0, 5.0],
        [4.0],
        [0.0, 0.1, 0.2, 0.3],
        [100.0, -100.0, 40.0, 40.0],
    )
    for values in batches:
        q_values = torch.tensor(values, requires_grad=True).reshape(-1, 1)
        reference_percentiles = torch.quantile(
            q_values.detach(), torch.tensor([0.05, 0.95]), dim=0,
            interpolation="linear",
        )
        reference_range = (
            reference_percentiles[1] - reference_percentiles[0]
        ).clamp(min=1.0)
        reference_scale.lerp_(reference_range, tau)

        instantaneous_range = agent._update_actor_loss_scale(q_values)

        torch.testing.assert_close(instantaneous_range, reference_range)
        torch.testing.assert_close(agent.actor_loss_scale, reference_scale)
        assert agent.actor_loss_scale.grad_fn is None
        assert not agent.actor_loss_scale.requires_grad


def test_disabled_scaler_preserves_v3_structure_metrics_and_skips_scale_math(
    monkeypatch,
):
    model = _model()
    agent = model.agent
    checkpoint = agent.checkpoint_state()
    assert checkpoint["checkpoint_version"] == 3
    assert "actor_loss_scale_spec" not in checkpoint
    assert "actor_loss_scale_state" not in checkpoint
    assert agent.actor_loss_scale is None
    assert not agent.actor_loss_scale_enabled

    def forbidden(*_args, **_kwargs):
        raise AssertionError("disabled actor scaling must not compute percentiles")

    monkeypatch.setattr(ambi_agent_module, "percentile_range", forbidden)
    zs, _, _ = _stub_actor_batch(monkeypatch, agent)
    metrics = agent._update_actor(zs)
    assert "actor_loss_scale" not in metrics
    assert "actor_effective_ent_coef" not in metrics


def test_outer_actor_logs_same_forward_per_action_ensemble_gap(monkeypatch):
    model = _model(outer_q_actor_reduction="min_all")
    agent = model.agent
    zs, _, _ = _stub_actor_batch(monkeypatch, agent)
    steps, batch = zs.shape[:2]
    head_zero = torch.arange(
        steps * batch,
        device=agent.device,
        dtype=zs.dtype,
    ).reshape(steps, batch, 1)
    head_one = head_zero + torch.linspace(
        2.0,
        8.0,
        steps * batch,
        device=agent.device,
        dtype=zs.dtype,
    ).reshape(steps, batch, 1)
    q_all = torch.stack((head_zero, head_one))
    calls = []

    def critic(z, action, **kwargs):
        calls.append(dict(kwargs))
        return q_all.to(device=z.device, dtype=z.dtype) + action[..., :1] * 0.0

    monkeypatch.setattr(agent.model, "Q", critic)
    metrics = agent._update_actor(zs)

    q_mean_all = q_all.mean(dim=0)
    q_min_all = q_all.min(dim=0).values
    assert calls == [{"reduction": "all", "detach": True}]
    torch.testing.assert_close(metrics["actor_q_mean"], q_min_all.mean())
    torch.testing.assert_close(metrics["actor_q_mean_all"], q_mean_all.mean())
    torch.testing.assert_close(metrics["actor_q_min_all"], q_min_all.mean())
    torch.testing.assert_close(
        metrics["actor_q_mean_all_minus_min_all"],
        (q_mean_all - q_min_all).mean(),
    )
    for key in (
        "actor_q_mean_all",
        "actor_q_min_all",
        "actor_q_mean_all_minus_min_all",
    ):
        assert not metrics[key].requires_grad
        assert metrics[key].grad_fn is None


def test_enabled_scaler_updates_from_depth_zero_and_scales_only_q(
    monkeypatch,
):
    model = _scaled_model(ent_coef=0.5)
    agent = model.agent
    zs, q_values, sampled_log_prob = _stub_actor_batch(monkeypatch, agent)
    initial_alpha = agent.alpha.detach().clone()

    metrics = agent._update_actor(zs)

    # With B=4, P5 and P95 interpolate to 1.5 and 28.5 respectively.
    expected_scale = torch.tensor([27.0], device=agent.device)
    torch.testing.assert_close(agent.actor_loss_scale, expected_scale)
    torch.testing.assert_close(metrics["actor_loss_scale"], expected_scale)
    torch.testing.assert_close(
        metrics["actor_effective_ent_coef"], initial_alpha.expand_as(expected_scale)
    )

    log_probs = torch.full_like(q_values, sampled_log_prob)
    per_time = (initial_alpha * log_probs - q_values / expected_scale).mean(
        dim=(1, 2)
    )
    expected_actor_loss = td_math.reduce_temporal_loss(
        per_time,
        agent.cfg.rho,
        include_terminal=True,
        legacy_order="vector_mean",
    )
    torch.testing.assert_close(metrics["actor_loss"], expected_actor_loss)

    full_objective_scaled = (
        (initial_alpha * log_probs - q_values) / expected_scale
    ).mean(dim=(1, 2))
    full_objective_scaled = td_math.reduce_temporal_loss(
        full_objective_scaled,
        agent.cfg.rho,
        include_terminal=True,
        legacy_order="vector_mean",
    )
    assert not torch.isclose(metrics["actor_loss"], full_objective_scaled)

    assert agent.log_ent_coef is None
    torch.testing.assert_close(agent.alpha, initial_alpha)


@pytest.mark.parametrize("horizon", [1, 2, 3, 4, 6])
@pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
def test_outer_actor_gradients_match_tdmpc2_q_only_scaling_formula(
    monkeypatch, horizon, rho
):
    agent = _scaled_model(
        ent_coef=0.5,
        rho=rho,
        train_unroll_horizon=horizon,
        inner_rollout_horizon=1,
        grad_clip_norm=1e6,
    ).agent
    steps = int(agent.cfg.train_unroll_horizon) + 1
    batch = int(agent.cfg.batch_size)
    feature = torch.arange(steps * batch, dtype=torch.float32).reshape(
        steps, batch, 1
    ) / 10.0
    q_offsets = torch.tensor([0.0, 10.0, 20.0, 30.0]).reshape(1, batch, 1)
    parameter = next(agent.model._pi.parameters())
    initial_theta = parameter.detach().reshape(-1)[:2].clone()

    def policy(z):
        theta = parameter.reshape(-1)[:2]
        action = theta[0] * feature + theta[1]
        log_prob = (theta[0] + 0.4).square() * (feature + 1.0) + theta[1]
        return action, {"log_prob": log_prob, "entropy": -log_prob}

    def critic(z, action, **kwargs):
        q_values = q_offsets + 2.0 * action
        assert kwargs["reduction"] == "all"
        return q_values.unsqueeze(0).expand(int(agent.cfg.num_q), -1, -1, -1)

    monkeypatch.setattr(agent.model, "pi", policy)
    monkeypatch.setattr(agent.model, "Q", critic)
    monkeypatch.setattr(agent.pi_optim, "step", lambda: None)
    zs = torch.zeros(steps, batch, agent.cfg.latent_dim)

    # Upstream update_pi scales Q before forming -(alpha * entropy + Q).
    # Supply identical entropy/Q tensors to isolate the scaling placement;
    # AMBI's distinct entropy estimator is deliberately outside this comparison.
    theta = initial_theta.requires_grad_()
    reference_action = theta[0] * feature + theta[1]
    reference_q = q_offsets + 2.0 * reference_action
    reference_log_prob = (
        (theta[0] + 0.4).square() * (feature + 1.0) + theta[1]
    )
    quantiles = torch.quantile(
        reference_q[0].detach(), torch.tensor([0.05, 0.95]), dim=0
    )
    reference_scale = (quantiles[1] - quantiles[0]).clamp(min=1.0)
    assert reference_scale.item() > 1.0
    rho_weights = torch.tensor([rho**t for t in range(steps)])
    reference_loss = (
        (0.5 * reference_log_prob - reference_q / reference_scale)
        .mean(dim=(1, 2)) * rho_weights
    ).mean()
    reference_gradient = torch.autograd.grad(reference_loss, theta)[0]

    metrics = agent._update_actor(zs)

    torch.testing.assert_close(metrics["actor_loss"], reference_loss.detach())
    torch.testing.assert_close(agent.actor_loss_scale, reference_scale)
    torch.testing.assert_close(parameter.grad.reshape(-1)[:2], reference_gradient)
    assert torch.count_nonzero(reference_gradient) == 2


def test_outer_temperature_uses_normalized_actor_temporal_weights(monkeypatch):
    model = _model(ent_coef="auto_0.5", rho=0.5)
    agent = model.agent
    zs, _, _ = _stub_actor_batch(monkeypatch, agent)
    depth_log_probs = torch.tensor(
        [-0.25, -1.25, -3.25], device=agent.device
    )
    assert depth_log_probs.shape[0] == zs.shape[0]

    def policy(z):
        anchor = next(agent.model._pi.parameters()).reshape(-1)[0]
        action = torch.zeros(
            *z.shape[:-1], agent.cfg.action_dim, device=z.device, dtype=z.dtype
        ) + anchor * 0.0
        sampled_log_prob = depth_log_probs.to(dtype=z.dtype).reshape(-1, 1, 1)
        sampled_log_prob = (
            sampled_log_prob.expand(*z.shape[:-1], 1) + anchor * 0.0
        )
        return action, {
            "log_prob": sampled_log_prob,
            "entropy": -sampled_log_prob,
        }

    monkeypatch.setattr(agent.model, "pi", policy)
    initial_log_alpha = agent.log_ent_coef.detach().clone()

    metrics = agent._update_actor(zs)

    normalized_weights = torch.pow(
        torch.as_tensor(agent.cfg.rho, device=agent.device),
        torch.arange(zs.shape[0], device=agent.device),
    )
    normalized_weights = normalized_weights / normalized_weights.sum()
    entropy_residuals = depth_log_probs + agent.target_entropy
    expected_temperature_loss = -(
        initial_log_alpha * (normalized_weights * entropy_residuals).sum()
    ).mean()
    uniform_temperature_loss = -(
        initial_log_alpha * entropy_residuals.mean()
    ).mean()
    torch.testing.assert_close(
        metrics["ent_coef_loss"], expected_temperature_loss
    )
    assert not torch.isclose(
        metrics["ent_coef_loss"], uniform_temperature_loss
    )


def test_enabled_scaler_preserves_raw_reward_only_bellman_target(monkeypatch):
    model = _scaled_model(ent_coef=0.5)
    agent = model.agent
    agent.actor_loss_scale.fill_(19.0)
    log_prob = -0.4
    next_q = 7.0

    def policy(z):
        action = torch.zeros(*z.shape[:-1], 1, device=z.device)
        sampled_log_prob = torch.full((*z.shape[:-1], 1), log_prob, device=z.device)
        return action, {"log_prob": sampled_log_prob}

    def critic(z, _action, **_kwargs):
        return torch.full((*z.shape[:-1], 1), next_q, device=z.device)

    monkeypatch.setattr(agent.model, "pi", policy)
    monkeypatch.setattr(agent.model, "Q", critic)
    next_z = torch.zeros(2, 4, agent.cfg.latent_dim, device=agent.device)
    reward = torch.full((2, 4, 1), 2.0, device=agent.device)
    terminated = torch.zeros_like(reward)

    target = agent._soft_td_target(next_z, reward, terminated)
    expected = reward + agent.discount * next_q
    torch.testing.assert_close(target, expected)
    torch.testing.assert_close(
        agent.actor_loss_scale, torch.tensor([19.0], device=agent.device)
    )


def test_enabled_checkpoint_and_exact_state_roundtrip_non_unit_scale():
    source = _scaled_model()
    source.agent.actor_loss_scale.fill_(3.25)
    checkpoint = _clone_tree(source.agent.checkpoint_state())
    assert checkpoint["checkpoint_version"] == 4
    assert checkpoint["actor_loss_scale_spec"] == {
        "mode": _MODE,
        "application": "q_only",
        "source": "decoded_outer_actor_q_depth0",
        "reduction": source.cfg.outer_q_actor_reduction,
        "percentiles": [5.0, 95.0],
        "tau": 1.0,
        "floor": 1.0,
    }

    portable = _scaled_model()
    portable.agent.load(checkpoint)
    torch.testing.assert_close(portable.agent.actor_loss_scale, torch.tensor([3.25]))

    source.agent.prepare_training_resume_boundary()
    exact = _clone_tree(source.agent.training_state_dict())
    assert exact["version"] == 4
    assert exact["outer"]["checkpoint_version"] == 4
    restored = _scaled_model()
    restored.agent.load_training_state_dict(exact)
    _assert_tree_equal(restored.agent.training_state_dict(), exact)


@pytest.mark.parametrize(
    ("corruption", "expected"),
    [
        ("spec", "specification"),
        ("missing", "incompatible schema"),
        ("shape", "shape"),
        ("dtype", "dtype"),
        ("nan", "finite"),
        ("below-floor", "floor"),
        ("percentiles", "percentiles differ"),
    ],
)
def test_enabled_scale_checkpoint_validation_is_transactional(corruption, expected):
    source = _scaled_model()
    source.agent.actor_loss_scale.fill_(4.0)
    invalid = _clone_tree(source.agent.checkpoint_state())
    if corruption == "spec":
        invalid["actor_loss_scale_spec"]["source"] = "wrong"
    elif corruption == "missing":
        invalid["actor_loss_scale_state"].pop("value")
    elif corruption == "shape":
        invalid["actor_loss_scale_state"]["value"] = torch.ones(2)
    elif corruption == "dtype":
        invalid["actor_loss_scale_state"]["value"] = torch.ones(
            1, dtype=torch.float64
        )
    elif corruption == "nan":
        invalid["actor_loss_scale_state"]["value"].fill_(float("nan"))
    elif corruption == "below-floor":
        invalid["actor_loss_scale_state"]["value"].fill_(0.5)
    else:
        invalid["actor_loss_scale_state"]["percentiles"][0] = 4.0

    target = _scaled_model()
    target.agent.actor_loss_scale.fill_(8.0)
    pristine = _clone_tree(target.agent.training_state_dict())
    with pytest.raises((TypeError, ValueError), match=expected):
        target.agent.load(invalid)
    _assert_tree_equal(target.agent.training_state_dict(), pristine)


@pytest.mark.parametrize("legacy_version", [1, 2, 3])
def test_legacy_structured_checkpoints_require_scaling_off_and_raw_weights_reset_scale(
    legacy_version,
):
    legacy_source = _model()
    legacy = _clone_tree(legacy_source.agent.checkpoint_state())
    legacy["checkpoint_version"] = legacy_version

    legacy_restored = _model()
    legacy_restored.agent.load(legacy)
    _assert_tree_equal(
        legacy_restored.agent.model.state_dict(),
        legacy_source.agent.model.state_dict(),
    )

    enabled = _scaled_model()
    enabled.agent.actor_loss_scale.fill_(7.0)
    pristine = _clone_tree(enabled.agent.training_state_dict())
    with pytest.raises(ValueError, match="version-4"):
        enabled.agent.load(legacy)
    _assert_tree_equal(enabled.agent.training_state_dict(), pristine)

    enabled.agent.load(_clone_tree(legacy_source.agent.model.state_dict()))
    torch.testing.assert_close(enabled.agent.actor_loss_scale, torch.ones(1))


def test_exact_scale_corruption_rejects_before_outer_or_inner_mutation():
    source = _scaled_model()
    source.agent.actor_loss_scale.fill_(4.0)
    source.agent.prepare_training_resume_boundary()
    invalid = _clone_tree(source.agent.training_state_dict())
    invalid["outer"]["actor_loss_scale_state"]["value"].fill_(float("inf"))

    target = _scaled_model()
    target.agent.actor_loss_scale.fill_(6.0)
    pristine = _clone_tree(target.agent.training_state_dict())
    with pytest.raises(ValueError, match="finite"):
        target.agent.load_training_state_dict(invalid)
    _assert_tree_equal(target.agent.training_state_dict(), pristine)


@pytest.mark.parametrize("exact_resume", [False, True])
def test_old_full_objective_scale_checkpoint_is_rejected_transactionally(exact_resume):
    source = _scaled_model()
    source.agent.actor_loss_scale.fill_(4.0)
    source.agent.prepare_training_resume_boundary()
    if exact_resume:
        invalid = _clone_tree(source.agent.training_state_dict())
        outer = invalid["outer"]
    else:
        invalid = _clone_tree(source.agent.checkpoint_state())
        outer = invalid
    outer["actor_loss_scale_spec"]["application"] = "full_sac_actor_objective"

    target = _scaled_model()
    target.agent.actor_loss_scale.fill_(8.0)
    pristine = _clone_tree(target.agent.training_state_dict())
    load = (
        target.agent.load_training_state_dict if exact_resume else target.agent.load
    )
    with pytest.raises(ValueError, match="actor-loss scale specification"):
        load(invalid)
    _assert_tree_equal(target.agent.training_state_dict(), pristine)


def test_v4_checkpoint_rejects_scaling_off_before_mutation():
    source = _scaled_model()
    checkpoint = _clone_tree(source.agent.checkpoint_state())
    target = _model()
    pristine = _clone_tree(target.agent.training_state_dict())

    with pytest.raises(ValueError, match="scaling is enabled"):
        target.agent.load(checkpoint)

    _assert_tree_equal(target.agent.training_state_dict(), pristine)
