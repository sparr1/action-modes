import math

import torch

from RL.tdmpc2_core.common import math as td_math
from tests.test_ambi_inner_decoupling import _model


def _set_saturated_outer_actor(agent, *, mean=50.0):
    """Make the real outer policy saturate while keeping its Gaussian resolvable."""

    action_dim = int(agent.cfg.action_dim)
    with torch.no_grad():
        for parameter in agent.model._pi.parameters():
            parameter.zero_()
        agent.model._pi[-1].bias[:action_dim].fill_(float(mean))
        # Unit pre-tanh standard deviation preserves the sampled epsilon in
        # float32 while the large mean still rounds every action to +1.
        agent.model._pi[-1].bias[action_dim:].zero_()


def _zero_online_critics(agent):
    with torch.no_grad():
        for parameter in agent.model._Qs.parameters():
            parameter.zero_()


def _reference_log_prob(policy_info):
    distribution = torch.distributions.Normal(
        policy_info["pre_tanh_mean"],
        policy_info["log_std"].exp(),
    )
    gaussian_log_prob = distribution.log_prob(
        policy_info["pre_tanh_action"]
    ).sum(dim=-1, keepdim=True)
    return gaussian_log_prob - td_math.tanh_log_abs_det_jacobian(
        policy_info["pre_tanh_action"]
    )


def test_outer_soft_target_uses_exact_saturated_policy_density(monkeypatch):
    model = _model(
        ent_coef=0.25,
        outer_critic_target="entropy_augmented",
    )
    agent = model.agent
    _set_saturated_outer_actor(agent)
    captured = {}
    original_pi = agent.model.pi

    def capture_policy(*args, **kwargs):
        action, info = original_pi(*args, **kwargs)
        captured["calls"] = captured.get("calls", 0) + 1
        captured["action"] = action.detach().clone()
        captured["info"] = {
            key: value.detach().clone() for key, value in info.items()
        }
        return action, info

    def constant_target_q(z, action, **_kwargs):
        return (z[..., :1] + action[..., :1] * 0.0).fill_(4.0)

    monkeypatch.setattr(agent.model, "pi", capture_policy)
    monkeypatch.setattr(agent.model, "Q", constant_target_q)
    next_z = torch.zeros(2, agent.cfg.latent_dim, device=agent.device)
    reward = torch.tensor([[1.0], [2.0]], device=agent.device)
    terminated = torch.tensor([[0.0], [1.0]], device=agent.device)

    torch.manual_seed(71)
    actual = agent._soft_td_target(next_z, reward, terminated)
    info = captured["info"]
    reference_log_prob = _reference_log_prob(info)
    expected = reward + agent.discount * (1.0 - terminated) * (
        4.0 - agent.alpha.detach() * reference_log_prob
    )

    assert captured["calls"] == 1
    assert torch.all(captured["action"].abs() == 1.0)
    torch.testing.assert_close(info["log_prob"], reference_log_prob)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[1], reward[1], rtol=0, atol=0)

    # This explicit historical formula makes the regression discriminative:
    # merely checking finiteness would allow the old +1e-6 density floor.
    legacy_jacobian = torch.log(
        torch.relu(1.0 - captured["action"].square()) + 1e-6
    ).sum(dim=-1, keepdim=True)
    gaussian_log_prob = reference_log_prob + td_math.tanh_log_abs_det_jacobian(
        info["pre_tanh_action"]
    )
    legacy_log_prob = gaussian_log_prob - legacy_jacobian
    legacy_target = reward + agent.discount * (1.0 - terminated) * (
        4.0 - agent.alpha.detach() * legacy_log_prob
    )
    assert not torch.isclose(actual[0], legacy_target[0]).item()


def test_outer_reward_only_target_ignores_policy_density(monkeypatch):
    model = _model(
        ent_coef=0.25,
        outer_critic_target="reward_only",
    )
    agent = model.agent

    def policy_with_poisoned_density(z):
        action = z.new_zeros(*z.shape[:-1], agent.cfg.action_dim)
        return action, {
            "log_prob": z.new_full((*z.shape[:-1], 1), float("nan")),
        }

    def constant_target_q(z, action, **_kwargs):
        return (z[..., :1] + action[..., :1] * 0.0).fill_(3.0)

    monkeypatch.setattr(agent.model, "pi", policy_with_poisoned_density)
    monkeypatch.setattr(agent.model, "Q", constant_target_q)
    next_z = torch.zeros(2, agent.cfg.latent_dim, device=agent.device)
    reward = torch.tensor([[1.0], [2.0]], device=agent.device)
    terminated = torch.tensor([[0.0], [1.0]], device=agent.device)

    actual = agent._soft_td_target(next_z, reward, terminated)
    expected = reward + agent.discount * (1.0 - terminated) * 3.0

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


def test_outer_actor_alpha_and_saturation_metrics_share_one_exact_sample(
    monkeypatch,
):
    model = _model(
        ent_coef="auto_0.5",
        ent_coef_lr=3e-4,
        target_entropy="auto",
    )
    agent = model.agent
    _set_saturated_outer_actor(agent)
    _zero_online_critics(agent)
    steps = int(agent.cfg.train_unroll_horizon) + 1
    zs = torch.zeros(
        steps,
        int(agent.cfg.batch_size),
        int(agent.cfg.latent_dim),
        device=agent.device,
    )

    original_pi = agent.model.pi
    torch.manual_seed(113)
    rng_before = torch.random.get_rng_state().clone()
    original_pi(zs)
    rng_after_one_policy_sample = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(rng_before)

    captured = {"calls": 0}

    def capture_policy(*args, **kwargs):
        action, info = original_pi(*args, **kwargs)
        captured["calls"] += 1
        captured["action"] = action.detach().clone()
        captured["info"] = {
            key: value.detach().clone() for key, value in info.items()
        }
        return action, info

    monkeypatch.setattr(agent.model, "pi", capture_policy)
    mean_bias = agent.model._pi[-1].bias[: int(agent.cfg.action_dim)]
    mean_bias_before = mean_bias.detach().clone()
    log_alpha_before = agent.log_ent_coef.detach().clone()

    metrics = agent._update_actor(zs)
    info = captured["info"]
    action = captured["action"]
    reference_log_prob = _reference_log_prob(info)

    assert captured["calls"] == 1
    torch.testing.assert_close(
        torch.random.get_rng_state(),
        rng_after_one_policy_sample,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(info["log_prob"], reference_log_prob)
    torch.testing.assert_close(
        metrics["actor_entropy"],
        -reference_log_prob.mean(),
    )
    assert torch.isfinite(metrics["actor_loss"])
    assert torch.isfinite(metrics["actor_grad_norm"])
    assert metrics["actor_grad_norm"] > 0.0
    assert torch.all(mean_bias.detach() < mean_bias_before)

    entropy_residual_per_time = (
        reference_log_prob + float(agent.target_entropy)
    ).mean(dim=(1, 2))
    entropy_weights = agent._actor_temporal_weights.to(
        dtype=entropy_residual_per_time.dtype
    )
    entropy_weights = entropy_weights / entropy_weights.sum()
    weighted_residual = (entropy_residual_per_time * entropy_weights).sum()
    expected_alpha_loss = -(log_alpha_before * weighted_residual).mean()
    torch.testing.assert_close(metrics["ent_coef_loss"], expected_alpha_loss)
    assert agent.log_ent_coef.detach() > log_alpha_before

    expected_saturation = td_math.tanh_saturation_statistics(
        info["pre_tanh_action"], action
    )
    metric_names = (
        "actor_pre_tanh_abs_mean",
        "actor_pre_tanh_abs_max",
        "actor_pre_tanh_abs_ge_7p6_fraction",
        "actor_action_exact_saturation_fraction",
    )
    for name, expected_value in zip(metric_names, expected_saturation):
        torch.testing.assert_close(metrics[name], expected_value)
        assert not metrics[name].requires_grad
    assert metrics["actor_pre_tanh_abs_ge_7p6_fraction"] == 1.0
    assert metrics["actor_action_exact_saturation_fraction"] == 1.0

    # The old floor would report a much smaller log-density at this action.
    legacy_correction = torch.log(
        torch.relu(1.0 - action.square()) + 1e-6
    ).sum(dim=-1, keepdim=True)
    gaussian_log_prob = reference_log_prob + td_math.tanh_log_abs_det_jacobian(
        info["pre_tanh_action"]
    )
    legacy_log_prob = gaussian_log_prob - legacy_correction
    assert not torch.allclose(reference_log_prob, legacy_log_prob)
    assert math.isfinite(float(metrics["actor_pre_tanh_abs_max"]))
