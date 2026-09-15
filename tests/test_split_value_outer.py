"""Outer component Bellman, optimization, and coefficient-timing contracts."""

from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.common import math as td_math
from tests.test_ambi_root_local_sac import _tiny_model


@pytest.fixture
def models():
    opened = []

    def create(**overrides):
        options = dict(
            critic_value_mode="return_entropy", q_representation="distributional",
            ent_coef=.5, inner_rounds=1, inner_updates_per_round=1,
            q_num_bins=11, num_bins=11, batch_size=4,
        )
        options.update(overrides)
        model = _tiny_model(**options)
        opened.append(model)
        return model.agent

    yield create
    for model in opened:
        model.env.close()


def batch(agent):
    h, b = agent.cfg.train_unroll_horizon, agent.cfg.batch_size
    return (torch.randn(h + 1, b, 3), torch.randn(h, b, 1).tanh(),
            torch.randn(h, b, 1), torch.zeros(h, b, 1))


def nonzero_outputs(agent):
    with torch.no_grad():
        generator = torch.Generator().manual_seed(471)
        for member in agent.model._Qs:
            member[-1].weight.copy_(torch.randn(member[-1].weight.shape, generator=generator) * .06)
        agent.model._target_Qs.load_state_dict(agent.model._Qs.state_dict())


def test_outer_targets_share_action_member_and_detach_all_inputs(models, monkeypatch):
    agent = models(discount=.9)
    calls = []

    def policy(z):
        calls.append("policy")
        return z[..., :1] * 0 + .75, {"log_prob": z[..., :1] * 0 - 2.}

    def values(z, action, **kwargs):
        calls.append(kwargs)
        torch.testing.assert_close(action, torch.full_like(action, .75))
        # Individually minimizing the return would choose the wrong member.
        return z.new_tensor([[[[0.], [10.]], [[0.], [10.]]],
                             [[[5.], [0.]], [[5.], [0.]]]])

    monkeypatch.setattr(agent.model, "pi", policy)
    monkeypatch.setattr(agent.model, "q_values", values)
    result = agent._soft_td_target(
        torch.zeros(2, agent.cfg.latent_dim, requires_grad=True),
        torch.tensor([[2.], [3.]], requires_grad=True), torch.tensor([[0.], [1.]]),
        alpha_snapshot=torch.tensor(.5, requires_grad=True), scale_snapshot=torch.tensor(4.),
    )
    torch.testing.assert_close(result, torch.tensor([[[6.5], [1.8]], [[3.], [0.]]]))
    assert calls == ["policy", {"target": True}]
    assert not result.requires_grad


def stub_actor(agent, monkeypatch):
    shape = (agent.cfg.train_unroll_horizon + 1, agent.cfg.batch_size, 1)
    returns = torch.tensor([0., 10., 20., 30.]).reshape(1, 4, 1).expand(shape).clone()
    returns[1:] *= 100.

    def policy(z):
        anchor = next(agent.model._pi.parameters()).flatten()[0]
        action = z[..., :1] * 0 + anchor * 0
        logprob = action - .25
        return action, {"log_prob": logprob, "entropy": -logprob}

    def values(z, action, **kwargs):
        assert kwargs == {"detach": True}
        first = torch.stack((returns, torch.full_like(returns, 4.)), dim=-2)
        second = torch.stack((returns + 4., torch.zeros_like(returns)), dim=-2)
        return torch.stack((first, second)) + action.unsqueeze(-2) * 0

    monkeypatch.setattr(agent.model, "pi", policy)
    monkeypatch.setattr(agent.model, "q_values", values)
    return torch.zeros(*shape[:-1], agent.cfg.latent_dim), returns


def test_actor_uses_entry_scale_then_updates_depth_zero_selected_return(models, monkeypatch):
    agent = models(sac_actor_loss_scale_mode="tdmpc2_percentile_range", sac_actor_loss_scale_tau=1.)
    agent._actor_loss_scale_value.fill_(4.)
    alpha, scale = agent._outer_value_coefficients()
    zs, returns = stub_actor(agent, monkeypatch)
    metrics = agent._update_actor(zs, alpha_snapshot=alpha, scale_snapshot=scale)
    expected = td_math.reduce_temporal_loss(
        (-.125 - (returns + 4.) / 4.).mean((1, 2)), agent.cfg.rho,
        include_terminal=True, legacy_order="vector_mean",
    )
    torch.testing.assert_close(metrics["actor_loss"], expected)
    assert metrics["actor_loss_scale_used"].item() == 4.
    assert metrics["actor_loss_scale_next"].item() == pytest.approx(27.)
    assert metrics["actor_beta_used"].item() == 2.
    assert metrics["actor_q_entropy"].item() == 0.
    # Changing the EMA must not mutate the tensor used during this backward.
    assert scale.item() == 4.
    next_metrics = agent._update_actor(zs)
    assert next_metrics["actor_loss_scale_used"].item() == pytest.approx(27.)
    assert next_metrics["actor_beta_used"].item() == pytest.approx(13.5)


def test_update_snapshots_once_before_critic_and_actor(models, monkeypatch):
    agent = models(sac_actor_loss_scale_mode="tdmpc2_percentile_range")
    agent._actor_loss_scale_value.fill_(3.)
    captured = []
    target, actor = agent._soft_td_target, agent._update_actor

    def target_spy(*args, **kwargs):
        captured.append(deepcopy(kwargs))
        result = target(*args, **kwargs)
        # Deliberately mutate live state between target and actor as a guard
        # against accidental aliases or recomputing coefficients mid-update.
        agent._actor_loss_scale_value.fill_(100.)
        agent.fixed_ent_coef.fill_(.9)
        return result

    def actor_spy(*args, **kwargs):
        captured.append({k: v.clone() for k, v in kwargs.items() if k.endswith("snapshot")})
        return actor(*args, **kwargs)

    monkeypatch.setattr(agent, "_soft_td_target", target_spy)
    monkeypatch.setattr(agent, "_update_actor", actor_spy)
    metrics = agent._update(*batch(agent))
    for used in captured:
        assert used["alpha_snapshot"].item() == .5
        assert used["scale_snapshot"].item() == 3.
    assert metrics["outer_beta_used"].item() == 1.5
    assert metrics["actor_beta_used"].item() == 1.5


def test_auto_alpha_update_is_isolated_and_effective_next_slot(models, monkeypatch):
    agent = models(ent_coef="auto_0.5")
    zs, _ = stub_actor(agent, monkeypatch)
    old_alpha = agent.alpha.detach().clone()
    metrics = agent._update_actor(zs)
    torch.testing.assert_close(metrics["actor_alpha_used"], old_alpha.reshape(()))
    assert metrics["actor_loss_scale_used"].item() == 1.
    assert agent.log_ent_coef.grad.item() == pytest.approx(.25 - agent.target_entropy)
    assert agent.alpha.item() != old_alpha.item()
    assert all(p.grad is None for p in agent.model._Qs.parameters())
    next_metrics = agent._update_actor(zs)
    torch.testing.assert_close(next_metrics["actor_alpha_used"], metrics["actor_alpha_next"])


@pytest.mark.parametrize("component", [0, 1])
def test_outer_component_loss_routes_to_encoder_and_dynamics(models, component):
    agent = models()
    nonzero_outputs(agent)
    obs, action, reward, terminated = batch(agent)
    next_z = agent.model.encode(obs[1:]).detach()
    targets = torch.ones(*reward.shape[:-1], 2, 1)
    result = agent._outer_update_kernel(obs[0], action, reward, terminated, next_z, targets)
    agent.model.critic_loss(result[2], targets, reduction="none")[..., component, :].mean().backward()
    for module in (agent.model._encoder, agent.model._dynamics):
        assert sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in agent.model._pi.parameters())
    assert all(p.grad is None for p in agent.model._target_Qs.parameters())


def test_outer_actor_freezes_critic_and_latents_with_nonzero_action_gradient(models):
    agent = models()
    nonzero_outputs(agent)
    zs = torch.randn(3, 4, agent.cfg.latent_dim, requires_grad=True)
    before = deepcopy(agent.model._Qs.state_dict())
    metrics = agent._update_actor(zs)
    assert metrics["actor_grad_norm"].item() > 0
    assert zs.grad is None
    assert all(p.grad is None for p in agent.model._Qs.parameters())
    for name, value in agent.model._Qs.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_outer_kernel_compiled_parity_and_component_loss_average(models, backend):
    agent = models()
    nonzero_outputs(agent)
    obs, action, reward, terminated = batch(agent)
    next_z = agent.model.encode(obs[1:]).detach()
    targets = agent._soft_td_target(next_z, reward, terminated)
    arguments = (obs[0], action, reward, terminated, next_z, targets)
    eager = agent._outer_update_kernel(*arguments)
    compiled = torch.compile(agent._outer_update_kernel, backend=backend, fullgraph=True)(*arguments)
    for left, right in zip(eager, compiled):
        if left is not None:
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    expected = []
    for component in (0, 1):
        losses = agent.model.critic_loss(eager[2], targets, reduction="none")[..., component, :]
        expected.append(td_math.reduce_temporal_loss(losses.mean((0, 2, 3)), agent.cfg.rho, legacy_order="vector_sum_divide"))
    torch.testing.assert_close(eager[7], torch.stack(expected).mean())


def test_update_reward_and_value_diagnostics_use_new_codec(models):
    agent = models(value_equivalence_diagnostics=True, value_equivalence_every_updates=1,
                   value_equivalence_mc_samples=2, inner_value_initialization="soft")
    nonzero_outputs(agent)
    with torch.no_grad():
        agent.model._reward[-1].bias.copy_(torch.linspace(-2, 2, agent.cfg.num_bins))
    info = agent._update(*batch(agent))
    assert "ve_prior_reward_rmse" in info
    assert all(torch.isfinite(torch.as_tensor(value)).all() for value in info.values())
    torch.testing.assert_close(info["critic_loss"], (info["critic_return_loss"] + info["critic_entropy_loss"]) / 2)
