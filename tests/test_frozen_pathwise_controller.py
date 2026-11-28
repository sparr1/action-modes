from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from RL.tdmpc2_core.frozen_pathwise import FrozenPathwiseController
from tests.test_inner_entropy_graph import assert_tree_close, make_agent, outer_snapshot
from utils.frozen_pathwise_settings import pathwise_settings


def _agent():
    agent = make_agent("squashed", "action", "per_update")
    agent.cfg.inner_critic_updates_per_round = 0
    agent.cfg.inner_critic_updates_per_action = 0
    return agent


def _controller(agent, *, detach=False):
    return FrozenPathwiseController(
        SimpleNamespace(cfg=agent.cfg, agent=agent),
        rollouts_per_update=3, detach_dynamics=detach,
    )


def test_detach_dynamics_changes_gradient_only(monkeypatch):
    agent = _agent()
    world = agent.model
    dims = int(agent.cfg.latent_dim) + int(agent.cfg.action_dim)
    reward = torch.nn.Linear(dims, 1, bias=False)
    dynamics = torch.nn.Linear(dims, agent.cfg.latent_dim, bias=False)
    with torch.no_grad():
        reward.weight.fill_(0.2)
        dynamics.weight.fill_(0.15)
    monkeypatch.setattr(world, "_reward", reward)
    monkeypatch.setattr(world, "_dynamics", dynamics)
    monkeypatch.setattr(world, "decode_reward", lambda logits: logits)
    world.eval()
    root = torch.zeros(1, agent.cfg.latent_dim, requires_grad=True)
    noise = torch.zeros(agent.cfg.inner_rollout_horizon + 1, 3, agent.cfg.action_dim)
    pair = world.q_backend.sample_pair_indices(agent.device, generator=torch.Generator().manual_seed(55))

    outputs = []
    for detach in (False, True):
        controller = _controller(agent, detach=detach)
        try:
            returns, entropy = controller._rollout(root, noise, pair)
            total = (returns + agent.alpha.detach() * entropy).mean()
            total.backward()
            gradients = [p.grad.flatten() for p in controller._actor.parameters()]
            outputs.append((returns.detach(), entropy.detach(), torch.cat(gradients)))
        finally:
            controller.close()
    torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=0, atol=0)
    assert torch.linalg.vector_norm(outputs[0][2] - outputs[1][2]) > 0
    assert root.grad is None
    assert all(p.grad is None for p in world.parameters())


@pytest.mark.parametrize("detach", [False, True])
def test_controller_owns_fresh_actor_adam_and_private_rng(detach):
    agent = _agent()
    engine = agent.inner_engine
    original = engine.act
    parameter = next(agent.model.parameters())
    parameter.grad = torch.full_like(parameter, 2.0)
    saved_gradient = parameter.grad.clone()
    frozen = outer_snapshot(agent)
    root = torch.linspace(-0.3, 0.3, agent.cfg.latent_dim).reshape(1, -1)
    ambient_rng = torch.get_rng_state().clone()
    controller = _controller(agent, detach=detach)
    try:
        assert engine.act == controller.act
        assert controller.settings == pathwise_settings(vars(agent.cfg), rollouts_per_update=3, detach_dynamics=detach)
        assert json.loads(json.dumps(controller.settings)) == controller.settings
        ctrl_settings = controller.settings
        ctrl_settings["discount_rule"].clear()
        assert controller.settings["discount_rule"]
        controller.reset_episode(55)
        with torch.no_grad():
            action, metrics, lengths = engine.act(root, eval_mode=True, collect_diagnostics=False)
            stream = controller._rng.get_state()
            engine.act(root, eval_mode=True, collect_diagnostics=False)
            assert not torch.equal(stream, controller._rng.get_state())
            controller.reset_episode(55)
            repeated = engine.act(root, eval_mode=True, collect_diagnostics=False)

        assert_tree_close((action, metrics, lengths), repeated)
        assert controller.calls == 3
        assert lengths == [] and torch.isfinite(action).all() and (action.abs() <= 1).all()
        assert metrics["inner_actor_optimizer_steps"] == 2
        assert metrics["inner_critic_optimizer_steps"] == metrics["inner_temperature_optimizer_steps"] == 0
        assert metrics["inner_model_steps"] == 2 * 3 * agent.cfg.inner_rollout_horizon
        assert metrics["inner_actor_loss_scale_initial"].item() == 4.0
        assert metrics["inner_actor_loss_scale_final"].item() >= 1.0
        assert torch.isfinite(metrics["inner_actor_loss"])
        assert_tree_close(frozen, outer_snapshot(agent))
        torch.testing.assert_close(parameter.grad, saved_gradient, rtol=0, atol=0)
        torch.testing.assert_close(ambient_rng, torch.get_rng_state(), rtol=0, atol=0)
    finally:
        controller.close()
    assert engine.act == original


def test_pathwise_rejects_unsupported_saved_protocol_and_missing_seed():
    agent = _agent()
    agent.cfg.inner_critic_updates_per_round = 1
    original = agent.inner_engine.act
    with pytest.raises(ValueError, match="inner_critic_updates_per_round"):
        _controller(agent)
    assert agent.inner_engine.act == original
    agent.cfg.inner_critic_updates_per_round = 0
    controller = _controller(agent)
    try:
        root = torch.zeros(1, agent.cfg.latent_dim)
        with pytest.raises(ValueError, match="Seed"):
            controller.act(root, eval_mode=True, collect_diagnostics=False)
        controller.reset_episode(55)
        with pytest.raises(ValueError, match="mean execution"):
            controller.act(root, eval_mode=False, collect_diagnostics=False)
        with pytest.raises(ValueError, match="without traces"):
            controller.act(root, eval_mode=True, collect_diagnostics=False, trace=object())
    finally:
        controller.close()
