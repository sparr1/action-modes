"""Auxiliary reward evaluation must not perturb detached primary learning."""

from copy import deepcopy

import pytest
import torch

from test_ambixqc_core import _batch
from test_ambixqc_prior_checkpoint import wrappers
from test_ambixqc_replay_archive import _assert_equal, _resident_rows, _rng_state


def _primary_state(agent):
    state = deepcopy(agent.checkpoint_state())
    state.pop("aux_return", None)
    state["semantic_signature"].pop("aux_return")
    state["module"] = {
        key: value for key, value in state["module"].items()
        if not key.startswith("aux_return.")
    }
    return state


@pytest.mark.parametrize("operator", ["none", "xqc"])
@pytest.mark.parametrize("ratio", [1, 2])
def test_detached_auxiliary_preserves_seeded_primary_training(wrappers, operator, ratio):
    outcomes = []
    for mode in ("off", "xqc"):
        model = wrappers(inner_operator=operator, aux_return_mode=mode, xqc_utd=ratio)
        # Gym otherwise creates OS-seeded streams lazily before learn() applies
        # the configured seed, so initialize those streams for this comparison.
        model.env.reset(seed=3)
        model.env.action_space.seed(3)
        initial = _primary_state(model.agent)
        initial_rng = _rng_state(model)
        model.learn(total_timesteps=10)
        outcomes.append({
            "initial": initial, "initial_rng": initial_rng,
            "trained": _primary_state(model.agent), "rng": _rng_state(model),
            "replay": _resident_rows(model.buffer),
            "replay_accounting": model.buffer._accounting_state(),
        })
        if mode == "xqc":
            assert model.agent.aux_return.update_step == model.agent.num_updates * ratio > 0
            assert model.agent.aux_return.critic is not model.agent.xqc_controller.critic
            assert not any("actor" in key or "temperature" in key
                           for key in model.agent.aux_return.state_dict())
    _assert_equal(outcomes[0], outcomes[1])


@pytest.mark.parametrize("detached", [False, True])
def test_auxiliary_representation_gradient_is_explicit(wrappers, detached):
    model = wrappers(aux_return_mode="xqc", aux_return_detach_representation=detached)
    agent = model.agent
    obs, action, reward, terminated = _batch(agent)
    with torch.no_grad():
        next_z = agent.model.encode(obs[1:])
    losses = agent._recurrent_world_and_value_losses(obs, action, reward, terminated, next_z)
    world_params = list(agent.model._encoder.parameters()) + list(agent.model._dynamics.parameters())
    gradients = torch.autograd.grad(
        losses["aux_return_loss"], world_params, allow_unused=True, retain_graph=True,
    )
    if detached:
        assert all(gradient is None for gradient in gradients)
    else:
        assert any(gradient is not None and bool(gradient.abs().sum() > 0) for gradient in gradients)
    critic_gradients = torch.autograd.grad(
        losses["aux_return_loss"], tuple(agent.aux_return.critic.parameters()),
    )
    assert any(bool(gradient.abs().sum() > 0) for gradient in critic_gradients)


@pytest.mark.parametrize("detached", [True, False])
def test_auxiliary_metrics_and_unscaled_optimizer_gradient(wrappers, detached):
    results = []
    for coefficient in (0.1, 0.7):
        model = wrappers(
            aux_return_mode="xqc", aux_return_critic_coef=coefficient,
            aux_return_detach_representation=detached,
        )
        info = model.agent._update(*_batch(model.agent))
        for key in (
            "aux_return_critic_loss", "aux_return_q_mean", "aux_return_q_target_mean",
            "aux_return_q_target_clip_fraction", "aux_return_grad_norm",
            "aux_return_learning_rate", "aux_return_target_updated", "aux_return_num_updates",
        ):
            assert key in info
            assert bool(torch.isfinite(torch.as_tensor(info[key])).all())
        results.append(deepcopy(model.agent.aux_return.state_dict()))
    for key in results[0]:
        torch.testing.assert_close(results[0][key], results[1][key], rtol=1e-5, atol=1e-6)


def test_shared_update_routes_weighted_world_and_unscaled_auxiliary_gradients(wrappers):
    settings = dict(
        aux_return_mode="xqc", aux_return_detach_representation=False,
        aux_return_critic_coef=0.3, grad_clip_norm=1e12,
    )
    reference, actual = wrappers(**settings), wrappers(**settings)
    obs, action, reward, terminated = _batch(reference.agent)
    with torch.no_grad():
        next_z = reference.agent.model.encode(obs[1:])
    losses = reference.agent._recurrent_world_and_value_losses(
        obs, action, reward, terminated, next_z,
    )
    expected_world = torch.autograd.grad(
        losses["total_loss"], tuple(reference.agent._world_params), retain_graph=True,
    )
    expected_auxiliary = torch.autograd.grad(
        losses["aux_return_loss"], tuple(reference.agent.aux_return.critic.parameters()),
    )
    actual.agent._update(obs, action, reward, terminated)
    for expected, parameter in zip(expected_world, actual.agent._world_params):
        torch.testing.assert_close(parameter.grad, expected, rtol=1e-5, atol=1e-7)
    for expected, parameter in zip(expected_auxiliary, actual.agent.aux_return.critic.parameters()):
        torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
