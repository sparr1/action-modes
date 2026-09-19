"""One-hot inner inputs preserve prior networks and ordinary compiled paths."""

from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.common.horizon_conditioning import (
    pack_horizon_input, reset_horizon_module, widen_horizon_module,
)
from RL.tdmpc2_core.common.layers import NormedLinear
from RL.tdmpc2_core.common.soft_world_model import SoftWorldModel
from tests.test_q_representation import model_cfg


def _model(representation="scalar"):
    torch.manual_seed(29)
    model = SoftWorldModel(model_cfg(q_representation=representation))
    # Learned priors have nonzero output kernels; the fresh model's zero critic
    # kernels would conceal conditioning and gradient-routing errors.
    with torch.no_grad():
        for head in model._Qs:
            head[-1].weight.normal_(std=0.1)
    return model.eval()


@pytest.mark.parametrize("horizon", [1, 3])
@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_horizon_inputs_preserve_every_prior_api_without_changing_rng(horizon, representation):
    model = _model(representation)
    outer_before = deepcopy(model.state_dict())
    actor, critic = deepcopy(model._pi), deepcopy(model._Qs)
    rng_before = torch.random.get_rng_state().clone()
    assert widen_horizon_module(actor, horizon) is actor
    assert widen_horizon_module(critic, horizon) is critic
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
    assert actor.horizon_conditioning_horizon == horizon
    assert critic.horizon_conditioning_horizon == horizon
    assert all(head.horizon_conditioning_horizon == horizon for head in critic)
    assert type(actor[0]) is NormedLinear
    assert all(type(head[0]) is NormedLinear for head in critic)

    z = torch.randn(2, 4, model.cfg.latent_dim)
    actions = torch.randn(2, 4, model.cfg.action_dim)
    noise = torch.randn_like(actions)
    expected_action, expected_info = model.pi(z, noise=noise)
    expected_q = model.q_predictions(z, actions)
    for remaining in range(1, horizon + 1):
        h = torch.full((*z.shape[:-1], 1), remaining, dtype=torch.long)
        kwargs = {"policy": actor, "remaining_horizon": h}
        action, info = model.pi(z, noise=noise, **kwargs)
        torch.testing.assert_close(action, expected_action)
        for name, value in info.items():
            torch.testing.assert_close(value, expected_info[name])
        torch.testing.assert_close(model.pi_action(z, noise=noise, **kwargs), expected_action)
        for name, value in model.policy_stats(z, **kwargs).items():
            torch.testing.assert_close(value, model.policy_stats(z)[name])
        native, native_info = model.pi_tdmpc2(z, noise=noise, **kwargs)
        native_prior, native_prior_info = model.pi_tdmpc2(z, noise=noise)
        torch.testing.assert_close(native, native_prior)
        for name, value in native_info.items():
            torch.testing.assert_close(value, native_prior_info[name])

        critic_kwargs = {"qs": critic, "remaining_horizon": h}
        predictions = model.q_predictions(z, actions, **critic_kwargs)
        torch.testing.assert_close(predictions, expected_q)
        torch.testing.assert_close(
            model.q_predictions_from_joint(model.joint_input(z, actions), **critic_kwargs), expected_q,
        )
        torch.testing.assert_close(model.q_values(z, actions, **critic_kwargs), model.q_values(z, actions))
        torch.testing.assert_close(model.Q(z, actions, **critic_kwargs), model.Q(z, actions))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, outer_before[name], rtol=0, atol=0)


@pytest.mark.parametrize("component", ["actor", "critic"])
def test_reset_restores_current_prior_and_zeroes_columns_in_existing_storage(component):
    model = _model()
    prior = model._pi if component == "actor" else model._Qs
    adapted = widen_horizon_module(deepcopy(prior), 3)
    parameter_ids = [id(parameter) for parameter in adapted.parameters()]
    storage = [parameter.data_ptr() for parameter in adapted.parameters()]
    with torch.no_grad():
        for parameter in adapted.parameters():
            parameter.add_(0.7)
        for parameter in prior.parameters():
            parameter.add_(0.2)
    rng_before = torch.random.get_rng_state().clone()
    assert reset_horizon_module(adapted, prior) is adapted
    assert [id(parameter) for parameter in adapted.parameters()] == parameter_ids
    assert [parameter.data_ptr() for parameter in adapted.parameters()] == storage
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
    adapted_heads = [adapted] if component == "actor" else list(adapted)
    prior_heads = [prior] if component == "actor" else list(prior)
    for head, prior_head in zip(adapted_heads, prior_heads):
        for name, value in head.state_dict().items():
            expected = prior_head.state_dict()[name]
            if name == "0.weight":
                torch.testing.assert_close(value[:, -3:], torch.zeros_like(value[:, -3:]), rtol=0, atol=0)
                value = value[:, :-3]
            torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_each_network_has_independent_columns_and_only_sampled_horizons_receive_gradient():
    model = _model()
    actor = widen_horizon_module(deepcopy(model._pi), 3)
    critic = widen_horizon_module(deepcopy(model._Qs), 3)
    z = torch.randn(4, model.cfg.latent_dim)
    action = torch.randn(4, model.cfg.action_dim)
    h = torch.full((4, 1), 2, dtype=torch.long)
    stats = model.policy_stats(z, policy=actor, remaining_horizon=h)
    stats["pre_tanh_mean"].sum().backward()
    model.q_predictions(z, action, qs=critic, remaining_horizon=h).sum().backward()
    first_layers = [actor[0], *(head[0] for head in critic)]
    assert len({layer.weight.data_ptr() for layer in first_layers}) == len(first_layers)
    for layer in first_layers:
        gradient = layer.weight.grad[:, -3:]
        assert gradient[:, 1].abs().sum() > 0
        torch.testing.assert_close(gradient[:, [0, 2]], torch.zeros_like(gradient[:, [0, 2]]), rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_nonzero_horizon_columns_change_state_dependent_predictions_and_actor_gradients():
    model = _model()
    actor = widen_horizon_module(deepcopy(model._pi), 3)
    critic = widen_horizon_module(deepcopy(model._Qs), 3)
    with torch.no_grad():
        for head in critic:
            head[0].weight[:, -3:].normal_(std=0.5)
        actor[0].weight[:, -3:].normal_(std=0.5)
    z = torch.randn(4, model.cfg.latent_dim)
    h1, h3 = torch.ones(4, 1, dtype=torch.long), torch.full((4, 1), 3, dtype=torch.long)
    action = model.pi_action(z, policy=actor, deterministic=True, remaining_horizon=h1)
    other_action = model.pi_action(z, policy=actor, deterministic=True, remaining_horizon=h3)
    assert not torch.allclose(action, other_action)
    first = model.Q(z, action, qs=critic, detach=True, remaining_horizon=h1)
    other = model.Q(z, action, qs=critic, detach=True, remaining_horizon=h3)
    assert not torch.allclose(first, other)
    first.sum().backward()
    assert actor[0].weight.grad[:, -3].abs().sum() > 0
    assert all(parameter.grad is None for parameter in critic.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_horizon_contract_rejects_missing_malformed_and_out_of_range_values():
    model = _model()
    actor = widen_horizon_module(deepcopy(model._pi), 3)
    z = torch.zeros(2, model.cfg.latent_dim)
    h = torch.ones(2, 1, dtype=torch.long)
    with pytest.raises(ValueError, match="requires remaining_horizon"):
        model.pi(z, policy=actor)
    with pytest.raises(ValueError, match="unconditioned"):
        model.pi(z, remaining_horizon=h)
    with pytest.raises(ValueError, match="unconditioned"):
        model.Q(z, torch.zeros(2, model.cfg.action_dim), remaining_horizon=h)
    with pytest.raises(ValueError, match="leading axes"):
        pack_horizon_input(actor, z, h.squeeze(-1))
    with pytest.raises(TypeError, match="integer dtype"):
        pack_horizon_input(actor, z, h.float())
    with pytest.raises(TypeError, match="integer tensor"):
        pack_horizon_input(actor, z, 2)
    for invalid in (0, 4):
        with pytest.raises(RuntimeError):
            pack_horizon_input(actor, z, torch.full_like(h, invalid))
    with pytest.raises(ValueError, match="already horizon conditioned"):
        widen_horizon_module(actor, 3)
    with pytest.raises(ValueError, match="positive integer"):
        widen_horizon_module(deepcopy(model._pi), True)
    assert pack_horizon_input(model._pi, z) is z


def test_strict_compilation_preserves_actor_gradients_through_detached_horizon_critics():
    torch._dynamo.reset()
    model = _model()
    eager_actor = widen_horizon_module(deepcopy(model._pi), 3)
    eager_critic = widen_horizon_module(deepcopy(model._Qs), 3)
    actor, critic = deepcopy(eager_actor), deepcopy(eager_critic)
    z = torch.randn(4, model.cfg.latent_dim)
    h = torch.tensor([[1], [2], [3], [1]])
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    def objective(selected_actor, selected_critic, latent, remaining):
        action = model.pi_action(
            latent, policy=selected_actor, deterministic=True, remaining_horizon=remaining,
        )
        return model.Q(
            latent, action, qs=selected_critic, detach=True,
            remaining_horizon=remaining, reduction="mean_all",
        ).mean()

    def kernel(latent, remaining):
        return objective(actor, critic, latent, remaining)

    compiled = torch.compile(kernel, backend=backend, fullgraph=True, dynamic=False)
    try:
        for remaining in (h, h.flip(0)):
            for module in (actor, eager_actor):
                module.zero_grad(set_to_none=True)
            expected = objective(eager_actor, eager_critic, z, remaining)
            actual = compiled(z, remaining)
            torch.testing.assert_close(actual, expected)
            expected.backward()
            actual.backward()
            for expected_parameter, parameter in zip(eager_actor.parameters(), actor.parameters()):
                torch.testing.assert_close(parameter.grad, expected_parameter.grad)
            assert actor[0].weight.grad[:, -3:].abs().sum() > 0
        assert len(graphs) == 1
        assert all(parameter.grad is None for parameter in critic.parameters())
        assert all(parameter.grad is None for parameter in model.parameters())
    finally:
        torch._dynamo.reset()
