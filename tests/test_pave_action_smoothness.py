"""Action-gradient consistency has separate coordinates and PAVE reductions."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from RL.tdmpc2_core import pave
from RL.tdmpc2_core.common.q_representation import QRepresentation
from tests.test_pave_math import Ensemble, ObservationModel, Polynomial, identity, tensor


def test_quadratic_action_smoothness_matches_hessian_and_parameter_gradients():
    critic = Ensemble(Polynomial(1., kind="quadratic"),
                      Polynomial(-2., kind="quadratic"))
    action = tensor([[.2, -.3], [.7, .4]])
    perturbed = tensor([[.3, -.5], [.4, .45]])
    delta_gradient = (perturbed - action) @ tensor([[1., 2.], [2., 3.]])
    per_head = delta_gradient.square().mean()
    loss, metrics, components = pave.critic_pave_loss(
        critic, identity, torch.zeros_like(action), action,
        noisy_action=perturbed, action_smoothness_coef=.3, return_components=True,
    )
    expected = (1. + 4.) * per_head
    torch.testing.assert_close(components["action_smoothness"], expected)
    torch.testing.assert_close(metrics["action_smoothness_loss"], expected)
    torch.testing.assert_close(loss, .3 * expected)
    torch.testing.assert_close(metrics["action_smoothness_weighted_loss"], loss.detach())
    assert all(not metric.requires_grad for metric in metrics.values())
    loss.backward()
    torch.testing.assert_close(critic.heads[0].coefficient.grad, .6 * per_head)
    torch.testing.assert_close(critic.heads[1].coefficient.grad, -1.2 * per_head)


@pytest.mark.parametrize("kind", ["linear", "constant", "bilinear"])
def test_action_affine_critics_have_zero_smoothness_with_backwardable_loss(kind):
    critic = Ensemble(Polynomial(2., kind=kind), Polynomial(-2., kind=kind))
    action = tensor([[.3, .7], [-.2, .4]])
    loss, metrics = pave.critic_pave_loss(
        critic, identity, tensor([[1., 2.], [3., 4.]]), action,
        noisy_action=action + .1, action_smoothness_coef=1.,
    )
    torch.testing.assert_close(loss, tensor(0.))
    torch.testing.assert_close(metrics["action_smoothness_loss"], tensor(0.))
    loss.backward()
    assert all(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
               for parameter in critic.parameters())


class CategoricalHead(nn.Module):
    def __init__(self, coefficient):
        super().__init__()
        self.coefficient = nn.Parameter(tensor(coefficient))

    def forward(self, joint):
        logit = self.coefficient * joint[..., -2:-1]
        return torch.cat((torch.zeros_like(logit), logit), dim=-1)


def test_distributional_smoothness_differentiates_decoded_raw_q_not_logits():
    decoder = QRepresentation("distributional", num_q=2, num_bins=2,
                              vmin=-1., vmax=1., codec="symexp_mean")
    critic = Ensemble(CategoricalHead(1.), CategoricalHead(-2.))
    action = tensor([[.2, -.3], [.7, .4]])
    perturbed = tensor([[.3, -.1], [.4, .5]])
    support_width = 2 * torch.expm1(tensor(1.))
    expected = tensor(0.)
    for coefficient in (1., -2.):
        p = (coefficient * action[:, 0]).sigmoid()
        pp = (coefficient * perturbed[:, 0]).sigmoid()
        # Only action coordinate 0 affects logits. The mean still includes both.
        delta = support_width * coefficient * (p * (1 - p) - pp * (1 - pp))
        expected += delta.square().mean() / action.shape[-1]
    loss, _ = pave.critic_pave_loss(
        critic, decoder.decode, torch.zeros_like(action), action,
        noisy_action=perturbed, action_smoothness_coef=1.,
    )
    torch.testing.assert_close(loss, expected)
    assert loss.item() > 0  # Raw logits are affine and would incorrectly give zero.
    loss.backward()
    assert all(head.coefficient.grad.abs().item() > 0 for head in critic.heads)


def test_clipped_action_view_uses_independent_leaf_and_only_updates_selected_critic():
    encoder = nn.Linear(2, 2, dtype=torch.float64)
    actor = nn.Linear(2, 2, dtype=torch.float64)
    obs = tensor([[.4, .5], [.7, .2]], requires_grad=True)
    with torch.no_grad():
        actor.weight.zero_()
        actor.bias.copy_(tensor([1., -1.]))
    action = actor(obs)
    perturbed = (action + tensor([[.4, .2], [.3, -.1]])).clamp(-1., 1.)
    critic = Ensemble(Polynomial(1., kind="quadratic", hessian=[[2., 0.], [0., 3.]]))
    unselected = Ensemble(Polynomial(4., kind="quadratic"))
    loss, _ = pave.critic_pave_loss(
        critic, identity, encoder(obs), action,
        noisy_action=perturbed, action_smoothness_coef=1.,
    )
    # Only one of four coordinates moves: its gradient changes by 3 * .2.
    torch.testing.assert_close(loss, tensor(.6 ** 2 / 4))
    loss.backward()
    torch.testing.assert_close(critic.heads[0].coefficient.grad, tensor(2 * .6 ** 2 / 4))
    assert obs.grad is None
    assert all(parameter.grad is None for module in (encoder, actor, unselected)
               for parameter in module.parameters())


def test_action_smoothness_can_coexist_without_changing_existing_pave_components():
    critic = Ensemble(Polynomial(1., kind="state_quadratic"),
                      Polynomial(-2., kind="state_quadratic"))
    z, action = tensor([[1., 2.], [3., 4.]]), tensor([[.2, -.3], [.7, .4]])
    options = dict(noisy_z=z + .2, next_z=z + .5, probe=torch.ones_like(action),
                   mpr_coef=.1, vfc_coef=.2, curvature_coef=.3, return_components=True)
    legacy, legacy_metrics, legacy_components = pave.critic_pave_loss(
        critic, identity, z, action, **options,
    )
    combined, metrics, components = pave.critic_pave_loss(
        critic, identity, z, action, noisy_action=action + .05,
        action_smoothness_coef=.4, **options,
    )
    for name, expected in legacy_components.items():
        torch.testing.assert_close(components[name], expected, rtol=0, atol=0)
        torch.testing.assert_close(metrics[f"{name}_loss"], legacy_metrics[f"{name}_loss"],
                                   rtol=0, atol=0)
    torch.testing.assert_close(combined, legacy + .4 * components["action_smoothness"])


@pytest.mark.parametrize("perturbed", [None, tensor([[0.]])])
def test_enabled_action_smoothness_requires_aligned_perturbed_actions(perturbed):
    critic = Ensemble(Polynomial())
    with pytest.raises(ValueError, match="perturbed action"):
        pave.critic_pave_loss(
            critic, identity, tensor([[0., 0.]]), tensor([[.1, .2]]),
            noisy_action=perturbed, action_smoothness_coef=.1,
        )


def test_action_noise_is_private_reproducible_and_preserves_earlier_sampling_draws():
    model = ObservationModel()
    obs = torch.arange(64, dtype=torch.float64).reshape(4, 4, 4)[..., :2]
    action = torch.zeros(3, 4, 2, dtype=torch.float64, requires_grad=True)
    kwargs = dict(seed=91, update_number=5, batch_size=7)
    rng = torch.random.get_rng_state().clone()
    legacy = pave.prepare_pave_inputs(model, obs, action, **kwargs)
    enabled = pave.prepare_pave_inputs(model, obs, action, needs_action_noise=True, **kwargs)
    repeated = pave.prepare_pave_inputs(model, obs, action, needs_action_noise=True, **kwargs)
    doubled = pave.prepare_pave_inputs(model, obs, action, needs_action_noise=True,
                                      action_noise_std=.04, **kwargs)
    clipped = pave.prepare_pave_inputs(model, obs, action, needs_action_noise=True,
                                      action_noise_std=10., **kwargs)
    for name in ("z", "action", "noisy_z", "next_z", "probe"):
        torch.testing.assert_close(enabled[name], legacy[name], rtol=0, atol=0)
    for name, value in enabled.items():
        torch.testing.assert_close(value, repeated[name], rtol=0, atol=0)
        assert not value.requires_grad
    assert legacy["noisy_action"] is None
    torch.testing.assert_close(doubled["noisy_action"], 2 * enabled["noisy_action"])
    torch.testing.assert_close(clipped["noisy_action"],
                               (500 * enabled["noisy_action"]).clamp(-1, 1))
    assert (clipped["noisy_action"].abs() == 1).any()
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


def test_zero_action_smoothness_preserves_old_loss_and_ignores_action_view():
    critic = Ensemble(Polynomial(kind="state_quadratic"))
    z, action = tensor([[1., 2.]]), tensor([[.2, -.3]])
    options = dict(noisy_z=z + .1, mpr_coef=.2, return_components=True)
    expected = pave.critic_pave_loss(critic, identity, z, action, **options)
    actual = pave.critic_pave_loss(
        critic, identity, z, action, noisy_action=tensor([[float("nan")]]),
        action_smoothness_coef=0., **options,
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert actual[1].keys() == expected[1].keys()
    assert actual[2].keys() == expected[2].keys()
    for name in expected[1]:
        torch.testing.assert_close(actual[1][name], expected[1][name], rtol=0, atol=0)


def test_disabled_extension_keeps_outer_noop_without_encoding_or_rng():
    model = ObservationModel()
    cfg = SimpleNamespace(outer_pave_action_smoothness_coef=0.,
                          aux_return_pave_action_smoothness_coef=0.,
                          outer_pave_action_noise_std=.5)
    rng = torch.random.get_rng_state().clone()
    loss, metrics = pave.outer_pave_loss(
        model, torch.zeros(3, 2, 2), torch.zeros(2, 2, 2), cfg, 1,
    )
    assert loss.item() == 0
    assert metrics == {}
    assert not model.encoded
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
