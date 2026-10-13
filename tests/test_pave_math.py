"""Analytic PAVE losses and paired observation sampling, independent of AMBI."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from RL.tdmpc2_core import pave


DTYPE = torch.float64


def tensor(values, **kwargs):
    return torch.tensor(values, dtype=DTYPE, **kwargs)


class Ensemble(nn.Module):
    """Deliberately forbid the potentially compiled ensemble entry point."""

    def __init__(self, *heads):
        super().__init__()
        self.heads = nn.ModuleList(heads)

    def __len__(self):
        return len(self.heads)

    def __iter__(self):
        return iter(self.heads)

    def forward(self, _):
        raise AssertionError("Higher derivatives must bypass compiled ensemble forward.")


class Polynomial(nn.Module):
    def __init__(self, coefficient=1.0, *, kind="bilinear", hessian=None):
        super().__init__()
        self.coefficient = nn.Parameter(tensor(coefficient))
        self.kind = kind
        self.register_buffer("hessian", tensor([[1., 2.], [2., 3.]])
                             if hessian is None else tensor(hessian))

    def forward(self, joint):
        z, action = joint[..., :2], joint[..., 2:]
        if self.kind == "bilinear":
            value = z[..., 0] * action[..., 0] + 2 * z[..., 1] * action[..., 1]
        elif self.kind == "state_quadratic":
            value = (z * action.square()).sum(-1) / 2
        elif self.kind == "quadratic":
            value = torch.einsum("bi,ij,bj->b", action, self.hessian, action) / 2
        elif self.kind == "linear":
            value = action.sum(-1)
        elif self.kind == "constant":
            value = torch.ones_like(joint[..., 0])
        else:
            raise AssertionError(self.kind)
        return (self.coefficient * value).unsqueeze(-1)


def identity(value):
    return value


def test_mpr_matches_squared_mixed_partials_and_sums_independent_heads():
    critic = Ensemble(Polynomial(1.), Polynomial(-2.))
    z = tensor([[0., 0.], [0., 0.]])
    noisy_z = tensor([[1., -1.], [2., 3.]])
    action = tensor([[.2, -.3], [.7, .4]])
    loss, metrics = pave.critic_pave_loss(
        critic, identity, z, action, noisy_z=noisy_z, mpr_coef=.2,
    )
    # First head's gradient changes are [1,-2], [2,6]: mean square 45/4.
    # The second head contributes four times as much, with no head averaging.
    expected = tensor((45 / 4) * (1 + 4))
    torch.testing.assert_close(metrics["mpr_loss"], expected)
    torch.testing.assert_close(loss, .2 * expected)
    torch.testing.assert_close(metrics["mpr_weighted_loss"], loss.detach())
    torch.testing.assert_close(metrics["weighted_loss"], loss.detach())
    loss.backward()
    torch.testing.assert_close(critic.heads[0].coefficient.grad, tensor(4.5))
    torch.testing.assert_close(critic.heads[1].coefficient.grad, tensor(-9.))
    assert all(not value.requires_grad for value in metrics.values())


def test_opposing_head_action_gradients_do_not_cancel():
    critic = Ensemble(Polynomial(1.), Polynomial(-1.))
    loss, metrics = pave.critic_pave_loss(
        critic, identity, tensor([[0., 0.]]), tensor([[.4, -.7]]),
        noisy_z=tensor([[1., 1.]]), mpr_coef=1.,
    )
    # The summed Q is identically zero, but both heads have nonzero MPR.
    torch.testing.assert_close(loss, tensor(5.))
    torch.testing.assert_close(metrics["mpr_loss"], tensor(5.))
    loss.backward()
    torch.testing.assert_close(critic.heads[0].coefficient.grad, tensor(5.))
    torch.testing.assert_close(critic.heads[1].coefficient.grad, tensor(-5.))


def test_vfc_uses_recorded_action_at_both_states():
    critic = Ensemble(Polynomial(1., kind="state_quadratic"),
                      Polynomial(-.5, kind="state_quadratic"))
    z = tensor([[1., 2.], [-1., 3.]])
    next_z = tensor([[2., -1.], [1., 5.]])
    action = tensor([[2., -1.], [3., 2.]])
    loss, metrics = pave.critic_pave_loss(
        critic, identity, z, action, next_z=next_z, vfc_coef=.4,
    )
    # grad_a Q = z*a, so paired differences are [-2,-3], [-6,-4].
    expected = tensor((4 + 9 + 36 + 16) / 4 * (1 + .25))
    torch.testing.assert_close(metrics["vfc_loss"], expected)
    torch.testing.assert_close(loss, .4 * expected)
    torch.testing.assert_close(metrics["vfc_weighted_loss"], loss.detach())


def test_curvature_matches_signed_hutchinson_hinge_including_off_diagonal_terms():
    critic = Ensemble(Polynomial(1., kind="quadratic"),
                      Polynomial(-1., kind="quadratic"))
    loss, metrics = pave.critic_pave_loss(
        critic, identity, tensor([[1., 2.], [3., 4.]]),
        tensor([[.3, .7], [-.2, .4]]), probe=tensor([[1., 1.], [1., -1.]]),
        curvature_coef=.3, curvature_margin=1.,
    )
    # H=[[1,2],[2,3]] gives v'Hv=[8,0]; -H gives [-8,0].
    # relu(v'Hv+1) averages to 5 and .5, then sums across heads.
    torch.testing.assert_close(metrics["curvature_loss"], tensor(5.5))
    torch.testing.assert_close(loss, tensor(1.65))
    torch.testing.assert_close(metrics["curvature_violation_fraction"], tensor(.75).float())
    loss.backward()
    torch.testing.assert_close(critic.heads[0].coefficient.grad, tensor(1.2))
    torch.testing.assert_close(critic.heads[1].coefficient.grad, tensor(0.))


def test_sufficiently_concave_critic_has_zero_curvature_penalty():
    critic = Ensemble(Polynomial(1., kind="quadratic", hessian=[[-2., 0.], [0., -2.]]))
    loss, metrics = pave.critic_pave_loss(
        critic, identity, tensor([[0., 0.]]), tensor([[.3, -.1]]),
        probe=tensor([[1., -1.]]), curvature_coef=1., curvature_margin=1.,
    )
    torch.testing.assert_close(loss, tensor(0.))
    assert metrics["curvature_violation_fraction"].item() == 0.
    loss.backward()
    torch.testing.assert_close(critic.heads[0].coefficient.grad, tensor(0.))


@pytest.mark.parametrize("kind", ["constant", "linear"])
def test_zero_hessian_has_margin_penalty_and_valid_zero_parameter_gradient(kind):
    critic = Ensemble(Polynomial(2., kind=kind), Polynomial(-3., kind=kind))
    loss, metrics = pave.critic_pave_loss(
        critic, identity, tensor([[1., 2.], [3., 4.]]), tensor([[.3, .7], [-.2, .4]]),
        probe=tensor([[1., 1.], [-1., 1.]]), curvature_coef=.5, curvature_margin=.7,
    )
    torch.testing.assert_close(metrics["curvature_loss"], tensor(1.4))
    torch.testing.assert_close(loss, tensor(.7))
    assert metrics["curvature_violation_fraction"].item() == 1.
    loss.backward()
    for parameter in critic.parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0


def test_pave_detaches_all_supplied_state_views_and_replay_actions():
    encoder = nn.Linear(2, 2, bias=False, dtype=DTYPE)
    with torch.no_grad():
        encoder.weight.copy_(torch.eye(2, dtype=DTYPE))
    obs = tensor([[.4, .5], [.7, .2]], requires_grad=True)
    action = tensor([[.3, .8], [.1, -.5]], requires_grad=True)
    selected = Ensemble(Polynomial(1., kind="state_quadratic"))
    unselected = Ensemble(Polynomial(4.))
    loss, _ = pave.critic_pave_loss(
        selected, identity, encoder(obs), action,
        noisy_z=encoder(obs + .2), next_z=encoder(obs + .5),
        probe=torch.ones_like(action), mpr_coef=.2, vfc_coef=.3, curvature_coef=.4,
    )
    loss.backward()
    assert selected.heads[0].coefficient.grad.abs().item() > 0
    assert all(parameter.grad is None for parameter in unselected.parameters())
    assert encoder.weight.grad is None
    assert obs.grad is None
    assert action.grad is None


class DropoutPolynomial(Polynomial):
    def __init__(self):
        super().__init__(kind="state_quadratic")
        self.dropout = nn.Dropout(.7)

    def forward(self, joint):
        assert not self.dropout.training
        return super().forward(self.dropout(joint))


@pytest.mark.parametrize("fail", [False, True])
def test_dropout_is_disabled_and_every_original_mode_is_restored_even_on_failure(fail):
    critic = Ensemble(DropoutPolynomial(), DropoutPolynomial())
    critic.train()
    critic.heads[0].eval()
    critic.heads[0].dropout.train()
    critic.heads[1].dropout.eval()
    modes = {id(module): module.training for module in critic.modules()}
    rng = torch.random.get_rng_state().clone()

    def decode(predictions):
        if fail:
            raise RuntimeError("intentional decode failure")
        return predictions

    kwargs = dict(noisy_z=tensor([[.8, 1.2]]), mpr_coef=1.)
    if fail:
        with pytest.raises(RuntimeError, match="intentional decode failure"):
            pave.critic_pave_loss(critic, decode, tensor([[.2, .3]]), tensor([[.4, .7]]), **kwargs)
    else:
        loss, _ = pave.critic_pave_loss(
            critic, decode, tensor([[.2, .3]]), tensor([[.4, .7]]), **kwargs,
        )
        loss.backward()
    assert {id(module): module.training for module in critic.modules()} == modes
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)


class ObservationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._Qs = Ensemble(Polynomial())
        self.q_backend = SimpleNamespace(decode=identity)
        self.encoded = []

    def encode(self, observation):
        self.encoded.append(observation.detach().clone())
        return 2 * observation + 3


def outer_config(**overrides):
    options = dict(
        seed=91, outer_pave_batch_size=4, outer_pave_obs_noise_std=.01,
        outer_pave_curvature_margin=1., outer_pave_mpr_coef=.1,
        outer_pave_vfc_coef=.2, outer_pave_curvature_coef=.3,
    )
    return SimpleNamespace(**{**options, **overrides})


def test_outer_sampling_preserves_real_transition_pairing_and_private_reproducible_draws(monkeypatch):
    model = ObservationModel()
    obs = torch.arange(24, dtype=DTYPE).reshape(4, 3, 2)
    actions = obs[:-1] + 1000
    captured = []

    def capture(critic, decode, z, action, **kwargs):
        captured.append({"z": z.clone(), "action": action.clone(), **{
            name: kwargs[name].clone() for name in ("noisy_z", "next_z", "probe")
        }})
        return action.new_zeros(()), {"weighted_loss": action.new_zeros(())}

    monkeypatch.setattr(pave, "critic_pave_loss", capture)
    rng = torch.random.get_rng_state().clone()
    for update in (7, 7, 8):
        _, metrics = pave.outer_pave_loss(model, obs, actions, outer_config(), update)
        assert metrics["outer_pave_sample_count"].item() == 4
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    for sampled in captured:
        current = (sampled["z"] - 3) / 2
        torch.testing.assert_close(sampled["action"], current + 1000)
        torch.testing.assert_close(sampled["next_z"], sampled["z"] + 12)
        assert torch.unique(current[:, 0]).numel() == 4
        assert set(sampled["probe"].flatten().tolist()) <= {-1., 1.}
        assert not torch.equal(sampled["noisy_z"], sampled["z"])
    for key in captured[0]:
        torch.testing.assert_close(captured[0][key], captured[1][key], rtol=0, atol=0)
    assert not torch.equal(captured[0]["noisy_z"], captured[2]["noisy_z"])
    # Encoder views are original observation, its perturbation, and real successor.
    torch.testing.assert_close(captured[0]["z"], model.encoded[0] * 2 + 3)
    torch.testing.assert_close(captured[0]["noisy_z"], model.encoded[1] * 2 + 3)
    torch.testing.assert_close(captured[0]["next_z"], model.encoded[2] * 2 + 3)


def test_disabled_outer_pave_neither_encodes_nor_consumes_rng():
    model = ObservationModel()
    obs = torch.zeros(3, 2, 2, dtype=DTYPE)
    actions = torch.zeros(2, 2, 2, dtype=DTYPE)
    rng = torch.random.get_rng_state().clone()
    loss, metrics = pave.outer_pave_loss(
        model, obs, actions, outer_config(outer_pave_mpr_coef=0.,
                                       outer_pave_vfc_coef=0., outer_pave_curvature_coef=0.), 1,
    )
    assert loss.item() == 0.
    assert metrics == {}
    assert model.encoded == []
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
