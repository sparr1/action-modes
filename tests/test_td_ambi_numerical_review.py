"""Independent numerical checks for the Humanoid TD-AMBI objective study.

The entropy oracle is the analytic Gaussian expectation, not another copy of
the implementation's squashing/ratio expression. Unit-magnitude noise makes
the sampled quadratic term equal its expectation exactly. The ratio is only
approximately cancellable away from its pole; literal near-pole parity is
covered separately in test_tdmpc2_scaled_entropy.py.
"""

import math
from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.common import math as td_math
from RL.tdmpc2_core.common.q_representation import QRepresentation
from tests.test_ambi_inner_decoupling import _model


@pytest.mark.parametrize("mean_value", [0.0, 10.0])
@pytest.mark.parametrize("log_std_value", [-10.0, -4.0, -2.4189385175704956, 2.0])
def test_humanoid_scaled_entropy_matches_gaussian_units_and_gradients(
    mean_value, log_std_value,
):
    dimension = 21
    mean = torch.full((16, dimension), mean_value, requires_grad=True)
    log_std = torch.full_like(mean, log_std_value, requires_grad=True)
    noise = torch.ones_like(mean)
    noise[:, ::2] = -1.0
    action = (mean + noise * log_std.exp()).tanh()
    actual = td_math.tdmpc2_scaled_entropy(
        td_math.gaussian_logprob(noise, log_std), action,
    )
    # With epsilon**2 = 1 the analytic Gaussian entropy is exact per sample.
    expected = dimension**2 * (0.5 * math.log(2.0 * math.pi * math.e) + log_std_value)
    torch.testing.assert_close(actual, torch.full_like(actual, expected), atol=5e-4, rtol=2e-6)
    mean_gradient, log_std_gradient = torch.autograd.grad(actual.sum(), (mean, log_std))
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(mean_gradient, torch.zeros_like(mean), atol=2e-4, rtol=0)
    torch.testing.assert_close(log_std_gradient, torch.full_like(log_std, dimension), atol=5e-4, rtol=2e-5)
    if log_std_value == -2.4189385175704956:
        assert actual.mean().item() == pytest.approx(-441.0, abs=2e-4)


@pytest.mark.parametrize("entropy_residual", [-441.0, 441.0])
def test_humanoid_inner_temperature_eighteen_steps_match_scalar_adam(
    monkeypatch, entropy_residual,
):
    holder = _model(
        ent_coef="auto_0.0001", outer_actor_entropy_mode="tdmpc2_scaled",
        target_entropy=-441.0, inner_actor_entropy_mode="tdmpc2_scaled",
        inner_temperature_mode="auto", inner_temperature_initialization="inherit_outer",
        inner_temperature_updates_per_action=1, inner_target_entropy=-441.0,
        inner_temperature_grad_clip_norm=None, inner_temperature_lr=3e-4,
        inner_adam_eps=1e-8, log_std_mapping="tdmpc2_tanh",
        log_std_min=-10.0, log_std_max=2.0,
    )
    try:
        cfg = deepcopy(holder.cfg)
        cfg.action_dim = 21
        agent = AMBITDMPC2Agent(cfg)
    finally:
        holder.env.close()
    engine = agent.inner_engine
    with engine.rng.fork("initialization"):
        engine._prepare_workspace(t0=True)
    desired_entropy = -441.0 + entropy_residual
    desired_log_std = desired_entropy / 21**2 - 0.5 * math.log(2.0 * math.pi * math.e)
    raw_std = math.atanh((desired_log_std + 10.0) / 6.0 - 1.0)
    with torch.no_grad():
        engine.state.actor[-1].weight.zero_()
        engine.state.actor[-1].bias.zero_()
        engine.state.actor[-1].bias[21:].fill_(raw_std)
    real_pi = engine.model.pi

    def unit_noise_pi(z, **kwargs):
        kwargs["noise"] = torch.ones((*z.shape[:-1], 21), device=z.device)
        return real_pi(z, **kwargs)

    monkeypatch.setattr(engine.model, "pi", unit_noise_pi)
    initial_outer_log_alpha = agent.log_ent_coef.detach().clone()
    expected_log_alpha = float(engine.state.log_alpha.detach())
    first_moment = second_moment = 0.0
    actor_before = [p.detach().clone() for p in engine.state.actor_params]
    for step in range(1, 19):
        metrics = engine._sac_policy_step(
            {"z": torch.zeros(512, cfg.latent_dim)}, update_actor=False,
            update_temperature=True, alpha=engine.alpha.detach(),
        )
        # Independent scalar Adam equations include both bias corrections.
        first_moment = 0.9 * first_moment + 0.1 * entropy_residual
        second_moment = 0.999 * second_moment + 0.001 * entropy_residual**2
        mean = first_moment / (1.0 - 0.9**step)
        rms = math.sqrt(second_moment / (1.0 - 0.999**step))
        expected_log_alpha -= 3e-4 * mean / (rms + 1e-8)
        # The inverse/forward float32 log-std mapping's sub-micro-unit error
        # is magnified by d**2; one milli-unit covers that rounding at d=21.
        assert metrics["actor_scaled_entropy"].item() == pytest.approx(desired_entropy, abs=1e-3)
        assert metrics["temperature_grad_norm"].item() == pytest.approx(abs(entropy_residual), abs=1e-3)
        assert engine.state.log_alpha.item() == pytest.approx(expected_log_alpha, abs=1e-5)
    assert (engine.alpha > agent.alpha).item() == (entropy_residual < 0)
    torch.testing.assert_close(agent.log_ent_coef, initial_outer_log_alpha, atol=0, rtol=0)
    for parameter, before in zip(engine.state.actor_params, actor_before):
        assert parameter.grad is None
        torch.testing.assert_close(parameter, before, atol=0, rtol=0)


def test_study_distributional_targets_at_zero_endpoints_and_outside_support():
    backend = QRepresentation(
        "distributional", num_q=5, pair_size=2, num_bins=101,
        vmin=-10.0, vmax=10.0,
    )
    bound = math.expm1(10.0)
    target = torch.tensor([-1e30, -bound, 0.0, bound, 1e30]).unsqueeze(-1)
    generator = torch.Generator().manual_seed(781)
    logits = (10.0 * torch.randn(5, 5, 101, generator=generator)).requires_grad_()
    actual = backend.loss(logits, target)
    # Clipped endpoint targets and exact zero must each become a point mass.
    indices = torch.tensor([0, 0, 50, 100, 100])
    expected = torch.stack([
        torch.nn.functional.cross_entropy(head, indices) for head in logits
    ]).mean()
    torch.testing.assert_close(actual, expected)
    actual_gradient, = torch.autograd.grad(actual, logits)
    labels = torch.nn.functional.one_hot(indices, 101).to(logits)
    expected_gradient = (logits.detach().softmax(-1) - labels) / 25.0
    assert torch.isfinite(actual) and torch.isfinite(actual_gradient).all()
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-8, rtol=1e-5)
