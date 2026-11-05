"""Acceleration must preserve gSDE actions, derivatives, and randomness."""

from copy import deepcopy

import pytest
import torch

from RL.tdmpc2_core.common.gsde import GsdeActor
from RL.tdmpc2_core.common.soft_world_model import SoftWorldModel
from tests.test_q_representation import model_cfg


def _log_domain_reference(features, log_scale):
    """Original independent log-sum-exp expression, without the GEMM path."""
    nonzero = features != 0
    safe = torch.where(nonzero, features.abs(), torch.ones_like(features))
    terms = 2 * (safe.log().unsqueeze(-1) + log_scale)
    terms = torch.where(nonzero.unsqueeze(-1), terms, torch.full_like(terms, -torch.inf))
    return 0.5 * torch.logsumexp(terms, dim=-2)


@pytest.mark.parametrize("shape", [(1, 512), (128, 512), (2, 3, 512)])
def test_cpu_gemm_variance_and_gradients_match_log_domain_reference(shape):
    torch.manual_seed(61)
    features = torch.randn(shape, requires_grad=True)
    with torch.no_grad():
        features[..., ::5] = 0
    log_scale = torch.randn(512, 21, requires_grad=True) * 0.3 - 3
    expected = _log_domain_reference(features, log_scale)
    actual = GsdeActor._marginal_log_std(features, log_scale)
    weights = torch.randn_like(actual)
    expected_gradients = torch.autograd.grad((expected * weights).sum(), (features, log_scale),
                                             retain_graph=True)
    actual_gradients = torch.autograd.grad((actual * weights).sum(), (features, log_scale))
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=8e-5, atol=3e-6)


@pytest.mark.parametrize("magnitude,log_scale", [(1e-30, -30.), (1e30, 30.), (1e30, -80.)])
def test_cpu_gemm_variance_retains_extreme_representable_products(magnitude, log_scale):
    features = torch.full((2, 512), magnitude)
    scales = torch.full((512, 21), log_scale)
    expected = _log_domain_reference(features, scales)
    actual = GsdeActor._marginal_log_std(features, scales)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def _model(source, dtype):
    torch.manual_seed(73)
    return SoftWorldModel(model_cfg(actor_distribution="gsde", gsde_feature_source=source)).to(dtype=dtype).eval()


@pytest.mark.parametrize("source", ["actor", "latent"])
@pytest.mark.parametrize("shape", [(1,), (4,), (2, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("deterministic", [False, True])
def test_action_only_matches_full_policy_without_consuming_rng(source, shape, dtype, deterministic):
    model = _model(source, dtype)
    z = torch.randn(*shape, model.cfg.latent_dim, dtype=dtype)
    matrix = torch.randn(model.gsde_noise_shape(batch_shape=shape), dtype=dtype)
    before = torch.random.get_rng_state().clone()
    actual = model.pi_action(z, sde_noise=matrix, deterministic=deterministic)
    expected, _ = model.pi(z, sde_noise=matrix, deterministic=deterministic)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)


@pytest.mark.parametrize("source", ["actor", "latent"])
@pytest.mark.parametrize("std_scale", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("detached", [False, True])
def test_action_only_preserves_input_and_parameter_gradients(source, std_scale, detached):
    actual_model = _model(source, torch.float64)
    expected_model = deepcopy(actual_model)
    actual_z = torch.randn(2, 3, actual_model.cfg.latent_dim, dtype=torch.float64, requires_grad=True)
    expected_z = actual_z.detach().clone().requires_grad_()
    matrix = torch.randn(actual_model.gsde_noise_shape(batch_shape=(2, 3)), dtype=torch.float64)
    actual = actual_model.pi_action(actual_z, sde_noise=matrix, std_scale=std_scale, detach_policy=detached)
    expected, _ = expected_model.pi(expected_z, sde_noise=matrix, std_scale=std_scale, detach_policy=detached)
    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(actual_z.grad, expected_z.grad, rtol=2e-8, atol=2e-9)
    for actual_parameter, expected_parameter in zip(actual_model._pi.parameters(), expected_model._pi.parameters()):
        if detached:
            assert actual_parameter.grad is None
            assert expected_parameter.grad is None
        else:
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("bad_noise", ["shape", "nonfinite", "two_sources"])
def test_action_only_does_not_skip_noise_validation(deterministic, bad_noise):
    model = _model("actor", torch.float32)
    z = torch.randn(2, model.cfg.latent_dim)
    matrix = torch.zeros(model.gsde_noise_shape(batch_shape=(2,)))
    options = {"deterministic": deterministic}
    if bad_noise == "shape":
        matrix = matrix[0]
    elif bad_noise == "nonfinite":
        matrix[0, 0, 0] = torch.inf
    else:
        options["noise"] = torch.zeros(2, model.cfg.action_dim)
    with pytest.raises((ValueError, RuntimeError)):
        model.pi_action(z, sde_noise=matrix, **options)


@pytest.mark.parametrize("std_scale", [0.0, -1., float("nan"), float("inf"), 1e300, 1e-300])
def test_action_only_rejects_invalid_or_singular_effective_scale(std_scale):
    model = _model("actor", torch.float32)
    z = torch.randn(2, model.cfg.latent_dim)
    matrix = torch.zeros(model.gsde_noise_shape(batch_shape=(2,)))
    with pytest.raises((ValueError, RuntimeError)):
        model.pi_action(z, sde_noise=matrix, std_scale=std_scale)


@pytest.mark.parametrize("source", ["actor", "latent"])
def test_action_only_fullgraph_compilation_preserves_actions_and_gradients(source, monkeypatch):
    # Backend discovery imports Inductor even for aot_eager. Avoid forking
    # compiler workers after numerical libraries have initialized on macOS.
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "1")
    import torch._inductor.config as inductor_config
    monkeypatch.setattr(inductor_config, "compile_threads", 1)
    model = _model(source, torch.float32)
    reference_model = deepcopy(model)
    action = torch.compile(
        lambda z, matrix: model.pi_action(z, sde_noise=matrix),
        backend="aot_eager", fullgraph=True, dynamic=False,
    )
    z = torch.randn(3, model.cfg.latent_dim, requires_grad=True)
    reference_z = z.detach().clone().requires_grad_()
    matrix = torch.randn(model.gsde_noise_shape(batch_shape=(3,)))
    before = torch.random.get_rng_state().clone()
    actual = action(z, matrix)
    expected, _ = reference_model.pi(reference_z, sde_noise=matrix)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(z.grad, reference_z.grad, rtol=3e-4, atol=3e-6)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
