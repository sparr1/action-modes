"""gSDE sampling, marginal densities, gradient ownership, and legacy behavior."""

from copy import deepcopy
import math

import pytest
import torch

from RL.tdmpc2_core.common.gsde import GsdeActor
from RL.tdmpc2_core.common.soft_world_model import SoftWorldModel
from RL.tdmpc2_core.common import math as td_math
from tests.test_q_representation import model_cfg


def _model(source="actor", dtype=torch.float64, **kwargs):
    torch.manual_seed(31)
    return SoftWorldModel(model_cfg(
        actor_distribution="gsde", gsde_feature_source=source, **kwargs,
    )).to(dtype=dtype).eval()


@pytest.mark.parametrize("source", ["actor", "latent"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_induced_variance_and_matrix_sample_match_density(source, dtype):
    model = _model(source, dtype=dtype)
    z = torch.randn(2, 3, model.cfg.latent_dim, dtype=dtype)
    actor = model._pi
    with torch.no_grad():
        actor.log_scale.copy_(torch.linspace(
            -3.7, -1.3, actor.log_scale.numel(), dtype=dtype,
        ).reshape_as(actor.log_scale))
    features = actor.trunk(z) if source == "actor" else z
    expected_std = (features.square() @ actor.log_scale.exp().square()).sqrt()
    matrix = torch.randn(model.gsde_noise_shape(batch_shape=z.shape[:-1]), dtype=dtype)
    action, info = model.pi(z, sde_noise=matrix)
    delta = (features.unsqueeze(-1) * actor.log_scale.exp() * matrix).sum(-2)
    torch.testing.assert_close(info["log_std"].exp(), expected_std)
    torch.testing.assert_close(info["pre_tanh_action"], info["pre_tanh_mean"] + delta)
    torch.testing.assert_close(action, info["pre_tanh_action"].tanh())
    expected_log_prob = torch.distributions.Normal(
        info["pre_tanh_mean"], expected_std,
    ).log_prob(info["pre_tanh_action"]).sum(-1, keepdim=True)
    expected_log_prob -= td_math.tanh_log_abs_det_jacobian(info["pre_tanh_action"])
    torch.testing.assert_close(info["log_prob"], expected_log_prob, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(model.pi_action(z, sde_noise=matrix), action)
    stats = model.policy_stats(z)
    for key in stats:
        torch.testing.assert_close(stats[key], info[key])
    assert action.dtype == dtype


def test_shared_matrix_has_the_predicted_cross_state_covariance():
    model = _model("latent")
    actor = model._pi
    with torch.no_grad():
        actor.log_scale.fill_(-0.5)
    count = 14000
    states = torch.tensor([
        [1., -2., 0., 1., 0., 0., 0., 0.],
        [-2., 1., 0., 1., 0., 0., 0., 0.],
    ], dtype=torch.float64)
    z = states.unsqueeze(0).expand(count, -1, -1)
    generator = torch.Generator().manual_seed(71)
    base = torch.randn(model.gsde_noise_shape(batch_shape=(count,)),
                       dtype=z.dtype, generator=generator)
    with torch.no_grad():
        _, info = model.pi(z, sde_noise=base.unsqueeze(1).expand(-1, 2, -1, -1))
    perturbation = info["pre_tanh_action"] - info["pre_tanh_mean"]
    covariance = torch.cov(perturbation[:, :, 0].T)
    expected = (states * actor.log_scale[:, 0].exp()) @ (
        states * actor.log_scale[:, 0].exp()
    ).T
    torch.testing.assert_close(covariance, expected, rtol=0.035, atol=0.025)
    assert covariance[0, 1] < 0


@pytest.mark.parametrize("source", ["actor", "latent"])
def test_persistent_base_draw_uses_current_parameters_without_cached_graph(source):
    model = _model(source)
    z = torch.randn(4, model.cfg.latent_dim, dtype=torch.float64)
    base = torch.randn(model.gsde_noise_shape(batch_shape=(4,)), dtype=z.dtype)
    original = base.clone()
    _, before = model.pi(z, sde_noise=base)
    for step in range(2):
        model._pi.zero_grad(set_to_none=True)
        action, info = model.pi(z, sde_noise=base)
        (action.sum() + info["log_std"].sum()).backward()
        assert model._pi.log_scale.grad is not None
        with torch.no_grad():
            model._pi.log_scale.add_(0.1)
            model._pi.trunk[0].bias.add_(0.01)
    _, after = model.pi(z, sde_noise=base)
    torch.testing.assert_close(base, original, rtol=0, atol=0)
    assert not torch.allclose(
        before["pre_tanh_action"] - before["pre_tanh_mean"],
        after["pre_tanh_action"] - after["pre_tanh_mean"],
    )


@pytest.mark.parametrize("source", ["actor", "latent"])
@pytest.mark.parametrize("matrix", [False, True])
def test_detached_policy_freezes_parameters_but_keeps_input_gradients(source, matrix):
    model = _model(source)
    z = torch.randn(3, model.cfg.latent_dim, dtype=torch.float64, requires_grad=True)
    kwargs = ({"sde_noise": torch.randn(model.gsde_noise_shape(batch_shape=(3,)), dtype=z.dtype)}
              if matrix else {"noise": torch.randn(3, model.cfg.action_dim, dtype=z.dtype)})
    action, info = model.pi(z, detach_policy=True, **kwargs)
    (action.sum() + info["log_std"].sum()).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    live_z = z.detach().clone().requires_grad_(True)
    live_action, live_info = model.pi(live_z, **kwargs)
    (live_action.sum() + live_info["log_std"].sum()).backward()
    torch.testing.assert_close(action, live_action, rtol=0, atol=0)
    torch.testing.assert_close(z.grad, live_z.grad)


def test_entropy_trains_actor_features_and_scales_without_training_encoder():
    model = _model()
    z = model.encode(torch.randn(5, 3, dtype=torch.float64)).detach()
    stats = model.policy_stats(z)
    entropy = (stats["log_std"] + 0.5 * math.log(2 * math.pi * math.e)).sum(-1).mean()
    entropy.backward()
    assert model._pi.log_scale.grad.abs().sum() > 0
    assert sum(p.grad.abs().sum() for p in model._pi.trunk.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in model._encoder.parameters())
    assert all(p.grad is None for p in model._pi.mean_head.parameters())


def test_zero_feature_coordinate_retains_linear_sample_gradient():
    model = _model("latent")
    z = torch.tensor([[1., 0., -2., 0., 0., 0., 0., 0.]],
                     dtype=torch.float64, requires_grad=True)
    base = torch.randn(model.gsde_noise_shape(batch_shape=(1,)), dtype=z.dtype)
    _, info = model.pi(z, sde_noise=base)
    delta = info["pre_tanh_action"] - info["pre_tanh_mean"]
    gradient, = torch.autograd.grad(delta.sum(), z)
    expected = (model._pi.log_scale.exp() * base).sum(-1)
    torch.testing.assert_close(gradient, expected)


@pytest.mark.parametrize("magnitude", [1e-25, 1e20])
def test_log_variance_preserves_tiny_and_large_nonzero_features(magnitude):
    model = _model("latent", dtype=torch.float32)
    z = torch.full((2, model.cfg.latent_dim), magnitude)
    stats = model.policy_stats(z)
    expected = math.log(magnitude) - 3 + 0.5 * math.log(model.cfg.latent_dim)
    torch.testing.assert_close(stats["log_std"], torch.full_like(stats["log_std"], expected))
    base = torch.ones(model.gsde_noise_shape(batch_shape=(2,)))
    _, info = model.pi(z, sde_noise=base)
    expected_delta = torch.full_like(info["pre_tanh_mean"], magnitude * math.exp(-3) * model.cfg.latent_dim)
    torch.testing.assert_close(info["pre_tanh_action"] - info["pre_tanh_mean"], expected_delta)


@pytest.mark.parametrize("source", ["actor", "latent"])
def test_singular_feature_row_is_rejected_without_jitter(source):
    model = _model(source)
    with pytest.raises(RuntimeError, match="singular Gaussian"):
        model.policy_stats(torch.zeros(2, model.cfg.latent_dim, dtype=torch.float64))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_matrix_parameters_and_noise_are_rejected(invalid):
    model = _model("latent")
    z = torch.ones(2, model.cfg.latent_dim, dtype=torch.float64)
    base = torch.zeros(model.gsde_noise_shape(batch_shape=(2,)), dtype=z.dtype)
    base[0, 0, 0] = invalid
    with pytest.raises(RuntimeError, match="noise must be finite"):
        model.pi(z, sde_noise=base)
    with torch.no_grad():
        model._pi.log_scale[0, 0] = invalid
    with pytest.raises(RuntimeError, match="matrix scales"):
        model.policy_stats(z)


def test_independent_branches_fresh_marginal_and_mean_execution():
    model = _model()
    z = torch.randn(1, model.cfg.latent_dim, dtype=torch.float64).expand(5, -1)
    base = torch.randn(model.gsde_noise_shape(batch_shape=(5,)), dtype=z.dtype)
    action, info = model.pi(z, sde_noise=base)
    assert not torch.equal(action[0], action[1])
    rng_before = torch.random.get_rng_state().clone()
    mean = model.pi_action(z, deterministic=True)
    torch.testing.assert_close(mean, info["mean"], rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
    eps = (info["pre_tanh_action"] - info["pre_tanh_mean"]) / info["log_std"].exp()
    marginal_action, marginal_info = model.pi(z, noise=eps)
    torch.testing.assert_close(marginal_action, action)
    torch.testing.assert_close(marginal_info["log_prob"], info["log_prob"])


def test_mean_clip_and_std_scale_apply_to_matrix_and_marginal_equally():
    model = _model()
    z = torch.randn(3, model.cfg.latent_dim, dtype=torch.float64)
    base = torch.randn(model.gsde_noise_shape(batch_shape=(3,)), dtype=z.dtype)
    with torch.no_grad():
        model._pi.mean_head.weight.zero_()
        model._pi.mean_head.bias.copy_(torch.tensor([4., -5.]))
    _, info = model.pi(z, sde_noise=base)
    _, scaled = model.pi(z, sde_noise=base, std_scale=2.5)
    torch.testing.assert_close(info["pre_tanh_mean"], torch.tensor([2., -2.], dtype=z.dtype).expand(3, -1))
    torch.testing.assert_close(scaled["log_std"], info["log_std"] + math.log(2.5))
    torch.testing.assert_close(scaled["pre_tanh_action"] - scaled["pre_tanh_mean"],
                               2.5 * (info["pre_tanh_action"] - info["pre_tanh_mean"]))
    model._pi.mean_clip = None
    torch.testing.assert_close(model.policy_stats(z)["pre_tanh_mean"],
                               torch.tensor([4., -5.], dtype=z.dtype).expand(3, -1))


def test_matrix_noise_shape_and_source_validation():
    model = _model()
    z = torch.ones(3, model.cfg.latent_dim, dtype=torch.float64)
    base = torch.zeros(model.gsde_noise_shape(batch_shape=(3,)))
    for kwargs in ({"noise": torch.zeros(3, 2)}, {"generator": torch.Generator()}):
        with pytest.raises(ValueError, match="multiple sources"):
            model.pi(z, sde_noise=base, **kwargs)
    with pytest.raises(ValueError, match="noise must have shape"):
        model.pi(z, sde_noise=base[:1])
    gaussian = SoftWorldModel(model_cfg()).double()
    with pytest.raises(ValueError, match="requires a gSDE actor"):
        gaussian.pi(z, sde_noise=base)


def test_generator_and_scalar_state_sampling_preserve_shapes_and_rng_isolation():
    model = _model()
    z = torch.randn(model.cfg.latent_dim, dtype=torch.float64)
    before = torch.random.get_rng_state().clone()
    first = model.pi_action(z, generator=torch.Generator().manual_seed(123))
    second = model.pi_action(z, generator=torch.Generator().manual_seed(123))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    assert first.shape == (model.cfg.action_dim,)
    matrix = torch.zeros(model.gsde_noise_shape(), dtype=z.dtype)
    torch.testing.assert_close(model.pi_action(z, sde_noise=matrix), model.policy_stats(z)["mean"])


def test_low_precision_distribution_arithmetic_uses_float32():
    model = _model(dtype=torch.bfloat16)
    z = torch.ones(2, model.cfg.latent_dim, dtype=torch.bfloat16)
    action, info = model.pi(z, noise=torch.zeros(2, model.cfg.action_dim))
    assert action.dtype == torch.float32
    assert info["log_std"].dtype == torch.float32
    assert torch.isfinite(info["log_prob"]).all()


def test_nonfinite_unclipped_mean_and_unrepresentable_scaled_std_are_rejected():
    model = _model(dtype=torch.float32)
    z = torch.ones(2, model.cfg.latent_dim)
    with pytest.raises(RuntimeError, match="Scaled gSDE"):
        model.policy_stats(z, std_scale=1e300)
    with torch.no_grad():
        model._pi.mean_head.bias.fill_(torch.inf)
    with pytest.raises(RuntimeError, match="mean outputs must be finite"):
        model.policy_stats(z)


@pytest.mark.parametrize("feature_magnitude,std_scale", [(1e-20, 1e50), (1e20, 1e-50)])
def test_std_scaling_combines_in_log_space_before_dtype_conversion(feature_magnitude, std_scale):
    model = _model("latent", dtype=torch.float32)
    z = torch.full((2, model.cfg.latent_dim), feature_magnitude)
    stats = model.policy_stats(z, std_scale=std_scale)
    expected_log_std = (
        math.log(feature_magnitude) - 3.0 + 0.5 * math.log(model.cfg.latent_dim)
        + math.log(std_scale)
    )
    torch.testing.assert_close(
        stats["log_std"], torch.full_like(stats["log_std"], expected_log_std),
    )
    assert torch.isfinite(model.pi_action(
        z, noise=torch.ones(2, model.cfg.action_dim), std_scale=std_scale,
    )).all()


def test_disabled_gsde_preserves_legacy_parameters_rng_and_gaussian_formula():
    torch.manual_seed(17)
    legacy = SoftWorldModel(model_cfg()).eval()
    legacy_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(17)
    explicit = SoftWorldModel(model_cfg(actor_distribution="gaussian")).eval()
    torch.testing.assert_close(torch.random.get_rng_state(), legacy_rng, rtol=0, atol=0)
    assert legacy.state_dict().keys() == explicit.state_dict().keys()
    for key, value in legacy.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
    z = torch.randn(3, legacy.cfg.latent_dim)
    noise = torch.randn(3, legacy.cfg.action_dim)
    action, info = explicit.pi(z, noise=noise)
    mean, raw_std = legacy._pi(z).chunk(2, dim=-1)
    log_std = raw_std.clamp(-20, 2)
    expected_action = (mean + noise * log_std.exp()).tanh()
    torch.testing.assert_close(action, expected_action, rtol=0, atol=0)
    assert set(info) == {"mean", "pre_tanh_mean", "pre_tanh_action", "log_std", "log_prob", "entropy"}


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("detach_policy", [False, True])
def test_gsde_compiled_forward_and_gradients_match_eager(matrix, detach_policy, monkeypatch):
    # Torch 2.3 discovers other backends while loading aot_eager. Avoid its
    # unrelated process-pool warmup after macOS framework initialization.
    import torch._inductor.config as inductor_config
    monkeypatch.setattr(inductor_config, "compile_threads", 1)
    model = _model(dtype=torch.float32)
    eager = deepcopy(model)
    z = torch.randn(3, model.cfg.latent_dim, requires_grad=True)
    eager_z = z.detach().clone().requires_grad_(True)
    kwargs = ({"sde_noise": torch.randn(model.gsde_noise_shape(batch_shape=(3,)))}
              if matrix else {"noise": torch.randn(3, model.cfg.action_dim)})
    compiled = torch.compile(model.pi, backend="aot_eager", fullgraph=True)
    action, info = compiled(z, detach_policy=detach_policy, **kwargs)
    expected_action, expected_info = eager.pi(eager_z, detach_policy=detach_policy, **kwargs)
    torch.testing.assert_close(action, expected_action)
    for key in info:
        torch.testing.assert_close(info[key], expected_info[key])
    (action.sum() + info["log_std"].sum()).backward()
    (expected_action.sum() + expected_info["log_std"].sum()).backward()
    for actual, expected in zip(model._pi.parameters(), eager._pi.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)
    torch.testing.assert_close(z.grad, eager_z.grad)


@pytest.mark.parametrize("kwargs", [
    {"feature_source": "raw"}, {"log_std_init": float("nan")},
    {"mean_clip": 0}, {"mean_clip": -2}, {"mean_clip": float("inf")},
])
def test_invalid_gsde_constructor_settings(kwargs):
    with pytest.raises(ValueError):
        GsdeActor(8, 16, 2, **kwargs)


def _reference_log_std(features, log_scale):
    nonzero = features != 0
    safe = torch.where(nonzero, features.abs(), torch.ones_like(features))
    terms = 2.0 * (safe.log().unsqueeze(-1) + log_scale)
    return 0.5 * torch.logsumexp(
        torch.where(nonzero.unsqueeze(-1), terms, torch.full_like(terms, -torch.inf)),
        dim=-2,
    )


@pytest.mark.parametrize("feature_magnitudes", [[1., 1., 1., 1.], [1e-30, 1e30, 1e-20, 1e20]])
def test_cpu_gemm_marginal_matches_log_domain_values_and_gradients(feature_magnitudes):
    torch.manual_seed(139)
    features = torch.randn(2, 2, 8) * torch.tensor(feature_magnitudes).reshape(2, 2, 1)
    features[..., 2] = 0
    features.requires_grad_(True)
    log_scale = torch.linspace(-25., 25., 24).reshape(8, 3).requires_grad_(True)
    reference_features = features.detach().double().requires_grad_(True)
    reference_scales = log_scale.detach().double().requires_grad_(True)
    result = GsdeActor._marginal_log_std(features, log_scale)
    reference = _reference_log_std(reference_features, reference_scales)
    torch.testing.assert_close(result, reference.float(), rtol=2e-6, atol=2e-6)
    weights = torch.linspace(-1., 1., result.numel()).reshape_as(result)
    gradients = torch.autograd.grad((result * weights).sum(), (features, log_scale))
    expected = torch.autograd.grad((reference * weights.double()).sum(),
                                   (reference_features, reference_scales))
    for actual, reference_gradient in zip(gradients, expected):
        torch.testing.assert_close(actual, reference_gradient.float(), rtol=3e-5, atol=2e-6)


def test_cpu_gemm_preserves_sparse_extreme_scales_without_a_floor():
    model = _model("latent", dtype=torch.float32)
    z = torch.zeros(2, model.cfg.latent_dim)
    z[:, 0] = 1
    with torch.no_grad():
        model._pi.log_scale.fill_(69.)
        model._pi.log_scale[0].fill_(-92.)
    stats = model.policy_stats(z)
    torch.testing.assert_close(stats["log_std"], torch.full_like(stats["log_std"], -92.))
    assert (stats["log_std"].exp() > 0).all()
    stats["log_std"].sum().backward()
    expected = torch.zeros_like(model._pi.log_scale)
    expected[0] = 2
    torch.testing.assert_close(model._pi.log_scale.grad, expected)


@pytest.mark.parametrize("source", ["actor", "latent"])
@pytest.mark.parametrize("std_scale", [1.0, 2.5])
@pytest.mark.parametrize("detach_policy", [False, True])
def test_action_only_matches_distribution_action_and_input_gradients(source, std_scale, detach_policy):
    model = _model(source, dtype=torch.float32)
    reference_model = deepcopy(model)
    z = torch.randn(2, 3, model.cfg.latent_dim, requires_grad=True)
    reference_z = z.detach().clone().requires_grad_(True)
    base = torch.randn(model.gsde_noise_shape(batch_shape=z.shape[:-1]))
    result = model._pi.action(z, sde_noise=base, detach_parameters=detach_policy, std_scale=std_scale)
    reference, _ = reference_model.pi(
        reference_z, sde_noise=base, detach_policy=detach_policy, std_scale=std_scale,
    )
    torch.testing.assert_close(result, reference, rtol=2e-6, atol=2e-6)
    result.square().sum().backward()
    reference.square().sum().backward()
    torch.testing.assert_close(z.grad, reference_z.grad, rtol=3e-5, atol=2e-6)
    for actual, expected in zip(model._pi.parameters(), reference_model._pi.parameters()):
        if expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=2e-6)


def test_action_only_mean_keeps_distribution_validation_without_projecting_noise(monkeypatch):
    model = _model("latent", dtype=torch.float32)
    z = torch.ones(2, model.cfg.latent_dim)
    expected = model.policy_stats(z)["mean"]

    def forbidden_projection(*args):
        raise AssertionError("Mean execution must not project exploration noise.")

    monkeypatch.setattr(model._pi, "_matrix_perturbation", forbidden_projection)
    base = torch.zeros(model.gsde_noise_shape(batch_shape=(2,)))
    before = torch.random.get_rng_state().clone()
    torch.testing.assert_close(model._pi.action(z, deterministic=True, sde_noise=base), expected)
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="singular"):
        model._pi.action(z * 0, deterministic=True)
    with pytest.raises(RuntimeError, match="Scaled gSDE"):
        model._pi.action(z, deterministic=True, std_scale=1e300)
    with pytest.raises(ValueError, match="explicit matrix"):
        model._pi.action(z)


@pytest.mark.parametrize("detach_parameters", [False, True])
def test_action_only_fullgraph_compilation_preserves_gradients(detach_parameters, monkeypatch):
    import torch._inductor.config as inductor_config
    monkeypatch.setattr(inductor_config, "compile_threads", 1)
    model = _model(dtype=torch.float32)
    eager_model = deepcopy(model)
    z = torch.randn(4, model.cfg.latent_dim, requires_grad=True)
    eager_z = z.detach().clone().requires_grad_(True)
    base = torch.randn(model.gsde_noise_shape(batch_shape=(4,)))
    compiled = torch.compile(model._pi.action, backend="aot_eager", fullgraph=True)
    result = compiled(z, sde_noise=base, detach_parameters=detach_parameters)
    reference = eager_model._pi.action(eager_z, sde_noise=base, detach_parameters=detach_parameters)
    torch.testing.assert_close(result, reference)
    result.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(z.grad, eager_z.grad)
