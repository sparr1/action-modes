"""Independent algebraic oracles for spectral handoff and its diagnostics."""

from copy import deepcopy
import json
import math

import pytest
import torch

from utils.spectral_transfer import spectral_energy, spectral_state, validate_spectral_spec


def states(dtype=torch.float64):
    prior = {"weight": torch.zeros((3, 3), dtype=dtype),
             "bias": torch.ones(3, dtype=dtype),
             "norm.weight": torch.ones(3, dtype=dtype),
             "matrix_buffer": torch.eye(3, dtype=dtype),
             "counter": torch.tensor(2)}
    donor = {key: value + 2 for key, value in prior.items()}
    donor["weight"] = torch.diag(torch.tensor([4., 2., 1.], dtype=dtype))
    return prior, donor


def transfer(prior, donor, *, method="svd", rank=1, **kwargs):
    spec_keys = {key: kwargs.pop(key) for key in ("strength", "norm_matched", "covariance_damping") if key in kwargs}
    return spectral_state(prior, donor, parameter_names={"weight", "bias", "norm.weight"},
                          spec={"method": method, "rank": rank, **spec_keys}, **kwargs)


def test_diagonal_svd_oracle_copies_matrix_only_and_preserves_inputs():
    prior, donor = states()
    saved_prior, saved_donor = deepcopy(prior), deepcopy(donor)
    metrics = {}
    result = transfer(prior, donor, metrics=metrics)
    torch.testing.assert_close(result["weight"], torch.diag(torch.tensor([4., 0., 0.], dtype=torch.float64)), rtol=0, atol=0)
    for key in prior:
        if key != "weight":
            assert torch.equal(result[key], prior[key])
        assert result[key].data_ptr() != prior[key].data_ptr()
        assert result[key].data_ptr() != donor[key].data_ptr()
        result[key].zero_()
        assert torch.equal(prior[key], saved_prior[key])
        assert torch.equal(donor[key], saved_donor[key])
    assert metrics["selected_rank_fraction"] == pytest.approx(1 / 3)
    assert metrics["transferred_energy_fraction"] == pytest.approx(16 / 21)
    assert metrics["relative_parameter_delta_l2"] is None


def test_activation_weighting_changes_the_selected_direction_and_is_weighted_optimal():
    prior, donor = states()
    x = torch.diag(torch.tensor([1., 3., 1.], dtype=torch.float64))
    damping = 1e-4
    metrics = {}
    result = transfer(prior, donor, method="activation", inputs={"weight": x},
                      covariance_damping=damping, metrics=metrics)["weight"]
    torch.testing.assert_close(result, torch.diag(torch.tensor([0., 2., 0.], dtype=torch.float64)), rtol=1e-12, atol=1e-12)
    covariance = x.T @ x / len(x)
    covariance += damping * covariance.diagonal().mean() * torch.eye(3, dtype=x.dtype)
    factor = torch.linalg.cholesky(covariance)
    weighted_delta = donor["weight"] @ factor
    # Eckart-Young's tail singular energy is an independent optimum oracle.
    optimum = torch.linalg.svdvals(weighted_delta)[1:].square().sum()
    objective = ((donor["weight"] - result) @ factor).square().sum()
    torch.testing.assert_close(objective, optimum)
    plain = transfer(prior, donor)["weight"]
    assert objective < ((donor["weight"] - plain) @ factor).square().sum()
    assert metrics["layers"]["weight"]["relative_activation_weighted_residual"] == pytest.approx(math.sqrt(float(optimum / weighted_delta.square().sum())))


def test_activation_general_nondiagonal_covariance_uses_correct_whitening_orientation():
    prior, donor = states()
    donor["weight"] = torch.tensor([[1., 2., 0.], [3., 1., 1.], [0., 4., 2.]], dtype=torch.float64)
    x = torch.tensor([[1., 2., 1.], [0., 1., 2.], [1., 0., 2.], [3., 1., 0.]], dtype=torch.float64)
    result = transfer(prior, donor, method="activation", inputs={"weight": x}, rank=2)["weight"]
    covariance = x.T @ x / len(x)
    covariance += 1e-4 * covariance.diagonal().mean() * torch.eye(3, dtype=x.dtype)
    factor = torch.linalg.cholesky(covariance)
    objective = ((donor["weight"] - result) @ factor).square().sum()
    optimum = torch.linalg.svdvals(donor["weight"] @ factor)[2:].square().sum()
    torch.testing.assert_close(objective, optimum, rtol=1e-10, atol=1e-10)
    assert torch.linalg.matrix_rank(result) == 2


def test_gradient_selects_benefit_not_magnitude_and_rejects_harmful_components():
    prior, donor = states()
    gradient = torch.diag(torch.tensor([1., -1., -3.], dtype=torch.float64))
    metrics = {}
    result = transfer(prior, donor, method="gradient", gradients={"weight": gradient}, metrics=metrics)
    torch.testing.assert_close(result["weight"], torch.diag(torch.tensor([0., 0., 1.], dtype=torch.float64)), rtol=0, atol=0)
    assert metrics["predicted_benefit"] == 3
    assert metrics["positive_benefit_count"] == 2
    assert metrics["scored_component_count"] == 3
    assert metrics["positive_benefit_fraction"] == pytest.approx(2 / 3)
    assert metrics["positive_benefit_energy_fraction"] == pytest.approx(5 / 21)
    assert metrics["layers"]["weight"]["selected_component_indices"] == [2]
    full_rank = transfer(prior, donor, method="gradient", rank=99, gradients={"weight": gradient})
    torch.testing.assert_close(full_rank["weight"], torch.diag(torch.tensor([0., 2., 1.], dtype=torch.float64)), rtol=0, atol=0)
    assert not torch.equal(full_rank["weight"], donor["weight"])
    negative = transfer(prior, donor, method="gradient", gradients={"weight": torch.eye(3, dtype=gradient.dtype)})
    assert torch.equal(negative["weight"], prior["weight"])
    zero = transfer(prior, donor, method="gradient", gradients={"weight": torch.zeros_like(gradient)})
    assert torch.equal(zero["weight"], prior["weight"])


@pytest.mark.parametrize("method", ["svd", "activation", "gradient"])
def test_norm_matched_dense_control_has_same_per_layer_norm(method):
    prior, donor = states()
    kwargs = {"method": method, "rank": 1, "strength": .5,
              "inputs": {"weight": torch.diag(torch.tensor([1., 3., 1.], dtype=torch.float64))},
              "gradients": {"weight": -torch.eye(3, dtype=torch.float64)}}
    filtered = transfer(prior, donor, **kwargs)["weight"]
    metrics = {}
    dense = transfer(prior, donor, **kwargs, norm_matched=True, metrics=metrics)["weight"]
    torch.testing.assert_close(dense.norm(), filtered.norm(), rtol=1e-12, atol=1e-12)
    expected_scale = float(filtered.norm() / donor["weight"].norm())
    torch.testing.assert_close(dense, donor["weight"] * expected_scale, rtol=1e-12, atol=1e-12)
    assert torch.linalg.matrix_rank(dense) == 3
    assert metrics["layers"]["weight"]["norm_matching_scale"] == pytest.approx(expected_scale)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_full_rank_and_zero_strength_endpoints_are_exact_and_detached(dtype):
    prior, donor = states(dtype)
    prior["weight"].fill_(1.3).requires_grad_()
    donor["weight"] += 1.3
    for matched in (False, True):
        full = transfer(prior, donor, rank=99, norm_matched=matched)
        zero = transfer(prior, donor, rank=2, strength=0, norm_matched=matched)
        assert torch.equal(full["weight"], donor["weight"])
        assert torch.equal(zero["weight"], prior["weight"])
        assert not full["weight"].requires_grad
        assert full["weight"].dtype == dtype


@pytest.mark.parametrize("method", ["svd", "activation", "gradient"])
def test_zero_delta_and_zero_covariance_are_finite_json_safe(method):
    prior, donor = states()
    donor["weight"].zero_()
    metrics = {}
    result = transfer(prior, donor, method=method, norm_matched=True, metrics=metrics,
                      inputs={"weight": torch.zeros((2, 3), dtype=torch.float64)},
                      gradients={"weight": torch.zeros((3, 3), dtype=torch.float64)})
    assert torch.equal(result["weight"], prior["weight"])
    json.dumps(metrics, allow_nan=False)
    info = metrics["layers"]["weight"]
    assert info["stable_rank"] == info["effective_rank"] == info["rank_99"] == 0
    assert info["energy_benefit_correlation"] is None
    assert info["energy_at_rank"]["128"] == 0


def test_spectral_energy_has_analytical_energy_ranks_and_benefit_statistics():
    delta = torch.diag(torch.tensor([4., 2., 1.], dtype=torch.float64))
    gradient = torch.diag(torch.tensor([1., -1., -3.], dtype=torch.float64))
    info = spectral_energy(delta, ranks=(1, 2, 3, 10), gradient=gradient)
    assert info["singular_values"] == [4, 2, 1]
    assert info["cumulative_energy"] == pytest.approx([16 / 21, 20 / 21, 1])
    assert info["stable_rank"] == pytest.approx(21 / 16)
    assert [info[f"rank_{p}"] for p in (50, 90, 95, 99)] == [1, 2, 2, 3]
    assert info["effective_rank"] == pytest.approx(math.exp(-sum(p * math.log(p) for p in (4/7, 2/7, 1/7))))
    assert info["component_benefits"] == [-4, 2, 3]
    assert info["positive_benefit_fraction"] == pytest.approx(2 / 3)
    assert info["positive_benefit_energy_fraction"] == pytest.approx(5 / 21)
    assert info["top_energy_benefit"] == {"1": -4, "2": -2, "3": 1, "10": 1}
    assert info["top_energy_positive_benefit_fraction"]["10"] == pytest.approx(2 / 3)
    assert info["energy_benefit_correlation"] < 0
    json.dumps(info, allow_nan=False)


def test_spectral_operations_do_not_consume_torch_rng():
    prior, donor = states()
    saved_rng = torch.random.get_rng_state().clone()
    for method in ("svd", "activation", "gradient"):
        transfer(prior, donor, method=method,
                 inputs={"weight": torch.eye(3, dtype=torch.float64)},
                 gradients={"weight": -torch.eye(3, dtype=torch.float64)})
    spectral_energy(donor["weight"])
    assert torch.equal(saved_rng, torch.random.get_rng_state())


@pytest.mark.parametrize("method", ["svd", "activation", "gradient"])
def test_zero_strength_needs_no_context_or_svd_and_reports_only_measured_quantities(method, monkeypatch):
    prior, donor = states()
    def unexpected(*args, **kwargs):
        pytest.fail("zero strength must not compute a decomposition")
    monkeypatch.setattr(torch.linalg, "svd", unexpected)
    metrics = {}
    result = transfer(prior, donor, method=method, strength=0, metrics=metrics)
    for name in prior:
        assert torch.equal(result[name], prior[name])
    assert metrics["layers"] == {}
    assert metrics["transferred_delta_l2"] == metrics["selected_rank_sum"] == 0
    assert metrics["donor_delta_l2"] == pytest.approx(math.sqrt(21))
    assert metrics["positive_benefit_fraction"] is None
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize("invalid", [
    {}, {"method": "random", "rank": 1}, {"method": "svd", "rank": True},
    {"method": "svd", "rank": 0}, {"method": "svd", "rank": 1.5},
    {"method": "svd", "rank": 1, "strength": True},
    {"method": "svd", "rank": 1, "strength": float("nan")},
    {"method": "svd", "rank": 1, "strength": 1.1},
    {"method": "svd", "rank": 1, "covariance_damping": 0},
    {"method": "svd", "rank": 1, "covariance_damping": float("inf")},
    {"method": "svd", "rank": 1, "norm_matched": 1},
    {"method": "svd", "rank": 1, "typo": 1},
])
def test_invalid_specifications_are_rejected(invalid):
    with pytest.raises(ValueError):
        validate_spectral_spec(invalid)


@pytest.mark.parametrize("bad", ["keys", "shape", "dtype", "nan_donor", "nan_buffer", "unknown_parameter", "missing_inputs", "missing_gradients", "input_width", "input_dtype", "gradient_shape", "extra_input"])
def test_invalid_states_and_probes_leave_inputs_and_metrics_untouched(bad):
    prior, donor = states()
    kwargs = {"parameter_names": {"weight"}, "spec": {"method": "svd", "rank": 1}}
    if bad == "keys":
        donor.pop("bias")
    elif bad == "shape":
        donor["weight"] = torch.zeros((3, 2), dtype=torch.float64)
    elif bad == "dtype":
        donor["weight"] = donor["weight"].float()
    elif bad == "nan_donor":
        donor["weight"][0, 0] = float("nan")
    elif bad == "nan_buffer":
        donor["matrix_buffer"][0, 0] = float("nan")
    elif bad == "unknown_parameter":
        kwargs["parameter_names"] = {"not_a_key"}
    elif bad == "missing_inputs":
        kwargs["spec"]["method"] = "activation"
    elif bad == "missing_gradients":
        kwargs["spec"]["method"] = "gradient"
    elif bad == "input_width":
        kwargs["inputs"] = {"weight": torch.ones((2, 2), dtype=torch.float64)}
    elif bad == "input_dtype":
        kwargs["inputs"] = {"weight": torch.eye(3, dtype=torch.float32)}
    elif bad == "gradient_shape":
        kwargs["gradients"] = {"weight": torch.ones((2, 3), dtype=torch.float64)}
    else:
        kwargs["inputs"] = {"matrix_buffer": torch.eye(3, dtype=torch.float64)}
    saved = deepcopy(prior)
    metrics = {"existing": 12}
    with pytest.raises(ValueError):
        spectral_state(prior, donor, metrics=metrics, **kwargs)
    assert metrics == {"existing": 12}
    for key in prior:
        torch.testing.assert_close(prior[key], saved[key], rtol=0, atol=0)


def test_multiple_layers_are_norm_matched_independently():
    prior = {"a": torch.zeros((2, 2)), "b": torch.zeros((2, 2))}
    donor = {"a": torch.diag(torch.tensor([3., 1.])), "b": torch.diag(torch.tensor([2., 2.]))}
    metrics = {}
    result = spectral_state(prior, donor, parameter_names=prior,
                            spec={"method": "svd", "rank": 1, "norm_matched": True}, metrics=metrics)
    torch.testing.assert_close(result["a"].norm(), torch.tensor(3.))
    torch.testing.assert_close(result["b"].norm(), torch.tensor(2.))
    assert metrics["layers"]["a"]["norm_matching_scale"] != metrics["layers"]["b"]["norm_matching_scale"]


@pytest.mark.parametrize("method", ["svd", "activation", "gradient"])
@pytest.mark.parametrize("norm_matched", [False, True])
def test_aggregate_only_metrics_preserve_transfer_and_skip_extra_activation_svd(method, norm_matched, monkeypatch):
    prior, donor = states()
    kwargs = dict(method=method, norm_matched=norm_matched,
        inputs={"weight": torch.diag(torch.tensor([1., 3., 1.], dtype=torch.float64))},
        gradients={"weight": torch.diag(torch.tensor([1., -1., -3.], dtype=torch.float64))})
    detailed, aggregate = {}, {}
    svd = torch.linalg.svd
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return svd(*args, **kwargs)
    monkeypatch.setattr(torch.linalg, "svd", counted)
    full = transfer(prior, donor, **kwargs, metrics=detailed)
    assert len(calls) == (2 if method == "activation" else 1)
    calls.clear()
    lean = transfer(prior, donor, **kwargs, metrics=aggregate, include_layer_metrics=False)
    assert len(calls) == 1
    assert aggregate["layers"] == {}
    for key in full:
        assert torch.equal(full[key], lean[key])
    for key in ("donor_delta_l2", "transferred_delta_l2", "transferred_energy_fraction", "predicted_benefit", "selected_rank_sum"):
        assert aggregate[key] == detailed[key]
    if method == "activation":
        assert aggregate["positive_benefit_fraction"] is None
    else:
        assert aggregate["positive_benefit_fraction"] == detailed["positive_benefit_fraction"]


def projection_states(dtype=torch.float64):
    prior = {"a": torch.ones((2, 2), dtype=dtype),
             "b": torch.ones((2, 2), dtype=dtype) * 2,
             "bias": torch.ones(2, dtype=dtype),
             "matrix_buffer": torch.eye(2, dtype=dtype)}
    donor = {key: value + 7 for key, value in prior.items()}
    donor["a"] = prior["a"] + torch.tensor([[3., 0.], [0., 1.]], dtype=dtype)
    donor["b"] = prior["b"] + torch.tensor([[0., -2.], [1., 0.]], dtype=dtype)
    gradients = {"a": torch.eye(2, dtype=dtype), "b": torch.ones((2, 2), dtype=dtype)}
    return prior, donor, gradients


def project(prior, donor, gradients=None, **kwargs):
    spec = {"method": "gradient_projection", **kwargs.pop("spec", {})}
    return spectral_state(prior, donor, parameter_names={"a", "b", "bias"},
                          gradients=gradients, spec=spec, **kwargs)


def test_gradient_projection_is_one_global_signed_direction_with_reset_vectors_and_buffers():
    prior, donor, gradients = projection_states()
    saved = deepcopy((prior, donor, gradients))
    metrics = {}
    result = project(prior, donor, gradients, metrics=metrics)
    # Global dot=4-1=3 and gradient squared norm=2+4=6. Per-layer
    # projections would incorrectly have coefficients 2 and -1/4.
    for key in ("a", "b"):
        torch.testing.assert_close(result[key], prior[key] + .5 * gradients[key], rtol=0, atol=0)
    for key in ("bias", "matrix_buffer"):
        assert torch.equal(result[key], prior[key])
    for original, copy in zip((prior, donor, gradients), saved):
        for key in original:
            assert torch.equal(original[key], copy[key])
    for key in prior:
        assert result[key].data_ptr() not in (prior[key].data_ptr(), donor[key].data_ptr())
    assert metrics["projection_coefficient"] == .5
    assert metrics["donor_gradient_inner_product"] == 3
    assert metrics["gradient_squared_norm"] == 6
    assert metrics["predicted_benefit"] == -3  # Uphill projection is retained.
    assert metrics["selected_rank_sum"] is metrics["selected_rank_fraction"] is None
    assert torch.linalg.matrix_rank(result["a"] - prior["a"]) == 2
    assert all(info["selected_rank"] is info["selected_component_indices"] is None
               for info in metrics["layers"].values())
    # Orthogonal projection preserves the donor's first-order loss change.
    assert metrics["predicted_benefit"] == -metrics["donor_gradient_inner_product"]
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize("scale", [.125, -3., 1e-30, 1e30])
def test_gradient_projection_is_invariant_to_gradient_scale_and_sign(scale):
    prior, donor, gradients = projection_states(torch.float32)
    result = project(prior, donor, gradients, include_layer_metrics=False)
    rescaled = project(prior, donor, {key: value * scale for key, value in gradients.items()},
                       include_layer_metrics=False)
    for key in result:
        torch.testing.assert_close(rescaled[key], result[key], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_gradient_projection_preserves_dtype_is_detached_and_projects_downhill(dtype):
    prior, donor, gradients = projection_states(dtype)
    for key in ("a", "b"):
        gradients[key].neg_()
        prior[key].requires_grad_()
        donor[key].requires_grad_()
    metrics = {}
    result = project(prior, donor, gradients, metrics=metrics, include_layer_metrics=False)
    assert metrics["projection_coefficient"] == -.5
    assert metrics["predicted_benefit"] == 3
    for key in result:
        assert result[key].dtype == prior[key].dtype
        assert not result[key].requires_grad


@pytest.mark.parametrize("norm_matched", [False, True])
def test_gradient_projection_zero_gradient_is_exact_prior(norm_matched):
    prior, donor, gradients = projection_states()
    metrics = {}
    result = project(prior, donor, {key: value * 0 for key, value in gradients.items()},
                     spec={"norm_matched": norm_matched}, metrics=metrics)
    for key in prior:
        assert torch.equal(result[key], prior[key])
    assert metrics["gradient_squared_norm"] == metrics["projection_coefficient"] == 0
    assert metrics["transferred_delta_l2"] == metrics["predicted_benefit"] == 0
    json.dumps(metrics, allow_nan=False)


def test_gradient_projection_dense_control_matches_global_norm_with_zero_donor_layer():
    prior, donor, gradients = projection_states()
    donor["b"] = prior["b"].clone()
    filtered_metrics, control_metrics = {}, {}
    filtered = project(prior, donor, gradients, spec={"strength": .25}, metrics=filtered_metrics)
    control = project(prior, donor, gradients, spec={"strength": .25, "norm_matched": True}, metrics=control_metrics)
    delta = torch.cat([(donor[key] - prior[key]).flatten() for key in ("a", "b")])
    projected = torch.cat([(filtered[key] - prior[key]).flatten() for key in ("a", "b")])
    dense = torch.cat([(control[key] - prior[key]).flatten() for key in ("a", "b")])
    torch.testing.assert_close(projected.norm(), dense.norm(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(dense, delta * (projected.norm() / delta.norm()), rtol=1e-12, atol=1e-12)
    assert not torch.equal(filtered["b"], prior["b"])
    assert torch.equal(control["b"], prior["b"])
    assert filtered_metrics["layers"]["b"]["transferred_energy_fraction"] is None
    assert filtered_metrics["layers"]["b"]["relative_parameter_residual"] is None
    assert control_metrics["norm_matching_scope"] == "component-global"
    assert all(info["norm_matching_scale"] == control_metrics["norm_matching_scale"]
               for info in control_metrics["layers"].values())
    # The dense control has the same norm but its own measured benefit.
    expected_benefit = -sum(float(((control[key] - prior[key]) * gradients[key]).sum()) for key in ("a", "b"))
    assert control_metrics["predicted_benefit"] == pytest.approx(expected_benefit)
    json.dumps(filtered_metrics, allow_nan=False)


def test_gradient_projection_controller_has_no_svd_and_same_results_as_diagnostics(monkeypatch):
    prior, donor, gradients = projection_states()
    full_metrics = {}
    full = project(prior, donor, gradients, metrics=full_metrics)
    rng = torch.random.get_rng_state().clone()
    def unexpected(*args, **kwargs):
        pytest.fail("gradient projection controller must not call SVD")
    monkeypatch.setattr(torch.linalg, "svd", unexpected)
    monkeypatch.setattr(torch.linalg, "svdvals", unexpected)
    metrics = {}
    lean = project(prior, donor, gradients, metrics=metrics, include_layer_metrics=False)
    no_metrics = project(prior, donor, gradients)
    for key in full:
        assert torch.equal(full[key], lean[key])
        assert torch.equal(full[key], no_metrics[key])
    assert metrics["layers"] == {}
    assert metrics["positive_benefit_fraction"] is None
    for key in ("donor_delta_l2", "transferred_delta_l2", "projection_coefficient",
                "donor_gradient_inner_product", "gradient_squared_norm", "predicted_benefit"):
        assert metrics[key] == full_metrics[key]
    assert torch.equal(rng, torch.random.get_rng_state())


def test_gradient_projection_zero_strength_needs_no_probe_and_does_not_claim_rank(monkeypatch):
    prior, donor, _ = projection_states()
    def unexpected(*args, **kwargs):
        pytest.fail("zero-strength projection must not call SVD")
    monkeypatch.setattr(torch.linalg, "svd", unexpected)
    metrics = {}
    result = project(prior, donor, spec={"strength": 0}, metrics=metrics)
    assert all(torch.equal(result[key], prior[key]) for key in prior)
    for name in ("projection_coefficient", "gradient_squared_norm", "donor_gradient_inner_product", "selected_rank_sum"):
        assert metrics[name] is None
    assert metrics["transferred_delta_l2"] == metrics["predicted_benefit"] == 0


@pytest.mark.parametrize("rank", [1, 0, -1, True, False, 1.0, "1", float("nan")])
def test_gradient_projection_rejects_matrix_rank(rank):
    with pytest.raises(ValueError, match="rank must be absent or None"):
        validate_spectral_spec({"method": "gradient_projection", "rank": rank})


def test_gradient_projection_normalizes_absent_and_null_rank_identically():
    assert validate_spectral_spec({"method": "gradient_projection"}) == validate_spectral_spec({"method": "gradient_projection", "rank": None})


@pytest.mark.parametrize("bad", ["missing", "shape", "dtype", "nonfinite", "overflow", "underflow"])
def test_gradient_projection_invalid_inputs_fail_atomically(bad):
    prior, donor, gradients = projection_states()
    if bad == "missing":
        gradients.pop("b")
    elif bad == "shape":
        gradients["b"] = torch.ones((3, 2), dtype=torch.float64)
    elif bad == "dtype":
        gradients["b"] = gradients["b"].float()
    elif bad == "nonfinite":
        gradients["b"][0, 0] = float("nan")
    elif bad == "overflow":
        gradients["b"].fill_(1e300)
    else:
        gradients = {key: value * 1e-300 for key, value in gradients.items()}
    saved = deepcopy((prior, donor, gradients))
    metrics = {"existing": 42}
    with pytest.raises(ValueError):
        project(prior, donor, gradients, metrics=metrics, include_layer_metrics=False)
    assert metrics == {"existing": 42}
    for actual, copy in zip((prior, donor, gradients), saved):
        for key in actual:
            torch.testing.assert_close(actual[key], copy[key], rtol=0, atol=0, equal_nan=True)


def gate_states(dtype=torch.float64):
    prior = {"weight": torch.ones((2, 3), dtype=dtype),
             "bias": torch.ones(2, dtype=dtype),
             "matrix_buffer": torch.ones((2, 3), dtype=dtype)}
    donor = {key: value + 7 for key, value in prior.items()}
    donor["weight"] = prior["weight"] + torch.tensor([[2., -3., 0.], [-4., 5., 6.]], dtype=dtype)
    gradients = {"weight": torch.tensor([[-1., 2., -3.], [-2., 0., 3.]], dtype=dtype)}
    return prior, donor, gradients


def gate(prior, donor, gradients=None, **kwargs):
    return spectral_state(prior, donor, parameter_names={"weight", "bias"},
        gradients=gradients, spec={"method": "gradient_gate", **kwargs.pop("spec", {})}, **kwargs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("strength", [.5, 1.])
def test_gradient_gate_strict_coordinate_oracle_resets_vectors_and_buffers(dtype, strength):
    prior, donor, gradients = gate_states(dtype)
    donor["weight"].requires_grad_()
    saved = deepcopy((prior, donor, gradients))
    metrics = {}
    output = gate(prior, donor, gradients, spec={"strength": strength}, metrics=metrics)
    expected_delta = torch.tensor([[2., -3., 0.], [0., 0., 0.]], dtype=dtype) * strength
    torch.testing.assert_close(output["weight"], prior["weight"] + expected_delta, rtol=0, atol=0)
    assert output["weight"].dtype == dtype and not output["weight"].requires_grad
    for key in ("bias", "matrix_buffer"):
        assert torch.equal(output[key], prior[key])
    for actual, original in zip((prior, donor, gradients), saved):
        for key in actual:
            assert torch.equal(actual[key], original[key])
    assert metrics["retained_fraction"] == pytest.approx(2 / 6)
    assert metrics["retained_parameter_count"] == 2 and metrics["parameter_count"] == 6
    assert metrics["donor_gradient_inner_product"] == 18
    assert metrics["predicted_benefit"] == 8 * strength
    assert metrics["gradient_squared_norm"] == 27
    assert metrics["transferred_energy_fraction"] == pytest.approx(13 * strength ** 2 / 90)
    assert metrics["selected_rank_sum"] is metrics["selected_rank_fraction"] is None
    assert metrics["layers"]["weight"]["selected_rank"] is None
    json.dumps(metrics, allow_nan=False)


def test_gradient_gate_dense_norm_control_matches_each_layer_and_may_remain_uphill():
    prior, donor, gradients = gate_states()
    prior["second"] = torch.ones((2, 3), dtype=torch.float64)
    donor["second"] = prior["second"] + 1
    gradients["second"] = -torch.ones((2, 3), dtype=torch.float64)
    outputs, metrics = [], []
    for matched in (False, True):
        current = {}
        outputs.append(spectral_state(prior, donor, parameter_names={"weight", "second", "bias"},
            spec={"method": "gradient_gate", "strength": .5, "norm_matched": matched},
            gradients=gradients, metrics=current))
        metrics.append(current)
    for key in ("weight", "second"):
        torch.testing.assert_close((outputs[0][key] - prior[key]).norm(), (outputs[1][key] - prior[key]).norm())
        expected_scale = .5 * math.sqrt(13 / 90) if key == "weight" else .5
        torch.testing.assert_close(outputs[1][key] - prior[key], (donor[key] - prior[key]) * expected_scale)
        assert metrics[1]["layers"][key]["norm_matching_scale"] == pytest.approx(expected_scale)
    assert metrics[1]["layers"]["weight"]["predicted_transfer_benefit"] < 0
    assert metrics[0]["retained_fraction"] == metrics[1]["retained_fraction"] == pytest.approx(8 / 12)
    assert metrics[1]["norm_matching_scope"] == "per-layer"


@pytest.mark.parametrize("matched", [False, True])
def test_gradient_gate_no_beneficial_coordinates_returns_exact_prior(matched):
    prior, donor, gradients = gate_states()
    for gradient in (torch.zeros_like(gradients["weight"]), donor["weight"] - prior["weight"]):
        metrics = {}
        output = gate(prior, donor, {"weight": gradient}, spec={"norm_matched": matched}, metrics=metrics)
        for key in prior:
            assert torch.equal(output[key], prior[key])
        assert metrics["retained_fraction"] == metrics["transferred_delta_l2"] == metrics["predicted_benefit"] == 0
        json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize("matched", [False, True])
def test_gradient_gate_controller_has_no_svd_or_randomness_and_matches_diagnostics(monkeypatch, matched):
    prior, donor, gradients = gate_states()
    detailed, lean = {}, {}
    rng = torch.random.get_rng_state().clone()
    full = gate(prior, donor, gradients, spec={"norm_matched": matched}, metrics=detailed)
    def unexpected(*args, **kwargs):
        pytest.fail("gradient-gated controller must not compute SVD")
    monkeypatch.setattr(torch.linalg, "svd", unexpected)
    output = gate(prior, donor, gradients, spec={"norm_matched": matched}, metrics=lean, include_layer_metrics=False)
    assert torch.equal(rng, torch.random.get_rng_state())
    for key in prior:
        assert torch.equal(full[key], output[key])
    for key in ("retained_fraction", "retained_parameter_count", "predicted_benefit", "donor_gradient_inner_product",
                "gradient_squared_norm", "donor_delta_l2", "transferred_delta_l2", "transferred_energy_fraction"):
        assert detailed[key] == lean[key]
    assert lean["layers"] == {} and lean["positive_benefit_fraction"] is None


def test_gradient_gate_zero_strength_needs_no_gradients_or_svd(monkeypatch):
    prior, donor, _ = gate_states()
    def unexpected(*args, **kwargs):
        pytest.fail("zero-strength gradient gate must not compute SVD")
    monkeypatch.setattr(torch.linalg, "svd", unexpected)
    metrics = {}
    output = gate(prior, donor, spec={"strength": 0}, metrics=metrics)
    for key in prior:
        assert torch.equal(output[key], prior[key])
    assert metrics["predicted_benefit"] == metrics["transferred_delta_l2"] == 0
    for name in ("retained_fraction", "retained_parameter_count", "donor_gradient_inner_product", "selected_rank_sum"):
        assert metrics[name] is None
    json.dumps(metrics, allow_nan=False)


def test_gradient_gate_uses_strict_sign_even_when_product_underflows():
    prior = {"weight": torch.zeros((1, 1), dtype=torch.float64)}
    donor = {"weight": torch.full((1, 1), 1e-200, dtype=torch.float64)}
    gradient = {"weight": torch.full((1, 1), -1e-200, dtype=torch.float64)}
    output = spectral_state(prior, donor, parameter_names={"weight"},
        spec={"method": "gradient_gate"}, gradients=gradient, include_layer_metrics=False)
    assert torch.equal(output["weight"], donor["weight"])


@pytest.mark.parametrize("rank", [0, 1, 4, -1, 1.5, True, "1"])
def test_gradient_gate_rejects_rank_and_normalizes_none(rank):
    with pytest.raises(ValueError, match="rank must be absent or None"):
        validate_spectral_spec({"method": "gradient_gate", "rank": rank})
    assert validate_spectral_spec({"method": "gradient_gate"}) == validate_spectral_spec({"method": "gradient_gate", "rank": None})


@pytest.mark.parametrize("bad", ["missing", "shape", "dtype", "nonfinite", "overflow"])
def test_gradient_gate_invalid_gradients_fail_atomically(bad):
    prior, donor, gradients = gate_states()
    if bad == "missing":
        gradients.clear()
    elif bad == "shape":
        gradients["weight"] = torch.ones((3, 2), dtype=torch.float64)
    elif bad == "dtype":
        gradients["weight"] = gradients["weight"].float()
    elif bad == "nonfinite":
        gradients["weight"][0, 0] = float("nan")
    else:
        gradients["weight"].fill_(1e300)
    metrics = {"existing": 7}
    with pytest.raises(ValueError):
        gate(prior, donor, gradients, metrics=metrics, include_layer_metrics=False)
    assert metrics == {"existing": 7}
