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
