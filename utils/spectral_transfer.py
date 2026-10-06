"""Deterministic matrix-delta transfer between frozen-prior inner solves.

Only named two-dimensional parameters are transferred. Biases, normalization
parameters and all buffers remain at the prior; the caller resets learner
state and resumes ordinary dense training. No operation samples randomness or
modifies its arguments.

Activation transfer minimizes ``||(delta - R) C**0.5||_F`` over rank-r R, with
``C = X.T @ X / N + damping * scale * I``. Scale is the unregularized mean
diagonal, or one for an all-zero covariance. Cholesky whitening and a triangular
solve implement the objective without explicitly inverting C. Gradient
transfer instead ranks donor SVD components by their *positive* first-order
loss reduction, ``-<G, sigma_i u_i v_i.T>``. Those scores are local predictions,
not guarantees of post-adaptation or environment return improvement.
"""

from collections.abc import Mapping, MutableMapping
import math
from numbers import Integral, Real

import torch


DEFAULT_ENERGY_RANKS = (1, 2, 4, 8, 16, 32, 64, 128)


def validate_spectral_spec(spec):
    """Validate a transfer rule without silently accepting misspelled options."""
    if not isinstance(spec, Mapping):
        raise ValueError("spectral specification must be a mapping")
    allowed = {"method", "rank", "strength", "norm_matched", "covariance_damping"}
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"Unknown spectral options: {sorted(unknown)}")
    method, rank = spec.get("method"), spec.get("rank")
    if method not in ("svd", "activation", "gradient"):
        raise ValueError("spectral method must be svd, activation or gradient")
    if isinstance(rank, bool) or not isinstance(rank, Integral) or rank < 1:
        raise ValueError("spectral rank must be a positive integer")
    strength = spec.get("strength", 1.0)
    damping = spec.get("covariance_damping", 1e-4)
    for name, value in (("strength", strength), ("covariance_damping", damping)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"spectral {name} must be a finite real number")
    if not 0 <= strength <= 1:
        raise ValueError("spectral strength must be in [0, 1]")
    if damping <= 0:
        raise ValueError("spectral covariance_damping must be positive")
    norm_matched = spec.get("norm_matched", False)
    if not isinstance(norm_matched, bool):
        raise ValueError("spectral norm_matched must be boolean")
    return {"method": method, "rank": int(rank), "strength": float(strength),
            "norm_matched": norm_matched, "covariance_damping": float(damping)}


def _finite_tensor(value, name):
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise ValueError(f"{name} must be a dense tensor")
    if value.is_complex() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain finite real values")


def _work_dtype(tensor):
    # CUDA supports fp32/fp64 exact SVD, but not half/bfloat16 SVD.
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _matrix(value, name):
    _finite_tensor(value, name)
    if value.ndim != 2 or min(value.shape) < 1 or not value.is_floating_point():
        raise ValueError(f"{name} must be a nonempty floating-point matrix")


def _validate_auxiliary(auxiliary, name, prior, matrix_names, *, required):
    if auxiliary is None:
        if required and matrix_names:
            raise ValueError(f"spectral transfer requires {name} for every matrix")
        return {}
    if not isinstance(auxiliary, Mapping):
        raise ValueError(f"spectral {name} must be a mapping")
    if set(auxiliary) - set(matrix_names):
        raise ValueError(f"spectral {name} has unknown or nonmatrix parameter keys")
    if required and set(auxiliary) != set(matrix_names):
        raise ValueError(f"spectral transfer requires {name} for every matrix")
    for key, value in auxiliary.items():
        _matrix(value, f"{name}[{key}]")
        reference = prior[key]
        expected = value.shape == reference.shape if name == "gradients" else value.shape[1] == reference.shape[1]
        if not expected or value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError(f"Incompatible spectral {name}[{key}] shape, device or dtype")
    return auxiliary


def _svd(delta):
    factors = torch.linalg.svd(delta, full_matrices=False)
    for factor in factors:
        _finite_tensor(factor, "SVD factor")
    return factors


def _benefits(u, singular_values, vh, gradient):
    scores = -singular_values * ((u.T @ gradient) * vh).sum(dim=1)
    _finite_tensor(scores, "spectral gradient scores")
    return scores


def _energy_from_svd(u, singular_values, vh, ranks, gradient, *, benefits=None):
    # Convert only compact spectra and scores to CPU; all matrix algebra stays
    # on its original device. Python arithmetic keeps the payload strict JSON.
    values = singular_values.detach().double().cpu().tolist()
    energies = [value * value for value in values]
    energy = math.fsum(energies)
    fractions = [value / energy for value in energies] if energy else [0.0] * len(values)
    cumulative, running = [], 0.0
    for fraction in fractions:
        running += fraction
        cumulative.append(min(1.0, running))
    nuclear = math.fsum(values)
    probabilities = [value / nuclear for value in values] if nuclear else []
    effective_rank = math.exp(-math.fsum(p * math.log(p) for p in probabilities if p > 0)) if nuclear else 0.0
    energy_effective_rank = math.exp(-math.fsum(p * math.log(p) for p in fractions if p > 0)) if energy else 0.0
    ranks = tuple(ranks)
    for rank in ranks:
        if isinstance(rank, bool) or not isinstance(rank, Integral) or rank < 1:
            raise ValueError("energy ranks must be positive integers")
    payload = {
        "singular_values": values, "cumulative_energy": cumulative,
        "delta_frobenius_norm": math.sqrt(energy),
        "stable_rank": energy / energies[0] if energy else 0.0,
        # Standard effective rank uses normalized singular values, while the
        # explicitly named energy version uses normalized squared values.
        "effective_rank": effective_rank, "energy_effective_rank": energy_effective_rank,
        "energy_at_rank": {str(rank): cumulative[min(int(rank), len(values)) - 1] if energy else 0.0 for rank in ranks},
    }
    for percent in (50, 90, 95, 99):
        payload[f"rank_{percent}"] = next((i + 1 for i, value in enumerate(cumulative) if value + 1e-15 >= percent / 100), len(values)) if energy else 0
    if gradient is not None:
        scores = (_benefits(u, singular_values, vh, gradient) if benefits is None else benefits).detach().double().cpu().tolist()
        positive = [score > 0 for score in scores]
        mean_energy, mean_score = energy / len(scores), math.fsum(scores) / len(scores)
        centered_energy = [value - mean_energy for value in energies]
        centered_scores = [value - mean_score for value in scores]
        denominator = math.sqrt(math.fsum(value * value for value in centered_energy) * math.fsum(value * value for value in centered_scores))
        payload.update({
            "component_benefits": scores,
            "predicted_full_delta_benefit": math.fsum(scores),
            "positive_benefit_fraction": sum(positive) / len(scores),
            "positive_benefit_energy_fraction": math.fsum(value for value, keep in zip(fractions, positive) if keep),
            "energy_benefit_correlation": math.fsum(a * b for a, b in zip(centered_energy, centered_scores)) / denominator if denominator else None,
            "top_energy_benefit": {str(rank): math.fsum(scores[:int(rank)]) for rank in ranks},
            "top_energy_positive_benefit_fraction": {str(rank): sum(positive[:int(rank)]) / min(int(rank), len(values)) for rank in ranks},
        })
    return payload


@torch.no_grad()
def spectral_energy(delta, ranks=DEFAULT_ENERGY_RANKS, gradient=None):
    """Return JSON-safe singular energy and optional component benefit metrics.

    Zero deltas have zero ranks and energy fractions. Correlation is ``None``
    when either vector is constant. Gradient scores depend on the chosen SVD
    basis inside a repeated-singular-value subspace. No randomized SVD is used.
    """
    _matrix(delta, "delta")
    if gradient is not None:
        _matrix(gradient, "gradient")
        if gradient.shape != delta.shape or gradient.dtype != delta.dtype or gradient.device != delta.device:
            raise ValueError("gradient must match delta shape, dtype and device")
    work = delta.detach().to(_work_dtype(delta))
    u, s, vh = _svd(work)
    g = gradient.detach().to(work.dtype) if gradient is not None else None
    return _energy_from_svd(u, s, vh, ranks, g)


@torch.no_grad()
def spectral_state(prior, donor, *, parameter_names, spec, inputs=None, gradients=None, metrics=None,
                   include_layer_metrics=True):
    """Return cloned prior state with selected matrix deltas added.

    ``inputs`` maps each matrix key to an N-by-input-width activation tensor;
    ``gradients`` maps each key to a gradient matrix. Both must match parameter
    device and dtype. Norm matching is per layer: it replaces filtered transfer
    by a scalar multiple of the dense donor delta with the same Frobenius norm.
    Norm matching may exceed unit dense strength for activation-weighted SVD.
    ``relative_parameter_delta_l2`` divides by the prior norm over transferred
    matrix parameters only, matching matrix-only baseline controls.
    Set ``include_layer_metrics=False`` for the every-decision controller path:
    full spectra and residual diagnostics are omitted, and activation filtering
    performs only the weighted SVD it needs. Uncomputed donor spectral benefit
    fractions are then ``None``; predicted benefit of the actual transfer is
    still available whenever a gradient is supplied.
    All validation precedes any metrics publication, and failures leave all
    inputs and the optional metrics mapping untouched.
    """
    normalized = validate_spectral_spec(spec)
    if not isinstance(include_layer_metrics, bool):
        raise ValueError("include_layer_metrics must be boolean")
    if not isinstance(prior, Mapping) or not isinstance(donor, Mapping) or set(prior) != set(donor):
        raise ValueError("Incompatible prior and donor state keys")
    try:
        parameter_names = set(parameter_names)
    except TypeError as exc:
        raise ValueError("parameter_names must be an iterable of state keys") from exc
    if parameter_names - set(prior):
        raise ValueError("parameter_names contains unknown state keys")
    if metrics is not None and not isinstance(metrics, MutableMapping):
        raise ValueError("metrics must be a mutable mapping")
    for key, value in prior.items():
        _finite_tensor(value, f"prior[{key}]")
        _finite_tensor(donor[key], f"donor[{key}]")
        if value.shape != donor[key].shape or value.dtype != donor[key].dtype or value.device != donor[key].device:
            raise ValueError(f"Incompatible donor tensor {key}")
    matrices = [key for key in prior if key in parameter_names and prior[key].ndim == 2]
    for key in matrices:
        _matrix(prior[key], f"prior[{key}]")
    method = normalized["method"]
    nonzero = normalized["strength"] > 0
    inputs = _validate_auxiliary(inputs, "inputs", prior, matrices, required=nonzero and method == "activation")
    gradients = _validate_auxiliary(gradients, "gradients", prior, matrices, required=nonzero and method == "gradient")
    result = {key: value.detach().clone() for key, value in prior.items()}
    if not nonzero:
        # The exact zero endpoint does not need a probe or decomposition. Do
        # not fill in unmeasured spectra or claim a selected nonzero rank.
        if metrics is not None:
            delta_square = math.fsum(float((donor[key].detach().double() - prior[key].detach().double()).square().sum().item()) for key in matrices)
            prior_square = math.fsum(float(prior[key].detach().double().square().sum().item()) for key in matrices)
            metrics.update({
                "layers": {}, "matrix_count": len(matrices),
                "selected_rank_sum": 0, "matrix_rank_capacity_sum": sum(min(prior[key].shape) for key in matrices),
                "selected_rank_fraction": 0.0, "donor_delta_l2": math.sqrt(delta_square),
                "transferred_delta_l2": 0.0, "transferred_energy_fraction": 0.0,
                "relative_parameter_delta_l2": 0.0 if prior_square else None,
                "predicted_benefit": 0.0, "positive_benefit_count": None,
                "scored_component_count": None, "positive_benefit_fraction": None,
                "positive_benefit_energy_fraction": None,
            })
        return result
    details = {}
    donor_square = transfer_square = 0.0
    prior_square = math.fsum(float(prior[key].detach().double().square().sum().item())
                            for key in matrices)
    selected_rank_sum = matrix_rank_capacity_sum = 0
    positive_benefit_count = scored_component_count = 0
    positive_benefit_energy = scored_donor_energy = 0.0
    predicted_benefit = 0.0 if gradients else None
    for key in matrices:
        base = prior[key].detach().to(_work_dtype(prior[key]))
        delta = donor[key].detach().to(base.dtype) - base
        _matrix(delta, f"delta[{key}]")
        rank = min(normalized["rank"], min(delta.shape))
        u, s, vh = (_svd(delta) if method != "activation" or (metrics is not None and include_layer_metrics)
                    else (None, None, None))
        gradient = gradients[key].detach().to(base.dtype) if key in gradients else None
        covariance_factor = None
        scores = None
        if method == "activation":
            x = inputs[key].detach().to(base.dtype)
            covariance = x.T @ x / x.shape[0]
            scale = covariance.diagonal().mean()
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            covariance = covariance + normalized["covariance_damping"] * scale * torch.eye(delta.shape[1], device=delta.device, dtype=delta.dtype)
            _matrix(covariance, f"activation covariance[{key}]")
            covariance_factor = torch.linalg.cholesky(covariance)
            a, b, c = _svd(delta @ covariance_factor)
            weighted = (a[:, :rank] * b[:rank]) @ c[:rank]
            filtered = torch.linalg.solve_triangular(covariance_factor.T, weighted.T, upper=True).T
            selected = list(range(rank))
        elif method == "gradient":
            scores = _benefits(u, s, vh, gradient)
            indices = torch.argsort(scores, descending=True, stable=True)
            indices = indices[scores[indices] > 0][:rank]
            filtered = (u[:, indices] * s[indices]) @ vh[indices]
            selected = indices.detach().cpu().tolist()
        else:
            filtered = (u[:, :rank] * s[:rank]) @ vh[:rank]
            selected = list(range(rank))
        # Exact full-rank carry endpoints avoid a needless reconstruction error.
        if method in ("svd", "activation") and rank == min(delta.shape):
            filtered = delta
        strength = normalized["strength"]
        transfer = filtered * strength
        delta_norm = float(torch.linalg.vector_norm(delta, dtype=torch.float64).item())
        transfer_norm = float(torch.linalg.vector_norm(transfer, dtype=torch.float64).item())
        norm_scale = transfer_norm / delta_norm if delta_norm else 0.0
        if normalized["norm_matched"]:
            transfer = delta * norm_scale
        if strength == 0:
            result[key] = prior[key].detach().clone()
        elif method in ("svd", "activation") and rank == min(delta.shape) and strength == 1:
            result[key] = donor[key].detach().clone()
        else:
            result[key] = (base + transfer).to(prior[key].dtype)
        _finite_tensor(result[key], f"transferred[{key}]")
        # Report actual state displacement, including destination dtype rounding.
        actual = result[key].to(base.dtype) - base
        actual_square = float(actual.double().square().sum().item())
        current_square = float(delta.double().square().sum().item())
        donor_square += current_square
        transfer_square += actual_square
        selected_rank_sum += len(selected)
        matrix_rank_capacity_sum += min(delta.shape)
        if metrics is not None:
            if gradient is not None:
                benefit = -float((gradient.double() * actual.double()).sum().item())
                predicted_benefit += benefit
                if u is not None:
                    scores = scores if scores is not None else _benefits(u, s, vh, gradient)
                    positive_benefit_count += int((scores > 0).sum().item())
                    scored_component_count += len(scores)
                    positive_benefit_energy += float(s[scores > 0].double().square().sum().item())
                    scored_donor_energy += current_square
            if not include_layer_metrics:
                continue
            info = _energy_from_svd(u, s, vh, DEFAULT_ENERGY_RANKS, gradient, benefits=scores)
            info.update({
                "requested_rank": normalized["rank"], "selected_rank": len(selected),
                "selected_component_indices": selected if method != "activation" else None,
                "selected_rank_fraction": len(selected) / min(delta.shape),
                "norm_matching_scale": norm_scale,
                "transferred_delta_l2": math.sqrt(actual_square),
                "transferred_energy_fraction": actual_square / current_square if current_square else 0.0,
                "relative_parameter_residual": float(torch.linalg.vector_norm(delta - actual).item()) / delta_norm if delta_norm else 0.0,
            })
            if covariance_factor is not None:
                weighted_norm = float(torch.linalg.vector_norm(delta @ covariance_factor).item())
                weighted_residual = float(torch.linalg.vector_norm((delta - actual) @ covariance_factor).item())
                info["relative_activation_weighted_residual"] = weighted_residual / weighted_norm if weighted_norm else 0.0
            if gradient is not None:
                info["predicted_transfer_benefit"] = benefit
            details[key] = info
    if metrics is not None:
        metrics.update({
            "layers": details, "matrix_count": len(matrices),
            "selected_rank_sum": selected_rank_sum, "matrix_rank_capacity_sum": matrix_rank_capacity_sum,
            "selected_rank_fraction": selected_rank_sum / matrix_rank_capacity_sum if matrix_rank_capacity_sum else 0.0,
            "donor_delta_l2": math.sqrt(donor_square),
            "transferred_delta_l2": math.sqrt(transfer_square),
            "transferred_energy_fraction": transfer_square / donor_square if donor_square else 0.0,
            "relative_parameter_delta_l2": math.sqrt(transfer_square / prior_square) if prior_square else None,
            "predicted_benefit": predicted_benefit,
            "positive_benefit_count": positive_benefit_count if scored_component_count else None,
            "scored_component_count": scored_component_count if scored_component_count else None,
            "positive_benefit_fraction": positive_benefit_count / scored_component_count if scored_component_count else None,
            "positive_benefit_energy_fraction": positive_benefit_energy / scored_donor_energy if scored_donor_energy else (0.0 if scored_component_count else None),
        })
    return result
