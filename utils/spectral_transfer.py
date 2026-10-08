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
Gradient projection instead retains the literal signed projection of the full
matrix-parameter delta onto one gradient direction, including uphill transfer.
Gradient gating keeps only coordinates whose donor change strictly opposes the
new-decision gradient. Its dense norm control matches each layer separately.
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
    if method not in ("svd", "activation", "gradient", "gradient_projection", "gradient_gate"):
        raise ValueError("spectral method must be svd, activation, gradient, gradient_projection or gradient_gate")
    if method in ("gradient_projection", "gradient_gate"):
        if rank is not None:
            raise ValueError(f"{method} rank must be absent or None; this method does not select a matrix rank")
    elif isinstance(rank, bool) or not isinstance(rank, Integral) or rank < 1:
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
    return {"method": method, "rank": int(rank) if rank is not None else None, "strength": float(strength),
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


def _finite_sum(values, name):
    """Accumulate matrix reductions without fp32 overflow or silent NaNs."""
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} is not representable as a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} is not representable as a finite real number")
    return result


def _check_finite_metrics(value):
    if isinstance(value, Mapping):
        for item in value.values():
            _check_finite_metrics(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_finite_metrics(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("gradient projection diagnostics are not finite")


def _gradient_projection_state(prior, donor, matrices, spec, gradients, metrics,
                               include_layer_metrics):
    """Project in the concatenated parameter space of one actor or critic."""
    result = {key: value.detach().clone() for key, value in prior.items()}
    # Form deltas and all reductions in double even for fp16/fp32 parameters.
    # This also avoids overflow in ordinary fp32 squared gradient norms.
    bases = {key: prior[key].detach().double() for key in matrices}
    deltas = {key: donor[key].detach().double() - bases[key] for key in matrices}
    for key, delta in deltas.items():
        _matrix(delta, f"delta[{key}]")
    donor_square = _finite_sum((float(value.square().sum().item()) for value in deltas.values()), "donor squared norm")
    prior_square = _finite_sum((float(value.square().sum().item()) for value in bases.values()), "prior squared norm")
    payload = {
        "layers": {}, "matrix_count": len(matrices),
        "selected_rank_sum": None,
        "matrix_rank_capacity_sum": sum(min(prior[key].shape) for key in matrices),
        "selected_rank_fraction": None, "donor_delta_l2": math.sqrt(donor_square),
        "transferred_delta_l2": 0.0, "transferred_energy_fraction": 0.0,
        "relative_parameter_delta_l2": 0.0 if prior_square else None,
        "predicted_benefit": 0.0, "positive_benefit_count": None,
        "scored_component_count": None, "positive_benefit_fraction": None,
        "positive_benefit_energy_fraction": None,
        "projection_coefficient": None, "donor_gradient_inner_product": None,
        "gradient_squared_norm": None, "norm_matching_scale": 0.0,
        "norm_matching_scope": "component-global",
    }
    if spec["strength"] == 0:
        if metrics is not None:
            metrics.update(payload)
        return result
    g = {key: gradients[key].detach().double() for key in matrices}
    inner = _finite_sum((float((deltas[key] * g[key]).sum().item()) for key in matrices), "donor-gradient inner product")
    gradient_square = _finite_sum((float(value.square().sum().item()) for value in g.values()), "gradient squared norm")
    # A numerically underflowed nonzero gradient must not silently mean G=0.
    if gradient_square == 0 and any(bool(value.ne(0).any()) for value in g.values()):
        raise ValueError("gradient squared norm underflows float64")
    coefficient = inner / gradient_square if gradient_square else 0.0
    if not math.isfinite(coefficient):
        raise ValueError("gradient projection coefficient is not finite")
    filtered = {key: value * coefficient * spec["strength"] for key, value in g.items()}
    for key, value in filtered.items():
        _matrix(value, f"projection[{key}]")
    projected_square = _finite_sum((float(value.square().sum().item()) for value in filtered.values()), "projected squared norm")
    # Projection can change a layer with zero donor delta. Only one global
    # dense scale can provide a valid matched-norm control in that case.
    norm_scale = math.sqrt(projected_square) / math.sqrt(donor_square) if donor_square else 0.0
    actuals, actual_squares, benefits = {}, {}, {}
    for key in matrices:
        transfer = deltas[key] * norm_scale if spec["norm_matched"] else filtered[key]
        result[key] = (bases[key] + transfer).to(prior[key].dtype)
        _finite_tensor(result[key], f"transferred[{key}]")
        actuals[key] = result[key].double() - bases[key]
        actual_squares[key] = float(actuals[key].square().sum().item())
        benefits[key] = -float((g[key] * actuals[key]).sum().item())
    transfer_square = _finite_sum(actual_squares.values(), "transferred squared norm")
    benefit = _finite_sum(benefits.values(), "predicted benefit")
    payload.update({
        "projection_coefficient": coefficient,
        "donor_gradient_inner_product": inner, "gradient_squared_norm": gradient_square,
        "norm_matching_scale": norm_scale, "predicted_benefit": benefit,
        "transferred_delta_l2": math.sqrt(transfer_square),
        "transferred_energy_fraction": transfer_square / donor_square if donor_square else 0.0,
        "relative_parameter_delta_l2": math.sqrt(transfer_square) / math.sqrt(prior_square) if prior_square else None,
    })
    if metrics is not None and include_layer_metrics:
        positive_count = scored_count = 0
        positive_energies = []
        for key in matrices:
            delta, actual = deltas[key], actuals[key]
            u, s, vh = _svd(delta)
            scores = _benefits(u, s, vh, g[key])
            info = _energy_from_svd(u, s, vh, DEFAULT_ENERGY_RANKS, g[key], benefits=scores)
            current_square = float(delta.square().sum().item())
            delta_norm = math.sqrt(current_square)
            info.update({
                "requested_rank": None, "selected_rank": None,
                "selected_component_indices": None, "selected_rank_fraction": None,
                "projection_coefficient": coefficient,
                "norm_matching_scale": norm_scale, "norm_matching_scope": "component-global",
                "transferred_delta_l2": math.sqrt(actual_squares[key]),
                # A globally projected direction can enter a donor-zero layer.
                "transferred_energy_fraction": actual_squares[key] / current_square if current_square else (None if actual_squares[key] else 0.0),
                "relative_parameter_residual": float(torch.linalg.vector_norm(delta - actual).item()) / delta_norm if delta_norm else (None if actual_squares[key] else 0.0),
                "predicted_transfer_benefit": benefits[key],
            })
            payload["layers"][key] = info
            positive_count += int((scores > 0).sum().item())
            scored_count += len(scores)
            positive_energies.append(float(s[scores > 0].square().sum().item()))
        payload.update({
            "positive_benefit_count": positive_count if scored_count else None,
            "scored_component_count": scored_count or None,
            "positive_benefit_fraction": positive_count / scored_count if scored_count else None,
            "positive_benefit_energy_fraction": _finite_sum(positive_energies, "positive-benefit energy") / donor_square if donor_square else (0.0 if scored_count else None),
        })
    _check_finite_metrics(payload)
    if metrics is not None:
        metrics.update(payload)
    return result


def _gradient_gate_state(prior, donor, matrices, spec, gradients, metrics,
                         include_layer_metrics):
    """Keep donor coordinates with strictly beneficial first-order changes."""
    result = {key: value.detach().clone() for key, value in prior.items()}
    bases = {key: prior[key].detach().double() for key in matrices}
    deltas = {key: donor[key].detach().double() - bases[key] for key in matrices}
    for key, delta in deltas.items():
        _matrix(delta, f"delta[{key}]")
    donor_square = _finite_sum((float(value.square().sum().item()) for value in deltas.values()), "donor squared norm")
    prior_square = _finite_sum((float(value.square().sum().item()) for value in bases.values()), "prior squared norm")
    parameter_count = sum(value.numel() for value in deltas.values())
    payload = {
        "layers": {}, "matrix_count": len(matrices), "selected_rank_sum": None,
        "matrix_rank_capacity_sum": sum(min(prior[key].shape) for key in matrices),
        "selected_rank_fraction": None, "donor_delta_l2": math.sqrt(donor_square),
        "transferred_delta_l2": 0.0, "transferred_energy_fraction": 0.0,
        "relative_parameter_delta_l2": 0.0 if prior_square else None,
        "predicted_benefit": 0.0, "positive_benefit_count": None,
        "scored_component_count": None, "positive_benefit_fraction": None,
        "positive_benefit_energy_fraction": None,
        "retained_fraction": None, "retained_parameter_count": None,
        "parameter_count": parameter_count, "donor_gradient_inner_product": None,
        "gradient_squared_norm": None, "norm_matching_scope": "per-layer",
    }
    if spec["strength"] == 0:
        if metrics is not None:
            metrics.update(payload)
        return result
    g = {key: gradients[key].detach().double() for key in matrices}
    gradient_square = _finite_sum((float(value.square().sum().item()) for value in g.values()), "gradient squared norm")
    actual_squares, benefits, donor_inners = [], [], []
    retained_count = positive_count = scored_count = 0
    positive_energies = []
    for key in matrices:
        base, delta, gradient = bases[key], deltas[key], g[key]
        # The sign comparison is exactly the real-valued G_j * D_j < 0 rule,
        # without numerical underflow turning a tiny negative product into -0.
        keep = ((gradient < 0) & (delta > 0)) | ((gradient > 0) & (delta < 0))
        selected = int(keep.sum().item())
        retained_count += selected
        filtered = torch.where(keep, delta, torch.zeros_like(delta)) * spec["strength"]
        current_square = float(delta.square().sum().item())
        filtered_square = float(filtered.square().sum().item())
        norm_scale = math.sqrt(filtered_square) / math.sqrt(current_square) if current_square else 0.0
        transfer = delta * norm_scale if spec["norm_matched"] else filtered
        if spec["strength"] == 1 and not spec["norm_matched"]:
            result[key] = torch.where(keep, donor[key].detach(), prior[key].detach())
        else:
            result[key] = (base + transfer).to(prior[key].dtype)
        _finite_tensor(result[key], f"transferred[{key}]")
        actual = result[key].double() - base
        actual_square = float(actual.square().sum().item())
        benefit = -float((gradient * actual).sum().item())
        donor_inner = float((gradient * delta).sum().item())
        actual_squares.append(actual_square)
        benefits.append(benefit)
        donor_inners.append(donor_inner)
        if metrics is not None and include_layer_metrics:
            # Full donor spectra belong only to sampled diagnostics. The
            # every-decision gate above has no decomposition or random draws.
            u, s, vh = _svd(delta)
            scores = _benefits(u, s, vh, gradient)
            info = _energy_from_svd(u, s, vh, DEFAULT_ENERGY_RANKS, gradient, benefits=scores)
            info.update({
                "requested_rank": None, "selected_rank": None,
                "selected_component_indices": None, "selected_rank_fraction": None,
                "retained_fraction": selected / delta.numel(),
                "retained_parameter_count": selected, "parameter_count": delta.numel(),
                "donor_gradient_inner_product": donor_inner,
                "norm_matching_scale": norm_scale, "norm_matching_scope": "per-layer",
                "transferred_delta_l2": math.sqrt(actual_square),
                "transferred_energy_fraction": actual_square / current_square if current_square else 0.0,
                "relative_parameter_residual": float(torch.linalg.vector_norm(delta - actual).item()) / math.sqrt(current_square) if current_square else 0.0,
                "predicted_transfer_benefit": benefit,
            })
            payload["layers"][key] = info
            positive_count += int((scores > 0).sum().item())
            scored_count += len(scores)
            positive_energies.append(float(s[scores > 0].square().sum().item()))
    transfer_square = _finite_sum(actual_squares, "transferred squared norm")
    payload.update({
        "retained_fraction": retained_count / parameter_count if parameter_count else 0.0,
        "retained_parameter_count": retained_count,
        "donor_gradient_inner_product": _finite_sum(donor_inners, "donor-gradient inner product"),
        "gradient_squared_norm": gradient_square,
        "predicted_benefit": _finite_sum(benefits, "predicted benefit"),
        "transferred_delta_l2": math.sqrt(transfer_square),
        "transferred_energy_fraction": transfer_square / donor_square if donor_square else 0.0,
        "relative_parameter_delta_l2": math.sqrt(transfer_square) / math.sqrt(prior_square) if prior_square else None,
        "positive_benefit_count": positive_count if scored_count else None,
        "scored_component_count": scored_count or None,
        "positive_benefit_fraction": positive_count / scored_count if scored_count else None,
        "positive_benefit_energy_fraction": (_finite_sum(positive_energies, "positive-benefit energy") / donor_square if donor_square else 0.0) if scored_count else None,
    })
    _check_finite_metrics(payload)
    if metrics is not None:
        metrics.update(payload)
    return result


@torch.no_grad()
def spectral_state(prior, donor, *, parameter_names, spec, inputs=None, gradients=None, metrics=None,
                   include_layer_metrics=True):
    """Return cloned prior state with selected matrix deltas added.

    ``inputs`` maps each matrix key to an N-by-input-width activation tensor;
    ``gradients`` maps each key to a gradient matrix. Both must match parameter
    device and dtype. Norm matching is per layer for SVD methods: it replaces
    filtered transfer by a scalar multiple of the dense donor delta with the
    same Frobenius norm.
    Norm matching may exceed unit dense strength for activation-weighted SVD.
    Gradient projection uses a single signed coefficient across all matrix
    parameters, and its dense control matches their combined norm. Its rank
    is None: a parameter-space direction need not be a rank-one matrix.
    Gradient gating instead keeps each coordinate only when its donor delta
    strictly opposes its gradient; its rank is None and norm matching is per
    layer. Its retained fraction counts selected matrix coordinates, including
    zero-delta coordinates in the denominator, before dense norm matching.
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
    gradients = _validate_auxiliary(gradients, "gradients", prior, matrices,
                                    required=nonzero and method in ("gradient", "gradient_projection", "gradient_gate"))
    if method == "gradient_projection":
        return _gradient_projection_state(prior, donor, matrices, normalized, gradients,
                                          metrics, include_layer_metrics)
    if method == "gradient_gate":
        return _gradient_gate_state(prior, donor, matrices, normalized, gradients,
                                    metrics, include_layer_metrics)
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
