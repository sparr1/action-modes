"""Frozen PAVE parameter-gradient diagnostics on real replay sequences.

This module measures gradient geometry; it never takes an optimizer step.
Candidate weights reuse one gradient basis, avoiding repeated higher-order
backward passes or changing the perturbations between comparisons.
"""

from contextlib import contextmanager
import math
import numbers
import random
import time

import numpy as np
import torch

from .pave import critic_pave_loss, prepare_pave_inputs


_TERMS = ("mpr", "vfc", "curvature")
_VARIANTS = {
    "reference": (.1, .1, .01),
    "stronger_curvature": (.1, .1, .1),
}
_CRITIC_ATTRS = {"primary": "_Qs", "auxiliary": "_aux_return_Qs"}


def _jsonable(value):
    """Strict JSON: unavailable/nonfinite diagnostics are null, never NaN."""
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _dot(left, right):
    if isinstance(left, torch.Tensor):
        return (left.double() * right.double()).sum()
    return sum((a.double() * b.double()).sum() for a, b in zip(left, right))


def _gram(vectors):
    """Gram matrix of flat tensors or aligned tuples of parameter gradients."""
    return torch.stack([
        torch.stack([_dot(left, right) for right in vectors]) for left in vectors
    ])


def _norm_from_squared(value):
    value = float(value)
    return math.sqrt(max(value, 0.0)) if math.isfinite(value) else float("nan")


def _candidate_statistics(gram, weights, td_norm_floor):
    """Evaluate a PAVE combination in weighted-TD/raw-PAVE Gram coordinates."""
    gram = torch.as_tensor(gram, dtype=torch.float64, device="cpu")
    if isinstance(weights, dict):
        weights = [weights[term] for term in _TERMS]
    weights = torch.as_tensor(weights, dtype=torch.float64, device="cpu")
    td_norm = _norm_from_squared(gram[0, 0])
    pave_squared = float(weights @ gram[1:, 1:] @ weights)
    dot = float(gram[0, 1:] @ weights)
    pave_norm = _norm_from_squared(pave_squared)
    reliable = math.isfinite(td_norm) and td_norm > td_norm_floor
    finite = math.isfinite(pave_norm) and math.isfinite(dot)
    sum_term_norms = sum(abs(float(weight)) * _norm_from_squared(gram[index + 1, index + 1])
                         for index, weight in enumerate(weights))
    return {
        "pave_gradient_norm": pave_norm,
        "ratio_to_td": pave_norm / td_norm if reliable and finite else None,
        "cosine_with_td": max(-1., min(1., dot / (td_norm * pave_norm)))
        if reliable and finite and pave_norm > 0. else None,
        "combined_critic_gradient_norm": _norm_from_squared(float(gram[0, 0]) + 2. * dot + pave_squared),
        "cancellation_ratio": pave_norm / sum_term_norms
        if finite and math.isfinite(sum_term_norms) and sum_term_norms > 0. else None,
    }


def _gradient(loss, parameters, *, retain_graph=False):
    active = [parameter for parameter in parameters if parameter.requires_grad]
    gradients = (torch.autograd.grad(loss, active, allow_unused=True, retain_graph=retain_graph)
                 if loss.requires_grad and active else [None] * len(active))
    by_id = {id(parameter): gradient for parameter, gradient in zip(active, gradients)}
    return tuple(by_id[id(parameter)].detach()
                 if by_id.get(id(parameter)) is not None else torch.zeros_like(parameter)
                 for parameter in parameters)


def _finite_fraction(gradients):
    count = sum(gradient.numel() for gradient in gradients)
    finite = sum(torch.isfinite(gradient).sum() for gradient in gradients)
    return float(finite) / count if count else 1.


def _clip_multiplier(norm, limit):
    return min(1., limit / (norm + 1e-6)) if math.isfinite(norm) else None


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def _frozen_probe(agent, seed):
    """Restore learned modes, RNG, and buffers even after failure.

    Compilation caches may warm, allowing callers to discard a cold probe and
    measure subsequent batches. Those operational caches are not learned state.
    """
    model = agent.model
    device = torch.device(agent.device)
    devices = ([device.index if device.index is not None else torch.cuda.current_device()]
               if device.type == "cuda" else [])
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    generators = []
    auxiliary = agent.aux_return
    if auxiliary is not None:
        generators = [(generator, generator.get_state()) for generator in
                      (auxiliary._cpu_generator, auxiliary._device_generator)
                      if generator is not None]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=devices):
        try:
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(seed + 1)
            if auxiliary is not None:
                auxiliary._cpu_generator.manual_seed(seed + 130363)
                if auxiliary._device_generator is not None:
                    auxiliary._device_generator.manual_seed(seed + 130364)
            with torch.enable_grad():
                yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            for generator, state in generators:
                generator.set_state(state)
            for module, training in modes:
                module.training = training
            with torch.no_grad():
                for buffer, value in buffers:
                    buffer.copy_(value)


def _validate(agent, obs, action, reward, terminated, critics, seed, update_number,
              batch_size, obs_noise_std, curvature_margin, scales, td_norm_floor):
    if torch.is_inference_mode_enabled():
        raise ValueError("PAVE gradient probes cannot run inside torch.inference_mode().")
    if agent.model.value_spec.is_split or getattr(agent.cfg, "obs", "state") != "state":
        raise ValueError("PAVE gradient probes require single-value state critics.")
    if float(getattr(agent.cfg, "value_equivalence_loss_coef", 0.)) != 0.:
        raise ValueError("PAVE gradient probes do not yet support active value_equivalence_loss_coef.")
    for name, value in (("seed", seed), ("update_number", update_number)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    if batch_size is not None and (isinstance(batch_size, bool)
                                  or not isinstance(batch_size, numbers.Integral) or batch_size <= 0):
        raise ValueError("batch_size must be a positive integer or None.")
    for name, value in (("obs_noise_std", obs_noise_std), ("curvature_margin", curvature_margin),
                        ("td_norm_floor", td_norm_floor)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value < 0.:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if isinstance(critics, str) or not critics or len(set(critics)) != len(critics) or any(
        name not in _CRITIC_ATTRS for name in critics
    ):
        raise ValueError("critics must contain unique primary/auxiliary selections.")
    selected = tuple(name for name in critics if hasattr(agent.model, _CRITIC_ATTRS[name]))
    if selected != tuple(critics):
        raise ValueError("A selected critic is unavailable in this checkpoint.")
    if not scales or any(isinstance(scale, bool) or not isinstance(scale, numbers.Real)
                         or not math.isfinite(scale) or scale < 0. for scale in scales):
        raise ValueError("scales must contain finite nonnegative numbers.")
    if any(not isinstance(value, torch.Tensor) for value in (obs, action, reward, terminated)):
        raise TypeError("PAVE probe inputs must be tensors.")
    horizon, batch = int(agent.cfg.train_unroll_horizon), int(agent.cfg.batch_size)
    if obs.ndim != 3 or obs.shape[:2] != (horizon + 1, batch):
        raise ValueError("Observation horizon/batch must match the configured training sequence.")
    if tuple(obs.shape[2:]) != tuple(agent.cfg.obs_shape["state"]):
        raise ValueError("Observation dimension must match the checkpoint state encoder.")
    if action.shape != (horizon, batch, int(agent.cfg.action_dim)):
        raise ValueError("Action shape must match the configured training horizon, batch, and action dimension.")
    if reward.shape != (horizon, batch, 1) or terminated.shape != reward.shape:
        raise ValueError("Rewards and terminated flags must have shape [H,B,1].")
    for name, value in (("obs", obs), ("action", action), ("reward", reward), ("terminated", terminated)):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite values.")
    if not obs.is_floating_point() or not action.is_floating_point() or not reward.is_floating_point():
        raise ValueError("Observations, actions, and rewards must be floating-point tensors.")
    if (action.abs() > 1.).any():
        raise ValueError("PAVE probe actions must use the normalized [-1,1] coordinates.")
    if not ((terminated == 0) | (terminated == 1)).all():
        raise ValueError("terminated must contain only zero or one.")
    return selected


def probe_batch(agent, obs, action, reward, terminated, *, seed, update_number,
                critics=None, batch_size=64, obs_noise_std=.01,
                curvature_margin=1., scales=(.01, .03, .1, .3, 1., 3., 10.),
                td_norm_floor=1e-10):
    """Return strict-JSONable diagnostics without changing learned agent state.

    ``gradient_gram`` uses [weighted TD, raw MPR, raw VFC, raw curvature].
    Candidate ``weights`` are the actual coefficients after multiplying by k.
    Ratios/cosines are null when their TD denominator is unreliable. Existing
    PAVE settings are replaced by the candidate grid, not added a second time.
    With ``critics=None``, all available critics are selected; explicit missing
    critics are rejected. Timing measures diagnostic forwards and differentiation, not a
    complete training update, and can include cold compilation. CUDA peak
    statistics describe this diagnostic process, not production training.
    """
    if critics is None:
        critics = tuple(name for name, attribute in _CRITIC_ATTRS.items() if hasattr(agent.model, attribute))
    selected = _validate(agent, obs, action, reward, terminated, critics, seed, update_number,
                         batch_size, obs_noise_std, curvature_margin, scales, td_norm_floor)
    device = torch.device(agent.device)
    parameter_dtype = next(agent.model.parameters()).dtype
    obs, action, reward, terminated = (
        value.detach().to(device=device, dtype=parameter_dtype)
        for value in (obs, action, reward, terminated)
    )
    parameters = tuple(agent._world_critic_params)
    parameter_indices = {id(parameter): index for index, parameter in enumerate(parameters)}
    critic_parameters = {name: tuple(getattr(agent.model, _CRITIC_ATTRS[name]).parameters())
                         for name in selected}
    peak_memory = {} if device.type == "cuda" else None
    with _frozen_probe(agent, int(seed)):
        _synchronize(device)
        if peak_memory is not None:
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        agent.model.eval()
        with torch.no_grad():
            next_z = agent.model.encode(obs[1:])
            td_target = agent._soft_td_target(next_z, reward, terminated)
            auxiliary_target = (agent.aux_return.td_target(next_z, reward, terminated)
                                if agent.aux_return is not None else None)
        agent.model.train()
        output = agent._outer_update_region(obs[0], action, reward, terminated, next_z, td_target)
        total = output[-1]
        losses = {"primary_td": output[7].detach()}
        if agent.aux_return is not None:
            auxiliary_loss, _ = agent.aux_return.critic_loss(output[0], action, auxiliary_target)
            total = total + float(agent.cfg.aux_return_critic_coef) * auxiliary_loss
            losses["auxiliary_td"] = auxiliary_loss.detach()
        losses["total"] = total.detach()
        baseline_gradient = _gradient(total, parameters)
        # With VE explicitly rejected, each critic's baseline block is exactly
        # its weighted TD gradient. Shared encoder/dynamics gradients remain in
        # the full vector used below for the actual global clipping calculation.
        td_gradients = {name: tuple(baseline_gradient[parameter_indices[id(parameter)]]
                                   for parameter in owned)
                        for name, owned in critic_parameters.items()}
        baseline_squared = float(_dot(baseline_gradient, baseline_gradient))
        baseline_norm = _norm_from_squared(baseline_squared)
        clip_limit = float(agent.cfg.grad_clip_norm)
        baseline_clip = _clip_multiplier(baseline_norm, clip_limit)
        baseline = dict(losses=losses, full_gradient_norm=baseline_norm,
                        clip_multiplier=baseline_clip,
                        finite_gradient_fraction=_finite_fraction(baseline_gradient))
        _synchronize(device)
        baseline_seconds = time.perf_counter() - start
        if peak_memory is not None:
            peak_memory["baseline_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        del output, total, td_target, auxiliary_target, next_z
        if agent.aux_return is not None:
            del auxiliary_loss
        _synchronize(device)
        if peak_memory is not None:
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        inputs = prepare_pave_inputs(
            agent.model, obs, action, seed=agent.cfg.seed, update_number=update_number,
            batch_size=batch_size, obs_noise_std=obs_noise_std,
        )
        critic_results, grams = {}, {}
        for name, owned in critic_parameters.items():
            _, diagnostics, components = critic_pave_loss(
                getattr(agent.model, _CRITIC_ATTRS[name]), agent.model.q_backend.decode,
                inputs["z"], inputs["action"], noisy_z=inputs["noisy_z"],
                next_z=inputs["next_z"], probe=inputs["probe"],
                mpr_coef=1., vfc_coef=1., curvature_coef=1.,
                curvature_margin=curvature_margin, return_components=True,
            )
            gradients = [_gradient(components[term], owned, retain_graph=index < len(_TERMS) - 1)
                         for index, term in enumerate(_TERMS)]
            vectors = [td_gradients[name], *gradients]
            gram = _gram(vectors).cpu()
            grams[name] = gram
            td_norm = _norm_from_squared(gram[0, 0])
            critic_results[name] = {
                "td_coefficient": float(agent.cfg.critic_coef if name == "primary"
                                        else agent.cfg.aux_return_critic_coef),
                "td_gradient_norm": td_norm,
                "td_norm_reliable": math.isfinite(td_norm) and td_norm > td_norm_floor,
                "raw_losses": {term: components[term].detach() for term in _TERMS},
                "weighted_term_gradient_norms": {
                    term: weight * _norm_from_squared(gram[index + 1, index + 1])
                    for index, (term, weight) in enumerate(zip(_TERMS, _VARIANTS["reference"]))
                },
                "gradient_gram": gram,
                "finite_gradient_fractions": {term: _finite_fraction(vector)
                                              for term, vector in zip(("td", *_TERMS), vectors)},
                "action_gradient_rms": diagnostics["action_gradient_rms"],
                "curvature_violation_fraction": diagnostics["curvature_violation_fraction"],
            }
            del components, gradients, vectors
        _synchronize(device)
        pave_seconds = time.perf_counter() - start
        if peak_memory is not None:
            peak_memory["pave_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)

        scopes = [(name, (name,)) for name in selected]
        if len(selected) > 1:
            scopes.append(("combined", selected))
        candidates = []
        for variant, base_weights in _VARIANTS.items():
            for scale in scales:
                weights = torch.tensor(base_weights, dtype=torch.float64) * float(scale)
                for scope, names in scopes:
                    squared = baseline_squared
                    for name in names:
                        gram = grams[name]
                        # Critic parameter sets are disjoint, so there is no
                        # primary/auxiliary PAVE cross term. TD/PAVE cross
                        # terms within each critic are retained explicitly.
                        squared += float(2. * gram[0, 1:] @ weights
                                         + weights @ gram[1:, 1:] @ weights)
                    norm = _norm_from_squared(squared)
                    clip = _clip_multiplier(norm, clip_limit)
                    candidates.append({
                        "variant": variant, "scope": scope, "scale": float(scale),
                        "weights": dict(zip(_TERMS, weights.tolist())),
                        "critics": {name: _candidate_statistics(grams[name], weights, td_norm_floor)
                                    for name in names},
                        "full_gradient_norm": norm, "clip_multiplier": clip,
                        "clip_multiplier_relative_to_baseline": clip / baseline_clip
                        if clip is not None and baseline_clip is not None and baseline_clip > 0. else None,
                    })
        report = {
            "schema": "ambi-pave-gradient-probe-batch", "version": 1,
            "seed": int(seed), "update_number": int(update_number),
            "sample_count": inputs["action"].shape[0],
            "settings": {
                "critics": selected, "batch_size": int(batch_size) if batch_size is not None else None,
                "obs_noise_std": float(obs_noise_std), "curvature_margin": float(curvature_margin),
                "td_norm_floor": float(td_norm_floor),
                "gram_order": ["td", *_TERMS], "gram_basis": "weighted_td_and_raw_pave",
                "td_seed": int(seed), "pave_training_seed": int(agent.cfg.seed),
                "train_unroll_horizon": int(agent.cfg.train_unroll_horizon),
                "td_batch_size": int(agent.cfg.batch_size), "action_dim": int(agent.cfg.action_dim),
                "num_q": int(agent.cfg.num_q), "grad_clip_norm": clip_limit,
                "compile": bool(getattr(agent.cfg, "compile", False)),
            },
            "baseline": baseline, "critics": critic_results, "candidates": candidates,
            "timing": {"baseline_seconds": baseline_seconds, "pave_seconds": pave_seconds,
                       "kind": "diagnostic_forward_and_gradients"},
            "cuda_memory": peak_memory,
        }
    return _jsonable(report)
