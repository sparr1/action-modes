"""Stateless parameter-noise primitives for TD-MPC2-style policy actors.

The current clone actor is a sequential MLP whose hidden ``nn.Linear``
modules own LayerNorm submodules and whose final linear head emits
``[pre_tanh_mean, log_std]``.  This module deliberately recognizes that
contract instead of perturbing every trainable tensor indiscriminately:

* hidden linear weights and biases receive independent Gaussian noise;
* LayerNorm affine parameters and every buffer remain exact;
* only the mean rows of the final joint policy head receive noise.

All perturbed parameters are functional tensors.  The source actor is never
modified, which lets a population of policies remain fixed for an imagined
rollout without allocating or mutating ``nn.Module`` copies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch
import torch.nn as nn
from torch.func import functional_call

from .layers import NormedLinear


@dataclass(frozen=True)
class ParameterNoiseSpec:
    """Validated structural description of one clone-policy actor."""

    action_dim: int
    input_dim: int
    parameter_names: tuple[str, ...]
    parameter_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    buffer_names: tuple[str, ...]
    buffer_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    hidden_linear_parameter_names: tuple[str, ...]
    layer_norm_parameter_names: tuple[str, ...]
    perturbable_names: tuple[str, ...]
    final_weight_name: str
    final_bias_name: str | None


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _module_parameter_name(module_name, leaf_name):
    return f"{module_name}.{leaf_name}" if module_name else leaf_name


def classify_parameter_noise_actor(
    actor: nn.Module, action_dim: int
) -> ParameterNoiseSpec:
    """Classify the supported clone actor and return an immutable noise spec.

    The check is intentionally strict.  A new actor architecture must be
    reviewed before parameter noise can silently act on newly introduced
    trainable tensors.
    """

    action_dim = _positive_integer(action_dim, "action_dim")
    if not isinstance(actor, nn.Sequential):
        raise TypeError(
            "Parameter noise currently supports the clone actor's nn.Sequential "
            f"MLP, got {type(actor).__name__}."
        )

    children = tuple(actor.named_children())
    if len(children) < 2:
        raise ValueError(
            "The parameter-noise actor must contain at least one hidden linear "
            "layer and one final linear head."
        )
    unsupported_hidden = [
        name
        for name, module in children[:-1]
        if type(module) is not NormedLinear
    ]
    if unsupported_hidden:
        raise ValueError(
            "Every hidden clone-actor stage must be exactly NormedLinear; "
            f"unsupported stages: {unsupported_hidden}."
        )
    stochastic_hidden = [
        name for name, module in children[:-1] if module.dropout is not None
    ]
    if stochastic_hidden:
        raise ValueError(
            "Parameter-noise actors must not contain hidden dropout; "
            f"stochastic stages: {stochastic_hidden}."
        )

    final_module_name, final_module = children[-1]
    if type(final_module) is not nn.Linear:
        raise ValueError(
            "The final clone-actor stage must be exactly nn.Linear, got "
            f"{type(final_module).__name__}."
        )
    if final_module.out_features != 2 * action_dim:
        raise ValueError(
            "The final clone-actor head must emit concatenated mean/log-std rows: "
            f"out_features={final_module.out_features}, expected {2 * action_dim}."
        )

    parameters = dict(actor.named_parameters())
    if not parameters:
        raise ValueError("The parameter-noise actor has no parameters.")
    devices = {parameter.device for parameter in parameters.values()}
    if len(devices) != 1:
        raise ValueError(
            "All clone-actor parameters must be on one device before sampling "
            f"parameter noise, got {sorted(map(str, devices))}."
        )
    dtypes = {parameter.dtype for parameter in parameters.values()}
    if len(dtypes) != 1:
        raise ValueError(
            "All clone-actor parameters must have one dtype before sampling "
            f"parameter noise, got {sorted(map(str, dtypes))}."
        )
    non_floating = [
        name
        for name, parameter in parameters.items()
        if not parameter.is_floating_point()
    ]
    if non_floating:
        raise TypeError(
            "Parameter noise requires floating-point actor parameters; "
            f"non-floating tensors: {non_floating}."
        )

    linear_parameter_names = []
    hidden_linear_parameter_names = []
    for index, (module_name, module) in enumerate(children):
        weight_name = _module_parameter_name(module_name, "weight")
        linear_parameter_names.append(weight_name)
        if index != len(children) - 1:
            hidden_linear_parameter_names.append(weight_name)
        if module.bias is not None:
            bias_name = _module_parameter_name(module_name, "bias")
            linear_parameter_names.append(bias_name)
            if index != len(children) - 1:
                hidden_linear_parameter_names.append(bias_name)

    layer_norm_parameter_names = []
    for module_name, module in actor.named_modules():
        if not isinstance(module, nn.LayerNorm):
            continue
        for leaf_name, _ in module.named_parameters(recurse=False):
            layer_norm_parameter_names.append(
                _module_parameter_name(module_name, leaf_name)
            )

    classified = set(linear_parameter_names) | set(layer_norm_parameter_names)
    actual = set(parameters)
    if classified != actual:
        missing = sorted(actual - classified)
        unknown = sorted(classified - actual)
        raise ValueError(
            "Unsupported clone-actor parameter layout; every parameter must belong "
            "to a top-level Linear or a LayerNorm. "
            f"unclassified={missing}, missing_from_actor={unknown}."
        )

    final_weight_name = _module_parameter_name(final_module_name, "weight")
    final_bias_name = (
        _module_parameter_name(final_module_name, "bias")
        if final_module.bias is not None
        else None
    )
    perturbable_set = set(hidden_linear_parameter_names) | {final_weight_name}
    if final_bias_name is not None:
        perturbable_set.add(final_bias_name)
    parameter_names = tuple(parameters)
    perturbable_names = tuple(
        name for name in parameter_names if name in perturbable_set
    )

    buffers = dict(actor.named_buffers())
    buffer_devices = {
        name: buffer.device
        for name, buffer in buffers.items()
        if buffer.device not in devices
    }
    if buffer_devices:
        raise ValueError(
            "Every clone-actor buffer must share the parameter device; "
            f"mismatches={buffer_devices}."
        )
    return ParameterNoiseSpec(
        action_dim=action_dim,
        input_dim=int(children[0][1].in_features),
        parameter_names=parameter_names,
        parameter_shapes=tuple(
            (name, tuple(parameter.shape)) for name, parameter in parameters.items()
        ),
        buffer_names=tuple(buffers),
        buffer_shapes=tuple(
            (name, tuple(buffer.shape)) for name, buffer in buffers.items()
        ),
        hidden_linear_parameter_names=tuple(hidden_linear_parameter_names),
        layer_norm_parameter_names=tuple(layer_norm_parameter_names),
        perturbable_names=perturbable_names,
        final_weight_name=final_weight_name,
        final_bias_name=final_bias_name,
    )


def _actor_state(actor, spec):
    if not isinstance(spec, ParameterNoiseSpec):
        raise TypeError(
            f"spec must be a ParameterNoiseSpec, got {type(spec).__name__}."
        )
    parameters = dict(actor.named_parameters())
    parameter_shapes = tuple(
        (name, tuple(parameter.shape)) for name, parameter in parameters.items()
    )
    buffers = dict(actor.named_buffers())
    buffer_shapes = tuple(
        (name, tuple(buffer.shape)) for name, buffer in buffers.items()
    )
    if (
        tuple(parameters) != spec.parameter_names
        or parameter_shapes != spec.parameter_shapes
    ):
        raise ValueError("The actor parameter layout no longer matches its noise spec.")
    if tuple(buffers) != spec.buffer_names or buffer_shapes != spec.buffer_shapes:
        raise ValueError("The actor buffer layout no longer matches its noise spec.")
    return parameters, buffers


@torch.no_grad()
def sample_parameter_deltas(
    actor: nn.Module,
    spec: ParameterNoiseSpec,
    population_size: int,
    *,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Sample independent standard-normal deltas for all allowed parameters.

    Returned tensors have a leading population dimension.  Final-head tensors
    retain the full actor-parameter shape, but their log-standard-deviation rows
    are exactly zero.
    """

    population_size = _positive_integer(population_size, "population_size")
    if not isinstance(generator, torch.Generator):
        raise TypeError(
            f"generator must be a torch.Generator, got {type(generator).__name__}."
        )
    parameters, _ = _actor_state(actor, spec)
    parameter_device = next(iter(parameters.values())).device
    generator_device = torch.device(generator.device)
    same_device_type = generator_device.type == parameter_device.type
    same_explicit_index = (
        generator_device.index is None
        or parameter_device.index is None
        or generator_device.index == parameter_device.index
    )
    if not same_device_type or not same_explicit_index:
        raise ValueError(
            "The private generator must live on the actor parameter device: "
            f"generator={generator_device}, actor={parameter_device}."
        )

    deltas = {}
    final_names = {spec.final_weight_name, spec.final_bias_name}
    for name in spec.perturbable_names:
        parameter = parameters[name]
        shape = (population_size, *parameter.shape)
        if name in final_names:
            delta = torch.zeros(shape, dtype=parameter.dtype, device=parameter.device)
            mean_shape = (population_size, spec.action_dim, *parameter.shape[1:])
            delta[:, : spec.action_dim] = torch.randn(
                mean_shape,
                dtype=parameter.dtype,
                device=parameter.device,
                generator=generator,
            )
        else:
            delta = torch.randn(
                shape,
                dtype=parameter.dtype,
                device=parameter.device,
                generator=generator,
            )
        deltas[name] = delta
    return deltas


def _validated_deltas(actor, spec, deltas):
    if not isinstance(deltas, Mapping):
        raise TypeError(f"deltas must be a mapping, got {type(deltas).__name__}.")
    parameters, _ = _actor_state(actor, spec)
    if set(deltas) != set(spec.perturbable_names):
        missing = sorted(set(spec.perturbable_names) - set(deltas))
        extra = sorted(set(deltas) - set(spec.perturbable_names))
        raise ValueError(
            f"Delta keys do not match the noise spec: missing={missing}, extra={extra}."
        )

    population_size = None
    final_names = {spec.final_weight_name, spec.final_bias_name}
    for name in spec.perturbable_names:
        delta = deltas[name]
        if not isinstance(delta, torch.Tensor):
            raise TypeError(f"Delta {name!r} must be a tensor.")
        parameter = parameters[name]
        if delta.ndim != parameter.ndim + 1:
            raise ValueError(
                f"Delta {name!r} must have one population dimension plus "
                f"parameter shape {tuple(parameter.shape)}, got {tuple(delta.shape)}."
            )
        if population_size is None:
            population_size = int(delta.shape[0])
            _positive_integer(population_size, "delta population size")
        expected_shape = (population_size, *parameter.shape)
        if tuple(delta.shape) != expected_shape:
            raise ValueError(
                f"Delta {name!r} has shape {tuple(delta.shape)}, expected "
                f"{expected_shape}."
            )
        if delta.device != parameter.device or delta.dtype != parameter.dtype:
            raise ValueError(
                f"Delta {name!r} must match its parameter device/dtype, got "
                f"{delta.device}/{delta.dtype} != {parameter.device}/{parameter.dtype}."
            )
        if not bool(torch.isfinite(delta).all().item()):
            raise ValueError(f"Delta {name!r} must contain only finite values.")
        if name in final_names and torch.count_nonzero(
            delta[:, spec.action_dim :]
        ).item():
            raise ValueError(
                f"Final-head delta {name!r} must be zero on all log-std rows."
            )
    return parameters, population_size


@torch.no_grad()
def make_perturbed_actor_parameters(
    actor: nn.Module,
    spec: ParameterNoiseSpec,
    deltas: Mapping[str, torch.Tensor],
    stddev: float,
) -> dict[str, torch.Tensor]:
    """Create a full batched functional parameter mapping without mutation."""

    if isinstance(stddev, bool) or not isinstance(stddev, Real):
        raise TypeError(f"stddev must be a real scalar, got {stddev!r}.")
    stddev = float(stddev)
    if not math.isfinite(stddev) or stddev < 0.0:
        raise ValueError(f"stddev must be finite and non-negative, got {stddev}.")
    parameters, population_size = _validated_deltas(actor, spec, deltas)
    final_names = {spec.final_weight_name, spec.final_bias_name}
    result = {}
    for name in spec.parameter_names:
        parameter = parameters[name].detach()
        expanded = parameter.unsqueeze(0).expand(population_size, *parameter.shape)
        if name not in deltas:
            # The functional population must remain fixed even if the clean
            # actor changes before evaluation, and callers must not be able to
            # mutate the actor through an expanded storage alias.
            result[name] = expanded.clone()
        elif name in final_names:
            mean = parameter[: spec.action_dim].unsqueeze(0) + (
                deltas[name][:, : spec.action_dim].detach() * stddev
            )
            log_std = parameter[spec.action_dim :].unsqueeze(0).expand(
                population_size, *parameter[spec.action_dim :].shape
            )
            result[name] = torch.cat((mean, log_std), dim=1)
        else:
            result[name] = expanded + deltas[name].detach() * stddev
    return result


def _base_functional_state(actor):
    parameters = {
        name: parameter.detach() for name, parameter in actor.named_parameters()
    }
    # Defensive clones ensure a stateful buffer update in a future actor cannot
    # mutate the source module through an aliased functional-call buffer.
    buffers = {name: buffer.detach().clone() for name, buffer in actor.named_buffers()}
    return parameters, buffers


def _require_joint_policy_output(output, action_dim):
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "The clone actor must return one tensor containing mean/log-std rows."
        )
    expected = 2 * action_dim
    if output.ndim == 0 or output.shape[-1] != expected:
        raise ValueError(
            f"The clone actor output must end in {expected} values, got "
            f"shape {tuple(output.shape)}."
        )
    return output


@torch.no_grad()
def actor_mean_raw(actor: nn.Module, latents: torch.Tensor, *, action_dim: int):
    """Evaluate the unperturbed actor's deterministic pre-tanh mean."""

    action_dim = _positive_integer(action_dim, "action_dim")
    if not isinstance(latents, torch.Tensor) or latents.ndim == 0:
        raise ValueError("latents must be a non-scalar tensor.")
    spec = classify_parameter_noise_actor(actor, action_dim)
    if latents.shape[-1] != spec.input_dim:
        raise ValueError(
            f"Actor latents must end in input width {spec.input_dim}, got "
            f"shape {tuple(latents.shape)}."
        )
    parameters, buffers = _base_functional_state(actor)
    output = functional_call(
        actor,
        (parameters, buffers),
        (latents,),
        strict=True,
    )
    return _require_joint_policy_output(output, action_dim)[..., :action_dim]


@torch.no_grad()
def deterministic_actor_actions(
    actor: nn.Module,
    latents: torch.Tensor,
    *,
    action_dim: int,
):
    """Evaluate the unperturbed actor's deterministic squashed action."""

    return torch.tanh(actor_mean_raw(actor, latents, action_dim=action_dim))


def _validated_batched_parameters(actor, spec, batched_parameters):
    if not isinstance(batched_parameters, Mapping):
        raise TypeError(
            "batched_parameters must be a full parameter mapping, got "
            f"{type(batched_parameters).__name__}."
        )
    parameters, _ = _actor_state(actor, spec)
    if set(batched_parameters) != set(spec.parameter_names):
        missing = sorted(set(spec.parameter_names) - set(batched_parameters))
        extra = sorted(set(batched_parameters) - set(spec.parameter_names))
        raise ValueError(
            "Batched parameter keys do not match the actor: "
            f"missing={missing}, extra={extra}."
        )
    population_size = None
    for name in spec.parameter_names:
        value = batched_parameters[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Batched parameter {name!r} must be a tensor.")
        parameter = parameters[name]
        if population_size is None:
            if value.ndim != parameter.ndim + 1:
                raise ValueError(
                    f"Batched parameter {name!r} lacks a population dimension."
                )
            population_size = int(value.shape[0])
            _positive_integer(population_size, "parameter population size")
        expected_shape = (population_size, *parameter.shape)
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"Batched parameter {name!r} has shape {tuple(value.shape)}, "
                f"expected {expected_shape}."
            )
        if value.device != parameter.device or value.dtype != parameter.dtype:
            raise ValueError(
                f"Batched parameter {name!r} must match actor device/dtype."
            )
    return population_size


@torch.no_grad()
def population_actor_mean_raw(
    actor: nn.Module,
    spec: ParameterNoiseSpec,
    batched_parameters: Mapping[str, torch.Tensor],
    latents: torch.Tensor,
    *,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Evaluate K functional actors on latents shaped ``[K, R, D]``."""

    population_size = _validated_batched_parameters(actor, spec, batched_parameters)
    if not isinstance(latents, torch.Tensor) or latents.ndim != 3:
        raise ValueError(
            "Population latents must have shape [K, R, D], got "
            f"{getattr(latents, 'shape', None)}."
        )
    if latents.shape[0] != population_size or latents.shape[1] <= 0:
        raise ValueError(
            "Population latents must have the same positive K as parameters and "
            f"a positive R, got {tuple(latents.shape)} and K={population_size}."
        )
    if latents.shape[-1] != spec.input_dim:
        raise ValueError(
            f"Population latents must end in input width {spec.input_dim}, got "
            f"shape {tuple(latents.shape)}."
        )
    first_parameter = next(iter(batched_parameters.values()))
    if (
        latents.device != first_parameter.device
        or latents.dtype != first_parameter.dtype
    ):
        raise ValueError("Population latents must match actor parameter device/dtype.")
    if chunk_size is None:
        chunk_size = population_size
    else:
        chunk_size = _positive_integer(chunk_size, "chunk_size")

    _, actor_buffers = _actor_state(actor, spec)
    buffers = {
        name: buffer.detach().clone() for name, buffer in actor_buffers.items()
    }

    def call_one(parameters, one_actor_latents):
        return functional_call(
            actor,
            (parameters, buffers),
            (one_actor_latents,),
            strict=True,
        )

    outputs = []
    for start in range(0, population_size, chunk_size):
        stop = min(start + chunk_size, population_size)
        chunk_parameters = {
            name: value[start:stop] for name, value in batched_parameters.items()
        }
        output = torch.vmap(
            call_one,
            in_dims=(0, 0),
            randomness="error",
        )(chunk_parameters, latents[start:stop])
        outputs.append(_require_joint_policy_output(output, spec.action_dim))
    return torch.cat(outputs, dim=0)[..., : spec.action_dim]


@torch.no_grad()
def deterministic_population_actions(
    actor: nn.Module,
    spec: ParameterNoiseSpec,
    batched_parameters: Mapping[str, torch.Tensor],
    latents: torch.Tensor,
    *,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Evaluate deterministic post-tanh actions for a functional population."""

    mean_raw = population_actor_mean_raw(
        actor,
        spec,
        batched_parameters,
        latents,
        chunk_size=chunk_size,
    )
    return torch.tanh(mean_raw)


@torch.no_grad()
def post_tanh_action_rms(
    reference_actions: torch.Tensor,
    perturbed_actions: torch.Tensor,
) -> torch.Tensor:
    """Return per-coordinate RMS displacement between bounded actions."""

    if not isinstance(reference_actions, torch.Tensor) or not isinstance(
        perturbed_actions, torch.Tensor
    ):
        raise TypeError("reference_actions and perturbed_actions must be tensors.")
    if reference_actions.ndim == 0 or perturbed_actions.ndim == 0:
        raise ValueError("Action tensors must have at least one dimension.")
    if reference_actions.shape[-1] != perturbed_actions.shape[-1]:
        raise ValueError(
            "Action tensors must have the same final action dimension, got "
            f"{reference_actions.shape[-1]} and {perturbed_actions.shape[-1]}."
        )
    try:
        reference_actions, perturbed_actions = torch.broadcast_tensors(
            reference_actions, perturbed_actions
        )
    except RuntimeError as error:
        raise ValueError("Action tensors are not broadcast-compatible.") from error
    if reference_actions.numel() == 0:
        raise ValueError("Action RMS is undefined for empty tensors.")
    return (perturbed_actions - reference_actions).square().mean().sqrt()


@torch.no_grad()
def parameter_noise_action_rms(
    actor: nn.Module,
    spec: ParameterNoiseSpec,
    batched_parameters: Mapping[str, torch.Tensor],
    latents: torch.Tensor,
    *,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Measure deterministic post-tanh displacement on common latent states."""

    perturbed_actions = deterministic_population_actions(
        actor,
        spec,
        batched_parameters,
        latents,
        chunk_size=chunk_size,
    )
    reference_actions = deterministic_actor_actions(
        actor,
        latents,
        action_dim=spec.action_dim,
    )
    return post_tanh_action_rms(reference_actions, perturbed_actions)


def adapt_parameter_noise_stddev(
    stddev: float,
    measured_action_rms: float,
    target_action_rms: float,
    *,
    adaptation_rate: float = 0.5,
    min_stddev: float = 1e-8,
    max_stddev: float = float("inf"),
    max_update_ratio: float = 2.0,
) -> float:
    """Take one bounded log-proportional action-space calibration step."""

    values = {
        "stddev": stddev,
        "measured_action_rms": measured_action_rms,
        "target_action_rms": target_action_rms,
        "adaptation_rate": adaptation_rate,
        "min_stddev": min_stddev,
        "max_stddev": max_stddev,
        "max_update_ratio": max_update_ratio,
    }
    converted = {}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real scalar, got {value!r}.")
        converted[name] = float(value)

    stddev = converted["stddev"]
    measured_action_rms = converted["measured_action_rms"]
    target_action_rms = converted["target_action_rms"]
    adaptation_rate = converted["adaptation_rate"]
    min_stddev = converted["min_stddev"]
    max_stddev = converted["max_stddev"]
    max_update_ratio = converted["max_update_ratio"]
    finite_nonnegative = {
        "stddev": stddev,
        "measured_action_rms": measured_action_rms,
        "target_action_rms": target_action_rms,
    }
    for name, value in finite_nonnegative.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}.")
    if not math.isfinite(adaptation_rate) or not 0.0 < adaptation_rate <= 1.0:
        raise ValueError("adaptation_rate must be finite and in (0, 1].")
    if not math.isfinite(min_stddev) or min_stddev <= 0.0:
        raise ValueError("min_stddev must be finite and positive.")
    if max_stddev < min_stddev or math.isnan(max_stddev):
        raise ValueError("max_stddev must be at least min_stddev.")
    if not math.isfinite(max_update_ratio) or max_update_ratio < 1.0:
        raise ValueError("max_update_ratio must be finite and at least one.")

    if target_action_rms == 0.0:
        return min_stddev
    current = min(max(stddev, min_stddev), max_stddev)
    if measured_action_rms == 0.0:
        update_ratio = max_update_ratio
    else:
        log_limit = math.log(max_update_ratio)
        log_update = adaptation_rate * (
            math.log(target_action_rms) - math.log(measured_action_rms)
        )
        update_ratio = math.exp(min(max(log_update, -log_limit), log_limit))
    return min(max(current * update_ratio, min_stddev), max_stddev)
