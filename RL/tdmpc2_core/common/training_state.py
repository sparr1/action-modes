"""Small schema checks for exact-resume payloads.

Durable checkpoints are already checksummed and tied to the resolved training
configuration. These helpers reject structural and dtype/layout changes before
PyTorch can silently cast or partially install an incompatible payload.
"""

from collections.abc import Mapping

import torch


def require_mapping(value, name):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")
    return value


def require_exact_keys(value, expected, name):
    value = require_mapping(value, name)
    expected = set(expected)
    missing = sorted(expected - set(value))
    unexpected = sorted(set(value) - expected)
    if missing or unexpected:
        raise ValueError(
            f"{name} has an incompatible schema: missing={missing}, "
            f"unexpected={unexpected}."
        )
    return value


def preflight_module_state(module, incoming, name):
    """Check keys, tensor shapes, and dtypes without mutating ``module``."""

    incoming = require_mapping(incoming, name)
    expected = module.state_dict()
    missing = sorted(set(expected) - set(incoming))
    unexpected = sorted(set(incoming) - set(expected))
    shapes = sorted(
        key
        for key in set(expected) & set(incoming)
        if torch.is_tensor(expected[key])
        and (
            not torch.is_tensor(incoming[key])
            or expected[key].shape != incoming[key].shape
        )
    )
    dtypes = sorted(
        key
        for key in set(expected) & set(incoming)
        if torch.is_tensor(expected[key])
        and torch.is_tensor(incoming[key])
        and expected[key].dtype != incoming[key].dtype
    )
    if missing or unexpected or shapes or dtypes:
        raise ValueError(
            f"{name} is incompatible before load: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}, shape_mismatches={shapes[:5]}, "
            f"dtype_mismatches={dtypes[:5]}."
        )


def require_tensor(value, name, *, shape=None, dtype=None):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor.")
    if shape is not None and value.shape != torch.Size(shape):
        raise ValueError(
            f"{name} has shape {tuple(value.shape)}; expected {tuple(shape)}."
        )
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} has dtype {value.dtype}; expected {dtype}.")
    return value


def _exact_value_equal(actual, expected):
    if torch.is_tensor(actual) or torch.is_tensor(expected):
        return (
            torch.is_tensor(actual)
            and torch.is_tensor(expected)
            and actual.shape == expected.shape
            and actual.dtype == expected.dtype
            and torch.equal(actual.detach().cpu(), expected.detach().cpu())
        )
    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and isinstance(expected, Mapping)
            and set(actual) == set(expected)
            and all(_exact_value_equal(actual[key], expected[key]) for key in expected)
        )
    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        return (
            type(actual) is type(expected)
            and len(actual) == len(expected)
            and all(
                _exact_value_equal(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected)
            )
        )
    return type(actual) is type(expected) and actual == expected


def _expected_adam_step(expected_steps, step):
    # Adam increments its floating step tensor in place. Once the unit
    # increment is below the dtype's resolution, the stored value stays at the
    # largest consecutive integer representable by that dtype.
    saturation = int(2.0 / torch.finfo(step.dtype).eps)
    return min(expected_steps, saturation)


def preflight_adam_state(optimizer, incoming, name, *, expected_steps):
    """Validate an exact Adam snapshot without mutating ``optimizer``."""

    if (
        isinstance(expected_steps, bool)
        or not isinstance(expected_steps, int)
        or expected_steps < 0
    ):
        raise TypeError(f"{name} expected_steps must be a non-negative integer.")
    incoming = require_exact_keys(incoming, {"state", "param_groups"}, name)
    expected = optimizer.state_dict()
    groups = incoming["param_groups"]
    if not isinstance(groups, list) or len(groups) != len(expected["param_groups"]):
        raise ValueError(f"{name} parameter-group layout is incompatible.")

    parameter_map = {}
    parameter_amsgrad = {}
    parameter_step_dtype = {}
    for incoming_group, expected_group, live_group in zip(
        groups, expected["param_groups"], optimizer.param_groups
    ):
        if (
            not isinstance(incoming_group, Mapping)
            or set(incoming_group) != set(expected_group)
        ):
            raise ValueError(f"{name} parameter-group fields are incompatible.")
        mismatched_options = sorted(
            key
            for key in expected_group
            if key != "params"
            and not _exact_value_equal(incoming_group[key], expected_group[key])
        )
        if mismatched_options:
            raise ValueError(
                f"{name} parameter-group hyperparameters are incompatible: "
                f"{mismatched_options}."
            )
        incoming_ids = incoming_group["params"]
        expected_ids = expected_group["params"]
        live_parameters = live_group["params"]
        if (
            not isinstance(incoming_ids, list)
            or len(incoming_ids) != len(expected_ids)
            or len(incoming_ids) != len(live_parameters)
        ):
            raise ValueError(f"{name} parameter-group layout is incompatible.")
        if incoming_ids != expected_ids:
            raise ValueError(f"{name} parameter identifiers/order are incompatible.")
        for parameter_id, parameter in zip(incoming_ids, live_parameters):
            if parameter_id in parameter_map:
                raise ValueError(f"{name} contains duplicate parameter identifiers.")
            parameter_map[parameter_id] = parameter
            parameter_amsgrad[parameter_id] = bool(
                incoming_group.get("amsgrad", False)
            )
            parameter_step_dtype[parameter_id] = (
                torch.float32
                if incoming_group.get("fused")
                or torch.get_default_dtype() != torch.float64
                else torch.float64
            )

    state = incoming["state"]
    if not isinstance(state, Mapping):
        raise TypeError(f"{name} optimizer state must be a mapping.")
    expected_state_ids = set(parameter_map) if expected_steps > 0 else set()
    if set(state) != expected_state_ids:
        raise ValueError(f"{name} optimizer state inventory is incomplete or unexpected.")

    for parameter_id, parameter_state in state.items():
        if not isinstance(parameter_state, Mapping):
            raise TypeError(f"{name} per-parameter state must be a mapping.")
        parameter = parameter_map[parameter_id]
        expected_fields = {"step", "exp_avg", "exp_avg_sq"}
        if parameter_amsgrad[parameter_id]:
            expected_fields.add("max_exp_avg_sq")
        if set(parameter_state) != expected_fields:
            raise ValueError(f"{name} per-parameter fields are incompatible.")
        step = parameter_state["step"]
        if (
            not torch.is_tensor(step)
            or step.shape != torch.Size([])
            or step.dtype != parameter_step_dtype[parameter_id]
            or not bool(torch.isfinite(step).item())
            or step.item() != _expected_adam_step(expected_steps, step)
        ):
            raise ValueError(
                f"{name} optimizer step does not match its scientific counter."
            )
        for field in expected_fields - {"step"}:
            value = parameter_state[field]
            if (
                not torch.is_tensor(value)
                or value.shape != parameter.shape
                or value.dtype != parameter.dtype
            ):
                raise ValueError(
                    f"{name} tensor {field!r} is incompatible with its parameter."
                )
    return incoming
