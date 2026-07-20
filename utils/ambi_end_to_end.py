"""Strict loader and deterministic materializer for end-to-end AMBI trials."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path


class EndToEndSuiteError(ValueError):
    """Raised when the end-to-end trial matrix is malformed."""


_CONDITION_FIELDS = {
    "group",
    "trial",
    "config",
    "train",
    "plan",
    "inner",
    "rounds",
    "rollouts_per_round",
    "updates_per_round",
    "batch",
    "transitions_per_round",
    "transitions_per_action",
    "updates_per_action",
    "replay_rows_per_action",
    "replay_capacity",
}
_INTEGER_FIELDS = _CONDITION_FIELDS - {"group", "trial", "config"}


def _reject_nonfinite_constant(value):
    raise EndToEndSuiteError(f"non-finite JSON number is not allowed: {value}")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise EndToEndSuiteError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path):
    path = Path(path)
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except OSError as exc:
        raise EndToEndSuiteError(f"Could not read JSON {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EndToEndSuiteError(f"Invalid JSON {path}: {exc}") from exc


def _positive_int(value, location):
    if isinstance(value, bool):
        raise EndToEndSuiteError(f"{location} must be a positive integer.")
    try:
        numeric = float(value)
        resolved = int(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EndToEndSuiteError(f"{location} must be a positive integer.") from exc
    if not math.isfinite(numeric) or resolved <= 0 or numeric != resolved:
        raise EndToEndSuiteError(f"{location} must be a positive integer.")
    return resolved


def validate_suite(matrix):
    """Validate the complete matrix and return it without mutation."""
    if not isinstance(matrix, dict):
        raise EndToEndSuiteError("suite must be a JSON object.")
    if matrix.get("schema_version") != 1:
        raise EndToEndSuiteError("schema_version must be 1.")
    if not isinstance(matrix.get("base_ambi_alg_config"), str):
        raise EndToEndSuiteError("base_ambi_alg_config must be a relative JSON path.")
    base = Path(matrix["base_ambi_alg_config"])
    if base.is_absolute() or base.suffix.lower() != ".json":
        raise EndToEndSuiteError("base_ambi_alg_config must be a relative JSON path.")
    if matrix.get("checked_in_training_seed") != 55:
        raise EndToEndSuiteError("checked_in_training_seed must be the exploratory seed 55.")

    conditions = matrix.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise EndToEndSuiteError("conditions must be a non-empty list.")

    configs = set()
    selectors = set()
    for index, condition in enumerate(conditions):
        location = f"conditions[{index}]"
        if not isinstance(condition, dict):
            raise EndToEndSuiteError(f"{location} must be an object.")
        missing = _CONDITION_FIELDS - set(condition)
        unknown = set(condition) - _CONDITION_FIELDS
        if missing or unknown:
            raise EndToEndSuiteError(
                f"{location} fields mismatch; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}."
            )
        for key in ("group", "trial", "config"):
            if not isinstance(condition[key], str) or not condition[key]:
                raise EndToEndSuiteError(f"{location}.{key} must be a non-empty string.")
        for key in _INTEGER_FIELDS:
            _positive_int(condition[key], f"{location}.{key}")

        selector = (condition["group"], condition["trial"])
        if selector in selectors:
            raise EndToEndSuiteError(f"duplicate condition selector: {selector}")
        if condition["config"] in configs:
            raise EndToEndSuiteError(f"duplicate condition config: {condition['config']}")
        selectors.add(selector)
        configs.add(condition["config"])

        per_round = condition["rollouts_per_round"] * condition["inner"]
        per_action = condition["rounds"] * per_round
        updates = condition["rounds"] * condition["updates_per_round"]
        replay_rows = updates * condition["batch"]
        if condition["transitions_per_round"] != per_round:
            raise EndToEndSuiteError(f"{location}.transitions_per_round must equal N*H.")
        if condition["transitions_per_action"] != per_action:
            raise EndToEndSuiteError(f"{location}.transitions_per_action must equal J*N*H.")
        if condition["updates_per_action"] != updates:
            raise EndToEndSuiteError(f"{location}.updates_per_action must equal J*G.")
        if condition["replay_rows_per_action"] != replay_rows:
            raise EndToEndSuiteError(
                f"{location}.replay_rows_per_action must equal J*G*inner_batch_size."
            )
        if condition["replay_capacity"] != per_action:
            raise EndToEndSuiteError(
                f"{location}.replay_capacity must equal J*N*H for action-local replay."
            )
    return matrix


def load_suite(path):
    return validate_suite(_load_json(path))


def _base_path(matrix_path, matrix):
    return (Path(matrix_path).resolve().parent / matrix["base_ambi_alg_config"]).resolve()


def resolve_condition(matrix_path, condition, matrix=None):
    """Return one full algorithm config generated from the canonical AMBI anchor."""
    matrix_path = Path(matrix_path).resolve()
    matrix = load_suite(matrix_path) if matrix is None else validate_suite(matrix)
    if isinstance(condition, str):
        matches = [item for item in matrix["conditions"] if item["config"] == condition]
        if len(matches) != 1:
            raise EndToEndSuiteError(f"Unknown condition config {condition!r}.")
        condition = matches[0]
    elif condition not in matrix["conditions"]:
        raise EndToEndSuiteError("condition must be a checked-in matrix entry or config name.")

    algorithm_config = _load_json(_base_path(matrix_path, matrix))
    if algorithm_config.get("alg") != "AMBITDMPC2/AMBITDMPC2":
        raise EndToEndSuiteError("base_ambi_alg_config must select AMBITDMPC2/AMBITDMPC2.")
    params = algorithm_config.get("alg_params")
    if not isinstance(params, dict):
        raise EndToEndSuiteError("base AMBI config must contain an alg_params object.")

    overrides = {
        "train_unroll_horizon": condition["train"],
        "outer_planning_horizon": condition["plan"],
        "inner_rollout_horizon": condition["inner"],
        "inner_rounds": condition["rounds"],
        "inner_rollouts_per_round": condition["rollouts_per_round"],
        "inner_updates_per_round": condition["updates_per_round"],
        "inner_batch_size": condition["batch"],
        "inner_replay_capacity": condition["replay_capacity"],
    }
    params.update(overrides)
    algorithm_config["seed"] = matrix["checked_in_training_seed"]
    algorithm_config["total_steps"] = 1_000_000
    return algorithm_config


def render_condition_configs(matrix_path):
    """Return deterministic filename-to-JSON-text output for every condition."""
    matrix_path = Path(matrix_path).resolve()
    matrix = load_suite(matrix_path)
    return {
        f"{condition['config']}.json": json.dumps(
            resolve_condition(matrix_path, condition, matrix=matrix),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
        for condition in matrix["conditions"]
    }


def materialize_condition_configs(matrix_path, output_dir):
    """Write the deterministic flat AMBI config set and return written paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, payload in render_condition_configs(matrix_path).items():
        path = output_dir / filename
        path.write_text(payload, encoding="utf-8")
        written.append(path)
    return written
