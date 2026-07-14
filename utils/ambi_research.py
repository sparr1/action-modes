"""Load and materialize one-axis-at-a-time AMBI research presets.

The preset file deliberately stores small overrides instead of duplicating full
algorithm JSON files.  Materialized configs have the same shape as files in
``configs/algs`` and can therefore be passed to the existing ``main.py``
experiment runner without a special code path.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path


_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_MAX_NUMPY_SEED = 2**32 - 1


class PresetMatrixError(ValueError):
    """Raised when a research preset matrix is malformed or ambiguous."""


def _reject_nonfinite_constant(value):
    raise PresetMatrixError(f"non-finite JSON number is not allowed: {value}")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PresetMatrixError(f"duplicate JSON key: {key}")
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
        raise PresetMatrixError(f"Could not read preset JSON {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PresetMatrixError(f"Invalid preset JSON {path}: {exc}") from exc


def _require_mapping(value, location):
    if not isinstance(value, dict):
        raise PresetMatrixError(f"{location} must be a JSON object.")
    return value


def _validate_name(value, location):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise PresetMatrixError(
            f"{location} must use lowercase letters, numbers, '-' or '_'; got {value!r}."
        )


def validate_preset_matrix(matrix):
    """Validate and return ``matrix`` without modifying it."""
    _require_mapping(matrix, "preset matrix")
    if matrix.get("schema_version") != 1:
        raise PresetMatrixError("preset matrix schema_version must be 1.")
    if not isinstance(matrix.get("base_alg_config"), str):
        raise PresetMatrixError("base_alg_config must be a relative JSON path.")
    if Path(matrix["base_alg_config"]).is_absolute():
        raise PresetMatrixError("base_alg_config must be relative to the matrix file.")

    _require_mapping(matrix.get("shared_alg_params", {}), "shared_alg_params")
    environment = _require_mapping(matrix.get("environment", {}), "environment")
    if not isinstance(environment.get("id"), str) or not environment["id"]:
        raise PresetMatrixError("environment.id must be a non-empty string.")
    _require_mapping(environment.get("params", {}), "environment.params")

    evaluation = _require_mapping(matrix.get("evaluation", {}), "evaluation")
    if "seeds" in evaluation:
        seeds = evaluation["seeds"]
        if not isinstance(seeds, list) or not seeds or any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= _MAX_NUMPY_SEED
            for seed in seeds
        ):
            raise PresetMatrixError(
                "evaluation.seeds must be non-empty and contain valid NumPy seed integers."
            )
        if len(set(seeds)) != len(seeds):
            raise PresetMatrixError("evaluation.seeds must not contain duplicates.")
    if "controller_seed" in evaluation and (
        isinstance(evaluation["controller_seed"], bool)
        or not isinstance(evaluation["controller_seed"], int)
        or not 0 <= evaluation["controller_seed"] <= _MAX_NUMPY_SEED
    ):
        raise PresetMatrixError("evaluation.controller_seed must be a valid NumPy seed.")
    if evaluation.get("max_steps") is not None and (
        isinstance(evaluation["max_steps"], bool)
        or not isinstance(evaluation["max_steps"], int)
        or evaluation["max_steps"] <= 0
    ):
        raise PresetMatrixError("evaluation.max_steps must be a positive integer or null.")

    comparisons = _require_mapping(matrix.get("comparisons"), "comparisons")
    if not comparisons:
        raise PresetMatrixError("comparisons must contain at least one comparison.")
    selectors = set()
    for comparison_name, comparison in comparisons.items():
        _validate_name(comparison_name, "comparison name")
        comparison = _require_mapping(comparison, f"comparisons.{comparison_name}")
        if "frozen_evaluation" in comparison and not isinstance(
            comparison["frozen_evaluation"], bool
        ):
            raise PresetMatrixError(
                f"comparisons.{comparison_name}.frozen_evaluation must be boolean."
            )
        if "frozen_evaluation_reason" in comparison and not isinstance(
            comparison["frozen_evaluation_reason"], str
        ):
            raise PresetMatrixError(
                f"comparisons.{comparison_name}.frozen_evaluation_reason must be a string."
            )
        variants = _require_mapping(
            comparison.get("variants"), f"comparisons.{comparison_name}.variants"
        )
        if not variants:
            raise PresetMatrixError(
                f"comparisons.{comparison_name}.variants must not be empty."
            )
        reference = comparison.get("reference")
        if reference not in variants:
            raise PresetMatrixError(
                f"comparisons.{comparison_name}.reference must name one of its variants."
            )
        for variant_name, variant in variants.items():
            _validate_name(variant_name, f"{comparison_name} variant name")
            variant = _require_mapping(
                variant, f"comparisons.{comparison_name}.variants.{variant_name}"
            )
            unknown = set(variant) - {"description", "alg_params", "run_params"}
            if unknown:
                raise PresetMatrixError(
                    f"Unknown fields in {comparison_name}/{variant_name}: {sorted(unknown)}."
                )
            _require_mapping(
                variant.get("alg_params", {}),
                f"comparisons.{comparison_name}.variants.{variant_name}.alg_params",
            )
            _require_mapping(
                variant.get("run_params", {}),
                f"comparisons.{comparison_name}.variants.{variant_name}.run_params",
            )
            if "alg_params" in variant.get("run_params", {}):
                raise PresetMatrixError(
                    f"{comparison_name}/{variant_name}.run_params cannot replace alg_params; "
                    "put algorithm overrides in its alg_params object."
                )
            selectors.add(f"{comparison_name}/{variant_name}")

    defaults = evaluation.get("default_presets", [])
    if not isinstance(defaults, list) or any(not isinstance(item, str) for item in defaults):
        raise PresetMatrixError("evaluation.default_presets must be a list of selectors.")
    unknown_defaults = sorted(set(defaults) - selectors)
    if unknown_defaults:
        raise PresetMatrixError(
            f"evaluation.default_presets contains unknown selectors: {unknown_defaults}."
        )
    return matrix


def load_preset_matrix(path):
    """Load a matrix with strict duplicate-key checking."""
    return validate_preset_matrix(_load_json(path))


def list_preset_selectors(matrix, comparisons=None):
    """Return stable ``comparison/variant`` selectors from ``matrix``."""
    validate_preset_matrix(matrix)
    requested = None if comparisons is None else set(comparisons)
    if requested is not None:
        unknown = requested - set(matrix["comparisons"])
        if unknown:
            raise PresetMatrixError(f"Unknown comparisons: {sorted(unknown)}.")
    return [
        f"{comparison_name}/{variant_name}"
        for comparison_name, comparison in matrix["comparisons"].items()
        if requested is None or comparison_name in requested
        for variant_name in comparison["variants"]
    ]


def normalize_selectors(matrix, selectors=None, comparisons=None):
    """Resolve explicit selectors, whole comparisons, or matrix defaults."""
    validate_preset_matrix(matrix)
    result = []
    if selectors:
        result.extend(selectors)
    if comparisons:
        result.extend(list_preset_selectors(matrix, comparisons=comparisons))
    if not result:
        result.extend(matrix.get("evaluation", {}).get("default_presets", []))
    if not result:
        result.extend(list_preset_selectors(matrix))

    known = set(list_preset_selectors(matrix))
    unknown = sorted(set(result) - known)
    if unknown:
        raise PresetMatrixError(f"Unknown preset selectors: {unknown}.")
    # Preserve user/matrix order while evaluating a duplicate only once.
    return list(dict.fromkeys(result))


def _base_config_path(matrix_path, matrix):
    matrix_path = Path(matrix_path).resolve()
    base = (matrix_path.parent / matrix["base_alg_config"]).resolve()
    if base.suffix.lower() != ".json":
        raise PresetMatrixError("base_alg_config must point to a JSON file.")
    return base


def resolve_preset(matrix_path, selector, matrix=None):
    """Materialize one preset as a normal AMBI algorithm configuration.

    Returns metadata plus ``algorithm_config``.  The source matrix and base
    configuration are never mutated, which makes repeated expansion safe.
    """
    matrix_path = Path(matrix_path).resolve()
    matrix = load_preset_matrix(matrix_path) if matrix is None else validate_preset_matrix(matrix)
    if selector.count("/") != 1:
        raise PresetMatrixError(
            f"Preset selector must be 'comparison/variant', got {selector!r}."
        )
    comparison_name, variant_name = selector.split("/", 1)
    try:
        comparison = matrix["comparisons"][comparison_name]
        variant = comparison["variants"][variant_name]
    except KeyError as exc:
        raise PresetMatrixError(f"Unknown preset selector {selector!r}.") from exc

    base_path = _base_config_path(matrix_path, matrix)
    algorithm_config = _load_json(base_path)
    _require_mapping(algorithm_config, f"base algorithm config {base_path}")
    alg_params = _require_mapping(
        algorithm_config.get("alg_params"), f"{base_path}.alg_params"
    )
    alg_params.update(copy.deepcopy(matrix.get("shared_alg_params", {})))
    alg_params.update(copy.deepcopy(variant.get("alg_params", {})))
    algorithm_config.update(copy.deepcopy(variant.get("run_params", {})))

    return {
        "selector": selector,
        "comparison": comparison_name,
        "variant": variant_name,
        "reference": comparison["reference"],
        "description": variant.get("description", ""),
        "algorithm_config": algorithm_config,
        "environment": copy.deepcopy(matrix["environment"]),
        "evaluation": copy.deepcopy(matrix.get("evaluation", {})),
    }


def materialize_presets(matrix_path, output_dir, selectors=None, comparisons=None):
    """Write resolved algorithm JSONs plus a matching experiment manifest."""
    matrix_path = Path(matrix_path).resolve()
    matrix = load_preset_matrix(matrix_path)
    selectors = normalize_selectors(matrix, selectors, comparisons)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for selector in selectors:
        resolved = resolve_preset(matrix_path, selector, matrix=matrix)
        output = output_dir / f"{selector.replace('/', '__')}.json"
        output.write_text(
            json.dumps(
                resolved["algorithm_config"],
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(output)

    experiment = {
        "overrides_alg": {"env": matrix["environment"]["id"]},
        "env_params": copy.deepcopy(matrix["environment"].get("params", {})),
        "trials": 1,
        "configs": [path.stem for path in written],
        "logs": "timestamp",
        "save_trials": "all",
        "log_info": False,
    }
    (output_dir / "AMBIResearchExperiment.json").write_text(
        json.dumps(experiment, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return written
