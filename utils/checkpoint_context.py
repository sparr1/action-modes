"""Portable checkpoint configuration shared by evaluation and rendering.

A present sidecar is authoritative: malformed metadata never falls back to
unrelated settings in a training directory. Legacy discovery remains a renderer
concern; research benchmarks require the saved sidecar explicitly.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping


METADATA_SCHEMA_VERSION = 1


class CheckpointContextError(RuntimeError):
    """An actionable checkpoint configuration error."""


@dataclass(frozen=True)
class CheckpointContext:
    trial_run_params: dict[str, Any]
    experiment_params: dict[str, Any]
    source: Path
    metadata: dict[str, Any] | None = None


def load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError as exc:
        raise CheckpointContextError(f"{description} does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointContextError(f"Could not read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointContextError(
            f"Malformed {description} {path}: expected a JSON object."
        )
    return value


def _is_int(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _validate_sidecar(metadata: dict[str, Any], path: Path) -> CheckpointContext:
    version = metadata.get("schema_version")
    if not _is_int(version) or int(version) != METADATA_SCHEMA_VERSION:
        raise CheckpointContextError(
            f"Unsupported checkpoint metadata schema_version={version!r} in {path}; "
            f"expected {METADATA_SCHEMA_VERSION}."
        )

    run_params = metadata.get("trial_run_params")
    experiment_params = metadata.get("experiment_params")
    checkpoint = metadata.get("checkpoint")
    if not isinstance(run_params, dict):
        raise CheckpointContextError(
            f"Malformed checkpoint metadata {path}: trial_run_params must be an object."
        )
    if not isinstance(experiment_params, dict):
        raise CheckpointContextError(
            f"Malformed checkpoint metadata {path}: experiment_params must be an object."
        )
    if not isinstance(checkpoint, dict):
        raise CheckpointContextError(
            f"Malformed checkpoint metadata {path}: checkpoint must be an object."
        )

    kind = checkpoint.get("kind")
    step = checkpoint.get("step")
    episode = checkpoint.get("episode")
    best_score = checkpoint.get("best_score")
    best_window = checkpoint.get("best_window")
    if not isinstance(kind, str) or not kind:
        raise CheckpointContextError(
            f"Malformed checkpoint metadata {path}: checkpoint.kind must be a non-empty string."
        )
    if not _is_int(step) or int(step) < 0:
        raise CheckpointContextError(
            f"Malformed checkpoint metadata {path}: checkpoint.step must be non-negative."
        )
    if not _is_int(episode) or int(episode) < 0:
        raise CheckpointContextError(
            f"Malformed checkpoint metadata {path}: checkpoint.episode must be non-negative."
        )
    if best_score is not None and (
        isinstance(best_score, bool)
        or not isinstance(best_score, Real)
        or not math.isfinite(float(best_score))
    ):
        raise CheckpointContextError(
            f"Malformed checkpoint metadata {path}: checkpoint.best_score must be finite or null."
        )
    if not _is_int(best_window) or int(best_window) <= 0:
        raise CheckpointContextError(
            f"Malformed checkpoint metadata {path}: checkpoint.best_window must be positive."
        )

    validate_context_params(run_params, experiment_params, path)
    return CheckpointContext(
        trial_run_params=copy.deepcopy(run_params),
        experiment_params=copy.deepcopy(experiment_params),
        source=path,
        metadata=copy.deepcopy(metadata),
    )


def validate_context_params(
    run_params: Mapping[str, Any],
    experiment_params: Mapping[str, Any],
    source: Path,
) -> None:
    algorithm = run_params.get("alg")
    if not isinstance(algorithm, str) or not algorithm:
        raise CheckpointContextError(
            f"Configuration from {source} is missing a non-empty trial_run_params.alg."
        )
    if not isinstance(run_params.get("env"), str) or not run_params.get("env"):
        raise CheckpointContextError(
            f"Configuration from {source} is missing a non-empty trial_run_params.env."
        )
    if not isinstance(run_params.get("alg_params", {}), dict):
        raise CheckpointContextError(
            f"Configuration from {source} has a non-object trial_run_params.alg_params."
        )
    if "env_params" in experiment_params and not isinstance(
        experiment_params["env_params"], dict
    ):
        raise CheckpointContextError(
            f"Configuration from {source} has a non-object experiment_params.env_params."
        )


def load_checkpoint_context(
    checkpoint: Path | str, metadata_path: Path | str | None = None
) -> CheckpointContext:
    """Load validated saved settings from an explicit or adjacent sidecar.

    The checkpoint's bytes are intentionally not read here; the evaluator or
    renderer owns checkpoint existence, content hashing, and weight loading.
    """
    path = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path is not None
        else Path(str(Path(checkpoint).expanduser()) + ".metadata.json")
    )
    return _validate_sidecar(load_json_object(path, "checkpoint metadata"), path)
