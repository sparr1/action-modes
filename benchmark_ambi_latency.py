"""Benchmark frozen-checkpoint AMBI action latency without environment steps.

The benchmark deliberately exercises only ``AMBITDMPC2.predict``.  A seeded
observation bank is built with environment resets during setup, then the
environment is never stepped and neither ``learn`` nor an outer update is
called.  Every J/N/G cell receives a fresh model loaded from the same outer
checkpoint.  For process-isolated cold-start measurements, invoke this runner
once per cell (the maintained Slurm launcher does so).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from evaluate_ambi_checkpoint import (
    _initialize_frozen_model,
    _make_env,
    _outer_state_digest,
    _seed_spaces,
)


BENCHMARK_NAME = "ambi-inner-latency"
SCHEMA_VERSION = 1
DEFAULT_CONFIG = (
    Path(__file__).resolve().parent
    / "configs"
    / "research"
    / "ambi_latency_benchmark.json"
)
_MAX_NUMPY_SEED = 2**32 - 1

PHASE_METRIC_KEYS = (
    "inner_action_seconds",
    "inner_setup_seconds",
    "inner_rollout_seconds",
    "inner_update_seconds",
    "inner_execution_seconds",
    "inner_mppi_seconds",
)
WORK_COUNTER_KEYS = (
    "inner_actions",
    "inner_rounds",
    "inner_iterations",
    "inner_rollouts",
    "inner_requested_rollouts",
    "inner_rollout_count",
    "inner_steps",
    "inner_model_steps",
    "inner_total_model_steps",
    "inner_nominal_model_steps",
    "inner_realized_model_steps",
    "inner_updates",
    "inner_update_slots",
    "inner_requested_update_slots",
    "inner_critic_optimizer_steps",
    "inner_actor_optimizer_steps",
    "inner_temperature_optimizer_steps",
    "inner_target_updates",
    "inner_critic_target_updates",
    "inner_actor_target_updates",
    "inner_policy_evaluations",
    "inner_q_evaluations",
    "inner_replay_draws",
    "inner_buffer_size",
    "inner_buffer_capacity",
)
COMPILE_FALLBACK_KEYS = (
    "inner_compile_rollout_fallback",
    "inner_compile_critic_fallback",
    "inner_compile_actor_fallback",
    "inner_compile_fallback",
)
EXPECTED_COUNTER_METRICS = {
    # The engine reports the allocated latent replay object as a buffer.
    "inner_replay_capacity": "inner_buffer_capacity",
}


class BenchmarkConfigError(ValueError):
    """Raised when a latency benchmark configuration is invalid."""


@dataclass(frozen=True)
class BenchmarkCell:
    """One explicit canonical AMBI J/N/G work cell."""

    name: str
    J: int
    N: int
    G: int
    source: Mapping[str, Any]

    @property
    def selector(self) -> str:
        return f"{self.J},{self.N},{self.G}"


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkConfigError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value):
    raise BenchmarkConfigError(f"non-finite JSON number is not allowed: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except OSError as exc:
        raise BenchmarkConfigError(f"Could not read JSON {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkConfigError(f"Invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkConfigError(f"{path} must contain a JSON object.")
    return value


def _positive_int(value, location):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise BenchmarkConfigError(f"{location} must be a positive integer.")
    value = int(value)
    if value <= 0:
        raise BenchmarkConfigError(f"{location} must be a positive integer.")
    return value


def _nonnegative_int(value, location):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise BenchmarkConfigError(f"{location} must be a non-negative integer.")
    value = int(value)
    if value < 0:
        raise BenchmarkConfigError(f"{location} must be a non-negative integer.")
    return value


def _seed(value, location):
    value = _nonnegative_int(value, location)
    if value > _MAX_NUMPY_SEED:
        raise BenchmarkConfigError(f"{location} must be a valid NumPy seed.")
    return value


def _resolve_relative(path_value, *, relative_to: Path, location: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise BenchmarkConfigError(f"{location} must be a non-empty path string.")
    path = Path(path_value)
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _parse_cell(value: str) -> tuple[int, int, int]:
    """Parse the launcher-facing ``J,N,G`` cell selector."""
    try:
        pieces = [piece.strip() for piece in value.split(",")]
        if len(pieces) != 3 or any(not piece for piece in pieces):
            raise ValueError
        J, N, G = (int(piece) for piece in pieces)
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"cell must be J,N,G with integer values, got {value!r}"
        ) from exc
    if J <= 0 or N <= 0 or G < 0:
        raise argparse.ArgumentTypeError(
            f"cell requires J>0, N>0, and G>=0, got {value!r}"
        )
    return J, N, G


def _normalize_cell(raw: Any, index: int) -> BenchmarkCell:
    if not isinstance(raw, dict):
        raise BenchmarkConfigError(f"cells[{index}] must be a JSON object.")
    J = _positive_int(raw.get("J"), f"cells[{index}].J")
    N = _positive_int(raw.get("N"), f"cells[{index}].N")
    G = _nonnegative_int(raw.get("G"), f"cells[{index}].G")
    name = raw.get("name", f"j{J}_n{N}_g{G}")
    if not isinstance(name, str) or not name:
        raise BenchmarkConfigError(f"cells[{index}].name must be a non-empty string.")
    return BenchmarkCell(name=name, J=J, N=N, G=G, source=copy.deepcopy(raw))


def load_benchmark_config(path: Path | str) -> dict[str, Any]:
    """Load and normalize the maintained direct-config benchmark specification."""
    path = Path(path).resolve()
    raw = _load_json(path)
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise BenchmarkConfigError(
            f"benchmark config schema_version must be {SCHEMA_VERSION}."
        )
    benchmark_name = raw.get("benchmark", BENCHMARK_NAME)
    if benchmark_name != BENCHMARK_NAME:
        raise BenchmarkConfigError(
            f"benchmark must be {BENCHMARK_NAME!r}, got {benchmark_name!r}."
        )
    base = raw.get("base")
    if not isinstance(base, dict):
        raise BenchmarkConfigError("base must be a JSON object.")
    algorithm_config_path = _resolve_relative(
        base.get("algorithm_config"),
        relative_to=path.parent,
        location="base.algorithm_config",
    )
    if not algorithm_config_path.is_file():
        raise BenchmarkConfigError(
            f"base.algorithm_config does not exist: {algorithm_config_path}"
        )
    algorithm_config = _load_json(algorithm_config_path)
    if not isinstance(algorithm_config.get("alg_params"), dict):
        raise BenchmarkConfigError(
            f"{algorithm_config_path}.alg_params must be a JSON object."
        )

    environment = base.get("environment", raw.get("environment"))
    if not isinstance(environment, dict):
        raise BenchmarkConfigError("base.environment must be a JSON object.")
    if not isinstance(environment.get("id"), str) or not environment["id"]:
        raise BenchmarkConfigError("base.environment.id must be a non-empty string.")
    if not isinstance(environment.get("params", {}), dict):
        raise BenchmarkConfigError("base.environment.params must be a JSON object.")

    settings = raw.get("settings", {})
    if not isinstance(settings, dict):
        raise BenchmarkConfigError("settings must be a JSON object.")
    cold_calls = _positive_int(settings.get("cold_calls", 1), "settings.cold_calls")
    if cold_calls != 1:
        raise BenchmarkConfigError(
            "settings.cold_calls must be 1: only the first action is a cold call."
        )
    normalized_settings = copy.deepcopy(settings)
    normalized_settings.update({
        "cold_calls": cold_calls,
        "warmup_calls": _nonnegative_int(
            settings.get("warmup_calls", 49), "settings.warmup_calls"
        ),
        "measured_calls": _positive_int(
            settings.get("measured_calls", 200), "settings.measured_calls"
        ),
        "observation_bank_size": _positive_int(
            settings.get("observation_bank_size", 32),
            "settings.observation_bank_size",
        ),
        "environment_seed": _seed(
            settings.get("environment_seed", 101), "settings.environment_seed"
        ),
        "controller_seed": _seed(
            settings.get("controller_seed", algorithm_config.get("seed", 0)),
            "settings.controller_seed",
        ),
        "action_mode": settings.get("action_mode", "training"),
        "include_samples": settings.get("include_samples", True),
    })
    if normalized_settings["action_mode"] not in {"training", "evaluation"}:
        raise BenchmarkConfigError(
            "settings.action_mode must be 'training' or 'evaluation'."
        )
    if not isinstance(normalized_settings["include_samples"], bool):
        raise BenchmarkConfigError("settings.include_samples must be boolean.")
    if settings.get("collect_diagnostics", False) is not False:
        raise BenchmarkConfigError(
            "settings.collect_diagnostics must be false for the latency benchmark."
        )
    if settings.get("wandb", False) is not False:
        raise BenchmarkConfigError("settings.wandb must be false for the latency benchmark.")
    if "device" in settings and (
        not isinstance(settings["device"], str) or not settings["device"]
    ):
        raise BenchmarkConfigError("settings.device must be a non-empty string.")
    base_params = algorithm_config["alg_params"]
    if "H" in settings:
        configured_horizon = _positive_int(settings["H"], "settings.H")
        if configured_horizon != int(base_params.get("inner_rollout_horizon", -1)):
            raise BenchmarkConfigError(
                "settings.H must match base inner_rollout_horizon."
            )
        normalized_settings["H"] = configured_horizon
    if "B" in settings:
        configured_batch = _positive_int(settings["B"], "settings.B")
        if configured_batch != int(base_params.get("inner_batch_size", -1)):
            raise BenchmarkConfigError("settings.B must match base inner_batch_size.")
        normalized_settings["B"] = configured_batch
    if "blocks" in settings:
        normalized_settings["blocks"] = _positive_int(
            settings["blocks"], "settings.blocks"
        )

    raw_cells = raw.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise BenchmarkConfigError("cells must be a non-empty JSON list.")
    cells = [_normalize_cell(cell, index) for index, cell in enumerate(raw_cells)]
    triples = [(cell.J, cell.N, cell.G) for cell in cells]
    names = [cell.name for cell in cells]
    if len(set(triples)) != len(triples):
        raise BenchmarkConfigError("cells must not contain duplicate J,N,G triples.")
    if len(set(names)) != len(names):
        raise BenchmarkConfigError("cells must not contain duplicate names.")

    return {
        "path": path,
        "raw": raw,
        "algorithm_config_path": algorithm_config_path,
        "algorithm_config": algorithm_config,
        "environment": copy.deepcopy(environment),
        "settings": normalized_settings,
        "cells": cells,
    }


def _select_cells(
    cells: Sequence[BenchmarkCell],
    selectors: Sequence[tuple[int, int, int]] | None,
) -> list[BenchmarkCell]:
    cells = list(cells)
    if not selectors:
        return cells
    by_triple = {(cell.J, cell.N, cell.G): cell for cell in cells}
    missing = [selector for selector in selectors if selector not in by_triple]
    if missing:
        available = ", ".join(cell.selector for cell in cells)
        requested = ", ".join(",".join(map(str, item)) for item in missing)
        raise BenchmarkConfigError(
            f"Unknown --cell selector(s): {requested}. Available cells: {available}."
        )
    # Preserve CLI order while running a duplicate selector only once.  Cells
    # retain arbitrary JSON annotations, so do not rely on dataclass hashing.
    selected = []
    seen = set()
    for selector in selectors:
        if selector not in seen:
            selected.append(by_triple[selector])
            seen.add(selector)
    return selected


def _resolved_cell_config(spec: Mapping[str, Any], cell: BenchmarkCell) -> dict[str, Any]:
    """Apply only J/N/G and the required action-local replay capacity."""
    algorithm_config = copy.deepcopy(spec["algorithm_config"])
    params = algorithm_config["alg_params"]
    operator = str(params.get("inner_operator", "sac")).lower()
    if operator not in {"sac", "td3"}:
        raise BenchmarkConfigError(
            f"J/N/G latency cells require SAC or TD3, not inner_operator={operator!r}."
        )
    horizon = _positive_int(
        params.get("inner_rollout_horizon"),
        "base algorithm_config.alg_params.inner_rollout_horizon",
    )
    params["inner_rounds"] = cell.J
    params["inner_rollouts_per_round"] = cell.N
    params["inner_updates_per_round"] = cell.G
    params["inner_replay_capacity"] = cell.J * cell.N * horizon
    # W&B is never initialized because learn() is not called.  Keep this hard
    # guard so future algorithm wrappers cannot silently change that contract.
    params["wandb"] = False
    return {
        "selector": f"latency/{cell.name}",
        "comparison": "latency",
        "variant": cell.name,
        "reference": cell.name,
        "description": "Frozen-checkpoint AMBI action-latency cell.",
        "algorithm_config": algorithm_config,
        "environment": copy.deepcopy(spec["environment"]),
        "evaluation": {},
    }


def _copy_observation(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _copy_observation(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_observation(item) for item in value)
    if isinstance(value, list):
        return [_copy_observation(item) for item in value]
    return copy.deepcopy(value)


def _build_observation_bank(env, *, seed: int, size: int) -> list[Any]:
    """Build fixed observations with seeded resets and no environment steps."""
    observations = []
    for offset in range(size):
        observation_seed = (int(seed) + offset) % (_MAX_NUMPY_SEED + 1)
        _seed_spaces(env, observation_seed)
        observation, _ = env.reset(seed=observation_seed)
        observations.append(_copy_observation(observation))
    return observations


def _digest_observation(digest, value):
    if torch.is_tensor(value):
        value = value.detach().contiguous().cpu().numpy()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"array")
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(repr(tuple(array.shape)).encode("utf-8"))
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=repr):
            _digest_observation(digest, key)
            _digest_observation(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("utf-8"))
        for item in value:
            _digest_observation(digest, item)
    else:
        digest.update(type(value).__name__.encode("utf-8"))
        digest.update(repr(value).encode("utf-8"))


def _observation_bank_digest(observations: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for observation in observations:
        _digest_observation(digest, observation)
    return digest.hexdigest()


def _numeric_float(value) -> float | None:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        value = value.detach().float().mean().cpu().item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _metric_subset(metrics: Mapping[str, Any] | None, keys: Sequence[str]):
    result = {}
    for key in keys:
        if key not in (metrics or {}):
            continue
        value = _numeric_float(metrics[key])
        if value is not None:
            result[key] = value
    return result


def _latency_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    if not np.all(np.isfinite(array)):
        raise RuntimeError("Benchmark samples must all be finite.")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _aggregate_group(samples: Sequence[Mapping[str, float]]):
    keys = sorted({key for sample in samples for key in sample})
    return {
        key: _latency_summary([sample[key] for sample in samples if key in sample])
        for key in keys
    }


def _default_synchronizer(device: torch.device) -> Callable[[], None]:
    if device.type != "cuda":
        return lambda: None
    if not torch.cuda.is_available():
        raise RuntimeError(f"Resolved CUDA device {device} but CUDA is unavailable.")

    def synchronize():
        torch.cuda.synchronize(device)

    return synchronize


def _default_memory_callbacks(device: torch.device):
    """Return measured-window memory probes without requiring CUDA in tests."""
    if device.type != "cuda":
        return (
            lambda: {"supported": False, "device_type": device.type},
            lambda: {},
        )

    def reset():
        # The caller has just synchronized the final warmup action.  Capture the
        # steady resident footprint, then reset peaks immediately before the
        # measured loop as required by the benchmark contract.
        baseline = {
            "supported": True,
            "device_type": "cuda",
            "baseline_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "baseline_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        }
        torch.cuda.reset_peak_memory_stats(device)
        return baseline

    def read():
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        current_allocated = int(torch.cuda.memory_allocated(device))
        current_reserved = int(torch.cuda.memory_reserved(device))
        return {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "current_allocated_bytes": current_allocated,
            "current_reserved_bytes": current_reserved,
        }

    return reset, read


def _timed_prediction(
    model,
    observation,
    *,
    deterministic: bool,
    episode_start: bool,
    synchronize: Callable[[], None],
    clock: Callable[[], float],
) -> dict[str, Any]:
    synchronize()
    started = clock()
    prediction = model.predict(
        observation,
        deterministic=deterministic,
        episode_start=episode_start,
        collect_diagnostics=False,
    )
    synchronize()
    wall_seconds = float(clock() - started)
    if not math.isfinite(wall_seconds) or wall_seconds < 0:
        raise RuntimeError(f"Invalid synchronized wall latency: {wall_seconds!r}")
    if not isinstance(prediction, tuple) or not prediction:
        raise RuntimeError("model.predict must return an (action, state) tuple.")
    action = np.asarray(prediction[0])
    expected_shape = getattr(model, "_action_shape", None)
    if action.size == 0 or not np.issubdtype(action.dtype, np.number):
        raise RuntimeError("model.predict returned an empty or non-numeric action.")
    if not np.all(np.isfinite(action)):
        raise RuntimeError("model.predict returned a non-finite action.")
    if expected_shape is not None and tuple(action.shape) != tuple(expected_shape):
        raise RuntimeError(
            "model.predict returned action shape "
            f"{tuple(action.shape)}, expected {tuple(expected_shape)}."
        )
    metrics = getattr(model.agent, "last_inner_metrics", {}) or {}
    return {
        "wall_seconds": wall_seconds,
        "phase_seconds": _metric_subset(metrics, PHASE_METRIC_KEYS),
        "work_counters": _metric_subset(metrics, WORK_COUNTER_KEYS),
        "compile_fallbacks": _metric_subset(metrics, COMPILE_FALLBACK_KEYS),
    }


def _aggregate_measurements(samples, *, include_samples: bool) -> dict[str, Any]:
    wall_values = [sample["wall_seconds"] for sample in samples]
    phases = _aggregate_group([sample["phase_seconds"] for sample in samples])
    work = _aggregate_group([sample["work_counters"] for sample in samples])
    fallbacks = _aggregate_group([sample["compile_fallbacks"] for sample in samples])
    for key, summary in fallbacks.items():
        summary["any"] = bool(summary["max"])
        summary["all"] = bool(summary["min"])
    result = {
        "count": len(samples),
        "wall_seconds": _latency_summary(wall_values),
        "phase_seconds": phases,
        "work_counters": work,
        "compile_fallbacks": fallbacks,
    }
    if include_samples:
        result["samples"] = {
            "wall_seconds": wall_values,
            "phase_seconds": {
                key: [sample["phase_seconds"].get(key) for sample in samples]
                for key in sorted({key for sample in samples for key in sample["phase_seconds"]})
            },
        }
    return result


def _validate_expected_counters(
    expected: Mapping[str, Any], measurements: Mapping[str, Any]
) -> dict[str, Any]:
    """Require each configured work counter on every measured action."""
    if not isinstance(expected, Mapping) or not expected:
        return {
            "passed": False,
            "details": {},
            "errors": ["cell.expected_counters must be a non-empty object"],
        }
    measured_count = int(measurements["count"])
    observed = measurements["work_counters"]
    details = {}
    errors = []
    for expected_key, expected_value in expected.items():
        metric = EXPECTED_COUNTER_METRICS.get(expected_key, expected_key)
        if (
            isinstance(expected_value, bool)
            or not isinstance(expected_value, (int, float))
            or not math.isfinite(float(expected_value))
        ):
            errors.append(f"{expected_key}: expected value is not a finite number")
            details[expected_key] = {
                "metric": metric,
                "expected": expected_value,
                "present": metric in observed,
                "constant": False,
                "matches": False,
            }
            continue
        summary = observed.get(metric)
        if summary is None:
            errors.append(f"{expected_key}: metric {metric!r} is missing")
            details[expected_key] = {
                "metric": metric,
                "expected": expected_value,
                "present": False,
                "constant": False,
                "matches": False,
            }
            continue
        complete = int(summary["count"]) == measured_count
        constant = complete and float(summary["min"]) == float(summary["max"])
        matches = constant and float(summary["min"]) == float(expected_value)
        details[expected_key] = {
            "metric": metric,
            "expected": expected_value,
            "present": True,
            "observed_count": int(summary["count"]),
            "observed_min": summary["min"],
            "observed_max": summary["max"],
            "constant": constant,
            "matches": matches,
        }
        if not matches:
            errors.append(
                f"{expected_key}: expected constant {expected_value}, observed "
                f"count={summary['count']}, min={summary['min']}, max={summary['max']}"
            )
    return {"passed": not errors, "details": details, "errors": errors}


def _validate_compile_fallbacks(measurements: Mapping[str, Any]) -> dict[str, Any]:
    """Require every sticky compile fallback flag to be present and zero."""
    measured_count = int(measurements["count"])
    observed = measurements["compile_fallbacks"]
    details = {}
    errors = []
    for key in COMPILE_FALLBACK_KEYS:
        summary = observed.get(key)
        if summary is None:
            details[key] = {"present": False, "zero_for_all_calls": False}
            errors.append(f"{key}: missing from measured action metrics")
            continue
        complete = int(summary["count"]) == measured_count
        zero = (
            complete
            and float(summary["min"]) == 0.0
            and float(summary["max"]) == 0.0
        )
        details[key] = {
            "present": True,
            "observed_count": int(summary["count"]),
            "observed_min": summary["min"],
            "observed_max": summary["max"],
            "zero_for_all_calls": zero,
        }
        if not zero:
            errors.append(
                f"{key}: expected max=0 across {measured_count} calls, observed "
                f"count={summary['count']}, min={summary['min']}, "
                f"max={summary['max']}"
            )
    return {"passed": not errors, "details": details, "errors": errors}


def benchmark_model(
    model,
    observations: Sequence[Any],
    *,
    warmup_calls: int,
    measured_calls: int,
    action_mode: str,
    include_samples: bool = True,
    synchronize: Callable[[], None] | None = None,
    reset_memory: Callable[[], Mapping[str, Any]] | None = None,
    read_memory: Callable[[], Mapping[str, Any]] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Benchmark a model; dependency injection keeps CPU fake tests cheap."""
    if not observations:
        raise ValueError("observations must contain at least one fixed observation.")
    warmup_calls = _nonnegative_int(warmup_calls, "warmup_calls")
    measured_calls = _positive_int(measured_calls, "measured_calls")
    if action_mode not in {"training", "evaluation"}:
        raise ValueError("action_mode must be 'training' or 'evaluation'.")
    resolved_device = torch.device(model.agent.device)
    if synchronize is None:
        synchronize = _default_synchronizer(resolved_device)
    if reset_memory is None or read_memory is None:
        default_reset, default_read = _default_memory_callbacks(resolved_device)
        reset_memory = default_reset if reset_memory is None else reset_memory
        read_memory = default_read if read_memory is None else read_memory
    deterministic = action_mode == "evaluation"
    call_index = 0

    def run_one():
        nonlocal call_index
        sample = _timed_prediction(
            model,
            observations[call_index % len(observations)],
            deterministic=deterministic,
            episode_start=(call_index == 0),
            synchronize=synchronize,
            clock=clock,
        )
        call_index += 1
        return sample

    cold_call = run_one()
    for _ in range(warmup_calls):
        run_one()
    # Every timed prediction synchronizes on return, so this reset sits exactly
    # between the warmup and measured windows.
    memory = dict(reset_memory())
    samples = [run_one() for _ in range(measured_calls)]
    memory.update(read_memory())
    if memory.get("supported"):
        memory["peak_additional_allocated_bytes"] = max(
            0,
            int(memory["peak_allocated_bytes"])
            - int(memory["baseline_allocated_bytes"]),
        )
        memory["peak_additional_reserved_bytes"] = max(
            0,
            int(memory["peak_reserved_bytes"])
            - int(memory["baseline_reserved_bytes"]),
        )
    measurements = _aggregate_measurements(samples, include_samples=include_samples)
    measurements["device_memory"] = memory
    return {
        "cold_call": cold_call,
        "warmup_calls": warmup_calls,
        "measurements": measurements,
        "total_action_calls": 1 + warmup_calls + measured_calls,
    }


def _resolved_work(model, cell: BenchmarkCell) -> dict[str, Any]:
    cfg = model.cfg
    J = int(cfg.inner_rounds)
    N = int(cfg.inner_rollouts_per_round)
    H = int(cfg.inner_rollout_horizon)
    G = int(cfg.inner_updates_per_round)
    if (J, N, G) != (cell.J, cell.N, cell.G):
        raise RuntimeError(
            "Resolved model schedule does not match requested cell: "
            f"requested={(cell.J, cell.N, cell.G)}, resolved={(J, N, G)}."
        )
    expected_capacity = J * N * H
    if int(cfg.inner_replay_capacity) != expected_capacity:
        raise RuntimeError(
            "Resolved replay capacity does not equal J*N*H: "
            f"{cfg.inner_replay_capacity} != {expected_capacity}."
        )
    batch_size = int(cfg.inner_batch_size)
    return {
        "rollout_paths": J * N,
        "imagined_transitions": expected_capacity,
        "update_slots": J * G,
        "inner_batch_size": batch_size,
        "replay_capacity": int(cfg.inner_replay_capacity),
        "nominal_replay_draws": J * G * batch_size,
    }


def _run_cell(
    spec,
    cell: BenchmarkCell,
    checkpoint: Path,
    *,
    settings: Mapping[str, Any],
    device: str | None,
) -> dict[str, Any]:
    resolved = _resolved_cell_config(spec, cell)
    env = _make_env(resolved)
    model = None
    try:
        observations = _build_observation_bank(
            env,
            seed=settings["environment_seed"],
            size=settings["observation_bank_size"],
        )
        observation_digest = _observation_bank_digest(observations)
        model, run_config = _initialize_frozen_model(
            resolved,
            env,
            checkpoint,
            settings["controller_seed"],
            device=device,
        )
        digest_before = _outer_state_digest(model)
        updates_before = int(model.agent.num_updates)
        resolved_device = torch.device(model.agent.device)
        timings = benchmark_model(
            model,
            observations,
            warmup_calls=settings["warmup_calls"],
            measured_calls=settings["measured_calls"],
            action_mode=settings["action_mode"],
            include_samples=settings["include_samples"],
            synchronize=_default_synchronizer(resolved_device),
        )
        digest_after = _outer_state_digest(model)
        updates_after = int(model.agent.num_updates)
        if digest_before != digest_after or updates_before != updates_after:
            raise RuntimeError(
                f"Frozen benchmark invariant failed for {cell.name}: outer state changed."
            )
        counter_validation = _validate_expected_counters(
            cell.source.get("expected_counters"), timings["measurements"]
        )
        compile_validation = _validate_compile_fallbacks(timings["measurements"])
        validation = {
            "passed": bool(
                counter_validation["passed"] and compile_validation["passed"]
            ),
            "expected_counters": counter_validation,
            "compile_fallbacks": compile_validation,
        }
        source = copy.deepcopy(dict(cell.source))
        source.update(
            name=cell.name,
            J=int(model.cfg.inner_rounds),
            N=int(model.cfg.inner_rollouts_per_round),
            H=int(model.cfg.inner_rollout_horizon),
            G=int(model.cfg.inner_updates_per_round),
            resolved_work=_resolved_work(model, cell),
            observation_bank_sha256=observation_digest,
            resolved_device=str(resolved_device),
            compile_enabled=bool(model.cfg.compile),
            collect_diagnostics=False,
            outer_updates_before=updates_before,
            outer_updates_after=updates_after,
            outer_state_unchanged=True,
            validation=validation,
            run_alg_params=copy.deepcopy(run_config["alg_params"]),
        )
        source.update(timings)
        return source
    finally:
        env.close()


def _git_metadata(repo: Path) -> dict[str, Any]:
    def run(*args):
        completed = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": None if status is None else bool(status),
        "status_porcelain": None if not status else status.splitlines(),
    }


def _hardware_metadata(device: str | None) -> dict[str, Any]:
    metadata = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "requested_device": device,
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata["cuda_device"] = {
            "index": index,
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "capability": [int(properties.major), int(properties.minor)],
        }
    else:
        metadata["cuda_device"] = None
    return metadata


def _checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    stat = checkpoint.stat()
    digest = hashlib.sha256()
    with checkpoint.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    stat_after = checkpoint.stat()
    if (stat.st_size, stat.st_mtime_ns) != (
        stat_after.st_size,
        stat_after.st_mtime_ns,
    ):
        raise RuntimeError(f"Checkpoint changed while hashing: {checkpoint}")
    return {
        "path": str(checkpoint),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark frozen-checkpoint AMBI action latency for explicit J/N/G cells."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--list-cells",
        action="store_true",
        help="Print headerless TSV rows (name, J, N, G) without loading a checkpoint.",
    )
    parser.add_argument(
        "--cell",
        action="append",
        type=_parse_cell,
        help="Select a configured J,N,G cell. Repeat to run several; defaults to all.",
    )
    parser.add_argument("--device", help="Override the base config device.")
    parser.add_argument("--warmup-calls", type=int)
    parser.add_argument("--measured-calls", type=int)
    parser.add_argument("--observation-bank-size", type=int)
    parser.add_argument("--environment-seed", type=int)
    parser.add_argument("--controller-seed", type=int)
    parser.add_argument(
        "--action-mode", choices=("training", "evaluation")
    )
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Omit raw timing samples while retaining all requested summaries.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def _settings_with_overrides(spec, args) -> dict[str, Any]:
    settings = copy.deepcopy(spec["settings"])
    for key in (
        "warmup_calls",
        "measured_calls",
        "observation_bank_size",
        "environment_seed",
        "controller_seed",
        "action_mode",
    ):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    settings["warmup_calls"] = _nonnegative_int(
        settings["warmup_calls"], "warmup_calls"
    )
    settings["measured_calls"] = _positive_int(
        settings["measured_calls"], "measured_calls"
    )
    settings["observation_bank_size"] = _positive_int(
        settings["observation_bank_size"], "observation_bank_size"
    )
    settings["environment_seed"] = _seed(
        settings["environment_seed"], "environment_seed"
    )
    settings["controller_seed"] = _seed(
        settings["controller_seed"], "controller_seed"
    )
    if args.no_samples:
        settings["include_samples"] = False
    return settings


def run_benchmark(args) -> dict[str, Any]:
    spec = load_benchmark_config(args.config)
    if args.checkpoint is None:
        raise BenchmarkConfigError("--checkpoint is required unless --list-cells is used.")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    settings = _settings_with_overrides(spec, args)
    cells = _select_cells(spec["cells"], args.cell)
    effective_device = args.device or settings.get("device")
    repo = Path(__file__).resolve().parent
    document = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git": _git_metadata(repo),
            "hardware": _hardware_metadata(effective_device),
            "checkpoint": _checkpoint_metadata(checkpoint),
            "config_path": str(spec["path"]),
            "algorithm_config_path": str(spec["algorithm_config_path"]),
            "process_id": os.getpid(),
            "benchmark_config_metadata": copy.deepcopy(
                spec["raw"].get("metadata", {})
            ),
        },
        "settings": {
            **settings,
            "device_override": args.device,
            "effective_device": effective_device,
            "selected_cells": [cell.selector for cell in cells],
            "fresh_model_per_cell": True,
            "process_isolation": len(cells) == 1,
            "collect_diagnostics": False,
            "environment_reset_calls_per_cell": settings["observation_bank_size"],
            "environment_step_calls": 0,
            "outer_update_calls": 0,
            "wandb_enabled": False,
            "timing_contract": {
                "wall_seconds": (
                    "time.perf_counter around public predict with device synchronization "
                    "immediately before and after"
                ),
                "phase_seconds": (
                    "existing AMBI inner phase metrics (CUDA events on CUDA, "
                    "perf_counter on CPU), finalized at the action boundary"
                ),
                "cold_call": "first action from a fresh checkpoint-loaded model",
            },
        },
        "cells": [],
    }
    expected_observation_digest = None
    for cell in cells:
        result = _run_cell(
            spec,
            cell,
            checkpoint,
            settings=settings,
            device=effective_device,
        )
        digest = result["observation_bank_sha256"]
        if expected_observation_digest is None:
            expected_observation_digest = digest
        elif digest != expected_observation_digest:
            raise RuntimeError(
                "Seeded observation bank differed between J/N/G cells."
            )
        document["cells"].append(result)
    document["settings"]["observation_bank_sha256"] = expected_observation_digest
    document["counter_formulas"] = copy.deepcopy(
        spec["raw"].get("counter_formulas", {})
    )
    failed_cells = [
        cell["name"] for cell in document["cells"] if not cell["validation"]["passed"]
    ]
    document["validation"] = {
        "passed": not failed_cells,
        "failed_cells": failed_cells,
    }
    return document


def _strict_json(document: Mapping[str, Any]) -> str:
    return json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_cells:
        spec = load_benchmark_config(args.config)
        for cell in _select_cells(spec["cells"], args.cell):
            print(f"{cell.name}\t{cell.J}\t{cell.N}\t{cell.G}")
        return 0
    document = run_benchmark(args)
    rendered = _strict_json(document)
    if args.output is None:
        print(rendered, end="")
    else:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"Wrote AMBI latency benchmark: {output}")
    if not document["validation"]["passed"]:
        failed = ", ".join(document["validation"]["failed_cells"])
        print(
            f"AMBI latency validation failed for cell(s): {failed}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
