#!/usr/bin/env python3
"""Validate, aggregate, and plot AMBI inner-loop latency benchmarks.

The benchmark contract deliberately stores summaries for many calls from one
process in each JSON file. Each file contributes one process observation tagged
by its ``block_N`` parent; action-level counts are never expanded into fake
statistical replicates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BENCHMARK = "ambi-inner-latency"
SCHEMA_VERSION = 1
STAT_FIELDS = ("count", "mean", "std", "p50", "p90", "p95", "min", "max")
PROTOCOL_SETTINGS = (
    "device",
    "effective_device",
    "device_override",
    "cold_calls",
    "warmup_calls",
    "measured_calls",
    "observation_bank_size",
    "environment_seed",
    "controller_seed",
    "action_mode",
    "include_samples",
    "H",
    "B",
    "blocks",
    "block_order_seed",
    "fresh_model_per_cell",
    "process_isolation",
    "collect_diagnostics",
    "wandb",
    "environment_reset_calls_per_cell",
    "environment_step_calls",
    "outer_update_calls",
    "wandb_enabled",
    "timing_contract",
    "observation_bank_sha256",
)
EXPECTED_DESIGN = (
    ("center_j2_n32_g4", 2, 32, 3, 4),
    ("g_sweep_g0", 2, 32, 3, 0),
    ("g_sweep_g2", 2, 32, 3, 2),
    ("g_sweep_g8", 2, 32, 3, 8),
    ("g_sweep_g16", 2, 32, 3, 16),
    ("n_sweep_n8", 2, 8, 3, 4),
    ("n_sweep_n16", 2, 16, 3, 4),
    ("n_sweep_n64", 2, 64, 3, 4),
    ("n_sweep_n128", 2, 128, 3, 4),
    ("natural_j1", 1, 32, 3, 4),
    ("natural_j4", 4, 32, 3, 4),
    ("natural_j8", 8, 32, 3, 4),
    ("matched_j1_n64_g8", 1, 64, 3, 8),
    ("matched_j4_n16_g2", 4, 16, 3, 2),
    ("matched_j8_n8_g1", 8, 8, 3, 1),
)
PHASE_ORDER = (
    "inner_setup_seconds",
    "inner_rollout_seconds",
    "inner_update_seconds",
    "inner_execution_seconds",
    "inner_mppi_seconds",
)
TOTAL_PHASE_NAMES = {
    "inner_action_seconds",
    "action_seconds",
    "wall_seconds",
    "total_seconds",
}


class ValidationError(ValueError):
    """Raised when benchmark inputs are unsafe to compare."""


@dataclass(frozen=True, order=True)
class CellKey:
    J: int
    N: int
    H: int
    G: int

    @property
    def rollout_paths(self) -> int:
        return self.J * self.N

    @property
    def imagined_transitions(self) -> int:
        return self.J * self.N * self.H

    @property
    def update_slots(self) -> int:
        return self.J * self.G


@dataclass(frozen=True)
class MetricStats:
    count: int
    mean: float
    std: float
    p50: float
    p90: float
    p95: float
    min: float
    max: float


@dataclass(frozen=True)
class BlockCell:
    source: Path
    block_id: int
    process_identity: tuple[str, int, str]
    name: str
    key: CellKey
    inner_batch_size: int
    replay_capacity: int
    measurement_count: int
    wall: MetricStats
    phases: Mapping[str, MetricStats]


@dataclass(frozen=True)
class Aggregate:
    center: float
    low: float
    high: float
    values: tuple[float, ...]


@dataclass(frozen=True)
class GroupSummary:
    name: str
    key: CellKey
    blocks: int
    samples_min: int
    samples_max: int
    wall: Mapping[str, Aggregate]
    phases_mean: Mapping[str, Aggregate]
    unattributed_mean: Aggregate
    phase_stack_seconds: Mapping[str, float]
    phase_representative_block: int


@dataclass(frozen=True)
class Dataset:
    records: tuple[BlockCell, ...]
    summaries: Mapping[CellKey, GroupSummary]
    reference: CellKey
    views: Mapping[str, tuple[CellKey, ...]]
    metadata: Mapping[str, Any]
    settings: Mapping[str, Any]


def _duplicate_guard(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON number {value!r} is not allowed")


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_duplicate_guard,
                parse_constant=_reject_constant,
            )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{path}: cannot read valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{path}: top level must be a JSON object")
    return value


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{where} must be an object")
    return value


def _exact_int(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{where} must be an integer")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValidationError(f"{where} must be a finite integer")
    result = int(number)
    if result < minimum:
        raise ValidationError(f"{where} must be >= {minimum}")
    return result


def _finite_float(value: Any, where: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{where} must be finite")
    if minimum is not None and result < minimum:
        raise ValidationError(f"{where} must be >= {minimum}")
    return result


def _metric_stats(value: Any, where: str, *, expected_count: int) -> MetricStats:
    obj = _mapping(value, where)
    missing = [field for field in STAT_FIELDS if field not in obj]
    if missing:
        raise ValidationError(f"{where} is missing summary fields: {', '.join(missing)}")
    count = _exact_int(obj["count"], f"{where}.count", minimum=1)
    if count != expected_count:
        raise ValidationError(
            f"{where}.count={count} disagrees with measurements.count={expected_count}"
        )
    values = {
        field: _finite_float(obj[field], f"{where}.{field}", minimum=0.0)
        for field in STAT_FIELDS
        if field != "count"
    }
    if values["std"] < 0:
        raise ValidationError(f"{where}.std must be nonnegative")
    ordered = [
        values["min"],
        values["p50"],
        values["p90"],
        values["p95"],
        values["max"],
    ]
    tolerance = max(1e-12, values["max"] * 1e-9)
    if any(left > right + tolerance for left, right in zip(ordered, ordered[1:])):
        raise ValidationError(f"{where} has inconsistent min/quantile/max ordering")
    if not values["min"] - tolerance <= values["mean"] <= values["max"] + tolerance:
        raise ValidationError(f"{where}.mean must lie between min and max")
    return MetricStats(count=count, **values)


def _validate_numeric_tree(value: Any, where: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_numeric_tree(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_numeric_tree(child, f"{where}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _finite_float(value, where)
    elif value is not None and not isinstance(value, (str, bool)):
        raise ValidationError(f"{where} contains unsupported value {type(value).__name__}")


def _fallback_peak(value: Any, where: str) -> float:
    if isinstance(value, dict):
        # Summary count is the number of observations, not the number of fallbacks.
        for field in ("max", "p95", "mean", "value"):
            if field in value:
                return _finite_float(value[field], f"{where}.{field}", minimum=0.0)
        peaks = [
            _fallback_peak(child, f"{where}.{key}")
            for key, child in value.items()
            if key != "count"
        ]
        return max(peaks, default=0.0)
    if value is None:
        return 0.0
    return _finite_float(value, where, minimum=0.0)


def _validate_fallbacks(value: Any, where: str) -> None:
    obj = _mapping(value, where)
    for key, summary in obj.items():
        peak = _fallback_peak(summary, f"{where}.{key}")
        if peak > 0:
            raise ValidationError(
                f"{where}.{key} reports a compile fallback ({peak:g}); "
                "fallback and compiled paths must not be pooled"
            )


def _resolved_int(
    resolved: Mapping[str, Any], key: str, expected: int, where: str
) -> int:
    if key not in resolved:
        raise ValidationError(f"{where} is missing {key!r}")
    actual = _exact_int(resolved[key], f"{where}.{key}", minimum=0)
    if actual != expected:
        raise ValidationError(
            f"{where}.{key}={actual} but J/N/H/G imply {expected}"
        )
    return actual


def _block_index(path: Path) -> int:
    matches = []
    for parent in path.resolve().parents:
        match = re.fullmatch(r"block_(\d+)", parent.name)
        if match:
            matches.append(int(match.group(1)))
    if len(matches) != 1:
        raise ValidationError(
            f"{path}: expected exactly one block_N parent directory, found {len(matches)}"
        )
    return matches[0]


def _process_identity(metadata: Mapping[str, Any], path: Path) -> tuple[str, int, str]:
    hardware = _mapping(metadata.get("hardware"), f"{path}: metadata.hardware")
    hostname = hardware.get("hostname")
    timestamp = metadata.get("timestamp_utc")
    if not isinstance(hostname, str) or not hostname:
        raise ValidationError(f"{path}: metadata.hardware.hostname is required")
    if not isinstance(timestamp, str) or not timestamp:
        raise ValidationError(f"{path}: metadata.timestamp_utc is required")
    process_id = _exact_int(
        metadata.get("process_id"), f"{path}: metadata.process_id", minimum=1
    )
    return hostname, process_id, timestamp


def _cell_from_json(
    path: Path,
    index: int,
    value: Any,
    *,
    block_id: int,
    process_identity: tuple[str, int, str],
) -> BlockCell:
    where = f"{path}: cells[{index}]"
    cell = _mapping(value, where)
    name = cell.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValidationError(f"{where}.name must be a non-empty string")
    key = CellKey(
        J=_exact_int(cell.get("J"), f"{where}.J", minimum=1),
        N=_exact_int(cell.get("N"), f"{where}.N", minimum=1),
        H=_exact_int(cell.get("H"), f"{where}.H", minimum=1),
        G=_exact_int(cell.get("G"), f"{where}.G", minimum=0),
    )
    resolved = _mapping(cell.get("resolved_work"), f"{where}.resolved_work")
    _resolved_int(resolved, "rollout_paths", key.rollout_paths, f"{where}.resolved_work")
    _resolved_int(
        resolved,
        "imagined_transitions",
        key.imagined_transitions,
        f"{where}.resolved_work",
    )
    _resolved_int(resolved, "update_slots", key.update_slots, f"{where}.resolved_work")
    inner_batch_size = _exact_int(
        resolved.get("inner_batch_size"),
        f"{where}.resolved_work.inner_batch_size",
        minimum=1,
    )
    replay_capacity = _exact_int(
        resolved.get("replay_capacity"),
        f"{where}.resolved_work.replay_capacity",
        minimum=1,
    )

    cold = _mapping(cell.get("cold_call"), f"{where}.cold_call")
    _finite_float(cold.get("wall_seconds"), f"{where}.cold_call.wall_seconds", minimum=0)
    _validate_numeric_tree(
        _mapping(cold.get("phase_seconds"), f"{where}.cold_call.phase_seconds"),
        f"{where}.cold_call.phase_seconds",
    )
    _validate_numeric_tree(
        _mapping(cold.get("work_counters"), f"{where}.cold_call.work_counters"),
        f"{where}.cold_call.work_counters",
    )
    # A fresh process can discover an unsupported compiled path on its cold
    # call. That is useful provenance, but it must not contaminate the measured
    # steady-state calls checked below.
    _validate_numeric_tree(
        _mapping(
            cold.get("compile_fallbacks", {}),
            f"{where}.cold_call.compile_fallbacks",
        ),
        f"{where}.cold_call.compile_fallbacks",
    )

    if "warmup_calls" not in cell:
        raise ValidationError(f"{where} is missing warmup_calls")
    _validate_numeric_tree(cell["warmup_calls"], f"{where}.warmup_calls")

    measurements = _mapping(cell.get("measurements"), f"{where}.measurements")
    count = _exact_int(
        measurements.get("count"), f"{where}.measurements.count", minimum=1
    )
    wall = _metric_stats(
        measurements.get("wall_seconds"),
        f"{where}.measurements.wall_seconds",
        expected_count=count,
    )
    phase_values = _mapping(
        measurements.get("phase_seconds"), f"{where}.measurements.phase_seconds"
    )
    phases = {
        phase: _metric_stats(
            summary,
            f"{where}.measurements.phase_seconds.{phase}",
            expected_count=count,
        )
        for phase, summary in phase_values.items()
    }
    if not phases:
        raise ValidationError(f"{where}.measurements.phase_seconds must not be empty")
    _validate_numeric_tree(
        _mapping(
            measurements.get("work_counters"),
            f"{where}.measurements.work_counters",
        ),
        f"{where}.measurements.work_counters",
    )
    _validate_fallbacks(
        measurements.get("compile_fallbacks", {}),
        f"{where}.measurements.compile_fallbacks",
    )
    validation_obj = _mapping(cell.get("validation"), f"{where}.validation")
    if validation_obj.get("passed") is not True:
        raise ValidationError(
            f"{where}.validation.passed must be true; runner work/fallback "
            "validation failed"
        )
    if cell.get("outer_state_unchanged") is not True:
        raise ValidationError(f"{where}.outer_state_unchanged must be true")

    component_mean = sum(
        stats.mean for phase, stats in phases.items() if phase not in TOTAL_PHASE_NAMES
    )
    if component_mean > wall.mean * 1.05 + 1e-9:
        raise ValidationError(
            f"{where}: component phase means ({component_mean:g}s) exceed wall mean "
            f"({wall.mean:g}s) by more than 5%; check overlapping phase metrics"
        )

    return BlockCell(
        source=path,
        block_id=block_id,
        process_identity=process_identity,
        name=name.strip(),
        key=key,
        inner_batch_size=inner_batch_size,
        replay_capacity=replay_capacity,
        measurement_count=count,
        wall=wall,
        phases=phases,
    )


def _gpu_identity(hardware: Any) -> Any:
    """Extract stable GPU model information while ignoring host names/UUIDs."""
    if isinstance(hardware, dict):
        cuda_device = hardware.get("cuda_device")
        if isinstance(cuda_device, dict):
            return {
                key: cuda_device[key]
                for key in ("name", "total_memory_bytes", "capability")
                if key in cuda_device
            }
        direct = []
        for key, value in sorted(hardware.items()):
            normalized = key.lower().replace("-", "_")
            if "gpu" in normalized and any(
                token in normalized for token in ("name", "model", "type")
            ):
                direct.append((normalized, value))
        if direct:
            return direct
        for key in ("gpu", "gpus", "cuda"):
            if key in hardware:
                nested = _gpu_identity(hardware[key])
                if nested is not None:
                    return nested
        for value in hardware.values():
            nested = _gpu_identity(value)
            if nested is not None:
                return nested
    elif isinstance(hardware, list):
        identities = [identity for item in hardware if (identity := _gpu_identity(item))]
        return identities or None
    return None


def _comparison_signature(document: Mapping[str, Any], path: Path) -> dict[str, Any]:
    metadata = _mapping(document.get("metadata"), f"{path}: metadata")
    settings = _mapping(document.get("settings"), f"{path}: settings")
    if "git" not in metadata:
        raise ValidationError(f"{path}: metadata.git is required")
    if "checkpoint" not in metadata:
        raise ValidationError(f"{path}: metadata.checkpoint is required")
    missing_settings = [key for key in PROTOCOL_SETTINGS if key not in settings]
    if missing_settings:
        raise ValidationError(
            f"{path}: settings missing comparison fields: {', '.join(missing_settings)}"
        )
    git = _mapping(metadata["git"], f"{path}: metadata.git")
    if not git.get("commit"):
        raise ValidationError(f"{path}: metadata.git.commit is required")
    if git.get("dirty") is not False:
        raise ValidationError(f"{path}: metadata.git.dirty must be false")
    hardware = _mapping(metadata.get("hardware"), f"{path}: metadata.hardware")
    runtime_fields = (
        "python",
        "torch",
        "cuda_available",
        "cuda_version",
        "cudnn_version",
        "requested_device",
    )
    missing_runtime = [key for key in runtime_fields if key not in hardware]
    if missing_runtime:
        raise ValidationError(
            f"{path}: metadata.hardware missing runtime fields: "
            f"{', '.join(missing_runtime)}"
        )
    gpu = _gpu_identity(hardware)
    if hardware["cuda_available"] is not True or gpu is None:
        raise ValidationError(f"{path}: benchmark hardware must resolve one CUDA GPU")
    for key in ("device", "effective_device", "device_override"):
        if settings[key] != "cuda":
            raise ValidationError(f"{path}: settings.{key} must be 'cuda'")
    if hardware["requested_device"] != settings["effective_device"]:
        raise ValidationError(
            f"{path}: hardware requested_device disagrees with effective_device"
        )
    if settings["action_mode"] != "training":
        raise ValidationError(f"{path}: settings.action_mode must be 'training'")
    if settings["process_isolation"] is not True:
        raise ValidationError(f"{path}: settings.process_isolation must be true")
    if settings["fresh_model_per_cell"] is not True:
        raise ValidationError(f"{path}: settings.fresh_model_per_cell must be true")
    for key in ("collect_diagnostics", "wandb", "wandb_enabled"):
        if settings[key] is not False:
            raise ValidationError(f"{path}: settings.{key} must be false")
    for key in ("environment_step_calls", "outer_update_calls"):
        if _exact_int(settings[key], f"{path}: settings.{key}") != 0:
            raise ValidationError(f"{path}: settings.{key} must be zero")
    if _exact_int(settings["cold_calls"], f"{path}: settings.cold_calls", minimum=1) != 1:
        raise ValidationError(f"{path}: settings.cold_calls must be one")
    for key in ("H", "B", "blocks"):
        _exact_int(settings[key], f"{path}: settings.{key}", minimum=1)
    if _exact_int(
        settings["environment_reset_calls_per_cell"],
        f"{path}: settings.environment_reset_calls_per_cell",
        minimum=1,
    ) != _exact_int(
        settings["observation_bank_size"],
        f"{path}: settings.observation_bank_size",
        minimum=1,
    ):
        raise ValidationError(
            f"{path}: environment_reset_calls_per_cell must equal observation_bank_size"
        )
    benchmark_metadata = _mapping(
        metadata.get("benchmark_config_metadata"),
        f"{path}: metadata.benchmark_config_metadata",
    )
    counter_formulas = _mapping(
        document.get("counter_formulas"), f"{path}: counter_formulas"
    )
    if not counter_formulas:
        raise ValidationError(f"{path}: counter_formulas must not be empty")
    return {
        "git": {key: git.get(key) for key in ("commit", "branch")},
        "checkpoint": metadata["checkpoint"],
        "config_path": metadata.get("config_path", settings.get("config_path")),
        "algorithm_config_path": metadata.get("algorithm_config_path"),
        "benchmark_config_metadata": benchmark_metadata,
        "counter_formulas": counter_formulas,
        "hardware": {
            **{key: hardware[key] for key in runtime_fields},
            "gpu": gpu,
        },
        "settings": {key: settings[key] for key in PROTOCOL_SETTINGS},
    }


def load_blocks(paths: Sequence[Path | str]) -> tuple[tuple[BlockCell, ...], dict[str, Any]]:
    """Load canonical single-process JSON files and validate the block matrix."""
    if not paths:
        raise ValidationError("at least one input JSON file is required")
    records: list[BlockCell] = []
    baseline_signature: dict[str, Any] | None = None
    baseline_path: Path | None = None
    first_document: Mapping[str, Any] | None = None
    for raw_path in paths:
        path = Path(raw_path)
        document = _load_json(path)
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ValidationError(
                f"{path}: schema_version must be {SCHEMA_VERSION}, got "
                f"{document.get('schema_version')!r}"
            )
        if document.get("benchmark") != BENCHMARK:
            raise ValidationError(
                f"{path}: benchmark must be {BENCHMARK!r}, got "
                f"{document.get('benchmark')!r}"
            )
        document_validation = _mapping(
            document.get("validation"), f"{path}: validation"
        )
        if document_validation.get("passed") is not True:
            raise ValidationError(f"{path}: validation.passed must be true")
        signature = _comparison_signature(document, path)
        if baseline_signature is None:
            baseline_signature = signature
            baseline_path = path
            first_document = document
        elif signature != baseline_signature:
            different = [
                key for key in signature if signature[key] != baseline_signature[key]
            ]
            raise ValidationError(
                f"{path}: comparison metadata/settings differ from {baseline_path}: "
                f"{', '.join(different)}"
            )
        cells = document.get("cells")
        if not isinstance(cells, list) or len(cells) != 1:
            raise ValidationError(
                f"{path}: cells must contain exactly one process-isolated cell"
            )
        metadata = _mapping(document.get("metadata"), f"{path}: metadata")
        block_id = _block_index(path)
        block_cells = [
            _cell_from_json(
                path,
                0,
                cells[0],
                block_id=block_id,
                process_identity=_process_identity(metadata, path),
            )
        ]
        measured_calls = _exact_int(
            document["settings"]["measured_calls"],
            f"{path}: settings.measured_calls",
            minimum=1,
        )
        for cell in block_cells:
            if path.stem != cell.name:
                raise ValidationError(
                    f"{path}: file stem must equal cell name {cell.name!r}"
                )
            resolved_device = cells[0].get("resolved_device")
            hardware = _mapping(metadata.get("hardware"), f"{path}: metadata.hardware")
            cuda_device = _mapping(
                hardware.get("cuda_device"), f"{path}: metadata.hardware.cuda_device"
            )
            cuda_index = _exact_int(
                cuda_device.get("index"),
                f"{path}: metadata.hardware.cuda_device.index",
            )
            if resolved_device != f"cuda:{cuda_index}":
                raise ValidationError(
                    f"{path}: cell resolved_device must match hardware CUDA index"
                )
            if cells[0].get("compile_enabled") is not True:
                raise ValidationError(f"{path}: cell compile_enabled must be true")
            if cell.measurement_count != measured_calls:
                raise ValidationError(
                    f"{path}: {cell.name} has {cell.measurement_count} measurements, "
                    f"expected settings.measured_calls={measured_calls}"
                )
            settings = document["settings"]
            if cell.key.H != _exact_int(
                settings["H"], f"{path}: settings.H", minimum=1
            ):
                raise ValidationError(f"{path}: cell H disagrees with settings.H")
            if cell.inner_batch_size != _exact_int(
                settings["B"], f"{path}: settings.B", minimum=1
            ):
                raise ValidationError(
                    f"{path}: resolved inner_batch_size disagrees with settings.B"
                )
            selected = settings.get("selected_cells")
            expected_selector = f"{cell.key.J},{cell.key.N},{cell.key.G}"
            if selected != [expected_selector]:
                raise ValidationError(
                    f"{path}: settings.selected_cells must equal [{expected_selector!r}]"
                )
        records.extend(block_cells)
    assert first_document is not None
    expected_block_count = _exact_int(
        first_document["settings"]["blocks"],
        f"{baseline_path}: settings.blocks",
        minimum=1,
    )
    actual_block_ids = {record.block_id for record in records}
    expected_block_ids = set(range(expected_block_count))
    if actual_block_ids != expected_block_ids:
        raise ValidationError(
            "Oscar block identities do not match settings.blocks: "
            f"expected {sorted(expected_block_ids)}, got {sorted(actual_block_ids)}"
        )
    identities: dict[tuple[str, int, str], Path] = {}
    for record in records:
        previous = identities.get(record.process_identity)
        if previous is not None:
            raise ValidationError(
                f"{record.source}: duplicate process artifact identity also used by {previous}"
            )
        identities[record.process_identity] = record.source
    expected_design = {
        (name, CellKey(J=J, N=N, H=H, G=G))
        for name, J, N, H, G in EXPECTED_DESIGN
    }
    by_block: dict[int, list[BlockCell]] = defaultdict(list)
    for record in records:
        by_block[record.block_id].append(record)
    for block_id in sorted(expected_block_ids):
        block_records = by_block[block_id]
        actual_design = {(record.name, record.key) for record in block_records}
        if len(block_records) != len(expected_design) or actual_design != expected_design:
            missing = sorted(name for name, key in expected_design - actual_design)
            unexpected = sorted(name for name, key in actual_design - expected_design)
            raise ValidationError(
                f"block_{block_id} does not contain the exact 15-cell design once; "
                f"missing={missing}, unexpected={unexpected}, files={len(block_records)}"
            )
    context = {
        "metadata": first_document["metadata"],
        "settings": first_document["settings"],
    }
    return tuple(records), context


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile needs at least one value")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    )


def _aggregate(
    values: Iterable[float], *, bootstrap_samples: int, bootstrap_seed: int, tag: str
) -> Aggregate:
    observations = tuple(float(value) for value in values)
    if not observations:
        raise ValidationError(f"cannot aggregate an empty metric ({tag})")
    center = float(statistics.median(observations))
    if len(observations) == 1 or bootstrap_samples == 0:
        return Aggregate(center, center, center, observations)
    digest = hashlib.blake2b(
        f"{bootstrap_seed}:{tag}".encode("utf-8"), digest_size=8
    ).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    replicate_medians = sorted(
        statistics.median(rng.choices(observations, k=len(observations)))
        for _ in range(bootstrap_samples)
    )
    return Aggregate(
        center=center,
        low=_percentile(replicate_medians, 0.025),
        high=_percentile(replicate_medians, 0.975),
        values=observations,
    )


def summarize_blocks(
    records: Sequence[BlockCell], *, bootstrap_samples: int = 5000, bootstrap_seed: int = 0
) -> dict[CellKey, GroupSummary]:
    if bootstrap_samples < 0:
        raise ValidationError("bootstrap_samples must be >= 0")
    grouped: dict[CellKey, list[BlockCell]] = defaultdict(list)
    for record in records:
        grouped[record.key].append(record)
    summaries: dict[CellKey, GroupSummary] = {}
    for key, group in grouped.items():
        names = {record.name for record in group}
        if len(names) != 1:
            raise ValidationError(
                f"condition {key} has inconsistent names across blocks: {sorted(names)}"
            )
        block_ids = {record.block_id for record in group}
        if len(block_ids) != len(group):
            raise ValidationError(f"condition {key} appears twice in one input block")
        batches = {record.inner_batch_size for record in group}
        capacities = {record.replay_capacity for record in group}
        if len(batches) != 1 or len(capacities) != 1:
            raise ValidationError(
                f"condition {key} has inconsistent batch/replay settings across blocks"
            )
        phase_sets = [set(record.phases) for record in group]
        if any(phases != phase_sets[0] for phases in phase_sets[1:]):
            raise ValidationError(f"condition {key} has inconsistent phase metrics across blocks")
        wall = {
            metric: _aggregate(
                (getattr(record.wall, metric) for record in group),
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                tag=f"{key}:wall:{metric}",
            )
            for metric in ("mean", "p50", "p90", "p95")
        }
        component_phases = [
            phase for phase in phase_sets[0] if phase not in TOTAL_PHASE_NAMES
        ]
        phases_mean = {
            phase: _aggregate(
                (record.phases[phase].mean for record in group),
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                tag=f"{key}:phase:{phase}:mean",
            )
            for phase in component_phases
        }
        residuals = [
            max(
                0.0,
                record.wall.mean
                - sum(record.phases[phase].mean for phase in component_phases),
            )
            for record in group
        ]
        wall_center = wall["mean"].center
        representative = min(
            group,
            key=lambda record: (
                abs(record.wall.mean - wall_center),
                record.block_id,
            ),
        )
        representative_components = {
            phase: representative.phases[phase].mean for phase in component_phases
        }
        component_total = sum(representative_components.values())
        source_wall = representative.wall.mean
        if component_total > source_wall and component_total > 0:
            representative_components = {
                phase: value * source_wall / component_total
                for phase, value in representative_components.items()
            }
            source_residual = 0.0
        else:
            source_residual = max(0.0, source_wall - component_total)
        if source_wall > 0:
            stack_scale = wall_center / source_wall
            phase_stack = {
                phase: value * stack_scale
                for phase, value in representative_components.items()
            }
            phase_stack["unattributed"] = source_residual * stack_scale
        else:
            phase_stack = {phase: 0.0 for phase in component_phases}
            phase_stack["unattributed"] = wall_center
        if not math.isclose(
            sum(phase_stack.values()), wall_center, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise AssertionError("paired phase stack must equal displayed wall mean")
        summaries[key] = GroupSummary(
            name=next(iter(names)),
            key=key,
            blocks=len(group),
            samples_min=min(record.measurement_count for record in group),
            samples_max=max(record.measurement_count for record in group),
            wall=wall,
            phases_mean=phases_mean,
            unattributed_mean=_aggregate(
                residuals,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                tag=f"{key}:phase:unattributed:mean",
            ),
            phase_stack_seconds=phase_stack,
            phase_representative_block=representative.block_id,
        )
    block_counts = {summary.blocks for summary in summaries.values()}
    if len(block_counts) != 1:
        detail = ", ".join(
            f"{summary.name}={summary.blocks}"
            for summary in sorted(summaries.values(), key=lambda item: item.name)
        )
        raise ValidationError(
            "unbalanced process blocks across conditions; refusing a partial matrix: "
            f"{detail}"
        )
    return summaries


def infer_reference(keys: Iterable[CellKey], reference_name: str | None, summaries: Mapping[CellKey, GroupSummary]) -> CellKey:
    keys = tuple(keys)
    if reference_name is not None:
        matches = [key for key in keys if summaries[key].name == reference_name]
        if len(matches) != 1:
            raise ValidationError(
                f"--reference-cell {reference_name!r} matched {len(matches)} conditions"
            )
        return matches[0]
    scores: dict[CellKey, int] = {}
    for candidate in keys:
        scores[candidate] = sum(
            other.H == candidate.H
            and sum(
                left != right
                for left, right in zip(
                    (candidate.J, candidate.N, candidate.G),
                    (other.J, other.N, other.G),
                )
            )
            == 1
            for other in keys
            if other != candidate
        )
    best_score = max(scores.values(), default=0)
    best = [key for key, score in scores.items() if score == best_score]
    if best_score < 3 or len(best) != 1:
        detail = ", ".join(f"{key}:{scores[key]}" for key in sorted(best))
        raise ValidationError(
            "cannot infer a unique reference cell from one-axis neighbors "
            f"(best score {best_score}; candidates {detail}); pass --reference-cell"
        )
    return best[0]


def infer_views(keys: Iterable[CellKey], reference: CellKey) -> dict[str, tuple[CellKey, ...]]:
    keys = tuple(keys)
    views = {
        "G": tuple(
            sorted(
                (key for key in keys if (key.J, key.N, key.H) == (reference.J, reference.N, reference.H)),
                key=lambda key: key.G,
            )
        ),
        "N": tuple(
            sorted(
                (key for key in keys if (key.J, key.H, key.G) == (reference.J, reference.H, reference.G)),
                key=lambda key: key.N,
            )
        ),
        "J": tuple(
            sorted(
                (key for key in keys if (key.N, key.H, key.G) == (reference.N, reference.H, reference.G)),
                key=lambda key: key.J,
            )
        ),
        "matched_J": tuple(
            sorted(
                (
                    key
                    for key in keys
                    if key.H == reference.H
                    and key.imagined_transitions == reference.imagined_transitions
                    and key.update_slots == reference.update_slots
                ),
                key=lambda key: key.J,
            )
        ),
    }
    for name, view in views.items():
        if len(view) < 2:
            raise ValidationError(
                f"view {name!r} has only {len(view)} condition(s) around reference {reference}"
            )
        variable = "J" if name in {"J", "matched_J"} else name
        levels = {getattr(key, variable) for key in view}
        if len(levels) != len(view):
            raise ValidationError(f"view {name!r} has duplicate {variable} levels")
    return views


def build_dataset(
    paths: Sequence[Path | str],
    *,
    reference_name: str | None = None,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 0,
) -> Dataset:
    records, context = load_blocks(paths)
    summaries = summarize_blocks(
        records,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    reference = infer_reference(summaries, reference_name, summaries)
    views = infer_views(summaries, reference)
    return Dataset(
        records=records,
        summaries=summaries,
        reference=reference,
        views=views,
        metadata=context["metadata"],
        settings=context["settings"],
    )


def _phase_label(name: str) -> str:
    if name == "unattributed":
        return "unattributed / overhead"
    label = re.sub(r"^inner_", "", name)
    label = re.sub(r"_seconds$", "", label)
    return label.replace("_", " ")


def _phase_sort_key(name: str) -> tuple[int, str]:
    try:
        return (PHASE_ORDER.index(name), name)
    except ValueError:
        return (len(PHASE_ORDER), name)


def _view_membership(dataset: Dataset, key: CellKey) -> str:
    return ";".join(name for name, keys in dataset.views.items() if key in keys)


def write_summary_csv(dataset: Dataset, path: Path) -> Path:
    phase_names = sorted(
        {
            phase
            for summary in dataset.summaries.values()
            for phase in summary.phase_stack_seconds
            if phase != "unattributed"
        },
        key=_phase_sort_key,
    )
    fields = [
        "name",
        "views",
        "is_reference",
        "J",
        "N",
        "H",
        "G",
        "rollout_paths",
        "imagined_transitions",
        "update_slots",
        "process_blocks",
        "samples_per_block_min",
        "samples_per_block_max",
        "phase_representative_block",
    ]
    for metric in ("mean", "p50", "p90", "p95"):
        fields.extend((f"wall_{metric}_ms", f"wall_{metric}_ci_low_ms", f"wall_{metric}_ci_high_ms"))
    fields.extend(
        f"phase_{_phase_label(phase).replace(' ', '_')}_paired_ms"
        for phase in phase_names
    )
    fields.append("phase_unattributed_paired_ms")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(dataset.summaries):
            summary = dataset.summaries[key]
            row: dict[str, Any] = {
                "name": summary.name,
                "views": _view_membership(dataset, key),
                "is_reference": key == dataset.reference,
                "J": key.J,
                "N": key.N,
                "H": key.H,
                "G": key.G,
                "rollout_paths": key.rollout_paths,
                "imagined_transitions": key.imagined_transitions,
                "update_slots": key.update_slots,
                "process_blocks": summary.blocks,
                "samples_per_block_min": summary.samples_min,
                "samples_per_block_max": summary.samples_max,
                "phase_representative_block": summary.phase_representative_block,
            }
            for metric, aggregate in summary.wall.items():
                row[f"wall_{metric}_ms"] = f"{aggregate.center * 1000:.9g}"
                row[f"wall_{metric}_ci_low_ms"] = f"{aggregate.low * 1000:.9g}"
                row[f"wall_{metric}_ci_high_ms"] = f"{aggregate.high * 1000:.9g}"
            for phase in phase_names:
                value = summary.phase_stack_seconds.get(phase)
                row[f"phase_{_phase_label(phase).replace(' ', '_')}_paired_ms"] = (
                    "" if value is None else f"{value * 1000:.9g}"
                )
            row["phase_unattributed_paired_ms"] = (
                f"{summary.phase_stack_seconds['unattributed'] * 1000:.9g}"
            )
            writer.writerow(row)
    return path


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "plotting requires matplotlib (install the repository requirements)"
        ) from exc
    return plt


def _view_specs() -> tuple[tuple[str, str, str], ...]:
    return (
        ("G", "Updates per round, G", "G sweep"),
        ("N", "Rollouts per round, N", "N sweep"),
        ("J", "Inner rounds, J", "J sweep (natural work)"),
        ("matched_J", "Inner rounds, J", "J sweep (matched work)"),
    )


def _x_labels(view_name: str, keys: Sequence[CellKey]) -> list[str]:
    if view_name == "matched_J":
        return [f"{key.J}\nN={key.N}, G={key.G}" for key in keys]
    variable = "J" if view_name == "J" else view_name
    return [str(getattr(key, variable)) for key in keys]


def _footer(dataset: Dataset) -> str:
    block_counts = [summary.blocks for summary in dataset.summaries.values()]
    block_text = (
        str(block_counts[0])
        if len(set(block_counts)) == 1
        else f"{min(block_counts)}–{max(block_counts)}"
    )
    calls = dataset.settings.get("measured_calls", "?")
    return (
        f"Point = median across {block_text} process blocks/cell; bars = process-block "
        f"bootstrap 95% CI; {calls} measured calls/block/cell."
    )


def _apply_plot_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _errorbar(ax, x: Sequence[int], aggregates: Sequence[Aggregate], **kwargs: Any) -> None:
    centers = [aggregate.center * 1000 for aggregate in aggregates]
    lower = [max(0.0, (aggregate.center - aggregate.low) * 1000) for aggregate in aggregates]
    upper = [max(0.0, (aggregate.high - aggregate.center) * 1000) for aggregate in aggregates]
    ax.errorbar(x, centers, yerr=[lower, upper], capsize=2.5, **kwargs)


def plot_scaling(dataset: Dataset, png_path: Path, pdf_path: Path, *, tail: str) -> tuple[Path, Path]:
    plt = _load_pyplot()
    _apply_plot_style(plt)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.7), constrained_layout=False)
    colors = {"p50": "#0072B2", tail: "#D55E00"}
    for ax, (view_name, xlabel, title) in zip(axes.flat, _view_specs()):
        keys = dataset.views[view_name]
        x = list(range(len(keys)))
        _errorbar(
            ax,
            x,
            [dataset.summaries[key].wall["p50"] for key in keys],
            color=colors["p50"],
            marker="o",
            linewidth=1.7,
            markersize=4.5,
            label="p50",
        )
        _errorbar(
            ax,
            x,
            [dataset.summaries[key].wall[tail] for key in keys],
            color=colors[tail],
            marker="s",
            linewidth=1.5,
            markersize=4.0,
            label=tail,
        )
        reference_index = keys.index(dataset.reference) if dataset.reference in keys else None
        if reference_index is not None:
            ax.axvline(reference_index, color="0.55", linestyle=":", linewidth=1.0)
        ax.set_xticks(x, _x_labels(view_name, keys))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Latency (ms/action)")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
    axes.flat[0].legend(loc="best", ncols=2)
    fig.suptitle("AMBI inner-loop latency scaling", fontsize=12, fontweight="semibold")
    fig.text(0.5, 0.012, _footer(dataset), ha="center", va="bottom", fontsize=7.2, color="0.3")
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_phases(dataset: Dataset, png_path: Path, pdf_path: Path) -> tuple[Path, Path]:
    plt = _load_pyplot()
    _apply_plot_style(plt)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.7), constrained_layout=False)
    phase_names = sorted(
        {
            phase
            for summary in dataset.summaries.values()
            for phase in summary.phase_stack_seconds
            if phase != "unattributed"
        },
        key=_phase_sort_key,
    )
    all_names = phase_names + ["unattributed"]
    palette = ["#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#0072B2", "#999999"]
    colors = {name: palette[index % len(palette)] for index, name in enumerate(all_names)}
    for ax, (view_name, xlabel, title) in zip(axes.flat, _view_specs()):
        keys = dataset.views[view_name]
        x = list(range(len(keys)))
        bottoms = [0.0] * len(keys)
        for phase in all_names:
            values = []
            for key in keys:
                summary = dataset.summaries[key]
                value = summary.phase_stack_seconds.get(phase)
                values.append(0.0 if value is None else value * 1000)
            ax.bar(
                x,
                values,
                bottom=bottoms,
                width=0.68,
                color=colors[phase],
                edgecolor="white",
                linewidth=0.35,
                label=_phase_label(phase),
            )
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        total_means = [dataset.summaries[key].wall["mean"].center * 1000 for key in keys]
        ax.scatter(x, total_means, marker="D", s=19, color="black", zorder=4, label="wall mean")
        ax.set_xticks(x, _x_labels(view_name, keys))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Paired representative mean (ms/action)")
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncols=min(4, len(labels)), fontsize=7.8)
    fig.suptitle("AMBI inner-loop latency phase breakdown", fontsize=12, fontweight="semibold")
    fig.text(
        0.5,
        0.012,
        _footer(dataset)
        + " Each additive phase stack uses the block nearest the median wall mean, "
        + "scaled to its displayed diamond.",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.86))
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def render_outputs(dataset: Dataset, output_dir: Path, *, prefix: str, tail: str) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scaling_png = output_dir / f"{prefix}_scaling.png"
    scaling_pdf = output_dir / f"{prefix}_scaling.pdf"
    phases_png = output_dir / f"{prefix}_phases.png"
    phases_pdf = output_dir / f"{prefix}_phases.pdf"
    csv_path = output_dir / f"{prefix}_summary.csv"
    plot_scaling(dataset, scaling_png, scaling_pdf, tail=tail)
    plot_phases(dataset, phases_png, phases_pdf)
    write_summary_csv(dataset, csv_path)
    return scaling_png, scaling_pdf, phases_png, phases_pdf, csv_path


def _expand_inputs(inputs: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            block_dirs = {
                candidate
                for candidate in (path, *path.rglob("block_*"))
                if candidate.is_dir() and re.fullmatch(r"block_\d+", candidate.name)
            }
            incomplete = sorted(
                block_dir for block_dir in block_dirs if not (block_dir / "COMPLETE").is_file()
            )
            if incomplete:
                raise ValidationError(
                    f"{path}: incomplete Oscar block directories lack COMPLETE: "
                    + ", ".join(str(block_dir) for block_dir in incomplete)
                )
            matches = sorted(path.rglob("*.json"))
            if not matches:
                raise ValidationError(f"{path}: directory contains no JSON files")
            paths.extend(matches)
        else:
            paths.append(path)
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)):
        raise ValidationError("the same input JSON file was supplied more than once")
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot process-block aggregates from AMBI inner latency JSON outputs."
    )
    parser.add_argument("inputs", nargs="+", help="Benchmark JSON files or directories")
    parser.add_argument("--output-dir", type=Path, default=Path("latency_plots"))
    parser.add_argument("--prefix", default="ambi_latency")
    parser.add_argument("--tail", choices=("p90", "p95"), default="p95")
    parser.add_argument("--reference-cell", help="Explicit cell name for the sweep center")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.prefix or Path(args.prefix).name != args.prefix:
        raise ValidationError("--prefix must be a non-empty file-name stem")
    paths = _expand_inputs(args.inputs)
    dataset = build_dataset(
        paths,
        reference_name=args.reference_cell,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    outputs = render_outputs(dataset, args.output_dir, prefix=args.prefix, tail=args.tail)
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
