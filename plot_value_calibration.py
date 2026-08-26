#!/usr/bin/env python3
"""Aggregate and plot AMBI value-calibration diagnostics across seeds.

Each input is one training seed. Histories must contain the same, unique
``env_step`` grid and all four value-diagnostic metrics. Aggregation is an
equal-weight mean over seeds with population standard deviation (``ddof=0``);
the utility deliberately performs no interpolation or smoothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ENV_STEP_KEY = "env_step"
DETERMINISTIC_MC_KEY = "eval/mc_value"
DETERMINISTIC_Q_KEY = "eval/q_value"
STOCHASTIC_MC_KEY = "eval/stochastic_mc_value"
STOCHASTIC_Q_KEY = "eval/stochastic_q_mean_all"
METRIC_KEYS = (
    DETERMINISTIC_MC_KEY,
    DETERMINISTIC_Q_KEY,
    STOCHASTIC_MC_KEY,
    STOCHASTIC_Q_KEY,
)
REQUIRED_HISTORY_KEYS = (ENV_STEP_KEY, *METRIC_KEYS)


# W&B stores both the original algorithm mapping and a fully resolved mapping.
# These are the fields needed to decide whether value-calibration histories have
# the same scientific meaning. Training seed and operational logging fields are
# intentionally absent: seeds are the population being aggregated.
_SEMANTIC_CONFIG_PATHS: Mapping[str, tuple[tuple[str, ...], ...]] = {
    "algorithm": (("algorithm",),),
    "recipe": (("run_params", "name"),),
    "environment": (("run_params", "env"), ("env",)),
    "task": (
        ("run_params", "env_params", "task"),
        ("run_params", "task"),
        ("config", "task"),
    ),
    "obs": (("config", "obs"), ("alg_params", "obs"), ("obs",)),
    "episode_length": (
        ("config", "episode_length"),
        ("alg_params", "episode_length"),
        ("episode_length",),
    ),
    "episodic": (
        ("config", "episodic"),
        ("alg_params", "episodic"),
        ("episodic",),
    ),
    "discount": (
        ("config", "discount"),
        ("alg_params", "discount"),
        ("discount",),
    ),
    "discount_min": (
        ("config", "discount_min"),
        ("alg_params", "discount_min"),
        ("discount_min",),
    ),
    "discount_max": (
        ("config", "discount_max"),
        ("alg_params", "discount_max"),
        ("discount_max",),
    ),
    "outer_critic_target": (
        ("config", "outer_critic_target"),
        ("alg_params", "outer_critic_target"),
        ("outer_critic_target",),
    ),
    "inner_sac_critic_target": (
        ("config", "inner_sac_critic_target"),
        ("alg_params", "inner_sac_critic_target"),
        ("inner_sac_critic_target",),
    ),
    "outer_q_target_reduction": (
        ("config", "outer_q_target_reduction"),
        ("alg_params", "outer_q_target_reduction"),
        ("outer_q_target_reduction",),
    ),
    "outer_q_actor_reduction": (
        ("config", "outer_q_actor_reduction"),
        ("alg_params", "outer_q_actor_reduction"),
        ("outer_q_actor_reduction",),
    ),
    "q_representation": (
        ("config", "q_representation"),
        ("alg_params", "q_representation"),
        ("q_representation",),
    ),
    "num_q": (("config", "num_q"), ("alg_params", "num_q"), ("num_q",)),
    "q_pair_size": (
        ("config", "q_pair_size"),
        ("alg_params", "q_pair_size"),
        ("q_pair_size",),
    ),
    "utd": (("config", "utd"), ("alg_params", "utd"), ("utd",)),
    "eval_freq": (
        ("config", "eval_freq"),
        ("alg_params", "eval_freq"),
        ("eval_freq",),
    ),
    "total_steps": (
        ("run_params", "total_steps"),
        ("config", "steps"),
        ("total_steps",),
        ("steps",),
    ),
    "eval_value": (
        ("config", "eval_value"),
        ("alg_params", "eval_value"),
        ("eval_value",),
    ),
    "eval_value_samples": (
        ("config", "eval_value_samples"),
        ("alg_params", "eval_value_samples"),
        ("eval_value_samples",),
    ),
    "eval_value_seed": (
        ("config", "eval_value_seed"),
        ("alg_params", "eval_value_seed"),
        ("eval_value_seed",),
    ),
    "eval_value_protocols": (
        ("config", "eval_value_protocols"),
        ("alg_params", "eval_value_protocols"),
        ("eval_value_protocols",),
    ),
    "outer_policy_episode_probability": (
        ("config", "outer_policy_episode_probability"),
        ("alg_params", "outer_policy_episode_probability"),
        ("outer_policy_episode_probability",),
    ),
}

_TRAINING_SEED_PATHS = (
    ("config", "seed"),
    ("run_params", "seed"),
    ("alg_params", "seed"),
    ("seed",),
)


@dataclass(frozen=True)
class SeedHistory:
    """One training seed's value-diagnostic history."""

    source: str
    env_step: Sequence[object]
    metrics: Mapping[str, Sequence[object]]
    semantic_config: Mapping[str, object] | None = None
    training_seed: int | None = None


@dataclass(frozen=True)
class MetricSummary:
    """Across-seed summary for one metric on the common step grid."""

    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class AggregatedHistory:
    """Equal-weight aggregate over validated training seeds."""

    env_step: np.ndarray
    metrics: Mapping[str, MetricSummary]
    n_seeds: int


def _load_pyplot():
    """Import matplotlib only for rendering and force its headless backend."""

    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError(
            "Plot rendering requires matplotlib; install the repository's root "
            "requirements or run aggregation without write_artifacts()."
        ) from exc
    matplotlib.use("Agg", force=True)
    try:
        import matplotlib.pyplot as pyplot
    except ImportError as exc:
        raise RuntimeError(
            "Plot rendering requires matplotlib; install the repository's root "
            "requirements or run aggregation without write_artifacts()."
        ) from exc
    return pyplot


def _path_value(config: Mapping[str, object], path: Sequence[str]) -> object:
    current: object = config
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _canonical_config_value(value: object) -> object:
    """Return a deterministic JSON-compatible representation."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Semantic config values must be finite.")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_config_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_config_value(item) for item in value]
    raise ValueError(
        "Semantic config values must be JSON-compatible; "
        f"got {type(value).__name__}."
    )


_MISSING = object()


def extract_semantic_config(config: Mapping[str, object] | None) -> dict[str, object] | None:
    """Extract comparable scientific fields from a W&B run config.

    ``None`` is returned when no semantic fields are available, as is normally
    the case for a plain offline history CSV. If aliases present in one config
    disagree, the run is rejected before cross-seed aggregation.
    """

    if not config:
        return None
    if not isinstance(config, Mapping):
        raise ValueError("Run config must be a mapping when supplied.")

    extracted: dict[str, object] = {}
    for semantic_key, paths in _SEMANTIC_CONFIG_PATHS.items():
        found: list[tuple[str, object]] = []
        for path in paths:
            value = _path_value(config, path)
            if value is not _MISSING:
                found.append((".".join(path), _canonical_config_value(value)))
        if not found:
            continue
        reference = found[0][1]
        conflicts = [(path, value) for path, value in found[1:] if value != reference]
        if conflicts:
            detail = ", ".join(
                f"{path}={value!r}" for path, value in [found[0], *conflicts]
            )
            raise ValueError(
                f"Run config has conflicting aliases for {semantic_key!r}: {detail}."
            )
        extracted[semantic_key] = reference
    return extracted or None


def extract_training_seed(config: Mapping[str, object] | None) -> int | None:
    """Extract one consistent non-negative training seed from a run config."""

    if not config:
        return None
    if not isinstance(config, Mapping):
        raise ValueError("Run config must be a mapping when supplied.")

    found: list[tuple[str, int]] = []
    for path in _TRAINING_SEED_PATHS:
        value = _path_value(config, path)
        if value is _MISSING:
            continue
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Training seed must be a non-negative integer.")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Training seed must be a non-negative integer.") from exc
        if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
            raise ValueError("Training seed must be a non-negative integer.")
        found.append((".".join(path), int(numeric)))
    if not found:
        return None
    reference = found[0][1]
    conflicts = [(path, seed) for path, seed in found[1:] if seed != reference]
    if conflicts:
        detail = ", ".join(
            f"{path}={seed}" for path, seed in [found[0], *conflicts]
        )
        raise ValueError(f"Run config has conflicting training-seed aliases: {detail}.")
    return reference


def _one_dimensional(values: Sequence[object], *, source: str, key: str) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{source}: {key!r} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{source}: history is empty.")
    return array


def _finite_metric(values: Sequence[object], *, source: str, key: str) -> np.ndarray:
    raw = _one_dimensional(values, source=source, key=key)
    try:
        result = raw.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{source}: {key!r} must contain only numeric values.") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{source}: {key!r} contains a non-finite value.")
    return result


def _environment_steps(values: Sequence[object], *, source: str) -> np.ndarray:
    raw = _one_dimensional(values, source=source, key=ENV_STEP_KEY)
    parsed: list[int] = []
    for value in raw.tolist():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{source}: {ENV_STEP_KEY!r} must contain integers, not booleans.")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{source}: {ENV_STEP_KEY!r} must contain non-negative integers."
            ) from exc
        if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
            raise ValueError(
                f"{source}: {ENV_STEP_KEY!r} must contain non-negative integers."
            )
        parsed.append(int(numeric))
    result = np.asarray(parsed, dtype=np.int64)
    unique, counts = np.unique(result, return_counts=True)
    duplicates = unique[counts > 1]
    if duplicates.size:
        rendered = ", ".join(str(int(step)) for step in duplicates)
        raise ValueError(f"{source}: duplicate env_step value(s): {rendered}.")
    return result


def _validate_configured_step_grid(
    env_step: np.ndarray,
    semantic_config: Mapping[str, object] | None,
    *,
    source: str,
) -> None:
    """Require the full scheduled grid when W&B config metadata is available."""

    if semantic_config is None:
        return
    if "eval_freq" not in semantic_config or "total_steps" not in semantic_config:
        return

    parsed: dict[str, int] = {}
    for key in ("eval_freq", "total_steps"):
        value = semantic_config[key]
        if isinstance(value, bool):
            raise ValueError(f"{source}: semantic {key!r} must be an integer.")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{source}: semantic {key!r} must be an integer."
            ) from exc
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{source}: semantic {key!r} must be an integer.")
        parsed[key] = int(numeric)

    eval_freq = parsed["eval_freq"]
    total_steps = parsed["total_steps"]
    if eval_freq <= 0 or total_steps < 0 or total_steps % eval_freq != 0:
        raise ValueError(
            f"{source}: configured eval_freq={eval_freq} and "
            f"total_steps={total_steps} do not define an exact complete grid."
        )
    expected = np.arange(0, total_steps + 1, eval_freq, dtype=np.int64)
    if not np.array_equal(env_step, expected):
        missing = np.setdiff1d(expected, env_step, assume_unique=True)
        unexpected = np.setdiff1d(env_step, expected, assume_unique=True)
        detail = []
        if missing.size:
            detail.append(
                "missing " + ", ".join(str(int(step)) for step in missing)
            )
        if unexpected.size:
            detail.append(
                "unexpected " + ", ".join(str(int(step)) for step in unexpected)
            )
        raise ValueError(
            f"{source}: env_step grid does not match configured step zero through "
            f"{total_steps} at cadence {eval_freq}: {'; '.join(detail)}."
        )


def validate_seed_history(history: SeedHistory) -> SeedHistory:
    """Validate one history and return it sorted by ``env_step``."""

    if not isinstance(history, SeedHistory):
        raise TypeError("Histories must be SeedHistory instances.")
    source = str(history.source).strip()
    if not source:
        raise ValueError("Every history must have a non-empty source name.")
    if not isinstance(history.metrics, Mapping):
        raise ValueError(f"{source}: metrics must be a mapping.")

    missing = [key for key in METRIC_KEYS if key not in history.metrics]
    if missing:
        raise ValueError(f"{source}: missing required metric key(s): {', '.join(missing)}.")

    env_step = _environment_steps(history.env_step, source=source)
    order = np.argsort(env_step, kind="stable")
    env_step = env_step[order]
    metrics: dict[str, np.ndarray] = {}
    for key in METRIC_KEYS:
        values = _finite_metric(history.metrics[key], source=source, key=key)
        if values.size != env_step.size:
            raise ValueError(
                f"{source}: {key!r} has {values.size} rows but {ENV_STEP_KEY!r} "
                f"has {env_step.size}."
            )
        metrics[key] = values[order]

    semantic_config = (
        None
        if history.semantic_config is None
        else _canonical_config_value(history.semantic_config)
    )
    if semantic_config is not None and not isinstance(semantic_config, Mapping):
        raise ValueError(f"{source}: semantic_config must be a mapping or None.")
    training_seed = history.training_seed
    if training_seed is not None:
        if isinstance(training_seed, (bool, np.bool_)) or not isinstance(
            training_seed, (int, np.integer)
        ):
            raise ValueError(f"{source}: training_seed must be a non-negative integer.")
        training_seed = int(training_seed)
        if training_seed < 0:
            raise ValueError(f"{source}: training_seed must be a non-negative integer.")
    _validate_configured_step_grid(
        env_step,
        semantic_config,
        source=source,
    )
    return SeedHistory(
        source=source,
        env_step=env_step,
        metrics=metrics,
        semantic_config=semantic_config,
        training_seed=training_seed,
    )


def _config_mismatch(reference: SeedHistory, candidate: SeedHistory) -> str:
    keys = sorted(
        set(reference.semantic_config or {}) | set(candidate.semantic_config or {})
    )
    differences = []
    for key in keys:
        left = (reference.semantic_config or {}).get(key, _MISSING)
        right = (candidate.semantic_config or {}).get(key, _MISSING)
        if left != right:
            left_text = "<missing>" if left is _MISSING else repr(left)
            right_text = "<missing>" if right is _MISSING else repr(right)
            differences.append(f"{key}: {left_text} != {right_text}")
    return "; ".join(differences)


def validate_histories(histories: Sequence[SeedHistory]) -> list[SeedHistory]:
    """Validate metrics, exact common grid, and available semantic configs."""

    if not histories:
        raise ValueError("At least one run or history CSV is required.")
    normalized = [validate_seed_history(history) for history in histories]

    sources = [history.source for history in normalized]
    if len(set(sources)) != len(sources):
        duplicates = sorted({source for source in sources if sources.count(source) > 1})
        raise ValueError(f"Duplicate input source(s): {', '.join(duplicates)}.")

    reference_grid = np.asarray(normalized[0].env_step, dtype=np.int64)
    for history in normalized[1:]:
        candidate_grid = np.asarray(history.env_step, dtype=np.int64)
        if not np.array_equal(candidate_grid, reference_grid):
            raise ValueError(
                f"{history.source}: env_step grid does not exactly match "
                f"{normalized[0].source}; interpolation is not permitted."
            )

    configured = [history for history in normalized if history.semantic_config is not None]
    if len(configured) > 1:
        reference = configured[0]
        for history in configured[1:]:
            if history.semantic_config != reference.semantic_config:
                detail = _config_mismatch(reference, history)
                raise ValueError(
                    f"{history.source}: semantic config is incompatible with "
                    f"{reference.source}: {detail}."
                )
    return normalized


def aggregate_histories(histories: Sequence[SeedHistory]) -> AggregatedHistory:
    """Compute equal-weight seed means and population SD on the exact grid."""

    normalized = validate_histories(histories)
    summaries: dict[str, MetricSummary] = {}
    for key in METRIC_KEYS:
        matrix = np.stack(
            [np.asarray(history.metrics[key], dtype=np.float64) for history in normalized],
            axis=0,
        )
        summaries[key] = MetricSummary(
            mean=matrix.mean(axis=0),
            std=matrix.std(axis=0, ddof=0),
        )
    return AggregatedHistory(
        env_step=np.asarray(normalized[0].env_step, dtype=np.int64).copy(),
        metrics=summaries,
        n_seeds=len(normalized),
    )


def _has_history_value(value: object) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip())


def history_from_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    source: str,
    semantic_config: Mapping[str, object] | None = None,
    training_seed: int | None = None,
) -> SeedHistory:
    """Build a history from W&B-like row mappings.

    Rows with none of the four evaluation metrics are ignored, allowing a full
    W&B history export. A row containing only part of the diagnostic payload is
    rejected instead of being forward-filled.
    """

    columns: dict[str, list[object]] = {key: [] for key in REQUIRED_HISTORY_KEYS}
    for row_index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"{source}: history row {row_index} is not a mapping.")
        if not any(_has_history_value(row.get(key)) for key in METRIC_KEYS):
            continue
        missing = [
            key for key in REQUIRED_HISTORY_KEYS if not _has_history_value(row.get(key))
        ]
        if missing:
            raise ValueError(
                f"{source}: diagnostic row {row_index} is missing required key(s): "
                f"{', '.join(missing)}."
            )
        for key in REQUIRED_HISTORY_KEYS:
            columns[key].append(row[key])

    if not columns[ENV_STEP_KEY]:
        raise ValueError(f"{source}: no complete value-diagnostic rows were found.")
    return SeedHistory(
        source=source,
        env_step=columns[ENV_STEP_KEY],
        metrics={key: columns[key] for key in METRIC_KEYS},
        semantic_config=semantic_config,
        training_seed=training_seed,
    )


def load_history_csv(path: str | Path) -> SeedHistory:
    """Load one offline W&B-style history CSV."""

    csv_path = Path(path)
    if not csv_path.is_file():
        raise ValueError(f"History CSV does not exist: {csv_path}.")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [key for key in REQUIRED_HISTORY_KEYS if key not in fieldnames]
        if missing:
            raise ValueError(
                f"{csv_path}: missing required CSV column(s): {', '.join(missing)}."
            )
        return history_from_rows(reader, source=str(csv_path))


def load_wandb_history(run_path: str, *, api: object | None = None) -> SeedHistory:
    """Load one run through the W&B Public API, importing W&B lazily."""

    run_path = str(run_path).strip()
    if not run_path:
        raise ValueError("W&B run paths must be non-empty.")
    if api is None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "Loading --run inputs requires the optional wandb dependency."
            ) from exc
        api = wandb.Api()
    run = api.run(run_path)
    raw_config = dict(getattr(run, "config", {}) or {})
    semantic_config = extract_semantic_config(raw_config)
    training_seed = extract_training_seed(raw_config)
    rows = run.scan_history(keys=list(REQUIRED_HISTORY_KEYS), page_size=10_000)
    return history_from_rows(
        rows,
        source=run_path,
        semantic_config=semantic_config,
        training_seed=training_seed,
    )


def write_aggregate_csv(aggregate: AggregatedHistory, path: str | Path) -> Path:
    """Write the exact-grid aggregate with mean, population SD, and seed count."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [ENV_STEP_KEY, "n"]
    for key in METRIC_KEYS:
        fieldnames.extend((f"{key}_mean", f"{key}_std"))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, step in enumerate(aggregate.env_step):
            row: dict[str, object] = {ENV_STEP_KEY: int(step), "n": aggregate.n_seeds}
            for key in METRIC_KEYS:
                row[f"{key}_mean"] = format(
                    float(aggregate.metrics[key].mean[index]), ".17g"
                )
                row[f"{key}_std"] = format(
                    float(aggregate.metrics[key].std[index]), ".17g"
                )
            writer.writerow(row)
    return output


def _plot_panel(
    axis,
    aggregate: AggregatedHistory,
    *,
    mc_key: str,
    q_key: str,
    title: str,
) -> None:
    steps = aggregate.env_step
    series = (
        (mc_key, "Monte Carlo value", "#0072B2", "-"),
        (q_key, "Q estimate", "#D55E00", "--"),
    )
    for key, label, color, linestyle in series:
        summary = aggregate.metrics[key]
        axis.plot(
            steps,
            summary.mean,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=f"{label} (n={aggregate.n_seeds})",
        )
        if aggregate.n_seeds > 1:
            axis.fill_between(
                steps,
                summary.mean - summary.std,
                summary.mean + summary.std,
                color=color,
                alpha=0.18,
                linewidth=0.0,
            )
    axis.set_title(title)
    axis.set_xlabel("Environment steps")
    axis.grid(True, alpha=0.25, linewidth=0.7)
    uncertainty_label = (
        "Mean ± 1 across-seed population SD"
        if aggregate.n_seeds > 1
        else "Mean; n=1 (no uncertainty band)"
    )
    axis.legend(title=uncertainty_label, frameon=False)
    axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))


def plot_aggregate(
    aggregate: AggregatedHistory,
    *,
    title: str = "AMBI value calibration",
):
    """Create the requested deterministic/stochastic two-panel figure."""

    pyplot = _load_pyplot()
    figure, axes = pyplot.subplots(1, 2, figsize=(12.0, 4.5), sharex=True)
    _plot_panel(
        axes[0],
        aggregate,
        mc_key=DETERMINISTIC_MC_KEY,
        q_key=DETERMINISTIC_Q_KEY,
        title="Paper deterministic protocol",
    )
    _plot_panel(
        axes[1],
        aggregate,
        mc_key=STOCHASTIC_MC_KEY,
        q_key=STOCHASTIC_Q_KEY,
        title="Stochastic Bellman protocol",
    )
    axes[0].set_ylabel("Discounted value")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def write_artifacts(
    aggregate: AggregatedHistory,
    output_prefix: str | Path,
    *,
    title: str = "AMBI value calibration",
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write PNG, PDF, and aggregate CSV artifacts for one aggregate."""

    prefix = Path(output_prefix)
    if not prefix.name:
        raise ValueError("output_prefix must include a file-name prefix.")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "png": Path(f"{prefix}.png"),
        "pdf": Path(f"{prefix}.pdf"),
        "csv": Path(f"{prefix}.csv"),
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise ValueError(
            f"Output artifact(s) already exist: {rendered}. Pass --overwrite "
            "to replace them."
        )
    figure = plot_aggregate(aggregate, title=title)
    pyplot = _load_pyplot()
    try:
        figure.savefig(outputs["png"], dpi=200, bbox_inches="tight")
        figure.savefig(outputs["pdf"], bbox_inches="tight")
    finally:
        pyplot.close(figure)
    write_aggregate_csv(aggregate, outputs["csv"])
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate exact-grid AMBI value-calibration histories across training "
            "seeds and write a two-panel PNG/PDF plus aggregate CSV."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="ENTITY/PROJECT/RUN_ID",
        help="W&B run path; repeat once per training seed.",
    )
    parser.add_argument(
        "--history-csv",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="Offline W&B history CSV; repeat once per training seed.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("value_calibration"),
        help="Output path without extension (default: value_calibration).",
    )
    parser.add_argument(
        "--title",
        default="AMBI value calibration",
        help="Figure title.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing PNG, PDF, and CSV artifacts for the output prefix.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.run and not args.history_csv:
        parser.error("provide at least one --run or --history-csv input")
    try:
        histories = [load_wandb_history(run_path) for run_path in args.run]
        histories.extend(load_history_csv(path) for path in args.history_csv)
        aggregate = aggregate_histories(histories)
        outputs = write_artifacts(
            aggregate,
            args.output_prefix,
            title=args.title,
            overwrite=args.overwrite,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {key: str(path.resolve()) for key, path in outputs.items()},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
