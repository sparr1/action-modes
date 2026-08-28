#!/usr/bin/env python3
"""Aggregate and plot vanilla-SAC value-calibration diagnostics.

The paper-facing deterministic probe and the SAC-target-matched stochastic
probe intentionally remain distinct.  The matched curve uses the finite soft
return plus its time-limit target-critic tail; the aggregate CSV retains every
supporting reward, finite-return, tail, critic, error, and timing metric.

Each input is one unique training seed.  Histories must use the same exact
``env_step`` grid.  When W&B config metadata is available, the complete
configured step-zero-through-budget grid and the SAC scientific semantics are
validated before equal-weight aggregation.  No interpolation or smoothing is
performed, and uncertainty bands are one across-seed population standard
deviation (``ddof=0``).
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

PAPER_MC_KEY = "eval/mc_value"
PAPER_MC_STD_KEY = "eval/mc_value_std"
PAPER_Q_KEY = "eval/q_value"
PAPER_Q_STD_KEY = "eval/q_value_std"
PAPER_BIAS_KEY = "eval/q_minus_mc"

REWARD_MC_KEY = "eval/stochastic_reward_mc_value"
REWARD_MC_STD_KEY = "eval/stochastic_reward_mc_value_std"
SOFT_MC_FINITE_KEY = "eval/stochastic_soft_mc_finite_value"
SOFT_MC_FINITE_STD_KEY = "eval/stochastic_soft_mc_finite_value_std"
SOFT_MC_BOOTSTRAPPED_KEY = "eval/stochastic_soft_mc_bootstrapped_value"
SOFT_MC_BOOTSTRAPPED_STD_KEY = (
    "eval/stochastic_soft_mc_bootstrapped_value_std"
)
SOFT_TRUNCATION_TAIL_KEY = "eval/stochastic_soft_truncation_tail"
SOFT_TRUNCATION_FRACTION_KEY = "eval/stochastic_soft_truncation_fraction"
SOFT_Q_MEAN_KEY = "eval/stochastic_soft_q_mean_all"
SOFT_Q_MEAN_STD_KEY = "eval/stochastic_soft_q_mean_all_std"
SOFT_Q_MIN_KEY = "eval/stochastic_soft_q_min_all"
SOFT_Q_HEAD_STD_KEY = "eval/stochastic_soft_q_head_std"
SOFT_Q_BIAS_MEAN_KEY = (
    "eval/stochastic_soft_q_minus_mc_bootstrapped_mean_all"
)
SOFT_Q_RMSE_MEAN_KEY = "eval/stochastic_soft_q_rmse_bootstrapped_mean_all"
SOFT_Q_BIAS_MIN_KEY = "eval/stochastic_soft_q_minus_mc_bootstrapped_min_all"
SOFT_Q_RMSE_MIN_KEY = "eval/stochastic_soft_q_rmse_bootstrapped_min_all"
SOFT_ALPHA_KEY = "eval/stochastic_soft_alpha"

VALUE_SAMPLES_KEY = "eval/value_samples"
VALUE_EVAL_SECONDS_KEY = "time/value_eval_seconds"

PLOT_METRIC_KEYS = (
    PAPER_MC_KEY,
    PAPER_Q_KEY,
    SOFT_MC_BOOTSTRAPPED_KEY,
    SOFT_Q_MEAN_KEY,
)

# Preserve the evaluator's full scientific payload in the aggregate CSV.  The
# order groups paper compatibility, raw/finite/corrected returns, critic
# summaries, corrected errors, and operational metadata.
METRIC_KEYS = (
    PAPER_MC_KEY,
    PAPER_MC_STD_KEY,
    PAPER_Q_KEY,
    PAPER_Q_STD_KEY,
    PAPER_BIAS_KEY,
    REWARD_MC_KEY,
    REWARD_MC_STD_KEY,
    SOFT_MC_FINITE_KEY,
    SOFT_MC_FINITE_STD_KEY,
    SOFT_MC_BOOTSTRAPPED_KEY,
    SOFT_MC_BOOTSTRAPPED_STD_KEY,
    SOFT_TRUNCATION_TAIL_KEY,
    SOFT_TRUNCATION_FRACTION_KEY,
    SOFT_Q_MEAN_KEY,
    SOFT_Q_MEAN_STD_KEY,
    SOFT_Q_MIN_KEY,
    SOFT_Q_HEAD_STD_KEY,
    SOFT_Q_BIAS_MEAN_KEY,
    SOFT_Q_RMSE_MEAN_KEY,
    SOFT_Q_BIAS_MIN_KEY,
    SOFT_Q_RMSE_MIN_KEY,
    SOFT_ALPHA_KEY,
    VALUE_SAMPLES_KEY,
    VALUE_EVAL_SECONDS_KEY,
)
REQUIRED_HISTORY_KEYS = (ENV_STEP_KEY, *METRIC_KEYS)

EXPECTED_PROTOCOLS = ["paper_deterministic", "stochastic_soft_bellman"]


# Native SAC currently records the original algorithm params, the resolved
# SACConfig, and run params.  Include runtime-metadata aliases so task and
# observation semantics remain available after main.py resolves the env.
_SEMANTIC_CONFIG_PATHS: Mapping[str, tuple[tuple[str, ...], ...]] = {
    "recipe": (("run_params", "name"),),
    "algorithm": (
        ("algorithm",),
        ("run_params", "alg"),
        ("alg",),
    ),
    "environment": (
        ("run_params", "env"),
        ("env",),
    ),
    "task": (
        ("run_params", "resolved_runtime", "observation", "task"),
        ("run_params", "env_params", "task"),
        ("experiment_params", "env_params", "task"),
        ("env_params", "task"),
        ("config", "task"),
        ("task",),
    ),
    "observation": (
        ("run_params", "resolved_runtime", "observation", "mode"),
        ("run_params", "env_params", "obs"),
        ("experiment_params", "env_params", "obs"),
        ("env_params", "obs"),
        ("alg_params", "obs"),
        ("config", "obs"),
        ("obs",),
    ),
    "action_repeat": (
        ("run_params", "resolved_runtime", "observation", "action_repeat"),
        ("run_params", "env_params", "action_repeat"),
        ("experiment_params", "env_params", "action_repeat"),
        ("env_params", "action_repeat"),
        ("action_repeat",),
    ),
    # The resolved SACConfig is the fail-closed training recipe.  It is
    # canonicalized below with only seed/device/verbosity removed so runs that
    # differ in architecture, optimizers, replay, target cadence, or any future
    # scientific field cannot be silently pooled as training seeds.
    "training_config": (("config",),),
    "train_frequency": (
        ("alg_params", "train_freq"),
        ("train_freq",),
    ),
    "gamma": (
        ("config", "gamma"),
        ("alg_params", "gamma"),
        ("gamma",),
    ),
    "alpha_mode": (
        ("config", "ent_coef"),
        ("alg_params", "ent_coef"),
        ("ent_coef",),
    ),
    "target_entropy": (
        ("config", "target_entropy"),
        ("alg_params", "target_entropy"),
        ("target_entropy",),
    ),
    "alpha_lr": (
        ("config", "alpha_lr"),
        ("alg_params", "alpha_lr"),
        ("alpha_lr",),
    ),
    "q_representation": (
        ("config", "q_representation"),
        ("alg_params", "q_representation"),
        ("q_representation",),
    ),
    "num_q": (
        ("config", "num_q"),
        ("alg_params", "num_q"),
        ("num_q",),
    ),
    "q_pair_size": (
        ("config", "q_pair_size"),
        ("alg_params", "q_pair_size"),
        ("q_pair_size",),
    ),
    "q_target_reduction": (
        ("config", "q_target_reduction"),
        ("alg_params", "q_target_reduction"),
        ("q_target_reduction",),
    ),
    "q_actor_reduction": (
        ("config", "q_actor_reduction"),
        ("alg_params", "q_actor_reduction"),
        ("q_actor_reduction",),
    ),
    "q_num_bins": (
        ("config", "q_num_bins"),
        ("alg_params", "q_num_bins"),
        ("q_num_bins",),
    ),
    "q_vmin": (
        ("config", "q_vmin"),
        ("alg_params", "q_vmin"),
        ("q_vmin",),
    ),
    "q_vmax": (
        ("config", "q_vmax"),
        ("alg_params", "q_vmax"),
        ("q_vmax",),
    ),
    "eval_freq": (
        ("config", "eval_freq"),
        ("alg_params", "eval_freq"),
        ("eval_freq",),
    ),
    "eval_episodes": (
        ("alg_params", "eval_episodes"),
        ("eval_episodes",),
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
}

_REQUIRED_SAC_SEMANTICS = frozenset(
    {
        "recipe",
        "algorithm",
        "environment",
        "task",
        "observation",
        "action_repeat",
        "training_config",
        "train_frequency",
        "gamma",
        "alpha_mode",
        "target_entropy",
        "alpha_lr",
        "q_representation",
        "num_q",
        "q_pair_size",
        "q_target_reduction",
        "q_actor_reduction",
        "eval_freq",
        "eval_episodes",
        "total_steps",
        "eval_value",
        "eval_value_samples",
        "eval_value_seed",
        "eval_value_protocols",
    }
)

_REQUIRED_TRAINING_CONFIG_FIELDS = frozenset(
    {
        "learning_rate",
        "buffer_size",
        "learning_starts",
        "batch_size",
        "tau",
        "gamma",
        "train_freq",
        "gradient_steps",
        "ent_coef",
        "target_entropy",
        "target_update_interval",
        "net_arch",
        "actor_net_arch",
        "critic_net_arch",
        "q_representation",
        "num_q",
        "q_pair_size",
        "q_target_reduction",
        "q_actor_reduction",
        "q_num_bins",
        "q_vmin",
        "q_vmax",
        "adam_eps",
        "actor_lr",
        "critic_lr",
        "alpha_lr",
        "actor_betas",
        "critic_betas",
        "alpha_betas",
        "log_std_min",
        "log_std_max",
    }
)

_TRAINING_SEED_PATHS = (
    ("config", "seed"),
    ("run_params", "seed"),
    ("alg_params", "seed"),
    ("seed",),
)
_CSV_SEED_KEYS = ("training_seed", "seed")
_MISSING = object()


@dataclass(frozen=True)
class SeedHistory:
    """One training seed's complete SAC calibration history."""

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
    """Equal-weight aggregate over validated unique training seeds."""

    env_step: np.ndarray
    metrics: Mapping[str, MetricSummary]
    n_seeds: int
    training_seeds: tuple[int, ...]


def _load_pyplot():
    """Import matplotlib lazily and force its headless backend."""

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


def _canonical_algorithm(value: object) -> object:
    value = _canonical_config_value(value)
    if not isinstance(value, str):
        return value
    normalized = value.strip().lower().replace("_", "")
    if normalized in {"sac", "sac/sac", "nativesac", "sacbaseline"}:
        return "SAC/SAC"
    return value


def _canonical_semantic_value(key: str, value: object) -> object:
    if key == "algorithm":
        return _canonical_algorithm(value)
    if key == "training_config":
        canonical = _canonical_config_value(value)
        if not isinstance(canonical, Mapping):
            return canonical
        canonical = dict(canonical)
        for operational_key in ("seed", "device", "verbose"):
            canonical.pop(operational_key, None)
        return canonical
    if key == "train_frequency":
        canonical = _canonical_config_value(value)
        if isinstance(canonical, list):
            if len(canonical) != 2:
                raise ValueError(
                    "train_freq must be a positive integer or [frequency, unit]."
                )
            frequency, unit = canonical
        else:
            frequency, unit = canonical, "step"
        frequency = _nonnegative_int(frequency, name="train_freq frequency")
        if frequency <= 0 or unit not in {"step", "episode"}:
            raise ValueError(
                "train_freq must be a positive integer or [frequency, "
                "'step'|'episode']."
            )
        return [frequency, unit]
    if key in {
        "q_representation",
        "q_target_reduction",
        "q_actor_reduction",
    } and isinstance(value, str):
        return value.lower()
    return _canonical_config_value(value)


def extract_semantic_config(
    config: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Extract and cross-check the scientific SAC config fields from W&B."""

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
                found.append(
                    (
                        ".".join(path),
                        _canonical_semantic_value(semantic_key, value),
                    )
                )
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


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(numeric)


def extract_training_seed(config: Mapping[str, object] | None) -> int | None:
    """Extract one consistent non-negative training seed from a run config."""

    if not config:
        return None
    if not isinstance(config, Mapping):
        raise ValueError("Run config must be a mapping when supplied.")

    found: list[tuple[str, int]] = []
    for path in _TRAINING_SEED_PATHS:
        value = _path_value(config, path)
        if value is not _MISSING:
            found.append(
                (
                    ".".join(path),
                    _nonnegative_int(value, name="Training seed"),
                )
            )
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


def _one_dimensional(
    values: Sequence[object], *, source: str, key: str
) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{source}: {key!r} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{source}: history is empty.")
    return array


def _finite_metric(
    values: Sequence[object], *, source: str, key: str
) -> np.ndarray:
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
    parsed = [
        _nonnegative_int(value, name=f"{source}: {ENV_STEP_KEY!r}")
        for value in raw.tolist()
    ]
    result = np.asarray(parsed, dtype=np.int64)
    unique, counts = np.unique(result, return_counts=True)
    duplicates = unique[counts > 1]
    if duplicates.size:
        rendered = ", ".join(str(int(step)) for step in duplicates)
        raise ValueError(f"{source}: duplicate env_step value(s): {rendered}.")
    return result


def _positive_number(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive finite number.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive finite number.") from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")
    return numeric


def _validate_sac_semantics(
    semantic_config: Mapping[str, object], *, source: str
) -> None:
    missing = sorted(_REQUIRED_SAC_SEMANTICS - set(semantic_config))
    if missing:
        raise ValueError(
            f"{source}: SAC semantic config is missing required field(s): "
            f"{', '.join(missing)}."
        )
    if semantic_config["algorithm"] != "SAC/SAC":
        raise ValueError(
            f"{source}: expected native SAC algorithm 'SAC/SAC', got "
            f"{semantic_config['algorithm']!r}."
        )
    for key in ("recipe", "environment", "task", "observation"):
        value = semantic_config[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{source}: {key} must be a non-empty string.")
    action_repeat = _nonnegative_int(
        semantic_config["action_repeat"], name=f"{source}: action_repeat"
    )
    if action_repeat <= 0:
        raise ValueError(f"{source}: action_repeat must be positive.")
    training_config = semantic_config["training_config"]
    if not isinstance(training_config, Mapping):
        raise ValueError(f"{source}: training_config must be a mapping.")
    missing_training = sorted(
        _REQUIRED_TRAINING_CONFIG_FIELDS - set(training_config)
    )
    if missing_training:
        raise ValueError(
            f"{source}: resolved SAC training config is missing required field(s): "
            f"{', '.join(missing_training)}."
        )
    if semantic_config["eval_value"] is not True:
        raise ValueError(f"{source}: SAC value calibration requires eval_value=true.")
    if semantic_config["eval_value_protocols"] != EXPECTED_PROTOCOLS:
        raise ValueError(
            f"{source}: expected eval_value_protocols={EXPECTED_PROTOCOLS!r}, got "
            f"{semantic_config['eval_value_protocols']!r}."
        )

    gamma = _positive_number(semantic_config["gamma"], name=f"{source}: gamma")
    if gamma > 1.0:
        raise ValueError(f"{source}: gamma must be at most 1.")
    _positive_number(semantic_config["alpha_lr"], name=f"{source}: alpha_lr")
    alpha_mode = semantic_config["alpha_mode"]
    if isinstance(alpha_mode, str):
        if not (
            alpha_mode == "auto" or alpha_mode.startswith("auto_")
        ):
            raise ValueError(
                f"{source}: alpha mode must be numeric or use 'auto[_initial]'."
            )
    else:
        _positive_number(alpha_mode, name=f"{source}: fixed alpha")

    train_frequency = semantic_config["train_frequency"]
    if (
        not isinstance(train_frequency, list)
        or len(train_frequency) != 2
        or train_frequency[1] not in {"step", "episode"}
    ):
        raise ValueError(f"{source}: train_frequency is invalid.")
    if _nonnegative_int(
        train_frequency[0], name=f"{source}: train_frequency"
    ) <= 0:
        raise ValueError(f"{source}: train_frequency must be positive.")

    for key in (
        "num_q",
        "q_pair_size",
        "eval_freq",
        "eval_episodes",
        "eval_value_samples",
    ):
        parsed = _nonnegative_int(semantic_config[key], name=f"{source}: {key}")
        if parsed <= 0:
            raise ValueError(f"{source}: {key} must be positive.")
    _nonnegative_int(
        semantic_config["eval_value_seed"], name=f"{source}: eval_value_seed"
    )
    total_steps = _nonnegative_int(
        semantic_config["total_steps"], name=f"{source}: total_steps"
    )
    eval_freq = int(semantic_config["eval_freq"])
    if total_steps % eval_freq != 0:
        raise ValueError(
            f"{source}: total_steps={total_steps} must be divisible by "
            f"eval_freq={eval_freq}."
        )
    if str(semantic_config["q_representation"]).lower() != "scalar":
        raise ValueError(
            f"{source}: vanilla SAC calibration requires q_representation='scalar'."
        )
    if int(semantic_config["num_q"]) != 2:
        raise ValueError(f"{source}: vanilla SAC calibration requires num_q=2.")
    if int(semantic_config["q_pair_size"]) != 2:
        raise ValueError(
            f"{source}: vanilla SAC calibration requires q_pair_size=2."
        )
    for key in ("q_target_reduction", "q_actor_reduction"):
        if semantic_config[key] != "min_pair":
            raise ValueError(
                f"{source}: vanilla SAC calibration requires {key}='min_pair'."
            )


def _validate_configured_step_grid(
    env_step: np.ndarray,
    semantic_config: Mapping[str, object] | None,
    *,
    source: str,
) -> None:
    if semantic_config is None:
        return
    eval_freq = _nonnegative_int(
        semantic_config["eval_freq"], name=f"{source}: eval_freq"
    )
    total_steps = _nonnegative_int(
        semantic_config["total_steps"], name=f"{source}: total_steps"
    )
    if eval_freq <= 0 or total_steps % eval_freq != 0:
        raise ValueError(
            f"{source}: configured eval_freq={eval_freq} and "
            f"total_steps={total_steps} do not define an exact complete grid."
        )
    expected = np.arange(0, total_steps + 1, eval_freq, dtype=np.int64)
    if np.array_equal(env_step, expected):
        return
    missing = np.setdiff1d(expected, env_step, assume_unique=True)
    unexpected = np.setdiff1d(env_step, expected, assume_unique=True)
    detail = []
    if missing.size:
        detail.append("missing " + ", ".join(str(int(step)) for step in missing))
    if unexpected.size:
        detail.append(
            "unexpected " + ", ".join(str(int(step)) for step in unexpected)
        )
    raise ValueError(
        f"{source}: env_step grid does not match configured step zero through "
        f"{total_steps} at cadence {eval_freq}: {'; '.join(detail)}."
    )


def validate_seed_history(history: SeedHistory) -> SeedHistory:
    """Validate one history and return it sorted by environment step."""

    if not isinstance(history, SeedHistory):
        raise TypeError("Histories must be SeedHistory instances.")
    source = str(history.source).strip()
    if not source:
        raise ValueError("Every history must have a non-empty source name.")
    if not isinstance(history.metrics, Mapping):
        raise ValueError(f"{source}: metrics must be a mapping.")

    missing = [key for key in METRIC_KEYS if key not in history.metrics]
    if missing:
        raise ValueError(
            f"{source}: missing required metric key(s): {', '.join(missing)}."
        )
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

    if history.training_seed is None:
        raise ValueError(
            f"{source}: training seed is required; W&B runs must record one and "
            "offline CSVs must use SEED=PATH or contain a seed column."
        )
    training_seed = _nonnegative_int(
        history.training_seed, name=f"{source}: training seed"
    )
    semantic_config = (
        None
        if history.semantic_config is None
        else _canonical_config_value(history.semantic_config)
    )
    if semantic_config is not None:
        if not isinstance(semantic_config, Mapping):
            raise ValueError(f"{source}: semantic_config must be a mapping or None.")
        _validate_sac_semantics(semantic_config, source=source)
    _validate_configured_step_grid(env_step, semantic_config, source=source)
    return SeedHistory(
        source=source,
        env_step=env_step,
        metrics=metrics,
        semantic_config=semantic_config,
        training_seed=training_seed,
    )


def _semantic_differences(
    left: Mapping[str, object], right: Mapping[str, object]
) -> str:
    differences = []
    for key in sorted(set(left) | set(right)):
        left_value = left.get(key, _MISSING)
        right_value = right.get(key, _MISSING)
        if left_value != right_value:
            left_text = "<missing>" if left_value is _MISSING else repr(left_value)
            right_text = "<missing>" if right_value is _MISSING else repr(right_value)
            differences.append(f"{key}: {left_text} != {right_text}")
    return "; ".join(differences)


def validate_histories(histories: Sequence[SeedHistory]) -> list[SeedHistory]:
    """Validate exact grids, unique seeds, and available SAC semantics."""

    if not histories:
        raise ValueError("At least one run or history CSV is required.")
    normalized = [validate_seed_history(history) for history in histories]

    sources = [history.source for history in normalized]
    if len(set(sources)) != len(sources):
        duplicates = sorted({source for source in sources if sources.count(source) > 1})
        raise ValueError(f"Duplicate input source(s): {', '.join(duplicates)}.")
    seeds = [int(history.training_seed) for history in normalized]
    if len(set(seeds)) != len(seeds):
        duplicates = sorted({seed for seed in seeds if seeds.count(seed) > 1})
        raise ValueError(
            "Training seeds must be unique; duplicate seed(s): "
            + ", ".join(str(seed) for seed in duplicates)
            + "."
        )

    reference_grid = np.asarray(normalized[0].env_step, dtype=np.int64)
    for history in normalized[1:]:
        if not np.array_equal(history.env_step, reference_grid):
            raise ValueError(
                f"{history.source}: env_step grid does not exactly match "
                f"{normalized[0].source}; interpolation is not permitted."
            )

    configured = [history for history in normalized if history.semantic_config is not None]
    if configured and len(configured) != len(normalized):
        missing = [
            history.source
            for history in normalized
            if history.semantic_config is None
        ]
        raise ValueError(
            "Cannot verify SAC semantic compatibility because config metadata is "
            f"missing for: {', '.join(missing)}."
        )
    if configured:
        reference = configured[0]
        for history in configured[1:]:
            if history.semantic_config != reference.semantic_config:
                detail = _semantic_differences(
                    reference.semantic_config, history.semantic_config
                )
                raise ValueError(
                    f"{history.source}: SAC semantic config is incompatible with "
                    f"{reference.source}: {detail}."
                )
    return normalized


def aggregate_histories(histories: Sequence[SeedHistory]) -> AggregatedHistory:
    """Compute equal-weight means and population SDs on the exact grid."""

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
        training_seeds=tuple(sorted(int(history.training_seed) for history in normalized)),
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
    """Build one strict history from W&B-like row mappings."""

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
        raise ValueError(f"{source}: no complete SAC value-diagnostic rows were found.")
    return SeedHistory(
        source=source,
        env_step=columns[ENV_STEP_KEY],
        metrics={key: columns[key] for key in METRIC_KEYS},
        semantic_config=semantic_config,
        training_seed=training_seed,
    )


def _seed_from_csv_rows(
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
    *,
    source: str,
) -> int | None:
    seed_key = next((key for key in _CSV_SEED_KEYS if key in fieldnames), None)
    if seed_key is None:
        return None
    values = {
        _nonnegative_int(row[seed_key], name=f"{source}: {seed_key}")
        for row in rows
        if _has_history_value(row.get(seed_key))
    }
    if not values:
        return None
    if len(values) != 1:
        raise ValueError(f"{source}: {seed_key!r} must be constant within one CSV.")
    return values.pop()


def load_history_csv(
    path: str | Path, *, training_seed: int | None = None
) -> SeedHistory:
    """Load one offline W&B-style CSV, with explicit or column seed metadata."""

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
        rows = list(reader)
    csv_seed = _seed_from_csv_rows(rows, fieldnames, source=str(csv_path))
    if training_seed is not None:
        training_seed = _nonnegative_int(training_seed, name="Training seed")
        if csv_seed is not None and csv_seed != training_seed:
            raise ValueError(
                f"{csv_path}: CSV seed {csv_seed} conflicts with supplied seed "
                f"{training_seed}."
            )
    else:
        training_seed = csv_seed
    return history_from_rows(
        rows,
        source=str(csv_path),
        training_seed=training_seed,
    )


def parse_seed_csv(specification: str) -> SeedHistory:
    """Load ``SEED=PATH`` or a path containing a constant seed column."""

    specification = str(specification).strip()
    if not specification:
        raise ValueError("History CSV specifications must be non-empty.")
    if "=" not in specification:
        return load_history_csv(specification)
    seed_text, path_text = specification.split("=", 1)
    if not seed_text.strip() or not path_text.strip():
        raise ValueError("History CSV specifications must use SEED=PATH.")
    seed = _nonnegative_int(seed_text.strip(), name="History CSV seed")
    return load_history_csv(path_text.strip(), training_seed=seed)


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
    rows = run.scan_history(keys=list(REQUIRED_HISTORY_KEYS), page_size=10_000)
    return history_from_rows(
        rows,
        source=run_path,
        semantic_config=extract_semantic_config(raw_config),
        training_seed=extract_training_seed(raw_config),
    )


def write_aggregate_csv(
    aggregate: AggregatedHistory, path: str | Path
) -> Path:
    """Write every primary and supporting metric with across-seed summaries."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [ENV_STEP_KEY, "n", "training_seeds"]
    for key in METRIC_KEYS:
        fieldnames.extend((f"{key}_mean", f"{key}_std"))
    seeds = ",".join(str(seed) for seed in aggregate.training_seeds)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, step in enumerate(aggregate.env_step):
            row: dict[str, object] = {
                ENV_STEP_KEY: int(step),
                "n": aggregate.n_seeds,
                "training_seeds": seeds,
            }
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
    mc_label: str,
    q_label: str,
    title: str,
) -> None:
    series = (
        (mc_key, mc_label, "#0072B2", "-"),
        (q_key, q_label, "#D55E00", "--"),
    )
    for key, label, color, linestyle in series:
        summary = aggregate.metrics[key]
        axis.plot(
            aggregate.env_step,
            summary.mean,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=f"{label} (n={aggregate.n_seeds})",
        )
        if aggregate.n_seeds > 1:
            axis.fill_between(
                aggregate.env_step,
                summary.mean - summary.std,
                summary.mean + summary.std,
                color=color,
                alpha=0.18,
                linewidth=0.0,
            )
    axis.set_title(title)
    axis.set_xlabel("Agent decisions")
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
    title: str = "Vanilla SAC value calibration",
):
    """Create paper-compatibility and SAC-target-matched panels."""

    pyplot = _load_pyplot()
    figure, axes = pyplot.subplots(1, 2, figsize=(12.0, 4.5), sharex=True)
    _plot_panel(
        axes[0],
        aggregate,
        mc_key=PAPER_MC_KEY,
        q_key=PAPER_Q_KEY,
        mc_label="Deterministic reward MC",
        q_label="Online paper-pair Q",
        title="Paper deterministic — not SAC-target matched",
    )
    _plot_panel(
        axes[1],
        aggregate,
        mc_key=SOFT_MC_BOOTSTRAPPED_KEY,
        q_key=SOFT_Q_MEAN_KEY,
        mc_label="Bootstrapped soft MC (corrected)",
        q_label="Online mean-all Q",
        title="Stochastic soft Bellman — SAC-target matched",
    )
    axes[0].set_ylabel("Discounted value")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def write_artifacts(
    aggregate: AggregatedHistory,
    output_prefix: str | Path,
    *,
    title: str = "Vanilla SAC value calibration",
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write a two-panel PNG/PDF and full-schema aggregate CSV."""

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
            "Aggregate exact-grid vanilla-SAC value-calibration histories "
            "across unique training seeds and write a two-panel PNG/PDF plus "
            "a full-schema aggregate CSV."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="ENTITY/PROJECT/RUN_ID",
        help="W&B run path; repeat once per unique training seed.",
    )
    parser.add_argument(
        "--history-csv",
        action="append",
        default=[],
        metavar="[SEED=]PATH",
        help=(
            "Offline W&B history CSV; repeat once per seed. Supply SEED=PATH "
            "unless the CSV contains one constant seed/training_seed column."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("sac_value_calibration"),
        help="Output path without extension (default: sac_value_calibration).",
    )
    parser.add_argument(
        "--title",
        default="Vanilla SAC value calibration",
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
        histories.extend(parse_seed_csv(specification) for specification in args.history_csv)
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
