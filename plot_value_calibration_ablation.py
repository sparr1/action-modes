#!/usr/bin/env python3
"""Plot the AMBI baseline against 50% outer-policy episode collection.

This companion utility reuses the single-condition value-calibration loaders
and aggregation contract. It requires three paired training seeds per
condition and the complete 21-point 0-to-1M evaluation grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

import plot_value_calibration as value_plot


BASELINE_PROBABILITY = 0.0
FIFTY_PERCENT_PROBABILITY = 0.5
PROBABILITY_KEY = "outer_policy_episode_probability"
EXPECTED_SEED_COUNT = 3
EXPECTED_TRAINING_SEEDS = (55, 56, 57)
EXPECTED_GRID = np.arange(0, 1_000_000 + 1, 50_000, dtype=np.int64)
_CROSS_CONDITION_EXCLUSIONS = frozenset({"recipe", PROBABILITY_KEY})
FIXED_WANDB_SEMANTICS: Mapping[str, object] = {
    "algorithm": "AMBITDMPC2",
    "environment": "DMControl-v0",
    "task": "humanoid-walk",
    "obs": "state",
    "episode_length": 500,
    "episodic": False,
    "discount": 0.99,
    "outer_critic_target": "reward_only",
    "inner_sac_critic_target": "reward_only",
    "outer_q_target_reduction": "min_all",
    "outer_q_actor_reduction": "min_all",
    "q_representation": "distributional",
    "num_q": 5,
    "q_pair_size": 2,
    "utd": 1,
    "eval_freq": 50_000,
    "total_steps": 1_000_000,
    "eval_value": True,
    "eval_value_samples": 100,
    "eval_value_seed": 12_345,
    "eval_value_protocols": [
        "paper_deterministic",
        "stochastic_bellman",
    ],
}


@dataclass(frozen=True)
class AblationAggregate:
    """Paired three-seed aggregates for the baseline and 50% condition."""

    baseline: value_plot.AggregatedHistory
    fifty_percent: value_plot.AggregatedHistory
    training_seeds: tuple[int, ...]


def _finite_probability(value: object, *, source: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{source}: {PROBABILITY_KEY} must be numeric.")
    try:
        probability = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{source}: {PROBABILITY_KEY} must be numeric.") from exc
    if not math.isfinite(probability):
        raise ValueError(f"{source}: {PROBABILITY_KEY} must be finite.")
    return probability


def _validate_probability(
    histories: Sequence[value_plot.SeedHistory],
    *,
    condition: str,
    expected: float,
) -> None:
    """Validate condition probability whenever run config metadata is present."""

    for history in histories:
        config = history.semantic_config
        if config is None:
            # A plain W&B history CSV has no config payload. Its condition is
            # supplied by the condition-specific CLI argument.
            continue
        if PROBABILITY_KEY not in config:
            raise ValueError(
                f"{history.source}: {condition} semantic config is missing "
                f"{PROBABILITY_KEY!r}."
            )
        actual = _finite_probability(config[PROBABILITY_KEY], source=history.source)
        if actual != expected:
            raise ValueError(
                f"{history.source}: {condition} requires {PROBABILITY_KEY}="
                f"{expected}, got {actual}."
            )


def _semantic_value_matches(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, (int, float)):
        if isinstance(actual, (bool, np.bool_)):
            return False
        try:
            numeric = float(actual)
        except (TypeError, ValueError, OverflowError):
            return False
        return math.isfinite(numeric) and numeric == float(expected)
    return actual == expected


def _validate_fixed_wandb_semantics(
    histories: Sequence[value_plot.SeedHistory],
    *,
    condition: str,
) -> None:
    """Enforce the confirmatory campaign contract on available run metadata."""

    for history in histories:
        config = history.semantic_config
        if config is None:
            # Plain history exports do not carry W&B config metadata.
            continue
        missing = [key for key in FIXED_WANDB_SEMANTICS if key not in config]
        if missing:
            raise ValueError(
                f"{history.source}: {condition} W&B metadata is missing fixed "
                f"semantic field(s): {', '.join(missing)}."
            )
        for key, expected in FIXED_WANDB_SEMANTICS.items():
            actual = config[key]
            if not _semantic_value_matches(actual, expected):
                raise ValueError(
                    f"{history.source}: fixed campaign requires {key}={expected!r}, "
                    f"got {actual!r}."
                )


def _validate_condition(
    histories: Sequence[value_plot.SeedHistory],
    *,
    condition: str,
    expected_probability: float,
) -> tuple[list[value_plot.SeedHistory], tuple[int, ...]]:
    normalized = value_plot.validate_histories(histories)
    if len(normalized) != EXPECTED_SEED_COUNT:
        raise ValueError(
            f"{condition} requires exactly {EXPECTED_SEED_COUNT} training seeds; "
            f"got {len(normalized)}."
        )

    missing_seed = [history.source for history in normalized if history.training_seed is None]
    if missing_seed:
        raise ValueError(
            f"{condition} input(s) lack a training seed: {', '.join(missing_seed)}. "
            "W&B runs must record config.seed and CSV inputs must use SEED=PATH."
        )
    seeds = tuple(sorted(int(history.training_seed) for history in normalized))
    if len(set(seeds)) != EXPECTED_SEED_COUNT:
        raise ValueError(f"{condition} training seeds must be unique; got {seeds}.")
    if seeds != EXPECTED_TRAINING_SEEDS:
        raise ValueError(
            f"{condition} requires exact training seeds {EXPECTED_TRAINING_SEEDS}; "
            f"got {seeds}."
        )

    grid = np.asarray(normalized[0].env_step, dtype=np.int64)
    if not np.array_equal(grid, EXPECTED_GRID):
        raise ValueError(
            f"{condition} must use the exact 21-point env_step grid from 0 to "
            "1,000,000 in increments of 50,000; interpolation is not permitted."
        )
    _validate_fixed_wandb_semantics(normalized, condition=condition)
    _validate_probability(
        normalized,
        condition=condition,
        expected=expected_probability,
    )
    return normalized, seeds


def _cross_condition_signature(
    config: Mapping[str, object],
) -> dict[str, object]:
    return {
        key: value
        for key, value in config.items()
        if key not in _CROSS_CONDITION_EXCLUSIONS
    }


def _semantic_differences(
    left: Mapping[str, object], right: Mapping[str, object]
) -> str:
    missing = object()
    differences = []
    for key in sorted(set(left) | set(right)):
        left_value = left.get(key, missing)
        right_value = right.get(key, missing)
        if left_value != right_value:
            left_text = "<missing>" if left_value is missing else repr(left_value)
            right_text = "<missing>" if right_value is missing else repr(right_value)
            differences.append(f"{key}: {left_text} != {right_text}")
    return "; ".join(differences)


def _validate_cross_condition_semantics(
    baseline: Sequence[value_plot.SeedHistory],
    fifty_percent: Sequence[value_plot.SeedHistory],
) -> None:
    baseline_configs = [
        history.semantic_config
        for history in baseline
        if history.semantic_config is not None
    ]
    fifty_configs = [
        history.semantic_config
        for history in fifty_percent
        if history.semantic_config is not None
    ]
    if not baseline_configs or not fifty_configs:
        return
    baseline_signature = _cross_condition_signature(baseline_configs[0])
    fifty_signature = _cross_condition_signature(fifty_configs[0])
    if baseline_signature != fifty_signature:
        detail = _semantic_differences(baseline_signature, fifty_signature)
        raise ValueError(
            "Baseline and 50% conditions have incompatible semantic config "
            f"outside the intended probability intervention: {detail}."
        )


def aggregate_ablation(
    baseline_histories: Sequence[value_plot.SeedHistory],
    fifty_percent_histories: Sequence[value_plot.SeedHistory],
) -> AblationAggregate:
    """Validate and aggregate the paired baseline/50% three-seed experiment."""

    baseline, baseline_seeds = _validate_condition(
        baseline_histories,
        condition="Baseline",
        expected_probability=BASELINE_PROBABILITY,
    )
    fifty_percent, fifty_seeds = _validate_condition(
        fifty_percent_histories,
        condition="50% condition",
        expected_probability=FIFTY_PERCENT_PROBABILITY,
    )
    if baseline_seeds != fifty_seeds:
        raise ValueError(
            "Baseline and 50% conditions must use the same paired training seeds; "
            f"got {baseline_seeds} and {fifty_seeds}."
        )
    _validate_cross_condition_semantics(baseline, fifty_percent)
    return AblationAggregate(
        baseline=value_plot.aggregate_histories(baseline),
        fifty_percent=value_plot.aggregate_histories(fifty_percent),
        training_seeds=baseline_seeds,
    )


def parse_seed_csv(specification: str) -> value_plot.SeedHistory:
    """Load a seed-qualified offline CSV argument of the form ``SEED=PATH``."""

    seed_text, separator, path_text = str(specification).partition("=")
    if not separator or not seed_text.strip() or not path_text.strip():
        raise ValueError(
            f"Invalid seed-qualified CSV {specification!r}; expected SEED=PATH."
        )
    try:
        seed = int(seed_text.strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid seed-qualified CSV {specification!r}; SEED must be a "
            "non-negative integer."
        ) from exc
    if seed < 0 or str(seed) != seed_text.strip():
        raise ValueError(
            f"Invalid seed-qualified CSV {specification!r}; SEED must be a "
            "non-negative integer."
        )
    history = value_plot.load_history_csv(Path(path_text.strip()))
    return value_plot.validate_seed_history(
        replace(
            history,
            source=f"seed={seed}:{history.source}",
            training_seed=seed,
        )
    )


def _validate_aggregate(aggregate: AblationAggregate) -> None:
    if len(aggregate.training_seeds) != EXPECTED_SEED_COUNT:
        raise ValueError("Ablation aggregate must contain exactly three training seeds.")
    if aggregate.baseline.n_seeds != EXPECTED_SEED_COUNT or (
        aggregate.fifty_percent.n_seeds != EXPECTED_SEED_COUNT
    ):
        raise ValueError("Both ablation conditions must aggregate exactly three seeds.")
    if not np.array_equal(aggregate.baseline.env_step, EXPECTED_GRID) or not (
        np.array_equal(aggregate.fifty_percent.env_step, EXPECTED_GRID)
    ):
        raise ValueError("Ablation aggregate must use the exact 21-point env_step grid.")


def _plot_condition_panel(
    axis,
    aggregate: value_plot.AggregatedHistory,
    *,
    mc_key: str,
    q_key: str,
    condition_label: str,
) -> None:
    series = (
        (mc_key, "Monte Carlo value", "#0072B2", "-"),
        (q_key, "Q estimate", "#D55E00", "--"),
    )
    for key, label, color, linestyle in series:
        summary = aggregate.metrics[key]
        axis.plot(
            aggregate.env_step,
            summary.mean,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
        axis.fill_between(
            aggregate.env_step,
            summary.mean - summary.std,
            summary.mean + summary.std,
            color=color,
            alpha=0.18,
            linewidth=0.0,
        )
    axis.text(
        0.02,
        0.96,
        condition_label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )
    axis.grid(True, alpha=0.25, linewidth=0.7)
    axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))


def plot_ablation(
    aggregate: AblationAggregate,
    *,
    title: str = "AMBI value calibration: baseline vs 50% outer-policy episodes",
):
    """Create a 2x2 condition-by-protocol calibration figure."""

    _validate_aggregate(aggregate)
    pyplot = value_plot._load_pyplot()
    figure, axes = pyplot.subplots(
        2,
        2,
        figsize=(12.0, 8.0),
        sharex=True,
        sharey="col",
    )
    conditions = (
        (aggregate.baseline, "Baseline (p=0.0)"),
        (aggregate.fifty_percent, "50% outer-policy episodes (p=0.5)"),
    )
    protocols = (
        (
            value_plot.DETERMINISTIC_MC_KEY,
            value_plot.DETERMINISTIC_Q_KEY,
            "Paper deterministic protocol",
        ),
        (
            value_plot.STOCHASTIC_MC_KEY,
            value_plot.STOCHASTIC_Q_KEY,
            "Stochastic Bellman protocol",
        ),
    )
    for row, (condition, condition_label) in enumerate(conditions):
        for column, (mc_key, q_key, protocol_label) in enumerate(protocols):
            axis = axes[row, column]
            _plot_condition_panel(
                axis,
                condition,
                mc_key=mc_key,
                q_key=q_key,
                condition_label=condition_label,
            )
            if row == 0:
                axis.set_title(protocol_label)
            if row == 1:
                axis.set_xlabel("Environment steps")
        axes[row, 0].set_ylabel("Discounted value")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        title="Equal-weight mean ± 1 population SD across training seeds (n=3)",
    )
    figure.suptitle(title)
    figure.tight_layout(rect=(0.0, 0.10, 1.0, 0.95))
    return figure


def write_combined_csv(aggregate: AblationAggregate, path: str | Path) -> Path:
    """Write both condition aggregates in one long-form CSV."""

    _validate_aggregate(aggregate)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        PROBABILITY_KEY,
        "training_seeds",
        value_plot.ENV_STEP_KEY,
        "n",
    ]
    for key in value_plot.METRIC_KEYS:
        fieldnames.extend((f"{key}_mean", f"{key}_std"))
    conditions = (
        ("baseline", BASELINE_PROBABILITY, aggregate.baseline),
        (
            "outer_policy_50_percent",
            FIFTY_PERCENT_PROBABILITY,
            aggregate.fifty_percent,
        ),
    )
    seed_text = ",".join(str(seed) for seed in aggregate.training_seeds)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition_name, probability, condition in conditions:
            for index, step in enumerate(condition.env_step):
                row: dict[str, object] = {
                    "condition": condition_name,
                    PROBABILITY_KEY: format(probability, ".1f"),
                    "training_seeds": seed_text,
                    value_plot.ENV_STEP_KEY: int(step),
                    "n": condition.n_seeds,
                }
                for key in value_plot.METRIC_KEYS:
                    row[f"{key}_mean"] = format(
                        float(condition.metrics[key].mean[index]), ".17g"
                    )
                    row[f"{key}_std"] = format(
                        float(condition.metrics[key].std[index]), ".17g"
                    )
                writer.writerow(row)
    return output


def write_artifacts(
    aggregate: AblationAggregate,
    output_prefix: str | Path,
    *,
    title: str = "AMBI value calibration: baseline vs 50% outer-policy episodes",
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write the 2x2 PNG/PDF and combined aggregate CSV."""

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
    figure = plot_ablation(aggregate, title=title)
    pyplot = value_plot._load_pyplot()
    try:
        figure.savefig(outputs["png"], dpi=200, bbox_inches="tight")
        figure.savefig(outputs["pdf"], bbox_inches="tight")
    finally:
        pyplot.close(figure)
    write_combined_csv(aggregate, outputs["csv"])
    return outputs


def _make_wandb_api():
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Loading W&B runs requires the optional wandb dependency."
        ) from exc
    return wandb.Api()


def _load_condition_inputs(
    run_paths: Sequence[str],
    csv_specs: Sequence[str],
    *,
    api: object | None,
) -> list[value_plot.SeedHistory]:
    histories = [
        value_plot.load_wandb_history(run_path, api=api) for run_path in run_paths
    ]
    histories.extend(parse_seed_csv(specification) for specification in csv_specs)
    return histories


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare three-seed AMBI value calibration for baseline p=0.0 and "
            "50% outer-policy episode collection p=0.5."
        )
    )
    parser.add_argument(
        "--baseline-run",
        action="append",
        default=[],
        metavar="ENTITY/PROJECT/RUN_ID",
        help="Baseline W&B run path; repeat for all three seeds.",
    )
    parser.add_argument(
        "--fifty-run",
        "--intervention-run",
        dest="fifty_run",
        action="append",
        default=[],
        metavar="ENTITY/PROJECT/RUN_ID",
        help="p=0.5 W&B run path; repeat for all three seeds.",
    )
    parser.add_argument(
        "--baseline-history-csv",
        action="append",
        default=[],
        metavar="SEED=PATH",
        help="Seed-qualified baseline history CSV; repeat for all three seeds.",
    )
    parser.add_argument(
        "--fifty-history-csv",
        "--intervention-history-csv",
        dest="fifty_history_csv",
        action="append",
        default=[],
        metavar="SEED=PATH",
        help="Seed-qualified p=0.5 history CSV; repeat for all three seeds.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("value_calibration_ablation"),
        help="Output path without extension (default: value_calibration_ablation).",
    )
    parser.add_argument(
        "--title",
        default="AMBI value calibration: baseline vs 50% outer-policy episodes",
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
    if not args.baseline_run and not args.baseline_history_csv:
        parser.error("provide baseline inputs")
    if not args.fifty_run and not args.fifty_history_csv:
        parser.error("provide 50% condition inputs")
    try:
        api = _make_wandb_api() if args.baseline_run or args.fifty_run else None
        baseline = _load_condition_inputs(
            args.baseline_run,
            args.baseline_history_csv,
            api=api,
        )
        fifty_percent = _load_condition_inputs(
            args.fifty_run,
            args.fifty_history_csv,
            api=api,
        )
        aggregate = aggregate_ablation(baseline, fifty_percent)
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
