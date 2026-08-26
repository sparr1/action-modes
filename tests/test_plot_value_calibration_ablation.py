import csv
from pathlib import Path

import numpy as np
import pytest

import plot_value_calibration as value_plot
import plot_value_calibration_ablation as ablation_plot


SEEDS = (55, 56, 57)
GRID = np.arange(0, 1_000_000 + 1, 50_000, dtype=np.int64)


def _semantic(probability, *, recipe, samples=100):
    return {
        "algorithm": "AMBITDMPC2",
        "recipe": recipe,
        "environment": "DMControl-v0",
        "task": "humanoid-walk",
        "obs": "state",
        "episode_length": 500,
        "episodic": False,
        "discount": 0.99,
        "discount_min": 0.99,
        "discount_max": 0.99,
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
        "eval_value_samples": samples,
        "eval_value_seed": 12_345,
        "eval_value_protocols": [
            "paper_deterministic",
            "stochastic_bellman",
        ],
        ablation_plot.PROBABILITY_KEY: probability,
    }


def _history(
    condition,
    seed,
    *,
    probability,
    condition_offset=0.0,
    grid=GRID,
    config=None,
):
    seed_offset = 2.0 * (seed - SEEDS[0])
    index = np.arange(len(grid), dtype=np.float64)
    metric_offsets = {
        value_plot.DETERMINISTIC_MC_KEY: 0.0,
        value_plot.DETERMINISTIC_Q_KEY: 100.0,
        value_plot.STOCHASTIC_MC_KEY: 200.0,
        value_plot.STOCHASTIC_Q_KEY: 300.0,
    }
    metrics = {
        key: index + metric_offset + condition_offset + seed_offset
        for key, metric_offset in metric_offsets.items()
    }
    if config is None:
        config = _semantic(probability, recipe=f"{condition}-recipe")
    return value_plot.SeedHistory(
        source=f"{condition}-seed-{seed}",
        env_step=np.asarray(grid),
        metrics=metrics,
        semantic_config=config,
        training_seed=seed,
    )


def _condition(condition, probability, *, condition_offset=0.0, seeds=SEEDS, **kwargs):
    return [
        _history(
            condition,
            seed,
            probability=probability,
            condition_offset=condition_offset,
            **kwargs,
        )
        for seed in seeds
    ]


def _paired():
    return (
        _condition("baseline", 0.0),
        _condition("fifty", 0.5, condition_offset=10.0),
    )


def _write_history_csv(path: Path, history: value_plot.SeedHistory):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[value_plot.ENV_STEP_KEY, *value_plot.METRIC_KEYS],
        )
        writer.writeheader()
        for index, step in enumerate(history.env_step):
            writer.writerow(
                {
                    value_plot.ENV_STEP_KEY: int(step),
                    **{
                        key: float(history.metrics[key][index])
                        for key in value_plot.METRIC_KEYS
                    },
                }
            )


def test_three_seed_ablation_uses_equal_weight_mean_and_population_sd():
    baseline, fifty = _paired()
    aggregate = ablation_plot.aggregate_ablation(baseline, fifty)

    assert aggregate.training_seeds == SEEDS
    assert aggregate.baseline.n_seeds == 3
    assert aggregate.fifty_percent.n_seeds == 3
    np.testing.assert_array_equal(aggregate.baseline.env_step, GRID)
    expected_std = np.sqrt(8.0 / 3.0)
    for key in value_plot.METRIC_KEYS:
        np.testing.assert_allclose(
            aggregate.baseline.metrics[key].mean,
            np.asarray(baseline[0].metrics[key]) + 2.0,
        )
        np.testing.assert_allclose(aggregate.baseline.metrics[key].std, expected_std)
        np.testing.assert_allclose(
            aggregate.fifty_percent.metrics[key].mean,
            np.asarray(fifty[0].metrics[key]) + 2.0,
        )
        np.testing.assert_allclose(
            aggregate.fifty_percent.metrics[key].std,
            expected_std,
        )


@pytest.mark.parametrize("condition", ["baseline", "fifty"])
def test_each_condition_requires_exactly_three_histories(condition):
    baseline, fifty = _paired()
    if condition == "baseline":
        baseline = baseline[:2]
    else:
        fifty = fifty[:2]

    label = "Baseline" if condition == "baseline" else "50% condition"
    with pytest.raises(ValueError, match=f"{label} requires exactly 3"):
        ablation_plot.aggregate_ablation(baseline, fifty)


def test_training_seeds_must_be_present_unique_and_paired():
    baseline, fifty = _paired()
    baseline[0] = value_plot.SeedHistory(
        source=baseline[0].source,
        env_step=baseline[0].env_step,
        metrics=baseline[0].metrics,
        semantic_config=baseline[0].semantic_config,
        training_seed=None,
    )
    with pytest.raises(ValueError, match="lack a training seed"):
        ablation_plot.aggregate_ablation(baseline, fifty)

    baseline, fifty = _paired()
    baseline[1] = value_plot.SeedHistory(
        source=baseline[1].source,
        env_step=baseline[1].env_step,
        metrics=baseline[1].metrics,
        semantic_config=baseline[1].semantic_config,
        training_seed=55,
    )
    with pytest.raises(ValueError, match="training seeds must be unique"):
        ablation_plot.aggregate_ablation(baseline, fifty)

    baseline, _ = _paired()
    mismatched_fifty = _condition(
        "fifty",
        0.5,
        condition_offset=10.0,
        seeds=(55, 56, 58),
    )
    with pytest.raises(ValueError, match="requires exact training seeds.*55, 56, 57"):
        ablation_plot.aggregate_ablation(baseline, mismatched_fifty)


def test_every_history_requires_the_exact_21_point_grid():
    shorter_grid = GRID[:-1]
    baseline = _condition("baseline", 0.0, grid=shorter_grid)
    fifty = _condition("fifty", 0.5, condition_offset=10.0)

    with pytest.raises(ValueError, match="grid does not match configured|exact 21-point"):
        ablation_plot.aggregate_ablation(baseline, fifty)


def test_condition_probabilities_must_be_exactly_zero_and_one_half():
    wrong_baseline = _condition("baseline", 0.5)
    fifty = _condition("fifty", 0.5, condition_offset=10.0)
    with pytest.raises(ValueError, match="Baseline requires.*=0.0, got 0.5"):
        ablation_plot.aggregate_ablation(wrong_baseline, fifty)

    baseline = _condition("baseline", 0.0)
    missing_probability = _semantic(0.5, recipe="fifty-recipe")
    missing_probability.pop(ablation_plot.PROBABILITY_KEY)
    fifty = _condition(
        "fifty",
        0.5,
        condition_offset=10.0,
        config=missing_probability,
    )
    with pytest.raises(ValueError, match="semantic config is missing.*outer_policy"):
        ablation_plot.aggregate_ablation(baseline, fifty)


@pytest.mark.parametrize(
    "key, bad_value",
    [
        ("utd", 2),
        ("obs", "rgb"),
        ("outer_critic_target", "entropy_augmented"),
        ("inner_sac_critic_target", "entropy_augmented"),
        ("q_representation", "scalar"),
        ("num_q", 2),
        ("q_pair_size", 3),
        ("discount", 0.95),
        ("eval_value_samples", 99),
        ("eval_value_seed", 1),
        ("eval_value_protocols", ["stochastic_bellman", "paper_deterministic"]),
    ],
)
def test_available_metadata_must_match_fixed_campaign_contract(key, bad_value):
    baseline_config = _semantic(0.0, recipe="baseline-recipe")
    baseline_config[key] = bad_value
    baseline = _condition("baseline", 0.0, config=baseline_config)
    fifty = _condition("fifty", 0.5, condition_offset=10.0)

    with pytest.raises(ValueError, match=f"fixed campaign requires {key}="):
        ablation_plot.aggregate_ablation(baseline, fifty)


def test_available_metadata_must_include_utd():
    baseline_config = _semantic(0.0, recipe="baseline-recipe")
    baseline_config.pop("utd")
    baseline = _condition("baseline", 0.0, config=baseline_config)
    fifty = _condition("fifty", 0.5, condition_offset=10.0)

    with pytest.raises(ValueError, match="missing fixed semantic field.*utd"):
        ablation_plot.aggregate_ablation(baseline, fifty)


def test_seed_qualified_inputs_require_exact_campaign_seed_set():
    baseline = _condition("baseline", 0.0, seeds=(54, 55, 56))
    fifty = _condition(
        "fifty",
        0.5,
        condition_offset=10.0,
        seeds=(54, 55, 56),
    )

    with pytest.raises(ValueError, match="requires exact training seeds.*55, 56, 57"):
        ablation_plot.aggregate_ablation(baseline, fifty)


def test_cross_condition_semantics_must_match_except_recipe_and_probability():
    baseline, fifty = _paired()
    assert ablation_plot.aggregate_ablation(baseline, fifty).training_seeds == SEEDS

    fifty_config = _semantic(0.5, recipe="fifty-recipe")
    fifty_config["discount_min"] = 0.95
    incompatible = _condition(
        "fifty",
        0.5,
        condition_offset=10.0,
        config=fifty_config,
    )
    with pytest.raises(ValueError, match="incompatible semantic config.*discount_min"):
        ablation_plot.aggregate_ablation(baseline, incompatible)


@pytest.mark.parametrize(
    "specification",
    ["history.csv", "=history.csv", "seed=history.csv", "-1=history.csv"],
)
def test_seed_qualified_csv_syntax_is_strict(specification):
    with pytest.raises(ValueError, match="SEED=PATH|SEED must be"):
        ablation_plot.parse_seed_csv(specification)


def test_seed_qualified_csv_loads_seed_and_history(tmp_path):
    path = tmp_path / "seed=55.csv"
    history = _history("offline", 55, probability=0.0)
    _write_history_csv(path, history)

    loaded = ablation_plot.parse_seed_csv(f"55={path}")

    assert loaded.training_seed == 55
    assert loaded.semantic_config is None
    np.testing.assert_array_equal(loaded.env_step, GRID)


def test_wandb_loader_carries_training_seed_and_probability():
    rows = []
    history = _history("wandb", 55, probability=0.0)
    for index, step in enumerate(GRID):
        rows.append(
            {
                value_plot.ENV_STEP_KEY: int(step),
                **{
                    key: float(history.metrics[key][index])
                    for key in value_plot.METRIC_KEYS
                },
            }
        )

    class Run:
        config = {
            "algorithm": "AMBITDMPC2",
            "run_params": {
                "name": "baseline-recipe",
                "env": "DMControl-v0",
                "seed": 55,
                "total_steps": 1_000_000,
            },
            "config": {
                "seed": 55,
                "eval_freq": 50_000,
                "steps": 1_000_000,
                "outer_policy_episode_probability": 0.0,
            },
        }

        def scan_history(self, *, keys, page_size):
            assert keys == list(value_plot.REQUIRED_HISTORY_KEYS)
            assert page_size == 10_000
            return iter(rows)

    class Api:
        def run(self, path):
            assert path == "entity/project/run"
            return Run()

    loaded = value_plot.load_wandb_history("entity/project/run", api=Api())

    assert loaded.training_seed == 55
    assert loaded.semantic_config[ablation_plot.PROBABILITY_KEY] == 0.0


def test_combined_csv_contains_both_21_point_aggregates(tmp_path):
    aggregate = ablation_plot.aggregate_ablation(*_paired())
    output = ablation_plot.write_combined_csv(aggregate, tmp_path / "combined.csv")

    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 42
    assert [row["condition"] for row in rows[:21]] == ["baseline"] * 21
    assert [row["condition"] for row in rows[21:]] == [
        "outer_policy_50_percent"
    ] * 21
    assert {row[ablation_plot.PROBABILITY_KEY] for row in rows} == {"0.0", "0.5"}
    assert {row["training_seeds"] for row in rows} == {"55,56,57"}
    assert {int(row["n"]) for row in rows} == {3}
    assert float(rows[0][f"{value_plot.DETERMINISTIC_MC_KEY}_mean"]) == pytest.approx(
        2.0
    )
    assert float(rows[21][f"{value_plot.DETERMINISTIC_MC_KEY}_mean"]) == pytest.approx(
        12.0
    )


def test_2x2_png_pdf_and_combined_csv_artifacts(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    aggregate = ablation_plot.aggregate_ablation(*_paired())
    figure = ablation_plot.plot_ablation(aggregate)
    try:
        assert matplotlib.get_backend().lower() == "agg"
        assert len(figure.axes) == 4
        assert all(len(axis.lines) == 2 for axis in figure.axes)
        assert all(len(axis.collections) == 2 for axis in figure.axes)
        assert len(figure.legends) == 1
        assert "population SD" in figure.legends[0].get_title().get_text()
        assert "n=3" in figure.legends[0].get_title().get_text()
    finally:
        plt.close(figure)

    outputs = ablation_plot.write_artifacts(aggregate, tmp_path / "ablation")
    assert outputs["png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert outputs["pdf"].read_bytes().startswith(b"%PDF")
    with outputs["csv"].open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 42


def test_offline_cli_accepts_three_seed_qualified_csvs_per_condition(
    tmp_path, capsys
):
    pytest.importorskip("matplotlib")
    arguments = []
    for condition, probability, option, offset in (
        ("baseline", 0.0, "--baseline-history-csv", 0.0),
        ("fifty", 0.5, "--fifty-history-csv", 10.0),
    ):
        for seed in SEEDS:
            path = tmp_path / f"{condition}-{seed}.csv"
            _write_history_csv(
                path,
                _history(
                    condition,
                    seed,
                    probability=probability,
                    condition_offset=offset,
                ),
            )
            arguments.extend((option, f"{seed}={path}"))
    prefix = tmp_path / "cli-ablation"
    arguments.extend(("--output-prefix", str(prefix)))

    assert ablation_plot.main(arguments) == 0
    assert Path(f"{prefix}.png").is_file()
    assert Path(f"{prefix}.pdf").is_file()
    assert Path(f"{prefix}.csv").is_file()
    assert '"csv"' in capsys.readouterr().out
