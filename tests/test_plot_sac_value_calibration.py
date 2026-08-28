import csv
from pathlib import Path

import numpy as np
import pytest

import plot_sac_value_calibration as sac_plot
from plot_sac_value_calibration import (
    ENV_STEP_KEY,
    METRIC_KEYS,
    PAPER_MC_KEY,
    PAPER_Q_KEY,
    REWARD_MC_KEY,
    SOFT_MC_BOOTSTRAPPED_KEY,
    SOFT_MC_FINITE_KEY,
    SOFT_Q_MEAN_KEY,
    SOFT_Q_MIN_KEY,
    SOFT_TRUNCATION_TAIL_KEY,
    SeedHistory,
    aggregate_histories,
    extract_semantic_config,
    load_history_csv,
    main,
    parse_seed_csv,
    plot_aggregate,
    write_artifacts,
)


def _training_config(**overrides):
    config = {
        "learning_rate": 1e-3,
        "buffer_size": 1_000_000,
        "learning_starts": 1_000,
        "batch_size": 512,
        "tau": 0.01,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto_0.1",
        "target_entropy": "auto",
        "target_update_interval": 2,
        "net_arch": [1024, 1024],
        "actor_net_arch": [1024, 1024],
        "critic_net_arch": [1024, 1024],
        "q_representation": "scalar",
        "num_q": 2,
        "q_pair_size": 2,
        "q_target_reduction": "min_pair",
        "q_actor_reduction": "min_pair",
        "q_num_bins": 101,
        "q_vmin": -10.0,
        "q_vmax": 10.0,
        "adam_eps": 1e-8,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "alpha_lr": 1e-4,
        "actor_betas": [0.9, 0.999],
        "critic_betas": [0.9, 0.999],
        "alpha_betas": [0.5, 0.999],
        "log_std_min": -5.0,
        "log_std_max": 2.0,
    }
    config.update(overrides)
    return config


def _semantic_config(**overrides):
    config = {
        "recipe": "sac_humanoid_walk_tdmpc_table5_value_calibration",
        "algorithm": "SAC/SAC",
        "environment": "DMControl-v0",
        "task": "humanoid-walk",
        "observation": "state",
        "action_repeat": 2,
        "training_config": _training_config(),
        "train_frequency": [1, "step"],
        "gamma": 0.99,
        "alpha_mode": "auto_0.1",
        "target_entropy": "auto",
        "alpha_lr": 1e-4,
        "q_representation": "scalar",
        "num_q": 2,
        "q_pair_size": 2,
        "q_target_reduction": "min_pair",
        "q_actor_reduction": "min_pair",
        "q_num_bins": 101,
        "q_vmin": -10.0,
        "q_vmax": 10.0,
        "eval_freq": 50_000,
        "eval_episodes": 1,
        "total_steps": 100_000,
        "eval_value": True,
        "eval_value_samples": 100,
        "eval_value_seed": 12_345,
        "eval_value_protocols": [
            "paper_deterministic",
            "stochastic_soft_bellman",
        ],
    }
    config.update(overrides)
    return config


def _metrics(offset=0.0, points=3):
    metrics = {}
    base = np.arange(1, points + 1, dtype=np.float64) + float(offset)
    for index, key in enumerate(METRIC_KEYS, start=1):
        metrics[key] = base + 0.01 * index
    # Keep count/fraction/dispersion-like fields physically valid in fixtures.
    metrics[sac_plot.VALUE_SAMPLES_KEY] = np.full(points, 100.0)
    metrics[sac_plot.SOFT_TRUNCATION_FRACTION_KEY] = np.linspace(
        0.0, 1.0, points
    )
    return metrics


_DEFAULT_CONFIG = object()


def _history(
    source,
    *,
    seed=55,
    steps=(0, 50_000, 100_000),
    offset=0.0,
    config=_DEFAULT_CONFIG,
    metrics=None,
):
    return SeedHistory(
        source=source,
        env_step=steps,
        metrics=_metrics(offset, len(steps)) if metrics is None else metrics,
        semantic_config=(
            _semantic_config() if config is _DEFAULT_CONFIG else config
        ),
        training_seed=seed,
    )


def _write_history_csv(path: Path, history: SeedHistory, *, include_seed=False):
    fieldnames = [ENV_STEP_KEY, *METRIC_KEYS]
    if include_seed:
        fieldnames.append("training_seed")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, step in enumerate(history.env_step):
            row = {
                ENV_STEP_KEY: step,
                **{key: history.metrics[key][index] for key in METRIC_KEYS},
            }
            if include_seed:
                row["training_seed"] = history.training_seed
            writer.writerow(row)


def test_metric_schema_contains_primary_and_all_requested_supporting_values():
    assert METRIC_KEYS == (
        "eval/mc_value",
        "eval/mc_value_std",
        "eval/q_value",
        "eval/q_value_std",
        "eval/q_minus_mc",
        "eval/stochastic_reward_mc_value",
        "eval/stochastic_reward_mc_value_std",
        "eval/stochastic_soft_mc_finite_value",
        "eval/stochastic_soft_mc_finite_value_std",
        "eval/stochastic_soft_mc_bootstrapped_value",
        "eval/stochastic_soft_mc_bootstrapped_value_std",
        "eval/stochastic_soft_truncation_tail",
        "eval/stochastic_soft_truncation_fraction",
        "eval/stochastic_soft_q_mean_all",
        "eval/stochastic_soft_q_mean_all_std",
        "eval/stochastic_soft_q_min_all",
        "eval/stochastic_soft_q_head_std",
        "eval/stochastic_soft_q_minus_mc_bootstrapped_mean_all",
        "eval/stochastic_soft_q_rmse_bootstrapped_mean_all",
        "eval/stochastic_soft_q_minus_mc_bootstrapped_min_all",
        "eval/stochastic_soft_q_rmse_bootstrapped_min_all",
        "eval/stochastic_soft_alpha",
        "eval/value_samples",
        "time/value_eval_seconds",
    )
    assert sac_plot.PLOT_METRIC_KEYS == (
        PAPER_MC_KEY,
        PAPER_Q_KEY,
        SOFT_MC_BOOTSTRAPPED_KEY,
        SOFT_Q_MEAN_KEY,
    )


def test_one_seed_preserves_all_metrics_and_uses_zero_population_sd():
    history = _history("seed-55")
    aggregate = aggregate_histories([history])

    assert aggregate.n_seeds == 1
    assert aggregate.training_seeds == (55,)
    np.testing.assert_array_equal(aggregate.env_step, [0, 50_000, 100_000])
    for key in METRIC_KEYS:
        np.testing.assert_allclose(aggregate.metrics[key].mean, history.metrics[key])
        np.testing.assert_array_equal(aggregate.metrics[key].std, [0.0, 0.0, 0.0])


def test_three_unique_seeds_are_equal_weighted_with_population_sd():
    histories = [
        _history("seed-55", seed=55, offset=0.0),
        _history("seed-56", seed=56, offset=2.0),
        _history("seed-57", seed=57, offset=4.0),
    ]
    aggregate = aggregate_histories(histories)
    expected_std = np.sqrt(8.0 / 3.0)

    assert aggregate.training_seeds == (55, 56, 57)
    for key in METRIC_KEYS:
        if key in {
            sac_plot.VALUE_SAMPLES_KEY,
            sac_plot.SOFT_TRUNCATION_FRACTION_KEY,
        }:
            continue
        np.testing.assert_allclose(
            aggregate.metrics[key].mean,
            np.asarray(histories[0].metrics[key]) + 2.0,
        )
        np.testing.assert_allclose(aggregate.metrics[key].std, expected_std)


def test_missing_supporting_metric_is_rejected():
    metrics = dict(_history("complete").metrics)
    metrics.pop(SOFT_TRUNCATION_TAIL_KEY)
    with pytest.raises(ValueError, match="missing required metric.*truncation_tail"):
        aggregate_histories([_history("missing", metrics=metrics)])


def test_duplicate_steps_and_mismatched_exact_grids_are_rejected():
    with pytest.raises(ValueError, match="duplicate env_step.*50000"):
        aggregate_histories(
            [_history("duplicate", steps=(0, 50_000, 50_000))]
        )

    with pytest.raises(ValueError, match="configured step zero.*missing 100000"):
        aggregate_histories(
            [_history("missing-final", steps=(0, 50_000))]
        )

    with pytest.raises(ValueError, match="does not exactly match.*interpolation"):
        aggregate_histories(
            [
                _history("seed-55", seed=55, config=None),
                _history(
                    "seed-56",
                    seed=56,
                    steps=(0, 25_000, 100_000),
                    config=None,
                ),
            ]
        )


def test_training_seeds_are_required_and_unique():
    with pytest.raises(ValueError, match="training seed is required"):
        aggregate_histories([_history("missing-seed", seed=None)])
    with pytest.raises(ValueError, match="seeds must be unique.*55"):
        aggregate_histories(
            [_history("first", seed=55), _history("second", seed=55)]
        )


def test_wandb_config_extraction_covers_sac_scientific_semantics():
    raw = {
        "run_params": {
            "name": "sac_humanoid_walk_tdmpc_table5_value_calibration",
            "alg": "SAC/SAC",
            "env": "DMControl-v0",
            "seed": 55,
            "total_steps": 100_000,
            "resolved_runtime": {
                "observation": {"task": "humanoid-walk", "action_repeat": 2}
            },
        },
        "experiment_params": {
            "env_params": {"task": "humanoid-walk", "obs": "state"}
        },
        "alg_params": {
            "train_freq": 1,
            "gamma": 0.99,
            "ent_coef": "auto_0.1",
            "target_entropy": "auto",
            "alpha_lr": 1e-4,
            "q_representation": "scalar",
            "num_q": 2,
            "q_pair_size": 2,
            "q_target_reduction": "min_pair",
            "q_actor_reduction": "min_pair",
            "eval_freq": 50_000,
            "eval_episodes": 1,
            "eval_value": True,
            "eval_value_samples": 100,
            "eval_value_seed": 12_345,
            "eval_value_protocols": [
                "paper_deterministic",
                "stochastic_soft_bellman",
            ],
        },
        "config": {
            **_training_config(),
            "seed": 55,
            "device": "cuda",
            "verbose": 1,
        },
    }

    semantic = extract_semantic_config(raw)
    assert semantic == _semantic_config()
    assert sac_plot.extract_training_seed(raw) == 55

    raw["alg_params"]["train_freq"] = [1, "episode"]
    raw["alg_params"]["eval_episodes"] = 10
    changed = extract_semantic_config(raw)
    assert changed["train_frequency"] == [1, "episode"]
    assert changed["eval_episodes"] == 10


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("recipe", "different_recipe"),
        ("gamma", 0.95),
        ("alpha_mode", "auto_1.0"),
        ("observation", "rgb"),
        ("action_repeat", 4),
        ("train_frequency", [1, "episode"]),
        ("eval_episodes", 10),
        ("eval_value_samples", 64),
        ("eval_value_seed", 999),
        ("task", "walker-walk"),
    ],
)
def test_incompatible_sac_semantics_are_rejected(field, replacement):
    changed = _semantic_config(**{field: replacement})
    with pytest.raises(ValueError, match=f"incompatible.*{field}"):
        aggregate_histories(
            [
                _history("seed-55", seed=55),
                _history("seed-56", seed=56, config=changed),
            ]
        )


def test_incompatible_resolved_training_configs_are_rejected():
    changed = _semantic_config(training_config=_training_config(net_arch=[256, 256]))
    with pytest.raises(ValueError, match="incompatible.*training_config"):
        aggregate_histories(
            [
                _history("seed-55", seed=55),
                _history("seed-56", seed=56, config=changed),
            ]
        )


def test_non_sac_and_wrong_protocol_configs_fail_closed():
    with pytest.raises(ValueError, match="expected native SAC"):
        aggregate_histories(
            [_history("ambi", config=_semantic_config(algorithm="AMBITDMPC2"))]
        )
    with pytest.raises(ValueError, match="expected eval_value_protocols"):
        aggregate_histories(
            [
                _history(
                    "wrong-protocol",
                    config=_semantic_config(
                        eval_value_protocols=["paper_deterministic"]
                    ),
                )
            ]
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("q_representation", "distributional", "q_representation='scalar'"),
        ("num_q", 5, "num_q=2"),
        ("q_pair_size", 1, "q_pair_size=2"),
        ("q_target_reduction", "mean_pair", "q_target_reduction='min_pair'"),
        ("q_actor_reduction", "mean_pair", "q_actor_reduction='min_pair'"),
    ],
)
def test_single_run_must_use_vanilla_twin_min_pair_semantics(
    field, replacement, message
):
    with pytest.raises(ValueError, match=message):
        aggregate_histories(
            [
                _history(
                    "non-vanilla",
                    config=_semantic_config(**{field: replacement}),
                )
            ]
        )


def test_mixed_presence_of_wandb_semantic_metadata_is_rejected():
    with pytest.raises(ValueError, match="metadata is missing for.*offline"):
        aggregate_histories(
            [
                _history("wandb", seed=55),
                _history("offline", seed=56, config=None),
            ]
        )


def test_seed_qualified_and_seed_column_csv_inputs(tmp_path):
    no_seed = tmp_path / "no-seed.csv"
    _write_history_csv(no_seed, _history("source", seed=55))
    loaded = parse_seed_csv(f"55={no_seed}")
    assert loaded.training_seed == 55

    with_seed = tmp_path / "with-seed.csv"
    _write_history_csv(with_seed, _history("source", seed=56), include_seed=True)
    loaded = load_history_csv(with_seed)
    assert loaded.training_seed == 56

    with pytest.raises(ValueError, match="conflicts with supplied seed"):
        parse_seed_csv(f"57={with_seed}")


def test_offline_csv_rejects_missing_required_soft_column(tmp_path):
    path = tmp_path / "missing.csv"
    fieldnames = [ENV_STEP_KEY, *METRIC_KEYS[:-1]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({ENV_STEP_KEY: 0, **{key: 1.0 for key in METRIC_KEYS[:-1]}})
    with pytest.raises(ValueError, match="missing required CSV column"):
        load_history_csv(path, training_seed=55)


@pytest.mark.parametrize("n_seeds, expected_band_count", [(1, 0), (3, 2)])
def test_plot_labels_target_matching_and_seed_bands(n_seeds, expected_band_count):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    histories = [
        _history(f"seed-{55 + index}", seed=55 + index, offset=2.0 * index)
        for index in range(n_seeds)
    ]
    figure = plot_aggregate(aggregate_histories(histories))
    try:
        assert [axis.get_title() for axis in figure.axes] == [
            "Paper deterministic — not SAC-target matched",
            "Stochastic soft Bellman — SAC-target matched",
        ]
        assert [line.get_label() for line in figure.axes[0].lines] == [
            f"Deterministic reward MC (n={n_seeds})",
            f"Online paper-pair Q (n={n_seeds})",
        ]
        assert [line.get_label() for line in figure.axes[1].lines] == [
            f"Bootstrapped soft MC (corrected) (n={n_seeds})",
            f"Online mean-all Q (n={n_seeds})",
        ]
        assert all(
            len(axis.collections) == expected_band_count for axis in figure.axes
        )
    finally:
        plt.close(figure)


def test_png_pdf_and_full_schema_aggregate_csv_artifacts(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    aggregate = aggregate_histories(
        [
            _history("seed-55", seed=55, offset=0.0),
            _history("seed-56", seed=56, offset=2.0),
            _history("seed-57", seed=57, offset=4.0),
        ]
    )
    outputs = write_artifacts(aggregate, tmp_path / "sac-calibration")

    assert outputs["png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert outputs["pdf"].read_bytes().startswith(b"%PDF")
    with outputs["csv"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["training_seeds"] == "55,56,57"
    assert int(rows[0]["n"]) == 3
    for key in (
        REWARD_MC_KEY,
        SOFT_MC_FINITE_KEY,
        SOFT_MC_BOOTSTRAPPED_KEY,
        SOFT_TRUNCATION_TAIL_KEY,
        SOFT_Q_MIN_KEY,
    ):
        assert f"{key}_mean" in rows[0]
        assert f"{key}_std" in rows[0]
    assert float(rows[0][f"{PAPER_MC_KEY}_mean"]) == pytest.approx(
        aggregate.metrics[PAPER_MC_KEY].mean[0]
    )
    assert matplotlib.get_backend().lower() == "agg"


def test_existing_artifact_is_rejected_before_matplotlib_is_needed(tmp_path):
    aggregate = aggregate_histories([_history("seed-55")])
    prefix = tmp_path / "existing"
    existing_pdf = Path(f"{prefix}.pdf")
    existing_pdf.write_bytes(b"keep-me")

    with pytest.raises(ValueError, match="already exist.*--overwrite"):
        write_artifacts(aggregate, prefix)
    assert existing_pdf.read_bytes() == b"keep-me"
    assert not Path(f"{prefix}.png").exists()
    assert not Path(f"{prefix}.csv").exists()


def test_cli_accepts_repeated_seed_qualified_histories(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    first = tmp_path / "seed-55.csv"
    second = tmp_path / "seed-56.csv"
    _write_history_csv(first, _history("seed-55", seed=55, config=None))
    _write_history_csv(second, _history("seed-56", seed=56, config=None))
    prefix = tmp_path / "cli-sac-calibration"

    assert main(
        [
            "--history-csv",
            f"55={first}",
            "--history-csv",
            f"56={second}",
            "--output-prefix",
            str(prefix),
        ]
    ) == 0
    assert Path(f"{prefix}.png").is_file()
    assert Path(f"{prefix}.pdf").is_file()
    assert Path(f"{prefix}.csv").is_file()
    assert '"png"' in capsys.readouterr().out


def test_cli_forwards_explicit_overwrite(tmp_path, monkeypatch, capsys):
    history_csv = tmp_path / "seed.csv"
    _write_history_csv(history_csv, _history("seed", config=None))
    calls = []

    def fake_write_artifacts(aggregate, output_prefix, *, title, overwrite):
        calls.append((aggregate.training_seeds, Path(output_prefix), title, overwrite))
        return {
            "png": Path(f"{output_prefix}.png"),
            "pdf": Path(f"{output_prefix}.pdf"),
            "csv": Path(f"{output_prefix}.csv"),
        }

    monkeypatch.setattr(sac_plot, "write_artifacts", fake_write_artifacts)
    prefix = tmp_path / "forwarded"
    assert main(
        [
            "--history-csv",
            f"55={history_csv}",
            "--output-prefix",
            str(prefix),
            "--overwrite",
        ]
    ) == 0
    assert calls == [
        ((55,), prefix, "Vanilla SAC value calibration", True)
    ]
    assert '"pdf"' in capsys.readouterr().out
