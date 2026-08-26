import csv
from pathlib import Path

import numpy as np
import pytest

import plot_value_calibration as value_plot
from plot_value_calibration import (
    DETERMINISTIC_MC_KEY,
    DETERMINISTIC_Q_KEY,
    ENV_STEP_KEY,
    METRIC_KEYS,
    STOCHASTIC_MC_KEY,
    STOCHASTIC_Q_KEY,
    SeedHistory,
    aggregate_histories,
    extract_semantic_config,
    load_history_csv,
    main,
    plot_aggregate,
    write_artifacts,
)


def _history(
    source,
    *,
    steps=(0, 50_000),
    offset=0.0,
    config=None,
    metrics=None,
):
    default_metrics = {
        DETERMINISTIC_MC_KEY: np.array([1.0, 2.0]) + offset,
        DETERMINISTIC_Q_KEY: np.array([3.0, 4.0]) + offset,
        STOCHASTIC_MC_KEY: np.array([5.0, 6.0]) + offset,
        STOCHASTIC_Q_KEY: np.array([7.0, 8.0]) + offset,
    }
    return SeedHistory(
        source=source,
        env_step=steps,
        metrics=default_metrics if metrics is None else metrics,
        semantic_config=config,
    )


def _write_history_csv(path: Path, history: SeedHistory):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[ENV_STEP_KEY, *METRIC_KEYS])
        writer.writeheader()
        for index, step in enumerate(history.env_step):
            writer.writerow(
                {
                    ENV_STEP_KEY: step,
                    **{key: history.metrics[key][index] for key in METRIC_KEYS},
                }
            )


def test_one_seed_preserves_values_and_uses_zero_population_sd():
    history = _history("seed-1")
    aggregate = aggregate_histories([history])

    assert aggregate.n_seeds == 1
    np.testing.assert_array_equal(aggregate.env_step, [0, 50_000])
    for key in METRIC_KEYS:
        np.testing.assert_allclose(aggregate.metrics[key].mean, history.metrics[key])
        np.testing.assert_array_equal(aggregate.metrics[key].std, [0.0, 0.0])


def test_three_seeds_are_equal_weighted_with_population_sd():
    histories = [
        _history("seed-1", offset=0.0),
        _history("seed-2", offset=2.0),
        _history("seed-3", offset=4.0),
    ]
    aggregate = aggregate_histories(histories)
    expected_std = np.sqrt(8.0 / 3.0)

    assert aggregate.n_seeds == 3
    for key in METRIC_KEYS:
        np.testing.assert_allclose(
            aggregate.metrics[key].mean,
            np.asarray(histories[0].metrics[key]) + 2.0,
        )
        np.testing.assert_allclose(aggregate.metrics[key].std, expected_std)


def test_missing_metric_is_rejected():
    metrics = dict(_history("complete").metrics)
    metrics.pop(STOCHASTIC_Q_KEY)
    with pytest.raises(ValueError, match="missing required metric.*stochastic_q_mean_all"):
        aggregate_histories([_history("missing", metrics=metrics)])


def test_duplicate_step_is_rejected():
    with pytest.raises(ValueError, match="duplicate env_step.*50000"):
        aggregate_histories([_history("duplicate", steps=(50_000, 50_000))])


def test_exact_common_grid_is_required_without_interpolation():
    with pytest.raises(ValueError, match="grid does not exactly match.*interpolation"):
        aggregate_histories(
            [
                _history("seed-1", steps=(0, 50_000)),
                _history("seed-2", steps=(0, 100_000)),
            ]
        )


def test_configured_grid_requires_step_zero_every_cadence_and_final_point():
    config = {"eval_freq": 50_000, "total_steps": 100_000}
    with pytest.raises(ValueError, match="grid does not match configured.*missing 100000"):
        aggregate_histories(
            [
                _history(
                    "missing-final",
                    steps=(0, 50_000),
                    config=config,
                )
            ]
        )


def test_incompatible_available_semantic_config_is_rejected():
    common = {
        "environment": "DMControl-v0",
        "task": "humanoid-walk",
        "eval_value_protocols": ["paper_deterministic", "stochastic_bellman"],
        "eval_value_samples": 100,
    }
    mismatched = {**common, "eval_value_samples": 64}
    with pytest.raises(ValueError, match="semantic config is incompatible.*eval_value_samples"):
        aggregate_histories(
            [
                _history("seed-1", config=common),
                _history("seed-2", config=mismatched),
            ]
        )


def test_semantic_config_extraction_ignores_training_seed_and_checks_protocol():
    raw = {
        "algorithm": "AMBITDMPC2",
        "run_params": {
            "name": "ambi_humanoid_walk_value_calibration",
            "env": "DMControl-v0",
            "seed": 55,
        },
        "config": {
            "seed": 55,
            "obs": "state",
            "utd": 1,
            "eval_value": True,
            "eval_freq": 50_000,
            "eval_value_samples": 100,
            "eval_value_seed": 123,
            "eval_value_protocols": [
                "paper_deterministic",
                "stochastic_bellman",
            ],
            "steps": 1_000_000,
        },
        "total_steps": 1_000_000,
    }
    semantic = extract_semantic_config(raw)

    assert "seed" not in semantic
    assert semantic["recipe"] == "ambi_humanoid_walk_value_calibration"
    assert semantic["environment"] == "DMControl-v0"
    assert semantic["eval_freq"] == 50_000
    assert semantic["utd"] == 1
    assert semantic["eval_value_samples"] == 100
    assert semantic["total_steps"] == 1_000_000
    assert semantic["eval_value_protocols"] == [
        "paper_deterministic",
        "stochastic_bellman",
    ]


def test_offline_csv_rejects_missing_required_column(tmp_path):
    path = tmp_path / "missing.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[ENV_STEP_KEY, *METRIC_KEYS[:-1]],
        )
        writer.writeheader()
        writer.writerow({ENV_STEP_KEY: 0, **{key: 1.0 for key in METRIC_KEYS[:-1]}})

    with pytest.raises(ValueError, match="missing required CSV column.*stochastic_q_mean_all"):
        load_history_csv(path)


def test_png_pdf_and_aggregate_csv_artifacts(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    aggregate = aggregate_histories(
        [_history("seed-1", offset=0), _history("seed-2", offset=2), _history("seed-3", offset=4)]
    )
    outputs = write_artifacts(aggregate, tmp_path / "calibration")

    assert outputs["png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert outputs["pdf"].read_bytes().startswith(b"%PDF")
    with outputs["csv"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row[ENV_STEP_KEY]) for row in rows] == [0, 50_000]
    assert [int(row["n"]) for row in rows] == [3, 3]
    assert float(rows[0][f"{DETERMINISTIC_MC_KEY}_mean"]) == pytest.approx(3.0)
    assert float(rows[0][f"{DETERMINISTIC_MC_KEY}_std"]) == pytest.approx(
        np.sqrt(8.0 / 3.0)
    )
    assert matplotlib.get_backend().lower() == "agg"

    with pytest.raises(ValueError, match="already exist.*--overwrite"):
        write_artifacts(aggregate, tmp_path / "calibration")
    overwritten = write_artifacts(
        aggregate,
        tmp_path / "calibration",
        overwrite=True,
    )
    assert overwritten == outputs


def test_existing_artifact_is_rejected_before_matplotlib_is_needed(tmp_path):
    aggregate = aggregate_histories([_history("seed-1")])
    prefix = tmp_path / "existing"
    existing_pdf = Path(f"{prefix}.pdf")
    existing_pdf.write_bytes(b"keep-me")

    with pytest.raises(ValueError, match="already exist.*--overwrite"):
        write_artifacts(aggregate, prefix)

    assert existing_pdf.read_bytes() == b"keep-me"
    assert not Path(f"{prefix}.png").exists()
    assert not Path(f"{prefix}.csv").exists()


@pytest.mark.parametrize("n_seeds, expected_band_count", [(1, 0), (3, 2)])
def test_plot_labels_seed_count_and_only_bands_multi_seed(n_seeds, expected_band_count):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    histories = [_history(f"seed-{index}", offset=2.0 * index) for index in range(n_seeds)]
    figure = plot_aggregate(aggregate_histories(histories))
    try:
        assert len(figure.axes) == 2
        assert all(len(axis.collections) == expected_band_count for axis in figure.axes)
        assert all(
            f"n={n_seeds}" in line.get_label()
            for axis in figure.axes
            for line in axis.lines
        )
        expected_title = (
            "Mean ± 1 across-seed population SD"
            if n_seeds > 1
            else "Mean; n=1 (no uncertainty band)"
        )
        assert all(
            axis.get_legend().get_title().get_text() == expected_title
            for axis in figure.axes
        )
    finally:
        plt.close(figure)


def test_cli_accepts_repeated_offline_histories_and_writes_all_artifacts(
    tmp_path, capsys
):
    pytest.importorskip("matplotlib")
    first = tmp_path / "seed-1.csv"
    second = tmp_path / "seed-2.csv"
    _write_history_csv(first, _history("seed-1", offset=0.0))
    _write_history_csv(second, _history("seed-2", offset=2.0))
    prefix = tmp_path / "cli-calibration"

    assert main(
        [
            "--history-csv",
            str(first),
            "--history-csv",
            str(second),
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
    _write_history_csv(history_csv, _history("seed"))
    calls = []

    def fake_write_artifacts(aggregate, output_prefix, *, title, overwrite):
        calls.append((aggregate.n_seeds, Path(output_prefix), title, overwrite))
        return {
            "png": Path(f"{output_prefix}.png"),
            "pdf": Path(f"{output_prefix}.pdf"),
            "csv": Path(f"{output_prefix}.csv"),
        }

    monkeypatch.setattr(value_plot, "write_artifacts", fake_write_artifacts)
    prefix = tmp_path / "forwarded"
    assert value_plot.main(
        [
            "--history-csv",
            str(history_csv),
            "--output-prefix",
            str(prefix),
            "--overwrite",
        ]
    ) == 0

    assert calls == [(1, prefix, "AMBI value calibration", True)]
    assert '"pdf"' in capsys.readouterr().out
