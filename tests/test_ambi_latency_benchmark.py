import argparse
import json
from pathlib import Path

import numpy as np
import pytest

import benchmark_ambi_latency as benchmark


class _FakeAgent:
    def __init__(self):
        self.device = "cpu"
        self.num_updates = 17
        self.last_inner_metrics = {}


class _FakeModel:
    def __init__(self):
        self.agent = _FakeAgent()
        self._action_shape = (2,)
        self.calls = []

    def predict(
        self,
        observation,
        deterministic=True,
        episode_start=None,
        *,
        collect_diagnostics=True,
    ):
        index = len(self.calls)
        self.calls.append(
            {
                "observation": observation,
                "deterministic": deterministic,
                "episode_start": episode_start,
                "collect_diagnostics": collect_diagnostics,
            }
        )
        self.agent.last_inner_metrics = {
            "inner_action_seconds": 0.010 + index * 0.001,
            "inner_rollout_seconds": 0.004 + index * 0.001,
            "inner_update_seconds": 0.003,
            "inner_diagnostic_seconds": 99.0,
            "inner_model_steps": 192.0,
            "inner_update_slots": 8.0,
            "inner_compile_fallback": float(index == 0),
            "unrelated_metric": 123.0,
        }
        return np.zeros(2, dtype=np.float32), None


class _SequenceClock:
    def __init__(self, values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


def _write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_parse_cell_accepts_explicit_j_n_g_and_rejects_invalid_values():
    assert benchmark._parse_cell("2, 32,4") == (2, 32, 4)
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark._parse_cell("2:32:4")
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark._parse_cell("0,32,4")
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark._parse_cell("2,32,-1")


def test_latency_summary_contains_required_percentiles():
    summary = benchmark._latency_summary([1.0, 2.0, 3.0, 4.0])
    assert summary == {
        "count": 4,
        "mean": 2.5,
        "std": pytest.approx(np.std([1.0, 2.0, 3.0, 4.0])),
        "p50": 2.5,
        "p90": pytest.approx(3.7),
        "p95": pytest.approx(3.85),
        "min": 1.0,
        "max": 4.0,
    }


def test_benchmark_model_separates_cold_warmup_and_measured_calls_on_cpu():
    model = _FakeModel()
    synchronizations = []
    memory_events = []
    # Six calls: one cold, two warmup, and three measured. Each clock pair is
    # exactly 0.1 seconds apart.
    clock = _SequenceClock(
        [value for index in range(6) for value in (float(index), float(index) + 0.1)]
    )

    result = benchmark.benchmark_model(
        model,
        ["observation-a", "observation-b"],
        warmup_calls=2,
        measured_calls=3,
        action_mode="training",
        synchronize=lambda: synchronizations.append(True),
        reset_memory=lambda: memory_events.append(("reset", len(model.calls))) or {
            "supported": False,
            "device_type": "cpu",
        },
        read_memory=lambda: memory_events.append(("read", len(model.calls))) or {},
        clock=clock,
    )

    assert len(model.calls) == 6
    assert len(synchronizations) == 12
    assert memory_events == [("reset", 3), ("read", 6)]
    assert [call["observation"] for call in model.calls] == [
        "observation-a",
        "observation-b",
        "observation-a",
        "observation-b",
        "observation-a",
        "observation-b",
    ]
    assert [call["episode_start"] for call in model.calls] == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert all(call["deterministic"] is False for call in model.calls)
    assert all(call["collect_diagnostics"] is False for call in model.calls)
    assert result["cold_call"]["compile_fallbacks"] == {
        "inner_compile_fallback": 1.0
    }
    assert "inner_diagnostic_seconds" not in result["cold_call"]["phase_seconds"]
    assert result["warmup_calls"] == 2
    assert result["total_action_calls"] == 6
    assert result["measurements"]["count"] == 3
    assert result["measurements"]["wall_seconds"]["p50"] == pytest.approx(0.1)
    assert result["measurements"]["phase_seconds"]["inner_action_seconds"][
        "p50"
    ] == pytest.approx(0.014)
    assert result["measurements"]["work_counters"]["inner_model_steps"][
        "mean"
    ] == 192.0
    assert result["measurements"]["compile_fallbacks"][
        "inner_compile_fallback"
    ]["any"] is False
    assert result["measurements"]["device_memory"] == {
        "supported": False,
        "device_type": "cpu",
    }


@pytest.mark.parametrize(
    "action, message",
    [
        (np.array([np.nan, 0.0], dtype=np.float32), "non-finite"),
        (np.zeros(3, dtype=np.float32), "action shape"),
    ],
)
def test_timed_prediction_rejects_invalid_actions(action, message):
    model = _FakeModel()
    model.predict = lambda *args, **kwargs: (action, None)

    with pytest.raises(RuntimeError, match=message):
        benchmark._timed_prediction(
            model,
            "observation",
            deterministic=False,
            episode_start=True,
            synchronize=lambda: None,
            clock=_SequenceClock([1.0, 1.1]),
        )


def test_direct_base_config_cell_override_changes_only_schedule_and_capacity():
    original = {
        "algorithm_config": {
            "seed": 55,
            "alg": "AMBITDMPC2/AMBITDMPC2",
            "alg_params": {
                "inner_operator": "sac",
                "inner_rounds": 2,
                "inner_rollouts_per_round": 32,
                "inner_rollout_horizon": 3,
                "inner_updates_per_round": 4,
                "inner_replay_capacity": 192,
                "inner_batch_size": 64,
                "inner_temperature_mode": "inherit_outer",
                "wandb": True,
                "unrelated": "preserved",
            },
        },
        "environment": {
            "id": "DMControl-v0",
            "params": {"task": "humanoid-walk", "obs": "state"},
        },
    }
    cell = benchmark.BenchmarkCell(
        name="j4_n16_g2",
        J=4,
        N=16,
        G=2,
        source={"name": "j4_n16_g2", "J": 4, "N": 16, "G": 2},
    )

    resolved = benchmark._resolved_cell_config(original, cell)
    params = resolved["algorithm_config"]["alg_params"]

    assert params["inner_rounds"] == 4
    assert params["inner_rollouts_per_round"] == 16
    assert params["inner_rollout_horizon"] == 3
    assert params["inner_updates_per_round"] == 2
    assert params["inner_replay_capacity"] == 4 * 16 * 3
    assert params["inner_batch_size"] == 64
    assert params["inner_temperature_mode"] == "inherit_outer"
    assert params["unrelated"] == "preserved"
    assert params["wandb"] is False
    assert original["algorithm_config"]["alg_params"]["inner_rounds"] == 2
    assert original["algorithm_config"]["alg_params"]["wandb"] is True


def test_config_loader_and_cell_selection_support_launcher_schema(tmp_path, capsys):
    base_path = tmp_path / "base.json"
    _write_json(
        base_path,
        {
            "seed": 55,
            "alg": "AMBITDMPC2/AMBITDMPC2",
            "alg_params": {
                "inner_operator": "sac",
                "inner_rounds": 2,
                "inner_rollouts_per_round": 32,
                "inner_rollout_horizon": 3,
                "inner_updates_per_round": 4,
                "inner_replay_capacity": 192,
                "inner_batch_size": 64,
            },
        },
    )
    config_path = tmp_path / "benchmark.json"
    _write_json(
        config_path,
        {
            "schema_version": 1,
            "benchmark": "ambi-inner-latency",
            "metadata": {"description": "test"},
            "base": {
                "algorithm_config": "base.json",
                "environment": {
                    "id": "DMControl-v0",
                    "params": {"task": "humanoid-walk", "obs": "state"},
                },
            },
            "settings": {
                "cold_calls": 1,
                "warmup_calls": 49,
                "measured_calls": 200,
                "observation_bank_size": 8,
                "environment_seed": 101,
                "controller_seed": 55,
                "action_mode": "training",
                "blocks": 3,
                "H": 3,
                "B": 64,
            },
            "cells": [
                {
                    "name": "center",
                    "J": 2,
                    "N": 32,
                    "G": 4,
                    "block": 0,
                    "families": ["center"],
                    "expected": {"imagined_transitions": 192},
                },
                {"name": "g8", "J": 2, "N": 32, "G": 8, "block": 1},
            ],
        },
    )

    spec = benchmark.load_benchmark_config(config_path)
    selected = benchmark._select_cells(spec["cells"], [(2, 32, 8)])

    assert spec["algorithm_config_path"] == base_path.resolve()
    assert spec["environment"]["params"]["task"] == "humanoid-walk"
    assert spec["settings"]["warmup_calls"] == 49
    assert selected[0].name == "g8"
    assert selected[0].source["block"] == 1
    with pytest.raises(benchmark.BenchmarkConfigError, match="Unknown --cell"):
        benchmark._select_cells(spec["cells"], [(9, 9, 9)])

    assert benchmark.main(["--config", str(config_path), "--list-cells"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "center\t2\t32\t4",
        "g8\t2\t32\t8",
    ]


def test_strict_json_rejects_nonfinite_outputs():
    rendered = benchmark._strict_json(
        {
            "schema_version": 1,
            "benchmark": "ambi-inner-latency",
            "cells": [],
        }
    )
    assert json.loads(rendered)["schema_version"] == 1
    with pytest.raises(ValueError):
        benchmark._strict_json({"bad": float("nan")})


def test_work_and_compile_validation_are_strict():
    measurements = {
        "count": 2,
        "work_counters": {
            "inner_model_steps": benchmark._latency_summary([192.0, 192.0]),
            "inner_buffer_capacity": benchmark._latency_summary([192.0, 192.0]),
        },
        "compile_fallbacks": {
            key: benchmark._latency_summary([0.0, 0.0])
            for key in benchmark.COMPILE_FALLBACK_KEYS
        },
    }

    counters = benchmark._validate_expected_counters(
        {"inner_model_steps": 192, "inner_replay_capacity": 192}, measurements
    )
    compile_fallbacks = benchmark._validate_compile_fallbacks(measurements)

    assert counters["passed"] is True
    assert counters["details"]["inner_replay_capacity"]["metric"] == (
        "inner_buffer_capacity"
    )
    assert compile_fallbacks["passed"] is True

    measurements["work_counters"]["inner_model_steps"] = benchmark._latency_summary(
        [192.0, 191.0]
    )
    measurements["compile_fallbacks"]["inner_compile_actor_fallback"] = (
        benchmark._latency_summary([0.0, 1.0])
    )
    assert benchmark._validate_expected_counters(
        {"inner_model_steps": 192}, measurements
    )["passed"] is False
    assert benchmark._validate_compile_fallbacks(measurements)["passed"] is False

    measurements["compile_fallbacks"]["inner_compile_actor_fallback"] = (
        benchmark._latency_summary([-1.0, 0.0])
    )
    assert benchmark._validate_compile_fallbacks(measurements)["passed"] is False


def test_checkpoint_metadata_includes_content_sha256(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen outer state")

    metadata = benchmark._checkpoint_metadata(checkpoint)

    assert metadata["size_bytes"] == len(b"frozen outer state")
    assert metadata["sha256"] == (
        "ebd526e7b8005f172d7dfc521f23ec943aac7d868bbf57f9ac0966c19f711c16"
    )
