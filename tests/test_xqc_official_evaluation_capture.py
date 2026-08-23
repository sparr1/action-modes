import argparse
import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ORACLE_PATH = (
    Path(__file__).resolve().parent / "oracles" / "run_official_xqc_smoke.py"
)
SPEC = importlib.util.spec_from_file_location(
    "xqc_official_evaluation_capture_oracle",
    ORACLE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


def _capture_args(**overrides):
    values = {
        "evaluation_csv": None,
        "base_seed": None,
        "num_seeds": None,
        "action_repeat": None,
        "expected_evaluation_rows": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_evaluation_capture_arguments_are_all_or_none(tmp_path):
    assert ORACLE._validate_evaluation_capture_args(_capture_args()) is False

    with pytest.raises(SystemExit, match="require --evaluation-csv"):
        ORACLE._validate_evaluation_capture_args(_capture_args(base_seed=0))
    with pytest.raises(SystemExit, match="requires --base-seed"):
        ORACLE._validate_evaluation_capture_args(
            _capture_args(
                evaluation_csv=tmp_path / "eval.csv",
                num_seeds=2,
                action_repeat=2,
                expected_evaluation_rows=22,
            )
        )
    with pytest.raises(SystemExit, match="action-repeat must be positive"):
        ORACLE._validate_evaluation_capture_args(
            _capture_args(
                evaluation_csv=tmp_path / "eval.csv",
                base_seed=0,
                num_seeds=2,
                action_repeat=0,
                expected_evaluation_rows=22,
            )
        )
    with pytest.raises(SystemExit, match="num-seeds must be positive"):
        ORACLE._validate_evaluation_capture_args(
            _capture_args(
                evaluation_csv=tmp_path / "eval.csv",
                base_seed=0,
                num_seeds=0,
                action_repeat=2,
                expected_evaluation_rows=22,
            )
        )
    with pytest.raises(SystemExit, match="expected-evaluation-rows must be positive"):
        ORACLE._validate_evaluation_capture_args(
            _capture_args(
                evaluation_csv=tmp_path / "eval.csv",
                base_seed=0,
                num_seeds=2,
                action_repeat=2,
                expected_evaluation_rows=0,
            )
        )
    with pytest.raises(SystemExit, match="must be divisible by --num-seeds"):
        ORACLE._validate_evaluation_capture_args(
            _capture_args(
                evaluation_csv=tmp_path / "eval.csv",
                base_seed=0,
                num_seeds=2,
                action_repeat=2,
                expected_evaluation_rows=21,
            )
        )

    assert ORACLE._validate_evaluation_capture_args(
        _capture_args(
            evaluation_csv=tmp_path / "eval.csv",
            base_seed=0,
            num_seeds=2,
            action_repeat=2,
            expected_evaluation_rows=22,
        )
    ) is True


def test_evaluation_capture_writes_two_seed_paper_schedule_durably(
    tmp_path,
    monkeypatch,
):
    fsynced_descriptors = []
    monkeypatch.setattr(
        ORACLE.os,
        "fsync",
        lambda descriptor: fsynced_descriptors.append(descriptor),
    )
    path = tmp_path / "official-evaluations.csv"
    capture = ORACLE._EvaluationCsvCapture(
        path,
        base_seed=0,
        num_seeds=2,
        action_repeat=2,
        expected_rows=22,
    )
    try:
        capture.record(20_000, {"critic_loss": np.array([1.0, 2.0])})
        assert len(fsynced_descriptors) == 1

        raw_frames = [2, *range(100_000, 1_000_001, 100_000)]
        for evaluation_index, raw_frame in enumerate(raw_frames):
            capture.record(
                raw_frame,
                {
                    "return": np.array(
                        [evaluation_index + 0.25, evaluation_index + 0.75],
                        dtype=np.float64,
                    )
                },
            )

        # Header creation and every evaluation event are independently durable.
        assert len(fsynced_descriptors) == 1 + len(raw_frames)
        capture.assert_expected_rows()
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    finally:
        capture.close()

    assert len(rows) == 22
    assert tuple(rows[0]) == ORACLE.EVALUATION_CSV_FIELDS
    assert rows[0] == {
        "implementation": "official-jax",
        "source_commit": ORACLE.OFFICIAL_COMMIT,
        "base_seed": "0",
        "num_seeds": "2",
        "seed_index": "0",
        "seed": "0",
        "evaluation_index": "0",
        "decision_step": "1",
        "raw_frame": "2",
        "paper_raw_frame": "0",
        "action_repeat": "2",
        "return": "0.25",
    }
    assert rows[1]["seed_index"] == "1"
    assert rows[1]["seed"] == "1"
    assert rows[-1]["evaluation_index"] == "10"
    assert rows[-1]["decision_step"] == "500000"
    assert rows[-1]["raw_frame"] == "1000000"
    assert rows[-1]["paper_raw_frame"] == "1000000"
    assert rows[-1]["seed"] == "1"
    assert rows[-1]["return"] == "10.75"


def test_evaluation_capture_exclusively_creates_and_checks_row_count(tmp_path):
    path = tmp_path / "existing.csv"
    path.write_text("user data\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ORACLE._EvaluationCsvCapture(
            path,
            base_seed=0,
            num_seeds=2,
            action_repeat=2,
            expected_rows=2,
        )
    assert path.read_text(encoding="utf-8") == "user data\n"

    incomplete = ORACLE._EvaluationCsvCapture(
        tmp_path / "incomplete.csv",
        base_seed=4,
        num_seeds=2,
        action_repeat=2,
        expected_rows=3,
    )
    try:
        incomplete.record(2, {"return": [1.0, 2.0]})
        with pytest.raises(SystemExit, match="wrote 2 rows; expected 3"):
            incomplete.assert_expected_rows()
    finally:
        incomplete.close()


@pytest.mark.parametrize(
    ("step", "returns", "message"),
    [
        (3, [1.0], "not divisible"),
        (float("nan"), [1.0], "finite, non-negative integer"),
        (2, [], "non-empty vector"),
        (2, [[1.0]], "non-empty vector"),
        (2, [1.0, 2.0], "return count"),
        (2, [float("inf")], "non-finite"),
        (2, ["bad"], "real numeric"),
        (2, [1.0 + 2.0j], "real numeric"),
    ],
)
def test_evaluation_capture_rejects_ambiguous_or_invalid_rows(
    tmp_path,
    step,
    returns,
    message,
):
    capture = ORACLE._EvaluationCsvCapture(
        tmp_path / "invalid.csv",
        base_seed=0,
        num_seeds=1,
        action_repeat=2,
        expected_rows=1,
    )
    try:
        with pytest.raises(SystemExit, match=message):
            capture.record(step, {"return": returns})
        assert capture.row_count == 0
        assert capture.evaluation_count == 0
    finally:
        capture.close()


def test_evaluation_logging_patch_forwards_and_restores_on_failure():
    forwarded = []
    captured = []

    def original(step, infos, fps=30):
        forwarded.append((step, infos, fps))
        return "logged"

    module = SimpleNamespace(log_multiple_seeds_to_wandb=original)
    capture = SimpleNamespace(
        record=lambda step, infos: captured.append((step, infos))
    )

    with pytest.raises(RuntimeError, match="official failure"):
        with ORACLE._patched_evaluation_logging(module, capture):
            assert module.log_multiple_seeds_to_wandb(
                2,
                {"return": [1.0, 2.0]},
                fps=17,
            ) == "logged"
            assert module.log_multiple_seeds_to_wandb is not original
            raise RuntimeError("official failure")

    assert module.log_multiple_seeds_to_wandb is original
    assert captured == [(2, {"return": [1.0, 2.0]})]
    assert forwarded == [(2, {"return": [1.0, 2.0]}, 17)]


def test_official_main_runs_directly_when_capture_is_absent():
    calls = []

    def official_main():
        calls.append("official")

    ORACLE._run_official_main(official_main)

    assert calls == ["official"]
