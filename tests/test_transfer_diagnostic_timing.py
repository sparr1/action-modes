"""Timing accounting and cleanup without running checkpoint-scale compute."""
import io
import json

import pytest

from profile_ambi_transfer_diagnostics import PhaseTimer, instrument
import evaluate_ambi_transfer_diagnostics as evaluator


def test_nested_exclusive_accounting_and_failure_receipt():
    ticks = iter([0., 1., 4., 10.])
    synchronizations = []
    stream = io.StringIO()
    timer = PhaseTimer(stream, lambda: synchronizations.append(1), lambda: next(ticks))
    with pytest.raises(RuntimeError):
        with timer.phase("outer"):
            with timer.phase("inner"):
                raise RuntimeError("failed measurement")
    assert len(synchronizations) == 4
    assert timer.stats["inner"]["seconds"] == 3
    assert timer.stats["outer"]["seconds"] == 10
    assert timer.stats["outer"]["exclusive_seconds"] == 7
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert events[-1]["status"] == "failed"


def test_solve_classification_and_wrapper_restoration():
    timer = PhaseTimer(io.StringIO())
    original = evaluator.solve_fork
    with pytest.raises(RuntimeError):
        with instrument(timer):
            assert evaluator.solve_fork is not original
            solve = timer.wrap(lambda: 42, "solve")
            assert solve() == 42
            with timer.phase("root_audit"):
                assert solve() == 42
                with timer.phase("replanning"):
                    assert solve() == 42
            raise RuntimeError("restore")
    assert evaluator.solve_fork is original
    for name in ("source_solve", "root_fork_solve", "replanning_solve"):
        assert timer.stats[name]["calls"] == 1
