"""The training throughput gate must distinguish intentional native actions."""

import pytest

from slurm.smoke_metaworld_mrsq_q5 import validate_graph_usage


def row(step, used, *, evaluation=False):
    return {"step": step, "evaluation": evaluation,
            "metrics": {"inner_cuda_graph_used": used}}


def test_graph_gate_accepts_native_evaluation_and_diagnostics():
    counts = validate_graph_usage([
        row(1, 0, evaluation=True), row(1000, 0), row(1102, 1),
    ], 1000, 1102, 1201)
    assert counts == {"training_graph": 1, "training_native_diagnostic": 1,
                      "evaluation_native": 1, "measured_training_graph": 1}


@pytest.mark.parametrize("record", [
    row(1102, 0), row(1, 1, evaluation=True), row(1000, 1),
])
def test_graph_gate_rejects_unexpected_execution_path(record):
    with pytest.raises(AssertionError, match="Unexpected graph usage"):
        validate_graph_usage([record], 1000, 1102, 1201)


def test_native_diagnostic_is_not_a_valid_timing_sample():
    with pytest.raises(AssertionError, match="Timed training must use"):
        validate_graph_usage([row(2000, 0)], 1000, 1901, 2000)
