import copy

import numpy as np
import pytest

from utils.training_diagnostics import TrainingDiagnostics
from utils.wandb_utils import WandbAccumulator


def test_action_boundary_thresholds_denominators_and_episode_reset():
    diagnostics, window = TrainingDiagnostics(), WandbAccumulator()
    first = np.array([.99, 1., 0., -.99])
    diagnostics.record_action(first, window, episode_start=True)
    first[:] = 0  # Observation owns its copy, not the caller's mutable buffer.
    diagnostics.record_action([-.99, -1., 0., .99], window)
    metrics = window.snapshot()
    prefix = "train/executed_action_"
    assert metrics[prefix + "samples"] == 2
    assert metrics[prefix + "finite_coordinates"] == 8
    assert metrics[prefix + "transition_samples"] == 1
    assert metrics[prefix + "transition_coordinates"] == 4
    assert metrics[prefix + "abs_ge_0p99_fraction"] == .75
    assert metrics[prefix + "exact_saturation_fraction"] == .25
    assert metrics[prefix + "sign_flip_fraction"] == .75
    assert metrics[prefix + "bound_flip_fraction"] == .75
    assert metrics[prefix + "delta_rms"] == pytest.approx(np.sqrt((1.98**2 * 2 + 4) / 4))
    diagnostics.complete_episode(1.)
    window.clear()
    diagnostics.record_action([1., 1., 1., 1.], window)
    assert window.snapshot()[prefix + "transition_coordinates"] == 0
    assert prefix + "delta_rms" not in window.snapshot()


def test_action_nonfinite_pairs_are_excluded_with_explicit_coverage():
    diagnostics, window = TrainingDiagnostics(), WandbAccumulator()
    diagnostics.record_action([np.nan, 0., -.5], window)
    diagnostics.record_action([1., -.5, .5], window)
    metrics = window.snapshot()
    prefix = "train/executed_action_"
    assert metrics[prefix + "nonfinite_fraction"] == pytest.approx(1 / 6)
    assert metrics[prefix + "finite_coordinates"] == 5
    assert metrics[prefix + "transition_coordinates"] == 2
    assert metrics[prefix + "sign_flip_fraction"] == .5
    assert metrics[prefix + "bound_flip_fraction"] == 0


def test_return_windows_use_completed_episodes_and_disjoint_comparison():
    diagnostics = TrainingDiagnostics()
    assert diagnostics.episode_metrics()["episode/return_window20_count"] == 0
    for _ in range(19):
        diagnostics.complete_episode(10.)
    partial = diagnostics.episode_metrics()
    assert partial["episode/return_window20_ready"] == 0
    assert "episode/return_window20_peak_mean" not in partial
    diagnostics.complete_episode(10.)
    for _ in range(20):
        diagnostics.complete_episode(4.)
    metrics = diagnostics.episode_metrics()
    assert metrics["episode/return_diagnostic_episodes"] == 40
    assert metrics["episode/return_window20_count"] == 20
    assert metrics["episode/return_window20_ready"] == 1
    assert metrics["episode/return_disjoint20_ready"] == 1
    assert metrics["episode/return_window20_mean"] == 4
    assert metrics["episode/return_window20_std"] == 0
    assert metrics["episode/return_window20_min"] == 4
    assert metrics["episode/return_window20_peak_mean"] == 10
    assert metrics["episode/return_window20_drawdown"] == 6
    assert metrics["episode/return_window20_drawdown_fraction"] == .6
    assert metrics["episode/return_previous20_mean"] == 10
    assert metrics["episode/return_disjoint20_change"] == -6
    assert metrics["episode/return_disjoint20_drop_fraction"] == .6
    # Repeated logging/snapshotting cannot count the same episode again.
    state = diagnostics.state_dict()
    assert diagnostics.episode_metrics() == metrics
    assert diagnostics.state_dict() == state


def test_return_ratios_are_undefined_for_nonpositive_reference():
    diagnostics = TrainingDiagnostics()
    for value in [-2.] * 20 + [-4.] * 20:
        diagnostics.complete_episode(value)
    metrics = diagnostics.episode_metrics()
    assert metrics["episode/return_window20_drawdown"] == 2
    assert metrics["episode/return_disjoint20_change"] == -2
    assert "episode/return_disjoint20_drop_fraction" not in metrics
    assert "episode/return_window20_drawdown_fraction" not in metrics


def test_history_roundtrip_preserves_peak_and_future_metrics_without_rng():
    diagnostics = TrainingDiagnostics()
    for value in [10.] * 20 + [2.] * 33:
        diagnostics.complete_episode(value)
    before = copy.deepcopy(np.random.get_state())
    restored = TrainingDiagnostics.from_state_dict(diagnostics.state_dict())
    for value in [1., 4., 5., 9.] * 7:
        diagnostics.complete_episode(value)
        restored.complete_episode(value)
        assert restored.episode_metrics() == diagnostics.episode_metrics()
        assert restored.state_dict() == diagnostics.state_dict()
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("corruption", [
    {"episodes": -1}, {"recent_returns": [1.]}, {"peak_window_mean": 1.},
    {"version": 2}, {"episodes": 1, "recent_returns": [float("nan")]},
])
def test_invalid_diagnostic_state_rejected(corruption):
    state = TrainingDiagnostics().state_dict()
    state.update(corruption)
    with pytest.raises(ValueError, match="diagnostics"):
        TrainingDiagnostics.from_state_dict(state)


def test_boundary_serialization_rejects_live_action_history():
    diagnostics = TrainingDiagnostics()
    diagnostics.record_action([1.], WandbAccumulator())
    with pytest.raises(RuntimeError, match="episode boundary"):
        diagnostics.state_dict()
    diagnostics.complete_episode(1.)
    assert diagnostics.state_dict()["episodes"] == 1
