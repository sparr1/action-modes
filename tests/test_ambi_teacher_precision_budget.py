"""CPU checks for the full-recipe CUDA gate's measured precision budget."""

import pytest
import torch

from tests.test_ambi_batched_reference_recipe import _assert_precision_budget


def test_precision_budget_accepts_measured_drift_and_small_roundoff():
    highest = {"weight": torch.ones(2, 3), "bias": torch.ones(2, 1)}
    reference = {key: value + 1e-3 for key, value in highest.items()}
    actual = {key: value + 5e-4 for key, value in reference.items()}
    metrics = _assert_precision_budget(actual, reference, highest)
    assert len(metrics["budget_rms_per_root"]) == 2
    assert all(drift < budget for drift, budget in zip(
        metrics["drift_rms_per_root"], metrics["budget_rms_per_root"],
    ))
    # An exact high/highest agreement still allows epsilon-scale roundoff in
    # nonzero values, with no absolute floor or cross-root tolerance pooling.
    epsilon = torch.finfo(torch.float32).eps
    actual = {key: value + 8 * epsilon for key, value in highest.items()}
    _assert_precision_budget(actual, highest, highest)
    zero = {"weight": torch.zeros(2, 3)}
    _assert_precision_budget(zero, zero, zero)


def test_precision_budget_rejects_excess_per_root_drift_zero_baselines_and_nonfinite():
    highest = {"weight": torch.ones(2, 3), "bias": torch.ones(2, 1)}
    reference = {key: value + 1e-3 for key, value in highest.items()}
    actual = {key: value.clone() for key, value in reference.items()}
    actual["weight"][1] += 2e-3
    with pytest.raises(AssertionError, match="exceeds reference precision budget"):
        _assert_precision_budget(actual, reference, highest)
    zero = {"weight": torch.zeros(2, 3)}
    with pytest.raises(AssertionError, match="exceeds reference precision budget"):
        _assert_precision_budget({"weight": torch.full((2, 3), 1e-12)}, zero, zero)
    for nonfinite in (float("nan"), float("inf")):
        invalid = {"weight": torch.full((2, 3), nonfinite)}
        for actual, reference, highest in ((invalid, zero, zero), (zero, invalid, zero), (zero, zero, invalid)):
            with pytest.raises(AssertionError, match="nonfinite"):
                _assert_precision_budget(actual, reference, highest)
