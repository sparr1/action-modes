"""MPPI action selection remains explicit without splitting historical means."""

from copy import deepcopy

import pytest

from utils.eval_series_data import planner_identity
from utils.resume_identity import scientific_trial_parameters


ALGORITHM = "AMBITDMPC2/AMBITDMPC2"


def _planner(params):
    return planner_identity(params, {}, ALGORITHM, "mean")


def _lineage(params):
    return scientific_trial_parameters({"alg": ALGORITHM, "alg_params": params})


@pytest.mark.parametrize("identity", [_planner, _lineage])
@pytest.mark.parametrize("spelling", ["mean", "MEAN"])
def test_mean_mppi_eval_preserves_omitted_historical_identity(identity, spelling):
    historical = {
        "inner_operator": "mppi",
        "inner_mppi_iterations": 8,
        "inner_mppi_num_samples": 512,
    }
    before = deepcopy(historical)
    expected = identity(historical)
    assert identity({**historical, "inner_mppi_eval_action": spelling}) == expected
    settings = expected["settings"] if identity is _planner else expected["alg_params"]
    assert "inner_mppi_eval_action" not in settings
    assert historical == before


@pytest.mark.parametrize("identity", [_planner, _lineage])
def test_weighted_elite_mppi_eval_has_distinct_case_normalized_identity(identity):
    historical = {"inner_operator": "mppi", "inner_mppi_iterations": 8}
    weighted = {**historical, "inner_mppi_eval_action": "weighted_elite"}
    before = deepcopy(weighted)
    result = identity(weighted)
    assert result != identity(historical)
    assert result == identity({**weighted, "inner_mppi_eval_action": "WEIGHTED_ELITE"})
    settings = result["settings"] if identity is _planner else result["alg_params"]
    assert settings["inner_mppi_eval_action"] == "weighted_elite"
    assert weighted == before


@pytest.mark.parametrize("operator", ["sac", "td3", "none"])
def test_inactive_mppi_eval_action_does_not_split_other_planner_curves(operator):
    config = {"inner_operator": operator}
    assert _planner({**config, "inner_mppi_eval_action": "weighted_elite"}) == _planner(config)
