"""Historical combined prior/MPPI bundles remain reusable prior references."""
import json

import pytest

from utils.ambi_benchmark import reference_returns


def test_combined_bundle_selects_only_unadapted_prior(tmp_path):
    prior = {"config": {"alg_params": {"inner_operator": "none"}},
             "status": "complete", "episodes": [{"seed": 101, "return": 12.0}]}
    mppi = {"config": {"alg_params": {"inner_operator": "none"},
                       "evaluation_controller": "mppi"},
            "status": "complete", "episodes": [{"seed": 101, "return": 99.0}]}
    manifest = {"schema_version": 1, "status": "complete", "checkpoint": {"sha256": "a" * 64},
                "protocol": {}, "runs": [prior, mppi]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert reference_returns(path, "a" * 64, {}) == {101: 12.0}
    manifest["runs"] = [mppi]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exactly one"):
        reference_returns(path, "a" * 64, {})
