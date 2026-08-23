import importlib.util
from pathlib import Path

import numpy as np
import pytest


ORACLE_PATH = (
    Path(__file__).resolve().parent / "oracles" / "run_official_xqc_smoke.py"
)
SPEC = importlib.util.spec_from_file_location("xqc_official_smoke_oracle", ORACLE_PATH)
assert SPEC is not None and SPEC.loader is not None
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


def test_official_smoke_finite_state_oracle_rejects_bad_leaves():
    ORACLE._assert_finite_leaves(
        "test state",
        [np.array([1.0, -2.0]), np.array([1, 2]), np.array(True)],
    )

    with pytest.raises(SystemExit, match="non-finite"):
        ORACLE._assert_finite_leaves("test state", [np.array([np.nan])])
    with pytest.raises(SystemExit, match="not numeric"):
        ORACLE._assert_finite_leaves("test state", [np.array(["bad"])])
    with pytest.raises(SystemExit, match="no numeric state"):
        ORACLE._assert_finite_leaves("test state", [])


def test_official_smoke_projection_oracle_matches_flax_column_axis_and_paths():
    flat_params = {
        "MLP_0/XQCBlock_0/Dense_0/kernel": np.array(
            [[3.0, 0.0], [4.0, 1.0]], dtype=np.float32
        ),
        "predictor_tanh_gauss/mean/kernel": np.eye(2, dtype=np.float32),
        "BatchNormEmbedder_0/BatchNorm_0/scale": np.array(
            [100.0, 100.0], dtype=np.float32
        ),
        "unprojected/Dense_0/kernel": np.full((2, 2), 100.0, dtype=np.float32),
    }

    assert ORACLE._projected_column_residual(flat_params) == pytest.approx(4.0)

    flat_params["predictor_tanh_gauss/mean/kernel"][0, 0] = np.nan
    with pytest.raises(SystemExit, match="non-finite"):
        ORACLE._projected_column_residual(flat_params)

    with pytest.raises(SystemExit, match="no projected kernels"):
        ORACLE._projected_column_residual(
            {"unprojected/Dense_0/kernel": np.eye(2, dtype=np.float32)}
        )
