"""AMBI defaults and provenance distinguish vmap from historical loop execution."""

from types import SimpleNamespace

import pytest
from torch import nn

import main as training_main
import utils.resume_identity as resume_identity
from RL.tdmpc2_core.common.layers import Ensemble
from tests.test_ambi_config_decoupling import _build_cfg
from tests.test_tdambi_initialization import _native_cfg
from utils.ensemble_identity import normalize_ensemble_backend_identity
from utils.resume_identity import canonical_json, scientific_trial_parameters


def _identity(**params):
    return scientific_trial_parameters({
        "alg": "AMBITDMPC2/AMBITDMPC2", "seed": 3, "alg_params": params,
    })


def test_default_is_vmap_for_ambi_and_tdambi_but_generic_ensemble_stays_loop():
    assert _build_cfg().critic_ensemble_backend == "vmap"
    assert _native_cfg().critic_ensemble_backend == "vmap"
    assert Ensemble([nn.Identity()]).backend == "loop"


@pytest.mark.parametrize("backend", ["loop", "vmap", "LOOP", "VMAP"])
def test_explicit_backend_is_validated_and_canonicalized_without_mutating_input(backend):
    config = {"critic_ensemble_backend": backend}
    assert _build_cfg(**config).critic_ensemble_backend == backend.lower()
    assert _native_cfg(**config).critic_ensemble_backend == backend.lower()
    assert config == {"critic_ensemble_backend": backend}


@pytest.mark.parametrize("value", [None, True, False, 0, 1, {}, [], "", "map", " vmap "])
def test_invalid_backends_fail_before_network_construction(value):
    with pytest.raises(ValueError, match="critic_ensemble_backend"):
        _build_cfg(critic_ensemble_backend=value)
    with pytest.raises(ValueError, match="critic_ensemble_backend"):
        normalize_ensemble_backend_identity({"critic_ensemble_backend": value})


def test_omitted_current_default_equals_explicit_vmap_and_differs_from_loop():
    assert _identity() == _identity(critic_ensemble_backend="vmap")
    assert _identity() == _identity(critic_ensemble_backend="VMAP")
    assert _identity() != _identity(critic_ensemble_backend="loop")
    assert _identity()["alg_params"]["critic_ensemble_backend"] == "vmap"
    assert "critic_ensemble_backend" not in _identity(critic_ensemble_backend="loop")["alg_params"]


@pytest.mark.parametrize("other", [{}, {"aux_return_mode": "sac"},
                                 {"outer_pave_curvature_coef": .01}])
def test_explicit_loop_preserves_pre_feature_scientific_identity(monkeypatch, other):
    current = canonical_json(_identity(**other, critic_ensemble_backend="loop"))
    # Reproduce the pre-feature projection, before this normalization existed.
    monkeypatch.setattr(resume_identity, "normalize_ensemble_backend_identity", dict)
    historical = canonical_json(_identity(**other))
    assert current == historical


def test_non_ambi_scientific_identity_does_not_acquire_vmap_default():
    native = scientific_trial_parameters({
        "alg": "TDMPC2/TDMPC2Baseline", "seed": 3, "alg_params": {},
    })
    assert "critic_ensemble_backend" not in native["alg_params"]
    assert "TDAMBI/TDAMBI" not in resume_identity.SUPPORTED_RESUME_ALGORITHMS


@pytest.mark.parametrize("backend", ["loop", "vmap"])
@pytest.mark.parametrize("algorithm", ["AMBITDMPC2/AMBITDMPC2", "TDAMBI/TDAMBI"])
def test_resolved_runtime_records_backend_outside_architecture_signature(backend, algorithm):
    signature = {"q_representation": "scalar", "num_q": 2}
    model = SimpleNamespace(
        cfg=SimpleNamespace(critic_ensemble_backend=backend),
        agent=SimpleNamespace(model=SimpleNamespace(critic_signature=signature)),
    )
    metadata = training_main._resolved_runtime_metadata(
        model, trial_run_params={"alg": algorithm, "seed": 3},
    )
    assert metadata["critic_ensemble_backend"] == backend
    assert "critic_ensemble_backend" not in metadata["critic"]
    assert signature == {"q_representation": "scalar", "num_q": 2}


def test_native_runtime_without_backend_field_retains_historical_shape():
    metadata = training_main._resolved_runtime_metadata(
        SimpleNamespace(cfg=SimpleNamespace()),
        trial_run_params={"alg": "TDMPC2/TDMPC2Baseline", "seed": 3},
    )
    assert "critic_ensemble_backend" not in metadata
