"""Inactive PAVE preserves lineages; active loss semantics identify resumes."""

import numpy as np
import pytest

from utils.pave_identity import PAVE_COEFFICIENT_KEYS, PAVE_DEFAULTS, pave_training_signature
from utils.resume_identity import canonical_json, scientific_trial_parameters


def identity(**config):
    return canonical_json(scientific_trial_parameters({
        "alg": "AMBITDMPC2/AMBITDMPC2", "seed": 3, "alg_params": config,
    }))


@pytest.mark.parametrize("base", [{}, {"aux_return_mode": "sac"},
                                  {"critic_value_mode": "return_entropy"}])
def test_disabled_pave_preserves_historical_identity_despite_inactive_knobs(base):
    assert identity(**base) == identity(**base, **PAVE_DEFAULTS)
    assert identity(**base) == identity(
        **base, outer_pave_batch_size=None, outer_pave_obs_noise_std=.5,
        outer_pave_curvature_margin=2, **dict.fromkeys(PAVE_COEFFICIENT_KEYS, 0),
    )


@pytest.mark.parametrize("key", PAVE_COEFFICIENT_KEYS)
def test_each_primary_or_auxiliary_loss_changes_scientific_identity(key):
    base = {"aux_return_mode": "sac"}
    assert identity(**base) != identity(**base, **{key: .1})
    assert identity(**base, **{key: .1}) != identity(**base, **{key: .2})


def test_active_defaults_and_numpy_scalar_types_hash_like_resolved_configuration():
    implicit = {"outer_pave_curvature_coef": 1}
    explicit = {**PAVE_DEFAULTS, **implicit, "outer_pave_curvature_coef": np.float64(1.),
                "outer_pave_batch_size": np.int64(64), "outer_pave_curvature_margin": 1}
    assert identity(**implicit) == identity(**explicit)


@pytest.mark.parametrize("key,value", [("outer_pave_batch_size", None),
                                       ("outer_pave_batch_size", 32),
                                       ("outer_pave_obs_noise_std", .02),
                                       ("outer_pave_curvature_margin", .5)])
def test_active_sampling_and_loss_controls_change_identity(key, value):
    base = {"outer_pave_mpr_coef": .1, "outer_pave_vfc_coef": .1,
            "outer_pave_curvature_coef": .01}
    assert identity(**base) != identity(**base, **{key: value})


def test_signature_records_sampling_reductions_and_seed_without_mutating_inputs():
    config = {"outer_pave_mpr_coef": .1, "seed": 3}
    original = dict(config)
    signature = pave_training_signature(config)
    assert config == original
    assert signature["seed"] == 3
    assert signature["head_reduction"] == "sum"
    assert signature["mpr_vfc_reduction"] == "mean_rows_and_actions"
    assert "terminal_pairs" in signature["transitions"]
    assert "without_replacement" in signature["sampling"]
    assert "shared_heads_and_critics" in signature["curvature_probe"]
    assert "seed" not in pave_training_signature(config, include_seed=False)
    assert pave_training_signature(config) != pave_training_signature({**config, "seed": 4})
    assert "seed" not in pave_training_signature({"outer_pave_mpr_coef": .1})
    assert pave_training_signature({}) is None


def test_auxiliary_normalization_cannot_drop_auxiliary_pave_weights():
    projected = scientific_trial_parameters({
        "alg": "AMBITDMPC2/AMBITDMPC2", "seed": 3,
        "alg_params": {"aux_return_mode": "sac", "aux_return_pave_vfc_coef": .1},
    })["alg_params"]
    assert projected["outer_pave_training"]["controls"]["aux_return_pave_vfc_coef"] == .1
    assert not any(key in projected for key in PAVE_DEFAULTS)
    assert "seed" not in projected["outer_pave_training"]
