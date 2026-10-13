"""PAVE controls and portable training semantics without learner imports."""

import math
from collections.abc import Mapping
from numbers import Integral, Real


PAVE_COEFFICIENT_KEYS = tuple(
    f"{prefix}_pave_{term}_coef"
    for prefix in ("outer", "aux_return")
    for term in ("mpr", "vfc", "curvature")
)
PAVE_DEFAULTS = {
    **dict.fromkeys(PAVE_COEFFICIENT_KEYS, 0.0),
    "outer_pave_batch_size": 64,
    "outer_pave_obs_noise_std": 0.01,
    "outer_pave_curvature_margin": 1.0,
}


def resolve_pave_controls(config):
    """Validate and canonicalize controls, including inactive numeric values."""
    if not isinstance(config, Mapping):
        config = vars(config)
    controls = {key: config.get(key, default) for key, default in PAVE_DEFAULTS.items()}
    batch = controls["outer_pave_batch_size"]
    if batch is not None:
        if isinstance(batch, bool) or not isinstance(batch, Integral) or batch <= 0:
            raise ValueError("outer_pave_batch_size must be a positive integer or null.")
        controls["outer_pave_batch_size"] = int(batch)
    for key in (*PAVE_COEFFICIENT_KEYS, "outer_pave_obs_noise_std",
                "outer_pave_curvature_margin"):
        value = controls[key]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{key} must be a finite non-negative number.")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite non-negative number.")
        controls[key] = value if value else 0.0
    return controls


def pave_training_signature(config, *, include_seed=True):
    """Describe active PAVE training; absence preserves pre-PAVE checkpoints."""
    if not isinstance(config, Mapping):
        config = vars(config)
    controls = resolve_pave_controls(config)
    if not any(controls[key] > 0 for key in PAVE_COEFFICIENT_KEYS):
        return None
    signature = {
        "version": 1,
        "controls": controls,
        "cadence": "every_outer_update",
        "transitions": "flatten_real_time_batch_include_terminal_pairs",
        "sampling": "uniform_without_replacement_when_capped",
        "perturbation": "isotropic_gaussian_observation_before_encoder",
        "representation_gradients": "detached",
        "actions": "same_recorded_normalized_action",
        "dropout": "disabled_for_geometry_only",
        "head_reduction": "sum",
        "mpr_vfc_reduction": "mean_rows_and_actions",
        "curvature_reduction": "mean_rows_relu_action_sum_vHv_plus_margin",
        "curvature_probe": "one_rademacher_per_row_shared_heads_and_critics",
        "rng": "seed_plus_1000003_update_plus_0x50415645_mod_2pow63minus1_v1",
    }
    if include_seed and "seed" in config:
        signature["seed"] = int(config["seed"])
    return signature


def normalize_pave_identity(config):
    """Drop disabled controls; hash active resolved settings and semantics."""
    result = dict(config)
    signature = pave_training_signature(result, include_seed=False)
    for key in PAVE_DEFAULTS:
        result.pop(key, None)
    result.pop("outer_pave_training", None)
    if signature is not None:
        # The trial already carries its seed. Do not inject a second default
        # seed when hashing an unresolved algorithm configuration.
        result["outer_pave_training"] = signature
    return result
