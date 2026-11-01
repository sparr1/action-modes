"""PAVE is opt-in and rejects unsupported observation/value semantics early."""

import numpy as np
import pytest

from RL.AMBITDMPC2 import AMBITDMPC2
from tests.test_ambi_config_decoupling import _build_cfg
from tests.test_tdmpc2_pixels import PixelEnv, _tiny_params
from utils.pave_identity import PAVE_COEFFICIENT_KEYS, PAVE_DEFAULTS, resolve_pave_controls


def test_pave_defaults_are_disabled_without_changing_single_or_auxiliary_modes():
    cfg = _build_cfg()
    assert {key: getattr(cfg, key) for key in PAVE_DEFAULTS} == PAVE_DEFAULTS
    assert cfg.critic_value_mode == "single"
    assert cfg.aux_return_mode == "off"
    assert cfg.outer_pave_action_smoothness_coef == 0.
    assert cfg.aux_return_pave_action_smoothness_coef == 0.
    assert cfg.outer_pave_action_noise_std == .02


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize("target", ["primary", "auxiliary", "both"])
def test_primary_and_auxiliary_pave_weights_are_independent(representation, target):
    options = {"q_representation": representation}
    if target in {"primary", "both"}:
        options.update(outer_pave_mpr_coef=.1, outer_pave_vfc_coef=.2,
                       outer_pave_curvature_coef=.03, outer_pave_action_smoothness_coef=.04)
    if target in {"auxiliary", "both"}:
        options.update(aux_return_mode="sac", aux_return_pave_mpr_coef=.4,
                       aux_return_pave_vfc_coef=.5, aux_return_pave_curvature_coef=.06,
                       aux_return_pave_action_smoothness_coef=.07,
                       aux_return_critic_coef=.7)
    cfg = _build_cfg(**options)
    for key in PAVE_COEFFICIENT_KEYS:
        assert getattr(cfg, key) == options.get(key, 0.)
    assert cfg.critic_coef == 1.
    assert cfg.aux_return_critic_coef == options.get("aux_return_critic_coef", .1)


@pytest.mark.parametrize("key", [key for key in PAVE_COEFFICIENT_KEYS if key.startswith("outer_")])
def test_active_primary_pave_rejects_split_value_critics(key):
    with pytest.raises(ValueError, match="PAVE.*critic_value_mode='single'"):
        _build_cfg(critic_value_mode="return_entropy", **{key: .1})


@pytest.mark.parametrize("key", [key for key in PAVE_COEFFICIENT_KEYS if key.startswith("aux_return_")])
def test_auxiliary_pave_requires_an_auxiliary_critic(key):
    with pytest.raises(ValueError, match="PAVE.*auxiliary return critic"):
        _build_cfg(**{key: .1})


def test_disabled_pave_does_not_restrict_split_value_critics():
    assert _build_cfg(critic_value_mode="return_entropy").critic_value_mode == "return_entropy"


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("key", ["outer_pave_mpr_coef", "outer_pave_action_smoothness_coef"])
def test_pave_rgb_contract_before_network_construction(active, key):
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = PixelEnv()
    algorithm.run_params = {"seed": 3, "device": "cpu", "total_steps": 10}
    algorithm.custom_params = {}
    params = _tiny_params(**{key: .1 if active else 0.})
    try:
        if active:
            with pytest.raises(ValueError, match="PAVE.*state observations"):
                algorithm._build_cfg(params)
        else:
            assert algorithm._build_cfg(params).obs == "rgb"
    finally:
        algorithm.env.close()


@pytest.mark.parametrize("key", (*PAVE_COEFFICIENT_KEYS, "outer_pave_obs_noise_std",
                                 "outer_pave_action_noise_std", "outer_pave_curvature_margin"))
@pytest.mark.parametrize("invalid", [-.1, float("nan"), float("inf"), True, "0.1", None])
def test_pave_numbers_are_finite_nonnegative_numbers_even_when_disabled(key, invalid):
    with pytest.raises(ValueError, match=key):
        resolve_pave_controls({key: invalid})


@pytest.mark.parametrize("invalid", [0, -1, True, np.bool_(True), 2., "2"])
def test_pave_batch_size_requires_positive_integer_or_null(invalid):
    with pytest.raises(ValueError, match="outer_pave_batch_size"):
        _build_cfg(outer_pave_batch_size=invalid)


@pytest.mark.parametrize("batch", [None, 1, np.int64(17)])
def test_full_batch_and_numpy_numeric_controls_resolve_canonically(batch):
    cfg = _build_cfg(outer_pave_batch_size=batch, outer_pave_mpr_coef=np.float64(.1),
                     outer_pave_obs_noise_std=0, outer_pave_curvature_margin=0,
                     outer_pave_action_smoothness_coef=np.float64(.2),
                     outer_pave_action_noise_std=np.float64(.03))
    assert cfg.outer_pave_batch_size == batch
    assert cfg.outer_pave_batch_size is None or type(cfg.outer_pave_batch_size) is int
    assert type(cfg.outer_pave_mpr_coef) is float
    assert type(cfg.outer_pave_obs_noise_std) is float
    assert type(cfg.outer_pave_curvature_margin) is float
    assert type(cfg.outer_pave_action_smoothness_coef) is float
    assert type(cfg.outer_pave_action_noise_std) is float
    assert cfg.outer_pave_action_noise_std == .03


@pytest.mark.parametrize("key", ["outer_pave_curvature_margin", "outer_pave_action_noise_std"])
def test_wrapper_validates_disabled_numeric_controls(key):
    with pytest.raises(ValueError, match=key):
        _build_cfg(**{key: float("nan")})
