"""Configuration and scientific identity for corrected BOOM prior learning."""

from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from tests.test_ambi_config_decoupling import _build_cfg
from utils.resume_identity import scientific_trial_parameters


def _boom_cfg(**overrides):
    return _build_cfg(**{
        "outer_actor_objective": "boom",
        "sac_actor_loss_scale_mode": "tdmpc2_percentile_range",
        "ent_coef": 0.004,
        "inner_temperature_mode": "inherit_outer",
        **overrides,
    })


def _identity(params):
    return scientific_trial_parameters({
        "alg": "AMBITDMPC2/AMBITDMPC2", "seed": 7, "alg_params": params,
    })


def _boom_cfg_for_experiment(experiment_params, **overrides):
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {"seed": 3, "device": "cpu", "env": "test-env", "total_steps": 12}
    algorithm.experiment_params = experiment_params
    params = {
        "device": "cpu",
        "outer_actor_objective": "boom",
        "sac_actor_loss_scale_mode": "tdmpc2_percentile_range",
        "ent_coef": 0.004,
        "inner_temperature_mode": "inherit_outer",
        **overrides,
    }
    algorithm.custom_params = params
    try:
        return algorithm._build_cfg(params)
    finally:
        algorithm.env.close()


def test_boom_is_opt_in_and_does_not_enable_teacher_reanalysis():
    default = _build_cfg()
    assert default.outer_actor_objective == "q_maximization"
    assert default.store_behavior_policy is False
    assert default.store_teacher_policy is False
    assert default.sac_actor_loss_scale_mode == "none"
    cfg = _boom_cfg(outer_actor_objective="BOOM")
    assert cfg.outer_actor_objective == "boom"
    assert cfg.store_behavior_policy is True
    assert cfg.store_teacher_policy is False
    assert cfg.outer_boom_coef == pytest.approx(1 / 50)
    assert cfg.outer_boom_min_std == 0.05
    assert cfg.outer_actor_entropy_mode == cfg.inner_actor_entropy_mode == "squashed"


@pytest.mark.parametrize("dimension,expected", [
    (1, 0.02), (23, 0.46), (24, 0.024), (38, 0.038), (48, 0.048), (49, 0.98),
])
def test_boom_auto_coefficient_uses_resolved_environment_action_dimension(dimension, expected):
    make = gym.make

    def make_with_action_dimension(*args, **kwargs):
        env = make(*args, **kwargs)
        env.action_space = gym.spaces.Box(-1.0, 1.0, (dimension,), dtype=np.float32)
        return env

    with patch("gymnasium.make", side_effect=make_with_action_dimension):
        cfg = _boom_cfg(outer_boom_coef="AUTO")
    assert cfg.action_dim == dimension
    assert cfg.outer_boom_coef == pytest.approx(expected)


@pytest.mark.parametrize("coefficient", [0, 0.123, 3])
def test_boom_supports_explicit_strength_including_zero_ablation(coefficient):
    cfg = _boom_cfg(outer_boom_coef=coefficient, outer_boom_min_std=0.1)
    assert cfg.outer_boom_coef == float(coefficient)
    assert cfg.outer_boom_min_std == 0.1
    assert cfg.store_behavior_policy


@pytest.mark.parametrize("name,value", [
    ("outer_boom_coef", -0.1), ("outer_boom_coef", True),
    ("outer_boom_coef", None), ("outer_boom_coef", float("nan")),
    ("outer_boom_coef", float("inf")), ("outer_boom_coef", "0.1"),
    ("outer_boom_coef", "default"), ("outer_boom_min_std", 0),
    ("outer_boom_min_std", -0.1), ("outer_boom_min_std", False),
    ("outer_boom_min_std", None), ("outer_boom_min_std", float("nan")),
    ("outer_boom_min_std", float("inf")),
])
def test_boom_controls_validate_numbers_strictly(name, value):
    with pytest.raises(ValueError, match=name):
        _boom_cfg(**{name: value})


def test_boom_requires_explicit_q_scaling_without_changing_existing_scale_contract():
    with pytest.raises(ValueError, match="requires explicit sac_actor_loss_scale_mode"):
        _build_cfg(outer_actor_objective="boom")
    with pytest.raises(ValueError, match="fixed temperatures"):
        _boom_cfg(ent_coef="auto")
    with pytest.raises(ValueError, match="fixed temperatures"):
        _boom_cfg(inner_temperature_mode="auto")


@pytest.mark.parametrize("options,match", [
    ({"inner_operator": "none"}, "inner_operator"),
    ({"inner_operator": "mppi"}, "inner_operator"),
    ({"inner_operator": "td3"}, "inner_operator"),
    ({"inner_execution_policy_source": "explorer"}, "inner_execution_policy_source"),
    ({"inner_explorer_mode": "frozen_random"}, "inner_explorer_mode"),
    ({"outer_behavior_policy_kl_schedule": "smooth"}, "outer_behavior_policy_kl_schedule"),
    ({"critic_value_mode": "return_entropy"}, "critic_value_mode"),
    ({"aux_return_mode": "sac"}, "aux_return_mode"),
    ({"aux_return_mode": "return_actor", "inner_actor_source": "return_actor"}, "inner_actor_source"),
])
def test_boom_rejects_unsupported_objective_combinations(options, match):
    with pytest.raises(ValueError, match=match):
        _boom_cfg(**options)


def test_boom_frozen_checkpoint_evaluation_allows_prior_only_controller():
    cfg = _boom_cfg_for_experiment(
        {"frozen_checkpoint_evaluation": True}, inner_operator="none",
    )
    assert cfg.inner_operator == "none"
    assert cfg.outer_actor_objective == "boom"
    assert cfg.sac_actor_loss_scale_mode == "tdmpc2_percentile_range"
    assert cfg.store_behavior_policy


@pytest.mark.parametrize("experiment_params", [
    {}, {"frozen_checkpoint_evaluation": False},
    {"frozen_checkpoint_evaluation": "true"}, {"frozen_checkpoint_evaluation": 1},
])
def test_boom_prior_only_exception_requires_explicit_frozen_evaluation(experiment_params):
    with pytest.raises(ValueError, match="inner_operator='sac'"):
        _boom_cfg_for_experiment(experiment_params, inner_operator="none")


@pytest.mark.parametrize("options,match", [
    ({"inner_operator": "mppi"}, "inner_operator='sac'"),
    ({"inner_operator": "td3"}, "inner_operator='sac'"),
    ({"outer_behavior_policy_kl_schedule": "smooth"}, "behavior-policy regularizer"),
    ({"aux_return_mode": "sac"}, "aux_return_mode"),
])
def test_boom_frozen_evaluation_preserves_other_requirements(options, match):
    with pytest.raises(ValueError, match=match):
        _boom_cfg_for_experiment(
            {"frozen_checkpoint_evaluation": True},
            **{"inner_operator": "none", **options},
        )


def test_boom_checkpoint_loads_into_frozen_prior_only_controller():
    architecture = {
        "model_size": None, "enc_dim": 32, "mlp_dim": 32, "latent_dim": 16,
        "num_enc_layers": 2, "simnorm_dim": 8, "num_bins": 11,
        "num_q": 2, "q_representation": "scalar", "dropout": 0.0,
        "compile": False,
    }
    source = AMBITDMPC2Agent(_boom_cfg_for_experiment({}, **architecture))
    source.actor_loss_scale.fill_(3.2)
    state = source.checkpoint_state()
    restored = AMBITDMPC2Agent(_boom_cfg_for_experiment(
        {"frozen_checkpoint_evaluation": True}, inner_operator="none", **architecture,
    ))
    restored.load(state)
    assert restored.cfg.inner_operator == "none"
    assert restored._boom_spec() == state["boom_spec"]
    assert restored._checkpoint_version() == state["checkpoint_version"] == 9
    torch.testing.assert_close(restored.actor_loss_scale, source.actor_loss_scale)
    for key, value in source.model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[key], value)


@pytest.mark.parametrize("execution", ["policy_sample", "mean", "mean_plus_gaussian"])
def test_boom_collects_policy_mean_for_stochastic_and_deterministic_execution(execution):
    cfg = _boom_cfg(inner_execution_action=execution)
    assert cfg.inner_execution_action == execution
    assert cfg.store_behavior_policy


def test_boom_does_not_require_fresh_or_active_inner_solve():
    inactive = _boom_cfg(inner_rounds=0)
    assert inactive.store_behavior_policy and inactive.inner_rounds == 0
    persistent = _boom_cfg(inner_actor_scope="episode")
    assert persistent.inner_actor_scope == "episode"


def test_boom_requires_nonnegative_temporal_weights():
    with pytest.raises(ValueError, match="rho >= 0"):
        _boom_cfg(rho=-0.1)
    assert _boom_cfg(rho=0.0).rho == 0.0
    assert _build_cfg(rho=-0.1).rho == -0.1


@pytest.mark.parametrize("objective", ["q_maximization", "distillation"])
def test_inactive_boom_controls_preserve_existing_fingerprints(objective):
    base = {"outer_actor_objective": objective}
    assert _identity(base) == _identity({
        **base, "outer_boom_coef": "auto", "outer_boom_min_std": 0.05,
    })
    assert _identity(base) == _identity({
        **base, "outer_boom_coef": 0.2, "outer_boom_min_std": 0.1,
    })


def test_boom_identity_normalizes_defaults_and_ignores_reanalysis_controls():
    base = {"outer_actor_objective": "boom"}
    default = _identity(base)
    assert default == _identity({
        "outer_actor_objective": "BOOM", "outer_boom_coef": "AUTO",
        "outer_boom_min_std": 0.05, "store_behavior_policy": True,
        "store_teacher_policy": False, "reanalyze_interval": 77,
        "reanalyze_batch_size": 8, "reanalyze_backend": "batched_cuda_graph",
        "reanalyze_parallel_roots": 3, "reanalyze_batched_optimizations": False,
        "reanalyze_batched_workers": 5,
    })
    assert default != _identity({})
    assert default != _identity({**base, "outer_boom_coef": 0.2})
    assert default != _identity({**base, "outer_boom_min_std": 0.1})
    assert _identity({**base, "outer_boom_coef": 2}) == _identity({
        **base, "outer_boom_coef": 2.0,
    })
