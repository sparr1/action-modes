"""Auxiliary learning and runtime routes have distinct scientific identities."""

import pytest

from utils.aux_return_identity import (
    AUX_RETURN_SOURCE_DEFAULTS,
    auxiliary_return_architecture,
    auxiliary_return_routing,
)
from utils.resume_identity import scientific_trial_parameters


def training_identity(**config):
    return scientific_trial_parameters({
        "alg": "AMBITDMPC2/AMBITDMPC2", "alg_params": config,
    })


@pytest.mark.parametrize("value_mode", ["single", "return_entropy"])
def test_off_preserves_existing_training_identity(value_mode):
    original = training_identity(critic_value_mode=value_mode)
    explicit_off = training_identity(
        critic_value_mode=value_mode, aux_return_mode="OFF",
        aux_return_actor_cfg={}, aux_return_ent_coef=0,
        **AUX_RETURN_SOURCE_DEFAULTS,
    )
    assert explicit_off == original


@pytest.mark.parametrize("mode", ["sac", "return_actor"])
def test_auxiliary_training_identity_retains_mode_and_training_changes(mode):
    base = {"aux_return_mode": mode, "aux_return_critic_coef": 0.1}
    assert training_identity(**base) != training_identity()
    assert training_identity(**base) != training_identity(
        **{**base, "aux_return_critic_coef": 0.2})
    assert training_identity(**base) == training_identity(
        **base, aux_return_actor_cfg={"critic_coef": 0.1},
        **AUX_RETURN_SOURCE_DEFAULTS)


@pytest.mark.parametrize("key,source", [
    ("inner_actor_source", "return_actor"),
    ("inner_critic_source", "aux_return"),
    ("inner_horizon_actor_source", "return_actor"),
    ("inner_horizon_critic_source", "aux_return"),
])
def test_each_active_route_changes_resume_identity_and_report(key, source):
    base = {"aux_return_mode": "return_actor", "inner_operator": "sac"}
    changed = {**base, key: source}
    assert training_identity(**base) != training_identity(**changed)
    assert auxiliary_return_routing(base) != auxiliary_return_routing(changed)
    assert training_identity(**base) == training_identity(**{**base, **AUX_RETURN_SOURCE_DEFAULTS})
    assert auxiliary_return_architecture(base) == auxiliary_return_architecture(changed)


def test_architecture_distinguishes_optional_actor_and_its_bounds():
    sac = {"aux_return_mode": "sac"}
    actor = {"aux_return_mode": "return_actor"}
    assert auxiliary_return_architecture({}) == ()
    assert auxiliary_return_architecture(sac) != auxiliary_return_architecture(actor)
    assert auxiliary_return_architecture(actor) != auxiliary_return_architecture(
        {**actor, "aux_return_log_std_min": -10})
    assert auxiliary_return_architecture(actor) == auxiliary_return_architecture(
        {**actor, "aux_return_actor_cfg": {"log_std_min": -20, "log_std_max": 2,
                                           "log_std_mapping": "direct_clamp"}})


def test_inherited_auxiliary_controls_equal_explicit_resolved_defaults():
    original = {"aux_return_mode": "return_actor", "actor_lr": 3e-4,
                "critic_lr": 1e-4, "ent_coef": 0.5,
                "outer_behavior_policy_kl_min_valid_count": "auto", "batch_size": 32}
    resolved = {**original, "aux_return_actor_lr": 3e-4, "aux_return_critic_lr": 1e-4,
                "aux_return_ent_coef": 0.5, "aux_return_critic_coef": 0.1,
                "aux_return_detach_representation": False,
                "aux_return_outer_behavior_policy_kl_min_valid_count": 32}
    assert training_identity(**original) == training_identity(**resolved)


def test_routing_reporting_is_absent_for_old_configs_and_explicit_when_enabled():
    assert auxiliary_return_routing({}) is None
    report = auxiliary_return_routing({"aux_return_mode": "sac", "inner_operator": "mppi",
                                      "inner_horizon_critic_source": "aux_return"})
    assert report == {"aux_return_mode": "sac", "inner_operator": "mppi",
                      **AUX_RETURN_SOURCE_DEFAULTS,
                      "inner_horizon_critic_source": "aux_return"}


def test_auxiliary_resume_identity_resolves_finite_horizon_and_actor_inheritance():
    base = {"aux_return_mode": "return_actor", "inner_actor_source": "return_actor",
            "aux_return_log_std_min": -8., "aux_return_log_std_max": 1.,
            "aux_return_log_std_mapping": "tdmpc2_tanh",
            "aux_return_outer_actor_entropy_mode": "tdmpc2_scaled"}
    assert training_identity(**base) == training_identity(
        **base, inner_finite_horizon=True, inner_log_std_min=-8., inner_log_std_max=1.,
        inner_log_std_mapping="tdmpc2_tanh", inner_actor_entropy_mode="tdmpc2_scaled")
