"""The auxiliary learner is independent of split value semantics and routing."""

import pytest

from tests.test_ambi_config_decoupling import _build_cfg


def test_off_defaults_do_not_enable_split_or_change_critic_weight():
    cfg = _build_cfg()
    assert cfg.aux_return_mode == "off"
    assert cfg.critic_value_mode == "single"
    assert cfg.critic_coef == 1.0
    assert not hasattr(cfg, "aux_return_actor_cfg")


@pytest.mark.parametrize("mode", ["sac", "return_actor"])
def test_auxiliary_defaults_and_explicit_independent_knobs(mode):
    cfg = _build_cfg(aux_return_mode=mode, actor_lr=.001, critic_lr=.002,
                     aux_return_actor_lr=.003, aux_return_ent_coef="off",
                     aux_return_log_std_min=-7, inner_actor_source="sac")
    assert cfg.inner_finite_horizon
    assert cfg.aux_return_critic_coef == .1
    assert cfg.aux_return_critic_lr == .002
    assert cfg.aux_return_actor_cfg["actor_lr"] == .003
    assert cfg.aux_return_actor_cfg["ent_coef"] == 0
    assert cfg.actor_lr == .001 and cfg.ent_coef == "auto"
    assert cfg.aux_return_actor_cfg["log_std_min"] == -7
    assert cfg.inner_log_std_min == cfg.log_std_min


def test_selected_actor_inherits_distribution_and_entropy_statistic():
    cfg = _build_cfg(aux_return_mode="return_actor", inner_actor_source="return_actor",
                     aux_return_log_std_mapping="tdmpc2_tanh",
                     aux_return_log_std_min=-8, aux_return_log_std_max=1,
                     aux_return_outer_actor_entropy_mode="tdmpc2_scaled",
                     aux_return_ent_coef=.01, inner_temperature_mode="fixed")
    assert cfg.inner_log_std_mapping == "tdmpc2_tanh"
    assert (cfg.inner_log_std_min, cfg.inner_log_std_max) == (-8, 1)
    assert cfg.inner_actor_entropy_mode == "tdmpc2_scaled"


def test_inner_entropy_off_is_independent_of_primary_soft_critic():
    cfg = _build_cfg(aux_return_mode="sac", inner_entropy_enabled=False)
    assert cfg.outer_critic_target == "entropy_augmented"
    assert cfg.inner_sac_critic_target == "reward_only"
    assert cfg.inner_temperature_updates_per_action == 0


def test_auxiliary_regularizer_alone_requests_behavior_metadata():
    cfg = _build_cfg(aux_return_mode="return_actor",
                     aux_return_outer_behavior_policy_kl_schedule="dual")
    assert cfg.outer_behavior_policy_kl_schedule == "none"
    assert cfg.store_behavior_policy


@pytest.mark.parametrize("options,match", [
    ({"critic_value_mode": "return_entropy"}, "split heads"),
    ({"inner_finite_horizon": False}, "finite_horizon"),
    ({"inner_operator": "td3"}, "inner_operator"),
    ({"inner_actor_initialization": "random"}, "initialization"),
    ({"inner_actor_scope": "run"}, "scope"),
    ({"inner_actor_writeback_coef": .1}, "writeback"),
    ({"aux_return_detach_representation": 1}, "bool"),
    ({"aux_return_ent_coef": -1}, "ent_coef"),
    ({"aux_return_ent_coef": "auto_-1"}, "ent_coef"),
    ({"aux_return_log_std_min": 3}, "log_std"),
])
def test_invalid_auxiliary_settings_rejected_before_construction(options, match):
    with pytest.raises(ValueError, match=match):
        _build_cfg(aux_return_mode="return_actor", **options)


@pytest.mark.parametrize("key,value", [
    ("inner_actor_source", "return_actor"),
    ("inner_horizon_actor_source", "return_actor"),
    ("inner_critic_source", "aux_return"),
    ("inner_horizon_critic_source", "aux_return"),
])
def test_unavailable_sources_are_errors(key, value):
    with pytest.raises(ValueError, match=key):
        _build_cfg(**{key: value})


def test_auxiliary_scale_can_be_selected_without_return_actor():
    cfg = _build_cfg(aux_return_mode="sac", inner_critic_source="aux_return",
                     aux_return_sac_actor_loss_scale_mode="tdmpc2_percentile_range",
                     inner_actor_loss_scale_update="per_update",
                     inner_temperature_mode="fixed")
    assert cfg.sac_actor_loss_scale_mode == "none"
    assert cfg.aux_return_actor_cfg["sac_actor_loss_scale_mode"] == "tdmpc2_percentile_range"


@pytest.mark.parametrize("operator", ["none", "mppi"])
def test_scaled_return_actor_can_supply_prior_and_mppi_actions(operator):
    cfg = _build_cfg(aux_return_mode="return_actor", inner_operator=operator,
                     inner_actor_source="return_actor",
                     aux_return_outer_actor_entropy_mode="tdmpc2_scaled", aux_return_ent_coef=.01)
    assert cfg.inner_operator == operator
    assert cfg.aux_return_actor_cfg["outer_actor_entropy_mode"] == "tdmpc2_scaled"


def test_scaled_inner_automatic_temperature_inherits_numeric_selected_actor_target():
    cfg = _build_cfg(aux_return_mode="return_actor", inner_actor_source="return_actor",
                     aux_return_outer_actor_entropy_mode="tdmpc2_scaled",
                     aux_return_target_entropy=-.7, aux_return_ent_coef="auto_0.1",
                     inner_temperature_mode="auto", inner_temperature_initialization="inherit_outer",
                     inner_target_entropy="inherit_outer")
    assert cfg.inner_actor_entropy_mode == "tdmpc2_scaled"
    assert cfg.inner_target_entropy == "inherit_outer"
