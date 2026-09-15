"""Configuration contracts for the opt-in split-value implementation."""

import json
from pathlib import Path

import pytest

from tests.test_ambi_config_decoupling import _build_cfg


def _split(**params):
    return _build_cfg(critic_value_mode="return_entropy", **params)


def test_legacy_defaults_remain_single_and_entropy_enabled():
    cfg = _build_cfg()
    assert cfg.critic_value_mode == "single"
    assert cfg.inner_entropy_enabled is True
    assert cfg.inner_value_initialization == "return"
    assert cfg.inner_temperature_mode == "auto"
    assert cfg.inner_finite_horizon is False


def test_split_defaults_resolve_reward_inner_and_finite_horizon():
    cfg = _split()
    assert cfg.inner_entropy_enabled is False
    assert cfg.inner_value_initialization == "return"
    assert cfg.inner_sac_critic_target == "reward_only"
    assert cfg.inner_temperature_updates_per_action == 0
    assert cfg.inner_finite_horizon is True
    assert cfg.inner_rebase_persistent is False
    assert cfg.q_num_bins == 101
    assert (cfg.q_vmin, cfg.q_vmax) == (-10, 10)
    assert cfg.critic_value_spec == {
        "critic_value_mode": "return_entropy",
        "value_components": ["return", "entropy"],
    }


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("initialization", ["return", "soft"])
@pytest.mark.parametrize("adaptation", ["clone", "lora_rl", "frozen"])
def test_inner_initialization_entropy_and_trainability_are_independent(
    enabled, initialization, adaptation,
):
    cfg = _split(
        inner_entropy_enabled=enabled,
        inner_value_initialization=initialization.upper(),
        inner_critic_adaptation=adaptation,
    )
    assert cfg.inner_value_initialization == initialization
    assert cfg.inner_critic_initialization == "prior"
    assert cfg.inner_critic_adaptation == adaptation
    if not enabled:
        assert cfg.inner_temperature_updates_per_action == 0
    if adaptation == "frozen":
        assert cfg.inner_critic_updates_per_action == 0


@pytest.mark.parametrize("mode", ["inherit_outer", "fixed", "auto"])
def test_inner_entropy_accepts_existing_temperature_modes(mode):
    cfg = _split(inner_entropy_enabled=True, inner_temperature_mode=mode)
    assert cfg.inner_sac_critic_target == "entropy_augmented"
    assert (cfg.inner_temperature_updates_per_action > 0) == (mode == "auto")


@pytest.mark.parametrize("mode", ["inherit_outer", "FIXED", "auto"])
def test_entropy_off_clears_inactive_temperature_mode_and_budget(mode):
    cfg = _split(inner_temperature_mode=mode,
                 inner_temperature_updates_per_action=7)
    assert cfg.inner_temperature_mode == "inherit_outer"
    assert cfg.inner_temperature_updates_per_action == 0
    assert cfg.inner_schedule_mode == "canonical"


@pytest.mark.parametrize("enabled", [False, True])
def test_fixed_temperatures_support_return_normalization(enabled):
    cfg = _split(
        inner_entropy_enabled=enabled,
        ent_coef=0.005,
        inner_temperature_mode="inherit_outer",
        sac_actor_loss_scale_mode="tdmpc2_percentile_range",
    )
    assert cfg.inner_actor_loss_scale_update == "per_action"


@pytest.mark.parametrize("params", [
    {"ent_coef": "auto"},
    {"ent_coef": 0.01, "inner_entropy_enabled": True, "inner_temperature_mode": "auto"},
])
def test_learned_temperatures_reject_moving_normalization(params):
    with pytest.raises(ValueError, match="fixed temperatures"):
        _split(sac_actor_loss_scale_mode="tdmpc2_percentile_range", **params)


def test_prior_only_split_has_no_inactive_finite_horizon_or_temperature_work():
    cfg = _split(inner_operator="none")
    assert cfg.inner_finite_horizon is False
    assert cfg.inner_rounds == cfg.inner_temperature_updates_per_action == 0


@pytest.mark.parametrize("key,value", [
    ("critic_value_mode", "return_soft"),
    ("critic_value_mode", None),
    ("inner_entropy_enabled", "false"),
    ("inner_entropy_enabled", 0),
    ("inner_value_initialization", "random"),
    ("inner_value_initialization", None),
])
def test_new_choices_are_strict(key, value):
    with pytest.raises(ValueError, match=key):
        _build_cfg(**{"critic_value_mode": "return_entropy", key: value})


@pytest.mark.parametrize("params", [
    {"inner_entropy_enabled": False},
    {"inner_value_initialization": "soft"},
])
def test_nondefault_new_semantics_require_split_mode(params):
    with pytest.raises(ValueError, match="require critic_value_mode"):
        _build_cfg(**params)


@pytest.mark.parametrize("key,value", [
    ("q_representation", "scalar"),
    ("inner_finite_horizon", False),
    ("outer_critic_target", "reward_only"),
    ("inner_sac_critic_target", "entropy_augmented"),
    ("inner_bootstrap_source", "outer_online"),
    ("inner_critic_target_initialization", "outer_target"),
    ("inner_actor_initialization", "random"),
    ("inner_critic_initialization", "random"),
    ("inner_rebase_persistent", True),
    ("inner_actor_writeback_coef", 0.1),
    ("inner_critic_writeback_coef", 0.1),
    ("value_equivalence_loss_coef", 0.1),
])
def test_unsupported_split_combinations_fail_before_model_construction(key, value):
    with pytest.raises(ValueError, match=key):
        _split(**{key: value})


@pytest.mark.parametrize("field", [
    "outer_q_target_reduction", "outer_q_actor_reduction",
    "inner_q_target_reduction", "inner_q_actor_reduction",
])
def test_split_reduction_requires_shared_min_pair(field):
    with pytest.raises(ValueError, match=field):
        _split(**{field: "mean_pair"})


@pytest.mark.parametrize("component", [
    "actor", "critic", "temperature", "replay",
    "actor_optimizer", "critic_optimizer", "temperature_optimizer",
])
def test_all_split_inner_state_is_action_local(component):
    params = {f"inner_{component}_scope": "episode"}
    if component.endswith("_optimizer"):
        params[f"inner_{component.removesuffix('_optimizer')}_scope"] = "episode"
    with pytest.raises(ValueError, match="scope"):
        _split(**params)


def test_split_per_update_scale_is_rejected():
    with pytest.raises(ValueError, match="inner_actor_loss_scale_update"):
        _split(ent_coef=0.01, inner_actor_loss_scale_update="per_update",
               sac_actor_loss_scale_mode="tdmpc2_percentile_range")


def test_native_tdambi_rejects_split_semantics_before_constructing_models():
    from RL.TDAMBI import TDAMBI

    native = object.__new__(TDAMBI)
    with pytest.raises(ValueError, match="critic_value_mode|split"):
        native._build_cfg({"critic_value_mode": "return_entropy"})


def test_split_component_metrics_use_critic_update_weighting():
    from tests.test_ambi_inner_decoupling import _model

    holder = _model()
    try:
        keys = (
            "inner_primary_critic_loss", "inner_primary_q_mean",
            "inner_primary_target_mean", "inner_primary_target_clip_fraction",
            "inner_initialization_residual_critic_loss",
            "inner_initialization_residual_q_mean",
            "inner_initialization_residual_target_mean",
            "inner_initialization_residual_target_clip_fraction",
            "inner_value_composition_coefficient", "inner_alpha_used",
            "inner_effective_entropy_coefficient",
        )
        for count, value in ((1, 2.0), (3, 6.0)):
            holder.agent.last_inner_metrics = {
                "inner_critic_optimizer_steps": count,
                "inner_actor_optimizer_steps": 7,
                **dict.fromkeys(keys, value),
            }
            holder._record_action_metrics(planned=True, action_seconds=0.0)
        snapshot = holder._wandb_train_window.snapshot()
        for key in keys:
            assert snapshot[f"train/{key}"] == pytest.approx(5.0)
            assert snapshot[f"train/{key}_count"] == 4
    finally:
        holder.close()


def test_split_actor_coefficients_use_actor_update_weighting():
    from tests.test_ambi_inner_decoupling import _model

    holder = _model()
    try:
        keys = ("inner_actor_alpha_used", "inner_actor_effective_entropy_coefficient",
                "inner_actor_value_composition_coefficient")
        for count, value in ((1, 2.0), (3, 6.0)):
            holder.agent.last_inner_metrics = {
                "inner_critic_optimizer_steps": 7,
                "inner_actor_optimizer_steps": count,
                **dict.fromkeys(keys, value),
            }
            holder._record_action_metrics(planned=True, action_seconds=0.0)
        snapshot = holder._wandb_train_window.snapshot()
        for key in keys:
            assert snapshot[f"train/{key}"] == pytest.approx(5.0)
    finally:
        holder.close()


def test_split_terminal_reduction_matches_frozen_return_handoff():
    assert _split().mppi_terminal_q_reduction == "min_pair"
    with pytest.raises(ValueError, match="mppi_terminal_q_reduction"):
        _split(mppi_terminal_q_reduction="mean_pair")


@pytest.mark.parametrize("variant", ["entropy_off", "entropy_fixed", "entropy_auto"])
def test_example_configs_resolve_without_starting_training(variant):
    path = (Path(__file__).resolve().parents[1] / "configs/dmcontrol/algs"
            / f"ambi_split_value_{variant}.json")
    config = json.loads(path.read_text())
    cfg = _build_cfg(**config["alg_params"])
    assert cfg.critic_value_mode == "return_entropy"
    assert cfg.inner_entropy_enabled == (variant != "entropy_off")
    assert config["alg_params"]["wandb"] is False
