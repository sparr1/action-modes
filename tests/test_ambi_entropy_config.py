"""Independent actor entropy selections and their temperature target units."""

import pytest

from tests.test_ambi_config_decoupling import _build_cfg


def test_entropy_defaults_remain_independent_squashed_objectives():
    cfg = _build_cfg()
    assert cfg.outer_actor_entropy_mode == cfg.inner_actor_entropy_mode == "squashed"
    assert cfg.target_entropy == "auto"
    assert cfg.inner_target_entropy == "inherit_outer"


@pytest.mark.parametrize("outer", ["squashed", "tdmpc2_scaled"])
@pytest.mark.parametrize("inner", ["squashed", "tdmpc2_scaled"])
def test_all_independent_entropy_combinations_with_explicit_targets(outer, inner):
    cfg = _build_cfg(
        outer_actor_entropy_mode=outer.upper(), inner_actor_entropy_mode=inner.upper(),
        target_entropy=3.5, inner_target_entropy=-2.0,
    )
    assert cfg.outer_actor_entropy_mode == outer
    assert cfg.inner_actor_entropy_mode == inner
    assert cfg.target_entropy == 3.5
    assert cfg.inner_target_entropy == -2.0


@pytest.mark.parametrize("key", ["outer_actor_entropy_mode", "inner_actor_entropy_mode"])
@pytest.mark.parametrize("value", [None, True, 0, "gaussian", "auto"])
def test_entropy_mode_is_a_strict_choice(key, value):
    with pytest.raises(ValueError, match=key):
        _build_cfg(**{key: value})


@pytest.mark.parametrize("coefficient", ["auto", "auto_0.03"])
@pytest.mark.parametrize("target", ["auto", "inherit_outer"])
def test_scaled_outer_auto_requires_numeric_target(coefficient, target):
    with pytest.raises(ValueError, match="explicit numeric target_entropy"):
        _build_cfg(
            outer_actor_entropy_mode="tdmpc2_scaled", ent_coef=coefficient,
            target_entropy=target, inner_target_entropy="auto",
        )


@pytest.mark.parametrize("outer", ["squashed", "tdmpc2_scaled"])
@pytest.mark.parametrize("target", ["auto", "inherit_outer"])
def test_scaled_inner_auto_requires_numeric_target_even_with_matching_outer(outer, target):
    with pytest.raises(ValueError, match="explicit numeric inner_target_entropy"):
        _build_cfg(
            outer_actor_entropy_mode=outer, target_entropy=3.0,
            inner_actor_entropy_mode="tdmpc2_scaled", inner_target_entropy=target,
        )


def test_squashed_inner_auto_cannot_inherit_a_scaled_outer_target():
    with pytest.raises(ValueError, match="inherit_outer target entropy"):
        _build_cfg(outer_actor_entropy_mode="tdmpc2_scaled", ent_coef=1e-4)
    for target in ("auto", -1.0):
        cfg = _build_cfg(
            outer_actor_entropy_mode="tdmpc2_scaled", ent_coef=1e-4,
            inner_target_entropy=target,
        )
        assert cfg.inner_actor_entropy_mode == "squashed"
        assert cfg.inner_target_entropy == target


@pytest.mark.parametrize("temperature_mode", ["fixed", "inherit_outer"])
def test_fixed_scaled_objectives_do_not_require_an_unused_target(temperature_mode):
    cfg = _build_cfg(
        outer_actor_entropy_mode="tdmpc2_scaled", ent_coef=1e-4,
        inner_actor_entropy_mode="tdmpc2_scaled",
        inner_temperature_mode=temperature_mode,
    )
    assert cfg.target_entropy == "auto"
    assert cfg.inner_target_entropy == "inherit_outer"


@pytest.mark.parametrize("operator", ["none", "td3", "mppi"])
def test_scaled_inner_rejects_non_sac_operators(operator):
    with pytest.raises(ValueError, match="inner_actor_entropy_mode.*inner_operator"):
        _build_cfg(inner_operator=operator, inner_actor_entropy_mode="tdmpc2_scaled")


def test_scaled_inner_rejects_shared_mixture_but_outer_remains_independent():
    with pytest.raises(ValueError, match="does not support.*shared_mixture"):
        _build_cfg(
            inner_actor_entropy_mode="tdmpc2_scaled", inner_target_entropy=2.0,
            inner_explorer_mode="shared_mixture",
        )
    cfg = _build_cfg(
        outer_actor_entropy_mode="tdmpc2_scaled", ent_coef=1e-4,
        inner_target_entropy="auto", inner_explorer_mode="shared_mixture",
    )
    assert cfg.inner_actor_entropy_mode == "squashed"


@pytest.mark.parametrize("explorer", ["none", "frozen_random", "adaptive_param_noise", "separate_critics"])
def test_scaled_inner_accepts_gaussian_explorers(explorer):
    extras = {"inner_param_noise_actor_count": 2} if explorer == "adaptive_param_noise" else {}
    cfg = _build_cfg(
        inner_actor_entropy_mode="tdmpc2_scaled", inner_target_entropy=2.0,
        inner_explorer_mode=explorer, **extras,
    )
    assert cfg.inner_actor_entropy_mode == "tdmpc2_scaled"


def test_documented_fixed_tdmpc2_recipe_resolves_without_changing_inner_entropy():
    cfg = _build_cfg(
        outer_actor_entropy_mode="tdmpc2_scaled", ent_coef=0.0001,
        outer_q_actor_reduction="mean_pair", num_q=5, q_pair_size=2,
        log_std_mapping="tdmpc2_tanh", log_std_min=-10, log_std_max=2,
        sac_actor_loss_scale_mode="tdmpc2_percentile_range",
        sac_actor_loss_scale_tau=0.01, rho=0.5, actor_lr=0.0003,
        actor_adam_eps=0.00001, grad_clip_norm=20,
        outer_behavior_policy_kl_schedule="none", inner_target_entropy="auto",
        inner_temperature_mode="inherit_outer",
        outer_critic_target="reward_only", inner_sac_critic_target="reward_only",
    )
    assert cfg.outer_actor_entropy_mode == "tdmpc2_scaled"
    assert cfg.inner_actor_entropy_mode == "squashed"
    assert cfg.inner_temperature_mode == "inherit_outer"
    assert cfg.inner_target_entropy == "auto"


@pytest.mark.parametrize("scale", ["none", "tdmpc2_percentile_range"])
@pytest.mark.parametrize("coefficient", [0.25, "auto", "auto_0.25"])
@pytest.mark.parametrize("temperature", ["fixed", "inherit_outer", "auto"])
@pytest.mark.parametrize("outer_target", ["reward_only", "entropy_augmented"])
@pytest.mark.parametrize("inner_target", ["reward_only", "entropy_augmented"])
def test_q_scaling_temperature_and_critic_target_matrix(
    scale, coefficient, temperature, outer_target, inner_target,
):
    params = dict(
        sac_actor_loss_scale_mode=scale, ent_coef=coefficient,
        inner_temperature_mode=temperature,
        outer_critic_target=outer_target, inner_sac_critic_target=inner_target,
    )
    allowed = scale == "none" or (
        isinstance(coefficient, float) and temperature != "auto"
    )
    if allowed:
        _build_cfg(**params)
    else:
        with pytest.raises(ValueError, match="requires fixed temperatures"):
            _build_cfg(**params)


def test_q_scaling_error_lists_all_conflicts_and_correction():
    with pytest.raises(ValueError) as error:
        _build_cfg(sac_actor_loss_scale_mode="tdmpc2_percentile_range")
    for field in ("ent_coef", "inner_temperature_mode"):
        assert field in str(error.value)
    assert "sac_actor_loss_scale_mode='none'" in str(error.value)


@pytest.mark.parametrize("operator", ["none", "td3", "mppi"])
def test_q_scaling_ignores_unused_inner_critic_target(operator):
    cfg = _build_cfg(
        sac_actor_loss_scale_mode="tdmpc2_percentile_range", ent_coef=0.25,
        outer_critic_target="reward_only", inner_operator=operator,
        inner_sac_critic_target="entropy_augmented",
    )
    assert cfg.inner_sac_critic_target == "entropy_augmented"


@pytest.mark.parametrize("explorer", ["shared_mixture", "separate_critics"])
def test_q_scaling_rejects_adaptive_explorer_temperatures(explorer):
    with pytest.raises(ValueError, match="inner_temperature_mode='auto'"):
        _build_cfg(
            sac_actor_loss_scale_mode="tdmpc2_percentile_range", ent_coef=0.25,
            outer_critic_target="reward_only",
            inner_sac_critic_target=("entropy_augmented" if explorer == "shared_mixture" else "reward_only"),
            inner_explorer_mode=explorer, inner_temperature_mode="auto",
        )


def test_behavior_quantile_tracking_allows_adaptive_soft_critics():
    cfg = _build_cfg(outer_behavior_policy_kl_schedule="quantile_gate")
    assert cfg.sac_actor_loss_scale_mode == "none"
    assert cfg.ent_coef == cfg.inner_temperature_mode == "auto"


@pytest.mark.parametrize("explorer", ["frozen_random", "adaptive_param_noise", "shared_mixture", "separate_critics"])
def test_q_scaled_entropy_critics_reject_explorer_combinations(explorer):
    extras = {"inner_param_noise_actor_count": 2} if explorer == "adaptive_param_noise" else {}
    with pytest.raises(ValueError, match="inner_explorer_mode='none'"):
        _build_cfg(
            sac_actor_loss_scale_mode="tdmpc2_percentile_range", ent_coef=0.25,
            inner_temperature_mode="inherit_outer", inner_explorer_mode=explorer,
            **extras,
        )


def test_scaled_auto_entropy_accepts_explicit_target_and_unclipped_temperature():
    cfg = _build_cfg(
        outer_actor_entropy_mode="tdmpc2_scaled", ent_coef="auto_0.0001",
        inner_actor_entropy_mode="tdmpc2_scaled", target_entropy=-441,
        inner_target_entropy=-441, inner_temperature_mode="auto",
        inner_temperature_initialization="inherit_outer",
        inner_temperature_grad_clip_norm=None,
    )
    assert cfg.inner_temperature_grad_clip_norm is None
    assert cfg.target_entropy == cfg.inner_target_entropy == -441.0
