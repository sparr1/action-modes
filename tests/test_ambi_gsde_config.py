"""Configuration boundaries for gSDE and analytic Gaussian entropy."""

import pytest

from tests.test_ambi_config_decoupling import _build_cfg
from utils.resume_identity import scientific_trial_parameters


GSDE_DEFAULTS = {
    "actor_distribution": "gaussian",
    "gsde_feature_source": "actor",
    "gsde_log_std_init": -3.0,
    "gsde_mean_clip": 2.0,
    "inner_gsde_noise_scope": "round",
    "inner_gsde_resample_steps": 8,
    "gsde_execution_resample_steps": 8,
}
ALGORITHM = "AMBITDMPC2/AMBITDMPC2"


def test_explicit_gaussian_defaults_preserve_resolved_config_and_identity():
    omitted = _build_cfg()
    explicit = _build_cfg(**GSDE_DEFAULTS)
    assert vars(omitted) == vars(explicit)
    assert scientific_trial_parameters({"alg": ALGORITHM, "alg_params": {}}) == scientific_trial_parameters(
        {"alg": ALGORITHM, "alg_params": GSDE_DEFAULTS},
    )


@pytest.mark.parametrize("feature", ["actor", "latent"])
@pytest.mark.parametrize("scope", ["round", "solve", "real_steps"])
@pytest.mark.parametrize("backend", ["loop", "vmap"])
def test_gsde_options_preserve_selected_training_schedule(feature, scope, backend):
    cfg = _build_cfg(
        actor_distribution="GSDE", gsde_feature_source=feature.upper(),
        inner_gsde_noise_scope=scope.upper(), critic_ensemble_backend=backend,
        inner_rounds=2, inner_rollouts_per_round=8, inner_rollout_horizon=3,
        inner_updates_per_round=2, inner_gsde_resample_steps=5,
        gsde_execution_resample_steps=7, gsde_mean_clip=None,
        gsde_log_std_init=-2.5,
    )
    assert cfg.actor_distribution == "gsde"
    assert cfg.gsde_feature_source == feature
    assert cfg.inner_gsde_noise_scope == scope
    assert cfg.gsde_mean_clip is None
    assert cfg.gsde_log_std_init == -2.5
    assert cfg.inner_gsde_resample_steps == 5
    assert cfg.gsde_execution_resample_steps == 7
    assert cfg.critic_ensemble_backend == backend
    assert cfg.inner_model_step_budget == 48
    assert cfg.inner_actor_updates_per_action == 4


@pytest.mark.parametrize("operator", ["none", "sac"])
def test_prior_only_and_zero_round_gsde_are_supported(operator):
    cfg = _build_cfg(actor_distribution="gsde", inner_operator=operator, inner_rounds=0)
    assert cfg.inner_rounds == 0
    assert cfg.actor_distribution == "gsde"


@pytest.mark.parametrize("episodic", [False, True])
@pytest.mark.parametrize("representation", ["scalar", "distributional"])
def test_gsde_supports_step_updates_and_ordinary_strict_compilation(episodic, representation):
    cfg = _build_cfg(
        actor_distribution="gsde", episodic=episodic,
        q_representation=representation, inner_update_timing="step",
        inner_steps_per_update=8, compile=True, compile_strict=True,
    )
    assert cfg.compile is True
    assert cfg.compile_strict is True
    assert cfg.inner_update_timing == "step"
    assert cfg.inner_steps_per_update == 8


@pytest.mark.parametrize("field,values", [
    ("actor_distribution", [None, True, "normal", 1]),
    ("gsde_feature_source", [None, False, "state", 1]),
    ("inner_gsde_noise_scope", [None, True, "episode", 1]),
    ("gsde_log_std_init", [None, True, "-3", float("inf"), float("nan")]),
    ("gsde_mean_clip", [True, "2", 0, -1, float("inf"), float("nan")]),
    ("inner_gsde_resample_steps", [None, True, 0, -1, 1.5, "8"]),
    ("gsde_execution_resample_steps", [None, True, 0, -1, 1.5, "8"]),
])
def test_gsde_controls_reject_invalid_values(field, values):
    for value in values:
        with pytest.raises(ValueError, match=field):
            _build_cfg(**{field: value})


@pytest.mark.parametrize("overrides", [
    {"inner_operator": "td3"},
    {"inner_operator": "mppi"},
    {"inner_sac_return_estimator": "retrace"},
    {"inner_explorer_mode": "frozen_random"},
    {"inner_explorer_mode": "shared_mixture"},
    {"inner_explorer_mode": "separate_critics"},
    {"inner_explorer_mode": "adaptive_param_noise", "inner_param_noise_actor_count": 8},
    {"inner_actor_initialization": "random"},
    {"inner_critic_initialization": "random"},
    {"inner_actor_adaptation": "frozen"},
    {"inner_critic_adaptation": "frozen"},
    {"inner_critic_adaptation": "lora_rl"},
    {"inner_horizon_conditioning": "one_hot"},
    {"aux_return_mode": "return_actor"},
    {"critic_value_mode": "return_entropy"},
    {"outer_actor_objective": "distillation"},
    {"outer_actor_objective": "po_mpc"},
    {"reanalyze_backend": "batched"},
    {"reanalyze_backend": "cuda_graph"},
    {"reanalyze_backend": "batched_cuda_graph"},
    {"outer_policy_episode_probability": 0.5},
    {"inner_actor_scope": "run"},
    {"inner_critic_scope": "episode"},
    {"inner_replay_scope": "run"},
    {"inner_model_step_budget": 8},
])
def test_unsupported_gsde_combinations_fail_during_resolution(overrides):
    with pytest.raises(ValueError):
        _build_cfg(actor_distribution="gsde", **overrides)


@pytest.mark.parametrize("scope", ["round", "solve", "real_steps"])
@pytest.mark.parametrize("timing", ["round", "step"])
@pytest.mark.parametrize("target", ["reward_only", "entropy_augmented"])
def test_gsde_collection_graph_accepts_static_noise_inputs(scope, timing, target):
    options = dict(
        actor_distribution="gsde", inner_cuda_graph=True, compile=True,
        compile_strict=True, inner_rounds=2, inner_rollouts_per_round=4,
        inner_rollout_horizon=2, inner_gsde_noise_scope=scope,
        inner_update_timing=timing, outer_actor_entropy_mode="pre_tanh",
        inner_actor_entropy_mode="pre_tanh", target_entropy=1.0,
        inner_target_entropy="inherit_outer", outer_critic_target=target,
        inner_sac_critic_target=target,
    )
    if timing == "step":
        options.update(inner_updates_per_round=None, inner_steps_per_update=4)
    else:
        options["inner_updates_per_round"] = 1
    cfg = _build_cfg(**options)
    assert cfg.inner_cuda_graph and cfg.inner_update_timing == timing
    assert cfg.inner_gsde_noise_scope == scope


@pytest.mark.parametrize("overrides", [
    {"compile": False}, {"compile_strict": False}, {"episodic": True},
    {"aux_return_mode": "sac"}, {"inner_execution_action": "mean"},
])
def test_gsde_collection_graph_keeps_unsupported_capture_boundaries(overrides):
    options = dict(actor_distribution="gsde", inner_cuda_graph=True,
                   compile=True, compile_strict=True, inner_rounds=1,
                   inner_updates_per_round=1)
    options.update(overrides)
    with pytest.raises(ValueError):
        _build_cfg(**options)


def test_gsde_keeps_gaussian_std_head_controls_inactive_without_rewriting_them():
    cfg = _build_cfg(
        actor_distribution="gsde", log_std_mapping="tdmpc2_tanh",
        log_std_min=-10, log_std_max=2, inner_log_std_mapping="direct_clamp",
        inner_log_std_min=-8, inner_log_std_max=1,
    )
    assert cfg.log_std_mapping == "tdmpc2_tanh"
    assert cfg.inner_log_std_mapping == "direct_clamp"
    assert cfg.inner_log_std_min == -8


@pytest.mark.parametrize("distribution", ["gaussian", "gsde"])
def test_pre_tanh_automatic_temperature_accepts_joint_numeric_inheritance(distribution):
    cfg = _build_cfg(
        actor_distribution=distribution,
        outer_actor_entropy_mode="PRE_TANH", inner_actor_entropy_mode="PRE_TANH",
        ent_coef="auto_0.1", target_entropy=1.25,
        inner_temperature_mode="auto", inner_target_entropy="inherit_outer",
    )
    assert cfg.outer_actor_entropy_mode == "pre_tanh"
    assert cfg.inner_actor_entropy_mode == "pre_tanh"
    assert cfg.target_entropy == 1.25
    assert cfg.inner_target_entropy == "inherit_outer"


def test_pre_tanh_requires_explicit_numeric_outer_target_for_auto_temperature():
    with pytest.raises(ValueError, match="numeric target_entropy"):
        _build_cfg(outer_actor_entropy_mode="pre_tanh", target_entropy="auto")


@pytest.mark.parametrize("target", ["auto", "inherit_outer"])
def test_pre_tanh_inner_cannot_inherit_implicit_or_other_mode_target(target):
    with pytest.raises(ValueError, match="numeric inner_target_entropy"):
        _build_cfg(
            outer_actor_entropy_mode="squashed", inner_actor_entropy_mode="pre_tanh",
            target_entropy=1.25, inner_target_entropy=target,
        )


def test_pre_tanh_fixed_temperatures_do_not_need_numeric_targets():
    cfg = _build_cfg(
        outer_actor_entropy_mode="pre_tanh", inner_actor_entropy_mode="pre_tanh",
        ent_coef=0.001, inner_temperature_mode="fixed",
        target_entropy="auto", inner_target_entropy="auto",
    )
    assert cfg.ent_coef == 0.001
    assert cfg.target_entropy == "auto"


def test_pre_tanh_does_not_enable_mixture_gaussian_entropy():
    with pytest.raises(ValueError, match="shared_mixture"):
        _build_cfg(inner_actor_entropy_mode="pre_tanh", inner_target_entropy=1.0,
                   inner_explorer_mode="shared_mixture")


def test_auxiliary_sac_critic_inherits_pre_tanh_config_without_second_actor():
    cfg = _build_cfg(
        actor_distribution="gsde", aux_return_mode="sac",
        inner_critic_source="aux_return", inner_horizon_critic_source="aux_return",
        outer_actor_entropy_mode="pre_tanh", inner_actor_entropy_mode="pre_tanh",
        ent_coef="auto_0.1", target_entropy=0.5,
    )
    assert cfg.aux_return_actor_cfg["outer_actor_entropy_mode"] == "pre_tanh"
    assert cfg.inner_actor_source == "sac"
    assert cfg.inner_finite_horizon is True


def test_gaussian_auxiliary_return_actor_can_select_analytic_entropy():
    cfg = _build_cfg(
        aux_return_mode="return_actor", inner_actor_source="return_actor",
        aux_return_outer_actor_entropy_mode="pre_tanh",
        aux_return_ent_coef="auto_0.1", aux_return_target_entropy=0.5,
    )
    assert cfg.inner_actor_entropy_mode == "pre_tanh"
    assert cfg.inner_target_entropy == "inherit_outer"


def test_gaussian_auxiliary_analytic_entropy_requires_numeric_auto_target():
    with pytest.raises(ValueError, match="aux_return_target_entropy"):
        _build_cfg(aux_return_mode="return_actor", aux_return_outer_actor_entropy_mode="pre_tanh")


@pytest.mark.parametrize("mode", ["frozen_random", "separate_critics", "adaptive_param_noise"])
def test_gaussian_pre_tanh_rejects_specialized_inner_actor_paths(mode):
    with pytest.raises(ValueError, match="inner_explorer_mode"):
        _build_cfg(
            inner_actor_entropy_mode="pre_tanh", inner_target_entropy=1.0,
            inner_explorer_mode=mode, inner_param_noise_actor_count=8,
        )


@pytest.mark.parametrize("key,value", [
    ("outer_actor_objective", "distillation"),
    ("outer_actor_objective", "po_mpc"),
    ("reanalyze_backend", "batched"),
    ("reanalyze_backend", "cuda_graph"),
    ("reanalyze_backend", "batched_cuda_graph"),
])
@pytest.mark.parametrize("learner", ["outer", "inner"])
def test_gaussian_pre_tanh_rejects_specialized_objective_or_teacher_backend(key, value, learner):
    with pytest.raises(ValueError, match=key):
        _build_cfg(**{
            f"{learner}_actor_entropy_mode": "pre_tanh", key: value,
            "ent_coef": 0.01, "target_entropy": 1.0, "inner_target_entropy": 1.0,
        })


def test_auxiliary_pre_tanh_cannot_bypass_specialized_backend_validation():
    with pytest.raises(ValueError, match="reanalyze_backend"):
        _build_cfg(
            aux_return_mode="return_actor", aux_return_outer_actor_entropy_mode="pre_tanh",
            aux_return_ent_coef=0.01, reanalyze_backend="batched",
        )


@pytest.mark.parametrize("key,value", [
    ("actor_distribution", "GSDE"),
    ("outer_actor_entropy_mode", "PRE_TANH"),
    ("inner_actor_entropy_mode", "PRE_TANH"),
])
def test_native_tdambi_rejects_ambi_policy_changes_before_sac_config_translation(key, value):
    from tests.test_tdambi_initialization import _native_cfg

    with pytest.raises(ValueError, match=f"Native TDAMBI.*{key}"):
        _native_cfg(**{key: value})
