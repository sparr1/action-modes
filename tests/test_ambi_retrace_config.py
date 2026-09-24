"""Retrace is an opt-in inner return estimator with stable historical identity."""

from copy import deepcopy

import pytest
import torch

from RL.TDAMBI import TDAMBI
from tests.test_ambi_config_decoupling import _build_cfg
from tests.test_ambi_latency_contract import _assert_tree_equal
from tests.test_ambi_root_local_sac import _tiny_model
from utils.eval_series_data import planner_identity
from utils.resume_identity import scientific_trial_parameters


ALGORITHM = "AMBITDMPC2/AMBITDMPC2"
SCOPES = (
    "actor", "critic", "temperature", "replay",
    "actor_optimizer", "critic_optimizer", "temperature_optimizer",
)
VALUE_SAMPLE_KEYS = ("inner_retrace_value_samples", "inner_retrace_boundary_value_samples")


def _cfg(**options):
    return _build_cfg(**{
        "inner_sac_return_estimator": "retrace",
        "inner_finite_horizon": True,
        **options,
    })


def test_retrace_defaults_leave_existing_one_step_configuration_unchanged():
    default = _build_cfg()
    explicit = _build_cfg(inner_sac_return_estimator="ONE_STEP")
    assert vars(default) == vars(explicit)
    assert default.inner_sac_return_estimator == "one_step"
    assert default.inner_retrace_lambda == 1.0
    assert default.inner_retrace_batch_trajectories is None
    assert default.inner_retrace_value_samples == 1
    assert default.inner_retrace_boundary_value_samples == 1


@pytest.mark.parametrize("interior,boundary", [(1, 1), (4, 1), (1, 16), (4, 16)])
def test_retrace_value_sample_counts_resolve_independently(interior, boundary):
    cfg = _cfg(inner_retrace_value_samples=interior,
               inner_retrace_boundary_value_samples=boundary)
    assert cfg.inner_retrace_value_samples == interior
    assert cfg.inner_retrace_boundary_value_samples == boundary


@pytest.mark.parametrize("key", VALUE_SAMPLE_KEYS)
@pytest.mark.parametrize("value", [None, True, False, 0, -1, 2.0, "2"])
def test_value_sample_count_requires_a_positive_integer(key, value):
    with pytest.raises(ValueError, match=key):
        _cfg(**{key: value})


@pytest.mark.parametrize("key", VALUE_SAMPLE_KEYS)
@pytest.mark.parametrize("operator", ["sac", "none", "mppi"])
def test_multiple_value_samples_reject_inactive_retrace(key, operator):
    with pytest.raises(ValueError, match=key):
        _build_cfg(inner_operator=operator, **{key: 4})


def test_retrace_value_sample_trace_catalog_describes_counts():
    from RL.tdmpc2_core.inner_trace import metric_catalog

    catalog = metric_catalog()
    for name in ("retrace_value_samples", "retrace_boundary_value_samples"):
        assert catalog[name]["unit"] == "count"
        assert catalog[name]["preferred_axis"] == "critic_updates"
        assert "action samples" in catalog[name]["definition"].lower()


@pytest.mark.parametrize("horizon,batch,expected", [(1, 128, 128), (3, 128, 43), (3, 6, 2)])
def test_retrace_default_batch_is_ceil_of_transition_batch_over_horizon(horizon, batch, expected):
    cfg = _cfg(inner_rollout_horizon=horizon, inner_batch_size=batch)
    assert cfg.inner_batch_size == batch
    assert cfg.inner_retrace_batch_trajectories == expected


def test_retrace_explicit_trajectory_batch_and_lambda_are_resolved():
    cfg = _cfg(inner_sac_return_estimator="RETRACE", inner_retrace_lambda=0,
               inner_retrace_batch_trajectories=7)
    assert cfg.inner_sac_return_estimator == "retrace"
    assert cfg.inner_retrace_lambda == 0.0
    assert cfg.inner_retrace_batch_trajectories == 7


def test_zero_work_default_capacity_still_holds_one_retrace_trajectory_slot():
    cfg = _cfg(inner_rounds=1, inner_rollouts_per_round=0, inner_updates_per_round=0,
               inner_replay_capacity=None, inner_rollout_horizon=3)
    assert cfg.inner_model_step_budget == 0
    assert cfg.inner_replay_capacity == 3
    old = _build_cfg(inner_rounds=1, inner_rollouts_per_round=0,
                     inner_updates_per_round=0, inner_replay_capacity=None)
    assert old.inner_replay_capacity == 1


@pytest.mark.parametrize("capacity", [1, 2])
def test_explicit_capacity_smaller_than_a_trajectory_is_rejected_even_without_updates(capacity):
    with pytest.raises(ValueError, match="one trajectory slot"):
        _cfg(inner_rounds=1, inner_rollouts_per_round=0, inner_updates_per_round=0,
             inner_replay_capacity=capacity, inner_rollout_horizon=3)


@pytest.mark.parametrize("value", [None, True, 1, "n_step", ""])
def test_invalid_return_estimator_is_rejected(value):
    with pytest.raises(ValueError, match="inner_sac_return_estimator"):
        _cfg(inner_sac_return_estimator=value)


@pytest.mark.parametrize("value", [True, False, None, "1", -0.1, 1.1, float("nan"), float("inf")])
def test_lambda_requires_finite_non_boolean_probability(value):
    with pytest.raises(ValueError, match="inner_retrace_lambda"):
        _cfg(inner_retrace_lambda=value)


@pytest.mark.parametrize("value", [True, False, 0, -1, 2.0, "2"])
def test_trajectory_batch_requires_positive_integer(value):
    with pytest.raises(ValueError, match="inner_retrace_batch_trajectories"):
        _cfg(inner_retrace_batch_trajectories=value)


@pytest.mark.parametrize("options", [
    {"inner_finite_horizon": False},
    {"inner_bootstrap_source": "outer_online"},
    {"inner_bootstrap_source": "outer_target"},
    {"inner_outer_replay_fraction": 0.25},
    {"inner_behavior_action": "mean"},
    {"inner_behavior_action": "mean_plus_gaussian"},
    {"inner_behavior_std_scale": 0.0},
    {"inner_explorer_mode": "frozen_random"},
    {"inner_actor_writeback_coef": 0.1},
    {"inner_critic_writeback_coef": 0.1},
    {"critic_value_mode": "return_entropy"},
    {"inner_execution_policy_source": "outer_soft_handoff"},
    {"inner_update_timing": "step", "inner_steps_per_update": 2},
    {"inner_model_step_budget": 18},
])
def test_retrace_rejects_unsupported_solver_combinations(options):
    with pytest.raises(ValueError, match="retrace|inner_execution_policy_source"):
        _cfg(**options)


@pytest.mark.parametrize("component", SCOPES)
@pytest.mark.parametrize("scope", ["episode", "run"])
def test_retrace_requires_every_active_component_to_be_action_local(component, scope):
    options = {f"inner_{component}_scope": scope}
    if component.endswith("_optimizer"):
        options[f"inner_{component.removesuffix('_optimizer')}_scope"] = scope
    with pytest.raises(ValueError, match="retrace"):
        _cfg(**options)


@pytest.mark.parametrize("operator", ["none", "td3", "mppi"])
def test_retrace_requires_inner_sac(operator):
    with pytest.raises(ValueError, match="retrace|only supported for inner_operator='sac'"):
        _cfg(inner_operator=operator)


def test_native_tdambi_rejects_retrace_before_its_sac_translation():
    native = object.__new__(TDAMBI)
    with pytest.raises(ValueError, match="inner_sac_return_estimator"):
        native._build_cfg({"inner_sac_return_estimator": "retrace"})


@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize("target", ["reward_only", "entropy_augmented"])
def test_existing_scalar_value_and_entropy_semantics_remain_configurable(representation, target):
    cfg = _cfg(q_representation=representation, inner_sac_critic_target=target,
               inner_behavior_std_scale=0.5)
    assert cfg.q_representation == representation
    assert cfg.inner_sac_critic_target == target
    assert cfg.inner_behavior_std_scale == 0.5


@pytest.mark.parametrize("initialization", ["prior", "random"])
@pytest.mark.parametrize("adaptation", ["clone", "lora_rl"])
def test_existing_initialization_and_critic_adaptation_choices_remain_supported(initialization, adaptation):
    cfg = _cfg(inner_actor_initialization=initialization,
               inner_critic_initialization=initialization,
               inner_critic_adaptation=adaptation,
               **({"inner_critic_lora_rank": 4} if adaptation == "lora_rl" else {}))
    assert cfg.inner_critic_initialization == initialization
    assert cfg.inner_critic_adaptation == adaptation


def test_retrace_rejects_unported_horizon_conditioning():
    with pytest.raises(ValueError, match="inner_horizon_conditioning"):
        _cfg(inner_horizon_conditioning="one_hot")


def test_retrace_preserves_auxiliary_source_routing():
    cfg = _cfg(aux_return_mode="return_actor", inner_actor_source="return_actor",
               inner_critic_source="aux_return", inner_horizon_actor_source="sac",
               inner_horizon_critic_source="aux_return")
    assert cfg.inner_actor_source == "return_actor"
    assert cfg.inner_critic_source == cfg.inner_horizon_critic_source == "aux_return"
    assert cfg.inner_horizon_actor_source == "sac"


def test_retrace_preserves_supported_entropy_surrogate_and_q_scaling():
    cfg = _cfg(ent_coef=0.2, inner_temperature_mode="inherit_outer",
               inner_actor_entropy_mode="tdmpc2_scaled",
               sac_actor_loss_scale_mode="tdmpc2_percentile_range")
    assert cfg.inner_actor_entropy_mode == "tdmpc2_scaled"
    assert cfg.sac_actor_loss_scale_mode == "tdmpc2_percentile_range"


def test_without_replacement_critic_only_uses_trajectory_batch_not_unused_actor_batch():
    cfg = _cfg(inner_rounds=1, inner_rollouts_per_round=2, inner_rollout_horizon=3,
               inner_batch_size=128, inner_retrace_batch_trajectories=2,
               inner_replay_capacity=6, inner_replay_sampling="without_replacement",
               inner_critic_updates_per_round=1, inner_actor_updates_per_round=0,
               inner_temperature_mode="inherit_outer")
    assert cfg.inner_retrace_batch_trajectories == 2


@pytest.mark.parametrize("capacity,match", [(6, "floor"), (9, "first update")])
def test_without_replacement_checks_whole_trajectory_capacity_and_first_collection(capacity, match):
    with pytest.raises(ValueError, match=match):
        _cfg(inner_rounds=1, inner_rollouts_per_round=2, inner_rollout_horizon=3,
             inner_batch_size=3, inner_retrace_batch_trajectories=3,
             inner_replay_capacity=capacity, inner_replay_sampling="without_replacement")


def test_without_replacement_still_checks_actor_transition_batch():
    with pytest.raises(ValueError, match="without-replacement inner replay"):
        _cfg(inner_rounds=1, inner_rollouts_per_round=2, inner_rollout_horizon=3,
             inner_batch_size=7, inner_retrace_batch_trajectories=1,
             inner_replay_capacity=6, inner_replay_sampling="without_replacement")


def _lineage(params):
    return scientific_trial_parameters({"alg": ALGORITHM, "alg_params": params})


def _planner(params):
    return planner_identity(params, {}, ALGORITHM, "tanh_mean")


@pytest.mark.parametrize("identity", [_lineage, _planner])
def test_inactive_retrace_preserves_historical_identity(identity):
    prior = {"inner_operator": "sac", "inner_finite_horizon": True,
             "inner_rollout_horizon": 3, "inner_batch_size": 128}
    explicit = {**prior, "inner_sac_return_estimator": "ONE_STEP",
                "inner_retrace_lambda": 0.7, "inner_retrace_batch_trajectories": 12}
    assert identity(explicit) == identity(prior)


@pytest.mark.parametrize("identity", [_lineage, _planner])
def test_active_retrace_identity_resolves_defaults_and_tracks_scientific_changes(identity):
    params = {"inner_operator": "sac", "inner_finite_horizon": True,
              "inner_sac_return_estimator": "retrace", "inner_rollout_horizon": 3,
              "inner_batch_size": 128}
    expected = identity(params)
    assert identity({**params, "inner_sac_return_estimator": "RETRACE",
                     "inner_retrace_lambda": 1, "inner_retrace_batch_trajectories": 43}) == expected
    assert identity({**params, "inner_retrace_lambda": 0.5}) != expected
    assert identity({**params, "inner_retrace_batch_trajectories": 44}) != expected
    assert identity({**params, "inner_sac_return_estimator": "one_step"}) != expected


@pytest.mark.parametrize("identity", [_lineage, _planner])
@pytest.mark.parametrize("estimator", ["one_step", "retrace"])
def test_single_value_samples_preserve_historical_identity(identity, estimator):
    params = {"inner_operator": "sac", "inner_sac_return_estimator": estimator,
              "inner_finite_horizon": True, "inner_rollout_horizon": 3}
    assert identity({**params, **dict.fromkeys(VALUE_SAMPLE_KEYS, 1)}) == identity(params)


@pytest.mark.parametrize("identity", [_lineage, _planner])
def test_each_active_value_sample_count_has_a_distinct_identity(identity):
    params = {"inner_operator": "sac", "inner_sac_return_estimator": "retrace",
              "inner_finite_horizon": True, "inner_rollout_horizon": 3}
    variants = [identity({**params, **dict(zip(VALUE_SAMPLE_KEYS, counts))})
                for counts in ((1, 1), (4, 1), (1, 4), (4, 16))]
    for index, variant in enumerate(variants):
        assert all(variant != previous for previous in variants[:index])


@pytest.fixture
def models():
    opened = []

    def create(**options):
        holder = _tiny_model(inner_finite_horizon=True, **options)
        opened.append(holder)
        return holder.agent

    yield create
    for holder in opened:
        holder.env.close()


def test_checkpoint_target_spec_records_only_active_retrace(models):
    old = models()._critic_target_spec()
    assert models(inner_sac_return_estimator="one_step")._critic_target_spec() == old
    assert "return_estimator" not in old["inner_solve"]
    active = models(inner_sac_return_estimator="retrace")._critic_target_spec()["inner_solve"]
    assert active["return_estimator"] == "retrace"
    assert active["retrace_lambda"] == 1.0
    assert active["retrace_batch_trajectories"] == 2
    assert active["retrace_protocol_version"] == 1


def test_checkpoint_target_spec_records_only_nondefault_value_sample_counts(models):
    default = models(inner_sac_return_estimator="retrace")._critic_target_spec()
    explicit = models(inner_sac_return_estimator="retrace",
                      **dict.fromkeys(VALUE_SAMPLE_KEYS, 1))._critic_target_spec()
    assert explicit == default
    assert "retrace_value_samples" not in default["inner_solve"]
    assert "retrace_boundary_value_samples" not in default["inner_solve"]
    interior = models(inner_sac_return_estimator="retrace",
                      inner_retrace_value_samples=4)._critic_target_spec()["inner_solve"]
    assert interior == {**default["inner_solve"], "retrace_value_samples": 4}
    boundary = models(inner_sac_return_estimator="retrace",
                      inner_retrace_boundary_value_samples=16)._critic_target_spec()["inner_solve"]
    assert boundary == {**default["inner_solve"], "retrace_boundary_value_samples": 16}


@pytest.mark.parametrize("change", [
    {"inner_sac_return_estimator": "one_step"},
    {"inner_retrace_lambda": 0.5},
    {"inner_retrace_batch_trajectories": 3},
    {"inner_retrace_value_samples": 4},
    {"inner_retrace_boundary_value_samples": 16},
])
def test_exact_preflight_rejects_changed_estimator_semantics_before_mutation(models, change):
    source = models(inner_sac_return_estimator="retrace")
    target = models(**{"inner_sac_return_estimator": "retrace", **change})
    before = deepcopy(target.checkpoint_state())
    with pytest.raises(ValueError, match="critic-target specification"):
        target._preflight_outer_training_state(source.checkpoint_state())
    _assert_tree_equal(target.checkpoint_state(), before)


@pytest.mark.parametrize("source_active", [False, True])
def test_portable_outer_weights_allow_a_new_inner_return_estimator(models, source_active):
    source = models(inner_sac_return_estimator="retrace" if source_active else "one_step")
    target = models(inner_sac_return_estimator="one_step" if source_active else "retrace")
    with torch.no_grad():
        for parameter in source.model.parameters():
            parameter.add_(0.01)
    target.load(deepcopy(source.checkpoint_state()))
    _assert_tree_equal(target.model.state_dict(), source.model.state_dict())
    assert target.cfg.inner_sac_return_estimator == ("one_step" if source_active else "retrace")


@pytest.mark.parametrize("interior,boundary", [(4, 1), (1, 16), (4, 16)])
def test_portable_outer_weights_allow_new_retrace_value_sample_counts(models, interior, boundary):
    source = models(inner_sac_return_estimator="retrace")
    target = models(inner_sac_return_estimator="retrace",
                    inner_retrace_value_samples=interior,
                    inner_retrace_boundary_value_samples=boundary)
    with torch.no_grad():
        for parameter in source.model.parameters():
            parameter.add_(0.01)
    source_state = deepcopy(source.checkpoint_state())
    target.load(source_state)
    _assert_tree_equal(target.model.state_dict(), source.model.state_dict())
    _assert_tree_equal(source.checkpoint_state(), source_state)
    assert target.cfg.inner_sac_return_estimator == "retrace"
    assert target.cfg.inner_retrace_value_samples == interior
    assert target.cfg.inner_retrace_boundary_value_samples == boundary
