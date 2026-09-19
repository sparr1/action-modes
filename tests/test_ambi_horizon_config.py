"""Horizon conditioning is opt-in scientific state with a bounded SAC scope."""

from copy import deepcopy
from types import SimpleNamespace

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


def _conditioned_cfg(**params):
    return _build_cfg(**{
        "inner_finite_horizon": True,
        "inner_horizon_conditioning": "one_hot",
        **params,
    })


def test_horizon_conditioning_defaults_preserve_existing_configuration():
    omitted = _build_cfg()
    explicit = _build_cfg(inner_horizon_conditioning="NONE")
    assert vars(omitted) == vars(explicit)
    assert omitted.inner_horizon_conditioning == "none"
    assert not hasattr(omitted, "horizon_conditioning_horizon")


def test_conditioning_width_is_derived_from_resolved_rollout_horizon():
    cfg = _conditioned_cfg(inner_rollout_horizon=1, horizon_conditioning_horizon=9)
    assert cfg.horizon_conditioning_horizon == 1


@pytest.mark.parametrize("value", [None, True, 1, "scalar", "embedding", ""])
def test_horizon_conditioning_rejects_invalid_modes(value):
    with pytest.raises(ValueError, match="inner_horizon_conditioning"):
        _build_cfg(inner_horizon_conditioning=value)


@pytest.mark.parametrize("horizon", [1, 3])
@pytest.mark.parametrize("representation", ["scalar", "distributional"])
@pytest.mark.parametrize("target", ["online", "outer_target"])
def test_horizon_conditioning_preserves_supported_critic_choices(horizon, representation, target):
    cfg = _conditioned_cfg(
        inner_horizon_conditioning="ONE_HOT", inner_rollout_horizon=horizon,
        q_representation=representation, inner_critic_target_initialization=target,
    )
    assert cfg.inner_horizon_conditioning == "one_hot"
    assert cfg.horizon_conditioning_horizon == horizon
    assert cfg.q_representation == representation
    assert cfg.inner_critic_target_initialization == target


@pytest.mark.parametrize("schedule", [
    {"inner_updates_per_round": 2},
    {"inner_critic_updates_per_round": 3, "inner_actor_updates_per_round": 2},
    {"inner_steps_per_update": 4, "inner_update_timing": "round"},
    {"inner_steps_per_update": 4, "inner_update_timing": "step"},
])
def test_horizon_conditioning_supports_existing_sac_update_timings(schedule):
    cfg = _conditioned_cfg(**schedule)
    assert cfg.inner_horizon_conditioning == "one_hot"
    assert cfg.inner_critic_updates_per_action > 0
    assert cfg.inner_actor_updates_per_action > 0


@pytest.mark.parametrize("options", [
    {"inner_finite_horizon": False},
    {"inner_actor_initialization": "random"},
    {"inner_critic_initialization": "random"},
    {"inner_actor_adaptation": "frozen"},
    {"inner_critic_adaptation": "frozen"},
    {"inner_critic_adaptation": "lora_rl"},
    {"inner_bootstrap_source": "outer_online"},
    {"inner_bootstrap_source": "outer_target"},
    {"inner_outer_replay_fraction": 0.25},
    {"inner_actor_writeback_coef": 0.1},
    {"inner_critic_writeback_coef": 0.1},
    {"inner_explorer_mode": "frozen_random"},
    {"critic_value_mode": "return_entropy"},
    {"aux_return_mode": "return_actor"},
    {"value_equivalence_diagnostics": True},
    {"value_equivalence_loss_coef": 0.1},
    {"inner_diagnostic_rollouts": 1},
])
def test_horizon_conditioning_rejects_unsupported_core_extensions(options):
    with pytest.raises(ValueError, match="inner_horizon_conditioning"):
        _conditioned_cfg(**options)


@pytest.mark.parametrize("component", SCOPES)
@pytest.mark.parametrize("scope", ["episode", "run"])
def test_horizon_conditioning_requires_fresh_state_in_every_scope(component, scope):
    options = {f"inner_{component}_scope": scope}
    if component.endswith("_optimizer"):
        options[f"inner_{component.removesuffix('_optimizer')}_scope"] = scope
    with pytest.raises(ValueError, match="inner_horizon_conditioning"):
        _conditioned_cfg(**options)


@pytest.mark.parametrize("operator", ["none", "td3", "mppi"])
def test_horizon_conditioning_requires_sac(operator):
    with pytest.raises(ValueError, match="inner_horizon_conditioning"):
        _conditioned_cfg(inner_operator=operator)


def test_horizon_conditioning_rejects_execution_handoff():
    with pytest.raises(ValueError, match="inner_execution_policy_source"):
        _conditioned_cfg(inner_execution_policy_source="outer_soft_handoff")


def test_horizon_conditioning_rejects_native_tdambi_before_sac_translation():
    algorithm = object.__new__(TDAMBI)
    with pytest.raises(ValueError, match="inner_horizon_conditioning"):
        algorithm._build_cfg({"inner_horizon_conditioning": "one_hot"})


def test_horizon_conditioning_retains_entropy_scale_and_kl_controls():
    cfg = _conditioned_cfg(
        ent_coef=0.1, inner_temperature_mode="inherit_outer",
        inner_actor_entropy_mode="tdmpc2_scaled",
        inner_sac_critic_target="reward_only",
        sac_actor_loss_scale_mode="tdmpc2_percentile_range",
        inner_actor_loss_scale_update="per_update", inner_outer_policy_kl_coef=0.2,
    )
    assert cfg.inner_actor_entropy_mode == "tdmpc2_scaled"
    assert cfg.inner_actor_loss_scale_update == "per_update"
    assert cfg.inner_outer_policy_kl_coef == 0.2


def _lineage(params):
    return scientific_trial_parameters({"alg": ALGORITHM, "alg_params": params})


def _planner(params):
    return planner_identity(params, {}, ALGORITHM, "tanh_mean")


@pytest.mark.parametrize("identity", [_lineage, _planner])
def test_disabled_conditioning_preserves_historical_scientific_identity(identity):
    historical = {"inner_operator": "sac", "inner_finite_horizon": True,
                  "inner_rollout_horizon": 3}
    expected = identity(historical)
    explicit = {**historical, "inner_horizon_conditioning": "NONE"}
    assert identity(explicit) == expected
    settings = expected["alg_params"] if identity is _lineage else expected["settings"]
    assert "inner_horizon_conditioning" not in settings
    assert "horizon_conditioning_horizon" not in settings


@pytest.mark.parametrize("identity", [_lineage, _planner])
def test_active_conditioning_records_normalized_mode_and_horizon(identity):
    params = {"inner_operator": "sac", "inner_finite_horizon": True,
              "inner_horizon_conditioning": "one_hot", "inner_rollout_horizon": 3}
    original = deepcopy(params)
    expected = identity(params)
    assert identity({**params, "inner_horizon_conditioning": "ONE_HOT"}) == expected
    settings = expected["alg_params"] if identity is _lineage else expected["settings"]
    assert settings["inner_horizon_conditioning"] == "one_hot"
    assert settings["horizon_conditioning_horizon"] == 3
    assert identity({**params, "inner_rollout_horizon": 1}) != expected
    assert identity({**params, "inner_horizon_conditioning": "none"}) != expected
    assert params == original


@pytest.mark.parametrize("mode", ["none", "one_hot"])
def test_runtime_metadata_records_only_active_conditioning(mode):
    from main import _resolved_runtime_metadata

    model = SimpleNamespace(cfg=_conditioned_cfg(inner_horizon_conditioning=mode))
    metadata = _resolved_runtime_metadata(model, trial_run_params={"alg": ALGORITHM, "seed": 3})
    inner = metadata["inner_budget"]
    if mode == "none":
        assert "inner_horizon_conditioning" not in inner
        assert "horizon_conditioning_horizon" not in inner
    else:
        assert inner["inner_horizon_conditioning"] == "one_hot"
        assert inner["horizon_conditioning_horizon"] == 3


@pytest.fixture
def models():
    opened = []

    def create(**options):
        model = _tiny_model(inner_finite_horizon=True, **options)
        opened.append(model)
        return model.agent

    yield create
    for model in opened:
        model.env.close()


def test_target_spec_records_active_conditioning_only(models):
    historical = models()._critic_target_spec()
    explicit = models(inner_horizon_conditioning="none")._critic_target_spec()
    assert explicit == historical
    assert "horizon_conditioning" not in historical["inner_solve"]
    conditioned = models(inner_horizon_conditioning="one_hot")._critic_target_spec()
    assert conditioned["inner_solve"]["horizon_conditioning"] == "one_hot"
    assert conditioned["inner_solve"]["horizon_conditioning_horizon"] == 2


@pytest.mark.parametrize("source_active", [False, True])
def test_exact_outer_preflight_rejects_changed_conditioning_before_mutation(models, source_active):
    source = models(inner_horizon_conditioning="one_hot" if source_active else "none")
    target = models(inner_horizon_conditioning="none" if source_active else "one_hot")
    before = deepcopy(target.checkpoint_state())
    with pytest.raises(ValueError, match="critic-target specification"):
        target._preflight_outer_training_state(source.checkpoint_state())
    _assert_tree_equal(target.checkpoint_state(), before)


def test_exact_outer_preflight_rejects_changed_conditioning_horizon(models):
    source = models(inner_horizon_conditioning="one_hot", inner_rollout_horizon=1)
    target = models(inner_horizon_conditioning="one_hot", inner_rollout_horizon=2)
    before = deepcopy(target.checkpoint_state())
    with pytest.raises(ValueError, match="critic-target specification"):
        target._preflight_outer_training_state(source.checkpoint_state())
    _assert_tree_equal(target.checkpoint_state(), before)


@pytest.mark.parametrize("source_active", [False, True])
def test_portable_outer_weights_allow_new_conditioning_choice(models, source_active):
    source = models(inner_horizon_conditioning="one_hot" if source_active else "none")
    target = models(inner_horizon_conditioning="none" if source_active else "one_hot")
    with torch.no_grad():
        for parameter in source.model.parameters():
            parameter.add_(0.01)
    target.load(deepcopy(source.checkpoint_state()))
    _assert_tree_equal(target.model.state_dict(), source.model.state_dict())
    assert target.cfg.inner_horizon_conditioning == ("none" if source_active else "one_hot")
