"""Scientific identity and portable interpretation of actor entropy choices."""

from itertools import product
from types import SimpleNamespace

import pytest

import main as training_main
from RL.tdmpc2_core.common.entropy import critic_entropy_spec
from utils.resume_identity import canonical_json, scientific_trial_parameters


def _trial(**params):
    return {
        "alg": "AMBITDMPC2/AMBITDMPC2",
        "seed": 7,
        "alg_params": {"batch_size": 32, **params},
    }


def _runtime(*, agent=None, **params):
    model = SimpleNamespace(
        cfg=SimpleNamespace(**{"action_dim": 3, "inner_operator": "sac", **params}),
        agent=agent,
        env=None,
    )
    return training_main._resolved_runtime_metadata(model, trial_run_params=_trial())


@pytest.mark.parametrize("spelling", ["squashed", "SQUASHED"])
def test_entropy_identity_preserves_omitted_and_explicit_historical_defaults(spelling):
    historical = scientific_trial_parameters(_trial())
    explicit = scientific_trial_parameters(
        _trial(outer_actor_entropy_mode=spelling, inner_actor_entropy_mode=spelling)
    )
    assert canonical_json(explicit) == canonical_json(historical)
    assert "outer_actor_entropy_mode" not in historical["alg_params"]
    assert "inner_actor_entropy_mode" not in historical["alg_params"]


def test_entropy_modes_are_independent_scientific_identity_fields():
    modes = ("squashed", "tdmpc2_scaled")
    identities = {
        canonical_json(scientific_trial_parameters(_trial(
            outer_actor_entropy_mode=outer,
            inner_actor_entropy_mode=inner,
        )))
        for outer, inner in product(modes, repeat=2)
    }
    assert len(identities) == 4
    lower = scientific_trial_parameters(_trial(outer_actor_entropy_mode="tdmpc2_scaled"))
    upper = scientific_trial_parameters(_trial(outer_actor_entropy_mode="TDMPC2_SCALED"))
    assert lower == upper


@pytest.mark.parametrize("field", ["target_entropy", "inner_target_entropy"])
def test_scaled_entropy_numeric_targets_remain_scientifically_distinct(field):
    common = {
        "outer_actor_entropy_mode": "tdmpc2_scaled",
        "inner_actor_entropy_mode": "tdmpc2_scaled",
        "target_entropy": 2.0,
        "inner_target_entropy": 3.0,
    }
    baseline = scientific_trial_parameters(_trial(**common))
    changed = scientific_trial_parameters(_trial(**{**common, field: 4.0}))
    assert baseline != changed


@pytest.mark.parametrize("outer,inner", product(("squashed", "tdmpc2_scaled"), repeat=2))
def test_runtime_entropy_modes_and_explicit_target_semantics(outer, inner):
    metadata = _runtime(
        outer_actor_entropy_mode=outer,
        inner_actor_entropy_mode=inner,
        ent_coef="auto_0.1",
        target_entropy=2.0,
        inner_temperature_mode="auto",
        inner_target_entropy=3.0,
    )["actor_entropy"]
    for learner, mode, target in (("outer", outer, 2.0), ("inner", inner, 3.0)):
        assert metadata[learner] == {
            "actor_entropy_mode": mode,
            "target_entropy_semantics": (
                "tdmpc2_scaled_entropy" if mode == "tdmpc2_scaled"
                else "squashed_action_entropy"
            ),
            "temperature_mode": "auto",
            "target_entropy_setting": target,
            "target_entropy": target,
            "target_active": True,
        }


def test_runtime_missing_historical_entropy_fields_mean_squashed():
    metadata = _runtime()["actor_entropy"]
    assert metadata["outer"] == {
        "actor_entropy_mode": "squashed",
        "target_entropy_semantics": "squashed_action_entropy",
        "temperature_mode": "auto",
        "target_entropy_setting": "auto",
        "target_entropy": -3.0,
        "target_active": True,
    }
    assert metadata["inner"] == {
        **metadata["outer"],
        "target_entropy_setting": "inherit_outer",
    }


def test_runtime_target_inheritance_records_resolved_outer_target():
    metadata = _runtime(agent=SimpleNamespace(target_entropy=-1.5))["actor_entropy"]
    assert metadata["outer"]["target_entropy"] == -1.5
    assert metadata["inner"]["target_entropy"] == -1.5
    assert metadata["inner"]["target_entropy_setting"] == "inherit_outer"


def test_runtime_tdmpc2_outer_recipe_preserves_independent_inner_auto_target():
    metadata = _runtime(
        outer_actor_entropy_mode="tdmpc2_scaled",
        ent_coef=0.0001,
        inner_target_entropy="auto",
    )["actor_entropy"]
    assert metadata["outer"]["actor_entropy_mode"] == "tdmpc2_scaled"
    assert metadata["outer"]["temperature_mode"] == "fixed"
    assert metadata["outer"]["target_active"] is False
    assert metadata["inner"]["actor_entropy_mode"] == "squashed"
    assert metadata["inner"]["target_entropy_semantics"] == "squashed_action_entropy"
    assert metadata["inner"]["target_entropy_setting"] == "auto"
    assert metadata["inner"]["target_entropy"] == -3.0
    assert metadata["inner"]["target_active"] is True


@pytest.mark.parametrize("algorithm", ["TDMPC2/TDMPC2Baseline", "TDAMBI/TDAMBI"])
def test_runtime_does_not_assign_ambi_entropy_switches_to_other_algorithms(algorithm):
    metadata = training_main._resolved_runtime_metadata(
        SimpleNamespace(cfg=SimpleNamespace(action_dim=3), agent=None, env=None),
        trial_run_params={"alg": algorithm, "seed": 7},
    )
    assert "actor_entropy" not in metadata
    assert "critic_entropy" not in metadata


@pytest.mark.parametrize("outer,inner", product(("squashed", "tdmpc2_scaled"), repeat=2))
@pytest.mark.parametrize("target", ["reward_only", "entropy_augmented"])
def test_runtime_records_critic_entropy_separately_from_actor_entropy(outer, inner, target):
    metadata = _runtime(
        outer_actor_entropy_mode=outer, inner_actor_entropy_mode=inner,
        outer_critic_target=target, inner_sac_critic_target=target,
    )
    for learner, mode in (("outer", outer), ("inner", inner)):
        expected = "none" if target == "reward_only" else (
            "tdmpc2_scaled_entropy" if mode == "tdmpc2_scaled" else "squashed_action_entropy"
        )
        assert metadata["critic_entropy"][learner] == expected


def test_runtime_distinguishes_shared_mixture_critic_entropy():
    metadata = _runtime(inner_explorer_mode="shared_mixture")
    assert metadata["critic_entropy"] == {
        "outer": "squashed_action_entropy", "inner": "squashed_mixture_entropy",
    }


@pytest.mark.parametrize("scale", ["none", "tdmpc2_percentile_range"])
@pytest.mark.parametrize("outer_target,inner_target", product(
    ("reward_only", "entropy_augmented"), repeat=2,
))
@pytest.mark.parametrize("mode", ["squashed", "tdmpc2_scaled"])
def test_runtime_records_scale_coefficient_units_only_for_active_bonuses(
    scale, outer_target, inner_target, mode,
):
    metadata = _runtime(
        sac_actor_loss_scale_mode=scale, sac_actor_loss_scale_tau=0.01,
        outer_critic_target=outer_target, inner_sac_critic_target=inner_target,
        outer_actor_entropy_mode=mode, inner_actor_entropy_mode=mode,
    )["critic_entropy"]
    entropy = (
        "tdmpc2_scaled_entropy" if mode == "tdmpc2_scaled"
        else "squashed_action_entropy"
    )
    expected = {
        "outer": entropy if outer_target == "entropy_augmented" else "none",
        "inner": entropy if inner_target == "entropy_augmented" else "none",
    }
    units = {}
    if scale == "tdmpc2_percentile_range":
        if outer_target == "entropy_augmented":
            units["outer"] = "alpha_times_outer_q_scale"
        if inner_target == "entropy_augmented":
            units["inner"] = "alpha_times_action_local_q_scale"
    if units:
        expected["coefficient_units"] = units
    assert metadata == expected


@pytest.mark.parametrize("operator", ["none", "td3", "mppi", "xqc", "tdambi"])
def test_runtime_omits_inner_scale_units_for_non_sac_operators(operator):
    metadata = _runtime(
        inner_operator=operator, sac_actor_loss_scale_mode="tdmpc2_percentile_range",
        sac_actor_loss_scale_tau=0.01,
    )["critic_entropy"]
    assert metadata == {
        "outer": "squashed_action_entropy",
        "inner": "squashed_action_entropy",
        "coefficient_units": {"outer": "alpha_times_outer_q_scale"},
    }


def test_scaled_critic_metadata_defaults_to_inner_sac_without_rewriting_old_defaults():
    assert critic_entropy_spec({}) == {
        "outer": "squashed_action_entropy", "inner": "squashed_action_entropy",
    }
    assert critic_entropy_spec({
        "sac_actor_loss_scale_mode": "tdmpc2_percentile_range",
    }) == {
        "outer": "squashed_action_entropy", "inner": "squashed_action_entropy",
        "coefficient_units": {
            "outer": "alpha_times_outer_q_scale",
            "inner": "alpha_times_action_local_q_scale",
        },
    }


@pytest.mark.parametrize("outer_target,inner_target", product(
    ("reward_only", "entropy_augmented"), repeat=2,
))
@pytest.mark.parametrize("explorer", ["none", "shared_mixture"])
def test_historical_critic_metadata_never_assumes_scale_coefficient_units(
    outer_target, inner_target, explorer,
):
    config = {
        "sac_actor_loss_scale_mode": "tdmpc2_percentile_range",
        "outer_actor_entropy_mode": "tdmpc2_scaled",
        "inner_actor_entropy_mode": "tdmpc2_scaled",
        "outer_critic_target": outer_target,
        "inner_sac_critic_target": inner_target,
        "inner_explorer_mode": explorer,
    }
    inner_entropy = (
        "squashed_mixture_entropy" if explorer == "shared_mixture"
        else "squashed_action_entropy"
    )
    assert critic_entropy_spec(config, historical=True) == {
        "outer": "none" if outer_target == "reward_only" else "squashed_action_entropy",
        "inner": "none" if inner_target == "reward_only" else inner_entropy,
    }


def test_runtime_q_scaled_shared_mixture_retains_its_entropy_statistic():
    metadata = _runtime(
        inner_explorer_mode="shared_mixture",
        sac_actor_loss_scale_mode="tdmpc2_percentile_range",
        sac_actor_loss_scale_tau=0.01,
    )["critic_entropy"]
    assert metadata == {
        "outer": "squashed_action_entropy", "inner": "squashed_mixture_entropy",
        "coefficient_units": {
            "outer": "alpha_times_outer_q_scale",
            "inner": "alpha_times_action_local_q_scale",
        },
    }
