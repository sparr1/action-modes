"""Auxiliary return critics have explicit configuration and evaluation identity."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import main as training_main
from test_ambixqc_wrapper import _build_cfg
from utils.ambi_benchmark import benchmark_run_labels
from utils.ambi_research import PresetMatrixError, resolve_preset
from utils.eval_series_data import planner_identity


def test_aux_defaults_preserve_main_xqc_learning_and_target():
    cfg = _build_cfg()
    assert cfg.aux_return_mode == "off"
    assert cfg.aux_return_detach_representation is True
    assert cfg.aux_return_critic_coef == pytest.approx(0.1)
    assert cfg.inner_critic_source == cfg.inner_horizon_critic_source == "xqc"
    assert cfg.inner_critic_target == "entropy_augmented"


@pytest.mark.parametrize("initial", ["xqc", "aux_return"])
@pytest.mark.parametrize("horizon", ["xqc", "aux_return"])
def test_inner_sources_are_independent_and_target_defaults_to_initial_source(initial, horizon):
    cfg = _build_cfg(
        aux_return_mode="xqc", inner_critic_source=initial,
        inner_horizon_critic_source=horizon, aux_return_detach_representation=False,
    )
    assert cfg.inner_critic_source == initial
    assert cfg.inner_horizon_critic_source == horizon
    assert cfg.inner_critic_target == (
        "entropy_augmented" if initial == "xqc" else "reward_only"
    )
    assert cfg.aux_return_detach_representation is False
    explicit = _build_cfg(
        aux_return_mode="xqc", inner_critic_source=initial,
        inner_horizon_critic_source=horizon, inner_critic_target="reward_only",
    )
    assert explicit.inner_critic_target == "reward_only"


@pytest.mark.parametrize("key", ["inner_critic_source", "inner_horizon_critic_source"])
@pytest.mark.parametrize("operator", ["xqc", "none"])
def test_aux_sources_require_trained_auxiliary_and_frozen_scale_even_when_dormant(key, operator):
    with pytest.raises(ValueError, match="requires aux_return_mode"):
        _build_cfg(**{key: "aux_return", "inner_operator": operator})
    with pytest.raises(ValueError, match="frozen_real_scale"):
        _build_cfg(**{key: "aux_return", "inner_operator": operator,
                     "aux_return_mode": "xqc", "inner_reward_normalization": "action_local_imagined"})
    # Training an unused auxiliary critic does not constrain the main-only inner solve.
    assert _build_cfg(
        aux_return_mode="xqc", inner_reward_normalization="action_local_imagined"
    ).inner_critic_source == "xqc"


@pytest.mark.parametrize("key,value", [
    ("aux_return_mode", "sac"), ("aux_return_mode", True), ("aux_return_mode", None),
    ("aux_return_detach_representation", 1), ("aux_return_detach_representation", "false"),
    ("aux_return_critic_coef", 0), ("aux_return_critic_coef", -1),
    ("aux_return_critic_coef", float("nan")), ("aux_return_critic_coef", float("inf")),
    ("aux_return_critic_coef", True), ("inner_critic_source", "outer"),
    ("inner_horizon_critic_source", "inner"), ("inner_critic_target", "soft"),
    ("aux_return_actor_lr", 1e-4), ("aux_return_critic_lr", 1e-4),
    ("aux_return_unknown", False), ("critic_value_mode", "scalar"),
    ("inner_actor_source", "aux_return"), ("outer_actor_source", "aux_return"),
])
def test_aux_configuration_rejects_ambiguous_or_unsupported_options(key, value):
    with pytest.raises(ValueError, match=key):
        _build_cfg(**{key: value})


def test_runtime_metadata_records_auxiliary_training_and_independent_inner_choices():
    cfg = _build_cfg(aux_return_mode="xqc", inner_horizon_critic_source="aux_return")
    metadata = training_main._resolved_runtime_metadata(
        SimpleNamespace(cfg=cfg), trial_run_params={"alg": "AMBIXQC/AMBIXQC", "seed": 3}
    )
    assert metadata["aux_return"] == {
        "aux_return_mode": "xqc", "aux_return_detach_representation": True,
        "aux_return_critic_coef": 0.1,
    }
    assert metadata["inner_budget"]["inner_critic_source"] == "xqc"
    assert metadata["inner_budget"]["inner_horizon_critic_source"] == "aux_return"
    assert metadata["inner_budget"]["inner_critic_target"] == "entropy_augmented"


def _preset(saved, overrides, *, shared=None):
    context = SimpleNamespace(
        source=Path("unused-checkpoint.metadata.json"),
        trial_run_params={"alg": "AMBIXQC/AMBIXQC", "env": "Unused-v0", "alg_params": saved},
        experiment_params={},
    )
    matrix = {
        "schema_version": 1, "base_alg_config": "checkpoint",
        "shared_alg_params": shared or {},
        "comparisons": {"test": {"reference": "candidate", "variants": {
            "candidate": {"alg_params": overrides},
        }}},
    }
    original = deepcopy(context.trial_run_params)
    result = resolve_preset("unused-matrix.json", "test/candidate", matrix,
                            checkpoint_context=context)
    assert context.trial_run_params == original
    return result["algorithm_config"]["alg_params"]


def test_evaluation_source_change_recomputes_target_without_changing_saved_configuration():
    saved = {"aux_return_mode": "xqc", "inner_critic_source": "xqc",
             "inner_critic_target": "entropy_augmented"}
    result = _preset(saved, {"inner_critic_source": "aux_return"})
    assert result["inner_critic_target"] == "reward_only"
    result = _preset(result, {"inner_critic_source": "xqc"})
    assert result["inner_critic_target"] == "entropy_augmented"
    result = _preset(saved, {"inner_critic_source": "aux_return"},
                     shared={"inner_critic_target": "entropy_augmented"})
    assert result["inner_critic_target"] == "entropy_augmented"
    unchanged = _preset({**saved, "inner_critic_target": "reward_only"},
                        {"inner_horizon_critic_source": "aux_return"})
    assert unchanged["inner_critic_target"] == "reward_only"
    reset = _preset(saved, {"inner_critic_source": "aux_return", "inner_critic_target": None})
    assert "inner_critic_target" not in reset


def test_evaluation_requires_auxiliary_in_checkpoint_and_rejects_outer_overrides():
    with pytest.raises(PresetMatrixError, match="checkpoint trained with"):
        _preset({}, {"inner_horizon_critic_source": "aux_return"})
    with pytest.raises(PresetMatrixError, match="frozen_real_scale"):
        _preset({"aux_return_mode": "xqc"}, {
            "inner_critic_source": "aux_return", "inner_reward_normalization": "action_local_imagined",
        })
    with pytest.raises(PresetMatrixError, match="override only inner controls"):
        _preset({}, {"aux_return_mode": "xqc"})


def test_new_default_fields_preserve_existing_planner_identity_and_nondefaults_differ():
    old = {"inner_operator": "xqc", "inner_rounds": 2}
    defaults = {**old, "inner_critic_source": "xqc", "inner_horizon_critic_source": "xqc",
                "inner_critic_target": "entropy_augmented"}
    identity = lambda config: planner_identity(config, {}, "ambixqc", "tanh_mean")
    assert identity(old) == identity(defaults)
    for key, value in (("inner_critic_source", "aux_return"),
                       ("inner_horizon_critic_source", "aux_return"),
                       ("inner_critic_target", "reward_only")):
        assert identity({**defaults, key: value}) != identity(defaults)


def test_evaluation_labels_distinguish_auxiliary_init_and_horizon_from_reward_target():
    labels = benchmark_run_labels({}, {}, {
        "alg": "AMBIXQC/AMBIXQC", "alg_params": {
            "inner_operator": "xqc", "inner_critic_source": "aux_return",
            "inner_horizon_critic_source": "xqc", "inner_critic_target": "reward_only",
        },
    }, "episodes")
    assert "auxiliary return init" in labels["name"]
    assert "reward-only target" in labels["name"]
    assert "critic-init:aux_return" in labels["tags"]
    assert "horizon-critic:xqc" in labels["tags"]
