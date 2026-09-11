"""Reconstruct entropy settings through real checkpoint evaluation APIs."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

import render_checkpoint as renderer
from tests.test_checkpoint_research_configs import _build_cfg
from utils.ambi_research import load_preset_matrix, resolve_preset


MATRIX = (
    Path(__file__).resolve().parents[1]
    / "configs/research/ambi_humanoid_inner_benchmark.json"
)


def _saved_context(tmp_path, **entropy_params):
    # Only sidecar configuration is read: no networks, replay, or episodes.
    checkpoint = tmp_path / "outer.pt"
    checkpoint.write_bytes(b"configuration-only checkpoint fixture")
    run = {
        "name": "ScaledEntropy",
        "alg": "AMBITDMPC2/AMBITDMPC2",
        "env": "Pendulum-v1",
        "seed": 7,
        "device": "cpu",
        "total_steps": 100,
        "alg_params": {
            "outer_actor_entropy_mode": "tdmpc2_scaled",
            "ent_coef": 0.0001,
            "target_entropy": "auto",
            "inner_actor_entropy_mode": "squashed",
            "inner_temperature_mode": "auto",
            "inner_target_entropy": "auto",
            "log_std_mapping": "tdmpc2_tanh",
            "log_std_min": -10,
            "log_std_max": 2,
            **entropy_params,
        },
    }
    metadata = {
        "schema_version": 1,
        "checkpoint": {
            "kind": "latest", "step": 100, "episode": 3,
            "best_score": None, "best_window": 100,
        },
        "trial_run_params": run,
        "experiment_params": {"env_params": {"max_episode_steps": 5}},
    }
    Path(f"{checkpoint}.metadata.json").write_text(json.dumps(metadata))
    return renderer.resolve_render_context(checkpoint)


def _resolved_run(context, matrix, selector="inner_budget/sac_1x"):
    return resolve_preset(
        MATRIX, selector, matrix, checkpoint_context=context,
    )["algorithm_config"]


@pytest.mark.parametrize("coefficient,target", [(0.0001, "auto"), ("auto_0.1", 3.0)])
def test_render_reconstruction_preserves_scaled_outer_mode_and_temperature_target(
    tmp_path, coefficient, target,
):
    context = _saved_context(tmp_path, ent_coef=coefficient, target_entropy=target)
    before = deepcopy(context)
    run, _ = renderer._prepare_run_params(
        context, backend="ambi_tdmpc2", device="cpu", controller_seed=61,
    )
    cfg = _build_cfg(run)
    assert cfg.outer_actor_entropy_mode == "tdmpc2_scaled"
    assert cfg.ent_coef == coefficient
    assert cfg.target_entropy == target
    assert cfg.inner_actor_entropy_mode == "squashed"
    assert cfg.inner_target_entropy == "auto"
    assert cfg.log_std_mapping == "tdmpc2_tanh"
    assert (cfg.log_std_min, cfg.log_std_max) == (-10, 2)
    assert run["seed"] == run["alg_params"]["seed"] == 61
    assert context == before


def test_outer_checkpoint_evaluation_permits_independent_scaled_inner_override(tmp_path):
    context = _saved_context(tmp_path)
    before = deepcopy(context)
    matrix = load_preset_matrix(MATRIX)
    variant = matrix["comparisons"]["inner_budget"]["variants"]["sac_1x"]
    variant["alg_params"].update(
        inner_actor_entropy_mode="tdmpc2_scaled", inner_target_entropy=2.5,
    )
    run = _resolved_run(context, matrix)
    cfg = _build_cfg(run)
    assert cfg.outer_actor_entropy_mode == "tdmpc2_scaled"
    assert cfg.ent_coef == 0.0001
    assert cfg.inner_actor_entropy_mode == "tdmpc2_scaled"
    assert cfg.inner_temperature_mode == "auto"
    assert cfg.inner_target_entropy == 2.5
    assert context == before
    assert context.trial_run_params["alg_params"]["inner_actor_entropy_mode"] == "squashed"


def test_scaled_outer_evaluation_rejects_inherited_squashed_target_until_explicit_auto(tmp_path):
    context = _saved_context(tmp_path)
    matrix = load_preset_matrix(MATRIX)
    original = deepcopy(matrix)
    inherited = _resolved_run(context, matrix)
    assert inherited["alg_params"]["outer_actor_entropy_mode"] == "tdmpc2_scaled"
    assert inherited["alg_params"]["inner_target_entropy"] == "inherit_outer"
    with pytest.raises(ValueError, match="inherit_outer target entropy"):
        _build_cfg(inherited)

    # Override a copy of the evaluation matrix; its saved preset is unchanged.
    variant = matrix["comparisons"]["inner_budget"]["variants"]["sac_1x"]
    variant["alg_params"]["inner_target_entropy"] = "auto"
    cfg = _build_cfg(_resolved_run(context, matrix))
    assert cfg.outer_actor_entropy_mode == "tdmpc2_scaled"
    assert cfg.inner_actor_entropy_mode == "squashed"
    assert cfg.inner_target_entropy == "auto"
    assert load_preset_matrix(MATRIX) == original


def test_prior_only_evaluation_of_scaled_inner_checkpoint_requires_explicit_squashed_override(
    tmp_path,
):
    context = _saved_context(
        tmp_path, inner_actor_entropy_mode="tdmpc2_scaled", inner_target_entropy=2.5,
    )
    matrix = load_preset_matrix(MATRIX)
    prior = _resolved_run(context, matrix, "inner_budget/prior")
    assert prior["alg_params"]["inner_operator"] == "none"
    assert prior["alg_params"]["inner_actor_entropy_mode"] == "tdmpc2_scaled"
    with pytest.raises(ValueError, match="inner_actor_entropy_mode.*inner_operator"):
        _build_cfg(prior)

    variant = matrix["comparisons"]["inner_budget"]["variants"]["prior"]
    variant["alg_params"]["inner_actor_entropy_mode"] = "squashed"
    cfg = _build_cfg(_resolved_run(context, matrix, "inner_budget/prior"))
    assert cfg.inner_operator == "none"
    assert cfg.inner_actor_entropy_mode == "squashed"
    assert cfg.outer_actor_entropy_mode == "tdmpc2_scaled"
    assert context.trial_run_params["alg_params"]["inner_actor_entropy_mode"] == "tdmpc2_scaled"
