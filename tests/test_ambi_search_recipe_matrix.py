"""End-to-end coverage for every named Humanoid Walk AMBI-search recipe.

The production matrix intentionally points at the full Humanoid configuration.
These tests retain each selector's complete search target/operator specification
while shrinking only the model and root-solve budget enough to run on CPU.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import gymnasium as gym
import pytest
import torch

from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core.common.search_config import resolve_inner_search_semantics
from tests.test_ambi_search_config import _build_cfg
from utils.ambi_research import (
    list_preset_selectors,
    load_preset_matrix,
    resolve_preset,
)


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs/research/ambi_search_humanoid_walk_v2.json"
MATRIX = load_preset_matrix(MATRIX_PATH)
SELECTORS = list_preset_selectors(MATRIX)


def _resolved_algorithm_config(selector):
    return resolve_preset(MATRIX_PATH, selector, matrix=MATRIX)["algorithm_config"]


def _expected_search_axes(selector):
    comparison, variant = selector.split("/", 1)
    if comparison == "controls":
        return {
            "operator": "none" if variant == "no_inner" else "sac",
            "q_objective": "legacy_continuing",
        }
    if comparison == "q_architecture_estimator":
        layout = next(
            candidate
            for candidate in ("depth_conditioned", "stage_heads", "shared")
            if variant.startswith(f"{candidate}_")
        )
        estimator = variant[len(layout) + 1 :]
        expected = {
            "operator": "sac",
            "q_objective": "finite_horizon",
            "critic_horizon_mode": layout,
            "return_estimator": estimator,
        }
        if estimator == "n_step":
            expected["return_steps"] = 2
        elif estimator == "lambda_return":
            expected["return_lambda"] = 0.5
        elif estimator == "retrace":
            expected.update(
                return_lambda=0.95,
                replay_retention="action",
                offpolicy_mode="per_decision_is",
            )
        elif estimator == "full_suffix":
            expected.update(bootstrap_critic="none", target_update_event="none")
        return expected
    if comparison == "replay_correction":
        choices = {
            "fresh_round": ("n_step", "round", "none"),
            "uncorrected": ("n_step", "action", "uncorrected"),
            "pdis": ("n_step", "action", "per_decision_is"),
            "resimulate": ("n_step", "action", "resimulate"),
            "retrace": ("retrace", "action", "per_decision_is"),
        }
        estimator, retention, correction = choices[variant]
        return {
            "operator": "sac",
            "q_objective": "finite_horizon",
            "critic_horizon_mode": "depth_conditioned",
            "return_estimator": estimator,
            "return_steps": 2 if estimator == "n_step" else None,
            "return_lambda": 0.95 if estimator == "retrace" else None,
            "replay_retention": retention,
            "offpolicy_mode": correction,
        }
    if comparison == "target_strategy":
        choices = {
            "online": ("shared", "online", "none", "mixed", 0.01),
            "frozen_target": (
                "shared",
                "frozen_target",
                "none",
                "mixed",
                0.01,
            ),
            "optimizer_step_ema": (
                "shared",
                "target",
                "optimizer_step",
                "mixed",
                0.01,
            ),
            "round_end_ema": (
                "shared",
                "target",
                "round_end",
                "mixed",
                0.01,
            ),
            "depth_stage_hard": (
                "stage_heads",
                "target",
                "depth_stage",
                "backward",
                1.0,
            ),
        }
        layout, bootstrap, event, order, tau = choices[variant]
        return {
            "operator": "sac",
            "q_objective": "finite_horizon",
            "critic_horizon_mode": layout,
            "return_estimator": "td0",
            "bootstrap_critic": bootstrap,
            "target_update_event": event,
            "depth_update_order": order,
            "critic_target_tau": tau,
        }
    if comparison == "leaf_q_source":
        return {
            "operator": "sac",
            "q_objective": "finite_horizon",
            "leaf_q_source": variant,
        }
    if comparison == "vtrace_layout_retention":
        layout, retention = variant.rsplit("_", 1)
        return {
            "operator": "vtrace",
            "q_objective": "finite_horizon",
            "critic_horizon_mode": layout,
            "replay_retention": retention,
            "offpolicy_mode": "per_decision_is",
            "return_lambda": 0.95,
            "inner_critic_target": "reward_only",
            "outer_critic_target": "reward_only",
        }
    raise AssertionError(f"Unhandled selector {selector!r}")


def _tiny_algorithm_config(selector):
    algorithm_config = deepcopy(_resolved_algorithm_config(selector))
    params = algorithm_config["alg_params"]
    operator = str(params.get("inner_operator", "sac")).lower()
    q_objective = str(
        params.get("inner_q_objective", "legacy_continuing")
    ).lower()
    retention = str(params.get("inner_search_replay_retention", "action")).lower()
    params.update(
        device="cpu",
        model_size=None,
        enc_dim=16,
        mlp_dim=16,
        latent_dim=8,
        num_enc_layers=2,
        simnorm_dim=4,
        num_bins=11,
        q_num_bins=11,
        q_vmin=-5,
        q_vmax=5,
        buffer_size=32,
        batch_size=2,
        compile=False,
        inner_diagnostic_rollouts=0,
    )
    if operator != "none":
        # Two rounds make retained replay genuinely cross-policy while round
        # retention stays at the minimum complete root solve.
        params.update(
            inner_rounds=(
                2
                if retention == "action" and q_objective == "finite_horizon"
                else 1
            ),
            inner_rollouts_per_round=2,
            inner_batch_size=2,
            inner_replay_capacity=12,
        )
    algorithm_config.update(seed=73, device="cpu", total_steps=10)
    return algorithm_config


def _tiny_model(selector):
    algorithm_config = _tiny_algorithm_config(selector)
    env = gym.make("Pendulum-v1", max_episode_steps=5)
    try:
        model = AMBITDMPC2(
            "AMBITDMPC2",
            env,
            algorithm_config["alg_params"],
            {
                "seed": algorithm_config["seed"],
                "device": "cpu",
                "env": "test",
                "total_steps": algorithm_config["total_steps"],
            },
            {},
        )
    except Exception:
        env.close()
        raise
    return model, env


def test_all_selector_outputs_have_unique_experiment_identity_and_config():
    resolved = [_resolved_algorithm_config(selector) for selector in SELECTORS]
    names = [config["alg_params"]["wandb_run_name"] for config in resolved]
    tags = [tuple(config["alg_params"]["wandb_tags"]) for config in resolved]
    assert len(SELECTORS) == 35
    assert len(names) == len(set(names)) == len(SELECTORS)
    assert len(tags) == len(set(tags)) == len(SELECTORS)
    # Generated identity fields make each materialized runnable unambiguous,
    # including deliberate scientific baselines shared by two comparisons.
    assert len({repr(config) for config in resolved}) == len(SELECTORS)


def test_cross_comparison_semantic_aliases_are_only_the_declared_references():
    groups = {}
    for selector in SELECTORS:
        cfg = _build_cfg(_resolved_algorithm_config(selector))
        semantics = resolve_inner_search_semantics(cfg)
        spec = (
            semantics.exact_spec(cfg)
            if semantics.is_search
            else {
                "operator": semantics.operator,
                "q_objective": semantics.q_objective,
            }
        )
        groups.setdefault(json.dumps(spec, sort_keys=True), []).append(selector)

    duplicates = {
        frozenset(selectors)
        for selectors in groups.values()
        if len(selectors) > 1
    }
    assert len(groups) == 31
    assert duplicates == {
        frozenset(
            {
                "q_architecture_estimator/shared_td0",
                "target_strategy/optimizer_step_ema",
                "leaf_q_source/outer_target",
            }
        ),
        frozenset(
            {
                "q_architecture_estimator/depth_conditioned_n_step",
                "replay_correction/fresh_round",
            }
        ),
        frozenset(
            {
                "q_architecture_estimator/depth_conditioned_retrace",
                "replay_correction/retrace",
            }
        ),
    }


@pytest.mark.parametrize("selector", SELECTORS, ids=SELECTORS)
def test_every_named_recipe_maps_to_its_declared_search_axes(selector):
    cfg = _build_cfg(_resolved_algorithm_config(selector))
    semantics = resolve_inner_search_semantics(cfg)
    actual = (
        {
            "operator": semantics.operator,
            "q_objective": semantics.q_objective,
        }
        if not semantics.is_search
        else semantics.exact_spec(cfg)
    )
    for field, expected in _expected_search_axes(selector).items():
        assert actual[field] == expected, f"{selector}: {field}"


@pytest.mark.parametrize("selector", SELECTORS, ids=SELECTORS)
def test_every_named_recipe_runs_one_tiny_finite_action(selector):
    production_cfg = _build_cfg(_resolved_algorithm_config(selector))
    tiny_config = _tiny_algorithm_config(selector)
    tiny_cfg = _build_cfg(tiny_config)
    production_semantics = resolve_inner_search_semantics(production_cfg)
    tiny_semantics = resolve_inner_search_semantics(tiny_cfg)

    # Compute/network reductions must not silently alter what target is solved.
    if production_semantics.is_search:
        assert tiny_semantics.exact_spec(tiny_cfg) == production_semantics.exact_spec(
            production_cfg
        )
    else:
        assert (tiny_semantics.operator, tiny_semantics.q_objective) == (
            production_semantics.operator,
            production_semantics.q_objective,
        )

    model, env = _tiny_model(selector)
    try:
        action = model.agent.act(
            torch.zeros(3),
            t0=True,
            eval_mode=False,
            collect_diagnostics=False,
        )
        metrics = model.agent.last_inner_metrics
        assert action.shape == (model.cfg.action_dim,)
        assert bool(torch.isfinite(action).all().item())

        if tiny_semantics.is_search:
            expected_rounds = int(model.cfg.inner_rounds)
            expected_model_steps = (
                expected_rounds
                * int(model.cfg.inner_rollouts_per_round)
                * int(model.cfg.inner_rollout_horizon)
            )
            assert metrics["inner_model_steps"] == expected_model_steps
            assert metrics["inner_critic_optimizer_steps"] > 0
            assert metrics["inner_actor_optimizer_steps"] > 0
            if tiny_semantics.replay_retention == "round":
                assert metrics["inner_buffer_size"] == 0
            else:
                assert metrics["inner_buffer_size"] == expected_model_steps
            if tiny_semantics.is_vtrace:
                assert metrics["inner_vtrace_distill_optimizer_steps"] == 64
                assert metrics["inner_vtrace_ratio_mean"] > 0
            if tiny_semantics.offpolicy_mode == "resimulate":
                assert metrics["inner_target_model_steps"] > 0
            if tiny_semantics.bootstrap_critic in {"none", "online"}:
                assert model.agent.inner_engine._action_pool.critic_target is None
            else:
                assert model.agent.inner_engine._action_pool.critic_target is not None
        elif tiny_semantics.operator == "none":
            assert metrics["inner_model_steps"] == 0
        else:
            assert metrics["inner_model_steps"] == 6
    finally:
        env.close()
