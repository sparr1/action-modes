import json
import warnings
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import pytest

import main
from RL.AMBITDMPC2 import AMBITDMPC2
from RL.tdmpc2_core.ambi_agent import AMBITDMPC2Agent
from RL.tdmpc2_core.common.search_config import resolve_inner_search_semantics
from utils.ambi_research import (
    list_preset_selectors,
    load_preset_matrix,
    resolve_preset,
)
from utils.resume_identity import scientific_trial_parameters


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/research/ambi_search_humanoid_walk_v2.json"


def _build_cfg(algorithm_config):
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {
        **algorithm_config,
        "seed": 3,
        "device": "cpu",
        "env": "test-env",
        "total_steps": 12,
    }
    algorithm.custom_params = algorithm_config["alg_params"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return algorithm._build_cfg(
                {"device": "cpu", **algorithm_config["alg_params"]}
            )
    finally:
        algorithm.env.close()


def _resolved(selector):
    matrix = load_preset_matrix(MATRIX)
    return resolve_preset(MATRIX, selector, matrix=matrix)["algorithm_config"]


def _mutated(selector, **updates):
    config = deepcopy(_resolved(selector))
    for key, value in updates.items():
        if value is None:
            config["alg_params"].pop(key, None)
        else:
            config["alg_params"][key] = value
    return config


def test_humanoid_search_matrix_is_complete_and_every_recipe_resolves():
    matrix = load_preset_matrix(MATRIX)
    selectors = list_preset_selectors(matrix)
    assert len(selectors) == 35
    assert set(matrix["comparisons"]) == {
        "controls",
        "q_architecture_estimator",
        "replay_correction",
        "target_strategy",
        "leaf_q_source",
        "vtrace_layout_retention",
    }

    q_variants = matrix["comparisons"]["q_architecture_estimator"]["variants"]
    assert set(q_variants) == {
        f"{layout}_{estimator}"
        for layout in ("shared", "depth_conditioned", "stage_heads")
        for estimator in (
            "td0",
            "n_step",
            "lambda_return",
            "full_suffix",
            "retrace",
        )
    }
    vtrace_variants = matrix["comparisons"]["vtrace_layout_retention"]["variants"]
    assert set(vtrace_variants) == {
        f"{layout}_{retention}"
        for layout in ("shared", "depth_conditioned", "stage_heads")
        for retention in ("round", "action")
    }

    for selector in selectors:
        cfg = _build_cfg(_resolved(selector))
        assert cfg.compile is True
        if selector == "controls/no_inner":
            assert cfg.inner_operator == "none"
        elif selector == "controls/legacy_continuing":
            assert cfg.inner_q_objective == "legacy_continuing"
            assert not cfg.inner_search_active
        else:
            assert cfg.inner_search_active
            assert cfg.inner_uses_outer_leaf
            assert cfg.inner_uses_structured_replay


def test_search_matrix_generates_unique_wandb_identity_without_expanded_jsons():
    matrix = load_preset_matrix(MATRIX)
    names = []
    unique_tags = []
    for selector in list_preset_selectors(matrix):
        params = resolve_preset(MATRIX, selector, matrix=matrix)["algorithm_config"][
            "alg_params"
        ]
        names.append(params["wandb_run_name"])
        selector_tag = f"ambisearch-humanoid-walk-v2-{selector.replace('/', '-')}"
        unique_tags.append(selector_tag)
        assert selector_tag in params["wandb_tags"]
    assert len(names) == len(set(names))
    assert len(unique_tags) == len(set(unique_tags))
    assert not list((ROOT / "configs/research").glob("**/generated/*.json"))


@pytest.mark.parametrize(
    ("selector", "updates", "message"),
    [
        (
            "q_architecture_estimator/shared_n_step",
            {"inner_return_steps": 1},
            "2 <= inner_return_steps",
        ),
        (
            "q_architecture_estimator/shared_lambda_return",
            {"inner_return_lambda": 1.0},
            "use td0 or full_suffix",
        ),
        (
            "q_architecture_estimator/shared_full_suffix",
            {"inner_search_bootstrap_critic": "target"},
            "full_suffix uses no inner bootstrap",
        ),
        (
            "replay_correction/uncorrected",
            {"inner_offpolicy_mode": "none"},
            "must explicitly select",
        ),
        (
            "replay_correction/pdis",
            {"inner_behavior_action": "mean"},
            "require exact policy_sample",
        ),
        (
            "q_architecture_estimator/shared_n_step",
            {"inner_behavior_action": "mean"},
            "Fresh-round multistep targets are on-policy only",
        ),
        (
            "q_architecture_estimator/shared_n_step",
            {"inner_behavior_std_scale": 0.5},
            "Fresh-round multistep targets are on-policy only",
        ),
        (
            "replay_correction/pdis",
            {"inner_actor_adaptation": "lora", "inner_actor_lora_dropout": 0.1},
            "require inner_actor_lora_dropout=0",
        ),
        (
            "target_strategy/frozen_target",
            {"inner_target_update_event": "optimizer_step"},
            "requires inner_target_update_event='none'",
        ),
        (
            "target_strategy/depth_stage_hard",
            {"inner_critic_horizon_mode": "shared"},
            "requires depth_conditioned or stage_heads",
        ),
        (
            "q_architecture_estimator/shared_td0",
            {"inner_sac_critic_target": "reward_only"},
            "matching outer_critic_target",
        ),
        (
            "vtrace_layout_retention/shared_round",
            {"outer_critic_target": "entropy_augmented"},
            "requires reward-only",
        ),
        (
            "vtrace_layout_retention/shared_round",
            {
                "inner_search_bootstrap_critic": "online",
                "inner_target_update_event": "none",
            },
            "requires a target value network",
        ),
        (
            "vtrace_layout_retention/shared_round",
            {"inner_depth_update_order": "backward"},
            "requires inner_depth_update_order='mixed'",
        ),
        (
            "q_architecture_estimator/depth_conditioned_td0",
            {"inner_critic_adaptation": "lora"},
            "Critic LoRA is supported only",
        ),
        (
            "q_architecture_estimator/shared_td0",
            {"inner_actor_scope": "episode"},
            "requires inner_actor_scope='action'",
        ),
        (
            "q_architecture_estimator/shared_td0",
            {"inner_bootstrap_source": "inner_target"},
            "belongs exclusively",
        ),
        (
            "q_architecture_estimator/shared_td0",
            {"inner_replay_capacity": 767},
            "exact multiple of inner_rollout_horizon",
        ),
        (
            "q_architecture_estimator/shared_td0",
            {
                "inner_replay_sampling": "without_replacement",
                "inner_batch_size": 97,
            },
            "eligible=96",
        ),
        (
            "vtrace_layout_retention/shared_action",
            {
                "inner_replay_sampling": "without_replacement",
                "inner_batch_size": 33,
            },
            "eligible=32",
        ),
        (
            "target_strategy/depth_stage_hard",
            {
                "inner_replay_sampling": "without_replacement",
                "inner_batch_size": 33,
            },
            "eligible=32",
        ),
        (
            "target_strategy/online",
            {
                "inner_depth_update_order": "backward",
                "inner_replay_sampling": "without_replacement",
                "inner_batch_size": 33,
            },
            "eligible=32",
        ),
    ],
)
def test_invalid_search_combinations_fail_loudly(selector, updates, message):
    with pytest.raises(ValueError, match=message):
        _build_cfg(_mutated(selector, **updates))


@pytest.mark.parametrize(
    ("selector", "batch_size"),
    [
        ("q_architecture_estimator/shared_td0", 96),
        ("vtrace_layout_retention/shared_action", 32),
        ("target_strategy/depth_stage_hard", 32),
    ],
)
def test_without_replacement_accepts_exact_first_round_population_boundary(
    selector, batch_size
):
    cfg = _build_cfg(
        _mutated(
            selector,
            inner_replay_sampling="without_replacement",
            inner_batch_size=batch_size,
        )
    )
    assert cfg.inner_batch_size == batch_size


def test_search_semantic_predicates_cover_target_inventory_and_events():
    online = _build_cfg(_resolved("target_strategy/online"))
    online_semantics = resolve_inner_search_semantics(online)
    assert online_semantics.uses_outer_leaf
    assert not online_semantics.creates_inner_target
    assert online_semantics.target_update_event == "none"

    frozen = _build_cfg(_resolved("target_strategy/frozen_target"))
    frozen_semantics = resolve_inner_search_semantics(frozen)
    assert frozen_semantics.creates_inner_target
    assert not frozen_semantics.updates_inner_target
    assert frozen_semantics.target_update_event == "none"

    ema = _build_cfg(_resolved("target_strategy/round_end_ema"))
    ema_semantics = resolve_inner_search_semantics(ema)
    assert ema_semantics.updates_inner_target
    assert ema_semantics.target_update_event == "round_end"

    suffix = _build_cfg(_resolved("q_architecture_estimator/shared_full_suffix"))
    suffix_semantics = resolve_inner_search_semantics(suffix)
    assert not suffix_semantics.creates_inner_target
    assert suffix_semantics.target_update_event == "none"


def test_backward_q_schedule_accounts_for_every_depth_stage():
    cfg = _build_cfg(
        _mutated(
            "target_strategy/online",
            inner_depth_update_order="backward",
        )
    )
    rounds = cfg.inner_rounds
    horizon = cfg.inner_rollout_horizon
    critic_per_depth = cfg.inner_critic_updates_per_round
    actor_per_round = cfg.inner_actor_updates_per_round

    assert cfg.inner_critic_depth_stages == horizon
    assert cfg.inner_effective_critic_updates_per_round == (
        horizon * critic_per_depth
    )
    assert cfg.inner_critic_updates_per_action == (
        rounds * horizon * critic_per_depth
    )
    assert cfg.inner_nominal_updates_per_round == (
        horizon * critic_per_depth + actor_per_round
    )
    assert cfg.inner_expected_update_slots == (
        rounds * (horizon * critic_per_depth + actor_per_round)
    )
    assert cfg.inner_nominal_critic_utd == pytest.approx(
        cfg.inner_critic_updates_per_action / cfg.inner_model_step_budget
    )
    model = SimpleNamespace(
        cfg=cfg,
        agent=SimpleNamespace(critic_signature={"num_q": cfg.num_q}),
    )
    metadata = main._resolved_runtime_metadata(
        model,
        trial_run_params={"alg": "AMBITDMPC2/AMBITDMPC2", "seed": 55},
    )["inner_budget"]
    assert metadata["inner_critic_depth_stages"] == horizon
    assert metadata["inner_effective_critic_updates_per_round"] == (
        horizon * critic_per_depth
    )
    assert metadata["inner_critic_updates_per_action"] == (
        rounds * horizon * critic_per_depth
    )


def test_exact_critic_target_spec_is_search_complete_but_legacy_stable():
    legacy_cfg = _build_cfg(_resolved("controls/legacy_continuing"))
    legacy_agent = object.__new__(AMBITDMPC2Agent)
    legacy_agent.cfg = legacy_cfg
    assert legacy_agent._critic_target_spec() == {
        "outer_critic_target": "entropy_augmented",
        "inner_sac_critic_target": "entropy_augmented",
    }

    search_cfg = _build_cfg(_resolved("replay_correction/pdis"))
    search_agent = object.__new__(AMBITDMPC2Agent)
    search_agent.cfg = search_cfg
    spec = search_agent._critic_target_spec()["inner_search"]
    assert spec["q_objective"] == "finite_horizon"
    assert spec["return_estimator"] == "n_step"
    assert spec["return_steps"] == 2
    assert spec["replay_retention"] == "action"
    assert spec["offpolicy_mode"] == "per_decision_is"
    assert spec["leaf_q_source"] == "outer_target"
    assert spec["bootstrap_critic"] == "target"
    assert spec["target_update_event"] == "optimizer_step"


def test_runtime_metadata_contains_every_search_semantic_field():
    cfg = _build_cfg(_resolved("vtrace_layout_retention/stage_heads_action"))
    agent = SimpleNamespace(critic_signature={"num_q": cfg.num_q})
    model = SimpleNamespace(cfg=cfg, agent=agent)
    metadata = main._resolved_runtime_metadata(
        model,
        trial_run_params={"alg": "AMBITDMPC2/AMBITDMPC2", "seed": 55},
    )["inner_budget"]
    expected = {
        "inner_q_objective",
        "inner_critic_horizon_mode",
        "inner_return_estimator",
        "inner_return_steps",
        "inner_return_lambda",
        "inner_leaf_q_source",
        "inner_leaf_value_samples",
        "inner_search_replay_retention",
        "inner_offpolicy_mode",
        "inner_search_bootstrap_critic",
        "inner_target_update_event",
        "inner_depth_update_order",
        "inner_critic_depth_stages",
        "inner_effective_critic_updates_per_round",
        "inner_vtrace_rho_clip",
        "inner_vtrace_c_clip",
        "inner_vtrace_pg_rho_clip",
        "inner_vtrace_distill_updates",
        "inner_vtrace_distill_action_samples",
    }
    assert expected <= set(metadata)
    assert metadata["inner_search_family"] == "vtrace"


def test_resume_identity_treats_omitted_and_explicit_search_defaults_equally():
    omitted = {
        "alg": "AMBITDMPC2/AMBITDMPC2",
        "alg_params": {},
    }
    explicit = deepcopy(omitted)
    explicit["alg_params"] = {
        "inner_operator": "SAC",
        "inner_q_objective": "LEGACY_CONTINUING",
        "inner_critic_horizon_mode": "SHARED",
        "inner_return_estimator": "TD0",
        "inner_return_steps": None,
        "inner_return_lambda": None,
        "inner_leaf_q_source": "OUTER_TARGET",
        "inner_leaf_value_samples": 1,
        "inner_search_replay_retention": "ACTION",
        "inner_offpolicy_mode": "NONE",
        "inner_search_bootstrap_critic": "TARGET",
        "inner_target_update_event": "OPTIMIZER_STEP",
        "inner_depth_update_order": "MIXED",
        "inner_vtrace_rho_clip": 1,
        "inner_vtrace_c_clip": 1,
        "inner_vtrace_pg_rho_clip": 1,
        "inner_vtrace_distill_updates": 64,
        "inner_vtrace_distill_action_samples": 4,
        "inner_bootstrap_source": "INNER_TARGET",
    }
    assert scientific_trial_parameters(omitted) == scientific_trial_parameters(
        explicit
    )
