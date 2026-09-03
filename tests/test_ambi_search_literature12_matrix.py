"""Contracts for the zero-KL, literature-shaped twelve-job campaign."""

from __future__ import annotations

import json
from pathlib import Path

from tests.test_ambi_search_config import _build_cfg
from utils.ambi_research import (
    list_preset_selectors,
    load_preset_matrix,
    materialize_presets,
    resolve_preset,
)


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = (
    ROOT / "configs/research/ambi_search_humanoid_walk_literature12.json"
)
MATERIALIZED_ROOT = (
    ROOT / "configs/research/ambi_search_humanoid_walk_literature12"
)
ALGORITHM_DIR = MATERIALIZED_ROOT / "algs"
EXPERIMENT_DIR = MATERIALIZED_ROOT / "experiments"
MATRIX = load_preset_matrix(MATRIX_PATH)
SELECTORS = list_preset_selectors(MATRIX)

EXPECTED_SELECTORS = [
    "full_suffix/shared_a1",
    "full_suffix/depth_a1",
    "full_suffix/stage_a1",
    "full_suffix/shared_a4",
    "full_suffix/depth_a4",
    "lambda_online/shared_a4",
    "lambda_online/depth_a4",
    "lambda_online/shared_a12",
    "vtrace/shared_a4",
    "vtrace/depth_a4",
    "hard_propagation/stage_td0",
    "polyak_ablation/depth_lambda_tau005",
]
EXPECTED_STEMS = [
    f"{index:02d}_{selector.replace('/', '__')}"
    for index, selector in enumerate(EXPECTED_SELECTORS)
]


def _resolved(selector):
    return resolve_preset(MATRIX_PATH, selector, matrix=MATRIX)[
        "algorithm_config"
    ]


def test_literature12_has_exactly_twelve_unique_runnable_cells():
    assert SELECTORS == EXPECTED_SELECTORS
    configs = [_resolved(selector) for selector in SELECTORS]
    params = [config["alg_params"] for config in configs]
    assert len({item["wandb_run_name"] for item in params}) == 12
    assert len({tuple(item["wandb_tags"]) for item in params}) == 12
    for config in configs:
        _build_cfg(config)


def test_literature12_common_compute_and_zero_kl_contract():
    for selector in SELECTORS:
        params = _resolved(selector)["alg_params"]
        cfg = _build_cfg(_resolved(selector))
        assert params["inner_outer_policy_kl_coef"] == 0.0
        assert params["outer_behavior_policy_kl_schedule"] == "none"
        assert params["outer_behavior_policy_kl_coef"] == 0.0
        assert params["inner_outer_action_l2_coef"] == 0.0
        assert params["inner_temperature_mode"] == "inherit_outer"
        assert params["inner_rounds"] == 8
        assert params["inner_rollouts_per_round"] == 512
        assert params["inner_rollout_horizon"] == 3
        assert params["inner_batch_size"] == 512
        assert params["inner_replay_capacity"] == 12288
        assert params["inner_replay_sampling"] == "without_replacement"
        assert params["inner_actor_lr"] == 0.00025
        assert params["inner_critic_lr"] == 0.00025
        assert params["inner_search_replay_retention"] == "round"
        assert params["inner_leaf_q_source"] == "outer_online"
        assert params["inner_leaf_value_samples"] == 1
        assert params["inner_behavior_action"] == "policy_sample"
        assert params["inner_behavior_std_scale"] == 1.0
        assert params["inner_behavior_noise_std"] == 0.0
        assert params["inner_execution_policy_source"] == "primary"
        assert params["inner_execution_std_scale"] == 1.0
        assert params["inner_execution_noise_std"] == 0.0
        assert params["inner_rebase_persistent"] is False
        assert cfg.inner_model_step_budget == 12288
        assert cfg.inner_temperature_updates_per_action == 0


def test_literature12_target_and_update_dose_contract():
    for selector in SELECTORS:
        params = _resolved(selector)["alg_params"]
        cfg = _build_cfg(_resolved(selector))
        if selector.startswith("full_suffix/"):
            assert params["inner_return_estimator"] == "full_suffix"
            assert params["inner_search_bootstrap_critic"] == "none"
            assert params["inner_target_update_event"] == "none"
        elif selector.startswith("lambda_online/"):
            assert params["inner_return_estimator"] == "lambda_return"
            assert params["inner_return_lambda"] == 0.95
            assert params["inner_search_bootstrap_critic"] == "online"
            assert params["inner_target_update_event"] == "none"
        elif selector.startswith("vtrace/"):
            assert params["inner_operator"] == "vtrace"
            assert params["inner_offpolicy_mode"] == "per_decision_is"
            assert params["outer_critic_target"] == "reward_only"
            assert params["inner_sac_critic_target"] == "reward_only"
            assert params["inner_critic_target_tau"] == 1.0
            assert cfg.inner_critic_updates_per_action == 8
        elif selector == "hard_propagation/stage_td0":
            assert params["inner_critic_horizon_mode"] == "stage_heads"
            assert params["inner_return_estimator"] == "td0"
            assert params["inner_target_update_event"] == "depth_stage"
            assert params["inner_depth_update_order"] == "backward"
            assert params["inner_critic_target_tau"] == 1.0
        else:
            assert selector == "polyak_ablation/depth_lambda_tau005"
            assert params["inner_search_bootstrap_critic"] == "target"
            assert params["inner_target_update_event"] == "optimizer_step"
            assert params["inner_critic_target_tau"] == 0.05

        if selector.startswith("vtrace/"):
            assert cfg.inner_critic_updates_per_action == 8
        else:
            assert cfg.inner_critic_updates_per_action == 24
        expected_actor = 96 if selector.endswith("shared_a12") else (
            32 if selector.endswith("a4") or selector.endswith("tau005") else 8
        )
        assert cfg.inner_actor_updates_per_action == expected_actor


def test_literature12_materializes_standard_manifest(tmp_path):
    written = materialize_presets(MATRIX_PATH, tmp_path)
    assert len(written) == 12
    manifest = json.loads((tmp_path / "AMBIResearchExperiment.json").read_text())
    assert manifest["configs"] == [path.stem for path in written]
    assert manifest["trials"] == 1
    assert manifest["overrides_alg"]["env"] == "DMControl-v0"
    assert manifest["env_params"]["task"] == "humanoid-walk"


def test_tracked_materialization_and_exact_resume_manifests_match_matrix():
    assert sorted(path.stem for path in ALGORITHM_DIR.glob("*.json")) == (
        EXPECTED_STEMS
    )
    assert sorted(path.stem for path in EXPERIMENT_DIR.glob("*.json")) == (
        EXPECTED_STEMS
    )
    for selector, stem in zip(EXPECTED_SELECTORS, EXPECTED_STEMS, strict=True):
        tracked = json.loads((ALGORITHM_DIR / f"{stem}.json").read_text())
        assert tracked == _resolved(selector)
        manifest = json.loads((EXPERIMENT_DIR / f"{stem}.json").read_text())
        assert manifest["configs"] == [stem]
        assert manifest["trials"] == 1
        assert manifest["logs"] == "none"
        assert manifest["save_trials"] == "none"
        assert manifest["overrides_alg"] == {
            "seed": 55,
            "device": "cuda",
            "env": "DMControl-v0",
            "total_steps": 14_000_000,
            "episodes": None,
        }
        assert manifest["env_params"] == {
            "task": "humanoid-walk",
            "obs": "state",
            "render_mode": None,
        }
