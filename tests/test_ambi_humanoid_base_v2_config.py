import copy
import json
from pathlib import Path

import gymnasium as gym
import pytest

from RL.AMBITDMPC2 import AMBITDMPC2


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_ROOT = ROOT / "configs/dmcontrol/experiments"
BASE_V1 = "ambi_humanoid_walk_base"
BASE_V2 = "ambi_humanoid_walk_base_v2"


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def _resolve_base_v2():
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {
        "seed": 55,
        "device": "cpu",
        "env": "test-env",
        "total_steps": 14_000_000,
    }
    params = _load(ALGORITHM_ROOT / f"{BASE_V2}.json")["alg_params"]
    algorithm.custom_params = params
    try:
        return algorithm._build_cfg({"device": "cpu", **params})
    finally:
        algorithm.env.close()


def test_base_v2_changes_only_its_declared_training_contract_from_base_v1():
    base_v1 = _load(ALGORITHM_ROOT / f"{BASE_V1}.json")
    actual = _load(ALGORITHM_ROOT / f"{BASE_V2}.json")
    expected = copy.deepcopy(base_v1)
    expected_params = expected["alg_params"]
    expected_params.update(
        {
            "eval_freq": None,
            "eval_inner_comparison": False,
            "eval_value": False,
            "value_equivalence_diagnostics": False,
            "value_equivalence_loss_coef": 0.0,
            "outer_policy_episode_probability": 0.0,
            "num_q": 5,
            "outer_critic_target": "entropy_augmented",
            "inner_sac_critic_target": "entropy_augmented",
            "sac_actor_loss_scale_mode": "none",
            "ent_coef_lr": 3e-4,
            "inner_rounds": 8,
            "inner_updates_per_round": 1,
            "inner_replay_capacity": 768,
            "inner_diagnostic_rollouts": 0,
            "inner_temperature_lr": 3e-4,
            "inner_critic_target_tau": 0.01,
            "wandb_run_name": (
                "AMBITDMPC2-humanoid-walk-base-v2-j8-n32-g1-seed55"
            ),
            "wandb_tags": actual["alg_params"]["wandb_tags"],
        }
    )

    assert actual == expected


def test_base_v2_inner_schedule_keeps_eight_updates_with_more_model_data():
    params = _load(ALGORITHM_ROOT / f"{BASE_V2}.json")["alg_params"]
    rounds = params["inner_rounds"]
    rollouts = params["inner_rollouts_per_round"]
    horizon = params["inner_rollout_horizon"]
    updates = params["inner_updates_per_round"]

    assert (rounds, rollouts, horizon, updates) == (8, 32, 3, 1)
    assert params["inner_replay_capacity"] == rounds * rollouts * horizon == 768
    assert rounds * updates == 8
    assert rounds * updates * params["inner_batch_size"] == 512
    assert updates / (rollouts * horizon) == 1 / 96


def test_base_v2_resolves_expected_inner_budgets_and_inherited_actor_tau():
    cfg = _resolve_base_v2()

    assert cfg.inner_nominal_transitions_per_round == 96
    assert cfg.inner_model_step_budget == 768
    assert cfg.inner_expected_update_slots == 8
    assert cfg.inner_critic_updates_per_action == 8
    assert cfg.inner_actor_updates_per_action == 8
    assert cfg.inner_temperature_updates_per_action == 8
    assert cfg.inner_nominal_critic_utd == pytest.approx(1 / 96)
    assert cfg.inner_critic_target_tau == pytest.approx(0.01)
    assert cfg.inner_actor_target_tau == pytest.approx(0.01)


def test_base_v2_disables_auxiliary_evaluation_and_matches_temperature_rates():
    params = _load(ALGORITHM_ROOT / f"{BASE_V2}.json")["alg_params"]

    assert params["eval_freq"] is None
    assert params["eval_inner_comparison"] is False
    assert params["eval_value"] is False
    assert params["value_equivalence_diagnostics"] is False
    assert params["value_equivalence_loss_coef"] == 0.0
    assert params["outer_policy_episode_probability"] == 0.0
    assert params["inner_diagnostic_rollouts"] == 0
    assert params["inner_diagnostics_every"] == 1_000
    assert params["ent_coef"] == "auto"
    assert params["inner_temperature_mode"] == "auto"
    assert params["ent_coef_lr"] == params["inner_temperature_lr"] == 3e-4
    assert params["tau"] == params["inner_critic_target_tau"] == 0.01
    assert "inner_actor_target_tau" not in params

    for key in (
        "eval_inner_comparison_episodes",
        "eval_inner_comparison_seed",
        "eval_value_samples",
        "eval_value_seed",
        "eval_value_protocols",
        "value_equivalence_mc_samples",
        "value_equivalence_loss_mc_samples",
    ):
        assert key not in params


def test_base_v2_has_unique_wandb_identity_and_descriptive_tags():
    params = _load(ALGORITHM_ROOT / f"{BASE_V2}.json")["alg_params"]
    tags = params["wandb_tags"]

    assert len(tags) == len(set(tags))
    assert {
        "base-v2",
        "training-focused",
        "14m-decisions",
        "j8",
        "n32",
        "g1",
        "fixed-update-slots-8",
        "outer-alpha-lr3e-4",
        "inner-alpha-lr3e-4",
        "inner-target-tau1e-2",
        "eval-off",
        "no-model-checkpoints",
        "single-seed",
        "seed55",
    } <= set(tags)
    assert params["wandb_run_name"] != _load(
        ALGORITHM_ROOT / f"{BASE_V1}.json"
    )["alg_params"]["wandb_run_name"]


def test_base_v2_manifest_disables_periodic_and_final_model_saves():
    base_v1 = _load(EXPERIMENT_ROOT / f"{BASE_V1}.json")
    actual = _load(EXPERIMENT_ROOT / f"{BASE_V2}.json")
    expected = copy.deepcopy(base_v1)
    expected.update(
        {
            "study_type": "single_seed_exploratory_training_base",
            "study_note": actual["study_note"],
            "configs": [BASE_V2],
            "checkpoint_every": None,
            "save_strat": "none",
        }
    )
    expected.pop("checkpoint_best_window")

    assert actual == expected
    assert actual["overrides_alg"]["total_steps"] == 14_000_000
    assert actual["env_params"] == {
        "task": "humanoid-walk",
        "obs": "state",
        "render_mode": None,
    }
    assert actual["trials"] == 1
    assert actual["save_trials"] == "none"
    assert actual["checkpoint_every"] is None
    assert actual["save_strat"] == "none"
    assert "checkpoint_best_window" not in actual

    note = actual["study_note"].lower()
    assert "j=8, n=32, h=3, and g=1" in note
    assert "paired outer-versus-fresh-inner" in note
    assert "q-versus-mc" in note
    assert "portable periodic and final model saves are disabled" in note
    assert "no confidence, significance, or confirmatory claim" in note
