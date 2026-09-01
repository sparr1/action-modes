"""Production D512-2 configuration contract for the nine explorer cells."""

import copy
import json
from pathlib import Path

import gymnasium as gym
import pytest

from RL.AMBITDMPC2 import AMBITDMPC2


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_ROOT = ROOT / "configs/dmcontrol/experiments"
ANCHOR = "ambi_humanoid_walk_base_v2_d512_2"
CAMPAIGN = "ambi_humanoid_walk_base_v2_explorer_d512_2"

MODES = {
    "frozen_random": {
        "slug": "frozen-random",
        "explorer_updates": (0, 0, 0),
        "optimizer_steps": 40,
    },
    "shared_mixture": {
        "slug": "shared-mixture",
        "explorer_updates": (1, 0, 0),
        "optimizer_steps": 48,
    },
    "separate_critics": {
        "slug": "separate-critics",
        "explorer_updates": (1, 3, 1),
        "optimizer_steps": 80,
    },
}

WEIGHTS = {
    "e0125": {"prior": 0.875, "primary": 448, "explorer": 64},
    "e0500": {"prior": 0.5, "primary": 256, "explorer": 256},
    "e0875": {"prior": 0.125, "primary": 64, "explorer": 448},
}

CASES = [
    (mode, weight_key, mode_row, weight_row)
    for mode, mode_row in MODES.items()
    for weight_key, weight_row in WEIGHTS.items()
]


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


def _stem(mode, weight_key):
    return f"{CAMPAIGN}_{mode}_{weight_key}"


def _resolve(params):
    algorithm = object.__new__(AMBITDMPC2)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {
        "seed": 55,
        "device": "cpu",
        "env": "test-env",
        "total_steps": 14_000_000,
    }
    algorithm.custom_params = params
    try:
        return algorithm._build_cfg({"device": "cpu", **params})
    finally:
        algorithm.env.close()


def test_campaign_has_exactly_nine_paired_one_cell_definitions():
    expected = {
        _stem(mode, weight_key)
        for mode in MODES
        for weight_key in WEIGHTS
    }
    pattern = f"{CAMPAIGN}_*_e*.json"
    algorithms = {path.stem for path in ALGORITHM_ROOT.glob(pattern)}
    experiments = {path.stem for path in EXPERIMENT_ROOT.glob(pattern)}

    assert algorithms == expected
    assert experiments == expected
    assert len(expected) == 9


@pytest.mark.parametrize(
    ("mode", "weight_key", "mode_row", "weight_row"),
    CASES,
)
def test_cell_changes_only_explorer_contract_and_identity(
    mode,
    weight_key,
    mode_row,
    weight_row,
):
    base = _load(ALGORITHM_ROOT / f"{ANCHOR}.json")
    actual = _load(ALGORITHM_ROOT / f"{_stem(mode, weight_key)}.json")
    expected = copy.deepcopy(base)
    params = expected["alg_params"]
    actual_params = actual["alg_params"]
    params.update(
        {
            "inner_explorer_mode": mode,
            "inner_prior_rollout_weight": weight_row["prior"],
            "inner_mixture_target_estimator": "stratified",
            "inner_execution_policy_source": "primary",
            "wandb_run_name": actual_params["wandb_run_name"],
            "wandb_tags": actual_params["wandb_tags"],
        }
    )

    assert actual == expected
    assert actual["seed"] == 55
    assert actual["total_steps"] == 14_000_000
    assert actual["episodes"] is None

    tags = actual_params["wandb_tags"]
    assert len(tags) == len(set(tags))
    assert {
        "d512-2",
        "j8",
        "n512",
        "batch512",
        "phased-updates",
        "c3",
        "a1",
        "alpha-follows-actor",
        "capacity-12288",
        "random-explorer-screen",
        mode_row["slug"],
        f"explorer-fraction-{weight_key[1:]}",
        f"p{weight_row['primary']}-r{weight_row['explorer']}",
        "primary-execution",
        "mixture-estimator-stratified",
    } <= set(tags)
    assert actual_params["wandb_run_name"] == (
        "AMBITDMPC2-humanoid-walk-base-v2-explorer-d512-2-"
        f"{mode_row['slug']}-{weight_key}-seed55"
    )


@pytest.mark.parametrize(
    ("mode", "weight_key", "mode_row", "weight_row"),
    CASES,
)
def test_cell_resolves_exact_population_and_d512_2_update_contract(
    mode,
    weight_key,
    mode_row,
    weight_row,
):
    params = _load(
        ALGORITHM_ROOT / f"{_stem(mode, weight_key)}.json"
    )["alg_params"]
    cfg = _resolve(params)

    assert cfg.inner_explorer_mode == mode
    assert cfg.inner_explorer_active is True
    assert cfg.inner_prior_rollout_weight == pytest.approx(weight_row["prior"])
    assert cfg.inner_primary_rollouts_per_round == weight_row["primary"]
    assert cfg.inner_explorer_rollouts_per_round == weight_row["explorer"]
    assert cfg.inner_primary_rollout_fraction == pytest.approx(
        weight_row["primary"] / 512
    )
    assert cfg.inner_explorer_rollout_fraction == pytest.approx(
        weight_row["explorer"] / 512
    )

    assert cfg.inner_schedule_mode == "canonical"
    assert cfg.inner_component_update_schedule is True
    assert cfg.inner_rounds == 8
    assert cfg.inner_rollouts_per_round == 512
    assert cfg.inner_rollout_horizon == 3
    assert cfg.inner_batch_size == 512
    assert cfg.inner_replay_capacity == 12_288
    assert cfg.inner_model_step_budget == 12_288
    assert cfg.inner_updates_per_round is None
    assert cfg.inner_critic_updates_per_round == 3
    assert cfg.inner_actor_updates_per_round == 1
    assert cfg.inner_primary_critic_updates_per_round == 3
    assert cfg.inner_primary_actor_updates_per_round == 1
    assert cfg.inner_primary_temperature_updates_per_round == 1
    assert cfg.inner_critic_updates_per_action == 24
    assert cfg.inner_actor_updates_per_action == 8
    assert cfg.inner_temperature_updates_per_action == 8
    assert cfg.inner_primary_optimizer_steps_per_action == 40
    assert cfg.inner_expected_update_slots == 32

    explorer_actor, explorer_critic, explorer_temperature = mode_row[
        "explorer_updates"
    ]
    assert cfg.inner_explorer_actor_updates_per_round == explorer_actor
    assert cfg.inner_explorer_critic_updates_per_round == explorer_critic
    assert (
        cfg.inner_explorer_temperature_updates_per_round
        == explorer_temperature
    )
    assert cfg.inner_explorer_actor_updates_per_action == 8 * explorer_actor
    assert cfg.inner_explorer_critic_updates_per_action == 8 * explorer_critic
    assert (
        cfg.inner_explorer_temperature_updates_per_action
        == 8 * explorer_temperature
    )
    assert cfg.inner_total_optimizer_steps_per_action == mode_row[
        "optimizer_steps"
    ]

    assert cfg.inner_mixture_target_estimator == "stratified"
    assert cfg.inner_execution_policy_source == "primary"
    if mode == "shared_mixture":
        assert cfg.inner_primary_target_rows_per_batch == weight_row["primary"]
        assert cfg.inner_explorer_target_rows_per_batch == weight_row["explorer"]
    else:
        assert cfg.inner_primary_target_rows_per_batch is None
        assert cfg.inner_explorer_target_rows_per_batch is None


@pytest.mark.parametrize(
    ("mode", "weight_key", "mode_row", "weight_row"),
    CASES,
)
def test_manifest_is_one_cell_and_preserves_production_run_contract(
    mode,
    weight_key,
    mode_row,
    weight_row,
):
    stem = _stem(mode, weight_key)
    base = _load(EXPERIMENT_ROOT / f"{ANCHOR}.json")
    actual = _load(EXPERIMENT_ROOT / f"{stem}.json")
    expected = copy.deepcopy(base)
    expected.update(
        {
            "study_type": "single_seed_exploratory_random_explorer_d512_2",
            "study_note": actual["study_note"],
            "configs": [stem],
        }
    )

    assert actual == expected
    assert actual["trials"] == 1
    assert actual["logs"] == "timestamp"
    assert actual["save_trials"] == actual["save_strat"] == "none"
    assert actual["checkpoint_every"] is None
    assert actual["overrides_alg"]["seed"] == 55
    assert actual["overrides_alg"]["total_steps"] == 14_000_000
    assert actual["env_params"] == {
        "task": "humanoid-walk",
        "obs": "state",
        "render_mode": None,
    }

    note = actual["study_note"]
    assert "J=8, N=512, H=3" in note
    assert "C=3 primary critic" in note
    assert "A=1 primary actor" in note
    assert "T=1 primary automatic-temperature" in note
    assert f"{weight_row['primary']} P and {weight_row['explorer']} R" in note
    assert "execution always samples P" in note
    assert "single-seed exploratory cell" in note


def test_campaign_run_names_are_unique():
    run_names = [
        _load(ALGORITHM_ROOT / f"{_stem(mode, weight_key)}.json")[
            "alg_params"
        ]["wandb_run_name"]
        for mode in MODES
        for weight_key in WEIGHTS
    ]

    assert len(run_names) == len(set(run_names)) == 9
