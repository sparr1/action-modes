"""Production D512-2 contract for the adaptive parameter-noise sweep."""

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
CAMPAIGN = "ambi_humanoid_walk_base_v2_adaptive_param_noise_d512_2_k8"
STUDY_TYPE = "single_seed_exploratory_adaptive_parameter_noise_d512_2"

COMMON_PARAMETER_NOISE_PARAMS = {
    "inner_explorer_mode": "adaptive_param_noise",
    "inner_param_noise_actor_count": 8,
    "inner_param_noise_target_action_rms": 0.1,
    "inner_param_noise_sigma_init": 0.001,
    "inner_param_noise_sigma_min": 0.000001,
    "inner_param_noise_sigma_max": 0.1,
    "inner_param_noise_calibration_directions": 8,
    "inner_param_noise_calibration_batch_size": 32,
    "inner_param_noise_calibration_max_probes": 8,
    "inner_execution_policy_source": "primary",
}

CELLS = {
    "e0250": {
        "prior": 0.75,
        "prior_tag": "0p75",
        "primary": 384,
        "explorer": 128,
        "rollouts_per_actor": 16,
    },
    "e0500": {
        "prior": 0.5,
        "prior_tag": "0p50",
        "primary": 256,
        "explorer": 256,
        "rollouts_per_actor": 32,
    },
    "e0750": {
        "prior": 0.25,
        "prior_tag": "0p25",
        "primary": 128,
        "explorer": 384,
        "rollouts_per_actor": 48,
    },
}


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


def _stem(cell):
    return f"{CAMPAIGN}_{cell}"


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


def test_campaign_has_exactly_three_paired_one_cell_definitions():
    expected = {_stem(cell) for cell in CELLS}
    pattern = f"{CAMPAIGN}_e*.json"
    algorithms = {path.stem for path in ALGORITHM_ROOT.glob(pattern)}
    experiments = {path.stem for path in EXPERIMENT_ROOT.glob(pattern)}

    assert algorithms == expected
    assert experiments == expected
    assert len(expected) == 3


@pytest.mark.parametrize(("cell", "row"), CELLS.items())
def test_cell_changes_only_parameter_noise_contract_and_identity(cell, row):
    base = _load(ALGORITHM_ROOT / f"{ANCHOR}.json")
    actual = _load(ALGORITHM_ROOT / f"{_stem(cell)}.json")
    actual_params = actual["alg_params"]
    expected = copy.deepcopy(base)
    expected["alg_params"].update(
        {
            **COMMON_PARAMETER_NOISE_PARAMS,
            "inner_prior_rollout_weight": row["prior"],
            "wandb_run_name": actual_params["wandb_run_name"],
            "wandb_tags": actual_params["wandb_tags"],
        }
    )

    assert actual == expected
    assert actual["seed"] == 55
    assert actual["total_steps"] == 14_000_000
    assert actual["episodes"] is None
    assert actual_params["wandb_run_name"] == (
        "AMBITDMPC2-humanoid-walk-base-v2-adaptive-param-noise-"
        f"d512-2-k8-{cell}-seed55"
    )

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
        "adaptive-param-noise-screen",
        "adaptive-param-noise",
        "param-noise-k8",
        "param-noise-target-rms-0p10",
        "clean-logstd",
        "primary-execution",
        f"explorer-fraction-{cell[1:]}",
        f"prior-weight-{row['prior_tag']}",
        f"p{row['primary']}-e{row['explorer']}",
        f"{row['rollouts_per_actor']}-rollouts-per-perturbed-actor",
    } <= set(tags)


@pytest.mark.parametrize(("cell", "row"), CELLS.items())
def test_cell_resolves_exact_population_and_d512_2_update_contract(cell, row):
    params = _load(ALGORITHM_ROOT / f"{_stem(cell)}.json")["alg_params"]
    cfg = _resolve(params)

    assert cfg.inner_explorer_mode == "adaptive_param_noise"
    assert cfg.inner_explorer_active is True
    assert cfg.inner_param_noise_active is True
    assert cfg.inner_explorer_trainable is False
    assert cfg.inner_explorer_has_separate_critic is False
    assert cfg.inner_prior_rollout_weight == pytest.approx(row["prior"])
    assert cfg.inner_primary_rollouts_per_round == row["primary"]
    assert cfg.inner_explorer_rollouts_per_round == row["explorer"]
    assert cfg.inner_param_noise_actor_count == 8
    assert cfg.inner_param_noise_rollouts_per_actor == row["rollouts_per_actor"]
    assert cfg.inner_primary_rollout_fraction == pytest.approx(
        row["primary"] / 512
    )
    assert cfg.inner_explorer_rollout_fraction == pytest.approx(
        row["explorer"] / 512
    )

    assert cfg.inner_param_noise_target_action_rms == pytest.approx(0.1)
    assert cfg.inner_param_noise_sigma_init == pytest.approx(0.001)
    assert cfg.inner_param_noise_sigma_min == pytest.approx(0.000001)
    assert cfg.inner_param_noise_sigma_max == pytest.approx(0.1)
    assert cfg.inner_param_noise_calibration_directions == 8
    assert cfg.inner_param_noise_calibration_batch_size == 32
    assert cfg.inner_param_noise_calibration_max_probes == 8
    assert cfg.inner_behavior_action == "policy_sample"
    assert cfg.inner_execution_policy_source == "primary"

    assert cfg.inner_schedule_mode == "canonical"
    assert cfg.inner_component_update_schedule is True
    assert cfg.inner_rounds == 8
    assert cfg.inner_rollouts_per_round == 512
    assert cfg.inner_rollout_horizon == 3
    assert cfg.inner_batch_size == 512
    assert cfg.inner_replay_capacity == 12_288
    assert cfg.inner_model_step_budget == 8 * 512 * 3 == 12_288
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
    for component in ("actor", "critic", "temperature"):
        assert getattr(cfg, f"inner_explorer_{component}_updates_per_round") == 0
        assert getattr(cfg, f"inner_explorer_{component}_updates_per_action") == 0
    assert cfg.inner_total_optimizer_steps_per_action == 40


@pytest.mark.parametrize(("cell", "row"), CELLS.items())
def test_manifest_is_one_cell_and_preserves_production_run_contract(cell, row):
    stem = _stem(cell)
    base = _load(EXPERIMENT_ROOT / f"{ANCHOR}.json")
    actual = _load(EXPERIMENT_ROOT / f"{stem}.json")
    expected = copy.deepcopy(base)
    expected.update(
        {
            "study_type": STUDY_TYPE,
            "study_note": actual["study_note"],
            "configs": [stem],
        }
    )

    assert actual == expected
    assert (ALGORITHM_ROOT / f"{actual['configs'][0]}.json").is_file()
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
    assert "C=3" in note
    assert "A=1" in note
    assert "T=1" in note
    assert "K=8" in note
    assert "0.10" in note
    assert "12,288" in note
    assert str(row["primary"]) in note
    assert str(row["explorer"]) in note
    assert str(row["rollouts_per_actor"]) in note
    assert "single-seed exploratory" in note


def test_campaign_run_names_and_cell_tags_are_unique():
    configs = [
        _load(ALGORITHM_ROOT / f"{_stem(cell)}.json")["alg_params"]
        for cell in CELLS
    ]
    run_names = [params["wandb_run_name"] for params in configs]
    tags = [tuple(params["wandb_tags"]) for params in configs]

    assert len(run_names) == len(set(run_names)) == 3
    assert len(tags) == len(set(tags)) == 3
    assert all(name.endswith("-seed55") for name in run_names)
