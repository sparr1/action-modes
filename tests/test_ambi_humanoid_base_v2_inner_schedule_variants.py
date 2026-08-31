import copy
import json
from pathlib import Path

import gymnasium as gym
import pytest

from RL.AMBITDMPC2 import AMBITDMPC2


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_ROOT = ROOT / "configs/dmcontrol/experiments"
BASE = "ambi_humanoid_walk_base_v2"

VARIANTS = {
    "d256_1_j6": {
        "id": "D256-1-J6",
        "role": "Half-scale 1:1 baseline",
        "j": 6,
        "n": 256,
        "batch": 256,
        "schedule": "shared",
        "g": 1,
        "critic_per_round": 1,
        "actor_per_round": 1,
        "critic_total": 6,
        "actor_total": 6,
        "alpha_total": 6,
        "slots": 6,
        "capacity": 4_608,
    },
    "d256_1": {
        "id": "D256-1",
        "role": "Half-scale 1:1 baseline",
        "j": 8,
        "n": 256,
        "batch": 256,
        "schedule": "shared",
        "g": 1,
        "critic_per_round": 1,
        "actor_per_round": 1,
        "critic_total": 8,
        "actor_total": 8,
        "alpha_total": 8,
        "slots": 8,
        "capacity": 6_144,
    },
    "d256_2": {
        "id": "D256-2",
        "role": "Half-scale critic-heavy",
        "j": 8,
        "n": 256,
        "batch": 256,
        "schedule": "phased",
        "critic_per_round": 3,
        "actor_per_round": 1,
        "critic_total": 24,
        "actor_total": 8,
        "alpha_total": 8,
        "slots": 32,
        "capacity": 6_144,
    },
    "d512_1_j6": {
        "id": "D512-1-J6",
        "role": "Full-width 1:1 baseline",
        "j": 6,
        "n": 512,
        "batch": 512,
        "schedule": "shared",
        "g": 1,
        "critic_per_round": 1,
        "actor_per_round": 1,
        "critic_total": 6,
        "actor_total": 6,
        "alpha_total": 6,
        "slots": 6,
        "capacity": 9_216,
    },
    "d512_1": {
        "id": "D512-1",
        "role": "Full-scale missing baseline",
        "j": 8,
        "n": 512,
        "batch": 512,
        "schedule": "shared",
        "g": 1,
        "critic_per_round": 1,
        "actor_per_round": 1,
        "critic_total": 8,
        "actor_total": 8,
        "alpha_total": 8,
        "slots": 8,
        "capacity": 12_288,
    },
    "d512_2": {
        "id": "D512-2",
        "role": "Primary critic-heavy proposal",
        "j": 8,
        "n": 512,
        "batch": 512,
        "schedule": "phased",
        "critic_per_round": 3,
        "actor_per_round": 1,
        "critic_total": 24,
        "actor_total": 8,
        "alpha_total": 8,
        "slots": 32,
        "capacity": 12_288,
    },
    "d512_3": {
        "id": "D512-3",
        "role": "Heavy 1:1 phased optimization",
        "j": 8,
        "n": 512,
        "batch": 512,
        "schedule": "phased",
        "critic_per_round": 3,
        "actor_per_round": 3,
        "critic_total": 24,
        "actor_total": 24,
        "alpha_total": 24,
        "slots": 48,
        "capacity": 12_288,
    },
    "d512_4_j6": {
        "id": "D512-4-J6",
        "role": "Heavy old-scheduler 1:1 baseline",
        "j": 6,
        "n": 512,
        "batch": 512,
        "schedule": "shared",
        "g": 3,
        "critic_per_round": 3,
        "actor_per_round": 3,
        "critic_total": 18,
        "actor_total": 18,
        "alpha_total": 18,
        "slots": 18,
        "capacity": 9_216,
    },
    "d512_4": {
        "id": "D512-4",
        "role": "Heavy old-scheduler 1:1 baseline",
        "j": 8,
        "n": 512,
        "batch": 512,
        "schedule": "shared",
        "g": 3,
        "critic_per_round": 3,
        "actor_per_round": 3,
        "critic_total": 24,
        "actor_total": 24,
        "alpha_total": 24,
        "slots": 24,
        "capacity": 12_288,
    },
    "d512_5": {
        "id": "D512-5",
        "role": "Batch-vs-iterations control",
        "j": 8,
        "n": 512,
        "batch": 256,
        "schedule": "phased",
        "critic_per_round": 6,
        "actor_per_round": 2,
        "critic_total": 48,
        "actor_total": 16,
        "alpha_total": 16,
        "slots": 64,
        "capacity": 12_288,
    },
}

TOTAL_ROLLOUTS = {
    "d256_1_j6": 1_536,
    "d256_1": 2_048,
    "d256_2": 2_048,
    "d512_1_j6": 3_072,
    "d512_1": 4_096,
    "d512_2": 4_096,
    "d512_3": 4_096,
    "d512_4_j6": 3_072,
    "d512_4": 4_096,
    "d512_5": 4_096,
}

REPLAY_ROWS_DRAWN = {
    "d256_1_j6": 1_536,
    "d256_1": 2_048,
    "d256_2": 8_192,
    "d512_1_j6": 3_072,
    "d512_1": 4_096,
    "d512_2": 16_384,
    "d512_3": 24_576,
    "d512_4_j6": 9_216,
    "d512_4": 12_288,
    "d512_5": 16_384,
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


def _stem(suffix):
    return f"{BASE}_{suffix}"


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


def test_variant_file_sets_are_complete_and_one_to_one():
    expected = {_stem(suffix) for suffix in VARIANTS}
    actual_algorithms = {
        path.stem for path in ALGORITHM_ROOT.glob(f"{BASE}_d*.json")
    }
    actual_experiments = {
        path.stem for path in EXPERIMENT_ROOT.glob(f"{BASE}_d*.json")
    }

    assert actual_algorithms == expected
    assert actual_experiments == expected


@pytest.mark.parametrize(("suffix", "row"), VARIANTS.items())
def test_variant_changes_only_declared_inner_schedule_and_identity(suffix, row):
    base = _load(ALGORITHM_ROOT / f"{BASE}.json")
    actual = _load(ALGORITHM_ROOT / f"{_stem(suffix)}.json")
    expected = copy.deepcopy(base)
    params = expected["alg_params"]
    actual_params = actual["alg_params"]

    params.update(
        {
            "inner_rounds": row["j"],
            "inner_rollouts_per_round": row["n"],
            "inner_batch_size": row["batch"],
            "inner_replay_capacity": row["capacity"],
            "wandb_run_name": actual_params["wandb_run_name"],
            "wandb_tags": actual_params["wandb_tags"],
        }
    )
    if row["schedule"] == "shared":
        params["inner_updates_per_round"] = row["g"]
        params.pop("inner_critic_updates_per_round", None)
        params.pop("inner_actor_updates_per_round", None)
    else:
        params.pop("inner_updates_per_round")
        params["inner_critic_updates_per_round"] = row["critic_per_round"]
        params["inner_actor_updates_per_round"] = row["actor_per_round"]

    assert actual == expected


@pytest.mark.parametrize(("suffix", "row"), VARIANTS.items())
def test_variant_resolves_requested_budget_updates_and_slots(suffix, row):
    params = _load(ALGORITHM_ROOT / f"{_stem(suffix)}.json")["alg_params"]
    cfg = _resolve(params)
    expected_transitions = row["j"] * row["n"] * 3

    assert cfg.inner_schedule_mode == "canonical"
    assert cfg.inner_operator == "sac"
    assert cfg.inner_rounds == row["j"]
    assert cfg.inner_rollouts_per_round == row["n"]
    assert cfg.inner_rollout_horizon == 3
    assert cfg.inner_batch_size == row["batch"]
    assert cfg.inner_replay_capacity == expected_transitions == row["capacity"]
    assert cfg.inner_nominal_transitions_per_round == row["n"] * 3
    assert cfg.inner_model_step_budget == expected_transitions
    assert row["j"] * row["n"] == TOTAL_ROLLOUTS[suffix]

    assert cfg.inner_critic_updates_per_action == row["critic_total"]
    assert cfg.inner_actor_updates_per_action == row["actor_total"]
    assert cfg.inner_temperature_updates_per_action == row["alpha_total"]
    assert cfg.inner_expected_update_slots == row["slots"]
    assert cfg.inner_nominal_critic_utd == pytest.approx(
        row["critic_total"] / row["capacity"]
    )
    assert row["slots"] * row["batch"] == REPLAY_ROWS_DRAWN[suffix]

    if row["schedule"] == "shared":
        assert cfg.inner_component_update_schedule is False
        assert cfg.inner_updates_per_round == row["g"]
        assert cfg.inner_critic_updates_per_round is None
        assert cfg.inner_actor_updates_per_round is None
        assert cfg.inner_nominal_updates_per_round == row["g"]
    else:
        assert cfg.inner_component_update_schedule is True
        assert cfg.inner_updates_per_round is None
        assert (
            cfg.inner_critic_updates_per_round == row["critic_per_round"]
        )
        assert cfg.inner_actor_updates_per_round == row["actor_per_round"]
        assert cfg.inner_nominal_updates_per_round == (
            row["critic_per_round"] + row["actor_per_round"]
        )

    assert cfg.inner_temperature_mode == "auto"
    assert cfg.ent_coef_lr == cfg.inner_temperature_lr == pytest.approx(3e-4)
    assert cfg.tau == cfg.inner_critic_target_tau == pytest.approx(0.01)
    assert cfg.inner_actor_target_tau == pytest.approx(0.01)
    assert cfg.eval_freq is None
    assert cfg.eval_inner_comparison is False
    assert cfg.eval_value is False
    assert cfg.value_equivalence_diagnostics is False


@pytest.mark.parametrize(("suffix", "row"), VARIANTS.items())
def test_variant_manifest_and_wandb_identity_are_unique(suffix, row):
    stem = _stem(suffix)
    base_manifest = _load(EXPERIMENT_ROOT / f"{BASE}.json")
    manifest = _load(EXPERIMENT_ROOT / f"{stem}.json")
    algorithm = _load(ALGORITHM_ROOT / f"{stem}.json")
    params = algorithm["alg_params"]
    expected_manifest = copy.deepcopy(base_manifest)
    expected_manifest.update(
        {
            "study_type": (
                "single_seed_exploratory_inner_loop_update_schedule"
            ),
            "study_note": manifest["study_note"],
            "configs": [stem],
        }
    )

    assert manifest == expected_manifest
    assert manifest["checkpoint_every"] is None
    assert manifest["save_trials"] == manifest["save_strat"] == "none"
    assert "checkpoint_best_window" not in manifest
    assert row["id"] in manifest["study_note"]
    assert row["role"] in manifest["study_note"]

    run_id = row["id"].lower()
    assert params["wandb_run_name"] == (
        f"AMBITDMPC2-humanoid-walk-base-v2-{run_id}-seed55"
    )
    tags = params["wandb_tags"]
    assert len(tags) == len(set(tags))
    assert {
        "base-v2",
        "training-focused",
        "inner-loop-update-sweep",
        run_id,
        f"j{row['j']}",
        f"n{row['n']}",
        f"batch{row['batch']}",
        f"critic-updates-{row['critic_total']}",
        f"actor-updates-{row['actor_total']}",
        f"alpha-updates-{row['alpha_total']}",
        f"update-slots-{row['slots']}",
        f"capacity-{row['capacity']}",
        "eval-off",
        "no-model-checkpoints",
        "single-seed",
        "seed55",
    } <= set(tags)
    assert not ({"n32", "fixed-update-slots-8"} & set(tags))


def test_all_variant_wandb_run_names_are_distinct():
    run_names = [
        _load(ALGORITHM_ROOT / f"{_stem(suffix)}.json")["alg_params"][
            "wandb_run_name"
        ]
        for suffix in VARIANTS
    ]

    assert len(run_names) == len(set(run_names)) == 10


def test_d512_3_is_phased_while_d512_4_uses_shared_joint_batches():
    phased = _load(
        ALGORITHM_ROOT / f"{_stem('d512_3')}.json"
    )["alg_params"]
    shared = _load(
        ALGORITHM_ROOT / f"{_stem('d512_4')}.json"
    )["alg_params"]

    assert "inner_updates_per_round" not in phased
    assert phased["inner_critic_updates_per_round"] == 3
    assert phased["inner_actor_updates_per_round"] == 3
    assert shared["inner_updates_per_round"] == 3
    assert "inner_critic_updates_per_round" not in shared
    assert "inner_actor_updates_per_round" not in shared
