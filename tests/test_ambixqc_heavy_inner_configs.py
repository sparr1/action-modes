import copy
import json
from pathlib import Path

import gymnasium as gym

from RL.AMBIXQC import AMBIXQC
from RL.xqc_core import OFFICIAL_XQC_COMMIT


ROOT = Path(__file__).resolve().parents[1]
ALG_DIR = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_DIR = ROOT / "configs/dmcontrol/experiments"
PRODUCTION = ALG_DIR / "ambixqc_humanoid_walk_state_compiled.json"
BASE_STEM = "ambixqc_humanoid_walk_heavy_inner_v1"
WANDB_GROUP = "ambixqc-humanoid-walk-heavy-inner-v1-14m"
COMMON_TAGS = [
    "ambi-xqc",
    "dmcontrol",
    "humanoid-walk",
    "state",
    "heavy-inner-v1",
    "xqc-native",
    "training-focused",
    "14m-decisions",
    "single-seed-exploratory",
    "seed55",
    "eval-off",
    "no-model-checkpoints",
    "no-exact-resume",
    "torch-compile-strict",
    "xqc-policy-delay3",
    "xqc-official-9a6832b",
    "inner-reward-normalization-action-local-imagined",
]
SCHEDULE_KEYS = (
    "inner_rounds",
    "inner_rollouts_per_round",
    "inner_updates_per_round",
    "inner_batch_size",
    "inner_replay_capacity",
)
NATIVE_PHASED_KEYS = {
    "inner_critic_updates_per_round",
    "inner_actor_updates_per_round",
    "inner_temperature_updates_per_round",
    "inner_critic_updates_per_action",
    "inner_actor_updates_per_action",
    "inner_temperature_updates_per_action",
}

# stem: (J rounds, N rollouts/round, B replay batch, G optimizer slots/round,
#        exact action-local replay capacity)
MATRIX = {
    BASE_STEM: (8, 32, 64, 1, 768),
    f"{BASE_STEM}_d256_g1_j6": (6, 256, 256, 1, 4_608),
    f"{BASE_STEM}_d256_g1": (8, 256, 256, 1, 6_144),
    f"{BASE_STEM}_d256_g3": (8, 256, 256, 3, 6_144),
    f"{BASE_STEM}_d512_g1_j6": (6, 512, 512, 1, 9_216),
    f"{BASE_STEM}_d512_g1": (8, 512, 512, 1, 12_288),
    f"{BASE_STEM}_d512_g3_j6": (6, 512, 512, 3, 9_216),
    f"{BASE_STEM}_d512_g3": (8, 512, 512, 3, 12_288),
    f"{BASE_STEM}_d512_b256_g6": (8, 512, 256, 6, 12_288),
}


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def _algorithm_config(stem):
    return _load_json(ALG_DIR / f"{stem}.json")


def _manifest(stem):
    return _load_json(EXPERIMENT_DIR / f"{stem}.json")


def _run_name(stem):
    cell = "base" if stem == BASE_STEM else stem.removeprefix(f"{BASE_STEM}_")
    return f"AMBIXQC-humanoid-walk-heavy-inner-v1-{cell.replace('_', '-')}-seed55"


def _expected_tags(schedule):
    rounds, rollouts, batch_size, updates, capacity = schedule
    horizon = 3
    slots = rounds * updates
    accepted = (slots - 1) // 3 + 1
    return [
        *COMMON_TAGS,
        f"j{rounds}",
        f"n{rollouts}",
        f"h{horizon}",
        f"g{updates}",
        f"batch{batch_size}",
        f"capacity{capacity}",
        f"critic-slots-{slots}",
        f"accepted-actor-steps-{accepted}",
        f"accepted-temperature-steps-{accepted}",
        f"rollouts-{rounds * rollouts}",
        f"model-transitions-{capacity}",
        f"replay-rows-{slots * batch_size}",
    ]


def _resolve(config):
    algorithm = object.__new__(AMBIXQC)
    algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
    algorithm.run_params = {
        "seed": config["seed"],
        "device": "cpu",
        "env": config["env"],
        "total_steps": config["total_steps"],
    }
    algorithm.experiment_params = {}
    algorithm.custom_params = copy.deepcopy(config["alg_params"])
    try:
        return algorithm._build_cfg(
            {"device": "cpu", **copy.deepcopy(config["alg_params"])}
        )
    finally:
        algorithm.env.close()


def test_heavy_inner_v1_has_exact_algorithm_and_manifest_file_sets():
    expected = {f"{stem}.json" for stem in MATRIX}
    assert {path.name for path in ALG_DIR.glob(f"{BASE_STEM}*.json")} == expected
    assert {
        path.name for path in EXPERIMENT_DIR.glob(f"{BASE_STEM}*.json")
    } == expected


def test_base_changes_only_declared_production_budget_inner_normalization_and_identity_fields():
    production = _load_json(PRODUCTION)
    base = _algorithm_config(BASE_STEM)
    normalized = copy.deepcopy(base)
    normalized["total_steps"] = production["total_steps"]
    params = normalized["alg_params"]
    production_params = production["alg_params"]
    params["eval_freq"] = production_params["eval_freq"]
    for key in SCHEDULE_KEYS:
        params[key] = production_params[key]
    assert params.pop("inner_reward_normalization") == "action_local_imagined"
    params["wandb_group"] = production_params["wandb_group"]
    params.pop("wandb_run_name")
    params["wandb_tags"] = production_params["wandb_tags"]

    assert normalized == production


def test_variants_change_only_the_xqc_schedule_and_wandb_cell_identity():
    base = _algorithm_config(BASE_STEM)
    base_params = base["alg_params"]

    for stem in MATRIX:
        if stem == BASE_STEM:
            continue
        normalized = copy.deepcopy(_algorithm_config(stem))
        params = normalized["alg_params"]
        for key in SCHEDULE_KEYS:
            params[key] = base_params[key]
        params["wandb_run_name"] = base_params["wandb_run_name"]
        params["wandb_tags"] = base_params["wandb_tags"]
        assert normalized == base, stem


def test_matrix_resolves_to_unique_strict_xqc_native_budgets():
    resolved_tuples = set()

    for stem, schedule in MATRIX.items():
        rounds, rollouts, batch_size, updates, capacity = schedule
        config = _algorithm_config(stem)
        params = config["alg_params"]
        cfg = _resolve(config)
        slots = rounds * updates
        accepted = (slots - 1) // 3 + 1

        assert config["seed"] == 55
        assert config["env"] == "DMControl-v0"
        assert config["alg"] == "AMBIXQC/AMBIXQC"
        assert config["device"] == "cuda"
        assert config["total_steps"] == 14_000_000
        assert config["episodes"] is None
        assert params["eval_freq"] is None
        assert params["inner_rollout_horizon"] == 3
        assert params["inner_replay_sampling"] == "with_replacement"
        assert params["inner_reward_normalization"] == "action_local_imagined"
        assert not (NATIVE_PHASED_KEYS & params.keys())

        assert cfg.compile is cfg.compile_strict is True
        assert cfg.inner_operator == "xqc"
        assert cfg.inner_schedule_mode == "canonical"
        assert cfg.inner_rounds == rounds
        assert cfg.inner_rollouts_per_round == rollouts
        assert cfg.inner_rollout_horizon == 3
        assert cfg.inner_updates_per_round == updates
        assert cfg.inner_batch_size == batch_size
        assert cfg.inner_model_step_budget == rounds * rollouts * 3 == capacity
        assert cfg.inner_replay_capacity == capacity
        assert cfg.inner_reward_normalization == "action_local_imagined"
        assert cfg.inner_expected_update_slots == slots
        assert cfg.inner_critic_updates_per_action == slots
        assert cfg.inner_actor_updates_per_action == accepted
        assert cfg.inner_temperature_updates_per_action == accepted
        assert f"replay-rows-{slots * batch_size}" in params["wandb_tags"]
        assert cfg.steps == cfg.xqc_lr_transition_steps == 14_000_000
        assert cfg.xqc_policy_delay == 3
        assert cfg.xqc_official_commit == OFFICIAL_XQC_COMMIT

        resolved_tuples.add(
            (
                cfg.inner_rounds,
                cfg.inner_rollouts_per_round,
                cfg.inner_batch_size,
                cfg.inner_updates_per_round,
                cfg.inner_replay_capacity,
                cfg.inner_critic_updates_per_action,
                cfg.inner_actor_updates_per_action,
                cfg.inner_temperature_updates_per_action,
            )
        )

    assert len(resolved_tuples) == len(MATRIX)


def test_wandb_identity_is_exact_truthful_and_not_native_phased_labeling():
    run_names = set()

    for stem, schedule in MATRIX.items():
        config = _algorithm_config(stem)
        params = config["alg_params"]
        run_name = _run_name(stem)

        assert params["wandb"] is True
        assert params["wandb_mode"] == "online"
        assert params["wandb_entity"] == "rwgao_b-brown-university"
        assert params["wandb_project"] == "ambi"
        assert params["wandb_group"] == WANDB_GROUP
        assert params["wandb_run_name"] == run_name
        assert params["wandb_tags"] == _expected_tags(schedule)
        assert len(params["wandb_tags"]) == len(set(params["wandb_tags"]))

        identity = " ".join([stem, run_name, *params["wandb_tags"]]).lower()
        for misleading in ("phased", "one-to-one", "1:1", "shared-joint"):
            assert misleading not in identity
        run_names.add(run_name)

    assert len(run_names) == len(MATRIX)


def test_manifests_are_exact_training_only_single_seed_cell_contracts():
    expected_keys = {
        "study_type",
        "study_note",
        "overrides_alg",
        "env_params",
        "trials",
        "configs",
        "logs",
        "save_trials",
        "checkpoint_every",
        "save_strat",
        "log_info",
        "log_type",
    }

    for stem, schedule in MATRIX.items():
        rounds, rollouts, batch_size, updates, _ = schedule
        slots = rounds * updates
        accepted = (slots - 1) // 3 + 1
        manifest = _manifest(stem)
        note = manifest["study_note"].lower()

        assert set(manifest) == expected_keys
        expected_study_type = (
            "ambixqc_humanoid_walk_heavy_inner_v1_base_single_seed_exploratory"
            if stem == BASE_STEM
            else "ambixqc_humanoid_walk_heavy_inner_v1_schedule_single_seed_exploratory"
        )
        assert manifest["study_type"] == expected_study_type
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
        assert manifest["trials"] == 1
        assert manifest["configs"] == [stem]
        assert manifest["logs"] == "timestamp"
        assert manifest["save_trials"] == "none"
        assert manifest["checkpoint_every"] is None
        assert manifest["save_strat"] == "none"
        assert manifest["log_info"] is False
        assert manifest["log_type"] == "summary"
        assert "checkpoint_best_window" not in manifest

        for phrase in (
            "xqc-native",
            "action-local imagined-return reward normalization",
            "no imagined normalization statistics write back",
            f"j={rounds}",
            f"n={rollouts}",
            "h=3",
            f"g={updates}",
            f"batch size {batch_size}",
            f"{slots} critic",
            f"{accepted} accepted actor optimizer",
            f"{accepted} accepted automatic-temperature optimizer",
            "14 million",
            "evaluation is disabled",
            "no model checkpoints",
            "exact trainer resume is unsupported",
            "single-seed",
            "non-confirmatory",
        ):
            assert phrase in note, (stem, phrase)
        assert "phased" not in note
        assert "1:1" not in note
