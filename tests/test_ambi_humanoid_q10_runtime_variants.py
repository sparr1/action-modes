import copy
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_ROOT = ROOT / "configs/dmcontrol/experiments"
BASE = "ambi_humanoid_walk_base_min_all_q10"
STUDY_TYPE = "single_seed_exploratory_ten_head_minimum_runtime_variant"
VARIANTS = {
    f"{BASE}_j2_n32_g6": {
        "inner_updates_per_round": 6,
    },
    f"{BASE}_j2_n64_g4": {
        "inner_rollouts_per_round": 64,
        "inner_replay_capacity": 384,
    },
    f"{BASE}_j4_n32_g4": {
        "inner_rounds": 4,
        "inner_replay_capacity": 384,
    },
}
SCHEDULES = {
    f"{BASE}_j2_n32_g6": (2, 32, 6),
    f"{BASE}_j2_n64_g4": (2, 64, 4),
    f"{BASE}_j4_n32_g4": (4, 32, 4),
}


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def test_q10_runtime_variants_change_only_the_selected_schedule_and_identity():
    baseline = _load(ALGORITHM_ROOT / f"{BASE}.json")
    baseline_params = baseline["alg_params"]
    run_names = {baseline_params["wandb_run_name"]}

    for name, runtime_changes in VARIANTS.items():
        actual = _load(ALGORITHM_ROOT / f"{name}.json")
        expected = copy.deepcopy(baseline)
        expected["alg_params"].update(runtime_changes)
        expected["alg_params"].update(
            {
                "wandb_run_name": actual["alg_params"]["wandb_run_name"],
                "wandb_tags": actual["alg_params"]["wandb_tags"],
            }
        )

        # Full equality makes the selected runtime fields and W&B identity the
        # only permitted differences from the ten-head min_all base.
        assert actual == expected

        params = actual["alg_params"]
        rounds, rollouts, updates = SCHEDULES[name]
        assert (
            params["inner_rounds"],
            params["inner_rollouts_per_round"],
            params["inner_updates_per_round"],
            params["inner_replay_capacity"],
        ) == (rounds, rollouts, updates, rounds * rollouts * 3)
        assert params["inner_rollout_horizon"] == 3
        assert params["num_q"] == 10
        assert {
            params["outer_q_target_reduction"],
            params["outer_q_actor_reduction"],
            params["inner_q_target_reduction"],
            params["inner_q_actor_reduction"],
        } == {"min_all"}

        run_name = params["wandb_run_name"]
        tags = set(params["wandb_tags"])
        schedule_slug = f"j{rounds}-n{rollouts}-g{updates}"
        schedule_tags = {
            tag for tag in tags if re.fullmatch(r"[jng]\d+", tag)
        }
        assert run_name == (
            "AMBITDMPC2-humanoid-walk-base-v1-"
            f"{schedule_slug}-min-all-q10-seed55"
        )
        assert len(params["wandb_tags"]) == len(tags)
        assert schedule_tags == {
            f"j{rounds}",
            f"n{rollouts}",
            f"g{updates}",
        }
        assert {"runtime-variant", "q-min-all", "q-heads-10"} <= tags
        assert run_name not in run_names
        run_names.add(run_name)


def test_q10_runtime_variant_manifests_are_isolated_exploratory_cells():
    baseline = _load(EXPERIMENT_ROOT / f"{BASE}.json")

    for name, (rounds, rollouts, updates) in SCHEDULES.items():
        actual = _load(EXPERIMENT_ROOT / f"{name}.json")
        expected = copy.deepcopy(baseline)
        expected.update(
            {
                "study_type": actual["study_type"],
                "study_note": actual["study_note"],
                "configs": [name],
            }
        )

        # A variant manifest may change only its study identity and exact
        # one-cell algorithm selection relative to the q10 base protocol.
        assert actual == expected
        assert actual["configs"] == [name]
        assert actual["trials"] == 1
        assert actual["overrides_alg"]["seed"] == 55
        assert actual["overrides_alg"]["total_steps"] == 14_000_000

        study_type = actual["study_type"]
        note = actual["study_note"].lower()
        assert study_type == STUDY_TYPE
        assert "single-seed exploratory" in note
        assert "14-million-decision" in note
        assert f"j={rounds}" in note
        assert f"n={rollouts}" in note
        assert "h=3" in note
        assert f"g={updates}" in note
        assert "no confidence" in note
        assert "significance" in note
        assert "confirmatory claims" in note
