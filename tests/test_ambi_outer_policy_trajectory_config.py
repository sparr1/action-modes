import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_ROOT = ROOT / "configs/dmcontrol/experiments"
BASE = "ambi_humanoid_walk_base_min_all_reward_only_value_calibration"
OUTER_50 = f"{BASE}_outer_policy_50"
MANIFEST = "ambi_humanoid_walk_value_calibration_outer_policy_ablation"
INTERVENTION_TAGS = [
    "outer-policy-trajectory-ablation",
    "50pct-outer-policy-trajectories",
]
STUDY_NOTE = (
    "Paired three-seed, one-million-decision Humanoid Walk value-calibration "
    "campaign comparing the unchanged AMBI baseline with a 50% outer-policy-"
    "trajectory intervention. In the intervention, one Bernoulli draw with "
    "probability 0.5 is made at each eligible fully post-seed episode "
    "start/reset and selects the collector for that entire episode/trajectory; "
    "the first episode that crosses the seed-collection boundary stays AMBI "
    "for its partial post-seed remainder, and the intervention is not a per-"
    "decision policy mixture. Both conditions retain the same reward-only "
    "five-head min_all recipe, evaluation seed and protocols, 50,000-decision "
    "evaluation/checkpoint grid, and seeds 55--57. The paper states that 50% "
    "of trajectories use the nominal policy, but its public repository does "
    "not release the collection patch, so this is an explicit operationalization "
    "rather than a byte-for-byte reproduction."
)


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


def test_outer_policy_50_config_is_an_exact_one_factor_derivative():
    baseline = _load(ALGORITHM_ROOT / f"{BASE}.json")
    actual = _load(ALGORITHM_ROOT / f"{OUTER_50}.json")
    expected = copy.deepcopy(baseline)
    expected["alg_params"].update(
        {
            "outer_policy_episode_probability": 0.5,
            "wandb_tags": [
                *baseline["alg_params"]["wandb_tags"],
                *INTERVENTION_TAGS,
            ],
        }
    )

    assert actual == expected
    assert "outer_policy_episode_probability" not in baseline["alg_params"]
    assert actual["alg_params"]["outer_policy_episode_probability"] == 0.5
    assert "wandb_run_name" not in actual["alg_params"]


def test_paired_manifest_changes_only_campaign_identity_and_config_pair():
    baseline = _load(EXPERIMENT_ROOT / f"{BASE}.json")
    actual = _load(EXPERIMENT_ROOT / f"{MANIFEST}.json")
    expected = copy.deepcopy(baseline)
    expected.update(
        {
            "study_type": "paired_three_seed_outer_policy_trajectory_ablation",
            "study_note": STUDY_NOTE,
            "configs": [BASE, OUTER_50],
        }
    )

    assert actual == expected
    assert actual["trials"] == 3
    assert actual["overrides_alg"] == {
        "seed": 55,
        "device": "cuda",
        "env": "DMControl-v0",
        "total_steps": 1_000_000,
        "episodes": None,
    }
    assert actual["checkpoint_every"] == 50_000
    assert actual["save_strat"] == ["all", "best", "latest"]
    assert actual["logs"] == "timestamp"
    assert actual["save_trials"] == "none"
    assert actual["log_type"] == "summary"


def test_paired_manifest_resolves_the_same_three_seeds_for_both_conditions():
    manifest = _load(EXPERIMENT_ROOT / f"{MANIFEST}.json")
    algorithms = {
        name: _load(ALGORITHM_ROOT / f"{name}.json")
        for name in manifest["configs"]
    }

    assert list(algorithms) == [BASE, OUTER_50]
    assert {
        name: [algorithm["seed"] + trial for trial in range(manifest["trials"])]
        for name, algorithm in algorithms.items()
    } == {
        BASE: [55, 56, 57],
        OUTER_50: [55, 56, 57],
    }
    assert all(
        algorithm["total_steps"] == 1_000_000
        for algorithm in algorithms.values()
    )
    assert all(
        algorithm["alg_params"]["eval_freq"] == 50_000
        for algorithm in algorithms.values()
    )
    assert all(
        algorithm["alg_params"]["eval_value_seed"] == 12_345
        for algorithm in algorithms.values()
    )
