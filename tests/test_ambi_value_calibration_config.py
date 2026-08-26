import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_ROOT = ROOT / "configs/dmcontrol/experiments"
BASE = "ambi_humanoid_walk_base_min_all_reward_only"
VARIANT = f"{BASE}_value_calibration"
PROTOCOLS = ["paper_deterministic", "stochastic_bellman"]
ADDED_TAGS = [
    "figure1-value-calibration",
    "eval-value-every-50k",
    "eval-value-samples-100",
    "paper-deterministic",
    "stochastic-bellman",
    "three-seed",
]
STUDY_NOTE = (
    "Three-seed, one-million-decision Humanoid Walk value-calibration run "
    "derived from the base-v1 G4 min_all reward-only configuration. It "
    "evaluates the paper-deterministic and stochastic-Bellman protocols at "
    "step zero and every 50,000 agent decisions with 100 fixed-seed samples "
    "per estimate; the paper-compatible MC and Q batches remain independent, "
    "while the stochastic protocol pairs each sampled first action with its "
    "rollout. All five Q heads, both reward-only critic targets, the cloned "
    "inner critic, automatic outer and inner entropy coefficients, and no "
    "actor-loss percentile scaling remain unchanged. This is a "
    "value-calibration protocol study, not a 14-million-decision performance "
    "comparison."
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


def test_value_calibration_algorithm_is_an_exact_reward_only_derivative():
    baseline = _load(ALGORITHM_ROOT / f"{BASE}.json")
    actual = _load(ALGORITHM_ROOT / f"{VARIANT}.json")
    expected = copy.deepcopy(baseline)
    expected["total_steps"] = 1_000_000
    params = expected["alg_params"]
    params.update(
        {
            "eval_freq": 50_000,
            "eval_episodes": 1,
            "eval_value": True,
            "eval_value_samples": 100,
            "eval_value_seed": 12_345,
            "eval_value_protocols": PROTOCOLS,
        }
    )
    params.pop("wandb_run_name")
    params["wandb_tags"] = [
        "1m-decisions" if tag == "14m-decisions" else tag
        for tag in baseline["alg_params"]["wandb_tags"]
    ] + ADDED_TAGS

    assert actual == expected
    assert "wandb_run_name" not in actual["alg_params"]
    assert actual["alg_params"]["eval_value_protocols"] == PROTOCOLS
    assert actual["alg_params"]["wandb_tags"].count("1m-decisions") == 1
    assert "14m-decisions" not in actual["alg_params"]["wandb_tags"]


def test_value_calibration_manifest_is_the_exact_three_seed_protocol():
    baseline = _load(EXPERIMENT_ROOT / f"{BASE}.json")
    actual = _load(EXPERIMENT_ROOT / f"{VARIANT}.json")
    expected = copy.deepcopy(baseline)
    expected.update(
        {
            "study_type": "three_seed_value_calibration",
            "study_note": STUDY_NOTE,
            "overrides_alg": {
                "seed": 55,
                "device": "cuda",
                "env": "DMControl-v0",
                "total_steps": 1_000_000,
                "episodes": None,
            },
            "trials": 3,
            "configs": [VARIANT],
            "checkpoint_every": 50_000,
            "save_strat": ["all", "best", "latest"],
        }
    )

    assert actual == expected
    assert [
        actual["overrides_alg"]["seed"] + trial
        for trial in range(actual["trials"])
    ] == [55, 56, 57]
    assert actual["logs"] == "timestamp"
    assert actual["save_trials"] == "none"
    assert actual["log_info"] is False
    assert actual["log_type"] == "summary"
    assert actual["checkpoint_every"] == 50_000
    assert actual["save_strat"] == ["all", "best", "latest"]


def test_value_calibration_budget_and_cadence_resolve_without_shallow_overrides():
    algorithm = _load(ALGORITHM_ROOT / f"{VARIANT}.json")
    manifest = _load(EXPERIMENT_ROOT / f"{VARIANT}.json")
    resolved = {**algorithm, **manifest["overrides_alg"]}

    assert "alg_params" not in manifest["overrides_alg"]
    assert resolved["total_steps"] == 1_000_000
    assert algorithm["alg_params"]["eval_freq"] == 50_000
    assert resolved["total_steps"] % algorithm["alg_params"]["eval_freq"] == 0
    assert manifest["checkpoint_every"] == algorithm["alg_params"]["eval_freq"]
