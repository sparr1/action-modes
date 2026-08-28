import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_ROOT = ROOT / "configs/dmcontrol/experiments"
BASE = "sac_humanoid_walk_tdmpc_table5"
VARIANT = f"{BASE}_value_calibration"
PROTOCOLS = ["paper_deterministic", "stochastic_soft_bellman"]
GROUP = "sac-humanoid-walk-state-tdmpc-table5-value-calibration-1m"
ADDED_TAGS = [
    "1m-decisions",
    "figure1-value-calibration",
    "eval-value-every-50k",
    "eval-value-samples-100",
    "paper-deterministic",
    "stochastic-soft-bellman",
    "three-seed",
]
STUDY_NOTE = (
    "Three-seed, one-million-decision Humanoid Walk value-calibration run "
    "derived from the direct-state native SAC TD-MPC Table-5 comparator. It "
    "evaluates paper_deterministic and stochastic_soft_bellman at step zero "
    "and every 50,000 agent decisions with 100 fixed-seed samples per "
    "estimate. paper_deterministic is compatibility-only and is not matched "
    "to SAC's entropy-regularized critic target; stochastic_soft_bellman is "
    "the primary target-matched protocol. Its corrected discounted timeout "
    "tail bootstraps the finite soft return at truncation and supplies the "
    "primary critic bias and RMSE metrics. All other Table-5 native SAC "
    "optimizer, network, entropy, twin-Q, and update-schedule settings remain "
    "unchanged. This is a calibration study, not a 14-million-decision "
    "performance comparison. Model checkpoints and trajectory logs are "
    "disabled."
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


def test_sac_value_calibration_is_an_exact_table5_derivative():
    baseline = _load(ALGORITHM_ROOT / f"{BASE}.json")
    actual = _load(ALGORITHM_ROOT / f"{VARIANT}.json")
    expected = copy.deepcopy(baseline)
    expected["seed"] = 55
    expected["total_steps"] = 1_000_000
    params = expected["alg_params"]
    params.update(
        {
            "buffer_size": 1_000_000,
            "eval_episodes": 1,
            "eval_value": True,
            "eval_value_samples": 100,
            "eval_value_seed": 12_345,
            "eval_value_protocols": PROTOCOLS,
            "wandb_group": GROUP,
            "wandb_tags": baseline["alg_params"]["wandb_tags"] + ADDED_TAGS,
        }
    )

    assert actual == expected
    assert baseline["seed"] == 1
    assert baseline["total_steps"] == 7_000_000
    assert baseline["alg_params"]["buffer_size"] == 7_000_000
    assert "eval_value" not in baseline["alg_params"]


def test_sac_value_calibration_manifest_is_the_exact_three_seed_campaign():
    actual = _load(EXPERIMENT_ROOT / f"{VARIANT}.json")

    assert actual == {
        "study_type": "native_sac_humanoid_walk_tdmpc_table5_value_calibration",
        "study_note": STUDY_NOTE,
        "overrides_alg": {
            "seed": 55,
            "device": "cuda",
            "env": "DMControl-v0",
            "total_steps": 1_000_000,
            "episodes": None,
        },
        "env_params": {
            "task": "humanoid-walk",
            "obs": "state",
            "render_mode": None,
        },
        "trials": 3,
        "configs": [VARIANT],
        "logs": "none",
        "save_trials": "none",
        "checkpoint_every": None,
        "save_strat": "none",
        "log_info": False,
        "log_type": "summary",
    }
    assert [
        actual["overrides_alg"]["seed"] + trial
        for trial in range(actual["trials"])
    ] == [55, 56, 57]


def test_primary_protocol_and_timeout_corrected_target_are_explicit():
    algorithm = _load(ALGORITHM_ROOT / f"{VARIANT}.json")
    manifest = _load(EXPERIMENT_ROOT / f"{VARIANT}.json")

    assert algorithm["alg_params"]["eval_value_protocols"] == PROTOCOLS
    assert "paper_deterministic is compatibility-only" in manifest["study_note"]
    assert "stochastic_soft_bellman is the primary target-matched" in manifest[
        "study_note"
    ]
    assert "corrected discounted timeout tail" in manifest["study_note"]
    assert "bias and RMSE" in manifest["study_note"]
    assert manifest["checkpoint_every"] is None
    assert manifest["logs"] == manifest["save_trials"] == "none"
