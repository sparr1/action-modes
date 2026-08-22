import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_ROOT = ROOT / "configs/dmcontrol/experiments"
BASE = "ambi_humanoid_walk_base_min_all_q10"
VARIANT = f"{BASE}_inner_critic_dropout_off"
RUN_NAME = (
    "AMBITDMPC2-humanoid-walk-base-v1-g4-min-all-q10-"
    "inner-critic-dropout-off-seed55"
)
STUDY_NOTE = (
    "Single-seed exploratory 14-million-decision Humanoid Walk base-v1 G4 "
    "ten-head min_all run changing only inner critic dropout from enabled to "
    "disabled while retaining outer critic dropout at 0.01, min_all for every "
    "outer and inner actor/target reduction, entropy-augmented critic targets, "
    "a trainable cloned inner critic, automatic outer and inner entropy "
    "coefficients, and no actor-loss percentile scaling. Make no confidence, "
    "significance, or confirmatory claims."
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


def test_q10_dropout_off_config_changes_only_inner_dropout_and_run_identity():
    baseline = _load(ALGORITHM_ROOT / f"{BASE}.json")
    actual = _load(ALGORITHM_ROOT / f"{VARIANT}.json")
    expected = copy.deepcopy(baseline)
    expected["alg_params"].update(
        {
            "inner_critic_dropout_enabled": False,
            "wandb_run_name": RUN_NAME,
            "wandb_tags": [
                *baseline["alg_params"]["wandb_tags"],
                "inner-critic-dropout-off",
            ],
        }
    )

    assert actual == expected
    params = actual["alg_params"]
    assert params["dropout"] == 0.01
    assert params["num_q"] == 10
    assert params["inner_critic_adaptation"] == "clone"
    assert {
        params["outer_q_target_reduction"],
        params["outer_q_actor_reduction"],
        params["inner_q_target_reduction"],
        params["inner_q_actor_reduction"],
    } == {"min_all"}


def test_q10_dropout_off_manifest_changes_only_study_identity():
    baseline = _load(EXPERIMENT_ROOT / f"{BASE}.json")
    actual = _load(EXPERIMENT_ROOT / f"{VARIANT}.json")
    expected = copy.deepcopy(baseline)
    expected.update(
        {
            "study_type": "single_seed_exploratory_inner_critic_dropout_ablation",
            "study_note": STUDY_NOTE,
            "configs": [VARIANT],
        }
    )

    assert actual == expected
    assert actual["trials"] == 1
    assert actual["overrides_alg"]["seed"] == 55
    assert actual["overrides_alg"]["total_steps"] == 14_000_000


def test_q10_dropout_off_launcher_is_one_a6000_job_for_the_exact_manifest():
    launcher = (
        ROOT
        / "slurm/run_ambi_humanoid_walk_base_min_all_q10_inner_critic_dropout_off.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --constraint=rtx_a6000" in launcher
    assert "#SBATCH --gres=gpu:1" in launcher
    assert "#SBATCH --array" not in launcher
    assert "SLURM_ARRAY_" not in launcher
    assert f"experiments/{VARIANT}.json" in launcher
    assert "--trial-index \"$TRIAL_INDEX\"" in launcher
    assert "--num-runs 1" in launcher
