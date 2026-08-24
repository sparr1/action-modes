import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALGORITHM_ROOT = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_ROOT = ROOT / "configs/dmcontrol/experiments"
BASE = "ambi_humanoid_walk_base_min_all_q10"
VARIANT = f"{BASE}_outer_actor_mean_all"
RUN_NAME = (
    "AMBITDMPC2-humanoid-walk-base-v1-g4-min-all-q10-"
    "outer-actor-mean-all-value-equivalence-seed55"
)
STUDY_TYPE = (
    "single_seed_exploratory_ten_head_outer_actor_mean_all_"
    "with_value_equivalence_diagnostics"
)
DIAGNOSTIC_SETTINGS = {
    "value_equivalence_diagnostics": True,
    "value_equivalence_every_updates": 1000,
    "value_equivalence_mc_samples": 4,
}
EXPECTED_TAGS = [
    "ambi",
    "dmcontrol",
    "humanoid-walk",
    "state",
    "base-v1",
    "14m-decisions",
    "g4",
    "critic-lr1e-4",
    "actor-lr5e-5",
    "q-heads-10",
    "q-target-min-all",
    "outer-q-actor-mean-all",
    "inner-q-actor-min-all",
    "critic-target-entropy-augmented",
    "inner-critic-clone",
    "outer-alpha-auto",
    "inner-alpha-auto",
    "actor-loss-scale-none",
    "value-equivalence-diagnostics",
    "value-equivalence-every-1000-updates",
    "value-equivalence-mc-samples-4",
]


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


def test_q10_outer_actor_mean_all_changes_only_learning_axis_and_telemetry():
    baseline = _load(ALGORITHM_ROOT / f"{BASE}.json")
    actual = _load(ALGORITHM_ROOT / f"{VARIANT}.json")
    expected = copy.deepcopy(baseline)
    expected["alg_params"].update(
        {
            "outer_q_actor_reduction": "mean_all",
            **DIAGNOSTIC_SETTINGS,
            "wandb_run_name": RUN_NAME,
            "wandb_tags": EXPECTED_TAGS,
        }
    )

    # Full equality permits one learning-axis change, explicit observational
    # telemetry, and W&B identity changes—nothing else.
    assert actual == expected

    params = actual["alg_params"]
    assert params["num_q"] == 10
    assert params["outer_q_actor_reduction"] == "mean_all"
    assert params["outer_q_target_reduction"] == "min_all"
    assert params["inner_q_target_reduction"] == "min_all"
    assert params["inner_q_actor_reduction"] == "min_all"
    assert {
        key: params[key] for key in DIAGNOSTIC_SETTINGS
    } == DIAGNOSTIC_SETTINGS
    assert len(params["wandb_tags"]) == len(set(params["wandb_tags"]))


def test_q10_outer_actor_mean_all_manifest_is_one_exploratory_seed():
    baseline = _load(EXPERIMENT_ROOT / f"{BASE}.json")
    actual = _load(EXPERIMENT_ROOT / f"{VARIANT}.json")
    expected = copy.deepcopy(baseline)
    expected.update(
        {
            "study_type": STUDY_TYPE,
            "study_note": actual["study_note"],
            "configs": [VARIANT],
        }
    )

    assert actual == expected
    assert actual["trials"] == 1
    assert actual["overrides_alg"]["seed"] == 55
    assert actual["overrides_alg"]["total_steps"] == 14_000_000

    note = actual["study_note"].lower()
    assert "sole learning-axis change" in note
    assert "outer policy-learning q reduction from min_all to mean_all" in note
    assert "both outer and inner target reductions" in note
    assert "inner actor reduction remain min_all" in note
    assert "observational value-equivalence diagnostics" in note
    assert "every 1,000 outer updates" in note
    assert "four monte carlo samples" in note
    assert "do not alter the learning objective" in note
    assert "single-seed exploratory" in note
    assert "no confidence" in note
    assert "significance" in note
    assert "confirmatory claims" in note


def test_q10_outer_actor_mean_all_oscar_launcher_is_exact_and_resumable():
    launcher = (
        ROOT
        / "slurm/"
        "run_ambi_humanoid_walk_base_min_all_q10_outer_actor_mean_all_oscar.sbatch"
    ).read_text(encoding="utf-8")
    canonical = (ROOT / "run_ambi_oscar.sh").read_text(encoding="utf-8")

    assert "#SBATCH --partition=gpu" in launcher
    assert "#SBATCH --gres=gpu:l40s:1" in launcher
    assert "#SBATCH --time=96:00:00" in launcher
    assert "#SBATCH --cpus-per-task=6" in launcher
    assert "#SBATCH --requeue" in launcher
    assert "#SBATCH --signal=USR1@3600" in launcher
    assert "git status --porcelain --untracked-files=all" in launcher
    assert f'experiments/{VARIANT}.json"' in launcher
    assert 'export AMBI_ALG_DIR="$algorithm_dir"' in launcher
    assert 'export AMBI_PYTHON="$ambi_python"' in launcher
    assert "environments/dmcontrol/.venv/bin/python" in launcher
    assert "export WANDB_MODE=online" in launcher
    assert "export WANDB_DISABLE_CODE=true" in launcher
    assert "source_commit=\"$(git rev-parse HEAD)\"" in launcher
    assert 'exec bash "$project_dir/run_ambi_oscar.sh"' in launcher

    # Delegated canonical behavior is part of this launcher's contract.
    assert "--num-runs 1" in canonical
    assert '--resume-mode "$resume_mode"' in canonical
    assert "--resume-wandb-mode online" in canonical
    assert "scontrol requeue" in canonical
