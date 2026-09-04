import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_ALGORITHM = (
    ROOT / "configs/dmcontrol/algs/tdmpc2_humanoid_walk_state.json"
)
BANK_ALGORITHM = (
    ROOT
    / "configs/dmcontrol/algs/tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m.json"
)
BANK_MANIFEST = (
    ROOT
    / "configs/dmcontrol/experiments/tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m.json"
)
BANK_LAUNCHER = (
    ROOT
    / "slurm/run_tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m_hydra.sbatch"
)
H5_ALGORITHM = (
    ROOT
    / "configs/dmcontrol/algs/tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m_train_h5.json"
)
H5_MANIFEST = (
    ROOT
    / "configs/dmcontrol/experiments/tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m_train_h5.json"
)
H5_LAUNCHER = (
    ROOT
    / "slurm/run_tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m_train_h5_hydra.sbatch"
)


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path):
    return json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_keys)


def test_checkpoint_bank_changes_only_budget_evaluation_and_wandb_identity():
    base = _load_json(BASE_ALGORITHM)
    actual = _load_json(BANK_ALGORITHM)
    expected = copy.deepcopy(base)
    expected["total_steps"] = 1_500_000
    expected["alg_params"]["eval_freq"] = None
    expected["alg_params"]["wandb_group"] = (
        "tdmpc2-humanoid-walk-state-checkpoint-bank-1p5m"
    )
    expected["alg_params"]["wandb_tags"] = [
        "tdmpc2",
        "dmcontrol",
        "humanoid-walk",
        "state",
        "single-seed-exploratory",
        "checkpoint-bank",
        "checkpoints-every-25k",
        "no-online-evaluation",
        "train-unroll-horizon-3",
        "1p5m-decisions",
    ]

    assert actual == expected
    assert actual["alg_params"]["mpc"] is True
    assert actual["alg_params"]["eval_episodes"] == 10


def test_checkpoint_bank_manifest_retains_sixty_numbered_snapshots():
    manifest = _load_json(BANK_MANIFEST)
    study_note = manifest["study_note"].lower()

    assert manifest["study_type"] == (
        "tdmpc2_humanoid_walk_state_checkpoint_bank"
    )
    assert "separate online evaluation episodes are disabled" in study_note
    assert "official single-task model-size-5 defaults" in study_note
    assert "8bbc14ebabdb32ea7ada5c801dc525d0dc73bafe" in study_note
    assert "compile=false" in study_note
    assert "not expected to reproduce the exact seeded trajectory" in study_note
    assert "not exact training resume" in study_note
    assert manifest["overrides_alg"] == {"env": "DMControl-v0"}
    assert manifest["env_params"] == {
        "task": "humanoid-walk",
        "obs": "state",
        "render_mode": None,
    }
    assert manifest["trials"] == 1
    assert manifest["configs"] == [
        "tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m"
    ]
    assert manifest["logs"] == "none"
    assert manifest["save_trials"] == "none"
    assert manifest["checkpoint_every"] == 25_000
    assert manifest["save_strat"] == ["all"]
    assert 1_500_000 // manifest["checkpoint_every"] == 60


def test_checkpoint_bank_hydra_launcher_is_one_guarded_a6000_job():
    contents = BANK_LAUNCHER.read_text()

    assert "#SBATCH --constraint=rtx_a6000" in contents
    assert "#SBATCH --gres=gpu:1" in contents
    assert "#SBATCH --time=5-00:00:00" in contents
    assert "#SBATCH --array" not in contents
    assert "SLURM_ARRAY_" not in contents
    assert "AMBI_DMC_PYTHON" in contents
    assert (
        "/cs/home/rgao48/projects/action-modes/environments/dmcontrol/.venv/bin/python"
        in contents
    )
    assert "tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m.json" in contents
    assert "Agent decisions: 1500000" in contents
    assert "Raw simulator control steps: 3000000" in contents
    assert "Training unroll horizon: 3" in contents
    assert "Outer planning horizon: 3" in contents
    assert "Checkpoint cadence: 25000" in contents
    assert "Expected numbered checkpoints: 60" in contents
    assert "AVAILABLE_KB < 4 * 1024 * 1024" in contents
    assert "tdmpc2-checkpoint-bank-1p5m" in contents
    assert (
        "#SBATCH --output=/cs/home/rgao48/projects/ambi-runs/slurm/%x-%j.out"
        in contents
    )
    assert (
        "#SBATCH --error=/cs/home/rgao48/projects/ambi-runs/slurm/%x-%j.err"
        in contents
    )
    assert "git status --porcelain --untracked-files=normal" in contents
    assert "SOURCE_COMMIT=$(git rev-parse HEAD)" in contents
    assert "export WANDB_MODE=online" in contents
    assert "export WANDB_DISABLE_CODE=true" in contents
    assert "TDMPC2_EVAL_CSV" not in contents
    assert "--alg-index 0" in contents
    assert '--trial-index "$TRIAL_INDEX"' in contents
    assert "--num-runs 1" in contents


def test_h5_checkpoint_bank_changes_only_the_training_unroll_and_tag():
    baseline = _load_json(BANK_ALGORITHM)
    actual = _load_json(H5_ALGORITHM)
    expected = copy.deepcopy(baseline)
    expected["alg_params"]["train_unroll_horizon"] = 5
    tag_index = expected["alg_params"]["wandb_tags"].index(
        "train-unroll-horizon-3"
    )
    expected["alg_params"]["wandb_tags"][tag_index] = (
        "train-unroll-horizon-5"
    )

    assert actual == expected
    assert actual["alg_params"]["outer_planning_horizon"] == 3
    assert actual["alg_params"]["inner_rollout_horizon"] == 3
    assert actual["alg_params"]["temporal_loss_normalization"] == (
        "reference_weighted_mean"
    )
    assert actual["alg_params"]["temporal_loss_reference_horizon"] == 3


def test_h5_checkpoint_bank_manifest_discloses_the_single_axis_change():
    baseline = _load_json(BANK_MANIFEST)
    actual = _load_json(H5_MANIFEST)

    assert actual["study_type"] == (
        "tdmpc2_humanoid_walk_state_checkpoint_bank_train_h5"
    )
    assert "differs from the matched horizon-3 bank only" in actual[
        "study_note"
    ]
    assert "train_unroll_horizon=5" in actual["study_note"]
    assert "outer_planning_horizon=3" in actual["study_note"]
    assert "temporal_loss_reference_horizon=3" in actual["study_note"]
    assert "compatibility-port-specific decoupled training-horizon ablation" in (
        actual["study_note"]
    )
    assert "legacy horizon=5 would also change MPPI planning depth" in actual[
        "study_note"
    ]
    assert actual["configs"] == [
        "tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m_train_h5"
    ]
    for key in (
        "overrides_alg",
        "env_params",
        "trials",
        "logs",
        "save_trials",
        "checkpoint_every",
        "save_strat",
        "log_info",
        "log_type",
    ):
        assert actual[key] == baseline[key]


def test_h5_checkpoint_bank_hydra_launcher_is_matched_and_isolated():
    baseline = BANK_LAUNCHER.read_text()
    contents = H5_LAUNCHER.read_text()

    for contract in (
        "#SBATCH --constraint=rtx_a6000",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --time=5-00:00:00",
        "Agent decisions: 1500000",
        "Raw simulator control steps: 3000000",
        "Checkpoint cadence: 25000",
        "Expected numbered checkpoints: 60",
        "git status --porcelain --untracked-files=normal",
        "AMBI_DMC_PYTHON",
        "export WANDB_MODE=online",
        "export WANDB_DISABLE_CODE=true",
        "--alg-index 0",
        '--trial-index "$TRIAL_INDEX"',
        "--num-runs 1",
    ):
        assert contract in baseline
        assert contract in contents
    assert "#SBATCH --array" not in contents
    assert "SLURM_ARRAY_" not in contents
    assert "Training unroll horizon: 5" in contents
    assert "Outer planning horizon: 3" in contents
    assert "tdmpc2-checkpoint-bank-1p5m-train-h5" in contents
    assert "TDMPC2_EVAL_CSV" not in contents
