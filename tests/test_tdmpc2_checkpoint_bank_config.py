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
        "1p5m-decisions",
    ]

    assert actual == expected
    assert actual["alg_params"]["mpc"] is True
    assert actual["alg_params"]["eval_episodes"] == 10


def test_checkpoint_bank_manifest_retains_sixty_numbered_snapshots():
    manifest = _load_json(BANK_MANIFEST)

    assert manifest["study_type"] == (
        "tdmpc2_humanoid_walk_state_checkpoint_bank"
    )
    assert "separate online evaluation episodes are disabled" in manifest[
        "study_note"
    ]
    assert "not expected to reproduce the exact seeded trajectory" in manifest[
        "study_note"
    ]
    assert "not exact training resume" in manifest["study_note"]
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
    assert "environments/dmcontrol/.venv/bin/python" in contents
    assert "tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m.json" in contents
    assert "Agent decisions: 1500000" in contents
    assert "Raw simulator control steps: 3000000" in contents
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
