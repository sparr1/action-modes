from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm/run_tdmpc2_mppi_eval_50k_150k_hydra.sbatch"


def test_newer_checkpoint_launcher_maps_five_steps_and_both_training_horizons():
    contents = LAUNCHER.read_text()

    assert "#SBATCH --array=0-8%9" in contents
    assert "0|1) STEP=50000" in contents
    assert "2|3) STEP=75000" in contents
    assert "4|5) STEP=100000" in contents
    assert "6|7) STEP=125000" in contents
    assert "8|9) STEP=150000" in contents
    assert "TASK_ID % 2 == 0" in contents
    assert "seed_1/job_34489" in contents
    assert "seed_1/job_34490" in contents
    assert "model:tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m_0_${STEP}" in contents
    assert (
        "model:tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m_train_h5_0_${STEP}"
        in contents
    )
    assert 'step_${STEP}' in contents


def test_newer_checkpoint_launcher_has_guarded_paired_then_same_state_protocol():
    contents = LAUNCHER.read_text()

    assert "git status --porcelain --untracked-files=normal" in contents
    assert '[[ ! -s "$CHECKPOINT" || ! -s "$CHECKPOINT.metadata.json" ]]' in contents
    assert '[[ -e "$PAIRED_OUTPUT" || -e "$ACTION_OUTPUT" ]]' in contents
    assert "--overwrite" not in contents

    paired_call = contents.index("evaluate_tdmpc2_mppi_checkpoint.py")
    action_call = contents.index("evaluate_tdmpc2_mppi_action_mc.py")
    assert paired_call < action_call
    assert '--behavior-json "$PAIRED_OUTPUT"' in contents
    assert contents.count("--episodes 12") == 2
    assert contents.count("--seed 101") == 2
    assert contents.count("--controller-seed 12345") == 2
    assert contents.count("--bootstrap-samples 20000") == 2
    assert contents.count("--device cuda") == 2
    assert "--block-size 25" in contents
    assert "--action-draws 1" in contents


def test_newer_checkpoint_launcher_uses_all_requested_hydra_gpu_nodes():
    contents = LAUNCHER.read_text()

    assert "#SBATCH --nodelist=gpu2501,gpu2301,gpu2201" in contents
    assert "#SBATCH --constraint" not in contents
    assert "#SBATCH --gres=gpu:1" in contents
    assert "#SBATCH --cpus-per-task=8" in contents
    assert "#SBATCH --mem=32G" in contents
    assert "#SBATCH --time=02:00:00" in contents
