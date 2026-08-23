import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm/run_xqc_humanoid_walk_hydra.sbatch"


def _contents():
    return LAUNCHER.read_text(encoding="utf-8")


def test_launcher_has_large_but_bounded_hydra_resources_and_no_node_pin():
    contents = _contents()

    assert "#SBATCH --partition=gpus" in contents
    assert "#SBATCH --gres=gpu:1" in contents
    assert "#SBATCH --cpus-per-task=8" in contents
    assert "#SBATCH --mem=32G" in contents
    assert "#SBATCH --time=30-00:00:00" in contents
    assert "#SBATCH --nodes=1" in contents
    assert "#SBATCH --ntasks=1" in contents
    assert "#SBATCH --nodelist" not in contents
    assert "#SBATCH -w" not in contents


def test_launcher_selects_exactly_one_implementation_and_pins_both_sources():
    contents = _contents()

    assert 'official|action)' in contents
    assert "IMPLEMENTATION must be exactly 'official' or 'action'" in contents
    assert "9a6832bb742ef01bbe9f1e06153a9338e612dae5" in contents
    assert "EXPECTED_ACTION_MODES_SHA" in contents
    assert "status --porcelain=v1 --untracked-files=all" in contents
    assert contents.count('require_clean_sha "$PROJECT_DIR_REAL"') == 2
    assert contents.count('require_clean_sha "$OFFICIAL_DIR_REAL"') == 2
    assert "f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6" in contents
    assert "bda38deffad85326e41382b44e06f2a2fc21396210a3232e17800bbaabf7bf85" in contents
    assert 'srun --ntasks=1 "$OFFICIAL_PYTHON"' in contents
    assert 'srun --ntasks=1 "$ACTION_PYTHON"' in contents


def test_launcher_runs_the_two_seed_paper_horizon_for_each_implementation():
    contents = _contents()

    assert "env=humanoid-walk" in contents
    assert "seed=0" in contents
    assert "num_seeds=2" in contents
    assert "max_steps=1000000" in contents
    assert "start_training=5000" in contents
    assert "updates_per_step=2" in contents
    assert "n_steps=1" in contents
    assert "eval_interval=50000" in contents
    assert "eval_episodes=10" in contents
    assert "log_interval_condition_number=null" in contents
    assert "--expected-updates 990000" in contents
    assert "--max-projection-residual 1e-6" in contents
    assert "--expected-evaluation-rows 22" in contents

    assert "xqc_humanoid_walk_state.json" in contents
    assert "for trial_index in 0 1" in contents
    assert '--trial-index "$trial_index"' in contents
    assert "--num-runs 1" in contents
    assert "len(rows) != 11" in contents
    assert "len(rows) != 22" in contents
    assert "range(50_000, 500_001, 50_000)" in contents
    assert "range(100_000, 1_000_001, 100_000)" in contents


def test_launcher_groups_online_wandb_runs_and_keeps_durable_csv_fallbacks():
    contents = _contents()

    assert 'readonly WANDB_PROJECT_NAME="ambi_humanoid"' in contents
    assert 'readonly WANDB_ENTITY_NAME="rwgao_b-brown-university"' in contents
    assert "XQC_COMPARISON_GROUP" in contents
    assert 'export WANDB_RUN_GROUP="$XQC_COMPARISON_GROUP"' in contents
    assert "export WANDB_MODE=online" in contents
    assert "wandb.mode=online" in contents
    assert "implementation-official-jax" in contents
    assert "implementation-action-pytorch" in contents
    assert "WANDB_DISABLED=true" not in contents
    assert 'export WANDB_DIR="$durable_root"' in contents
    assert 'export WANDB_CACHE_DIR="$scratch_root/cache"' in contents
    assert 'hydra.job.env_set.WANDB_DIR="$RUN_ROOT/wandb"' in contents
    assert "+hydra.job.env_set.WANDB_RUN_GROUP=${oc.env:WANDB_RUN_GROUP}" in contents
    assert "+hydra.job.env_set.WANDB_TAGS=${oc.env:WANDB_TAGS}" in contents
    assert '--evaluation-csv "$RUN_ROOT/evaluations.csv"' in contents
    assert 'export XQC_EVAL_CSV="$SEED_ROOT/evaluations.csv"' in contents


def test_launcher_refuses_artifact_reuse_and_uses_unique_durable_job_roots():
    contents = _contents()

    assert 'readonly RUN_ROOT="$GROUP_ROOT/$IMPLEMENTATION-$SLURM_JOB_ID"' in contents
    assert 'if [[ -e "$RUN_ROOT" ]]' in contents
    assert "Refusing to overwrite existing result directory" in contents
    assert 'if [[ -e "$JOB_SCRATCH" ]]' in contents
    assert "Refusing to reuse existing job scratch" in contents
    assert 'exec > >(tee "$RUN_ROOT/job.log") 2>&1' in contents
    assert 'SCRATCH_BASE="${SLURM_TMPDIR:-/tmp}"' in contents
    assert "rm -rf" not in contents


def test_launcher_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
