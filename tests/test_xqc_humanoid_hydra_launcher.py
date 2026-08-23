import json
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm/run_xqc_humanoid_walk_hydra.sbatch"
SUBMITTER = ROOT / "slurm/submit_xqc_humanoid_walk_hydra.sh"
ACTION_CONFIG = ROOT / "configs/dmcontrol/algs/xqc_humanoid_walk_state.json"
WALKER_CONFIG = ROOT / "configs/dmcontrol/algs/xqc_walker_walk_state.json"


def _launcher_contents():
    return LAUNCHER.read_text(encoding="utf-8")


def _submitter_contents():
    return SUBMITTER.read_text(encoding="utf-8")


def test_launcher_has_large_but_bounded_hydra_resources_and_no_node_pin():
    contents = _launcher_contents()

    assert "#SBATCH --partition=gpus" in contents
    assert "#SBATCH --gres=gpu:1" in contents
    assert "#SBATCH --cpus-per-task=8" in contents
    assert "#SBATCH --mem=32G" in contents
    assert "#SBATCH --time=30-00:00:00" in contents
    assert "#SBATCH --nodes=1" in contents
    assert "#SBATCH --ntasks=1" in contents
    assert "#SBATCH --nodelist" not in contents
    assert "#SBATCH -w" not in contents


def test_launcher_runs_exactly_one_requested_implementation_and_seed():
    contents = _launcher_contents()

    assert 'case "$IMPLEMENTATION" in' in contents
    assert 'case "$SEED" in' in contents
    assert "IMPLEMENTATION must be exactly 'official' or 'action'" in contents
    assert "SEED must be exactly 0 or 1" in contents
    assert '--base-seed "$SEED"' in contents
    assert "--num-seeds 1" in contents
    assert 'seed="$SEED"' in contents
    assert "num_seeds=1" in contents
    assert '--trial-index "$SEED"' in contents
    assert "--num-runs 1" in contents
    assert "for trial_index in 0 1" not in contents
    assert "num_seeds=2" not in contents


def test_launcher_preserves_the_paper_horizon_and_per_seed_acceptance_checks():
    contents = _launcher_contents()

    assert "env=humanoid-walk" in contents
    assert "max_steps=1000000" in contents
    assert "start_training=5000" in contents
    assert "updates_per_step=2" in contents
    assert "n_steps=1" in contents
    assert "eval_interval=50000" in contents
    assert "eval_episodes=10" in contents
    assert "log_interval_condition_number=null" in contents
    assert "--expected-updates 990000" in contents
    assert "--max-projection-residual 1e-6" in contents
    assert "--expected-evaluation-rows 11" in contents
    assert "len(rows) != 11" in contents
    assert "range(50_000, 500_001, 50_000)" in contents
    assert "range(100_000, 1_000_001, 100_000)" in contents
    assert 'validate_official_csv "$RUN_ROOT/evaluations.csv" "$SEED"' in contents
    assert 'validate_action_csv "$XQC_EVAL_CSV" "$SEED"' in contents


def test_launcher_pins_clean_sources_and_both_dependency_locks():
    contents = _launcher_contents()

    assert "9a6832bb742ef01bbe9f1e06153a9338e612dae5" in contents
    assert "EXPECTED_ACTION_MODES_SHA" in contents
    assert "status --porcelain=v1 --untracked-files=all" in contents
    assert contents.count('require_clean_sha "$PROJECT_DIR_REAL"') == 2
    assert contents.count('require_clean_sha "$OFFICIAL_DIR_REAL"') == 2
    assert "f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6" in contents
    assert "bda38deffad85326e41382b44e06f2a2fc21396210a3232e17800bbaabf7bf85" in contents
    assert 'srun --ntasks=1 "$OFFICIAL_PYTHON"' in contents
    assert 'srun --ntasks=1 "$ACTION_PYTHON"' in contents


def test_launcher_uses_canonical_method_groups_names_tags_and_metadata():
    contents = _launcher_contents()

    assert 'readonly WANDB_PROJECT_NAME="ambi_humanoid"' in contents
    assert 'readonly WANDB_ENTITY_NAME="rwgao_b-brown-university"' in contents
    assert 'readonly METHOD_GROUP="$XQC_COMPARISON_ID-$IMPLEMENTATION_LABEL"' in contents
    assert 'export WANDB_RUN_GROUP="$METHOD_GROUP"' in contents
    assert 'readonly RUN_NAME="xqc-$IMPLEMENTATION_LABEL-humanoid-walk-seed$SEED"' in contents
    assert 'export WANDB_NAME="$RUN_NAME"' in contents
    assert 'export WANDB_JOB_TYPE="$IMPLEMENTATION_LABEL"' in contents
    assert "comparison-$XQC_COMPARISON_ID" in contents
    assert "implementation-$IMPLEMENTATION_LABEL" in contents
    assert "seed-$SEED" in contents
    assert 'export XQC_IMPLEMENTATION="$IMPLEMENTATION_LABEL"' in contents
    assert 'export XQC_TASK="humanoid-walk"' in contents
    assert 'export XQC_SOURCE_SHA="$SOURCE_SHA"' in contents
    assert "export XQC_ACTION_REPEAT=2" in contents
    assert "--canonical-wandb" in contents
    assert "--task humanoid-walk" in contents
    assert "--implementation official-jax" in contents
    assert '--source-sha "$OFFICIAL_XQC_SHA"' in contents


def test_action_job_explicitly_uses_the_optimized_cuda_learner_path():
    for path in (ACTION_CONFIG, WALKER_CONFIG):
        config = json.loads(path.read_text(encoding="utf-8"))
        params = config["alg_params"]

        assert params["debug_checks"] is False
        assert params["compile"] is True
        assert params["compile_strict"] is True
        assert params["optimizer_backend"] == "auto"


def test_launcher_uses_unique_durable_per_job_artifacts():
    contents = _launcher_contents()

    assert 'readonly COMPARISON_ROOT="$RESULTS_BASE_REAL/$XQC_COMPARISON_ID"' in contents
    assert 'readonly RUN_ROOT="$COMPARISON_ROOT/$IMPLEMENTATION_LABEL-seed$SEED-job$SLURM_JOB_ID"' in contents
    assert 'if [[ -e "$RUN_ROOT" ]]' in contents
    assert "Refusing to overwrite existing result directory" in contents
    assert 'if [[ -e "$JOB_SCRATCH" ]]' in contents
    assert "Refusing to reuse existing job scratch" in contents
    assert 'exec > >(tee "$RUN_ROOT/job.log") 2>&1' in contents
    assert 'export WANDB_DIR="$durable_root"' in contents
    assert 'export WANDB_CACHE_DIR="$scratch_root/cache"' in contents
    assert '--evaluation-csv "$RUN_ROOT/evaluations.csv"' in contents
    assert 'export XQC_EVAL_CSV="$RUN_ROOT/evaluations.csv"' in contents
    assert "rm -rf" not in contents


def test_submitter_submits_the_four_cells_as_independent_jobs():
    contents = _submitter_contents()

    assert "readonly JOB_COUNT=4" in contents
    assert "RUN_IMPLEMENTATIONS=(official official action action)" in contents
    assert "RUN_SEEDS=(0 1 0 1)" in contents
    assert "sbatch" in contents
    assert "--parsable" in contents
    assert '--job-name="$job_name"' in contents
    assert '--nodelist="$node"' in contents
    assert '--export="$export_spec"' in contents
    assert "IMPLEMENTATION=$implementation" in contents
    assert "SEED=$seed" in contents
    assert "XQC_COMPARISON_ID=$XQC_COMPARISON_ID" in contents
    assert "scancel" not in contents


def test_submitter_prefers_gpu2301_then_falls_back_to_gpu2201_for_four_slots():
    contents = _submitter_contents()

    assert 'readonly PREFERRED_NODE="gpu2301"' in contents
    assert 'readonly FALLBACK_NODE="gpu2201"' in contents
    assert 'scontrol show node -o "$node"' in contents
    assert "preferred_slots >= JOB_COUNT" in contents
    assert "fallback_slots >= JOB_COUNT" in contents
    assert "preferred_slots + fallback_slots >= JOB_COUNT" in contents
    assert "Need four concurrent Hydra slots" in contents


def test_submitter_fails_before_submission_on_source_or_artifact_mismatch():
    contents = _submitter_contents()
    first_sbatch = contents.index("    sbatch\n")

    assert 'require_clean_sha "$PROJECT_DIR" "$EXPECTED_ACTION_MODES_SHA"' in contents
    assert 'require_clean_sha "$XQC_OFFICIAL_DIR" "$OFFICIAL_XQC_SHA"' in contents
    assert 'require_lock_hash "$PROJECT_DIR/environments/dmcontrol/uv.lock"' in contents
    assert 'require_lock_hash "$XQC_OFFICIAL_DIR/uv.lock"' in contents
    assert 'if [[ -e "$COMPARISON_ROOT" || -L "$COMPARISON_ROOT" ]]' in contents
    assert contents.index('require_clean_sha "$PROJECT_DIR"') < first_sbatch
    assert contents.index('if [[ -e "$COMPARISON_ROOT"') < first_sbatch


def test_launcher_and_submitter_have_valid_bash_syntax():
    assert LAUNCHER.stat().st_mode & stat.S_IXUSR
    assert SUBMITTER.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    subprocess.run(["bash", "-n", str(SUBMITTER)], check=True)
