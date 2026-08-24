import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "slurm/run_xqc_humanoid_walk_1m_hydra.sbatch"
SUBMITTER_PATH = ROOT / "slurm/submit_xqc_humanoid_walk_1m_hydra.sh"


def test_launcher_is_one_unpinned_long_hydra_gpu_job():
    contents = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "#SBATCH --partition=gpus" in contents
    assert "#SBATCH --gres=gpu:1" in contents
    assert "#SBATCH --cpus-per-task=8" in contents
    assert "#SBATCH --mem=32G" in contents
    assert "#SBATCH --time=30-00:00:00" in contents
    assert "#SBATCH --nodes=1" in contents
    assert "#SBATCH --ntasks=1" in contents
    assert "#SBATCH --array" not in contents
    assert "#SBATCH --nodelist" not in contents
    assert "#SBATCH -w" not in contents

    assert ": \"${XQC_BASELINE_PYTHON:?" in contents
    assert "XQC_BASELINE_PYTHON must be an executable at an absolute path" in contents
    assert "environments/dmcontrol/.venv/bin/python" not in contents
    assert "uv run" not in contents
    assert "conda activate" not in contents
    assert "export MUJOCO_GL=egl" in contents
    assert "export WANDB_MODE=online" in contents
    assert "export WANDB_ENTITY=rwgao_b-brown-university" in contents
    assert "export WANDB_PROJECT=ambi" in contents
    assert "export WANDB_RUN_GROUP=ambixqc-humanoid-walk-state-1m" in contents
    assert 'export XQC_EVAL_CSV="$EVAL_CSV"' in contents
    assert "xqc_humanoid_walk_state_1m.json" in contents
    assert '--trial-index 0' in contents
    assert "--num-runs 1" in contents
    assert "Agent decisions: 1000000" in contents
    assert "Raw control frames: 2000000" in contents


def test_launcher_guards_source_locks_artifacts_and_wandb_identity():
    contents = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "EXPECTED_ACTION_MODES_SHA" in contents
    assert "^[0-9a-f]{40}$" in contents
    assert "status --porcelain=v1 --untracked-files=all" in contents
    assert contents.count(
        'require_clean_sha "$PROJECT_DIR_REAL" "$EXPECTED_ACTION_MODES_SHA"'
    ) == 2
    lock_hash = "f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6"
    assert lock_hash in contents
    assert 'require_lock_hash "$PROJECT_DIR_REAL/environments/dmcontrol/uv.lock"' in contents
    assert 'require_lock_hash "$ENV_PROJECT_REAL/uv.lock"' in contents
    assert "Durable results must be outside the Git checkout" in contents
    assert 'if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]' in contents
    assert 'if [[ -e "$JOB_SCRATCH" || -L "$JOB_SCRATCH" ]]' in contents
    assert 'exec > >(tee "$RUN_ROOT/job.log") 2>&1' in contents
    assert "rm -rf" not in contents

    for cache in (
        "XDG_CACHE_HOME",
        "TORCH_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "MPLCONFIGDIR",
        "NUMBA_CACHE_DIR",
        "TMPDIR",
        "WANDB_DIR",
        "WANDB_CACHE_DIR",
        "WANDB_DATA_DIR",
        "WANDB_ARTIFACT_DIR",
    ):
        assert f"export {cache}=" in contents
    assert 'export WANDB_RUN_ID="$WANDB_UNIQUE_ID"' in contents
    assert "export WANDB_RESUME=never" in contents
    assert "unset XQC_COMPARISON_ID" in contents
    assert 'config["alg_params"]["wandb_run_name"] = run_name' in contents
    assert "job$SLURM_JOB_ID" in contents
    assert '--alg-dir "$SCRATCH_ALG_DIR"' in contents


def test_launcher_validates_cadence_updates_finiteness_and_projection():
    contents = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert 'expected_steps = [1, *range(50_000, 1_000_001, 50_000)]' in contents
    assert "len(rows) != 21" in contents
    assert 'int(row["seed"]) != 55' in contents
    assert "math.isfinite" in contents
    assert "expected_updates = 2 * (1_000_000 - 5_000)" in contents
    assert "expected_delayed = (expected_updates - 1) // 3 + 1" in contents
    assert 'normalizer_count != 1_000_000.0' in contents
    assert 'require_finite(agent, "agent")' in contents
    assert 'require_finite(state.get("metrics", {}), "metrics")' in contents
    assert 'optimizer_steps(agent.get("critic_optimizer", {}), expected_updates' in contents
    assert 'len(target_means) != 10 or len(target_vars) != 10' in contents
    assert "len(projected) != 16" in contents
    assert "maximum_residual > 1e-6" in contents
    assert 'root.rglob("*_latest.pt")' in contents
    assert 'root.rglob("*_best.pt")' in contents
    assert 'checkpoint_metadata.get("step") != 1_000_000' in contents

    assert 'sys.version_info[:2] == (3,10)' in contents
    assert 'task="humanoid-walk",obs="state"' in contents
    assert 'obs.shape == (67,)' in contents
    assert 'env.action_space.shape == (21,)' in contents
    assert 'env.unwrapped.action_repeat == 2' in contents
    assert "finally:" in contents and "env.close()" in contents


def test_submitter_prefers_gpu2301_then_gpu2201_and_submits_one_job():
    contents = SUBMITTER_PATH.read_text(encoding="utf-8")

    assert 'readonly PREFERRED_NODE="gpu2301"' in contents
    assert 'readonly FALLBACK_NODE="gpu2201"' in contents
    assert 'scontrol show node -o "$node"' in contents
    assert "preferred_slots >= 1" in contents
    assert "fallback_slots >= 1" in contents
    assert 'readonly SELECTED_NODE="$PREFERRED_NODE"' in contents
    assert 'readonly SELECTED_NODE="$FALLBACK_NODE"' in contents
    assert '--nodelist="$SELECTED_NODE"' in contents
    assert "sbatch" in contents
    assert "--parsable" in contents
    assert '--export="$export_spec"' in contents
    assert "EXPECTED_ACTION_MODES_SHA=$EXPECTED_ACTION_MODES_SHA" in contents
    assert "XQC_BASELINE_RESULTS_ROOT=$XQC_BASELINE_RESULTS_ROOT" in contents
    assert "XQC_BASELINE_RUN_ID=$XQC_BASELINE_RUN_ID" in contents
    assert "XQC_BASELINE_PYTHON=$XQC_BASELINE_PYTHON" in contents
    assert "--python)" in contents
    assert "--python is required" in contents
    assert "--dry-run" in contents
    assert "scancel" not in contents

    first_sbatch = contents.index("  sbatch\n")
    assert contents.index('require_clean_sha "$PROJECT_DIR"') < first_sbatch
    assert contents.index('if [[ -e "$STUDY_ROOT"') < first_sbatch


def test_launcher_and_submitter_are_executable_valid_bash():
    for path in (LAUNCHER_PATH, SUBMITTER_PATH):
        assert path.stat().st_mode & stat.S_IXUSR
        subprocess.run(["bash", "-n", str(path)], check=True)
