import json
import stat
import subprocess
from pathlib import Path

from utils.checkpointing import CheckpointTracker


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "slurm/run_ambixqc_cuda_gate_hydra.sbatch"
SMOKE_ALGORITHM = (
    ROOT / "configs/dmcontrol/algs/ambixqc_humanoid_walk_state_smoke.json"
)
SMOKE_MANIFEST = (
    ROOT / "configs/dmcontrol/experiments/ambixqc_humanoid_walk_state_smoke.json"
)


def _contents():
    return LAUNCHER.read_text(encoding="utf-8")


def test_gate_requests_one_unpinned_two_hour_gpu_allocation():
    contents = _contents()

    assert "#SBATCH --partition=gpus" in contents
    assert "#SBATCH --gres=gpu:1" in contents
    assert "#SBATCH --cpus-per-task=8" in contents
    assert "#SBATCH --mem=32G" in contents
    assert "#SBATCH --time=02:00:00" in contents
    assert "#SBATCH --nodes=1" in contents
    assert "#SBATCH --ntasks=1" in contents
    assert "#SBATCH --nodelist" not in contents
    assert "#SBATCH -w" not in contents


def test_gate_rejects_wrong_or_dirty_source_and_wrong_dependency_lock():
    contents = _contents()

    assert "EXPECTED_ACTION_MODES_SHA" in contents
    assert "^[0-9a-f]{40}$" in contents
    assert "status --porcelain=v1 --untracked-files=all" in contents
    assert contents.count(
        'require_clean_sha "$PROJECT_DIR_REAL" "$EXPECTED_ACTION_MODES_SHA"'
    ) == 2
    assert "f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6" in contents
    assert 'require_lock_hash "$PROJECT_DIR_REAL/environments/dmcontrol/uv.lock"' in contents
    assert 'require_lock_hash "$ENV_PROJECT_REAL/uv.lock"' in contents


def test_gate_keeps_generated_state_outside_source_and_disables_wandb():
    contents = _contents()

    assert "Gate artifacts must be outside the Git checkout" in contents
    assert 'readonly JOB_SCRATCH="$SCRATCH_BASE/ambixqc-cuda-gate-$SLURM_JOB_ID"' in contents
    assert 'exec > >(tee "$RUN_ROOT/gate.log") 2>&1' in contents
    assert 'export XDG_CACHE_HOME="$JOB_SCRATCH/cache"' in contents
    assert 'export TORCH_HOME="$JOB_SCRATCH/cache/torch"' in contents
    assert "export WANDB_MODE=disabled" in contents
    assert "export WANDB_DISABLED=true" in contents
    assert 'printf \'PASS\\n\' > "$RUN_ROOT/PASS"' in contents
    assert "rm -rf" not in contents


def test_gate_runs_cuda_regression_focused_suites_and_humanoid_smoke():
    contents = _contents()

    assert "torch.cuda.is_available()" in contents
    assert "torch.get_default_dtype() == torch.float32" in contents
    assert "not torch.is_autocast_enabled()" in contents
    assert "task=\"humanoid-walk\"" in contents
    assert "test_cuda_workspace_projects_live_weights_and_moves_the_live_target" in contents
    for path in (
        "tests/test_ambixqc_core.py",
        "tests/test_ambixqc_inner.py",
        "tests/test_ambixqc_wrapper.py",
        "tests/test_xqc_correctness.py",
        "tests/test_xqc_integration.py",
        "tests/test_tdmpc2_correctness.py",
        "tests/test_tdmpc2_checkpoint.py",
        "tests/test_tdmpc2_training_state.py",
    ):
        assert path in contents
    assert "AMBI_XQC_TEST_DEVICE=cuda AMBI_XQC_REQUIRE_CUDA=1" in contents
    assert "ambixqc_humanoid_walk_state_smoke.json" in contents
    assert "--num-runs 1" in contents


def test_gate_validates_final_checkpoint_semantics_and_numerics():
    contents = _contents()

    assert 'AMBIXQC_VALIDATE_SMOKE_ROOT="$SMOKE_ROOT"' in contents
    assert 'AMBIXQC_VALIDATE_OFFICIAL_XQC_SHA="$OFFICIAL_XQC_SHA"' in contents
    assert '\nSMOKE_ROOT="$SMOKE_ROOT"' not in contents
    assert 'checkpoint.get("step") != 502' in contents
    assert 'trial.get("alg") != "AMBIXQC/AMBIXQC"' in contents
    assert "expected_counters = (2, 2, 2, 1, 1, 1)" in contents
    assert 'reward_state.get("count") != 502.0' in contents
    assert "assert_finite(state)" in contents
    assert "len(projected) != 16" in contents
    assert "residual > 1e-6" in contents
    assert "xqc_controller.log_temperature" in contents


def test_smoke_final_latest_alias_represents_step_502_despite_periodic_step_500():
    algorithm = json.loads(SMOKE_ALGORITHM.read_text(encoding="utf-8"))
    manifest = json.loads(SMOKE_MANIFEST.read_text(encoding="utf-8"))
    assert algorithm["total_steps"] == 502
    assert manifest["checkpoint_every"] == 500
    assert manifest["save_strat"] == ["latest"]

    tracker = CheckpointTracker(
        manifest["checkpoint_every"],
        "/tmp/not-written-by-this-test",
        "ambixqc-smoke",
        save_strat=manifest["save_strat"],
    )
    periodic = tracker.targets(500)
    final = tracker.targets(502, final=True)
    assert [(target.kind, target.metadata["checkpoint"]["step"]) for target in periodic] == [
        ("latest", 500)
    ]
    assert [(target.kind, target.metadata["checkpoint"]["step"]) for target in final] == [
        ("latest", 502)
    ]


def test_gate_launcher_is_executable_and_has_valid_bash_syntax():
    assert LAUNCHER.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
