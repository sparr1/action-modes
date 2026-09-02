import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    ROOT / "slurm/run_ambixqc_humanoid_walk_heavy_inner_pair_hydra.sbatch"
)
SUBMITTER = (
    ROOT / "slurm/submit_ambixqc_humanoid_walk_heavy_inner_pair_hydra.sh"
)
README = ROOT / "README.md"
ALG_DIR = ROOT / "configs/dmcontrol/algs"
EXPERIMENT_DIR = ROOT / "configs/dmcontrol/experiments"
ALLOWLIST = {
    0: (
        "ambixqc_humanoid_walk_heavy_inner_v1_d512_g3_j6",
        (6, 512, 3, 3, 512, 9_216),
    ),
    1: (
        "ambixqc_humanoid_walk_heavy_inner_v1_d512_g3",
        (8, 512, 3, 3, 512, 12_288),
    ),
}


def _contents(path):
    return path.read_text(encoding="utf-8")


def _load_json(path):
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )


def test_launcher_is_exact_atomic_two_cell_hydra_array():
    contents = _contents(LAUNCHER)

    assert "#SBATCH --partition=gpus" in contents
    assert "#SBATCH --gres=gpu:1" in contents
    assert "#SBATCH --cpus-per-task=8" in contents
    assert "#SBATCH --mem=64G" in contents
    assert "#SBATCH --time=30-00:00:00" in contents
    assert "#SBATCH --nodes=1" in contents
    assert "#SBATCH --ntasks=1" in contents
    assert "#SBATCH --array=0-1%2" in contents
    assert "#SBATCH --no-requeue" in contents
    assert "#SBATCH --nodelist" not in contents
    assert "#SBATCH -w" not in contents
    assert "%A_%a" in contents


def test_array_index_is_the_only_cell_selector_and_exact_allowlist():
    contents = _contents(LAUNCHER)

    assert 'case "$SLURM_ARRAY_TASK_ID" in' in contents
    assert 'fail "SLURM_ARRAY_TASK_ID must be exactly 0 or 1"' in contents
    assert "AMBIXQC_CONFIG" not in contents
    assert "AMBIXQC_MANIFEST" not in contents
    assert "--manifest" not in contents
    for index, (stem, _) in ALLOWLIST.items():
        assert f"  {index})" in contents
        assert f'CONFIG_STEM="{stem}"' in contents

    for forbidden in (
        "ambixqc_humanoid_walk_heavy_inner_v1_d256_g1",
        "ambixqc_humanoid_walk_heavy_inner_v1_d256_g1_j6",
        "ambixqc_humanoid_walk_heavy_inner_v1_d256_g3",
        "ambixqc_humanoid_walk_heavy_inner_v1_d512_g1",
        "ambixqc_humanoid_walk_heavy_inner_v1_d512_g1_j6",
        "ambixqc_humanoid_walk_heavy_inner_v1_d512_b256_g6",
    ):
        assert forbidden not in contents


def test_allowlisted_files_freeze_the_expected_heavy_contract():
    for _, (stem, schedule) in ALLOWLIST.items():
        rounds, rollouts, horizon, updates, batch_size, capacity = schedule
        config = _load_json(ALG_DIR / f"{stem}.json")
        manifest = _load_json(EXPERIMENT_DIR / f"{stem}.json")
        params = config["alg_params"]

        assert config["seed"] == 55
        assert config["device"] == "cuda"
        assert config["total_steps"] == 14_000_000
        assert params["utd"] == 1
        assert params["eval_freq"] is None
        assert params["compile"] is params["compile_strict"] is True
        assert params["inner_reward_normalization"] == "action_local_imagined"
        assert (
            params["inner_rounds"],
            params["inner_rollouts_per_round"],
            params["inner_rollout_horizon"],
            params["inner_updates_per_round"],
            params["inner_batch_size"],
            params["inner_replay_capacity"],
        ) == (rounds, rollouts, horizon, updates, batch_size, capacity)
        assert manifest["configs"] == [stem]
        assert manifest["trials"] == 1
        assert manifest["checkpoint_every"] is None
        assert manifest["save_strat"] == "none"
        assert manifest["save_trials"] == "none"


def test_launcher_preflights_strict_science_and_runs_one_cell_per_task():
    contents = _contents(LAUNCHER)

    assert 'params.get("compile") is not True' in contents
    assert 'params.get("compile_strict") is not True' in contents
    assert 'params.get("utd") != 1' in contents
    assert 'params.get("eval_freq") is not None' in contents
    assert 'params.get("inner_reward_normalization") != "action_local_imagined"' in contents
    assert 'manifest.get("checkpoint_every") is not None' in contents
    assert 'manifest.get("save_strat") != "none"' in contents
    assert '"total_steps": 14_000_000' in contents
    assert '"task": "humanoid-walk"' in contents
    assert '"obs": "state"' in contents
    assert 'params["wandb_run_name"] = run_name' in contents
    assert '--run "$MANIFEST"' in contents
    assert '--alg-dir "$SCRATCH_ALG_DIR"' in contents
    assert "--alg-index 0" in contents
    assert "--trial-index 0" in contents
    assert "--num-runs 1" in contents
    assert "validate_evaluations" not in contents
    assert "validate_final_checkpoint" not in contents
    assert "TDMPC2_EVAL_CSV=" not in contents
    assert "XQC_EVAL_CSV=" not in contents


def test_launcher_guards_source_environment_paths_space_and_artifacts():
    contents = _contents(LAUNCHER)

    assert "EXPECTED_ACTION_MODES_SHA" in contents
    assert "^[0-9a-f]{40}$" in contents
    assert contents.count(
        'require_clean_sha "$PROJECT_DIR_REAL" "$EXPECTED_ACTION_MODES_SHA"'
    ) == 2
    assert "status --porcelain=v1 --untracked-files=all" in contents
    lock = "f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6"
    assert lock in contents
    assert '"$PROJECT_DIR_REAL/environments/dmcontrol/uv.lock"' in contents
    assert '"$ENV_PROJECT_REAL/uv.lock"' in contents
    assert '"$PYTHON" == "$ENV_PROJECT_REAL/.venv/bin/python"' in contents
    assert "durable results must be outside the Git checkout" in contents
    assert "study root must be one existing non-symlinked directory" in contents
    assert '[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]]' in contents
    assert '[[ ! -e "$JOB_SCRATCH" && ! -L "$JOB_SCRATCH" ]]' in contents
    assert 'readonly MIN_DURABLE_FREE_KIB=$((4 * 1024 * 1024))' in contents
    assert 'readonly MIN_SCRATCH_FREE_KIB=$((8 * 1024 * 1024))' in contents
    assert 'df -Pk "$path"' in contents
    assert "rm -rf" not in contents
    assert "git pull" not in contents
    assert "git fetch" not in contents
    assert "uv sync" not in contents


def test_launcher_isolates_compile_wandb_and_miscellaneous_caches():
    contents = _contents(LAUNCHER)

    for cache in (
        "XDG_CACHE_HOME",
        "TORCH_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "NUMBA_CACHE_DIR",
        "MPLCONFIGDIR",
        "TMPDIR",
        "WANDB_DIR",
        "WANDB_CACHE_DIR",
        "WANDB_DATA_DIR",
        "WANDB_ARTIFACT_DIR",
    ):
        assert f"export {cache}=" in contents
    assert "export PYTHONNOUSERSITE=1" in contents
    assert "export MUJOCO_GL=egl" in contents
    assert "export MUJOCO_EGL_DEVICE_ID=0" in contents
    assert "export WANDB_MODE=online" in contents
    assert 'export WANDB_RUN_ID="$WANDB_UNIQUE_ID"' in contents
    assert "export WANDB_RESUME=never" in contents
    assert 'readonly JOB_KEY="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"' in contents
    assert 'readonly RUN_ROOT="$STUDY_ROOT/$CONFIG_STEM-job$JOB_KEY"' in contents
    assert 'exec > >(tee "$RUN_ROOT/job.log") 2>&1' in contents
    assert "torch.cuda.is_available()" in contents
    assert "torch.get_default_dtype() == torch.float32" in contents
    assert "not torch.is_autocast_enabled()" in contents
    assert "env.unwrapped.action_repeat == 2" in contents
    assert 'readonly EXPECTED_NODE="gpu2301"' in contents
    assert '"${SLURM_NODELIST:-}" == "$EXPECTED_NODE"' in contents
    assert 'props.name == "NVIDIA L40"' in contents
    assert "props.total_memory >= 44 * 1024**3" in contents
    assert "torch.cuda.device_count() == 1" in contents
    assert (
        "unset TORCHDYNAMO_DISABLE CUDA_LAUNCH_BLOCKING "
        "PYTORCH_NO_CUDA_MEMORY_CACHING"
    ) in contents
    assert "unset TORCH_COMPILE_DEBUG TORCH_LOGS" in contents
    assert "Compilation: strict; any graph or runtime fallback is fatal" in contents
    assert "Optimizer backend: auto; fused Adam is intended on CUDA" in contents
    assert "printf 'PASS\\n' > \"$RUN_ROOT/PASS\"" in contents


def test_submitter_is_one_atomic_gpu2301_only_submission():
    contents = _contents(SUBMITTER)

    assert 'readonly TARGET_NODE="gpu2301"' in contents
    assert 'readonly JOB_COUNT=2' in contents
    assert 'readonly JOB_CPUS=8' in contents
    assert 'readonly JOB_MEMORY_MIB=65536' in contents
    assert 'readonly MIN_DURABLE_FREE_KIB=$((8 * 1024 * 1024))' in contents
    assert "gpu2201" not in contents
    assert 'scontrol show node -o "$node"' in contents
    assert "available_slots >= JOB_COUNT" in contents
    assert "need two immediate one-GPU/eight-CPU/64-GiB slots" in contents
    assert '--nodelist="$TARGET_NODE"' in contents
    assert '--chdir="$PROJECT_DIR"' in contents
    assert 'cd "$PROJECT_DIR"' in contents
    assert '--output="$STUDY_ROOT/slurm-%x-%A_%a.out"' in contents
    assert '--error="$STUDY_ROOT/slurm-%x-%A_%a.err"' in contents
    assert '--export="$export_spec"' in contents
    assert "EXPECTED_ACTION_MODES_SHA=$EXPECTED_ACTION_MODES_SHA" in contents
    assert "AMBIXQC_HEAVY_RESULTS_ROOT=$AMBIXQC_HEAVY_RESULTS_ROOT" in contents
    assert "AMBIXQC_HEAVY_RUN_ID=$AMBIXQC_HEAVY_RUN_ID" in contents
    assert "AMBIXQC_HEAVY_PYTHON=$AMBIXQC_HEAVY_PYTHON" in contents
    assert contents.count("submission=\"") == 1
    assert "for ((index" not in contents
    assert "--dry-run" in contents
    assert "scancel" not in contents


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="forking bash after the macOS Torch runtime can abort in libomp",
)
def test_node_capacity_function_uses_gpu_cpu_memory_and_health_limits():
    contents = _contents(SUBMITTER)
    body = contents.split("node_job_slots() {", 1)[1].split(
        '\n}\n\n[[ "$AMBIXQC_HEAVY_PYTHON"', 1
    )[0]
    function = f"node_job_slots() {{{body}\n}}"
    records = [
        (
            "NodeName=gpu2301 CPUAlloc=0 CPUTot=64 RealMemory=524288 "
            "AllocMem=0 State=IDLE CfgTRES=cpu=64,mem=512G,gres/gpu=4 "
            "AllocTRES=",
            4,
        ),
        (
            "NodeName=gpu2301 CPUAlloc=16 CPUTot=64 RealMemory=524288 "
            "AllocMem=131072 State=MIXED CfgTRES=cpu=64,mem=512G,gres/gpu=4 "
            "AllocTRES=cpu=16,mem=128G,gres/gpu=2",
            2,
        ),
        (
            "NodeName=gpu2301 CPUAlloc=16 CPUTot=24 RealMemory=524288 "
            "AllocMem=0 State=MIXED CfgTRES=cpu=24,mem=512G,gres/gpu=4 "
            "AllocTRES=cpu=16,gres/gpu=1",
            1,
        ),
        (
            "NodeName=gpu2301 CPUAlloc=0 CPUTot=64 RealMemory=196608 "
            "AllocMem=131072 State=MIXED CfgTRES=cpu=64,mem=192G,gres/gpu=4 "
            "AllocTRES=mem=128G",
            1,
        ),
        (
            "NodeName=gpu2301 CPUAlloc=0 CPUTot=64 RealMemory=524288 "
            "AllocMem=0 State=DOWN CfgTRES=cpu=64,mem=512G,gres/gpu=4 "
            "AllocTRES=",
            0,
        ),
    ]

    for record, expected in records:
        script = f'''\
JOB_CPUS=8
JOB_MEMORY_MIB=65536
scontrol() {{ printf '%s\\n' "$NODE_RECORD"; }}
{function}
node_job_slots gpu2301
'''
        result = subprocess.run(
            ["bash"],
            input=script,
            text=True,
            capture_output=True,
            check=True,
            env={**os.environ, "NODE_RECORD": record},
        )
        assert result.stdout.strip() == str(expected)


def test_submitter_rejects_unlocked_or_reused_inputs_before_sbatch():
    contents = _contents(SUBMITTER)
    first_sbatch = contents.index("  sbatch\n")

    assert "submission requires a clean checkout" in contents
    assert 'require_clean_sha "$PROJECT_DIR" "$EXPECTED_ACTION_MODES_SHA"' in contents
    assert '"$PROJECT_DIR/environments/dmcontrol/uv.lock"' in contents
    assert '"$ENV_PROJECT/uv.lock"' in contents
    assert '"$AMBIXQC_HEAVY_PYTHON" == "$ENV_PROJECT/.venv/bin/python"' in contents
    assert "durable results must be outside the Git checkout" in contents
    assert 'if [[ -e "$STUDY_ROOT" || -L "$STUDY_ROOT" ]]' in contents
    assert "exported paths and IDs cannot contain commas or newlines" in contents
    assert 'df -Pk "$path"' in contents
    assert contents.index('require_clean_sha "$PROJECT_DIR"') < first_sbatch
    assert contents.index('if [[ -e "$STUDY_ROOT"') < first_sbatch
    assert contents.index('available_slots="$(node_job_slots') < first_sbatch
    assert "rm -rf" not in contents
    assert "git pull" not in contents
    assert "git fetch" not in contents
    assert "uv sync" not in contents


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="forking bash after the macOS Torch runtime can abort in libomp",
)
def test_scripts_are_executable_valid_bash_and_documented():
    for path in (LAUNCHER, SUBMITTER):
        assert path.stat().st_mode & stat.S_IXUSR
        subprocess.run(["bash", "-n", str(path)], check=True)

    readme = _contents(README)
    normalized_readme = " ".join(readme.split())
    assert "submit_ambixqc_humanoid_walk_heavy_inner_pair_hydra.sh" in readme
    assert "gpu2301" in readme
    assert "`d512_g3_j6`" in readme
    assert "`d512_g3`" in readme
    assert "two immediately available" in normalized_readme
    assert "64 GiB" in normalized_readme
    assert "30 days" in normalized_readme
    assert "no fallback node" in normalized_readme
    assert "eight GiB of durable free space" in normalized_readme
    assert "four GiB" in normalized_readme
    assert "NVIDIA L40" in normalized_readme
    assert "strict compilation" in normalized_readme
    assert "auto" in normalized_readme
    assert "fused Adam" in normalized_readme
    assert "No job is submitted by adding these launch files" in normalized_readme
