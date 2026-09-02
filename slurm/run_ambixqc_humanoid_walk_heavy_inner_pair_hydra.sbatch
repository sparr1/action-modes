#!/usr/bin/env bash
#SBATCH --job-name=ambixqc-hw-heavy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=30-00:00:00
#SBATCH --array=0-1%2
#SBATCH --no-requeue
#SBATCH --output=/tmp/ambixqc-humanoid-walk-heavy-%A_%a.out
#SBATCH --error=/tmp/ambixqc-humanoid-walk-heavy-%A_%a.err

set -Eeuo pipefail
umask 077

# Submit only through submit_ambixqc_humanoid_walk_heavy_inner_pair_hydra.sh.
# The array index is the allowlist: callers cannot select another manifest.

readonly ACTION_LOCK_SHA="f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6"
readonly EXPECTED_NODE="gpu2301"
readonly MIN_DURABLE_FREE_KIB=$((4 * 1024 * 1024))
readonly MIN_SCRATCH_FREE_KIB=$((8 * 1024 * 1024))
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

: "${EXPECTED_ACTION_MODES_SHA:?Export the exact pushed Action Modes commit SHA.}"
: "${AMBIXQC_HEAVY_RESULTS_ROOT:?Export an existing absolute durable result directory.}"
: "${AMBIXQC_HEAVY_RUN_ID:?Export a unique run ID.}"
: "${AMBIXQC_HEAVY_PYTHON:?Export an absolute locked DMControl Python path.}"
: "${SLURM_JOB_ID:?This launcher must run inside a Slurm allocation.}"
: "${SLURM_ARRAY_JOB_ID:?This launcher must run as a Slurm array.}"
: "${SLURM_ARRAY_TASK_ID:?This launcher must run as a Slurm array.}"

fail() {
  echo "AMBI-XQC heavy-inner Hydra error: $*" >&2
  exit 2
}

[[ "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]] || fail \
  "EXPECTED_ACTION_MODES_SHA must be a full lowercase 40-character SHA"
[[ "$AMBIXQC_HEAVY_RESULTS_ROOT" == /* ]] || fail \
  "AMBIXQC_HEAVY_RESULTS_ROOT must be absolute"
[[ -d "$AMBIXQC_HEAVY_RESULTS_ROOT" ]] || fail \
  "AMBIXQC_HEAVY_RESULTS_ROOT must already exist"
[[ ! -L "$AMBIXQC_HEAVY_RESULTS_ROOT" ]] || fail \
  "AMBIXQC_HEAVY_RESULTS_ROOT must not be a symlink"
[[ -w "$AMBIXQC_HEAVY_RESULTS_ROOT" ]] || fail \
  "AMBIXQC_HEAVY_RESULTS_ROOT is not writable"
[[ "$AMBIXQC_HEAVY_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail \
  "AMBIXQC_HEAVY_RUN_ID must use only letters, digits, '.', '_', and '-'"
(( ${#AMBIXQC_HEAVY_RUN_ID} <= 64 )) || fail \
  "AMBIXQC_HEAVY_RUN_ID must contain at most 64 characters"
[[ "$AMBIXQC_HEAVY_PYTHON" == /* && -x "$AMBIXQC_HEAVY_PYTHON" ]] || fail \
  "AMBIXQC_HEAVY_PYTHON must be an executable at an absolute path"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    CONFIG_STEM="ambixqc_humanoid_walk_heavy_inner_v1_d512_g3_j6"
    CELL_LABEL="d512-g3-j6"
    EXPECTED_ROUNDS=6
    EXPECTED_CAPACITY=9216
    EXPECTED_SLOTS=18
    EXPECTED_DELAYED_STEPS=6
    ;;
  1)
    CONFIG_STEM="ambixqc_humanoid_walk_heavy_inner_v1_d512_g3"
    CELL_LABEL="d512-g3"
    EXPECTED_ROUNDS=8
    EXPECTED_CAPACITY=12288
    EXPECTED_SLOTS=24
    EXPECTED_DELAYED_STEPS=8
    ;;
  *)
    fail "SLURM_ARRAY_TASK_ID must be exactly 0 or 1"
    ;;
esac
readonly CONFIG_STEM CELL_LABEL EXPECTED_ROUNDS EXPECTED_CAPACITY
readonly EXPECTED_SLOTS EXPECTED_DELAYED_STEPS

PROJECT_DIR_REAL="$(cd "$PROJECT_DIR" && pwd -P)"
RESULTS_BASE_REAL="$(cd "$AMBIXQC_HEAVY_RESULTS_ROOT" && pwd -P)"
PYTHON="$(cd "$(dirname "$AMBIXQC_HEAVY_PYTHON")" && pwd -P)/$(basename "$AMBIXQC_HEAVY_PYTHON")"
ENV_PROJECT_REAL="$(cd "$(dirname "$PYTHON")/../.." && pwd -P)"
readonly PROJECT_DIR_REAL RESULTS_BASE_REAL PYTHON ENV_PROJECT_REAL
readonly MANIFEST="$PROJECT_DIR_REAL/configs/dmcontrol/experiments/$CONFIG_STEM.json"
readonly SOURCE_ALG_CONFIG="$PROJECT_DIR_REAL/configs/dmcontrol/algs/$CONFIG_STEM.json"
readonly STUDY_ROOT="$RESULTS_BASE_REAL/$AMBIXQC_HEAVY_RUN_ID"
readonly JOB_KEY="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
readonly RUN_ROOT="$STUDY_ROOT/$CONFIG_STEM-job$JOB_KEY"
readonly SCRATCH_BASE="${SLURM_TMPDIR:-/tmp}"
readonly JOB_SCRATCH="$SCRATCH_BASE/ambixqc-heavy-inner-$JOB_KEY"
readonly SCRATCH_ALG_DIR="$JOB_SCRATCH/algs"
readonly WANDB_LOCAL_ROOT="$JOB_SCRATCH/wandb"
readonly WANDB_UNIQUE_NAME="AMBIXQC-humanoid-walk-heavy-inner-v1-$CELL_LABEL-seed55-$AMBIXQC_HEAVY_RUN_ID-job$JOB_KEY"
readonly WANDB_UNIQUE_ID="axh-${EXPECTED_ACTION_MODES_SHA:0:8}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

require_clean_sha() {
  local checkout="$1"
  local expected="$2"
  local actual
  actual="$(git -C "$checkout" rev-parse HEAD)"
  [[ "$actual" == "$expected" ]] || fail \
    "Action Modes commit mismatch: expected $expected, found $actual"
  if [[ -n "$(git -C "$checkout" status --porcelain=v1 --untracked-files=all)" ]]; then
    git -C "$checkout" status --short >&2
    fail "Action Modes checkout must be clean"
  fi
}

require_lock_hash() {
  local lockfile="$1"
  local expected="$2"
  local label="$3"
  local actual
  [[ -f "$lockfile" ]] || fail "missing $label lock: $lockfile"
  actual="$(sha256sum "$lockfile" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || fail \
    "$label lock mismatch: expected $expected, found $actual"
}

require_free_kib() {
  local path="$1"
  local minimum="$2"
  local label="$3"
  local available
  available="$(df -Pk "$path" | awk 'NR == 2 {print $4}')"
  [[ "$available" =~ ^[0-9]+$ ]] || fail \
    "could not determine free space for $label: $path"
  (( available >= minimum )) || fail \
    "$label has less than $((minimum / 1024 / 1024)) GiB free: $path"
}

for command in git sha256sum df awk nvidia-smi srun; do
  command -v "$command" >/dev/null 2>&1 || fail "missing required command: $command"
done

[[ "$PYTHON" == "$ENV_PROJECT_REAL/.venv/bin/python" ]] || fail \
  "AMBIXQC_HEAVY_PYTHON must point to the locked environment's .venv/bin/python"
[[ -f "$MANIFEST" && -f "$SOURCE_ALG_CONFIG" ]] || fail \
  "allowlisted manifest or algorithm configuration is missing"
[[ -d "$STUDY_ROOT" && ! -L "$STUDY_ROOT" ]] || fail \
  "study root must be one existing non-symlinked directory created by the submitter"
[[ "${SLURM_NODELIST:-}" == "$EXPECTED_NODE" ]] || fail \
  "launcher must run on $EXPECTED_NODE, found ${SLURM_NODELIST:-unset}"
case "$RESULTS_BASE_REAL/" in
  "$PROJECT_DIR_REAL/"*) fail "durable results must be outside the Git checkout" ;;
esac
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || fail \
  "refusing to overwrite an existing run directory: $RUN_ROOT"
[[ -d "$SCRATCH_BASE" && -w "$SCRATCH_BASE" ]] || fail \
  "node-local scratch base must be an existing writable directory: $SCRATCH_BASE"
[[ ! -e "$JOB_SCRATCH" && ! -L "$JOB_SCRATCH" ]] || fail \
  "refusing to reuse node-local job scratch: $JOB_SCRATCH"

require_clean_sha "$PROJECT_DIR_REAL" "$EXPECTED_ACTION_MODES_SHA"
require_lock_hash \
  "$PROJECT_DIR_REAL/environments/dmcontrol/uv.lock" "$ACTION_LOCK_SHA" \
  "Action Modes"
require_lock_hash "$ENV_PROJECT_REAL/uv.lock" "$ACTION_LOCK_SHA" \
  "DMControl environment"
require_free_kib "$RESULTS_BASE_REAL" "$MIN_DURABLE_FREE_KIB" \
  "durable result filesystem"
require_free_kib "$SCRATCH_BASE" "$MIN_SCRATCH_FREE_KIB" \
  "node-local scratch filesystem"

mkdir "$RUN_ROOT"
mkdir "$JOB_SCRATCH"
mkdir "$SCRATCH_ALG_DIR"
mkdir -p \
  "$JOB_SCRATCH/cache/torch" \
  "$JOB_SCRATCH/cache/torchinductor" \
  "$JOB_SCRATCH/cache/triton" \
  "$JOB_SCRATCH/cache/cuda" \
  "$JOB_SCRATCH/cache/numba" \
  "$JOB_SCRATCH/matplotlib" \
  "$JOB_SCRATCH/tmp" \
  "$WANDB_LOCAL_ROOT/cache" \
  "$WANDB_LOCAL_ROOT/data" \
  "$WANDB_LOCAL_ROOT/artifacts"
exec > >(tee "$RUN_ROOT/job.log") 2>&1

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export XDG_CACHE_HOME="$JOB_SCRATCH/cache"
export TORCH_HOME="$JOB_SCRATCH/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$JOB_SCRATCH/cache/torchinductor"
export TRITON_CACHE_DIR="$JOB_SCRATCH/cache/triton"
export CUDA_CACHE_PATH="$JOB_SCRATCH/cache/cuda"
export NUMBA_CACHE_DIR="$JOB_SCRATCH/cache/numba"
export MPLCONFIGDIR="$JOB_SCRATCH/matplotlib"
export TMPDIR="$JOB_SCRATCH/tmp"
export WANDB_MODE=online
export WANDB_ENTITY=rwgao_b-brown-university
export WANDB_PROJECT=ambi
export WANDB_DIR="$WANDB_LOCAL_ROOT"
export WANDB_CACHE_DIR="$WANDB_LOCAL_ROOT/cache"
export WANDB_DATA_DIR="$WANDB_LOCAL_ROOT/data"
export WANDB_ARTIFACT_DIR="$WANDB_LOCAL_ROOT/artifacts"
export WANDB_DISABLE_CODE=true
export WANDB_RUN_ID="$WANDB_UNIQUE_ID"
export WANDB_RESUME=never
unset TDMPC2_EVAL_CSV XQC_EVAL_CSV XQC_COMPARISON_ID
unset WANDB_DISABLED WANDB_NAME WANDB_TAGS WANDB_RUN_GROUP WANDB_JOB_TYPE
unset TORCHDYNAMO_DISABLE CUDA_LAUNCH_BLOCKING PYTORCH_NO_CUDA_MEMORY_CACHING
unset TORCH_COMPILE_DEBUG TORCH_LOGS

# Validate the full scientific and artifact contract before starting a
# checkpoint-free 14M-decision run. Only the operational W&B display name is
# changed in the scratch copy consumed by main.py.
"$PYTHON" -c '
import json
import sys
from pathlib import Path


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit(f"duplicate JSON key: {key}")
        result[key] = value
    return result


source = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
destination = Path(sys.argv[3])
stem = sys.argv[4]
run_name = sys.argv[5]
rounds, capacity, slots, delayed = map(int, sys.argv[6:10])
config = json.loads(
    source.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
)
manifest = json.loads(
    manifest_path.read_text(encoding="utf-8"),
    object_pairs_hook=reject_duplicate_keys,
)
params = config.get("alg_params", {})
expected_schedule = {
    "inner_rounds": rounds,
    "inner_rollouts_per_round": 512,
    "inner_rollout_horizon": 3,
    "inner_updates_per_round": 3,
    "inner_batch_size": 512,
    "inner_replay_capacity": capacity,
}
actual_schedule = {key: params.get(key) for key in expected_schedule}
if actual_schedule != expected_schedule:
    raise SystemExit(
        f"unexpected heavy-inner schedule: {actual_schedule} != {expected_schedule}"
    )
expected_top_level = {
    "seed": 55,
    "env": "DMControl-v0",
    "alg": "AMBIXQC/AMBIXQC",
    "device": "cuda",
    "total_steps": 14_000_000,
    "episodes": None,
}
if any(config.get(key) != value for key, value in expected_top_level.items()):
    raise SystemExit("unexpected heavy-inner top-level algorithm contract")
if (
    params.get("compile") is not True
    or params.get("compile_strict") is not True
    or params.get("utd") != 1
    or params.get("eval_freq") is not None
    or params.get("mpc") is not False
    or params.get("xqc_policy_delay") != 3
    or params.get("xqc_optimizer_backend") != "auto"
    or params.get("inner_reward_normalization") != "action_local_imagined"
    or params.get("wandb") is not True
    or params.get("wandb_mode") != "online"
):
    raise SystemExit("unexpected heavy-inner strict-XQC runtime contract")
if (rounds * 3, (rounds * 3 - 1) // 3 + 1) != (slots, delayed):
    raise SystemExit("declared optimizer-slot metadata is inconsistent")
if manifest.get("configs") != [stem] or manifest.get("trials") != 1:
    raise SystemExit("manifest must select exactly the allowlisted single-seed cell")
if manifest.get("overrides_alg") != {
    "seed": 55,
    "device": "cuda",
    "env": "DMControl-v0",
    "total_steps": 14_000_000,
    "episodes": None,
}:
    raise SystemExit("unexpected heavy-inner manifest overrides")
if manifest.get("env_params") != {
    "task": "humanoid-walk",
    "obs": "state",
    "render_mode": None,
}:
    raise SystemExit("unexpected heavy-inner environment contract")
if (
    manifest.get("logs") != "timestamp"
    or manifest.get("save_trials") != "none"
    or manifest.get("checkpoint_every") is not None
    or manifest.get("save_strat") != "none"
    or "checkpoint_best_window" in manifest
):
    raise SystemExit("heavy-inner run must remain checkpoint-free")
params["wandb_run_name"] = run_name
with destination.open("x", encoding="utf-8") as stream:
    json.dump(config, stream, indent=2)
    stream.write("\n")
' \
  "$SOURCE_ALG_CONFIG" \
  "$MANIFEST" \
  "$SCRATCH_ALG_DIR/$CONFIG_STEM.json" \
  "$CONFIG_STEM" \
  "$WANDB_UNIQUE_NAME" \
  "$EXPECTED_ROUNDS" \
  "$EXPECTED_CAPACITY" \
  "$EXPECTED_SLOTS" \
  "$EXPECTED_DELAYED_STEPS"

echo "Array job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Cell: $CONFIG_STEM"
echo "Task: humanoid-walk (state)"
echo "Seed: 55"
echo "Agent decisions: 14000000"
echo "Inner schedule: J${EXPECTED_ROUNDS}/N512/H3/G3/B512"
echo "Compilation: strict; any graph or runtime fallback is fatal"
echo "Optimizer backend: auto; fused Adam is intended on CUDA, with runtime selection authoritative"
echo "Action Modes: $PROJECT_DIR_REAL @ $EXPECTED_ACTION_MODES_SHA"
echo "Python: $PYTHON"
echo "DMControl environment: $ENV_PROJECT_REAL"
echo "Manifest: $MANIFEST"
echo "Durable results: $RUN_ROOT"
echo "Node-local scratch: $JOB_SCRATCH"
echo "W&B run name: $WANDB_UNIQUE_NAME"
echo "W&B run ID: $WANDB_UNIQUE_ID (resume=never)"
git -C "$PROJECT_DIR_REAL" show --no-patch --format=fuller HEAD
sha256sum \
  "$PROJECT_DIR_REAL/environments/dmcontrol/uv.lock" \
  "$ENV_PROJECT_REAL/uv.lock"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total \
  --format=csv,noheader | tee "$RUN_ROOT/gpu.txt"

cd "$PROJECT_DIR_REAL"
srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON" -c \
  'import sys, torch; from importlib.metadata import version; assert sys.version_info[:2] == (3, 10); assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; assert torch.get_default_dtype() == torch.float32; assert not torch.is_autocast_enabled(); props = torch.cuda.get_device_properties(0); assert props.name == "NVIDIA L40", props.name; assert props.total_memory >= 44 * 1024**3, props.total_memory; print(sys.version); print({name: version(name) for name in ("dm-control", "gymnasium", "mujoco", "numpy", "torch", "wandb")}); print(torch.version.cuda, props.name, props.total_memory)' \
  | tee "$RUN_ROOT/runtime.txt"
srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON" -c \
  'import gymnasium as gym, numpy as np, domains; env = gym.make("DMControl-v0", task="humanoid-walk", obs="state", render_mode=None); error = None
try:
 obs, _ = env.reset(seed=55); assert obs.shape == (67,) and np.isfinite(obs).all(); assert env.action_space.shape == (21,); assert env.unwrapped.action_repeat == 2; assert np.isfinite(env.action_space.low).all() and np.isfinite(env.action_space.high).all(); print("DMControl probe", obs.shape, env.action_space.shape, env.unwrapped.action_repeat)
finally:
 env.close()' \
  | tee "$RUN_ROOT/dmcontrol.txt"

srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON" main.py \
  --run "$MANIFEST" \
  --alg-dir "$SCRATCH_ALG_DIR" \
  --log-dir "$RUN_ROOT/training" \
  --alg-index 0 \
  --trial-index 0 \
  --num-runs 1

require_clean_sha "$PROJECT_DIR_REAL" "$EXPECTED_ACTION_MODES_SHA"
printf 'PASS\n' > "$RUN_ROOT/PASS"
echo "AMBI-XQC heavy-inner cell completed. Results: $RUN_ROOT"
