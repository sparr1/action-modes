#!/usr/bin/env bash
#SBATCH --job-name=axqc-return-mppi
#SBATCH --partition=gpu
#SBATCH --qos=pri-gpu+
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0
#SBATCH --no-requeue
#SBATCH --output=/tmp/ambixqc-return-mppi-%A_%a.out
#SBATCH --error=/tmp/ambixqc-return-mppi-%A_%a.err

set -Eeuo pipefail
umask 077

# One task evaluates return-only MPPI; production reuses the completed prior episodes.
# The root operator chooses the array concurrency from live cluster capacity.
fail() { echo "AMBI-XQC return-only MPPI error: $*" >&2; exit 2; }
: "${EXPECTED_ACTION_MODES_SHA:?Export the exact tested and pushed commit.}"
: "${AMBIXQC_ACTION_MODES_DIR:?Export the clean checkout path.}"
: "${AMBIXQC_RESULTS_ROOT:?Export an existing durable directory outside the checkout.}"
: "${AMBIXQC_PYTHON:?Export the existing locked nested DMControl Python.}"
: "${AMBIXQC_CHECKPOINT_MANIFEST:?Export the immutable six-bank manifest.}"
: "${SLURM_JOB_ID:?Run inside an allocated Slurm job.}"
: "${SLURM_ARRAY_TASK_ID:?Submit the explicit checkpoint array.}"
readonly MODE="${AMBIXQC_MODE:-production}"
[[ "$MODE" == smoke || "$MODE" == production ]] || fail "mode must be smoke or production"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^(0|[1-9][0-9]*)$ ]] || fail "checkpoint index must be an integer"
((SLURM_ARRAY_TASK_ID < 80)) || fail "checkpoint index must be 0 through 79"
if [[ "$MODE" == smoke ]]; then
  case "$SLURM_ARRAY_TASK_ID" in 0|39|40|79) ;; *) fail "smoke indices must be 0,39,40,79" ;; esac
else
  : "${AMBIXQC_SMOKE_ROOT:?Export the validated four-cell smoke directory.}"
  : "${EVAL_RUN_MAP:?Prepare four explicit New evaluation runs.}"
  [[ "$AMBIXQC_SMOKE_ROOT" == /* && -d "$AMBIXQC_SMOKE_ROOT" ]] || fail "smoke root must be an absolute existing directory"
  : "${AMBIXQC_REFERENCE_INDEX:?Export the pinned prior-reference index.}"
  [[ "$AMBIXQC_REFERENCE_INDEX" == /* && -f "$AMBIXQC_REFERENCE_INDEX" ]] || fail "reference index must be an absolute existing file"
  [[ "$EVAL_RUN_MAP" == /* && -f "$EVAL_RUN_MAP" ]] || fail "run map must be an absolute existing file"
fi
[[ "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]] || fail "expected SHA must be full lowercase hexadecimal"
[[ "$AMBIXQC_ACTION_MODES_DIR" == /* && -d "$AMBIXQC_ACTION_MODES_DIR" ]] || fail "checkout must be an existing absolute directory"
[[ "$AMBIXQC_RESULTS_ROOT" == /* && -d "$AMBIXQC_RESULTS_ROOT" ]] || fail "result root must be an existing absolute directory"
[[ "$AMBIXQC_PYTHON" == /* && -x "$AMBIXQC_PYTHON" ]] || fail "Python must be an absolute executable"
[[ "$AMBIXQC_CHECKPOINT_MANIFEST" == /* && -f "$AMBIXQC_CHECKPOINT_MANIFEST" ]] || fail "manifest must be an absolute existing file"
readonly PROJECT_DIR="$(cd "$AMBIXQC_ACTION_MODES_DIR" && pwd -P)"
readonly RESULTS_ROOT="$(cd "$AMBIXQC_RESULTS_ROOT" && pwd -P)"
# Canonicalize the containing directory, not the executable symlink (which may
# point outside its venv). This also accepts Hydra's /home -> /cs/home alias.
readonly PYTHON="$(cd "$(dirname "$AMBIXQC_PYTHON")" && pwd -P)/$(basename "$AMBIXQC_PYTHON")"
readonly ENV_PROJECT="$(cd "$(dirname "$PYTHON")/../.." && pwd -P)"
[[ "$PYTHON" == "$ENV_PROJECT/.venv/bin/python" ]] || fail "Python must belong to a nested locked .venv"
case "$RESULTS_ROOT/" in "$PROJECT_DIR/"*) fail "results must be outside the checkout" ;; esac
readonly ARRAY_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
readonly JOB_ROOT="$RESULTS_ROOT/$MODE/job${ARRAY_ID}-task${SLURM_ARRAY_TASK_ID}"
readonly CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/ambixqc-return-mppi-${ARRAY_ID}-${SLURM_ARRAY_TASK_ID}"
[[ ! -e "$JOB_ROOT" && ! -L "$JOB_ROOT" ]] || fail "refusing to overwrite job outputs"
[[ ! -e "$CACHE_ROOT" && ! -L "$CACHE_ROOT" ]] || fail "refusing to reuse caches"
readonly LOCK_SHA=f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6
require_clean_sha() {
  [[ "$(git -C "$PROJECT_DIR" rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || fail "checkout commit mismatch"
  [[ -z "$(git -C "$PROJECT_DIR" status --porcelain=v1 --untracked-files=all)" ]] || fail "checkout must be clean"
}
for command in git sha256sum nvidia-smi srun; do
  command -v "$command" >/dev/null || fail "missing command: $command"
done
require_clean_sha
for lock in "$PROJECT_DIR/environments/dmcontrol/uv.lock" "$ENV_PROJECT/uv.lock"; do
  [[ -f "$lock" && "$(sha256sum "$lock" | awk '{print $1}')" == "$LOCK_SHA" ]] || fail "runtime lock mismatch: $lock"
done
cd "$PROJECT_DIR"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
if [[ "$MODE" == production ]]; then
  "$PYTHON" run_ambixqc_return_mppi_evaluation.py \
    --manifest "$AMBIXQC_CHECKPOINT_MANIFEST" --source-sha "$EXPECTED_ACTION_MODES_SHA" \
    --verify-smoke-root "$AMBIXQC_SMOKE_ROOT"
fi
mkdir -p "$RESULTS_ROOT/$MODE"
mkdir "$JOB_ROOT" "$CACHE_ROOT"
mkdir -p "$CACHE_ROOT/wandb/cache" "$CACHE_ROOT/wandb/data" "$CACHE_ROOT/wandb/artifacts" \
  "$CACHE_ROOT/wandb/config" "$CACHE_ROOT/xdg" "$CACHE_ROOT/torchinductor" \
  "$CACHE_ROOT/triton" "$CACHE_ROOT/cuda" "$CACHE_ROOT/tmp" "$CACHE_ROOT/matplotlib"
exec > >(tee "$JOB_ROOT/job.log") 2>&1
trap 'status=$?; if ((status != 0)); then printf "FAILED exit=%s\n" "$status" > "$JOB_ROOT/FAILED"; fi' EXIT
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" OPENBLAS_NUM_THREADS=1
export WANDB_DIR="$CACHE_ROOT/wandb" WANDB_CACHE_DIR="$CACHE_ROOT/wandb/cache"
export WANDB_DATA_DIR="$CACHE_ROOT/wandb/data" WANDB_ARTIFACT_DIR="$CACHE_ROOT/wandb/artifacts"
export WANDB_CONFIG_DIR="$CACHE_ROOT/wandb/config" WANDB_CONFIG_PATHS=""
export XDG_CACHE_HOME="$CACHE_ROOT/xdg" XDG_CONFIG_HOME="$CACHE_ROOT/xdg"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor" TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export CUDA_CACHE_PATH="$CACHE_ROOT/cuda" MPLCONFIGDIR="$CACHE_ROOT/matplotlib"
export TMPDIR="$CACHE_ROOT/tmp" WANDB_DISABLE_CODE=true
unset TDMPC2_EVAL_CSV TDMPC2_COMPUTE_TIMING_OUTPUT WANDB_NAME WANDB_RUN_ID WANDB_RESUME PYTEST_ADDOPTS
# GPU workers only stage portable results. The sole Oscar CPU owner publishes.
export WANDB_MODE=disabled WANDB_DISABLED=true
printf 'AMBI-XQC return-only MPPI %s: job %s task %s node %s source %s\n' \
  "$MODE" "$ARRAY_ID" "$SLURM_ARRAY_TASK_ID" "${SLURM_NODELIST:-unknown}" "$EXPECTED_ACTION_MODES_SHA"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader | tee "$JOB_ROOT/gpu.txt"
srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON" -c '
import json, sys, torch
from pathlib import Path
assert sys.version_info[:2] == (3, 10)
assert torch.__version__.split("+")[0] == "2.3.1"
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert "L40S" in torch.cuda.get_device_name(0)
assert Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve() / ".venv"
assert torch.ones(1, device="cuda").sum().item() == 1
print(json.dumps({"python": sys.version, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}))
' "$ENV_PROJECT" | tee "$JOB_ROOT/runtime.json"
if [[ "$MODE" == smoke && "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
  srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON" -m pytest -q -rs --color=no -p no:cacheprovider \
    tests/test_mppi_helper.py tests/test_xqc_mppi.py tests/test_ambixqc_mppi_evaluation.py \
    tests/test_ambixqc_controller.py tests/test_ambixqc_prior_checkpoint.py \
    tests/test_ambixqc_backbone_mppi_launchers.py \
    tests/test_ambixqc_backbone_mppi_evaluation.py \
    tests/test_xqc_return_mppi.py tests/test_ambixqc_return_mppi_evaluation.py | tee "$JOB_ROOT/pytest.log"
fi
args=(--manifest "$AMBIXQC_CHECKPOINT_MANIFEST" --index "$SLURM_ARRAY_TASK_ID" \
  --result-root "$JOB_ROOT" --mode "$MODE" --device cuda)
if [[ -n "${AMBIXQC_CHECKPOINT_ROOT:-}" ]]; then
  [[ "$AMBIXQC_CHECKPOINT_ROOT" == /* && -d "$AMBIXQC_CHECKPOINT_ROOT" ]] || fail "checkpoint root override must be absolute existing directory"
  args+=(--checkpoint-root "$AMBIXQC_CHECKPOINT_ROOT")
fi
if [[ "$MODE" == production ]]; then
  args+=(--eval-run-map "$EVAL_RUN_MAP" --reference-index "$AMBIXQC_REFERENCE_INDEX")
fi
srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON" \
  run_ambixqc_return_mppi_evaluation.py "${args[@]}"
require_clean_sha
printf 'PASS\n' > "$JOB_ROOT/PASS"
echo "Return-only backbone MPPI $MODE passed: $JOB_ROOT"
