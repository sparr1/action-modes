#!/usr/bin/env bash
#SBATCH --job-name=ambixqc-prior-mppi
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
#SBATCH --output=/tmp/ambixqc-mppi-%A_%a.out
#SBATCH --error=/tmp/ambixqc-mppi-%A_%a.err

set -Eeuo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the clean repository root.}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the exact tested and pushed commit.}"
: "${CHECKPOINT_MANIFEST:?Supply the absolute immutable checkpoint manifest.}"
: "${RESULT_ROOT:?Supply an absolute output directory outside the checkout.}"
: "${SLURM_ARRAY_TASK_ID:?Missing checkpoint array index.}"
PYTHON_BIN="${PYTHON_BIN:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
MODE="${AMBIXQC_EVAL_MODE:-production}"
[[ "$MODE" == smoke || "$MODE" == production ]] || { echo "Invalid evaluation mode" >&2; exit 2; }
[[ "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$CHECKPOINT_MANIFEST" == /* && "$RESULT_ROOT" == /* && "$PYTHON_BIN" == /* ]] || exit 2
[[ -f "$CHECKPOINT_MANIFEST" && -x "$PYTHON_BIN" ]] || exit 2
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || exit 2
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || { echo "Source checkout is dirty" >&2; exit 2; }
readonly LOCK_SHA=f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6
ENV_PROJECT="$(cd "$(dirname "$PYTHON_BIN")/../.." && pwd -P)"
[[ "$PYTHON_BIN" == "$ENV_PROJECT/.venv/bin/python" ]] || exit 2
for lock in environments/dmcontrol/uv.lock "$ENV_PROJECT/uv.lock"; do
  [[ "$(sha256sum "$lock" | awk '{print $1}')" == "$LOCK_SHA" ]] || { echo "Runtime lock mismatch" >&2; exit 2; }
done

readonly CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/ambixqc-mppi-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
[[ ! -e "$CACHE_ROOT" && ! -L "$CACHE_ROOT" ]] || exit 2
mkdir -p "$CACHE_ROOT/wandb/cache" "$CACHE_ROOT/wandb/data" "$CACHE_ROOT/wandb/artifacts" "$CACHE_ROOT/tmp"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" OPENBLAS_NUM_THREADS=1
export WANDB_DIR="$CACHE_ROOT/wandb" WANDB_CACHE_DIR="$CACHE_ROOT/wandb/cache"
export WANDB_DATA_DIR="$CACHE_ROOT/wandb/data" WANDB_ARTIFACT_DIR="$CACHE_ROOT/wandb/artifacts"
export TMPDIR="$CACHE_ROOT/tmp" WANDB_DISABLE_CODE=true
unset WANDB_NAME WANDB_RUN_ID WANDB_RESUME WANDB_DISABLED PYTEST_ADDOPTS
export WANDB_MODE=online
[[ "$MODE" != smoke ]] || export WANDB_MODE=disabled WANDB_DISABLED=true

echo "AMBI-XQC MPPI $MODE: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}; node ${SLURM_NODELIST}; commit $EXPECTED_ACTION_MODES_SHA"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON_BIN" -c '
import sys, torch
assert sys.version_info[:2] == (3, 10)
assert torch.__version__.split("+")[0] == "2.3.1"
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert "L40S" in torch.cuda.get_device_name(0)
assert torch.ones(1, device="cuda").sum().item() == 1
print("Verified locked CUDA runtime", torch.__version__, torch.cuda.get_device_name(0))
'
if [[ "$MODE" == smoke ]]; then
  srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON_BIN" -m pytest -q -rs -p no:cacheprovider \
    tests/test_mppi_helper.py tests/test_xqc_mppi.py tests/test_ambixqc_mppi_evaluation.py \
    tests/test_ambixqc_controller.py tests/test_ambixqc_prior_checkpoint.py
fi
args=(--manifest "$CHECKPOINT_MANIFEST" --index "$SLURM_ARRAY_TASK_ID" --result-root "$RESULT_ROOT" --mode "$MODE" --device cuda)
[[ "$MODE" != production ]] || args+=(--wandb)
srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$PYTHON_BIN" run_ambixqc_mppi_evaluation.py "${args[@]}"
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]] || { echo "Source changed during evaluation" >&2; exit 2; }
echo "AMBI-XQC paired MPPI $MODE completed successfully."
