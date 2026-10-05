#!/usr/bin/env bash
#SBATCH --job-name=transfer-curves
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from synchronized repository}"
: "${CAMPAIGN_ROOT:?}" "${EXPECTED_ACTION_MODES_SHA:?}"
PYTHON_BIN="${PYTHON_BIN:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
runtime_root="${CAMPAIGN_ROOT%/*}/runtime/${SLURM_JOB_ID:?}-${SLURM_ARRAY_TASK_ID:?}"
export TMPDIR="$runtime_root/tmp" XDG_CACHE_HOME="$runtime_root/cache"
export TORCHINDUCTOR_CACHE_DIR="$runtime_root/inductor" TRITON_CACHE_DIR="$runtime_root/triton"
export WANDB_DIR="$runtime_root/wandb" WANDB_CACHE_DIR="$runtime_root/wandb-cache"
export WANDB_DATA_DIR="$runtime_root/wandb-data" WANDB_MODE=disabled
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
unset WANDB_RUN_ID WANDB_NAME WANDB_GROUP WANDB_SERVICE _WANDB_SERVICE WANDB_RESUME
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
args=(worker --root "$CAMPAIGN_ROOT" --index "$SLURM_ARRAY_TASK_ID" --expected-source-sha "$EXPECTED_ACTION_MODES_SHA")
if [[ "${EVAL_SMOKE:-0}" == 1 ]]; then args+=(--smoke); fi
exec "$PYTHON_BIN" slurm/ambi_transfer_curve_campaign.py "${args[@]}"
