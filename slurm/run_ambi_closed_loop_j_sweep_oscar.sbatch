#!/usr/bin/env bash
#SBATCH --job-name=closed-loop-j650
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --no-requeue

# Select CPU prepare/watch or GPU worker resources explicitly at submission.
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the synchronized repository root}"
: "${CAMPAIGN_ROOT:?Supply a new campaign directory}"
: "${EXPECTED_ACTION_MODES_SHA:?Pin the tested source commit}"
: "${EVAL_MODE:?Choose prepare, worker, or watch}"
PYTHON_BIN="${PYTHON_BIN:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]

# W&B artifact staging and compilation must not consume the home quota.
runtime_root="${CAMPAIGN_ROOT%/*}/runtime/${SLURM_JOB_ID:?}-${SLURM_ARRAY_TASK_ID:-single}"
export TMPDIR="$runtime_root/tmp" WANDB_DIR="$runtime_root/wandb"
export WANDB_CACHE_DIR="$runtime_root/wandb-cache" WANDB_DATA_DIR="$runtime_root/wandb-data"
export XDG_CACHE_HOME="$runtime_root/cache" TORCHINDUCTOR_CACHE_DIR="$runtime_root/inductor"
export TRITON_CACHE_DIR="$runtime_root/triton"
mkdir -p "$TMPDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" \
  "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
unset WANDB_RUN_ID WANDB_NAME WANDB_GROUP WANDB_SERVICE _WANDB_SERVICE
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" TORCHINDUCTOR_COMPILE_THREADS=1
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0

case "$EVAL_MODE" in
  prepare)
    args=(prepare --root "$CAMPAIGN_ROOT"
      --inventory "${CHECKPOINT_INVENTORY:?}" --references "${EVAL_REFERENCES_PATH:?}"
      --registry /oscar/scratch/rgao48/ambi/evaluation-series/runs)
    if [[ -n "${EVAL_MATRIX_PATH:-}" ]]; then args+=(--matrix "$EVAL_MATRIX_PATH"); fi
    if [[ -n "${EVAL_HORIZON:-}" ]]; then args+=(--horizon "$EVAL_HORIZON"); fi
    if [[ -n "${EVAL_CAMPAIGN_GROUP:-}" ]]; then args+=(--group "$EVAL_CAMPAIGN_GROUP"); fi
    if [[ -n "${EVAL_CAMPAIGN_LABEL:-}" ]]; then args+=(--label "$EVAL_CAMPAIGN_LABEL"); fi
    exec "$PYTHON_BIN" slurm/ambi_closed_loop_j_sweep.py "${args[@]}"
    ;;
  worker)
    args=(worker --root "$CAMPAIGN_ROOT" --index "${SLURM_ARRAY_TASK_ID:?}")
    if [[ "${EVAL_SMOKE:-0}" == 1 ]]; then args+=(--smoke); fi
    exec "$PYTHON_BIN" slurm/ambi_closed_loop_j_sweep.py "${args[@]}"
    ;;
  watch)
    exec "$PYTHON_BIN" slurm/ambi_closed_loop_j_sweep_publish.py "$EVAL_MODE" --root "$CAMPAIGN_ROOT"
    ;;
  *) echo "Unsupported EVAL_MODE: $EVAL_MODE" >&2; exit 2 ;;
esac
