#!/usr/bin/env bash
#SBATCH --job-name=warm-calibration575
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --no-requeue

# Submit resources and array concurrency using current Oscar capacity. This
# dispatcher intentionally supplies neither a GPU partition nor an array cap.
set -Eeuo pipefail
: "${CAMPAIGN_ROOT:?Set the campaign scratch directory}"
: "${EXPECTED_ACTION_MODES_SHA:?Pin the tested source commit}"
: "${EVAL_MODE:?Choose prepare, worker, report, publish, or watch}"
SOURCE_DIR="${SOURCE_DIR:-${SLURM_SUBMIT_DIR:?}}"
PYTHON_BIN="${PYTHON_BIN:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
cd "$SOURCE_DIR"
[[ -x "$PYTHON_BIN" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]

runtime_root="${CAMPAIGN_ROOT%/*}/runtime/${SLURM_JOB_ID:?}-${SLURM_ARRAY_TASK_ID:-single}"
export TMPDIR="$runtime_root/tmp" WANDB_DIR="$runtime_root/wandb"
export WANDB_CACHE_DIR="$runtime_root/wandb-cache" WANDB_DATA_DIR="$runtime_root/wandb-data"
export XDG_CACHE_HOME="$runtime_root/cache" TORCHINDUCTOR_CACHE_DIR="$runtime_root/inductor"
export TRITON_CACHE_DIR="$runtime_root/triton" MPLCONFIGDIR="$runtime_root/matplotlib"
mkdir -p "$TMPDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" \
  "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
unset WANDB_RUN_ID WANDB_NAME WANDB_GROUP WANDB_SERVICE _WANDB_SERVICE
unset WANDB_RESUME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" TORCHINDUCTOR_COMPILE_THREADS=1
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0

case "$EVAL_MODE" in
  prepare)
    export WANDB_MODE=disabled
    exec "$PYTHON_BIN" slurm/warm_actor_calibration_campaign.py prepare --root "$CAMPAIGN_ROOT" "$@"
    ;;
  worker)
    export WANDB_MODE=disabled
    : "${KIND:?Choose capture, prefix, or replan}"
    case "$KIND" in capture|prefix|replan|smoke) ;; *) echo "Invalid KIND: $KIND" >&2; exit 2 ;; esac
    exec "$PYTHON_BIN" slurm/warm_actor_calibration_campaign.py worker --root "$CAMPAIGN_ROOT" \
      --kind "$KIND" --index "${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:?}}" "$@"
    ;;
  report|publish|watch)
    export OMP_NUM_THREADS=1
    if [[ "$EVAL_MODE" == report ]]; then export WANDB_MODE=disabled; else export WANDB_MODE=online; fi
    exec "$PYTHON_BIN" slurm/warm_actor_calibration_publish.py "$EVAL_MODE" --root "$CAMPAIGN_ROOT" "$@"
    ;;
  *) echo "Invalid EVAL_MODE: $EVAL_MODE" >&2; exit 2 ;;
esac
