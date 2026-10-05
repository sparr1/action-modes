#!/usr/bin/env bash
#SBATCH --job-name=transfer-diagnostic-time
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${TIMING_ROOT:?}" "${CHECKPOINT_PATH:?}" "${EXPECTED_ACTION_MODES_SHA:?}"
PYTHON_BIN="${PYTHON_BIN:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
index="${SLURM_ARRAY_TASK_ID:?}"
[[ "$index" =~ ^[0-8]$ ]]
horizons=(1 2 3)
rounds=(1 4 10)
h="${horizons[$((index / 3))]}"
j="${rounds[$((index % 3))]}"
measurement_args=(--decisions 25 --max-steps 26 --mc-rollouts 32 --action-count 8
  --fit-steps 32 --fit-states 16 --real-rollouts 3 --real-tail-steps 1000
  --replan-steps 25 --replan-repeats 3)
if [[ "${TIMING_SMOKE:-0}" == 1 ]]; then
  [[ "$index" == 0 ]]
  measurement_args=(--decisions 1 --max-steps 2 --mc-rollouts 2 --action-count 4
    --fit-steps 1 --fit-states 4 --real-rollouts 1 --real-tail-steps 2
    --replan-steps 2 --replan-repeats 1)
fi
cell="$TIMING_ROOT/h${h}-j${j}"
mkdir "$cell"
runtime="$cell/runtime"
export TMPDIR="$runtime/tmp" XDG_CACHE_HOME="$runtime/cache"
export WANDB_DIR="$runtime/wandb" WANDB_CACHE_DIR="$runtime/wandb-cache"
export WANDB_DATA_DIR="$runtime/wandb-data" WANDB_MODE=disabled
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
unset WANDB_RUN_ID WANDB_NAME WANDB_GROUP WANDB_SERVICE _WANDB_SERVICE WANDB_RESUME
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS=1
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "$cell/hardware.csv"
/usr/bin/time -v -o "$cell/process.time" "$PYTHON_BIN" profile_ambi_transfer_diagnostics.py \
  --checkpoint "$CHECKPOINT_PATH" --horizons "$h" --rounds "$j" \
  --histories joint --seeds 101 --controller-seed 55 \
  --device cuda --data-lanes common natural --full-trace-probes "${measurement_args[@]}" \
  --save-snapshots --output-dir "$cell/diagnostics" --profile-dir "$cell/profile"
