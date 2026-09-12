#!/usr/bin/env bash
#SBATCH --job-name=ambi-scratch-takeoff
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --no-requeue

# Submit a smoke first, then --array=0-13%<live-available-GPUs>. Tasks 0..6
# evaluate full episodes; 7..13 run real calibration at the same checkpoints.
# Supply persistent --output/--error paths outside the source checkout.
set -Eeuo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested commit}"
: "${AMBI_TAKEOFF_MANIFEST:?Supply the verified seven-checkpoint inventory}"
: "${AMBI_TAKEOFF_ROOT:?Supply the absolute campaign output root}"
: "${AMBI_TAKEOFF_ATTEMPT:?Supply an explicit new attempt label}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || { echo "Unexpected commit" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || { echo "Dirty checkout" >&2; exit 1; }
PYTHON_BIN="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing locked Python" >&2; exit 1; }
[[ "${AMBI_TAKEOFF_SMOKE:-0}" == 0 || "${AMBI_TAKEOFF_SMOKE:-0}" == 1 ]] || { echo "AMBI_TAKEOFF_SMOKE must be 0 or 1" >&2; exit 1; }
arguments=(worker --inventory "$AMBI_TAKEOFF_MANIFEST" --output-root "$AMBI_TAKEOFF_ROOT"
    --attempt-label "$AMBI_TAKEOFF_ATTEMPT" --task-index "${SLURM_ARRAY_TASK_ID:?Supply a task index}")
if [[ "${AMBI_TAKEOFF_SMOKE:-0}" == 1 ]]; then arguments+=(--smoke); fi
if [[ -n "${EVAL_RUN_MAP:-}" ]]; then arguments+=(--eval-run-map "$EVAL_RUN_MAP"); fi
if [[ -n "${AMBI_TAKEOFF_MATRIX:-}" ]]; then arguments+=(--matrix "$AMBI_TAKEOFF_MATRIX"); fi
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/ambi-takeoff-${SLURM_JOB_ID}"
# GPU workers never initialize W&B. Curve pointers are staged for a CPU owner.
export WANDB_MODE=offline
exec "$PYTHON_BIN" slurm/ambi_takeoff_campaign.py "${arguments[@]}"
