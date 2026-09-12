#!/usr/bin/env bash
#SBATCH --job-name=ambi-round-scaling
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested commit}"
: "${AMBI_ROUND_SCALING_CAMPAIGN:?Supply the immutable campaign manifest}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || { echo "Unexpected commit" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || { echo "Dirty checkout" >&2; exit 1; }
PYTHON_BIN="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing locked Python" >&2; exit 1; }
[[ "${AMBI_ROUND_SCALING_SMOKE:-0}" == 0 || "${AMBI_ROUND_SCALING_SMOKE:-0}" == 1 ]] || exit 1
arguments=(worker --campaign "$AMBI_ROUND_SCALING_CAMPAIGN" --task-index "${SLURM_ARRAY_TASK_ID:?Supply task index}")
if [[ "${AMBI_ROUND_SCALING_SMOKE:-0}" == 1 ]]; then arguments+=(--smoke); fi
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/ambi-round-scaling-${SLURM_JOB_ID}"
export WANDB_MODE=offline
exec "$PYTHON_BIN" slurm/ambi_round_scaling_campaign.py "${arguments[@]}"
