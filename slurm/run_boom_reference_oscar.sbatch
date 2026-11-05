#!/usr/bin/env bash
#SBATCH --job-name=boom-reference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=default
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=16G
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Use Slurm}"
: "${BOOM_OBSERVER_COMMIT:?Pin observer source}"
: "${BOOM_REFERENCE_ROOT:?Set scratch root}"
: "${BOOM_REFERENCE_PYTHON:?Set isolated upstream interpreter}"
mode="${1:?validate, benchmark, or train}"
[[ "$(git rev-parse HEAD)" == "$BOOM_OBSERVER_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1 WANDB_DISABLE_CODE=true
cache="$BOOM_REFERENCE_ROOT/cache/$SLURM_JOB_ID"
mkdir -p "$cache/inductor" "$cache/triton" "$cache/tmp" "$cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="$cache/inductor" TRITON_CACHE_DIR="$cache/triton"
export TMPDIR="$cache/tmp" XDG_CACHE_HOME="$cache/xdg"
if [[ "$mode" == train ]]; then
  : "${WANDB_RUN_ID:?Set unique scientific run ID}"
  : "${BOOM_WORKSPACE_PYTHON:?Set publisher interpreter}"
  "$BOOM_WORKSPACE_PYTHON" slurm/boom_reference_workspace.py "$BOOM_REFERENCE_ROOT"
  export WANDB_MODE=online
elif [[ "$mode" == validate || "$mode" == benchmark ]]; then
  export WANDB_MODE=disabled
else
  exit 2
fi
srun --kill-on-bad-exit=1 "$BOOM_REFERENCE_PYTHON" slurm/boom_upstream_reference.py "$mode" \
  --upstream "$BOOM_REFERENCE_ROOT/upstream" --output "$BOOM_REFERENCE_ROOT/$mode-$SLURM_JOB_ID"
if [[ "$mode" == train ]]; then
  "$BOOM_WORKSPACE_PYTHON" slurm/boom_reference_workspace.py "$BOOM_REFERENCE_ROOT"
fi

