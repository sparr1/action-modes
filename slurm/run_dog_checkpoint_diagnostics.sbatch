#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=default
#SBATCH --partition=gpu
#SBATCH --qos=pri-gpu+
#SBATCH --gres=gpu:l40s:1
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Use Slurm}"
: "${AMBI_EXPECTED_COMMIT:?Pin source}"
: "${AMBI_DIAGNOSTIC_ROOT:?Set output root}"
: "${AMBI_DMCONTROL_PYTHON:?Use locked Python}"
: "${SLURM_CPUS_PER_TASK:?Use measured CPU request}"
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
export WANDB_MODE=disabled MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
task_cache="$AMBI_DIAGNOSTIC_ROOT/cache/$SLURM_JOB_ID"
mkdir -p "$task_cache/"{tmp,inductor,triton,xdg}
export TMPDIR="$task_cache/tmp" XDG_CACHE_HOME="$task_cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="$task_cache/inductor" TRITON_CACHE_DIR="$task_cache/triton"
srun --kill-on-bad-exit=1 "$AMBI_DMCONTROL_PYTHON" diagnose_dog_checkpoints.py "$@" --threads "$SLURM_CPUS_PER_TASK"
