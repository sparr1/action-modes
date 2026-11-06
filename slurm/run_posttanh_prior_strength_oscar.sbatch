#!/usr/bin/env bash
#SBATCH --job-name=prior-strength
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=default
#SBATCH --partition=gpu
#SBATCH --qos=pri-gpu+
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Use Slurm}"
: "${AMBI_EXPECTED_COMMIT:?Pin source}"
: "${AMBI_PRIOR_CAMPAIGN:?Set campaign scratch root}"
: "${AMBI_DMCONTROL_PYTHON:?Set locked interpreter}"
: "${AMBI_WORKSPACE_PYTHON:?Set W&B publisher interpreter}"
index="${1:?arm index}"
export AMBI_PRIOR_VARIANT="${AMBI_PRIOR_VARIANT:-strength}"
case "$AMBI_PRIOR_VARIANT" in
  strength)
    [[ "$index" =~ ^[0-3]$ ]]
    [[ "$AMBI_PRIOR_CAMPAIGN" == /oscar/scratch/rgao48/ambi/posttanh-prior-strength-20261105 ]]
    ;;
  strength_x003)
    [[ "$index" == 0 ]]
    [[ "$AMBI_PRIOR_CAMPAIGN" == /oscar/scratch/rgao48/ambi/posttanh-prior-strength-x003-20261106 ]]
    ;;
  *) exit 2 ;;
esac
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
unset CUDA_LAUNCH_BLOCKING WANDB_RESUME WANDB_RUN_ID
export WANDB_MODE=disabled
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
scratch_cache="$AMBI_PRIOR_CAMPAIGN/cache/$SLURM_JOB_ID"
mkdir "$scratch_cache"
mkdir -p "$scratch_cache/inductor" "$scratch_cache/triton" "$scratch_cache/cache" "$scratch_cache/wandb" "$scratch_cache/data" "$scratch_cache/artifacts" "$scratch_cache/tmp"
export TMPDIR="$scratch_cache/tmp" TORCHINDUCTOR_CACHE_DIR="$scratch_cache/inductor" TRITON_CACHE_DIR="$scratch_cache/triton"
export XDG_CACHE_HOME="$scratch_cache/cache" WANDB_DIR="$scratch_cache/wandb" WANDB_CACHE_DIR="$scratch_cache/cache"
export WANDB_DATA_DIR="$scratch_cache/data" WANDB_ARTIFACT_DIR="$scratch_cache/artifacts"
export WANDB_DISABLE_CODE=true WANDB__REQUIRE_CORE=true
srun --kill-on-bad-exit=1 "$AMBI_DMCONTROL_PYTHON" slurm/posttanh_prior_strength_oscar.py start --index "$index"
flock "$AMBI_PRIOR_CAMPAIGN/workspace.lock" "$AMBI_WORKSPACE_PYTHON" slurm/posttanh_prior_fixes_workspace.py "$AMBI_PRIOR_CAMPAIGN/submission.json"
