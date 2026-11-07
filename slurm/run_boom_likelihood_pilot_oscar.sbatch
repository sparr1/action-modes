#!/usr/bin/env bash
#SBATCH --job-name=boom-likelihood
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=default
#SBATCH --partition=gpu
#SBATCH --qos=pri-gpu+
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=16G
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Use Slurm}"
: "${AMBI_EXPECTED_COMMIT:?Pin source}"
: "${AMBI_PRIOR_CAMPAIGN:?Set campaign scratch root}"
: "${AMBI_DMCONTROL_PYTHON:?Set locked interpreter}"
mode="${1:?gate or train}"
shift
[[ "$mode" == gate || "$mode" == train ]]
[[ "$AMBI_PRIOR_CAMPAIGN" == /oscar/scratch/rgao48/ambi/boom-likelihood-pilot-20261107 ]]
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
unset CUDA_LAUNCH_BLOCKING WANDB_RESUME
export AMBI_PRIOR_VARIANT=likelihood
if [[ "$mode" == gate ]]; then
    unset WANDB_RUN_ID
    export WANDB_MODE=disabled
else
    : "${WANDB_RUN_ID:?Assign a unique training run ID}"
    export WANDB_MODE=online
fi
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
srun --kill-on-bad-exit=1 "$AMBI_DMCONTROL_PYTHON" slurm/boom_likelihood_pilot_oscar.py "$mode" "$@"
if [[ -n "${AMBI_WORKSPACE_PYTHON:-}" ]]; then
    flock "$AMBI_PRIOR_CAMPAIGN/workspace.lock" "$AMBI_WORKSPACE_PYTHON" slurm/posttanh_prior_fixes_workspace.py "$AMBI_PRIOR_CAMPAIGN/submission.json"
fi
