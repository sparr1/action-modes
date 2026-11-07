#!/usr/bin/env bash
#SBATCH --job-name=boom-calibrated
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
: "${AMBI_CALIBRATED_CAMPAIGN:?Set scratch root}"
: "${AMBI_DMCONTROL_PYTHON:?Set locked interpreter}"
mode="${1:?gate or train}"
shift
[[ "$mode" == gate || "$mode" == train ]]
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
unset CUDA_LAUNCH_BLOCKING WANDB_RESUME
if [[ "$mode" == gate ]]; then
    unset WANDB_RUN_ID
    export WANDB_MODE=disabled
else
    : "${WANDB_RUN_ID:?Assign unique training run ID}"
    export WANDB_MODE=online
fi
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
task_cache="$AMBI_CALIBRATED_CAMPAIGN/cache/$SLURM_JOB_ID"
mkdir -p "$task_cache"/{inductor,triton,cache,wandb,data,artifacts,tmp}
export TMPDIR="$task_cache/tmp" TORCHINDUCTOR_CACHE_DIR="$task_cache/inductor" TRITON_CACHE_DIR="$task_cache/triton"
export XDG_CACHE_HOME="$task_cache/cache" WANDB_DIR="$task_cache/wandb" WANDB_CACHE_DIR="$task_cache/cache"
export WANDB_DATA_DIR="$task_cache/data" WANDB_ARTIFACT_DIR="$task_cache/artifacts"
export WANDB_DISABLE_CODE=true WANDB__REQUIRE_CORE=true
srun --kill-on-bad-exit=1 "$AMBI_DMCONTROL_PYTHON" slurm/boom_likelihood_calibrated_oscar.py "$mode" "$@"
