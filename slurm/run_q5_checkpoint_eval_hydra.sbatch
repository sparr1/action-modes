#!/usr/bin/env bash
#SBATCH --job-name=q5-checkpoint-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
# Set --time from the measured smoke throughput when submitting.
set -Eeuo pipefail
umask 077
: "${AMBI_EVAL_CAMPAIGN:?Set campaign root in public storage}"
: "${AMBI_EXPECTED_COMMIT:?Pin source}"
: "${AMBI_DMCONTROL_PYTHON:?Set locked interpreter}"
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain)" ]]
unset CUDA_LAUNCH_BLOCKING WANDB_RESUME
export WANDB_MODE=disabled MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
task_cache="${SLURM_TMPDIR:-/tmp}/q5-eval-${UID}-${SLURM_JOB_ID:?}-${SLURM_ARRAY_TASK_ID:-smoke}"
mkdir -p "$task_cache/inductor" "$task_cache/triton" "$task_cache/tmp" "$task_cache/cache" "$task_cache/wandb" "$task_cache/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="$task_cache/inductor" TRITON_CACHE_DIR="$task_cache/triton" TMPDIR="$task_cache/tmp"
export XDG_CACHE_HOME="$task_cache/cache" WANDB_DIR="$task_cache/wandb" WANDB_CACHE_DIR="$task_cache/cache"
export MPLCONFIGDIR="$task_cache/matplotlib" CUDA_CACHE_PATH="$task_cache/cache/cuda"
args=(--campaign "$AMBI_EVAL_CAMPAIGN" --index "${SLURM_ARRAY_TASK_ID:-0}")
if [[ "${1:-}" == smoke ]]; then
    srun --kill-on-bad-exit=1 "$AMBI_DMCONTROL_PYTHON" -m pytest -q -p no:cacheprovider \
        tests/test_q5_checkpoint_evaluation.py::test_cuda_q5_humanoid_mean_graph_parity
    args+=(--smoke)
fi
srun --kill-on-bad-exit=1 "$AMBI_DMCONTROL_PYTHON" slurm/graph_checkpoint_campaign.py worker "${args[@]}"
