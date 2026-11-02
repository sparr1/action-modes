#!/usr/bin/env bash
#SBATCH --job-name=inner-entropy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=default
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=16G
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Use Slurm}"
: "${AMBI_EXPECTED_COMMIT:?Pin the source commit}"
: "${AMBI_ENTROPY_CAMPAIGN:?Set campaign scratch root}"
: "${AMBI_DMCONTROL_PYTHON:?Set the locked Python interpreter}"
mode="${1:?gate or train}"
index="${2:?arm index}"
[[ "$index" =~ ^[0-5]$ ]]
[[ "$AMBI_ENTROPY_CAMPAIGN" == /oscar/scratch/rgao48/ambi/inner-entropy-20261102 ]]
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
unset CUDA_LAUNCH_BLOCKING WANDB_RESUME
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
scratch_cache="$AMBI_ENTROPY_CAMPAIGN/cache/$SLURM_JOB_ID"
mkdir "$scratch_cache"
mkdir -p "$scratch_cache/inductor" "$scratch_cache/triton" "$scratch_cache/cache" "$scratch_cache/wandb" "$scratch_cache/data" "$scratch_cache/artifacts" "$scratch_cache/tmp"
export TMPDIR="$scratch_cache/tmp" TORCHINDUCTOR_CACHE_DIR="$scratch_cache/inductor" TRITON_CACHE_DIR="$scratch_cache/triton"
export XDG_CACHE_HOME="$scratch_cache/cache" WANDB_DIR="$scratch_cache/wandb" WANDB_CACHE_DIR="$scratch_cache/cache"
export WANDB_DATA_DIR="$scratch_cache/data" WANDB_ARTIFACT_DIR="$scratch_cache/artifacts"
export WANDB_DISABLE_CODE=true WANDB__REQUIRE_CORE=true
if [[ "$mode" == gate ]]; then
  export WANDB_MODE=disabled
  srun --kill-on-bad-exit=1 "$AMBI_DMCONTROL_PYTHON" slurm/inner_entropy_oscar.py gate \
    --index "$index" --output "$AMBI_ENTROPY_CAMPAIGN/gates/$index"
elif [[ "$mode" == train ]]; then
  : "${AMBI_WORKSPACE_PYTHON:?Set the isolated W&B publisher interpreter}"
  "$AMBI_WORKSPACE_PYTHON" slurm/inner_entropy_workspace.py "$AMBI_ENTROPY_CAMPAIGN/submission.json"
  export WANDB_MODE=online
  export WANDB_RUN_ID="$("$AMBI_DMCONTROL_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["runs"][int(sys.argv[2])]["id"])' "$AMBI_ENTROPY_CAMPAIGN/submission.json" "$index")"
  srun --kill-on-bad-exit=1 "$AMBI_DMCONTROL_PYTHON" slurm/inner_entropy_oscar.py train \
    --index "$index" --gate "$AMBI_ENTROPY_CAMPAIGN/gates/$index/receipt.json" \
    --output "$AMBI_ENTROPY_CAMPAIGN/production/$index"
  "$AMBI_WORKSPACE_PYTHON" slurm/inner_entropy_workspace.py "$AMBI_ENTROPY_CAMPAIGN/submission.json"
else
  echo "Unknown launch mode: $mode" >&2
  exit 2
fi
