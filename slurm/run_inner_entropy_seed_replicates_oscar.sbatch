#!/usr/bin/env bash
#SBATCH --job-name=posttanh-seed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=default
#SBATCH --gres=gpu:l40s:1
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Use Slurm}"
: "${AMBI_EXPECTED_COMMIT:?Pin the source commit}"
: "${AMBI_ENTROPY_CAMPAIGN:?Set a new campaign scratch root}"
: "${AMBI_REFERENCE_CAMPAIGN:?Set the original validated campaign}"
: "${AMBI_DMCONTROL_PYTHON:?Set the locked training interpreter}"
: "${AMBI_WORKSPACE_PYTHON:?Set the existing W&B workspace interpreter}"
: "${SLURM_CPUS_PER_TASK:?Use the measured CPU allocation}"
index="${1:?seed index 0 or 1}"
[[ "$index" =~ ^[01]$ ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
unset CUDA_LAUNCH_BLOCKING WANDB_RESUME
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
scratch_cache="$AMBI_ENTROPY_CAMPAIGN/cache/$SLURM_JOB_ID"
mkdir "$scratch_cache"
mkdir -p "$scratch_cache/"{inductor,triton,cache,wandb,data,artifacts,tmp}
export TMPDIR="$scratch_cache/tmp" TORCHINDUCTOR_CACHE_DIR="$scratch_cache/inductor" TRITON_CACHE_DIR="$scratch_cache/triton"
export XDG_CACHE_HOME="$scratch_cache/cache" WANDB_DIR="$scratch_cache/wandb" WANDB_CACHE_DIR="$scratch_cache/cache"
export WANDB_DATA_DIR="$scratch_cache/data" WANDB_ARTIFACT_DIR="$scratch_cache/artifacts"
export WANDB_DISABLE_CODE=true WANDB__REQUIRE_CORE=true
# Serialize publication so two seed jobs cannot create competing workspaces.
publish() {
  flock "$AMBI_ENTROPY_CAMPAIGN/workspace.lock" "$AMBI_WORKSPACE_PYTHON" \
    slurm/inner_entropy_workspace.py "$AMBI_ENTROPY_CAMPAIGN/submission.json"
}
publish
export WANDB_MODE=online
export WANDB_RUN_ID="$("$AMBI_DMCONTROL_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["runs"][int(sys.argv[2])]["id"])' "$AMBI_ENTROPY_CAMPAIGN/submission.json" "$index")"
srun --kill-on-bad-exit=1 "$AMBI_DMCONTROL_PYTHON" slurm/inner_entropy_seed_replicates.py "$index"
publish
