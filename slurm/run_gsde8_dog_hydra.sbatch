#!/usr/bin/env bash
#SBATCH --job-name=gsde8-dog-walk
#SBATCH --partition=gpus
#SBATCH --account=default
#SBATCH --nodelist=gpu2501
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Use Slurm}"
: "${AMBI_EXPECTED_COMMIT:?Pin the source}"
: "${AMBI_GSDE_CAMPAIGN:?Set node-local campaign storage}"
: "${AMBI_DMCONTROL_PYTHON:?Set locked Python}"
mode="${AMBI_GSDE_MODE:-gate}"
[[ "$mode" == gate || "$mode" == train ]]
[[ "$(hostname -s)" == gpu2501 && "$AMBI_GSDE_CAMPAIGN" == /ltmp/rgao48/* ]]
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
unset CUDA_LAUNCH_BLOCKING WANDB_RESUME
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
scratch_cache="$AMBI_GSDE_CAMPAIGN/cache/$SLURM_JOB_ID"
mkdir -p "$scratch_cache"/{inductor,triton,cache,wandb,data,artifacts,tmp}
export TMPDIR="$scratch_cache/tmp" TORCHINDUCTOR_CACHE_DIR="$scratch_cache/inductor" TRITON_CACHE_DIR="$scratch_cache/triton"
export XDG_CACHE_HOME="$scratch_cache/cache" WANDB_DIR="$scratch_cache/wandb" WANDB_CACHE_DIR="$scratch_cache/cache"
export WANDB_DATA_DIR="$scratch_cache/data" WANDB_ARTIFACT_DIR="$scratch_cache/artifacts"
export WANDB_DISABLE_CODE=true WANDB__REQUIRE_CORE=true
if [[ "$mode" == gate ]]; then
  export WANDB_MODE=disabled
  release_args=()
  if [[ -n "${AMBI_TRAIN_JOB_ID:-}" ]]; then release_args=(--release-job "$AMBI_TRAIN_JOB_ID"); fi
  exec "$AMBI_DMCONTROL_PYTHON" slurm/gsde8_dog_hydra.py gate --output "$AMBI_GSDE_CAMPAIGN/gate" "${release_args[@]}"
else
  : "${WANDB_RUN_ID:?Set the unique training identity}"
  export WANDB_MODE=online
  exec "$AMBI_DMCONTROL_PYTHON" slurm/gsde8_dog_hydra.py train --gate "$AMBI_GSDE_CAMPAIGN/gate/receipt.json" --output "$AMBI_GSDE_CAMPAIGN/production/0"
fi
