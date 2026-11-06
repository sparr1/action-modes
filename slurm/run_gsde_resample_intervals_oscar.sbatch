#!/usr/bin/env bash
#SBATCH --job-name=gsde-intervals
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=default
#SBATCH --partition=gpu
#SBATCH --qos=pri-gpu+
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Use Slurm}"
: "${AMBI_EXPECTED_COMMIT:?Pin source}"
: "${AMBI_GSDE_CAMPAIGN:?Set campaign scratch root}"
: "${AMBI_DMCONTROL_PYTHON:?Set locked Python}"
: "${AMBI_WORKSPACE_PYTHON:?Set publisher Python}"
index="${1:?arm index}"
[[ "$index" =~ ^[0-3]$ ]]
[[ "$AMBI_GSDE_CAMPAIGN" == /oscar/scratch/rgao48/ambi/gsde-resample-intervals-20261106 ]]
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
unset CUDA_LAUNCH_BLOCKING WANDB_RESUME
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
scratch_cache="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/ambi-gsde-intervals-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "$scratch_cache/inductor" "$scratch_cache/triton" "$scratch_cache/cache" "$scratch_cache/wandb" "$scratch_cache/data" "$scratch_cache/artifacts" "$scratch_cache/tmp"
export TMPDIR="$scratch_cache/tmp" TORCHINDUCTOR_CACHE_DIR="$scratch_cache/inductor" TRITON_CACHE_DIR="$scratch_cache/triton"
export XDG_CACHE_HOME="$scratch_cache/cache" WANDB_DIR="$scratch_cache/wandb" WANDB_CACHE_DIR="$scratch_cache/cache"
export WANDB_DATA_DIR="$scratch_cache/data" WANDB_ARTIFACT_DIR="$scratch_cache/artifacts"
export WANDB_DISABLE_CODE=true WANDB__REQUIRE_CORE=true WANDB_MODE=disabled
srun --kill-on-bad-exit=1 "$AMBI_DMCONTROL_PYTHON" slurm/gsde_resample_intervals_oscar.py validate \
  --index "$index" --output "$AMBI_GSDE_CAMPAIGN/gates/$index"
"$AMBI_WORKSPACE_PYTHON" slurm/gsde_noise_scopes_workspace.py "$AMBI_GSDE_CAMPAIGN/submission.json"
export WANDB_MODE=online WANDB_RESUME=never
export WANDB_RUN_ID="$("$AMBI_DMCONTROL_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["runs"][int(sys.argv[2])]["id"])' "$AMBI_GSDE_CAMPAIGN/submission.json" "$index")"
training_status=0
srun --kill-on-bad-exit=1 "$AMBI_DMCONTROL_PYTHON" slurm/gsde_resample_intervals_oscar.py train \
  --index "$index" --gate "$AMBI_GSDE_CAMPAIGN/gates/$index/receipt.json" \
  --output "$AMBI_GSDE_CAMPAIGN/production/$index" || training_status=$?
publisher_status=0
"$AMBI_WORKSPACE_PYTHON" slurm/gsde_noise_scopes_workspace.py "$AMBI_GSDE_CAMPAIGN/submission.json" || publisher_status=$?
if [[ "$training_status" != 0 ]]; then exit "$training_status"; fi
exit "$publisher_status"
