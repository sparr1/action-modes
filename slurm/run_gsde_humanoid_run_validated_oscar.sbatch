#!/usr/bin/env bash
#SBATCH --job-name=gsde-humanoid-run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=default
#SBATCH --partition=gpu
#SBATCH --qos=pri-gpu+
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=59:00:00
#SBATCH --hint=nomultithread
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Use Slurm}"
: "${AMBI_EXPECTED_COMMIT:?Pin the source commit}"
: "${AMBI_GSDE_CAMPAIGN:?Set campaign scratch root}"
: "${AMBI_DMCONTROL_PYTHON:?Set the locked Python interpreter}"
: "${AMBI_WORKSPACE_PYTHON:?Set the W&B publisher interpreter}"
: "${AMBI_EXPECTED_LAUNCHER_COMMIT:?Pin the launcher commit}"
launcher_root="${AMBI_LAUNCHER_ROOT:?Set the pinned launcher checkout}"
[[ "$(git -C "$launcher_root" rev-parse HEAD)" == "$AMBI_EXPECTED_LAUNCHER_COMMIT" ]]
[[ -z "$(git -C "$launcher_root" status --porcelain --untracked-files=all)" ]]
[[ "$AMBI_GSDE_CAMPAIGN" == /oscar/scratch/rgao48/ambi/gsde-humanoid-run-20261108 ]]
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
unset CUDA_LAUNCH_BLOCKING WANDB_RESUME WANDB_RUN_ID
export AMBI_GSDE_VARIANT=humanoid_run WANDB_MODE=disabled
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
scratch_cache="$AMBI_GSDE_CAMPAIGN/cache/$SLURM_JOB_ID"
mkdir "$scratch_cache"
mkdir -p "$scratch_cache/inductor" "$scratch_cache/triton" "$scratch_cache/cache" "$scratch_cache/wandb" "$scratch_cache/data" "$scratch_cache/artifacts" "$scratch_cache/tmp"
export TMPDIR="$scratch_cache/tmp" TORCHINDUCTOR_CACHE_DIR="$scratch_cache/inductor" TRITON_CACHE_DIR="$scratch_cache/triton"
export XDG_CACHE_HOME="$scratch_cache/cache" WANDB_DIR="$scratch_cache/wandb" WANDB_CACHE_DIR="$scratch_cache/cache"
export WANDB_DATA_DIR="$scratch_cache/data" WANDB_ARTIFACT_DIR="$scratch_cache/artifacts"
export WANDB_DISABLE_CODE=true WANDB__REQUIRE_CORE=true
# The full-size Humanoid Run gate already passed on the unchanged pinned learner.
# Validate its persisted receipt in the production entry point rather than repeat it.
"$AMBI_WORKSPACE_PYTHON" slurm/gsde_noise_scopes_workspace.py "$AMBI_GSDE_CAMPAIGN/submission.json"
export WANDB_MODE=online WANDB_RESUME=never
export WANDB_RUN_ID="$("$AMBI_DMCONTROL_PYTHON" - "$AMBI_GSDE_CAMPAIGN/submission.json" <<'CHECK'
import json, os, sys
s=json.load(open(sys.argv[1])); arm=s['runs'][0]
assert s['commit']==os.environ['AMBI_EXPECTED_COMMIT']
assert str(arm['job'])==os.environ['SLURM_JOB_ID']
assert arm['allocated_hours']==59 and arm['cpus']==1 and arm['memory_gib']==8
assert s['workspace_api_verified'] and s['workspace_browser_verified']
print(arm['id'])
CHECK
)"
srun --kill-on-bad-exit=1 "$AMBI_DMCONTROL_PYTHON" slurm/gsde_noise_scopes_oscar.py train \
  --index 0 --gate "$AMBI_GSDE_CAMPAIGN/gates/0/receipt.json" \
  --output "$AMBI_GSDE_CAMPAIGN/production/0"
"$AMBI_WORKSPACE_PYTHON" slurm/gsde_noise_scopes_workspace.py "$AMBI_GSDE_CAMPAIGN/submission.json"
