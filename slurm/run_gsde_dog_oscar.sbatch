#!/usr/bin/env bash
#SBATCH --job-name=gsde8-dog
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
: "${AMBI_EXPECTED_COMMIT:?Pin source commit}"
: "${AMBI_DMCONTROL_PYTHON:?Set the locked training interpreter}"
: "${AMBI_GSDE_VARIANT:?dog_trot or dog_run}"
[[ "$AMBI_GSDE_VARIANT" == dog_trot || "$AMBI_GSDE_VARIANT" == dog_run ]]
[[ "$AMBI_GSDE_CAMPAIGN" == "/oscar/scratch/rgao48/ambi/gsde8-dog-trot-run-20261116/$AMBI_GSDE_VARIANT" ]]
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
mode="${1:?gate or train}"
unset CUDA_LAUNCH_BLOCKING WANDB_RESUME WANDB_RUN_ID
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
scratch_cache="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/ambi-gsde-dog-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "$scratch_cache/inductor" "$scratch_cache/triton" "$scratch_cache/cache" "$scratch_cache/wandb" "$scratch_cache/data" "$scratch_cache/artifacts" "$scratch_cache/tmp"
export TMPDIR="$scratch_cache/tmp" TORCHINDUCTOR_CACHE_DIR="$scratch_cache/inductor" TRITON_CACHE_DIR="$scratch_cache/triton"
export XDG_CACHE_HOME="$scratch_cache/cache" WANDB_DIR="$scratch_cache/wandb" WANDB_CACHE_DIR="$scratch_cache/cache"
export WANDB_DATA_DIR="$scratch_cache/data" WANDB_ARTIFACT_DIR="$scratch_cache/artifacts"
export WANDB_DISABLE_CODE=true WANDB__REQUIRE_CORE=true
if [[ "$mode" == gate ]]; then
  export WANDB_MODE=disabled
  srun --kill-on-bad-exit=1 "$AMBI_DMCONTROL_PYTHON" slurm/gsde_noise_scopes_oscar.py gate \
    --index 0 --output "$AMBI_GSDE_CAMPAIGN/gates/0"
elif [[ "$mode" == train ]]; then
  : "${AMBI_WORKSPACE_PYTHON:?Set publisher interpreter}"
  "$AMBI_WORKSPACE_PYTHON" slurm/gsde_noise_scopes_workspace.py "$AMBI_GSDE_CAMPAIGN/submission.json"
  export WANDB_MODE=online WANDB_RESUME=never
  export WANDB_RUN_ID="$("$AMBI_DMCONTROL_PYTHON" - "$AMBI_GSDE_CAMPAIGN/submission.json" <<'CHECK'
import json, os, sys
s=json.load(open(sys.argv[1])); arm=s['runs'][0]
assert s['commit']==os.environ['AMBI_EXPECTED_COMMIT']
assert str(arm['job'])==os.environ['SLURM_JOB_ID']
assert s['workspace_api_verified'] and s['workspace_browser_verified']
assert arm['cpus']==int(os.environ['SLURM_CPUS_PER_TASK'])
print(arm['id'])
CHECK
)"
  training_status=0
  srun --kill-on-bad-exit=1 "$AMBI_DMCONTROL_PYTHON" slurm/gsde_noise_scopes_oscar.py train \
    --index 0 --gate "$AMBI_GSDE_CAMPAIGN/gates/0/receipt.json" \
    --output "$AMBI_GSDE_CAMPAIGN/production/0" || training_status=$?
  publisher_status=0
  "$AMBI_WORKSPACE_PYTHON" slurm/gsde_noise_scopes_workspace.py "$AMBI_GSDE_CAMPAIGN/submission.json" || publisher_status=$?
  if [[ "$training_status" != 0 ]]; then exit "$training_status"; fi
  exit "$publisher_status"
else
  echo "Unknown mode: $mode" >&2
  exit 2
fi
