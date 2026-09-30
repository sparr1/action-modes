#!/usr/bin/env bash
#SBATCH --job-name=axqc-update-sweep
#SBATCH --partition=gpu
#SBATCH --qos=pri-gpu+
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
: "${AMBI_EXECUTION_ROOT:?Original clean scientific checkout required}"
: "${AMBI_SOURCE_SHA:?Original full source commit required}"
: "${AMBI_TOOLING_ROOT:?Clean tooling checkout required}"
: "${AMBI_TOOLING_SHA:?Full tooling commit required}"
: "${AMBI_UPDATE_SWEEP_PLAN:?Immutable update-sweep plan required}"
: "${AMBIXQC_PYTHON:?Existing locked Python required}"
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${SLURM_ARRAY_TASK_ID:?Submit explicit update-sweep array}"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]]
[[ "$(git -C "$AMBI_EXECUTION_ROOT" rev-parse HEAD)" == "$AMBI_SOURCE_SHA" ]]
[[ "$(git -C "$AMBI_TOOLING_ROOT" rev-parse HEAD)" == "$AMBI_TOOLING_SHA" ]]
[[ -z "$(git -C "$AMBI_EXECUTION_ROOT" status --porcelain=v1 --untracked-files=all)" ]]
[[ -z "$(git -C "$AMBI_TOOLING_ROOT" status --porcelain=v1 --untracked-files=all)" ]]
readonly ENV_PROJECT="$(cd "$(dirname "$AMBIXQC_PYTHON")/../.." && pwd -P)"
readonly LOCK_SHA=f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6
for lock in "$AMBI_EXECUTION_ROOT/environments/dmcontrol/uv.lock" "$ENV_PROJECT/uv.lock"; do
  [[ "$(sha256sum "$lock" | awk '{print $1}')" == "$LOCK_SHA" ]]
done
readonly TASK_CACHE="${SLURM_TMPDIR:-/tmp}/ambixqc-update-sweep-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
[[ ! -e "$TASK_CACHE" ]]
mkdir -p "$TASK_CACHE/wandb" "$TASK_CACHE/cache" "$TASK_CACHE/data" "$TASK_CACHE/artifacts" \
  "$TASK_CACHE/config" "$TASK_CACHE/tmp" "$TASK_CACHE/torchinductor" "$TASK_CACHE/triton" \
  "$TASK_CACHE/cuda" "$TASK_CACHE/matplotlib"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" OPENBLAS_NUM_THREADS=1
export WANDB_DIR="$TASK_CACHE/wandb" WANDB_CACHE_DIR="$TASK_CACHE/cache" WANDB_DATA_DIR="$TASK_CACHE/data"
export WANDB_ARTIFACT_DIR="$TASK_CACHE/artifacts" WANDB_CONFIG_DIR="$TASK_CACHE/config" WANDB_CONFIG_PATHS=""
export XDG_CACHE_HOME="$TASK_CACHE/cache" XDG_CONFIG_HOME="$TASK_CACHE/config" TMPDIR="$TASK_CACHE/tmp"
export TORCHINDUCTOR_CACHE_DIR="$TASK_CACHE/torchinductor" TRITON_CACHE_DIR="$TASK_CACHE/triton"
export CUDA_CACHE_PATH="$TASK_CACHE/cuda" MPLCONFIGDIR="$TASK_CACHE/matplotlib"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_DISABLE_CODE=true
unset TDMPC2_EVAL_CSV TDMPC2_COMPUTE_TIMING_OUTPUT WANDB_NAME WANDB_RUN_ID WANDB_RESUME PYTEST_ADDOPTS
cd "$AMBI_EXECUTION_ROOT"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
srun --unbuffered --kill-on-bad-exit=1 --ntasks=1 "$AMBIXQC_PYTHON" \
  "$AMBI_TOOLING_ROOT/run_ambixqc_update_sweep.py" worker \
  --execution-root "$AMBI_EXECUTION_ROOT" --source-sha "$AMBI_SOURCE_SHA" \
  --tooling-sha "$AMBI_TOOLING_SHA" --plan "$AMBI_UPDATE_SWEEP_PLAN" --index "$SLURM_ARRAY_TASK_ID"
[[ -z "$(git -C "$AMBI_EXECUTION_ROOT" status --porcelain=v1 --untracked-files=all)" ]]
[[ -z "$(git -C "$AMBI_TOOLING_ROOT" status --porcelain=v1 --untracked-files=all)" ]]
