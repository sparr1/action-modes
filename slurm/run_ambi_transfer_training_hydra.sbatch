#!/usr/bin/env bash
#SBATCH --job-name=ambi-transfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpus
#SBATCH --constraint=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --no-requeue

# Submit one AMBI_TRANSFER_MODE=smoke job, then all six training cells with
# --array=0-5 --dependency=afterok:<smoke>. Supply absolute output/error paths.
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${AMBI_PROJECT_DIR:?Set the clean source checkout.}"
: "${AMBI_PYTHON:?Set the locked DMControl interpreter.}"
: "${AMBI_EXPECTED_COMMIT:?Set the full tested commit.}"
: "${AMBI_TRANSFER_OUTPUT:?Set durable campaign storage outside source.}"
mode="${AMBI_TRANSFER_MODE:-train}"
[[ "$mode" == smoke || "$mode" == train ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
[[ "$AMBI_PROJECT_DIR" == /* && "$AMBI_PYTHON" == /* && "$AMBI_TRANSFER_OUTPUT" == /* ]]
cd "$AMBI_PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "$AMBI_TRANSFER_OUTPUT" != "$PWD" && "$AMBI_TRANSFER_OUTPUT" != "$PWD/"* ]]
mkdir -p "$AMBI_TRANSFER_OUTPUT"
available_kb="$(df -Pk "$AMBI_TRANSFER_OUTPUT" | awk 'NR==2 {print $4}')"
(( available_kb >= 10 * 1024 * 1024 ))
local_root="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/ambi-transfer-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "$local_root/inductor" "$local_root/triton" "$local_root/cache" "$local_root/wandb" "$local_root/data" "$local_root/artifacts" "$local_root/tmp"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$local_root/inductor" TRITON_CACHE_DIR="$local_root/triton"
export XDG_CACHE_HOME="$local_root/cache" TMPDIR="$local_root/tmp"
export WANDB_DISABLE_CODE=true WANDB_DIR="$local_root/wandb" WANDB_CACHE_DIR="$local_root/cache"
export WANDB_DATA_DIR="$local_root/data" WANDB_ARTIFACT_DIR="$local_root/artifacts"
if [[ "$mode" == smoke ]]; then
  [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]
  export WANDB_MODE=disabled
  srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" slurm/ambi_transfer_training.py smoke --output "$AMBI_TRANSFER_OUTPUT/gate"
else
  : "${SLURM_ARRAY_TASK_ID:?Train as a six-cell array.}"
  [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-5]$ ]]
  export WANDB_MODE=online WANDB_RESUME=never
  export WANDB_RUN_ID="full${SLURM_ARRAY_JOB_ID}x${SLURM_ARRAY_TASK_ID}"
  exec srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" slurm/ambi_transfer_training.py train \
    --index "$SLURM_ARRAY_TASK_ID" --output "$AMBI_TRANSFER_OUTPUT/cell-${SLURM_ARRAY_TASK_ID}" \
    --gate "$AMBI_TRANSFER_OUTPUT/gate/receipt.json"
fi
