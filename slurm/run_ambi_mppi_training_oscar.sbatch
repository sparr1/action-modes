#!/usr/bin/env bash
#SBATCH --job-name=ambi-mppi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --hint=nomultithread
#SBATCH --no-requeue

# No default CPU, GPU, memory or walltime reservations: supply the actual
# allocation explicitly to sbatch after checking live QOS headroom. The short
# gate needs 2 physical CPUs and one GPU; production needs its measured 1 or 2
# CPUs, that same GPU model, and a throughput-derived walltime.
# Run AMBI_MPPI_MODE=smoke first, then train --array=0-1 with its receipt.
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${SLURM_CPUS_PER_TASK:?Specify a measured CPU allocation in sbatch.}"
: "${AMBI_PROJECT_DIR:?Set the clean source checkout.}"
: "${AMBI_PYTHON:?Set the locked DMControl interpreter.}"
: "${AMBI_EXPECTED_COMMIT:?Set the full tested commit.}"
: "${AMBI_MPPI_OUTPUT:?Set campaign storage outside source.}"
mode="${AMBI_MPPI_MODE:-train}"
manifest="${AMBI_MPPI_MANIFEST:-configs/dmcontrol/experiments/ambi_mppi_training.json}"
[[ "$mode" == smoke || "$mode" == train ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
[[ "$AMBI_PROJECT_DIR" == /* && "$AMBI_PYTHON" == /* && "$AMBI_MPPI_OUTPUT" == /* ]]
cd "$AMBI_PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "$AMBI_MPPI_OUTPUT" != "$PWD" && "$AMBI_MPPI_OUTPUT" != "$PWD/"* ]]
mkdir -p "$AMBI_MPPI_OUTPUT"
available_kb="$(df -Pk "$AMBI_MPPI_OUTPUT" | awk 'NR==2 {print $4}')"
(( available_kb >= 10 * 1024 * 1024 ))
local_root="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/ambi-mppi-${SLURM_JOB_ID}.XXXXXX")"
wandb_root="$AMBI_MPPI_OUTPUT/wandb-job-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-gate}"
mkdir -p "$local_root/inductor" "$local_root/triton" "$local_root/cache" "$local_root/data" "$local_root/artifacts" "$local_root/tmp" "$wandb_root"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$local_root/inductor" TRITON_CACHE_DIR="$local_root/triton"
export XDG_CACHE_HOME="$local_root/cache" TMPDIR="$local_root/tmp"
# Keep the sync journal after Slurm removes node-local temporary storage.
export WANDB_DISABLE_CODE=true WANDB_DIR="$wandb_root" WANDB_CACHE_DIR="$local_root/cache"
export WANDB_DATA_DIR="$local_root/data" WANDB_ARTIFACT_DIR="$local_root/artifacts"
if [[ "$mode" == smoke ]]; then
  [[ -z "${SLURM_ARRAY_TASK_ID:-}" && "$SLURM_CPUS_PER_TASK" == 2 ]]
  export WANDB_MODE=disabled
  exec srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" slurm/ambi_mppi_training.py smoke \
    --manifest "$manifest" --output "$AMBI_MPPI_OUTPUT/gate"
else
  : "${SLURM_ARRAY_TASK_ID:?Train as a two-cell array.}"
  [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-1]$ ]]
  [[ "$SLURM_CPUS_PER_TASK" == 1 || "$SLURM_CPUS_PER_TASK" == 2 ]]
  export WANDB_MODE=online WANDB_RESUME=never
  export WANDB_RUN_ID="mppi${SLURM_ARRAY_JOB_ID}x${SLURM_ARRAY_TASK_ID}"
  exec srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" slurm/ambi_mppi_training.py train \
    --manifest "$manifest" --index "$SLURM_ARRAY_TASK_ID" --output "$AMBI_MPPI_OUTPUT/cell-${SLURM_ARRAY_TASK_ID}" \
    --gate "$AMBI_MPPI_OUTPUT/gate/receipt.json"
fi
