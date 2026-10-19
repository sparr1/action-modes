#!/usr/bin/env bash
#SBATCH --job-name=ambi-h3-j8-k3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --hint=nomultithread
#SBATCH --requeue

# Supply CPU/GPU/memory/walltime explicitly from the measured gate and live QOS.
# Smoke uses 3 physical CPUs; production uses a measured count (1,2,3).
# Add --signal=USR1@900 only to the production submission as a drain fallback.
# No array: this is exactly one seed55 aux-return H3/J8/K3 training lineage.
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${SLURM_CPUS_PER_TASK:?Specify measured CPU resources.}"
: "${AMBI_PROJECT_DIR:?Set the clean pinned source checkout.}"
: "${AMBI_PYTHON:?Set the locked DMControl interpreter.}"
: "${AMBI_EXPECTED_COMMIT:?Set the full tested commit.}"
: "${AMBI_HELD_OUTPUT:?Set persistent campaign output outside source.}"
mode="${AMBI_HELD_MODE:-train}"
manifest="${AMBI_HELD_MANIFEST:-configs/dmcontrol/experiments/ambi_held_training.json}"
[[ "$mode" == smoke || "$mode" == train ]]
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]
[[ "$AMBI_PROJECT_DIR" == /* && "$AMBI_PYTHON" == /* && "$AMBI_HELD_OUTPUT" == /* ]]
cd "$AMBI_PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "$AMBI_HELD_OUTPUT" != "$PWD" && "$AMBI_HELD_OUTPUT" != "$PWD/"* ]]
mkdir -p "$AMBI_HELD_OUTPUT"
local_root="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/ambi-held-${SLURM_JOB_ID}.XXXXXX")"
wandb_root="$AMBI_HELD_OUTPUT/wandb-${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}"
mkdir -p "$local_root/inductor" "$local_root/triton" "$local_root/cache" "$local_root/data" "$local_root/artifacts" "$local_root/tmp" "$wandb_root"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$local_root/inductor" TRITON_CACHE_DIR="$local_root/triton"
export XDG_CACHE_HOME="$local_root/cache" TMPDIR="$local_root/tmp"
export WANDB__REQUIRE_CORE=true WANDB_DISABLE_CODE=true WANDB_DIR="$wandb_root"
export WANDB_CACHE_DIR="$local_root/cache" WANDB_DATA_DIR="$local_root/data" WANDB_ARTIFACT_DIR="$local_root/artifacts"
if [[ "$mode" == smoke ]]; then
  [[ "$SLURM_CPUS_PER_TASK" == 3 && "${SLURM_RESTART_COUNT:-0}" == 0 ]]
  export WANDB_MODE=disabled
  exec srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" slurm/ambi_held_training.py smoke \
    --manifest "$manifest" --output "$AMBI_HELD_OUTPUT/gate"
else
  [[ "$SLURM_CPUS_PER_TASK" == 1 || "$SLURM_CPUS_PER_TASK" == 2 || "$SLURM_CPUS_PER_TASK" == 3 ]]
  : "${AMBI_LINEAGE_DIR:?Set a unique durable lineage under /oscar/home/rgao48/ambi-durable.}"
  : "${AMBI_DRAIN_AFTER_SECONDS:?Set an explicit segment budget below requested walltime.}"
  export AMBI_DURABLE_ROOT=/oscar/home/rgao48/ambi-durable
  export AMBI_DURABLE_QUOTA_LABEL=rgao48 AMBI_DURABLE_QUOTA_PATH=/oscar/home
  export AMBI_RESUME_CHECKPOINT_MINUTES="${AMBI_RESUME_CHECKPOINT_MINUTES:-30}"
  export WANDB_MODE=online
  # TrainingResumeSession assigns and restores the lineage's W&B identity.
  unset WANDB_RUN_ID WANDB_RESUME
  exec "$AMBI_PYTHON" slurm/ambi_held_training.py train \
    --manifest "$manifest" --output "$AMBI_HELD_OUTPUT" \
    --gate "${AMBI_HELD_GATE:-$AMBI_HELD_OUTPUT/gate/receipt.json}"
fi
