#!/usr/bin/env bash
#SBATCH --job-name=ambi-return-only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --hint=nomultithread
#SBATCH --requeue

# Explicit submission resources: smoke needs one GPU/two physical CPUs;
# production uses the measured one- or two-core allocation and measured time.
# Supply --signal=USR1@900 for production. Each job selects one one-cell
# manifest and one durable lineage; independent SAC/MPPI gates may run together.
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${SLURM_CPUS_PER_TASK:?Specify measured CPU resources.}"
: "${AMBI_PROJECT_DIR:?Set the clean pinned source checkout.}"
: "${AMBI_PYTHON:?Set the locked DMControl interpreter.}"
: "${AMBI_EXPECTED_COMMIT:?Set the full tested commit.}"
: "${AMBI_RETURN_OUTPUT:?Set persistent campaign output outside source.}"
: "${AMBI_RETURN_MANIFEST:?Select the single-cell SAC or MPPI manifest.}"
mode="${AMBI_RETURN_MODE:-train}"
[[ "$mode" == smoke || "$mode" == train ]]
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]
[[ "$AMBI_PROJECT_DIR" == /* && "$AMBI_PYTHON" == /* && "$AMBI_RETURN_OUTPUT" == /* ]]
cd "$AMBI_PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "$AMBI_RETURN_OUTPUT" != "$PWD" && "$AMBI_RETURN_OUTPUT" != "$PWD/"* ]]
mkdir -p "$AMBI_RETURN_OUTPUT"
local_root="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/ambi-return-${SLURM_JOB_ID}.XXXXXX")"
wandb_root="$AMBI_RETURN_OUTPUT/wandb-${SLURM_JOB_ID}.${SLURM_RESTART_COUNT:-0}"
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
  [[ "$SLURM_CPUS_PER_TASK" == 2 && "${SLURM_RESTART_COUNT:-0}" == 0 ]]
  export WANDB_MODE=disabled
  exec srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" slurm/ambi_reward_training.py smoke \
    --manifest "$AMBI_RETURN_MANIFEST" --output "$AMBI_RETURN_OUTPUT/gate"
else
  [[ "$SLURM_CPUS_PER_TASK" == 1 || "$SLURM_CPUS_PER_TASK" == 2 ]]
  : "${AMBI_LINEAGE_DIR:?Set a unique durable lineage under /oscar/home/rgao48/ambi-durable.}"
  : "${AMBI_DRAIN_AFTER_SECONDS:?Set an explicit segment budget below requested walltime.}"
  export AMBI_DURABLE_ROOT=/oscar/home/rgao48/ambi-durable
  export AMBI_DURABLE_QUOTA_LABEL=rgao48 AMBI_DURABLE_QUOTA_PATH=/oscar/home
  export AMBI_RESUME_CHECKPOINT_MINUTES="${AMBI_RESUME_CHECKPOINT_MINUTES:-30}"
  export WANDB_MODE=online
  unset WANDB_RUN_ID WANDB_RESUME
  exec "$AMBI_PYTHON" slurm/ambi_reward_training.py train \
    --manifest "$AMBI_RETURN_MANIFEST" --output "$AMBI_RETURN_OUTPUT" \
    --gate "${AMBI_RETURN_GATE:-$AMBI_RETURN_OUTPUT/gate/receipt.json}"
fi
