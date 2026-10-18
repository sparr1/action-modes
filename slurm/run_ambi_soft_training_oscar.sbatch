#!/usr/bin/env bash
#SBATCH --job-name=ambi-soft-sac
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --hint=nomultithread
#SBATCH --no-requeue

# Supply explicit CPU/GPU/memory/walltime after checking live QOS headroom.
# Gate: AMBI_SOFT_MODE=smoke, four physical CPUs and one GPU. Production:
# AMBI_SOFT_MODE=train, one measured CPU count (1/2/4), same GPU model, and a
# throughput-derived walltime. This is one run, never an array.
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${SLURM_CPUS_PER_TASK:?Specify a measured CPU allocation in sbatch.}"
: "${AMBI_PROJECT_DIR:?Set the clean source checkout.}"
: "${AMBI_PYTHON:?Set the locked DMControl interpreter.}"
: "${AMBI_EXPECTED_COMMIT:?Set the full tested commit.}"
: "${AMBI_SOFT_OUTPUT:?Set campaign storage outside source.}"
mode="${AMBI_SOFT_MODE:-train}"
manifest="${AMBI_SOFT_MANIFEST:-configs/dmcontrol/experiments/ambi_soft_training.json}"
[[ "$mode" == smoke || "$mode" == train ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 && -z "${SLURM_ARRAY_TASK_ID:-}" ]]
[[ "$AMBI_PROJECT_DIR" == /* && "$AMBI_PYTHON" == /* && "$AMBI_SOFT_OUTPUT" == /* ]]
cd "$AMBI_PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "$AMBI_SOFT_OUTPUT" != "$PWD" && "$AMBI_SOFT_OUTPUT" != "$PWD/"* ]]
mkdir -p "$AMBI_SOFT_OUTPUT"
available_kb="$(df -Pk "$AMBI_SOFT_OUTPUT" | awk 'NR==2 {print $4}')"
(( available_kb >= 5 * 1024 * 1024 ))
local_root="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/ambi-soft-${SLURM_JOB_ID}.XXXXXX")"
wandb_root="$AMBI_SOFT_OUTPUT/wandb-job-${SLURM_JOB_ID}"
mkdir -p "$local_root/inductor" "$local_root/triton" "$local_root/cache" "$local_root/data" "$local_root/artifacts" "$local_root/tmp" "$wandb_root"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$local_root/inductor" TRITON_CACHE_DIR="$local_root/triton"
export XDG_CACHE_HOME="$local_root/cache" TMPDIR="$local_root/tmp"
# Preserve the sync journal outside node-local storage and use the proven Core backend.
export WANDB__REQUIRE_CORE=true WANDB_DISABLE_CODE=true WANDB_DIR="$wandb_root"
export WANDB_CACHE_DIR="$local_root/cache" WANDB_DATA_DIR="$local_root/data" WANDB_ARTIFACT_DIR="$local_root/artifacts"
if [[ "$mode" == smoke ]]; then
  [[ "$SLURM_CPUS_PER_TASK" == 4 ]]
  export WANDB_MODE=disabled
  exec srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" slurm/ambi_soft_training.py smoke \
    --manifest "$manifest" --output "$AMBI_SOFT_OUTPUT/gate"
else
  [[ "$SLURM_CPUS_PER_TASK" == 1 || "$SLURM_CPUS_PER_TASK" == 2 || "$SLURM_CPUS_PER_TASK" == 4 ]]
  export WANDB_MODE=online WANDB_RESUME=never WANDB_RUN_ID="softsac${SLURM_JOB_ID}"
  exec srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" slurm/ambi_soft_training.py train \
    --manifest "$manifest" --output "$AMBI_SOFT_OUTPUT/cell-0" --gate "$AMBI_SOFT_OUTPUT/gate/receipt.json"
fi
