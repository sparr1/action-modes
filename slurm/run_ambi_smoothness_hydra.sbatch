#!/usr/bin/env bash
#SBATCH --job-name=ambi-smoothness
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2501
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --no-requeue

# Gate each of the four cells before production. Submit --output/--error paths
# outside the checkout. Override train CPU/time from that cell's gate measurements
# and set AMBI_TRAIN_WALLTIME_SECONDS to the requested allocation in seconds.
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${AMBI_PROJECT_DIR:?Set the clean source checkout.}"
: "${AMBI_PYTHON:?Set the locked DMControl interpreter.}"
: "${AMBI_EXPECTED_COMMIT:?Set the full tested source commit.}"
: "${AMBI_CAMPAIGN_ROOT:?Set campaign storage under home, outside the checkout.}"
: "${AMBI_ARTIFACT_ROOT:?Set campaign storage under /ltmp/rgao48 on gpu2501.}"
: "${AMBI_MODE:?Select gate or train.}"
export AMBI_CELL_INDEX="${SLURM_ARRAY_TASK_ID:-${AMBI_CELL_INDEX:?Select a cell in 0..3.}}"
[[ "$AMBI_CELL_INDEX" =~ ^[0-3]$ ]]
[[ "$AMBI_MODE" == gate || "$AMBI_MODE" == train ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
[[ "$AMBI_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$AMBI_PROJECT_DIR" == /* && "$AMBI_PYTHON" == /* && "$AMBI_CAMPAIGN_ROOT" == /* ]]
[[ "$AMBI_ARTIFACT_ROOT" == /ltmp/rgao48/* ]]
cd "$AMBI_PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "$AMBI_CAMPAIGN_ROOT" != "$PWD" && "$AMBI_CAMPAIGN_ROOT" != "$PWD/"* ]]
[[ -x "$AMBI_PYTHON" ]]
export AMBI_CACHE_ROOT="$AMBI_ARTIFACT_ROOT/cache/cell-$AMBI_CELL_INDEX/$AMBI_EXPECTED_COMMIT/$SLURM_JOB_ID"
mkdir -p "$AMBI_CACHE_ROOT" "$AMBI_CAMPAIGN_ROOT/cell-$AMBI_CELL_INDEX/console"
for AMBI_CACHE_PART in inductor triton cache data artifacts tmp wandb; do
  mkdir -p "$AMBI_CACHE_ROOT/$AMBI_CACHE_PART"
done
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?Slurm CPU allocation missing.}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$AMBI_CACHE_ROOT/inductor" TRITON_CACHE_DIR="$AMBI_CACHE_ROOT/triton"
export XDG_CACHE_HOME="$AMBI_CACHE_ROOT/cache" TMPDIR="$AMBI_CACHE_ROOT/tmp"
export WANDB_DISABLE_CODE=true WANDB_DIR="$AMBI_CACHE_ROOT/wandb" WANDB_CACHE_DIR="$AMBI_CACHE_ROOT/cache"
export WANDB_DATA_DIR="$AMBI_CACHE_ROOT/data" WANDB_ARTIFACT_DIR="$AMBI_CACHE_ROOT/artifacts"
export WANDB_MODE=disabled
if [[ "$AMBI_MODE" == train ]]; then
  : "${AMBI_TRAIN_WALLTIME_SECONDS:?Set the measured production allocation in seconds.}"
  export WANDB_MODE=online
fi
export AMBI_CONSOLE="$AMBI_CAMPAIGN_ROOT/cell-$AMBI_CELL_INDEX/console/$AMBI_MODE-$SLURM_JOB_ID.log"
[[ ! -e "$AMBI_CONSOLE" ]]
exec > >(tee "$AMBI_CONSOLE") 2>&1
exec srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" slurm/ambi_smoothness_launch.py
