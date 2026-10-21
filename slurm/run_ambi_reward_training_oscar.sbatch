#!/usr/bin/env bash
#SBATCH --job-name=ambi-return-only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --hint=nomultithread
#SBATCH --requeue

# Explicit resources: smoke needs one GPU and two physical CPUs for SAC/MPPI,
# or four for the no-inner baseline. Production uses a measured allocation.
# Supply --signal=USR1@900 for production. Each job selects one one-cell
# manifest and one durable lineage; independent SAC/MPPI gates may run together.
# For ordinary GPU-eligible replay without continuation, set
# AMBI_RETURN_CONTINUATION=off and submit with --no-requeue; see REPLAY_PLACEMENT.md.
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${SLURM_CPUS_PER_TASK:?Specify measured CPU resources.}"
: "${AMBI_PROJECT_DIR:?Set the clean pinned source checkout.}"
: "${AMBI_PYTHON:?Set the locked DMControl interpreter.}"
: "${AMBI_EXPECTED_COMMIT:?Set the full tested commit.}"
: "${AMBI_RETURN_OUTPUT:?Set persistent campaign output outside source.}"
: "${AMBI_RETURN_MANIFEST:?Select the single-cell SAC, MPPI or no-inner manifest.}"
mode="${AMBI_RETURN_MODE:-train}"
continuation="${AMBI_RETURN_CONTINUATION:-exact}"
[[ "$mode" == smoke || "$mode" == train ]]
[[ "$continuation" == exact || "$continuation" == off ]]
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]
if [[ "$continuation" == off ]]; then
  [[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
  job_record="$(scontrol show job "$SLURM_JOB_ID" --oneliner)"
  [[ " $job_record " == *" Requeue=0 "* ]] || { echo "Continuation off requires sbatch --no-requeue" >&2; exit 2; }
fi
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
  [[ ( "$SLURM_CPUS_PER_TASK" == 2 || "$SLURM_CPUS_PER_TASK" == 4 ) && "${SLURM_RESTART_COUNT:-0}" == 0 ]]
  export WANDB_MODE=disabled
  exec srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" slurm/ambi_reward_training.py smoke \
    --manifest "$AMBI_RETURN_MANIFEST" --output "$AMBI_RETURN_OUTPUT/gate" --continuation "$continuation"
else
  [[ "$SLURM_CPUS_PER_TASK" == 1 || "$SLURM_CPUS_PER_TASK" == 2 || "$SLURM_CPUS_PER_TASK" == 4 ]]
  if [[ "$continuation" == off ]]; then
    export WANDB_MODE=online
    unset WANDB_RUN_ID WANDB_RESUME
    exec srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" slurm/ambi_reward_training.py train \
      --manifest "$AMBI_RETURN_MANIFEST" --output "$AMBI_RETURN_OUTPUT" --continuation off \
      --gate "${AMBI_RETURN_GATE:-$AMBI_RETURN_OUTPUT/gate/receipt.json}"
  fi
  : "${AMBI_LINEAGE_DIR:?Set a unique lineage below the approved storage root.}"
  : "${AMBI_DRAIN_AFTER_SECONDS:?Set an explicit segment budget below requested walltime.}"
  export AMBI_DURABLE_ROOT="${AMBI_DURABLE_ROOT:-/oscar/home/rgao48/ambi-durable}"
  case "$AMBI_DURABLE_ROOT" in
    /oscar/home/rgao48/ambi-durable) expected_quota_path=/oscar/home ;;
    /oscar/scratch/rgao48/ambi-training-lineages) expected_quota_path=/oscar/scratch ;;
    *) echo "Unapproved AMBI storage root" >&2; exit 2 ;;
  esac
  export AMBI_DURABLE_QUOTA_LABEL="${AMBI_DURABLE_QUOTA_LABEL:-rgao48}"
  export AMBI_DURABLE_QUOTA_PATH="${AMBI_DURABLE_QUOTA_PATH:-$expected_quota_path}"
  [[ "$AMBI_DURABLE_QUOTA_LABEL" == rgao48 && "$AMBI_DURABLE_QUOTA_PATH" == "$expected_quota_path" ]]
  export AMBI_RESUME_CHECKPOINT_MINUTES="${AMBI_RESUME_CHECKPOINT_MINUTES:-30}"
  export WANDB_MODE=online
  unset WANDB_RUN_ID WANDB_RESUME
  exec "$AMBI_PYTHON" slurm/ambi_reward_training.py train \
    --manifest "$AMBI_RETURN_MANIFEST" --output "$AMBI_RETURN_OUTPUT" \
    --gate "${AMBI_RETURN_GATE:-$AMBI_RETURN_OUTPUT/gate/receipt.json}"
fi
