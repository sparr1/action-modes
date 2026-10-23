#!/usr/bin/env bash
#SBATCH --job-name=ambi-alpha-carry
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=16G
#SBATCH --time=4-00:00:00
#SBATCH --no-requeue

# CPU and memory requests are overridden at submission after the campaign's
# 1/2/4-core benchmark. Gate jobs must receive a short explicit walltime.
# Smoke can use AMBI_ALPHA_CARRY_INDICES="0 4" in independent allocations.
# After all eight case files exist, run mode=seal to produce gate/receipt.json.
set -Eeuo pipefail
umask 077
batch_started=$SECONDS
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${AMBI_PROJECT_DIR:?Set the clean source checkout.}"
: "${AMBI_PYTHON:?Set the locked DMControl interpreter.}"
: "${AMBI_EXPECTED_COMMIT:?Set the full tested commit.}"
: "${AMBI_ALPHA_CARRY_OUTPUT:?Set campaign storage outside source.}"
mode="${AMBI_ALPHA_CARRY_MODE:-train}"
[[ "$mode" == smoke || "$mode" == seal || "$mode" == benchmark || "$mode" == train ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
[[ "$AMBI_PROJECT_DIR" == /* && "$AMBI_PYTHON" == /* && "$AMBI_ALPHA_CARRY_OUTPUT" == /* ]]
cd "$AMBI_PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "$AMBI_ALPHA_CARRY_OUTPUT" != "$PWD" && "$AMBI_ALPHA_CARRY_OUTPUT" != "$PWD/"* ]]
mkdir -p "$AMBI_ALPHA_CARRY_OUTPUT"
local_root="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/ambi-alpha-carry-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "$local_root/inductor" "$local_root/triton" "$local_root/cache" "$local_root/wandb" "$local_root/data" "$local_root/artifacts" "$local_root/tmp"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$local_root/inductor" TRITON_CACHE_DIR="$local_root/triton"
export XDG_CACHE_HOME="$local_root/cache" TMPDIR="$local_root/tmp"
export WANDB_DISABLE_CODE=true WANDB_DIR="$local_root/wandb" WANDB_CACHE_DIR="$local_root/cache"
export WANDB_DATA_DIR="$local_root/data" WANDB_ARTIFACT_DIR="$local_root/artifacts"
helper=slurm/ambi_alpha_carry_training.py
case "$mode" in
  smoke)
    export WANDB_MODE=disabled
    smoke_indices="${AMBI_ALPHA_CARRY_INDICES:-${SLURM_ARRAY_TASK_ID:-}}"
    if [[ -n "$smoke_indices" ]]; then
      for index in $smoke_indices; do
        [[ "$index" =~ ^[0-7]$ ]]
        srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" "$helper" smoke \
          --index "$index" --output "$AMBI_ALPHA_CARRY_OUTPUT/gate"
      done
    else
      srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" "$helper" smoke \
        --output "$AMBI_ALPHA_CARRY_OUTPUT/gate"
    fi
    ;;
  seal)
    export WANDB_MODE=disabled
    "$AMBI_PYTHON" "$helper" seal --output "$AMBI_ALPHA_CARRY_OUTPUT/gate"
    ;;
  benchmark)
    export WANDB_MODE=disabled
    srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" "$helper" benchmark \
      --output "$AMBI_ALPHA_CARRY_OUTPUT/cpu-benchmark"
    ;;
  train)
    : "${SLURM_ARRAY_TASK_ID:?Train as an eight-cell array.}"
    [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-7]$ ]]
    # This campaign uses scratch for large generations and checkpoint banks;
    # the older home-only launcher's policy remains unchanged.
    [[ "$AMBI_ALPHA_CARRY_OUTPUT" == /oscar/scratch/rgao48/ambi/alpha-carry-training/* ]]
    export AMBI_LINEAGE_DIR="$AMBI_ALPHA_CARRY_OUTPUT/lineages/cell-${SLURM_ARRAY_TASK_ID}"
    export AMBI_RESUME_MODE="${AMBI_RESUME_MODE:-new}"
    [[ "$AMBI_RESUME_MODE" == new || "$AMBI_RESUME_MODE" == required ]]
    quota_output="$(checkquota)"
    printf '%s\n' "$quota_output" | "$AMBI_PYTHON" -m utils.oscar_resume_launcher quota \
      --allocation rgao48 --filesystem-path /oscar/scratch
    "$AMBI_PYTHON" -m utils.oscar_resume_launcher storage \
      --durable-root "$AMBI_ALPHA_CARRY_OUTPUT" --lineage-dir "$AMBI_LINEAGE_DIR"
    if [[ -e "$AMBI_LINEAGE_DIR/DONE" ]]; then
      "$AMBI_PYTHON" -m utils.oscar_resume_launcher done --lineage-dir "$AMBI_LINEAGE_DIR"
      exit 0
    fi
    export AMBI_SEGMENT_ID="${SLURM_JOB_ID}.0"
    export AMBI_SEGMENT_CONSOLE_DIR="$AMBI_ALPHA_CARRY_OUTPUT/segment-console/$AMBI_SEGMENT_ID"
    mkdir -p "$(dirname -- "$AMBI_SEGMENT_CONSOLE_DIR")"
    mkdir "$AMBI_SEGMENT_CONSOLE_DIR"
    export AMBI_DRAIN_AFTER_SECONDS=$((341700 - (SECONDS - batch_started)))
    (( AMBI_DRAIN_AFTER_SECONDS > 0 ))
    export WANDB_MODE=online
    # The native lineage owns its W&B run UUID and resume/publication policy.
    unset WANDB_RUN_ID WANDB_RESUME
    set +e
    srun --unbuffered --kill-on-bad-exit=1 \
      --output="$AMBI_SEGMENT_CONSOLE_DIR/stdout.log" \
      --error="$AMBI_SEGMENT_CONSOLE_DIR/stderr.log" \
      "$AMBI_PYTHON" "$helper" train --index "$SLURM_ARRAY_TASK_ID" \
      --output "$AMBI_ALPHA_CARRY_OUTPUT/launches/$AMBI_SEGMENT_ID" \
      --gate "$AMBI_ALPHA_CARRY_OUTPUT/gate/receipt.json"
    status=$?
    set -e
    case "$status" in
      0)
        "$AMBI_PYTHON" -m utils.oscar_resume_launcher done --lineage-dir "$AMBI_LINEAGE_DIR"
        echo "Verified 1M-decision completion."
        ;;
      75)
        "$AMBI_PYTHON" -m utils.oscar_resume_launcher handoff --lineage-dir "$AMBI_LINEAGE_DIR" \
          --slurm-job-id "$SLURM_JOB_ID" --segment-id "$AMBI_SEGMENT_ID"
        echo "Verified clean checkpoint handoff; target not yet complete. No automatic requeue."
        exit 75
        ;;
      *)
        echo "Training failed with status $status; no automatic requeue." >&2
        exit "$status"
        ;;
    esac
    ;;
esac
