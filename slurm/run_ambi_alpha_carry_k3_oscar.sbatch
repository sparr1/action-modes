#!/usr/bin/env bash
#SBATCH --job-name=ambi-alpha-k3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --no-requeue

# The default is a short validation allocation. Production must override --time
# and --cpus-per-task from the K3 CPU benchmark, supplying the same walltime in
# AMBI_TRAIN_WALLTIME_SECONDS. No production array or implicit requeue is used.
set -Eeuo pipefail
umask 077
batch_started=$SECONDS
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${AMBI_PROJECT_DIR:?Set the clean source checkout.}"
: "${AMBI_PYTHON:?Set the locked DMControl interpreter.}"
: "${AMBI_EXPECTED_COMMIT:?Set the full tested commit.}"
: "${AMBI_ALPHA_CARRY_K3_OUTPUT:?Set campaign storage outside source.}"
mode="${AMBI_ALPHA_CARRY_K3_MODE:-train}"
export AMBI_ALPHA_CARRY_K3_MANIFEST="${AMBI_ALPHA_CARRY_K3_MANIFEST:-configs/dmcontrol/experiments/ambi_alpha_carry_k3_training.json}"
case "$AMBI_ALPHA_CARRY_K3_MANIFEST" in
  configs/dmcontrol/experiments/ambi_alpha_carry_k3_training.json|configs/dmcontrol/experiments/ambi_alpha_carry_k3_j10_training.json) ;;
  *) echo "Unsupported K3 campaign manifest." >&2; exit 1 ;;
esac
[[ "$mode" == smoke || "$mode" == benchmark || "$mode" == train ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
[[ "$AMBI_PROJECT_DIR" == /* && "$AMBI_PYTHON" == /* ]]
[[ "$AMBI_ALPHA_CARRY_K3_OUTPUT" == /oscar/scratch/rgao48/ambi/alpha-carry-k3/* ]]
cd "$AMBI_PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "$AMBI_ALPHA_CARRY_K3_OUTPUT" != "$PWD" && "$AMBI_ALPHA_CARRY_K3_OUTPUT" != "$PWD/"* ]]
mkdir -p "$AMBI_ALPHA_CARRY_K3_OUTPUT"
local_root="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/ambi-alpha-k3-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "$local_root/inductor" "$local_root/triton" "$local_root/cache" "$local_root/wandb" "$local_root/data" "$local_root/artifacts" "$local_root/tmp"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
# Benchmark subprocesses share disk caches across the1/2CPU cases.
export TORCHINDUCTOR_CACHE_DIR="$local_root/inductor" TRITON_CACHE_DIR="$local_root/triton"
export XDG_CACHE_HOME="$local_root/cache" TMPDIR="$local_root/tmp"
export WANDB_DISABLE_CODE=true WANDB_DIR="$local_root/wandb" WANDB_CACHE_DIR="$local_root/cache"
export WANDB_DATA_DIR="$local_root/data" WANDB_ARTIFACT_DIR="$local_root/artifacts"
export WANDB__REQUIRE_CORE=true WANDB_INIT_TIMEOUT=90 WANDB_HTTP_TIMEOUT=30 WANDB__SERVICE_WAIT=60
helper=slurm/ambi_alpha_carry_k3_training.py
case "$mode" in
  smoke|benchmark)
    export WANDB_MODE=disabled
    if [[ "$mode" == smoke ]]; then destination=gate; else destination=cpu-benchmark; fi
    srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" "$helper" "$mode" \
      --output "$AMBI_ALPHA_CARRY_K3_OUTPUT/$destination"
    ;;
  train)
    : "${AMBI_TRAIN_WALLTIME_SECONDS:?Set the measured production Slurm walltime in seconds.}"
    [[ "$AMBI_TRAIN_WALLTIME_SECONDS" =~ ^[0-9]+$ ]]
    # Verify requested --time agrees with the declared production duration.
    allocation_time="$(squeue --noheader --jobs "$SLURM_JOB_ID" --format '%l')"
    "$AMBI_PYTHON" - "$allocation_time" "$AMBI_TRAIN_WALLTIME_SECONDS" <<'PY'
import sys
value = sys.argv[1].strip()
days, clock = value.split('-') if '-' in value else ('0', value)
parts = list(map(int, clock.split(':')))
assert len(parts) in (2, 3), value
if len(parts) == 2:
    parts.insert(0, 0)
actual = int(days) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
assert actual == int(sys.argv[2]), (actual, sys.argv[2])
PY
    export AMBI_LINEAGE_DIR="$AMBI_ALPHA_CARRY_K3_OUTPUT/lineage"
    export AMBI_RESUME_MODE="${AMBI_RESUME_MODE:-new}"
    [[ "$AMBI_RESUME_MODE" == new || "$AMBI_RESUME_MODE" == required ]]
    quota_output="$(checkquota)"
    printf '%s\n' "$quota_output" | "$AMBI_PYTHON" -m utils.oscar_resume_launcher quota \
      --allocation rgao48 --filesystem-path /oscar/scratch
    "$AMBI_PYTHON" -m utils.oscar_resume_launcher storage \
      --durable-root "$AMBI_ALPHA_CARRY_K3_OUTPUT" --lineage-dir "$AMBI_LINEAGE_DIR"
    if [[ -e "$AMBI_LINEAGE_DIR/DONE" ]]; then
      "$AMBI_PYTHON" -m utils.oscar_resume_launcher done --lineage-dir "$AMBI_LINEAGE_DIR"
      exit 0
    fi
    export AMBI_SEGMENT_ID="${SLURM_JOB_ID}.0"
    export AMBI_SEGMENT_CONSOLE_DIR="$AMBI_ALPHA_CARRY_K3_OUTPUT/segment-console/$AMBI_SEGMENT_ID"
    mkdir -p "$(dirname -- "$AMBI_SEGMENT_CONSOLE_DIR")"
    mkdir "$AMBI_SEGMENT_CONSOLE_DIR"
    # Drain at least30minutes before the allocation ends; permit an earlier
    # operator-selected drain for a planned exact-resume segment.
    requested_drain="${AMBI_DRAIN_AFTER_SECONDS:-$((AMBI_TRAIN_WALLTIME_SECONDS - 1800))}"
    [[ "$requested_drain" =~ ^[0-9]+$ ]]
    (( requested_drain <= AMBI_TRAIN_WALLTIME_SECONDS - 1800 ))
    export AMBI_DRAIN_AFTER_SECONDS=$((requested_drain - (SECONDS - batch_started)))
    (( AMBI_DRAIN_AFTER_SECONDS > 0 ))
    export WANDB_MODE=online
    unset WANDB_RUN_ID WANDB_RESUME
    set +e
    srun --unbuffered --kill-on-bad-exit=1 \
      --output="$AMBI_SEGMENT_CONSOLE_DIR/stdout.log" \
      --error="$AMBI_SEGMENT_CONSOLE_DIR/stderr.log" \
      "$AMBI_PYTHON" "$helper" train \
      --output "$AMBI_ALPHA_CARRY_K3_OUTPUT/launches/$AMBI_SEGMENT_ID" \
      --gate "$AMBI_ALPHA_CARRY_K3_OUTPUT/gate/receipt.json" \
      --benchmark "$AMBI_ALPHA_CARRY_K3_OUTPUT/cpu-benchmark/results.json"
    status=$?
    set -e
    case "$status" in
      0)
        "$AMBI_PYTHON" -m utils.oscar_resume_launcher done --lineage-dir "$AMBI_LINEAGE_DIR"
        echo "Verified1M-decision completion."
        ;;
      75)
        "$AMBI_PYTHON" -m utils.oscar_resume_launcher handoff --lineage-dir "$AMBI_LINEAGE_DIR" \
          --slurm-job-id "$SLURM_JOB_ID" --segment-id "$AMBI_SEGMENT_ID"
        echo "Verified clean checkpoint handoff; no automatic requeue."
        exit 75
        ;;
      *)
        echo "Training failed with status $status; no automatic requeue." >&2
        exit "$status"
        ;;
    esac
    ;;
esac
