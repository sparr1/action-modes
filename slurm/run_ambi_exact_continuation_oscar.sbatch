#!/usr/bin/env bash
#SBATCH --job-name=ambi-exact-continuation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --hint=nomultithread
#SBATCH --requeue
#SBATCH --signal=USR1@1800

# Submit with explicit measured CPU, GPU, memory, walltime and batch-log paths.
# This launcher may come from a newer tooling checkout: all training and
# checkpoint verification execute in the original, clean AMBI_PROJECT_DIR.
set -Eeuo pipefail
umask 077
batch_started=$SECONDS

fail() { echo "Exact continuation launcher error: $*" >&2; exit 2; }
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${SLURM_CPUS_PER_TASK:?Specify the measured CPU allocation.}"
: "${AMBI_PROJECT_DIR:?Set the original clean training checkout.}"
: "${AMBI_EXPECTED_COMMIT:?Set the original full training commit.}"
: "${AMBI_PYTHON:?Set the shared locked DMControl interpreter.}"
: "${AMBI_OUTPUT_ROOT:?Set an existing scratch root containing the lineage.}"
: "${AMBI_LINEAGE_DIR:?Set the existing lineage to continue.}"
: "${AMBI_RUN_CONFIG:?Set the single-cell training manifest.}"
: "${AMBI_ALG_DIR:?Set its algorithm configuration directory.}"
: "${AMBI_QUOTA_PATH:?Set /oscar/scratch for these scratch lineages.}"
: "${AMBI_DRAIN_AFTER_SECONDS:?Set the segment budget below requested walltime.}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || fail "SLURM_JOB_ID must be numeric"
[[ "$SLURM_CPUS_PER_TASK" == 1 ]] || fail "these continuations require the measured one-CPU allocation"
[[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || fail "submit one job per existing lineage"
[[ "$AMBI_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail "expected commit must be a full SHA"
[[ "$AMBI_DRAIN_AFTER_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail "drain budget must be positive"
[[ "$AMBI_QUOTA_PATH" == /oscar/scratch ]] || fail "quota path must be /oscar/scratch"
for path in "$AMBI_PROJECT_DIR" "$AMBI_PYTHON" "$AMBI_OUTPUT_ROOT" "$AMBI_LINEAGE_DIR"; do
  [[ "$path" == /* ]] || fail "source, Python, output and lineage paths must be absolute"
done
[[ -x "$AMBI_PYTHON" ]] || fail "locked Python is not executable"
[[ -d "$AMBI_LINEAGE_DIR" && -f "$AMBI_LINEAGE_DIR/LINEAGE.json" && -f "$AMBI_LINEAGE_DIR/LATEST" ]] || fail "existing lineage and LATEST are required"
cd "$AMBI_PROJECT_DIR"
project_dir="$(pwd -P)"
git_root="$(git rev-parse --show-toplevel)"
[[ "$(cd "$git_root" && pwd -P)" == "$project_dir" ]] || fail "training directory must be the Git root"
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]] || fail "training checkout does not match expected commit"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || fail "training checkout must be clean"
[[ -f "$AMBI_RUN_CONFIG" && -d "$AMBI_ALG_DIR" ]] || fail "manifest or algorithm directory is missing"
output_root="$(cd "$AMBI_OUTPUT_ROOT" && pwd -P)"
[[ "$output_root" != / && "$output_root" != "$project_dir" && "$output_root" != "$project_dir/"* ]] || fail "output root must be outside source"
if [[ -d /oscar/scratch ]]; then
  scratch_root="$(cd /oscar/scratch && pwd -P)"
  [[ "$output_root" == "$scratch_root/"* ]] || fail "output root must be below the selected scratch filesystem"
fi
restart_count="${SLURM_RESTART_COUNT:-0}"
[[ "$restart_count" =~ ^[0-9]+$ ]] || fail "restart count must be non-negative"

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
quota_output="$(checkquota)" || fail "checkquota failed"
printf '%s\n' "$quota_output" | "$AMBI_PYTHON" -m utils.oscar_resume_launcher quota \
  --allocation rgao48 --filesystem-path "$AMBI_QUOTA_PATH"
"$AMBI_PYTHON" -m utils.oscar_resume_launcher storage \
  --durable-root "$output_root" --lineage-dir "$AMBI_LINEAGE_DIR"
if [[ -e "$AMBI_LINEAGE_DIR/DONE" ]]; then
  "$AMBI_PYTHON" -m utils.oscar_resume_launcher done --lineage-dir "$AMBI_LINEAGE_DIR"
  echo "Verified DONE; training target is already complete."
  exit 0
fi
if (( restart_count > 0 )); then
  "$AMBI_PYTHON" -m utils.oscar_resume_launcher handoff \
    --lineage-dir "$AMBI_LINEAGE_DIR" --slurm-job-id "$SLURM_JOB_ID" \
    --segment-id "${SLURM_JOB_ID}.$((restart_count - 1))"
fi

export AMBI_SEGMENT_ID="${SLURM_JOB_ID}.${restart_count}"
segment_console_dir="$output_root/segment-console/$AMBI_SEGMENT_ID"
wandb_root="$output_root/wandb/$AMBI_SEGMENT_ID"
mkdir -p "$(dirname "$segment_console_dir")" "$(dirname "$wandb_root")"
mkdir "$segment_console_dir" "$wandb_root" || fail "segment output already exists"
export AMBI_SEGMENT_CONSOLE_DIR="$segment_console_dir"
local_root="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/ambi-continuation-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "$local_root/inductor" "$local_root/triton" "$local_root/cache" \
  "$local_root/data" "$local_root/artifacts" "$local_root/tmp"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$local_root/inductor" TRITON_CACHE_DIR="$local_root/triton"
export XDG_CACHE_HOME="$local_root/cache" TMPDIR="$local_root/tmp"
export WANDB_MODE=online WANDB__REQUIRE_CORE=true WANDB_DISABLE_CODE=true
export WANDB_DIR="$wandb_root" WANDB_CACHE_DIR="$local_root/cache"
export WANDB_DATA_DIR="$local_root/data" WANDB_ARTIFACT_DIR="$local_root/artifacts"
unset WANDB_RUN_ID WANDB_RESUME

# Imports, quota probes and filesystem setup consume the segment's budget.
remaining=$((AMBI_DRAIN_AFTER_SECONDS - (SECONDS - batch_started)))
(( remaining > 0 )) || fail "drain budget expired during startup"
echo "Continuing $AMBI_LINEAGE_DIR at $AMBI_EXPECTED_COMMIT in segment $AMBI_SEGMENT_ID."
set +e
srun --unbuffered --kill-on-bad-exit=1 \
  --output="$segment_console_dir/stdout.log" --error="$segment_console_dir/stderr.log" \
  "$AMBI_PYTHON" main.py --run "$AMBI_RUN_CONFIG" --alg-dir "$AMBI_ALG_DIR" \
  --num-runs 1 --lineage-dir "$AMBI_LINEAGE_DIR" \
  --resume-mode required --resume-wandb-mode online \
  --resume-checkpoint-minutes 15 --drain-after-seconds "$remaining"
status=$?
set -e
case "$status" in
  0)
    "$AMBI_PYTHON" -m utils.oscar_resume_launcher done --lineage-dir "$AMBI_LINEAGE_DIR"
    echo "Training target complete; DONE verified against LATEST."
    ;;
  75)
    "$AMBI_PYTHON" -m utils.oscar_resume_launcher handoff \
      --lineage-dir "$AMBI_LINEAGE_DIR" --slurm-job-id "$SLURM_JOB_ID" \
      --segment-id "$AMBI_SEGMENT_ID"
    echo "Clean handoff verified; requeueing $SLURM_JOB_ID."
    scontrol requeue "$SLURM_JOB_ID"
    ;;
  *)
    echo "Training failed with status $status; automatic requeue suppressed." >&2
    exit "$status"
    ;;
esac
