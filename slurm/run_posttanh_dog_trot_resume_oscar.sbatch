#!/usr/bin/env bash
#SBATCH --job-name=posttanh-trot-s56-resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=default
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --exclude=gpu2708
#SBATCH --requeue
#SBATCH --signal=USR1@1800
set -Eeuo pipefail
umask 077
batch_started=$SECONDS
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${AMBI_EXPECTED_COMMIT:?Pin the tested source}"
: "${AMBI_PYTHON:?Use the locked DMControl interpreter}"
: "${AMBI_RESUME_CAMPAIGN:?Set campaign scratch storage}"
mode="${1:-train}"
[[ "$mode" == benchmark || "$mode" == train ]]
cd "${SLURM_SUBMIT_DIR:?}"
[[ "$(git rev-parse --show-toplevel)" == "$PWD" ]]
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "$AMBI_RESUME_CAMPAIGN" == /oscar/scratch/rgao48/ambi/* ]]
restart="${SLURM_RESTART_COUNT:-0}"
[[ "$restart" =~ ^[0-9]+$ ]]
cache="$AMBI_RESUME_CAMPAIGN/cache/$SLURM_JOB_ID.$restart"
mkdir -p "$(dirname "$cache")"
mkdir "$cache"
mkdir -p "$cache/"{inductor,triton,cache,tmp,wandb,data,artifacts}
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
export TMPDIR="$cache/tmp" TORCHINDUCTOR_CACHE_DIR="$cache/inductor" TRITON_CACHE_DIR="$cache/triton"
export XDG_CACHE_HOME="$cache/cache" WANDB_DIR="$cache/wandb" WANDB_CACHE_DIR="$cache/cache"
export WANDB_DATA_DIR="$cache/data" WANDB_ARTIFACT_DIR="$cache/artifacts"
export WANDB_DISABLE_CODE=true WANDB__REQUIRE_CORE=true
export WANDB_INIT_TIMEOUT=90 WANDB_HTTP_TIMEOUT=30 WANDB__SERVICE_WAIT=60
unset CUDA_LAUNCH_BLOCKING WANDB_RUN_ID WANDB_RESUME
helper=slurm/posttanh_dog_trot_resume.py
if [[ "$mode" == benchmark ]]; then
  export WANDB_MODE=disabled
  (( SLURM_CPUS_PER_TASK >= 2 ))
  srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" "$helper" benchmark \
    --output "$AMBI_RESUME_CAMPAIGN/benchmark"
  exit 0
fi
: "${AMBI_LINEAGE_DIR:?Set a fresh durable lineage, or its original path for required resume}"
: "${AMBI_WORKSPACE_PYTHON:?Use the existing workspace publication environment}"
: "${AMBI_TRAIN_WALLTIME_SECONDS:?Set the measured allocation duration}"
[[ "$AMBI_LINEAGE_DIR" == /oscar/home/rgao48/ambi-durable/lineages/* ]]
[[ "$AMBI_TRAIN_WALLTIME_SECONDS" =~ ^[0-9]+$ ]]
(( AMBI_TRAIN_WALLTIME_SECONDS > 3600 ))
"$AMBI_PYTHON" "$helper" allocation --output "$AMBI_RESUME_CAMPAIGN/benchmark"
quota_output="$(checkquota)"
printf '%s\n' "$quota_output" | "$AMBI_PYTHON" -m utils.oscar_resume_launcher quota \
  --allocation rgao48 --filesystem-path /oscar/home
"$AMBI_PYTHON" -m utils.oscar_resume_launcher storage \
  --durable-root /oscar/home/rgao48/ambi-durable --lineage-dir "$AMBI_LINEAGE_DIR"
if [[ -e "$AMBI_LINEAGE_DIR/DONE" ]]; then
  "$AMBI_PYTHON" -m utils.oscar_resume_launcher done --lineage-dir "$AMBI_LINEAGE_DIR"
  exit 0
fi
resume_mode="${AMBI_RESUME_MODE:-new}"
if (( restart > 0 )); then
  "$AMBI_PYTHON" -m utils.oscar_resume_launcher handoff --lineage-dir "$AMBI_LINEAGE_DIR" \
    --slurm-job-id "$SLURM_JOB_ID" --segment-id "$SLURM_JOB_ID.$((restart-1)).train"
  resume_mode=required
fi
[[ "$resume_mode" == new || "$resume_mode" == required ]]
export WANDB_MODE=disabled
srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" "$helper" node-check \
  --output "$AMBI_RESUME_CAMPAIGN/node-checks/$SLURM_JOB_ID.$restart.json"
export WANDB_MODE=online
run_segment() {
  export AMBI_SEGMENT_ID="$SLURM_JOB_ID.$restart.$1"
  export AMBI_SEGMENT_CONSOLE_DIR="$AMBI_RESUME_CAMPAIGN/console/$AMBI_SEGMENT_ID"
  mkdir -p "$(dirname "$AMBI_SEGMENT_CONSOLE_DIR")"
  mkdir "$AMBI_SEGMENT_CONSOLE_DIR"
  set +e
  srun --unbuffered --kill-on-bad-exit=1 \
    --output="$AMBI_SEGMENT_CONSOLE_DIR/stdout.log" --error="$AMBI_SEGMENT_CONSOLE_DIR/stderr.log" \
    "$AMBI_PYTHON" "$helper" train --output "$AMBI_RESUME_CAMPAIGN/launches/$AMBI_SEGMENT_ID" \
    --resume-mode "$resume_mode" --snapshot-root "$AMBI_RESUME_CAMPAIGN/models" \
    --drain-seconds "$2" --checkpoint-minutes 15
  segment_status=$?
  set -e
}
verify_handoff() {
  "$AMBI_PYTHON" -m utils.oscar_resume_launcher handoff --lineage-dir "$AMBI_LINEAGE_DIR" \
    --slurm-job-id "$SLURM_JOB_ID" --segment-id "$AMBI_SEGMENT_ID"
}
publish() {
  "$AMBI_PYTHON" "$helper" publish --output "$AMBI_RESUME_CAMPAIGN"
}
if [[ "$resume_mode" == new ]]; then
  # This is the first part of the real training run, with unchanged seed,
  # target and update dose. Resume it in the same allocation after a verified
  # durable handoff; no separate W&B validation run is created.
  run_segment verify 600
  if [[ "$segment_status" != 75 ]]; then
    echo "First segment did not produce a clean handoff (status $segment_status)." >&2
    exit 2
  fi
  verify_handoff
  "$AMBI_PYTHON" "$helper" record-handoff --output "$AMBI_RESUME_CAMPAIGN"
  publish
  resume_mode=required
fi
remaining=$((AMBI_TRAIN_WALLTIME_SECONDS - 1800 - (SECONDS - batch_started)))
(( remaining > 0 ))
run_segment train "$remaining"
case "$segment_status" in
  0)
    "$AMBI_PYTHON" -m utils.oscar_resume_launcher done --lineage-dir "$AMBI_LINEAGE_DIR"
    publish
    ;;
  75)
    verify_handoff
    publish
    scontrol requeue "$SLURM_JOB_ID"
    ;;
  *)
    echo "Training failed with status $segment_status; no automatic requeue." >&2
    exit "$segment_status"
    ;;
esac
