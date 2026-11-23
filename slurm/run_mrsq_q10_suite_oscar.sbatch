#!/usr/bin/env bash
#SBATCH --job-name=mrsq-q10-suite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=default
#SBATCH --partition=gpu
#SBATCH --qos=pri-gpu+
#SBATCH --gres=gpu:l40s:1
#SBATCH --hint=nomultithread
#SBATCH --exclude=gpu2708
set -Eeuo pipefail
umask 077
batch_started=$SECONDS
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${SLURM_CPUS_PER_TASK:?Set measured CPUs}"
: "${AMBI_EXPECTED_COMMIT:?Pin the tested source}"
: "${AMBI_PYTHON:?Use the locked DMControl interpreter}"
: "${AMBI_Q10_CAMPAIGN:?Set campaign scratch storage}"
mode="${1:?Use benchmark or train}"
task="${2:?Set task}"
[[ "$mode" == benchmark || "$mode" == train ]]
case "$task" in humanoid-walk|humanoid-run|dog-trot|dog-run) ;; *) exit 2 ;; esac
[[ "$AMBI_Q10_CAMPAIGN" == /oscar/scratch/rgao48/ambi/mrsq-q10-suite-20261122 ]]
cd "${SLURM_SUBMIT_DIR:?}"
[[ "$(git rev-parse --show-toplevel)" == "$PWD" ]]
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
restart="${SLURM_RESTART_COUNT:-0}"
[[ "$restart" =~ ^[0-9]+$ ]]
task_root="$AMBI_Q10_CAMPAIGN/$task"
cache="$task_root/cache/$SLURM_JOB_ID.$restart"
mkdir -p -- "$(dirname "$cache")"
mkdir -- "$cache"
mkdir -- "$cache/"{inductor,triton,cache,tmp,wandb,data,artifacts}
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
export WANDB_MODE=disabled
helper=slurm/mrsq_q10_suite.py
if [[ "$mode" == benchmark ]]; then
    (( SLURM_CPUS_PER_TASK >= 2 ))
    srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" "$helper" benchmark "$task"
    exit 0
fi

walltime="$("$AMBI_PYTHON" "$helper" allocation "$task")"
(( walltime > 3600 ))
lineage="$task_root/lineage"
checkquota | "$AMBI_PYTHON" -m utils.oscar_resume_launcher quota \
    --allocation rgao48 --filesystem-path /oscar/scratch
"$AMBI_PYTHON" -m utils.oscar_resume_launcher storage \
    --durable-root "$task_root" --lineage-dir "$lineage"
if [[ -e "$lineage/DONE" ]]; then
    "$AMBI_PYTHON" -m utils.oscar_resume_launcher done --lineage-dir "$lineage"
    exit 0
fi

resume_mode="${AMBI_RESUME_MODE:-new}"
if (( restart > 0 )); then
    "$AMBI_PYTHON" -m utils.oscar_resume_launcher handoff --lineage-dir "$lineage" \
        --slurm-job-id "$SLURM_JOB_ID" --segment-id "$SLURM_JOB_ID.$((restart-1)).train"
    resume_mode=required
fi
[[ "$resume_mode" == new || "$resume_mode" == required ]]
if [[ "$resume_mode" == new ]]; then
    [[ ! -e "$lineage" ]]
else
    [[ -s "$lineage/LINEAGE.json" && -s "$lineage/LATEST" ]]
fi
srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" "$helper" node-check "$task" \
    --output "$task_root/node-checks/$SLURM_JOB_ID.$restart.json"
export WANDB_MODE=online

run_segment() {
    export AMBI_SEGMENT_ID="$SLURM_JOB_ID.$restart.$1"
    export AMBI_SEGMENT_CONSOLE_DIR="$task_root/console/$AMBI_SEGMENT_ID"
    mkdir -p -- "$(dirname "$AMBI_SEGMENT_CONSOLE_DIR")"
    mkdir -- "$AMBI_SEGMENT_CONSOLE_DIR"
    segment_status=0
    srun --unbuffered --kill-on-bad-exit=1 \
        --output="$AMBI_SEGMENT_CONSOLE_DIR/stdout.log" --error="$AMBI_SEGMENT_CONSOLE_DIR/stderr.log" \
        "$AMBI_PYTHON" "$helper" train "$task" --output "$task_root/launches/$AMBI_SEGMENT_ID" \
        --resume-mode "$resume_mode" --drain-seconds "$2" || segment_status=$?
}

verify_handoff() {
    "$AMBI_PYTHON" -m utils.oscar_resume_launcher handoff --lineage-dir "$lineage" \
        --slurm-job-id "$SLURM_JOB_ID" --segment-id "$AMBI_SEGMENT_ID"
}

if [[ "$resume_mode" == new ]]; then
    # Exercise the actual trained state and restore it inside the same allocation.
    run_segment verify 1200
    if [[ "$segment_status" != 75 ]]; then
        echo "First segment did not produce a clean handoff (status $segment_status)." >&2
        exit 2
    fi
    verify_handoff
    "$AMBI_PYTHON" "$helper" record-handoff "$task"
    resume_mode=required
fi

remaining=$((walltime - 1800 - (SECONDS - batch_started)))
(( remaining > 0 ))
run_segment train "$remaining"
case "$segment_status" in
    0) "$AMBI_PYTHON" -m utils.oscar_resume_launcher done --lineage-dir "$lineage" ;;
    75) verify_handoff; scontrol requeue "$SLURM_JOB_ID" ;;
    *) echo "Training failed with status $segment_status. Inspect this lineage before retrying." >&2
       exit "$segment_status" ;;
esac
