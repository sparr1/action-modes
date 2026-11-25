#!/usr/bin/env bash
#SBATCH --partition=gpus
#SBATCH --account=default
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --hint=nomultithread
#SBATCH --requeue
#SBATCH --signal=USR1@1800
#SBATCH --open-mode=append
set -Eeuo pipefail
umask 077
task="${1:?Set a Q10 task}"
seed="${2:-56}"
case "$seed:$task" in
    56:dog-stand|56:dog-trot|56:dog-walk) campaign=mrsq-q10-dogs-s56-hydra-20261124 ;;
    57:dog-trot|57:dog-walk|57:humanoid-stand) campaign=mrsq-q10-seed57-hydra-20261125 ;;
    *) exit 2 ;;
esac
: "${AMBI_EXPECTED_COMMIT:?Pin the tested checkout}"
: "${AMBI_PYTHON:?Use the locked DMControl interpreter}"
cd "${SLURM_SUBMIT_DIR:?}"
[[ "$(git rev-parse --show-toplevel)" == "$PWD" ]]
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]

root="/data/public/rgao48/ambi/$campaign/$task-s$seed"
mkdir -p "$root"
exec 9>"$root/launcher.lock"
flock -n 9
lineage="$root/lineage"
prefix="$SLURM_JOB_ID.${SLURM_RESTART_COUNT:-0}"
source slurm/hydra_q10_local_cache.sh
setup_hydra_q10_local_cache /ltmp "$prefix"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
export WANDB_DISABLE_CODE=true WANDB__REQUIRE_CORE=true WANDB_MODE=disabled
export WANDB_INIT_TIMEOUT=90 WANDB_HTTP_TIMEOUT=30 WANDB__SERVICE_WAIT=60
unset CUDA_LAUNCH_BLOCKING WANDB_RUN_ID WANDB_RESUME

check_resume=("$AMBI_PYTHON" -B -m utils.oscar_resume_launcher)
"${check_resume[@]}" storage --durable-root "$root" --lineage-dir "$lineage" --print-metrics
if [[ -e "$lineage/DONE" ]]; then
    "${check_resume[@]}" done --lineage-dir "$lineage"
    exit 0
fi
resume_mode=new
if [[ -s "$lineage/LATEST" && -s "$lineage/LINEAGE.json" ]]; then
    resume_mode=required
elif [[ -e "$lineage" ]]; then
    echo "Incomplete lineage without a committed checkpoint: $lineage" >&2
    exit 2
fi

end_time="${SLURM_JOB_END_TIME:-}"
if [[ ! "$end_time" =~ ^[0-9]+$ ]]; then
    iso_end=$(scontrol show job -o "$SLURM_JOB_ID" | tr ' ' '\n' | sed -n 's/^EndTime=//p')
    end_time=$(date -d "$iso_end" +%s)
fi
helper=slurm/mrsq_q10_hydra.py
srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -B "$helper" node-check "$task" --seed "$seed"
export WANDB_MODE=online

run_segment() {
    export AMBI_SEGMENT_ID="$prefix.$1"
    local console="$root/console/$AMBI_SEGMENT_ID"
    mkdir -p "$(dirname "$console")"
    mkdir "$console"
    segment_status=0
    srun --unbuffered --kill-on-bad-exit=1 --output="$console/stdout.log" --error="$console/stderr.log" \
        "$AMBI_PYTHON" -B "$helper" train "$task" --seed "$seed" --resume-mode "$resume_mode" --drain-seconds "$2" \
        || segment_status=$?
}

verify_handoff() {
    "${check_resume[@]}" handoff --lineage-dir "$lineage" \
        --slurm-job-id "$SLURM_JOB_ID" --segment-id "$AMBI_SEGMENT_ID"
}

if [[ "$resume_mode" == new ]]; then
    run_segment verify 1200
    if [[ "$segment_status" != 75 ]]; then
        echo "First segment did not produce a clean handoff (status $segment_status)." >&2
        exit 2
    fi
    verify_handoff
    "$AMBI_PYTHON" -B "$helper" record-handoff "$task" --seed "$seed"
    resume_mode=required
fi

remaining=$((end_time - $(date +%s) - 1800))
(( remaining > 0 ))
run_segment train "$remaining"
case "$segment_status" in
    0) "${check_resume[@]}" done --lineage-dir "$lineage" ;;
    75) verify_handoff; cleanup_hydra_q10_local_cache; scontrol requeue "$SLURM_JOB_ID" ;;
    *) echo "Training failed with status $segment_status. Full checkpoints are in $lineage." >&2
       exit "$segment_status" ;;
esac
