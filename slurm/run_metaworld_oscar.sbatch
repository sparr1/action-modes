#!/usr/bin/env bash
#SBATCH --job-name=ambi-metaworld
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=USR1@900
# Supply measured --cpus-per-task, --mem and --time and external log paths at submission.
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit via Slurm}"
: "${SLURM_ARRAY_JOB_ID:?Submit as a job array}"
: "${SLURM_ARRAY_TASK_ID:?Missing task index}"
: "${AMBI_PROJECT_DIR:?Set the clean pinned checkout}"
: "${AMBI_EXPECTED_COMMIT:?Set the full tested commit}"
: "${AMBI_METAWORLD_ROOT:?Set an absolute shared campaign directory outside source}"
tasks=(assembly hand-insert push stick-pull)
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-3]$ ]]
task=${tasks[SLURM_ARRAY_TASK_ID]}
seed=${AMBI_METAWORLD_SEED-55}
[[ "$seed" =~ ^(0|[1-9][0-9]*)$ ]] || {
    echo "AMBI_METAWORLD_SEED must be a non-negative integer without leading zeros" >&2
    exit 1
}

cd "$AMBI_PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "$AMBI_METAWORLD_ROOT" == /* ]]
root=$(realpath -m "$AMBI_METAWORLD_ROOT")
[[ "$root" != "$PWD" && "$root" != "$PWD/"* ]]
cell="$root/$task/ambi_state/seed_$seed"
lineage="$cell/lineage"
mkdir -p "$cell"
exec 9>"$cell/launcher.lock"
if ! flock -n 9; then
    echo "Another allocation owns $cell" >&2
    exit 1
fi

if [[ -n "${AMBI_METAWORLD_NATIVE_ENV:-}" ]]; then
    source "$AMBI_METAWORLD_NATIVE_ENV"
fi
py=${AMBI_METAWORLD_PYTHON:-$PWD/environments/metaworld/python}
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export WANDB_MODE=online WANDB_DISABLE_CODE=true WANDB__REQUIRE_CORE=true
unset WANDB_RUN_ID WANDB_RESUME TDMPC2_EVAL_CSV
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
export AMBI_SEGMENT_ID="slurm-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}"
step_log="$cell/console/$AMBI_SEGMENT_ID"
cache_root="${SLURM_TMPDIR:-/tmp}/ambi-metaworld-$UID-$AMBI_SEGMENT_ID"
mkdir -p "$cache_root/inductor" "$cache_root/triton" "$cache_root/tmp"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor" TRITON_CACHE_DIR="$cache_root/triton" TMPDIR="$cache_root/tmp"

check_resume=("$py" -m utils.oscar_resume_launcher)
if [[ -f "$lineage/DONE" ]]; then
    "${check_resume[@]}" done --lineage-dir "$lineage"
    exit 0
fi
mode=new
if [[ -f "$lineage/LATEST" ]]; then
    mode=required
elif [[ -e "$lineage" ]]; then
    echo "Incomplete lineage without LATEST: $lineage" >&2
    exit 1
fi
"${check_resume[@]}" storage --durable-root "$root" --lineage-dir "$lineage"

# Reserve time to finish an episode, commit replay, and verify the handoff.
end_time=${SLURM_JOB_END_TIME:-}
if [[ ! "$end_time" =~ ^[0-9]+$ ]]; then
    iso_end=$(scontrol show job -o "$SLURM_JOB_ID" | tr ' ' '\n' | sed -n 's/^EndTime=//p')
    end_time=$(date -d "$iso_end" +%s)
fi
drain=$((end_time - $(date +%s) - 900))
(( drain > 0 )) || { echo "Allocation has less than 15 minutes remaining" >&2; exit 1; }

mkdir -p "$cell/console"
mkdir "$step_log"
export WANDB_DIR="$cache_root/wandb" WANDB_CACHE_DIR="$cache_root/wandb-cache"
export WANDB_DATA_DIR="$cache_root/wandb-data" WANDB_ARTIFACT_DIR="$cache_root/wandb-artifacts"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR"
"$py" slurm/metaworld_campaign.py production-config \
    --task "$task" --seed "$seed" --output "$step_log/ambi_state.json"
status=0
srun --unbuffered --output="$step_log/stdout.log" --error="$step_log/stderr.log" \
    "$py" -c '
import os, runpy, torch
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
torch.set_num_interop_threads(1)
torch.set_float32_matmul_precision("high")
runpy.run_path("main.py", run_name="__main__")
' --run "configs/metaworld/experiments/$task.json" \
    --alg-dir "$step_log" --num-runs 1 --resume-wandb-mode online \
    --lineage-dir "$lineage" --resume-mode "$mode" \
    --resume-checkpoint-minutes 30 --resume-checkpoint-steps 25000 --drain-after-seconds "$drain" || status=$?
case "$status" in
    0) "${check_resume[@]}" done --lineage-dir "$lineage" ;;
    75)
        "${check_resume[@]}" handoff --lineage-dir "$lineage" \
            --slurm-job-id "$SLURM_JOB_ID" --segment-id "$AMBI_SEGMENT_ID"
        scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
        ;;
    *) echo "Training exited with status $status; see $step_log" >&2; exit "$status" ;;
esac
