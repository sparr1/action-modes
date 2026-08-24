#!/bin/bash
#SBATCH --job-name=ambi_ant_resume
#SBATCH --output=logs/ambi_ant_%j.out
#SBATCH --error=logs/ambi_ant_%j.err
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --requeue
#SBATCH --signal=USR1@3600

set -Eeuo pipefail
umask 077
batch_started=$SECONDS

fail() {
	echo "Oscar resume launcher error: $*" >&2
	exit 2
}

: "${AMBI_DURABLE_ROOT:?Set AMBI_DURABLE_ROOT to an operator-approved durable allocation directory.}"
: "${AMBI_LINEAGE_DIR:?Set AMBI_LINEAGE_DIR to a lineage below AMBI_DURABLE_ROOT.}"
: "${AMBI_DURABLE_QUOTA_LABEL:?Set AMBI_DURABLE_QUOTA_LABEL; rgao48 AMBI runs use rgao48, never data+rbalestr.}"
: "${AMBI_DURABLE_QUOTA_PATH:?Set AMBI_DURABLE_QUOTA_PATH; rgao48 AMBI runs use /oscar/home.}"
: "${SLURM_JOB_ID:?This launcher must run as a Slurm batch job.}"
[[ "$AMBI_DURABLE_QUOTA_LABEL" != "data+rbalestr" ]] || fail \
	"data+rbalestr is not an approved AMBI durable allocation; use the quota row that owns AMBI_DURABLE_ROOT"

project_dir="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
cd "$project_dir"

git_root="$(git rev-parse --show-toplevel 2>/dev/null)" || fail "not a Git checkout"
git_root="$(cd -- "$git_root" && pwd -P)"
[[ "$git_root" == "$project_dir" ]] || fail "submit directory is not the Git root"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || fail \
	"production resume requires a clean checkout"

restart_count="${SLURM_RESTART_COUNT:-0}"
[[ "$restart_count" =~ ^[0-9]+$ ]] || fail "SLURM_RESTART_COUNT must be non-negative"
if (( restart_count > 0 )); then
	resume_mode=required
else
	resume_mode="${AMBI_RESUME_MODE:-new}"
fi
[[ "$resume_mode" == new || "$resume_mode" == required ]] || fail \
	"AMBI_RESUME_MODE must be new or required"

resume_generation=""
if [[ -n "${AMBI_RESUME_GENERATION:-}" ]]; then
	if (( restart_count == 0 )); then
		[[ "$resume_mode" == required ]] || fail \
			"AMBI_RESUME_GENERATION requires initial required mode"
		resume_generation="$AMBI_RESUME_GENERATION"
	else
		echo "Ignoring previously consumed AMBI_RESUME_GENERATION; resuming LATEST."
	fi
fi

checkpoint_minutes="${AMBI_RESUME_CHECKPOINT_MINUTES:-60}"
drain_budget="${AMBI_DRAIN_AFTER_SECONDS:-341700}"
[[ "$checkpoint_minutes" =~ ^[1-9][0-9]*$ ]] || fail \
	"AMBI_RESUME_CHECKPOINT_MINUTES must be positive"
[[ "$drain_budget" =~ ^[1-9][0-9]*$ ]] || fail "AMBI_DRAIN_AFTER_SECONDS must be positive"

mkdir -p logs
module load miniforge3/25.3.0-3
: "${MAMBA_ROOT_PREFIX:?The miniforge module did not set MAMBA_ROOT_PREFIX.}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh"
conda activate ambi
ambi_python="${AMBI_PYTHON:-python}"

quota_output="$(checkquota)" || fail "checkquota failed"
quota_args=(
	--allocation "$AMBI_DURABLE_QUOTA_LABEL"
	--filesystem-path "$AMBI_DURABLE_QUOTA_PATH"
)
printf '%s\n' "$quota_output" | "$ambi_python" -m utils.oscar_resume_launcher quota \
	"${quota_args[@]}"
"$ambi_python" -m utils.oscar_resume_launcher storage \
	--durable-root "$AMBI_DURABLE_ROOT" --lineage-dir "$AMBI_LINEAGE_DIR"

if [[ -e "$AMBI_LINEAGE_DIR/DONE" ]]; then
	"$ambi_python" -m utils.oscar_resume_launcher done --lineage-dir "$AMBI_LINEAGE_DIR"
	echo "Verified DONE; the experiment is already complete."
	exit 0
fi

if (( restart_count > 0 )); then
	previous_segment="${SLURM_JOB_ID}.$((restart_count - 1))"
	"$ambi_python" -m utils.oscar_resume_launcher handoff \
		--lineage-dir "$AMBI_LINEAGE_DIR" \
		--slurm-job-id "$SLURM_JOB_ID" \
		--segment-id "$previous_segment"
	echo "Verified prior clean handoff from $previous_segment."
fi

export AMBI_SEGMENT_ID="${SLURM_JOB_ID}.${restart_count}"
segment_console_dir="$AMBI_DURABLE_ROOT/segment-console/$AMBI_SEGMENT_ID"
mkdir -p "$(dirname -- "$segment_console_dir")"
mkdir "$segment_console_dir" || fail "segment console directory already exists"
export AMBI_SEGMENT_CONSOLE_DIR="$segment_console_dir"

# The deadline is anchored at batch-script startup, so imports and preflight
# consume the same budget as training.
elapsed=$((SECONDS - batch_started))
remaining=$((drain_budget - elapsed))
(( remaining > 0 )) || fail "drain budget expired during startup"

run_config="${AMBI_RUN_CONFIG:-configs/ambi/experiments/ambi_anchor.json}"
algorithm_dir="${AMBI_ALG_DIR:-configs/ambi/algs}"
main_args=(
	"$ambi_python" main.py
	--run "$run_config"
	--alg-dir "$algorithm_dir"
	--num-runs 1
	--lineage-dir "$AMBI_LINEAGE_DIR"
	--resume-mode "$resume_mode"
	--resume-wandb-mode online
	--resume-checkpoint-minutes "$checkpoint_minutes"
	--drain-after-seconds "$remaining"
)
[[ -z "$resume_generation" ]] || main_args+=(--resume-generation "$resume_generation")

set +e
srun --unbuffered --kill-on-bad-exit=1 \
	--output="$segment_console_dir/stdout.log" \
	--error="$segment_console_dir/stderr.log" \
	"${main_args[@]}"
status=$?
set -e

case "$status" in
	0)
		"$ambi_python" -m utils.oscar_resume_launcher done \
			--lineage-dir "$AMBI_LINEAGE_DIR"
		echo "Training target complete; DONE matches LATEST and the target step."
		;;
	75)
		"$ambi_python" -m utils.oscar_resume_launcher handoff \
			--lineage-dir "$AMBI_LINEAGE_DIR" \
			--slurm-job-id "$SLURM_JOB_ID" \
			--segment-id "$AMBI_SEGMENT_ID"
		echo "Clean handoff verified; requeueing job $SLURM_JOB_ID once."
		scontrol requeue "$SLURM_JOB_ID"
		;;
	*)
		echo "Training failed with status $status; automatic requeue suppressed." >&2
		exit "$status"
		;;
esac
