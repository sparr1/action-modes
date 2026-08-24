#!/bin/bash
#SBATCH --job-name=ambi_resume_canary
#SBATCH --output=logs/ambi_resume_canary_%j.out
#SBATCH --error=logs/ambi_resume_canary_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6

set -Eeuo pipefail
umask 077

fail() {
	echo "Oscar resume canary error: $*" >&2
	exit 2
}

[[ "${AMBI_RUN_OSCAR_RESUME_CANARY:-0}" == 1 ]] || fail \
	"set AMBI_RUN_OSCAR_RESUME_CANARY=1 to authorize the live gpu-debug canary"
: "${AMBI_DURABLE_ROOT:?Set AMBI_DURABLE_ROOT to an operator-approved durable allocation directory.}"
: "${AMBI_LINEAGE_DIR:?Set AMBI_LINEAGE_DIR to a new lineage below AMBI_DURABLE_ROOT.}"
: "${AMBI_DURABLE_QUOTA_LABEL:?Set AMBI_DURABLE_QUOTA_LABEL; rgao48 AMBI runs use rgao48, never data+rbalestr.}"
: "${AMBI_DURABLE_QUOTA_PATH:?Set AMBI_DURABLE_QUOTA_PATH; rgao48 AMBI runs use /oscar/home.}"
: "${SLURM_JOB_ID:?This launcher must run as a Slurm batch job.}"
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]] || fail "canary requires a fresh allocation"
approved_durable_root="/oscar/home/rgao48/ambi-durable"
[[ "$AMBI_DURABLE_QUOTA_LABEL" != "data+rbalestr" ]] || fail \
	"data+rbalestr is not an approved AMBI durable allocation; use the quota row that owns AMBI_DURABLE_ROOT"
[[ "$AMBI_DURABLE_QUOTA_LABEL" == "rgao48" ]] || fail \
	"AMBI_DURABLE_QUOTA_LABEL must be rgao48 for Oscar AMBI runs"
[[ "$AMBI_DURABLE_QUOTA_PATH" == "/oscar/home" ]] || fail \
	"AMBI_DURABLE_QUOTA_PATH must be /oscar/home for Oscar AMBI runs"
if [[ -d /oscar/home/rgao48 ]]; then
	[[ "$AMBI_DURABLE_ROOT" == "$approved_durable_root" ]] || fail \
		"AMBI_DURABLE_ROOT must be $approved_durable_root on Oscar"
fi

project_dir="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
cd "$project_dir"
git_root="$(git rev-parse --show-toplevel 2>/dev/null)" || fail "not a Git checkout"
git_root="$(cd -- "$git_root" && pwd -P)"
[[ "$git_root" == "$project_dir" ]] || fail "submit directory is not the Git root"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || fail \
	"the live canary requires a clean checkout"

drain_seconds="${AMBI_CANARY_DRAIN_AFTER_SECONDS:-300}"
checkpoint_minutes="${AMBI_RESUME_CHECKPOINT_MINUTES:-60}"
[[ "$drain_seconds" =~ ^[1-9][0-9]*$ ]] || fail "canary drain must be positive"
(( drain_seconds <= 900 )) || fail "canary drain must not exceed 900 seconds"
[[ "$checkpoint_minutes" =~ ^[1-9][0-9]*$ ]] || fail \
	"checkpoint cadence must be positive"

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
	--durable-root "$AMBI_DURABLE_ROOT" \
	--lineage-dir "$AMBI_LINEAGE_DIR" \
	--print-metrics
[[ ! -e "$AMBI_LINEAGE_DIR" ]] || fail "AMBI_LINEAGE_DIR must not exist"

console_root="$AMBI_DURABLE_ROOT/resume-canary/$SLURM_JOB_ID"
mkdir -p "$(dirname -- "$console_root")"
mkdir "$console_root" || fail "canary output directory already exists"

canary_run="${AMBI_CANARY_RUN_CONFIG:-configs/experiments/AntAMBITDMPC2ResumeCanary.json}"
canary_alg_dir="${AMBI_CANARY_ALG_DIR:-configs/algs}"

run_segment() {
	local index="$1"
	local mode="$2"
	local segment_id="${SLURM_JOB_ID}.canary.${index}"
	local console_dir="$console_root/segment-$index"
	local status
	mkdir "$console_dir"
	export AMBI_SEGMENT_ID="$segment_id"
	export AMBI_SEGMENT_CONSOLE_DIR="$console_dir"
	set +e
	srun --unbuffered --kill-on-bad-exit=1 \
		--output="$console_dir/stdout.log" \
		--error="$console_dir/stderr.log" \
		"$ambi_python" main.py \
		--run "$canary_run" \
		--alg-dir "$canary_alg_dir" \
		--num-runs 1 \
		--lineage-dir "$AMBI_LINEAGE_DIR" \
		--resume-mode "$mode" \
		--resume-wandb-mode online \
		--resume-checkpoint-minutes "$checkpoint_minutes" \
		--drain-after-seconds "$drain_seconds"
	status=$?
	set -e
	[[ "$status" == 75 ]] || fail \
		"segment $segment_id returned $status instead of clean handoff 75"
	"$ambi_python" -m utils.oscar_resume_launcher handoff \
		--lineage-dir "$AMBI_LINEAGE_DIR" \
		--slurm-job-id "$SLURM_JOB_ID" \
		--segment-id "$segment_id"
}

run_segment 0 new
run_segment 1 required

"$ambi_python" -m utils.oscar_resume_canary verify-lineage \
	--lineage-dir "$AMBI_LINEAGE_DIR" \
	--first-segment "${SLURM_JOB_ID}.canary.0" \
	--second-segment "${SLURM_JOB_ID}.canary.1" \
	--minimum-first-step 500

benchmark_dir="$console_root/replay-benchmark"
mkdir "$benchmark_dir"
srun --unbuffered --kill-on-bad-exit=1 \
	--output="$benchmark_dir/stdout.log" \
	--error="$benchmark_dir/stderr.log" \
	"$ambi_python" -m utils.oscar_resume_canary benchmark-replay \
	--run configs/ambi/experiments/ambi_anchor.json \
	--algorithm configs/ambi/algs/ambi_anchor.json \
	--output "$console_root/REPLAY_BENCHMARK.json" \
	--durable-root "$AMBI_DURABLE_ROOT" \
	--shard-rows 100000 \
	--maximum-estimated-bytes 4000000000

echo "Two-segment resume smoke passed. Replay measurements: $console_root/REPLAY_BENCHMARK.json"
