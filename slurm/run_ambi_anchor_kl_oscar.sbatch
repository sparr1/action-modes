#!/usr/bin/env bash
#SBATCH --job-name=ambi-hw-anchor-kl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --array=0-8%6
#SBATCH --requeue
#SBATCH --signal=USR1@3600
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err

set -Eeuo pipefail

fail() {
  echo "Oscar behavior-KL launcher error: $*" >&2
  exit 2
}

: "${SLURM_JOB_ID:?This launcher must run as a Slurm batch job.}"
: "${SLURM_ARRAY_JOB_ID:?This launcher must run as a Slurm array.}"
: "${SLURM_ARRAY_TASK_ID:?This launcher must run as a Slurm array.}"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] || fail \
  "SLURM_ARRAY_TASK_ID must be an integer"
(( SLURM_ARRAY_TASK_ID <= 8 )) || fail \
  "SLURM_ARRAY_TASK_ID must be in the configured range 0--8"

# Submit from the clean action-modes repository root:
#   sbatch --export=ALL,AMBI_VALUE_CALIBRATION_CAMPAIGN=<unique-campaign> \
#     slurm/run_ambi_anchor_kl_oscar.sbatch
# Each array cell selects one tracked single-trial manifest and delegates
# signal-aware draining, exact resume, and requeue to run_ambi_oscar.sh.
project_dir="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
cd "$project_dir"

git_root="$(git rev-parse --show-toplevel 2>/dev/null)" || fail \
  "not a Git checkout"
git_root="$(cd -- "$git_root" && pwd -P)"
[[ "$git_root" == "$project_dir" ]] || fail \
  "submit directory is not the Git root"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || fail \
  "production launch requires a clean checkout"

algorithm_dir="configs/dmcontrol/algs"
default_python="$project_dir/environments/dmcontrol/.venv/bin/python"
ambi_python="${AMBI_DMC_PYTHON:-$default_python}"
[[ -x "$ambi_python" ]] || fail \
  "missing locked DMControl interpreter: $ambi_python"

algorithm_configs=(
  ambi_anchor_kl_smooth
  ambi_anchor_kl_quantile
  ambi_anchor_kl_dual
)
schedule_labels=(smooth quantile_gate dual)
variant_index=$((SLURM_ARRAY_TASK_ID % 3))
trial_index=$((SLURM_ARRAY_TASK_ID / 3))
seed=$((55 + trial_index))
algorithm_config="${algorithm_configs[$variant_index]}"
schedule_label="${schedule_labels[$variant_index]}"
manifest="configs/dmcontrol/experiments/${algorithm_config}_14m_seed${seed}.json"
[[ -f "$manifest" ]] || fail "missing manifest: $manifest"
[[ -f "$algorithm_dir/$algorithm_config.json" ]] || fail \
  "missing algorithm config: $algorithm_dir/$algorithm_config.json"

campaign="${AMBI_VALUE_CALIBRATION_CAMPAIGN:-job_${SLURM_ARRAY_JOB_ID}}"
[[ "$campaign" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail \
  "AMBI_VALUE_CALIBRATION_CAMPAIGN must be a filesystem-safe basename"
approved_durable_root="/oscar/home/rgao48/ambi-durable"
lineage_dir="$approved_durable_root/value-calibration/$campaign/task_${SLURM_ARRAY_TASK_ID}"

wandb_local_root="${SLURM_TMPDIR:-/tmp}/ambi-anchor-kl-wandb-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p \
  "$wandb_local_root" \
  "$wandb_local_root/cache" \
  "$wandb_local_root/data" \
  "$wandb_local_root/artifacts"

source_commit="$(git rev-parse HEAD)" || fail "could not resolve source commit"
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || fail \
  "source commit is not a full Git SHA"

export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
export WANDB_DIR="$wandb_local_root"
export WANDB_CACHE_DIR="$wandb_local_root/cache"
export WANDB_DATA_DIR="$wandb_local_root/data"
export WANDB_ARTIFACT_DIR="$wandb_local_root/artifacts"
export WANDB_DISABLE_CODE=true
export AMBI_DURABLE_ROOT="$approved_durable_root"
export AMBI_LINEAGE_DIR="$lineage_dir"
export AMBI_DURABLE_QUOTA_LABEL="rgao48"
export AMBI_DURABLE_QUOTA_PATH="/oscar/home"
export AMBI_RUN_CONFIG="$manifest"
export AMBI_ALG_DIR="$algorithm_dir"
export AMBI_PYTHON="$ambi_python"

echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Task: humanoid-walk"
echo "Behavior-KL schedule: $schedule_label"
echo "Algorithm config: $algorithm_config"
echo "Variant index: $variant_index"
echo "Trial index: $trial_index"
echo "Seed: $seed"
echo "Agent decisions: 14000000"
echo "Raw simulator steps: 28000000"
echo "Manifest: $manifest"
echo "Campaign: $campaign"
echo "Durable lineage: $lineage_dir"
echo "W&B mode: $WANDB_MODE"
echo "W&B local staging: $wandb_local_root"
echo "Source commit: $source_commit"

# The canonical Oscar launcher owns quota/storage validation, hourly durable
# checkpoints, signal-aware draining, W&B continuity, and one-cell requeue.
exec bash "$project_dir/run_ambi_oscar.sh"
