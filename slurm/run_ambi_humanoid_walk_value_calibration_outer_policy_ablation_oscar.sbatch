#!/usr/bin/env bash
#SBATCH --job-name=ambi-hw-valcal-ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --array=0-5%6
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err

set -Eeuo pipefail

fail() {
  echo "Oscar value-calibration launcher error: $*" >&2
  exit 2
}

: "${SLURM_JOB_ID:?This launcher must run as a Slurm batch job.}"
: "${SLURM_ARRAY_JOB_ID:?This launcher must run as a Slurm array.}"
: "${SLURM_ARRAY_TASK_ID:?This launcher must run as a Slurm array.}"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] || fail \
  "SLURM_ARRAY_TASK_ID must be an integer"
(( SLURM_ARRAY_TASK_ID <= 5 )) || fail \
  "SLURM_ARRAY_TASK_ID must be in the configured range 0--5"

# Submit from the clean action-modes repository root:
#   sbatch slurm/run_ambi_humanoid_walk_value_calibration_outer_policy_ablation_oscar.sbatch
# Training logs and model checkpoints live outside the source checkout under
# the approved AMBI durable tree. Set AMBI_VALUE_CALIBRATION_CAMPAIGN to a
# stable, filesystem-safe name when a human-readable campaign identifier is
# preferable to the default array-job identifier.
# This direct array deliberately has no Slurm requeue contract: the exact-resume
# path currently accepts only one-algorithm, one-trial manifests, whereas this
# launcher selects cells from a two-algorithm, three-trial manifest.
project_dir="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
cd "$project_dir"

git_root="$(git rev-parse --show-toplevel 2>/dev/null)" || fail \
  "not a Git checkout"
git_root="$(cd -- "$git_root" && pwd -P)"
[[ "$git_root" == "$project_dir" ]] || fail \
  "submit directory is not the Git root"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || fail \
  "production launch requires a clean checkout"

manifest="configs/dmcontrol/experiments/ambi_humanoid_walk_value_calibration_outer_policy_ablation.json"
algorithm_dir="configs/dmcontrol/algs"
default_python="$project_dir/environments/dmcontrol/.venv/bin/python"
ambi_python="${AMBI_DMC_PYTHON:-$default_python}"
[[ -f "$manifest" ]] || fail "missing manifest: $manifest"
[[ -x "$ambi_python" ]] || fail \
  "missing locked DMControl interpreter: $ambi_python"

algorithm_configs=(
  ambi_humanoid_walk_base_min_all_reward_only_value_calibration
  ambi_humanoid_walk_base_min_all_reward_only_value_calibration_outer_policy_50
)
condition_labels=(baseline outer_policy_50)
alg_index=$((SLURM_ARRAY_TASK_ID / 3))
trial_index=$((SLURM_ARRAY_TASK_ID % 3))
seed=$((55 + trial_index))
algorithm_config="${algorithm_configs[$alg_index]}"
condition_label="${condition_labels[$alg_index]}"

approved_output_root="/oscar/home/rgao48/ambi-durable/value-calibration"
output_root="${AMBI_VALUE_CALIBRATION_ROOT:-$approved_output_root}"
[[ "$output_root" == /* ]] || fail \
  "AMBI_VALUE_CALIBRATION_ROOT must be an absolute path"
[[ "$output_root" != "/oscar/home" ]] || fail \
  "AMBI_VALUE_CALIBRATION_ROOT must not be the broad /oscar/home directory"
[[ "$output_root" != "/oscar/home/rgao48" ]] || fail \
  "AMBI_VALUE_CALIBRATION_ROOT must not be the broad user home directory"
if [[ -d /oscar/home/rgao48 ]]; then
  [[ "$output_root" == "$approved_output_root" ]] || fail \
    "AMBI_VALUE_CALIBRATION_ROOT must be $approved_output_root on Oscar"
fi
mkdir -p "$output_root"
output_root="$(cd -- "$output_root" && pwd -P)"

campaign="${AMBI_VALUE_CALIBRATION_CAMPAIGN:-job_${SLURM_ARRAY_JOB_ID}}"
[[ "$campaign" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail \
  "AMBI_VALUE_CALIBRATION_CAMPAIGN must be a filesystem-safe basename"
cell_output_root="$output_root/$campaign/task_${SLURM_ARRAY_TASK_ID}"
run_root="$cell_output_root"

available_kb="$(df -Pk "$output_root" | awk 'NR == 2 {print $4}')"
[[ "$available_kb" =~ ^[0-9]+$ ]] || fail \
  "could not determine free durable storage"
(( available_kb >= 2 * 1024 * 1024 )) || fail \
  "refusing to start with less than 2 GiB free in $output_root"

wandb_local_root="${SLURM_TMPDIR:-/tmp}/ambi-valcal-wandb-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
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

echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Task: humanoid-walk"
echo "Condition: $condition_label"
echo "Algorithm config: $algorithm_config"
echo "Algorithm index: $alg_index"
echo "Trial index: $trial_index"
echo "Seed: $seed"
echo "Agent decisions: 1000000"
echo "Manifest: $manifest"
echo "Campaign: $campaign"
echo "Durable output root: $output_root"
echo "Run root: $run_root"
echo "W&B mode: $WANDB_MODE"
echo "W&B local staging: $wandb_local_root"
echo "Source commit: $source_commit"
"$ambi_python" --version
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
"$ambi_python" -c 'import os, torch; print("torch:", torch.__version__); print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES")); print("visible CUDA devices:", torch.cuda.device_count()); print("cuda:", torch.cuda.get_device_name())'

srun --unbuffered --kill-on-bad-exit=1 \
  "$ambi_python" main.py \
  --run "$manifest" \
  --alg-dir "$algorithm_dir" \
  --log-dir "$run_root" \
  --alg-index "$alg_index" \
  --trial-index "$trial_index" \
  --num-runs 1
