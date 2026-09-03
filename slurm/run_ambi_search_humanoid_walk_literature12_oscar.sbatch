#!/usr/bin/env bash
#SBATCH --job-name=ambi-search-hw-lit12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --array=0-11%12
#SBATCH --requeue
#SBATCH --signal=USR1@3600
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err

set -Eeuo pipefail
umask 077

fail() {
  echo "Oscar AMBI Search literature12 launcher error: $*" >&2
  exit 2
}

: "${SLURM_JOB_ID:?This launcher must run as a Slurm batch job.}"
: "${SLURM_ARRAY_JOB_ID:?This launcher must run as a Slurm array.}"
: "${SLURM_ARRAY_TASK_ID:?This launcher must run as a Slurm array.}"
: "${AMBI_EXPECTED_COMMIT:?Set AMBI_EXPECTED_COMMIT to the exact pushed commit SHA.}"
: "${AMBI_SEARCH_CAMPAIGN:?Set AMBI_SEARCH_CAMPAIGN to a unique campaign basename.}"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] || fail \
  "SLURM_ARRAY_TASK_ID must be an integer"
(( SLURM_ARRAY_TASK_ID <= 11 )) || fail \
  "SLURM_ARRAY_TASK_ID must be in the configured range 0--11"
[[ "$AMBI_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail \
  "AMBI_EXPECTED_COMMIT must be one full lowercase Git SHA"
[[ "$AMBI_SEARCH_CAMPAIGN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail \
  "AMBI_SEARCH_CAMPAIGN must be a filesystem-safe basename"

# Submit from a clean Oscar checkout whose HEAD and origin/ambisearch both
# equal AMBI_EXPECTED_COMMIT. Each task owns one tracked single-cell manifest
# and one durable exact-resume lineage.
project_dir="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
cd "$project_dir"

git_root="$(git rev-parse --show-toplevel 2>/dev/null)" || fail \
  "not a Git checkout"
git_root="$(cd -- "$git_root" && pwd -P)"
[[ "$git_root" == "$project_dir" ]] || fail \
  "submit directory is not the Git root"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || fail \
  "production launch requires a clean checkout"

source_commit="$(git rev-parse HEAD)" || fail \
  "could not resolve the source commit"
origin_commit="$(git rev-parse 'origin/ambisearch^{commit}')" || fail \
  "could not resolve origin/ambisearch"
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || fail \
  "source commit is not a full Git SHA"
[[ "$origin_commit" =~ ^[0-9a-f]{40}$ ]] || fail \
  "origin/ambisearch is not a full Git SHA"
[[ "$source_commit" == "$AMBI_EXPECTED_COMMIT" ]] || fail \
  "HEAD does not match AMBI_EXPECTED_COMMIT"
[[ "$origin_commit" == "$AMBI_EXPECTED_COMMIT" ]] || fail \
  "origin/ambisearch does not match AMBI_EXPECTED_COMMIT"

algorithm_configs=(
  00_full_suffix__shared_a1
  01_full_suffix__depth_a1
  02_full_suffix__stage_a1
  03_full_suffix__shared_a4
  04_full_suffix__depth_a4
  05_lambda_online__shared_a4
  06_lambda_online__depth_a4
  07_lambda_online__shared_a12
  08_vtrace__shared_a4
  09_vtrace__depth_a4
  10_hard_propagation__stage_td0
  11_polyak_ablation__depth_lambda_tau005
)
critic_layouts=(
  shared depth_conditioned stage_heads shared depth_conditioned shared
  depth_conditioned shared shared depth_conditioned stage_heads depth_conditioned
)
return_estimators=(
  full_suffix full_suffix full_suffix full_suffix full_suffix lambda_return
  lambda_return lambda_return vtrace vtrace td0 lambda_return
)
bootstrap_strategies=(
  none none none none none online online online target_hard target_hard
  depth_stage_hard target_polyak_tau005
)
critic_updates=(3 3 3 3 3 3 3 3 1 1 1 3)
actor_updates=(1 1 1 4 4 4 4 12 4 4 1 4)

algorithm_config="${algorithm_configs[$SLURM_ARRAY_TASK_ID]}"
critic_layout="${critic_layouts[$SLURM_ARRAY_TASK_ID]}"
return_estimator="${return_estimators[$SLURM_ARRAY_TASK_ID]}"
bootstrap_strategy="${bootstrap_strategies[$SLURM_ARRAY_TASK_ID]}"
critic_updates_per_round="${critic_updates[$SLURM_ARRAY_TASK_ID]}"
actor_updates_per_round="${actor_updates[$SLURM_ARRAY_TASK_ID]}"
campaign_root="configs/research/ambi_search_humanoid_walk_literature12"
algorithm_dir="$campaign_root/algs"
manifest="$campaign_root/experiments/${algorithm_config}.json"
[[ -f "$manifest" ]] || fail "missing manifest: $manifest"
[[ -f "$algorithm_dir/$algorithm_config.json" ]] || fail \
  "missing algorithm config: $algorithm_dir/$algorithm_config.json"
[[ -f "$project_dir/run_ambi_oscar.sh" ]] || fail \
  "missing canonical Oscar launcher: $project_dir/run_ambi_oscar.sh"

default_python="$project_dir/environments/dmcontrol/.venv/bin/python"
ambi_python="${AMBI_DMCONTROL_PYTHON:-$default_python}"
[[ -x "$ambi_python" ]] || fail \
  "missing locked DMControl interpreter: $ambi_python"
restart_count="${SLURM_RESTART_COUNT:-0}"
[[ "$restart_count" =~ ^[0-9]+$ ]] || fail \
  "SLURM_RESTART_COUNT must be non-negative"
wandb_staging_parent="${SLURM_TMPDIR:-/tmp}"
[[ "$wandb_staging_parent" == /* ]] || fail \
  "SLURM_TMPDIR must be an absolute path when set"

approved_durable_root="/oscar/scratch/rgao48/ambi-durable"
lineage_dir="$approved_durable_root/ambisearch-literature12/$AMBI_SEARCH_CAMPAIGN/task_${SLURM_ARRAY_TASK_ID}"
wandb_local_root="$wandb_staging_parent/ambi-search-lit12-wandb-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}-${restart_count}"
mkdir -p \
  "$wandb_local_root" \
  "$wandb_local_root/cache" \
  "$wandb_local_root/data" \
  "$wandb_local_root/artifacts"

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
export AMBI_DURABLE_QUOTA_PATH="/oscar/scratch"
export AMBI_RUN_CONFIG="$manifest"
export AMBI_ALG_DIR="$algorithm_dir"
export AMBI_PYTHON="$ambi_python"

echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Task: humanoid-walk"
echo "Algorithm config: $algorithm_config"
echo "Critic layout: $critic_layout"
echo "Return estimator: $return_estimator"
echo "Bootstrap/target: $bootstrap_strategy"
echo "Search rounds: 8"
echo "Rollouts per round: 512"
echo "Rollout horizon: 3"
echo "Model transitions per real action: 12288"
echo "Inner batch size: 512"
echo "Critic updates per round: $critic_updates_per_round"
echo "Actor updates per round: $actor_updates_per_round"
echo "Inner actor learning rate: 0.00025"
echo "Inner critic learning rate: 0.00025"
echo "Inner policy-to-prior KL coefficient: 0.0"
echo "Outer behavior-policy KL schedule: none"
echo "Outer behavior-policy KL coefficient: 0.0"
echo "Inner temperature mode: inherit_outer"
echo "Agent decisions: 14000000"
echo "Raw simulator steps: 28000000"
echo "Manifest: $manifest"
echo "Campaign: $AMBI_SEARCH_CAMPAIGN"
echo "Durable lineage: $lineage_dir"
echo "W&B mode: $WANDB_MODE"
echo "W&B local staging: $wandb_local_root"
echo "Source commit: $source_commit"
echo "origin/ambisearch: $origin_commit"

exec bash "$project_dir/run_ambi_oscar.sh"
