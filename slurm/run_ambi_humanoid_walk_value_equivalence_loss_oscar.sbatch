#!/usr/bin/env bash
#SBATCH --job-name=ambi-hw-ve-loss
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --requeue
#SBATCH --signal=USR1@3600
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -Eeuo pipefail

fail() {
  echo "Oscar value-equivalence-loss launcher error: $*" >&2
  exit 2
}

: "${SLURM_JOB_ID:?This launcher must run as a Slurm batch job.}"
: "${AMBI_EXPECTED_COMMIT:?Set AMBI_EXPECTED_COMMIT to the exact pushed commit SHA.}"
: "${AMBI_VALUE_EQUIVALENCE_CAMPAIGN:?Set AMBI_VALUE_EQUIVALENCE_CAMPAIGN to a unique campaign basename.}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]] || fail \
  "SLURM_JOB_ID must identify one non-array job"
[[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || fail \
  "the value-equivalence-loss screen must not run as a Slurm array"
[[ "$AMBI_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail \
  "AMBI_EXPECTED_COMMIT must be one full lowercase Git SHA"
[[ "$AMBI_VALUE_EQUIVALENCE_CAMPAIGN" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail \
  "AMBI_VALUE_EQUIVALENCE_CAMPAIGN must be a filesystem-safe basename"

# Submit this single-cell screen from a clean Oscar checkout whose HEAD and
# origin/ambi-inner-loop both equal AMBI_EXPECTED_COMMIT. The specialized
# wrapper performs only source/configuration selection and staging; the
# canonical launcher owns durable-storage validation, exact resume, signal
# draining, and requeue.
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
origin_commit="$(git rev-parse 'origin/ambi-inner-loop^{commit}')" || fail \
  "could not resolve origin/ambi-inner-loop"
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || fail \
  "source commit is not a full Git SHA"
[[ "$origin_commit" =~ ^[0-9a-f]{40}$ ]] || fail \
  "origin/ambi-inner-loop is not a full Git SHA"
[[ "$source_commit" == "$AMBI_EXPECTED_COMMIT" ]] || fail \
  "HEAD does not match AMBI_EXPECTED_COMMIT"
[[ "$origin_commit" == "$AMBI_EXPECTED_COMMIT" ]] || fail \
  "origin/ambi-inner-loop does not match AMBI_EXPECTED_COMMIT"

algorithm_config="ambi_humanoid_walk_q5_pair2_ve_loss_c0p1_mc4_1m_seed55"
manifest="configs/dmcontrol/experiments/${algorithm_config}.json"
algorithm_dir="configs/dmcontrol/algs"
default_python="$project_dir/environments/dmcontrol/.venv/bin/python"
ambi_python="${AMBI_DMCONTROL_PYTHON:-$default_python}"
[[ -f "$manifest" ]] || fail "missing manifest: $manifest"
[[ -f "$algorithm_dir/$algorithm_config.json" ]] || fail \
  "missing algorithm config: $algorithm_dir/$algorithm_config.json"
[[ -f "$project_dir/run_ambi_oscar.sh" ]] || fail \
  "missing canonical Oscar launcher: $project_dir/run_ambi_oscar.sh"
[[ -x "$ambi_python" ]] || fail \
  "missing locked DMControl interpreter: $ambi_python"

restart_count="${SLURM_RESTART_COUNT:-0}"
[[ "$restart_count" =~ ^[0-9]+$ ]] || fail \
  "SLURM_RESTART_COUNT must be non-negative"
wandb_staging_parent="${SLURM_TMPDIR:-/tmp}"
[[ "$wandb_staging_parent" == /* ]] || fail \
  "SLURM_TMPDIR must be an absolute path when set"

approved_durable_root="/oscar/home/rgao48/ambi-durable"
lineage_dir="$approved_durable_root/value-equivalence-loss/$AMBI_VALUE_EQUIVALENCE_CAMPAIGN/seed_55"
wandb_local_root="$wandb_staging_parent/ambi-ve-loss-wandb-${SLURM_JOB_ID}-${restart_count}"
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
export AMBI_DURABLE_QUOTA_PATH="/oscar/home"
export AMBI_RUN_CONFIG="$manifest"
export AMBI_ALG_DIR="$algorithm_dir"
export AMBI_PYTHON="$ambi_python"

echo "Job: $SLURM_JOB_ID"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Task: humanoid-walk"
echo "Algorithm config: $algorithm_config"
echo "Seed: 55"
echo "Agent decisions: 1000000"
echo "Value-equivalence loss coefficient: 0.1"
echo "Value-equivalence loss MC samples: 4"
echo "Manifest: $manifest"
echo "Campaign: $AMBI_VALUE_EQUIVALENCE_CAMPAIGN"
echo "Durable lineage: $lineage_dir"
echo "W&B mode: $WANDB_MODE"
echo "W&B local staging: $wandb_local_root"
echo "Source commit: $source_commit"
echo "origin/ambi-inner-loop: $origin_commit"

exec bash "$project_dir/run_ambi_oscar.sh"
