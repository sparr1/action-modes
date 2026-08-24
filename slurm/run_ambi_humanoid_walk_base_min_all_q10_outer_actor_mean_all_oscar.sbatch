#!/usr/bin/env bash
#SBATCH --job-name=ambi-hw-q10-outer-mean
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
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

fail() {
  echo "Oscar Q10 outer-mean launcher error: $*" >&2
  exit 2
}

: "${SLURM_JOB_ID:?This launcher must run as a Slurm batch job.}"

project_dir="$(cd -- "${SLURM_SUBMIT_DIR:-$PWD}" && pwd -P)"
cd "$project_dir"

git_root="$(git rev-parse --show-toplevel 2>/dev/null)" || fail "not a Git checkout"
git_root="$(cd -- "$git_root" && pwd -P)"
[[ "$git_root" == "$project_dir" ]] || fail "submit directory is not the Git root"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || fail \
  "production launch requires a clean checkout"

manifest="configs/dmcontrol/experiments/ambi_humanoid_walk_base_min_all_q10_outer_actor_mean_all.json"
algorithm_dir="configs/dmcontrol/algs"
ambi_python="$project_dir/environments/dmcontrol/.venv/bin/python"
[[ -f "$manifest" ]] || fail "missing manifest: $manifest"
[[ -x "$ambi_python" ]] || fail "missing locked DMControl interpreter: $ambi_python"

source_commit="$(git rev-parse HEAD)" || fail "could not resolve source commit"
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || fail "source commit is not a full Git SHA"

export AMBI_RUN_CONFIG="$manifest"
export AMBI_ALG_DIR="$algorithm_dir"
export AMBI_PYTHON="$ambi_python"
export WANDB_MODE=online
export WANDB_DISABLE_CODE=true
export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

echo "Job: $SLURM_JOB_ID"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Task: humanoid-walk"
echo "Seed: 55"
echo "Critic heads: 10"
echo "Outer actor Q reduction: mean_all"
echo "Outer/inner target Q reductions: min_all"
echo "Inner actor Q reduction: min_all"
echo "Value-equivalence diagnostics: every 1000 outer updates, 4 MC samples"
echo "Manifest: $AMBI_RUN_CONFIG"
echo "Source commit: $source_commit"
echo "W&B mode: $WANDB_MODE"

# The canonical Oscar launcher owns durable lineage validation, exact resume,
# signal-aware draining, W&B run continuity, and one-cell requeue behavior.
exec bash "$project_dir/run_ambi_oscar.sh"
