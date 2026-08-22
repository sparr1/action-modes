#!/usr/bin/env bash
#SBATCH --job-name=ambi-hw-q10-nodrop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpus
#SBATCH --constraint=rtx_a6000
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=21-00:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -Eeuo pipefail

# Submit from a clean action-modes checkout. This is one seed-55 exploratory
# Humanoid Walk run, not an array.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"

if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Refusing to run from a dirty checkout:" >&2
  git status --short >&2
  exit 1
fi

DEFAULT_PYTHON="$PROJECT_DIR/environments/dmcontrol/.venv/bin/python"
PYTHON="${AMBI_DMCONTROL_PYTHON:-$DEFAULT_PYTHON}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing locked DMControl interpreter: $PYTHON" >&2
  echo "Set AMBI_DMCONTROL_PYTHON or run: uv sync --project environments/dmcontrol --locked" >&2
  exit 1
fi

AVAILABLE_KB=$(df -Pk "$PROJECT_DIR" | awk 'NR == 2 {print $4}')
if (( AVAILABLE_KB < 2 * 1024 * 1024 )); then
  echo "Refusing to start with less than 2 GiB free in $PROJECT_DIR" >&2
  exit 1
fi

MANIFEST="configs/dmcontrol/experiments/ambi_humanoid_walk_base_min_all_q10_inner_critic_dropout_off.json"
TRIAL_INDEX=0
SEED=55
RUN_ROOT="$PROJECT_DIR/logs/dmcontrol/ambi_q10_min_all_inner_critic_dropout_off/humanoid_walk/seed_${SEED}/job_${SLURM_JOB_ID}"
WANDB_LOCAL_ROOT="${SLURM_TMPDIR:-/tmp}/ambi-q10-nodrop-wandb-${SLURM_JOB_ID}"
mkdir -p "$WANDB_LOCAL_ROOT" "$WANDB_LOCAL_ROOT/cache" "$WANDB_LOCAL_ROOT/data" "$WANDB_LOCAL_ROOT/artifacts"

export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
export WANDB_DIR="$WANDB_LOCAL_ROOT"
export WANDB_CACHE_DIR="$WANDB_LOCAL_ROOT/cache"
export WANDB_DATA_DIR="$WANDB_LOCAL_ROOT/data"
export WANDB_ARTIFACT_DIR="$WANDB_LOCAL_ROOT/artifacts"
export WANDB_DISABLE_CODE=true

echo "Job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Task: humanoid-walk"
echo "Seed: $SEED"
echo "Agent decisions: 14000000"
echo "Raw simulator steps: 28000000"
echo "Critic heads: 10"
echo "Q reductions: min_all"
echo "Inner critic dropout: disabled"
echo "Outer critic dropout: 0.01"
echo "Manifest: $MANIFEST"
echo "Run root: $RUN_ROOT"
echo "W&B local staging: $WANDB_LOCAL_ROOT"
echo "Source commit: $(git rev-parse HEAD)"
"$PYTHON" --version
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
"$PYTHON" -c 'import os, torch; print("torch:", torch.__version__); print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES")); print("visible CUDA devices:", torch.cuda.device_count()); print("cuda:", torch.cuda.get_device_name())'

srun "$PYTHON" main.py \
  --run "$MANIFEST" \
  --alg-dir configs/dmcontrol/algs \
  --log-dir "$RUN_ROOT" \
  --alg-index 0 \
  --trial-index "$TRIAL_INDEX" \
  --num-runs 1
