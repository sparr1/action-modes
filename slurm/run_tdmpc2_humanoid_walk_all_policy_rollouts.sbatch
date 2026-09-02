#!/usr/bin/env bash
#SBATCH --job-name=tdmpc2-hw-allpi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=30-00:00:00
#SBATCH --array=0-2%3
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err

set -Eeuo pipefail

# Select the requested Hydra node at submission time, for example:
#   sbatch --nodelist=gpu2301 slurm/run_tdmpc2_humanoid_walk_all_policy_rollouts.sbatch
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"

PYTHON="$PROJECT_DIR/environments/dmcontrol/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing locked DMControl interpreter: $PYTHON" >&2
  echo "Run: uv sync --project environments/dmcontrol --locked" >&2
  exit 1
fi

AVAILABLE_KB=$(df -Pk "$PROJECT_DIR" | awk 'NR == 2 {print $4}')
if (( AVAILABLE_KB < 2 * 1024 * 1024 )); then
  echo "Refusing to start with less than 2 GiB free in $PROJECT_DIR" >&2
  exit 1
fi

MANIFEST="configs/dmcontrol/experiments/tdmpc2_humanoid_walk_all_policy_rollouts.json"
TRIAL_INDEX="$SLURM_ARRAY_TASK_ID"
SEED=$((TRIAL_INDEX + 1))
RUN_ROOT="$PROJECT_DIR/logs/dmcontrol/tdmpc2_all_policy_rollouts/humanoid_walk/seed_${SEED}/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
EVAL_ROOT="$PROJECT_DIR/results/dmcontrol/tdmpc2_all_policy_rollouts/humanoid_walk/seed_${SEED}"
EVAL_CSV="$EVAL_ROOT/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"
WANDB_LOCAL_ROOT="${SLURM_TMPDIR:-/tmp}/tdmpc2-allpi-wandb-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$EVAL_ROOT"
mkdir -p "$WANDB_LOCAL_ROOT" "$WANDB_LOCAL_ROOT/cache" "$WANDB_LOCAL_ROOT/data" "$WANDB_LOCAL_ROOT/artifacts"

export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TDMPC2_EVAL_CSV="$EVAL_CSV"
export WANDB_MODE=online
export WANDB_DIR="$WANDB_LOCAL_ROOT"
export WANDB_CACHE_DIR="$WANDB_LOCAL_ROOT/cache"
export WANDB_DATA_DIR="$WANDB_LOCAL_ROOT/data"
export WANDB_ARTIFACT_DIR="$WANDB_LOCAL_ROOT/artifacts"
export WANDB_DISABLE_CODE=true

echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $SLURM_NODELIST"
echo "Task: humanoid-walk"
echo "Seed: $SEED"
echo "Agent decisions: 14000000"
echo "Policy proposal trajectories: 512 / 512"
echo "Manifest: $MANIFEST"
echo "Evaluation CSV: $EVAL_CSV"
echo "W&B local staging: $WANDB_LOCAL_ROOT"
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
