#!/usr/bin/env bash
#SBATCH --job-name=tdmpc2-dmc6-state
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=192:00:00
#SBATCH --array=0-17%4
#SBATCH --output=slurm/%x-%A_%a.out
#SBATCH --error=slurm/%x-%A_%a.err

set -Eeuo pipefail

# Submit from the existing action-modes repository root:
#   sbatch slurm/run_tdmpc2_dmcontrol_state.sbatch
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"

PYTHON="$PROJECT_DIR/environments/dmcontrol/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing locked DMControl interpreter: $PYTHON" >&2
  echo "Run: uv sync --project environments/dmcontrol --locked" >&2
  exit 1
fi

TASKS=(
  cartpole_swingup
  cheetah_run
  cup_catch
  finger_spin
  reacher_easy
  walker_walk
)

TASK_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
TRIAL_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
TASK_NAME="${TASKS[$TASK_INDEX]}"
SEED=$((TRIAL_INDEX + 1))
MANIFEST="configs/dmcontrol/experiments/state/${TASK_NAME}.json"
RUN_ROOT="$PROJECT_DIR/logs/dmcontrol/tdmpc2_state/${TASK_NAME}/seed_${SEED}/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
EVAL_ROOT="$PROJECT_DIR/results/dmcontrol/tdmpc2_state/${TASK_NAME}/seed_${SEED}"
EVAL_CSV="$EVAL_ROOT/job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"
WANDB_LOCAL_ROOT="${SLURM_TMPDIR:-/tmp}/tdmpc2-wandb-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
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
echo "Task: $TASK_NAME"
echo "Seed: $SEED"
echo "Manifest: $MANIFEST"
echo "Evaluation CSV: $EVAL_CSV"
echo "W&B local staging: $WANDB_LOCAL_ROOT"

srun "$PYTHON" main.py \
  --run "$MANIFEST" \
  --alg-dir configs/dmcontrol/algs \
  --log-dir "$RUN_ROOT" \
  --trial-index "$TRIAL_INDEX" \
  --num-runs 1
