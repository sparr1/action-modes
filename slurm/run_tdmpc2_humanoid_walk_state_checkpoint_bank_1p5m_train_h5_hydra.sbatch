#!/usr/bin/env bash
#SBATCH --job-name=tdmpc2-hw-h5-bank
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2501
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --output=/cs/home/rgao48/projects/ambi-runs/slurm/%x-%j.out
#SBATCH --error=/cs/home/rgao48/projects/ambi-runs/slurm/%x-%j.err

set -Eeuo pipefail

# Submit from a clean checkout of the recorded experiment commit.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"

PYTHON="${AMBI_DMC_PYTHON:-/cs/home/rgao48/projects/action-modes/environments/dmcontrol/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing locked DMControl interpreter: $PYTHON" >&2
  echo "Set AMBI_DMC_PYTHON to a locked DMControl interpreter." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Refusing to train from a checkout with uncommitted files." >&2
  git status --short --untracked-files=normal >&2
  exit 1
fi

MANIFEST="configs/dmcontrol/experiments/tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m_train_h5.json"
TRIAL_INDEX=0
SEED=1
ARTIFACT_ROOT="/cs/home/rgao48/projects/ambi-runs/tdmpc2-checkpoint-bank-1p5m-train-h5"
RUN_ROOT="$ARTIFACT_ROOT/seed_${SEED}/job_${SLURM_JOB_ID}"
WANDB_LOCAL_ROOT="${SLURM_TMPDIR:-/tmp}/tdmpc2-h5-ckptbank-wandb-${SLURM_JOB_ID}"
mkdir -p "$ARTIFACT_ROOT"
mkdir -p "$WANDB_LOCAL_ROOT" "$WANDB_LOCAL_ROOT/cache" "$WANDB_LOCAL_ROOT/data" "$WANDB_LOCAL_ROOT/artifacts"

AVAILABLE_KB=$(df -Pk "$ARTIFACT_ROOT" | awk 'NR == 2 {print $4}')
if (( AVAILABLE_KB < 4 * 1024 * 1024 )); then
  echo "Refusing to start with less than 4 GiB free for checkpoint artifacts." >&2
  exit 1
fi

export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
export WANDB_DIR="$WANDB_LOCAL_ROOT"
export WANDB_CACHE_DIR="$WANDB_LOCAL_ROOT/cache"
export WANDB_DATA_DIR="$WANDB_LOCAL_ROOT/data"
export WANDB_ARTIFACT_DIR="$WANDB_LOCAL_ROOT/artifacts"
export WANDB_DISABLE_CODE=true

SOURCE_COMMIT=$(git rev-parse HEAD)
echo "Job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Source commit: $SOURCE_COMMIT"
echo "Task: humanoid-walk"
echo "Seed: $SEED"
echo "Agent decisions: 1500000"
echo "Raw simulator control steps: 3000000"
echo "Training unroll horizon: 5"
echo "Outer planning horizon: 3"
echo "Checkpoint cadence: 25000"
echo "Expected numbered checkpoints: 60"
echo "Manifest: $MANIFEST"
echo "Artifact root: $RUN_ROOT"
echo "W&B local staging: $WANDB_LOCAL_ROOT"
echo "Python: $PYTHON"
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
