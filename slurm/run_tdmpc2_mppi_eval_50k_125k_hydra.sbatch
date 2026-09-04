#!/usr/bin/env bash
#SBATCH --job-name=tdmpc2-mppi-50k125k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2501
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/cs/home/rgao48/projects/ambi-runs/slurm/%x-%A_%a.out
#SBATCH --error=/cs/home/rgao48/projects/ambi-runs/slurm/%x-%A_%a.err

set -Eeuo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"

PYTHON="${AMBI_DMC_PYTHON:-/cs/home/rgao48/projects/action-modes/environments/dmcontrol/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing locked DMControl interpreter: $PYTHON" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Refusing to evaluate from a checkout with uncommitted files." >&2
  git status --short --untracked-files=normal >&2
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:?missing array task id}"
case "$TASK_ID" in
  0|1) STEP=50000 ;;
  2|3) STEP=75000 ;;
  4|5) STEP=100000 ;;
  6|7) STEP=125000 ;;
  8|9) STEP=150000 ;;
  *)
    echo "Unsupported array task: $TASK_ID" >&2
    exit 1
    ;;
esac

if (( TASK_ID % 2 == 0 )); then
  LABEL="train-h3"
  MODEL_DIR="/cs/home/rgao48/projects/ambi-runs/tdmpc2-checkpoint-bank-1p5m/seed_1/job_34489/tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m_2026-09-04_14-18-37/models"
  CHECKPOINT="$MODEL_DIR/model:tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m_0_${STEP}"
else
  LABEL="train-h5"
  MODEL_DIR="/cs/home/rgao48/projects/ambi-runs/tdmpc2-checkpoint-bank-1p5m-train-h5/seed_1/job_34490/tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m_train_h5_2026-09-04_14-18-37/models"
  CHECKPOINT="$MODEL_DIR/model:tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m_train_h5_0_${STEP}"
fi

if [[ ! -s "$CHECKPOINT" || ! -s "$CHECKPOINT.metadata.json" ]]; then
  echo "Missing checkpoint or metadata sidecar: $CHECKPOINT" >&2
  exit 1
fi

PAIRED_ROOT="/cs/home/rgao48/projects/ambi-runs/tdmpc2-mppi-mc-eval/step_${STEP}"
ACTION_ROOT="/cs/home/rgao48/projects/ambi-runs/tdmpc2-mppi-one-step-mc-eval/step_${STEP}"
PAIRED_OUTPUT="$PAIRED_ROOT/${LABEL}_paired_mc12_env101-112_controller12345.json"
ACTION_OUTPUT="$ACTION_ROOT/${LABEL}_behavior12_blocks20_realized-action_prior-continuation_exact-prefix.json"

if [[ -e "$PAIRED_OUTPUT" || -e "$ACTION_OUTPUT" ]]; then
  echo "Refusing to replace an existing evaluation output." >&2
  echo "Paired output: $PAIRED_OUTPUT" >&2
  echo "Same-state output: $ACTION_OUTPUT" >&2
  exit 1
fi
mkdir -p "$PAIRED_ROOT" "$ACTION_ROOT"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

echo "Job: $SLURM_JOB_ID"
echo "Array task: $TASK_ID"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Source commit: $(git rev-parse HEAD)"
echo "Checkpoint: $CHECKPOINT"
echo "Training horizon label: $LABEL"
echo "Training decisions: $STEP"
echo "Paired output: $PAIRED_OUTPUT"
echo "Same-state output: $ACTION_OUTPUT"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

"$PYTHON" -c 'import torch; print("torch:", torch.__version__); print("visible CUDA devices:", torch.cuda.device_count()); print("cuda:", torch.cuda.get_device_name(0))'

"$PYTHON" evaluate_tdmpc2_mppi_checkpoint.py "$CHECKPOINT" \
  --output "$PAIRED_OUTPUT" \
  --episodes 12 \
  --seed 101 \
  --controller-seed 12345 \
  --bootstrap-samples 20000 \
  --device cuda

"$PYTHON" evaluate_tdmpc2_mppi_action_mc.py "$CHECKPOINT" \
  --behavior-json "$PAIRED_OUTPUT" \
  --output "$ACTION_OUTPUT" \
  --episodes 12 \
  --seed 101 \
  --controller-seed 12345 \
  --block-size 25 \
  --action-draws 1 \
  --bootstrap-samples 20000 \
  --device cuda
