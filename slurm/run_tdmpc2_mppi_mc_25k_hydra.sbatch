#!/usr/bin/env bash
#SBATCH --job-name=tdmpc2-mppi-mc25k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2301
#SBATCH --constraint=l40
#SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-1%2
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

case "${SLURM_ARRAY_TASK_ID:?missing array task id}" in
  0)
    LABEL="train-h3"
    CHECKPOINT="/cs/home/rgao48/projects/ambi-runs/tdmpc2-checkpoint-bank-1p5m/seed_1/job_34489/tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m_2026-09-04_14-18-37/models/model:tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m_0_25000"
    ;;
  1)
    LABEL="train-h5"
    CHECKPOINT="/cs/home/rgao48/projects/ambi-runs/tdmpc2-checkpoint-bank-1p5m-train-h5/seed_1/job_34490/tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m_train_h5_2026-09-04_14-18-37/models/model:tdmpc2_humanoid_walk_state_checkpoint_bank_1p5m_train_h5_0_25000"
    ;;
  *)
    echo "Unsupported array task: $SLURM_ARRAY_TASK_ID" >&2
    exit 1
    ;;
esac

if [[ ! -s "$CHECKPOINT" || ! -s "$CHECKPOINT.metadata.json" ]]; then
  echo "Missing checkpoint or metadata sidecar: $CHECKPOINT" >&2
  exit 1
fi

RESULT_ROOT="/cs/home/rgao48/projects/ambi-runs/tdmpc2-mppi-mc-eval/step_25000"
OUTPUT="$RESULT_ROOT/${LABEL}_paired_mc12_env101-112_controller12345.json"
mkdir -p "$RESULT_ROOT"

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

echo "Job: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Source commit: $(git rev-parse HEAD)"
echo "Checkpoint: $CHECKPOINT"
echo "Protocol: policy-prior mean vs native eval-mode MPPI"
echo "Paired full episodes: 12"
echo "Environment seeds: 101-112"
echo "Controller seed base: 12345"
echo "Output: $OUTPUT"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

"$PYTHON" -c 'import torch; print("torch:", torch.__version__); print("visible CUDA devices:", torch.cuda.device_count()); print("cuda:", torch.cuda.get_device_name(0))'

"$PYTHON" evaluate_tdmpc2_mppi_checkpoint.py "$CHECKPOINT" \
  --output "$OUTPUT" \
  --episodes 12 \
  --seed 101 \
  --controller-seed 12345 \
  --bootstrap-samples 20000 \
  --device cuda
