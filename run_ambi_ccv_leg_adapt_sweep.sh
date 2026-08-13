#!/bin/bash
#SBATCH --job-name=leg_adapt_sweep
#SBATCH --output=logs/leg_adapt_sweep_%A_%a.out
#SBATCH --error=logs/leg_adapt_sweep_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --constraint=geforce3090
#SBATCH --array=0-3

# Each array index runs a separate LegAdaptAnt-v0 experiment config, so all
# four algorithms train concurrently as independent SLURM jobs, each logging
# its own wandb run:
#   0 -> AntLegAdaptSAC
#   1 -> AntLegAdaptTDMPC2
#   2 -> AntLegAdaptAMBITDMPC2
#   3 -> AntLegAdaptAMBITDMPC2LoRA
CONFIGS=(
  "AntLegAdaptSAC"
  "AntLegAdaptTDMPC2"
  "AntLegAdaptAMBITDMPC2"
  "AntLegAdaptAMBITDMPC2LoRA"
)
CONFIG_NAME="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

set -Eeuo pipefail

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Array task ID: $SLURM_ARRAY_TASK_ID -> $CONFIG_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate uv venv
source venv_ccv/bin/activate

# Run the training script
echo "Starting AMBI leg-adaptation sweep: $CONFIG_NAME..."
python main.py \
  --run "configs/experiments/${CONFIG_NAME}.json" \
  --alg-dir configs/algs

echo "End time: $(date)"
echo "Job completed!"
