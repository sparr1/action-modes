#!/usr/bin/env bash
#SBATCH --job-name=ambi_branch_n16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=300:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodelist=gpu[2001]


set -Eeuo pipefail

# Submit this file from the repository root:
#   sbatch slurm/run_ambi_branch_n16.sbatch
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"



# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs


source ~/miniforge3/etc/profile.d/conda.sh
conda activate ambi


# Run the training script
echo "Starting AMBI training..."

python main.py \
  --run configs/ambi/experiments/ambi_branch_n16.json \
  --alg-dir configs/ambi/algs

echo "End time: $(date)"
echo "Job completed!"
