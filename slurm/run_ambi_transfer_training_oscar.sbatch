#!/usr/bin/env bash
#SBATCH --job-name=ambi-transfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --no-requeue

# Share the tested runtime body; only the directives above apply on Oscar.
# Submit one smoke job, then --array=0-5 --dependency=afterok:<smoke>.
# AMBI_TRANSFER_MANIFEST is forwarded to the shared launcher; omission keeps J6.
set -Eeuo pipefail
: "${AMBI_PROJECT_DIR:?Set the clean source checkout.}"
exec bash "$AMBI_PROJECT_DIR/slurm/run_ambi_transfer_training_hydra.sbatch"
