#!/bin/bash
#SBATCH --job-name=ambi_ant
#SBATCH --output=logs/ambi_ant_%j.out
#SBATCH --error=logs/ambi_ant_%j.err
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --constraint=geforce3090 

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

module load miniforge3/25.3.0-3
source ${MAMBA_ROOT_PREFIX}/etc/profile.d/conda.sh
conda activate ambi


# Run the training script
echo "Starting AMBI training..."
python main.py --run configs/experiments/AntAMBI.json
if [ $? -ne 0 ]; then
    echo "ERROR: Training script failed"
fi

echo "End time: $(date)"
echo "Job completed!"