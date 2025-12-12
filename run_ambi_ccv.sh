#!/bin/bash
#SBATCH --job-name=ambi_ant
#SBATCH --output=logs/ambi_ant_%j.out
#SBATCH --error=logs/ambi_ant_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Load modules
module load python/3.11.0s-ixrhc3q
module load cuda/12.1.1

# Create logs directory if it doesn't exist
mkdir -p logs

# Create and activate virtual environment
if [ ! -d "venv_ccv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv_ccv
fi

source venv_ccv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip 2>&1 | tee -a logs/pip_install.log
if [ $? -ne 0 ]; then
    echo "ERROR: pip upgrade failed"
    exit 1
fi

pip install -r requirements.txt 2>&1 | tee -a logs/pip_install.log
if [ $? -ne 0 ]; then
    echo "ERROR: requirements installation failed"
    exit 1
fi

echo "Dependencies installed successfully!"
echo ""

# Run the training script
echo "Starting AMBI training..."
python main.py --run configs/experiments/AntAMBI.json
if [ $? -ne 0 ]; then
    echo "ERROR: Training script failed"
    exit 1
fi

echo "End time: $(date)"
echo "Job completed!"
