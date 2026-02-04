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

set -e

# ============================================
# JOB INFO
# ============================================
echo "============================================"
echo "AMBI Training Job"
echo "============================================"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       ${SLURM_NODELIST:-$(hostname)}"
echo "Start time: $(date)"
echo "Working dir: $(pwd)"
echo "============================================"
echo ""

# ============================================
# ENVIRONMENT SETUP
# ============================================
module load python/3.11.0s-ixrhc3q
module load cuda/12.1.1

mkdir -p logs

# Create venv if needed
if [ ! -d "venv_ccv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv_ccv
fi

source venv_ccv/bin/activate
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Install dependencies (only if requirements changed)
REQS_HASH=$(md5sum requirements.txt 2>/dev/null | cut -d' ' -f1 || md5 -q requirements.txt 2>/dev/null || echo "unknown")
LAST_HASH_FILE="venv_ccv/.reqs_hash"

if [ ! -f "$LAST_HASH_FILE" ] || [ "$(cat $LAST_HASH_FILE)" != "$REQS_HASH" ]; then
    echo "Installing/updating dependencies..."
    pip install --upgrade pip 2>&1 | tee -a logs/pip_install.log
    pip install -r requirements.txt 2>&1 | tee -a logs/pip_install.log
    echo "$REQS_HASH" > "$LAST_HASH_FILE"
    echo "Dependencies installed."
else
    echo "Dependencies up to date (skipping install)."
fi

echo ""

# ============================================
# GPU INFO
# ============================================
echo "--- GPU Info ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "No GPU detected"
echo ""

# ============================================
# TRAINING
# ============================================
echo "============================================"
echo "Starting AMBI training..."
echo "Log file: logs/ambi_ant_${SLURM_JOB_ID:-local}.out"
echo "============================================"
echo ""

# PYTHONUNBUFFERED + python -u for real-time output
export PYTHONUNBUFFERED=1
python -u train_ambi_experiment.py
EXIT_CODE=$?

echo ""
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "============================================"

exit $EXIT_CODE
