#!/bin/bash
#SBATCH --job-name=collapse-diagnostics
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
: "${AMBI_PYTHON:?Set AMBI_PYTHON to the existing locked DMControl interpreter}"
export WANDB_MODE=disabled WANDB_DISABLED=true MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
exec srun --cpu-bind=cores "$AMBI_PYTHON" -u scripts/run_collapse_diagnostic_task.py "$@"
