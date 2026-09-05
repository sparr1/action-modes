#!/usr/bin/env bash
#SBATCH --job-name=ambi-inner-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=1-5
#SBATCH --no-requeue
#SBATCH --output=/tmp/ambi-inner-eval-%A_%a.out
#SBATCH --error=/tmp/ambi-inner-eval-%A_%a.err

# Oscar resources and runtime only. Keep the evaluation protocol, configuration
# selection, guards, reference pairing, and outputs in the shared launcher body.
# Submit from the clean checkout with persistent --output/--error paths.
set -Eeuo pipefail
export AMBI_DMC_PYTHON="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
exec bash "${SLURM_SUBMIT_DIR:?Submit from the repository root.}/slurm/run_ambi_inner_benchmark_hydra.sbatch" "$@"
