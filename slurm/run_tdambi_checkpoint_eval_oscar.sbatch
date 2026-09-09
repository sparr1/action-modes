#!/usr/bin/env bash
#SBATCH --job-name=tdambi-checkpoint-eval
#SBATCH --partition=gpu
#SBATCH --qos=pri-gpu+
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=2-20
#SBATCH --no-requeue
#SBATCH --output=/tmp/tdambi-eval-%A_%a.out
#SBATCH --error=/tmp/tdambi-eval-%A_%a.err

# Reuse the validated checkpoint, reference, CUDA and publication protocol.
# Supply durable log paths and TDAMBI_PRESET when submitting from a clean checkout.
# Leave concurrency to live QOS/resource limits unless contention requires a cap.
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
exec bash slurm/run_tdambi_checkpoint_eval_hydra.sbatch
