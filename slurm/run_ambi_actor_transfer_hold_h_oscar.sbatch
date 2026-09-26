#!/usr/bin/env bash
#SBATCH --job-name=actor-hold-h575
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --no-requeue

# Same validated CPU/GPU ownership and scratch-runtime setup as uniform-J.
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the synchronized repository root}"
export EVAL_MATRIX_PATH="${EVAL_MATRIX_PATH:-$PWD/configs/research/ambi_actor_transfer_hold_h_575k.json}"
export EVAL_CAMPAIGN_GROUP="${EVAL_CAMPAIGN_GROUP:-actor-transfer-hold-h-575k-20260926}"
export EVAL_CAMPAIGN_LABEL="${EVAL_CAMPAIGN_LABEL:-575k actor transfer hold-H | H1/2/3 J1/2/4/6/8/10 | solve every H decisions}"
exec bash slurm/run_ambi_actor_transfer_oscar.sbatch
