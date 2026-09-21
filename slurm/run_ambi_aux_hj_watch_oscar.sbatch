#!/usr/bin/env bash
#SBATCH --job-name=aux625k-hj-publish
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
cd "${AMBI_PROJECT_DIR:?}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_ACTION_MODES_SHA:?}" && -z "$(git status --porcelain)" ]]
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 WANDB_DIR="${AMBI_CAMPAIGN_ROOT:?}/wandb"
mkdir -p "$WANDB_DIR"
unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS
exec "${AMBI_DMC_PYTHON:?}" slurm/ambi_aux_hj_sweep.py watch --root "$AMBI_CAMPAIGN_ROOT" "$@"
