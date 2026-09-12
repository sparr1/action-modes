#!/usr/bin/env bash
#SBATCH --job-name=ambi-rounds-publish
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested commit}"
: "${AMBI_ROUND_SCALING_CAMPAIGN:?Supply the immutable campaign manifest}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
PYTHON_BIN="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export WANDB_DIR="${SLURM_TMPDIR:-/tmp}/ambi-rounds-publish-${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR"
exec "$PYTHON_BIN" slurm/ambi_round_scaling_campaign.py publish \
  --campaign "$AMBI_ROUND_SCALING_CAMPAIGN" --cell-index "${SLURM_ARRAY_TASK_ID:?Supply cell index}"
