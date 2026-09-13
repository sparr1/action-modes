#!/usr/bin/env bash
#SBATCH --job-name=ambi-entropy-publish
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
# Start alongside the worker array to make its explicit W&B run visible immediately.
# The CLI watches campaign.compute_job_id and coverage, then validates all results
# before publishing final aggregate metrics. No scheduler dependency is required.
set -Eeuo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested commit}"
: "${AMBI_ENTROPY_CAMPAIGN:?Supply the absolute campaign.json path}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || { echo "Unexpected commit" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || { echo "Dirty checkout" >&2; exit 1; }
[[ "$AMBI_ENTROPY_CAMPAIGN" == /* && -f "$AMBI_ENTROPY_CAMPAIGN" ]] || { echo "Campaign must be an existing absolute file" >&2; exit 1; }
PYTHON_BIN="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing locked Python" >&2; exit 1; }
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WANDB_MODE=online WANDB__SERVICE_WAIT=300
export WANDB_DIR="${SLURM_TMPDIR:-/tmp}/ambi-entropy-publish-${SLURM_JOB_ID:?Supply job ID}"
mkdir -p "$WANDB_DIR"
exec "$PYTHON_BIN" evaluate_ambi_entropy.py publish --campaign "$AMBI_ENTROPY_CAMPAIGN"
