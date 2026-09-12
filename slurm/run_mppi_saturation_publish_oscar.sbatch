#!/usr/bin/env bash
#SBATCH --job-name=mppi-saturation-publish
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
: "${AMBI_MPPI_BUNDLE:?Supply the completed production diagnostic directory}"
: "${AMBI_MPPI_ATTEMPT:?Supply an explicit attempt label}"
: "${AMBI_MPPI_RUN_ID:?Supply an explicit unique W&B run ID}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || { echo "Unexpected commit" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || { echo "Dirty checkout" >&2; exit 1; }
PYTHON_BIN="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing locked Python" >&2; exit 1; }
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export WANDB_MODE=online WANDB__SERVICE_WAIT=300
export WANDB_DIR="${SLURM_TMPDIR:-/tmp}/mppi-saturation-publish-${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR"
exec "$PYTHON_BIN" publish_mppi_saturation.py --bundle "$AMBI_MPPI_BUNDLE" \
    --attempt-label "$AMBI_MPPI_ATTEMPT" --run-id "$AMBI_MPPI_RUN_ID" --mode online
