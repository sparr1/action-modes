#!/usr/bin/env bash
#SBATCH --job-name=ambi-takeoff-publisher
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --no-requeue

# Submit with --dependency=afterany:<compute-array> to drain complete diagnostic
# bundles even when another checkpoint fails. Optional --array=0-13 publishes
# independent bundles in parallel; successful publication receipts are reused.
set -Eeuo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested commit}"
: "${AMBI_TAKEOFF_ROOT:?Supply the existing absolute campaign output root}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || { echo "Unexpected commit" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || { echo "Dirty checkout" >&2; exit 1; }
PYTHON_BIN="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing locked Python" >&2; exit 1; }
arguments=(publish --output-root "$AMBI_TAKEOFF_ROOT")
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then arguments+=(--task-index "$SLURM_ARRAY_TASK_ID"); fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export WANDB_DIR="${SLURM_TMPDIR:-/tmp}/ambi-takeoff-wandb-${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR"
exec "$PYTHON_BIN" slurm/ambi_takeoff_campaign.py "${arguments[@]}"
