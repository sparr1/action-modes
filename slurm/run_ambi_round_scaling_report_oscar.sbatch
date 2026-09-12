#!/usr/bin/env bash
#SBATCH --job-name=ambi-rounds-comparison
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested commit}"
: "${AMBI_ROUND_SCALING_CAMPAIGN:?Supply the immutable campaign manifest}"
: "${AMBI_ROUND_SCALING_COMPARISON_ID:?Supply the reserved comparison run ID}"
: "${AMBI_ROUND_SCALING_COMPUTE_JOBS:?Supply compute job IDs}"
: "${AMBI_ROUND_SCALING_MERGE_JOBS:?Supply merge job IDs}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
PYTHON_BIN="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
read -r -a compute_jobs <<< "$AMBI_ROUND_SCALING_COMPUTE_JOBS"
read -r -a merge_jobs <<< "$AMBI_ROUND_SCALING_MERGE_JOBS"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export WANDB_DIR="${SLURM_TMPDIR:-/tmp}/ambi-rounds-comparison-${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR"
exec "$PYTHON_BIN" slurm/ambi_round_scaling_report.py --campaign "$AMBI_ROUND_SCALING_CAMPAIGN" \
  --wandb-run-id "$AMBI_ROUND_SCALING_COMPARISON_ID" --owner oscar-rgao48 --mode online --watch \
  --compute-jobs "${compute_jobs[@]}" --merge-jobs "${merge_jobs[@]}"
