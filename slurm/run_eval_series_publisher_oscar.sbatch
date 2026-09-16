#!/usr/bin/env bash
#SBATCH --job-name=eval-curve-publisher
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@60

# One CPU publisher per evaluation run. Override wall time for the campaign.
# Submit from the synchronized checkout and supply persistent Slurm log paths.
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${EVAL_RUN_DIR:?Supply the existing authoritative evaluation run directory}"
: "${EVAL_COMPUTE_JOBS:?Supply space-separated compute job IDs}"
PYTHON_BIN="${PYTHON_BIN:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing locked Python" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || { echo "Dirty checkout" >&2; exit 1; }
[[ -z "${EXPECTED_ACTION_MODES_SHA:-}" || "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || { echo "Unexpected commit" >&2; exit 1; }
read -r -a compute_jobs <<< "$EVAL_COMPUTE_JOBS"
publisher_args=()
if [[ -n "${EVAL_PUBLICATION_OWNER:-}" ]]; then
  publisher_args+=(--owner "$EVAL_PUBLICATION_OWNER")
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec "$PYTHON_BIN" eval_series.py publish "$EVAL_RUN_DIR" "${publisher_args[@]}" --watch --poll-seconds 15 --jobs "${compute_jobs[@]}"
