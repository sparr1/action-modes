#!/usr/bin/env bash
#SBATCH --job-name=ambi-mean-prefix-merge
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested commit}"
: "${AMBI_MEAN_PREFIX_MANIFEST:?Supply the verified inventory}"
: "${AMBI_MEAN_PREFIX_ROOT:?Supply the campaign output root}"
: "${AMBI_MEAN_PREFIX_ATTEMPT:?Supply an explicit new attempt}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || { echo "Unexpected commit" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || { echo "Dirty checkout" >&2; exit 1; }
PYTHON_BIN="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing locked Python" >&2; exit 1; }
arguments=(merge --inventory "$AMBI_MEAN_PREFIX_MANIFEST" --output-root "$AMBI_MEAN_PREFIX_ROOT"
    --attempt-label "$AMBI_MEAN_PREFIX_ATTEMPT" --checkpoint-index "${SLURM_ARRAY_TASK_ID:?Supply checkpoint index}")
if [[ "${AMBI_MEAN_PREFIX_SMOKE:-0}" == 1 ]]; then
    arguments+=(--smoke --shard-indices 0 1)
else
    arguments+=(--eval-run-map "${EVAL_RUN_MAP:?Supply the full-panel evaluation run map}")
fi
if [[ -n "${AMBI_MEAN_PREFIX_MATRIX:-}" ]]; then arguments+=(--matrix "$AMBI_MEAN_PREFIX_MATRIX"); fi
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 WANDB_MODE=offline
exec "$PYTHON_BIN" slurm/ambi_mean_prefix_campaign.py "${arguments[@]}"
