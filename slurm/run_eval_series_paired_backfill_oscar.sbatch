#!/usr/bin/env bash
#SBATCH --job-name=eval-paired-backfill
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
set -Eeuo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
: "${EXPECTED_ACTION_MODES_SHA:?}"
: "${EVAL_RUN_DIR:?}"
: "${EVAL_PUBLICATION_OWNER:?}"
: "${AMBI_PRIOR_OUTPUT_ROOT:?}"
: "${CHECKPOINT_MANIFEST:?}"
: "${BACKFILL_RECEIPT:?}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" && -z "$(git status --porcelain)" ]]
PYTHON_BIN="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS
args=("$EVAL_RUN_DIR" --prior-bundles "$AMBI_PRIOR_OUTPUT_ROOT"/step_*/bundle
      --checkpoint-inventory "$CHECKPOINT_MANIFEST" --owner "$EVAL_PUBLICATION_OWNER")
"$PYTHON_BIN" backfill_eval_series.py "${args[@]}" --receipt "${BACKFILL_RECEIPT%.json}-preflight.json"
exec "$PYTHON_BIN" backfill_eval_series.py "${args[@]}" --publish --receipt "$BACKFILL_RECEIPT"
