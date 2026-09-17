#!/usr/bin/env bash
#SBATCH --job-name=aux625k-merge-publish
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_ACTION_MODES_SHA:?}" && -z "$(git status --porcelain)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" TORCHINDUCTOR_COMPILE_THREADS=1
export WANDB_DIR="${AMBI_CAMPAIGN_ROOT:?}/wandb"
mkdir -p "$WANDB_DIR"
unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS
case "${1:-}" in
  '') mode=merge ;;
  --publish-only) mode=publish ;;
  *) exit 2 ;;
esac
exec "${AMBI_DMC_PYTHON:?}" slurm/ambi_aux_closed_loop.py "$mode" \
  --root "$AMBI_CAMPAIGN_ROOT" --inventory "${CHECKPOINT_MANIFEST:?}" \
  --horizon "${AMBI_HORIZON:-1}" --alpha-mode "${AMBI_ALPHA_MODE:-zero}" \
  --critic-mode "${AMBI_CRITIC_MODE:-soft_q}" \
  --update-budget "${AMBI_UPDATE_BUDGET:-c32_a4}" \
  --run-dir "${EVAL_RUN_DIR:?}" --publish
