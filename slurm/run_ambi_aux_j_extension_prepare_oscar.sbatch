#!/usr/bin/env bash
#SBATCH --job-name=aux625k-j1216-prepare
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
cd "${AMBI_PROJECT_DIR:?}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_ACTION_MODES_SHA:?}" && -z "$(git status --porcelain)" ]]
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
"${AMBI_DMC_PYTHON:?}" slurm/ambi_aux_hj_sweep.py prepare \
  --root "${AMBI_CAMPAIGN_ROOT:?}" --checkpoint "${AMBI_CHECKPOINT:?}" \
  --inventory "${CHECKPOINT_MANIFEST:?}" --reference "${AMBI_PRIOR_REFERENCE:?}" \
  --registry /oscar/scratch/rgao48/ambi/evaluation-series/runs \
  --matrix "${AMBI_J_EXTENSION_MATRIX:-configs/research/ambi_aux_soft_j1216_625k.json}" \
  --baseline-campaign "${AMBI_J16_C16_CAMPAIGN:?}" \
  --baseline-extension-campaign "${AMBI_J16_C32_CAMPAIGN:?}" \
  --baseline-kind j_extension \
  --group "${AMBI_J16_GROUP:?}" --label "${AMBI_J_EXTENSION_LABEL:-625k soft/soft scaling: J12/J16 uniform replay}"
cp "${AMBI_J16_LAUNCH_ROOT:?}/submission.json" "$AMBI_CAMPAIGN_ROOT/submission.json"
cp "$AMBI_J16_LAUNCH_ROOT/campaign.env" "$AMBI_CAMPAIGN_ROOT/campaign.env"
git rev-parse HEAD HEAD^{tree} > "$AMBI_CAMPAIGN_ROOT/source-sha.txt"
git status --porcelain > "$AMBI_CAMPAIGN_ROOT/source-status.txt"
