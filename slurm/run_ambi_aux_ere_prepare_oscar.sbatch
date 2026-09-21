#!/usr/bin/env bash
#SBATCH --job-name=aux625k-ere-prepare
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
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
  --matrix configs/research/ambi_aux_ere_625k.json \
  --baseline-campaign "${AMBI_ERE_BASELINE_CAMPAIGN:?}" \
  --baseline-extension-campaign "${AMBI_ERE_C32_CAMPAIGN:?}" \
  --group "${AMBI_ERE_GROUP:?}" --label "625k soft/soft: whole-round ERE | H1/H2/H3 J8 C8/C16/C32"
cp "${AMBI_ERE_LAUNCH_ROOT:?}/submission.json" "$AMBI_CAMPAIGN_ROOT/submission.json"
cp "$AMBI_ERE_LAUNCH_ROOT/campaign.env" "$AMBI_CAMPAIGN_ROOT/campaign.env"
git rev-parse HEAD HEAD^{tree} > "$AMBI_CAMPAIGN_ROOT/source-sha.txt"
git status --porcelain > "$AMBI_CAMPAIGN_ROOT/source-status.txt"
