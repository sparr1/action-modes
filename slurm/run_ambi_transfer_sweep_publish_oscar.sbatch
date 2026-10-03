#!/usr/bin/env bash
#SBATCH --job-name=transfer575-publish
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --no-requeue

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the separate publisher checkout}"
: "${CAMPAIGN_ROOT:?Existing immutable evaluation campaign}"
: "${PUBLICATION_ROOT:?Separate publisher state directory}"
: "${EXPECTED_PUBLISHER_SHA:?Pin publisher source commit}"
: "${GPU_JOB_IDS:?Space-separated production Slurm job IDs}"
PYTHON_BIN="${PYTHON_BIN:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_PUBLISHER_SHA" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]

runtime_root="${PUBLICATION_ROOT%/*}/publisher-runtime/${SLURM_JOB_ID:?}"
export TMPDIR="$runtime_root/tmp" WANDB_DIR="$runtime_root/wandb"
export WANDB_CACHE_DIR="$runtime_root/wandb-cache" WANDB_DATA_DIR="$runtime_root/wandb-data"
export XDG_CACHE_HOME="$runtime_root/cache" WANDB_MODE=online
mkdir -p "$TMPDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$XDG_CACHE_HOME"
unset WANDB_RUN_ID WANDB_NAME WANDB_GROUP WANDB_RESUME WANDB_SERVICE _WANDB_SERVICE
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
args=(--root "$CAMPAIGN_ROOT" --publication-root "$PUBLICATION_ROOT")
if [[ -n "${REFERENCE_ROOT:-}${REFERENCE_PUBLICATION_ROOT:-}${REFERENCE_AUDIT:-}" ]]; then
  : "${REFERENCE_ROOT:?Hidden comparison requires reference campaign}"
  : "${REFERENCE_PUBLICATION_ROOT:?Hidden comparison requires original publisher state}"
  : "${REFERENCE_AUDIT:?Hidden comparison requires pinned reference audit}"
  args+=(--reference-root "$REFERENCE_ROOT" --reference-publication-root "$REFERENCE_PUBLICATION_ROOT" --reference-audit "$REFERENCE_AUDIT")
fi
read -r -a jobs <<< "$GPU_JOB_IDS"
for job in "${jobs[@]}"; do args+=(--gpu-job-id "$job"); done
exec "$PYTHON_BIN" slurm/ambi_transfer_sweep_publish.py "${args[@]}"
