#!/usr/bin/env bash
#SBATCH --job-name=800k-j10-publish
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from synchronized repository}"
: "${CAMPAIGN_ROOT:?}" "${PUBLICATION_ROOT:?}" "${EXPECTED_ACTION_MODES_SHA:?}"
: "${REFERENCE_J4_ROOT:?}" "${REFERENCE_J6_ROOT:?}" "${REFERENCE_J8_ROOT:?}"
PYTHON_BIN="${PYTHON_BIN:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
runtime_root="${CAMPAIGN_ROOT%/*}/runtime/publisher-${SLURM_JOB_ID:?}"
export TMPDIR="$runtime_root/tmp" XDG_CACHE_HOME="$runtime_root/cache"
export WANDB_DIR="$runtime_root/wandb" WANDB_CACHE_DIR="$runtime_root/wandb-cache"
export WANDB_DATA_DIR="$runtime_root/wandb-data"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
unset WANDB_RUN_ID WANDB_NAME WANDB_GROUP WANDB_SERVICE _WANDB_SERVICE WANDB_RESUME
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
args=(--root "$CAMPAIGN_ROOT" --publication-root "$PUBLICATION_ROOT"
  --reference-publication-root "$REFERENCE_J4_ROOT"
  --reference-publication-root "$REFERENCE_J6_ROOT"
  --reference-publication-root "$REFERENCE_J8_ROOT")
if [[ "${VALIDATE_ONLY:-0}" == 1 ]]; then
  export WANDB_MODE=disabled
  args+=(--validate-only)
else
  export WANDB_MODE=online
  args+=(--gpu-job-id "${GPU_JOB_ID:?}")
fi
exec "$PYTHON_BIN" slurm/publish_ambi_transfer_j10.py "${args[@]}"
