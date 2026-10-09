#!/usr/bin/env bash
#SBATCH --job-name=transfer-curves-publish
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from synchronized repository}"
: "${CAMPAIGN_ROOT:?}" "${EXPECTED_ACTION_MODES_SHA:?}" "${CAMPAIGN_MODE:?}"
PYTHON_BIN="${PYTHON_BIN:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
runtime_root="${CAMPAIGN_ROOT%/*}/runtime/cpu-${SLURM_JOB_ID:?}"
export TMPDIR="$runtime_root/tmp" XDG_CACHE_HOME="$runtime_root/cache"
export WANDB_DIR="$runtime_root/wandb" WANDB_CACHE_DIR="$runtime_root/wandb-cache"
export WANDB_DATA_DIR="$runtime_root/wandb-data"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
unset WANDB_RUN_ID WANDB_NAME WANDB_GROUP WANDB_SERVICE _WANDB_SERVICE WANDB_RESUME
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
case "$CAMPAIGN_MODE" in
  prepare)
    export WANDB_MODE=disabled
    exec "$PYTHON_BIN" slurm/ambi_transfer_curve_campaign.py prepare \
      --root "$CAMPAIGN_ROOT" --inventory "${CHECKPOINT_INVENTORY:?}" \
      --config "${CAMPAIGN_CONFIG:-configs/research/ambi_transfer_checkpoint_curves.json}" \
      --expected-source-sha "$EXPECTED_ACTION_MODES_SHA"
    ;;
  publication_prepare)
    export WANDB_MODE=disabled
    args=(prepare --root "$CAMPAIGN_ROOT" --publication-root "${PUBLICATION_ROOT:?}"
      --attempt-label "${ATTEMPT_LABEL:?}" --project ambi-inner-bench)
    if [[ -n "${COMPARISON_HOST_PUBLICATION_ROOT:-}" ]]; then
      args+=(--comparison-host-publication-root "$COMPARISON_HOST_PUBLICATION_ROOT")
    fi
    exec "$PYTHON_BIN" slurm/ambi_transfer_checkpoint_publish.py "${args[@]}"
    ;;
  publish)
    export WANDB_MODE=online
    args=(watch --root "$CAMPAIGN_ROOT" --publication-root "${PUBLICATION_ROOT:?}")
    if [[ "${PUBLISH_ONCE:-0}" == 1 ]]; then
      args+=(--once)
    else
      args+=(--gpu-job-id "${GPU_JOB_ID:?}")
    fi
    exec "$PYTHON_BIN" slurm/ambi_transfer_checkpoint_publish.py "${args[@]}"
    ;;
  *) echo "Unknown CAMPAIGN_MODE: $CAMPAIGN_MODE" >&2; exit 2 ;;
esac
