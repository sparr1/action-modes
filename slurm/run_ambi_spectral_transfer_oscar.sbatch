#!/usr/bin/env bash
#SBATCH --job-name=spectral-transfer575
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the synchronized repository}"
: "${CAMPAIGN_ROOT:?}"
: "${EXPECTED_ACTION_MODES_SHA:?}"
: "${EVAL_MODE:?}"
PYTHON_BIN="${PYTHON_BIN:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain --untracked-files=normal)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
runtime_root="${CAMPAIGN_ROOT%/*}/runtime/${SLURM_JOB_ID:?}-${SLURM_ARRAY_TASK_ID:-single}"
export TMPDIR="$runtime_root/tmp" XDG_CACHE_HOME="$runtime_root/cache"
export TORCHINDUCTOR_CACHE_DIR="$runtime_root/inductor" TRITON_CACHE_DIR="$runtime_root/triton"
export WANDB_DIR="$runtime_root/wandb" WANDB_CACHE_DIR="$runtime_root/wandb-cache"
export WANDB_DATA_DIR="$runtime_root/wandb-data" WANDB_MODE=disabled
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
  "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"
unset WANDB_RUN_ID WANDB_NAME WANDB_GROUP WANDB_SERVICE _WANDB_SERVICE WANDB_RESUME
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
case "$EVAL_MODE" in
  prepare)
    # Legacy all-parameter references must not leak into matrix-only campaigns.
    [[ -z "${REFERENCE_CAMPAIGN_ROOT:-}${BERNOULLI_REFERENCE_CAMPAIGN_ROOT:-}${BERNOULLI_J8_REFERENCE_CAMPAIGN_ROOT:-}${BERNOULLI_J10_REFERENCE_CAMPAIGN_ROOT:-}" ]]
    args=(prepare --root "$CAMPAIGN_ROOT" --checkpoint "${CHECKPOINT_PATH:?}"
      --matrix "${CAMPAIGN_MATRIX:-configs/research/ambi_spectral_transfer_575k.json}"
      --expected-source "$EXPECTED_ACTION_MODES_SHA")
    if [[ -n "${SPECTRAL_REFERENCE_CAMPAIGN_ROOT:-}" ]]; then
      args+=(--spectral-reference-root "$SPECTRAL_REFERENCE_CAMPAIGN_ROOT")
    fi
    exec "$PYTHON_BIN" slurm/ambi_spectral_transfer_campaign.py "${args[@]}"
    ;;
  worker)
    # Fail before evaluation if the allocation cannot actually execute CUDA.
    # A visible GPU in nvidia-smi alone does not prevent CPU fallback.
    "$PYTHON_BIN" - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable; refusing CPU fallback"
assert torch.cuda.device_count() == 1, "Expected one allocated GPU"
assert "L40S" in torch.cuda.get_device_name(0), "Expected L40S timing hardware"
probe = torch.ones(1024, device="cuda").square().sum()
torch.cuda.synchronize()
assert probe.item() == 1024.0, "CUDA arithmetic preflight failed"
print("CUDA health preflight passed: " + torch.cuda.get_device_name(0), flush=True)
PY
    args=(worker --root "$CAMPAIGN_ROOT" --index "${SLURM_ARRAY_TASK_ID:?}")
    if [[ "${EVAL_SMOKE:-0}" == 1 ]]; then args+=(--smoke); fi
    exec "$PYTHON_BIN" slurm/ambi_transfer_discovery_campaign.py "${args[@]}"
    ;;
  publish)
    export WANDB_MODE=online OMP_NUM_THREADS=1
    exec "$PYTHON_BIN" slurm/ambi_transfer_discovery_publish.py --root "$CAMPAIGN_ROOT" \
      --publication-root "${PUBLICATION_ROOT:?}" --gpu-job-id "${GPU_JOB_ID:?}"
    ;;
  *) echo "Unknown EVAL_MODE: $EVAL_MODE" >&2; exit 2 ;;
esac
