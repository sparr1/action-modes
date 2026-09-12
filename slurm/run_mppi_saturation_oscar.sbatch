#!/usr/bin/env bash
#SBATCH --job-name=mppi-saturation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested commit}"
: "${AMBI_MPPI_CHECKPOINT:?Supply the frozen 200k checkpoint}"
: "${AMBI_MPPI_ROOT_BANK:?Supply the complete verified prior-root bank}"
: "${AMBI_MPPI_OUTPUT_ROOT:?Supply an absolute fresh output root}"
[[ "$AMBI_MPPI_OUTPUT_ROOT" == /* ]] || exit 1
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || { echo "Unexpected commit" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || { echo "Dirty checkout" >&2; exit 1; }
PYTHON_BIN="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing locked Python" >&2; exit 1; }
[[ "${AMBI_MPPI_SMOKE:-0}" == 0 || "${AMBI_MPPI_SMOKE:-0}" == 1 ]] || exit 1
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/mppi-saturation-${SLURM_JOB_ID}"
export WANDB_MODE=offline
"$PYTHON_BIN" -c "import torch; assert torch.cuda.is_available(); print({'device':torch.cuda.get_device_name(0),'torch':torch.__version__})"
arguments=(--checkpoint "$AMBI_MPPI_CHECKPOINT" --root-bank "$AMBI_MPPI_ROOT_BANK" --device cuda)
if [[ "${AMBI_MPPI_SMOKE:-0}" == 1 ]]; then
    "$PYTHON_BIN" -m pytest -q tests/test_mppi_saturation.py tests/test_mppi_helper.py tests/test_mppi_saturation_publication.py
    arguments+=(--smoke --output-dir "$AMBI_MPPI_OUTPUT_ROOT/smoke")
else
    arguments+=(--output-dir "$AMBI_MPPI_OUTPUT_ROOT/production")
fi
exec "$PYTHON_BIN" evaluate_mppi_saturation.py "${arguments[@]}"
