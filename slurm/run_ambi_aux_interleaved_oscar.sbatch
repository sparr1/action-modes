#!/usr/bin/env bash
#SBATCH --job-name=aux625k-interleaved
#SBATCH --partition=gpu
#SBATCH --account=default
#SBATCH --qos=pri-gpu+
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
cd "${AMBI_PROJECT_DIR:?}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_ACTION_MODES_SHA:?}" && -z "$(git status --porcelain)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cache_root="${AMBI_CAMPAIGN_ROOT:?}/compiler-cache/${SLURM_JOB_ID:?}-${SLURM_ARRAY_TASK_ID:-single}"
mkdir "$cache_root"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/inductor" TRITON_CACHE_DIR="$cache_root/triton"
unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS
case "${1:-}" in
  --controls)
    replica=a; extra=()
    case "${SLURM_ARRAY_TASK_ID:?}" in
      0) replica=a ;;
      1) replica=b ;;
      2) replica=phased; extra=(--phased-control) ;;
      *) exit 2 ;;
    esac
    exec "${AMBI_DMC_PYTHON:?}" slurm/ambi_aux_hj_sweep.py worker --root "$AMBI_CAMPAIGN_ROOT" --smoke --replica "$replica" "${extra[@]}" ;;
  --gate) exec "${AMBI_DMC_PYTHON:?}" slurm/ambi_aux_interleaved.py --root "$AMBI_CAMPAIGN_ROOT" ;;
  '') [[ -f "$AMBI_CAMPAIGN_ROOT/control-completion.json" ]]
    exec "${AMBI_DMC_PYTHON:?}" slurm/ambi_aux_hj_sweep.py worker --root "$AMBI_CAMPAIGN_ROOT" --index "${SLURM_ARRAY_TASK_ID:?}" ;;
  *) exit 2 ;;
esac
