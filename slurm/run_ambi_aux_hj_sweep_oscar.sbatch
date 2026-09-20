#!/usr/bin/env bash
#SBATCH --job-name=aux625k-hj
#SBATCH --partition=gpu
#SBATCH --account=default
#SBATCH --qos=pri-gpu+
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a6000|a40|a5500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
cd "${AMBI_PROJECT_DIR:?}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_ACTION_MODES_SHA:?}" && -z "$(git status --porcelain)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" TORCHINDUCTOR_COMPILE_THREADS=1
unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS
case "${1:-}" in
  --smoke) extra=(--smoke) ;;
  --smoke-index) extra=(--smoke --index "${SLURM_ARRAY_TASK_ID:?}") ;;
  '') extra=(--index "${SLURM_ARRAY_TASK_ID:?}") ;;
  *) exit 2 ;;
esac
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
exec "${AMBI_DMC_PYTHON:?}" slurm/ambi_aux_hj_sweep.py worker --root "${AMBI_CAMPAIGN_ROOT:?}" "${extra[@]}"
