#!/usr/bin/env bash
#SBATCH --job-name=aux625k-closed-loop
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|a6000|a40|a5500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_ACTION_MODES_SHA:?}" && -z "$(git status --porcelain)" ]]
[[ "${SLURM_RESTART_COUNT:-0}" == 0 ]]
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" TORCHINDUCTOR_COMPILE_THREADS=1
unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS
python_bin="${AMBI_DMC_PYTHON:?}"
common=(--root "${AMBI_CAMPAIGN_ROOT:?}" --inventory "${CHECKPOINT_MANIFEST:?}")
case "${1:-}" in
  --smoke) seed=101; extra=(--smoke) ;;
  '')
    [[ "${SLURM_ARRAY_TASK_ID:?}" =~ ^[1-5]$ ]]
    seed=$((100 + SLURM_ARRAY_TASK_ID)); extra=()
    ;;
  *) exit 2 ;;
esac
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
exec "$python_bin" slurm/ambi_aux_closed_loop.py worker "${common[@]}" \
  --checkpoint "${AMBI_CHECKPOINT:?}" --reference "${AMBI_PRIOR_REFERENCE:?}" --seed "$seed" "${extra[@]}"
