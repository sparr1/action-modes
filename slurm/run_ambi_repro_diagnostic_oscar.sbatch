#!/usr/bin/env bash
#SBATCH --job-name=ambi-repro-diagnostic
#SBATCH --partition=gpu
#SBATCH --account=default
#SBATCH --qos=pri-gpu+
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
set -Eeuo pipefail
cd "${AMBI_PROJECT_DIR:?}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_ACTION_MODES_SHA:?}" && -z "$(git status --porcelain)" ]]
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=1
mkdir -p "${AMBI_DIAGNOSTIC_ROOT:?}"
for profile in production compile-a compile-b deterministic eager; do
  extra=(--mode compile)
  [[ "$profile" != deterministic ]] || extra+=(--deterministic)
  [[ "$profile" != eager ]] || extra=(--mode eager)
  [[ "$profile" != production ]] || extra+=(--no-record)
  "${AMBI_DMC_PYTHON:?}" diagnose_ambi_reproducibility.py \
    --bundle "${AMBI_DIAGNOSTIC_BUNDLE:?}" \
    --output "$AMBI_DIAGNOSTIC_ROOT/$profile" "${extra[@]}"
done
