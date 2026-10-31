#!/usr/bin/env bash
#SBATCH --job-name=orig253-aux
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2301
#SBATCH --constraint=l40
#SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/cs/home/rgao48/projects/ambi-runs/slurm/%x-%j.out
#SBATCH --error=/cs/home/rgao48/projects/ambi-runs/slurm/%x-%j.err

set -Eeuo pipefail
umask 077
MODE="${1:?Specify benchmark or train}"
[[ "$MODE" == benchmark || "$MODE" == train ]] || exit 2
cd "${SLURM_SUBMIT_DIR:?Submit from the source checkout}"
PYTHON="${AMBI_DMC_PYTHON:-/cs/home/rgao48/projects/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON" ]] || { echo "Missing locked runtime" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || { echo "Dirty source checkout" >&2; exit 1; }
ROOT="${AMBI_AUX_REPRO_ROOT:-/cs/home/rgao48/projects/ambi-runs/original-prior-aux-reproduction-20261030}"
STAGING="${SLURM_TMPDIR:-/tmp}/ambi-original-prior-aux-${SLURM_JOB_ID}"
mkdir -p "$STAGING"/{wandb,cache,data,artifacts,inductor,triton}
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export WANDB_DIR="$STAGING/wandb" WANDB_CACHE_DIR="$STAGING/cache" WANDB_DATA_DIR="$STAGING/data" WANDB_ARTIFACT_DIR="$STAGING/artifacts"
export TORCHINDUCTOR_CACHE_DIR="$STAGING/inductor" TRITON_CACHE_DIR="$STAGING/triton"
export WANDB_MODE=online WANDB__REQUIRE_CORE=true WANDB_DISABLE_CODE=true
unset WANDB_RESUME
if [[ "$MODE" == benchmark ]]; then
  srun --unbuffered --kill-on-bad-exit=1 "$PYTHON" slurm/reproduce_original_prior_aux.py benchmark --output "$ROOT/benchmark-${SLURM_JOB_ID}"
else
  : "${AMBI_AUX_REPRO_GATE:?Set completed benchmark.json path}"
  : "${WANDB_RUN_ID:?Set a verified-unused run ID}"
  srun --unbuffered --kill-on-bad-exit=1 "$PYTHON" slurm/reproduce_original_prior_aux.py train --output "$ROOT/seed55/job-${SLURM_JOB_ID}" --gate "$AMBI_AUX_REPRO_GATE"
fi
