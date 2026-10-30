#!/usr/bin/env bash
#SBATCH --job-name=original-prior
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
[[ -x "$PYTHON" ]] || { echo "Missing locked Python runtime" >&2; exit 1; }
[[ "$SLURM_JOB_NODELIST" == gpu2301 ]] || { echo "Unexpected node" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || { echo "Dirty source checkout" >&2; exit 1; }
ROOT="${AMBI_REPRO_ROOT:-/cs/home/rgao48/projects/ambi-runs/original-prior-reproduction-20261030}"
STAGING="${SLURM_TMPDIR:-/tmp}/ambi-original-prior-${SLURM_JOB_ID}"
mkdir -p "$STAGING"/{wandb,cache,data,artifacts,inductor,triton}
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export WANDB_DIR="$STAGING/wandb" WANDB_CACHE_DIR="$STAGING/cache" WANDB_DATA_DIR="$STAGING/data" WANDB_ARTIFACT_DIR="$STAGING/artifacts"
export TORCHINDUCTOR_CACHE_DIR="$STAGING/inductor" TRITON_CACHE_DIR="$STAGING/triton" WANDB_DISABLE_CODE=true
if [[ "$MODE" == benchmark ]]; then
  export WANDB_MODE=disabled
  srun --unbuffered --kill-on-bad-exit=1 "$PYTHON" slurm/reproduce_original_prior.py benchmark --output "$ROOT/benchmark-${SLURM_JOB_ID}"
else
  : "${AMBI_REPRO_GATE:?Set completed benchmark.json path}"
  export WANDB_MODE=online
  srun --unbuffered --kill-on-bad-exit=1 "$PYTHON" slurm/reproduce_original_prior.py train --output "$ROOT/seed55/job-${SLURM_JOB_ID}" --gate "$AMBI_REPRO_GATE"
fi
