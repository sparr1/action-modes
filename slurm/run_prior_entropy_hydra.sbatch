#!/usr/bin/env bash
#SBATCH --job-name=prior-entropy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpus
#SBATCH --account=default
#SBATCH --nodelist=gpu2301
#SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --no-requeue
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${AMBI_EXPECTED_COMMIT:?Pin source}"
: "${AMBI_ENTROPY_OUTPUT:?Set output directory outside checkout}"
: "${AMBI_DMCONTROL_PYTHON:?Use the existing locked environment}"
test "$(git rev-parse HEAD)" = "$AMBI_EXPECTED_COMMIT"
test -z "$(git status --porcelain --untracked-files=all)"
test "${SLURM_RESTART_COUNT:-0}" = 0
mode="${1:?gate or train}"
unset CUDA_LAUNCH_BLOCKING WANDB_RESUME
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MUJOCO_GL=egl
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TORCHINDUCTOR_COMPILE_THREADS=1
local_root="$(mktemp -d "${SLURM_TMPDIR:-/ltmp}/prior-entropy-${USER}-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "$local_root/inductor" "$local_root/triton" "$local_root/cache" "$local_root/wandb" "$local_root/data" "$local_root/artifacts"
export TMPDIR="$local_root" TORCHINDUCTOR_CACHE_DIR="$local_root/inductor" TRITON_CACHE_DIR="$local_root/triton"
export XDG_CACHE_HOME="$local_root/cache" WANDB_DIR="$local_root/wandb" WANDB_CACHE_DIR="$local_root/cache"
export WANDB_DATA_DIR="$local_root/data" WANDB_ARTIFACT_DIR="$local_root/artifacts"
export WANDB_DISABLE_CODE=true WANDB__REQUIRE_CORE=true
if [[ "$mode" == gate ]]; then
  export WANDB_MODE=disabled
  exec srun --kill-on-bad-exit=1 "$AMBI_DMCONTROL_PYTHON" slurm/prior_entropy_hydra.py gate --output "$AMBI_ENTROPY_OUTPUT"
elif [[ "$mode" == train ]]; then
  : "${AMBI_ENTROPY_GATE:?Successful measured gate required}"
  : "${AMBI_ENTROPY_INDEX:?Arm index required}"
  : "${WANDB_RUN_ID:?Fresh run identity required}"
  export WANDB_MODE=online
  exec srun --kill-on-bad-exit=1 "$AMBI_DMCONTROL_PYTHON" slurm/prior_entropy_hydra.py train --output "$AMBI_ENTROPY_OUTPUT" --gate "$AMBI_ENTROPY_GATE" --index "$AMBI_ENTROPY_INDEX"
else
  echo "Unknown mode: $mode" >&2
  exit 2
fi
