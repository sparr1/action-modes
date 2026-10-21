#!/usr/bin/env bash
#SBATCH --job-name=ambi-replay-timing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --hint=nomultithread
#SBATCH --no-requeue

# Submit with explicit GPU, CPU, memory and walltime resources. The paired
# placement comparison uses the same allocated GPU and one physical CPU.
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${AMBI_PROJECT_DIR:?Set the clean source checkout.}"
: "${AMBI_PYTHON:?Set the locked DMControl interpreter.}"
: "${AMBI_EXPECTED_COMMIT:?Set the full tested commit.}"
: "${AMBI_REPLAY_OUTPUT:?Set a new output directory outside source.}"
[[ "${SLURM_CPUS_PER_TASK:-0}" == 1 && "${SLURM_RESTART_COUNT:-0}" == 0 ]]
[[ "$AMBI_PROJECT_DIR" == /* && "$AMBI_PYTHON" == /* && "$AMBI_REPLAY_OUTPUT" == /* ]]
cd "$AMBI_PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
local_root="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/ambi-replay-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "$local_root/cache" "$local_root/tmp"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
export XDG_CACHE_HOME="$local_root/cache" TMPDIR="$local_root/tmp"
exec srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" \
  slurm/benchmark_ambi_replay.py --output "$AMBI_REPLAY_OUTPUT" "$@"
