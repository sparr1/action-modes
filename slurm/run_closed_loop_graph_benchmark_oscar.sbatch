#!/usr/bin/env bash
#SBATCH --job-name=closed-loop-graph-timing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --hint=nomultithread

set -Eeuo pipefail
: "${AMBI_PROJECT_DIR:?}"
: "${AMBI_EXPECTED_COMMIT:?}"
: "${AMBI_PYTHON:?}"
: "${AMBI_CHECKPOINT:?}"
: "${AMBI_BENCH_OUTPUT:?}"
[[ "$SLURM_CPUS_PER_TASK" == 1 ]]
cd "$AMBI_PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$AMBI_EXPECTED_COMMIT" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
mkdir -p "$AMBI_BENCH_OUTPUT"
local_root="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/closed-loop-graph-${SLURM_JOB_ID}.XXXXXX")"
mkdir -p "$local_root/inductor" "$local_root/triton" "$local_root/cache" "$local_root/tmp"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$local_root/inductor" TRITON_CACHE_DIR="$local_root/triton"
export XDG_CACHE_HOME="$local_root/cache" TMPDIR="$local_root/tmp"
unset CUDA_LAUNCH_BLOCKING WANDB_RUN_ID WANDB_RESUME
nvidia-smi --query-gpu=name,uuid --format=csv > "$AMBI_BENCH_OUTPUT/gpu.csv"
exec srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -m tests.benchmark_closed_loop_graph \
  --checkpoint "$AMBI_CHECKPOINT" --output "$AMBI_BENCH_OUTPUT/result.json"
