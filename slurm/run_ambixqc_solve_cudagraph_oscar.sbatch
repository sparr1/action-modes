#!/bin/bash
#SBATCH --job-name=ambixqc-solve-graph
#SBATCH --partition=gpu
#SBATCH --account=default
#SBATCH --qos=pri-gpu+
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00

# One GPU/physical CPU for a same-allocation comparison. Previous compiled
# K3 cases took ~2.4 minutes including cold startup. Budget ten minutes for
# the small and full-shape CUDA gates, capture startup and two timed cases.
set -euo pipefail
: "${XQC_TIMING_SHA:?exact tested commit required}"
: "${XQC_TIMING_PYTHON:?locked interpreter required}"
: "${XQC_TIMING_RESULTS:?external result root required}"
test "$(git rev-parse HEAD)" = "$XQC_TIMING_SHA"
test -z "$(git status --porcelain)"
export WANDB_MODE=disabled WANDB_DISABLED=true MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/xqc-solve-${SLURM_JOB_ID}/inductor"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/xqc-solve-${SLURM_JOB_ID}/triton"
export AMBI_RUN_XQC_SOLVE_CUDA_GATE=1
git rev-parse HEAD HEAD^{tree}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
srun --cpu-bind=cores "$XQC_TIMING_PYTHON" -m pytest -q -s \
    tests/test_ambixqc_cudagraph_solve.py \
    > "$XQC_TIMING_RESULTS/gate-${SLURM_JOB_ID}.log" 2>&1
srun --cpu-bind=cores "$XQC_TIMING_PYTHON" \
    tests/benchmarks/ambixqc_j8_training_throughput.py \
    --graph-comparison --solve-interval 3 --warmup 21 --measured 108 \
    --output "$XQC_TIMING_RESULTS/job-${SLURM_JOB_ID}"
