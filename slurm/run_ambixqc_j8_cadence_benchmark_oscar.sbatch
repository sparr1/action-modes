#!/bin/bash
#SBATCH --job-name=ambixqc-j8-cadence
#SBATCH --partition=gpu
#SBATCH --account=default
#SBATCH --qos=pri-gpu+
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=8G
#SBATCH --time=00:05:00

# Submit from a clean exact-SHA checkout. Set XQC_TIMING_SHA,
# XQC_TIMING_PYTHON and XQC_TIMING_RESULTS, and supply external stdout/err paths.
# One physical CPU suffices from the measured J8 CPU scaling comparison.
# The five-minute bound includes a cold compiler plus the measured <3-minute
# K1 training window; K3 cases finish sooner. Each task owns its output/cache.
set -euo pipefail
: "${XQC_TIMING_SHA:?exact tested commit required}"
: "${XQC_TIMING_PYTHON:?locked DMControl interpreter required}"
: "${XQC_TIMING_RESULTS:?external result root required}"
test "$(git rev-parse HEAD)" = "$XQC_TIMING_SHA"
test -z "$(git status --porcelain)"
test -x "$XQC_TIMING_PYTHON"
case "$SLURM_ARRAY_TASK_ID" in
    0) interval=1; compilation=--compile ;;
    1) interval=3; compilation=--compile ;;
    2) interval=3; compilation=--no-compile ;;
    *) exit 2 ;;
esac
export PYTHONDONTWRITEBYTECODE=1
git rev-parse HEAD HEAD^{tree}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
exec srun --cpu-bind=cores "$XQC_TIMING_PYTHON" \
    tests/benchmarks/ambixqc_j8_training_throughput.py \
    --single-case --mode j8 "$compilation" --solve-interval "$interval" \
    --warmup 21 --measured 108 \
    --output "$XQC_TIMING_RESULTS/job-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
