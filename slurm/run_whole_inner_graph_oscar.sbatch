#!/usr/bin/env bash
#SBATCH --job-name=ambi-whole-graph
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --hint=nomultithread
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${AMBI_PROJECT_DIR:?Supply the clean source checkout.}"
: "${AMBI_PYTHON:?Supply the locked existing DMControl interpreter.}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested full commit SHA.}"
: "${AMBI_BENCH_OUTPUT:?Supply a fresh output directory outside the checkout.}"
cd "$AMBI_PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "$AMBI_BENCH_OUTPUT" == /* && "$AMBI_BENCH_OUTPUT" != "$AMBI_PROJECT_DIR/"* ]]
mkdir "$AMBI_BENCH_OUTPUT"
graph_cache="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/ambi-whole-graph-${SLURM_JOB_ID}.XXXXXX")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$graph_cache/inductor" TRITON_CACHE_DIR="$graph_cache/triton"
export XDG_CACHE_HOME="$graph_cache/cache" WANDB_MODE=disabled MUJOCO_GL=egl
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$AMBI_BENCH_OUTPUT/gpu.csv"
for graph_case in tiny-eager tiny-compiled full-compiled; do
    args=()
    case "$graph_case" in
        tiny-eager) args=(--tiny --eager --steps 3 --repeats 1);;
        tiny-compiled) args=(--tiny --steps 3 --repeats 1);;
        full-compiled) args=(--steps 32 --repeats 3);;
    esac
    srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -m tests.benchmark_whole_inner_graph \
        "${args[@]}" --output "$AMBI_BENCH_OUTPUT/$graph_case.json" \
        > "$AMBI_BENCH_OUTPUT/$graph_case.log" 2>&1
done
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
