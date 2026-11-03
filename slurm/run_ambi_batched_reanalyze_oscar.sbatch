#!/usr/bin/env bash
#SBATCH --job-name=batched-reanalyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

# A bounded correctness/timing workload. No checkpoints or W&B publication.
set -Eeuo pipefail
umask 077
ulimit -c 0
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${AMBI_PROJECT_DIR:?Supply a clean source checkout.}"
: "${AMBI_PYTHON:?Supply the locked interpreter.}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested full SHA.}"
: "${AMBI_BENCH_OUTPUT:?Supply a fresh output directory outside source.}"
: "${AMBI_BENCH_MODE:?Choose gate, width, or training.}"
[[ "$AMBI_PROJECT_DIR" == /* && "$AMBI_PYTHON" == /* && "$AMBI_BENCH_OUTPUT" == /* ]]
[[ "$AMBI_BENCH_MODE" == gate || "$AMBI_BENCH_MODE" == width || "$AMBI_BENCH_MODE" == training ]]
cd "$AMBI_PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "$AMBI_BENCH_OUTPUT" != "$AMBI_PROJECT_DIR" && "$AMBI_BENCH_OUTPUT" != "$AMBI_PROJECT_DIR/"* ]]
mkdir "$AMBI_BENCH_OUTPUT"
benchmark_cache="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/batched-reanalyze-${SLURM_JOB_ID}.XXXXXX")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$benchmark_cache/inductor" TRITON_CACHE_DIR="$benchmark_cache/triton"
export XDG_CACHE_HOME="$benchmark_cache/cache" WANDB_MODE=disabled MUJOCO_GL=egl
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$AMBI_BENCH_OUTPUT/gpu.csv"
"$AMBI_PYTHON" - <<'PY' > "$AMBI_BENCH_OUTPUT/runtime.json"
import json, os, platform, subprocess
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.get_device_name(0) == "NVIDIA L40S"
assert torch.__version__.split("+")[0] == "2.3.1"
print(json.dumps(dict(host=platform.node(), torch=torch.__version__, cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(0), job_id=os.environ["SLURM_JOB_ID"],
    cpus=os.environ["SLURM_CPUS_PER_TASK"],
    sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
    tree=subprocess.check_output(["git","rev-parse","HEAD^{tree}"],text=True).strip()),indent=2))
PY
if [[ "$AMBI_BENCH_MODE" == gate || "${AMBI_VALIDATE_FIRST:-0}" == 1 ]]; then
  srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -m pytest -q --tb=short -p no:cacheprovider \
    tests/test_ambi_batched_teacher.py tests/test_ambi_reanalyze_backends.py \
    --basetemp="$benchmark_cache/pytest" --junitxml="$AMBI_BENCH_OUTPUT/gate.xml" \
    > "$AMBI_BENCH_OUTPUT/gate.log" 2>&1
fi
if [[ "$AMBI_BENCH_MODE" == width ]]; then
  : "${AMBI_BATCH_WIDTH:?Supply a positive root batch width.}"
  [[ "$AMBI_BATCH_WIDTH" =~ ^[1-9][0-9]*$ ]]
  read -r -a benchmark_widths <<< "${AMBI_BATCH_WIDTHS:-$AMBI_BATCH_WIDTH}"
  for width in "${benchmark_widths[@]}"; do [[ "$width" =~ ^[1-9][0-9]*$ ]]; done
  # Separate processes isolate compilation and CUDA failures. Record the new
  # solver first; a failing old baseline must not erase completed measurements.
  for width in "${benchmark_widths[@]}"; do
    srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -m tests.benchmark_ambi_reanalyze \
      --device cuda --profile reference --compile --full-refresh --roots 60 \
      --batched-backend batched_cuda_graph --parallel-roots "$width" \
      --warmup 1 --repeats 5 --output "$AMBI_BENCH_OUTPUT/batched-$width.json" \
      > "$AMBI_BENCH_OUTPUT/batched-$width.log" 2>&1
  done
  srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -m tests.benchmark_ambi_reanalyze \
    --device cuda --profile reference --compile --graph-only --full-refresh --roots 60 \
    --parallel-roots 1 --warmup 1 --repeats 5 --output "$AMBI_BENCH_OUTPUT/serial-graph.json" \
    > "$AMBI_BENCH_OUTPUT/serial-graph.log" 2>&1
elif [[ "$AMBI_BENCH_MODE" == training ]]; then
  : "${AMBI_BATCH_WIDTH:?Supply the measured root batch width.}"
  collection_backend="${AMBI_COLLECTION_BACKEND:-cuda_graph}"
  [[ "$collection_backend" == native || "$collection_backend" == cuda_graph ]]
  for interval in 0 10; do
    srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -m tests.benchmark_ambi_distillation_training \
      --collection "$collection_backend" --refresh-interval "$interval" \
      --reanalyze-backend batched_cuda_graph --parallel-roots "$AMBI_BATCH_WIDTH" \
      --warmup 510 --decisions 100 --output "$AMBI_BENCH_OUTPUT/interval-${interval}.json" \
      > "$AMBI_BENCH_OUTPUT/interval-${interval}.log" 2>&1
  done
fi
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
printf 'Benchmark complete: %s\n' "$AMBI_BENCH_OUTPUT"
