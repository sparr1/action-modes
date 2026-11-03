#!/usr/bin/env bash
#SBATCH --job-name=distill-timing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

# Independent modes can run concurrently on separate L40S GPUs. Paired training
# arms run sequentially on the same GPU. No checkpoints or W&B publication.
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${AMBI_PROJECT_DIR:?Supply a clean absolute source checkout.}"
: "${AMBI_PYTHON:?Supply the locked DMControl interpreter.}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested full SHA.}"
: "${AMBI_BENCH_OUTPUT:?Supply a fresh output directory outside source.}"
: "${AMBI_BENCH_MODE:?Choose native, cuda_graph, parallel, or parallel_wide.}"
[[ "$AMBI_PROJECT_DIR" == /* && "$AMBI_PYTHON" == /* && "$AMBI_BENCH_OUTPUT" == /* ]]
[[ "$AMBI_BENCH_MODE" == native || "$AMBI_BENCH_MODE" == cuda_graph || "$AMBI_BENCH_MODE" == parallel || "$AMBI_BENCH_MODE" == parallel_wide ]]
cd "$AMBI_PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "$AMBI_BENCH_OUTPUT" != "$AMBI_PROJECT_DIR" && "$AMBI_BENCH_OUTPUT" != "$AMBI_PROJECT_DIR/"* ]]
mkdir "$AMBI_BENCH_OUTPUT"
benchmark_cache="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/distill-timing-${SLURM_JOB_ID}.XXXXXX")"
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
if [[ "$AMBI_BENCH_MODE" == parallel || "$AMBI_BENCH_MODE" == parallel_wide ]]; then
  if [[ "$AMBI_BENCH_MODE" == parallel ]]; then
    srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -m tests.benchmark_ambi_reanalyze \
      --device cuda --profile reference --compile --graph-only --roots 1 --parallel-roots 1 \
      --warmup 2 --repeats 20 --output "$AMBI_BENCH_OUTPUT/one-root.json" \
      > "$AMBI_BENCH_OUTPUT/one-root.log" 2>&1
    widths=(1 4 16 60)
  else
    widths=(16 60)
  fi
  srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -m tests.benchmark_ambi_reanalyze \
    --device cuda --profile reference --compile --graph-only --full-refresh --roots 60 \
    --parallel-roots "${widths[@]}" --compile-cache-limit 128 --reuse-graph-pool --warmup 1 --repeats 3 \
    --output "$AMBI_BENCH_OUTPUT/parallel.json" > "$AMBI_BENCH_OUTPUT/parallel.log" 2>&1
else
  if [[ "$AMBI_BENCH_MODE" == cuda_graph ]]; then
    srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -m pytest -q -p no:cacheprovider \
      tests/test_ambi_distillation_training_benchmark.py \
      --basetemp="$benchmark_cache/pytest" --junitxml="$AMBI_BENCH_OUTPUT/collection-gate.xml" \
      > "$AMBI_BENCH_OUTPUT/collection-gate.log" 2>&1
  fi
  for interval in 0 10; do
    srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -m tests.benchmark_ambi_distillation_training \
      --collection "$AMBI_BENCH_MODE" --refresh-interval "$interval" \
      --warmup 510 --decisions 100 --output "$AMBI_BENCH_OUTPUT/interval-${interval}.json" \
      > "$AMBI_BENCH_OUTPUT/interval-${interval}.log" 2>&1
  done
fi
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
printf 'Benchmark complete: %s\n' "$AMBI_BENCH_OUTPUT"
