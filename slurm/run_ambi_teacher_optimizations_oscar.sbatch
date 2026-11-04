#!/usr/bin/env bash
#SBATCH --job-name=teacher-optimization
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -Eeuo pipefail
umask 077
ulimit -c 0
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${AMBI_PROJECT_DIR:?Supply a clean source checkout.}"
: "${AMBI_PYTHON:?Supply the locked interpreter.}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested full SHA.}"
: "${AMBI_BENCH_OUTPUT:?Supply a fresh output directory outside source.}"
cd "$AMBI_PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "$AMBI_BENCH_OUTPUT" == /* && "$AMBI_BENCH_OUTPUT" != "$AMBI_PROJECT_DIR" && "$AMBI_BENCH_OUTPUT" != "$AMBI_PROJECT_DIR/"* ]]
mkdir "$AMBI_BENCH_OUTPUT"
benchmark_cache="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/teacher-optimizations-${SLURM_JOB_ID}.XXXXXX")"
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
if [[ "${AMBI_PRECISION_DIAGNOSTIC:-0}" == 1 ]]; then
  srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -m tests.diagnose_ambi_teacher_precision \
    --output "$AMBI_BENCH_OUTPUT/precision.json" > "$AMBI_BENCH_OUTPUT/precision.log" 2>&1
  [[ -z "$(git status --porcelain --untracked-files=all)" ]]
  exit 0
fi
srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -m pytest -q --tb=short -p no:cacheprovider \
  tests/test_ambi_batched_teacher.py tests/test_ambi_batched_optimizations.py \
  tests/test_ambi_batched_pool.py tests/test_ambi_reanalyze_backends.py \
  tests/test_ambi_batched_reference_recipe.py tests/test_ambi_batched_cuda_semantics.py \
  --basetemp="$benchmark_cache/pytest" --junitxml="$AMBI_BENCH_OUTPUT/gate.xml" \
  > "$AMBI_BENCH_OUTPUT/gate.log" 2>&1
if [[ "${AMBI_GATE_ONLY:-0}" != 1 ]]; then
  for arm in reference:1 heads:1 heads-cache:1 all:1 all:2 all:4; do
    features="${arm%:*}"
    workers="${arm#*:}"
    srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -m tests.benchmark_ambi_teacher_optimizations \
      --features "$features" --workers "$workers" --width 4 --roots 60 --warmup 1 --repeats 5 \
      --output "$AMBI_BENCH_OUTPUT/${features}-workers${workers}.json" \
      > "$AMBI_BENCH_OUTPUT/${features}-workers${workers}.log" 2>&1
  done
fi
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
printf 'Benchmark complete: %s\n' "$AMBI_BENCH_OUTPUT"
