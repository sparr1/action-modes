#!/usr/bin/env bash
#SBATCH --job-name=reanalyze-latency
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --no-requeue

# One GPU keeps all timing arms on identical hardware. Setup and three warmed
# 60-root sweeps per arm fit a bounded allocation at the measured native rate.
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${AMBI_PROJECT_DIR:?Supply an absolute clean source checkout.}"
: "${AMBI_PYTHON:?Supply the existing locked DMControl interpreter.}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested full commit SHA.}"
: "${AMBI_BENCH_OUTPUT:?Supply a fresh output directory outside source.}"
[[ "$AMBI_PROJECT_DIR" == /* && "$AMBI_PYTHON" == /* && "$AMBI_BENCH_OUTPUT" == /* ]]
cd "$AMBI_PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "$AMBI_BENCH_OUTPUT" != "$AMBI_PROJECT_DIR" && "$AMBI_BENCH_OUTPUT" != "$AMBI_PROJECT_DIR/"* ]]
mkdir "$AMBI_BENCH_OUTPUT"
benchmark_cache="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/reanalyze-bench-${SLURM_JOB_ID}.XXXXXX")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$benchmark_cache/inductor" TRITON_CACHE_DIR="$benchmark_cache/triton"
export XDG_CACHE_HOME="$benchmark_cache/cache" WANDB_MODE=disabled MUJOCO_GL=egl
export AMBI_BENCH_OUTPUT EXPECTED_ACTION_MODES_SHA
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$AMBI_BENCH_OUTPUT/gpu.csv"
"$AMBI_PYTHON" - <<'PY' > "$AMBI_BENCH_OUTPUT/runtime.json"
import json, os, platform, subprocess
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.get_device_name(0) == "NVIDIA L40S"
assert torch.__version__.split("+")[0] == "2.3.1"
print(json.dumps(dict(
    host=platform.node(), platform=platform.platform(), torch=torch.__version__,
    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
    git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    git_tree=subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip(),
    job_id=os.environ["SLURM_JOB_ID"], cpus=os.environ["SLURM_CPUS_PER_TASK"],
), indent=2))
PY
srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_ambi_teacher_graph.py tests/test_ambi_parallel_teacher.py \
  --basetemp="$benchmark_cache/pytest" --junitxml="$AMBI_BENCH_OUTPUT/cuda-parity.xml" \
  > "$AMBI_BENCH_OUTPUT/cuda-parity.log" 2>&1
srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -m tests.benchmark_ambi_reanalyze \
  --device cuda --profile reference --compile --full-refresh --roots 60 --parallel-roots 1 4 8 \
  --threads 1 --warmup 1 --repeats 3 --output "$AMBI_BENCH_OUTPUT/latency.json" \
  > "$AMBI_BENCH_OUTPUT/latency.log" 2>&1
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
printf 'Benchmark complete: %s\n' "$AMBI_BENCH_OUTPUT"
