#!/usr/bin/env bash
#SBATCH --job-name=pave-view-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --no-requeue

# One GPU keeps parity and all paired timings on the same device. The prior
# joint-critic benchmark took seven minutes with two CPUs; this budget includes
# the added higher-derivative gate and both geometry batch sizes.
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
pave_cache="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/pave-views-${SLURM_JOB_ID}.XXXXXX")"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$pave_cache/inductor" TRITON_CACHE_DIR="$pave_cache/triton"
export XDG_CACHE_HOME="$pave_cache/cache" WANDB_MODE=disabled MUJOCO_GL=egl
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$AMBI_BENCH_OUTPUT/gpu.csv"
"$AMBI_PYTHON" - <<'PY' > "$AMBI_BENCH_OUTPUT/runtime.json"
import json, os, platform, subprocess
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.get_device_name(0) == "NVIDIA L40S"
assert torch.__version__.split("+")[0] == "2.3.1"
print(json.dumps(dict(
    host=platform.node(), torch=torch.__version__, cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(0), job_id=os.environ["SLURM_JOB_ID"],
    cpus=os.environ["SLURM_CPUS_PER_TASK"],
    git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    git_tree=subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip(),
), indent=2))
PY
srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -m pytest -q -p no:cacheprovider \
    tests/test_pave_view_batching.py tests/test_pave_view_batching_update.py -k cuda \
    --basetemp="$pave_cache/pytest" > "$AMBI_BENCH_OUTPUT/cuda-parity.log" 2>&1
for pave_rows in 64 768; do
    srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -m tests.benchmark_pave_views \
        --device cuda --compile --pave-rows "$pave_rows" --threads 1 \
        --warmup 10 --steps 100 --repeats 4 \
        --output "$AMBI_BENCH_OUTPUT/rows-$pave_rows.json" \
        > "$AMBI_BENCH_OUTPUT/rows-$pave_rows.log" 2>&1
done
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
printf 'PAVE validation complete: %s\n' "$AMBI_BENCH_OUTPUT"
