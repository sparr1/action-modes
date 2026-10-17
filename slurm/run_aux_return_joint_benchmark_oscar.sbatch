#!/usr/bin/env bash
#SBATCH --job-name=aux-joint-vmap-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

# Submit a clean tested checkout, with logs/output outside source. The benchmark
# runs eager and strict Inductor modes sequentially on the same allocated GPU.
set -Eeuo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit through Slurm.}"
: "${AMBI_PROJECT_DIR:?Supply an absolute clean source checkout.}"
: "${AMBI_PYTHON:?Supply the existing locked DMControl interpreter.}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested full commit SHA.}"
: "${AMBI_BENCH_OUTPUT:?Supply a fresh absolute output directory outside source.}"
[[ "$AMBI_PROJECT_DIR" == /* && "$AMBI_PYTHON" == /* && "$AMBI_BENCH_OUTPUT" == /* ]]
cd "$AMBI_PROJECT_DIR"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]]
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
[[ "$AMBI_BENCH_OUTPUT" != "$AMBI_PROJECT_DIR" && "$AMBI_BENCH_OUTPUT" != "$AMBI_PROJECT_DIR/"* ]]
mkdir "$AMBI_BENCH_OUTPUT"
benchmark_cache="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/aux-joint-bench-${SLURM_JOB_ID}.XXXXXX")"
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
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
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
    tests/test_aux_return_joint_cuda.py --basetemp="$benchmark_cache/pytest" \
    > "$AMBI_BENCH_OUTPUT/cuda-parity.log" 2>&1
benchmark_args=(--device cuda --operator mppi --width 512 --encoder-width 256 \
    --latent-dim 512 --obs-dim 67 --action-dim 21 --bins 101 --heads 5 \
    --batch-size 256 --horizon 3 --mppi-horizon 3 --mppi-samples 128 \
    --mppi-iterations 2 --mppi-elites 8 --steps 100 --warmup 10 --repeats 4 --threads 1)
for benchmark_mode in eager compiled; do
    compile_args=()
    if [[ "$benchmark_mode" == compiled ]]; then compile_args=(--compile); fi
    srun --unbuffered --kill-on-bad-exit=1 "$AMBI_PYTHON" -m tests.benchmark_aux_return_joint_vmap \
        "${benchmark_args[@]}" "${compile_args[@]}" \
        --output "$AMBI_BENCH_OUTPUT/$benchmark_mode.json" \
        > "$AMBI_BENCH_OUTPUT/$benchmark_mode.log" 2>&1
done
[[ -z "$(git status --porcelain --untracked-files=all)" ]]
printf 'Benchmark complete: %s\n' "$AMBI_BENCH_OUTPUT"
