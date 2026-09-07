#!/usr/bin/env bash
#SBATCH --job-name=tdambi-checkpoint-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --no-requeue

# Submit from a clean, tested Git checkout. Supply durable Slurm log paths.
set -Eeuo pipefail
umask 077
fail() { printf '%s\n' "$*" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the tested commit}"
: "${TDAMBI_CHECKPOINT:?Supply an absolute native TD-MPC2 checkpoint path}"
: "${TDAMBI_SMOKE_OUTPUT_ROOT:?Supply a fresh absolute scratch output directory}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || fail "Checkout does not match tested commit."
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "Smoke requires a clean checkout."
[[ "$TDAMBI_CHECKPOINT" == /* && -f "$TDAMBI_CHECKPOINT" && -f "$TDAMBI_CHECKPOINT.metadata.json" ]] || fail "Checkpoint and adjacent sidecar must exist at absolute paths."
[[ "$TDAMBI_SMOKE_OUTPUT_ROOT" == /* && ! -e "$TDAMBI_SMOKE_OUTPUT_ROOT" ]] || fail "Output directory must be fresh and absolute."
[[ "${TDAMBI_MEASURE_TRACE:-0}" == 0 || "${TDAMBI_MEASURE_TRACE:-0}" == 1 ]] || fail "TDAMBI_MEASURE_TRACE must be 0 or 1."
PYTHON_BIN="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || fail "Locked Python runtime is not executable."
mkdir -p "$(dirname "$TDAMBI_SMOKE_OUTPUT_ROOT")"
mkdir "$TDAMBI_SMOKE_OUTPUT_ROOT"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/tdambi-${SLURM_JOB_ID}"

"$PYTHON_BIN" -m pytest -q tests/test_tdambi_inner.py tests/test_tdambi_checkpoint.py \
    tests/test_tdambi_publication.py
"$PYTHON_BIN" evaluate_ambi_checkpoint.py \
    --matrix configs/research/tdambi_humanoid_inner_benchmark.json \
    --preset inner_budget/tdambi_3 --checkpoint "$TDAMBI_CHECKPOINT" \
    --controller-seed 12345 --seeds 101 --max-steps 3 --device cuda \
    --bundle-dir "$TDAMBI_SMOKE_OUTPUT_ROOT/bundle" \
    --output "$TDAMBI_SMOKE_OUTPUT_ROOT/results.json"
"$PYTHON_BIN" report_ambi_benchmark.py \
    --bundle "$TDAMBI_SMOKE_OUTPUT_ROOT/bundle" \
    --output "$TDAMBI_SMOKE_OUTPUT_ROOT/report.html" --title "TDAMBI checkpoint smoke"

if [[ "${TDAMBI_MEASURE_TRACE:-0}" == 1 ]]; then
    export AMBI_RUN_TRACE_CUDA_BENCHMARK=1
    export AMBI_TRACE_BENCHMARK_CHECKPOINT="$TDAMBI_CHECKPOINT"
    export AMBI_TRACE_BENCHMARK_MATRIX="$PWD/configs/research/tdambi_humanoid_inner_benchmark.json"
    export AMBI_TRACE_BENCHMARK_PRESET=inner_budget/tdambi_3
    export AMBI_TRACE_BENCHMARK_CONTROLLER_SEED=12345
    export AMBI_TRACE_BENCHMARK_OUTPUT="$TDAMBI_SMOKE_OUTPUT_ROOT/trace-overhead.json"
    "$PYTHON_BIN" -m pytest -s -q tests/test_ambi_inner_trace.py::test_cuda_trace_overhead_measurement
fi
