#!/usr/bin/env bash
#SBATCH --job-name=ambi-inner-eval
#SBATCH --partition=gpus
#SBATCH --constraint=rtx_a6000
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=1-5
#SBATCH --no-requeue
#SBATCH --output=/tmp/ambi-inner-eval-%A_%a.out
#SBATCH --error=/tmp/ambi-inner-eval-%A_%a.err

# One checkpoint per task: 100k, 200k, 300k, 400k, 500k. Submit from the
# clean checkout, with durable --output/--error paths supplied to sbatch.
set -Eeuo pipefail
umask 077
fail() { echo "AMBI benchmark: $*" >&2; exit 2; }
: "${AMBI_CHECKPOINT_PREFIX:?Set the absolute checkpoint filename prefix, including its final underscore.}"
: "${AMBI_BENCHMARK_OUTPUT_ROOT:?Set an absolute, fresh campaign output directory.}"
: "${EXPECTED_ACTION_MODES_SHA:?Set the exact pushed commit.}"
: "${SLURM_ARRAY_TASK_ID:?Submit as an array with task indices 1 through 5.}"
PYTHON="${AMBI_DMC_PYTHON:-/cs/home/rgao48/projects/action-modes/environments/dmcontrol/.venv/bin/python}"
preset="${AMBI_BENCHMARK_PRESET:-named_run/d512_4_j6}"
matrix="${AMBI_BENCHMARK_MATRIX:-configs/research/ambi_humanoid_inner_benchmark.json}"
cd "${SLURM_SUBMIT_DIR:?}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || fail "commit mismatch"
[[ -z "$(git status --porcelain)" ]] || fail "checkout must be clean"
[[ "$AMBI_CHECKPOINT_PREFIX" == /* && "$AMBI_BENCHMARK_OUTPUT_ROOT" == /* ]] || fail "paths must be absolute"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[1-5]$ ]] || fail "array index must be 1 through 5"
[[ $# -le 1 ]] || fail "expected at most one mode"
case "${1:-}" in
    ''|--smoke|--bank-only|--bank-smoke) ;;
    *) fail "supported modes: --smoke, --bank-only, --bank-smoke" ;;
esac
checkpoint_step=$((SLURM_ARRAY_TASK_ID * 100000))
checkpoint="${AMBI_CHECKPOINT_PREFIX}${checkpoint_step}"
output="${AMBI_BENCHMARK_OUTPUT_ROOT}/step_${checkpoint_step}"
[[ -f "$checkpoint" && -f "$checkpoint.metadata.json" ]] || fail "checkpoint or sidecar missing"
mkdir -p "$AMBI_BENCHMARK_OUTPUT_ROOT"
mkdir "$output" || fail "output already exists: $output"

export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TORCHINDUCTOR_COMPILE_THREADS=1
unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS WANDB_DISABLED
export WANDB_DIR="${SLURM_TMPDIR:-/tmp}/ambi-inner-benchmark-${SLURM_JOB_ID}"
export WANDB_CACHE_DIR="$WANDB_DIR/cache" WANDB_DATA_DIR="$WANDB_DIR/data"
export WANDB_ARTIFACT_DIR="$WANDB_DIR/artifacts" WANDB_DISABLE_CODE=true
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/ambi-inner-compile-${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR"

common=(--matrix "$matrix" --checkpoint "$checkpoint" --device cuda)
if [[ "${1:-}" == --smoke ]]; then
    common+=(--seeds 101 --max-steps 3)
elif [[ "${1:-}" == --bank-smoke ]]; then
    # Keep the bank's 500-decision protocol; shorten repetitions, not episodes.
    common+=(--bank-repetitions 1)
else
    common+=(--wandb)
fi

if [[ "${1:-}" == --bank-only || "${1:-}" == --bank-smoke ]]; then
    : "${AMBI_BENCHMARK_REFERENCE_ROOT:?Set the original evaluation directory containing shared banks.}"
    [[ "$AMBI_BENCHMARK_REFERENCE_ROOT" == /* ]] || fail "reference root must be absolute"
    bank="$AMBI_BENCHMARK_REFERENCE_ROOT/step_${checkpoint_step}/roots.json"
    [[ -f "$bank" ]] || fail "shared observation bank missing: $bank"
    read -r -a bank_presets <<< "${AMBI_BENCHMARK_PRESETS:-$preset}"
    selected=()
    for bank_preset in "${bank_presets[@]}"; do
        selected+=(--preset "$bank_preset")
    done
    [[ ${#selected[@]} -gt 0 ]] || fail "select at least one bank preset"
    "$PYTHON" evaluate_ambi_checkpoint.py "${common[@]}" "${selected[@]}" \
        --root-bank "$bank" --bank-only --bundle-dir "$output/bank" \
        --output "$output/bank-results.json"
    "$PYTHON" report_ambi_benchmark.py --bundle "$output/bank" \
        --output "$output/comparison.html"
    exit 0
fi

prior="$output/prior"
if [[ -n "${AMBI_BENCHMARK_REFERENCE_ROOT:-}" ]]; then
    [[ "$AMBI_BENCHMARK_REFERENCE_ROOT" == /* ]] || fail "reference root must be absolute"
    prior="$AMBI_BENCHMARK_REFERENCE_ROOT/step_${checkpoint_step}/prior"
    [[ -f "$prior/manifest.json" ]] || fail "prior reference missing: $prior"
else
    "$PYTHON" evaluate_ambi_checkpoint.py "${common[@]}" \
        --preset named_run/prior --bundle-dir "$prior" \
        --save-root-bank "$output/roots.json" --output "$output/prior-results.json"
fi
"$PYTHON" evaluate_ambi_checkpoint.py "${common[@]}" \
    --preset "$preset" --bundle-dir "$output/inner" \
    --reference-bundle "$prior" --output "$output/inner-results.json"
"$PYTHON" report_ambi_benchmark.py --bundle "$prior" \
    --bundle "$output/inner" --output "$output/comparison.html"
