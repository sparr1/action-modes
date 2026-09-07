#!/usr/bin/env bash
#SBATCH --job-name=ambi-prior-mppi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=1-5
#SBATCH --no-requeue
#SBATCH --output=/tmp/ambi-prior-mppi-%A_%a.out
#SBATCH --error=/tmp/ambi-prior-mppi-%A_%a.err

# Frozen original AMBI prior: five checkpoints, existing prior references.
set -Eeuo pipefail
umask 077
fail() { echo "AMBI prior MPPI: $*" >&2; exit 2; }
: "${EXPECTED_ACTION_MODES_SHA:?Set the tested, pushed commit.}"
: "${AMBI_CHECKPOINT_PREFIX:?Absolute checkpoint prefix ending with underscore.}"
: "${AMBI_BENCHMARK_OUTPUT_ROOT:?Fresh absolute result directory.}"
: "${CHECKPOINT_MANIFEST:?Verified source-run/checkpoint inventory.}"
: "${SLURM_ARRAY_TASK_ID:?Use array indices 1 through 5.}"
cd "${SLURM_SUBMIT_DIR:?Submit from the source checkout.}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" && -z "$(git status --porcelain)" ]] || fail "source commit/cleanliness mismatch"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[1-5]$ && $# -le 1 ]] || fail "expected indices 1 through 5 and at most one mode"
[[ "$AMBI_CHECKPOINT_PREFIX" == /* && "$AMBI_BENCHMARK_OUTPUT_ROOT" == /* && "$CHECKPOINT_MANIFEST" == /* ]] || fail "paths must be absolute"
[[ -f "$CHECKPOINT_MANIFEST" ]] || fail "checkpoint inventory missing"
case "${1:-}" in ''|--smoke) ;; *) exit 2 ;; esac
step=$((SLURM_ARRAY_TASK_ID * 100000))
checkpoint="${AMBI_CHECKPOINT_PREFIX}${step}"
output="${AMBI_BENCHMARK_OUTPUT_ROOT}/step_${step}"
[[ -f "$checkpoint" && -f "$checkpoint.metadata.json" ]] || fail "checkpoint or sidecar missing"
mkdir -p "$AMBI_BENCHMARK_OUTPUT_ROOT"
mkdir "$output"
PYTHON="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/ambi-prior-mppi-${SLURM_JOB_ID}"
unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS
common=(--matrix configs/research/ambi_humanoid_native_mppi_benchmark.json
        --checkpoint "$checkpoint" --checkpoint-inventory "$CHECKPOINT_MANIFEST"
        --device cuda --bundle-dir "$output/bundle" --output "$output/results.json")
if [[ "${1:-}" == --smoke ]]; then
    # Short paired smoke only; production reuses the existing full prior episodes.
    "$PYTHON" evaluate_ambi_checkpoint.py "${common[@]}" --preset controller/prior --preset controller/mppi \
        --seeds 101 102 --max-steps 3
    report=(--bundle "$output/bundle")
else
    : "${AMBI_BENCHMARK_REFERENCE_ROOT:?Existing prior bundles, never repeated automatically.}"
    : "${EVAL_RUN_MAP:?Explicit New/Append run assignment prepared before jobs launch.}"
    [[ "$AMBI_BENCHMARK_REFERENCE_ROOT" == /* && "$EVAL_RUN_MAP" == /* && -f "$EVAL_RUN_MAP" ]] || fail "reference/run-map paths invalid"
    prior="$AMBI_BENCHMARK_REFERENCE_ROOT/step_${step}/prior"
    [[ -f "$prior/manifest.json" ]] || fail "prior reference missing"
    "$PYTHON" evaluate_ambi_checkpoint.py "${common[@]}" --preset controller/mppi \
        --reference-bundle "$prior" --eval-run-map "$EVAL_RUN_MAP" --seeds 101 102 103 104 105 --max-steps 500
    report=(--bundle "$prior" --bundle "$output/bundle")
fi
"$PYTHON" report_ambi_benchmark.py "${report[@]}" --output "$output/comparison.html" \
    --title "Original AMBI prior versus native MPPI — checkpoint $step"
