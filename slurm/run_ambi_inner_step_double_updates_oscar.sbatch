#!/usr/bin/env bash
#SBATCH --job-name=ambi-step-double
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=1-10%3
#SBATCH --no-requeue

# Two updates per parallel step; checkpoints 50k through 500k and J1/J3/J6.
# Prepare explicit New run assignments and verified reference links first.
set -Eeuo pipefail
umask 077
fail() { echo "AMBI two-update step-round evaluation: $*" >&2; exit 2; }
: "${AMBI_CHECKPOINT_PREFIX:?Supply the absolute checkpoint filename prefix.}"
: "${AMBI_BENCHMARK_OUTPUT_ROOT:?Supply a fresh absolute campaign output root.}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the exact tested and pushed commit.}"
: "${CHECKPOINT_MANIFEST:?Supply the verified source checkpoint inventory.}"
cd "${SLURM_SUBMIT_DIR:?Submit from the clean repository root.}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || fail "commit mismatch"
[[ -z "$(git status --porcelain)" ]] || fail "checkout must be clean"
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^([1-9]|10)$ ]] || fail "array index must be 1 through 10"
[[ $# -le 1 && ( $# -eq 0 || "$1" == --smoke ) ]] || fail "only --smoke is supported"
for path in "$AMBI_CHECKPOINT_PREFIX" "$AMBI_BENCHMARK_OUTPUT_ROOT" "$CHECKPOINT_MANIFEST"; do
    [[ "$path" == /* ]] || fail "paths must be absolute"
done
[[ -f "$CHECKPOINT_MANIFEST" ]] || fail "checkpoint inventory missing"
PYTHON="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
matrix=configs/research/ambi_humanoid_inner_step_double_updates.json
step=$((SLURM_ARRAY_TASK_ID * 50000))
checkpoint="${AMBI_CHECKPOINT_PREFIX}${step}"
output="${AMBI_BENCHMARK_OUTPUT_ROOT}/step_${step}"
[[ -f "$checkpoint" && -f "$checkpoint.metadata.json" ]] || fail "checkpoint or sidecar missing"
if [[ "${1:-}" != --smoke ]]; then
    : "${EVAL_RUN_MAP:?Prepare explicit New evaluation runs before production submission.}"
    : "${AMBI_BENCHMARK_REFERENCE_ROOT:?Supply the verified paired-reference root.}"
    [[ "$EVAL_RUN_MAP" == /* && -f "$EVAL_RUN_MAP" ]] || fail "run map missing"
    [[ "$AMBI_BENCHMARK_REFERENCE_ROOT" == /* ]] || fail "reference root must be absolute"
    prior="$AMBI_BENCHMARK_REFERENCE_ROOT/step_${step}/prior"
    [[ -f "$prior/manifest.json" ]] || fail "paired prior reference missing"
fi
mkdir -p "$AMBI_BENCHMARK_OUTPUT_ROOT"
mkdir "$output" || fail "output already exists: $output"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/ambi-step-double-${SLURM_JOB_ID}"
unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS WANDB_DISABLED
common=(--checkpoint "$checkpoint" --checkpoint-inventory "$CHECKPOINT_MANIFEST"
        --device cuda --controller-seed 55)
if [[ "${1:-}" == --smoke ]]; then
    common+=(--seeds 101 102 --max-steps 3)
    prior="$output/prior"
else
    common+=(--seeds 101 102 103 104 105 --max-steps 500)
fi
if [[ ! -f "$prior/manifest.json" ]]; then
    # Production requires every completed prior reference; only smoke makes a short prior.
    prior="$output/prior"
    "$PYTHON" evaluate_ambi_checkpoint.py "${common[@]}" \
        --matrix configs/research/ambi_humanoid_inner_benchmark.json \
        --preset named_run/prior --bundle-dir "$prior" --output "$output/prior-results.json"
fi
read -r -a presets <<< "${AMBI_BENCHMARK_PRESETS:-step_double_updates/j1 step_double_updates/j3 step_double_updates/j6}"
selected=()
for preset in "${presets[@]}"; do
    [[ "$preset" =~ ^step_double_updates/j(1|3|6)$ ]] || fail "unsupported round-count preset"
    selected+=(--preset "$preset")
done
[[ ${#selected[@]} -gt 0 ]] || fail "select at least one preset"
sac_common=("${common[@]}")
if [[ "${1:-}" != --smoke ]]; then
    sac_common+=(--eval-run-map "$EVAL_RUN_MAP")
fi
"$PYTHON" evaluate_ambi_checkpoint.py "${sac_common[@]}" \
    --matrix "$matrix" "${selected[@]}" --reference-bundle "$prior" \
    --bundle-dir "$output/inner" --output "$output/inner-results.json"
"$PYTHON" report_ambi_benchmark.py --bundle "$prior" --bundle "$output/inner" \
    --output "$output/comparison.html"
