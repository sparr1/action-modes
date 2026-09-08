#!/usr/bin/env bash
#SBATCH --job-name=ambi-lora-step
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-29
#SBATCH --no-requeue

# One complete checkpoint/J cell per task; set concurrency from live QOS limits.
# Default groups: indices 0-9 -> J6, 10-19 -> J3, 20-29 -> J1, each 50k-500k.
# AMBI_ROUND_VALUES='6' with --array=0-9 selects only J6, without duplicate work.
set -Eeuo pipefail
umask 077
fail() { echo "AMBI LoRA step-round evaluation: $*" >&2; exit 2; }
: "${AMBI_CHECKPOINT_PREFIX:?Supply the absolute checkpoint filename prefix.}"
: "${AMBI_BENCHMARK_OUTPUT_ROOT:?Supply a fresh absolute campaign output root.}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the exact tested and pushed commit.}"
: "${CHECKPOINT_MANIFEST:?Supply the verified source checkpoint inventory.}"
cd "${SLURM_SUBMIT_DIR:?Submit from the clean repository root.}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || fail "commit mismatch"
[[ -z "$(git status --porcelain)" ]] || fail "checkout must be clean"
[[ $# -le 1 && ( $# -eq 0 || "$1" == --smoke ) ]] || fail "only --smoke is supported"
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^(0|[1-9][0-9]*)$ ]] || fail "array index must be a nonnegative integer"
round_values="${AMBI_ROUND_VALUES-6 3 1}"
[[ "$round_values" != *$'\n'* ]] || fail "AMBI_ROUND_VALUES must be a single line"
read -r -a rounds <<< "$round_values"
[[ ${#rounds[@]} -gt 0 ]] || fail "select at least one round count"
seen=" "
for round in "${rounds[@]}"; do
    [[ "$round" =~ ^(1|3|6)$ ]] || fail "AMBI_ROUND_VALUES supports only 1, 3, and 6"
    [[ "$seen" != *" $round "* ]] || fail "duplicate round count: $round"
    seen+="$round "
done
[[ "$SLURM_ARRAY_TASK_ID" -lt $((${#rounds[@]} * 10)) ]] || fail "array index exceeds selected checkpoint/round cells"
for path in "$AMBI_CHECKPOINT_PREFIX" "$AMBI_BENCHMARK_OUTPUT_ROOT" "$CHECKPOINT_MANIFEST"; do
    [[ "$path" == /* ]] || fail "paths must be absolute"
done
[[ -f "$CHECKPOINT_MANIFEST" ]] || fail "checkpoint inventory missing"
PYTHON="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
matrix=configs/research/ambi_humanoid_inner_lora_step_rounds.json
step=$(((SLURM_ARRAY_TASK_ID % 10 + 1) * 50000))
j="${rounds[$((SLURM_ARRAY_TASK_ID / 10))]}"
checkpoint="${AMBI_CHECKPOINT_PREFIX}${step}"
output="${AMBI_BENCHMARK_OUTPUT_ROOT}/step_${step}/j${j}"
[[ -f "$checkpoint" && -f "$checkpoint.metadata.json" ]] || fail "checkpoint or sidecar missing"
if [[ "${1:-}" != --smoke ]]; then
    : "${EVAL_RUN_MAP:?Prepare explicit New/Append evaluation runs before production submission.}"
    : "${AMBI_BENCHMARK_REFERENCE_ROOT:?Supply the verified paired-reference root.}"
    [[ "$EVAL_RUN_MAP" == /* && -f "$EVAL_RUN_MAP" ]] || fail "run map missing"
    [[ "$AMBI_BENCHMARK_REFERENCE_ROOT" == /* ]] || fail "reference root must be absolute"
    prior="$AMBI_BENCHMARK_REFERENCE_ROOT/step_${step}/prior"
    [[ -f "$prior/manifest.json" ]] || fail "paired prior reference missing; production never regenerates prior episodes"
fi
mkdir -p "$AMBI_BENCHMARK_OUTPUT_ROOT/step_${step}"
mkdir "$output" || fail "output already exists: $output"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/ambi-lora-step-${SLURM_JOB_ID}"
unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS WANDB_DISABLED
common=(--checkpoint "$checkpoint" --checkpoint-inventory "$CHECKPOINT_MANIFEST"
        --device cuda --controller-seed 55)
if [[ "${1:-}" == --smoke ]]; then
    common+=(--seeds 101 102 --max-steps 3)
    prior="$output/prior"
    # Only smoke creates its own private short prior; no production prior is run.
    "$PYTHON" evaluate_ambi_checkpoint.py "${common[@]}" \
        --matrix configs/research/ambi_humanoid_inner_benchmark.json \
        --preset named_run/prior --bundle-dir "$prior" --output "$output/prior-results.json"
else
    common+=(--seeds 101 102 103 104 105 --max-steps 500)
fi
# Array mapping owns the only selected preset; inherited benchmark overrides
# cannot expand a task into another cell or bypass the chosen round subset.
selected=(--preset "lora_step_rounds/j${j}")
sac_common=("${common[@]}")
if [[ "${1:-}" != --smoke ]]; then
    sac_common+=(--eval-run-map "$EVAL_RUN_MAP")
fi
"$PYTHON" evaluate_ambi_checkpoint.py "${sac_common[@]}" \
    --matrix "$matrix" "${selected[@]}" --reference-bundle "$prior" \
    --bundle-dir "$output/inner" --output "$output/inner-results.json"
"$PYTHON" report_ambi_benchmark.py --bundle "$prior" --bundle "$output/inner" \
    --output "$output/comparison.html"
