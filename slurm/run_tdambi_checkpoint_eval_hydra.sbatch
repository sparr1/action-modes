#!/usr/bin/env bash
#SBATCH --job-name=tdambi-native-inner-g3
#SBATCH --partition=gpus
#SBATCH --constraint=rtx_a6000
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=2-10
#SBATCH --no-requeue
#SBATCH --output=/tmp/tdambi-eval-%A_%a.out
#SBATCH --error=/tmp/tdambi-eval-%A_%a.err

# One frozen checkpoint per worker: 100k–500k every 50k. Supply durable
# scheduler logs and an explicitly prepared evaluation-curve run directory.
set -Eeuo pipefail
umask 077
fail() { printf 'TDAMBI evaluation: %s\n' "$*" >&2; exit 1; }
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
: "${EXPECTED_ACTION_MODES_SHA:?Supply the exact tested commit}"
: "${AMBI_DMC_PYTHON:?Supply the locked DMControl Python interpreter}"
: "${TDAMBI_CHECKPOINT_PREFIX:?Supply the absolute native checkpoint filename prefix}"
: "${TDAMBI_REFERENCE_ROOT:?Supply the absolute native paired-results root}"
: "${TDAMBI_OUTPUT_ROOT:?Supply an absolute campaign output directory}"
: "${CHECKPOINT_INVENTORY:?Supply the validated native checkpoint inventory}"
: "${EVAL_RUN_DIR:?Supply the explicitly prepared evaluation curve run directory}"
: "${SLURM_ARRAY_TASK_ID:?Use array indices 2 through 10}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || fail "Checkout does not match tested commit."
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || fail "Evaluation requires a clean checkout."
[[ "$SLURM_ARRAY_TASK_ID" =~ ^([2-9]|10)$ ]] || fail "Array index must be 2 through 10."
[[ "$AMBI_DMC_PYTHON" == /* && -x "$AMBI_DMC_PYTHON" ]] || fail "Locked Python must be executable at an absolute path."
for variable in TDAMBI_CHECKPOINT_PREFIX TDAMBI_REFERENCE_ROOT TDAMBI_OUTPUT_ROOT CHECKPOINT_INVENTORY EVAL_RUN_DIR; do
    [[ "${!variable}" == /* ]] || fail "$variable must be absolute."
done
step=$((50000 * SLURM_ARRAY_TASK_ID))
checkpoint="${TDAMBI_CHECKPOINT_PREFIX}${step}"
reference="$TDAMBI_REFERENCE_ROOT/step_${step}/paired.json"
output="$TDAMBI_OUTPUT_ROOT/step_${step}"
[[ -f "$checkpoint" && -f "$checkpoint.metadata.json" ]] || fail "Checkpoint or adjacent sidecar is missing."
[[ -f "$reference" ]] || fail "Matching native prior reference is missing."
[[ -f "$CHECKPOINT_INVENTORY" && -f "$EVAL_RUN_DIR/run.json" ]] || fail "Checkpoint inventory or prepared curve registry is missing."
[[ ! -e "$output" ]] || fail "Checkpoint output already exists; choose a fresh campaign output directory."
"$AMBI_DMC_PYTHON" - "$CHECKPOINT_INVENTORY" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    inventory = json.load(handle)
if inventory.get("source_run") != "rwgao_b-brown-university/ambi/xq3zva9u":
    raise SystemExit("TDAMBI requires the validated native TD-MPC2 xq3zva9u backbone inventory.")
PY
mkdir -p "$TDAMBI_OUTPUT_ROOT"
mkdir "$output"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" TORCHINDUCTOR_COMPILE_THREADS=1
"$AMBI_DMC_PYTHON" evaluate_ambi_checkpoint.py \
    --matrix configs/research/tdambi_humanoid_inner_benchmark.json \
    --preset inner_budget/tdambi_3 --checkpoint "$checkpoint" \
    --checkpoint-inventory "$CHECKPOINT_INVENTORY" \
    --controller-seed 12345 --seeds 101 102 103 104 105 --max-steps 500 --device cuda \
    --reference-bundle "$reference" --eval-run-dir "$EVAL_RUN_DIR" \
    --bundle-dir "$output/bundle" --output "$output/results.json"
"$AMBI_DMC_PYTHON" report_ambi_benchmark.py \
    --bundle "$output/bundle" --output "$output/report.html" \
    --title "TDAMBI G3 on TD-MPC2 prior — checkpoint ${step}"
