#!/usr/bin/env bash
#SBATCH --job-name=td-ambi-prior-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-79
#SBATCH --no-requeue
#SBATCH --output=/tmp/td-ambi-prior-eval-%A_%a.out
#SBATCH --error=/tmp/td-ambi-prior-eval-%A_%a.err

# One checkpoint and planner stage per task. Submission controls live capacity
# and dependencies; fixed/adaptive reuse the completed matching MPPI prior.
set -Eeuo pipefail
umask 077
fail() { echo "TD-AMBI prior evaluation: $*" >&2; exit 2; }
: "${EXPECTED_ACTION_MODES_SHA:?Set the tested, pushed commit.}"
: "${BANK_MATRIX:?Set the bank-specific checkpoint matrix.}"
: "${CHECKPOINT_MANIFEST:?Set the verified absolute checkpoint inventory.}"
: "${AMBI_BENCHMARK_OUTPUT_ROOT:?Set the fresh absolute campaign output directory.}"
: "${SLURM_ARRAY_TASK_ID:?Use array indices 0 through 79.}"
cd "${SLURM_SUBMIT_DIR:?Submit from the source checkout.}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" && -z "$(git status --porcelain)" ]] || fail "source commit/cleanliness mismatch"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^([0-9]|[1-7][0-9])$ && $# -le 1 ]] || fail "expected indices 0 through 79 and at most one mode"
case "${1:-}" in ''|--smoke) ;; *) fail "only --smoke is supported" ;; esac
mode="${TD_AMBI_EVAL_MODE:-mppi}"
case "$mode" in mppi|fixed|adaptive) ;; *) fail "TD_AMBI_EVAL_MODE must be mppi, fixed, or adaptive" ;; esac
[[ "$CHECKPOINT_MANIFEST" == /* && "$AMBI_BENCHMARK_OUTPUT_ROOT" == /* ]] || fail "inventory/output paths must be absolute"
[[ -f "$CHECKPOINT_MANIFEST" && -f "$BANK_MATRIX" ]] || fail "checkpoint inventory or bank matrix missing"
PYTHON="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON" ]] || fail "locked DMControl interpreter missing"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/td-ambi-eval-${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID}"
unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS

# Assignment captures Python's exit status, so a failed hash/schema preflight
# cannot be hidden by shell process substitution. Tab/newline paths are rejected.
row=$("$PYTHON" - "$CHECKPOINT_MANIFEST" "$SLURM_ARRAY_TASK_ID" "$BANK_MATRIX" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

manifest_path, index, matrix_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
inventory = json.loads(Path(manifest_path).read_text())
rows = inventory["checkpoints"]
if len(rows) != 80 or [row.get("step") for row in rows] != list(range(25000, 2000001, 25000)):
    raise ValueError("Inventory must contain the ordered 80-checkpoint 25k-through-2M bank.")
row = rows[index]
matrix = json.loads(Path(matrix_path).read_text())
source = row["source_run"]
if not isinstance(source, str) or not source or source != matrix.get("source_run"):
    raise ValueError("Checkpoint row source_run must match the selected bank matrix.")
checkpoint = Path(row["path"])
if not checkpoint.is_absolute() or any(char in str(checkpoint) for char in "\t\r\n"):
    raise ValueError("Checkpoint paths must be absolute and contain no tabs/newlines.")
for path, key in ((checkpoint, "sha256"), (Path(str(checkpoint) + ".metadata.json"), "metadata_sha256")):
    expected = row[key]
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Invalid {key} in checkpoint inventory.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"Checkpoint inventory hash mismatch: {path}")
print(f"{row['step']}\t{checkpoint}")
PY
)
IFS=$'\t' read -r step checkpoint <<< "$row"
[[ "$step" =~ ^[0-9]+$ && -f "$checkpoint" ]] || fail "invalid checkpoint selection"

if [[ "${1:-}" == --smoke ]]; then
    stage=smoke
    selectors=(--preset controller/prior --preset controller/mppi --preset inner/fixed --preset inner/adaptive)
    evaluation=(--seeds 101 102 --max-steps 3)
else
    : "${EVAL_RUN_MAP:?Set the explicit New/Append run assignment before production.}"
    [[ "$EVAL_RUN_MAP" == /* && -f "$EVAL_RUN_MAP" ]] || fail "run-map path must be an existing absolute file"
    stage="$mode"
    evaluation=(--eval-run-map "$EVAL_RUN_MAP" --seeds 101 102 103 104 105 --max-steps 500)
    if [[ "$mode" == mppi ]]; then
        selectors=(--preset controller/prior --preset controller/mppi)
    else
        selectors=(--preset "inner/$mode")
        prior="$AMBI_BENCHMARK_OUTPUT_ROOT/mppi/step_${step}/bundle"
        [[ -f "$prior/manifest.json" ]] || fail "completed MPPI/prior reference missing: $prior"
        evaluation+=(--reference-bundle "$prior")
    fi
fi
output="$AMBI_BENCHMARK_OUTPUT_ROOT/$stage/step_${step}"
mkdir -p "$AMBI_BENCHMARK_OUTPUT_ROOT/$stage"
mkdir "$output"  # Exclusive task ownership; preserve partial/completed results.
"$PYTHON" evaluate_ambi_checkpoint.py --matrix "$BANK_MATRIX" --checkpoint "$checkpoint" \
    --checkpoint-inventory "$CHECKPOINT_MANIFEST" --device cuda \
    --bundle-dir "$output/bundle" --output "$output/results.json" \
    "${selectors[@]}" "${evaluation[@]}"
report=(--bundle "$output/bundle")
if [[ "$stage" != smoke && "$mode" != mppi ]]; then
    report=(--bundle "$prior" "${report[@]}")
fi
"$PYTHON" report_ambi_benchmark.py "${report[@]}" \
    --output "$output/comparison.html" --title "TD-AMBI $stage - checkpoint $step"
