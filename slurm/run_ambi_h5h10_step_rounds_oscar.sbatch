#!/usr/bin/env bash
#SBATCH --job-name=ambi-h5h10-step
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

# Each manifest row owns one complete checkpoint/controller result. Submit
# prior rows first; submit inner rows after their prior jobs succeed. Choose
# array concurrency from live account/QOS limits, including other campaigns.
set -Eeuo pipefail
umask 077
fail() { echo "AMBI H5/H10 step evaluation: $*" >&2; exit 2; }
: "${EXPECTED_ACTION_MODES_SHA:?Supply the exact tested and pushed commit.}"
: "${AMBI_EVAL_MANIFEST:?Supply the absolute evaluation manifest.}"
: "${AMBI_BENCHMARK_OUTPUT_ROOT:?Supply a fresh absolute campaign output root.}"
cd "${SLURM_SUBMIT_DIR:?Submit from the clean repository root.}"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_ACTION_MODES_SHA" ]] || fail "commit mismatch"
[[ -z "$(git status --porcelain)" ]] || fail "checkout must be clean"
[[ $# -le 1 && ( $# -eq 0 || "$1" == --smoke ) ]] || fail "only --smoke is supported"
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^(0|[1-9][0-9]*)$ ]] || fail "array index must be nonnegative"
for path in "$AMBI_EVAL_MANIFEST" "$AMBI_BENCHMARK_OUTPUT_ROOT"; do
    [[ "$path" == /* ]] || fail "paths must be absolute"
done
[[ -f "$AMBI_EVAL_MANIFEST" ]] || fail "evaluation manifest missing"
PYTHON="${AMBI_DMC_PYTHON:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

# Parse as data, never as shell code. Validate the selected checkpoint's exact
# bytes and source sidecar before allocating a learner or producing a curve.
fields=$("$PYTHON" - "$AMBI_EVAL_MANIFEST" "$SLURM_ARRAY_TASK_ID" "${1:-}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

def require(value, message):
    if not value:
        raise SystemExit(message)

def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs)

def absolute(value, label):
    require(isinstance(value, str) and value.startswith("/") and
            not any(char in value for char in "\t\r\n"), label + " must be an absolute single-line path")
    return value

manifest = read(sys.argv[1])
require(manifest.get("schema_version") == 1, "unsupported evaluation manifest schema")
rows = manifest.get("rows")
require(isinstance(rows, list) and rows, "evaluation manifest needs rows")
index = int(sys.argv[2])
require(index < len(rows), "array index outside manifest rows")
sources = {"h5": "rwgao_b-brown-university/ambi/8vlf8z3w",
           "h10": "rwgao_b-brown-university/ambi/gky2rxlf"}
owners = set()
for row in rows:
    require(isinstance(row, dict), "manifest row must be an object")
    backbone, stage, step, selector = (row.get(key) for key in ("backbone", "stage", "step", "selector"))
    require(backbone in sources, "unsupported backbone")
    require(stage in {"prior", "inner"}, "unsupported evaluation stage")
    require(type(step) is int and step in range(50000, 500001, 50000), "checkpoint must be on the 50k through 500k grid")
    valid = {"named_run/prior"} if stage == "prior" else {"step_rounds/j1", "step_rounds/j3", "step_rounds/j6"}
    require(selector in valid, "selector does not match stage")
    owner = (backbone, step, selector)
    require(owner not in owners, "duplicate checkpoint/controller output in manifest")
    owners.add(owner)
row = rows[index]
backbone, stage, step, selector = (row[key] for key in ("backbone", "stage", "step", "selector"))
checkpoint = Path(absolute(row.get("checkpoint"), "checkpoint"))
sidecar = Path(str(checkpoint) + ".metadata.json")
inventory_path = absolute(row.get("checkpoint_inventory"), "checkpoint inventory")
inventory = read(inventory_path)
require(inventory.get("source_run") == sources[backbone], "checkpoint inventory source does not match backbone")
matches = [item for item in inventory.get("checkpoints", []) if item.get("step") == step]
require(len(matches) == 1, "inventory must identify exactly one checkpoint at selected step")
for path, key in ((checkpoint, "sha256"), (sidecar, "metadata_sha256")):
    require(path.is_file(), "checkpoint or sidecar missing: " + str(path))
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    require(matches[0].get(key) == digest.hexdigest(), "checkpoint inventory hash mismatch: " + key)
metadata = read(sidecar)
require(metadata.get("checkpoint", {}).get("step") == step, "checkpoint metadata step mismatch")
saved = metadata.get("trial_run_params", {})
require(saved.get("alg") == "AMBITDMPC2/AMBITDMPC2", "checkpoint is not an AMBI SAC backbone")
params = saved.get("alg_params", {})
require(params.get("train_unroll_horizon", params.get("horizon")) == int(backbone[1:]),
        "checkpoint training horizon does not match backbone")
require(params.get("inner_operator") == "none", "checkpoint was not trained with the prior-only controller")
run_dir = "-" if sys.argv[3] == "--smoke" else absolute(row.get("eval_run_dir"), "explicit New evaluation run")
print("\t".join(map(str, (backbone, stage, step, selector, checkpoint, inventory_path, run_dir))))
PY
) || fail "manifest/checkpoint validation failed"
IFS=$'\t' read -r backbone stage step selector checkpoint inventory run_dir <<< "$fields"

checkpoint_output="$AMBI_BENCHMARK_OUTPUT_ROOT/$backbone/step_$step"
prior_matrix="configs/research/ambi_humanoid_${backbone}_prior_benchmark.json"
prior="$checkpoint_output/prior"
if [[ "$stage" == prior ]]; then
    matrix="$prior_matrix"
    output="$checkpoint_output/prior-evaluation"
    bundle="$prior"
else
    matrix="configs/research/ambi_humanoid_${backbone}_inner_step_rounds.json"
    output="$checkpoint_output/${selector#step_rounds/}"
    bundle="$output/inner"
fi
if [[ "${1:-}" != --smoke ]]; then
    [[ -f "$run_dir/run.json" ]] || fail "explicit New evaluation run missing"
    if [[ "$stage" == inner ]]; then
        [[ -f "$prior/manifest.json" ]] || fail "completed paired prior missing; submit after prior stage succeeds"
    fi
elif [[ "$stage" == inner ]]; then
    prior="$output/prior"
fi
[[ ! -e "$bundle" ]] || fail "bundle already exists: $bundle"
mkdir -p "$checkpoint_output"
mkdir "$output" || fail "output already exists: $output"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}" TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/ambi-h5h10-step-${SLURM_JOB_ID}"
unset WANDB_RUN_ID WANDB_RESUME WANDB_NAME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_TAGS WANDB_DISABLED
common=(--checkpoint "$checkpoint" --checkpoint-inventory "$inventory" --device cuda --controller-seed 55)
if [[ "${1:-}" == --smoke ]]; then
    common+=(--seeds 101 102 --max-steps 3)
else
    common+=(--seeds 101 102 103 104 105 --max-steps 500)
fi
# This metadata-only path resolves the planner and verifies source provenance;
# GPU evaluation repeats the compatibility guard when assigned its explicit run.
"$PYTHON" evaluate_ambi_checkpoint.py "${common[@]}" --matrix "$matrix" --preset "$selector" \
    --eval-series-spec-dir "$output/specification"
evaluation=("${common[@]}" --matrix "$matrix" --preset "$selector")
if [[ "${1:-}" != --smoke ]]; then
    "$PYTHON" eval_series.py append "$run_dir" --spec "$output/specification/${selector//\//__}.json"
    evaluation+=(--eval-run-dir "$run_dir")
elif [[ "$stage" == inner ]]; then
    "$PYTHON" evaluate_ambi_checkpoint.py "${common[@]}" --matrix "$prior_matrix" \
        --preset named_run/prior --bundle-dir "$prior" --output "$output/prior-results.json"
fi
if [[ "$stage" == inner ]]; then
    evaluation+=(--reference-bundle "$prior")
fi
"$PYTHON" evaluate_ambi_checkpoint.py "${evaluation[@]}" \
    --bundle-dir "$bundle" --output "$output/results.json"
reports=(--bundle "$bundle")
if [[ "$stage" == inner ]]; then
    reports=(--bundle "$prior" "${reports[@]}")
fi
"$PYTHON" report_ambi_benchmark.py "${reports[@]}" --output "$output/comparison.html"
