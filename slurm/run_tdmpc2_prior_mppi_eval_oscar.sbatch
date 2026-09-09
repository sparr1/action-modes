#!/usr/bin/env bash
#SBATCH --job-name=tdmpc2-prior-mppi
#SBATCH --partition=gpu
#SBATCH --qos=pri-gpu+
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --array=0
#SBATCH --no-requeue
#SBATCH --output=/tmp/tdmpc2-prior-mppi-%A_%a.out
#SBATCH --error=/tmp/tdmpc2-prior-mppi-%A_%a.err

# Submit from the clean checkout, supplying persistent Slurm output/error paths.
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root.}"

PYTHON_BIN="${PYTHON_BIN:-/oscar/home/rgao48/action-modes/environments/dmcontrol/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing locked DMControl Python: $PYTHON_BIN" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || {
  echo "Refusing to evaluate from a checkout with uncommitted files." >&2; exit 1;
}
SOURCE_COMMIT=$(git rev-parse HEAD)
SOURCE_TREE=$(git rev-parse 'HEAD^{tree}')
[[ -z "${EXPECTED_ACTION_MODES_SHA:-}" || "$SOURCE_COMMIT" == "$EXPECTED_ACTION_MODES_SHA" ]] || {
  echo "Source commit differs from EXPECTED_ACTION_MODES_SHA." >&2; exit 1;
}

: "${CHECKPOINT_MANIFEST:?Supply the absolute checkpoint manifest path.}"
: "${RESULT_ROOT:?Supply the absolute campaign result directory.}"
: "${CAMPAIGN:?Supply a descriptive campaign identifier.}"
: "${SLURM_ARRAY_TASK_ID:?Missing array task ID.}"
export CHECKPOINT_MANIFEST RESULT_ROOT CAMPAIGN SOURCE_COMMIT SOURCE_TREE
export NO_WARM_START="${NO_WARM_START:-0}"
export PRIOR_REFERENCE_ROOT="${PRIOR_REFERENCE_ROOT:-}"
[[ "$NO_WARM_START" == 0 || "$NO_WARM_START" == 1 ]] || { echo "NO_WARM_START must be 0 or 1" >&2; exit 2; }
export REQUIRE_CUDA="${REQUIRE_CUDA:-$NO_WARM_START}"
if [[ "${WANDB_MODE:-online}" != disabled ]]; then
  : "${EVAL_RUN_MAP:?Prepare explicit New/Append prior and MPPI runs before submission.}"
  [[ "$EVAL_RUN_MAP" == /* && -f "$EVAL_RUN_MAP" ]] || exit 2
  export EVAL_RUN_MAP
fi
export EPISODES="${EPISODES:-5}" MAX_STEPS="${MAX_STEPS:-500}"
export EXPECTED_MAX_STEP="${EXPECTED_MAX_STEP:-1500000}"

# Select one immutable input and check the saved settings; never override the
# checkpoint's architecture or silently change the native planning protocol.
SELECTION=$("$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import tempfile
from publish_tdmpc2_mppi_eval import STEPS, _expected_steps, evaluation_source_fingerprint

def require(condition, message):
    if not condition:
        raise SystemExit(message)

if os.environ.get("EVAL_RUN_MAP"):
    from utils.eval_series import load_run
    assignments = json.loads(Path(os.environ["EVAL_RUN_MAP"]).read_text())
    expected_controllers = {"native_mppi"} if os.environ["PRIOR_REFERENCE_ROOT"] else {"policy_prior", "native_mppi"}
    require(set(assignments) == expected_controllers, "Run map must assign exactly the evaluated controllers")
    require(len(set(assignments.values())) == len(expected_controllers), "Prior and MPPI require distinct runs")
    for run_dir in assignments.values():
        load_run(run_dir)

manifest = Path(os.environ["CHECKPOINT_MANIFEST"])
root = Path(os.environ["RESULT_ROOT"])
require(manifest.is_absolute() and root.is_absolute(), "Input and result paths must be absolute.")
manifest_data = json.loads(manifest.read_text())
source_run = "rwgao_b-brown-university/ambi/xq3zva9u"
require(manifest_data.get("source_run", source_run) == source_run, "Foreign source run in manifest.")
rows = manifest_data["checkpoints"]
steps = [row["step"] for row in rows]
require(bool(steps) and all(type(step) is int and step in STEPS for step in steps)
        and steps == sorted(set(steps)),
        "Expected ordered unique checkpoints on the 100k–1.5M 50k grid.")
expected_steps = _expected_steps(int(os.environ["EXPECTED_MAX_STEP"]))
require(all(step in expected_steps for step in steps), "Checkpoint exceeds expected campaign maximum.")
index = int(os.environ["SLURM_ARRAY_TASK_ID"])
require(0 <= index < len(rows), "Invalid checkpoint array index.")
require(int(os.environ["EPISODES"]) > 0 and 0 < int(os.environ["MAX_STEPS"]) <= 500,
        "Episodes must be positive and max steps must be between 1 and 500.")
row = rows[index]
if os.environ["PRIOR_REFERENCE_ROOT"]:
    reference_root = Path(os.environ["PRIOR_REFERENCE_ROOT"])
    require(reference_root.is_absolute() and (reference_root / f"step_{row['step']}" / "paired.json").is_file(),
            "Missing absolute prior-reference bundle")
checkpoint = Path(row["path"])
require(checkpoint.is_absolute() and checkpoint.is_file(), "Checkpoint must be an existing absolute file.")
require("\n" not in str(checkpoint), "Checkpoint path must not contain a newline.")
digest = hashlib.sha256()
with checkpoint.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
require(digest.hexdigest() == row["sha256"], "Checkpoint content hash differs from the manifest.")
metadata_bytes = Path(str(checkpoint) + ".metadata.json").read_bytes()
metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
if "metadata_sha256" in row:
    require(metadata_sha256 == row["metadata_sha256"], "Checkpoint metadata hash differs from the manifest.")
metadata = json.loads(metadata_bytes)
trial = metadata["trial_run_params"]
params = trial["alg_params"]
env = metadata["experiment_params"]["env_params"]
require(metadata["checkpoint"]["step"] == row["step"], "Checkpoint step differs from the manifest.")
require(trial["alg"] == "TDMPC2/TDMPC2Baseline" and trial["env"] == "DMControl-v0"
        and trial["seed"] == 55, "Expected the seed-55 TD-MPC2 backbone.")
require(env["task"] == "humanoid-walk" and env["obs"] == "state", "Expected Humanoid Walk state observations.")
expected = {"obs": "state", "model_size": 5, "mpc": False, "episodic": False,
            "train_unroll_horizon": 3, "outer_planning_horizon": 3,
            "iterations": 6, "num_samples": 512, "num_elites": 64,
            "num_pi_trajs": 24, "min_std": 0.05, "max_std": 2.0, "temperature": 0.5}
for key, value in expected.items():
    require(params.get(key) == value, f"Saved {key} must equal {value!r}; received {params.get(key)!r}.")
if os.environ.get("EVAL_RUN_MAP"):
    from utils.eval_series import validate_identity
    from utils.eval_series_data import identity_for_tdmpc2_checkpoint
    protocol = {"environment_seeds": list(range(101, 101 + int(os.environ["EPISODES"]))),
                "max_steps": int(os.environ["MAX_STEPS"]), "controller_seed": 12345}
    for controller, run_dir in assignments.items():
        identity = identity_for_tdmpc2_checkpoint(
            {**row, "source_run": source_run}, metadata, controller, protocol,
            {"code_sha": os.environ["SOURCE_COMMIT"]}, path=checkpoint,
            inventory_path=manifest, source_run=source_run,
            warm_start=os.environ["NO_WARM_START"] == "0",
        )
        validate_identity(run_dir, identity)
require(not (root / f"step_{row['step']}" / "paired.json").exists(), "Evaluation output already exists.")
directory = root / f"step_{row['step']}"
directory.mkdir(parents=True, exist_ok=True)
provenance = {"code_sha": os.environ["SOURCE_COMMIT"], "code_tree": os.environ["SOURCE_TREE"],
              "evaluation_source_sha256": evaluation_source_fingerprint(
                  os.environ["SOURCE_COMMIT"], os.environ["SOURCE_TREE"]),
              "source_run": source_run, "campaign": os.environ["CAMPAIGN"],
              "checkpoint_sha256": row["sha256"], "metadata_sha256": metadata_sha256}
target = directory / "provenance.json"
if target.exists():
    require(json.loads(target.read_text()) == provenance, "Existing result provenance does not match this evaluation.")
else:
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, prefix=".provenance-", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(provenance, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, target)
    finally:
        temporary.unlink()
print(row["step"])
print(checkpoint)
PY
)

STEP="${SELECTION%%$'\n'*}"
CHECKPOINT="${SELECTION#*$'\n'}"
OUTPUT="$RESULT_ROOT/step_${STEP}/paired.json"
mkdir -p "$(dirname "$OUTPUT")"
WANDB_LOCAL_ROOT="${SLURM_TMPDIR:-/tmp}/tdmpc2-prior-mppi-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$WANDB_LOCAL_ROOT" "$WANDB_LOCAL_ROOT/cache" "$WANDB_LOCAL_ROOT/data" "$WANDB_LOCAL_ROOT/artifacts"
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-online}" WANDB_DISABLE_CODE=true
export WANDB_DIR="$WANDB_LOCAL_ROOT" WANDB_CACHE_DIR="$WANDB_LOCAL_ROOT/cache"
export WANDB_DATA_DIR="$WANDB_LOCAL_ROOT/data" WANDB_ARTIFACT_DIR="$WANDB_LOCAL_ROOT/artifacts"

echo "Source commit: $SOURCE_COMMIT"
echo "Checkpoint: $CHECKPOINT"
echo "Result: $OUTPUT"
echo "Native MPPI: H3, N512, elites64, policy24, configured6/effective8 iterations"
echo "Previous-plan warm start disabled: $NO_WARM_START"

if [[ "$REQUIRE_CUDA" == 1 ]]; then
  "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; x=torch.ones(2, 2, device="cuda"); assert (x @ x).sum().item() == 8'
fi
evaluation_options=(--device cuda)
[[ "$NO_WARM_START" == 0 ]] || evaluation_options+=(--no-warm-start)
[[ -z "$PRIOR_REFERENCE_ROOT" ]] || evaluation_options+=(--prior-reference "$PRIOR_REFERENCE_ROOT/step_${STEP}/paired.json")

"$PYTHON_BIN" evaluate_tdmpc2_mppi_checkpoint.py "$CHECKPOINT" \
  --output "$OUTPUT" --episodes "$EPISODES" --max-steps "$MAX_STEPS" \
  --seed 101 --controller-seed 12345 --bootstrap-samples 20000 "${evaluation_options[@]}"

staging=("$PYTHON_BIN" publish_tdmpc2_mppi_eval.py "$OUTPUT")
[[ -z "${EVAL_RUN_MAP:-}" ]] || staging+=(--eval-run-map "$EVAL_RUN_MAP")
"${staging[@]}" \
  --project ambi-inner-bench \
  --source-run rwgao_b-brown-university/ambi/xq3zva9u \
  --campaign "$CAMPAIGN" --expected-max-step "$EXPECTED_MAX_STEP"
