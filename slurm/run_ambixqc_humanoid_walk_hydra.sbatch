#!/usr/bin/env bash
#SBATCH --job-name=ambixqc-hwalk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=30-00:00:00
#SBATCH --output=/tmp/ambixqc-humanoid-walk-%j.out
#SBATCH --error=/tmp/ambixqc-humanoid-walk-%j.err

set -Eeuo pipefail

# This runs one seed-55 exploratory AMBI-XQC job. Prefer submitting through
# submit_ambixqc_humanoid_walk_hydra.sh so node selection and durable Slurm
# output are handled consistently.

readonly ACTION_LOCK_SHA="f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6"
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

: "${EXPECTED_ACTION_MODES_SHA:?Export the exact pushed Action Modes commit SHA.}"
: "${AMBIXQC_RESULTS_ROOT:?Export an existing absolute durable result directory.}"
: "${AMBIXQC_RUN_ID:?Export a unique run ID.}"
: "${SLURM_JOB_ID:?This launcher must run inside a Slurm allocation.}"

if [[ ! "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPECTED_ACTION_MODES_SHA must be a full lowercase 40-character SHA." >&2
  exit 1
fi
if [[ "$AMBIXQC_RESULTS_ROOT" != /* || ! -d "$AMBIXQC_RESULTS_ROOT" ]]; then
  echo "AMBIXQC_RESULTS_ROOT must be an existing absolute directory." >&2
  exit 1
fi
if [[ ! -w "$AMBIXQC_RESULTS_ROOT" ]]; then
  echo "AMBIXQC_RESULTS_ROOT is not writable: $AMBIXQC_RESULTS_ROOT" >&2
  exit 1
fi
if [[ ! "$AMBIXQC_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "AMBIXQC_RUN_ID must use only letters, digits, '.', '_', and '-'." >&2
  exit 1
fi
if ((${#AMBIXQC_RUN_ID} > 64)); then
  echo "AMBIXQC_RUN_ID must contain at most 64 characters." >&2
  exit 1
fi

PROJECT_DIR_REAL="$(cd "$PROJECT_DIR" && pwd -P)"
RESULTS_BASE_REAL="$(cd "$AMBIXQC_RESULTS_ROOT" && pwd -P)"
readonly PROJECT_DIR_REAL RESULTS_BASE_REAL
readonly DEFAULT_PYTHON="$PROJECT_DIR_REAL/environments/dmcontrol/.venv/bin/python"
readonly PYTHON="${AMBIXQC_PYTHON:-$DEFAULT_PYTHON}"
readonly MANIFEST="$PROJECT_DIR_REAL/configs/dmcontrol/experiments/ambixqc_humanoid_walk_state.json"
readonly SOURCE_ALG_CONFIG="$PROJECT_DIR_REAL/configs/dmcontrol/algs/ambixqc_humanoid_walk_state.json"
readonly STUDY_ROOT="$RESULTS_BASE_REAL/$AMBIXQC_RUN_ID"
readonly RUN_ROOT="$STUDY_ROOT/seed55-job$SLURM_JOB_ID"
readonly SCRATCH_BASE="${SLURM_TMPDIR:-/tmp}"
readonly JOB_SCRATCH="$SCRATCH_BASE/ambixqc-humanoid-walk-$SLURM_JOB_ID"
readonly EVAL_CSV="$RUN_ROOT/evaluations.csv"
readonly WANDB_LOCAL_ROOT="$JOB_SCRATCH/wandb"
readonly SCRATCH_ALG_DIR="$JOB_SCRATCH/algs"
readonly WANDB_UNIQUE_NAME="AMBIXQC-humanoid-walk-state-1m-j2-n32-g4-seed55-$AMBIXQC_RUN_ID-job$SLURM_JOB_ID"
readonly WANDB_UNIQUE_ID="axqc-${EXPECTED_ACTION_MODES_SHA:0:8}-$SLURM_JOB_ID"

require_clean_sha() {
  local checkout="$1"
  local expected="$2"
  local actual
  actual="$(git -C "$checkout" rev-parse HEAD)"
  if [[ "$actual" != "$expected" ]]; then
    echo "Action Modes commit mismatch: expected $expected, found $actual" >&2
    exit 1
  fi
  if [[ -n "$(git -C "$checkout" status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "Action Modes checkout is not clean: $checkout" >&2
    git -C "$checkout" status --short >&2
    exit 1
  fi
}

require_lock_hash() {
  local lockfile="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$lockfile" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "DMControl lock hash mismatch: expected $expected, found $actual" >&2
    exit 1
  fi
}

validate_evaluations() {
  srun --ntasks=1 "$PYTHON" -c '
import csv
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    if reader.fieldnames != ["step", "reward", "seed"]:
        raise SystemExit(f"unexpected evaluation fields: {reader.fieldnames}")
    rows = list(reader)
# The inherited TD-MPC2 harness evaluates the seeded initial observation
# before collecting a real transition, so its first evaluation is step zero.
expected_steps = [0, *range(100_000, 1_000_001, 100_000)]
if len(rows) != len(expected_steps):
    raise SystemExit(f"evaluation CSV has {len(rows)} rows; expected {len(expected_steps)}")
if [int(row["step"]) for row in rows] != expected_steps:
    raise SystemExit("evaluation steps differ from the one-million-decision protocol")
if any(int(row["seed"]) != 55 for row in rows):
    raise SystemExit("evaluation CSV is not seed 55")
if not all(math.isfinite(float(row["reward"])) for row in rows):
    raise SystemExit("evaluation CSV contains a non-finite return")
print(f"Validated {len(rows)} finite evaluation rows through step {expected_steps[-1]}")
' "$EVAL_CSV"
}

if [[ "$PYTHON" != /* ]]; then
  echo "AMBIXQC_PYTHON must resolve to an absolute path: $PYTHON" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing locked DMControl interpreter: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST" || ! -f "$SOURCE_ALG_CONFIG" ]]; then
  echo "Missing AMBI-XQC Humanoid Walk manifest: $MANIFEST" >&2
  exit 1
fi
case "$RESULTS_BASE_REAL/" in
  "$PROJECT_DIR_REAL/"*)
    echo "Durable results must be outside the Git checkout." >&2
    exit 1
    ;;
esac

require_clean_sha "$PROJECT_DIR_REAL" "$EXPECTED_ACTION_MODES_SHA"
require_lock_hash "$PROJECT_DIR_REAL/environments/dmcontrol/uv.lock" "$ACTION_LOCK_SHA"

if [[ -L "$STUDY_ROOT" ]]; then
  echo "Refusing a symlinked study directory: $STUDY_ROOT" >&2
  exit 1
fi
mkdir -p "$STUDY_ROOT"
if [[ -L "$STUDY_ROOT" ]]; then
  echo "Refusing a symlinked study directory: $STUDY_ROOT" >&2
  exit 1
fi
if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite an existing run directory: $RUN_ROOT" >&2
  exit 1
fi
if [[ -e "$JOB_SCRATCH" || -L "$JOB_SCRATCH" ]]; then
  echo "Refusing to reuse job scratch: $JOB_SCRATCH" >&2
  exit 1
fi
mkdir "$RUN_ROOT"
mkdir "$JOB_SCRATCH"
mkdir -p "$WANDB_LOCAL_ROOT/cache" "$WANDB_LOCAL_ROOT/data" "$WANDB_LOCAL_ROOT/artifacts"
mkdir "$SCRATCH_ALG_DIR"
exec > >(tee "$RUN_ROOT/job.log") 2>&1

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TDMPC2_EVAL_CSV="$EVAL_CSV"
export WANDB_MODE=online
export WANDB_DIR="$WANDB_LOCAL_ROOT"
export WANDB_CACHE_DIR="$WANDB_LOCAL_ROOT/cache"
export WANDB_DATA_DIR="$WANDB_LOCAL_ROOT/data"
export WANDB_ARTIFACT_DIR="$WANDB_LOCAL_ROOT/artifacts"
export WANDB_DISABLE_CODE=true
export WANDB_RUN_ID="$WANDB_UNIQUE_ID"
export WANDB_RESUME=never
unset WANDB_DISABLED

# Make the display name unique without mutating the SHA-guarded checkout.
# The immutable scientific config is copied to job scratch and receives only
# this operational W&B name; main.py still reads the tracked manifest.
"$PYTHON" -c '
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
run_name = sys.argv[3]
config = json.loads(source.read_text(encoding="utf-8"))
config["alg_params"]["wandb_run_name"] = run_name
destination.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
' "$SOURCE_ALG_CONFIG" "$SCRATCH_ALG_DIR/ambixqc_humanoid_walk_state.json" "$WANDB_UNIQUE_NAME"

echo "Job: $SLURM_JOB_ID"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Task: humanoid-walk (state)"
echo "Seed: 55"
echo "Agent decisions: 1000000"
echo "Inner schedule: J2/N32/H3/G4"
echo "Action Modes: $PROJECT_DIR_REAL @ $EXPECTED_ACTION_MODES_SHA"
echo "Python: $PYTHON"
echo "Manifest: $MANIFEST"
echo "Durable results: $RUN_ROOT"
echo "W&B local staging: $WANDB_LOCAL_ROOT"
echo "W&B run name: $WANDB_UNIQUE_NAME"
echo "W&B run ID: $WANDB_UNIQUE_ID (resume=never)"
sha256sum "$PROJECT_DIR_REAL/environments/dmcontrol/uv.lock"
nvidia-smi

cd "$PROJECT_DIR_REAL"
srun --ntasks=1 "$PYTHON" -c \
  'import sys,torch; from importlib.metadata import version; assert torch.cuda.is_available(); assert not torch.is_autocast_enabled(); print(sys.version); print({name:version(name) for name in ("dm-control","gymnasium","mujoco","numpy","torch","wandb")}); print(torch.version.cuda,torch.get_default_dtype(),torch.cuda.get_device_name(0))'

srun --ntasks=1 "$PYTHON" main.py \
  --run "$MANIFEST" \
  --alg-dir "$SCRATCH_ALG_DIR" \
  --log-dir "$RUN_ROOT/training" \
  --alg-index 0 \
  --trial-index 0 \
  --num-runs 1

validate_evaluations
require_clean_sha "$PROJECT_DIR_REAL" "$EXPECTED_ACTION_MODES_SHA"
echo "AMBI-XQC Humanoid Walk seed 55 completed. Results: $RUN_ROOT"
