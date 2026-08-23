#!/usr/bin/env bash
#SBATCH --job-name=xqc-hwalk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=30-00:00:00
#SBATCH --output=/tmp/xqc-humanoid-walk-%j.out
#SBATCH --error=/tmp/xqc-humanoid-walk-%j.err

set -Eeuo pipefail

# One invocation runs exactly one implementation and one seed. Use
# submit_xqc_humanoid_walk_hydra.sh to submit the four comparison jobs.

readonly OFFICIAL_XQC_SHA="9a6832bb742ef01bbe9f1e06153a9338e612dae5"
readonly ACTION_LOCK_SHA="f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6"
readonly OFFICIAL_LOCK_SHA="bda38deffad85326e41382b44e06f2a2fc21396210a3232e17800bbaabf7bf85"
readonly WANDB_PROJECT_NAME="ambi_humanoid"
readonly WANDB_ENTITY_NAME="rwgao_b-brown-university"
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

: "${IMPLEMENTATION:?Export IMPLEMENTATION=official or IMPLEMENTATION=action.}"
: "${SEED:?Export SEED=0 or SEED=1.}"
: "${EXPECTED_ACTION_MODES_SHA:?Export the exact pushed Action Modes commit SHA.}"
: "${XQC_OFFICIAL_DIR:?Export the absolute path to the pinned official XQC checkout.}"
: "${XQC_RESULTS_ROOT:?Export an existing durable result directory.}"
: "${XQC_COMPARISON_ID:?Export one unique comparison ID shared by all four jobs.}"
: "${SLURM_JOB_ID:?This launcher must run inside a Slurm allocation.}"

case "$IMPLEMENTATION" in
  official)
    readonly IMPLEMENTATION_LABEL="official-jax"
    readonly SOURCE_SHA="$OFFICIAL_XQC_SHA"
    ;;
  action)
    readonly IMPLEMENTATION_LABEL="action-pytorch"
    readonly SOURCE_SHA="$EXPECTED_ACTION_MODES_SHA"
    ;;
  *)
    echo "IMPLEMENTATION must be exactly 'official' or 'action'." >&2
    exit 1
    ;;
esac

case "$SEED" in
  0|1) ;;
  *)
    echo "SEED must be exactly 0 or 1." >&2
    exit 1
    ;;
esac

if [[ ! "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPECTED_ACTION_MODES_SHA must be a full lowercase 40-character SHA." >&2
  exit 1
fi
if [[ ! "$XQC_COMPARISON_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "XQC_COMPARISON_ID must use only letters, digits, '.', '_', and '-'." >&2
  exit 1
fi
if [[ "$XQC_RESULTS_ROOT" != /* || ! -d "$XQC_RESULTS_ROOT" ]]; then
  echo "XQC_RESULTS_ROOT must be an existing absolute directory." >&2
  exit 1
fi
if [[ ! -w "$XQC_RESULTS_ROOT" ]]; then
  echo "XQC_RESULTS_ROOT is not writable: $XQC_RESULTS_ROOT" >&2
  exit 1
fi

PROJECT_DIR_REAL="$(cd "$PROJECT_DIR" && pwd -P)"
OFFICIAL_DIR_REAL="$(cd "$XQC_OFFICIAL_DIR" && pwd -P)"
RESULTS_BASE_REAL="$(cd "$XQC_RESULTS_ROOT" && pwd -P)"
readonly PROJECT_DIR_REAL OFFICIAL_DIR_REAL RESULTS_BASE_REAL
readonly ACTION_PYTHON="$PROJECT_DIR_REAL/environments/dmcontrol/.venv/bin/python"
readonly OFFICIAL_PYTHON="$OFFICIAL_DIR_REAL/.venv/bin/python"
readonly ACTION_MANIFEST="$PROJECT_DIR_REAL/configs/dmcontrol/experiments/xqc_humanoid_walk_state.json"
readonly ACTION_ALG_DIR="$PROJECT_DIR_REAL/configs/dmcontrol/algs"
readonly SCRATCH_BASE="${SLURM_TMPDIR:-/tmp}"
readonly JOB_SCRATCH="$SCRATCH_BASE/xqc-humanoid-walk-$IMPLEMENTATION_LABEL-seed$SEED-$SLURM_JOB_ID"
readonly COMPARISON_ROOT="$RESULTS_BASE_REAL/$XQC_COMPARISON_ID"
readonly RUN_ROOT="$COMPARISON_ROOT/$IMPLEMENTATION_LABEL-seed$SEED-job$SLURM_JOB_ID"
readonly RUN_NAME="xqc-$IMPLEMENTATION_LABEL-humanoid-walk-seed$SEED"
readonly METHOD_GROUP="$XQC_COMPARISON_ID-$IMPLEMENTATION_LABEL"

require_clean_sha() {
  local checkout="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(git -C "$checkout" rev-parse HEAD)"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label commit mismatch: expected $expected, found $actual" >&2
    exit 1
  fi
  if [[ -n "$(git -C "$checkout" status --porcelain=v1 --untracked-files=all)" ]]; then
    echo "$label checkout is not clean: $checkout" >&2
    git -C "$checkout" status --short >&2
    exit 1
  fi
}

require_lock_hash() {
  local lockfile="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(sha256sum "$lockfile" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label lock hash mismatch: expected $expected, found $actual" >&2
    exit 1
  fi
}

configure_wandb_storage() {
  local durable_root="$1"
  local scratch_root="$2"
  if [[ -e "$durable_root" || -e "$scratch_root" ]]; then
    echo "Refusing to reuse W&B storage: $durable_root or $scratch_root" >&2
    exit 1
  fi
  mkdir -p "$durable_root/data" "$durable_root/artifacts" "$scratch_root/cache"
  export WANDB_DIR="$durable_root"
  export WANDB_DATA_DIR="$durable_root/data"
  export WANDB_ARTIFACT_DIR="$durable_root/artifacts"
  export WANDB_CACHE_DIR="$scratch_root/cache"
}

validate_action_csv() {
  local csv_path="$1"
  local expected_seed="$2"
  srun --ntasks=1 "$ACTION_PYTHON" -c '
import csv
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_seed = int(sys.argv[2])
with path.open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    if reader.fieldnames != ["step", "reward", "seed"]:
        raise SystemExit(f"unexpected Action evaluation fields: {reader.fieldnames}")
    rows = list(reader)
expected_steps = [1, *range(50_000, 500_001, 50_000)]
if len(rows) != 11:
    raise SystemExit(f"Action evaluation CSV has {len(rows)} rows; expected 11")
if [int(row["step"]) for row in rows] != expected_steps:
    raise SystemExit("Action evaluation steps differ")
if any(int(row["seed"]) != expected_seed for row in rows):
    raise SystemExit(f"Action evaluation CSV is not seed {expected_seed}")
if not all(math.isfinite(float(row["reward"])) for row in rows):
    raise SystemExit("Action evaluation CSV contains a non-finite return")
print(f"Validated Action seed {expected_seed}: {len(rows)} evaluation rows")
' "$csv_path" "$expected_seed"
}

validate_official_csv() {
  local csv_path="$1"
  local expected_seed="$2"
  srun --ntasks=1 "$ACTION_PYTHON" -c '
import csv
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_seed = int(sys.argv[2])
with path.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != 11:
    raise SystemExit(f"official evaluation CSV has {len(rows)} rows; expected 11")
expected_decisions = [1, *range(50_000, 500_001, 50_000)]
expected_raw_frames = [2, *range(100_000, 1_000_001, 100_000)]
expected_paper_frames = list(range(0, 1_000_001, 100_000))
if any(int(row["base_seed"]) != expected_seed for row in rows):
    raise SystemExit("official evaluation base seed differs")
if any(int(row["num_seeds"]) != 1 for row in rows):
    raise SystemExit("official evaluation did not use num_seeds=1")
if any(int(row["seed_index"]) != 0 for row in rows):
    raise SystemExit("official evaluation seed index differs")
if any(int(row["seed"]) != expected_seed for row in rows):
    raise SystemExit("official evaluation seed differs")
if [int(row["evaluation_index"]) for row in rows] != list(range(11)):
    raise SystemExit("official evaluation indices differ")
if [int(row["decision_step"]) for row in rows] != expected_decisions:
    raise SystemExit("official evaluation decision steps differ")
if [int(row["raw_frame"]) for row in rows] != expected_raw_frames:
    raise SystemExit("official evaluation raw-frame steps differ")
if [int(row["paper_raw_frame"]) for row in rows] != expected_paper_frames:
    raise SystemExit("official evaluation paper-frame steps differ")
if any(int(row["action_repeat"]) != 2 for row in rows):
    raise SystemExit("official evaluation action repeat differs")
if any(row["implementation"] != "official-jax" for row in rows):
    raise SystemExit("official evaluation implementation differs")
if any(row["source_commit"] != "9a6832bb742ef01bbe9f1e06153a9338e612dae5" for row in rows):
    raise SystemExit("official evaluation source commit differs")
if not all(math.isfinite(float(row["return"])) for row in rows):
    raise SystemExit("official evaluation contains a non-finite return")
print(f"Validated official XQC seed {expected_seed}: {len(rows)} evaluation rows")
' "$csv_path" "$expected_seed"
}

if [[ ! -x "$ACTION_PYTHON" ]]; then
  echo "Missing locked Action Modes interpreter: $ACTION_PYTHON" >&2
  exit 1
fi
if [[ ! -x "$OFFICIAL_PYTHON" ]]; then
  echo "Missing locked official XQC interpreter: $OFFICIAL_PYTHON" >&2
  exit 1
fi
if [[ ! -f "$ACTION_MANIFEST" ]]; then
  echo "Missing Humanoid Walk manifest: $ACTION_MANIFEST" >&2
  exit 1
fi

require_clean_sha "$PROJECT_DIR_REAL" "$EXPECTED_ACTION_MODES_SHA" "Action Modes"
require_clean_sha "$OFFICIAL_DIR_REAL" "$OFFICIAL_XQC_SHA" "Official XQC"
require_lock_hash "$PROJECT_DIR_REAL/environments/dmcontrol/uv.lock" "$ACTION_LOCK_SHA" "Action Modes"
require_lock_hash "$OFFICIAL_DIR_REAL/uv.lock" "$OFFICIAL_LOCK_SHA" "Official XQC"

if [[ -L "$COMPARISON_ROOT" ]]; then
  echo "Refusing a symlinked comparison result directory: $COMPARISON_ROOT" >&2
  exit 1
fi
mkdir -p "$COMPARISON_ROOT"
if [[ -L "$COMPARISON_ROOT" ]]; then
  echo "Refusing a symlinked comparison result directory: $COMPARISON_ROOT" >&2
  exit 1
fi
if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite existing result directory: $RUN_ROOT" >&2
  exit 1
fi
if [[ -e "$JOB_SCRATCH" ]]; then
  echo "Refusing to reuse existing job scratch: $JOB_SCRATCH" >&2
  exit 1
fi
mkdir "$RUN_ROOT"
mkdir "$JOB_SCRATCH"
exec > >(tee "$RUN_ROOT/job.log") 2>&1

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=online
export WANDB_PROJECT="$WANDB_PROJECT_NAME"
export WANDB_ENTITY="$WANDB_ENTITY_NAME"
export WANDB_RUN_GROUP="$METHOD_GROUP"
export WANDB_NAME="$RUN_NAME"
export WANDB_JOB_TYPE="$IMPLEMENTATION_LABEL"
export WANDB_TAGS="xqc-humanoid-walk-parity,comparison-$XQC_COMPARISON_ID,implementation-$IMPLEMENTATION_LABEL,seed-$SEED"
export WANDB_DISABLE_CODE=true
export XQC_IMPLEMENTATION="$IMPLEMENTATION_LABEL"
export XQC_TASK="humanoid-walk"
export XQC_SOURCE_SHA="$SOURCE_SHA"
export XQC_ACTION_REPEAT=2
unset WANDB_DISABLED WANDB_RUN_ID WANDB_RESUME

configure_wandb_storage "$RUN_ROOT/wandb" "$JOB_SCRATCH/wandb"

echo "Job: $SLURM_JOB_ID"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Implementation: $IMPLEMENTATION_LABEL"
echo "Seed: $SEED"
echo "Run name: $RUN_NAME"
echo "Comparison ID: $XQC_COMPARISON_ID"
echo "W&B method group: $METHOD_GROUP"
echo "W&B: $WANDB_ENTITY_NAME/$WANDB_PROJECT_NAME"
echo "Action Modes: $PROJECT_DIR_REAL @ $EXPECTED_ACTION_MODES_SHA"
echo "Official XQC: $OFFICIAL_DIR_REAL @ $OFFICIAL_XQC_SHA"
echo "Durable results: $RUN_ROOT"
echo "Scratch: $JOB_SCRATCH"
sha256sum "$PROJECT_DIR_REAL/environments/dmcontrol/uv.lock" "$OFFICIAL_DIR_REAL/uv.lock"
nvidia-smi

case "$IMPLEMENTATION" in
  official)
    srun --ntasks=1 "$OFFICIAL_PYTHON" -c \
      'import jax,sys,wandb; from importlib.metadata import version; assert jax.default_backend() == "gpu"; assert not jax.config.x64_enabled; print(sys.version); print({name:version(name) for name in ("dm-control","flax","jax","jaxlib","mujoco","numpy","optax","wandb")}); print(jax.devices())'

    cd "$OFFICIAL_DIR_REAL"
    srun --ntasks=1 "$OFFICIAL_PYTHON" \
      "$PROJECT_DIR_REAL/tests/oracles/run_official_xqc_smoke.py" \
      --official-repo "$OFFICIAL_DIR_REAL" \
      --expected-updates 990000 \
      --max-projection-residual 1e-6 \
      --evaluation-csv "$RUN_ROOT/evaluations.csv" \
      --base-seed "$SEED" \
      --num-seeds 1 \
      --action-repeat 2 \
      --expected-evaluation-rows 11 \
      --canonical-wandb \
      --task humanoid-walk \
      --implementation official-jax \
      --source-sha "$OFFICIAL_XQC_SHA" \
      env=humanoid-walk \
      seed="$SEED" \
      num_seeds=1 \
      start_training=5000 \
      max_steps=1000000 \
      replay_buffer_size=1000000 \
      batch_size=256 \
      updates_per_step=2 \
      n_steps=1 \
      eval_interval=50000 \
      eval_episodes=10 \
      eval_video=false \
      log_interval_condition_number=null \
      wandb.mode=online \
      wandb.project_name="$WANDB_PROJECT_NAME" \
      wandb.entity="$WANDB_ENTITY_NAME" \
      hydra.job.env_set.WANDB_DIR="$RUN_ROOT/wandb" \
      '+hydra.job.env_set.WANDB_RUN_GROUP=${oc.env:WANDB_RUN_GROUP}' \
      '+hydra.job.env_set.WANDB_NAME=${oc.env:WANDB_NAME}' \
      '+hydra.job.env_set.WANDB_JOB_TYPE=${oc.env:WANDB_JOB_TYPE}' \
      '+hydra.job.env_set.WANDB_TAGS=${oc.env:WANDB_TAGS}' \
      "hydra.run.dir=$RUN_ROOT/hydra" \
      hydra.output_subdir=null
    validate_official_csv "$RUN_ROOT/evaluations.csv" "$SEED"
    ;;

  action)
    srun --ntasks=1 "$ACTION_PYTHON" -c \
      'import sys,torch; from importlib.metadata import version; assert torch.cuda.is_available(); assert not torch.is_autocast_enabled(); print(sys.version); print({name:version(name) for name in ("dm-control","gymnasium","mujoco","numpy","torch","wandb")}); print(torch.version.cuda,torch.get_default_dtype(),torch.cuda.get_device_name(0))'

    export XQC_EVAL_CSV="$RUN_ROOT/evaluations.csv"
    cd "$PROJECT_DIR_REAL"
    srun --ntasks=1 "$ACTION_PYTHON" main.py \
      --run "$ACTION_MANIFEST" \
      --alg-dir "$ACTION_ALG_DIR" \
      --log-dir "$RUN_ROOT/action-logs" \
      --trial-index "$SEED" \
      --num-runs 1
    validate_action_csv "$XQC_EVAL_CSV" "$SEED"
    ;;
esac

require_clean_sha "$PROJECT_DIR_REAL" "$EXPECTED_ACTION_MODES_SHA" "Action Modes"
require_clean_sha "$OFFICIAL_DIR_REAL" "$OFFICIAL_XQC_SHA" "Official XQC"
echo "XQC Humanoid Walk $IMPLEMENTATION_LABEL seed $SEED passed. Results: $RUN_ROOT"
