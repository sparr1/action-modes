#!/usr/bin/env bash
#SBATCH --job-name=xqc-humanoid-walk
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

# Submit this file exactly twice from a clean Action Modes checkout. Both jobs
# must receive the same XQC_COMPARISON_GROUP and XQC_RESULTS_ROOT; choose the
# preferred node at submission time rather than baking cluster state into this
# launcher:
#
#   sbatch --nodelist=gpu2301 \
#     --export=ALL,IMPLEMENTATION=official,EXPECTED_ACTION_MODES_SHA="$SHA",XQC_OFFICIAL_DIR="$OFFICIAL",XQC_RESULTS_ROOT="$RESULTS",XQC_COMPARISON_GROUP="$GROUP" \
#     slurm/run_xqc_humanoid_walk_hydra.sbatch
#   sbatch --nodelist=gpu2301 \
#     --export=ALL,IMPLEMENTATION=action,EXPECTED_ACTION_MODES_SHA="$SHA",XQC_OFFICIAL_DIR="$OFFICIAL",XQC_RESULTS_ROOT="$RESULTS",XQC_COMPARISON_GROUP="$GROUP" \
#     slurm/run_xqc_humanoid_walk_hydra.sbatch

readonly OFFICIAL_XQC_SHA="9a6832bb742ef01bbe9f1e06153a9338e612dae5"
readonly ACTION_LOCK_SHA="f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6"
readonly OFFICIAL_LOCK_SHA="bda38deffad85326e41382b44e06f2a2fc21396210a3232e17800bbaabf7bf85"
readonly WANDB_PROJECT_NAME="ambi_humanoid"
readonly WANDB_ENTITY_NAME="rwgao_b-brown-university"
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

: "${IMPLEMENTATION:?Export IMPLEMENTATION=official or IMPLEMENTATION=action.}"
: "${EXPECTED_ACTION_MODES_SHA:?Export the exact pushed Action Modes commit SHA.}"
: "${XQC_OFFICIAL_DIR:?Export the absolute path to the pinned official XQC checkout.}"
: "${XQC_RESULTS_ROOT:?Export an existing durable result directory.}"
: "${XQC_COMPARISON_GROUP:?Export one shared W&B comparison group for both jobs.}"
: "${SLURM_JOB_ID:?This launcher must run inside a Slurm allocation.}"

case "$IMPLEMENTATION" in
  official|action) ;;
  *)
    echo "IMPLEMENTATION must be exactly 'official' or 'action'." >&2
    exit 1
    ;;
esac

if [[ ! "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPECTED_ACTION_MODES_SHA must be a full lowercase 40-character SHA." >&2
  exit 1
fi
if [[ ! "$XQC_COMPARISON_GROUP" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "XQC_COMPARISON_GROUP must use only letters, digits, '.', '_', and '-'." >&2
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
readonly JOB_SCRATCH="$SCRATCH_BASE/xqc-humanoid-walk-$SLURM_JOB_ID"
readonly GROUP_ROOT="$RESULTS_BASE_REAL/$XQC_COMPARISON_GROUP"
readonly RUN_ROOT="$GROUP_ROOT/$IMPLEMENTATION-$SLURM_JOB_ID"

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
steps = [int(row["step"]) for row in rows]
if steps != expected_steps:
    raise SystemExit(f"Action evaluation steps differ: {steps}")
if any(int(row["seed"]) != expected_seed for row in rows):
    raise SystemExit(f"Action evaluation CSV does not contain only seed {expected_seed}")
if not all(math.isfinite(float(row["reward"])) for row in rows):
    raise SystemExit("Action evaluation CSV contains a non-finite return")
print(f"Validated Action seed {expected_seed}: {len(rows)} evaluation rows")
' "$csv_path" "$expected_seed"
}

validate_official_csv() {
  local csv_path="$1"
  srun --ntasks=1 "$ACTION_PYTHON" -c '
import csv
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != 22:
    raise SystemExit(f"official evaluation CSV has {len(rows)} rows; expected 22")
expected_decisions = [1, *range(50_000, 500_001, 50_000)]
expected_raw_frames = [2, *range(100_000, 1_000_001, 100_000)]
expected_paper_frames = list(range(0, 1_000_001, 100_000))
for seed in (0, 1):
    seed_rows = [row for row in rows if int(row["seed"]) == seed]
    if len(seed_rows) != 11:
        raise SystemExit(f"official seed {seed} has {len(seed_rows)} rows; expected 11")
    if [int(row["evaluation_index"]) for row in seed_rows] != list(range(11)):
        raise SystemExit(f"official seed {seed} evaluation indices differ")
    if [int(row["decision_step"]) for row in seed_rows] != expected_decisions:
        raise SystemExit(f"official seed {seed} decision steps differ")
    if [int(row["raw_frame"]) for row in seed_rows] != expected_raw_frames:
        raise SystemExit(f"official seed {seed} raw-frame steps differ")
    if [int(row["paper_raw_frame"]) for row in seed_rows] != expected_paper_frames:
        raise SystemExit(f"official seed {seed} paper-frame steps differ")
    if not all(math.isfinite(float(row["return"])) for row in seed_rows):
        raise SystemExit(f"official seed {seed} contains a non-finite return")
if any(row["implementation"] != "official-jax" for row in rows):
    raise SystemExit("official evaluation CSV has an unexpected implementation")
if any(row["source_commit"] != "9a6832bb742ef01bbe9f1e06153a9338e612dae5" for row in rows):
    raise SystemExit("official evaluation CSV has an unexpected source commit")
print("Validated official XQC: 22 evaluation rows across seeds 0 and 1")
' "$csv_path"
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

if [[ -L "$GROUP_ROOT" ]]; then
  echo "Refusing a symlinked comparison-group result directory: $GROUP_ROOT" >&2
  exit 1
fi
mkdir -p "$GROUP_ROOT"
if [[ -L "$GROUP_ROOT" ]]; then
  echo "Refusing a symlinked comparison-group result directory: $GROUP_ROOT" >&2
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
export WANDB_RUN_GROUP="$XQC_COMPARISON_GROUP"
export WANDB_DISABLE_CODE=true
unset WANDB_DISABLED WANDB_RUN_ID WANDB_RESUME

echo "Job: $SLURM_JOB_ID"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Implementation: $IMPLEMENTATION"
echo "Comparison group: $XQC_COMPARISON_GROUP"
echo "W&B: $WANDB_ENTITY_NAME/$WANDB_PROJECT_NAME"
echo "Action Modes: $PROJECT_DIR_REAL @ $EXPECTED_ACTION_MODES_SHA"
echo "Official XQC: $OFFICIAL_DIR_REAL @ $OFFICIAL_XQC_SHA"
echo "Durable results: $RUN_ROOT"
echo "Scratch: $JOB_SCRATCH"
sha256sum "$PROJECT_DIR_REAL/environments/dmcontrol/uv.lock" "$OFFICIAL_DIR_REAL/uv.lock"
nvidia-smi

case "$IMPLEMENTATION" in
  official)
    export WANDB_TAGS="xqc-humanoid-walk-parity,implementation-official-jax,seeds-0-1"
    export WANDB_JOB_TYPE="official-jax"
    configure_wandb_storage "$RUN_ROOT/wandb" "$JOB_SCRATCH/wandb"
    srun --ntasks=1 "$OFFICIAL_PYTHON" -c \
      'import jax,sys,wandb; from importlib.metadata import version; assert jax.default_backend() == "gpu"; assert not jax.config.x64_enabled; print(sys.version); print({name:version(name) for name in ("dm-control","flax","jax","jaxlib","mujoco","numpy","optax","wandb")}); print(jax.devices())'

    cd "$OFFICIAL_DIR_REAL"
    srun --ntasks=1 "$OFFICIAL_PYTHON" \
      "$PROJECT_DIR_REAL/tests/oracles/run_official_xqc_smoke.py" \
      --official-repo "$OFFICIAL_DIR_REAL" \
      --expected-updates 990000 \
      --max-projection-residual 1e-6 \
      --evaluation-csv "$RUN_ROOT/evaluations.csv" \
      --base-seed 0 \
      --num-seeds 2 \
      --action-repeat 2 \
      --expected-evaluation-rows 22 \
      env=humanoid-walk \
      seed=0 \
      num_seeds=2 \
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
      '+hydra.job.env_set.WANDB_TAGS=${oc.env:WANDB_TAGS}' \
      "hydra.run.dir=$RUN_ROOT/hydra" \
      hydra.output_subdir=null
    validate_official_csv "$RUN_ROOT/evaluations.csv"
    ;;

  action)
    export WANDB_JOB_TYPE="action-pytorch"
    srun --ntasks=1 "$ACTION_PYTHON" -c \
      'import sys,torch; from importlib.metadata import version; assert torch.cuda.is_available(); assert not torch.is_autocast_enabled(); print(sys.version); print({name:version(name) for name in ("dm-control","gymnasium","mujoco","numpy","torch","wandb")}); print(torch.version.cuda,torch.get_default_dtype(),torch.cuda.get_device_name(0))'

    cd "$PROJECT_DIR_REAL"
    for trial_index in 0 1; do
      SEED_ROOT="$RUN_ROOT/seed-$trial_index"
      SEED_SCRATCH="$JOB_SCRATCH/seed-$trial_index"
      if [[ -e "$SEED_ROOT" || -e "$SEED_SCRATCH" ]]; then
        echo "Refusing to overwrite seed $trial_index storage." >&2
        exit 1
      fi
      mkdir "$SEED_ROOT"
      mkdir "$SEED_SCRATCH"
      configure_wandb_storage "$SEED_ROOT/wandb" "$SEED_SCRATCH/wandb"
      export WANDB_TAGS="xqc-humanoid-walk-parity,implementation-action-pytorch,seed-$trial_index"
      export XQC_EVAL_CSV="$SEED_ROOT/evaluations.csv"

      srun --ntasks=1 "$ACTION_PYTHON" main.py \
        --run "$ACTION_MANIFEST" \
        --alg-dir "$ACTION_ALG_DIR" \
        --log-dir "$SEED_ROOT/action-logs" \
        --trial-index "$trial_index" \
        --num-runs 1
      validate_action_csv "$XQC_EVAL_CSV" "$trial_index"
    done
    ;;
esac

require_clean_sha "$PROJECT_DIR_REAL" "$EXPECTED_ACTION_MODES_SHA" "Action Modes"
require_clean_sha "$OFFICIAL_DIR_REAL" "$OFFICIAL_XQC_SHA" "Official XQC"
echo "XQC Humanoid Walk $IMPLEMENTATION job passed. Results: $RUN_ROOT"
