#!/usr/bin/env bash
#SBATCH --job-name=xqc-hwalk-1m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=30-00:00:00
#SBATCH --output=/tmp/xqc-humanoid-walk-1m-%j.out
#SBATCH --error=/tmp/xqc-humanoid-walk-1m-%j.err

set -Eeuo pipefail

# One seed-55 vanilla XQC baseline at the AMBI-XQC one-million-decision
# comparison budget. Submit through submit_xqc_humanoid_walk_1m_hydra.sh.

readonly ACTION_LOCK_SHA="f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6"
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

: "${EXPECTED_ACTION_MODES_SHA:?Export the exact pushed Action Modes commit SHA.}"
: "${XQC_BASELINE_RESULTS_ROOT:?Export an existing absolute durable result directory.}"
: "${XQC_BASELINE_RUN_ID:?Export a unique run ID.}"
: "${XQC_BASELINE_PYTHON:?Export an absolute locked DMControl Python path.}"
: "${SLURM_JOB_ID:?This launcher must run inside a Slurm allocation.}"

if [[ ! "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPECTED_ACTION_MODES_SHA must be a full lowercase 40-character SHA." >&2
  exit 1
fi
if [[ "$XQC_BASELINE_RESULTS_ROOT" != /* || ! -d "$XQC_BASELINE_RESULTS_ROOT" ]]; then
  echo "XQC_BASELINE_RESULTS_ROOT must be an existing absolute directory." >&2
  exit 1
fi
if [[ ! -w "$XQC_BASELINE_RESULTS_ROOT" ]]; then
  echo "XQC_BASELINE_RESULTS_ROOT is not writable: $XQC_BASELINE_RESULTS_ROOT" >&2
  exit 1
fi
if [[ ! "$XQC_BASELINE_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "XQC_BASELINE_RUN_ID must use only letters, digits, '.', '_', and '-'." >&2
  exit 1
fi
if ((${#XQC_BASELINE_RUN_ID} > 64)); then
  echo "XQC_BASELINE_RUN_ID must contain at most 64 characters." >&2
  exit 1
fi
if [[ "$XQC_BASELINE_PYTHON" != /* || ! -x "$XQC_BASELINE_PYTHON" ]]; then
  echo "XQC_BASELINE_PYTHON must be an executable at an absolute path." >&2
  exit 1
fi

PROJECT_DIR_REAL="$(cd "$PROJECT_DIR" && pwd -P)"
RESULTS_BASE_REAL="$(cd "$XQC_BASELINE_RESULTS_ROOT" && pwd -P)"
PYTHON="$(cd "$(dirname "$XQC_BASELINE_PYTHON")" && pwd -P)/$(basename "$XQC_BASELINE_PYTHON")"
ENV_PROJECT_REAL="$(cd "$(dirname "$PYTHON")/../.." && pwd -P)"
readonly PROJECT_DIR_REAL RESULTS_BASE_REAL PYTHON ENV_PROJECT_REAL
readonly MANIFEST="$PROJECT_DIR_REAL/configs/dmcontrol/experiments/xqc_humanoid_walk_state_1m.json"
readonly SOURCE_ALG_CONFIG="$PROJECT_DIR_REAL/configs/dmcontrol/algs/xqc_humanoid_walk_state_1m.json"
readonly STUDY_ROOT="$RESULTS_BASE_REAL/$XQC_BASELINE_RUN_ID"
readonly RUN_ROOT="$STUDY_ROOT/seed55-job$SLURM_JOB_ID"
readonly SCRATCH_BASE="${SLURM_TMPDIR:-/tmp}"
readonly JOB_SCRATCH="$SCRATCH_BASE/xqc-humanoid-walk-1m-$SLURM_JOB_ID"
readonly SCRATCH_ALG_DIR="$JOB_SCRATCH/algs"
readonly WANDB_LOCAL_ROOT="$JOB_SCRATCH/wandb"
readonly EVAL_CSV="$RUN_ROOT/evaluations.csv"
readonly WANDB_UNIQUE_NAME="XQC-vanilla-humanoid-walk-state-1m-seed55-$XQC_BASELINE_RUN_ID-job$SLURM_JOB_ID"
readonly WANDB_UNIQUE_ID="xqc-${EXPECTED_ACTION_MODES_SHA:0:8}-$SLURM_JOB_ID"

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
  local label="$3"
  local actual
  if [[ ! -f "$lockfile" ]]; then
    echo "Missing $label lock: $lockfile" >&2
    exit 1
  fi
  actual="$(sha256sum "$lockfile" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "$label lock hash mismatch: expected $expected, found $actual" >&2
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
expected_steps = [1, *range(50_000, 1_000_001, 50_000)]
if len(rows) != 21:
    raise SystemExit(f"evaluation CSV has {len(rows)} rows; expected 21")
if [int(row["step"]) for row in rows] != expected_steps:
    raise SystemExit("evaluation steps differ from vanilla XQC cadence")
if any(int(row["seed"]) != 55 for row in rows):
    raise SystemExit("evaluation CSV is not seed 55")
if not all(math.isfinite(float(row["reward"])) for row in rows):
    raise SystemExit("evaluation CSV contains a non-finite return")
print(f"Validated {len(rows)} finite evaluation rows through decision {expected_steps[-1]}")
' "$EVAL_CSV"
}

validate_final_checkpoint() {
  XQC_VALIDATE_ROOT="$RUN_ROOT/training" \
    srun --ntasks=1 "$PYTHON" -c '
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

import torch


root = Path(os.environ["XQC_VALIDATE_ROOT"])
latest = [
    path
    for path in root.rglob("*_latest.pt")
    if path.is_file() and not path.name.endswith(".metadata.json")
]
best = [
    path
    for path in root.rglob("*_best.pt")
    if path.is_file() and not path.name.endswith(".metadata.json")
]
if len(latest) != 1:
    raise SystemExit(f"expected one latest checkpoint, found {latest}")
if len(best) != 1:
    raise SystemExit(f"expected one best checkpoint, found {best}")
for path in (*latest, *best):
    if not Path(f"{path}.metadata.json").is_file():
        raise SystemExit(f"missing portable checkpoint sidecar: {path}")

path = latest[0]
metadata = json.loads(Path(f"{path}.metadata.json").read_text(encoding="utf-8"))
checkpoint_metadata = metadata.get("checkpoint", {})
if checkpoint_metadata.get("kind") != "latest" or checkpoint_metadata.get("step") != 1_000_000:
    raise SystemExit(f"latest checkpoint is not the final step: {checkpoint_metadata}")
trial = metadata.get("trial_run_params", {})
if (
    trial.get("alg") != "XQC/XQC"
    or trial.get("seed") != 55
    or trial.get("total_steps") != 1_000_000
):
    raise SystemExit(f"unexpected resolved XQC trial: {trial}")

state = torch.load(path, map_location="cpu", weights_only=False)
if (
    state.get("checkpoint_version") != 1
    or state.get("algorithm") != "XQC"
    or state.get("official_xqc_commit")
    != "9a6832bb742ef01bbe9f1e06153a9338e612dae5"
    or state.get("num_timesteps") != 1_000_000
):
    raise SystemExit("unexpected final XQC checkpoint identity or step")
agent = state.get("agent", {})
expected_updates = 2 * (1_000_000 - 5_000)
expected_delayed = (expected_updates - 1) // 3 + 1
actual_counters = (
    agent.get("update_step"),
    agent.get("actor_optimizer_steps"),
    agent.get("temperature_optimizer_steps"),
)
if actual_counters != (expected_updates, expected_delayed, expected_delayed):
    raise SystemExit(f"unexpected optimizer counters: {actual_counters}")

def optimizer_steps(optimizer_state, expected, name):
    parameter_states = optimizer_state.get("state", {})
    if not parameter_states:
        raise SystemExit(f"{name} optimizer state is empty")
    steps = []
    for parameter_state in parameter_states.values():
        value = parameter_state.get("step")
        if value is None:
            raise SystemExit(f"{name} optimizer state is missing a step")
        steps.append(int(value.item() if torch.is_tensor(value) else value))
    if set(steps) != {expected}:
        raise SystemExit(f"{name} optimizer steps differ: {set(steps)}")

optimizer_steps(agent.get("critic_optimizer", {}), expected_updates, "critic")
optimizer_steps(agent.get("actor_optimizer", {}), expected_delayed, "actor")
optimizer_steps(
    agent.get("temperature_optimizer", {}), expected_delayed, "temperature"
)
normalizer = agent.get("reward_normalizer", {})
normalizer_count = normalizer.get("count")
if normalizer_count != 1_000_000.0:
    raise SystemExit(f"unexpected reward-normalizer count: {normalizer_count}")

def require_finite(value, location="state"):
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all().item()):
            raise SystemExit(f"non-finite tensor at {location}")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            require_finite(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            require_finite(child, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise SystemExit(f"non-finite scalar at {location}")

require_finite(agent, "agent")
require_finite(state.get("metrics", {}), "metrics")

target = agent.get("critic_target", {})
target_means = [value for name, value in target.items() if name.endswith("running_mean")]
target_vars = [value for name, value in target.items() if name.endswith("running_var")]
if len(target_means) != 10 or len(target_vars) != 10:
    raise SystemExit("target critic does not contain the expected 20 BN buffers")
if any(not torch.equal(value, torch.zeros_like(value)) for value in target_means):
    raise SystemExit("target critic running means mutated")
if any(not torch.equal(value, torch.ones_like(value)) for value in target_vars):
    raise SystemExit("target critic running variances mutated")

projected = []
for family in ("actor", "critic"):
    module_state = agent.get(family, {})
    for name, value in module_state.items():
        if name.endswith(".weight") and torch.is_tensor(value) and value.ndim == 2:
            residual = torch.linalg.vector_norm(value.float(), dim=1).sub(1.0).abs().max()
            projected.append((f"{family}.{name}", float(residual)))
if len(projected) != 16:
    raise SystemExit(f"expected 16 projected matrices, found {len(projected)}")
maximum_residual = max(residual for _, residual in projected)
if maximum_residual > 1e-6:
    raise SystemExit(f"projected-row residual {maximum_residual} exceeds 1e-6")
print(
    "Validated final XQC state: "
    f"updates={expected_updates}, delayed={expected_delayed}, "
    f"projection_residual={maximum_residual:.3g}"
)
'
}

case "$RESULTS_BASE_REAL/" in
  "$PROJECT_DIR_REAL/"*)
    echo "Durable results must be outside the Git checkout." >&2
    exit 1
    ;;
esac
if [[ "$PYTHON" != "$ENV_PROJECT_REAL/.venv/bin/python" ]]; then
  echo "XQC_BASELINE_PYTHON must be the locked environment's .venv/bin/python." >&2
  exit 1
fi
if [[ ! -f "$MANIFEST" || ! -f "$SOURCE_ALG_CONFIG" ]]; then
  echo "Missing XQC one-million-decision Humanoid Walk configuration." >&2
  exit 1
fi

require_clean_sha "$PROJECT_DIR_REAL" "$EXPECTED_ACTION_MODES_SHA"
require_lock_hash "$PROJECT_DIR_REAL/environments/dmcontrol/uv.lock" "$ACTION_LOCK_SHA" "Action Modes"
require_lock_hash "$ENV_PROJECT_REAL/uv.lock" "$ACTION_LOCK_SHA" "DMControl environment"

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
mkdir "$SCRATCH_ALG_DIR"
mkdir -p \
  "$JOB_SCRATCH/cache" \
  "$JOB_SCRATCH/matplotlib" \
  "$JOB_SCRATCH/tmp" \
  "$WANDB_LOCAL_ROOT/cache" \
  "$WANDB_LOCAL_ROOT/data" \
  "$WANDB_LOCAL_ROOT/artifacts"
exec > >(tee "$RUN_ROOT/job.log") 2>&1

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export XDG_CACHE_HOME="$JOB_SCRATCH/cache"
export TORCH_HOME="$JOB_SCRATCH/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$JOB_SCRATCH/cache/torchinductor"
export TRITON_CACHE_DIR="$JOB_SCRATCH/cache/triton"
export MPLCONFIGDIR="$JOB_SCRATCH/matplotlib"
export NUMBA_CACHE_DIR="$JOB_SCRATCH/cache/numba"
export TMPDIR="$JOB_SCRATCH/tmp"
export XQC_EVAL_CSV="$EVAL_CSV"
export XQC_IMPLEMENTATION=action-pytorch
export XQC_TASK=humanoid-walk
export XQC_SOURCE_SHA="$EXPECTED_ACTION_MODES_SHA"
export XQC_ACTION_REPEAT=2
export WANDB_MODE=online
export WANDB_ENTITY=rwgao_b-brown-university
export WANDB_PROJECT=ambi
export WANDB_RUN_GROUP=ambixqc-humanoid-walk-state-1m
export WANDB_DIR="$WANDB_LOCAL_ROOT"
export WANDB_CACHE_DIR="$WANDB_LOCAL_ROOT/cache"
export WANDB_DATA_DIR="$WANDB_LOCAL_ROOT/data"
export WANDB_ARTIFACT_DIR="$WANDB_LOCAL_ROOT/artifacts"
export WANDB_DISABLE_CODE=true
export WANDB_JOB_TYPE=vanilla-xqc-baseline
export WANDB_RUN_ID="$WANDB_UNIQUE_ID"
export WANDB_RESUME=never
unset XQC_COMPARISON_ID WANDB_DISABLED WANDB_NAME WANDB_TAGS

# Inject only operational W&B identity into job scratch; the source checkout
# stays clean and the scientific configuration remains SHA-guarded.
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
' "$SOURCE_ALG_CONFIG" "$SCRATCH_ALG_DIR/xqc_humanoid_walk_state_1m.json" "$WANDB_UNIQUE_NAME"

echo "Job: $SLURM_JOB_ID"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Task: humanoid-walk (state)"
echo "Seed: 55"
echo "Agent decisions: 1000000"
echo "Raw control frames: 2000000"
echo "Learner: vanilla XQC, UTD 2, warmup 5000 decisions"
echo "Action Modes: $PROJECT_DIR_REAL @ $EXPECTED_ACTION_MODES_SHA"
echo "Python: $PYTHON"
echo "DMControl environment: $ENV_PROJECT_REAL"
echo "Manifest: $MANIFEST"
echo "Durable results: $RUN_ROOT"
echo "W&B local staging: $WANDB_LOCAL_ROOT"
echo "W&B run name: $WANDB_UNIQUE_NAME"
echo "W&B run ID: $WANDB_UNIQUE_ID (resume=never)"
echo "W&B: $WANDB_ENTITY/$WANDB_PROJECT group=$WANDB_RUN_GROUP"
sha256sum \
  "$PROJECT_DIR_REAL/environments/dmcontrol/uv.lock" \
  "$ENV_PROJECT_REAL/uv.lock"
nvidia-smi

cd "$PROJECT_DIR_REAL"
srun --ntasks=1 "$PYTHON" -c \
  'import sys,torch; from importlib.metadata import version; assert sys.version_info[:2] == (3,10); assert torch.cuda.is_available(); assert torch.get_default_dtype() == torch.float32; assert not torch.is_autocast_enabled(); print(sys.version); print({name:version(name) for name in ("dm-control","gymnasium","mujoco","numpy","torch","wandb")}); print(torch.version.cuda,torch.cuda.get_device_name(0))'
srun --ntasks=1 "$PYTHON" -c \
  'import gymnasium as gym,numpy as np,domains; env=gym.make("DMControl-v0",task="humanoid-walk",obs="state",render_mode=None); error=None
try:
 obs,_=env.reset(seed=55); assert obs.shape == (67,) and np.isfinite(obs).all(); assert env.action_space.shape == (21,); assert env.unwrapped.action_repeat == 2; assert np.isfinite(env.action_space.low).all() and np.isfinite(env.action_space.high).all(); print("DMControl probe",obs.shape,env.action_space.shape,env.unwrapped.action_repeat)
finally:
 env.close()'

srun --ntasks=1 "$PYTHON" main.py \
  --run "$MANIFEST" \
  --alg-dir "$SCRATCH_ALG_DIR" \
  --log-dir "$RUN_ROOT/training" \
  --alg-index 0 \
  --trial-index 0 \
  --num-runs 1

validate_evaluations
validate_final_checkpoint
require_clean_sha "$PROJECT_DIR_REAL" "$EXPECTED_ACTION_MODES_SHA"
echo "Vanilla XQC Humanoid Walk seed 55 completed. Results: $RUN_ROOT"
