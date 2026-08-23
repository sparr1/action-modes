#!/usr/bin/env bash
#SBATCH --job-name=ambixqc-gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/tmp/ambixqc-cuda-gate-%j.out
#SBATCH --error=/tmp/ambixqc-cuda-gate-%j.err

set -Eeuo pipefail

# Submit this only from a clean, pushed AMBI-XQC checkout. The environment may
# come from another clean Action Modes checkout when its lock is byte-identical:
#
#   EXPECTED_ACTION_MODES_SHA="$(git rev-parse HEAD)" \
#   AMBIXQC_GATE_ROOT=/absolute/durable/path \
#   AMBIXQC_ENV_PROJECT=/absolute/path/to/environments/dmcontrol \
#     sbatch --export=ALL,EXPECTED_ACTION_MODES_SHA,AMBIXQC_GATE_ROOT,AMBIXQC_ENV_PROJECT \
#       slurm/run_ambixqc_cuda_gate_hydra.sbatch

readonly ACTION_LOCK_SHA="f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6"
readonly OFFICIAL_XQC_SHA="9a6832bb742ef01bbe9f1e06153a9338e612dae5"
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

: "${EXPECTED_ACTION_MODES_SHA:?Export the exact pushed Action Modes commit SHA.}"
: "${AMBIXQC_GATE_ROOT:?Export an existing absolute durable gate-artifact directory.}"
: "${SLURM_JOB_ID:?This launcher must run inside a Slurm allocation.}"

if [[ ! "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPECTED_ACTION_MODES_SHA must be a full lowercase 40-character SHA." >&2
  exit 1
fi
if [[ "$AMBIXQC_GATE_ROOT" != /* || ! -d "$AMBIXQC_GATE_ROOT" ]]; then
  echo "AMBIXQC_GATE_ROOT must be an existing absolute directory." >&2
  exit 1
fi
if [[ ! -w "$AMBIXQC_GATE_ROOT" ]]; then
  echo "AMBIXQC_GATE_ROOT is not writable: $AMBIXQC_GATE_ROOT" >&2
  exit 1
fi

PROJECT_DIR_REAL="$(cd "$PROJECT_DIR" && pwd -P)"
GATE_ROOT_REAL="$(cd "$AMBIXQC_GATE_ROOT" && pwd -P)"
ENV_PROJECT_REAL="$(cd "${AMBIXQC_ENV_PROJECT:-$PROJECT_DIR_REAL/environments/dmcontrol}" && pwd -P)"
readonly PROJECT_DIR_REAL GATE_ROOT_REAL ENV_PROJECT_REAL
readonly PYTHON="$ENV_PROJECT_REAL/.venv/bin/python"
readonly MANIFEST="$PROJECT_DIR_REAL/configs/dmcontrol/experiments/ambixqc_humanoid_walk_state_smoke.json"
readonly ALG_DIR="$PROJECT_DIR_REAL/configs/dmcontrol/algs"
readonly RUN_ROOT="$GATE_ROOT_REAL/job-$SLURM_JOB_ID"
readonly SCRATCH_BASE="${SLURM_TMPDIR:-/tmp}"
readonly JOB_SCRATCH="$SCRATCH_BASE/ambixqc-cuda-gate-$SLURM_JOB_ID"
readonly SMOKE_ROOT="$RUN_ROOT/smoke"

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
  if [[ ! -f "$lockfile" ]]; then
    echo "Missing DMControl lock: $lockfile" >&2
    exit 1
  fi
  actual="$(sha256sum "$lockfile" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "DMControl lock hash mismatch: expected $expected, found $actual" >&2
    exit 1
  fi
}

case "$GATE_ROOT_REAL/" in
  "$PROJECT_DIR_REAL/"*)
    echo "Gate artifacts must be outside the Git checkout." >&2
    exit 1
    ;;
esac
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing locked DMControl interpreter: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing AMBI-XQC Humanoid Walk smoke manifest: $MANIFEST" >&2
  exit 1
fi

require_clean_sha "$PROJECT_DIR_REAL" "$EXPECTED_ACTION_MODES_SHA"
require_lock_hash "$PROJECT_DIR_REAL/environments/dmcontrol/uv.lock" "$ACTION_LOCK_SHA"
require_lock_hash "$ENV_PROJECT_REAL/uv.lock" "$ACTION_LOCK_SHA"

if [[ -e "$RUN_ROOT" || -L "$RUN_ROOT" ]]; then
  echo "Refusing to overwrite existing gate artifacts: $RUN_ROOT" >&2
  exit 1
fi
if [[ -e "$JOB_SCRATCH" || -L "$JOB_SCRATCH" ]]; then
  echo "Refusing to reuse existing job scratch: $JOB_SCRATCH" >&2
  exit 1
fi
mkdir "$RUN_ROOT"
mkdir "$JOB_SCRATCH"
mkdir -p \
  "$JOB_SCRATCH/cache" \
  "$JOB_SCRATCH/matplotlib" \
  "$JOB_SCRATCH/tmp" \
  "$JOB_SCRATCH/wandb/cache" \
  "$JOB_SCRATCH/wandb/data" \
  "$JOB_SCRATCH/wandb/artifacts"
exec > >(tee "$RUN_ROOT/gate.log") 2>&1

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export XDG_CACHE_HOME="$JOB_SCRATCH/cache"
export TORCH_HOME="$JOB_SCRATCH/cache/torch"
export MPLCONFIGDIR="$JOB_SCRATCH/matplotlib"
export NUMBA_CACHE_DIR="$JOB_SCRATCH/cache/numba"
export TMPDIR="$JOB_SCRATCH/tmp"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_DIR="$JOB_SCRATCH/wandb"
export WANDB_CACHE_DIR="$JOB_SCRATCH/wandb/cache"
export WANDB_DATA_DIR="$JOB_SCRATCH/wandb/data"
export WANDB_ARTIFACT_DIR="$JOB_SCRATCH/wandb/artifacts"
export WANDB_DISABLE_CODE=true
unset WANDB_RUN_ID WANDB_RESUME

echo "Job: $SLURM_JOB_ID"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Action Modes: $PROJECT_DIR_REAL @ $EXPECTED_ACTION_MODES_SHA"
echo "DMControl environment: $ENV_PROJECT_REAL"
echo "Smoke: humanoid-walk state, 502 decisions"
echo "Gate artifacts: $RUN_ROOT"
git -C "$PROJECT_DIR_REAL" show --no-patch --format=fuller HEAD
sha256sum \
  "$PROJECT_DIR_REAL/environments/dmcontrol/uv.lock" \
  "$ENV_PROJECT_REAL/uv.lock"
nvidia-smi

cd "$PROJECT_DIR_REAL"
srun --ntasks=1 "$PYTHON" -c \
  'import sys, torch; from importlib.metadata import version; assert torch.cuda.is_available(); assert torch.get_default_dtype() == torch.float32; assert not torch.is_autocast_enabled(); print(sys.version); print({name: version(name) for name in ("dm-control", "gymnasium", "mujoco", "numpy", "torch", "wandb")}); print(torch.version.cuda, torch.cuda.get_device_name(0))'
srun --ntasks=1 "$PYTHON" -c \
  'import gymnasium as gym, numpy as np, domains; env = gym.make("DMControl-v0", task="humanoid-walk", obs="state", render_mode=None); obs, _ = env.reset(seed=1); assert obs.ndim == 1 and np.isfinite(obs).all(); assert env.action_space.shape and np.isfinite(env.action_space.low).all() and np.isfinite(env.action_space.high).all(); print("DMControl probe", obs.shape, obs.dtype, env.action_space.shape); env.close()'

# The first test is the dedicated live-CUDA projection/EMA regression for the
# cached-parameter bug found after the local CPU suite passed.
srun --ntasks=1 "$PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_ambixqc_controller.py::test_cuda_workspace_projects_live_weights_and_moves_the_live_target

srun --ntasks=1 "$PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_ambixqc_hydra_gate_launcher.py \
  tests/test_ambixqc_humanoid_walk_config.py \
  tests/test_ambixqc_controller.py \
  tests/test_ambixqc_core.py \
  tests/test_ambixqc_inner.py \
  tests/test_ambixqc_wrapper.py \
  tests/test_render_checkpoint.py \
  tests/test_tdmpc2_correctness.py \
  tests/test_tdmpc2_checkpoint.py \
  tests/test_tdmpc2_training_state.py

AMBI_XQC_TEST_DEVICE=cuda AMBI_XQC_REQUIRE_CUDA=1 \
  srun --ntasks=1 "$PYTHON" -m pytest -q -p no:cacheprovider \
    tests/test_xqc_correctness.py \
    tests/test_xqc_integration.py

srun --ntasks=1 "$PYTHON" main.py \
  --run "$MANIFEST" \
  --alg-dir "$ALG_DIR" \
  --log-dir "$SMOKE_ROOT" \
  --alg-index 0 \
  --trial-index 0 \
  --num-runs 1

SMOKE_ROOT="$SMOKE_ROOT" OFFICIAL_XQC_SHA="$OFFICIAL_XQC_SHA" \
  srun --ntasks=1 "$PYTHON" -c '
import json
import math
import os
from pathlib import Path

import numpy as np
import torch


root = Path(os.environ["SMOKE_ROOT"])
paths = [
    path
    for path in root.rglob("*_latest")
    if path.is_file() and not path.name.endswith(".metadata.json")
]
if len(paths) != 1:
    raise SystemExit(f"expected one final latest checkpoint, found {paths}")
path = paths[0]
metadata_path = Path(f"{path}.metadata.json")
if not metadata_path.is_file():
    raise SystemExit(f"missing checkpoint metadata: {metadata_path}")
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
checkpoint = metadata.get("checkpoint", {})
if checkpoint.get("kind") != "latest" or checkpoint.get("step") != 502:
    raise SystemExit(f"latest checkpoint is not the final step-502 state: {checkpoint}")
trial = metadata.get("trial_run_params", {})
if (
    trial.get("alg") != "AMBIXQC/AMBIXQC"
    or trial.get("total_steps") != 502
    or trial.get("seed") != 1
):
    raise SystemExit(f"unexpected resolved smoke trial: {trial}")

state = torch.load(path, map_location="cpu", weights_only=False)
expected_fields = {
    "checkpoint_version",
    "semantic_signature",
    "module",
    "world_optimizer",
    "xqc_workspace",
    "reward_normalizer",
    "outer_generator",
    "num_updates",
    "outer_version",
    "inner",
}
if set(state) != expected_fields or state["checkpoint_version"] != 1:
    raise SystemExit(f"unexpected AMBI-XQC checkpoint schema: {set(state)}")
signature = state["semantic_signature"]
if (
    signature.get("algorithm") != "AMBIXQC"
    or signature.get("official_xqc_commit") != os.environ["OFFICIAL_XQC_SHA"]
    or signature.get("observation", {}).get("mode") != "state"
):
    raise SystemExit(f"unexpected AMBI-XQC semantics: {signature}")

workspace = state["xqc_workspace"]
actual_counters = (
    state["num_updates"],
    state["outer_version"],
    workspace.get("update_step"),
    workspace.get("actor_optimizer_steps"),
    workspace.get("temperature_optimizer_steps"),
    state["inner"].get("action_index"),
)
expected_counters = (2, 2, 2, 1, 1, 1)
if actual_counters != expected_counters:
    raise SystemExit(
        f"unexpected update/action counters: {actual_counters}, expected {expected_counters}"
    )
reward_state = state["reward_normalizer"]
if reward_state.get("count") != 502.0:
    raise SystemExit(
        f"reward normalizer did not see all real transitions: {reward_state}"
    )

def assert_finite(value, location="state"):
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all()
        ):
            raise SystemExit(f"non-finite tensor at {location}")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.inexact) and not bool(np.isfinite(value).all()):
            raise SystemExit(f"non-finite array at {location}")
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise SystemExit(f"non-finite scalar at {location}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{location}[{index}]")

assert_finite(state)
module = state["module"]
projected = {
    key: value
    for key, value in module.items()
    if (
        key.startswith("xqc_controller.actor.")
        or key.startswith("xqc_controller.critic.")
    )
    and key.endswith(
        (".linear.weight", ".mean.weight", ".log_std.weight", ".value.weight")
    )
}
if len(projected) != 16:
    raise SystemExit(f"expected 16 projected online weights, found {sorted(projected)}")
residual = max(
    float((torch.linalg.vector_norm(weight, dim=1) - 1.0).abs().max())
    for weight in projected.values()
)
if residual > 1e-6:
    raise SystemExit(f"projected-row residual {residual} exceeds 1e-6")
temperature = module.get("xqc_controller.log_temperature")
if temperature is None or not bool(torch.isfinite(temperature).all()) or not bool(
    temperature.exp() > 0
):
    raise SystemExit("learned XQC temperature is not positive and finite")
print(
    "Validated final AMBI-XQC smoke checkpoint",
    path,
    "counters=",
    actual_counters,
    "projection_residual=",
    residual,
)
'

require_clean_sha "$PROJECT_DIR_REAL" "$EXPECTED_ACTION_MODES_SHA"
printf 'PASS\n' > "$RUN_ROOT/PASS"
echo "AMBI-XQC Hydra CUDA gate passed. Artifacts: $RUN_ROOT"
