#!/usr/bin/env bash
#SBATCH --job-name=ambi-hw-noinner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpus
#SBATCH --nodelist=gpu2201
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=30-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/tmp/ambi-hw-noinner-%j.out
#SBATCH --error=/tmp/ambi-hw-noinner-%j.err

set -Eeuo pipefail
umask 077

readonly ACTION_LOCK_SHA="f123ba99aadde092401c0e912dbeb88994f00ae420680c69c18003965485efe6"
readonly EXPECTED_NODE="gpu2201"
readonly CONFIG_STEM="ambi_humanoid_walk_outer_prior_no_inner_d512_2_14m"
readonly MIN_DURABLE_FREE_KIB=$((2 * 1024 * 1024))
readonly MIN_SCRATCH_FREE_KIB=$((4 * 1024 * 1024))
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

: "${EXPECTED_ACTION_MODES_SHA:?Export the exact pushed Action Modes commit SHA.}"
: "${AMBI_NO_INNER_RESULTS_ROOT:?Export an existing absolute durable result directory.}"
: "${AMBI_NO_INNER_RUN_ID:?Export a unique run ID.}"
: "${AMBI_NO_INNER_PYTHON:?Export an absolute locked DMControl Python path.}"
: "${SLURM_JOB_ID:?This launcher must run inside a Slurm allocation.}"

fail() {
  echo "AMBI no-inner Hydra error: $*" >&2
  exit 2
}

require_free_kib() {
  local path="$1"
  local minimum="$2"
  local label="$3"
  local available
  available="$(df -Pk "$path" | awk 'NR == 2 {print $4}')"
  [[ "$available" =~ ^[0-9]+$ ]] || fail \
    "could not determine free space for $label: $path"
  (( available >= minimum )) || fail \
    "$label has less than $((minimum / 1024 / 1024)) GiB free: $path"
}

[[ "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]] || fail \
  "EXPECTED_ACTION_MODES_SHA must be a full lowercase 40-character SHA"
[[ "$AMBI_NO_INNER_RESULTS_ROOT" == /* ]] || fail \
  "AMBI_NO_INNER_RESULTS_ROOT must be absolute"
[[ -d "$AMBI_NO_INNER_RESULTS_ROOT" ]] || fail \
  "AMBI_NO_INNER_RESULTS_ROOT must already exist"
[[ ! -L "$AMBI_NO_INNER_RESULTS_ROOT" ]] || fail \
  "AMBI_NO_INNER_RESULTS_ROOT must not be a symlink"
[[ -w "$AMBI_NO_INNER_RESULTS_ROOT" ]] || fail \
  "AMBI_NO_INNER_RESULTS_ROOT is not writable"
[[ "$AMBI_NO_INNER_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail \
  "AMBI_NO_INNER_RUN_ID must be filesystem-safe"
(( ${#AMBI_NO_INNER_RUN_ID} <= 64 )) || fail \
  "AMBI_NO_INNER_RUN_ID must contain at most 64 characters"
[[ "$AMBI_NO_INNER_PYTHON" == /* && -x "$AMBI_NO_INNER_PYTHON" ]] || fail \
  "AMBI_NO_INNER_PYTHON must be an executable absolute path"
[[ "${SLURM_NODELIST:-}" == "$EXPECTED_NODE" ]] || fail \
  "launcher must run on $EXPECTED_NODE, found ${SLURM_NODELIST:-unset}"

for command in git sha256sum df awk nvidia-smi srun tee; do
  command -v "$command" >/dev/null 2>&1 || fail "missing required command: $command"
done

PROJECT_DIR_REAL="$(cd "$PROJECT_DIR" && pwd -P)"
RESULTS_ROOT_REAL="$(cd "$AMBI_NO_INNER_RESULTS_ROOT" && pwd -P)"
PYTHON="$(cd "$(dirname "$AMBI_NO_INNER_PYTHON")" && pwd -P)/$(basename "$AMBI_NO_INNER_PYTHON")"
ENV_PROJECT_REAL="$(cd "$(dirname "$PYTHON")/../.." && pwd -P)"
readonly PROJECT_DIR_REAL RESULTS_ROOT_REAL PYTHON ENV_PROJECT_REAL
readonly MANIFEST="$PROJECT_DIR_REAL/configs/dmcontrol/experiments/$CONFIG_STEM.json"
readonly ALGORITHM_CONFIG="$PROJECT_DIR_REAL/configs/dmcontrol/algs/$CONFIG_STEM.json"
readonly PARENT_CONFIG="$PROJECT_DIR_REAL/configs/dmcontrol/algs/ambi_humanoid_walk_base_v2_d512_2.json"
readonly STUDY_ROOT="$RESULTS_ROOT_REAL/$AMBI_NO_INNER_RUN_ID"
readonly RUN_ROOT="$STUDY_ROOT/training"
readonly SCRATCH_BASE="${SLURM_TMPDIR:-/tmp}"
readonly JOB_SCRATCH="$SCRATCH_BASE/ambi-no-inner-$SLURM_JOB_ID"
readonly WANDB_LOCAL_ROOT="$JOB_SCRATCH/wandb"

cd "$PROJECT_DIR_REAL"

[[ "$PYTHON" == "$ENV_PROJECT_REAL/.venv/bin/python" ]] || fail \
  "AMBI_NO_INNER_PYTHON must point to a locked environment's .venv/bin/python"
[[ -f "$MANIFEST" && -f "$ALGORITHM_CONFIG" && -f "$PARENT_CONFIG" ]] || fail \
  "allowlisted manifest, algorithm configuration, or parent is missing"
case "$RESULTS_ROOT_REAL/" in
  "$PROJECT_DIR_REAL/"*) fail "durable results must be outside the Git checkout" ;;
esac
[[ ! -e "$STUDY_ROOT" && ! -L "$STUDY_ROOT" ]] || fail \
  "refusing to reuse study artifacts: $STUDY_ROOT"
[[ -d "$SCRATCH_BASE" && -w "$SCRATCH_BASE" ]] || fail \
  "node-local scratch base is not writable: $SCRATCH_BASE"
[[ ! -e "$JOB_SCRATCH" && ! -L "$JOB_SCRATCH" ]] || fail \
  "refusing to reuse node-local scratch: $JOB_SCRATCH"

actual_sha="$(git -C "$PROJECT_DIR_REAL" rev-parse HEAD)"
[[ "$actual_sha" == "$EXPECTED_ACTION_MODES_SHA" ]] || fail \
  "commit mismatch: expected $EXPECTED_ACTION_MODES_SHA, found $actual_sha"
if [[ -n "$(git -C "$PROJECT_DIR_REAL" status --porcelain=v1 --untracked-files=all)" ]]; then
  git -C "$PROJECT_DIR_REAL" status --short >&2
  fail "Action Modes checkout must be clean"
fi

for lockfile in \
  "$PROJECT_DIR_REAL/environments/dmcontrol/uv.lock" \
  "$ENV_PROJECT_REAL/uv.lock"; do
  [[ -f "$lockfile" ]] || fail "missing DMControl lock: $lockfile"
  actual_lock_sha="$(sha256sum "$lockfile" | awk '{print $1}')"
  [[ "$actual_lock_sha" == "$ACTION_LOCK_SHA" ]] || fail \
    "DMControl lock mismatch for $lockfile"
done

require_free_kib "$RESULTS_ROOT_REAL" "$MIN_DURABLE_FREE_KIB" \
  "durable result filesystem"
require_free_kib "$SCRATCH_BASE" "$MIN_SCRATCH_FREE_KIB" \
  "node-local scratch filesystem"

# Fail before creating artifacts if the scientific contract drifted.
"$PYTHON" -c '
import copy
import json
import sys
from pathlib import Path

import gymnasium as gym

from RL.AMBITDMPC2 import AMBITDMPC2


def load(path):
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SystemExit(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=reject_duplicates)


config = load(sys.argv[1])
manifest = load(sys.argv[2])
parent = load(sys.argv[3])
stem = sys.argv[4]
params = config.get("alg_params", {})
expected = copy.deepcopy(parent)
expected_params = expected["alg_params"]
expected_params.update(
    {
        "inner_operator": "none",
        "inner_rounds": None,
        "inner_rollouts_per_round": None,
        "inner_temperature_mode": "inherit_outer",
        "wandb_run_name": params.get("wandb_run_name"),
        "wandb_tags": params.get("wandb_tags"),
    }
)
expected_params.pop("inner_critic_updates_per_round")
expected_params.pop("inner_actor_updates_per_round")
if config != expected:
    raise SystemExit("no-inner config is not an exact scientific D512-2 derivative")
expected_top = {
    "seed": 55,
    "env": "DMControl-v0",
    "alg": "AMBITDMPC2/AMBITDMPC2",
    "device": "cuda",
    "total_steps": 14_000_000,
    "episodes": None,
}
if any(config.get(key) != value for key, value in expected_top.items()):
    raise SystemExit("unexpected top-level no-inner algorithm contract")
required = {
    "obs": "state",
    "model_size": 5,
    "utd": 1,
    "compile": True,
    "compile_strict": False,
    "mpc": False,
    "eval_freq": None,
    "inner_operator": "none",
    "inner_rounds": None,
    "inner_rollouts_per_round": None,
    "inner_temperature_mode": "inherit_outer",
    "inner_execution_action": "policy_sample",
    "outer_policy_episode_probability": 0.0,
    "inner_batch_size": 512,
    "inner_replay_capacity": 12_288,
    "wandb": True,
    "wandb_mode": "online",
}
if {key: params.get(key) for key in required} != required:
    raise SystemExit("unexpected D512-2-derived no-inner parameter contract")
for forbidden in ("inner_critic_updates_per_round", "inner_actor_updates_per_round"):
    if forbidden in params:
        raise SystemExit(f"forbidden active inner schedule field: {forbidden}")
if manifest.get("configs") != [stem] or manifest.get("trials") != 1:
    raise SystemExit("manifest must select exactly one seed-55 no-inner cell")
expected_manifest = {
    "seed": 55,
    "device": "cuda",
    "env": "DMControl-v0",
    "total_steps": 14_000_000,
    "episodes": None,
}
if manifest.get("overrides_alg") != expected_manifest:
    raise SystemExit("unexpected no-inner manifest overrides")
if manifest.get("env_params") != {
    "task": "humanoid-walk",
    "obs": "state",
    "render_mode": None,
}:
    raise SystemExit("unexpected no-inner environment contract")
if (
    manifest.get("logs") != "timestamp"
    or manifest.get("save_trials") != "none"
    or manifest.get("checkpoint_every") is not None
    or manifest.get("save_strat") != "none"
):
    raise SystemExit("unexpected no-inner artifact policy")

algorithm = object.__new__(AMBITDMPC2)
algorithm.env = gym.make("Pendulum-v1", max_episode_steps=5)
algorithm.run_params = {
    "seed": 55,
    "device": "cpu",
    "env": "launch-preflight",
    "total_steps": 14_000_000,
}
algorithm.custom_params = params
try:
    cfg = algorithm._build_cfg({"device": "cpu", **params})
finally:
    algorithm.env.close()
resolved_zero = {
    "inner_rounds": 0,
    "inner_rollouts_per_round": 0,
    "inner_updates_per_round": 0,
    "inner_mppi_iterations": 0,
    "inner_model_step_budget": 0,
    "inner_expected_update_slots": 0,
    "inner_critic_updates_per_action": 0,
    "inner_actor_updates_per_action": 0,
    "inner_temperature_updates_per_action": 0,
}
if cfg.inner_operator != "none" or cfg.inner_component_update_schedule:
    raise SystemExit("resolved controller still selects an inner operator")
if any(getattr(cfg, key) != value for key, value in resolved_zero.items()):
    raise SystemExit("resolved controller contains nonzero inner work")
if (
    cfg.inner_execution_action != "policy_sample"
    or cfg.inner_execution_std_scale != 1.0
    or cfg.inner_execution_noise_std != 0.0
):
    raise SystemExit("resolved controller does not sample the outer policy")
' "$ALGORITHM_CONFIG" "$MANIFEST" "$PARENT_CONFIG" "$CONFIG_STEM"

mkdir "$STUDY_ROOT"
mkdir "$JOB_SCRATCH"
mkdir -p \
  "$JOB_SCRATCH/cache/torch" \
  "$JOB_SCRATCH/cache/torchinductor" \
  "$JOB_SCRATCH/cache/triton" \
  "$JOB_SCRATCH/cache/cuda" \
  "$JOB_SCRATCH/cache/numba" \
  "$JOB_SCRATCH/matplotlib" \
  "$JOB_SCRATCH/tmp" \
  "$WANDB_LOCAL_ROOT/cache" \
  "$WANDB_LOCAL_ROOT/data" \
  "$WANDB_LOCAL_ROOT/artifacts"
exec > >(tee "$STUDY_ROOT/job.log") 2>&1

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export XDG_CACHE_HOME="$JOB_SCRATCH/cache"
export TORCH_HOME="$JOB_SCRATCH/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$JOB_SCRATCH/cache/torchinductor"
export TRITON_CACHE_DIR="$JOB_SCRATCH/cache/triton"
export CUDA_CACHE_PATH="$JOB_SCRATCH/cache/cuda"
export NUMBA_CACHE_DIR="$JOB_SCRATCH/cache/numba"
export MPLCONFIGDIR="$JOB_SCRATCH/matplotlib"
export TMPDIR="$JOB_SCRATCH/tmp"
export WANDB_MODE=online
export WANDB_ENTITY=rwgao_b-brown-university
export WANDB_PROJECT=ambi
export WANDB_DIR="$WANDB_LOCAL_ROOT"
export WANDB_CACHE_DIR="$WANDB_LOCAL_ROOT/cache"
export WANDB_DATA_DIR="$WANDB_LOCAL_ROOT/data"
export WANDB_ARTIFACT_DIR="$WANDB_LOCAL_ROOT/artifacts"
export WANDB_DISABLE_CODE=true
export WANDB_RUN_ID="ani-${EXPECTED_ACTION_MODES_SHA:0:8}-${SLURM_JOB_ID}"
export WANDB_RESUME=never
unset TDMPC2_EVAL_CSV XQC_EVAL_CSV XQC_COMPARISON_ID
unset WANDB_DISABLED WANDB_NAME WANDB_TAGS WANDB_RUN_GROUP WANDB_JOB_TYPE
unset TORCHDYNAMO_DISABLE CUDA_LAUNCH_BLOCKING PYTORCH_NO_CUDA_MEMORY_CACHING
unset TORCH_COMPILE_DEBUG TORCH_LOGS

echo "Job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Task: humanoid-walk"
echo "Condition: D512-2-derived outer-prior execution with no action-local inner SAC"
echo "Persistent outer SAC-style learning: enabled"
echo "Seed: 55"
echo "Agent decisions: 14000000"
echo "Inner model steps per action: 0"
echo "Manifest: $MANIFEST"
echo "Run root: $RUN_ROOT"
echo "W&B local staging: $WANDB_LOCAL_ROOT"
echo "Source commit: $actual_sha"
"$PYTHON" --version
"$PYTHON" -c '
import os
import sys

import gymnasium as gym
import torch
import wandb

import domains  # noqa: F401 - registers DMControl-v0


if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"expected Python 3.10, found {sys.version}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("expected exactly one scheduler-visible CUDA device")
device_name = torch.cuda.get_device_name()
if "RTX A6000" not in device_name:
    raise SystemExit(f"expected an RTX A6000 on gpu2201, found {device_name}")
env = gym.make(
    "DMControl-v0",
    task="humanoid-walk",
    obs="state",
    render_mode=None,
)
try:
    observation, _ = env.reset(seed=55)
    if observation.shape != (67,) or env.action_space.shape != (21,):
        raise SystemExit(
            f"unexpected Humanoid Walk spaces: obs={observation.shape}, "
            f"action={env.action_space.shape}"
        )
    if env.unwrapped.action_repeat != 2 or env.spec.max_episode_steps != 500:
        raise SystemExit("unexpected Humanoid Walk action-repeat/horizon contract")
finally:
    env.close()
print("torch:", torch.__version__)
print("gymnasium:", gym.__version__)
print("wandb:", wandb.__version__)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("cuda:", device_name)
print("Humanoid Walk contract: obs=67 action=21 repeat=2 horizon=500")
'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

srun --unbuffered --kill-on-bad-exit=1 \
  "$PYTHON" main.py \
  --run "$MANIFEST" \
  --alg-dir "$PROJECT_DIR_REAL/configs/dmcontrol/algs" \
  --log-dir "$RUN_ROOT" \
  --alg-index 0 \
  --trial-index 0 \
  --num-runs 1
