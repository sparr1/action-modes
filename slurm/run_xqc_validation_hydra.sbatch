#!/usr/bin/env bash
#SBATCH --job-name=xqc-validation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/tmp/xqc-validation-%j.out
#SBATCH --error=/tmp/xqc-validation-%j.err

set -Eeuo pipefail

# From a clean, pushed Action Modes checkout:
#   EXPECTED_ACTION_MODES_SHA="$(git rev-parse HEAD)" \
#   XQC_OFFICIAL_DIR=/absolute/path/to/xqc \
#     sbatch --export=ALL,EXPECTED_ACTION_MODES_SHA,XQC_OFFICIAL_DIR \
#       slurm/run_xqc_validation_hydra.sbatch

readonly OFFICIAL_XQC_SHA="9a6832bb742ef01bbe9f1e06153a9338e612dae5"
readonly PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
: "${EXPECTED_ACTION_MODES_SHA:?Export the exact pushed Action Modes commit SHA.}"
: "${XQC_OFFICIAL_DIR:?Export the absolute path to the pinned official XQC checkout.}"

PROJECT_DIR_REAL="$(cd "$PROJECT_DIR" && pwd -P)"
OFFICIAL_DIR_REAL="$(cd "$XQC_OFFICIAL_DIR" && pwd -P)"
readonly PROJECT_DIR_REAL OFFICIAL_DIR_REAL
readonly ACTION_PYTHON="$PROJECT_DIR_REAL/environments/dmcontrol/.venv/bin/python"
readonly OFFICIAL_PYTHON="$OFFICIAL_DIR_REAL/.venv/bin/python"
readonly SCRATCH_BASE="${SLURM_TMPDIR:-/tmp}"
readonly JOB_SCRATCH="$SCRATCH_BASE/xqc-validation-$SLURM_JOB_ID"
readonly ACTION_RUN_ROOT="$JOB_SCRATCH/action-modes"
readonly OFFICIAL_RUN_ROOT="$JOB_SCRATCH/official-xqc"
readonly WANDB_LOCAL_ROOT="$JOB_SCRATCH/wandb"

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

if [[ ! "$EXPECTED_ACTION_MODES_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPECTED_ACTION_MODES_SHA must be a full lowercase 40-character SHA." >&2
  exit 1
fi
if [[ ! -x "$ACTION_PYTHON" ]]; then
  echo "Missing locked Action Modes interpreter: $ACTION_PYTHON" >&2
  exit 1
fi
if [[ ! -x "$OFFICIAL_PYTHON" ]]; then
  echo "Missing official XQC interpreter: $OFFICIAL_PYTHON" >&2
  exit 1
fi

require_clean_sha "$PROJECT_DIR_REAL" "$EXPECTED_ACTION_MODES_SHA" "Action Modes"
require_clean_sha "$OFFICIAL_DIR_REAL" "$OFFICIAL_XQC_SHA" "Official XQC"

mkdir -p "$ACTION_RUN_ROOT" "$OFFICIAL_RUN_ROOT" "$WANDB_LOCAL_ROOT/cache" \
  "$WANDB_LOCAL_ROOT/data" "$WANDB_LOCAL_ROOT/artifacts"

export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_DIR="$WANDB_LOCAL_ROOT"
export WANDB_CACHE_DIR="$WANDB_LOCAL_ROOT/cache"
export WANDB_DATA_DIR="$WANDB_LOCAL_ROOT/data"
export WANDB_ARTIFACT_DIR="$WANDB_LOCAL_ROOT/artifacts"
export WANDB_DISABLE_CODE=true
export XQC_EVAL_CSV="$ACTION_RUN_ROOT/evaluation.csv"

echo "Job: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Action Modes: $PROJECT_DIR_REAL @ $EXPECTED_ACTION_MODES_SHA"
echo "Official XQC: $OFFICIAL_DIR_REAL @ $OFFICIAL_XQC_SHA"
echo "Scratch: $JOB_SCRATCH"
sha256sum "$PROJECT_DIR_REAL/environments/dmcontrol/uv.lock" "$OFFICIAL_DIR_REAL/uv.lock"
nvidia-smi

cd "$PROJECT_DIR_REAL"
srun --ntasks=1 "$ACTION_PYTHON" -c \
  'import sys,torch; from importlib.metadata import version; assert torch.cuda.is_available(); print(sys.version); print({name:version(name) for name in ("dm-control","gymnasium","mujoco","numpy","torch")}); print(torch.version.cuda,torch.get_default_dtype(),torch.cuda.get_device_name(0))'
srun --ntasks=1 "$OFFICIAL_PYTHON" -c \
  'import sys,jax; from importlib.metadata import version; assert jax.default_backend() == "gpu"; print(sys.version); print({name:version(name) for name in ("dm-control","flax","jax","jaxlib","mujoco","numpy","optax","tensorflow-probability")}); print(jax.devices())'
JAX_PLATFORMS=cpu srun --ntasks=1 "$OFFICIAL_PYTHON" \
  "$PROJECT_DIR_REAL/tests/oracles/generate_xqc_official_fixture.py" \
  --official-repo "$OFFICIAL_DIR_REAL" \
  --output "$JOB_SCRATCH/xqc_official_fixture.json"
srun --ntasks=1 "$ACTION_PYTHON" \
  "$PROJECT_DIR_REAL/tests/oracles/compare_xqc_fixture.py" \
  "$PROJECT_DIR_REAL/tests/fixtures/xqc_official_9a6832b.json" \
  "$JOB_SCRATCH/xqc_official_fixture.json" \
  --atol 1e-6 \
  --rtol 1e-5
srun --ntasks=1 "$ACTION_PYTHON" -c \
  'import gymnasium as gym, domains; env=gym.make("DMControl-v0", task="walker-walk", obs="state", render_mode=None); obs,info=env.reset(seed=1); print(obs.shape, obs.dtype, env.action_space.shape); env.close()'

AMBI_XQC_TEST_DEVICE=cpu srun --ntasks=1 "$ACTION_PYTHON" -m pytest -q \
  -p no:cacheprovider \
  tests/test_xqc_correctness.py \
  tests/test_xqc_integration.py \
  tests/test_xqc_official_smoke_oracle.py
AMBI_XQC_TEST_DEVICE=cuda AMBI_XQC_REQUIRE_CUDA=1 \
  srun --ntasks=1 "$ACTION_PYTHON" -m pytest -q \
    -p no:cacheprovider \
    tests/test_xqc_correctness.py \
    tests/test_xqc_integration.py \
    tests/test_xqc_official_smoke_oracle.py
srun --ntasks=1 "$ACTION_PYTHON" -m pytest -q \
  -p no:cacheprovider \
  tests/test_sac_correctness.py \
  tests/test_sac_distributional_q.py \
  tests/test_sac_target_update_cadence.py \
  tests/test_sac_online_eval.py \
  tests/test_dmcontrol_env.py \
  tests/test_dmcontrol_benchmark_configs.py

cd "$OFFICIAL_DIR_REAL"
srun --ntasks=1 "$OFFICIAL_PYTHON" \
  "$PROJECT_DIR_REAL/tests/oracles/run_official_xqc_smoke.py" \
  --official-repo "$OFFICIAL_DIR_REAL" \
  --expected-updates 40 \
  --max-projection-residual 1e-6 \
  env=walker-walk \
  seed=1 \
  num_seeds=1 \
  max_steps=10040 \
  replay_buffer_size=10000 \
  batch_size=256 \
  updates_per_step=2 \
  eval_interval=50000 \
  eval_episodes=1 \
  wandb.mode=disabled \
  "hydra.run.dir=$OFFICIAL_RUN_ROOT" \
  hydra.output_subdir=null

cd "$PROJECT_DIR_REAL"
srun --ntasks=1 "$ACTION_PYTHON" main.py \
  --run configs/dmcontrol/experiments/xqc_walker_walk_state_smoke.json \
  --alg-dir configs/dmcontrol/algs \
  --log-dir "$ACTION_RUN_ROOT" \
  --num-runs 1

ACTION_RUN_ROOT="$ACTION_RUN_ROOT" srun --ntasks=1 "$ACTION_PYTHON" -c \
  'import glob,math,os,numpy as np,torch; from torch.utils._pytree import tree_leaves; paths=glob.glob(os.environ["ACTION_RUN_ROOT"]+"/**/*_latest.pt",recursive=True); assert len(paths)==1,paths; state=torch.load(paths[0],map_location="cpu",weights_only=False); agent=state["agent"]; assert state["algorithm"]=="XQC" and state["official_xqc_commit"]=="9a6832bb742ef01bbe9f1e06153a9338e612dae5"; assert state["num_timesteps"]==5020,state["num_timesteps"]; assert (agent["update_step"],agent["actor_optimizer_steps"],agent["temperature_optimizer_steps"])==(40,14,14); leaves=tree_leaves(state); assert all(not torch.is_tensor(x) or not (x.is_floating_point() or x.is_complex()) or bool(torch.isfinite(x).all()) for x in leaves); assert all(not isinstance(x,(float,np.floating,np.ndarray)) or bool(np.isfinite(x).all()) for x in leaves); weights=[value for group in (agent["actor"],agent["critic"]) for key,value in group.items() if key.endswith((".linear.weight","mean.weight","log_std.weight",".value.weight"))]; residual=max(float((torch.linalg.vector_norm(value,dim=1)-1).abs().max()) for value in weights); assert residual<=1e-6,residual; print(paths[0],agent["update_step"],residual)'

require_clean_sha "$PROJECT_DIR_REAL" "$EXPECTED_ACTION_MODES_SHA" "Action Modes"
require_clean_sha "$OFFICIAL_DIR_REAL" "$OFFICIAL_XQC_SHA" "Official XQC"
echo "XQC Hydra validation passed. Artifacts: $JOB_SCRATCH"
